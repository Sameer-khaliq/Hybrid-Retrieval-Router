# Hybrid Retrieval & Query Routing System — Implementation Plan

**Status:** Ready to build · **Scope:** Solo build, portfolio-grade, free-tier constrained, live-demoable
**Sequencing:** Steps are ordered strictly by technical dependency, not by day. Phases group steps for readability only — a phase boundary is not permission to reorder steps inside it.

**Prerequisites:** Docker + Docker Compose, `uv`, Python 3.11+, and free-tier accounts/keys for Groq, Google AI Studio (Gemini), and Tavily.

## Sequencing Notes — read before Step 1

Decisions made in translating the requirements plus your three addenda into a strict build order, flagged the same way REQUIREMENTS.md flags its own filled gaps:

1. **Phase 7 is new** — not in your suggested phase list. FR-27–32 ("Extended Capabilities — finalized, not v2") have real dependencies on the v1 core: FR-28 needs FR-20's logs to exist *with volume in them*, and FR-30 is explicitly gated on Risk #5's mitigation being verified. Folding them into Phases 2–5 would mean hand-waving those dependencies away, so they're built last, against an already-validated core.
2. **Addendum #2's "~400ms" ceiling** — NFR-1 states p95 = 350ms for the LLM-router call specifically (400ms is closer to the *embedding* call's stated p95, one row up in that table). The plan keeps 400ms as the configured default anyway: it sits above the documented 350ms rather than at it — a ceiling set exactly at a stated p95 would, by construction, still cut off ~5% of calls that are behaving entirely within spec — and it's independently justified by retrieval-side timing (see Step 16). Flagging the mismatch rather than silently "correcting" the number without a paper trail.
3. **Two gaps the requirements doc leaves open, filled here:**
   - FR-9's "classified as needing current/external information" trigger has no stated detection mechanism. Built as a second cheap rule-based check (Step 20), independent of the RRF-confidence-floor check — not a dedicated LLM call, to protect NFR-4's 4-call budget.
   - FR-32 requires feedback to be "retrievable and correctly associated" with a trace ID. Stdout JSON logs alone aren't queryable after the fact, so Step 31 adds a small SQLite-backed store — the smallest addition that satisfies this without a new Docker service.
4. **FR-24 vs. FR-9 (your addendum #3)** — built as two structurally independent decision points: Step 14 (pre-retrieval, hard gate, zero retrieval calls) and Step 20 (post-retrieval, confidence/currency-based). Called out again at both steps so the independence survives implementation, not just design.

---

## Phase 1: Environment & Foundational Setup

### Step 1 — Project scaffolding & dependency management
**Builds:** `pyproject.toml` (via `uv init`), the full package layout (`config/`, `src/{ingestion,retrieval,routing,gating,generation,api,observability,eval,pipeline,common}/`, `tests/`, `data/`), `.gitignore`. Core deps added via `uv add`: `qdrant-client`, `bm25s`, `sentence-transformers`, `groq`, `google-genai`, `tavily-python`, `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`, `pytest`, `pytest-asyncio`.
**Why here:** every later step depends on a working `uv` environment and a stable import path. Note for `pydantic-settings`: in Pydantic v2, `BaseSettings` lives in the separate `pydantic-settings` package, not `pydantic` itself — a common break point. Also: install everything via `uv add`, never a bare local `pip install` — packages added outside `pyproject.toml` build fine locally and then silently fail in the Step 3 Docker build, a gotcha you've hit before.
**Done when:**
```bash
uv sync
uv run python -c "import qdrant_client, bm25s, sentence_transformers, groq, google.genai, tavily, fastapi, pydantic_settings"
```
Exits 0, no import errors.
**Pausable:** Yes.

### Step 2 — Central configuration module (FR-19 foundation)
**Builds:** `config/settings.py` — a `pydantic-settings` `BaseSettings` class (the one sanctioned exception to "no custom classes," per REQUIREMENTS.md §0.2) covering: chunk size/overlap bounds (FR-1), top-N per stage (default 20, FR-4/5), RRF `k` (default 60, FR-6), rerank top-K/top-K′ (15 / 5, FR-6), routing thresholds τ_low/τ_high (0.3/0.6, FR-12), the LLM-router timeout ceiling (400ms default — addendum #2, see Step 16 for the full justification), the Tavily confidence floor (FR-9), the Gemini embedding model/version/task_type/dimension pinned in one place (Risk #5), Groq model IDs for fast/deep and mid-tier placeholders (FR-27), max query length (2000 chars, FR-17), and the API-call budget ceiling (4, NFR-4). `.env.example` documents every variable. Also note: a trailing whitespace character in a `.env` value breaks the Docker Compose env-file parser silently — worth a comment in `.env.example` itself given it's bitten a prior build.
**Why here:** FR-19 requires every tunable to be externally configurable, never hardcoded. Building this before any pipeline logic means every module from Step 5 onward reads from `Settings` from its first line, instead of retrofitting config into already-hardcoded modules later.
**Done when:**
```bash
uv run python -c "from config.settings import Settings; s = Settings(); print(s.rrf_k, s.tau_low, s.tau_high, s.router_timeout_ms)"
```
Prints the configured defaults; changing one value in `.env` and re-running shows the new value with no source edit — this doubles as FR-19's own verify step, re-affirmed structurally by every later step that reads `Settings` instead of a literal.
**Pausable:** Yes.

### Step 3 — Qdrant via Docker Compose
**Builds:** `docker-compose.yml` (official `qdrant/qdrant` image, persistent volume, exposed port), `src/common/qdrant_client.py` — a connection-factory function reading host/port from `Settings`.
**Why here:** dense retrieval (Phase 2) needs a live Qdrant instance to write into; standing it up before any ingestion code means ingestion is tested against the real store from its first commit, not a mock.
**Done when:**
```bash
docker compose up -d qdrant
uv run python -c "from src.common.qdrant_client import get_client; print(get_client().get_collections())"
```
Returns an empty collections list, no error.
**Pausable:** Yes.

### Step 4 — Structured JSON logging skeleton (NFR-11 foundation)
**Builds:** `src/observability/logging_setup.py` — stdlib `logging` plus a JSON formatter emitting to stdout, and a `get_logger(trace_id=...)` helper every later module imports.
**Why here:** NFR-11 and FR-20 require trace-ID-tagged structured logs on every query from day one. Every module built from here forward should log through this from its first line, not have logging bolted on retroactively — retrofitting structured logging into already-written modules is exactly the rework this step is designed to avoid.
**Done when:**
```bash
uv run python -c "from src.observability.logging_setup import get_logger; get_logger(trace_id='t1').info('smoke_test', stage='setup')" | python -m json.tool
```
Output is valid, parseable JSON with a `trace_id` field.
**Pausable:** Yes.

---

## Phase 2: Ingestion & Dual Retrieval Core

### Step 5 — Document loading & chunking (FR-1)
**Builds:** `src/ingestion/loaders.py` (PDF, TXT, Markdown readers → raw text), `src/ingestion/chunking.py` (token-based splitter, configurable size/overlap bounds pulled from `Settings`).
**Why here:** everything else in ingestion and both retrieval legs consumes chunks — this is the first concrete artifact in the corpus pipeline.
**Done when:**
```bash
uv run pytest tests/test_chunking.py -v
```
A fixture document's every chunk falls within the configured token range, and measured overlap matches config ±1 token — FR-1's own stated verify, encoded directly as the test.
**Pausable:** Yes.

### Step 6 — Metadata tagging & content-hash dedup (FR-2, FR-3)
**Builds:** `src/ingestion/pipeline.py` — attaches metadata (source doc ID, chunk index, character offset range, ingestion timestamp) to each chunk from Step 5; a content-hash function (SHA-256 over normalized chunk text) checked against a persisted hash set before a chunk is accepted for storage.
**Why here:** must exist before chunks are ever written to Qdrant or BM25 (Steps 7–8). Sequenced ahead of both so no persistent store ever receives un-deduplicated data in the first place.
**Done when:**
```bash
uv run pytest tests/test_ingestion_pipeline.py -v
```
Covers FR-2 (query stored metadata after a fixture ingest, no null fields) and FR-3 (run the pipeline against the same fixture file twice, chunk count identical after the second run) — both FRs' own stated verify.
**Pausable:** Yes.

### Step 7 — Dense embedding & Qdrant write path (FR-5 write half, Risk #5 pinning)
**Builds:** `src/ingestion/embedder.py` — a Gemini embedding call wrapper pinning model/version/`task_type=document`/output-dimension in one place per `Settings` (Risk Register #5); extends `pipeline.py` to embed each deduped chunk and upsert into Qdrant with Step 6's metadata as payload, also logging the embedding config alongside each stored vector (feeds Step 10's consistency check).
**Why here:** needs deduped, metadata-tagged chunks (Step 6) as input; must exist before dense retrieval (Step 9) has anything to query against.
**Done when:**
```bash
uv run python -m src.ingestion.pipeline --path tests/fixtures/sample.pdf
uv run python -c "from src.common.qdrant_client import get_client; print(get_client().count('chunks'))"
```
Shows a non-zero, expected point count.
**Pausable:** Yes.

### Step 8 — BM25 sparse index: build, disk persistence, hot-reload (FR-4, addendum #1)
**Builds:** `src/retrieval/sparse_bm25.py` — `build_index(chunks)` using `bm25s`; `save_index()` / `load_index()` via `bm25s`'s native save/load to disk; `get_or_build_index()`, which loads from disk on startup if a persisted index exists and only falls back to a full-corpus tokenize+build when it doesn't (first run, or an explicit `--rebuild` flag); a tokenization cache (`chunk_id → tokens`, persisted alongside the index) so a later rebuild after new ingestion only re-tokenizes *new* chunks — worth being precise about what "incremental" means here: `bm25s` has no true single-document incremental-update API (BM25's IDF and average-doc-length statistics are corpus-wide, so any new document changes them), so the statistical fit itself is still recomputed on every rebuild. What's genuinely incremental is the tokenization cost, not the index-mutation algorithm. A `rebuild_and_swap()` function builds the new index off the request-serving path, writes it to a temp file then atomically `os.rename`s it into place, then swaps the in-process index reference the query path (Step 9/12) reads from — so a query arriving mid-rebuild is served by the last-good index, never a half-built one.
**Why here:** this is addendum #1's core requirement; built directly on the ingestion output (Steps 5–7) so persistence/reload is validated against a real chunk corpus, not a mock. Must exist before Step 12 wires BM25 in as a live concurrent dependency.
**Done when:**
```bash
uv run python -m src.retrieval.sparse_bm25 --build
uv run pytest tests/test_bm25_persistence.py -v
```
Confirms: (a) a fresh Python process calling `load_index()` returns results identical to the pre-persist in-memory index; (b) after ingesting one more fixture doc and calling `rebuild_and_swap()`, the new chunk is retrievable and the tokenization-cache hit count for pre-existing chunks equals the pre-existing chunk count (they weren't re-tokenized); (c) a query fired concurrently during the rebuild does not error or return partial results.
**Pausable:** **No — finish `rebuild_and_swap()` as one unbroken unit.** A partial implementation (disk persistence wired but the in-memory pointer swap not yet implemented, or vice versa) is exactly the "un-persisted index sync" failure mode named in your instructions: the live process keeps serving a stale index while believing it's current, or crashes on next query against a not-yet-written reference. The `build_index`/`save_index`/`load_index` trio, by contrast, is safe to pause after individually.

### Step 9 — Dense retrieval query path (FR-5 query half)
**Builds:** `src/retrieval/dense_qdrant.py` — embeds the query (reusing Step 7's embedder with `task_type=query`, per Risk #5's document/query distinction), cosine-similarity search against Qdrant, top-N default 20 from `Settings`.
**Why here:** depends on Step 7's embeddings already sitting in Qdrant to search against; independent of Step 8, sequenced here because it shares the embedder module with Step 7.
**Done when:**
```bash
uv run pytest tests/test_dense_retrieval.py -v
```
Against a fixture corpus, returned chunk IDs match expected nearest neighbors within tolerance — FR-5's own verify.
**Pausable:** Yes.

### Step 10 — Embedding-config consistency check (Risk #5 mitigation, ties to FR-19)
**Builds:** `src/common/config_check.py` — a startup check comparing the embedding config logged at ingestion time (Step 7) against the config the query path (Step 9) is about to use; fails loudly (raises, non-zero exit) on any mismatch instead of silently returning meaningless similarity scores.
**Why here:** needs both the ingest-side (Step 7) and query-side (Step 9) embedding calls to already exist so there's something concrete to compare; must exist before Step 9's output is trusted for anything downstream.
**Done when:**
```bash
uv run pytest tests/test_config_consistency.py -v
```
Deliberately mismatching `task_type` between a mocked ingest-config and query-config causes the check to raise before any Qdrant call is made; matching configs pass silently.
**Pausable:** Yes.

### Step 11 — Embedding-failure fallback & shared retry/backoff (FR-8, NFR-8 foundation)
**Builds:** `src/common/resilience.py` — a shared exponential-backoff retry wrapper (max 2 retries on 429/5xx/timeout, NFR-8) intended for reuse by every external call in the system; `src/retrieval/fallback.py` wraps Step 9's embedding call in it, and on exhausted retries returns a `degraded: true` sparse-only result rather than raising.
**Why here:** needs both sparse (Step 8) and dense (Step 9) paths to exist so "fall back to sparse when dense fails" has a real sparse path to fall back to. Also the natural place to build the *shared* retry/backoff helper — it's needed here first and gets reused by Step 16's router call, Step 20's Tavily call, and Step 22's generation calls. Building it once now avoids four divergent retry implementations later.
**Done when:**
```bash
uv run pytest tests/test_embedding_fallback.py -v
```
Fault-injection test mocking embedding failure across all retries; a response is still returned with `degraded: true` and sparse-only results — FR-8's own verify.
**Pausable:** Yes.

### Step 12 — Concurrent sparse+dense retrieval orchestration (FR-7)
**Builds:** `src/retrieval/orchestrator.py` — runs Step 8's BM25 query and Step 9's dense query (through Step 11's fallback wrapper) concurrently via `asyncio.gather`.
**Why here:** both legs must independently exist and be independently tested *before* wiring concurrency — a concurrency bug in a leg that hasn't already been verified correct in isolation is much harder to debug than concurrency alone once both legs are already trustworthy.
**Done when:**
```bash
uv run pytest tests/test_retrieval_concurrency.py -v
```
Instrumented timing (using an artificial delay injected into a mock to make the assertion unambiguous) shows retrieval-stage wall-clock ≈ max(sparse_time, dense_time), not their sum — FR-7's own verify.
**Pausable:** **No — finish as one unbroken unit.** A partially-wired `asyncio.gather` (only one leg actually awaited, or Step 11's fallback wrapper accidentally bypassed by the concurrency refactor) is a classic silent-bug source: code that runs without error but returns partial or sequential-not-concurrent results, surfacing only later as an unexplained NFR-2 latency-budget miss.

### Step 13 — RRF fusion — math only (FR-6, first half)
**Builds:** `src/retrieval/fusion.py` — `rrf_fuse(sparse_ranked, dense_ranked, k=60)`, combining Step 12's two ranked lists.
**Why here:** needs both ranked lists to fuse. Deliberately scoped to *only* the RRF math — FR-6's conditional cross-encoder-invocation half is deferred to Step 19 (Phase 4) because it depends on the routing decision, which doesn't exist until Phase 3. Building all of FR-6 here would mean either hardcoding a placeholder routing decision (hand-wavy) or blocking this step on Phase 3 finishing first, breaking strict dependency order.
**Done when:**
```bash
uv run pytest tests/test_rrf_fusion.py -v
```
Unit test with synthetic rank lists confirms `score = Σ 1/(k + rank)` exactly — FR-6's fusion-only verify.
**Pausable:** Yes.

---

## Phase 3: Query Gating & Routing Layer

### Step 14 — Query gating: non-corpus intent, abuse, credential-solicitation, out-of-scope (FR-21, FR-22, FR-23, FR-24, FR-25)
**Builds:** `src/gating/prefilter.py` — four independent rule-based/regex matchers (FR-25 mandates local-only matching, no LLM call, in the common case): non-corpus intent (FR-21: greetings/meta-questions/gratitude/out-of-scope domains), abusive language (FR-22, logged as a flagged category rather than stored verbatim), credential-solicitation as a **request-pattern** match, not a bare keyword match (FR-23 — must match "give me your API key" but *not* "explain how password hashing works"; needs a small labeled fixture set of both trigger and non-trigger phrasings to get this distinction right, not just a keyword list), and domain-relevance (FR-24). Each returns either `None` (pass through) or a fixed response object.
**Design note (addendum #3):** FR-24's matcher here is a hard pre-retrieval domain-relevance gate. It must not share state or a scoring function with FR-9's post-retrieval RRF-confidence check (Step 20) — wire them as two independent decision points, not two call sites reading the same "relevance score." Conflating them risks a low RRF score looking like an FR-24 gate match, or a query FR-24 correctly let through getting double-gated later by mistake.
**Why here:** functionally self-contained (operates only on the raw query string), but sequenced here rather than Phase 1 because NFR-13's "zero retrieval-pipeline stage invocations logged in `latency_breakdown`" verify criterion needs Phase 2's retrieval pipeline — with Step 12's instrumentation — already in place to assert against. Testing "zero invocations" against a pipeline that doesn't exist yet is a vacuous test.
**Done when:**
```bash
uv run pytest tests/test_gating.py -v
uv run pytest tests/test_gating_latency.py -v
```
The first covers FR-21/22/23/24's fixture sets matching correctly, including FR-23's non-trigger set ("how does password hashing work" must *not* gate). The second asserts p95 <50ms per NFR-13 and, via Step 12's instrumentation, confirms zero retrieval-pipeline calls logged in `latency_breakdown` for each matched case.
**Pausable:** Yes — the four matchers are independently built and unit-tested here. The partial-wiring risk flagged in the instructions applies to Step 21, where these matchers get connected to the live request path — flagged there, not here.

### Step 15 — Composite query-complexity score (FR-11)
**Builds:** `src/routing/features.py` — four feature functions (normalized token count, comparison/multi-hop trigger-term presence, question-word category, clause count) combined via §3.4's weighted formula: `score = 0.20·token_count_feat + 0.35·keyword_feat + 0.25·question_word_feat + 0.20·clause_count_feat`, with all weights pulled from `Settings`.
**Why here:** a pure function of query text only — no dependency on retrieval or gating, so it *could* have been built earlier. Sequenced here, adjacent to its only consumer (Step 16), to keep the routing narrative together.
**Done when:**
```bash
uv run pytest tests/test_complexity_score.py -v
```
Unit tests over a labeled query set assert scores fall in the expected band per query — FR-11's own verify.
**Pausable:** Yes.

### Step 16 — Two-layer routing cascade: thresholds, LLM fallback, timeout ceiling (FR-12, FR-13, FR-14 setup, addendum #2)
**Builds:** `src/routing/layer0_rules.py` (FR-12: score < τ_low → fast, score > τ_high → deep, else → Layer 1); `src/routing/layer1_llm.py` (FR-13: a single Groq `llama-3.1-8b-instant` call at temperature 0, returning `{path, confidence, reason}` validated against a Pydantic schema). The Layer-1 call is wrapped in `asyncio.wait_for(..., timeout=settings.router_timeout_ms)` — default 400ms — with Step 11's `resilience.py` retry/backoff (NFR-8) running *inside* that same window, not layered on top of it: if a 429 retry-with-backoff would blow past the ceiling, the outer `wait_for` still cancels at 400ms. Two distinct, deliberately different failure branches: **(a)** the call returns but fails schema validation → default **fast-path**, `degraded: true` (FR-13's stated default, per NFR-9); **(b)** the call exceeds the 400ms ceiling entirely, with or without exhausting retries first → default **deep-path** (addendum #2, grounded in Risk Register #2's stated bias toward deep-path under routing uncertainty). Each branch logs a distinct `deciding_layer` value (`"malformed-fallback"` vs. `"timeout-fallback"`) so FR-20's log field is unambiguous about which happened.
**Design note — the 400ms figure:** NFR-1 states p95 = 350ms for this exact call, not ~400ms (see Sequencing Note #2 above). 400ms is used as the configured ceiling anyway, for two reasons: it sits *above* the documented p95 rather than at it, and it's independently consistent with the retrieval side of the pipeline — query-embedding p95 (400ms, NFR-1) plus retrieval-stage p95 (60ms, NFR-1) puts fusion-readiness at roughly 460ms p95, so a 400ms router ceiling keeps Layer-1 comfortably off the critical path at the p95 level, satisfying FR-14 without needing a tighter, riskier number.
**Why here:** depends on Step 15 (the score Layer 0 thresholds against) and Step 11's `resilience.py`; must exist before Step 19 wires the routing decision into the conditional-rerank logic that finishes FR-6.
**Done when:**
```bash
uv run pytest tests/test_routing_cascade.py -v
```
Boundary-value tests at and around τ_low/τ_high confirm correct path selection (FR-12's verify); a schema-validation test with malformed mocked LLM output confirms the fast-path default (FR-13's verify); a mocked-hang test (a call that never returns) confirms the deep-path default fires at exactly the configured ceiling, not meaningfully before or after.
**Pausable:** **No — finish as one unbroken unit.** The two failure branches are easy to accidentally collapse into a single "on any router problem, do X" handler if built incrementally — silently violating one of the two explicitly different requirements (FR-13/NFR-9's fast-path default vs. addendum #2's deep-path-on-timeout default). Implement and test both branches together before moving on.

### Step 17 — Routing-retrieval concurrency wiring (FR-14)
**Builds:** extends `src/retrieval/orchestrator.py` — kicks off Step 16's routing cascade via `asyncio.gather` alongside Step 12's sparse+dense retrieval, rather than awaiting retrieval first and only then starting routing.
**Why here:** needs Step 12 (retrieval concurrency) and Step 16 (routing cascade) fully built and independently tested first — same reasoning as Step 12: concurrency bugs are far easier to isolate once both sides of the `gather` are already individually correct.
**Done when:**
```bash
uv run pytest tests/test_routing_retrieval_concurrency.py -v
```
Instrumented timing shows routing start time ≈ retrieval start time, within a few ms — FR-14's own verify.
**Pausable:** **No — finish as one unbroken unit.** Same risk class as Step 12: a partially-wired `gather` here can silently degrade to sequential execution without raising, surfacing only later as an unexplained NFR-2 latency miss.

---

## Phase 4: Fusion, Reranking & Fallbacks

### Step 18 — Cross-encoder reranker (FR-10)
**Builds:** `src/retrieval/rerank.py` — a local `sentence-transformers` cross-encoder (an `ms-marco-MiniLM-L-6`-class model, per REQUIREMENTS.md §0.2), scoring each `(query, candidate)` pair independently and returning candidates ordered by descending relevance.
**Why here:** a standalone scoring function needing only candidate chunks (available since Phase 2) and a query string — genuinely independent of routing. Sequenced here because its only real consumer, Step 19's conditional-invocation wiring, needs the routing decision (Phase 3, now complete) to decide whether to call it at all.
**Done when:**
```bash
uv run pytest tests/test_reranker.py -v
```
A synthetic case with a known relevant/irrelevant pair confirms correct ordering (FR-10's own verify); a separate timing test logs raw CPU latency for a top-15 rerank against the eventual demo corpus size, feeding Risk Register #3's "benchmark actual CPU latency in week one, not week four" mitigation directly.
**Pausable:** Yes.

### Step 19 — Conditional rerank wiring — completes FR-6
**Builds:** extends `src/retrieval/fusion.py` — given Step 13's fused list and Step 16/17's routing decision: deep-path invokes Step 18's reranker over top-K (default 15); fast-path skips it or applies it to a reduced top-K′ (default 5), per `Settings`.
**Why here:** the second, previously-deferred half of FR-6 (see Step 13). Now buildable because both its dependencies — fusion output and a routing decision — exist.
**Done when:**
```bash
uv run pytest tests/test_conditional_rerank.py -v
```
Two fixture queries pre-labeled fast/deep confirm the reranker is invoked or skipped as expected, with the expected K — FR-6's conditional-rerank verify, completing what Step 13 left open.
**Pausable:** Yes.

### Step 20 — Tavily fallback leg (FR-9)
**Builds:** `src/retrieval/tavily_fallback.py` — triggers on either **(a)** Step 19's fused/reranked top-1 RRF score falling below `settings.confidence_floor`, or **(b)** a cheap rule-based "needs current/external info" check (keyword/pattern signals — "today," "latest," "current," date-relative phrasing — deliberately *not* a dedicated LLM call, to protect NFR-4's ≤4-call budget). Tags results `"web"` vs. `"corpus"`; reuses `resilience.py` (Step 11) for retry/backoff.
**Design note (addendum #3, reinforced):** this trigger logic reads Step 19's already-computed fusion/rerank confidence — it must not re-derive or share the domain-relevance signal from Step 14's FR-24 gate. A query reaching this step has, by construction, already passed FR-24, or it would have short-circuited before retrieval ever ran. FR-9 firing here is strictly a post-retrieval confidence/currency decision, never a re-litigation of FR-24's relevance gate.
**Why here:** depends on Step 19 for a real fused/reranked result and its confidence score to evaluate against the floor.
**Done when:**
```bash
uv run pytest tests/test_tavily_fallback.py -v
```
A fixture query designed to score below the floor triggers a mocked Tavily call, with sources correctly tagged `"web"`/`"corpus"` (FR-9's own verify); a second fixture confirms a normal high-confidence corpus query does *not* trigger Tavily.
**Pausable:** Yes.

---

## Phase 5: Generation, API, Streaming & Observability

### Step 21 — Full pipeline wiring: gating → routing/retrieval → fusion → rerank → Tavily
**Builds:** `src/pipeline/run_query.py` — the end-to-end internal function: Step 14's gating (checked first, short-circuits immediately if matched) → if passed, Step 17's concurrent routing+retrieval → Step 19's conditional rerank → Step 20's Tavily check → assembled context ready for generation.
**Why here:** the first point where every earlier phase's output actually has to compose into one call path. Deliberately kept separate from the HTTP layer (Step 23) so pipeline logic is testable without spinning up FastAPI.
**Done when:**
```bash
uv run pytest tests/test_pipeline_e2e.py -v
```
A full fixture run through gating-pass → routing → retrieval → fusion → rerank → Tavily-check asserts every stage's output correctly feeds the next; a separate gating fixture confirms a matched FR-21–24 case returns immediately without touching Steps 17/19/20 at all.
**Pausable:** **No — finish as one unbroken unit.** This is the "split state" risk from your instructions at the whole-pipeline level: if gating's short-circuit is wired for some FR-21–24 categories but not others (the risk flagged forward from Step 14), or if Step 17's routing decision isn't actually threaded through to Step 19's conditional-rerank call, the pipeline runs end-to-end without erroring while silently doing the wrong thing — e.g., always reranking regardless of path, or leaking a gated query through to a live Groq call. Build and integration-test the full chain before considering this step done.

### Step 22 — Generation: streaming & degraded-mode fallback (FR-26, NFR-9 generation half)
**Builds:** `src/generation/groq_stream.py` — a Groq streaming completion call for both fast-path and deep-path models. On deep-path failure or rate-limit after Step 11's `resilience.py` retries are exhausted, falls back to the fast-path model and flags `degraded: true` (NFR-9). Gated responses (Step 14) bypass this module entirely and return as one atomic, non-streamed response — FR-26's explicit carve-out.
**Why here:** needs Step 21's assembled context (retrieved and reranked chunks) as generation input, and Step 16's routing decision to pick the fast or deep model.
**Done when:**
```bash
uv run pytest tests/test_generation_streaming.py -v
```
The client receives multiple ordered partial chunks before the final chunk, for both paths (FR-26's verify); a fault-injection test on deep-path failure confirms fallback to fast-path output with `degraded: true` (NFR-9's verify); a gated-response fixture confirms single, atomic, non-chunked delivery.
**Pausable:** Yes.

### Step 23 — FastAPI surface: `/query`, `/health`, input validation (FR-15, FR-16, FR-17)
**Builds:** `src/api/main.py`, `src/api/schemas.py` — `POST /query` wired to Steps 21+22's pipeline, returning `{answer, sources[], routing_metadata: {path, confidence, reason}, latency_breakdown: {stage: ms}}`; `GET /health` checking Qdrant/Groq/Gemini/Tavily reachability; input validation (empty or >2000-char query → 4xx, zero downstream calls) running as a dependency *before* the pipeline is invoked at all — distinct from Step 14's content-based gating, which runs *inside* the pipeline.
**Why here:** the first externally-callable surface — depends on the full pipeline (Step 21) and generation (Step 22) to wire against.
**Done when:**
```bash
docker compose up -d
uv run uvicorn src.api.main:app &
curl -s -X POST localhost:8000/query -d '{"query":"test"}' | python -m json.tool
curl -s -X POST localhost:8000/query -d '{"query":""}'
curl -s localhost:8000/health | python -m json.tool
```
The first call returns the documented response schema; the second returns 4xx; the third shows per-dependency status — FR-15/16/17's combined verify.
**Pausable:** Yes.

### Step 24 — Observability logging consolidation (FR-20, NFR-11, NFR-4 tracking half)
**Builds:** consolidates logging across every module from Steps 5–23 through Step 4's `get_logger(trace_id=...)`; assembles the full `latency_breakdown` dict and an `api_call_count` field inside `run_query.py` (Step 21).
**Why here:** lightweight logging calls were added incrementally as each module was written, per Step 4's foundation — this step is the *consolidation and verification* point. Confirming trace IDs are actually consistent across every stage of one request, and that NFR-11's full field set is present, is only meaningfully testable once the full pipeline (Step 23) exists end-to-end.
**Done when:**
```bash
uv run pytest tests/test_observability.py -v
```
After one test query through `/query`: trace ID is identical across every log line for that query; all NFR-11 fields are present (per-stage latency, routing decision + reason + deciding layer, retrieved chunk IDs pre/post-rerank, degraded flags, API-call count) — FR-20/NFR-11's combined verify. A separate assertion confirms `api_call_count` ≤ 4 for a fixture query exercising the worst case (embedding + LLM-router + generation + Tavily all firing) — NFR-4's DoD gate, made concrete and testable rather than aspirational.
**Pausable:** Yes.

---

## Phase 6: Evaluation & Hardening

### Step 25 — Batch evaluation harness (FR-18)
**Builds:** `src/eval/metrics.py` (Recall@k, MRR, routing-decision-accuracy implementations), `src/eval/run_eval.py` (a CLI entrypoint running a labeled query set through Step 21's pipeline, writing a report file). Uses the small fixture corpora/query sets already established for unit and integration testing throughout Phase 2 — not the larger curated demo corpus built in Step 26.
**Why here:** needs the full pipeline (Step 21) to exist and be callable end-to-end. This is also the first point where NFR-5 (Recall@10 ≥ 0.85) and NFR-6 (≥80% routing agreement) become measurable rather than aspirational.
**Done when:**
```bash
uv run python -m src.eval.run_eval --set tests/fixtures/eval_set.jsonl
uv run pytest tests/test_eval_harness.py -v
```
The first produces a report file; the second runs against a fixture eval set with precomputed expected metric values, asserting output matches within tolerance — FR-18's own verify.
**Pausable:** Yes.

### Step 26 — Demo corpus curation & query-set rehearsal (Risk #4 mitigation)
**Builds:** `data/corpus/` — a first-class curated demo corpus (not an afterthought, per Risk Register #4), sized to comfortably fit BM25's in-memory constraint (documented properly in Step 27's `SCALABILITY_BOUNDARY.md`); a rehearsed query set in `tests/fixtures/demo_queries.jsonl` deliberately exercising fast-path, deep-path, the LLM-router mid-band, FR-9's Tavily trigger, and each of FR-21–24's gating categories, so the demo has a concrete, rehearsed case for every major branch in the system.
**Why here:** distinct from Step 25's small test fixtures — this is a separate, larger corpus built specifically for the live client walkthrough, so it's only meaningful to curate once there's a working pipeline (Step 21) and eval harness (Step 25) to validate it against. Curating a corpus before there's a system to test it against risks discovering misalignment (Risk #4's stated failure mode) too late to fix. Running the full rehearsed query set back-to-back through Groq is also exactly the load shape that hit the Groq TPM ceiling mid-run on a prior build — run this rehearsal early enough to catch that here, not live in front of a client.
**Done when:**
```bash
uv run python -m src.eval.run_eval --set tests/fixtures/demo_queries.jsonl
```
Manually confirm, from the report and log output, that each rehearsed category routes, gates, or falls back as intended.
**Pausable:** Yes.

### Step 27 — DoD-gate validation pass (NFR-2, NFR-4, NFR-5, NFR-6, NFR-8, NFR-9, NFR-11, NFR-12)
**Builds:** no new application code — `scripts/validate_dod.sh` (or a pytest marker suite) running Step 25's eval harness plus targeted latency tests against Step 23's live `/query` endpoint, checking each DoD-gate NFR against its stated number; `docs/SCALABILITY_BOUNDARY.md` — NFR-12's explicit documentation requirement ("must be explicitly documented, not discovered by the client"): in-memory BM25's RAM ceiling, single-node Qdrant's lack of HA/replication, and free-tier rate limits as the real concurrency ceiling, framed as what the architecture *would* scale to on paid infrastructure, not what it already does.
**Why here:** the final gate before calling v1 done. Every prior step produced a component; this step measures them together, end-to-end, against the actual numeric DoD gates — echoing the requirements doc's own warning that CorpMind's projected numbers came in ~57x over once actually load-tested. Measure here, don't assume.
**Done when:**
```bash
bash scripts/validate_dod.sh
```
A single run producing a pass/fail line for each of NFR-2, 4, 5, 6, 8, 9, 11, 12 (NFR-13 is already covered by Step 14's own test); `docs/SCALABILITY_BOUNDARY.md` exists and has been reviewed.
**Pausable:** Yes.

---

## Phase 7: Extended Capabilities (FR-27–32)

Everything below builds on a v1 core that Step 27 has already validated. Ordered by which extensions have real data/measurement dependencies (FR-28, FR-30) versus which are pure additions (FR-27, FR-29, FR-31, FR-32).

### Step 28 — Weighted linear fusion, alternative to RRF (FR-29)
**Builds:** extends `src/retrieval/fusion.py` — `weighted_fuse(sparse_ranked, dense_ranked, sparse_weight, dense_weight)` over normalized scores, selectable via a `fusion_strategy` config flag (default `"rrf"`).
**Why here:** a pure extension of Steps 13/19's fusion module with no dependency on anything built after Phase 2 — technically buildable earlier, but held until after Step 27 so the config flag can't accidentally change behavior during the DoD-gate measurement pass.
**Done when:**
```bash
uv run pytest tests/test_weighted_fusion.py -v
```
Switching the config flag changes fusion behavior on a fixture query set; RRF remains default when unset — FR-29's own verify.
**Pausable:** Yes.

### Step 29 — Three-tier routing: fast/mid/deep (FR-27)
**Builds:** extends `src/routing/layer0_rules.py` — a second threshold on Step 15's existing continuous complexity score (per §3.3's explicit design intent: "a threshold change, not a redesign"), mapping to `llama-3.1-8b-instant` (fast) / `llama-3.3-70b-versatile` or `gpt-oss-20b` (mid) / `gpt-oss-120b` (deep); extends Step 22's generation module to accept a third model tier.
**Why here:** depends only on Phase 3's routing (Steps 15–16) — buildable earlier, but sequenced here for the same reason as Step 28: don't destabilize the binary-routing numbers Step 27 just validated.
**Done when:**
```bash
uv run pytest tests/test_three_tier_routing.py -v
```
Three fixture queries spanning the complexity range route to three distinct model tiers; boundary values at both thresholds resolve correctly — FR-27's own verify.
**Pausable:** Yes.

### Step 30 — LangSmith tracing (FR-31)
**Builds:** `src/observability/tracing.py` — per-stage spans (retrieval, fusion, routing, rerank-if-invoked, generation) matching Step 24's logged fields. Can reuse the same `@traceable`-plus-singleton-agent-instantiation pattern already proven out in `researchpilot-ai`, rather than designing a new tracing structure from scratch.
**Why here:** depends on Step 24's field set as the source of truth for what a span needs to capture. Purely additive, so sequenced after the DoD-validated core (Step 27) rather than before.
**Done when:**
```bash
uv run pytest tests/test_tracing.py -v
```
A test query produces a trace with spans for every stage, with timestamps consistent with `latency_breakdown` — FR-31's own verify.
**Pausable:** Yes.

### Step 31 — Feedback capture (FR-32)
**Builds:** `src/api/feedback.py` (`POST /feedback`) plus `src/common/trace_store.py` — a small SQLite-backed store (functional wrapper functions over `sqlite3`, no ORM class, consistent with the "no custom classes" constraint) keyed by trace ID, persisting each query's routing decision and retrieved chunk IDs so feedback can be looked up and associated after the fact.
**Filled gap:** REQUIREMENTS.md specifies FR-32's persistence and retrievability requirement but not a storage mechanism — stdout JSON logs alone aren't queryable by trace ID after the fact. SQLite is proposed as the smallest addition that satisfies "retrievable... correctly associated" without a new Docker service or a break from the free-tier/solo-build constraints, flagged the same way REQUIREMENTS.md flags its own filled gaps in §0.2.
**Why here:** depends on Step 23 (trace-ID-bearing `/query` responses) and Step 24 (the fields being persisted). The doc explicitly scopes FR-32 to capture-only — this step stops at persistence and retrieval, nothing more.
**Done when:**
```bash
uv run pytest tests/test_feedback.py -v
```
A feedback submission against a known trace ID is retrievable and correctly associated; submission against an unknown trace ID returns 4xx — FR-32's own verify.
**Pausable:** Yes.

### Step 32 — k-NN routing signal, Layer 2 (FR-28)
**Builds:** `src/routing/layer2_knn.py` — k-NN against a labeled reference set, reusing Step 9's already-computed query embedding (no new embedding call, protecting NFR-4); a bootstrap script building the initial labeled reference set from Steps 24/31's accumulated routing-decision logs.
**Why here:** the hardest real dependency in the plan — it needs logged routing decisions to exist *in volume*, which means the system needs to have actually been run (Step 26's rehearsal onward, plus whatever real usage follows) before this step has real data to bootstrap from. Also depends on Step 25's eval harness to measure the layer's agreement rate before deciding whether to merge it live.
**Done when:**
```bash
uv run python -m src.routing.layer2_knn --bootstrap
uv run pytest tests/test_knn_routing.py -v
```
The bootstrap builds the reference set from logged data; the test measures held-out agreement against Step 25's Layer-0/1-alone baseline — FR-28's own verify. Note this is explicitly conditional: per the requirements doc, the layer is only merged into the live cascade if it improves accuracy (NFR-6), so a "pass" here may legitimately mean "built, measured, and correctly *not* merged."
**Pausable:** Yes.

### Step 33 — Semantic caching, gated on embedding-config stability (FR-30)
**Builds:** `src/retrieval/cache.py` — an embedding-similarity-threshold cache for retrieval+generation results, behind a `caching_enabled` flag defaulting to `false` until Step 10's consistency check has a clean, passing run logged.
**Why here:** last step by explicit design. REQUIREMENTS.md states this "SHALL NOT be enabled until embedding-config stability... is verified," and sequencing it after Step 27's DoD validation means the DoD-gate latency/cost numbers (NFR-2, NFR-4) were measured *without* caching's help — an honest uncached baseline before caching is layered on as an optimization.
**Done when:**
```bash
uv run pytest tests/test_semantic_cache.py -v
```
A repeated near-duplicate (paraphrased) query is served from cache; a genuinely distinct query is not; `api_call_count` (from Step 24's logging) drops for the cached case — FR-30's own verify.
**Pausable:** Yes.

---

## Final Checklist — FR/NFR → Step Mapping

| ID | What it covers | Step(s) |
|---|---|---|
| FR-1 | Chunking: 200–500 tok, 10–15% overlap | 5, 6 |
| FR-2 | Chunk metadata | 6 |
| FR-3 | Content-hash dedup | 6 |
| FR-4 | BM25 top-N | 8 |
| FR-5 | Dense top-N (cosine) | 7, 9 |
| FR-6 | RRF fusion + conditional rerank | 13, 19 |
| FR-7 | Concurrent sparse ∥ dense | 12 |
| FR-8 | Embedding-failure → sparse-only fallback | 11 |
| FR-9 | Tavily fallback leg | 20 |
| FR-10 | Cross-encoder reranker | 18 |
| FR-11 | Composite complexity score | 15 |
| FR-12 | Threshold routing (τ_low/τ_high) | 16 |
| FR-13 | LLM-router fallback | 16 |
| FR-14 | Routing ∥ retrieval concurrency | 17 |
| FR-15 | `POST /query` contract | 23 |
| FR-16 | `GET /health` | 23 |
| FR-17 | Input validation (empty/oversized) | 23 |
| FR-18 | Batch eval entrypoint | 25 |
| FR-19 | Externalized config | 2 (foundation, enforced by every later step) |
| FR-20 | Per-query structured logging | 4 (foundation), 24 (consolidated & verified) |
| FR-21 | Non-corpus-intent gate | 14 |
| FR-22 | Abusive-language gate | 14 |
| FR-23 | Credential-solicitation gate | 14 |
| FR-24 | Domain-relevance gate | 14 |
| FR-25 | Local-only gating (no LLM call) | 14 |
| FR-26 | Token-by-token streaming | 22 |
| FR-27 | 3-tier routing (fast/mid/deep) | 29 |
| FR-28 | k-NN routing signal | 32 |
| FR-29 | Weighted-linear fusion alternative | 28 |
| FR-30 | Semantic caching | 33 |
| FR-31 | LangSmith tracing | 30 |
| FR-32 | Feedback capture | 31 |
| NFR-2 | End-to-end latency (DoD gate) | 27 (validated); depends on 12, 17, 22 |
| NFR-4 | ≤4 API calls/query (DoD gate) | 24 (tracked), 27 (validated) |
| NFR-5 | Recall@10 ≥ 0.85 (DoD gate) | 25 (measured), 27 (validated) |
| NFR-6 | Routing accuracy ≥ 80% (DoD gate) | 25 (measured), 27 (validated) |
| NFR-8 | Retry/backoff (DoD gate) | 11 (built); reused by 16, 20, 22 |
| NFR-9 | Degraded-mode fallback (DoD gate) | 11 (embedding half), 16 (router half), 22 (generation half) |
| NFR-11 | Observability fields (DoD gate) | 24 |
| NFR-12 | Scalability boundary documented (DoD gate) | 27 |
| NFR-13 | Gating latency <50ms (DoD gate) | 14 |

Step 27 is the actual "v1 done" line — everything before it is the DoD-gated core; everything in Phase 7 is additive on top of a system that has already been measured, not assumed, to meet spec.
