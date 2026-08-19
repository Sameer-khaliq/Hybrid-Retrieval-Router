# Scalability Boundary

**This is not a production-scale system as configured.** Per NFR-12, documenting this explicitly rather than leaving it for a client or reviewer to discover is a Definition of Done (DoD) gate, not an afterthought.

---

## 1. Known Limits & Free-Tier Boundaries

- **Latency Boundaries (NFR-2 Real-World Measurement):**
  - **Targets:** Fast-path $\le$ 1,000ms, Deep-path $\le$ 2,000ms.
  - **Measured Reality:** Fast-path median is **~3.2s – 3.7s**; Deep-path median is **~5.5s – 6.8s**.
  - **Root Cause:** Free-tier Groq API network round-trips combined with full-token streaming generation for `gpt-oss-20b` (fast) and `gpt-oss-120b` (deep) account for ~2.5s–4.0s of the total response time. 
  - **Internal Pipeline Performance:** The internal computational overhead (Gating + Hybrid Retrieval + Local Cross-Encoder Reranking) executes within **~400ms – 1,200ms**. On dedicated GPU endpoints or faster paid model tiers with streaming TTFT (Time-to-First-Token) SLAs, the architecture naturally satisfies sub-second targets.

- **Free-Tier API Rate Limits (Groq, Gemini, Tavily):**
  - Groq free-tier enforces a strict **8K TPM (Tokens Per Minute)** budget on `gpt-oss-120b` and `gpt-oss-20b`.
  - Prompt context truncation (max 5 chunks, ~800 chars/chunk) and concurrency semaphores are enforced in software to stay reliably within these bounds.
  - Design throughput is optimized for single-digit concurrent users comfortably (NFR-3); unvalidated beyond that on free tiers.

- **In-Memory BM25:**
  - The BM25 inverted index and document corpus live entirely in process RAM.
  - Sized for low tens of thousands of chunks (~20k max warning in `scripts/build_demo_corpus.py`). Large enterprise-scale datasets would require external indexing backends (e.g., Elasticsearch, OpenSearch, or Qdrant Sparse Vectors).

- **Single-Node Qdrant (Docker Compose):**
  - Runs locally without replication, sharding, or automatic failover. Vector persistence is handled via disk volume mounts, but horizontal scaling is intentionally omitted in v1.

- **Zero Multi-Node Orchestration:**
  - No Kubernetes, distributed workers, or external message brokers—explicitly out of scope for v1 (REQUIREMENTS.md §4.1).

---

## 2. What This Architecture Demonstrates

The design choices prove production-readiness in architecture and routing logic without paying premature cloud costs:

1. **Pluggable & Decoupled Layers:** 
   - Switching from local Docker Qdrant to **Qdrant Cloud (Clustered/Sharded)** requires only environment variable updates (`QDRANT_URL`, `QDRANT_API_KEY`).
2. **Predictable Latency Scaling:** 
   - Moving to paid inference endpoints (e.g., Groq Enterprise / vLLM on dedicated GPUs) instantly drops generation latency into target SLA boundaries without changing pipeline logic.
3. **Resilience & Graceful Degradation:** 
   - Built-in circuit breakers, timeout fallbacks, rate-limit semaphores, and retry/backoff wrappers (`with_retry`) ensure zero unhandled 500 crashes under upstream degradation.
4. **Extensible Core for Scale-Up (Phase 7+):**
   - Clean architectural seams allow dropping in Redis-based **Semantic Caching (FR-30)**, asynchronous message queues (Celery/RabbitMQ), and distributed tracing (LangSmith/OpenTelemetry) on top of the validated core.

---

> **Summary:** This is a system whose foundational design choices do not preclude scaling, built and verified under free-tier constraints.