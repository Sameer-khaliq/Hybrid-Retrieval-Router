# Scalability Boundary

**This is not a production-scale system as configured.** Per NFR-12,
documenting this explicitly rather than leaving it for a client to
discover is a DoD gate, not an afterthought.

## Known limits

- **In-memory BM25**: the full corpus must fit in RAM. Fine for low
  tens of thousands of chunks; not enterprise document volumes.
  `scripts/build_demo_corpus.py` warns if the corpus crosses ~20k
  chunks, but the real ceiling depends on available machine RAM, not a
  fixed number this system enforces.
- **Single-node Qdrant** via docker-compose: no horizontal scaling, no
  replication, no failover. A container restart loses in-flight state
  (though the persistent volume preserves vector data across restarts).
- **Free-tier API rate limits** (Groq, Gemini, Tavily) cap real
  concurrent throughput far below what the architecture could
  theoretically support on paid infrastructure. Design target is
  single-digit concurrent users comfortably (NFR-3) - untested and
  unvalidated beyond that.
- **No horizontal scaling, multi-node deployment, or managed cloud
  infra** - explicitly out of scope for v1 (REQUIREMENTS.md §4.1).

## What this project demonstrates

The architecture and reasoning that *would* scale:
- A managed/clustered vector store (e.g. Qdrant Cloud, or a sharded
  self-hosted cluster) in place of single-node docker-compose Qdrant.
- Paid model tiers removing the free-tier rate-limit ceiling.
- A queue + horizontal API replicas in front of the FastAPI surface.
- Semantic caching (FR-30, Step 33) reducing redundant API calls at
  scale, once embedding-config stability is verified (Risk #5).

This is **not** a system that already scales - it's a system whose
design choices don't preclude scaling later, demonstrated at a scope
appropriate for a solo portfolio build under free-tier constraints.
