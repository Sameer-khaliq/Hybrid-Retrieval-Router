# Hybrid Retrieval & Query Routing System

Portfolio project: hybrid (BM25 + dense) retrieval with RRF fusion, local
cross-encoder reranking, and a two-layer complexity-based query router,
built solo on free-tier APIs (Groq, Gemini, Tavily) with a self-hosted
Qdrant vector store.

See `IMPLEMENTATION_PLAN.md` for the full dependency-ordered build sequence
and `docs/SCALABILITY_BOUNDARY.md` for the documented v1 limits (NFR-12).

**Status:** scaffolded, not yet built.
