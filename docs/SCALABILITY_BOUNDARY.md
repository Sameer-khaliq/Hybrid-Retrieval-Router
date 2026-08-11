# Scalability Boundary (NFR-12)

Documented per NFR-12 -- these are the v1 boundaries by design, not gaps
meant to be discovered later by a client.

## In-memory BM25 (RAM ceiling)
<!-- TODO: measured RAM footprint at demo-corpus size, and the chunk-count
     point at which this stops being viable -->

## Single-node Qdrant (no HA / no replication)
<!-- TODO: what happens on container restart, and what the paid/production
     topology would look like -->

## Free-tier rate limits (the real concurrency ceiling)
<!-- TODO: Groq / Gemini / Tavily free-tier RPM & TPM limits, and how many
     concurrent demo users that actually supports -->

## What this would scale to on paid infrastructure
<!-- TODO: one paragraph per boundary above, naming the specific upgrade -->
