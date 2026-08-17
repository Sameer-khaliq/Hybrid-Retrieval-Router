# Notes on Hybrid Retrieval Systems

Retrieval systems that answer questions over a document collection generally
fall into two broad families. The first family relies on lexical matching,
where a query and a candidate passage are compared based on shared words or
subword units. The second family relies on dense vector representations,
where both the query and the passage are embedded into a shared vector space
and compared by distance or similarity in that space. Each family has
strengths the other lacks, which is exactly why combining them tends to
outperform either one alone.

## Lexical retrieval and BM25

Lexical retrieval methods, of which BM25 is the most widely deployed, score
a passage based on how often query terms appear in it, adjusted for how rare
those terms are across the whole collection and how long the passage is.
BM25 is fast, requires no training, and handles rare or specific terms very
well - a product code, a proper noun, or an exact phrase will usually be
found reliably by a lexical method even when no training data mentions it.
The weakness is that BM25 has no notion of meaning: a query about a "car"
will not match a passage that only says "automobile" unless both words
happen to appear together somewhere in the collection.

## Dense retrieval and embeddings

Dense retrieval addresses exactly that weakness. An embedding model maps
text into a continuous vector space such that semantically related text ends
up close together, regardless of whether the exact same words were used.
This lets a dense retriever find a passage about "automobile maintenance"
even when the query says "car repair." The cost is that dense retrieval can
be less precise on queries containing specific identifiers, unusual proper
nouns, or exact phrases the embedding model was never trained to distinguish
finely, and it requires an embedding call for every query and every document
at ingestion time, which adds latency and, for hosted embedding APIs, cost.

## Why fuse the two

Because BM25 and dense retrieval fail in different, largely uncorrelated
ways, fusing their outputs tends to recover documents that either method
alone would have missed. A common fusion technique is Reciprocal Rank
Fusion, which combines two ranked lists by scoring each document according
to the inverse of its rank in each list, rather than trying to normalize and
compare raw scores that are not on the same scale to begin with. This makes
fusion robust even when the two underlying methods produce scores with very
different distributions.

## Reranking as a refinement stage

After fusion produces a single merged ranking, a cross-encoder reranker can
be applied to the top candidates. Unlike a dense retriever, which embeds the
query and each passage independently and compares the resulting vectors, a
cross-encoder processes the query and a candidate passage together in a
single forward pass, letting it model fine-grained interactions between the
two. This tends to produce a substantially more accurate ranking than either
retrieval method alone, but it is far more computationally expensive per
candidate, since the model must run once per candidate passage rather than
once per query. For that reason, cross-encoder reranking is almost always
restricted to a small number of top candidates rather than the full
retrieved set.

## Routing retrieval depth by query complexity

Not every query benefits equally from the expensive reranking stage. A
short factual query with a single unambiguous answer is often already well
served by the fused top few results, while a query that requires comparing
multiple entities or reasoning across several pieces of context benefits
much more from reranking, and potentially from a more capable generation
model as well. A router that estimates query complexity before retrieval
completes can therefore decide, per query, whether to spend the extra
latency on reranking and a stronger model, or to skip straight to a fast
answer. This adaptive behavior is what allows a system to keep average
latency low without sacrificing answer quality on the harder queries that
actually need the extra computation.

## Handling low-confidence retrieval

Even a well-tuned hybrid retrieval pipeline will sometimes fail to find
anything genuinely relevant in the underlying document collection, either
because the collection does not cover the topic at all or because the query
asks about something that has changed since the collection was last
updated. In those situations, forcing a generation model to answer from
weak or irrelevant retrieved context tends to produce a confident-sounding
but incorrect answer, which is often worse than not answering at all. A
practical mitigation is to monitor the confidence of the fused retrieval
results and fall back to an external, up-to-date source when that
confidence falls below a configured floor, tagging results by their origin
so the final answer can be transparent about whether it came from the
internal collection or from an external lookup.

## Operational considerations

Beyond the core retrieval logic, a production system has to account for
failure modes that never show up in a clean benchmark. Embedding calls to a
hosted API can fail or time out, in which case a system that depends on
dense retrieval for every query will produce no results at all unless it
has a fallback path. Sparse indexes built with in-memory libraries need to
be persisted to disk and reloaded on restart, or every deployment restart
becomes a full reindex from scratch. Configuration values that are pinned at
ingestion time - most notably the embedding model version and its output
dimensionality - must stay consistent with the configuration used at query
time, since a silent mismatch does not raise an error, it just makes every
similarity score meaningless without any visible symptom. None of these
concerns are exotic, but each one is easy to overlook until it causes a
failure in front of a live audience rather than in a test suite.