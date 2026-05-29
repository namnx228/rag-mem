# Project Memory: rag-mem

Standalone repo (own `.git`) at `source_code/rag-mem`. A RAG memory system over a
directory of Markdown files. Exposes three retrievers on `KnowledgeBase`:
`semantic_search`, `bm25_search`, `graphrag_search`. v0.1.

## Current Status
- **v0.2 slice 1 (2026-05-29): LanceDB vector store + tag filtering.** Semantic search
  swapped from in-memory NumPy cosine to an embedded, on-disk **LanceDB** table
  (`<kb>/.ragmem/vectors.lance`, cosine flat search); the NumPy `SemanticIndex` +
  `embeddings.npy` cache were removed. `persistence.py` now just holds a freshness
  manifest (`content_hash` + model + chunk_ids). `Chunk` gained a `tags` field and
  `semantic_search(query, k, tags=[...])` + CLI `--tag` filter results by tag (OR
  semantics, prefilter). 73 tests passing; pytest uses a **real** LanceDB on `tmp_path`
  (only OpenAI is mocked, like `bm25s`). Version 0.2.0. lancedb 0.33.0 + pyarrow 24.0.0
  installed (ARM aarch64 wheels). Tag *population* from source (front-matter) deferred.
- v0.1 complete. 66 tests passing; validated end-to-end via live smoke with real
  OpenAI + Anthropic on `tests/fixtures/mini_kb` — all three retrievers return
  correct, query-specific results.
- Built in 10 TDD slices, each its own commit. Library + CLI (`build`/`search`/
  `info`; `ragmem` console script and `python -m ragmem`).

## Key Decisions
- Semantic: OpenAI `text-embedding-3-small`, **embedded LanceDB vector table** (on-disk,
  cosine flat search, native tag filtering) — v0.2; replaced v0.1 in-memory numpy cosine.
  BM25: `bm25s` (local, tested for real). GraphRAG: llama-index `PropertyGraphIndex`
  over embedded **Kuzu**, extraction by Anthropic `claude-haiku-4-5`.
- v0.2 slice-1 calls (mine, flagged to user): tag filter = **OR** (chunk matches any
  requested tag), applied as a LanceDB **prefilter**; **flat/exact** cosine search (no
  ANN index yet); tag *population* from source deferred; LanceDB DI'd as a connection
  into `LanceSemanticIndex.build/open` so tests use a real local table.
- llama-index confined to `ragmem/index/graphrag.py` (lazy-imported); semantic +
  bm25 hand-rolled. One markdown-header chunker feeds all three.
- GraphRAG: Kuzu vector index **disabled** (`use_vector_index=False`,
  `embed_kg_nodes=False`) — avoids the "VECTOR_INDEX only supports FLOAT ARRAY"
  binder error and is cheaper. Retrieval = LLM synonym match → triples →
  attribute back to source chunks by entity text, filtered to query-relevant
  triples so results vary per query. Direct chunk_id path kept as fallback.
- Graph persisted under `<kb>/.ragmem/graph` with a content-hash marker; reopened
  via `PropertyGraphIndex.from_existing` while source unchanged (no
  re-extraction). Embedding matrix cached under `<kb>/.ragmem` too.
- Read-only KB for v0.1. Out of scope: writing/updating memory, non-`.md`
  inputs, community/global GraphRAG, MCP, cross-retriever fusion/reranking.

## Notes
- Installed: llama-index-core 0.14.22, kuzu 0.11.3, bm25s 0.3.9, openai 2.38.0,
  lancedb 0.33.0, pyarrow 24.0.0. Python 3.12; venv is **uv-managed** (no `pip` inside —
  install with `uv pip install --python venv/bin/python <pkg>`). Keys read from env
  (OPENAI_API_KEY, ANTHROPIC_API_KEY).
- Tests: mock OpenAI at the SDK boundary, graphrag at the llama-index retriever
  boundary; bm25 against the real lib. No network in pytest; live smoke is the
  only API-touching step (manual).
- Known v0.1 limitations / future work: re-enable Kuzu vector index (needs
  `embed_dimension`) for hybrid graph+vector retrieval; `search --bm25` still
  builds embeddings (facade builds semantic+bm25 together) — make per-retriever
  build lazy; incremental updates; MCP wrapper over the same facade.

## Design direction (v0.2 — decided 2026-05-29 with the user)
Guiding principle: **embedded only, no database service** to operate, ever.
- **Vectors: LanceDB** — DONE (v0.2 slice 1); replaced in-memory numpy — embedded, serverless, scales to
  millions on disk, native metadata/tag filtering. Chosen after rejecting (all
  verified online): pgvector (needs a Postgres service); Qdrant (embedded/local
  mode caps ~20k points, a few million needs the Qdrant *server*); sqlite-vec
  (brute-force only, slow past ~1M); vectorlite (beta, rowid-only filtering, index
  not stored in the SQLite file, SIMD off on ARM → 3–4× slower).
- **Lexical: keep bm25s** — real BM25 (Postgres native FTS is not BM25 / no IDF).
  Persist to disk + `mmap`; reindex **in batch or on explicit user request**, not
  live (bm25s has no incremental insert — full rebuild on change).
- **Graph: keep Kuzu** (embedded).
- **Conversations (future): SQLite** (embedded, no service).
- **Tag filtering** — DONE (v0.2 slice 1): `Chunk` has `tags`; `semantic_search` + CLI
  `--tag` filter via LanceDB metadata (OR, prefilter). Tag *population* from source
  (front-matter) is a later slice — chunks carry no tags until then.
- **Reranking** planned: over-retrieve → cross-encoder/LLM rerank → top-k, behind
  the facade (highest-leverage RAG quality gain).
- Scale target: a few million chunks. **User is on ARM (aarch64)** — relevant to
  any SIMD-sensitive choice.
- Migration shape: keep the `KnowledgeBase` facade; swap the semantic store
  (numpy → LanceDB) behind it so the 3 functions don't change.
