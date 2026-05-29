# Project Memory: rag-mem

Standalone repo (own `.git`) at `source_code/rag-mem`. A RAG memory system over a
directory of Markdown files. Exposes three retrievers on `KnowledgeBase`:
`semantic_search`, `bm25_search`, `graphrag_search`. v0.1.

## Current Status
- v0.1 complete. 66 tests passing; validated end-to-end via live smoke with real
  OpenAI + Anthropic on `tests/fixtures/mini_kb` — all three retrievers return
  correct, query-specific results.
- Built in 10 TDD slices, each its own commit. Library + CLI (`build`/`search`/
  `info`; `ragmem` console script and `python -m ragmem`).

## Key Decisions
- Semantic: OpenAI `text-embedding-3-small`, in-memory numpy cosine. BM25:
  `bm25s` (local, tested for real). GraphRAG: llama-index `PropertyGraphIndex`
  over embedded **Kuzu**, extraction by Anthropic `claude-haiku-4-5`.
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
- Installed: llama-index-core 0.14.22, kuzu 0.11.3, bm25s 0.3.9, openai 2.38.0.
  Python 3.12; venv at `venv/`. Keys read from env (OPENAI_API_KEY,
  ANTHROPIC_API_KEY).
- Tests: mock OpenAI at the SDK boundary, graphrag at the llama-index retriever
  boundary; bm25 against the real lib. No network in pytest; live smoke is the
  only API-touching step (manual).
- Known v0.1 limitations / future work: re-enable Kuzu vector index (needs
  `embed_dimension`) for hybrid graph+vector retrieval; `search --bm25` still
  builds embeddings (facade builds semantic+bm25 together) — make per-retriever
  build lazy; incremental updates; MCP wrapper over the same facade.
