# RAG-Mem v0.1 — Implementation Plan

Date: 2026-05-29

## Goal

A standalone repository (own `.git`) at `source_code/rag-mem`. It ingests a
directory of Markdown (`.md`) files as a knowledge base and exposes **three
retrieval functions**: `semantic_search`, `bm25_search`, `graphrag_search`.
v0.1 is **read-only** retrieval over a fixed directory.

## Decisions (confirmed with the user)

- **Interface:** Python library + CLI.
- **Semantic:** OpenAI `text-embedding-3-small`; in-memory numpy vectors; cosine similarity.
- **BM25:** `bm25s`, hand-rolled index.
- **GraphRAG:** llama-index `PropertyGraphIndex`, **graphrag-only scope** — the
  framework is confined behind our facade and used solely for the graph
  retriever. Embedded **Kuzu** graph store. Extraction LLM Anthropic
  `claude-haiku-4-5`. OpenAI embeddings reused for the graph's vector retrieval.
- **One markdown-aware chunker** feeds all three retrievers. Our `Chunk` objects
  are handed to llama-index as pre-built `TextNode`s, so there is a single
  chunking system.
- **Persistence:** built artifacts cached under `<kb>/.ragmem`, keyed by a
  content hash of the source files; rebuild only when files change.

## Architecture

```
dir of .md ─► loader ─► markdown-aware chunker ─► 3 indexes ─► KnowledgeBase (3 funcs)
                                                  ├─ semantic  (OpenAI embed → numpy cosine top-k)
                                                  ├─ bm25       (bm25s lexical top-k)
                                                  └─ graphrag   (llama-index PropertyGraphIndex + Kuzu)
```

## Module responsibilities

- `types.py` — `Document`, `Chunk`, `SearchResult`.
- `loader.py` — `load_markdown(dir) -> list[Document]`. Recursive, `.md` only,
  utf-8, skips hidden dirs and `.ragmem`. Paths relative to the KB root.
- `chunker.py` — `chunk_document(doc) -> list[Chunk]`. Header-aware: splits on
  ATX headings, carries the heading breadcrumb + start/end lines + a stable id.
- `embeddings.py` — `make_embedder(client=None) -> Embedder` (DI OpenAI client).
  `embed_texts(list[str]) -> np.ndarray`, batched.
- `llm.py` — `make_extraction_llm(...)` returns a llama-index Anthropic LLM (DI).
- `index/semantic.py` — `SemanticIndex.build(chunks, embedder)` / `.search(query, k)`.
- `index/bm25.py` — `Bm25Index.build(chunks)` / `.search(query, k)`.
- `index/graphrag.py` — `GraphRagIndex` wraps `PropertyGraphIndex` over Kuzu;
  lazy-imports llama-index; `.build(chunks, llm, embed_model, store)` / `.search(query, k)`.
- `knowledge_base.py` — `KnowledgeBase.from_directory(path, ...)`; exposes
  `semantic_search` / `bm25_search` / `graphrag_search`. The public API.
- `persistence.py` — save/load artifacts; content-hash freshness check.
- `cli.py` — argparse subcommands: `build`, `search` (`--semantic`/`--bm25`/`--graphrag`, `-k`), `info`.

## TDD slices (one commit each; suite green at every boundary)

1. **bootstrap** — repo, config, types, deps.
2. **loader** — RED→GREEN.
3. **chunker** — header-aware.
4. **bm25** — first real index (no API).
5. **semantic** — mock OpenAI at the SDK boundary.
6. **graphrag** — mock the llama-index retriever boundary.
7. **knowledge_base** — facade exposing the 3 functions.
8. **persistence** — `.ragmem` cache.
9. **cli** — argparse subcommands.
10. **README live-smoke + polish**.

## Testing strategy

- pytest, **no live network**. Mock the OpenAI client (a `SimpleNamespace`
  shaped like `embeddings.create(...).data[i].embedding`). Mock the llama-index
  retriever for graphrag (return canned `NodeWithScore`). DI everywhere.
- Live smoke documented in the README as a manual step.

## Dependencies

`numpy`, `bm25s`, `openai`, `anthropic`, `llama-index-core`,
`llama-index-llms-anthropic`, `llama-index-embeddings-openai`,
`llama-index-graph-stores-kuzu`, `kuzu`; `pytest` (dev).

## Out of scope (v0.1)

- Writing / updating memory (read-only KB only).
- Non-`.md` inputs (PDF, web, code).
- Full GraphRAG community detection / global search.
- MCP server (CLI only; MCP can wrap the same facade later).
- Cross-retriever fusion / reranking.
