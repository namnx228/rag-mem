# rag-mem

A small **RAG memory system** over a directory of Markdown files.

You point it at a folder of `.md` files and it exposes **three retrieval
functions** over that knowledge base:

| Function | What it does | Backend |
|---|---|---|
| `semantic_search` | Dense vector similarity (+ tag filter) | OpenAI `text-embedding-3-small` + embedded **LanceDB** |
| `bm25_search` | Lexical / keyword ranking | `bm25s` |
| `graphrag_search` | Entity-graph retrieval | llama-index `PropertyGraphIndex` + embedded **Kuzu** (extraction by Anthropic `claude-haiku-4-5`) |

Read-only retrieval over a fixed directory (`.md` only). v0.2 stores the vectors in an
embedded, on-disk **LanceDB** table and supports tag-filtered semantic search.

## Install

```bash
python3 -m venv venv
uv pip install --python venv/bin/python -e .        # or: venv/bin/pip install -e .
venv/bin/pip install pytest                          # dev
cp .env.example .env                                 # then fill in keys
```

Required keys (in `.env` or the environment):

- `OPENAI_API_KEY` — semantic embeddings.
- `ANTHROPIC_API_KEY` — GraphRAG entity/relation extraction.

## Library usage

```python
from ragmem import KnowledgeBase

kb = KnowledgeBase.from_directory("path/to/knowledge")   # loads, chunks, builds indexes

for r in kb.semantic_search("how does auth work?", k=5):
    print(f"{r.score:.3f}  {r.chunk.doc_path}  [{r.chunk.heading_breadcrumb}]")

kb.bm25_search("refresh token", k=5)
kb.graphrag_search("who owns the billing service?", k=5)

# Restrict semantic results to chunks carrying any of the given tags (OR):
kb.semantic_search("how does auth work?", k=5, tags=["billing", "auth"])
```

Each call returns a list of `SearchResult(chunk, score, source, detail)`.

## CLI

```bash
ragmem build  path/to/knowledge                       # build + cache the indexes
ragmem search path/to/knowledge --semantic "query" -k 5
ragmem search path/to/knowledge --semantic "query" --tag billing --tag auth  # tag-filtered
ragmem search path/to/knowledge --bm25     "query"
ragmem search path/to/knowledge --graphrag "query"
ragmem info   path/to/knowledge                       # docs / chunks / cache status
```

The built indexes are cached under `<kb>/.ragmem` and reused until the source
files change.

## Development

Strict TDD; official SDKs only; dependency injection everywhere; mock at the SDK
/ library boundary (never HTTP). See `CLAUDE.md` for the full conventions.

```bash
./venv/bin/pytest -v
```

No test touches the network. The live smoke below is the only step that calls
real APIs.

### Live smoke (manual — costs a few cents of API calls)

A small fixture knowledge base ships at `tests/fixtures/mini_kb` (a fictional
robotics company). `build` runs Haiku extraction + embeddings once and caches
everything under `tests/fixtures/mini_kb/.ragmem`:

```bash
ragmem build  tests/fixtures/mini_kb
ragmem search tests/fixtures/mini_kb --semantic "who runs the company?" -k 3
ragmem search tests/fixtures/mini_kb --bm25     "Beacon fleet controller" -k 3
ragmem search tests/fixtures/mini_kb --graphrag "Who leads the Pathfinder team?" -k 3
```

A search on an un-built KB builds (and caches) the indexes first, so `build`
once and then `search` repeatedly.

## How GraphRAG works (v0.1)

1. **Extract** — Anthropic `claude-haiku-4-5` extracts `subject -> relation ->
   object` triples from each chunk (llama-index `SimpleLLMPathExtractor`).
2. **Store** — triples go into an embedded **Kuzu** property graph under
   `<kb>/.ragmem/graph`, reused on the next build while the source is unchanged
   (no re-extraction).
3. **Retrieve** — the query's keywords are matched to graph entities; the
   connected relationships are attributed back to the source chunks whose text
   mentions the query-relevant entities, and chunks are ranked by matched score.

v0.1 is deliberately lightweight: no Kuzu vector index, and no community/global
summarization. Graph retrieval gets more selective as the knowledge base grows.

## Design decisions

**Guiding principle: embedded only — no database *service* to run, now or later.**
Every component is in-process / file-based, so the system is a library you import,
not infrastructure you operate.

| Concern | Choice | Status | Why |
|---|---|---|---|
| Vector / semantic | **LanceDB** (embedded, on-disk, flat search + metadata filtering) | **v0.2** | Serverless and scales to millions on a single node, with native tag filtering. Replaced the v0.1 in-memory NumPy cosine. (Flat/exact search for now; an ANN index is future work.) |
| Lexical | **bm25s** | v0.1 (disk persist planned) | Real BM25 — Postgres native FTS is *not* BM25 (no IDF). Persists to disk with `mmap`; reindexed **in batch or on explicit request**, not live. |
| Graph | **Kuzu** (embedded property graph) | v0.1 | Embedded "SQLite for graphs"; extraction by `claude-haiku-4-5`, persisted + reused. |
| Conversations | **SQLite** | planned | Embedded, no service — fits the principle. |
| Embeddings | OpenAI `text-embedding-3-small` | v0.1 | Quality/cost; the only hosted call on the read path. |

**Scale target:** a few million chunks.

**Why LanceDB for vectors** (rejected alternatives):
- **pgvector** — strong, but needs a **running Postgres service**. Rejected: no services.
- **Qdrant** — its embedded/local mode caps at ~20k points; a few million needs the **Qdrant server** (a service). Rejected for the same reason.
- **sqlite-vec** — truly one-SQLite-file, but **brute-force only** today → too slow past ~1M vectors.
- **vectorlite** — SQLite + HNSW, but beta, rowid-only filtering, the index lives *outside* the DB (so not really one file), and **SIMD is disabled on ARM (3–4× slower)**.
- **LanceDB** — the only option that is embedded *and* scales to millions *and* has first-class metadata/tag filtering.

**Tag filtering:** LanceDB filters vector results by metadata, so `semantic_search`
takes an optional `tags` argument and `Chunk` carries a `tags` field. A chunk matches if
it has *any* of the requested tags (OR), applied as a prefilter so the top-k is taken
over matching rows. Populating tags from the source (e.g. YAML front-matter) is a later
slice — chunks carry no tags until then.

**Reranking (planned):** a post-retrieval reranker (cross-encoder / LLM) behind the
facade — over-retrieve top-N → rerank → top-k — since reranking is the
highest-leverage RAG quality gain.

> **v0.2 (shipped):** LanceDB on-disk vectors + tag filtering, replacing the v0.1
> in-memory NumPy cosine. Still agreed but not yet implemented: reranking,
> disk-persisted `bm25s`, SQLite conversations, tag population from source, and an ANN
> index (flat/exact cosine search for now).

## Design notes

See `docs/superpowers/plans/2026-05-29-rag-mem-system-v0.1.md`.
