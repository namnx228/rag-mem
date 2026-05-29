# rag-mem

A small **RAG memory system** over a directory of Markdown files.

You point it at a folder of `.md` files and it exposes **three retrieval
functions** over that knowledge base:

| Function | What it does | Backend |
|---|---|---|
| `semantic_search` | Dense vector similarity | OpenAI `text-embedding-3-small` + numpy cosine |
| `bm25_search` | Lexical / keyword ranking | `bm25s` |
| `graphrag_search` | Entity-graph retrieval | llama-index `PropertyGraphIndex` + embedded **Kuzu** (extraction by Anthropic `claude-haiku-4-5`) |

v0.1 is **read-only** retrieval over a fixed directory (`.md` only).

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
```

Each call returns a list of `SearchResult(chunk, score, source, detail)`.

## CLI

```bash
ragmem build  path/to/knowledge                       # build + cache the indexes
ragmem search path/to/knowledge --semantic "query" -k 5
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

### Live smoke (manual — costs API calls)

```bash
ragmem build tests/fixtures/mini_kb
ragmem search tests/fixtures/mini_kb --semantic "..." -k 3
ragmem search tests/fixtures/mini_kb --graphrag "..." -k 3
```

## Design

See `docs/superpowers/plans/2026-05-29-rag-mem-system-v0.1.md`.
