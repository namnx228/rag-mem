# CLAUDE.md — rag-mem

RAG memory system over a directory of Markdown files. Exposes three retrievers on a
`KnowledgeBase`: `semantic_search` (LanceDB, tag-filterable), `bm25_search`, `graphrag_search`.

## Iron rules
- **TDD always.** Failing pytest → confirm RED → minimal impl → green → refactor. No production code before a red test.
- **Official SDKs only.** `openai`, `anthropic`, and the `llama-index-*` integration packages. No raw HTTP to any LLM/embedding provider.
- **DI every external client.** Factories take the client(s) as args and default to building from env. Tests pass `MagicMock` / `SimpleNamespace`.
- **Mock at the SDK / library boundary, not HTTP.** Mock the OpenAI client's `embeddings.create` shape; mock the llama-index retriever for GraphRAG. Never patch `httpx`.
- **No live network in pytest.** Live smoke is a separate manual step in the README.
- **llama-index is confined to GraphRAG** (`ragmem/index/graphrag.py`), imported lazily. Semantic and BM25 stay hand-rolled.

## Layout
- Source in `ragmem/`, tests mirror it in `tests/`. `pythonpath = ["."]` — `from ragmem.types import Chunk` works anywhere.
- venv at `venv/`. Use `./venv/bin/python` and `./venv/bin/pytest`.
- Never commit `venv/`, `.env`, `.ragmem/`, `.pytest_cache/`.

## Providers
- Semantic embeddings: OpenAI `text-embedding-3-small`, stored in an embedded **LanceDB** table on disk (`<kb>/.ragmem/vectors.lance`); cosine flat search, native tag filtering. `manifest.json` (content hash + model) tracks freshness so re-embedding only happens when the source/chunking/model changes. LanceDB + `pyarrow` are local libraries (no service), mocked-free in tests by using a real table on `tmp_path`.
- BM25: `bm25s` (lexical).
- GraphRAG: llama-index `PropertyGraphIndex` over an embedded **Kuzu** graph store; extraction LLM Anthropic `claude-haiku-4-5`, embeddings reuse OpenAI.

## Commands
- Test: `./venv/bin/pytest -v`
- CLI: `./venv/bin/ragmem build <kb_dir>` ; `./venv/bin/ragmem search <kb_dir> --semantic "query" -k 5`

## Commits
- Format `<tag>: description`. Tags: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `setup`. **No Claude co-author.**
- Small red→green→refactor commits; suite green at every boundary.

## Plans
- `docs/superpowers/plans/YYYY-MM-DD-<kebab-topic>.md`.

## Out of scope (v0.1)
- Writing/updating memory (read-only KB), non-`.md` inputs, full GraphRAG community/global search, MCP server, cross-retriever fusion/reranking.
