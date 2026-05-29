"""Command-line interface for rag-mem: ``build``, ``search``, ``info``.

``main`` takes an injectable ``kb_factory`` (defaulting to
``KnowledgeBase.from_directory``) so dispatch and output formatting are tested
without touching the network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from ragmem.chunker import chunk_document
from ragmem.knowledge_base import KnowledgeBase
from ragmem.loader import load_markdown
from ragmem.persistence import content_hash
from ragmem.types import SearchResult

_SNIPPET_LEN = 140


def main(argv: Sequence[str] | None = None, *, kb_factory: Callable[..., Any] | None = None) -> int:
    kb_factory = kb_factory or KnowledgeBase.from_directory
    args = build_parser().parse_args(argv)
    if args.command == "build":
        return _cmd_build(args, kb_factory)
    if args.command == "search":
        return _cmd_search(args, kb_factory)
    return _cmd_info(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragmem", description="RAG memory over a directory of Markdown files."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build and cache the indexes for a knowledge base.")
    build.add_argument("kb_dir")
    build.add_argument(
        "--no-graphrag", action="store_true", help="Skip GraphRAG (no Anthropic calls)."
    )
    build.add_argument("--persist-dir", default=None)

    search = sub.add_parser("search", help="Search a knowledge base with one retriever.")
    search.add_argument("kb_dir")
    search.add_argument("query")
    mode = search.add_mutually_exclusive_group(required=True)
    mode.add_argument("--semantic", action="store_true", help="Dense embedding search.")
    mode.add_argument("--bm25", action="store_true", help="Lexical BM25 search.")
    mode.add_argument("--graphrag", action="store_true", help="Entity-graph search.")
    search.add_argument("-k", "--top-k", type=int, default=5, dest="top_k")
    search.add_argument(
        "--tag",
        action="append",
        dest="tags",
        metavar="TAG",
        help="Restrict semantic results to chunks carrying any of these tags (repeatable).",
    )
    search.add_argument("--persist-dir", default=None)

    info = sub.add_parser("info", help="Show document/chunk counts and cache status.")
    info.add_argument("kb_dir")
    info.add_argument("--persist-dir", default=None)
    return parser


def format_results(results: list[SearchResult]) -> str:
    if not results:
        return "No results."
    lines: list[str] = []
    for rank, result in enumerate(results, 1):
        chunk = result.chunk
        crumb = chunk.heading_breadcrumb or "(no heading)"
        lines.append(f"{rank}. [{result.score:.3f}] {chunk.doc_path}:{chunk.start_line}  {crumb}")
        lines.append(f"    {_snippet(chunk.text)}")
    return "\n".join(lines)


def _cmd_build(args: argparse.Namespace, kb_factory: Callable[..., Any]) -> int:
    kb = kb_factory(
        args.kb_dir, build_graphrag=not args.no_graphrag, persist_dir=args.persist_dir
    )
    graph = "off" if args.no_graphrag else "on"
    print(f"Built knowledge base at {args.kb_dir}: {len(kb.chunks)} chunks (graphrag: {graph}).")
    return 0


def _cmd_search(args: argparse.Namespace, kb_factory: Callable[..., Any]) -> int:
    mode = "semantic" if args.semantic else "bm25" if args.bm25 else "graphrag"
    kb = kb_factory(
        args.kb_dir, build_graphrag=(mode == "graphrag"), persist_dir=args.persist_dir
    )
    if mode == "semantic":
        results = kb.semantic_search(args.query, args.top_k, tags=args.tags)
    else:
        results = getattr(kb, f"{mode}_search")(args.query, args.top_k)
    print(format_results(results))
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    docs = load_markdown(args.kb_dir)
    chunks = [chunk for doc in docs for chunk in chunk_document(doc)]
    persist_dir = Path(args.persist_dir) if args.persist_dir else Path(args.kb_dir) / ".ragmem"
    print(f"Knowledge base: {args.kb_dir}")
    print(f"Documents: {len(docs)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding cache: {_cache_status(persist_dir, chunks)}")
    return 0


def _cache_status(persist_dir: Path, chunks: list) -> str:
    manifest = persist_dir / "manifest.json"
    if not manifest.exists():
        return "absent"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return "fresh" if data.get("content_hash") == content_hash(chunks) else "stale"


def _snippet(text: str, length: int = _SNIPPET_LEN) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= length else collapsed[: length - 1] + "…"


if __name__ == "__main__":
    raise SystemExit(main())
