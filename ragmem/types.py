"""Core data types shared across the loader, chunker, and the three retrievers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """A single Markdown file loaded from the knowledge base."""

    path: str  # path relative to the knowledge base root, e.g. "notes/intro.md"
    text: str  # raw file contents (utf-8)


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text extracted from a Document.

    A chunk is a heading-delimited section of a Markdown document. The same
    chunks feed all three retrievers (semantic, bm25, graphrag).
    """

    id: str  # stable id, e.g. "notes/intro.md::0"
    doc_path: str  # path relative to the knowledge base root
    heading_path: tuple[str, ...]  # breadcrumb of headings, outermost first
    text: str  # the section body (heading line excluded)
    start_line: int  # 1-based line in the source document where the chunk starts
    end_line: int  # 1-based inclusive end line
    tags: tuple[str, ...] = ()  # metadata labels for filtered semantic search

    @property
    def heading_breadcrumb(self) -> str:
        """Human-readable heading trail, e.g. 'Intro > Setup > Install'."""
        return " > ".join(self.heading_path)

    @property
    def embedding_text(self) -> str:
        """Text used for embedding / indexing: breadcrumb prepended to body.

        Prepending the heading trail gives lexical and semantic retrievers the
        section context that the body alone may omit.
        """
        if self.heading_path:
            return f"{self.heading_breadcrumb}\n\n{self.text}".strip()
        return self.text.strip()


@dataclass(frozen=True)
class SearchResult:
    """A scored chunk returned by any of the three retrievers."""

    chunk: Chunk
    score: float
    source: str  # "semantic" | "bm25" | "graphrag"
    detail: dict[str, Any] = field(default_factory=dict)  # retriever-specific extras
