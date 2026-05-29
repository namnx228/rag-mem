"""rag-mem: a RAG memory system over a directory of Markdown files."""

from __future__ import annotations

from ragmem.knowledge_base import KnowledgeBase
from ragmem.types import Chunk, Document, SearchResult

__all__ = ["Chunk", "Document", "KnowledgeBase", "SearchResult"]
__version__ = "0.2.0"
