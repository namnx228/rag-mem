"""Dense semantic retrieval backed by an embedded, on-disk LanceDB table.

Replaces the v0.1 in-memory NumPy cosine index. LanceDB persists the vectors (and
the chunk metadata needed to reconstruct results) to disk, so the corpus need not
fit in memory and metadata/tag filtering is native.

The LanceDB *connection* is dependency-injected: ``build`` / ``open`` take a ``db``
handle (``lancedb.connect(...)`` in production, a real local DB on ``tmp_path`` in
tests). This module never opens a connection itself, and only the OpenAI embedder
touches the network — so it can be mocked at the SDK boundary like the old index.

Scoring: LanceDB cosine returns a *distance* (``1 - cosine_similarity``); we report
``1 - distance`` so callers keep the v0.1 cosine-*similarity* contract (1.0 == identical).
"""

from __future__ import annotations

from typing import Any, Sequence

import pyarrow as pa

from ragmem.types import Chunk, SearchResult

TABLE_NAME = "vectors"


class LanceSemanticIndex:
    def __init__(self, table: Any | None, embedder: Any):
        self._table = table  # None for an empty corpus / missing table
        self._embedder = embedder

    @property
    def is_empty(self) -> bool:
        """True when there is no backing table (empty corpus, or table absent)."""
        return self._table is None

    @classmethod
    def build(
        cls, chunks: list[Chunk], embedder: Any, db: Any, *, table_name: str = TABLE_NAME
    ) -> "LanceSemanticIndex":
        """Embed *chunks* and (over)write them into ``<db>/<table_name>``."""
        chunks = list(chunks)
        if not chunks:
            _drop_table(db, table_name)
            return cls(None, embedder)
        vectors = embedder.embed_texts([c.embedding_text for c in chunks])
        rows = [_to_row(chunk, vector) for chunk, vector in zip(chunks, vectors)]
        table = db.create_table(
            table_name, data=rows, schema=_vector_schema(int(vectors.shape[1])), mode="overwrite"
        )
        return cls(table, embedder)

    @classmethod
    def open(cls, db: Any, embedder: Any, *, table_name: str = TABLE_NAME) -> "LanceSemanticIndex":
        """Wrap the already-persisted table without re-embedding (None if absent)."""
        if table_name not in _table_names(db):
            return cls(None, embedder)
        return cls(db.open_table(table_name), embedder)

    def search(self, query: str, k: int = 5, tags: Sequence[str] | None = None) -> list[SearchResult]:
        if self._table is None or k <= 0:
            return []
        vector = [float(x) for x in self._embedder.embed_query(query)]
        search = self._table.search(vector).metric("cosine")
        if tags:
            search = search.where(_tags_filter(tags), prefilter=True)
        rows = search.limit(k).to_list()
        return [
            SearchResult(
                chunk=_row_to_chunk(row), score=1.0 - float(row["_distance"]), source="semantic"
            )
            for row in rows
        ]


def _vector_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("doc_path", pa.string()),
            pa.field("heading_path", pa.list_(pa.string())),
            pa.field("text", pa.string()),
            pa.field("start_line", pa.int64()),
            pa.field("end_line", pa.int64()),
            pa.field("tags", pa.list_(pa.string())),
        ]
    )


def _to_row(chunk: Chunk, vector: Any) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "vector": [float(x) for x in vector],
        "doc_path": chunk.doc_path,
        "heading_path": list(chunk.heading_path),
        "text": chunk.text,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "tags": list(chunk.tags),
    }


def _row_to_chunk(row: dict[str, Any]) -> Chunk:
    return Chunk(
        id=row["id"],
        doc_path=row["doc_path"],
        heading_path=tuple(row["heading_path"]),
        text=row["text"],
        start_line=int(row["start_line"]),
        end_line=int(row["end_line"]),
        tags=tuple(row["tags"]),
    )


def _tags_filter(tags: Sequence[str]) -> str:
    """A chunk matches if it carries *any* of the tags (OR semantics)."""
    return " OR ".join(f"array_has(tags, '{_escape(tag)}')" for tag in tags)


def _escape(value: str) -> str:
    return value.replace("'", "''")  # SQL single-quote escaping for the filter literal


def _table_names(db: Any) -> list[str]:
    # lancedb >=0.33 list_tables() returns a paginated result (``.tables`` + page token);
    # older builds / table_names() return a plain list. Normalize to a list of names.
    lister = getattr(db, "list_tables", None) or db.table_names
    result = lister()
    return list(getattr(result, "tables", result))


def _drop_table(db: Any, table_name: str) -> None:
    if table_name in _table_names(db):
        db.drop_table(table_name)
