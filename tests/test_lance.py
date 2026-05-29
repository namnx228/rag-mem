"""LanceSemanticIndex over a real (embedded, local) LanceDB table.

LanceDB is embedded — like ``bm25s`` — so these tests use a real table on
``tmp_path`` and mock only the OpenAI embedder (no network). Cosine similarity is
reported as ``1 - _distance`` so the score contract matches the old NumPy index.
"""

import lancedb
import numpy as np

from ragmem.index.lance import LanceSemanticIndex
from ragmem.types import Chunk


def _chunk(i, text, heading=(), tags=()):
    return Chunk(
        id=f"d.md::{i}",
        doc_path="d.md",
        heading_path=heading,
        text=text,
        start_line=1,
        end_line=1,
        tags=tags,
    )


class FakeEmbedder:
    """Maps known text -> vector; KeyErrors on anything unexpected."""

    model = "fake-model"

    def __init__(self, table):
        self.table = table

    def embed_texts(self, texts):
        return np.asarray([self.table[t] for t in texts], dtype=np.float32)

    def embed_query(self, text):
        return np.asarray(self.table[text], dtype=np.float32)


def _db(tmp_path, name=".ragmem"):
    return lancedb.connect(str(tmp_path / name))


def _animals_index(tmp_path):
    table = {
        "cat": [1.0, 0.0, 0.0],
        "dog": [0.5, 0.5, 0.0],
        "car": [0.0, 0.0, 1.0],
        "q": [1.0, 0.0, 0.0],
        "feline animal": [0.95, 0.05, 0.0],
    }
    chunks = [
        _chunk(0, "cat", tags=("animal",)),
        _chunk(1, "dog", tags=("animal", "pet")),
        _chunk(2, "car", tags=("vehicle",)),
    ]
    return LanceSemanticIndex.build(chunks, FakeEmbedder(table), _db(tmp_path))


def test_returns_most_similar_chunk_first(tmp_path):
    results = _animals_index(tmp_path).search("feline animal", k=2)
    assert results[0].source == "semantic"
    assert [r.chunk.id for r in results] == ["d.md::0", "d.md::1"]


def test_scores_are_cosine_similarity(tmp_path):
    table = {"x": [1.0, 0.0], "q": [1.0, 0.0]}
    index = LanceSemanticIndex.build([_chunk(0, "x")], FakeEmbedder(table), _db(tmp_path))
    assert abs(index.search("q", k=1)[0].score - 1.0) < 1e-5


def test_k_is_clamped_to_corpus_size(tmp_path):
    table = {"a": [1.0, 0.0], "q": [1.0, 0.0]}
    index = LanceSemanticIndex.build([_chunk(0, "a")], FakeEmbedder(table), _db(tmp_path))
    assert len(index.search("q", k=5)) == 1


def test_empty_corpus_returns_empty_without_embedding_query(tmp_path):
    # FakeEmbedder({}) would KeyError if the query were embedded.
    index = LanceSemanticIndex.build([], FakeEmbedder({}), _db(tmp_path))
    assert index.search("q", k=3) == []


def test_indexes_breadcrumb_text_not_just_body(tmp_path):
    table = {"Topic\n\nbody": [1.0, 0.0], "q": [1.0, 0.0]}
    chunk = _chunk(0, "body", heading=("Topic",))
    index = LanceSemanticIndex.build([chunk], FakeEmbedder(table), _db(tmp_path))
    results = index.search("q", k=1)
    assert results[0].chunk.id == "d.md::0"
    assert results[0].chunk.heading_path == ("Topic",)
    assert results[0].chunk.text == "body"


def test_tag_filter_restricts_results(tmp_path):
    results = _animals_index(tmp_path).search("q", k=5, tags=["pet"])
    assert [r.chunk.id for r in results] == ["d.md::1"]
    assert results[0].chunk.tags == ("animal", "pet")


def test_multiple_tags_use_or_semantics(tmp_path):
    results = _animals_index(tmp_path).search("q", k=5, tags=["pet", "vehicle"])
    assert {r.chunk.id for r in results} == {"d.md::1", "d.md::2"}
    assert results[0].chunk.id == "d.md::1"  # higher cosine to the query


def test_no_tags_returns_all_chunks(tmp_path):
    results = _animals_index(tmp_path).search("q", k=5)
    assert {r.chunk.id for r in results} == {"d.md::0", "d.md::1", "d.md::2"}


def test_persisted_table_reopens_without_rebuild(tmp_path):
    table = {"cat": [1.0, 0.0, 0.0], "q": [1.0, 0.0, 0.0]}
    LanceSemanticIndex.build([_chunk(0, "cat", tags=("animal",))], FakeEmbedder(table), _db(tmp_path))
    reopened = LanceSemanticIndex.open(_db(tmp_path), FakeEmbedder(table))
    results = reopened.search("q", k=1)
    assert results[0].chunk.id == "d.md::0"
    assert results[0].chunk.tags == ("animal",)


def test_open_missing_table_returns_empty(tmp_path):
    index = LanceSemanticIndex.open(_db(tmp_path, "fresh"), FakeEmbedder({}))
    assert index.search("q", k=3) == []
