import numpy as np

from ragmem.index.semantic import SemanticIndex
from ragmem.types import Chunk


def _chunk(i, text, heading=()):
    return Chunk(
        id=f"d.md::{i}",
        doc_path="d.md",
        heading_path=heading,
        text=text,
        start_line=1,
        end_line=1,
    )


class FakeEmbedder:
    """Maps known text -> vector; KeyErrors on anything unexpected."""

    def __init__(self, table):
        self.table = table

    def embed_texts(self, texts):
        return np.asarray([self.table[t] for t in texts], dtype=np.float32)

    def embed_query(self, text):
        return np.asarray(self.table[text], dtype=np.float32)


def test_returns_most_similar_chunk_first():
    table = {
        "cat": [1.0, 0.0, 0.0],
        "dog": [0.5, 0.5, 0.0],
        "car": [0.0, 0.0, 1.0],
        "feline animal": [0.95, 0.05, 0.0],
    }
    chunks = [_chunk(0, "cat"), _chunk(1, "dog"), _chunk(2, "car")]
    results = SemanticIndex.build(chunks, FakeEmbedder(table)).search("feline animal", k=2)
    assert results[0].source == "semantic"
    assert [r.chunk.id for r in results] == ["d.md::0", "d.md::1"]


def test_scores_are_cosine_similarity():
    table = {"x": [1.0, 0.0], "q": [1.0, 0.0]}
    r = SemanticIndex.build([_chunk(0, "x")], FakeEmbedder(table)).search("q", k=1)[0]
    assert abs(r.score - 1.0) < 1e-6


def test_k_is_clamped_to_corpus_size():
    table = {"a": [1.0, 0.0], "q": [1.0, 0.0]}
    results = SemanticIndex.build([_chunk(0, "a")], FakeEmbedder(table)).search("q", k=5)
    assert len(results) == 1


def test_empty_corpus_returns_empty_without_embedding_query():
    # FakeEmbedder({}) would KeyError if the query were embedded.
    assert SemanticIndex.build([], FakeEmbedder({})).search("q", k=3) == []


def test_indexes_breadcrumb_text_not_just_body():
    table = {"Topic\n\nbody": [1.0, 0.0], "q": [1.0, 0.0]}
    chunk = _chunk(0, "body", heading=("Topic",))
    results = SemanticIndex.build([chunk], FakeEmbedder(table)).search("q", k=1)
    assert results[0].chunk.id == "d.md::0"
