from ragmem.index.bm25 import Bm25Index
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


def test_ranks_lexically_relevant_chunk_first():
    chunks = [
        _chunk(0, "the cat sat on the mat"),
        _chunk(1, "python dataclasses and type hints"),
        _chunk(2, "kubernetes pods and deployments"),
    ]
    results = Bm25Index.build(chunks).search("python type hints", k=2)
    assert results[0].chunk.id == "d.md::1"
    assert results[0].source == "bm25"
    assert results[0].score > 0


def test_returns_empty_when_no_token_matches():
    idx = Bm25Index.build([_chunk(0, "alpha beta gamma")])
    assert idx.search("zzzznomatch", k=5) == []


def test_k_larger_than_corpus_is_clamped():
    idx = Bm25Index.build([_chunk(0, "alpha"), _chunk(1, "beta")])
    results = idx.search("alpha", k=10)
    assert [r.chunk.id for r in results] == ["d.md::0"]


def test_empty_corpus_search_returns_empty():
    assert Bm25Index.build([]).search("anything", k=5) == []


def test_more_term_occurrences_rank_higher():
    chunks = [
        _chunk(0, "apple apple apple"),
        _chunk(1, "apple"),
        _chunk(2, "completely unrelated text"),
    ]
    results = Bm25Index.build(chunks).search("apple", k=3)
    assert [r.chunk.id for r in results] == ["d.md::0", "d.md::1"]
    assert results[0].score >= results[1].score


def test_heading_breadcrumb_is_searchable():
    chunks = [
        _chunk(0, "no relevant words here", heading=("Installation", "Docker")),
        _chunk(1, "more unrelated content"),
    ]
    results = Bm25Index.build(chunks).search("docker", k=1)
    assert results[0].chunk.id == "d.md::0"
