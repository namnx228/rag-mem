"""Unit tests for the GraphRAG retriever's custom logic.

We mock at the llama-index retriever boundary: a FakeRetriever returns real
``NodeWithScore`` objects, and we assert how GraphRagIndex maps them back to our
chunks. The real PropertyGraphIndex build/extraction is exercised in the manual
live smoke (see README), never in pytest.
"""

from llama_index.core.schema import NodeWithScore, TextNode

from ragmem.index.graphrag import GraphRagIndex, _chunks_to_nodes
from ragmem.types import Chunk


def _chunk(i, text="body", heading=()):
    return Chunk(
        id=f"d.md::{i}",
        doc_path="d.md",
        heading_path=heading,
        text=text,
        start_line=1,
        end_line=1,
    )


class FakeRetriever:
    def __init__(self, nodes):
        self._nodes = nodes
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return self._nodes


def _nws(chunk_id, score, *, with_metadata=True):
    metadata = {"chunk_id": chunk_id} if with_metadata else {}
    return NodeWithScore(node=TextNode(text="t", id_=chunk_id, metadata=metadata), score=score)


def test_maps_retrieved_nodes_to_source_chunks():
    retr = FakeRetriever([_nws("d.md::1", 0.9), _nws("d.md::0", 0.4)])
    results = GraphRagIndex.from_retriever([_chunk(0), _chunk(1)], retr).search("q", k=5)
    assert [r.chunk.id for r in results] == ["d.md::1", "d.md::0"]
    assert all(r.source == "graphrag" for r in results)
    assert retr.queries == ["q"]


def test_dedups_same_chunk_keeping_highest_score():
    retr = FakeRetriever([_nws("d.md::0", 0.3), _nws("d.md::0", 0.8)])
    results = GraphRagIndex.from_retriever([_chunk(0)], retr).search("q", k=5)
    assert len(results) == 1
    assert results[0].score == 0.8


def test_skips_nodes_without_a_known_chunk_id():
    retr = FakeRetriever([_nws("entity::Foo", 0.9), _nws("d.md::0", 0.5)])
    results = GraphRagIndex.from_retriever([_chunk(0)], retr).search("q", k=5)
    assert [r.chunk.id for r in results] == ["d.md::0"]


def test_truncates_to_k_by_score():
    nodes = [_nws(f"d.md::{i}", 1.0 - i * 0.1) for i in range(3)]
    results = GraphRagIndex.from_retriever([_chunk(i) for i in range(3)], FakeRetriever(nodes)).search("q", k=2)
    assert [r.chunk.id for r in results] == ["d.md::0", "d.md::1"]


def test_recovers_chunk_id_from_node_id_when_metadata_missing():
    retr = FakeRetriever([_nws("d.md::0", 0.7, with_metadata=False)])
    results = GraphRagIndex.from_retriever([_chunk(0)], retr).search("q", k=5)
    assert [r.chunk.id for r in results] == ["d.md::0"]


def test_score_none_is_treated_as_zero():
    retr = FakeRetriever([_nws("d.md::0", None)])
    results = GraphRagIndex.from_retriever([_chunk(0)], retr).search("q", k=5)
    assert results[0].score == 0.0


def test_empty_corpus_returns_empty_without_retrieving():
    retr = FakeRetriever([_nws("d.md::0", 1.0)])
    assert GraphRagIndex.from_retriever([], retr).search("q", k=5) == []
    assert retr.queries == []


def test_chunks_to_nodes_carry_chunk_id_and_breadcrumb_text():
    nodes = _chunks_to_nodes([_chunk(0, text="hello", heading=("H",))])
    assert len(nodes) == 1
    assert nodes[0].metadata["chunk_id"] == "d.md::0"
    assert "hello" in nodes[0].text and "H" in nodes[0].text
    # chunk_id must not leak into the text fed to the LLM / embeddings
    assert "chunk_id" in nodes[0].excluded_embed_metadata_keys
    assert "chunk_id" in nodes[0].excluded_llm_metadata_keys
