import numpy as np
import pytest

from ragmem.knowledge_base import KnowledgeBase
from ragmem.types import Chunk


def _write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


class FakeEmbedder:
    """Deterministic bag-of-words embedder over a tiny fixed vocab (no network)."""

    VOCAB = ["alpha", "beta", "gamma", "delta"]

    def _vec(self, text):
        low = text.lower()
        return [float(low.count(w)) for w in self.VOCAB]

    def embed_texts(self, texts):
        return np.asarray([self._vec(t) for t in texts], dtype=np.float32)

    def embed_query(self, text):
        return np.asarray(self._vec(text), dtype=np.float32)


class StubIndex:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, k):
        self.calls.append((query, k))
        return self.results


def test_from_directory_builds_semantic_and_bm25(tmp_path):
    _write(tmp_path / "a.md", "# Alpha\nalpha alpha beta")
    _write(tmp_path / "b.md", "# Gamma\ngamma delta")
    kb = KnowledgeBase.from_directory(tmp_path, embedder=FakeEmbedder(), build_graphrag=False)

    sem = kb.semantic_search("alpha", k=1)
    assert sem[0].chunk.doc_path == "a.md"
    assert sem[0].source == "semantic"

    bm = kb.bm25_search("gamma", k=1)
    assert bm[0].chunk.doc_path == "b.md"
    assert bm[0].source == "bm25"


def test_graphrag_search_raises_when_not_built(tmp_path):
    _write(tmp_path / "a.md", "# A\nalpha")
    kb = KnowledgeBase.from_directory(tmp_path, embedder=FakeEmbedder(), build_graphrag=False)
    with pytest.raises(RuntimeError):
        kb.graphrag_search("alpha", k=1)


def test_each_method_delegates_to_its_index():
    sem, bm, gr = StubIndex(["S"]), StubIndex(["B"]), StubIndex(["G"])
    kb = KnowledgeBase([], semantic=sem, bm25=bm, graphrag=gr)

    assert kb.semantic_search("q1", k=3) == ["S"]
    assert kb.bm25_search("q2", k=4) == ["B"]
    assert kb.graphrag_search("q3", k=5) == ["G"]
    assert sem.calls == [("q1", 3)]
    assert bm.calls == [("q2", 4)]
    assert gr.calls == [("q3", 5)]


def test_chunks_property_exposes_loaded_chunks(tmp_path):
    _write(tmp_path / "a.md", "# A\nalpha\n## B\nbeta")
    kb = KnowledgeBase.from_directory(tmp_path, embedder=FakeEmbedder(), build_graphrag=False)
    chunks = kb.chunks
    assert len(chunks) == 2
    assert all(isinstance(c, Chunk) for c in chunks)


class CountingEmbedder(FakeEmbedder):
    model = "fake-model"

    def __init__(self):
        self.embed_texts_calls = 0

    def embed_texts(self, texts):
        self.embed_texts_calls += 1
        return super().embed_texts(texts)


def test_second_build_reuses_cached_embeddings(tmp_path):
    _write(tmp_path / "a.md", "# A\nalpha beta")
    e1 = CountingEmbedder()
    KnowledgeBase.from_directory(tmp_path, embedder=e1, build_graphrag=False)
    assert e1.embed_texts_calls == 1  # first build embeds + caches

    e2 = CountingEmbedder()
    kb2 = KnowledgeBase.from_directory(tmp_path, embedder=e2, build_graphrag=False)
    assert e2.embed_texts_calls == 0  # second build loads matrix from cache
    assert kb2.semantic_search("alpha", k=1)[0].chunk.doc_path == "a.md"


def test_cache_invalidated_when_markdown_changes(tmp_path):
    _write(tmp_path / "a.md", "# A\nalpha")
    KnowledgeBase.from_directory(tmp_path, embedder=CountingEmbedder(), build_graphrag=False)
    _write(tmp_path / "a.md", "# A\nbeta gamma")
    e2 = CountingEmbedder()
    KnowledgeBase.from_directory(tmp_path, embedder=e2, build_graphrag=False)
    assert e2.embed_texts_calls == 1  # content changed -> rebuild


def test_use_cache_false_always_rebuilds(tmp_path):
    _write(tmp_path / "a.md", "# A\nalpha")
    e1 = CountingEmbedder()
    KnowledgeBase.from_directory(tmp_path, embedder=e1, use_cache=False, build_graphrag=False)
    e2 = CountingEmbedder()
    KnowledgeBase.from_directory(tmp_path, embedder=e2, use_cache=False, build_graphrag=False)
    assert e1.embed_texts_calls == 1
    assert e2.embed_texts_calls == 1
