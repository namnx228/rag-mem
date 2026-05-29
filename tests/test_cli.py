import pytest

from ragmem.cli import format_results, main
from ragmem.types import Chunk, SearchResult


def _result(doc, score, heading=(), text="body", source="semantic"):
    chunk = Chunk(
        id=f"{doc}::0", doc_path=doc, heading_path=heading, text=text, start_line=1, end_line=2
    )
    return SearchResult(chunk=chunk, score=score, source=source)


class FakeKB:
    def __init__(self, results, chunks=None):
        self._results = results
        self._chunks = chunks or []
        self.calls = {}

    def semantic_search(self, q, k, tags=None):
        self.calls["semantic"] = (q, k)
        self.semantic_tags = tags
        return self._results

    def bm25_search(self, q, k):
        self.calls["bm25"] = (q, k)
        return self._results

    def graphrag_search(self, q, k):
        self.calls["graphrag"] = (q, k)
        return self._results

    @property
    def chunks(self):
        return self._chunks


# ---- format_results ----------------------------------------------------------

def test_format_results_includes_score_path_and_breadcrumb():
    out = format_results([_result("a.md", 0.4242, heading=("H", "S"))])
    assert "0.424" in out
    assert "a.md" in out
    assert "H > S" in out


def test_format_results_empty():
    assert "no results" in format_results([]).lower()


# ---- search ------------------------------------------------------------------

def test_search_semantic_dispatches_and_prints(capsys):
    kb = FakeKB([_result("a.md", 0.9)])
    rc = main(["search", "kb", "--semantic", "my query", "-k", "3"], kb_factory=lambda *a, **k: kb)
    assert rc == 0
    assert kb.calls["semantic"] == ("my query", 3)
    assert "a.md" in capsys.readouterr().out


def test_search_semantic_passes_tags():
    kb = FakeKB([_result("a.md", 0.9)])
    main(
        ["search", "kb", "--semantic", "q", "--tag", "billing", "--tag", "auth"],
        kb_factory=lambda *a, **k: kb,
    )
    assert kb.semantic_tags == ["billing", "auth"]


def test_search_without_tag_passes_none():
    kb = FakeKB([_result("a.md", 0.9)])
    main(["search", "kb", "--semantic", "q"], kb_factory=lambda *a, **k: kb)
    assert kb.semantic_tags is None


def test_search_bm25_uses_default_k():
    kb = FakeKB([_result("b.md", 1.0, source="bm25")])
    main(["search", "kb", "--bm25", "q"], kb_factory=lambda *a, **k: kb)
    assert kb.calls["bm25"] == ("q", 5)


def test_search_graphrag_requests_graph_build():
    captured = {}

    def factory(path, **kw):
        captured.update(kw)
        return FakeKB([_result("c.md", 0.5, source="graphrag")])

    main(["search", "kb", "--graphrag", "q"], kb_factory=factory)
    assert captured["build_graphrag"] is True


def test_search_nongraph_skips_graph_build():
    captured = {}

    def factory(path, **kw):
        captured.update(kw)
        return FakeKB([])

    main(["search", "kb", "--bm25", "q"], kb_factory=factory)
    assert captured["build_graphrag"] is False


def test_search_requires_a_mode_flag():
    with pytest.raises(SystemExit):
        main(["search", "kb", "q"], kb_factory=lambda *a, **k: FakeKB([]))


# ---- build -------------------------------------------------------------------

def test_build_reports_chunk_count(capsys):
    kb = FakeKB([], chunks=[_result("a.md", 0).chunk, _result("b.md", 0).chunk])
    rc = main(["build", "kb"], kb_factory=lambda *a, **k: kb)
    assert rc == 0
    assert "2 chunks" in capsys.readouterr().out


def test_build_default_builds_graphrag():
    captured = {}

    def factory(path, **kw):
        captured.update(kw)
        return FakeKB([], chunks=[])

    main(["build", "kb"], kb_factory=factory)
    assert captured["build_graphrag"] is True


def test_build_no_graphrag_flag():
    captured = {}

    def factory(path, **kw):
        captured.update(kw)
        return FakeKB([], chunks=[])

    main(["build", "kb", "--no-graphrag"], kb_factory=factory)
    assert captured["build_graphrag"] is False


# ---- info (no network, real files) ------------------------------------------

def test_info_reports_counts_and_absent_cache(tmp_path, capsys):
    (tmp_path / "a.md").write_text("# A\nalpha\n## B\nbeta", encoding="utf-8")
    rc = main(["info", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Documents: 1" in out
    assert "Chunks: 2" in out
    assert "absent" in out
