from pathlib import Path

import pytest

from ragmem.loader import load_markdown
from ragmem.types import Document


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_loads_md_files_recursively(tmp_path: Path):
    _write(tmp_path / "a.md", "# A\nalpha")
    _write(tmp_path / "sub" / "b.md", "# B\nbeta")

    docs = load_markdown(tmp_path)

    assert [d.path for d in docs] == ["a.md", "sub/b.md"]
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].text == "# A\nalpha"


def test_ignores_non_md_files(tmp_path: Path):
    _write(tmp_path / "keep.md", "# keep")
    _write(tmp_path / "skip.txt", "nope")
    _write(tmp_path / "skip.py", "print()")

    docs = load_markdown(tmp_path)

    assert [d.path for d in docs] == ["keep.md"]


def test_md_match_is_case_insensitive(tmp_path: Path):
    _write(tmp_path / "upper.MD", "# upper")

    docs = load_markdown(tmp_path)

    assert [d.path for d in docs] == ["upper.MD"]


def test_skips_hidden_and_cache_dirs(tmp_path: Path):
    _write(tmp_path / "visible.md", "# v")
    _write(tmp_path / ".git" / "config.md", "# hidden")
    _write(tmp_path / ".ragmem" / "cache.md", "# cache")

    docs = load_markdown(tmp_path)

    assert [d.path for d in docs] == ["visible.md"]


def test_results_are_sorted_by_path(tmp_path: Path):
    for name in ["c.md", "a.md", "b.md"]:
        _write(tmp_path / name, name)

    docs = load_markdown(tmp_path)

    assert [d.path for d in docs] == ["a.md", "b.md", "c.md"]


def test_empty_dir_returns_empty_list(tmp_path: Path):
    assert load_markdown(tmp_path) == []


def test_missing_dir_raises(tmp_path: Path):
    with pytest.raises((FileNotFoundError, NotADirectoryError)):
        load_markdown(tmp_path / "does-not-exist")
