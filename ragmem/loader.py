"""Load a directory of Markdown files into Document objects."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ragmem.types import Document

_MD_SUFFIX = ".md"


def load_markdown(root: str | Path) -> list[Document]:
    """Recursively load all Markdown (`.md`) files under *root*.

    - Recurses into subdirectories, skipping hidden dirs (name starts with "."),
      which includes the `.ragmem` cache dir and `.git`.
    - Matches the `.md` suffix case-insensitively.
    - Reads files as utf-8.
    - Returns Documents sorted by their path relative to *root* (posix style).

    Raises ``FileNotFoundError`` if *root* does not exist and
    ``NotADirectoryError`` if it exists but is not a directory.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    docs = [
        Document(path=path.relative_to(root).as_posix(), text=path.read_text(encoding="utf-8"))
        for path in _iter_markdown_files(root)
    ]
    docs.sort(key=lambda d: d.path)
    return docs


def _iter_markdown_files(directory: Path) -> Iterator[Path]:
    for entry in directory.iterdir():
        if entry.is_dir():
            if entry.name.startswith("."):
                continue
            yield from _iter_markdown_files(entry)
        elif entry.is_file() and entry.suffix.lower() == _MD_SUFFIX:
            yield entry
