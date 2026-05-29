"""Split a Markdown Document into heading-delimited Chunks.

Each ATX heading (``#`` .. ``######``) starts a new chunk whose body is the text
directly under it, up to the next heading of any level. The chunk carries the
breadcrumb of enclosing headings (``heading_path``). Text before the first
heading becomes a preamble chunk with an empty breadcrumb. Sections whose body
is empty/whitespace-only are skipped. Headings inside fenced code blocks
(``` ``` ``` / ``~~~``) are ignored. Setext headings are not handled in v0.1.
"""

from __future__ import annotations

import re

from ragmem.types import Chunk, Document

# Up to 3 leading spaces, 1-6 hashes, required whitespace, title, optional
# trailing closing hashes (CommonMark ATX heading).
_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
# A fenced code block delimiter: >=3 backticks or tildes (up to 3 leading spaces).
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def chunk_document(doc: Document) -> list[Chunk]:
    """Split *doc* into a list of Chunks (see module docstring for the rules)."""
    lines = doc.text.split("\n")
    headings = _find_headings(lines)  # list of (line_index, level, title)

    chunks: list[Chunk] = []
    n = len(lines)
    heading_idxs = [h[0] for h in headings]

    # Preamble: everything before the first heading.
    first = heading_idxs[0] if headings else n
    _add_chunk(chunks, doc, (), lines[0:first], start_line=1, end_line=first)

    stack: list[tuple[int, str]] = []  # (level, title) of the enclosing headings
    for i, (h_idx, level, title) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        heading_path = tuple(t for _, t in stack)

        next_idx = heading_idxs[i + 1] if i + 1 < len(headings) else n
        _add_chunk(
            chunks,
            doc,
            heading_path,
            lines[h_idx + 1 : next_idx],
            start_line=h_idx + 1,
            end_line=next_idx,
        )

    return chunks


def _find_headings(lines: list[str]) -> list[tuple[int, int, str]]:
    headings: list[tuple[int, int, str]] = []
    in_fence = False
    fence_char = ""
    for idx, line in enumerate(lines):
        fence = _FENCE_RE.match(line)
        if fence:
            ch = fence.group(1)[0]
            if not in_fence:
                in_fence, fence_char = True, ch
                continue
            if ch == fence_char:
                in_fence, fence_char = False, ""
                continue
            # A different fence char while inside a fence is just content.
        if in_fence:
            continue
        m = _HEADING_RE.match(line)
        if m:
            headings.append((idx, len(m.group(1)), m.group(2).strip()))
    return headings


def _add_chunk(
    chunks: list[Chunk],
    doc: Document,
    heading_path: tuple[str, ...],
    body_lines: list[str],
    *,
    start_line: int,
    end_line: int,
) -> None:
    text = "\n".join(body_lines).strip()
    if not text:
        return
    chunks.append(
        Chunk(
            id=f"{doc.path}::{len(chunks)}",
            doc_path=doc.path,
            heading_path=heading_path,
            text=text,
            start_line=start_line,
            end_line=end_line,
        )
    )
