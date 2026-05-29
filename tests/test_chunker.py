from ragmem.chunker import chunk_document
from ragmem.types import Document


def _summ(chunks):
    return [(c.heading_path, c.text) for c in chunks]


def test_splits_on_headings_with_breadcrumb():
    doc = Document("x.md", "# Title\nintro text\n## Section\nsection body\n")
    chunks = chunk_document(doc)
    assert _summ(chunks) == [
        (("Title",), "intro text"),
        (("Title", "Section"), "section body"),
    ]


def test_preamble_before_first_heading_has_empty_breadcrumb():
    doc = Document("x.md", "preface line\n\n# Title\nbody\n")
    chunks = chunk_document(doc)
    assert _summ(chunks) == [
        ((), "preface line"),
        (("Title",), "body"),
    ]


def test_breadcrumb_pops_on_same_or_shallower_level():
    doc = Document("x.md", "# A\n## B\nb body\n## C\nc body\n# D\nd body")
    chunks = chunk_document(doc)
    assert _summ(chunks) == [
        (("A", "B"), "b body"),
        (("A", "C"), "c body"),
        (("D",), "d body"),
    ]


def test_empty_body_sections_are_skipped():
    doc = Document("x.md", "# A\n# B\nbody B")
    chunks = chunk_document(doc)
    assert _summ(chunks) == [(("B",), "body B")]


def test_headings_inside_code_fences_are_ignored():
    doc = Document("x.md", "# Real\n```\n# not a heading\n```\nafter")
    chunks = chunk_document(doc)
    assert _summ(chunks) == [(("Real",), "```\n# not a heading\n```\nafter")]


def test_trailing_hashes_stripped_from_title():
    doc = Document("x.md", "## Section ##\nbody")
    chunks = chunk_document(doc)
    assert chunks[0].heading_path == ("Section",)


def test_hash_without_space_is_not_a_heading():
    doc = Document("x.md", "#nothashheading\nbody")
    chunks = chunk_document(doc)
    assert _summ(chunks) == [((), "#nothashheading\nbody")]


def test_line_numbers_span_heading_to_section_end():
    doc = Document("x.md", "# Title\nintro\n## Section\nbody")
    chunks = chunk_document(doc)
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 2)
    assert (chunks[1].start_line, chunks[1].end_line) == (3, 4)


def test_ids_are_contiguous_and_use_doc_path():
    doc = Document("notes/x.md", "# A\nbody a\n## B\nbody b")
    chunks = chunk_document(doc)
    assert [c.id for c in chunks] == ["notes/x.md::0", "notes/x.md::1"]


def test_empty_document_yields_no_chunks():
    assert chunk_document(Document("x.md", "")) == []
    assert chunk_document(Document("x.md", "   \n\n")) == []
