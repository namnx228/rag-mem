import numpy as np

from ragmem.persistence import content_hash, load_semantic, save_semantic
from ragmem.types import Chunk


def _chunk(i, text):
    return Chunk(
        id=f"d.md::{i}", doc_path="d.md", heading_path=(), text=text, start_line=1, end_line=1
    )


def test_content_hash_is_deterministic_and_text_sensitive():
    a = [_chunk(0, "hello"), _chunk(1, "world")]
    b = [_chunk(0, "hello"), _chunk(1, "world")]
    c = [_chunk(0, "hello"), _chunk(1, "CHANGED")]
    assert content_hash(a) == content_hash(b)
    assert content_hash(a) != content_hash(c)


def test_save_then_load_roundtrips_matrix(tmp_path):
    chunks = [_chunk(0, "a"), _chunk(1, "b")]
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    save_semantic(tmp_path / ".ragmem", chunks, matrix, "model-x")
    loaded = load_semantic(tmp_path / ".ragmem", chunks, "model-x")
    assert loaded is not None
    assert np.array_equal(loaded, matrix)


def test_load_returns_none_without_cache(tmp_path):
    assert load_semantic(tmp_path / ".ragmem", [_chunk(0, "a")], "m") is None


def test_load_returns_none_when_text_changed(tmp_path):
    d = tmp_path / ".ragmem"
    save_semantic(d, [_chunk(0, "a")], np.zeros((1, 2), dtype=np.float32), "m")
    assert load_semantic(d, [_chunk(0, "DIFFERENT")], "m") is None


def test_load_returns_none_when_model_changed(tmp_path):
    d = tmp_path / ".ragmem"
    save_semantic(d, [_chunk(0, "a")], np.zeros((1, 2), dtype=np.float32), "m1")
    assert load_semantic(d, [_chunk(0, "a")], "m2") is None


def test_load_returns_none_when_chunk_set_changed(tmp_path):
    d = tmp_path / ".ragmem"
    save_semantic(d, [_chunk(0, "a")], np.zeros((1, 2), dtype=np.float32), "m")
    assert load_semantic(d, [_chunk(0, "a"), _chunk(1, "b")], "m") is None
