from ragmem.persistence import content_hash, manifest_fresh, save_manifest
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


def test_manifest_is_fresh_after_save(tmp_path):
    chunks = [_chunk(0, "a"), _chunk(1, "b")]
    save_manifest(tmp_path / ".ragmem", chunks, "model-x")
    assert manifest_fresh(tmp_path / ".ragmem", chunks, "model-x") is True


def test_not_fresh_without_manifest(tmp_path):
    assert manifest_fresh(tmp_path / ".ragmem", [_chunk(0, "a")], "m") is False


def test_not_fresh_when_text_changed(tmp_path):
    d = tmp_path / ".ragmem"
    save_manifest(d, [_chunk(0, "a")], "m")
    assert manifest_fresh(d, [_chunk(0, "DIFFERENT")], "m") is False


def test_not_fresh_when_model_changed(tmp_path):
    d = tmp_path / ".ragmem"
    save_manifest(d, [_chunk(0, "a")], "m1")
    assert manifest_fresh(d, [_chunk(0, "a")], "m2") is False


def test_not_fresh_when_chunk_set_changed(tmp_path):
    d = tmp_path / ".ragmem"
    save_manifest(d, [_chunk(0, "a")], "m")
    assert manifest_fresh(d, [_chunk(0, "a"), _chunk(1, "b")], "m") is False
