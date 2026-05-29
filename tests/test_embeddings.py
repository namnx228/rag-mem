from types import SimpleNamespace

import numpy as np

from ragmem.embeddings import make_embedder


class FakeEmbeddings:
    def __init__(self, table):
        self.table = table
        self.calls = []

    def create(self, model, input):
        self.calls.append((model, list(input)))
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=self.table[t]) for t in input]
        )


class FakeClient:
    def __init__(self, table):
        self.embeddings = FakeEmbeddings(table)


def test_embed_texts_returns_matrix():
    emb = make_embedder(client=FakeClient({"a": [1.0, 0.0], "b": [0.0, 1.0]}), model="m")
    mat = emb.embed_texts(["a", "b"])
    assert mat.shape == (2, 2)
    assert np.allclose(mat, [[1.0, 0.0], [0.0, 1.0]])


def test_embed_texts_uses_model_and_batches_over_100():
    table = {str(i): [float(i)] for i in range(250)}
    client = FakeClient(table)
    emb = make_embedder(client=client, model="text-embedding-3-small")
    emb.embed_texts([str(i) for i in range(250)])
    assert len(client.embeddings.calls) == 3  # 100 + 100 + 50
    assert all(model == "text-embedding-3-small" for model, _ in client.embeddings.calls)


def test_embed_query_returns_1d_vector():
    emb = make_embedder(client=FakeClient({"q": [1.0, 2.0, 3.0]}), model="m")
    v = emb.embed_query("q")
    assert v.shape == (3,)
    assert np.allclose(v, [1.0, 2.0, 3.0])


def test_empty_texts_returns_empty_matrix_without_api_call():
    client = FakeClient({})
    emb = make_embedder(client=client, model="m")
    mat = emb.embed_texts([])
    assert mat.shape == (0, 0)
    assert client.embeddings.calls == []
