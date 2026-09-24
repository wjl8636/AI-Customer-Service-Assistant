from app.core import embeddings


class _FakeEmb:
    def __init__(self, vec):
        self.embedding = vec


class _FakeResp:
    def __init__(self, vecs):
        self.data = [_FakeEmb(v) for v in vecs]


class _FakeEmbeddings:
    def __init__(self):
        self.calls = []

    async def create(self, model, input):
        self.calls.append((model, list(input)))
        return _FakeResp([[float(i), 0.0, 1.0] for i, _ in enumerate(input)])


class _FakeClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddings()


async def test_embed_texts_passes_model_and_input(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(embeddings, "_client", lambda: fake)
    out = await embeddings.embed_texts(["a", "b"])
    assert out == [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
    assert fake.embeddings.calls == [("BAAI/bge-m3", ["a", "b"])]


async def test_embed_query_returns_single_vector(monkeypatch):
    monkeypatch.setattr(embeddings, "_client", lambda: _FakeClient())
    v = await embeddings.embed_query("邮费")
    assert v == [0.0, 0.0, 1.0]
