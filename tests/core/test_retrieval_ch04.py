# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import pytest

from app.core import retrieval


def test_arrange_head_tail():
    # 输入按相关度降序 [a,b,c,d,e] → a 首、b 尾、剩 c,d,e 居中
    out = retrieval.arrange_head_tail(["a", "b", "c", "d", "e"])
    assert out[0] == "a" and out[-1] == "b"
    assert set(out[1:-1]) == {"c", "d", "e"}


@pytest.mark.asyncio
async def test_strategy_vector(monkeypatch):
    async def fake_embed(q): return [0.1] * 1024
    monkeypatch.setattr("app.core.embeddings.embed_query", fake_embed)
    called = {}
    def fake_dense(client, vec, top_k, category=None, collection="knowledge"):
        called["dense"] = True
        return [{"id": 1, "score": 0.9, "question": "q", "answer": "a",
                 "section_path": "p", "content_type": "faq", "category": "运费"}]
    monkeypatch.setattr("app.kb.milvus_client.dense_search", fake_dense)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    out = await retrieval.search_knowledge("运费", strategy="vector", client=object())
    assert called.get("dense") and out[0]["id"] == 1


@pytest.mark.asyncio
async def test_strategy_hybrid_rerank(monkeypatch):
    async def fake_embed(q): return [0.1] * 1024
    monkeypatch.setattr("app.core.embeddings.embed_query", fake_embed)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    def fake_hybrid(client, vec, text, top_k, recall=50, category=None, collection="knowledge"):
        return [
            {"id": 1, "score": 0.5, "question": "运费", "answer": "满99包邮",
             "section_path": "p1", "content_type": "faq", "category": "运费"},
            {"id": 2, "score": 0.4, "question": "型号", "answer": "Pro自动清理",
             "section_path": "p2", "content_type": "manual", "category": "商品"},
        ]
    monkeypatch.setattr("app.kb.milvus_client.hybrid_search", fake_hybrid)
    async def fake_rerank(q, docs, top_n=None):
        return [(1, 0.95), (0, 0.2)]  # 第 2 条(index1)最相关
    monkeypatch.setattr("app.core.rerank.rerank", fake_rerank)
    out = await retrieval.search_knowledge("Pro 型号", strategy="hybrid_rerank", client=object())
    assert out[0]["id"] == 2 and out[0]["rerank_score"] == 0.95
