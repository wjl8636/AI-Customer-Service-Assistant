import pytest

from app.config import settings
from app.core import rerank as rr


async def _noop_sleep(_):
    """退避不真睡,测试只关心重试了几次。"""
    return None


@pytest.mark.parametrize("base, want", [
    # 默认那份:带 /v1,直接接 /rerank
    ("https://api.siliconflow.cn/v1", "https://api.siliconflow.cn/v1/rerank"),
    ("https://api.siliconflow.cn/v1/", "https://api.siliconflow.cn/v1/rerank"),
    # 「上游地址」按字面填、没带版本段:补一个 /v1,否则打到 /rerank 直接 404,
    # 而且只会在查询时以 hybrid_rerank 失败的形式冒出来,排障要绕一圈
    ("https://api.siliconflow.cn", "https://api.siliconflow.cn/v1/rerank"),
    ("https://api.siliconflow.cn/", "https://api.siliconflow.cn/v1/rerank"),
    # 已经带了版本段的别乱动:Cohere 的 rerank 在 /v2 下
    ("https://api.cohere.com/v2", "https://api.cohere.com/v2/rerank"),
])
def test_rerank_url_容忍带不带版本段(monkeypatch, base, want):
    monkeypatch.setattr(settings, "rerank_base_url", base)
    assert rr._rerank_url() == want


def test_rerank_url_每次现算_不在导入时冻住(monkeypatch):
    # 冻在模块级的话,改了配置要重启进程才生效,测试里也没法换上游
    monkeypatch.setattr(settings, "rerank_base_url", "https://a.example.com/v1")
    first = rr._rerank_url()
    monkeypatch.setattr(settings, "rerank_base_url", "https://b.example.com/v1")
    assert first != rr._rerank_url() == "https://b.example.com/v1/rerank"


@pytest.mark.asyncio
async def test_rerank_orders_and_truncates(monkeypatch):
    async def fake_post(url, json, headers, timeout):
        class R:
            def raise_for_status(self): pass
            def json(self):
                return {"results": [
                    {"index": 0, "relevance_score": 0.1},
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 2, "relevance_score": 0.5},
                ]}
        return R()
    monkeypatch.setattr("app.core.rerank._post", fake_post)
    out = await rr.rerank("q", ["a", "b", "c"], top_n=2)
    assert out == [(1, 0.9), (2, 0.5)]


@pytest.mark.asyncio
async def test_rerank_empty_docs():
    assert await rr.rerank("q", []) == []


async def test_断连也退避重试_不是只认状态码(monkeypatch):
    """上游把连接掐了跟回 503 一样是瞬时故障,一样得重试。

    早先 _post 只在拿到 resp 之后判状态码,c.post 抛出来就直接冒到顶:上游回 503
    会重试,上游断连反而一次都不试。eval-rag 跑到第 200 多题被断一次,整轮白跑。
    """
    calls = {"n": 0}

    class _Resp:
        status_code = 200

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise rr.httpx.RemoteProtocolError("Server disconnected")
            return _Resp()

    monkeypatch.setattr(rr.httpx, "AsyncClient", lambda *a, **kw: _Client())
    monkeypatch.setattr(rr.asyncio, "sleep", _noop_sleep)

    resp = await rr._post("http://x/rerank", {}, {}, 5)
    assert resp.status_code == 200
    assert calls["n"] == 2          # 第一次断连,退避后第二次成功


async def test_断连重试到底仍失败就抛出去(monkeypatch):
    """别把连接故障吞成 None——评估脚本靠异常把这一轮记成失败,静默返回会被算成低分。"""
    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **kw):
            raise rr.httpx.ConnectError("connection refused")

    monkeypatch.setattr(rr.httpx, "AsyncClient", lambda *a, **kw: _Client())
    monkeypatch.setattr(rr.asyncio, "sleep", _noop_sleep)

    with pytest.raises(rr.httpx.ConnectError):
        await rr._post("http://x/rerank", {}, {}, 5)
