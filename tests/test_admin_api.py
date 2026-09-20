# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""后台首页聚合 API:六张卡齐出,某一块依赖没起只让它自己那张卡显示读数失败,不连坐整页。"""
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api import admin as admin_api

CARDS = ("kb", "rageval", "review", "observability", "topics", "classifier")


@pytest.fixture
def app():
    a = FastAPI()
    a.include_router(admin_api.router)
    return a


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        yield c


@pytest.fixture(autouse=True)
def _no_milvus(monkeypatch):
    async def offline():
        return {"online": False, "count": None, "detail": "stub"}
    monkeypatch.setattr("app.api.kb.milvus_state", offline)


async def test_all_deps_down_still_returns_every_card(client, monkeypatch):
    """mysql 全挂:整页仍然出得来,每张卡自己说自己读不到,后台首页恰恰是这种时候要用。"""
    async def boom(*a, **kw):
        raise RuntimeError("mysql 没起")
    for fn in ("knowledge_stats", "list_review_queue", "topic_distribution", "list_eval_runs"):
        monkeypatch.setattr(f"app.db.repository.{fn}", boom)

    body = (await client.get("/api/admin/overview")).json()
    assert [m["key"] for m in body["modules"]] == list(CARDS)
    for key in ("kb", "review", "topics", "observability"):
        m = {c["key"]: c for c in body["modules"]}[key]
        assert m["status"] == "error" and "mysql 没起" in m["note"]


async def test_empty_db_reads_as_missing_not_error(client, db_session_factory):
    """空库不是故障:知识库与飞轮该显示「没数据」,提示下一步动作,不是红字报错。"""
    body = (await client.get("/api/admin/overview")).json()
    by_key = {m["key"]: m for m in body["modules"]}
    assert by_key["kb"]["status"] == "missing"
    assert by_key["review"]["status"] == "missing"
    assert by_key["topics"]["status"] == "missing"
    assert by_key["kb"]["page"] == "/kb"


async def test_topics_headline_ranks_by_count(client, monkeypatch):
    """「占前三」得真按问题量排:topic_distribution 给的是权威类目表顺序,不是排行榜。"""
    async def dist(*a, **kw):
        return {"total": 9, "latest": None, "classes": [
            {"label": "退换货", "count": 1, "samples": []},
            {"label": "物流", "count": 3, "samples": []},
            {"label": "商品信息", "count": 5, "samples": []},
            {"label": "评价", "count": 0, "samples": []},
        ]}
    monkeypatch.setattr("app.db.repository.topic_distribution", dist)

    body = (await client.get("/api/admin/overview")).json()
    card = {m["key"]: m for m in body["modules"]}["topics"]
    assert card["headline"] == "占前三:商品信息 5、物流 3、退换货 1"
