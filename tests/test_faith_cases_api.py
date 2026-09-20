# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch04 编造个案台账:一题一行、复发退回未解决、分页、角标原文如实带出、处置状态流转。"""
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api import rageval as rageval_api
from app.db import repository

CITS = [{"n": 1, "chunk_id": 141, "section_path": "退货退款政策 / 退换货运费承担",
         "question": "退换货运费承担", "answer": "无理由退货的退回运费由买家承担…"},
        {"n": 2, "chunk_id": 140, "section_path": "退货退款政策 / 换货政策 / 换货运费",
         "question": "换货运费", "answer": "非质量问题的换货,往返运费由买家承担…"}]


@pytest.fixture
def app():
    a = FastAPI()
    a.include_router(rageval_api.router)
    return a


@pytest.fixture
async def client(app, db_session_factory):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c


async def _add(eval_id="A43", **kw):
    kw.setdefault("bucket", "A_policy")
    kw.setdefault("query", "少发漏发导致的退换货,运费由谁承担")
    kw.setdefault("answer", "换货往返运费也由平台承担[3]。")
    kw.setdefault("reason", "证据区分了非质量问题由买家承担,客服遗漏条件范围。")
    return await repository.upsert_faith_case(eval_id, **kw)


async def test_首次写入默认未解决(client):
    cid, reopened = await _add(citations=CITS)
    assert reopened is False
    d = (await client.get("/api/rag-eval/faith-cases")).json()
    assert d["counts"]["未解决"] == 1 and d["total"] == 1
    item = d["items"][0]
    assert item["id"] == cid and item["status"] == "未解决" and item["seen_count"] == 1
    # 角标原文如实带出来:追溯时要能看到当时喂进去的是哪几块
    assert [c["n"] for c in item["citations"]] == [1, 2]
    assert item["citations"][1]["section_path"].endswith("换货运费")


async def test_同一题再判编造不新增行只累加(client):
    await _add()
    await _add(answer="第二版答案", reason="第二次理由")
    d = (await client.get("/api/rag-eval/faith-cases")).json()
    assert d["total"] == 1
    item = d["items"][0]
    assert item["seen_count"] == 2 and item["answer"] == "第二版答案"


async def test_已解决的题再判出来会退回未解决并标复发(client):
    cid, _ = await _add()
    r = await client.post(f"/api/rag-eval/faith-cases/{cid}/status", json={"status": "已解决", "resolution": "库里补了那一格"})
    assert r.json()["status"] == "已解决" and r.json()["reopened"] is False
    _, reopened = await _add(answer="又编了")          # 下一轮又被判出来
    assert reopened is True
    item = (await client.get("/api/rag-eval/faith-cases")).json()["items"][0]
    assert item["status"] == "未解决" and item["reopened"] is True


async def test_没记快照的老个案citations为空(client):
    await _add()
    item = (await client.get("/api/rag-eval/faith-cases")).json()["items"][0]
    assert not item["citations"]


async def test_按状态筛选与分页(client):
    for i in range(7):
        await _add(f"X{i}")
    cid, _ = await _add("Y1")
    await client.post(f"/api/rag-eval/faith-cases/{cid}/status", json={"status": "无需解决", "resolution": "裁判判严了,答案没错"})

    d = (await client.get("/api/rag-eval/faith-cases?size=5&page=1")).json()
    assert d["total"] == 8 and d["pages"] == 2 and len(d["items"]) == 5
    assert all(i["status"] == "未解决" for i in d["items"][:5])   # 未解决排前面
    d2 = (await client.get("/api/rag-eval/faith-cases?size=5&page=2")).json()
    assert len(d2["items"]) == 3

    only = (await client.get("/api/rag-eval/faith-cases?status=无需解决")).json()
    assert only["total"] == 1 and only["items"][0]["eval_id"] == "Y1"
    assert only["counts"] == {"未解决": 7, "已解决": 0, "无需解决": 1}


async def test_退回未解决会清掉处置时间(client):
    cid, _ = await _add()
    await client.post(f"/api/rag-eval/faith-cases/{cid}/status", json={"status": "已解决", "resolution": "库里补了那一格"})
    back = (await client.post(f"/api/rag-eval/faith-cases/{cid}/status",
                              json={"status": "未解决"})).json()
    assert back["status"] == "未解决" and back["resolved_at"] is None and back["reopened"] is False


async def test_处置必须写说明(client):
    # 半年后回头看,一个没写理由的「无需解决」和没处理过没区别
    cid, _ = await _add()
    for st in ("已解决", "无需解决"):
        r = await client.post(f"/api/rag-eval/faith-cases/{cid}/status", json={"status": st})
        assert r.status_code == 400
        blank = await client.post(f"/api/rag-eval/faith-cases/{cid}/status",
                                  json={"status": st, "resolution": "   "})
        assert blank.status_code == 400          # 空白也不算交代
    item = (await client.get("/api/rag-eval/faith-cases")).json()["items"][0]
    assert item["status"] == "未解决"             # 挡住了就别把状态改脏


async def test_处置说明落库并带出来(client):
    cid, _ = await _add()
    r = await client.post(f"/api/rag-eval/faith-cases/{cid}/status",
                          json={"status": "已解决", "resolution": "库里给 Lite 补了废砂盒容量约 5 天"})
    assert r.json()["resolution"] == "库里给 Lite 补了废砂盒容量约 5 天"
    item = (await client.get("/api/rag-eval/faith-cases?status=已解决")).json()["items"][0]
    assert item["resolution"].startswith("库里给 Lite")


async def test_退回未解决会连说明一起清掉(client):
    cid, _ = await _add()
    await client.post(f"/api/rag-eval/faith-cases/{cid}/status",
                      json={"status": "无需解决", "resolution": "常识类,不进库"})
    back = (await client.post(f"/api/rag-eval/faith-cases/{cid}/status",
                              json={"status": "未解决"})).json()
    assert back["resolution"] is None and back["resolved_at"] is None


async def test_幻觉率两个口径(client, monkeypatch):
    # 分母 = 可作答评上 100 + 库外 20;库外有 2 条该拒没拒 → 两个口径都要含进去
    monkeypatch.setattr(rageval_api, "_read", lambda: {
        "generation": {"faithfulness": {"A_policy": {"answered": 60}, "B_model": {"answered": 40}},
                       "refusal": {"total": 20, "correct": 18},
                       "faithfulness_cases": [{"id": f"H{i}"} for i in range(4)]}})
    for i in range(4):
        await _add(f"H{i}")
    ids = [i["id"] for i in (await client.get("/api/rag-eval/faith-cases")).json()["items"]]
    await client.post(f"/api/rag-eval/faith-cases/{ids[0]}/status",
                      json={"status": "已解决", "resolution": "库里补了那一格"})
    await client.post(f"/api/rag-eval/faith-cases/{ids[1]}/status",
                      json={"status": "无需解决", "resolution": "裁判判严了"})
    h = (await client.get("/api/rag-eval/faith-cases")).json()["hallucination"]
    assert h["evaluated"] == 120 and h["graded"] == 100 and h["absent"] == 20
    assert h["refusal_missed"] == 2          # 库外 20 题里 2 条该拒没拒 = 无据而答
    assert h["cases_judged"] == 4 and h["cases_confirmed"] == 1
    assert h["dismissed"] == 1 and h["pending"] == 2
    assert h["judged"] == 6 and h["judged_rate"] == 0.05        # (4+2)/120:线索量
    assert h["confirmed"] == 3 and h["confirmed_rate"] == 0.025  # (1+2)/120:真账
    assert h["ledger"] == {"total": 4, "未解决": 2, "已解决": 1, "无需解决": 1}


async def test_幻觉率只算本轮判出的_台账累计另算(client, monkeypatch):
    # 台账有 3 条(历史两条早改完了),但这一轮只判出 1 条 → 分子是 1,不是 3。
    # 不这么算的话:改掉的旧个案会一直摊到每一轮头上,判出率越修越高
    monkeypatch.setattr(rageval_api, "_read", lambda: {
        "generation": {"faithfulness": {"A_policy": {"answered": 240}},
                       "refusal": {"total": 60, "correct": 60},
                       "faithfulness_cases": [{"id": "A30"}]}})
    for eid in ("C2", "E15", "A30"):
        await _add(eid)
    ids = {i["eval_id"]: i["id"] for i in (await client.get("/api/rag-eval/faith-cases")).json()["items"]}
    for eid in ("C2", "E15"):        # 上几轮确认真编造并且已经改掉
        await client.post(f"/api/rag-eval/faith-cases/{ids[eid]}/status",
                          json={"status": "已解决", "resolution": "库里补了那一格"})

    h = (await client.get("/api/rag-eval/faith-cases")).json()["hallucination"]
    assert h["cases_judged"] == 1 and h["judged"] == 1
    assert h["judged_rate"] == round(1 / 300, 4)          # 本轮 1/300,不是台账 3/300
    assert h["pending"] == 1 and h["dismissed"] == 0      # A30 还没过目
    assert h["cases_confirmed"] == 0 and h["confirmed_rate"] == 0.0
    # 台账累计单独报:3 条里 2 条已解决,那是跨轮的管理视图
    assert h["ledger"] == {"total": 3, "未解决": 1, "已解决": 2, "无需解决": 0}


async def test_库外该拒没拒直接算真幻觉(client, monkeypatch):
    # 编造个案一条没有,但库外有 3 条该拒没拒 → 确认幻觉率不能是 0
    monkeypatch.setattr(rageval_api, "_read", lambda: {
        "generation": {"faithfulness": {"A_policy": {"answered": 60}},
                       "refusal": {"total": 60, "correct": 57}}})
    h = (await client.get("/api/rag-eval/faith-cases")).json()["hallucination"]
    assert h["cases_judged"] == 0 and h["refusal_missed"] == 3
    assert h["evaluated"] == 120 and h["confirmed"] == 3 and h["confirmed_rate"] == 0.025


async def test_报告没跑过时幻觉率分母为空(client, monkeypatch):
    # 分母不知道就回 None,不编一个出来。没有报告也就不知道「本轮判出了哪几题」,
    # 两个比率一起回 None——但台账累计照样报,它不依赖报告
    monkeypatch.setattr(rageval_api, "_read", lambda: None)
    await _add()
    h = (await client.get("/api/rag-eval/faith-cases")).json()["hallucination"]
    assert h["evaluated"] is None and h["confirmed_rate"] is None and h["judged_rate"] is None
    assert h["judged"] == 0                       # 不知道本轮判了什么,别拿台账顶上
    assert h["ledger"]["total"] == 1


async def test_非法状态与不存在的个案(client):
    assert (await client.get("/api/rag-eval/faith-cases?status=瞎写")).status_code == 400
    r = await client.post("/api/rag-eval/faith-cases/99999/status", json={"status": "已解决", "resolution": "库里补了那一格"})
    assert r.status_code == 404
    cid, _ = await _add()
    bad = await client.post(f"/api/rag-eval/faith-cases/{cid}/status", json={"status": "关闭"})
    assert bad.status_code == 422          # Literal 校验挡住,不写脏状态进库
