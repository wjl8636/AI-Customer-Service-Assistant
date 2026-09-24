"""ch10 主题 API:分布(17 类全出、计数正确)与类目问题列表(分页、多标签、未知类目挡住)。"""
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api import topics as topics_api
from app.db import repository


@pytest.fixture
def app():
    a = FastAPI()
    a.include_router(topics_api.router)
    return a


@pytest.fixture
async def client(app, db_session_factory):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        yield c


async def test_distribution_empty_returns_all_17(client):
    resp = await client.get("/api/topics/distribution")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert len(body["classes"]) == 17


async def test_distribution_counts(client):
    qid = await repository.insert_low_confidence(None, "猫窝买大了想退", "retrieval_low_conf", None)
    await repository.insert_topic_classifications(
        [{"question_id": qid, "labels": ["尺码", "退换货"]}])
    resp = await client.get("/api/topics/distribution")
    body = resp.json()
    by_label = {c["label"]: c for c in body["classes"]}
    assert body["total"] == 1
    assert by_label["尺码"]["count"] == 1 and by_label["退换货"]["count"] == 1
    assert by_label["物流"]["count"] == 0


async def test_distribution_samples_are_deduped(client):
    """归并后同一句标准化问法对应池里好几行:样例列表里只出现一次,计数照旧按条算。"""
    for _ in range(3):
        qid = await repository.insert_low_confidence(None, "废砂盒多久倒", "retrieval_low_conf", None)
        await repository.insert_topic_classifications([{"question_id": qid, "labels": ["商品信息"]}])

    body = (await client.get("/api/topics/distribution")).json()
    card = {c["label"]: c for c in body["classes"]}["商品信息"]
    assert card["count"] == 3 and card["samples"] == ["废砂盒多久倒"]


async def _classify(text: str, labels: list[str]) -> int:
    qid = await repository.insert_low_confidence(None, text, "retrieval_low_conf", None)
    await repository.insert_topic_classifications([{"question_id": qid, "labels": labels}])
    return qid


async def test_questions_paginates_within_one_class(client):
    """一类下的问题按页给:总数与页数都按这一类算,不是全表条数。"""
    for i in range(7):
        await _classify(f"猫窝能不能机洗 {i}", ["商品信息"])

    body = (await client.get("/api/topics/questions?label=商品信息&size=3&page=2")).json()
    assert body["total"] == 7 and body["pages"] == 3 and body["page"] == 2
    assert len(body["items"]) == 3
    tail = (await client.get("/api/topics/questions?label=商品信息&size=3&page=3")).json()
    assert len(tail["items"]) == 1


async def test_questions_of_multi_label_show_in_every_hit_class(client):
    """多标签问题在它命中的每个类目下都要出现,并带上同伴类目——和分布图的计数口径一致。"""
    await _classify("猫窝买大了想退", ["尺码", "退换货"])

    for label in ("尺码", "退换货"):
        body = (await client.get(f"/api/topics/questions?label={label}")).json()
        assert body["total"] == 1
        assert body["items"][0]["labels"] == ["尺码", "退换货"]
        assert body["items"][0]["text"] == "猫窝买大了想退"


async def test_questions_falls_back_to_raw_when_not_merged(client):
    """没归并的问题没有标准化问法:显示原话,并标出这条还没进待审队列。"""
    await _classify("那个猫窝啊到底能不能机洗啊", ["商品信息"])

    item = (await client.get("/api/topics/questions?label=商品信息")).json()["items"][0]
    assert item["text"] == "那个猫窝啊到底能不能机洗啊"
    assert item["normalized"] is False and item["review_status"] is None
    assert item["source"] == "retrieval_low_conf"


async def test_questions_uses_normalized_text_after_merge(client):
    """归并过的显示标准化问法,原话另给一份:列表读起来是 FAQ 式短句,不是一堆口语。"""
    qid = await _classify("那个猫窝啊到底能不能机洗啊", ["商品信息"])
    rid = await repository.insert_review_item("猫窝能不能机洗", "分内胆与外套…")
    await repository.set_matched_review(qid, rid)

    item = (await client.get("/api/topics/questions?label=商品信息")).json()["items"][0]
    assert item["text"] == "猫窝能不能机洗" and item["normalized"] is True
    assert item["raw_question"] == "那个猫窝啊到底能不能机洗啊"
    assert item["review_status"] == "待审"


async def test_unknown_label_is_rejected(client):
    """类目名写错(或旧链接)直接 400:静静回一页空列表会被当成「这类没问题」。"""
    resp = await client.get("/api/topics/questions?label=不存在的类")
    assert resp.status_code == 400
    assert "未知类目" in resp.json()["detail"]
