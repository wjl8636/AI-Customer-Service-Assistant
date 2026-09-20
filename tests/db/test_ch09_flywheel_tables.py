# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch09 地基:review_queue / eval_runs ORM 与 repository 飞轮方法。"""
from app.db import repository


async def test_insert_low_confidence_with_snapshot(db_session_factory):
    snap = [{"question": "退货运费谁出", "answer": "满99包邮", "rerank_score": 0.91, "section_path": "售后 / 退货"}]
    lcq_id = await repository.insert_low_confidence(None, "猫窝能水洗吗", "retrieval_low_conf", "top1=0.2", retrieved_chunks=snap)
    rows = await repository.fetch_unmatched_low_conf(10)
    assert [r.id for r in rows] == [lcq_id]
    assert rows[0].retrieved_chunks[0]["rerank_score"] == 0.91
    assert rows[0].matched_review_id is None


async def test_insert_low_confidence_without_snapshot(db_session_factory):
    await repository.insert_low_confidence(None, "没走检索的问题", "user_feedback", None)
    rows = await repository.fetch_unmatched_low_conf(10)
    assert rows[0].retrieved_chunks is None


async def test_review_item_create_and_merge(db_session_factory):
    rid = await repository.insert_review_item("猫窝是否支持水洗", "可以,答案备查")
    cands = await repository.list_review_candidates()
    assert cands == [{"id": rid, "normalized_question": "猫窝是否支持水洗"}]
    await repository.increment_occurrence(rid)
    await repository.increment_occurrence(rid)
    (item, raws) = await repository.get_review_detail(rid)
    assert item.occurrence_count == 3
    assert item.review_status == "待审"
    assert raws == []


async def test_set_matched_review_links_raws(db_session_factory):
    rid = await repository.insert_review_item("猫窝是否支持水洗", None)
    lcq_id = await repository.insert_low_confidence(None, "猫窝可以洗吗", "user_feedback", None)
    await repository.set_matched_review(lcq_id, rid)
    _, raws = await repository.get_review_detail(rid)
    assert [r.id for r in raws] == [lcq_id]
    assert not await repository.fetch_unmatched_low_conf(10)  # 归并后不再是待处理


async def test_update_review_status_only_from_pending(db_session_factory):
    rid = await repository.insert_review_item("猫窝是否支持水洗", None)
    assert await repository.update_review_status(rid, "通过", approved_answer="可以水洗") is True
    item, _ = await repository.get_review_detail(rid)
    assert item.review_status == "通过" and item.approved_answer == "可以水洗"
    # 已通过的不许再改(终审)
    assert await repository.update_review_status(rid, "驳回") is False


async def test_list_review_queue_orders_by_occurrence(db_session_factory):
    a = await repository.insert_review_item("问题A", None)
    b = await repository.insert_review_item("问题B", None)
    await repository.increment_occurrence(b)
    rows = await repository.list_review_queue("待审")
    assert [r.id for r in rows] == [b, a]
    await repository.update_review_status(a, "驳回")
    assert [r.id for r in await repository.list_review_queue("待审")] == [b]
    assert len(await repository.list_review_queue(None)) == 2


async def test_eval_runs_roundtrip(db_session_factory):
    await repository.insert_eval_run("手动", 40, {"recall_at_10": 0.85, "mrr": 0.72})
    await repository.insert_eval_run("定时", 40, {"recall_at_10": 0.88, "mrr": 0.74})
    runs = await repository.list_eval_runs()
    assert len(runs) == 2
    assert runs[0].metrics["recall_at_10"] == 0.88  # 最新在前
    assert runs[1].triggered_by == "手动"
