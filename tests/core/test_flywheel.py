# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch09 飞轮核心:标准化+查重一次输出;命中累加不新建、不复活;幻觉 id 跳过;串行防同批重复。"""
from app.core import flywheel
from app.core.flywheel import NormalizeResult
from app.db import repository


def _stub_llm(monkeypatch, results):
    """按序弹出预设结果;可混入 Exception 模拟坏 JSON/解析失败。"""
    queue = list(results)

    async def fake(raw_question, candidates):
        r = queue.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(flywheel, "normalize_and_match", fake)


async def test_new_question_creates_pending_item(db_session_factory, monkeypatch):
    await repository.insert_low_confidence(None, "猫窝能水洗吗", "retrieval_low_conf", None)
    _stub_llm(monkeypatch, [NormalizeResult(
        normalized_question="猫窝是否支持水洗", matched_question_id=None,
        ai_suggested_answer="以平台售后规则为准")])
    stats = await flywheel.process_pending()
    assert stats == {"processed": 1, "merged": 0, "created": 1, "skipped": 0}
    rows = await repository.list_review_queue("待审")
    assert rows[0].normalized_question == "猫窝是否支持水洗"
    assert not await repository.fetch_unmatched_low_conf(10)   # 游标已推进


async def test_match_merges_and_never_revives(db_session_factory, monkeypatch):
    rid = await repository.insert_review_item("猫窝是否支持水洗", None)
    await repository.update_review_status(rid, "驳回")        # 已驳回=终审
    await repository.insert_low_confidence(None, "猫窝可以洗吗", "user_feedback", None)
    _stub_llm(monkeypatch, [NormalizeResult(
        normalized_question="猫窝是否支持水洗", matched_question_id=rid, ai_suggested_answer="")])
    stats = await flywheel.process_pending()
    assert stats["merged"] == 1 and stats["created"] == 0
    item, raws = await repository.get_review_detail(rid)
    assert item.occurrence_count == 2
    assert item.review_status == "驳回"                        # 只累加,不复活
    assert [r.raw_question for r in raws] == ["猫窝可以洗吗"]   # 归并落点记回原话


async def test_hallucinated_id_skips_and_keeps_cursor(db_session_factory, monkeypatch):
    await repository.insert_low_confidence(None, "猫窝可以洗吗", "self_check", None)
    _stub_llm(monkeypatch, [NormalizeResult(
        normalized_question="猫窝是否支持水洗", matched_question_id=99999, ai_suggested_answer="")])
    stats = await flywheel.process_pending()
    assert stats["skipped"] == 1 and stats["processed"] == 0
    assert len(await repository.fetch_unmatched_low_conf(10)) == 1   # 游标留在原地,下轮重试
    assert await repository.list_review_queue(None) == []


async def test_llm_failure_skips_row(db_session_factory, monkeypatch):
    await repository.insert_low_confidence(None, "问题一", "self_check", None)
    await repository.insert_low_confidence(None, "问题二", "self_check", None)
    _stub_llm(monkeypatch, [RuntimeError("坏 JSON"),
                            NormalizeResult(normalized_question="问题二标准版",
                                            matched_question_id=None, ai_suggested_answer="")])
    stats = await flywheel.process_pending()
    assert stats == {"processed": 1, "merged": 0, "created": 1, "skipped": 1}


async def test_same_batch_duplicates_merge_serially(db_session_factory, monkeypatch):
    """同批两条同义:串行处理,第二条的候选里已有第一条刚建的行 → 归并而非重复建行。"""
    await repository.insert_low_confidence(None, "猫窝能水洗吗", "retrieval_low_conf", None)
    await repository.insert_low_confidence(None, "猫窝可以洗吗", "user_feedback", None)
    seen_candidates = []

    async def fake(raw_question, candidates):
        seen_candidates.append(list(candidates))
        if candidates:
            return NormalizeResult(normalized_question="猫窝是否支持水洗",
                                   matched_question_id=candidates[0]["id"], ai_suggested_answer="")
        return NormalizeResult(normalized_question="猫窝是否支持水洗",
                               matched_question_id=None, ai_suggested_answer="")
    monkeypatch.setattr(flywheel, "normalize_and_match", fake)
    stats = await flywheel.process_pending()
    assert stats == {"processed": 2, "merged": 1, "created": 1, "skipped": 0}
    assert seen_candidates[0] == [] and len(seen_candidates[1]) == 1
    rows = await repository.list_review_queue(None)
    assert len(rows) == 1 and rows[0].occurrence_count == 2
