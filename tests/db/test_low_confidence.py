# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import pytest

from app.db import repository


@pytest.mark.asyncio
async def test_insert_low_confidence(db_session_factory):
    cid = await repository.create_conversation("u1")
    lid = await repository.insert_low_confidence(cid, "邮费到底多少啊", "self_check", "证据不足")
    assert lid > 0
    factory = db_session_factory
    from sqlalchemy import text
    async with factory() as s:
        row = (await s.execute(text("SELECT raw_question, source, reason FROM low_confidence_questions WHERE id=:i"), {"i": lid})).first()
    assert row.source == "self_check" and row.raw_question == "邮费到底多少啊"


@pytest.mark.asyncio
async def test_insert_low_confidence_null_conv(db_session_factory):
    lid = await repository.insert_low_confidence(None, "无会话问题", "retrieval_low_conf", None)
    assert lid > 0
