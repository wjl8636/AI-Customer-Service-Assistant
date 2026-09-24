import pytest

from app.config import settings
from app.db import repository


@pytest.mark.asyncio
async def test_segments_append_with_increasing_seq(db_session_factory):
    cid = await repository.create_conversation("u_seg")
    assert await repository.append_summary_segment(cid, 1, 10, "第一段") == 1
    assert await repository.append_summary_segment(cid, 11, 20, "第二段") == 2


@pytest.mark.asyncio
async def test_projection_has_no_duplicate(db_session_factory):
    """投影拼的是「已有的最近几段 + 这一段」，autoflush 曾让新段重复出现一次。"""
    cid = await repository.create_conversation("u_seg")
    for i in range(1, 5):
        await repository.append_summary_segment(cid, i * 10 - 9, i * 10, f"第{i}段")
    conv = await repository.get_conversation(cid)
    lines = conv.summary.split("\n")
    assert len(lines) == settings.summary_inject_segments
    assert len(lines) == len(set(lines))
    assert lines[-1] == "第4段"


@pytest.mark.asyncio
async def test_old_segments_kept_but_not_injected(db_session_factory):
    """老段落留在库里可查，只是不再注入 —— 也不会被重压。"""
    cid = await repository.create_conversation("u_seg")
    for i in range(1, 6):
        await repository.append_summary_segment(cid, i * 10 - 9, i * 10, f"第{i}段")
    assert len(await repository.list_summary_segments(cid)) == settings.summary_inject_segments
    assert len(await repository.list_summary_segments(cid, limit=99)) == 5


@pytest.mark.asyncio
async def test_boundary_advances_with_each_segment(db_session_factory):
    cid = await repository.create_conversation("u_seg")
    await repository.append_summary_segment(cid, 1, 100, "第一段")
    assert (await repository.get_conversation(cid)).summary_upto_msg_id == 100
    await repository.append_summary_segment(cid, 101, 250, "第二段")
    assert (await repository.get_conversation(cid)).summary_upto_msg_id == 250
