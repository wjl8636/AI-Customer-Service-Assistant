# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import pytest

from app.config import settings
from app.core import summarizer


class _Conv:
    def __init__(self, upto=0, layer1=0):
        self.summary_upto_msg_id = upto
        self.layer1_from_msg_id = layer1


class _Msg:
    def __init__(self, mid, content, role="user"):
        self.id, self.content, self.role = mid, content, role


@pytest.mark.asyncio
async def test_layer2_over_budget_triggers(monkeypatch):
    """层 2 撑过预算就压一段，不用等条数攒够。"""
    msgs = [_Msg(i, "证" * 400) for i in range(1, 21)]
    monkeypatch.setattr(summarizer.repository, "list_dialog_messages",
                        lambda cid: _async(msgs))
    monkeypatch.setattr(summarizer.repository, "count_messages_after",
                        lambda cid, upto: _async(0))          # 条数判据故意不满足
    monkeypatch.setattr(settings, "layer1_ratio", 0.7)
    monkeypatch.setattr(summarizer.budget, "compute", lambda: _Budget(10_000))
    assert await summarizer._should_summarize(1, _Conv(upto=0, layer1=20))


@pytest.mark.asyncio
async def test_条数攒得再多_层2是空的也不压(monkeypatch):
    """双层时代那条「新增满 N 条就压」的定时器已经删掉。

    留在三层里会坏事:层 2 刚被降级填进东西就被它清空,中间层等于白设。
    判据只有一条,层 2 超预算才压。"""
    monkeypatch.setattr(summarizer.repository, "list_dialog_messages", lambda cid: _async([]))
    assert not await summarizer._should_summarize(1, _Conv())


class _Budget:
    def __init__(self, sliding):
        self.sliding = sliding


async def _async(v):
    return v
