"""ch07 后台摘要:触发判据 / 边界 / 防抖 / 任务体(LLM 用假实现,不打真上游)。"""
import asyncio
from types import SimpleNamespace

from app.core import summarizer
from app.db import repository as repo


def _msg(mid, role, content="x"):
    return SimpleNamespace(id=mid, role=role, content=content)


def _dialog_rows(n_turns, start_id=1):
    rows, i = [], start_id
    for t in range(n_turns):
        rows.append(_msg(i, "user", f"q{t}"))
        rows.append(_msg(i + 1, "assistant", f"a{t}"))
        i += 2
    return rows


def _small_budget(monkeypatch, sliding: int) -> None:
    """把滑窗预算打小,好让层 2 在几条消息内就超预算。

    只用到 .sliding 一个字段,拿个轻量对象顶上就够,不必真去算一遍预算。"""
    monkeypatch.setattr(summarizer.budget, "compute",
                        lambda *a, **k: SimpleNamespace(sliding=sliding))


async def _seed(cid, n_turns=12, content="问题"):
    for t in range(n_turns):
        await repo.append_message(cid, "user", content=f"{content}{t}")
        await repo.append_message(cid, "assistant", content=f"回答{t}")


async def _fake_llm(old_summary, dialog):
    return "摘要"


async def test_摘要边界就是层1起点(db_session_factory, db_clean, monkeypatch):
    """摘要吃的就是层 2,不另算一套边界。

    以前这里按轮数算边界(留最近 K 轮、更早的压掉),那是双层时代的规则。改成从窗口
    倒推之后两者脱节:按轮数那条每次都伸得更远,一口把还在层 1 的内容吞掉,层 2 于是
    永远是空的。讲的是三层,跑起来只有两层,而且从外面一点看不出来。
    """
    cid = await repo.create_conversation("u1")
    await _seed(cid)
    monkeypatch.setattr(summarizer, "summarize_dialog", _fake_llm)

    msgs = await repo.list_dialog_messages(cid)
    layer1_from = msgs[9].id                      # 前十条降级到了层 2
    await repo.update_layer1_from(cid, layer1_from)

    await summarizer.run_summary(cid)

    conv = await repo.get_conversation(cid)
    assert conv.summary_upto_msg_id == layer1_from     # 追到层 2 上沿,层 2 清空
    segs = await repo.list_summary_segments(cid)
    assert len(segs) == 1


async def test_层2是空的就不压(db_session_factory, db_clean, monkeypatch):
    """层 1 起点没动过,说明一条都没降级下来,没有可压的。"""
    cid = await repo.create_conversation("u1")
    await _seed(cid)
    monkeypatch.setattr(summarizer, "summarize_dialog", _fake_llm)

    await summarizer.run_summary(cid)

    conv = await repo.get_conversation(cid)
    assert not conv.summary_upto_msg_id
    assert await repo.list_summary_segments(cid) == []


async def test_run_summary_不重复压同一批(db_session_factory, db_clean, monkeypatch):
    cid = await repo.create_conversation("u1")
    await _seed(cid, content="问题 订单100")
    monkeypatch.setattr(summarizer, "summarize_dialog", _fake_llm)
    msgs = await repo.list_dialog_messages(cid)
    await repo.update_layer1_from(cid, msgs[9].id)

    await summarizer.run_summary(cid)
    conv = await repo.get_conversation(cid)
    await summarizer.run_summary(cid)              # 边界没前进 → 跳过
    conv2 = await repo.get_conversation(cid)
    assert conv2.summary == conv.summary
    assert len(await repo.list_summary_segments(cid)) == 1


async def test_层2没超预算就不压(db_session_factory, db_clean, monkeypatch):
    """装得下就别压。压缩是成本不是美德,层 2 还没满就调一次模型纯属浪费。"""
    cid = await repo.create_conversation("u1")
    await _seed(cid)
    _small_budget(monkeypatch, sliding=100_000)    # 层 2 预算 3 万,几十个字远够不着
    conv = await repo.get_conversation(cid)
    msgs = await repo.list_dialog_messages(cid)
    await repo.update_layer1_from(cid, msgs[9].id)
    conv = await repo.get_conversation(cid)

    assert not await summarizer._should_summarize(cid, conv)


async def test_层2超预算才压(db_session_factory, db_clean, monkeypatch):
    cid = await repo.create_conversation("u1")
    await _seed(cid)
    _small_budget(monkeypatch, sliding=20)         # 层 2 预算 6 token,十条消息稳超
    msgs = await repo.list_dialog_messages(cid)
    await repo.update_layer1_from(cid, msgs[9].id)
    conv = await repo.get_conversation(cid)

    assert await summarizer._should_summarize(cid, conv)


async def test_不再按条数定时压(db_session_factory, db_clean, monkeypatch):
    """双层时代那条「新增满 N 条就压」的定时器已经删掉,别有人顺手加回来。

    它留在三层里会坏事:层 2 刚被降级填进东西就被它清空,中间层等于白设。
    这里堆足够多的消息,只要层 2 是空的,就一条都不该压。
    """
    cid = await repo.create_conversation("u1")
    await _seed(cid, n_turns=60)                   # 120 条,远超当年那个 30 的阈值
    _small_budget(monkeypatch, sliding=20)
    conv = await repo.get_conversation(cid)

    assert not await summarizer._should_summarize(cid, conv)


async def test_层2是空时不起后台任务(db_session_factory, db_clean, monkeypatch):
    cid = await repo.create_conversation("u1")
    await repo.append_message(cid, "user", content="q")
    called = []

    async def fake_run(c):
        called.append(c)
    monkeypatch.setattr(summarizer, "run_summary", fake_run)

    await summarizer.maybe_schedule_summary(cid)   # 层 2 是空的 → 不起任务
    assert summarizer._running.get(cid) is None and called == []


async def test_maybe_schedule_fires_and_debounces(db_session_factory, db_clean, monkeypatch):
    cid = await repo.create_conversation("u1")
    await _seed(cid, n_turns=2)
    _small_budget(monkeypatch, sliding=20)
    msgs = await repo.list_dialog_messages(cid)
    await repo.update_layer1_from(cid, msgs[-1].id)

    gate = asyncio.Event()
    ran = []

    async def slow_run(c):
        ran.append(c)
        await gate.wait()
    monkeypatch.setattr(summarizer, "run_summary", slow_run)

    await summarizer.maybe_schedule_summary(cid)
    await summarizer.maybe_schedule_summary(cid)   # 在跑 → 防抖不重复起
    await asyncio.sleep(0)                         # 让 task 起跑
    assert ran == [cid]
    gate.set()
    await summarizer._running[cid]                 # 等任务收尾
    assert cid not in summarizer._running          # done 后清防抖表


async def test_maybe_schedule_never_raises_into_reply_path(db_session_factory, db_clean, monkeypatch):
    """触发检查在用户回复路径上,DB 抖动不得把已完成的回答炸成 500(spec §8 红线)。"""
    async def boom(cid):
        raise RuntimeError("db down")
    monkeypatch.setattr(summarizer.repository, "get_conversation", boom)
    await summarizer.maybe_schedule_summary(1)     # 不抛即过


async def test_maybe_schedule_concurrent_calls_single_task(db_session_factory, db_clean, monkeypatch):
    """TOCTOU:检查与 create_task 之间有 await,同会话并发触发只允许起一个任务。"""
    cid = await repo.create_conversation("u1")
    await _seed(cid, n_turns=2)
    _small_budget(monkeypatch, sliding=20)
    msgs = await repo.list_dialog_messages(cid)
    await repo.update_layer1_from(cid, msgs[-1].id)

    gate = asyncio.Event()
    ran = []

    async def slow_run(c):
        ran.append(c)
        await gate.wait()
    monkeypatch.setattr(summarizer, "run_summary", slow_run)

    await asyncio.gather(summarizer.maybe_schedule_summary(cid),
                         summarizer.maybe_schedule_summary(cid))
    await asyncio.sleep(0)
    assert ran == [cid]                            # 并发双触发只起一个
    gate.set()
    await summarizer._running[cid]
    assert cid not in summarizer._running


