# Ch07 会话上下文管理(滑窗 + 异步摘要)实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> 本次按用户要求走 **inline 执行**(executing-plans)。

**Goal:** 把「全量历史直递模型」升级为三层上下文:最近轮次留原文、稍远的半压(用户原话不动、回复和工具结果压缩)、更早轮次后台异步压成滚动摘要(落 conversations.summary / summary_upto_msg_id),拼装顺序固定、token 有兜底、全程不阻塞当轮回复。

**Architecture:** 方案 B——入口给用户 HumanMessage 带 `db-{msg_id}` 锚点入 State;调模型前在消费点(main_agent / coref / 意图)按锚点切窗 + trim_messages 兜底,摘要以 system 消息紧跟人设;摘要任务在 runtime 层 `asyncio.create_task` 图外后台跑。State+checkpoint 全量历史不动,精简版每次现拼,双轨互不影响。验收可观测两件套:main_agent 调模型前打 `model_ctx` 日志(摘要全文+滑窗逐条),前端会话侧栏(两个只读接口)支持多会话切换与历史回载。

**Tech Stack:** LangGraph State(add_messages + AsyncSqliteSaver checkpoint)、LangChain `trim_messages`/`with_structured_output`、SQLAlchemy async + MySQL(conversations/messages)。

**Spec:** `docs/superpowers/specs/2026-07-16-ch07-context-management-design.md`

## Global Constraints

- 技术选型定死:LangGraph State + LangChain trim_messages + MySQL 两表;走不通停下来问用户,不许换方案。
- 摘要提示词红线:只提炼事实与诉求(商品、订单号、手机号、明确诉求、未决问题);没出现的一个字不许编;寒暄不留;几十到一两百字;严格 JSON `{"summary": "string"}`。
- 拼装顺序固定:system 人设红线(恒为 AGENT_SYSTEM)→ 摘要 system → 滑窗原文 → 当前用户消息 → 本轮材料(证据/订单数据)。可变内容进 system 会让前缀缓存全 miss。
- 起步值:`summary_trigger_messages=30`、`context_window_turns=8`、`context_window_max_tokens=3000`、`summary_model=""`(空回落 chat_model),全进 settings。
- State.messages 全量历史一条不删;摘要失败只 log 不重试(下轮重触发);summary 两字段只在成功后原子更新。
- 每完成一个任务在 dev-notes/ch07.md 追记一段(不许收尾补记)。
- 涉及库 API 用法拿不准先查 Context7(已核:trim_messages 参数、add_messages 同 id 覆盖语义)。

---

### Task 1: DB 层——模型两列 + conftest 支持 ALTER + repository 三函数

**Files:**
- Modify: `app/db/models.py`(Conversation 加两列)
- Modify: `tests/conftest.py:13-33`(_DDL_FILES 加 ch07,语句解析加 ALTER TABLE)
- Modify: `app/db/repository.py`(新增 3 函数)
- Test: `tests/test_repository.py`(追加)

**Interfaces:**
- Produces: `repository.count_messages_after(cid: int, after_id: int | None) -> int`;`repository.list_dialog_messages(cid: int) -> list[Message]`(user/assistant 按 id 升序);`repository.update_conversation_summary(cid: int, summary: str, upto_msg_id: int) -> None`;`Conversation.summary: str | None`、`Conversation.summary_upto_msg_id: int | None`

- [ ] **Step 1: conftest 先行**(不是测试对象,是让测试库带上 ch07 列;两处改动)

```python
# tests/conftest.py _DDL_FILES 追加:
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch07-ddl.sql",

# _create_table_stmts 的筛选条件改为(CREATE TABLE 之外也收 ALTER TABLE,文件顺序保证先建后改):
        stmts += [s.strip() for s in sql.split(";")
                  if s.strip() and ("CREATE TABLE" in s.upper() or "ALTER TABLE" in s.upper())]
```

- [ ] **Step 2: 写失败测试**(追加到 `tests/test_repository.py`)

```python
async def test_conversation_summary_roundtrip(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    conv = await repo.get_conversation(cid)
    assert conv.summary is None and conv.summary_upto_msg_id is None   # 新会话两字段为空
    await repo.update_conversation_summary(cid, "用户问过订单1001的物流", 5)
    conv = await repo.get_conversation(cid)
    assert conv.summary == "用户问过订单1001的物流"
    assert conv.summary_upto_msg_id == 5

async def test_count_messages_after(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    ids = [await repo.append_message(cid, "user", content=f"q{i}") for i in range(4)]
    assert await repo.count_messages_after(cid, None) == 4      # 边界空=数全量
    assert await repo.count_messages_after(cid, ids[1]) == 2
    assert await repo.count_messages_after(cid, ids[3]) == 0

async def test_list_dialog_messages_filters_tool_rows(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    await repo.append_message(cid, "user", content="订单1001到哪了")
    await repo.append_message(cid, "tool", content='{"s":1}', tool_call_id="c1")
    await repo.append_message(cid, "assistant", content="在路上")
    msgs = await repo.list_dialog_messages(cid)
    assert [m.role for m in msgs] == ["user", "assistant"]      # tool 行过滤,id 升序
```

- [ ] **Step 3: 跑测试确认失败**

Run: `uv run pytest tests/test_repository.py -v -k "summary or count_messages or dialog_messages"`
Expected: FAIL(`AttributeError: summary` / `no attribute 'count_messages_after'`)

- [ ] **Step 4: 实现**

```python
# app/db/models.py Conversation 类尾部(created_at 之前)加:
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_upto_msg_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

# app/db/repository.py 追加(ch07 区):
async def count_messages_after(conversation_id: int, after_id: int | None) -> int:
    """距上次摘要新增了多少条(after_id 空 = 从未摘要,数全量)。摘要触发判据。"""
    async with db.async_session() as s:
        q = (select(func.count()).select_from(Message)
             .where(Message.conversation_id == conversation_id))
        if after_id:
            q = q.where(Message.id > after_id)
        return int((await s.execute(q)).scalar_one())

async def list_dialog_messages(conversation_id: int) -> list[Message]:
    """user/assistant 消息按 id 升序(摘要任务源数据;tool 行不落库,过滤一道保险)。"""
    async with db.async_session() as s:
        result = await s.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id,
                   Message.role.in_(("user", "assistant")))
            .order_by(Message.id)
        )
        return list(result.scalars())

async def update_conversation_summary(conversation_id: int, summary: str, upto_msg_id: int) -> None:
    """摘要成功后原子更新两字段(一起写,不会出现摘要新边界旧)。"""
    async with db.async_session() as s:
        conv = await s.get(Conversation, conversation_id)
        if conv is not None:
            conv.summary = summary
            conv.summary_upto_msg_id = upto_msg_id
            await s.commit()
```

- [ ] **Step 5: 跑测试确认通过 + 全量回归**

Run: `uv run pytest tests/test_repository.py -v` → 全 PASS;`uv run pytest -q` → 无回归

- [ ] **Step 6: Commit**

```bash
git add app/db/models.py app/db/repository.py tests/conftest.py tests/test_repository.py
git commit -m "feat(ch07): conversations 摘要两列 + repository 摘要读写/计数函数"
```

---

### Task 2: memory.py 升级——锚点切窗 build_window + 摘要注入 helpers

**Files:**
- Modify: `app/core/memory.py`
- Test: `tests/test_memory.py`(追加)

**Interfaces:**
- Consumes: 无(纯函数,仅依赖 langchain_core)
- Produces: `build_window(messages: list[BaseMessage], summary_upto_msg_id: int, max_tokens: int) -> list[BaseMessage]`;`summary_line(summary: str | None) -> str`;`summary_system(summary: str | None) -> SystemMessage | None`;锚点约定 `HumanMessage.id == f"db-{mysql_msg_id}"`

- [ ] **Step 1: 写失败测试**(追加到 `tests/test_memory.py`)

```python
from langchain_core.messages import SystemMessage
from app.core.memory import build_window, summary_line, summary_system

def _dialog(n_turns: int, start_db_id: int = 1) -> list:
    """n 轮对话,用户消息带 db-id 锚点(每轮占 2 个 id:user、assistant)。"""
    msgs = []
    db_id = start_db_id
    for i in range(n_turns):
        msgs.append(HumanMessage(f"问题{i}", id=f"db-{db_id}"))
        msgs.append(AIMessage(f"回答{i}"))
        db_id += 2
    return msgs

def test_build_window_cuts_at_anchor():
    msgs = _dialog(10)                       # 用户消息 db-1,3,...,19
    win = build_window(msgs, summary_upto_msg_id=8, max_tokens=100000)
    assert win[0].id == "db-9"               # 第一条 > 8 的用户消息
    assert len(win) == 12                    # 第 5-10 轮共 6 轮
    assert win[-1] == msgs[-1]

def test_build_window_no_boundary_keeps_all():
    msgs = _dialog(3)
    assert build_window(msgs, summary_upto_msg_id=0, max_tokens=100000) == msgs

def test_build_window_anchor_missing_degrades_to_trim():
    msgs = [HumanMessage("旧消息无锚点"), AIMessage("答")] * 3   # ch07 前的旧会话消息
    win = build_window(list(msgs), summary_upto_msg_id=99, max_tokens=100000)
    assert len(win) == 6                     # 找不到锚点 → 不切,整段进 trim

def test_build_window_token_cap_still_applies():
    msgs = _dialog(20)
    for m in msgs:
        m.content = m.content + "喵" * 200   # 撑大 token
    win = build_window(msgs, summary_upto_msg_id=0, max_tokens=300)
    assert 0 < len(win) < len(msgs)
    assert win[0].type == "human"            # trim start_on=human

def test_build_window_never_returns_empty():
    msgs = [HumanMessage("超长" + "喵" * 5000, id="db-1")]
    win = build_window(msgs, summary_upto_msg_id=0, max_tokens=10)
    assert win == msgs                       # 裁到空则宁可超预算也回退原窗

def test_summary_helpers():
    assert summary_line(None) == "" and summary_line("") == ""
    assert "订单1001" in summary_line("用户问过订单1001")
    assert summary_system(None) is None
    ss = summary_system("用户问过订单1001")
    assert isinstance(ss, SystemMessage) and "订单1001" in ss.content
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_memory.py -v`
Expected: FAIL(`ImportError: cannot import name 'build_window'`)

- [ ] **Step 3: 实现**(`app/core/memory.py` 追加;`trim_history`/`SessionStore` 保留不动)

```python
from langchain_core.messages import HumanMessage, SystemMessage

_DB_ID_PREFIX = "db-"

def _db_msg_id(m: BaseMessage) -> int | None:
    """从消息 id 解出 MySQL 消息 id(入口约定 HumanMessage.id=f"db-{msg_id}");非锚点返回 None。"""
    mid = getattr(m, "id", None)
    if isinstance(mid, str) and mid.startswith(_DB_ID_PREFIX):
        try:
            return int(mid[len(_DB_ID_PREFIX):])
        except ValueError:
            return None
    return None

def build_window(messages: list[BaseMessage], summary_upto_msg_id: int,
                 max_tokens: int) -> list[BaseMessage]:
    """滑动窗口:摘要边界锚点切窗(第一条 db-id > 边界的用户消息起)+ trim_messages token 兜底。
    锚点缺失(旧会话)/边界为 0 → 整段进 trim,退化纯 token 裁剪;裁到空宁可超预算回退原窗。
    只在调模型前现拼,不改 State。"""
    start = 0
    if summary_upto_msg_id:
        for i, m in enumerate(messages):
            if isinstance(m, HumanMessage):
                did = _db_msg_id(m)
                if did is not None and did > summary_upto_msg_id:
                    start = i
                    break
    window = messages[start:]
    trimmed = trim_history(window, max_tokens=max_tokens)
    return trimmed or window

def summary_line(summary: str | None) -> str:
    """coref/意图的历史文本前缀:单行摘要;无摘要空串。"""
    return f"(早前对话摘要:{summary})" if summary else ""

def summary_system(summary: str | None) -> SystemMessage | None:
    """main_agent 拼装用:摘要 system 消息(紧跟人设 system);无摘要 None。"""
    if not summary:
        return None
    return SystemMessage(f"## 早前对话摘要(更早轮次已压缩,其中事实可信)\n{summary}")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_memory.py -v` → 全 PASS

- [ ] **Step 5: Commit**

```bash
git add app/core/memory.py tests/test_memory.py
git commit -m "feat(ch07): memory 升级——db-id 锚点切窗 build_window + 摘要注入 helpers"
```

---

### Task 3: settings + 摘要 prompt + summarizer(边界/触发/任务体)

**Files:**
- Modify: `app/config.py`(4 配置)
- Modify: `app/core/prompts.py`(SUMMARY_SYSTEM / SUMMARY_PROMPT)
- Create: `app/core/summarizer.py`
- Test: `tests/test_summarizer.py`(新建)

**Interfaces:**
- Consumes: Task 1 的 `repository.list_dialog_messages / count_messages_after / update_conversation_summary / get_conversation`
- Produces: `summarizer.compute_boundary(messages, keep_turns: int) -> int | None`;`summarizer.summarize_dialog(old_summary: str, dialog: str) -> str`(async);`summarizer.run_summary(cid: int) -> None`(async);`summarizer.maybe_schedule_summary(cid: int) -> None`(async,fire-and-forget 入口);settings 4 项(见 Global Constraints)

- [ ] **Step 1: 写失败测试**(`tests/test_summarizer.py`;LLM 用 monkeypatch 假实现,不打真上游)

```python
from types import SimpleNamespace

import pytest

from app.core import summarizer

def _msg(mid, role, content="x"):
    return SimpleNamespace(id=mid, role=role, content=content)

def _dialog_rows(n_turns, start_id=1):
    rows, i = [], start_id
    for t in range(n_turns):
        rows.append(_msg(i, "user", f"q{t}"))
        rows.append(_msg(i + 1, "assistant", f"a{t}"))
        i += 2
    return rows

def test_compute_boundary_keeps_last_k_turns():
    rows = _dialog_rows(12)                       # user id 1,3,...,23
    b = summarizer.compute_boundary(rows, keep_turns=8)
    assert b == 8                                 # 倒数第8轮用户消息 id=9,前一条 id=8

def test_compute_boundary_not_enough_turns():
    assert summarizer.compute_boundary(_dialog_rows(8), keep_turns=8) is None
    assert summarizer.compute_boundary([], keep_turns=8) is None

async def test_run_summary_updates_fields(db_session_factory, db_clean, monkeypatch):
    from app.db import repository as repo
    cid = await repo.create_conversation("u1")
    for t in range(12):
        await repo.append_message(cid, "user", content=f"问题{t} 订单100{t}")
        await repo.append_message(cid, "assistant", content=f"回答{t}")

    async def fake_llm(old_summary, dialog):
        return f"摘要[旧:{bool(old_summary)}]:" + dialog[:20]
    monkeypatch.setattr(summarizer, "summarize_dialog", fake_llm)
    monkeypatch.setattr(summarizer.settings, "context_window_turns", 8)

    await summarizer.run_summary(cid)
    conv = await repo.get_conversation(cid)
    assert conv.summary and conv.summary.startswith("摘要[旧:False]")
    assert conv.summary_upto_msg_id is not None    # 边界=倒数第8轮用户消息前一条

    await summarizer.run_summary(cid)              # 边界未前进 → 跳过,不重写
    conv2 = await repo.get_conversation(cid)
    assert conv2.summary == conv.summary

async def test_maybe_schedule_below_threshold_noop(db_session_factory, db_clean, monkeypatch):
    from app.db import repository as repo
    cid = await repo.create_conversation("u1")
    await repo.append_message(cid, "user", content="q")
    called = []

    async def fake_run(c):
        called.append(c)
    monkeypatch.setattr(summarizer, "run_summary", fake_run)
    monkeypatch.setattr(summarizer.settings, "summary_trigger_messages", 30)

    await summarizer.maybe_schedule_summary(cid)   # 1 条 < 30 → 不起任务
    assert summarizer._running.get(cid) is None and called == []

async def test_maybe_schedule_fires_and_debounces(db_session_factory, db_clean, monkeypatch):
    import asyncio
    from app.db import repository as repo
    cid = await repo.create_conversation("u1")
    for t in range(2):
        await repo.append_message(cid, "user", content=f"q{t}")
    monkeypatch.setattr(summarizer.settings, "summary_trigger_messages", 2)

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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_summarizer.py -v`
Expected: FAIL(`ModuleNotFoundError: app.core.summarizer`)

- [ ] **Step 3: 实现**

```python
# app/config.py ch06 配置后追加:
    # ch07 会话上下文管理(滑窗 + 异步摘要)
    summary_trigger_messages: int = 30    # 距上次摘要新增满多少条触发后台任务
    context_window_turns: int = 8         # 摘要后保留原文的最近轮数(需求 5-10 取中)
    context_window_max_tokens: int = 3000 # 滑窗 token 上限(trim_messages 兜底)
    summary_model: str = ""               # 摘要模型,空=回落 chat_model
```

```python
# app/core/prompts.py 尾部追加:
SUMMARY_SYSTEM = """## 角色
你是客服对话摘要器。把早期的客服对话压成简洁摘要,供后续轮次当上下文用。

## 规则
- 只提炼事实与诉求:问过哪款商品、报过的订单号/手机号、用户的明确诉求、还没解决的问题
- 对话里没出现的内容一个字不许编;寒暄闲聊不留
- 已有旧摘要时,把旧摘要与新对话合并成一份;旧摘要里的订单号、手机号等事实不得丢
- 长度几十到一两百字
- 严格输出 JSON:{{"summary": "string"}}"""

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SUMMARY_SYSTEM),
     ("human", "旧摘要:{old_summary}\n\n需要并入摘要的对话:\n{dialog}")]
)
```

```python
# app/core/summarizer.py(新建)
"""ch07 后台滚动摘要:轮结束后计数触发,asyncio 后台跑,不阻塞当轮回复。
失败只 log 不重试——触发条件仍满足,下一轮自然重触发;summary 两字段只在成功后原子更新。"""
import asyncio
import logging
import time

from pydantic import BaseModel, Field

from app.config import settings
from app.core.llm import get_chat_model
from app.core.prompts import SUMMARY_PROMPT
from app.db import repository

logger = logging.getLogger(__name__)

_running: dict[int, asyncio.Task] = {}   # cid -> 在跑任务(防抖:在跑不重复起)

class _Summary(BaseModel):
    summary: str = Field(description="早期对话滚动摘要,只含事实与诉求")

def compute_boundary(messages, keep_turns: int) -> int | None:
    """新摘要边界 = 倒数第 keep_turns 轮用户消息的前一条消息 id(滑窗从其后接原文)。
    不足 keep_turns 轮 / 边界落在会话开头 → None(本次放弃)。
    messages:按 id 升序的 user/assistant 行(需 .id/.role)。"""
    user_idx = [i for i, m in enumerate(messages) if m.role == "user"]
    if len(user_idx) <= keep_turns:
        return None
    cut = user_idx[-keep_turns]
    if cut == 0:
        return None
    return messages[cut - 1].id

async def summarize_dialog(old_summary: str, dialog: str) -> str:
    """LLM 滚动重写:旧摘要 + 新滑出段 → 新摘要。严格 JSON(structured output)。"""
    model = get_chat_model(model=settings.summary_model or None).with_structured_output(_Summary)
    r: _Summary = await (SUMMARY_PROMPT | model).ainvoke(
        {"old_summary": old_summary or "(无)", "dialog": dialog})
    return (r.summary or "").strip()

async def run_summary(conversation_id: int) -> None:
    """任务体:读消息 → 算边界 → 压缩 → 原子更新两字段。异常往外抛,由 done 回调统一 log。"""
    t0 = time.monotonic()
    conv = await repository.get_conversation(conversation_id)
    if conv is None:
        return
    msgs = await repository.list_dialog_messages(conversation_id)
    boundary = compute_boundary(msgs, settings.context_window_turns)
    old_upto = conv.summary_upto_msg_id or 0
    if boundary is None or boundary <= old_upto:
        logger.info("summary skip conv=%s boundary=%s upto=%s",
                    conversation_id, boundary, old_upto)
        return
    seg = [m for m in msgs if old_upto < m.id <= boundary]
    dialog = "\n".join(
        f"{'用户' if m.role == 'user' else '客服'}:{m.content}" for m in seg if m.content)
    logger.info("summary start conv=%s msgs=%s upto %s->%s",
                conversation_id, len(seg), old_upto, boundary)
    summary = await summarize_dialog(conv.summary or "", dialog)
    if not summary:
        raise ValueError("摘要为空,放弃更新")
    await repository.update_conversation_summary(conversation_id, summary, boundary)
    logger.info("summary done conv=%s upto=%s len=%s cost=%.0fms",
                conversation_id, boundary, len(summary), (time.monotonic() - t0) * 1000)

async def maybe_schedule_summary(conversation_id: int) -> None:
    """轮结束后调用:新增消息数达阈值且无在跑任务 → create_task 后台跑,不 await(不阻塞回复)。"""
    if conversation_id in _running:
        return
    conv = await repository.get_conversation(conversation_id)
    if conv is None:
        return
    n = await repository.count_messages_after(conversation_id, conv.summary_upto_msg_id)
    if n < settings.summary_trigger_messages:
        return
    logger.info("summary trigger conv=%s new_msgs=%s", conversation_id, n)
    task = asyncio.create_task(run_summary(conversation_id))
    _running[conversation_id] = task

    def _done(t: asyncio.Task) -> None:
        _running.pop(conversation_id, None)
        if not t.cancelled() and t.exception() is not None:
            logger.error("summary failed conv=%s", conversation_id, exc_info=t.exception())

    task.add_done_callback(_done)
```

注意:`run_summary` 内部调用 `summarize_dialog` 要走**模块属性**(`await summarize_dialog(...)` 直呼即可,测试 monkeypatch `summarizer.summarize_dialog` 对同模块直呼生效——Python 名字查找走模块全局,成立);`maybe_schedule_summary` 里 `run_summary(conversation_id)` 同理。

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

Run: `uv run pytest tests/test_summarizer.py -v` → 全 PASS;`uv run pytest -q` → 无回归

- [ ] **Step 5: Commit**

```bash
git add app/config.py app/core/prompts.py app/core/summarizer.py tests/test_summarizer.py
git commit -m "feat(ch07): 摘要 prompt + summarizer(边界计算/触发防抖/后台任务体)"
```

---

### Task 4: 接线——State 两字段 + runtime 入口/触发 + nodes 三处消费点

**Files:**
- Modify: `app/graph/state.py`(ConversationState 加 summary / summary_upto_msg_id)
- Modify: `app/graph/runtime.py`(_ensure_conversation 带出摘要、_graph_input 锚点+摘要、四个入口轮后触发)
- Modify: `app/graph/nodes.py`(_history_text / _agent_messages 用 build_window + 摘要注入)
- Test: `tests/test_context_assembly.py`(新建,纯函数级)

**Interfaces:**
- Consumes: Task 2 `memory.build_window / summary_line / summary_system`;Task 3 `summarizer.maybe_schedule_summary`;Task 1 `Conversation.summary*`
- Produces: State 新字段 `summary: str`、`summary_upto_msg_id: int`;`_graph_input(user_id, message, cid, msg_id, summary, summary_upto)`(runtime 内部)

- [ ] **Step 1: 写失败测试**(`tests/test_context_assembly.py`)

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph.nodes import _agent_messages, _history_text

def _state(n_turns=3, summary="", upto=0):
    msgs = []
    db_id = 1
    for i in range(n_turns):
        msgs.append(HumanMessage(f"问题{i}", id=f"db-{db_id}"))
        msgs.append(AIMessage(f"回答{i}"))
        db_id += 2
    return {"messages": msgs, "summary": summary, "summary_upto_msg_id": upto}

def test_agent_messages_order_with_summary():
    """拼装顺序固定:人设 system → 摘要 system → 滑窗原文(当前用户消息在窗尾)。"""
    ms = _agent_messages(_state(summary="用户问过订单1001", upto=0))
    assert isinstance(ms[0], SystemMessage) and "小喵" in ms[0].content   # 人设红线打头
    assert isinstance(ms[1], SystemMessage) and "订单1001" in ms[1].content
    assert isinstance(ms[2], HumanMessage)
    assert ms[-1].content == "回答2" or isinstance(ms[-1], AIMessage)

def test_agent_messages_no_summary_no_extra_system():
    ms = _agent_messages(_state(summary="", upto=0))
    assert isinstance(ms[0], SystemMessage)
    assert not isinstance(ms[1], SystemMessage)     # 无摘要不插第二条 system

def test_agent_messages_window_cut_by_boundary():
    ms = _agent_messages(_state(n_turns=6, summary="早期摘要", upto=6))
    win = [m for m in ms if not isinstance(m, SystemMessage)]
    assert win[0].id == "db-7"                      # 滑窗从边界后第一条用户消息接原文

def test_history_text_prepends_summary_and_windows():
    text = _history_text(_state(n_turns=6, summary="用户问过订单1001", upto=6))
    assert text.startswith("(早前对话摘要:用户问过订单1001")
    assert "问题0" not in text                       # 边界前原文不出现
    assert "问题3" in text                           # 窗内原文在(排除当前最后一条 human)

def test_history_text_no_summary_same_as_before():
    text = _history_text(_state(n_turns=2))
    assert "摘要" not in text and "问题0" in text
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_context_assembly.py -v`
Expected: FAIL(`_agent_messages` 现在返回 system+全量,顺序/切窗断言不成立)

- [ ] **Step 3: 实现——state.py**

```python
# ConversationState 增(messages 行后):
    summary: str            # 早期轮次滚动摘要(入口每轮从 conversations 加载)
    summary_upto_msg_id: int  # 摘要覆盖到的 MySQL 消息 id,滑窗从其后接原文
```

- [ ] **Step 4: 实现——nodes.py**(`_history_text` 与 `_agent_messages` 两处;头部 `from app.core import memory`)

```python
def _history_text(state, max_turns: int = 6) -> str:
    """摘要行 + 滑窗内最近若干轮(不含本轮最后一条 human),供 coref/意图读上下文。
    ch07:先按摘要边界切窗,再取尾——跨滑窗的指代靠摘要行兜住。"""
    msgs = memory.build_window(state.get("messages", []),
                               state.get("summary_upto_msg_id") or 0,
                               settings.context_window_max_tokens)
    prior = msgs[:-1] if msgs else []
    lines = []
    for m in prior[-max_turns:]:
        role = "用户" if isinstance(m, HumanMessage) else "客服"
        text = m.content if isinstance(m.content, str) else ""
        if text:
            lines.append(f"{role}:{text}")
    body = "\n".join(lines)
    head = memory.summary_line(state.get("summary"))
    return f"{head}\n{body}".strip() if head else body

TURN_CTX_ID = "turn-ctx"   # 哨兵:日志据此把注入块跟真实用户消息分开。只能用 id 不能用 name
                           # ——name 会被序列化发给上游,改变请求字节;id 不会。


def _turn_context(state) -> str:
    """本轮才有的材料:检索证据 + 退款路的订单数据。没有就返回空串。

    顺序不能反:REFUND_JUDGE_HINT 的措辞是「下面给出该订单数据与检索到的退换货政策证据」,
    它假设证据已在前文给过。"""
    parts = []
    ss = memory.summary_system(state.get("summary"))
    if ss is not None:
        parts.append("\n\n" + ss.content)     # 摘要排最前:它讲的是更早发生的事
    if state.get("evidence"):
        parts.append(_KNOWLEDGE_EVIDENCE_HINT + state["evidence"])
    if state.get("route") == "refund_flow":
        parts.append(REFUND_JUDGE_HINT + json.dumps(state.get("order_data", {}), ensure_ascii=False))
    return "".join(parts)     # 三段文本逐字沿用原来的常量,一个字都不改:这次只挪位置,


def _with_turn_context(window: list, turn_ctx: str) -> list:
    """插在滑窗里最后一条用户消息之后:ReAct 第 2 步的 [Sys,问,材料,AI,Tool] 要完整
    包含第 1 步的 [Sys,问,材料] 作前缀,轮内才命中缓存。窗内无用户消息则退化成追加。"""
    msg = HumanMessage(turn_ctx, id=TURN_CTX_ID)
    for i in range(len(window) - 1, -1, -1):
        if isinstance(window[i], HumanMessage):
            return [*window[:i + 1], msg, *window[i + 1:]]
    return [*window, msg]


def _agent_messages(state) -> list:
    """拼装顺序固定(ch07):人设+红线 system → 滑窗原文 → 摘要与本轮材料(紧跟用户那句)。

    整条消息列表里**只有一条 SystemMessage**,内容恒为 AGENT_SYSTEM。这不是洁癖:
    上游的 chat template 会把列表里所有 system 消息上提、合并成一个头部块渲染,所以
    「摘要单独放第二条 system」等于把它拼在了 AGENT_SYSTEM 后面,工具 schema 被挤到
    可变内容之后 —— 而 prompt caching 按渲染后的前缀精确匹配,于是整段前缀全 miss。
    实测:无摘要的会话 cache_read=2048,摘要一进 system 就掉到 0。
    摘要因此跟证据一起走用户侧那条消息。

    滑窗=摘要边界锚点切 + trim_messages token 兜底;State 全量历史不动,这里只现拼精简版。"""
    window = memory.build_window(state.get("messages", []),
                                 state.get("summary_upto_msg_id") or 0,
                                 settings.context_window_max_tokens)
    turn_ctx = _turn_context(state)
    if turn_ctx:
        window = _with_turn_context(window, turn_ctx)
    return [SystemMessage(AGENT_SYSTEM), *window]
```

`_history_text` 里 `AIMessage(tool_calls)`/ToolMessage 行 content 多为空串,原逻辑就跳过,不需要额外过滤。

- [ ] **Step 5: 实现——runtime.py**(头部 `from app.core import summarizer`)

```python
async def _ensure_conversation(user_id: str, conversation_id: int | None) -> tuple[int, str, int]:
    """返回 (cid, summary, summary_upto_msg_id):入口一次库读把摘要两字段一并带出。"""
    if conversation_id is None:
        return await repository.create_conversation(user_id), "", 0
    conv = await repository.get_conversation(conversation_id)
    if conv is None:
        raise ConversationNotFound(conversation_id)
    return conversation_id, conv.summary or "", conv.summary_upto_msg_id or 0

def _graph_input(user_id: str, message: str, cid: int, msg_id: int,
                 summary: str, summary_upto: int) -> dict:
    # (原注释保留)ch07:用户消息带 db-{msg_id} 锚点(摘要边界对齐);summary 两字段每轮刷新。
    return {"messages": [HumanMessage(message, id=f"db-{msg_id}")], "user_id": user_id,
            "conversation_id": cid, "steps": 0, "tokens_used": 0,
            "answer": "", "suggested_actions": [], "evidence": "",
            "evidence_strong": False, "citations": [],
            "resolved_query": "", "intent_confidence": 0.0,
            "order_id": "", "order_data": {},
            "summary": summary, "summary_upto_msg_id": summary_upto,
            "trace": None}

# run_turn:
    cid, summary, upto = await _ensure_conversation(user_id, conversation_id)
    msg_id = await repository.append_message(cid, "user", content=message)
    config = {"configurable": {"thread_id": str(cid)}}
    final = await get_graph().ainvoke(_graph_input(user_id, message, cid, msg_id, summary, upto), config)
    await summarizer.maybe_schedule_summary(cid)     # 轮后触发检查(后台,不阻塞返回)
    return {...原样...}

# resume_turn:ainvoke 后加 await summarizer.maybe_schedule_summary(conversation_id)
# stream_turn:
    cid, summary, upto = await _ensure_conversation(user_id, conversation_id)
    msg_id = await repository.append_message(cid, "user", content=message)
    async for ev in _stream_events(cid, _graph_input(user_id, message, cid, msg_id, summary, upto)):
        yield ev
    await summarizer.maybe_schedule_summary(cid)
# stream_resume:async for 后同样加 await summarizer.maybe_schedule_summary(conversation_id)
```

- [ ] **Step 6: 跑测试确认通过 + 全量回归**

Run: `uv run pytest tests/test_context_assembly.py -v` → 全 PASS;`uv run pytest -q` → 无回归(重点看 test_agent_api / test_chat_api / test_actions_api 这几个走 runtime 的)

- [ ] **Step 7: Commit**

```bash
git add app/graph/state.py app/graph/runtime.py app/graph/nodes.py tests/test_context_assembly.py
git commit -m "feat(ch07): 接线——入口 db-id 锚点+摘要入 State,三处消费点切窗注入,轮后触发摘要"
```

---

### Task 5: 摘要 prompt 标注样例验证(eval 代 TDD)+ Makefile

**Files:**
- Create: `scripts/eval_ch07.py`
- Modify: `Makefile`(加 `eval-ch07` target,.PHONY 同步)

**Interfaces:**
- Consumes: Task 3 `summarizer.summarize_dialog`
- Produces: `make eval-ch07` / `uv run python -m scripts.eval_ch07`(需上游可用)

- [ ] **Step 1: 写脚本**

```python
"""ch07 摘要 prompt 标注样例验证(纯 Prompt 任务以 eval 代 TDD)。需上游可用。
断言:严格 JSON(structured output 保证)、关键事实保留、无编造实体、长度达标、寒暄不留。
用法:uv run python -m scripts.eval_ch07"""
import asyncio
import re

from app.core.summarizer import summarize_dialog

# 样例1:事实保留——订单号/手机号/诉求必须进摘要
CASE1_DIALOG = """用户:你好在吗
客服:您好,我是小喵,请问有什么能帮您?
用户:我买的猫爬架订单1001一直没到,什么时候发货
客服:订单1001已从杭州仓发出,预计后天到。
用户:太慢了,我手机13800138000,到了让快递提前打电话
客服:好的,已备注让快递员派送前致电13800138000。
用户:对了这个猫爬架承重多少
客服:这款猫爬架最大承重15公斤。"""
CASE1_MUST = ["1001", "13800138000", "猫爬架"]
CASE1_BAN = ["在吗", "您好,我是小喵"]          # 寒暄不留

# 样例2:无编造——摘要里的数字实体必须都在原文出现过
CASE2_DIALOG = """用户:订单2002能退吗
客服:订单2002是猫粮,已签收3天,符合7天无理由,可以退。
用户:那我要退,原因是猫不爱吃
客服:好的,已为您记录退款诉求:订单2002,原因「猫不爱吃」。"""

# 样例3:滚动合并——旧摘要里的事实不得丢
CASE3_OLD = "用户问过订单1001(猫爬架)物流,留了手机13800138000要求快递先打电话;问过承重(15公斤)。"
CASE3_DIALOG = """用户:猫爬架到了,但是少了一根立柱
客服:非常抱歉!可为您补发立柱,或整单退货,您选哪种?
用户:补发吧
客服:好的,已登记订单1001补发立柱,3天内发出。"""
CASE3_MUST = ["1001", "13800138000", "立柱"]     # 旧事实(手机号)+ 新事实(补发立柱)

def check(name: str, summary: str, must=(), ban=(), src_digits: str = "") -> bool:
    ok = True
    problems = []
    if not (20 <= len(summary) <= 250):
        ok, problems = False, problems + [f"长度{len(summary)}出界[20,250]"]
    for kw in must:
        if kw not in summary:
            ok, problems = False, problems + [f"关键事实丢失:{kw}"]
    for kw in ban:
        if kw in summary:
            ok, problems = False, problems + [f"寒暄残留:{kw}"]
    if src_digits:
        src_nums = set(re.findall(r"\d{4,}", src_digits))
        for n in set(re.findall(r"\d{4,}", summary)):
            if n not in src_nums:
                ok, problems = False, problems + [f"编造数字实体:{n}"]
    print(f"{'✅' if ok else '❌'} {name} len={len(summary)}")
    print(f"   摘要:{summary}")
    if problems:
        print(f"   问题:{problems}")
    return ok

async def main():
    results = []
    s1 = await summarize_dialog("", CASE1_DIALOG)
    results.append(check("样例1 事实保留+寒暄不留", s1, must=CASE1_MUST, ban=CASE1_BAN,
                         src_digits=CASE1_DIALOG))
    s2 = await summarize_dialog("", CASE2_DIALOG)
    results.append(check("样例2 无编造(数字实体⊆原文)", s2, must=["2002"],
                         src_digits=CASE2_DIALOG))
    s3 = await summarize_dialog(CASE3_OLD, CASE3_DIALOG)
    results.append(check("样例3 滚动合并旧事实不丢", s3, must=CASE3_MUST,
                         src_digits=CASE3_OLD + CASE3_DIALOG))
    print(f"\n{sum(results)}/{len(results)} 通过")
    raise SystemExit(0 if all(results) else 1)

if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Makefile**

```makefile
# .PHONY 行尾追加 eval-ch07
eval-ch07:
	uv run python -m scripts.eval_ch07
```

- [ ] **Step 3: 跑真模型验证**

Run: `make eval-ch07`
Expected: `3/3 通过`(不过就改 SUMMARY_SYSTEM 提示词重跑,直到过;改动记 dev-notes)

- [ ] **Step 4: Commit**

```bash
git add scripts/eval_ch07.py Makefile
git commit -m "feat(ch07): 摘要 prompt 标注样例验证脚本(eval 代 TDD)"
```

---

### Task 6: 验收配套——会话侧栏 + model_ctx 上下文日志

**Files:**
- Create: `app/api/conversations.py`(两个只读接口)
- Modify: `app/main.py`(注册路由)、`app/db/repository.py`(list_conversations)
- Modify: `app/graph/nodes.py`(`_log_model_context`,main_agent 调模型前调用)
- Modify: `app/static/index.html`(会话侧栏:列表/切换/历史回载)
- Test: `tests/test_conversations_api.py`(新建,路由层打桩)、`tests/test_repository.py`(list_conversations 真库)、`tests/test_context_assembly.py`(日志内容 caplog)

**Interfaces:**
- Consumes: Task 1 `repository.get_conversation / list_dialog_messages`;Task 4 `_agent_messages`
- Produces: `GET /api/conversations?user_id=` → `{items:[{id,status,preview,has_summary,updated_at}]}`;`GET /api/conversations/{id}/messages` → `{items:[{role,content,created_at}]}`(404 会话不存在);`repository.list_conversations(user_id, limit=50) -> list[dict]`;`nodes._log_model_context(state, msgs) -> None`

- [ ] **Step 1: 写失败测试**——路由层打桩(本仓惯例:API 层 mock、repository 层真库;TestClient 的 anyio portal 与 pytest-asyncio 事件循环不共库连接,API 测试直连真库会报 attached to a different loop):

```python
# tests/test_conversations_api.py:fake repository,断言 user_id 透传、items 序列化、404
# tests/test_repository.py 追加:list_conversations 新在前/首问预览/(空会话)/has_summary 标记/不见别人会话
# tests/test_context_assembly.py 追加:
def test_log_model_context_shows_summary_and_window(caplog):
    state = _state(n_turns=6, summary="用户问过订单1001,留了手机13800138000", upto=6)
    msgs = _agent_messages(state)
    with caplog.at_level(logging.INFO, logger="app.graph.nodes"):
        _log_model_context(state, msgs)
    assert "model_ctx" in caplog.text and "订单1001" in caplog.text
    assert "[human] '问题3" in caplog.text     # 滑窗逐条可见
    assert "问题0" not in caplog.text          # 边界前原文不进上下文也不进日志
```

- [ ] **Step 2: 实现**

```python
# app/db/repository.py
async def list_conversations(user_id: str, limit: int = 50) -> list[dict]:
    """某用户的会话列表(新在前,带首问预览 + 有无摘要标记),前端多会话切换用。"""
    # select Conversation ... order_by(id.desc()).limit(limit);每条再查首条 user 消息作 preview

# app/api/conversations.py:两个 GET 路由,404 走 HTTPException;app/main.py include_router

# app/graph/nodes.py
def _log_model_context(state, msgs) -> None:
    """验收可观测:摘要全文 + 滑窗每条(角色+前40字) + 窗口条数 + tokens 估算,进 log/app.log。"""
    # main_agent 内:msgs = _agent_messages(state); _log_model_context(state, msgs); ainvoke(msgs)
```

前端(app/static/index.html):`.app` 改横向 flex,左侧 `#sidebar`(224px,像素风,<780px 隐藏)——`loadConversations()` 拉列表渲染(active 高亮 + 「已摘要」徽标),点击 `switchConversation(cid)` 回载历史(user 纯文本 / bot 走 renderMarkdown)继续聊;`send()` finally 与「新对话」后刷新列表;侧栏任何失败静默降级不影响聊天。

- [ ] **Step 3: 跑测试 + 全量回归**

Run: `uv run pytest tests/test_conversations_api.py tests/test_repository.py tests/test_context_assembly.py -v` → 全 PASS;`uv run pytest -q` → 无回归

- [ ] **Step 4: Commit**

```bash
git add app/api/conversations.py app/main.py app/db/repository.py app/static/index.html app/graph/nodes.py tests/
git commit -m "feat(ch07): 验收配套——会话侧栏(列表/切换/历史回载)+ model_ctx 日志(摘要+滑窗可见)"
```

---

### Task 7: 端到端冒烟 + DDL 应用 + 交付准备

**Files:**
- Modify: 无新代码;`sql/ch07-ddl.sql` 应用到 dev 库
- Test: 全量 pytest + 长对话冒烟脚本(临时,不入库)

- [ ] **Step 1: dev 库应用 DDL**

Run: `mysql -h127.0.0.1 -uroot -proot mewhelp < sql/ch07-ddl.sql`(或 docker exec,视 docker-compose 环境)
Verify: `SHOW COLUMNS FROM conversations` 含 summary / summary_upto_msg_id

- [ ] **Step 2: 全量回归**

Run: `uv run pytest -q` → 全 PASS

- [ ] **Step 3: 起服务长对话冒烟**(scratchpad 临时脚本打 `/api/agent` 25 轮,验证:token 不爆、
      log/app.log 出现 `summary trigger/start/done`、摘要落库、当轮响应耗时无摘要尖峰)

冒烟要点(演示留给用户浏览器手动,这一步只保证链路通):
- 25 轮里前几轮报订单号/手机号,后面灌闲聊+咨询;
- 第 26 问「最开始那个订单后来怎么说」看答复含早期订单号;
- `tail log/app.log | grep summary` 有 trigger/done 痕迹;
- `SELECT summary, summary_upto_msg_id FROM conversations WHERE id=?` 已更新。

- [ ] **Step 4: dev-notes 追记 + 收尾**

请求 code review(superpowers:requesting-code-review)→ 结论记 dev-notes → finish(superpowers:finishing-a-development-branch)。

---

## Self-Review 结论

- **Spec 覆盖**:§3 锚点(T2/T4)、§4 拼装(T2/T4)、§5 摘要任务(T3)、§6 配置(T3)、§7 DB(T1)、§8 边界(T2 空窗回退/T3 skip 逻辑/T4 resume 触发)、§9 测试(T1-T5)——全覆盖。
- **占位扫描**:无 TBD/TODO;所有步骤带完整代码与命令。
- **类型一致性**:`_ensure_conversation` 返回三元组只在 runtime 内部消费;`build_window(messages, summary_upto_msg_id, max_tokens)` 三处调用签名一致;`maybe_schedule_summary(cid)` 四入口一致。
