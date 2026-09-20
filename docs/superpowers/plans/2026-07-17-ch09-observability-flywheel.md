# Ch09 可观测性与数据飞轮 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给客服系统接 Langfuse 全链路观测 + 按意图 token 账,并建成"三入口低置信度问题池 → 标准化查重待审队列 → 人工审核写回知识库"的数据飞轮闭环,配自动化评估趋势流水线。

**Architecture:** 方案 A"回调为主、脚本为辅"——观测靠 `compile().with_config({"callbacks":[CallbackHandler()]})` 编译时挂一次吃全图(课程 README 姿势);飞轮/评估/成本统计是 make 驱动的批处理脚本;审核后台 = 静态单页 + REST。Spec:`docs/superpowers/specs/2026-07-17-ch09-observability-flywheel-design.md`,课程对齐:`mewhelp-course/ch09-observability-flywheel/README.md`。

**Tech Stack:** Langfuse v3 自部署(独立 docker compose)、langfuse Python SDK(`langfuse.langchain.CallbackHandler`)、LangGraph、SQLAlchemy 2.0 async、FastAPI、复用 ch03 dualwrite / ch04 评估集与指标 / ch05 图节点。

## Global Constraints

- **技术选型定死**:Langfuse 自部署(不用 LangSmith/云版);实现中发现走不通,停下来问用户,不许自行换方案。
- **API 用法先查证**:每个任务动手前,涉及 Langfuse/LangGraph/LangChain/FastAPI/SQLAlchemy 的具体 API,先用 Context7 MCP 查最新文档核对(计划里已按 2026-07 文档写好,执行时若与实际安装版本不符,以 Context7 + 实测为准并在 dev-notes 记录偏差)。
- **观测可选降级**:Langfuse 三配置(public_key/secret_key/host)不齐时一律不挂回调、不初始化 client,系统照常跑;所有 langfuse 调用点必须 guard,不许成为启动/请求硬依赖(测试环境无 Langfuse)。
- **DDL 已定稿**:`sql/ch09-ddl.sql`(用户手写,已提交)。表结构不改;ORM/代码适配它。`review_queue.review_status` 与 `eval_runs.triggered_by` 是中文值 ENUM。
- **查重语义(用户拍板)**:候选 = review_queue 全部行(截 200);命中任何状态的行都只 `occurrence_count+1`、状态不变(驳回即终审);未命中新建"待审"行。
- **飞轮是定时批处理**(用户拍板):脚本 + make;不引 APScheduler;不做落池后近实时触发。
- **快照语义**:`retrieved_snapshot` 只要走了知识检索就写 state(精排 Top3);落池时随条写入 `low_confidence_questions.retrieved_chunks`;没走检索的入口为 NULL。
- **阈值不拍脑袋**:`evidence_confidence_threshold` 默认值必须来自校准脚本在 ch04 评估集上的实跑输出(Task 6),不许随手写死。
- **中文 JSON 不转义**:所有面向人看的输出 `ensure_ascii=False`。
- **测试命令**:`uv run pytest -v`(全量)/ `uv run pytest tests/xxx -v`(单测)。每任务结束全量测试零回归再 commit。
- **过程留痕**:每任务完成即在 `dev-notes/ch09.md` 追记一段(四样:用户原话/关键产出/纠偏/翻车),不许收尾补记。dev-notes 目录被 gitignore,不随代码 commit。
- **前端例外**:Task 12(审核页)与 Task 7 的前端部分走 Vibe Coding,不套 TDD;做完浏览器人工验收。

---

### Task 1: ORM + conftest + repository 地基

**Files:**
- Modify: `app/db/models.py`(加 `ReviewQueue`、`EvalRun`,`LowConfidenceQuestion` 加两列)
- Modify: `app/db/repository.py`(`insert_low_confidence` 加快照参数 + 新增 10 个方法)
- Modify: `tests/conftest.py`(`_DDL_FILES` 挂 ch09-ddl.sql,`_TABLES` 加两表)
- Test: `tests/db/test_ch09_flywheel_tables.py`

**Interfaces:**
- Consumes: `sql/ch09-ddl.sql`(已定稿);`app.db.base`(Base/async_session 既有)。
- Produces(后续任务依赖的确切签名):
  - `class ReviewQueue`:`id:int, normalized_question:str, ai_suggested_answer:str|None, occurrence_count:int, review_status:str('待审'|'通过'|'驳回'), approved_answer:str|None, created_at, updated_at`
  - `class EvalRun`:`id:int, triggered_by:str('定时'|'手动'), dataset_size:int, metrics:dict, created_at`
  - `LowConfidenceQuestion` 新增:`retrieved_chunks: list|None`(JSON)、`matched_review_id: int|None`
  - `repository.insert_low_confidence(conversation_id, raw_question, source, reason, retrieved_chunks=None) -> int`
  - `repository.fetch_unmatched_low_conf(limit: int) -> list[LowConfidenceQuestion]`(matched_review_id IS NULL,按 id 升序)
  - `repository.list_review_candidates(limit: int = 200) -> list[dict]`(`[{"id":int,"normalized_question":str}]`,updated_at 倒序)
  - `repository.insert_review_item(normalized_question: str, ai_suggested_answer: str|None) -> int`
  - `repository.increment_occurrence(review_id: int) -> None`
  - `repository.set_matched_review(lcq_id: int, review_id: int) -> None`
  - `repository.list_review_queue(status: str|None) -> list[ReviewQueue]`(occurrence_count 降序;status=None 全量)
  - `repository.get_review_detail(review_id: int) -> tuple[ReviewQueue, list[LowConfidenceQuestion]]|None`
  - `repository.update_review_status(review_id: int, status: str, approved_answer: str|None = None) -> bool`(仅当前状态为"待审"才更新,返回是否更新)
  - `repository.insert_eval_run(triggered_by: str, dataset_size: int, metrics: dict) -> int`
  - `repository.list_eval_runs(limit: int = 10) -> list[EvalRun]`(created_at/id 倒序)

- [ ] **Step 1: conftest 挂 ch09 DDL**

`tests/conftest.py` 改两处:

```python
_DDL_FILES = [
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch02-ddl.sql",
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch03-ddl.sql",
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch04-ddl.sql",
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch07-ddl.sql",
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch08-ddl.sql",
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch09-ddl.sql",
]
# 删除顺序:先子表后父表;low_confidence_questions 有 FK 指向 review_queue,排它前面
_TABLES = ["low_confidence_questions", "review_queue", "eval_runs", "messages", "tickets",
           "tool_audit_logs", "conversations", "faq", "qa_extraction_staging", "knowledge_chunks"]
```

- [ ] **Step 2: 写失败测试**

`tests/db/test_ch09_flywheel_tables.py`:

```python
"""ch09 地基:review_queue / eval_runs ORM 与 repository 飞轮方法。"""
import pytest

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
```

- [ ] **Step 3: 跑测试确认失败**

Run: `uv run pytest tests/db/test_ch09_flywheel_tables.py -v`
Expected: FAIL(`ReviewQueue` 不存在 / repository 无新方法 / 表缺列)。

- [ ] **Step 4: 实现 ORM**

`app/db/models.py` 追加(并给 `LowConfidenceQuestion` 加两列):

```python
class ReviewQueue(Base):
    """ch09 飞轮待审队列:一行 = 一个去重后的知识缺口;查重命中累加 occurrence_count 不新建行。"""
    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    normalized_question: Mapped[str] = mapped_column(String(512))
    ai_suggested_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    occurrence_count: Mapped[int] = mapped_column(Integer, server_default="1")
    review_status: Mapped[str] = mapped_column(Enum("待审", "通过", "驳回"), server_default="待审")
    approved_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class EvalRun(Base):
    """ch09 自动化评估流水线:一行 = 一轮评估;metrics JSON 收各指标,按时间连成趋势。"""
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    triggered_by: Mapped[str] = mapped_column(Enum("定时", "手动"), server_default="定时")
    dataset_size: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

`LowConfidenceQuestion` 加两列(放 `reason` 之后,与 DDL 对齐):

```python
    retrieved_chunks: Mapped[list | None] = mapped_column(JSON, nullable=True)   # ch09 落池时召回快照
    matched_review_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("review_queue.id"), nullable=True)               # ch09 查重归并落点
```

- [ ] **Step 5: 实现 repository 方法**

`app/db/repository.py`:import 行加 `EvalRun, ReviewQueue`;`insert_low_confidence` 加参数;文件尾追加 ch09 段:

```python
async def insert_low_confidence(
    conversation_id: int | None, raw_question: str, source: str, reason: str | None,
    retrieved_chunks: list | None = None,
) -> int:
    async with db.async_session() as s:
        row = LowConfidenceQuestion(
            conversation_id=conversation_id, raw_question=raw_question,
            source=source, reason=reason, retrieved_chunks=retrieved_chunks,
        )
        s.add(row)
        await s.commit()
        await s.refresh(row)
        return row.id


# ---- ch09 数据飞轮(待审队列 + 评估轮次)----


async def fetch_unmatched_low_conf(limit: int) -> list[LowConfidenceQuestion]:
    """飞轮处理游标:尚未归并(matched_review_id IS NULL)的池内问题,按 id 升序。"""
    async with db.async_session() as s:
        result = await s.execute(
            select(LowConfidenceQuestion)
            .where(LowConfidenceQuestion.matched_review_id.is_(None))
            .order_by(LowConfidenceQuestion.id).limit(limit)
        )
        return list(result.scalars())


async def list_review_candidates(limit: int = 200) -> list[dict]:
    """查重候选:全部状态的缺口行(用户拍板:比全部,驳回即终审),updated_at 倒序截断防 token 爆。"""
    async with db.async_session() as s:
        result = await s.execute(
            select(ReviewQueue.id, ReviewQueue.normalized_question)
            .order_by(ReviewQueue.updated_at.desc()).limit(limit)
        )
        return [{"id": r.id, "normalized_question": r.normalized_question} for r in result]


async def insert_review_item(normalized_question: str, ai_suggested_answer: str | None) -> int:
    async with db.async_session() as s:
        row = ReviewQueue(normalized_question=normalized_question,
                          ai_suggested_answer=ai_suggested_answer)
        s.add(row)
        await s.commit()
        return row.id


async def increment_occurrence(review_id: int) -> None:
    """查重命中:只累加次数,状态不动(命中已驳回/已通过也一样——驳回即终审)。"""
    async with db.async_session() as s:
        row = await s.get(ReviewQueue, review_id)
        if row is not None:
            row.occurrence_count = row.occurrence_count + 1
            await s.commit()


async def set_matched_review(lcq_id: int, review_id: int) -> None:
    async with db.async_session() as s:
        row = await s.get(LowConfidenceQuestion, lcq_id)
        if row is not None:
            row.matched_review_id = review_id
            await s.commit()


async def list_review_queue(status: str | None) -> list[ReviewQueue]:
    """审核页列表:按出现次数降序(频次=优先级),同频新的在前。"""
    async with db.async_session() as s:
        q = select(ReviewQueue).order_by(
            ReviewQueue.occurrence_count.desc(), ReviewQueue.id.desc())
        if status:
            q = q.where(ReviewQueue.review_status == status)
        return list((await s.execute(q)).scalars())


async def get_review_detail(review_id: int) -> tuple[ReviewQueue, list[LowConfidenceQuestion]] | None:
    """详情:缺口行 + 归并进来的原话流水(带 source/快照,审核人判断真缺还是没检到)。"""
    async with db.async_session() as s:
        item = await s.get(ReviewQueue, review_id)
        if item is None:
            return None
        raws = list((await s.execute(
            select(LowConfidenceQuestion)
            .where(LowConfidenceQuestion.matched_review_id == review_id)
            .order_by(LowConfidenceQuestion.id)
        )).scalars())
        return item, raws


async def update_review_status(review_id: int, status: str, approved_answer: str | None = None) -> bool:
    """仅「待审」可流转(通过/驳回都是终态);返回是否真的更新了。"""
    async with db.async_session() as s:
        row = await s.get(ReviewQueue, review_id)
        if row is None or row.review_status != "待审":
            return False
        row.review_status = status
        if approved_answer is not None:
            row.approved_answer = approved_answer
        await s.commit()
        return True


async def insert_eval_run(triggered_by: str, dataset_size: int, metrics: dict) -> int:
    async with db.async_session() as s:
        row = EvalRun(triggered_by=triggered_by, dataset_size=dataset_size, metrics=metrics)
        s.add(row)
        await s.commit()
        return row.id


async def list_eval_runs(limit: int = 10) -> list[EvalRun]:
    async with db.async_session() as s:
        result = await s.execute(select(EvalRun).order_by(EvalRun.id.desc()).limit(limit))
        return list(result.scalars())
```

- [ ] **Step 6: 跑测试确认通过 + 全量零回归**

Run: `uv run pytest tests/db/test_ch09_flywheel_tables.py -v` → 全 PASS。
Run: `uv run pytest -v` → 零回归(注意既有 LCQ 相关测试不受 insert_low_confidence 新可选参影响)。

- [ ] **Step 7: 给 dev 库应用 DDL + Commit**

```bash
docker exec -i mewhelp-mysql mysql --default-character-set=utf8mb4 -uroot -proot mewhelp < sql/ch09-ddl.sql
git add app/db/models.py app/db/repository.py tests/conftest.py tests/db/test_ch09_flywheel_tables.py
git commit -m "feat(ch09): 飞轮地基——review_queue/eval_runs ORM + 问题池快照/归并列 + repository 十方法"
```

---

### Task 2: Langfuse 部署物 + 配置 + 编译时挂回调

**Files:**
- Create: `docker-compose.langfuse.yml`
- Create: `app/core/observability.py`
- Modify: `app/config.py`(3 个 langfuse 配置)
- Modify: `app/graph/runtime.py`(编译后挂回调 + 三处 config 加 session metadata)
- Modify: `Makefile`(langfuse-up / langfuse-down)
- Modify: `pyproject.toml`(uv add langfuse)
- Test: `tests/core/test_observability.py`

**Interfaces:**
- Consumes: `build_graph(checkpointer)`(既有);`settings`。
- Produces:
  - `observability.langfuse_enabled() -> bool`
  - `observability.attach_observability(graph) -> graph`(未启用原样返回;启用则初始化 Langfuse 单例 + `graph.with_config({"callbacks":[CallbackHandler()]})`)
  - `observability.tag_intent(intent: str, confidence: float) -> None`(Task 3 用;本任务一并产出)
  - `observability.get_langfuse()`(启用时返回已初始化的 langfuse client 单例,供 cost 脚本用;未启用返回 None)
  - runtime 三处 invoke/astream 的 config 均含 `"metadata": {"langfuse_session_id": str(cid)}`

- [ ] **Step 1: Context7 查证(执行时必做)**

用 Context7 查 `/langfuse/langfuse-python` 确认两点(2026-07 文档口径,执行时以实装版本为准):
1. `Langfuse(public_key=..., secret_key=..., ...)` 的地址构造参名(`host` 还是 `base_url`)——环境变量口径按课程 README 与 SDK 文档一致的 `LANGFUSE_BASE_URL`(settings 字段 `langfuse_base_url`),构造参名以 SDK 实际签名为准;
2. `langfuse.langchain.CallbackHandler`(无参构造,复用已初始化单例)与 `get_client().update_current_trace(metadata=..., tags=...)` 可用性。

- [ ] **Step 2: 加依赖**

```bash
uv add langfuse
```

- [ ] **Step 3: settings 加配置**

`app/config.py` 追加(ch08 段之后):

```python
    # ch09 可观测(Langfuse 自部署;三者齐全才挂回调,缺省时系统照常跑、测试环境不依赖)
    # 环境变量名与课程 README 一致:LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = ""     # 如 http://localhost:3000(自部署地址,数据不出门)
```

- [ ] **Step 4: 写失败测试**

`tests/core/test_observability.py`:

```python
"""ch09 观测挂载:缺配置不挂回调、不碰 langfuse;齐配置才 with_config。"""
from app.core import observability


def test_disabled_when_keys_missing(monkeypatch):
    monkeypatch.setattr("app.config.settings.langfuse_public_key", "")
    monkeypatch.setattr("app.config.settings.langfuse_secret_key", "sk")
    monkeypatch.setattr("app.config.settings.langfuse_base_url", "http://x")
    assert observability.langfuse_enabled() is False


def test_attach_returns_graph_unchanged_when_disabled(monkeypatch):
    monkeypatch.setattr("app.config.settings.langfuse_public_key", "")
    sentinel = object()
    assert observability.attach_observability(sentinel) is sentinel


def test_tag_intent_noop_when_disabled(monkeypatch):
    monkeypatch.setattr("app.config.settings.langfuse_public_key", "")
    observability.tag_intent("商品咨询", 0.9)   # 不许抛异常、不许要求 langfuse 已装好服务


def test_attach_wraps_with_callbacks_when_enabled(monkeypatch):
    monkeypatch.setattr("app.config.settings.langfuse_public_key", "pk")
    monkeypatch.setattr("app.config.settings.langfuse_secret_key", "sk")
    monkeypatch.setattr("app.config.settings.langfuse_base_url", "http://localhost:3000")

    calls = {}

    class FakeGraph:
        def with_config(self, cfg):
            calls["cfg"] = cfg
            return "wrapped"

    monkeypatch.setattr(observability, "_init_client", lambda: object())
    monkeypatch.setattr(observability, "_make_handler", lambda: "HANDLER")
    assert observability.attach_observability(FakeGraph()) == "wrapped"
    assert calls["cfg"] == {"callbacks": ["HANDLER"]}
```

Run: `uv run pytest tests/core/test_observability.py -v` → FAIL(模块不存在)。

- [ ] **Step 5: 实现 observability 模块**

`app/core/observability.py`:

```python
"""ch09 可观测:Langfuse 挂载与 trace 标注,全部可选降级。

课程 README 姿势:设三个环境变量 + 图编译时挂一次回调,节点业务代码零侵入。
项目配置经 pydantic-settings(.env),不依赖 os.environ,故显式传参初始化单例。
所有对外函数在未配置 Langfuse 时必须是安全 no-op——观测是增强,不是依赖。
"""
import logging

from app.config import settings

logger = logging.getLogger(__name__)
_client = None


def langfuse_enabled() -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key
                and settings.langfuse_base_url)


def _init_client():
    """初始化(或复用)Langfuse 单例。构造参名以 Context7 查证为准(host / base_url)。"""
    global _client
    if _client is None:
        from langfuse import Langfuse
        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_base_url,   # 构造参名 host/base_url 以 Context7 查证为准
        )
    return _client


def _make_handler():
    from langfuse.langchain import CallbackHandler
    return CallbackHandler()


def get_langfuse():
    """启用时返回已初始化 client(cost 脚本用 client.api 查询);未启用返回 None。"""
    if not langfuse_enabled():
        return None
    return _init_client()


def attach_observability(graph):
    """编译后挂一次 Langfuse 回调,全图自动 trace(README:「编译时挂一次,节点里一行不用动」)。
    未配置时原样返回——不挂、不 import langfuse。"""
    if not langfuse_enabled():
        return graph
    _init_client()
    return graph.with_config({"callbacks": [_make_handler()]})


def tag_intent(intent: str, confidence: float) -> None:
    """把意图写进当前 trace 的 metadata + tag(Cost Control 按意图分堆的钩子)。
    任何异常静默降级——观测失败不许影响业务流。"""
    if not langfuse_enabled():
        return
    try:
        from langfuse import get_client
        get_client().update_current_trace(
            metadata={"intent": intent, "intent_confidence": confidence},
            tags=[f"intent:{intent}"],
        )
    except Exception:
        logger.debug("langfuse tag_intent 失败(已忽略)", exc_info=True)
```

- [ ] **Step 6: runtime 挂载 + session 归属**

`app/graph/runtime.py`:
1. import 加 `from app.core.observability import attach_observability`;
2. `init_graph()` 里 `_graph = build_graph(checkpointer=checkpointer)` 改为:

```python
    _graph = attach_observability(build_graph(checkpointer=checkpointer))
```

3. 三处 config(`run_turn` L107、`resume_turn` L118、`_stream_events` L129)统一改为带 session 归属(langfuse 未启用时多余 metadata 无害):

```python
    config = {"configurable": {"thread_id": str(cid)},
              "metadata": {"langfuse_session_id": str(cid)}}
```

(`resume_turn`/`_stream_events` 里变量名分别是 `conversation_id`/`cid`,照各自变量写。)

- [ ] **Step 7: 跑测试**

Run: `uv run pytest tests/core/test_observability.py -v` → PASS;`uv run pytest -v` → 零回归。

- [ ] **Step 8: 部署物(compose + make)**

以官方 v3 compose 为底拿到 `docker-compose.langfuse.yml`:

```bash
curl -fsSL https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml -o docker-compose.langfuse.yml
```

然后按下面清单改(执行时对照 Context7 `/websites/langfuse_self-hosting` 的 headless initialization 文档核对变量名):
1. 给 `langfuse-web` 服务的 environment 追加 headless 初始化变量,起来即有组织/项目/固定 key,免手工点界面:

```yaml
      LANGFUSE_INIT_ORG_ID: mewhelp
      LANGFUSE_INIT_ORG_NAME: MewHelp
      LANGFUSE_INIT_PROJECT_ID: mewhelp-ch09
      LANGFUSE_INIT_PROJECT_NAME: mewhelp
      LANGFUSE_INIT_PROJECT_PUBLIC_KEY: pk-lf-mewhelp-local
      LANGFUSE_INIT_PROJECT_SECRET_KEY: sk-lf-mewhelp-local
      LANGFUSE_INIT_USER_EMAIL: admin@mewhelp.local
      LANGFUSE_INIT_USER_NAME: admin
      LANGFUSE_INIT_USER_PASSWORD: mewhelp123
```

2. 若与现有 mysql/milvus 栈端口冲突(3000/5432/6379/9000/9090),只改宿主侧映射,容器内不动;minio 与 milvus 的 minio 同名冲突时给服务/卷名加 `langfuse-` 前缀。
3. 文件头加注释:`# ch09 Langfuse 自部署(独立栈,数据不出门):make langfuse-up / langfuse-down`。

`Makefile` `.PHONY` 行追加 `langfuse-up langfuse-down`,并加:

```makefile
# ch09: Langfuse 自部署观测栈(web:3000 + worker + postgres + clickhouse + redis + minio)
langfuse-up:
	docker compose -f docker-compose.langfuse.yml up -d
	@echo "Langfuse 起中: http://localhost:3000 (admin@mewhelp.local / mewhelp123)"
	@echo "首次就绪约 2-3 分钟;key 已 headless 预置,写 .env:"
	@echo "  LANGFUSE_PUBLIC_KEY=pk-lf-mewhelp-local"
	@echo "  LANGFUSE_SECRET_KEY=sk-lf-mewhelp-local"
	@echo "  LANGFUSE_BASE_URL=http://localhost:3000"

langfuse-down:
	docker compose -f docker-compose.langfuse.yml down
```

- [ ] **Step 9: 冒烟 + Commit**

```bash
make langfuse-up   # 等 2-3 分钟
curl -sf http://localhost:3000/api/public/health && echo OK
```

`.env` 写入三个 LANGFUSE_ 变量(settings 字段名小写对应)。起服务(`make dev`,需 milvus/mysql 在跑、上游可用)发一条消息,登录 localhost:3000 确认出现 trace 树(节点层级 + LLM span 带 prompt/usage)。**这是验收 1 的首次点亮**,截图路径记进 dev-notes。

```bash
git add docker-compose.langfuse.yml app/core/observability.py app/config.py app/graph/runtime.py Makefile pyproject.toml uv.lock tests/core/test_observability.py
git commit -m "feat(ch09): Langfuse 自部署栈 + 编译时挂一次回调,全图 trace 零侵入"
```

---

### Task 3: token usage 收口 + 意图打进 trace

**Files:**
- Modify: `app/core/llm.py`(`stream_usage=True` 进构造)
- Modify: `app/graph/nodes.py`(`main_agent` 去重复 bind;`classify_intent` 调 `tag_intent`)
- Test: `tests/core/test_llm_usage.py`

**Interfaces:**
- Consumes: `observability.tag_intent`(Task 2)。
- Produces: `get_chat_model(streaming=False, model=None) -> ChatOpenAI`(行为不变 + 流式回传 usage);`classify_intent` 返回值形状不变。

- [ ] **Step 1: Context7 查证**

查 `/langchain-ai/langchain` 或 langchain-openai 文档:确认 `ChatOpenAI(stream_usage=True)` 构造参存在且等效于 `stream_options={"include_usage": True}`(现装 langchain 1.3 系)。若该参不存在,回退方案:保留 main_agent 的 `.bind(stream_options=...)` 原样,仅在 llm.py 加注释说明为何不收口,并在 dev-notes 记录。

- [ ] **Step 2: 写失败测试**

`tests/core/test_llm_usage.py`:

```python
"""ch09 Cost Control 前提:所有模型调用(含流式)都回传 usage,按意图统计才不漏账。"""
from app.core.llm import get_chat_model


def test_streaming_model_requests_usage():
    m = get_chat_model(streaming=True)
    assert m.stream_usage is True


def test_non_streaming_model_also_flags_usage():
    assert get_chat_model().stream_usage is True
```

Run: `uv run pytest tests/core/test_llm_usage.py -v` → FAIL。

- [ ] **Step 3: 实现**

`app/core/llm.py` 构造参数加一行(docstring 补一句"ch09:usage 统一收口,流式也回传 token,Langfuse 按意图统计才不漏账"):

```python
        streaming=streaming,
        stream_usage=True,   # ch09:流式也回传 usage(等效 stream_options.include_usage)
        temperature=0.3,
```

`app/graph/nodes.py` `main_agent`:删掉 `.bind(stream_options={"include_usage": True})` 一行(收口后冗余),model 构造变为:

```python
    model = get_chat_model(streaming=True).bind_tools([s.tool for s in specs])
```

docstring 里那句"须显式 bind stream_options"改为"usage 回传已在 get_chat_model 统一收口(ch09)"。

- [ ] **Step 4: classify_intent 打意图**

`app/graph/nodes.py` 顶部 import 加 `from app.core.observability import tag_intent`;`classify_intent` return 前加:

```python
    tag_intent(intent, conf)   # ch09:意图进当前 trace 的 metadata+tag,Cost Control 按意图分堆
```

- [ ] **Step 5: 测试 + 冒烟 + Commit**

Run: `uv run pytest -v` → 零回归(重点看 tests/graph 里 main_agent/意图相关用例)。
冒烟:起服务发一条消息,Langfuse 里该 trace 的 metadata 出现 `intent`、tags 出现 `intent:xxx`,main_agent 的 generation span 有 usage 数字。

```bash
git add app/core/llm.py app/graph/nodes.py tests/core/test_llm_usage.py
git commit -m "feat(ch09): usage 统一收口 + 意图写进 trace metadata/tag(Cost Control 地基)"
```

---

### Task 4: evidence_confidence 纯函数(TDD)

**Files:**
- Create: `app/core/confidence.py`
- Modify: `app/config.py`(阈值配置,占位默认 0.5,Task 6 校准后改)
- Test: `tests/core/test_confidence.py`

**Interfaces:**
- Consumes: 无(纯函数,吃 `search_knowledge` 的 hit dict:`{question, answer, rerank_score, ...}`)。
- Produces:
  - `@dataclass EvidenceConfidence: score: float; signals: dict`
  - `compute_evidence_confidence(hits: list[dict]) -> EvidenceConfidence`
  - `settings.evidence_confidence_threshold: float`
  - `confidence.snapshot_from_hits(hits: list[dict], top_n: int = 3) -> list[dict]`(落池快照的统一出口)

- [ ] **Step 1: 写失败测试**

`tests/core/test_confidence.py`:

```python
"""ch09 置信度闸:四信号(Top1 分/有效证据数/分差/关键条款)加权,纯函数可复现。"""
import pytest

from app.core.confidence import EvidenceConfidence, compute_evidence_confidence, snapshot_from_hits


def _hit(score, q="退货运费谁出", a="满99包邮,退货运费买家承担"):
    return {"question": q, "answer": a, "rerank_score": score,
            "section_path": "售后 / 退货", "id": 1, "content_type": "faq"}


def test_empty_hits_zero_confidence():
    r = compute_evidence_confidence([])
    assert r.score == 0.0
    assert r.signals == {"top1_score": 0.0, "valid_count": 0, "margin": 0.0, "key_clause_hit": False}


def test_strong_evidence_scores_high():
    r = compute_evidence_confidence([_hit(0.95), _hit(0.40), _hit(0.35)])
    assert r.score > 0.7
    assert r.signals["top1_score"] == 0.95
    assert r.signals["margin"] == pytest.approx(0.55)
    assert r.signals["key_clause_hit"] is True   # 「退货」「运费」命中关键条款词表


def test_weak_flat_evidence_scores_low():
    hits = [_hit(0.22, q="猫粮口味", a="三文鱼味与鸡肉味"),
            _hit(0.21, q="猫粮口味", a="三文鱼味与鸡肉味")]
    r = compute_evidence_confidence(hits)
    assert r.score < 0.4
    assert r.signals["valid_count"] == 0          # 全部低于有效线
    assert r.signals["key_clause_hit"] is False


def test_single_hit_margin_falls_back_to_top1():
    r = compute_evidence_confidence([_hit(0.8)])
    assert r.signals["margin"] == pytest.approx(0.8)


def test_score_monotonic_in_top1():
    low = compute_evidence_confidence([_hit(0.3), _hit(0.2)])
    high = compute_evidence_confidence([_hit(0.9), _hit(0.2)])
    assert high.score > low.score


def test_score_bounded_zero_one():
    r = compute_evidence_confidence([_hit(1.0), _hit(0.0)])
    assert 0.0 <= r.score <= 1.0


def test_snapshot_from_hits_top3_shape():
    hits = [_hit(0.9), _hit(0.8), _hit(0.7), _hit(0.6)]
    snap = snapshot_from_hits(hits)
    assert len(snap) == 3
    assert snap[0] == {"question": "退货运费谁出", "answer": "满99包邮,退货运费买家承担",
                       "rerank_score": 0.9, "section_path": "售后 / 退货"}


def test_snapshot_empty_hits():
    assert snapshot_from_hits([]) == []
```

Run: `uv run pytest tests/core/test_confidence.py -v` → FAIL。

- [ ] **Step 2: 实现**

`app/core/confidence.py`:

```python
"""ch09 证据置信度(README 四信号):量化「这批证据到底符不符合用户问题」。

信号(全部来自检索/精排结果,零额外模型调用):
  top1_score     精排 Top1 的 rerank_score(证据里最强的一条有多强)
  valid_count    有效证据数:rerank_score >= VALID_SCORE_FLOOR 的条数(强证据是孤证还是成群)
  margin         Top1 - Top2 分差(证据聚焦度:断崖式领先 vs 一堆似是而非;单条时取 top1 自身)
  key_clause_hit Top3 文本是否命中关键条款词表(复用 ch03 入库同款 _KEY_TERMS 启发式——
                 hit 里没有 is_key_clause 字段,按同词表现算,口径与入库标记一致)

组合:固定权重线性加权归一到 0-1。权重是模块常量不进 settings——评估集校准的是
阈值(settings.evidence_confidence_threshold),不是权重(YAGNI,别造两层可调)。
"""
from dataclasses import dataclass

from app.kb.documents import _KEY_TERMS

VALID_SCORE_FLOOR = 0.3   # 有效证据线(沿用 ch04 rerank_min_score 的经验位)
VALID_COUNT_CAP = 3       # 有效证据数归一封顶:3 条及以上算满分
W_TOP1, W_VALID, W_MARGIN, W_KEY = 0.5, 0.2, 0.2, 0.1


@dataclass
class EvidenceConfidence:
    score: float    # 0-1 总分
    signals: dict   # 各原始信号,落池 reason 与 trace 留痕用


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_evidence_confidence(hits: list[dict]) -> EvidenceConfidence:
    if not hits:
        return EvidenceConfidence(0.0, {"top1_score": 0.0, "valid_count": 0,
                                        "margin": 0.0, "key_clause_hit": False})
    scores = [float(h.get("rerank_score", 0.0)) for h in hits]
    top1 = scores[0]
    margin = top1 - scores[1] if len(scores) > 1 else top1
    valid_count = sum(1 for s in scores if s >= VALID_SCORE_FLOOR)
    key_hit = any(
        any(t in f"{h.get('question', '')}{h.get('answer', '')}" for t in _KEY_TERMS)
        for h in hits[:3]
    )
    score = (W_TOP1 * _clip01(top1)
             + W_VALID * min(valid_count, VALID_COUNT_CAP) / VALID_COUNT_CAP
             + W_MARGIN * _clip01(margin)
             + W_KEY * (1.0 if key_hit else 0.0))
    return EvidenceConfidence(round(_clip01(score), 4),
                              {"top1_score": top1, "valid_count": valid_count,
                               "margin": round(margin, 4), "key_clause_hit": key_hit})


def snapshot_from_hits(hits: list[dict], top_n: int = 3) -> list[dict]:
    """落池召回快照的统一出口:Top N 的原文与得分(审核页给人看的那份)。"""
    return [{"question": h.get("question", ""), "answer": h.get("answer", ""),
             "rerank_score": float(h.get("rerank_score", 0.0)),
             "section_path": h.get("section_path", "")}
            for h in hits[:top_n]]
```

`app/config.py` ch09 段追加:

```python
    # ch09 置信度闸(正式版):阈值由 make calibrate-confidence 在 ch04 评估集上校准后回填,不拍脑袋
    evidence_confidence_threshold: float = 0.5   # 占位;Task 6 校准后改为实测推荐值
```

注意 `test_weak_flat_evidence_scores_low` 里 0.22/0.21 的组合分 ≈ 0.5*0.22+0+0.2*0.01+0 ≈ 0.112 < 0.4,`test_strong_evidence_scores_high` ≈ 0.5*0.95+0.2*(1/3)+0.2*0.55+0.1 ≈ 0.752 > 0.7,断言与权重自洽;若调权重需同步核这两条。

- [ ] **Step 3: 跑测试 + Commit**

Run: `uv run pytest tests/core/test_confidence.py -v` → PASS;全量零回归。

```bash
git add app/core/confidence.py app/config.py tests/core/test_confidence.py
git commit -m "feat(ch09): evidence_confidence 四信号纯函数 + 快照出口(TDD)"
```

---

### Task 5: 置信度闸接线(state + retrieve_knowledge + fallback_reply)

**Files:**
- Modify: `app/graph/state.py`(3 个新字段)
- Modify: `app/graph/runtime.py`(`_graph_input` 重置新字段)
- Modify: `app/graph/nodes.py`(`retrieve_knowledge`、`fallback_reply`)
- Test: `tests/graph/test_ch09_confidence_gate.py`

**Interfaces:**
- Consumes: `compute_evidence_confidence` / `snapshot_from_hits`(Task 4);`repository.insert_low_confidence(..., retrieved_chunks=)`(Task 1)。
- Produces(state 新字段,Task 7 回捞与 fallback 落池依赖):
  - `evidence_confidence: float`(本轮检索证据总分)
  - `fallback_source: str`("retrieval_low_conf" | "self_check";兜底落池的 source)
  - `retrieved_snapshot: list`(走了检索就有,Top3 快照;没走检索为空列表)

- [ ] **Step 1: 写失败测试**

`tests/graph/test_ch09_confidence_gate.py`(参考 tests/graph 既有 monkeypatch 风格,mock 掉检索/自评/落池,不碰网络):

```python
"""ch09 闸升级:置信度低→retrieval_low_conf;置信度过但自评不过→self_check;
快照只要走了检索就写 state;fallback 落池带对 source 与快照。"""
import pytest

from app.graph import nodes


def _mk_hits(*scores):
    return [{"id": i, "question": f"q{i}", "answer": f"a{i}", "rerank_score": s,
             "section_path": "s", "content_type": "faq"} for i, s in enumerate(scores)]


@pytest.fixture
def stub_retrieval(monkeypatch):
    async def fake_understand(q):
        return {"standard": q, "expanded": []}
    monkeypatch.setattr(nodes.query_understanding, "understand", fake_understand)

    def set_hits(hits):
        async def fake_search(*a, **k):
            return hits
        monkeypatch.setattr(nodes.retrieval, "search_knowledge", fake_search)
    return set_hits


@pytest.fixture
def stub_selfcheck(monkeypatch):
    def set_result(useful, reason=""):
        async def fake_check(q, ev):
            return {"useful": useful, "reason": reason}
        monkeypatch.setattr(nodes.selfcheck, "check_sufficient", fake_check)
    return set_result


async def test_low_confidence_blocks_with_source_and_snapshot(stub_retrieval, stub_selfcheck, monkeypatch):
    monkeypatch.setattr("app.config.settings.evidence_confidence_threshold", 0.5)
    stub_retrieval(_mk_hits(0.15, 0.14))          # 弱证据 → 置信度闸拦下
    stub_selfcheck(True)
    out = await nodes.retrieve_knowledge({"messages": [], "conversation_id": 1})
    assert out["evidence_strong"] is False
    assert out["fallback_source"] == "retrieval_low_conf"
    assert len(out["retrieved_snapshot"]) == 2    # 走了检索就有快照
    assert out["evidence_confidence"] < 0.5
    assert "confidence_signals" in out["trace"]


async def test_selfcheck_fail_labels_self_check(stub_retrieval, stub_selfcheck, monkeypatch):
    monkeypatch.setattr("app.config.settings.evidence_confidence_threshold", 0.3)
    stub_retrieval(_mk_hits(0.9, 0.3, 0.3))
    stub_selfcheck(False, "缺关键信息")
    out = await nodes.retrieve_knowledge({"messages": [], "conversation_id": 1})
    assert out["evidence_strong"] is False
    assert out["fallback_source"] == "self_check"
    assert out["retrieved_snapshot"]              # 自评不过同样有快照


async def test_strong_evidence_keeps_snapshot_for_feedback(stub_retrieval, stub_selfcheck, monkeypatch):
    monkeypatch.setattr("app.config.settings.evidence_confidence_threshold", 0.3)
    stub_retrieval(_mk_hits(0.9, 0.4, 0.3))
    stub_selfcheck(True)
    out = await nodes.retrieve_knowledge({"messages": [], "conversation_id": 1})
    assert out["evidence_strong"] is True
    assert len(out["retrieved_snapshot"]) == 3    # 闸都过也留快照,供事后 👎 回捞
    assert out["citations"]


async def test_fallback_reply_persists_source_and_snapshot(monkeypatch):
    recorded = {}
    async def fake_insert(cid, q, source, reason, retrieved_chunks=None):
        recorded.update(cid=cid, q=q, source=source, reason=reason, chunks=retrieved_chunks)
        return 1
    monkeypatch.setattr(nodes.repository, "insert_low_confidence", fake_insert)
    from langchain_core.messages import HumanMessage
    state = {"messages": [HumanMessage("猫窝能水洗吗")], "conversation_id": 7,
             "fallback_source": "self_check", "evidence_confidence": 0.41,
             "retrieved_snapshot": [{"question": "x", "answer": "y", "rerank_score": 0.4, "section_path": "s"}],
             "trace": {}}
    out = await nodes.fallback_reply(state)
    assert recorded["source"] == "self_check"
    assert recorded["chunks"] and recorded["chunks"][0]["question"] == "x"
    assert "0.41" in recorded["reason"] or "0.410" in recorded["reason"]
    assert out["suggested_actions"] == [{"type": "transfer_human"}]


async def test_fallback_reply_defaults_source_when_missing(monkeypatch):
    """非知识路兜底(如无 fallback_source 的旧路径)不许炸,回落 retrieval_low_conf。"""
    recorded = {}
    async def fake_insert(cid, q, source, reason, retrieved_chunks=None):
        recorded["source"] = source; recorded["chunks"] = retrieved_chunks
        return 1
    monkeypatch.setattr(nodes.repository, "insert_low_confidence", fake_insert)
    from langchain_core.messages import HumanMessage
    out = await nodes.fallback_reply({"messages": [HumanMessage("嗯")], "conversation_id": 1, "trace": {}})
    assert recorded["source"] == "retrieval_low_conf"
    assert recorded["chunks"] is None             # 没快照就存 NULL,不存 []
```

Run: `uv run pytest tests/graph/test_ch09_confidence_gate.py -v` → FAIL。

- [ ] **Step 2: state 新字段 + 入口重置**

`app/graph/state.py` `ConversationState` 追加:

```python
    evidence_confidence: float  # ch09 正式置信度闸:四信号加权总分(0-1)
    fallback_source: str        # ch09 兜底落池 source:retrieval_low_conf | self_check
    retrieved_snapshot: list    # ch09 召回快照(Top3 原文+得分):走了检索就有,落池/👎回捞共用
```

`app/graph/runtime.py` `_graph_input` 返回 dict 里补重置(与 `evidence_strong` 同段):

```python
            "evidence_strong": False, "citations": [],
            "evidence_confidence": 0.0, "fallback_source": "", "retrieved_snapshot": [],
```

- [ ] **Step 3: 改 retrieve_knowledge**

`app/graph/nodes.py`:import 加 `from app.core.confidence import compute_evidence_confidence, snapshot_from_hits`。`retrieve_knowledge` 整体替换为:

```python
async def retrieve_knowledge(state) -> dict:
    """知识类强制检索:产出编号证据 + 证据强弱信号。ch09 把 ch05 最简机械闸升级成
    正式置信度闸(四信号加权,阈值评估集校准),闸位不动:检索后、进 Agent 前。
    快照规则:只要走了检索就存 Top3 快照——闸不过随落池写库;闸都过留给事后 👎 回捞。"""
    query_raw = _user_text(state)
    u = await query_understanding.understand(query_raw)
    query = u["standard"]
    bm25_text = query + (" " + " ".join(u["expanded"]) if u["expanded"] else "")

    hits = await retrieval.search_knowledge(
        query, strategy="hybrid_rerank", bm25_text=bm25_text)
    conf = compute_evidence_confidence(hits)
    snapshot = snapshot_from_hits(hits)
    base = {"evidence_confidence": conf.score, "retrieved_snapshot": snapshot}

    # 置信度闸(原机械闸升级):四信号加权总分低于校准阈值 → 证据弱,拒答落池
    if conf.score < settings.evidence_confidence_threshold:
        return {**base, "evidence_strong": False, "fallback_source": "retrieval_low_conf",
                "trace": {"forced_rag": True, "evidence_confidence": conf.score,
                          "confidence_signals": conf.signals}}

    # 语义闸:生成前自评证据够不够(README 的 useful 判断;ch09 修正:这才是 self_check 入口)
    ev_texts = [f"{h['question']} {h['answer']}" for h in hits]
    chk = await selfcheck.check_sufficient(query, ev_texts)
    if not chk["useful"]:
        return {**base, "evidence_strong": False, "fallback_source": "self_check",
                "trace": {"forced_rag": True, "evidence_confidence": conf.score,
                          "confidence_signals": conf.signals, "self_check": chk["reason"]}}

    arranged = retrieval.arrange_head_tail(hits)
    citations = [
        {"n": i + 1, "id": h["id"], "section_path": h["section_path"],
         "question": h["question"], "answer": h["answer"], "content_type": h["content_type"]}
        for i, h in enumerate(arranged)
    ]
    evidence = "\n".join(f"[{c['n']}] {c['question']}: {c['answer']}" for c in citations)
    return {**base, "evidence_strong": True, "evidence": evidence, "citations": citations,
            "trace": {"forced_rag": True, "evidence_confidence": conf.score,
                      "confidence_signals": conf.signals}}
```

- [ ] **Step 4: 改 fallback_reply**

```python
async def fallback_reply(state) -> dict:
    """置信度兜底:证据弱回兜底话术,并把问题落低置信池留给数据飞轮(ch09:source 打对标签
    retrieval_low_conf/self_check,随条存召回快照给审核页;话术与转人工按钮不变)。"""
    source = state.get("fallback_source") or "retrieval_low_conf"
    signals = state.get("trace", {}).get("confidence_signals") or {}
    reason = (f"evidence_confidence={state.get('evidence_confidence', 0.0):.3f} "
              f"signals={json.dumps(signals, ensure_ascii=False)}")
    if source == "self_check":
        reason += f" self_check={state.get('trace', {}).get('self_check', '')}"
    snapshot = state.get("retrieved_snapshot") or None   # 没走检索存 NULL,不存 []
    await repository.insert_low_confidence(
        state.get("conversation_id"), _user_text(state), source, reason,
        retrieved_chunks=snapshot,
    )
    return {"answer": FALLBACK_REPLY,
            "suggested_actions": [{"type": "transfer_human"}],
            "trace": {"route": "fallback"}}
```

(nodes.py 已 import json——若没有则补。)

- [ ] **Step 5: 测试 + Commit**

Run: `uv run pytest tests/graph/test_ch09_confidence_gate.py -v` → PASS;`uv run pytest -v` → 零回归(既有引用 `evidence_top` 的测试若有,按新 trace 键 `evidence_confidence` 修正——这是口径升级,不是行为破坏)。

```bash
git add app/graph/state.py app/graph/runtime.py app/graph/nodes.py tests/graph/test_ch09_confidence_gate.py
git commit -m "feat(ch09): 置信度闸接线——四信号总分拦截 + source 打对标签 + 快照随池落库"
```

---

### Task 6: 阈值校准脚本(评估集实跑,标注样例验证型)

**Files:**
- Create: `scripts/calibrate_confidence.py`
- Modify: `Makefile`(calibrate-confidence)
- Modify: `app/config.py`(回填实测推荐阈值)
- Output: `dev-notes/ch09-confidence-calibration.txt`

**Interfaces:**
- Consumes: `tests/data/eval_ch04.jsonl`(A/B/C 可答桶 + D 应拒桶);`compute_evidence_confidence`;`retrieval.search_knowledge`。
- Produces: `settings.evidence_confidence_threshold` 的实测默认值 + 校准报告。

**前置**:milvus 在线、上游可用,知识库已建(`make milvus-up && make kb-build && make kb-vectorize`)。本任务无单测(数据校准类,产物即验证)。

- [ ] **Step 1: 写校准脚本**

`scripts/calibrate_confidence.py`:

```python
"""ch09 置信度阈值校准:拿 ch04 评估集,A/B/C 桶(应可答)与 D 桶(应拒答)分别算
evidence_confidence 分布,扫阈值取 Youden J(=可答通过率 - 应拒放行率)最大的分离点。
输出分布表 + 推荐阈值 → 人工回填 settings.evidence_confidence_threshold(不拍脑袋)。
运行:make calibrate-confidence(需 milvus + 上游可用 + 知识库已建)。
产物:dev-notes/ch09-confidence-calibration.txt
"""
import asyncio
import json
import pathlib
import sys

from app.core import retrieval
from app.core.confidence import compute_evidence_confidence

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OUT = _ROOT / "dev-notes/ch09-confidence-calibration.txt"
ANSWERABLE = {"A_policy", "B_model", "C_colloquial"}
_SEM = asyncio.Semaphore(8)
_LINES: list[str] = []


def _log(msg=""):
    print(msg, flush=True)
    _LINES.append(msg)


async def _conf(sample) -> tuple[str, float]:
    async with _SEM:
        hits = await retrieval.search_knowledge(sample["query"], strategy="hybrid_rerank")
    return sample["bucket"], compute_evidence_confidence(hits).score


def _dist(name, xs):
    xs = sorted(xs)
    if not xs:
        return
    p = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]
    _log(f"{name:12s} n={len(xs):3d} min={xs[0]:.3f} p25={p(.25):.3f} "
         f"p50={p(.5):.3f} p75={p(.75):.3f} max={xs[-1]:.3f}")


async def main():
    samples = [json.loads(ln) for ln in open("tests/data/eval_ch04.jsonl")]
    results = await asyncio.gather(*(_conf(s) for s in samples))
    answerable = [c for b, c in results if b in ANSWERABLE]
    absent = [c for b, c in results if b == "D_absent"]

    _log("=== evidence_confidence 分布(hybrid_rerank)===")
    _dist("可答(ABC)", answerable)
    _dist("应拒(D)", absent)

    _log("\n=== 阈值扫描(通过率=conf>=t 的可答占比;放行率=conf>=t 的应拒占比)===")
    _log(f"{'t':>6s} {'可答通过率':>10s} {'应拒放行率':>10s} {'YoudenJ':>8s}")
    best_t, best_j = 0.0, -1.0
    for i in range(5, 96):
        t = i / 100
        tpr = sum(1 for c in answerable if c >= t) / len(answerable)
        fpr = sum(1 for c in absent if c >= t) / len(absent)
        j = tpr - fpr
        if i % 5 == 0 or j > best_j:
            _log(f"{t:6.2f} {tpr:10.3f} {fpr:10.3f} {j:8.3f}")
        if j > best_j:
            best_t, best_j = t, j
    _log(f"\n推荐阈值 evidence_confidence_threshold = {best_t:.2f}(Youden J={best_j:.3f})")
    _log("请回填 app/config.py 默认值,并在 dev-notes/ch09.md 记录本次校准。")
    _OUT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    _log(f"报告已落 {_OUT.relative_to(_ROOT)}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
```

Makefile(`.PHONY` 同步加):

```makefile
calibrate-confidence:  ## ch09 置信度阈值校准(需 milvus + 上游可用 + 知识库已建)
	PYTHONPATH=. uv run python scripts/calibrate_confidence.py
```

- [ ] **Step 2: 实跑校准**

Run: `make calibrate-confidence`
Expected: 两桶分布明显分离(可答 p50 显著高于应拒 p50),输出推荐阈值。若两桶重叠严重(J < 0.3),停下来把分布表给用户看,共同定夺(可能要调权重或信号),不许硬选。

- [ ] **Step 3: 回填阈值 + Commit**

把 `app/config.py` 的 `evidence_confidence_threshold` 默认值改为实测推荐值,注释写明校准来源:

```python
    evidence_confidence_threshold: float = <实测值>   # make calibrate-confidence 在 eval_ch04 上校准(YoudenJ 最优,见 dev-notes/ch09-confidence-calibration.txt)
```

Run: `uv run pytest -v` → 零回归(test_ch09_confidence_gate 用 monkeypatch 阈值,不受影响)。

```bash
git add scripts/calibrate_confidence.py Makefile app/config.py
git commit -m "feat(ch09): 置信度阈值评估集校准——分布/扫描/YoudenJ 选点,回填实测默认值"
```

---

### Task 7: 用户反馈入口(👎 API + 快照回捞 + 前端接线)

**Files:**
- Create: `app/api/feedback.py`、`app/schemas/feedback.py`
- Modify: `app/graph/runtime.py`(加 `get_turn_snapshot`)
- Modify: `app/main.py`(挂 feedback router)
- Modify: `app/static/index.html`(👎/👍 发请求;Vibe Coding,不套 TDD)
- Test: `tests/test_feedback_api.py`

**Interfaces:**
- Consumes: `repository.insert_low_confidence(..., retrieved_chunks=)`;state 的 `retrieved_snapshot`(Task 5)。
- Produces:
  - `POST /api/feedback` body `{"conversation_id": int, "rating": "up"|"down", "question": str}` → `{"ok": true, "pooled": bool}`
  - `runtime.get_turn_snapshot(conversation_id: int) -> dict`(`{"question": str, "snapshot": list}`,读 checkpointer 终态)

- [ ] **Step 1: 写失败测试**

`tests/test_feedback_api.py`:

```python
"""ch09 入口3:👎 落池(source=user_feedback,快照尽力回捞);👍 只记日志不落库。"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import feedback as feedback_api


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(feedback_api.router)
    return TestClient(app)


@pytest.fixture
def spy_insert(monkeypatch):
    calls = []
    async def fake_insert(cid, q, source, reason, retrieved_chunks=None):
        calls.append({"cid": cid, "q": q, "source": source, "chunks": retrieved_chunks})
        return 1
    monkeypatch.setattr(feedback_api.repository, "insert_low_confidence", fake_insert)
    return calls


def _stub_snapshot(monkeypatch, question, snapshot):
    async def fake_get(cid):
        return {"question": question, "snapshot": snapshot}
    monkeypatch.setattr(feedback_api.runtime, "get_turn_snapshot", fake_get)


def test_down_pools_with_snapshot_when_question_matches(client, spy_insert, monkeypatch):
    snap = [{"question": "x", "answer": "y", "rerank_score": 0.4, "section_path": "s"}]
    _stub_snapshot(monkeypatch, "猫窝能水洗吗", snap)
    r = client.post("/api/feedback", json={"conversation_id": 1, "rating": "down", "question": "猫窝能水洗吗"})
    assert r.status_code == 200 and r.json()["pooled"] is True
    assert spy_insert[0]["source"] == "user_feedback"
    assert spy_insert[0]["chunks"] == snap        # 问题对得上 → 快照带走


def test_down_pools_without_snapshot_when_question_differs(client, spy_insert, monkeypatch):
    _stub_snapshot(monkeypatch, "另一个问题", [{"question": "x", "answer": "y", "rerank_score": 0.4, "section_path": "s"}])
    r = client.post("/api/feedback", json={"conversation_id": 1, "rating": "down", "question": "猫窝能水洗吗"})
    assert r.status_code == 200
    assert spy_insert[0]["chunks"] is None        # 对不上 → 空着(尽力而已,不硬塞)


def test_down_survives_snapshot_failure(client, spy_insert, monkeypatch):
    async def boom(cid):
        raise RuntimeError("checkpointer 不可用")
    monkeypatch.setattr(feedback_api.runtime, "get_turn_snapshot", boom)
    r = client.post("/api/feedback", json={"conversation_id": 1, "rating": "down", "question": "猫窝能水洗吗"})
    assert r.status_code == 200                   # 回捞失败不拦落池
    assert spy_insert[0]["chunks"] is None


def test_up_logs_only(client, spy_insert, monkeypatch):
    _stub_snapshot(monkeypatch, "q", [])
    r = client.post("/api/feedback", json={"conversation_id": 1, "rating": "up", "question": "q"})
    assert r.status_code == 200 and r.json()["pooled"] is False
    assert spy_insert == []                       # 👍 不落库


def test_rejects_bad_rating(client, spy_insert):
    r = client.post("/api/feedback", json={"conversation_id": 1, "rating": "meh", "question": "q"})
    assert r.status_code == 422
```

Run: `uv run pytest tests/test_feedback_api.py -v` → FAIL。

- [ ] **Step 2: runtime 加回捞**

`app/graph/runtime.py` 追加(get_graph 之后):

```python
async def get_turn_snapshot(conversation_id: int) -> dict:
    """👎 快照尽力回捞:读该会话 checkpointer 终态,返回最近一轮的用户问题与召回快照。
    调用方拿去比对被踩的 question,对得上才用快照——踩历史消息时快照可能已是别轮的,不硬塞。"""
    config = {"configurable": {"thread_id": str(conversation_id)}}
    st = await get_graph().aget_state(config)
    values = getattr(st, "values", None) or {}
    question = ""
    for m in reversed(values.get("messages", [])):
        if getattr(m, "type", "") == "human":
            question = m.content if isinstance(m.content, str) else ""
            break
    return {"question": question, "snapshot": values.get("retrieved_snapshot") or []}
```

- [ ] **Step 3: schema + API**

`app/schemas/feedback.py`:

```python
from typing import Literal

from pydantic import BaseModel


class FeedbackRequest(BaseModel):
    conversation_id: int
    rating: Literal["up", "down"]
    question: str          # 被评价那轮的用户原话(前端从气泡上带上来)


class FeedbackResponse(BaseModel):
    ok: bool = True
    pooled: bool           # down 且成功落池才为 True
```

`app/api/feedback.py`:

```python
"""ch09 飞轮入口3:用户反馈没解决。👎 落 low_confidence_questions(source=user_feedback),
快照尽力回捞(checkpointer 里最近一轮的 retrieved_snapshot,问题对得上才用);👍 只记日志。"""
import logging

from fastapi import APIRouter

from app.db import repository
from app.graph import runtime
from app.schemas.feedback import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _same_question(a: str, b: str) -> bool:
    return "".join(a.split()) == "".join(b.split())


@router.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest) -> FeedbackResponse:
    if req.rating == "up":
        logger.info("feedback up conv=%s q=%r(只记日志,不落库)", req.conversation_id, req.question[:40])
        return FeedbackResponse(pooled=False)

    snapshot = None
    try:
        turn = await runtime.get_turn_snapshot(req.conversation_id)
        if turn["snapshot"] and _same_question(turn["question"], req.question):
            snapshot = turn["snapshot"]
    except Exception:
        logger.warning("feedback 快照回捞失败 conv=%s(照常落池)", req.conversation_id, exc_info=True)

    await repository.insert_low_confidence(
        req.conversation_id, req.question, "user_feedback", "用户反馈未解决",
        retrieved_chunks=snapshot,
    )
    logger.info("feedback down conv=%s 已落池 快照=%s", req.conversation_id,
                "有" if snapshot else "无")
    return FeedbackResponse(pooled=True)
```

`app/main.py`:import `from app.api.feedback import router as feedback_router`,`app.include_router(feedback_router)`。

- [ ] **Step 4: 跑测试**

Run: `uv run pytest tests/test_feedback_api.py -v` → PASS;全量零回归。

- [ ] **Step 5: 前端接线(Vibe Coding)**

`app/static/index.html` `addFeedbackBar(bubble)`:
1. 函数签名不变;在 `give()` 内、`chosen === down` 时发请求。该轮用户问题:向上找 bubble 之前最近的一条 user 气泡文本(现有 DOM 结构里用户行 class 与 bot 行对称,照现有 addRow 类名取);`conversationId` 用页面现有全局会话变量(index.html 已有,搜 `conversation_id` 取同名)。
2. 代码形状(细节按现有 DOM 命名适配):

```javascript
      function userQuestionBefore(bubble) {
        let row = bubble.closest(".row, .msg");        // 按现有行容器 class 适配
        while (row && (row = row.previousElementSibling)) {
          if (row.classList.contains("user")) return row.textContent.trim();
        }
        return "";
      }
      function give(chosen, other) {
        if (bar.dataset.given) return;
        bar.dataset.given = "1";
        chosen.classList.add("active"); other.classList.add("dim");
        up.disabled = true; down.disabled = true;
        done.hidden = false;
        const rating = chosen === down ? "down" : "up";
        fetch("/api/feedback", {
          method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({conversation_id: currentConversationId,
                                rating, question: userQuestionBefore(bubble)}),
        }).catch(() => {});   // 失败静默,反馈条 UI 行为不变
      }
```

3. 浏览器验收:发一问 → 点 👎 → `curl` 或 mysql 查 `low_confidence_questions` 出现 `user_feedback` 行、走过检索的轮次 `retrieved_chunks` 非空。效果给用户看,按用户描述改。

- [ ] **Step 6: Commit**

```bash
git add app/api/feedback.py app/schemas/feedback.py app/graph/runtime.py app/main.py app/static/index.html tests/test_feedback_api.py
git commit -m "feat(ch09): 👎 接后端落池——source=user_feedback + checkpointer 快照尽力回捞"
```

---

### Task 8: 飞轮流水线(标准化+查重一次输出 + 批处理脚本 + 样例验证)

**Files:**
- Create: `app/core/flywheel.py`
- Create: `scripts/flywheel_pipeline.py`、`scripts/validate_flywheel_samples.py`
- Create: `tests/data/flywheel_samples.json`
- Modify: `app/core/prompts.py`(FLYWHEEL_NORMALIZE_PROMPT)
- Modify: `Makefile`(flywheel / flywheel-samples)
- Test: `tests/core/test_flywheel.py`

**Interfaces:**
- Consumes: Task 1 全部 repository 飞轮方法;`get_chat_model()`。
- Produces:
  - `flywheel.normalize_and_match(raw_question: str, candidates: list[dict]) -> NormalizeResult`(pydantic:`normalized_question: str, matched_question_id: int|None, ai_suggested_answer: str`)
  - `flywheel.process_pending(limit: int = 50) -> dict`(`{"processed","merged","created","skipped"}`)
  - `make flywheel`(验收 2/4 的触发命令)

- [ ] **Step 1: 写 prompt**

`app/core/prompts.py` 追加(README 的 JSON 形状,一次输出标准化+查重+示例答案):

```python
FLYWHEEL_NORMALIZE_SYSTEM = """## 角色
你是客服知识库的问题标准化与查重器。输入一条用户原话和一批候选标准问题,你做三件事一次输出:

1. normalized_question:把原话去噪——剥掉情绪、口语、无关细节,只留核心诉求,改写成一句
   FAQ 式标准问题(如「我上周买的鞋跑两次就开胶了太坑了能退吗」→「商品出现质量问题(如开胶)能否退货」)。
2. matched_question_id:逐条比对候选,判断当前问题与哪条候选是同一个意图(问法不同不要紧,
   问的是同一件事就算命中)。命中填那条候选的 id(整数);都不是同类填 null。
   只能填候选列表里出现过的 id,严禁编造。宁可 null 也不要硬凑。
3. ai_suggested_answer:给这个标准问题写一条简短的示例答案备查(客服口吻,不臆造政策数字,
   拿不准的表述用「以平台售后规则为准」)。

严格输出 JSON,不含其他文本:
{{"normalized_question": "...", "matched_question_id": 128 或 null, "ai_suggested_answer": "..."}}"""

FLYWHEEL_NORMALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", FLYWHEEL_NORMALIZE_SYSTEM),
     ("human", "候选标准问题(可空):\n{candidates}\n\n用户原话:{raw_question}")]
)
```

- [ ] **Step 2: 写失败测试**

`tests/core/test_flywheel.py`:

```python
"""ch09 飞轮核心:标准化+查重一次输出;命中累加不新建、不复活;幻觉 id 跳过;串行防同批重复。"""
import pytest

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
```

Run: `uv run pytest tests/core/test_flywheel.py -v` → FAIL。

- [ ] **Step 3: 实现 flywheel 核心**

`app/core/flywheel.py`:

```python
"""ch09 数据飞轮流水线:问题标准化 + 查重(模型一次输出,README 形状)→ 待审队列。

批处理语义(用户拍板:定时批处理,不做近实时):
- 游标 = low_confidence_questions.matched_review_id IS NULL,处理完回写,天然幂等可重跑;
- 串行逐条:同批同义问题第二条能命中第一条刚建的行,不重复建缺口;
- 命中任何状态的候选都只累加 occurrence_count(驳回即终审,不复活);
- 解析失败/幻觉 id:该条跳过并告警,游标留在原地下轮重试。
"""
import json
import logging

from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import FLYWHEEL_NORMALIZE_PROMPT
from app.db import repository

logger = logging.getLogger(__name__)


class NormalizeResult(BaseModel):
    normalized_question: str = Field(description="FAQ 式标准问题")
    matched_question_id: int | None = Field(default=None, description="命中候选 id,无同类为 null")
    ai_suggested_answer: str = Field(default="", description="示例答案备查")


def _chain():
    return FLYWHEEL_NORMALIZE_PROMPT | get_chat_model().with_structured_output(NormalizeResult)


async def normalize_and_match(raw_question: str, candidates: list[dict]) -> NormalizeResult:
    """一次模型调用输出 {normalized_question, matched_question_id, ai_suggested_answer}。"""
    cand_text = "\n".join(
        f"- id={c['id']}: {c['normalized_question']}" for c in candidates) or "(无候选)"
    return await _chain().ainvoke({"raw_question": raw_question, "candidates": cand_text})


async def process_pending(limit: int = 50) -> dict:
    """扫未归并的池内问题,逐条标准化+查重,写待审队列并回写归并落点。返回本轮统计。"""
    rows = await repository.fetch_unmatched_low_conf(limit)
    stats = {"processed": 0, "merged": 0, "created": 0, "skipped": 0}
    for row in rows:
        candidates = await repository.list_review_candidates()   # 每条现拉:同批新建的行进得了候选
        try:
            r = await normalize_and_match(row.raw_question, candidates)
        except Exception:
            logger.warning("flywheel 标准化失败 lcq=%s(跳过,下轮重试)", row.id, exc_info=True)
            stats["skipped"] += 1
            continue
        mid = r.matched_question_id
        if mid is not None:
            if not any(c["id"] == mid for c in candidates):
                logger.warning("flywheel 幻觉 id=%s lcq=%s(候选里不存在,跳过下轮重试)", mid, row.id)
                stats["skipped"] += 1
                continue
            await repository.increment_occurrence(mid)
            review_id = mid
            stats["merged"] += 1
        else:
            review_id = await repository.insert_review_item(
                r.normalized_question, r.ai_suggested_answer or None)
            stats["created"] += 1
        await repository.set_matched_review(row.id, review_id)
        stats["processed"] += 1
        logger.info("flywheel lcq=%s → review=%s(%s)%s", row.id, review_id,
                    "归并" if mid is not None else "新建",
                    "" if len(candidates) < 200 else " [候选已截断200,查重覆盖不全]")
    return stats
```

- [ ] **Step 4: 跑测试**

Run: `uv run pytest tests/core/test_flywheel.py -v` → PASS;全量零回归。

- [ ] **Step 5: 批处理脚本 + make**

`scripts/flywheel_pipeline.py`:

```python
"""ch09 飞轮批处理:扫问题池未归并条目 → 标准化+查重 → 待审队列。
运行:make flywheel(需 mysql + 上游可用)。定时跑给 cron 示例:
  */30 * * * * cd /path/to/mewhelp && make flywheel >> log/flywheel.log 2>&1
"""
import asyncio
import sys

from app.core.flywheel import process_pending


async def main():
    stats = await process_pending(limit=200)
    print(f"本轮处理 {stats['processed']} 条:新建缺口 {stats['created']},"
          f"归并 {stats['merged']},跳过待重试 {stats['skipped']}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
```

Makefile(`.PHONY` 同步):

```makefile
flywheel:  ## ch09 飞轮批处理:问题池 → 标准化查重 → 待审队列(需 mysql + 上游可用)
	PYTHONPATH=. uv run python scripts/flywheel_pipeline.py

flywheel-samples:  ## ch09 标准化查重 prompt 标注样例验证(通过率 ≥ 80%)
	PYTHONPATH=. uv run python scripts/validate_flywheel_samples.py
```

- [ ] **Step 6: 标注样例集 + 验证脚本(Prompt 类任务的"TDD 替代")**

`tests/data/flywheel_samples.json`(12 条,覆盖:口语去噪、情绪剥离、同义命中、近似但不同义不命中、空候选、多诉求取主诉求):

```json
[
  {"raw_question": "我上周买的那双鞋跑了两次就开胶了这也太坑了吧能不能给我退了啊",
   "candidates": [], "expect_match": null, "expect_keywords": ["质量", "退"]},
  {"raw_question": "猫窝可以水洗吗",
   "candidates": [{"id": 1, "normalized_question": "猫窝是否支持水洗"}],
   "expect_match": 1, "expect_keywords": []},
  {"raw_question": "猫窝脏了怎么洗才不会坏",
   "candidates": [{"id": 1, "normalized_question": "猫窝是否支持水洗"}],
   "expect_match": 1, "expect_keywords": []},
  {"raw_question": "猫窝有什么颜色",
   "candidates": [{"id": 1, "normalized_question": "猫窝是否支持水洗"}],
   "expect_match": null, "expect_keywords": ["颜色"]},
  {"raw_question": "你们家猫爬架的保修到底是几年啊问了好几次都没人理",
   "candidates": [{"id": 3, "normalized_question": "猫爬架保修期是多久"}],
   "expect_match": 3, "expect_keywords": []},
  {"raw_question": "猫爬架坏了修一次要多少钱",
   "candidates": [{"id": 3, "normalized_question": "猫爬架保修期是多久"}],
   "expect_match": null, "expect_keywords": ["维修", "费用"]},
  {"raw_question": "智能猫砂盆第二代和第一代比升级了啥",
   "candidates": [{"id": 5, "normalized_question": "智能猫砂盆二代与一代的区别"}],
   "expect_match": 5, "expect_keywords": []},
  {"raw_question": "买的猫粮拆封了还能退吗急",
   "candidates": [{"id": 7, "normalized_question": "猫粮拆封后能否退货"}],
   "expect_match": 7, "expect_keywords": []},
  {"raw_question": "双十一买的东西什么时候发货啊等好久了",
   "candidates": [{"id": 7, "normalized_question": "猫粮拆封后能否退货"}],
   "expect_match": null, "expect_keywords": ["发货"]},
  {"raw_question": "猫别墅要自己装吗看着好复杂的样子装不好能找人吗",
   "candidates": [], "expect_match": null, "expect_keywords": ["安装"]},
  {"raw_question": "上次那个客服说可以退结果到现在钱都没到我要投诉你们",
   "candidates": [{"id": 9, "normalized_question": "退款多久到账"}],
   "expect_match": 9, "expect_keywords": []},
  {"raw_question": "宠物烘干箱猫咪用会不会太热有没有温度上限",
   "candidates": [], "expect_match": null, "expect_keywords": ["温度"]}
]
```

`scripts/validate_flywheel_samples.py`:

```python
"""ch09 Prompt 标注样例验证(工作要求:纯 Prompt 任务拿标注样例跑一遍代替 TDD)。
判分:matched_question_id 与 expect_match 一致 = 查重对;normalized_question 含全部
expect_keywords = 标准化对(仅新建样例查关键词)。两项都对才算过,通过率 ≥ 80% 视为可用。
运行:make flywheel-samples(需上游可用)。
"""
import asyncio
import json
import sys

from app.core.flywheel import normalize_and_match

THRESHOLD = 0.8


async def main():
    samples = json.load(open("tests/data/flywheel_samples.json"))
    passed = 0
    for s in samples:
        try:
            r = await normalize_and_match(s["raw_question"], s["candidates"])
        except Exception as e:
            print(f"✗ {s['raw_question'][:24]}… 调用失败 {type(e).__name__}")
            continue
        match_ok = r.matched_question_id == s["expect_match"]
        kw_ok = all(k in r.normalized_question for k in s["expect_keywords"])
        ok = match_ok and kw_ok
        passed += ok
        mark = "✓" if ok else "✗"
        print(f"{mark} {s['raw_question'][:24]}… → {r.normalized_question} "
              f"match={r.matched_question_id}(期望 {s['expect_match']})")
    rate = passed / len(samples)
    print(f"\n通过 {passed}/{len(samples)} = {rate:.0%}(线 {THRESHOLD:.0%})")
    return 0 if rate >= THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

Run: `make flywheel-samples` → 通过率 ≥ 80%。不到线就改 prompt(加边界样例说明)重跑,把翻车记 dev-notes;连续两轮不到线停下来问用户。

- [ ] **Step 7: Commit**

```bash
git add app/core/flywheel.py app/core/prompts.py scripts/flywheel_pipeline.py scripts/validate_flywheel_samples.py tests/data/flywheel_samples.json tests/core/test_flywheel.py Makefile
git commit -m "feat(ch09): 飞轮流水线——标准化+查重一次输出/批处理脚本/标注样例验证"
```

---

### Task 9: 审核 API + 通过写回知识库

**Files:**
- Create: `app/api/review.py`、`app/schemas/review.py`
- Modify: `app/kb/dualwrite.py`(`vectorize_pending` 的 Milvus 调用改走 `acall`,server 内可安全调)
- Modify: `app/main.py`(挂 review router + `/review` 页面路由)
- Test: `tests/test_review_api.py`

**Interfaces:**
- Consumes: Task 1 repository;`dualwrite.write_pending` / `vectorize_pending`;`Chunk`(app/kb/documents.py)。
- Produces:
  - `GET /api/review/queue?status=待审` → `{"items": [{"id","normalized_question","ai_suggested_answer","occurrence_count","review_status","created_at"}]}`(occurrence_count 降序;status 省略=全量)
  - `GET /api/review/{id}` → 队列行 + `raws: [{"raw_question","source","reason","created_at","retrieved_chunks"}]`
  - `POST /api/review/{id}/approve` body `{"approved_answer": str}` → 200 `{"ok":true,"chunk_ids":[...]}`;非待审 409;写库/向量化失败 502 且状态保持待审
  - `POST /api/review/{id}/reject` → 200;非待审 409

- [ ] **Step 1: 写失败测试**

`tests/test_review_api.py`:

```python
"""ch09 审核 API:列表/详情/通过(写回知识库)/驳回;只有待审可流转;写回失败状态不动。"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import review as review_api
from app.db import repository


@pytest.fixture
def client(db_session_factory):
    app = FastAPI()
    app.include_router(review_api.router)
    return TestClient(app)


@pytest.fixture
def stub_kb(monkeypatch):
    written = {}
    async def fake_write(chunks):
        written["chunks"] = chunks
        return [101]
    async def fake_vectorize():
        written["vectorized"] = True
        return 1
    monkeypatch.setattr(review_api, "_write_chunks", fake_write)
    monkeypatch.setattr(review_api, "_vectorize", fake_vectorize)
    return written


async def _seed():
    rid = await repository.insert_review_item("猫窝是否支持水洗", "可以,以平台售后规则为准")
    lcq = await repository.insert_low_confidence(
        None, "猫窝可以洗吗", "user_feedback", None,
        retrieved_chunks=[{"question": "q", "answer": "a", "rerank_score": 0.3, "section_path": "s"}])
    await repository.set_matched_review(lcq, rid)
    return rid


def test_queue_lists_pending(client):
    import asyncio
    rid = asyncio.get_event_loop().run_until_complete(_seed())
    r = client.get("/api/review/queue", params={"status": "待审"})
    assert r.status_code == 200
    items = r.json()["items"]
    assert items[0]["id"] == rid and items[0]["occurrence_count"] == 1


def test_detail_carries_raws_and_snapshot(client):
    import asyncio
    rid = asyncio.get_event_loop().run_until_complete(_seed())
    r = client.get(f"/api/review/{rid}")
    assert r.status_code == 200
    d = r.json()
    assert d["normalized_question"] == "猫窝是否支持水洗"
    assert d["raws"][0]["source"] == "user_feedback"
    assert d["raws"][0]["retrieved_chunks"][0]["rerank_score"] == 0.3
    assert client.get("/api/review/99999").status_code == 404


def test_approve_writes_kb_and_flips_status(client, stub_kb):
    import asyncio
    rid = asyncio.get_event_loop().run_until_complete(_seed())
    r = client.post(f"/api/review/{rid}/approve", json={"approved_answer": "可以水洗,注意阴干"})
    assert r.status_code == 200 and r.json()["chunk_ids"] == [101]
    chunk = stub_kb["chunks"][0]
    assert chunk.questions == "猫窝是否支持水洗" and chunk.answer == "可以水洗,注意阴干"
    assert chunk.content_type == "faq"
    assert stub_kb["vectorized"] is True
    r2 = client.get(f"/api/review/{rid}")
    assert r2.json()["review_status"] == "通过"
    # 终审后再操作 → 409
    assert client.post(f"/api/review/{rid}/reject").status_code == 409


def test_approve_failure_keeps_pending(client, monkeypatch):
    import asyncio
    rid = asyncio.get_event_loop().run_until_complete(_seed())
    async def boom(chunks):
        raise RuntimeError("嵌入上游不可用")
    monkeypatch.setattr(review_api, "_write_chunks", boom)
    r = client.post(f"/api/review/{rid}/approve", json={"approved_answer": "x"})
    assert r.status_code == 502
    assert client.get(f"/api/review/{rid}").json()["review_status"] == "待审"   # 可重试


def test_reject_flips_status(client):
    import asyncio
    rid = asyncio.get_event_loop().run_until_complete(_seed())
    assert client.post(f"/api/review/{rid}/reject").status_code == 200
    assert client.get(f"/api/review/{rid}").json()["review_status"] == "驳回"
    assert client.post(f"/api/review/{rid}/approve", json={"approved_answer": "x"}).status_code == 409
```

(注:TestClient 同步跑异步 app 没问题;seed 用 loop 直跑是本仓库测试里没有的新形状,若 event loop 冲突,改成 `anyio`/`pytest.mark.asyncio` 的 async 测试 + `httpx.AsyncClient`,以实际跑通为准,断言不变。)

Run: `uv run pytest tests/test_review_api.py -v` → FAIL。

- [ ] **Step 2: dualwrite 的 Milvus 调用改走 acall**

`app/kb/dualwrite.py`:`vectorize_pending` 目前直接在事件循环里同步调 `milvus_client.upsert_vectors/flush`,且要求调用方先拿 client——脚本里没问题,server 内违反"Milvus 同步调用固定单线程执行"的既有约定(见 retrieval.py 注)。改成 client 参数可选、Milvus 操作走 `acall`:

```python
async def vectorize_pending(client=None, batch_size: int = 64, collection: str = milvus_client.COLLECTION) -> int:
    """幂等可重跑:取 pending → 拼 category+questions+answer → 嵌入 + BM25 text → Milvus upsert(PK=id)
    → 回填 vector_id、status=done。
    ch09:client 缺省时在 Milvus 专用线程内惰性创建,upsert/flush 也走 acall——
    server 进程(审核通过写回)与脚本两个场景共用一条安全路径。"""
    pending = await repository.list_pending_chunks()
    done = 0
    for batch in _batches(pending, batch_size):
        texts = [f"{r.category}\n{r.questions}\n{r.answer}" for r in batch]
        vectors = await embeddings.embed_texts(texts)
        rows = [
            {"id": r.id, "dense": v, "text": t,
             "question": r.questions, "answer": r.answer,
             "section_path": r.section_path or "", "content_type": r.content_type or "",
             "category": r.category or ""}
            for r, v, t in zip(batch, vectors, texts)
        ]

        def work_upsert(rows=rows):
            c = client or milvus_client.get_client()
            milvus_client.ensure_collection(c, collection=collection)
            milvus_client.upsert_vectors(c, rows, collection=collection)
        await milvus_client.acall(work_upsert)
        for r in batch:
            await repository.mark_chunk_vectorized(r.id, str(r.id))
        done += len(batch)
    if done:
        def work_flush():
            c = client or milvus_client.get_client()
            milvus_client.flush(c, collection=collection)  # 刷盘,数据方可被 BM25 检索
        await milvus_client.acall(work_flush)
    return done
```

检查调用方:`grep -rn "vectorize_pending" scripts/ app/ tests/`——`scripts/vectorize_kb.py` 等传了显式 client 的保持兼容(client 参数仍在);若脚本此前自建 client 传入,行为不变。跑 `uv run pytest tests/kb -v` 确认 kb 测试零回归。

- [ ] **Step 3: schema + API 实现**

`app/schemas/review.py`:

```python
from pydantic import BaseModel


class ApproveRequest(BaseModel):
    approved_answer: str


class ReviewItemOut(BaseModel):
    id: int
    normalized_question: str
    ai_suggested_answer: str | None
    occurrence_count: int
    review_status: str
    created_at: str
```

`app/api/review.py`:

```python
"""ch09 审核后台 API:待审队列列表/详情/通过(写回知识库)/驳回。
通过 = 核准答案以 QA chunk 走 ch03 落库流程(write_pending → vectorize_pending 同步),
向量化成功才置「通过」——保证审核页点了通过,下一问就能检索命中(验收 3)。"""
import logging

from fastapi import APIRouter, HTTPException

from app.db import repository
from app.kb import dualwrite
from app.kb.documents import Chunk, _is_key
from app.schemas.review import ApproveRequest

logger = logging.getLogger(__name__)
router = APIRouter()


async def _write_chunks(chunks: list[Chunk]) -> list[int]:
    return await dualwrite.write_pending(chunks)


async def _vectorize() -> int:
    return await dualwrite.vectorize_pending()


def _item_out(r) -> dict:
    return {"id": r.id, "normalized_question": r.normalized_question,
            "ai_suggested_answer": r.ai_suggested_answer,
            "occurrence_count": r.occurrence_count, "review_status": r.review_status,
            "created_at": r.created_at.isoformat() if r.created_at else None}


@router.get("/api/review/queue")
async def review_queue(status: str | None = None) -> dict:
    rows = await repository.list_review_queue(status)
    return {"items": [_item_out(r) for r in rows]}


@router.get("/api/review/{review_id}")
async def review_detail(review_id: int) -> dict:
    detail = await repository.get_review_detail(review_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="缺口不存在")
    item, raws = detail
    out = _item_out(item)
    out["approved_answer"] = item.approved_answer
    out["raws"] = [{"raw_question": r.raw_question, "source": r.source, "reason": r.reason,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "retrieved_chunks": r.retrieved_chunks}
                   for r in raws]
    return out


@router.post("/api/review/{review_id}/approve")
async def approve(review_id: int, req: ApproveRequest) -> dict:
    detail = await repository.get_review_detail(review_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="缺口不存在")
    item, _ = detail
    if item.review_status != "待审":
        raise HTTPException(status_code=409, detail=f"当前状态为「{item.review_status}」,不可再审")

    chunk = Chunk(
        category="飞轮沉淀", questions=item.normalized_question, answer=req.approved_answer,
        section_path=f"飞轮沉淀 / {item.normalized_question}", content_type="faq",
        is_key_clause=_is_key(item.normalized_question, req.approved_answer),
    )
    try:
        chunk_ids = await _write_chunks([chunk])
        await _vectorize()
    except Exception:
        logger.exception("审核写回知识库失败 review=%s(状态保持待审,可重试)", review_id)
        raise HTTPException(status_code=502, detail="写回知识库失败(检查上游/Milvus),状态未变可重试")

    await repository.update_review_status(review_id, "通过", approved_answer=req.approved_answer)
    logger.info("审核通过 review=%s → knowledge_chunks %s(已向量化,下一问可检索)", review_id, chunk_ids)
    return {"ok": True, "chunk_ids": chunk_ids}


@router.post("/api/review/{review_id}/reject")
async def reject(review_id: int) -> dict:
    if not await repository.update_review_status(review_id, "驳回"):
        detail = await repository.get_review_detail(review_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="缺口不存在")
        raise HTTPException(status_code=409, detail="仅待审状态可驳回")
    return {"ok": True}
```

`app/main.py`:挂 router + 审核页路由:

```python
from app.api.feedback import router as feedback_router
from app.api.review import router as review_router
...
app.include_router(feedback_router)
app.include_router(review_router)


@app.get("/review", include_in_schema=False)
async def review_page() -> FileResponse:
    return FileResponse(_STATIC_DIR / "review.html")
```

(review.html Task 12 才建;先建一个占位文件避免 404:`echo '审核页施工中' > app/static/review.html` 或直接在 Task 12 前不点这个路由。)

- [ ] **Step 4: 测试 + Commit**

Run: `uv run pytest tests/test_review_api.py tests/kb -v` → PASS;全量零回归。

```bash
git add app/api/review.py app/schemas/review.py app/kb/dualwrite.py app/main.py app/static/review.html tests/test_review_api.py
git commit -m "feat(ch09): 审核 API——通过即写回知识库(dualwrite 走 acall),驳回终审"
```

---

### Task 10: 自动化评估流水线(eval_runs + 趋势对比)

**Files:**
- Create: `scripts/eval_flywheel.py`
- Modify: `Makefile`(eval-flywheel)
- Output: `dev-notes/ch09-eval-trend.txt`

**Interfaces:**
- Consumes: `scripts/eval_ch04.py` 的 `_load/_hit_rank/_mean/_format_evidence/_refusal_one/GRADED_BUCKETS/K`(import 复用,不复制粘贴);`FAITHFULNESS_PROMPT/RAG_ANSWER_PROMPT`;`repository.insert_eval_run/list_eval_runs`。
- Produces: `make eval-flywheel [TRIGGER=手动|定时]`;`eval_runs` 每轮一行 `metrics = {"recall_at_10","mrr","faithfulness","refusal_rate"}`;趋势对比输出(↑/↓/→,下滑标 ⚠)。

**前置**:milvus + mysql 在线、上游可用,知识库已建。无单测(评估流水线本身就是验证工具;趋势格式跑两轮肉眼验收)。

- [ ] **Step 1: 写评估脚本**

`scripts/eval_flywheel.py`:

```python
"""ch09 自动化评估流水线:复用 ch04 评估集与指标,定期跑、落 eval_runs、连趋势。
指标:检索段 Recall@10 / MRR(hybrid_rerank,可答桶),生成段 Faithfulness + D 桶拒答率。
运行:make eval-flywheel(TRIGGER=手动|定时,默认手动)。cron 示例:
  0 6 * * * cd /path/to/mewhelp && make eval-flywheel TRIGGER=定时 >> log/eval.log 2>&1
趋势:读最近 10 轮,对比上一轮涨跌;任何指标下滑标 ⚠——README:「重点全在这条趋势线上」。
产物:dev-notes/ch09-eval-trend.txt
"""
import argparse
import asyncio
import pathlib
import sys

from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import FAITHFULNESS_PROMPT, RAG_ANSWER_PROMPT
from app.db import repository
from scripts.eval_ch04 import (
    GRADED_BUCKETS, K, _format_evidence, _hit_rank, _load, _mean, _refusal_one, _retrieve, _try,
)

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OUT = _ROOT / "dev-notes/ch09-eval-trend.txt"
_LINES: list[str] = []


def _log(msg=""):
    print(msg, flush=True)
    _LINES.append(msg)


class _Faith(BaseModel):
    faithful: bool = Field(description="是否忠实于证据")
    reason: str = Field(default="")


async def _gen_faith(s, hits, model, judge):
    evidence = _format_evidence(hits)
    ans = await _try((RAG_ANSWER_PROMPT | model).ainvoke(
        {"query": s["query"], "evidence": evidence}), f"gen:{s['id']}")
    if ans is None:
        return None
    text = ans.content if isinstance(ans.content, str) else str(ans.content)
    fa = await _try((FAITHFULNESS_PROMPT | judge).ainvoke(
        {"evidence": evidence, "answer": text}), f"faith:{s['id']}")
    return None if fa is None else bool(fa.faithful)


async def run_once() -> dict:
    samples = _load()
    graded = [s for s in samples if s["bucket"] in GRADED_BUCKETS]
    absent = [s for s in samples if s["bucket"] == "D_absent"]

    hits_list = await asyncio.gather(*(_retrieve("hybrid_rerank", s) for s in graded))
    ranks = [_hit_rank(h, s["expect_section"]) for s, h in zip(graded, hits_list)]
    recall = _mean([1.0 if 0 < rk <= K else 0.0 for rk in ranks])
    mrr = _mean([(1.0 / rk if rk else 0.0) for rk in ranks])

    model = get_chat_model()
    judge = get_chat_model().with_structured_output(_Faith)
    faiths = await asyncio.gather(*(
        _gen_faith(s, h, model, judge) for s, h in zip(graded, hits_list)))
    faithfulness = _mean([1.0 if f else 0.0 for f in faiths if f is not None])

    refusals = await asyncio.gather(*(_refusal_one(s) for s in absent))
    got = [r for r in refusals if isinstance(r, bool)]
    refusal_rate = sum(got) / len(got) if got else 0.0

    return {"dataset_size": len(samples),
            "metrics": {"recall_at_10": round(recall, 3), "mrr": round(mrr, 3),
                        "faithfulness": round(faithfulness, 3),
                        "refusal_rate": round(refusal_rate, 3)}}


def _trend(runs) -> None:
    """runs 按新→旧;打印每轮各指标,并与上一轮(更旧一行)对比涨跌,下滑标 ⚠。"""
    _log("\n=== 评估趋势(新在上)===")
    names = ["recall_at_10", "mrr", "faithfulness", "refusal_rate"]
    _log(f"{'轮次':>4s} {'时间':16s} {'触发':4s} " + "".join(f"{n:>16s}" for n in names))
    for i, r in enumerate(runs):
        prev = runs[i + 1].metrics if i + 1 < len(runs) else None
        row = f"#{r.id:>3d} {r.created_at:%m-%d %H:%M}   {r.triggered_by:4s} "
        for n in names:
            v = r.metrics.get(n)
            cell = f"{v:.3f}" if v is not None else "—"
            if prev and v is not None and prev.get(n) is not None:
                d = v - prev[n]
                cell += " ↑" if d > 0.005 else (" ⚠↓" if d < -0.005 else " →")
            row += f"{cell:>16s}"
        _log(row)
    latest, older = runs[0].metrics, (runs[1].metrics if len(runs) > 1 else None)
    if older:
        drops = [n for n in names if latest.get(n, 0) < older.get(n, 0) - 0.005]
        _log(f"\n⚠ 下滑指标:{', '.join(drops)}" if drops else "\n所有指标持平或上涨。")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triggered-by", default="手动", choices=["手动", "定时"])
    args = ap.parse_args()

    _log(f"=== ch09 评估流水线(触发:{args.triggered_by})===")
    result = await run_once()
    m = result["metrics"]
    _log(f"本轮:Recall@10={m['recall_at_10']:.3f} MRR={m['mrr']:.3f} "
         f"Faithfulness={m['faithfulness']:.3f} 拒答率={m['refusal_rate']:.3f}")
    await repository.insert_eval_run(args.triggered_by, result["dataset_size"], m)
    _trend(await repository.list_eval_runs(limit=10))
    _OUT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    _log(f"\n趋势报告已落 {_OUT.relative_to(_ROOT)}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
```

Makefile(`.PHONY` 同步):

```makefile
eval-flywheel:  ## ch09 评估流水线:复用 ch04 评估集,落 eval_runs 连趋势(TRIGGER=手动|定时)
	PYTHONPATH=. uv run python scripts/eval_flywheel.py --triggered-by $(or $(TRIGGER),手动)
```

注意:`from scripts.eval_ch04 import ...` 依赖 `PYTHONPATH=.` 下 scripts 作为命名空间包可导入,且 eval_ch04 模块级只有常量定义无副作用——执行时先 `uv run python -c "from scripts import eval_ch04"` 验证;若 import 失败(目录不是包),在 `scripts/` 下建空 `__init__.py`。

- [ ] **Step 2: 实跑两轮(验收 6 的素材)**

```bash
make eval-flywheel                 # 第一轮
make eval-flywheel TRIGGER=定时    # 第二轮
```

Expected: 第二轮输出趋势表,两行可对比,涨跌符号正确;`eval_runs` 表两行。截取输出记 dev-notes。

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_flywheel.py Makefile
git commit -m "feat(ch09): 评估流水线——ch04 指标定期跑落 eval_runs,趋势对比下滑标红"
```

---

### Task 11: Cost Control 统计脚本(按意图 token 账)

**Files:**
- Create: `scripts/cost_by_intent.py`
- Modify: `Makefile`(cost-report)
- Output: `dev-notes/ch09-cost-report.txt`

**Interfaces:**
- Consumes: `observability.get_langfuse()`(Task 2);Langfuse Metrics API(`client.api.metrics.get(query=...)`,v2 Metrics)。
- Produces: `make cost-report [DAYS=7]` → 意图 × 请求数 × 总 token × 平均 token 表。

**前置**:Langfuse 在跑且已积累带 `intent:` tag 的 trace(Task 3 之后聊过几句)。无单测(对外 API 查询脚本,实跑验证)。

- [ ] **Step 1: Context7 查证(执行时必做)**

查 `/langfuse/langfuse-docs` Metrics API:traces 视图按 `tags` 维度分组 + `totalTokens` sum 的确切 query 形状与返回行结构(tags 是数组维度,确认分组行为);若 tags 分组不可用,回退方案:逐意图 tag 过滤(`filters: [{"column":"tags","operator":"any of","value":["intent:物流"],"type":"arrayOptions"}]`,九类循环九次查询,不分组)。以实跑结果为准,脚本里两条路都留(主路失败自动走回退)。

- [ ] **Step 2: 写脚本**

`scripts/cost_by_intent.py`:

```python
"""ch09 Cost Control:按意图统计 token 花销(README:「按意图把 token 分堆一算,
哪类意图最烧钱立马现形」)。数据源 = Langfuse Metrics API(意图在 Task 3 打成 intent:xxx tag)。
运行:make cost-report(DAYS=N 窗口天数,默认 7;需 Langfuse 在跑且 .env 配好三变量)。
产物:dev-notes/ch09-cost-report.txt
说明:自定义模型名(glm-5.2 等)在 Langfuse 无内置单价,统计以 token 数为准;
要看钱在 Langfuse 界面配模型单价即可,不在本脚本范围。
"""
import argparse
import json
import pathlib
import sys
from datetime import datetime, timedelta, timezone

from app.core.observability import get_langfuse

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OUT = _ROOT / "dev-notes/ch09-cost-report.txt"
INTENTS = ["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "人工", "闲聊", "其他"]


def _window(days: int) -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return (now - timedelta(days=days)).strftime(fmt), now.strftime(fmt)


def _query_grouped_by_tags(client, frm: str, to: str) -> list[dict] | None:
    """主路:traces 视图按 tags 分组聚合 totalTokens/count。返回 None 表示不可用,走回退。"""
    try:
        resp = client.api.metrics.get(query=json.dumps({
            "view": "traces",
            "metrics": [{"measure": "totalTokens", "aggregation": "sum"},
                        {"measure": "count", "aggregation": "count"}],
            "dimensions": [{"field": "tags"}],
            "filters": [],
            "fromTimestamp": frm, "toTimestamp": to,
        }))
        data = getattr(resp, "data", None) or (resp.get("data") if isinstance(resp, dict) else None)
        if data is None:
            return None
        rows = []
        for row in data:
            row = row if isinstance(row, dict) else row.dict()
            tag = str(row.get("tags", ""))
            if tag.startswith("intent:"):
                rows.append({"intent": tag.removeprefix("intent:"),
                             "tokens": int(float(row.get("sum_totalTokens") or 0)),
                             "count": int(float(row.get("count_count") or 0))})
        return rows
    except Exception as e:
        print(f"[主路 tags 分组不可用:{type(e).__name__},走逐意图回退] {e}")
        return None


def _query_per_intent(client, frm: str, to: str) -> list[dict]:
    """回退:九类意图逐个 tag 过滤聚合(九次查询,不分组)。"""
    rows = []
    for intent in INTENTS:
        resp = client.api.metrics.get(query=json.dumps({
            "view": "traces",
            "metrics": [{"measure": "totalTokens", "aggregation": "sum"},
                        {"measure": "count", "aggregation": "count"}],
            "filters": [{"column": "tags", "operator": "any of",
                         "value": [f"intent:{intent}"], "type": "arrayOptions"}],
            "fromTimestamp": frm, "toTimestamp": to,
        }))
        data = getattr(resp, "data", None) or (resp.get("data") if isinstance(resp, dict) else [])
        tokens = count = 0
        for row in data:
            row = row if isinstance(row, dict) else row.dict()
            tokens += int(float(row.get("sum_totalTokens") or 0))
            count += int(float(row.get("count_count") or 0))
        if count:
            rows.append({"intent": intent, "tokens": tokens, "count": count})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()

    client = get_langfuse()
    if client is None:
        print("Langfuse 未配置(.env 三变量),无账可查。")
        return 1
    frm, to = _window(args.days)
    rows = _query_grouped_by_tags(client, frm, to)
    if rows is None:
        rows = _query_per_intent(client, frm, to)
    rows.sort(key=lambda r: r["tokens"], reverse=True)

    total = sum(r["tokens"] for r in rows) or 1
    lines = [f"=== 按意图 token 花销(近 {args.days} 天,数据源 Langfuse)===",
             f"{'意图':6s} {'请求数':>8s} {'总tokens':>12s} {'平均tokens':>12s} {'占比':>7s}"]
    for i, r in enumerate(rows):
        mark = "  ← 最烧钱" if i == 0 else ""
        lines.append(f"{r['intent']:6s} {r['count']:>8d} {r['tokens']:>12,d} "
                     f"{r['tokens'] // max(r['count'], 1):>12,d} {r['tokens'] / total:>6.0%}{mark}")
    if not rows:
        lines.append("(窗口内没有带 intent tag 的 trace——先聊几句再来)")
    out = "\n".join(lines)
    print(out)
    _OUT.write_text(out + "\n", encoding="utf-8")
    print(f"\n报告已落 {_OUT.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Makefile(`.PHONY` 同步):

```makefile
cost-report:  ## ch09 按意图 token 账(需 Langfuse 在跑;DAYS=窗口天数)
	PYTHONPATH=. uv run python scripts/cost_by_intent.py --days $(or $(DAYS),7)
```

- [ ] **Step 3: 实跑验证 + Commit**

先在聊天页用不同意图各发几条(物流/商品咨询/闲聊…),再:

Run: `make cost-report`
Expected: 表格按 token 降序,最烧钱意图有标记;数字与 Langfuse 界面 traces 大致对得上。若主路/回退的 API 形状与文档不符,按实测调整并记 dev-notes(这是外部 API,以实跑为准)。

```bash
git add scripts/cost_by_intent.py Makefile
git commit -m "feat(ch09): Cost Control——Langfuse 按意图聚合 token 账,主路分组+逐意图回退"
```

---

### Task 12: 审核后台页(Vibe Coding,不套 TDD)

**Files:**
- Create/Rewrite: `app/static/review.html`(Task 9 的占位换成真页面)

**Interfaces:**
- Consumes: Task 9 的四个 REST 接口。
- Produces: `/review` 页面,验收 2/3/4 的人工操作台。

- [ ] **Step 1: 实现页面**

单文件 HTML(照 index.html 的猫咪 CSS 变量与卡片风格,可直接复制其 `:root` 变量与基础样式段),功能清单:

1. **顶栏**:标题「飞轮待审队列」+ 状态筛选(待审/通过/驳回/全部,默认待审)+ 刷新按钮。
2. **三道闸提示条**(README 审核要点,静态文案):「审核前先过三道闸:① 垃圾过滤——乱输入、测试胡打、不当言论直接驳回;② 时效——强时效问题(活动截止类)不沉淀,驳回;③ 频次——低频冷门问题不值得占库,驳回。」
3. **列表**:表格列 = 标准化问题 / 出现次数(降序,高频标橙色徽标)/ 示例答案摘要(一行截断)/ 状态 / 操作(通过、驳回,仅待审行可点)。数据源 `GET /api/review/queue?status=`。
4. **行详情展开**(点行 toggle):调 `GET /api/review/{id}`,展示归并原话列表——每条带 source 标签(检索置信度低/模型自评不足/用户反馈,三色徽标)、时间、以及召回片段快照卡(每条快照:question/answer 原文 + rerank_score 得分条 + section_path;`retrieved_chunks` 为 null 显示「该入口未走检索,无快照」)。旁注一句:「对着快照判断:知识库真缺这块,还是有但没检到。」
5. **通过弹框**:点通过弹 modal,textarea 预填 `ai_suggested_answer`,确认后 `POST /api/review/{id}/approve` body `{"approved_answer": ...}`;成功 toast「已写回知识库,下次同类问题可直接检索命中」并刷新列表;失败(502)红 toast 显示 detail(上游/Milvus 不在线可重试)。
6. **驳回**:直接 `POST .../reject`,刷新行状态。
7. 无框架、原生 fetch,`ensure_ascii` 无关(前端);中文界面。
8. **挂上后台管理导航**:引 `app/static/admin.js`,`mountAdminNav("/review")` 把 ch03 那份共用导航摆在顶栏底下,并在导航注册表里加上本页一行,页面自己就不必再手写跨页按钮。页面路径仍是 `/review`。

- [ ] **Step 2: 浏览器验收(和用户 Vibe 迭代)**

起服务开 `http://localhost:8000/review`:造一条待审(问一个答不上的问题 + `make flywheel`),走一遍列表→详情→快照→通过/驳回。**截图给用户看效果,按用户描述改到满意为止**——本任务的完成判据是用户点头,不是测试绿。

- [ ] **Step 3: Commit**

```bash
git add app/static/review.html app/static/admin.js
git commit -m "feat(ch09): 审核后台页——队列/详情/快照/三道闸提示,通过驳回一站式"
```

---

### Task 13: 观测与成本页(三张报表进后台 + 只读 API + 作业注册)

**Files:**
- Create: `app/api/observability.py`、`app/static/observability.html`、`tests/test_observability_api.py`
- Modify: `scripts/cost_by_intent.py`、`scripts/calibrate_confidence.py`、`scripts/eval_flywheel.py`(产物落 `data/ch09/reports/`,前两个多写一份 json)、`app/core/jobs.py`(注册三个作业)、`app/api/admin.py`(多一张卡)、`app/static/admin.js`(导航多一格)、`app/main.py`(路由 + include)、`tests/test_admin_api.py`、`tests/test_static.py`

**Interfaces:**
- Consumes: Task 6 的校准产物、Task 10 的 `eval_runs` 表、Task 11 的成本产物、ch03 的作业运行器与后台外壳。
- Produces: `GET /api/observability/overview`、`/observability` 页面,验收 5/6/7 的看板。

- [ ] **Step 1: 产物落地(脚本侧)**

三个脚本的输出改到 `data/ch09/reports/`:`cost_by_intent.{txt,json}`、`confidence_calibration.{txt,json}`、`eval_trend.txt`。json 里带的数就是终端那份数,平均 token、占比、Youden J、推荐阈值一律在脚本里算完写进去,页面不再算一遍。评估趋势不落 json:`eval_runs` 表就是它的权威源,页面直接读表。

- [ ] **Step 2: 只读 API(TDD)**

RED 先写 `tests/test_observability_api.py`:①三块都没产物 → 200 + 各自 `present=false` + 该按哪个作业;②成本产物原样透出(`rows` 逐字段相等,最烧钱那路取产物第一行);③跑过但窗口内没 trace → `status=missing` + 提示先聊几句;④趋势读 `eval_runs` 新在上;⑤推荐阈值与 `settings.evidence_confidence_threshold` 不一致时 `in_sync=false`;⑥产物写坏半个 json 按「没跑过」处理,不 500;⑦`list_eval_runs` 抛错只让趋势那块 `status=error`,成本与校准照样出。GREEN 实现 `app/api/observability.py`。

- [ ] **Step 3: 作业注册 + 页面**

`app/core/jobs.py` 加三条:`cost-report`(需 Langfuse 在跑)、`eval-flywheel`、`calibrate-confidence`(后两个 `heavy=True`,页面二次确认)。页面 `observability.html` 复用 `acceptance.css` 与 `acceptance.js`(`jobRow` / `missingBox` / `toast`),三块各一个 panel:成本账给占比条 + 明细表,趋势给最近十轮表格与涨跌箭头,校准给分布表 + 阈值扫描双曲线(选定线与在用线各画一条)。每块底下一个重跑按钮 + 共用日志窗口。

- [ ] **Step 4: 挂进后台外壳**

`admin.js` 导航在「飞轮待审」后面加一格「观测与成本」;`admin.py` 多一张卡(最烧钱占比 / 评估轮次 / 忠实度 / 在用阈值四个数,推荐值与在用值不一致时标要干活);`main.py` 加 `/observability` 路由与 router include。`tests/test_admin_api.py` 的卡片清单加 `observability`(并把 `list_eval_runs` 也纳入「mysql 全挂」那条),`tests/test_static.py` 加 `/observability` 一行。

- [ ] **Step 5: 浏览器验收**

`make dev` 后开 `http://localhost:8000/observability`:三块的数与终端 `make cost-report` / `make eval-flywheel` / `make calibrate-confidence` 的输出逐个对上;按一次「重跑 意图成本账」,日志尾在页面上回显、跑完自动刷新。

- [ ] **Step 6: Commit**

```bash
uv run pytest -q
git add app/api/observability.py app/static/observability.html tests/test_observability_api.py \
        scripts/cost_by_intent.py scripts/calibrate_confidence.py scripts/eval_flywheel.py \
        app/core/jobs.py app/api/admin.py app/static/admin.js app/main.py \
        tests/test_admin_api.py tests/test_static.py data/ch09/reports
git commit -m "feat(ch09): 观测与成本页——意图成本账、评估趋势、置信度校准三张报表进后台"
```

---

### Task 13.5: 图底下那句「读图」交给模型写(产物落盘时生成 + 数字校验)

**Files:**
- Create: `app/core/read_notes.py`、`tests/test_read_notes.py`
- Modify: `scripts/eval_ch04.py`、`scripts/cost_by_intent.py`、`scripts/calibrate_confidence.py`、`scripts/eval_flywheel.py`、`app/api/rageval.py`、`app/api/observability.py`、`app/static/acceptance.js`、`app/static/rageval.html`、`app/static/observability.html`、`tests/test_rageval_api.py`、`tests/test_observability_api.py`

**Interfaces:**
- Consumes: Task 13 的三份产物与 `/rag-eval` 的评估产物、`app/core/llm.get_chat_model`。
- Produces: 各产物里的 `read_notes` 字段 + `eval_trend_note.json`,两页的「读图」那一行。

- [ ] **Step 1: 生成器 + 数字校验(TDD)**

RED 先写 `tests/test_read_notes.py`:①原样 / 四舍五入 / 百分比 / 千分位逗号 / 比例余数都算「数据里有的数」;②标识符里的数字(`p25`、`Recall@10`、`bge-m3`)不当数字看;③数据里没有的数 → 整条丢;④超长丢;⑤上游抛错回 `None` 不算失败;⑥全角逗号与分号归一成半角(跟页面兜底句同一口径);⑦每类图都得交代「画的是什么」与「读者要做什么判断」。GREEN 实现 `app/core/read_notes.py`(`KINDS` / `verify` / `generate` / `generate_all`,超时 120s——产物落盘时的一次性开销,不是页面等待)。

- [ ] **Step 2: 四个脚本落注**

各脚本在写产物前调 `generate`,payload **给中文标签**(`纯 BM25` / `口语类` / `单均 token`),不喂字段名。`rag_eval.json` / `cost_by_intent.json` / `confidence_calibration.json` 多一个 `read_notes`;趋势的权威源是 `eval_runs` 表,注只能旁挂 `eval_trend_note.json` 且带上它描述的轮次 id。生成失败只打一行日志,产物照样落盘。

- [ ] **Step 3: API 透出(TDD)**

`/api/rag-eval/overview` 加 `read_notes`(缺则空字典);`/api/observability/overview` 三块各加 `read_note`,趋势那句只在注记的轮次仍是最新一轮时才端。测试:注原样透出、缺注是 `null` 不是错误、注记的轮次过期就不端。

- [ ] **Step 4: 页面优先用注**

`acceptance.js` 加 `readNoteHtml(note, fallbackHtml)`:转义后把数字自动加粗,没注就用页面自己那句。两页四处 `.read` 改成走它。

- [ ] **Step 5: 浏览器验收**

跑一次带注的产物,`/rag-eval` 与 `/observability` 上的读图换成模型写的那句,数字加粗;把注从产物里删掉再刷新,页面回落兜底句,不白屏。

- [ ] **Step 6: Commit**

```bash
uv run pytest -q
git add app/core/read_notes.py tests/test_read_notes.py scripts app/api app/static tests data
git commit -m "feat(ch09): 图底下那句读图改由模型看着当轮的数写,落盘时生成并校验每个数字"
```

---

### Task 14: 端到端验收 + code review + finish

**Files:** 无新代码(修复除外);产物为 dev-notes 验收记录。

- [ ] **Step 1: 全量测试**

`uv run pytest -v` 全绿;`make flywheel-samples` ≥ 80%。

- [ ] **Step 2: 六条验收逐条实跑**

前置:`make langfuse-up` + mysql/milvus/MCP + `make dev` 全起。

| # | 验收 | 操作 | 通过判据 |
|---|---|---|---|
| 1 | 完整链路 | 聊天页发「猫爬架保修多久」,开 localhost:3000 | trace 树铺开:resolve_reference→classify_intent→retrieve_knowledge→…每节点 prompt/检索结果/token/耗时可见;session 归属该会话 |
| 2 | 答不上→兜底+入队+详情 | 问知识库没有的(如「猫窝能机洗吗」)→ 收到兜底话术 → `make flywheel` → /review 详情 | 兜底话术+转人工按钮;待审队列出现标准化问题;详情见原话 + source=检索置信度低 + 快照(原文+得分) |
| 3 | 飞轮整圈 | 在 /review 给上一条填核准答案点通过 → 聊天页再问同一问题 | 答案命中新知识(带引用),不再兜底 |
| 4 | 👎 落池入队 | 对任一回答点 👎 → `make flywheel` → /review | low_confidence_questions 出现 user_feedback 行(走过检索的带快照);标准化查重后出现在待审队列(与已有缺口同义则次数+1) |
| 5 | 意图 token 账 | 各意图聊几句 → `make cost-report` → 开 /observability | 表格按意图汇总 token,烧钱大户置顶;页面上那张成本账与终端输出同一份数 |
| 6 | 评估趋势 | `make eval-flywheel` 两轮(Task 10 已跑过则再补一轮) | 趋势表 ≥2 行,涨跌符号/⚠ 正确;eval_runs 表有对应行,页面趋势那块同源 |
| 7 | 报表在页面上看也在页面上按 | 开 /observability,按「重跑 意图成本账」 | 三块都有数;日志尾回显在页面上,跑完自动刷新;校准那块的推荐值与在用阈值对得上 |

每条验收把命令与关键输出记进 `dev-notes/ch09.md`;验收 1/2/3/7 截图。

- [ ] **Step 3: code review**

按 superpowers:requesting-code-review 对 ch09 全部 diff 跑一轮 review,修复确认的问题(修复后重跑全量测试),结论记 dev-notes。

- [ ] **Step 4: finish**

按 superpowers:finishing-a-development-branch 收尾;交付物汇总(演示命令清单、测试结果、dev-notes/ch09.md 路径)交给用户。

---

## Self-Review 记录(计划自审)

1. **Spec 覆盖**:spec §3(部署/配置/挂载/usage 收口/意图元数据)→ Task 2/3;§4 Cost → Task 11;§5 置信度闸(纯函数/校准/接线/落池)→ Task 4/5/6;§6 反馈入口 → Task 7;§7 飞轮 → Task 1/8;§8 审核+写回 → Task 9;§9 评估 → Task 10;§10 前端 → Task 7 Step5 + Task 12,§10.2 观测与成本页 → Task 13;§11 测试策略 → 各任务内嵌;§12 验收 → Task 14。无缺口。
2. **占位符扫描**:Task 6 阈值「<实测值>」是校准产出的回填位,非占位符(不许拍脑袋是需求);Task 2/11 的 Context7 查证步骤是工作要求 3 的显式步骤。其余步骤代码齐全。
3. **类型一致性**:`insert_low_confidence(..., retrieved_chunks=None)` 在 Task 1 定义、Task 5/7 按同签名调;`NormalizeResult` 字段与 README JSON 三键一致;`get_review_detail` 返回 tuple 在 Task 9 解包一致;`retrieved_snapshot`(state)/`retrieved_chunks`(DB 列)两个名字刻意区分层次,各任务用法一致。
