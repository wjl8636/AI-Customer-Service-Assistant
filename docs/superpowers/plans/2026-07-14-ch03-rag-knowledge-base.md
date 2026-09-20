# Ch03 知识库向量语义检索 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `query_faq` 的内部实现从关键词查表升级为 BGE-M3 向量语义检索,建起 MySQL 权威源 + Milvus 双写的知识库,入参出参契约不变;建库这条链落一页后台 `/kb`,录入、切块预览、补向量、检索自测都在浏览器里做。

**Architecture:** 两条离线链路(文档结构感知切分 / 历史对话挖 QA)把知识写进 MySQL `knowledge_chunks`(status=pending),一个幂等可重跑的 vectorize job 取 pending → BGE-M3 嵌入 → Milvus `knowledge` 集合(按 id upsert)→ 回填 `vector_id`、status=done。在线 `query_faq` 把问题向量化后到 Milvus 取 Top-K,直接从 Milvus 取回 question/answer。 录入页 `/kb` 复用同一套切块与双写:手工贴的正文走 `/api/kb/ingest`(先写 MySQL 再向量化),离线那几步走 `/api/jobs` 的 make 白名单;几块后台共用一份导航,聚合首页在 `/admin`。

**Tech Stack:** FastAPI + LangChain 1.3(text-splitters)+ openai SDK(直连 SiliconFlow BGE-M3)+ pymilvus(Milvus Lite 嵌入式)+ SQLAlchemy 2.0 async / MySQL。

## Global Constraints

- Python `>=3.12`,包管理 `uv`(`uv run` / `uv add`)。
- **应用只说 OpenAI 协议**;嵌入走 openai SDK 打 `settings.embed_base_url` 的 `/v1/embeddings`,地址和 key 只从配置来,不写死在代码里。
- **`query_faq` 入参出参契约必须一字不改**:入参 `keyword: str`;出参命中 `{"hits": [{"question", "answer"}]}`、未命中 `{"hits": [], "message": ...}`。
- **固定技术选型**:嵌入 BGE-M3(SiliconFlow)、向量库 Milvus(Lite)、MySQL 权威源。本章只做 dense 单路;**不做**关键词召回 / 混合检索 / 重排;**不引入** LangGraph / Langfuse。实现中若发现选型矛盾或走不通,**停下问用户,不自行换方案**。
- **密钥**:`SILICONFLOW_API_KEY` 只存 `.env`(已 gitignore),严禁写入任何提交文件 / 文档 / 留痕。
- **表结构以 `sql/ch03-ddl.sql` 为权威**:ORM 只映射不反向建表。
- **不改动 ch01/ch02 行为**:`app/api/chat.py`、`app/core/agent.py` 编排、`/api/agent`、工具注册表逐字节不动;本章只改 `query_faq` 的**内部实现**。
- **测试策略**:确定性代码走 TDD;纯 Prompt(挖知识抽取)与语义召回质量走标注样例 / eval。**Milvus Lite 是本地文件、单测用临时 db 跑真实向量库**;**嵌入在单测里 mock**(不打真实 API),真实链路只在冒烟 / eval / 验收里验。
- **页面上的数只从各模块自己的读数函数来,不在接口里另算一份**:切块预览与录入都调 `documents.build_chunks`,向量化都调 `dualwrite.vectorize_pending`,材料清单只有 `app/kb/sources.py` 一份——页面显示的和终端跑出来的必须是同一个真相。
- **页面有重跑按钮,但没有 shell**:作业名走白名单,argv 写死在 `app/core/jobs.py`,前端只传作业名;命令一律走 make,配方是仓库里那一份。
- **依赖缺失是状态而不是错误**:mysql / Milvus / 嵌入上游任一不可用,只让相关那几个数显示「读不到」,不连坐整页,也不把「读不到」说成「对不上」。
- 每个任务结束提交一次。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `app/config.py`(改) | 加 `embed_model / milvus_uri / retrieval_top_k / retrieval_min_score` |
| `.env.example`(改) | 加 `SILICONFLOW_API_KEY` 占位符 |
| `.gitignore`(改) | 加 `data/*.db*`(Milvus Lite 文件) |
| `pyproject.toml`(改) | 加 `pymilvus`、`openai` 依赖 |
| `app/core/embeddings.py` | 直连 `/v1/embeddings` 调 bge-m3:`embed_texts` / `embed_query` |
| `app/kb/milvus_client.py` | Milvus Lite 客户端 + 集合 ensure / upsert / search |
| `app/db/models.py`(改) | 加 `KnowledgeChunk`、`QaExtractionStaging` ORM(只映射) |
| `app/db/repository.py`(改) | 加知识 chunk / 暂存表 CRUD |
| `app/kb/chunking.py` | 切分原语:标题切 / 递归切 / 句末重叠 / 表格按行切复表头 |
| `app/kb/documents.py` | 源 `.md` → `Chunk` 记录(字段映射) |
| `app/kb/dualwrite.py` | 写 pending + vectorize job(幂等可重跑) |
| `app/kb/dedup.py` | 暂存表文本归一化去重 |
| `app/kb/mining.py` | 对话 → LLM 抽 QA(结构化输出) |
| `scripts/smoke_embed.py` | 冒烟:直连 SiliconFlow,验维度 |
| `scripts/build_kb.py` | CLI:`data/kb/*.md` → chunks → MySQL pending |
| `scripts/vectorize_kb.py` | CLI:pending → Milvus → done(幂等) |
| `scripts/mine_knowledge.py` | CLI:对话 → staging → 去重 → MySQL pending(可重跑 job) |
| `scripts/eval_retrieval.py` | eval:换说法 → 期望命中块(验收1) |
| `scripts/eval_mining.py` | eval:样例对话 → 期望问答对 |
| `data/kb/*.md` | 源知识文档(夹具,提交) |
| `tests/data/{retrieval,mining}_samples.json` | 标注样例 |
| `tests/conftest.py`(改) | 测试库也建 ch03 两表 |
| `app/kb/sources.py` | 建库材料清单(文件 → content_type),CLI 与录入页同读 |
| `app/api/kb.py` | 录入页 API:概览 / 切块预览 / 录入 / 向量化 / 检索自测 / 暂存表 |
| `app/core/jobs.py` | 作业运行器:白名单 make 目标 + 日志落盘 + 长活能停 |
| `app/api/jobs.py` | 作业 API:发起 / 状态与日志尾 / 停止 |
| `app/api/admin.py` | 后台首页聚合:每模块一张卡,依赖没起只影响它自己 |
| `app/static/kb.html` | 知识库录入页(录入 / 材料 / 挖知识 / 双写 / 检索自测 / 最近入库) |
| `app/static/admin.html` | 后台管理首页(模块卡片) |
| `app/static/admin.js` | 后台各页共用导航(自带样式注入) |
| `app/main.py`(改) | 加 `/kb`、`/admin` 页面路由与静态目录挂载 |
| `Makefile`(改) | 加 `kb-preview / kb-build / kb-vectorize / kb-mine / kb-reset / eval-retrieval` 目标 |

---

## Task 1: 冒烟风险闸 + 基础设施接线(go/no-go)

**非 TDD**(照 ch02 Task 1,真实服务冒烟由控制器/主会话亲自做)。验 SiliconFlow BGE-M3 能否跑通、维度是否 1024。跑不通则**停下问用户**,只在 SiliconFlow/BGE-M3 内换法。

**Files:**
- Modify: `.env.example`, `.gitignore`, `pyproject.toml`, `app/config.py`
- Create: `scripts/smoke_embed.py`

**Interfaces:**
- Produces: `settings.embed_base_url / embed_api_key / embed_model / milvus_uri / retrieval_top_k / retrieval_min_score`。

- [ ] **Step 1: 加依赖**

Run: `uv add pymilvus openai`
Expected: `pyproject.toml` 出现 `pymilvus`、`openai`,`uv.lock` 更新。

- [ ] **Step 2: .env.example 加嵌入那一组,.gitignore 加 Milvus 文件**

第 1 章只配了聊天那一组,这一章加嵌入。Edit `.env.example`:

```bash
# 嵌入 BAAI/bge-m3。BASE_URL 有内置默认值(硅基流动),换上游才需要填
EMBED_API_KEY=sk-xxx
# EMBED_BASE_URL=https://api.siliconflow.cn/v1
```

Edit `.gitignore`,追加一行:

```
data/*.db*
```

(真实 key 已在 `.env`,勿动、勿提交。)

- [ ] **Step 3: config.py 加配置项**

Edit `app/config.py`,在 `test_database_url` 下方加字段:

```python
    embed_base_url: str = "https://api.siliconflow.cn/v1"
    embed_api_key: str                      # 跟着账号走,不给默认值
    embed_model: str = "BAAI/bge-m3"        # 上游真实名,没有别名这一层
    milvus_uri: str = "data/milvus_knowledge.db"
    retrieval_top_k: int = 3
    retrieval_min_score: float = 0.4
```

`embed_model` 这里要注意:写的是上游认的真名 `BAAI/bge-m3`,带 `BAAI/` 前缀。写成 `bge-m3`
不会在启动时报错,而是第一次调嵌入时回一句 `Model does not exist`。

- [ ] **Step 4: 写冒烟脚本**

Create `scripts/smoke_embed.py`:

```python
"""冒烟:直连 SiliconFlow BGE-M3,验证连通与维度。
需 .env 里填好 EMBED_API_KEY。
运行:PYTHONPATH=. uv run python scripts/smoke_embed.py"""
import asyncio

from openai import AsyncOpenAI

from app.config import settings


async def main() -> None:
    client = AsyncOpenAI(base_url=settings.embed_base_url, api_key=settings.embed_api_key)
    resp = await client.embeddings.create(model=settings.embed_model, input=["邮费是多少", "运费怎么算"])
    dims = [len(d.embedding) for d in resp.data]
    print(f"返回 {len(resp.data)} 条向量,维度={dims}")
    assert dims and dims[0] == 1024, f"期望维度 1024,实际 {dims}"
    print("✅ 冒烟通过:直连 SiliconFlow BGE-M3,维度 1024")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 5: 跑冒烟**

Run:
```bash
sleep 5
PYTHONPATH=. uv run python scripts/smoke_embed.py
```
Expected: `✅ 冒烟通过:直连 SiliconFlow BGE-M3,维度 1024`。
**若失败**(鉴权/路由/维度不符):停下,把报错 surface 给用户,在 SiliconFlow/BGE-M3 内讨论换法(如 `Pro/BAAI/bge-m3`),不自行换选型。

- [ ] **Step 7: 提交**

```bash
git add pyproject.toml uv.lock .env.example .gitignore app/config.py scripts/smoke_embed.py
git commit -m "feat(ch03): 基础设施接线 + SiliconFlow BGE-M3 冒烟(维度 1024 GO)"
```

---

## Task 2: 嵌入客户端 `app/core/embeddings.py`

**TDD**(单测 mock openai 客户端,不打真实 API)。

**Files:**
- Create: `app/core/embeddings.py`
- Test: `tests/test_embeddings.py`

**Interfaces:**
- Consumes: `settings.embed_base_url / embed_api_key / embed_model`。
- Produces: `async embed_texts(texts: list[str]) -> list[list[float]]`;`async embed_query(text: str) -> list[float]`;模块级 `_client() -> AsyncOpenAI`(供测试 monkeypatch)。

- [ ] **Step 1: 写失败测试**

Create `tests/test_embeddings.py`:

```python
import pytest

from app.core import embeddings


class _FakeEmb:
    def __init__(self, vec):
        self.embedding = vec


class _FakeResp:
    def __init__(self, vecs):
        self.data = [_FakeEmb(v) for v in vecs]


class _FakeEmbeddings:
    def __init__(self):
        self.calls = []

    async def create(self, model, input):
        self.calls.append((model, list(input)))
        return _FakeResp([[float(i), 0.0, 1.0] for i, _ in enumerate(input)])


class _FakeClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddings()


async def test_embed_texts_passes_model_and_input(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(embeddings, "_client", lambda: fake)
    out = await embeddings.embed_texts(["a", "b"])
    assert out == [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
    assert fake.embeddings.calls == [("bge-m3", ["a", "b"])]


async def test_embed_query_returns_single_vector(monkeypatch):
    monkeypatch.setattr(embeddings, "_client", lambda: _FakeClient())
    v = await embeddings.embed_query("邮费")
    assert v == [0.0, 0.0, 1.0]
```

- [ ] **Step 2: 运行看失败**

Run: `uv run pytest tests/test_embeddings.py -v`
Expected: FAIL(`ModuleNotFoundError: app.core.embeddings`)。

- [ ] **Step 3: 实现**

Create `app/core/embeddings.py`:

```python
from openai import AsyncOpenAI

from app.config import settings


def _client() -> AsyncOpenAI:
    """直连嵌入上游(硅基流动的 bge-m3,OpenAI 兼容)。"""
    return AsyncOpenAI(base_url=settings.embed_base_url, api_key=settings.embed_api_key)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = await _client().embeddings.create(model=settings.embed_model, input=texts)
    return [d.embedding for d in resp.data]


async def embed_query(text: str) -> list[float]:
    return (await embed_texts([text]))[0]
```

- [ ] **Step 4: 运行看通过**

Run: `uv run pytest tests/test_embeddings.py -v`
Expected: PASS(2 passed)。

- [ ] **Step 5: 提交**

```bash
git add app/core/embeddings.py tests/test_embeddings.py
git commit -m "feat(ch03): 嵌入客户端(直连 bge-m3,embed_texts/embed_query)"
```

---

## Task 3: Milvus 客户端 `app/kb/milvus_client.py`

**TDD**,单测用临时 Milvus Lite 文件跑**真实**向量库。

**Files:**
- Create: `app/kb/__init__.py`, `app/kb/milvus_client.py`
- Test: `tests/test_milvus_client.py`

**Interfaces:**
- Consumes: `settings.milvus_uri`。
- Produces:
  - `COLLECTION = "knowledge"`、`DIM = 1024`
  - `get_client(uri: str | None = None) -> MilvusClient`
  - `ensure_collection(client) -> None`
  - `upsert_vectors(client, rows: list[dict]) -> None`(row 键:`id:int, vector:list[float], question:str, answer:str`)
  - `search(client, vector: list[float], top_k: int) -> list[dict]`(返回 `{"id","score","question","answer"}`)
  - `count(client) -> int`

- [ ] **Step 1: 写失败测试**

Create `tests/test_milvus_client.py`:

```python
import pytest

from app.kb import milvus_client as mc


@pytest.fixture()
def client(tmp_path):
    c = mc.get_client(uri=str(tmp_path / "t.db"))
    mc.ensure_collection(c)
    return c


def _vec(seed: float) -> list[float]:
    # 造 1024 维、方向可区分的向量
    v = [0.0] * mc.DIM
    v[0] = seed
    v[1] = 1.0 - seed
    return v


def test_upsert_then_search_returns_fields(client):
    mc.upsert_vectors(client, [
        {"id": 1, "vector": _vec(1.0), "question": "运费怎么算", "answer": "满99包邮"},
        {"id": 2, "vector": _vec(0.0), "question": "发货时效", "answer": "48小时内发货"},
    ])
    hits = mc.search(client, _vec(0.98), top_k=1)
    assert len(hits) == 1
    assert hits[0]["id"] == 1
    assert hits[0]["question"] == "运费怎么算"
    assert hits[0]["answer"] == "满99包邮"
    assert isinstance(hits[0]["score"], float)


def test_upsert_is_idempotent_by_pk(client):
    row = {"id": 1, "vector": _vec(1.0), "question": "q", "answer": "a"}
    mc.upsert_vectors(client, [row])
    mc.upsert_vectors(client, [row])  # 同 id 重写
    assert mc.count(client) == 1
```

- [ ] **Step 2: 运行看失败**

Run: `uv run pytest tests/test_milvus_client.py -v`
Expected: FAIL(`ModuleNotFoundError: app.kb.milvus_client`)。

- [ ] **Step 3: 实现**

Create `app/kb/__init__.py`(空文件)。

Create `app/kb/milvus_client.py`:

```python
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from app.config import settings

COLLECTION = "knowledge"
DIM = 1024


def get_client(uri: str | None = None) -> MilvusClient:
    return MilvusClient(uri=uri or settings.milvus_uri)


def ensure_collection(client: MilvusClient) -> None:
    """幂等:已存在则确保加载;不存在则按 schema 建集合 + AUTOINDEX/COSINE 索引 + 加载。
    主键 id = MySQL knowledge_chunks.id(auto_id=False,由我方提供),使重跑按 id upsert 幂等。"""
    if client.has_collection(COLLECTION):
        client.load_collection(COLLECTION)
        return
    schema = CollectionSchema([
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema("vector", DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema("question", DataType.VARCHAR, max_length=2048),
        FieldSchema("answer", DataType.VARCHAR, max_length=8192),
    ])
    client.create_collection(COLLECTION, schema=schema)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_index(COLLECTION, index_params)
    client.load_collection(COLLECTION)


def upsert_vectors(client: MilvusClient, rows: list[dict]) -> None:
    if rows:
        client.upsert(COLLECTION, rows)


def search(client: MilvusClient, vector: list[float], top_k: int) -> list[dict]:
    res = client.search(
        COLLECTION, data=[vector], limit=top_k,
        output_fields=["question", "answer"],
        search_params={"metric_type": "COSINE"},
    )
    return [
        {"id": h["id"], "score": float(h["distance"]),
         "question": h["entity"]["question"], "answer": h["entity"]["answer"]}
        for h in res[0]
    ]


def count(client: MilvusClient) -> int:
    return client.query(COLLECTION, filter="id >= 0", output_fields=["count(*)"])[0]["count(*)"]
```

- [ ] **Step 4: 运行看通过**

Run: `uv run pytest tests/test_milvus_client.py -v`
Expected: PASS(2 passed)。若 `search`/`count` 结果结构与断言不符,按真实 Milvus Lite 返回结构修正解析(这正是本任务用真实 Lite 兜住的点)。

- [ ] **Step 5: 提交**

```bash
git add app/kb/__init__.py app/kb/milvus_client.py tests/test_milvus_client.py
git commit -m "feat(ch03): Milvus Lite 客户端(集合 ensure/upsert/search,id 主键 COSINE)"
```

---

## Task 4: ORM 模型 + 测试库建 ch03 表

**TDD**(DB 往返)。先扩 conftest 让测试库也建 ch03 两表,再加 ORM。

**Files:**
- Modify: `app/db/models.py`, `tests/conftest.py`
- Test: `tests/test_kb_models.py`

**Interfaces:**
- Produces: `KnowledgeChunk`(字段同 `sql/ch03-ddl.sql`)、`QaExtractionStaging` ORM 类。

- [ ] **Step 1: 扩 conftest 建 ch03 表**

Edit `tests/conftest.py`:把单一 `_DDL` 改为两个 DDL 文件,`_TABLES` 加 ch03 两表,`_create_table_stmts` 遍历两文件。

替换顶部 `_DDL` / `_TABLES` 定义与 `_create_table_stmts`:

```python
_DDL_FILES = [
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch02-ddl.sql",
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch03-ddl.sql",
]
# 删除顺序:先子表后父表;knowledge_chunks 自引用 FK 靠 FOREIGN_KEY_CHECKS=0 兜
_TABLES = ["messages", "tickets", "conversations", "faq",
           "qa_extraction_staging", "knowledge_chunks"]


def _create_table_stmts() -> list[str]:
    stmts: list[str] = []
    for ddl in _DDL_FILES:
        raw = ddl.read_text(encoding="utf-8")
        sql = "\n".join(ln for ln in raw.splitlines() if not ln.lstrip().startswith("--"))
        stmts += [s.strip() for s in sql.split(";") if s.strip() and "CREATE TABLE" in s.upper()]
    return stmts
```

- [ ] **Step 2: 写失败测试**

Create `tests/test_kb_models.py`:

```python
from sqlalchemy import select

from app.db.models import KnowledgeChunk, QaExtractionStaging


async def test_knowledge_chunk_roundtrip(db_session_factory):
    async with db_session_factory() as s:
        row = KnowledgeChunk(category="售后政策", questions="运费说明", answer="满99包邮")
        s.add(row)
        await s.commit()
        await s.refresh(row)
        assert row.id is not None
        assert row.vectorize_status == "pending"
        assert row.is_key_clause == 0


async def test_staging_roundtrip(db_session_factory):
    async with db_session_factory() as s:
        row = QaExtractionStaging(batch_no="b1", question="邮费多少", answer="满99包邮")
        s.add(row)
        await s.commit()
        got = (await s.execute(select(QaExtractionStaging))).scalars().all()
        assert len(got) == 1
        assert got[0].status == "extracted"
```

- [ ] **Step 3: 运行看失败**

Run: `uv run pytest tests/test_kb_models.py -v`
Expected: FAIL(`ImportError: cannot import name 'KnowledgeChunk'`)。

- [ ] **Step 4: 实现 ORM**

Edit `app/db/models.py`,末尾追加(顶部 import 需含 `Integer`;`BigInteger, DateTime, Enum, String, Text, func` 已在):

```python
from sqlalchemy import Integer  # 若未导入则加


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(255))
    questions: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_key_clause: Mapped[int] = mapped_column(Integer, server_default="0")
    prev_chunk_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    next_chunk_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vectorize_status: Mapped[str] = mapped_column(
        Enum("pending", "done"), server_default="pending"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class QaExtractionStaging(Base):
    __tablename__ = "qa_extraction_staging"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    batch_no: Mapped[str] = mapped_column(String(64))
    source_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        Enum("extracted", "kept", "discarded"), server_default="extracted"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

- [ ] **Step 5: 运行看通过**

Run: `uv run pytest tests/test_kb_models.py -v`
Expected: PASS(2 passed)。

- [ ] **Step 6: 提交**

```bash
git add app/db/models.py tests/conftest.py tests/test_kb_models.py
git commit -m "feat(ch03): KnowledgeChunk/QaExtractionStaging ORM + 测试库建 ch03 表"
```

---

## Task 5: 知识 / 暂存 repository CRUD

**TDD**(DB 往返)。repository 内部用 `db.async_session`(模块属性,受 conftest monkeypatch)。

**Files:**
- Modify: `app/db/repository.py`
- Test: `tests/test_kb_repository.py`

**Interfaces:**
- Produces(全部 async):
  - `insert_knowledge_chunk(category, questions, answer, section_path=None, content_type=None, is_key_clause=0) -> int`
  - `list_pending_chunks() -> list[KnowledgeChunk]`
  - `mark_chunk_vectorized(chunk_id: int, vector_id: str) -> None`
  - `set_chunk_neighbors(chunk_id: int, prev_id: int | None, next_id: int | None) -> None`
  - `count_chunks_by_status(status: str) -> int`
  - `list_all_questions() -> list[str]`(knowledge_chunks 已有问法,供去重)
  - `insert_staging(batch_no, source_ref, question, answer) -> int`
  - `list_staging_by_status(status: str) -> list[QaExtractionStaging]`
  - `set_staging_status(ids: list[int], status: str) -> None`

- [ ] **Step 1: 写失败测试**

Create `tests/test_kb_repository.py`:

```python
from app.db import repository


async def test_insert_and_list_pending(db_session_factory):
    cid = await repository.insert_knowledge_chunk("cat", "运费说明", "满99包邮", content_type="policy")
    pending = await repository.list_pending_chunks()
    assert [c.id for c in pending] == [cid]
    assert pending[0].questions == "运费说明"


async def test_mark_vectorized_removes_from_pending(db_session_factory):
    cid = await repository.insert_knowledge_chunk("cat", "q", "a")
    await repository.mark_chunk_vectorized(cid, str(cid))
    assert await repository.list_pending_chunks() == []
    assert await repository.count_chunks_by_status("done") == 1


async def test_set_neighbors(db_session_factory):
    a = await repository.insert_knowledge_chunk("c", "qa", "aa")
    b = await repository.insert_knowledge_chunk("c", "qb", "ab")
    await repository.set_chunk_neighbors(b, prev_id=a, next_id=None)
    pending = {c.id: c for c in await repository.list_pending_chunks()}
    assert pending[b].prev_chunk_id == a


async def test_staging_flow(db_session_factory):
    i1 = await repository.insert_staging("b1", "conv:1", "邮费多少", "满99包邮")
    await repository.insert_staging("b1", "conv:2", "怎么退货", "7天无理由")
    extracted = await repository.list_staging_by_status("extracted")
    assert len(extracted) == 2
    await repository.set_staging_status([i1], "discarded")
    assert len(await repository.list_staging_by_status("extracted")) == 1
    assert len(await repository.list_staging_by_status("discarded")) == 1


async def test_list_all_questions(db_session_factory):
    await repository.insert_knowledge_chunk("c", "运费怎么算", "满99包邮")
    assert "运费怎么算" in await repository.list_all_questions()
```

- [ ] **Step 2: 运行看失败**

Run: `uv run pytest tests/test_kb_repository.py -v`
Expected: FAIL(`AttributeError: module 'app.db.repository' has no attribute 'insert_knowledge_chunk'`)。

- [ ] **Step 3: 实现**

Edit `app/db/repository.py`,顶部 import 加 `KnowledgeChunk, QaExtractionStaging`,末尾追加:

```python
from app.db.models import KnowledgeChunk, QaExtractionStaging  # 合并进已有 import 行


async def insert_knowledge_chunk(
    category: str, questions: str, answer: str,
    section_path: str | None = None, content_type: str | None = None,
    is_key_clause: int = 0,
) -> int:
    async with db.async_session() as s:
        row = KnowledgeChunk(
            category=category, questions=questions, answer=answer,
            section_path=section_path, content_type=content_type,
            is_key_clause=is_key_clause,
        )
        s.add(row)
        await s.commit()
        return row.id


async def list_pending_chunks() -> list[KnowledgeChunk]:
    async with db.async_session() as s:
        result = await s.execute(
            select(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == "pending")
            .order_by(KnowledgeChunk.id)
        )
        return list(result.scalars())


async def mark_chunk_vectorized(chunk_id: int, vector_id: str) -> None:
    async with db.async_session() as s:
        row = await s.get(KnowledgeChunk, chunk_id)
        if row is not None:
            row.vector_id = vector_id
            row.vectorize_status = "done"
            await s.commit()


async def set_chunk_neighbors(chunk_id: int, prev_id: int | None, next_id: int | None) -> None:
    async with db.async_session() as s:
        row = await s.get(KnowledgeChunk, chunk_id)
        if row is not None:
            row.prev_chunk_id = prev_id
            row.next_chunk_id = next_id
            await s.commit()


async def count_chunks_by_status(status: str) -> int:
    async with db.async_session() as s:
        result = await s.execute(
            select(func.count()).select_from(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == status)
        )
        return int(result.scalar_one())


async def list_all_questions() -> list[str]:
    async with db.async_session() as s:
        result = await s.execute(select(KnowledgeChunk.questions))
        return list(result.scalars())


async def insert_staging(batch_no: str, source_ref: str | None, question: str, answer: str) -> int:
    async with db.async_session() as s:
        row = QaExtractionStaging(
            batch_no=batch_no, source_ref=source_ref, question=question, answer=answer
        )
        s.add(row)
        await s.commit()
        return row.id


async def list_staging_by_status(status: str) -> list[QaExtractionStaging]:
    async with db.async_session() as s:
        result = await s.execute(
            select(QaExtractionStaging)
            .where(QaExtractionStaging.status == status)
            .order_by(QaExtractionStaging.id)
        )
        return list(result.scalars())


async def set_staging_status(ids: list[int], status: str) -> None:
    if not ids:
        return
    async with db.async_session() as s:
        for i in ids:
            row = await s.get(QaExtractionStaging, i)
            if row is not None:
                row.status = status
        await s.commit()
```

注:`func` 已在 `app.db.repository` 的既有 import(`from sqlalchemy import select`;需补 `func` → 改为 `from sqlalchemy import func, select`)。

- [ ] **Step 4: 运行看通过**

Run: `uv run pytest tests/test_kb_repository.py -v`
Expected: PASS(5 passed)。

- [ ] **Step 5: 提交**

```bash
git add app/db/repository.py tests/test_kb_repository.py
git commit -m "feat(ch03): 知识 chunk + 暂存表 repository CRUD"
```

---

## Task 6: 切分原语 — 标题切 + 递归切

**TDD**(纯逻辑)。

**Files:**
- Create: `app/kb/chunking.py`
- Test: `tests/test_chunking_split.py`

**Interfaces:**
- Produces:
  - `HEADERS`(headers_to_split_on 配置)、`CJK_SEPARATORS`
  - `split_sections(md: str) -> list[Document]`(每个 `Document.metadata` 含命中的 `h1..h4`,`page_content` 为正文)
  - `recursive_split(text: str, chunk_size: int, chunk_overlap: int = 0) -> list[str]`

- [ ] **Step 1: 写失败测试**

Create `tests/test_chunking_split.py`:

```python
from app.kb import chunking


def test_split_sections_keeps_header_path():
    md = "# 售后手册\n\n## 退货政策\n\n支持7天无理由。\n\n## 运费说明\n\n满99包邮。"
    secs = chunking.split_sections(md)
    운 = [d for d in secs if d.metadata.get("h2") == "运费说明"]
    assert 운, "应切出「运费说明」小节"
    assert 운[0].metadata.get("h1") == "售后手册"
    assert "满99包邮" in 운[0].page_content


def test_recursive_split_breaks_oversized():
    text = "句子内容。" * 60  # 360 字
    parts = chunking.recursive_split(text, chunk_size=50, chunk_overlap=0)
    assert len(parts) > 1
    assert max(len(p) for p in parts) <= 50
```

(测试变量名用中文亦可;若嫌怪,改成 ascii 名。)

- [ ] **Step 2: 运行看失败**

Run: `uv run pytest tests/test_chunking_split.py -v`
Expected: FAIL(`ModuleNotFoundError: app.kb.chunking`)。

- [ ] **Step 3: 实现**

Create `app/kb/chunking.py`:

```python
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
# 中文无词边界,分隔符优先段落/换行,再句末标点,最后逐字
CJK_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "!", "?", ";", "，", " ", ""]


def split_sections(md: str) -> list[Document]:
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=True)
    return splitter.split_text(md)


def recursive_split(text: str, chunk_size: int, chunk_overlap: int = 0) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=CJK_SEPARATORS, is_separator_regex=False, length_function=len,
    )
    return splitter.split_text(text)
```

- [ ] **Step 4: 运行看通过**

Run: `uv run pytest tests/test_chunking_split.py -v`
Expected: PASS(2 passed)。

- [ ] **Step 5: 提交**

```bash
git add app/kb/chunking.py tests/test_chunking_split.py
git commit -m "feat(ch03): 切分原语 标题切 split_sections + 递归切 recursive_split"
```

---

## Task 7: 切分原语 — 句末重叠裁剪

**TDD**(纯逻辑)。重叠取上一块结尾的**完整句子**(不超过 overlap 字数),句中不截断。

**Files:**
- Modify: `app/kb/chunking.py`
- Test: `tests/test_chunking_overlap.py`

**Interfaces:**
- Produces: `apply_sentence_overlap(chunks: list[str], overlap: int) -> list[str]`(首块原样;后续块前置上一块的尾部完整句)。

- [ ] **Step 1: 写失败测试**

Create `tests/test_chunking_overlap.py`:

```python
from app.kb import chunking


def test_overlap_is_whole_trailing_sentence():
    chunks = ["前面很多内容。中间一句话。最后收尾句。", "下一块正文。"]
    out = chunking.apply_sentence_overlap(chunks, overlap=6)
    assert out[0] == chunks[0]
    # 末6字符恰为「最后收尾句。」,是完整句 → 整句作为重叠前缀
    assert out[1] == "最后收尾句。下一块正文。"


def test_overlap_never_starts_mid_sentence():
    chunks = ["这是一个非常非常非常长的句子没有中间标点结尾才有。", "新块。"]
    out = chunking.apply_sentence_overlap(chunks, overlap=5)
    # overlap=5 放不下整句,但不能留半句 → 宁可整句(优先不留半截话)
    assert out[1].startswith("这是") or out[1] == "新块。"
    assert "非常非常" not in out[1] or out[1].startswith("这是")  # 不出现句中片段起头
```

- [ ] **Step 2: 运行看失败**

Run: `uv run pytest tests/test_chunking_overlap.py -v`
Expected: FAIL(`AttributeError: ... 'apply_sentence_overlap'`)。

- [ ] **Step 3: 实现**

Edit `app/kb/chunking.py`,加 `import re` 与:

```python
import re

_SENT_RE = re.compile(r"[^。！？!?…\n]*[。！？!?…\n]|[^。！？!?…\n]+$")


def _split_sentences(text: str) -> list[str]:
    return [m for m in _SENT_RE.findall(text) if m]


def _trailing_sentences(text: str, max_chars: int) -> str:
    """取 text 结尾若干**完整句**作为重叠,总长尽量不超过 max_chars;
    单句超长则整句保留(优先「不留半截话」)。"""
    out: list[str] = []
    total = 0
    for s in reversed(_split_sentences(text)):
        if out and total + len(s) > max_chars:
            break
        out.insert(0, s)
        total += len(s)
    return "".join(out)


def apply_sentence_overlap(chunks: list[str], overlap: int) -> list[str]:
    if not chunks:
        return []
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        ov = _trailing_sentences(chunks[i - 1], overlap)
        out.append(ov + chunks[i] if ov else chunks[i])
    return out
```

- [ ] **Step 4: 运行看通过**

Run: `uv run pytest tests/test_chunking_overlap.py -v`
Expected: PASS(2 passed)。

- [ ] **Step 5: 提交**

```bash
git add app/kb/chunking.py tests/test_chunking_overlap.py
git commit -m "feat(ch03): 句末重叠裁剪 apply_sentence_overlap(整句重叠,不留半截话)"
```

---

## Task 8: 切分原语 — 大表格按行切 + 复制表头

**TDD**(纯逻辑)。

**Files:**
- Modify: `app/kb/chunking.py`
- Test: `tests/test_chunking_table.py`

**Interfaces:**
- Produces:
  - `is_table_block(text: str) -> bool`
  - `split_table_rows(table_md: str, max_rows: int) -> list[str]`(超 max_rows 数据行则按组切,每组重贴表头行 + 分隔行)

- [ ] **Step 1: 写失败测试**

Create `tests/test_chunking_table.py`:

```python
from app.kb import chunking


TABLE = "\n".join([
    "| 商品 | 价格 |",
    "| --- | --- |",
    "| A | 1 |",
    "| B | 2 |",
    "| C | 3 |",
])


def test_is_table_block():
    assert chunking.is_table_block(TABLE)
    assert not chunking.is_table_block("普通段落文字。")


def test_split_table_replicates_header():
    out = chunking.split_table_rows(TABLE, max_rows=2)
    assert len(out) == 2
    assert out[0] == "| 商品 | 价格 |\n| --- | --- |\n| A | 1 |\n| B | 2 |"
    assert out[1] == "| 商品 | 价格 |\n| --- | --- |\n| C | 3 |"
    for block in out:
        assert block.startswith("| 商品 | 价格 |\n| --- | --- |")  # 每块都带表头


def test_split_table_small_stays_whole():
    assert chunking.split_table_rows(TABLE, max_rows=10) == [TABLE]
```

- [ ] **Step 2: 运行看失败**

Run: `uv run pytest tests/test_chunking_table.py -v`
Expected: FAIL(`AttributeError: ... 'is_table_block'`)。

- [ ] **Step 3: 实现**

Edit `app/kb/chunking.py`,追加:

```python
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def is_table_block(text: str) -> bool:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return (
        len(lines) >= 2
        and lines[0].lstrip().startswith("|")
        and bool(_TABLE_SEP_RE.match(lines[1])) and "-" in lines[1]
    )


def split_table_rows(table_md: str, max_rows: int) -> list[str]:
    lines = [ln for ln in table_md.strip().splitlines() if ln.strip()]
    header, sep, rows = lines[0], lines[1], lines[2:]
    if len(rows) <= max_rows:
        return [table_md.strip()]
    out: list[str] = []
    for i in range(0, len(rows), max_rows):
        out.append("\n".join([header, sep, *rows[i:i + max_rows]]))
    return out
```

- [ ] **Step 4: 运行看通过**

Run: `uv run pytest tests/test_chunking_table.py -v`
Expected: PASS(3 passed)。

- [ ] **Step 5: 提交**

```bash
git add app/kb/chunking.py tests/test_chunking_table.py
git commit -m "feat(ch03): 大表格按行切 split_table_rows(每块复制表头)"
```

---

## Task 9: 源文档 → Chunk 记录 `app/kb/documents.py`

**TDD**(纯逻辑)。组合 Task 6/7/8,做字段映射。

**Files:**
- Create: `app/kb/documents.py`
- Test: `tests/test_documents.py`

**Interfaces:**
- Consumes: `chunking.split_sections / recursive_split / apply_sentence_overlap / is_table_block / split_table_rows`。
- Produces:
  - `@dataclass Chunk(category, questions, answer, section_path, content_type, is_key_clause=0)`
  - `build_chunks(md: str, content_type: str, chunk_size=400, overlap=60, table_max_rows=10) -> list[Chunk]`

- [ ] **Step 1: 写失败测试**

Create `tests/test_documents.py`:

```python
from app.kb.documents import Chunk, build_chunks


def test_policy_maps_heading_and_parent_path():
    md = "# 售后政策\n\n## 运费说明\n\n单笔订单满99元包邮,未满收取10元运费。偏远地区另计。"
    chunks = build_chunks(md, content_type="policy")
    c = [c for c in chunks if c.questions == "运费说明"][0]
    assert c.category == "售后政策"
    assert c.section_path == "售后政策 / 运费说明"
    assert "满99" in c.answer
    assert c.content_type == "policy"
    assert c.is_key_clause == 1  # 含「运费」关键条款


def test_table_section_splits_by_rows_with_header():
    rows = "\n".join(f"| 商品{i} | {i} |" for i in range(1, 15))
    md = f"# 商品价格表\n\n## 价目\n\n| 商品 | 价格 |\n| --- | --- |\n{rows}"
    chunks = build_chunks(md, content_type="manual", table_max_rows=5)
    price = [c for c in chunks if c.questions == "价目"]
    assert len(price) >= 2  # 14 行按 5 切成 >=3 块
    for c in price:
        assert c.answer.startswith("| 商品 | 价格 |")  # 每块复制表头


def test_returns_chunk_dataclass():
    chunks = build_chunks("# A\n\n## B\n\n正文。", content_type="faq")
    assert isinstance(chunks[0], Chunk)
```

- [ ] **Step 2: 运行看失败**

Run: `uv run pytest tests/test_documents.py -v`
Expected: FAIL(`ModuleNotFoundError: app.kb.documents`)。

- [ ] **Step 3: 实现**

Create `app/kb/documents.py`:

```python
from dataclasses import dataclass

from app.kb import chunking

_KEY_TERMS = ("退款", "退货", "时效", "运费", "邮费", "费用", "保修", "赔偿", "期限", "包邮")


@dataclass
class Chunk:
    category: str
    questions: str
    answer: str
    section_path: str
    content_type: str
    is_key_clause: int = 0


def _is_key(title: str, body: str) -> int:
    head = title + body[:40]
    return int(any(t in head for t in _KEY_TERMS))


def build_chunks(
    md: str, content_type: str,
    chunk_size: int = 400, overlap: int = 60, table_max_rows: int = 10,
) -> list[Chunk]:
    out: list[Chunk] = []
    for sec in chunking.split_sections(md):
        path = [sec.metadata[k] for k in ("h1", "h2", "h3", "h4") if sec.metadata.get(k)]
        section_path = " / ".join(path)
        title = path[-1] if path else content_type
        # category:政策/手册用上级标题路径;顶层无上级则用 content_type
        category = " / ".join(path[:-1]) if len(path) > 1 else (path[0] if path else content_type)
        body = sec.page_content.strip()
        if not body:
            continue
        if chunking.is_table_block(body):
            pieces = chunking.split_table_rows(body, table_max_rows)
        else:
            base = chunking.recursive_split(body, chunk_size, chunk_overlap=0)
            pieces = chunking.apply_sentence_overlap(base, overlap)
        for piece in pieces:
            out.append(Chunk(
                category=category, questions=title, answer=piece,
                section_path=section_path, content_type=content_type,
                is_key_clause=_is_key(title, piece),
            ))
    return out
```

- [ ] **Step 4: 运行看通过**

Run: `uv run pytest tests/test_documents.py -v`
Expected: PASS(3 passed)。

- [ ] **Step 5: 提交**

```bash
git add app/kb/documents.py tests/test_documents.py
git commit -m "feat(ch03): 源文档→Chunk 记录 build_chunks(标题/表格分派 + 字段映射)"
```

---

## Task 10: 双写 `app/kb/dualwrite.py` + vectorize CLI(幂等恢复=验收2)

**TDD**(真实 test MySQL + 临时 Milvus Lite;嵌入 mock)。**验收2 的核心任务**。

**Files:**
- Create: `app/kb/dualwrite.py`, `scripts/vectorize_kb.py`
- Test: `tests/test_dualwrite.py`

**Interfaces:**
- Consumes: `repository.*`(Task 5)、`milvus_client.*`(Task 3)、`embeddings.embed_texts`(Task 2)、`documents.Chunk`(Task 9)。
- Produces:
  - `async write_pending(chunks: list[Chunk]) -> list[int]`(顺序插入 + 同文档内连 prev/next)
  - `async vectorize_pending(client, batch_size=64) -> int`(取 pending → embed → Milvus upsert → 标 done;幂等可重跑)

- [ ] **Step 1: 写失败测试**

Create `tests/test_dualwrite.py`:

```python
import pytest

from app.kb import dualwrite, milvus_client
from app.kb.documents import Chunk
from app.db import repository


def _chunk(q, a):
    return Chunk(category="c", questions=q, answer=a, section_path="c / " + q, content_type="policy")


@pytest.fixture()
def milvus(tmp_path):
    c = milvus_client.get_client(uri=str(tmp_path / "t.db"))
    milvus_client.ensure_collection(c)
    return c


async def test_write_pending_links_neighbors(db_session_factory):
    ids = await dualwrite.write_pending([_chunk("q1", "a1"), _chunk("q2", "a2"), _chunk("q3", "a3")])
    assert len(ids) == 3
    by_id = {c.id: c for c in await repository.list_pending_chunks()}
    assert by_id[ids[1]].prev_chunk_id == ids[0]
    assert by_id[ids[1]].next_chunk_id == ids[2]
    assert by_id[ids[0]].prev_chunk_id is None


async def test_vectorize_resumes_after_crash(db_session_factory, milvus, monkeypatch):
    await dualwrite.write_pending([_chunk(f"q{i}", f"a{i}") for i in range(4)])

    # 第一趟:embed 在第 2 批抛错(batch_size=2 → 前 2 条 done,后 2 条仍 pending)
    real_calls = {"n": 0}

    async def flaky_embed(texts):
        real_calls["n"] += 1
        if real_calls["n"] == 2:
            raise RuntimeError("嵌入服务中断")
        return [[float(i), 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3) for i, _ in enumerate(texts)]

    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", flaky_embed)
    with pytest.raises(RuntimeError):
        await dualwrite.vectorize_pending(milvus, batch_size=2)
    assert await repository.count_chunks_by_status("done") == 2
    assert await repository.count_chunks_by_status("pending") == 2

    # 第二趟:恢复正常 → 只捡剩下 2 条 pending,补齐;Milvus 无重(按 id upsert)
    async def ok_embed(texts):
        return [[9.0, 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3) for _ in texts]

    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", ok_embed)
    done_now = await dualwrite.vectorize_pending(milvus, batch_size=2)
    assert done_now == 2
    assert await repository.count_chunks_by_status("pending") == 0
    assert await repository.count_chunks_by_status("done") == 4
    assert milvus_client.count(milvus) == 4  # 无重无漏
```

- [ ] **Step 2: 运行看失败**

Run: `uv run pytest tests/test_dualwrite.py -v`
Expected: FAIL(`ModuleNotFoundError: app.kb.dualwrite`)。

- [ ] **Step 3: 实现**

Create `app/kb/dualwrite.py`:

```python
from app.core import embeddings
from app.db import repository
from app.kb import milvus_client
from app.kb.documents import Chunk


async def write_pending(chunks: list[Chunk]) -> list[int]:
    """按顺序把一份文档的 chunks 写入 MySQL(status=pending),并在同文档内连 prev/next 指针。"""
    ids: list[int] = []
    for c in chunks:
        cid = await repository.insert_knowledge_chunk(
            c.category, c.questions, c.answer,
            section_path=c.section_path, content_type=c.content_type,
            is_key_clause=c.is_key_clause,
        )
        ids.append(cid)
    for i, cid in enumerate(ids):
        prev_id = ids[i - 1] if i > 0 else None
        next_id = ids[i + 1] if i < len(ids) - 1 else None
        await repository.set_chunk_neighbors(cid, prev_id, next_id)
    return ids


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


async def vectorize_pending(client, batch_size: int = 64) -> int:
    """幂等可重跑:取 pending → 拼 category+questions+answer 向量化 → Milvus upsert(PK=id)
    → 回填 vector_id、status=done。崩在任意批,重跑只捡剩余 pending(Milvus 按 id upsert 无重)。"""
    pending = await repository.list_pending_chunks()
    done = 0
    for batch in _batches(pending, batch_size):
        texts = [f"{r.category}\n{r.questions}\n{r.answer}" for r in batch]
        vectors = await embeddings.embed_texts(texts)
        rows = [
            {"id": r.id, "vector": v, "question": r.questions, "answer": r.answer}
            for r, v in zip(batch, vectors)
        ]
        milvus_client.upsert_vectors(client, rows)
        for r in batch:
            await repository.mark_chunk_vectorized(r.id, str(r.id))
        done += len(batch)
    return done
```

Create `scripts/vectorize_kb.py`:

```python
"""CLI:把 knowledge_chunks 里 pending 的块向量化写入 Milvus(幂等可重跑)。
运行:PYTHONPATH=. uv run python scripts/vectorize_kb.py"""
import asyncio

from app.kb import dualwrite, milvus_client


async def main() -> None:
    client = milvus_client.get_client()
    milvus_client.ensure_collection(client)
    n = await dualwrite.vectorize_pending(client)
    print(f"✅ 本次向量化 {n} 块;Milvus 现有 {milvus_client.count(client)} 条")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: 运行看通过**

Run: `uv run pytest tests/test_dualwrite.py -v`
Expected: PASS(2 passed)。恢复测试证明验收2:中断后重跑,漏向量化的块被捡起补齐、无重无漏。

- [ ] **Step 5: 提交**

```bash
git add app/kb/dualwrite.py scripts/vectorize_kb.py tests/test_dualwrite.py
git commit -m "feat(ch03): 双写 write_pending + vectorize_pending(幂等恢复)+ vectorize CLI"
```

---

## Task 11: 去重 `app/kb/dedup.py`

**TDD**(纯逻辑)。文本归一化:去空白/标点,保留中文。

**Files:**
- Create: `app/kb/dedup.py`
- Test: `tests/test_dedup.py`

**Interfaces:**
- Produces:
  - `normalize_question(q: str) -> str`
  - `dedupe(items: list, existing_questions: list[str]) -> tuple[list, list]`(返回 `(kept, discarded)`;`items` 是有 `.question` 属性的对象)

- [ ] **Step 1: 写失败测试**

Create `tests/test_dedup.py`:

```python
from dataclasses import dataclass

from app.kb import dedup


@dataclass
class Item:
    question: str


def test_normalize_strips_punct_keeps_chinese():
    assert dedup.normalize_question(" 邮费,是多少? ") == "邮费是多少"


def test_dedupe_within_batch_and_against_existing():
    items = [Item("邮费是多少"), Item("邮费是多少?"), Item("怎么退货"), Item("运费怎么算")]
    kept, discarded = dedup.dedupe(items, existing_questions=["运费怎么算"])
    kept_q = [i.question for i in kept]
    assert "邮费是多少" in kept_q          # 首次出现保留
    assert "怎么退货" in kept_q
    assert "运费怎么算" not in kept_q       # 与库内已有重复 → 丢弃
    assert len(kept) == 2 and len(discarded) == 2
```

- [ ] **Step 2: 运行看失败**

Run: `uv run pytest tests/test_dedup.py -v`
Expected: FAIL(`ModuleNotFoundError: app.kb.dedup`)。

- [ ] **Step 3: 实现**

Create `app/kb/dedup.py`:

```python
import re

_STRIP_RE = re.compile(r"[\s\W_]+", re.UNICODE)  # \W 不含中文(中文属 \w),故只去空白/标点


def normalize_question(q: str) -> str:
    return _STRIP_RE.sub("", q.strip().lower())


def dedupe(items: list, existing_questions: list[str]) -> tuple[list, list]:
    seen = {normalize_question(q) for q in existing_questions}
    kept, discarded = [], []
    for item in items:
        key = normalize_question(item.question)
        if not key or key in seen:
            discarded.append(item)
        else:
            seen.add(key)
            kept.append(item)
    return kept, discarded
```

- [ ] **Step 4: 运行看通过**

Run: `uv run pytest tests/test_dedup.py -v`
Expected: PASS(2 passed)。

- [ ] **Step 5: 提交**

```bash
git add app/kb/dedup.py tests/test_dedup.py
git commit -m "feat(ch03): 暂存表文本归一化去重 dedupe"
```

---

## Task 12: 对话挖知识 `app/kb/mining.py` + CLI(Prompt → eval)

**纯 Prompt 任务:不写单测,用标注样例 eval 验证**(work-req 1)。抽取用 `PROMPT | model.with_structured_output(...)`(ch01 extract 已验证 glm-5.2 支持)。

**Files:**
- Create: `app/kb/mining.py`, `scripts/mine_knowledge.py`
- Modify: `app/core/prompts.py`(加 `MINING_SYSTEM` / `MINING_PROMPT`)

**Interfaces:**
- Consumes: `get_chat_model`、`repository`(staging/knowledge)、`dedup.dedupe`、`dualwrite.write_pending`、`documents.Chunk`。
- Produces:
  - Pydantic `QaPair(question, answer)` / `QaExtraction(pairs: list[QaPair])`
  - `async extract_qa(conversation_texts: list[str], model=None) -> list[QaPair]`
  - `async mine(batch_size=20, model=None) -> dict`(读对话→分批抽取写 staging→整体去重→kept 写 knowledge pending;返回统计)

- [ ] **Step 1: 加抽取 prompt**

Edit `app/core/prompts.py`,追加:

```python
MINING_SYSTEM = """你是客服知识库构建助手。下面是若干条历史客服对话(用户问 + 客服答)。
请从中抽取「可复用的问答对」,用于沉淀到 FAQ 知识库。要求:
- 只抽有普适价值的问答(政策、流程、时效、费用等),忽略闲聊、纯个案(如某具体订单号的状态)。
- question 用简洁的通用问法(去掉具体订单号/人名),answer 忠于客服原答、不编造承诺。
- 一条对话可能不含任何可复用问答,此时不要硬抽。
- 退款/售后时效统一表述为「以平台售后规则为准」。"""

MINING_PROMPT = ChatPromptTemplate.from_messages(
    [("system", MINING_SYSTEM), ("human", "历史对话:\n{conversations}")]
)
```

- [ ] **Step 2: 实现 mining**

Create `app/kb/mining.py`:

```python
from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import MINING_PROMPT
from app.db import repository
from app.kb import dedup, dualwrite
from app.kb.documents import Chunk


class QaPair(BaseModel):
    question: str = Field(description="通用问法,去掉具体订单号/人名")
    answer: str = Field(description="客服原答,不编造")


class QaExtraction(BaseModel):
    pairs: list[QaPair] = Field(default_factory=list, description="抽出的问答对,可为空")


async def extract_qa(conversation_texts: list[str], model=None) -> list[QaPair]:
    model = model or get_chat_model()
    chain = MINING_PROMPT | model.with_structured_output(QaExtraction)
    result: QaExtraction = await chain.ainvoke({"conversations": "\n---\n".join(conversation_texts)})
    return result.pairs


async def _load_conversation_texts() -> list[tuple[str, str]]:
    """返回 [(source_ref, 对话文本)];对话文本由该会话的 user/assistant 消息拼成。"""
    convs = await repository.list_conversations_with_messages()
    out = []
    for conv_id, msgs in convs:
        lines = [f"{m.role}: {m.content}" for m in msgs if m.content]
        if lines:
            out.append((f"conv:{conv_id}", "\n".join(lines)))
    return out


async def mine(batch_size: int = 20, model=None) -> dict:
    sources = await _load_conversation_texts()
    batch_no = f"mine-{len(sources)}"  # 无 Date.now 可用;以来源数标批,重跑可覆盖语义
    # 1) 分批抽取 → 写 staging(extracted)
    for start in range(0, len(sources), batch_size):
        batch = sources[start:start + batch_size]
        pairs = await extract_qa([t for _, t in batch], model=model)
        for p in pairs:
            await repository.insert_staging(batch_no, batch[0][0], p.question, p.answer)
    # 2) 整体去重(staging extracted 内 + 对已有 knowledge)
    staged = await repository.list_staging_by_status("extracted")
    existing = await repository.list_all_questions()
    kept, discarded = dedup.dedupe(staged, existing)
    await repository.set_staging_status([s.id for s in kept], "kept")
    await repository.set_staging_status([s.id for s in discarded], "discarded")
    # 3) kept 写 knowledge_chunks(pending, content_type=mined, questions=真实问法)
    chunks = [Chunk(category="历史对话", questions=s.question, answer=s.answer,
                    section_path="mined", content_type="mined") for s in kept]
    if chunks:
        await dualwrite.write_pending(chunks)
    return {"sources": len(sources), "extracted": len(staged),
            "kept": len(kept), "discarded": len(discarded)}
```

- [ ] **Step 3: 加 repository 依赖方法**

Edit `app/db/repository.py`,追加(mining 需要按会话取消息):

```python
async def list_conversations_with_messages() -> list[tuple[int, list[Message]]]:
    async with db.async_session() as s:
        conv_ids = list((await s.execute(select(Conversation.id).order_by(Conversation.id))).scalars())
        out = []
        for cid in conv_ids:
            msgs = list((await s.execute(
                select(Message).where(Message.conversation_id == cid).order_by(Message.id)
            )).scalars())
            out.append((cid, msgs))
        return out
```

- [ ] **Step 4: 写 CLI**

Create `scripts/mine_knowledge.py`:

```python
"""CLI(可重跑 job):从历史对话挖 QA → 暂存表 → 去重 → 写 knowledge_chunks(pending)。
之后跑 scripts/vectorize_kb.py 向量化。运行:PYTHONPATH=. uv run python scripts/mine_knowledge.py"""
import asyncio

from app.kb import mining


async def main() -> None:
    stats = await mining.mine()
    print(f"✅ 挖知识:{stats}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 5: 冒烟(结构验证,不算 eval)**

先确保有历史对话(Task 16 会 seed;若还没有,本步可跑通 0 条)。
Run: `PYTHONPATH=. uv run python scripts/mine_knowledge.py`
Expected: 打印统计 dict,不报错。真正的抽取质量在 Task 16 的 `eval_mining.py` 里用标注样例验。

- [ ] **Step 6: 提交**

```bash
git add app/kb/mining.py scripts/mine_knowledge.py app/core/prompts.py app/db/repository.py
git commit -m "feat(ch03): 对话挖知识 mining(结构化抽取→staging→去重→pending)+ CLI"
```

---

## Task 13: 在线检索 `app/core/retrieval.py`

**TDD**(mock embed + 临时 Milvus Lite)。

**Files:**
- Create: `app/core/retrieval.py`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `embeddings.embed_query`、`milvus_client.*`、`settings.retrieval_top_k / retrieval_min_score`。
- Produces: `async search_knowledge(query: str, top_k: int | None = None, min_score: float | None = None, client=None) -> list[dict]`(返回 `{"id","score","question","answer"}`,按 min_score 过滤)。

- [ ] **Step 1: 写失败测试**

Create `tests/test_retrieval.py`:

```python
import pytest

from app.core import retrieval
from app.kb import milvus_client


@pytest.fixture()
def milvus(tmp_path):
    c = milvus_client.get_client(uri=str(tmp_path / "t.db"))
    milvus_client.ensure_collection(c)
    v_hit = [1.0, 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3)
    v_other = [0.0, 1.0, 0.0] + [0.0] * (milvus_client.DIM - 3)
    milvus_client.upsert_vectors(c, [
        {"id": 1, "vector": v_hit, "question": "运费怎么算", "answer": "满99包邮"},
        {"id": 2, "vector": v_other, "question": "发货时效", "answer": "48小时"},
    ])
    return c


async def test_search_returns_top_hits_above_threshold(monkeypatch, milvus):
    query_vec = [1.0, 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3)
    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query",
                        lambda q: _coro(query_vec))
    hits = await retrieval.search_knowledge("邮费是多少", top_k=1, min_score=0.5, client=milvus)
    assert hits and hits[0]["question"] == "运费怎么算"


async def test_below_threshold_filtered(monkeypatch, milvus):
    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query",
                        lambda q: _coro([1.0, 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3)))
    hits = await retrieval.search_knowledge("邮费", top_k=2, min_score=0.999999, client=milvus)
    assert hits == [] or all(h["score"] >= 0.999999 for h in hits)


async def _coro(v):
    return v
```

(注:`monkeypatch` 用同步 lambda 返回协程,替 async `embed_query`。)

- [ ] **Step 2: 运行看失败**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: FAIL(`ModuleNotFoundError: app.core.retrieval`)。

- [ ] **Step 3: 实现**

Create `app/core/retrieval.py`:

```python
from app.config import settings
from app.core import embeddings
from app.kb import milvus_client


async def search_knowledge(
    query: str, top_k: int | None = None,
    min_score: float | None = None, client=None,
) -> list[dict]:
    top_k = top_k or settings.retrieval_top_k
    min_score = settings.retrieval_min_score if min_score is None else min_score
    vector = await embeddings.embed_query(query)
    client = client or milvus_client.get_client()
    milvus_client.ensure_collection(client)
    hits = milvus_client.search(client, vector, top_k)
    return [h for h in hits if h["score"] >= min_score]
```

- [ ] **Step 4: 运行看通过**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: PASS(2 passed)。

- [ ] **Step 5: 提交**

```bash
git add app/core/retrieval.py tests/test_retrieval.py
git commit -m "feat(ch03): 在线检索 search_knowledge(embed→Milvus Top-K→阈值过滤)"
```

---

## Task 14: `query_faq` 改写(契约不变)

**TDD**(mock retrieval)。只换 `query_faq` 内部,签名/出参/docstring 不动;registry 不动。

**Files:**
- Modify: `app/tools/business.py`
- Test: `tests/test_tool_faq_vector.py`

**Interfaces:**
- Consumes: `retrieval.search_knowledge`。
- Produces: `query_faq(keyword)` 出参不变:命中 `{"hits": [{"question","answer"}]}`,未命中 `{"hits": [], "message": ...}`。

- [ ] **Step 1: 写失败测试**

Create `tests/test_tool_faq_vector.py`:

```python
from app.tools.business import query_faq


async def test_query_faq_maps_hits_to_contract(monkeypatch):
    async def fake_search(keyword, **kw):
        return [{"id": 1, "score": 0.8, "question": "运费怎么算", "answer": "满99包邮"}]
    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    out = await query_faq.ainvoke({"keyword": "邮费是多少"})
    assert out == {"hits": [{"question": "运费怎么算", "answer": "满99包邮"}]}


async def test_query_faq_empty_returns_message(monkeypatch):
    async def fake_search(keyword, **kw):
        return []
    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    out = await query_faq.ainvoke({"keyword": "无关问题"})
    assert out["hits"] == []
    assert "未找到" in out["message"]
```

- [ ] **Step 2: 运行看失败**

Run: `uv run pytest tests/test_tool_faq_vector.py -v`
Expected: FAIL(`AttributeError: module 'app.tools.business' has no attribute 'retrieval'`)。

- [ ] **Step 3: 实现**

Edit `app/tools/business.py`:顶部把 `from app.db import repository` 旁加 `from app.core import retrieval`;改写 `query_faq` 函数体(**docstring 与 `FaqInput` / 签名保持不变**),把调 `repository.search_faq` 换成 `retrieval.search_knowledge`:

```python
@tool(args_schema=FaqInput)
async def query_faq(keyword: str) -> dict:
    """根据关键词查询常见问题解答(FAQ)。用于用户咨询政策、规则、操作流程等通用问题时。"""
    hits = await retrieval.search_knowledge(keyword)
    if not hits:
        return {"hits": [], "message": f"未找到与「{keyword}」相关的常见问题"}
    return {"hits": [{"question": h["question"], "answer": h["answer"]} for h in hits]}
```

(`repository` 若在 business.py 已无其他用处可留着不删,避免动别的工具。)

- [ ] **Step 4: 运行看通过 + 全量回归**

Run: `uv run pytest tests/test_tool_faq_vector.py -v && uv run pytest -q`
Expected: 新测 PASS;全量绿(ch01/ch02 不回归。旧 `tests/test_tool_faq.py` 若断言查表实现,已被本任务语义取代——如与新实现冲突,改造为对 `retrieval.search_knowledge` 的 mock 断言,不保留查表断言)。

- [ ] **Step 5: 提交**

```bash
git add app/tools/business.py tests/test_tool_faq_vector.py
git commit -m "feat(ch03): query_faq 内部改向量检索(入参出参契约不变)"
```

---

## Task 15: 源文档夹具 + build_kb CLI + Makefile

**集成/夹具**(主会话)。造 `data/kb/*.md`,写 build CLI,加 Makefile 目标。

**Files:**
- Create: `data/kb/returns-policy.md`, `data/kb/after-sales-manual.md`, `data/kb/product-faq.md`, `scripts/build_kb.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `documents.build_chunks`、`dualwrite.write_pending`。

- [ ] **Step 1: 造源文档**

Create `data/kb/product-faq.md`(FAQ:标题=问法;**必须含运费/邮费块**,验收1 的召回目标;把 ch02 六条 faq 内容并入):

```markdown
# 商品与购物 FAQ

## 退货政策是什么

支持 7 天无理由退货,商品需保持完好、不影响二次销售,具体以平台售后规则为准。

## 如何申请退款

在「我的订单」找到对应订单点击「申请退款」,按提示提交,审核通过后原路退回。

## 换货流程怎么走

收到商品 7 天内可申请换货,联系客服登记后将商品寄回,平台核验无误后补发新品。

## 发货时效多久

现货商品付款后 48 小时内发货,预售商品以商品详情页标注的发货时间为准。

## 运费怎么算

单笔订单满 99 元包邮,未满收取 10 元运费,偏远地区运费另计。邮费即运费,按此规则收取。

## 发票如何开具

在「我的订单」-「申请开票」提交抬头与税号,电子发票将于 3 个工作日内发送至邮箱。
```

Create `data/kb/returns-policy.md`(政策:标题=本节标题,category=上级路径;含关键条款):

```markdown
# 退货退款政策

## 无理由退货

### 适用范围

自签收之日起 7 天内,商品完好、不影响二次销售的,支持无理由退货。定制类、生鲜类商品除外。

### 退款时效

退货商品经平台核验通过后,退款原路退回,到账时效以平台售后规则为准。
```

Create `data/kb/after-sales-manual.md`(手册:含一张**大表格**练按行切):

```markdown
# 售后手册

## 常见问题处理时限

| 问题类型 | 首次响应 | 处理时限 |
| --- | --- | --- |
| 退款 | 2 小时 | 3 个工作日 |
| 换货 | 2 小时 | 5 个工作日 |
| 维修 | 4 小时 | 7 个工作日 |
| 投诉 | 1 小时 | 3 个工作日 |
| 发票 | 4 小时 | 3 个工作日 |
| 物流异常 | 2 小时 | 5 个工作日 |
| 商品咨询 | 1 小时 | 当日 |
| 价格保护 | 2 小时 | 3 个工作日 |
| 账户问题 | 2 小时 | 3 个工作日 |
| 其他 | 4 小时 | 7 个工作日 |

## 保修说明

电子类商品保修 12 个月,自签收日起算;人为损坏不在保修范围。
```

- [ ] **Step 2: 写 build_kb CLI**

Create `scripts/build_kb.py`:

```python
"""CLI:把 data/kb/*.md 切块写入 knowledge_chunks(pending)。之后跑 vectorize_kb.py。
运行:PYTHONPATH=. uv run python scripts/build_kb.py"""
import asyncio
import pathlib

from app.kb import documents, dualwrite

KB_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "kb"
# 文件 → content_type
DOCS = {
    "product-faq.md": "faq",
    "returns-policy.md": "policy",
    "after-sales-manual.md": "manual",
}


async def main() -> None:
    total = 0
    for fname, ctype in DOCS.items():
        md = (KB_DIR / fname).read_text(encoding="utf-8")
        chunks = documents.build_chunks(md, content_type=ctype)
        ids = await dualwrite.write_pending(chunks)
        total += len(ids)
        print(f"  {fname}: {len(ids)} 块")
    print(f"✅ 建库(pending):共 {total} 块。下一步:scripts/vectorize_kb.py")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: 加 Makefile 目标**

Edit `Makefile`,`.PHONY` 行补目标,并追加:

```makefile
kb-build:
	PYTHONPATH=. uv run python scripts/build_kb.py

kb-vectorize:
	PYTHONPATH=. uv run python scripts/vectorize_kb.py

kb-mine:
	PYTHONPATH=. uv run python scripts/mine_knowledge.py

eval-retrieval:
	PYTHONPATH=. uv run python scripts/eval_retrieval.py
```

- [ ] **Step 4: 提交**

```bash
git add data/kb/ scripts/build_kb.py Makefile
git commit -m "feat(ch03): 源知识文档夹具 + build_kb CLI + Makefile kb-* 目标"
```

---

## Task 16: 标注样例 + eval 脚本 + 合成对话 seed

**eval**(主会话)。给验收1 建检索 eval;给挖知识建抽取 eval;seed 合成历史对话喂 mining。

**Files:**
- Create: `tests/data/retrieval_samples.json`, `tests/data/mining_samples.json`, `scripts/eval_retrieval.py`, `scripts/eval_mining.py`, `sql/ch03-seed.sql`
- Modify: `Makefile`(加 `seed-conv` / `eval-mining`)

- [ ] **Step 1: 检索标注样例(换说法 → 期望命中)**

Create `tests/data/retrieval_samples.json`:

```json
[
  {"query": "邮费是多少", "expect_answer_contains": "包邮"},
  {"query": "快递费怎么收", "expect_answer_contains": "运费"},
  {"query": "多久能发货", "expect_answer_contains": "48"},
  {"query": "东西不想要了能退吗", "expect_answer_contains": "7 天"},
  {"query": "怎么开发票", "expect_answer_contains": "开票"}
]
```

- [ ] **Step 2: 检索 eval 脚本(直接调 search_knowledge,纯测向量召回)**

Create `scripts/eval_retrieval.py`:

```python
"""验收1 eval:对换说法的问题跑向量检索,核对是否召回期望内容。
需已建库并向量化(make kb-build && make kb-vectorize)+ 嵌入上游可用(真实嵌入)。
运行:PYTHONPATH=. uv run python scripts/eval_retrieval.py"""
import asyncio
import json
import pathlib
import sys

from app.core import retrieval

SAMPLES = pathlib.Path(__file__).resolve().parent.parent / "tests/data/retrieval_samples.json"


async def main() -> int:
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    failures = 0
    for s in samples:
        hits = await retrieval.search_knowledge(s["query"])
        top = hits[0] if hits else None
        ok = bool(top) and s["expect_answer_contains"] in top["answer"]
        failures += not ok
        print(f"{'✅' if ok else '❌'} {s['query']!r} -> "
              f"{(top['question'] + ' | ' + top['answer'][:30]) if top else '(空)'}")
    print(f"\n召回正确 {len(samples) - failures}/{len(samples)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 3: 挖知识标注样例 + eval**

Create `tests/data/mining_samples.json`(样例对话 → 期望能抽到的问法要点):

```json
[
  {
    "conversation": "user: 你们运费怎么算啊\nassistant: 单笔订单满99元包邮,未满收取10元运费,偏远地区另计。",
    "expect_question_contains": "运费"
  },
  {
    "conversation": "user: 我的订单MH123到哪了\nassistant: 您的包裹在广州分拨中心,预计明天送达。",
    "expect_empty": true
  }
]
```

Create `scripts/eval_mining.py`:

```python
"""挖知识 eval:对样例对话跑抽取,核对该抽的抽到、不该抽的(纯个案)不硬抽。
需聊天上游可用。运行:PYTHONPATH=. uv run python scripts/eval_mining.py"""
import asyncio
import json
import pathlib
import sys

from app.kb import mining

SAMPLES = pathlib.Path(__file__).resolve().parent.parent / "tests/data/mining_samples.json"


async def main() -> int:
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    failures = 0
    for s in samples:
        pairs = await mining.extract_qa([s["conversation"]])
        if s.get("expect_empty"):
            ok = len(pairs) == 0
        else:
            ok = any(s["expect_question_contains"] in p.question for p in pairs)
        failures += not ok
        print(f"{'✅' if ok else '❌'} 抽到 {len(pairs)} 对:{[p.question for p in pairs]}")
    print(f"\n抽取符合预期 {len(samples) - failures}/{len(samples)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 4: 合成历史对话 seed**

Create `sql/ch03-seed.sql`(三段式 + `SET NAMES utf8mb4`,给 mining 喂料):

```sql
-- ch03 · 合成历史客服对话(喂挖知识 job)。三段式 + 字面量。
SET NAMES utf8mb4;

SELECT COUNT(*) AS before_conv FROM conversations;

INSERT INTO conversations (user_id, status) VALUES ('seed-u1', '已结束'), ('seed-u2', '已结束');
SET @c1 = (SELECT id FROM conversations WHERE user_id='seed-u1' ORDER BY id DESC LIMIT 1);
SET @c2 = (SELECT id FROM conversations WHERE user_id='seed-u2' ORDER BY id DESC LIMIT 1);

INSERT INTO messages (conversation_id, role, content) VALUES
  (@c1, 'user', '你们发货一般多久啊'),
  (@c1, 'assistant', '现货商品付款后48小时内发货,预售以详情页为准。'),
  (@c2, 'user', '满多少包邮'),
  (@c2, 'assistant', '单笔订单满99元包邮,未满收取10元运费。');

SELECT COUNT(*) AS after_conv FROM conversations;
```

Edit `Makefile`,追加:

```makefile
seed-conv:
	docker exec -i mewhelp-mysql mysql --default-character-set=utf8mb4 -uroot -proot mewhelp < sql/ch03-seed.sql

eval-mining:
	PYTHONPATH=. uv run python scripts/eval_mining.py
```

- [ ] **Step 5: 提交**

```bash
git add tests/data/retrieval_samples.json tests/data/mining_samples.json scripts/eval_retrieval.py scripts/eval_mining.py sql/ch03-seed.sql Makefile
git commit -m "feat(ch03): 检索/挖知识标注样例 + eval 脚本 + 合成对话 seed"
```

---

## Task 17: 录入 API + 作业运行器(TDD)

**TDD**。页面要的读数与写入都在这一层:库存盘点、切块预览(dry-run)、录入落库、补向量、检索自测;外加一个只能跑白名单 make 目标的作业运行器。

**Files:**
- `app/kb/sources.py`(新):`KB_DIR` + `SOURCE_TYPES`(文件 → content_type)+ `CONTENT_TYPES`;`scripts/build_kb.py` / `scripts/show_kb.py` 改为从这里读,不各留一份清单
- `app/db/repository.py`(改):`knowledge_stats` / `list_recent_chunks` / `list_chunk_pairs` / `staging_stats`
- `app/core/jobs.py`(新):`JobSpec` 白名单 + `start/stop/status/tail`
- `app/api/jobs.py`(新):`GET /api/jobs`、`POST /api/jobs/{name}`、`GET /api/jobs/{name}`、`POST /api/jobs/{name}/stop`
- `app/api/kb.py`(新):`GET /api/kb/overview`、`POST /api/kb/preview`、`POST /api/kb/ingest`、`POST /api/kb/vectorize`、`POST /api/kb/search`、`GET /api/kb/staging`
- `app/api/admin.py`(新):`GET /api/admin/overview`
- `tests/test_kb_api.py`、`tests/test_jobs_api.py`、`tests/test_admin_api.py`(新)
- `Makefile`(改):补 `kb-preview`、`kb-reset`

- [ ] **Step 1: 测试先行 —— 录入 API 的四条口径**

```python
# tests/test_kb_api.py
async def test_preview_is_dry_run(client):
    """预览只切不写:按完预览再看库存,一块都不该多。"""

async def test_ingest_then_reingest_is_idempotent(client):
    """同一份正文重录:第二次 inserted=0、skipped=全部,库里总数不变。"""

async def test_ingest_keeps_multiple_pieces_of_one_section(client):
    """大表格按行拆出的多块共用节标题——只按问法查重会误杀,这里锁住不许。"""

async def test_preview_rejects_bad_input(client):
    """空正文 / 非法 content_type / 不在材料清单里的文件名(含路径穿越)一律 400。"""
```
Expected: 四条全红(模块还不存在)。

- [ ] **Step 2: 测试先行 —— 作业运行器的安全边界**

```python
# tests/test_jobs_api.py
async def test_unknown_job_is_rejected(client):
    """作业名不在白名单 → 404;命令片段没有任何地方能传进来。"""

async def test_running_job_refuses_reentry(client, monkeypatch):
    """同名作业还在跑 → 409,别让两份进程抢同一批产物。"""

def test_every_job_target_exists_in_makefile():
    """注册表里每个作业都得是 Makefile 里真有的目标,免得按钮指向不存在的配方。"""
```
Expected: 三条全红。

- [ ] **Step 3: 实现到全绿**

要点:
- `preview` / `ingest` 都走 `documents.build_chunks`,`vectorize` 走 `dualwrite.vectorize_pending`——不在 API 里另写一套。
- 查重指纹 = `normalize(questions) + "|" + normalize(answer)`,复用 `app/kb/dedup.py` 的归一化。
- `ingest` 顺序:写 MySQL(pending)→ 向量化;向量化失败回 502 且**不回滚**,提示「已入库 N 块,补跑一次即可」。
- `overview` 里 mysql 与 Milvus 各自 try/except:任一读不到,`consistent` 回 `None`(读不到),不回 `False`(对不上)。
- 作业:`start_new_session=True` 起独立会话(停止时 killpg 连 make → uv → python 一起收);日志覆盖写 `log/acceptance/<job>.log`;`kb-mine` / `kb-reset` 标 heavy。

```bash
uv run pytest tests/test_kb_api.py tests/test_jobs_api.py tests/test_admin_api.py -q
git add -A && git commit -m "feat(ch03): 知识库录入 API + 白名单作业运行器(预览 dry-run、录入按指纹幂等)"
```
Expected: 全绿。

---

## Task 18: 知识库录入页 + 后台管理外壳

**非 TDD(Vibe Coding)**:页面按 spec §10 的六块布局,数全来自 Task 17 的接口;验收在浏览器里点,不写前端单测。静态页只加一条「能打开且引了共用外壳」的可达性断言。

**Files:**
- `app/static/kb.html`(新)、`app/static/admin.html`(新)、`app/static/admin.js`(新)
- `app/static/acceptance.css` / `acceptance.js`(复用:配色、闸条、表格、作业按钮与日志窗口)
- `app/static/index.html`(改):聊天页顶栏加「后台管理」入口
- `app/main.py`(改):`/kb`、`/admin` 路由
- `tests/test_static.py`(改):后台各页可达 + 都引了 `/static/admin.js`

- [ ] **Step 1: 录入页六块**

① 手工录入(内容类型 + 正文 + 预览 / 入库 + 「入库后顺手向量化」)、② 建库材料(清单 + 就地切块 + 逐份「看切块」+ 作业按钮)、③ 对话挖知识(暂存三态 + 作业按钮 + 逐条展开)、④ 向量化与双写(pending/done/Milvus 三数 + 补向量按钮 + 作业按钮)、⑤ 检索自测(问句 + 路线 + Top-K + 预设问句)、⑥ 最近入库。

预览里逐块标出:节路径、问法、正文、字数、是否关键条款、是否表格块、是否与库里重复;顶部标出三个切块特性有没有触发。

- [ ] **Step 2: 后台外壳**

`admin.js` 导出 `mountAdminNav(active)`:自带样式注入(挂在样式内联的旧页上也不打架)、包在 IIFE 里(不与宿主页的 `$` / `el` 撞名)、两层导航(模块行 + 当前模块的子页行)。`/admin` 聚合首页每模块一张卡:状态药丸(正常 / 要干活 / 没数据 / 读数失败)+ 一句结论 + 几个关键数 + 进入按钮。**各模块页面保持原路径**,首页只收入口。

- [ ] **Step 3: 作业日志不许在跑完那一刻消失**

作业进终态会回调页面重新取数,取数把按钮与日志窗口整个重建。按作业名缓存日志尾,重建时贴回去——不然最该读的结论正好被刷掉。

```bash
uv run pytest tests/test_static.py -q
git add -A && git commit -m "feat(ch03): 知识库录入页 + 后台管理聚合首页与共用导航"
```

---

## Task 19: 真实建库全链路 + 三项验收(全在浏览器里)

**集成验收**(主会话,真实服务)。起 `make dev` 之后一律在浏览器里走:建库、补向量、检索自测、语义召回,截图留档 `dev-notes/generated-images/`。

**Files:** 无新增(必要时 `README.md` 补 ch03 启动/验收段)。

- [ ] **Step 1: 起服务 + 在页面上建库**

`make dev`(应用;MySQL 复用 mewhelp-mysql),开 `/kb`:
1. ② 建库材料按「重跑 材料清单与切块预览」→ 日志窗口里看到各文件块数与「表格按行拆已触发」;页面表格里的块数与日志里的总块数对得上。
2. 按「重跑 离线建库」→ chunks 落 pending;顶部闸条 `待向量化` 有数。
3. ④ 区按「向量化待补块」→ pending 归零、`done` 与 `Milvus 条数` 相等、`双写` 显示「一致」。

Expected: 三步都在页面上完成,不回终端;库存闸条五个数自洽。

- [ ] **Step 2: 验收1(语义召回)—— 页面自测 + 聊天页**

1. `/kb` ⑤ 检索自测,路线选 `dense 向量单路`,点预设「邮费是多少」:Top-1 落在运费/包邮那块,分数与其余候选拉开;截图。
2. 聊天页 `/` 输入「邮费是多少」:出现「🔧 调用了 query_faq」徽章 + 答出满 99 包邮 / 运费规则;截图。
3. `make eval-retrieval` 作为回归线(≥4/5 通过)。

Expected: 换说法能召回,页面与聊天页两处一致。

- [ ] **Step 3: 验收2(中断重跑)—— 页面上演一遍**

1. ④ 区按「重跑 清库重建」(heavy,二次确认)→ 两表清空、集合 drop。
2. 按「重跑 离线建库」→ 只有 pending,`双写` 显示「对不上」。
3. 按「向量化待补块」中途停掉(或临时改错 key 让它失败)→ 闸条显示 done 与 pending 并存。
4. 修好后再按一次 → 只捡剩余 pending 补齐,全部 done,`Milvus 条数 == 知识块数`,无重无漏。

Expected: 幂等恢复的实证不用回终端看 SQL——闸条那三个数就是账。
(该性质已被 Task 10 的 `test_vectorize_resumes_after_crash` 单测证明;此处做一次真实链路演示。)

- [ ] **Step 4: 验收3(手工录入)—— 贴一段进去**

1. ① 区「填一段示例」→ 内容类型选 `policy` → 「切块预览」:两节两块,关键条款标出来,表格块标出来,查重显示「无重复,可入库」。
2. 「录入入库」(勾了顺手向量化)→ toast 报入库块数与向量化块数;闸条总数增加、pending 仍为 0。
3. 再按一次「录入入库」→ 预览区每块标「重复」,入库 0 跳过全部——录入天然幂等。
4. ⑤ 检索自测再问一次「邮费是多少」:刚录进去的那块出现在 Top-K 里(带它的 id)。

Expected: 四步都在浏览器里,截图留档。

- [ ] **Step 5: 挖知识全链路(需聊天上游)**

`/kb` ③ 区:「灌历史会话种子」→「对话挖知识」(heavy,二次确认)→ 暂存三态出现落差(抽出 / 保留 / 丢弃),「看暂存表逐条」能逐条对照;再按 ④ 区补向量。
`make eval-mining` 作为抽取质量回归线。

Expected: 去重那一刀在页面上看得见;新挖的块向量化后可被检索到。

- [ ] **Step 6: 后台首页与全量回归**

`/admin` 上知识库那张卡出得来:关掉 Milvus 只让它说「Milvus 离线」,不连坐整页(后面每往后台加一块,这页多一张卡)。

```bash
uv run pytest -q
git add -A && git commit -m "test(ch03): 建库全链在浏览器里跑通,三项验收(语义召回 / 中断重跑 / 手工录入幂等)通过"
```

---

## Self-Review(计划对照 spec)

**1. Spec coverage:**
- 功能1 文档处理 → Task 6/7/8/9(标题切/递归切/句末重叠/表格行切复表头 + 字段映射)✓
- 功能2 对话挖知识 job → Task 12(mining + CLI,staging→去重→pending)+ Task 16(seed + eval)✓
- 功能3 落库结构(三格拼向量 + 四类元数据)→ Task 4/5(ORM+CRUD 全字段)、Task 9(字段映射)、Task 10(向量化文本 = category+questions+answer)✓
- 功能4 双写幂等 → Task 10(write_pending + vectorize_pending 按 id upsert,恢复单测)✓
- 功能5 在线检索替换 → Task 13(search_knowledge)+ Task 14(query_faq 契约不变)✓
- 功能6 录入页 → Task 17(录入 API + 作业运行器,TDD)+ Task 18(六块页面 + 后台外壳)✓
- 基础设施(Milvus Lite / 嵌入上游 / env / deps)→ Task 1 ✓
- 验收1 → Task 16 eval + Task 19 Step 2(录入页自测 + 聊天页);验收2 → Task 10 单测 + Task 19 Step 3(页面上演);验收3(建库全链在浏览器里)→ Task 19 Step 1/4/5 ✓
- spec §10 不变量各有归属:切块一份实现(Task 17 Step 3)、双写顺序不许反与失败不回滚(Task 17 Step 3)、指纹查重不误杀同节多块(Task 17 Step 1 用例)、读不到 ≠ 对不上(Task 17 Step 3)、没有 shell(Task 17 Step 2 用例)✓

**2. Placeholder scan:** 无 TBD/TODO;每个代码步给出完整测试+实现;夹具给出具体 markdown/JSON/SQL。`retrieval_min_score=0.4` 为初值,Task 19 按 eval 结果微调(已注明)。

**3. Type consistency:** `Chunk`(documents.py)贯穿 documents/dualwrite/mining;`search_knowledge`/`milvus_client.search` 返回 `{id,score,question,answer}` 在 retrieval/query_faq/eval 一致;repository 方法名(`insert_knowledge_chunk`/`list_pending_chunks`/`mark_chunk_vectorized`/`set_chunk_neighbors`/`count_chunks_by_status`/`list_all_questions`/`insert_staging`/`list_staging_by_status`/`set_staging_status`/`list_conversations_with_messages`)跨任务一致。

**已知需实现时留意:** Milvus Lite `search`/`count` 返回结构以 Task 3 真实 Lite 跑通为准修正;`test_tool_faq.py`(ch02 查表断言)在 Task 14 按新实现改造,不保留查表断言。
