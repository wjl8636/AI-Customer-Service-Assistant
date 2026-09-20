# Ch02 Function Calling 工具链 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 ch01 纯对话客服装上「查数据」能力,而且能力长在前端在用的聊天入口上——用 LangChain `@tool` 定义 5 个业务工具,让 `glm-5.2` 经 Function Calling 自己决定调哪个工具,后端单轮执行并把结果回灌;用户在 ch01 那个 SSE 流式聊天页问一句就触发,最终回答仍逐 token 流式吐出,气泡显示工具轨迹徽章。

**Architecture:** 编排抽成共享核心 `_prepare_turn`(会话身份→落 user→turn1 `bind_tools` 定工具→落 assistant→并行执行工具落 tool),被两个出口共用——`stream_agent_turn`(流式,喂 `POST /api/chat` 前端主入口)与 `run_agent_turn`(非流式,喂 `POST /api/agent` 程序化/测试出口)。收敛那步都**不 bind_tools**(强制单轮):流式用 `astream` 逐 token 吐,非流式用 `ainvoke` 一次性返回。工具执行经基础设施(注册/校验/超时/重试/错误回灌)。会话/消息落 MySQL。`/api/extract` 不动。

**Tech Stack:** Python 3.12 · FastAPI · SQLAlchemy 2.0 async(asyncmy 驱动)· MySQL(Docker)· LangChain `@tool`/`bind_tools`/`astream`。前端为 ch01 静态聊天页(原生 JS + SSE),改造走 Vibe Coding。LLM 链路沿用 ch01:`ChatOpenAI` 直连 `CHAT_BASE_URL` 指定的上游。

## Global Constraints

以下为项目级约束,每个任务都隐含遵守(值从 spec 逐字抄录):

- Python `>=3.12`;依赖管理 `uv`。
- 技术选型定死:FastAPI + SQLAlchemy 2.0 async + MySQL(Docker)+ LangChain `@tool`。**不用 Redis**、**不用 LangGraph**。实现中发现矛盾或走不通,**停下来问用户,不自行换方案**。
- 新增依赖:`sqlalchemy[asyncio]>=2.0.0`、`asyncmy>=0.2.9`(若 `asyncmy` 在本机 uv 编译失败,停下问用户是否改 `aiomysql`——属技术选型变更)。
- LLM 链路沿用 ch01:应用层 `ChatOpenAI` 说 OpenAI 协议,直连 `CHAT_BASE_URL` 指定的上游;个别模型对 `temperature` 有限制(glm 只接受 `temperature=1`)。
- 单轮硬约束对两个出口都成立:收敛那步**不** `bind_tools`,天然收敛,不出现第二轮工具执行;流式出口收敛用 `astream` 逐 token。
- `create_ticket.conversation_id` 由系统注入(`InjectedToolArg`),**不暴露给模型**;写类工具(`create_ticket`)**不自动重试**。
- 建表以 [sql/ch02-ddl.sql](sql/ch02-ddl.sql) 为**权威**,ORM 只映射、**不用 `metadata.create_all`**。
- `/api/chat` 升级为带工具链的 SSE 流式主入口(见 Task 12);`/api/extract`、`app/core/llm.py` **保持 ch01 原样**。内存 `SessionStore` 与 `CUSTOMER_SERVICE_PROMPT` 弃用但**不删文件**(chat 不再引用)。
- 前端聊天页改造走 **Vibe Coding**(不套 brainstorm/TDD/code review);验收在浏览器聊天页跑三条。
- 涉及库 API 已在计划前用 Context7 核对(LangChain 1.3 / SQLAlchemy 2.0);实现中若遇 API 偏差以官方文档+实测为准。
- 测试:可单测代码走 TDD(先红后绿);纯 Prompt/模型行为走标注样例评估。DB 测试连 Docker MySQL 的 `mewhelp_test` 库(不用 SQLite——ENUM/`ON UPDATE`/JSON 行为不一致)。

---

## 任务清单概览

1. **glm-5.2 tool-calling 冒烟**(go/no-go 风险闸)
2. **项目基建与 DB 连接**(docker-compose + 依赖 + config + db/base + 测试 fixture)
3. **ORM 模型**(4 张表映射)
4. **数据访问层 repository**(会话/消息/faq/工单 CRUD)
5. **faq 种子数据**(ch02-seed.sql,含验收 2/3 数据设计)
6. **三个 mock 工具**(query_order/product/logistics)
7. **query_faq 工具**(查 faq 表 LIKE)
8. **create_ticket 工具**(InjectedToolArg 注入 + 写库 + 会话状态流转)
9. **工具注册表 + 基础设施**(注册/校验/超时/重试/错误回灌)
10. **客服+工具 system prompt**
11. **编排 core/agent.py**(共享 `_prepare_turn` + `run_agent_turn` 非流式 + `stream_agent_turn` 流式)
12. **接口层**(`/api/chat` 升级为带工具链 SSE 流式主入口 + `/api/agent` 非流式程序化出口 + schemas + 挂载)
13. **前端聊天页**(Vibe Coding:工具徽章 + markdown 渲染 + 工具期间打字点 + conversation_id 持久化)
14. **标注样例评估 + 浏览器验收 + demo + dev-notes 留痕**

---

## Task 1: glm-5.2 tool-calling 冒烟(go/no-go 风险闸)

**为什么第一个**:上游模型的 tool calling 支持未验证。这是整个 ch02 的地基,若不支持必须立刻停下问用户。此任务**不依赖 DB**,只用 ch01 已有的 LLM 链路。

**Files:**
- Create: `scripts/smoke_toolcall.py`

**Interfaces:**
- Consumes: `app.core.llm.get_chat_model`(ch01 已有)
- Produces: 无(一次性验证脚本);结论写入 dev-notes

- [ ] **Step 1: 确认上游可用**

Run: `curl -s "$CHAT_BASE_URL/models" -H "Authorization: Bearer $CHAT_API_KEY"` (若无输出,先查 `.env` 里 `CHAT_*` 三项)
Expected: 返回含 `glm-5.2` 的 JSON。

- [ ] **Step 2: 写冒烟脚本**

```python
# scripts/smoke_toolcall.py
"""真实验证当前上游模型能否返回结构化 tool_calls。go/no-go 风险闸。"""
import asyncio

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from app.core.llm import get_chat_model

class AddInput(BaseModel):
    a: int = Field(description="第一个加数")
    b: int = Field(description="第二个加数")

@tool(args_schema=AddInput)
def add(a: int, b: int) -> int:
    """把两个整数相加。"""
    return a + b

async def main() -> None:
    model = get_chat_model()  # 非流式,直连 settings.chat_base_url
    bound = model.bind_tools([add])
    ai = await bound.ainvoke("请用工具计算 23 加 19 等于多少")
    print("content:", repr(ai.content))
    print("tool_calls:", ai.tool_calls)
    if ai.tool_calls and ai.tool_calls[0]["name"] == "add":
        print("✅ GO:glm-5.2 支持 tool calling,选中 add,args=", ai.tool_calls[0]["args"])
    else:
        print("❌ NO-GO:未返回预期 tool_calls —— 停下来问用户,不自行换方案")

if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: 跑冒烟**

Run: `uv run python scripts/smoke_toolcall.py`
Expected(GO):打印 `✅ GO`,`tool_calls` 含 `{'name':'add','args':{'a':23,'b':19},'id':...}`。

- [ ] **Step 4: 判定 go/no-go**

- 若 GO:在 `dev-notes/ch02.md` 追加「阶段 3 · Task 1」段,记录冒烟通过、tool_calls 原始结构。继续 Task 2。
- 若 NO-GO(无 tool_calls / 报错 / 抖动多次不稳):**停止执行,回报用户**——这是 spec 标注的红线(技术选型走不通),不自行换方案/换模型。

- [ ] **Step 5: Commit**

```bash
git add scripts/smoke_toolcall.py dev-notes/ch02.md
git commit -m "test(ch02): glm-5.2 tool-calling 冒烟通过(go/no-go 风险闸)"
```

---

## Task 2: 项目基建与 DB 连接

**Files:**
- Create: `docker-compose.yml`
- Modify: `pyproject.toml`(加依赖)
- Modify: `app/config.py`(加 `database_url` / `test_database_url`)
- Modify: `.env.example`(加 DB 变量)
- Create: `app/db/__init__.py`(空)
- Create: `app/db/base.py`
- Create: `tests/conftest.py`(DB fixture)
- Test: `tests/test_db_conn.py`

**Interfaces:**
- Produces:
  - `app.db.base.Base`(`DeclarativeBase` 子类)
  - `app.db.base.engine`(`AsyncEngine`)
  - `app.db.base.async_session`(`async_sessionmaker[AsyncSession]`)
  - `app.config.settings.database_url: str`、`settings.test_database_url: str`
  - pytest fixture `db_clean`(function 级,每测清空 4 张表)

- [ ] **Step 1: 加依赖**

Run:
```bash
uv add "sqlalchemy[asyncio]>=2.0.0" "asyncmy>=0.2.9"
```
Expected:`pyproject.toml` 的 `dependencies` 新增两项,`uv.lock` 更新。若 asyncmy 编译失败 → 停下问用户。

- [ ] **Step 2: 写 docker-compose.yml**

```yaml
services:
  mysql:
    image: mysql:8.0
    container_name: mewhelp-mysql
    environment:
      MYSQL_ROOT_PASSWORD: root
      MYSQL_DATABASE: mewhelp
    ports:
      - "3306:3306"
    volumes:
      - mewhelp-mysql-data:/var/lib/mysql
      - ./sql:/docker-entrypoint-initdb.d:ro   # 首启按文件名序执行 ch02-ddl.sql → ch02-seed.sql
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost", "-proot"]
      interval: 3s
      timeout: 3s
      retries: 20

volumes:
  mewhelp-mysql-data:
```

- [ ] **Step 3: 起 MySQL 并确认建表**

Run:
```bash
docker compose up -d
docker compose exec -T mysql sh -c 'until mysqladmin ping -proot --silent; do sleep 1; done'
docker compose exec -T mysql mysql -uroot -proot mewhelp -e "SHOW TABLES;"
```
Expected:列出 `conversations`、`faq`、`messages`、`tickets` 四张表(initdb 自动跑了 ch02-ddl.sql)。

- [ ] **Step 4: 扩展 config 与 .env.example**

`app/config.py` 在 `Settings` 内新增(其余不动):
```python
    database_url: str = "mysql+asyncmy://root:root@localhost:3306/mewhelp"
    test_database_url: str = "mysql+asyncmy://root:root@localhost:3306/mewhelp_test"
```
`.env.example` 追加:
```
DATABASE_URL=mysql+asyncmy://root:root@localhost:3306/mewhelp
TEST_DATABASE_URL=mysql+asyncmy://root:root@localhost:3306/mewhelp_test
```

- [ ] **Step 5: 写 db/base.py**

```python
# app/db/base.py
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

class Base(DeclarativeBase):
    pass

engine = create_async_engine(settings.database_url, pool_pre_ping=True)
async_session: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine, expire_on_commit=False
)
```

- [ ] **Step 6: 写 conftest.py(测试库 fixture)**

```python
# tests/conftest.py
import pathlib

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings

_DDL = pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch02-ddl.sql"
_TABLES = ["messages", "tickets", "conversations", "faq"]  # 删除顺序:先子表后父表

@pytest_asyncio.fixture(scope="session")
async def _test_engine():
    # 用 server-url(不带库名)建 mewhelp_test 库
    server_url = settings.test_database_url.rsplit("/", 1)[0]
    admin = create_async_engine(server_url, isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(text("DROP DATABASE IF EXISTS mewhelp_test"))
        await conn.execute(text("CREATE DATABASE mewhelp_test CHARACTER SET utf8mb4"))
    await admin.dispose()

    engine = create_async_engine(settings.test_database_url, pool_pre_ping=True)
    # 跑 DDL 建表(以 ch02-ddl.sql 为权威;按 ; 分割逐句执行)
    stmts = [s.strip() for s in _DDL.read_text(encoding="utf-8").split(";") if s.strip()]
    async with engine.begin() as conn:
        for s in stmts:
            if s.lstrip().upper().startswith("CREATE TABLE"):
                await conn.execute(text(s))
    yield engine
    await engine.dispose()

@pytest_asyncio.fixture()
async def db_session_factory(_test_engine, monkeypatch):
    """把 repository 用到的 async_session 指向测试库。"""
    factory = async_sessionmaker(_test_engine, expire_on_commit=False)
    monkeypatch.setattr("app.db.base.async_session", factory)
    return factory

@pytest_asyncio.fixture()
async def db_clean(_test_engine):
    """每个测试后清空 4 张表(关外键检查以便 TRUNCATE)。"""
    yield
    async with _test_engine.begin() as conn:
        await conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        for t in _TABLES:
            await conn.execute(text(f"TRUNCATE TABLE {t}"))
        await conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
```

> 注:`monkeypatch.setattr` 让 repository 里 `from app.db.base import async_session` 的引用换成测试库工厂。repository 必须**在函数内部引用 `app.db.base.async_session`**(见 Task 4 注),否则 patch 不生效。

- [ ] **Step 7: 写连接测试**

```python
# tests/test_db_conn.py
from sqlalchemy import text

async def test_test_db_reachable_and_tables_exist(_test_engine, db_clean):
    async with _test_engine.connect() as conn:
        rows = (await conn.execute(text("SHOW TABLES"))).scalars().all()
    assert {"conversations", "messages", "faq", "tickets"} <= set(rows)
```

- [ ] **Step 8: 跑测试**

Run: `uv run pytest tests/test_db_conn.py -v`
Expected: PASS(测试库建库、建表、连接均正常)。

- [ ] **Step 9: Commit**

```bash
git add docker-compose.yml pyproject.toml uv.lock app/config.py .env.example app/db/ tests/conftest.py tests/test_db_conn.py
git commit -m "feat(ch02): MySQL docker-compose + SQLAlchemy async 连接骨架与测试 fixture"
```

---

## Task 3: ORM 模型(4 张表映射)

**Files:**
- Create: `app/db/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `app.db.base.Base`、`app.db.base.async_session`(测试用)
- Produces: `Conversation`、`Message`、`Faq`、`Ticket`(字段见下)

- [ ] **Step 1: 写映射测试(先红)**

```python
# tests/test_models.py
from app.db.models import Conversation, Message, Ticket

async def test_conversation_defaults_and_autoincrement(db_session_factory, db_clean):
    async with db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        await s.commit()
        assert conv.id is not None          # BIGINT 自增
        assert conv.status == "进行中"        # DB DEFAULT 读回
        assert conv.created_at is not None

async def test_message_json_tool_calls_roundtrip(db_session_factory, db_clean):
    async with db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        await s.flush()
        msg = Message(
            conversation_id=conv.id,
            role="assistant",
            content=None,
            tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}],
        )
        s.add(msg)
        await s.commit()
        got = await s.get(Message, msg.id)
        assert got.tool_calls[0]["name"] == "query_order"   # JSON 往返
        assert got.role == "assistant"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL(`ModuleNotFoundError: app.db.models`)。

- [ ] **Step 3: 写 models.py**

```python
# app/db/models.py
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Enum, ForeignKey, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(
        Enum("进行中", "已转人工", "已结束"), server_default="进行中"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id")
    )
    role: Mapped[str] = mapped_column(Enum("user", "assistant", "tool"))
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

class Faq(Base):
    __tablename__ = "faq"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(512))
    answer: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

class Ticket(Base):
    __tablename__ = "tickets"

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id")
    )
    description: Mapped[str] = mapped_column(Text)
    ticket_type: Mapped[str] = mapped_column(Enum("售后", "投诉", "咨询"))
    status: Mapped[str] = mapped_column(
        Enum("待处理", "已处理"), server_default="待处理"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

> 注:`server_default` 仅作声明对齐 DDL;因不 `create_all`,实际默认值由 MySQL 的 DDL 提供。insert 不传 `status`/`created_at` 时由 DB 填,再读回。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS(2 项)。

- [ ] **Step 5: Commit**

```bash
git add app/db/models.py tests/test_models.py
git commit -m "feat(ch02): 4 张表 ORM 映射(Conversation/Message/Faq/Ticket)"
```

---

## Task 4: 数据访问层 repository

**Files:**
- Create: `app/db/repository.py`
- Test: `tests/test_repository.py`

**Interfaces:**
- Consumes: `app.db.models`、`app.db.base.async_session`
- Produces(全部 async):
  - `create_conversation(user_id: str) -> int`
  - `get_conversation(conversation_id: int) -> Conversation | None`
  - `append_message(conversation_id: int, role: str, content: str | None = None, tool_calls: list | None = None, tool_call_id: str | None = None) -> int`
  - `list_messages(conversation_id: int) -> list[Message]`
  - `search_faq(keyword: str) -> list[Faq]`
  - `create_ticket(conversation_id: int, description: str, ticket_type: str) -> str`(生成 `ticket_no`,写库,并把会话 `status` 置「已转人工」)

- [ ] **Step 1: 写 repository 测试(先红)**

```python
# tests/test_repository.py
from app.db import repository as repo
from app.db.models import Conversation, Faq, Ticket

async def test_create_and_get_conversation(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    conv = await repo.get_conversation(cid)
    assert conv is not None and conv.user_id == "u1"
    assert await repo.get_conversation(999999) is None

async def test_append_and_list_messages_in_order(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    await repo.append_message(cid, "user", content="订单 1001 到哪了")
    await repo.append_message(cid, "assistant", tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])
    await repo.append_message(cid, "tool", content='{"status":"运输中"}', tool_call_id="c1")
    msgs = await repo.list_messages(cid)
    assert [m.role for m in msgs] == ["user", "assistant", "tool"]
    assert msgs[2].tool_call_id == "c1"

async def test_search_faq_like_hit_and_miss(db_session_factory, db_clean):
    async with db_session_factory() as s:
        s.add(Faq(question="退货政策", answer="7 天无理由退货", category="售后"))
        await s.commit()
    assert len(await repo.search_faq("退货")) == 1          # 命中
    assert await repo.search_faq("鞋子") == []              # 漏召回(字面不含)

async def test_create_ticket_writes_and_flips_conversation_status(db_session_factory, db_clean):
    cid = await repo.create_conversation("u1")
    no = await repo.create_ticket(cid, "要退货", "售后")
    assert no.startswith("T")
    async with db_session_factory() as s:
        t = await s.get(Ticket, no)
        assert t.ticket_type == "售后" and t.status == "待处理"
        conv = await s.get(Conversation, cid)
        assert conv.status == "已转人工"          # 会话状态流转
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_repository.py -v`
Expected: FAIL(`ModuleNotFoundError: app.db.repository`)。

- [ ] **Step 3: 写 repository.py**

```python
# app/db/repository.py
from datetime import datetime

from sqlalchemy import select

import app.db.base as db          # 用模块属性引用,便于测试 monkeypatch async_session
from app.db.models import Conversation, Faq, Message, Ticket

_TICKET_SEQ = 0

def _gen_ticket_no() -> str:
    global _TICKET_SEQ
    _TICKET_SEQ += 1
    return f"T{datetime.now():%Y%m%d%H%M%S}{_TICKET_SEQ:03d}"

async def create_conversation(user_id: str) -> int:
    async with db.async_session() as s:
        conv = Conversation(user_id=user_id)
        s.add(conv)
        await s.commit()
        return conv.id

async def get_conversation(conversation_id: int) -> Conversation | None:
    async with db.async_session() as s:
        return await s.get(Conversation, conversation_id)

async def append_message(
    conversation_id: int,
    role: str,
    content: str | None = None,
    tool_calls: list | None = None,
    tool_call_id: str | None = None,
) -> int:
    async with db.async_session() as s:
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )
        s.add(msg)
        await s.commit()
        return msg.id

async def list_messages(conversation_id: int) -> list[Message]:
    async with db.async_session() as s:
        result = await s.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        )
        return list(result.scalars())

async def search_faq(keyword: str) -> list[Faq]:
    async with db.async_session() as s:
        result = await s.execute(
            select(Faq).where(Faq.question.like(f"%{keyword}%"))
        )
        return list(result.scalars())

async def create_ticket(conversation_id: int, description: str, ticket_type: str) -> str:
    ticket_no = _gen_ticket_no()
    async with db.async_session() as s:
        s.add(
            Ticket(
                ticket_no=ticket_no,
                conversation_id=conversation_id,
                description=description,
                ticket_type=ticket_type,
            )
        )
        conv = await s.get(Conversation, conversation_id)
        if conv is not None:
            conv.status = "已转人工"
        await s.commit()
    return ticket_no
```

> 注:conftest 的 `db_session_factory` 用 `monkeypatch.setattr("app.db.base.async_session", ...)`;因此 repository 必须通过 `db.async_session()`(模块属性)调用,不能 `from app.db.base import async_session` 顶层绑定——否则 patch 不生效。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_repository.py -v`
Expected: PASS(4 项)。

- [ ] **Step 5: Commit**

```bash
git add app/db/repository.py tests/test_repository.py
git commit -m "feat(ch02): 数据访问层 repository(会话/消息/faq/工单 CRUD)"
```

---

## Task 5: faq 种子数据

**Files:**
- Create: `sql/ch02-seed.sql`
- Modify: `Makefile`(加 `seed` target)

**Interfaces:**
- Produces: `faq` 表测试数据;运维可 `make seed` 重灌(幂等)

**数据设计(命中验收 2、暴露验收 3 漏召回)**:FAQ 用**书面语**,question 里**不含「鞋子」等具体品类词**。这样「退货政策是什么」→ `query_faq(keyword="退货政策"/"退货")` 命中;「买的鞋子能不能退」→ 模型多半提取「鞋子」→ `question LIKE '%鞋子%'` 查不到,暴露关键词检索的语义鸿沟。

- [ ] **Step 1: 写 seed 脚本**(按 sql-checklist:INSERT VALUES 字面量 + 幂等)

```sql
-- sql/ch02-seed.sql
-- faq 种子数据。幂等靠 question 唯一;重复执行不新增。
-- 运维对账:执行后应有 6 行。

INSERT INTO faq (question, answer, category)
SELECT * FROM (
  SELECT '退货政策' AS q, '支持 7 天无理由退货,商品需保持完好、不影响二次销售,以平台售后规则为准。' AS a, '售后' AS c
  UNION ALL SELECT '如何申请退款', '在「我的订单」找到对应订单点击「申请退款」,按提示提交,审核通过后原路退回。', '售后'
  UNION ALL SELECT '换货流程', '收到商品 7 天内可申请换货,联系客服登记后寄回,平台核验后补发。', '售后'
  UNION ALL SELECT '发货时效', '现货商品付款后 48 小时内发货,预售商品以详情页标注时间为准。', '物流'
  UNION ALL SELECT '运费怎么算', '单笔订单满 99 元包邮,未满收取 10 元运费,偏远地区另计。', '物流'
  UNION ALL SELECT '发票如何开具', '在「我的订单」-「申请开票」提交抬头与税号,电子发票 3 个工作日内发送至邮箱。', '账务'
) AS seed
WHERE NOT EXISTS (SELECT 1 FROM faq f WHERE f.question = seed.q);
```

- [ ] **Step 2: 加 Makefile target**

`Makefile` 追加(`.PHONY` 行补上 `seed`):
```makefile
seed:
	docker compose exec -T mysql mysql -uroot -proot mewhelp < sql/ch02-seed.sql
```

- [ ] **Step 3: 灌数并验证行数**

Run:
```bash
make seed
docker compose exec -T mysql mysql -uroot -proot mewhelp -e "SELECT COUNT(*) FROM faq;"
```
Expected: `6`。再跑一次 `make seed`,仍为 `6`(幂等)。

- [ ] **Step 4: 验证验收 2/3 的 LIKE 行为**

Run:
```bash
docker compose exec -T mysql mysql -uroot -proot mewhelp -e "SELECT question FROM faq WHERE question LIKE '%退货%';"
docker compose exec -T mysql mysql -uroot -proot mewhelp -e "SELECT question FROM faq WHERE question LIKE '%鞋子%';"
```
Expected:第一条返回「退货政策」;第二条**空**(漏召回,符合预期)。

- [ ] **Step 5: Commit**

```bash
git add sql/ch02-seed.sql Makefile
git commit -m "feat(ch02): faq 种子数据(命中退货政策/暴露鞋子漏召回)"
```

---

## Task 6: 三个 mock 工具(query_order / query_product / query_logistics)

**Files:**
- Create: `app/tools/__init__.py`(空)
- Create: `app/tools/business.py`(本任务先放 3 个 mock 工具,后续任务往同文件加)
- Test: `tests/test_tools_mock.py`

**Interfaces:**
- Produces:`query_order`、`query_product`、`query_logistics`(均为 LangChain `BaseTool`,async;`.name` 分别等于函数名)
- 约定:mock 以入参为随机种子 → **同一入参多次调用结果一致**(eval 可复现)

- [ ] **Step 1: 写 mock 工具测试(先红)**

```python
# tests/test_tools_mock.py
from app.tools.business import query_logistics, query_order, query_product

async def test_query_order_deterministic_and_shaped():
    r1 = await query_order.ainvoke({"order_id": "1001"})
    r2 = await query_order.ainvoke({"order_id": "1001"})
    assert r1 == r2                                  # 同种子可复现
    assert r1["order_id"] == "1001"
    assert r1["status"] in {"待付款", "已付款", "已发货", "已签收"}

async def test_query_product_and_logistics_names():
    assert query_product.name == "query_product"
    assert query_logistics.name == "query_logistics"
    p = await query_product.ainvoke({"product_name": "猫粮"})
    assert p["product_name"] == "猫粮" and "price" in p
    lg = await query_logistics.ainvoke({"order_id": "1001"})
    assert lg["order_id"] == "1001" and "status" in lg and "timeline" in lg
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_tools_mock.py -v`
Expected: FAIL(`ModuleNotFoundError: app.tools.business`)。

- [ ] **Step 3: 写 3 个 mock 工具**

```python
# app/tools/business.py
import random

from langchain_core.tools import tool
from pydantic import BaseModel, Field

class OrderInput(BaseModel):
    order_id: str = Field(description="订单号,例如 1001")

class ProductInput(BaseModel):
    product_name: str = Field(description="商品名称或关键词,例如 猫粮")

class LogisticsInput(BaseModel):
    order_id: str = Field(description="订单号,用于查询该订单的物流轨迹")

@tool(args_schema=OrderInput)
async def query_order(order_id: str) -> dict:
    """查询订单的状态、金额、下单时间和商品名。用于用户询问某个订单情况时。"""
    rng = random.Random(f"order:{order_id}")
    return {
        "order_id": order_id,
        "status": rng.choice(["待付款", "已付款", "已发货", "已签收"]),
        "amount": rng.randint(50, 2000),
        "created_at": f"2026-07-{rng.randint(1, 12):02d} 10:00",
        "product": rng.choice(["智能猫砂盆", "猫粮 5kg", "猫爬架", "自动饮水机"]),
    }

@tool(args_schema=ProductInput)
async def query_product(product_name: str) -> dict:
    """查询商品的价格、库存和规格。用于用户咨询某商品是否有货、多少钱时。"""
    rng = random.Random(f"product:{product_name}")
    return {
        "product_name": product_name,
        "price": rng.randint(20, 999),
        "stock": rng.randint(0, 500),
        "spec": rng.choice(["标准装", "家庭装", "试用装"]),
    }

@tool(args_schema=LogisticsInput)
async def query_logistics(order_id: str) -> dict:
    """查询订单的物流状态、当前位置和轨迹。用于用户询问物流/快递到哪了时。"""
    rng = random.Random(f"logistics:{order_id}")
    status = rng.choice(["已揽件", "运输中", "派送中", "已签收"])
    city = rng.choice(["深圳", "广州", "杭州", "上海", "成都"])
    return {
        "order_id": order_id,
        "status": status,
        "location": f"{city}分拨中心",
        "timeline": [f"{city}分拨中心 已发出", f"当前状态:{status}"],
    }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_tools_mock.py -v`
Expected: PASS(2 项)。

- [ ] **Step 5: Commit**

```bash
git add app/tools/__init__.py app/tools/business.py tests/test_tools_mock.py
git commit -m "feat(ch02): 三个 mock 工具(order/product/logistics,入参种子可复现)"
```

---

## Task 7: query_faq 工具

**Files:**
- Modify: `app/tools/business.py`(追加 `query_faq`)
- Test: `tests/test_tool_faq.py`

**Interfaces:**
- Consumes: `app.db.repository.search_faq`
- Produces: `query_faq`(async `BaseTool`);返回 `{"hits": [{"question","answer"}], "message"?: str}`

- [ ] **Step 1: 写测试(先红)**

```python
# tests/test_tool_faq.py
from app.db.models import Faq
from app.tools.business import query_faq

async def test_query_faq_hit(db_session_factory, db_clean):
    async with db_session_factory() as s:
        s.add(Faq(question="退货政策", answer="7 天无理由", category="售后"))
        await s.commit()
    r = await query_faq.ainvoke({"keyword": "退货"})
    assert r["hits"] and r["hits"][0]["question"] == "退货政策"

async def test_query_faq_miss_returns_message(db_session_factory, db_clean):
    async with db_session_factory() as s:
        s.add(Faq(question="退货政策", answer="7 天无理由", category="售后"))
        await s.commit()
    r = await query_faq.ainvoke({"keyword": "鞋子"})
    assert r["hits"] == [] and "未找到" in r["message"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_tool_faq.py -v`
Expected: FAIL(`ImportError: cannot import name 'query_faq'`)。

- [ ] **Step 3: 追加 query_faq**

`app/tools/business.py` 顶部导入追加:
```python
from app.db import repository
```
文件末尾追加:
```python
class FaqInput(BaseModel):
    keyword: str = Field(description="用于在常见问题库中检索的关键词,如『退货』『发货时效』")

@tool(args_schema=FaqInput)
async def query_faq(keyword: str) -> dict:
    """根据关键词查询常见问题解答(FAQ)。用于用户咨询政策、规则、操作流程等通用问题时。"""
    rows = await repository.search_faq(keyword)
    if not rows:
        return {"hits": [], "message": f"未找到与「{keyword}」相关的常见问题"}
    return {"hits": [{"question": r.question, "answer": r.answer} for r in rows]}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_tool_faq.py -v`
Expected: PASS(2 项)。

- [ ] **Step 5: Commit**

```bash
git add app/tools/business.py tests/test_tool_faq.py
git commit -m "feat(ch02): query_faq 工具(查 faq 表 LIKE,未命中返回提示)"
```

---

## Task 8: create_ticket 工具(InjectedToolArg 注入 + 写库)

**Files:**
- Modify: `app/tools/business.py`(追加 `create_ticket`)
- Test: `tests/test_tool_ticket.py`

**Interfaces:**
- Consumes: `app.db.repository.create_ticket`、`langchain_core.tools.InjectedToolArg`
- Produces: `create_ticket`(async `BaseTool`);模型可见参数仅 `description`、`ticket_type`;`conversation_id` 为 `InjectedToolArg`(执行时代码注入),返回 `{"ticket_no", "status"}`

- [ ] **Step 1: 写测试(先红)——含「conversation_id 不在模型 schema」断言**

```python
# tests/test_tool_ticket.py
from app.db.models import Conversation, Ticket
from app.tools.business import create_ticket

def test_conversation_id_hidden_from_model_schema():
    # InjectedToolArg 的参数不应出现在暴露给 LLM 的 args schema 里
    schema = create_ticket.get_input_schema().model_json_schema()
    props = schema.get("properties", {})
    assert "description" in props and "ticket_type" in props
    assert "conversation_id" not in props        # 关键:模型看不到会话主键

async def test_create_ticket_injects_conversation_id_and_writes(db_session_factory, db_clean):
    async with db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        await s.commit()
        cid = conv.id
    r = await create_ticket.ainvoke(
        {"description": "商品损坏要退货", "ticket_type": "售后", "conversation_id": cid}
    )
    assert r["ticket_no"].startswith("T")
    async with db_session_factory() as s:
        t = await s.get(Ticket, r["ticket_no"])
        assert t is not None and t.conversation_id == cid
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_tool_ticket.py -v`
Expected: FAIL(`ImportError: cannot import name 'create_ticket'`)。

- [ ] **Step 3: 追加 create_ticket(InjectedToolArg)**

`app/tools/business.py` 顶部导入追加:
```python
from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg
```
文件末尾追加:
```python
@tool
async def create_ticket(
    description: str,
    ticket_type: Literal["售后", "投诉", "咨询"],
    conversation_id: Annotated[int, InjectedToolArg],
) -> dict:
    """当用户问题需要人工介入(投诉、无法自助解决、明确要求人工)时,创建人工工单。
    description 填用户问题描述,ticket_type 从 售后/投诉/咨询 中选。"""
    ticket_no = await repository.create_ticket(conversation_id, description, ticket_type)
    return {"ticket_no": ticket_no, "status": "已转人工"}
```

> Fallback(仅当 Step 4 证明 `InjectedToolArg` 在本版本无法从 schema 排除该参):去掉 `Annotated[..., InjectedToolArg]`,改由 Task 9 基础设施在执行前把 `conversation_id` 注入 `args`,并在本工具 docstring 明确「conversation_id 由系统填写,模型不要提供」。二选一,不两条都留。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_tool_ticket.py -v`
Expected: PASS(2 项)。若 `test_conversation_id_hidden_from_model_schema` 失败 → 走 Step 3 的 Fallback。

- [ ] **Step 5: Commit**

```bash
git add app/tools/business.py tests/test_tool_ticket.py
git commit -m "feat(ch02): create_ticket 工具(InjectedToolArg 注入 conversation_id + 写库)"
```

---

## Task 9: 工具注册表 + 基础设施

**Files:**
- Create: `app/tools/registry.py`
- Create: `app/tools/infra.py`
- Test: `tests/test_tools_infra.py`

**Interfaces:**
- Consumes: `app.tools.business` 的 5 个工具、`langchain_core.messages.ToolMessage`
- Produces:
  - `registry.get_all_tools() -> list[BaseTool]`、`registry.get_tool(name) -> BaseTool | None`
  - `registry.NO_RETRY: set[str]`(含 `create_ticket`)、`registry.INJECT_CONVERSATION: set[str]`(含 `create_ticket`,供 Fallback;InjectedToolArg 生效时留空亦可)
  - `infra.ToolRun`(dataclass:`tool_call_id`、`name`、`ok: bool`、`tool_message: ToolMessage`)
  - `infra.execute_tool_call(tool_call: dict, conversation_id: int, timeout: float = 5.0, max_retries: int = 2) -> ToolRun`

- [ ] **Step 1: 写基础设施测试(先红)**

```python
# tests/test_tools_infra.py
import asyncio

import pytest

from app.tools import infra, registry

def test_registry_has_five_tools():
    names = {t.name for t in registry.get_all_tools()}
    assert names == {"query_order", "query_product", "query_logistics", "query_faq", "create_ticket"}

async def test_execute_unknown_tool_returns_error_run():
    run = await infra.execute_tool_call({"name": "nope", "args": {}, "id": "c1"}, conversation_id=1)
    assert run.ok is False and run.tool_call_id == "c1"
    assert "未知工具" in run.tool_message.content

async def test_timeout_then_error(monkeypatch):
    async def slow(_args):
        await asyncio.sleep(1)
    fake = type("T", (), {"name": "query_order", "ainvoke": staticmethod(slow)})()
    monkeypatch.setattr(registry, "get_tool", lambda n: fake)
    run = await infra.execute_tool_call({"name": "query_order", "args": {"order_id": "1"}, "id": "c1"},
                                        conversation_id=1, timeout=0.05, max_retries=1)
    assert run.ok is False and "失败" in run.tool_message.content

async def test_retry_succeeds_on_second_attempt(monkeypatch):
    calls = {"n": 0}
    async def flaky(_args):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("瞬时错误")
        return {"ok": True}
    fake = type("T", (), {"name": "query_faq", "ainvoke": staticmethod(flaky)})()
    monkeypatch.setattr(registry, "get_tool", lambda n: fake)
    run = await infra.execute_tool_call({"name": "query_faq", "args": {"keyword": "x"}, "id": "c1"},
                                        conversation_id=1, timeout=1.0, max_retries=2)
    assert run.ok is True and calls["n"] == 2

async def test_create_ticket_not_retried(monkeypatch):
    calls = {"n": 0}
    async def always_fail(_args):
        calls["n"] += 1
        raise RuntimeError("boom")
    fake = type("T", (), {"name": "create_ticket", "ainvoke": staticmethod(always_fail)})()
    monkeypatch.setattr(registry, "get_tool", lambda n: fake)
    run = await infra.execute_tool_call({"name": "create_ticket", "args": {"description": "x", "ticket_type": "售后"}, "id": "c1"},
                                        conversation_id=1, timeout=1.0, max_retries=2)
    assert run.ok is False and calls["n"] == 1      # 写类工具不重试
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_tools_infra.py -v`
Expected: FAIL(`ModuleNotFoundError: app.tools.registry`)。

- [ ] **Step 3: 写 registry.py**

```python
# app/tools/registry.py
from langchain_core.tools import BaseTool

from app.tools.business import (
    create_ticket,
    query_faq,
    query_logistics,
    query_order,
    query_product,
)

_ALL: list[BaseTool] = [query_order, query_product, query_logistics, query_faq, create_ticket]
_BY_NAME: dict[str, BaseTool] = {t.name: t for t in _ALL}

NO_RETRY: set[str] = {"create_ticket"}            # 写类工具不自动重试
INJECT_CONVERSATION: set[str] = {"create_ticket"}  # 需注入会话主键的工具

def get_all_tools() -> list[BaseTool]:
    return _ALL

def get_tool(name: str) -> BaseTool | None:
    return _BY_NAME.get(name)
```

- [ ] **Step 4: 写 infra.py**

```python
# app/tools/infra.py
import asyncio
import json
import logging
from dataclasses import dataclass

from langchain_core.messages import ToolMessage

from app.tools import registry

logger = logging.getLogger(__name__)

@dataclass
class ToolRun:
    tool_call_id: str
    name: str
    ok: bool
    tool_message: ToolMessage

def _error_run(tc_id: str, name: str, msg: str) -> ToolRun:
    return ToolRun(
        tool_call_id=tc_id,
        name=name,
        ok=False,
        tool_message=ToolMessage(content=f"工具执行失败:{msg}", tool_call_id=tc_id, name=name, status="error"),
    )

async def execute_tool_call(
    tool_call: dict, conversation_id: int, timeout: float = 5.0, max_retries: int = 2
) -> ToolRun:
    name = tool_call["name"]
    tc_id = tool_call["id"]
    args = dict(tool_call.get("args") or {})

    tool = registry.get_tool(name)
    if tool is None:
        return _error_run(tc_id, name, f"未知工具 {name}")

    # Fallback 注入(InjectedToolArg 生效时 INJECT_CONVERSATION 可为空,这段不改变行为)
    if name in registry.INJECT_CONVERSATION:
        args["conversation_id"] = conversation_id

    retries = 0 if name in registry.NO_RETRY else max_retries
    attempt = 0
    while True:
        try:
            result = await asyncio.wait_for(tool.ainvoke(args), timeout=timeout)
            content = json.dumps(result, ensure_ascii=False, default=str)
            return ToolRun(
                tool_call_id=tc_id, name=name, ok=True,
                tool_message=ToolMessage(content=content, tool_call_id=tc_id, name=name),
            )
        except Exception as e:  # noqa: BLE001 - 统一兜底转错误回灌
            attempt += 1
            if attempt > retries:
                logger.exception("工具执行失败 name=%s", name)
                return _error_run(tc_id, name, type(e).__name__)
            await asyncio.sleep(0.2 * attempt)
```

> InjectedToolArg 生效时:`create_ticket` 的 schema 不含 `conversation_id`,但执行仍需该值,故 `INJECT_CONVERSATION` 保留 `create_ticket`——注入的是 injected 参数,LangChain 允许 `ainvoke` 传入 injected args。若走了 Task 8 的 Fallback(去掉 InjectedToolArg),这段逻辑同样适用,无需改动。

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_tools_infra.py -v`
Expected: PASS(5 项)。

- [ ] **Step 6: Commit**

```bash
git add app/tools/registry.py app/tools/infra.py tests/test_tools_infra.py
git commit -m "feat(ch02): 工具注册表 + 执行基础设施(超时/重试/错误回灌/写类不重试)"
```

---

## Task 10: 客服+工具 system prompt

**Files:**
- Modify: `app/core/prompts.py`(追加 `AGENT_SYSTEM` / `AGENT_PROMPT`)
- Test: `tests/test_prompts.py`(追加断言,验证要点在场)

**说明**:纯 Prompt 产出,按用户工作要求 1「TDD 换成标注样例/评估集验证」——真实选工具效果在 Task 14 的 eval 验证。本任务只补一个轻量结构测试(确认关键约束在 prompt 文本中在场),不打模型。

**Interfaces:**
- Consumes: `langchain_core.prompts.ChatPromptTemplate`、`MessagesPlaceholder`(ch01 已用)
- Produces: `AGENT_SYSTEM: str`、`AGENT_PROMPT: ChatPromptTemplate`(占位符名 `history`)

- [ ] **Step 1: 写在场断言测试(先红)**

```python
# tests/test_prompts.py 追加
from app.core.prompts import AGENT_SYSTEM, AGENT_PROMPT

def test_agent_system_covers_tool_principles():
    for kw in ["小喵", "工具", "query_faq", "create_ticket", "不要臆造"]:
        assert kw in AGENT_SYSTEM

def test_agent_prompt_has_history_placeholder():
    assert any(getattr(m, "variable_name", None) == "history"
               for m in AGENT_PROMPT.messages)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: FAIL(`ImportError: AGENT_SYSTEM`)。

- [ ] **Step 3: 追加 AGENT_SYSTEM / AGENT_PROMPT**

`app/core/prompts.py` 末尾追加:
```python
AGENT_SYSTEM = """你是「喵喵优选」电商平台的智能客服「小喵」。你可以调用工具查询真实数据来回答用户。

## 工具使用原则
- 需要订单/商品/物流的具体信息时,调用对应工具查询(query_order / query_product / query_logistics),不要臆造数据。
- 用户咨询政策、规则、操作流程等通用问题时,用 query_faq 按关键词检索常见问答。
- 用户明确要求人工介入、投诉、或问题无法自助解决时,用 create_ticket 建人工工单(工单关联的会话号由系统填写,你不要编造)。
- 能直接回答的闲聊或超出电商客服范围的问题,礼貌回应或引导回购物话题,不必调用工具。
- 拿到工具结果后,用简洁、亲切、专业的中文组织回答;工具查不到时如实告知并给出下一步建议,不要编造。
- 退款/售后时效统一表述为「以平台售后规则为准」,不承诺无法保证的赔偿。"""

AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", AGENT_SYSTEM),
        MessagesPlaceholder("history"),
    ]
)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: PASS(含 ch01 原有用例)。

- [ ] **Step 5: Commit**

```bash
git add app/core/prompts.py tests/test_prompts.py
git commit -m "feat(ch02): 客服+工具 system prompt(工具使用原则)"
```

---

## Task 11: 编排 core/agent.py(共享核心 + 非流式/流式两出口)

**Files:**
- Create: `app/core/agent.py`
- Test: `tests/test_agent_orchestration.py`(非流式)、`tests/test_agent_stream.py`(流式)

**Interfaces:**
- Consumes: `app.core.llm.get_chat_model`、`app.core.prompts.AGENT_SYSTEM`、`app.tools.registry.get_all_tools`、`app.tools.infra.execute_tool_call`、`app.db.repository`、`app.core.memory.trim_history`(ch01 已有)
- Produces:
  - 异常 `ConversationNotFound(Exception)`
  - dataclass `AgentResult`(`conversation_id: int`、`answer: str`、`tool_calls: list[dict]`、`tool_runs: list[ToolRun]`)
  - `async _prepare_turn(user_id, message, conversation_id, model) -> (cid, messages, ai, runs)`(共享编排前半)
  - `async run_agent_turn(user_id, message, conversation_id, model=None) -> AgentResult`(非流式出口)
  - `async stream_agent_turn(user_id, message, conversation_id, model=None) -> AsyncIterator[dict]`(流式出口,产出事件 dict:`{"type":"tool","name"}` / `{"type":"delta","text"}` / `{"type":"done","conversation_id"}`)

**编排契约(硬约束)**:`_prepare_turn` 里 turn1 用 `model.bind_tools(get_all_tools()).ainvoke`(必须非流式才能拿完整 `tool_calls`);两个出口的收敛那步都用**未 bind** 的 `model`(强制单轮)——`run_agent_turn` 用 `ainvoke`、`stream_agent_turn` 用 `astream` 逐 token。跨轮历史只回放 `user` 与「`content` 非空且**无** `tool_calls`」的 assistant(带 tool_calls 的 assistant 常夹带 preamble,不跨轮回放,避免污染续接)。`model` 参数用于测试注入 FakeModel;非流式默认 `get_chat_model()`,流式默认 `get_chat_model(streaming=True)`。

- [ ] **Step 1: 写编排测试(先红,用 Fake 模型不打真实上游)**

FakeModel 需同时支持 `ainvoke`(turn1 + 非流式收敛)与 `astream`(流式收敛),并用 `bind_calls` 计数锁住「收敛不 bind」的硬约束:

```python
# tests/test_agent_orchestration.py
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.core import agent
from app.db import repository as repo

class FakeModel:
    """turn1 返回 scripted[0](可带 tool_calls);非流式收敛返回 scripted[1];
    流式收敛按 stream_tokens 逐个 yield AIMessageChunk。
    bind_calls 记录 bind_tools 次数(收敛不 bind → 应恒为 1)。"""
    def __init__(self, scripted, stream_tokens=None):
        self._scripted = list(scripted)
        self._stream_tokens = list(stream_tokens or [])
        self.bind_calls = 0
        self.invoke_messages = []
        self.astream_messages = []

    def bind_tools(self, tools):
        self.bind_calls += 1
        return self

    async def ainvoke(self, messages):
        self.invoke_messages.append(list(messages))
        return self._scripted.pop(0)

    async def astream(self, messages):
        self.astream_messages.append(list(messages))
        for t in self._stream_tokens:
            yield AIMessageChunk(content=t)

async def test_no_tool_calls_returns_direct_answer(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="你好,喵~ 有什么可以帮您?")])
    res = await agent.run_agent_turn("u1", "你好", None, model=model)
    assert res.answer.startswith("你好") and res.tool_calls == []
    msgs = await repo.list_messages(res.conversation_id)
    assert [m.role for m in msgs] == ["user", "assistant"]

async def test_tool_call_flow_executes_and_converges(db_session_factory, db_clean):
    first = AIMessage(content="", tool_calls=[
        {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])
    model = FakeModel([first, AIMessage(content="您的订单 1001 正在运输中。")])
    res = await agent.run_agent_turn("u1", "订单 1001 到哪了", None, model=model)
    assert res.answer == "您的订单 1001 正在运输中。"
    assert res.tool_calls[0]["name"] == "query_logistics"
    assert res.tool_runs[0].ok is True
    assert model.bind_calls == 1                       # 收敛不 bind
    msgs = await repo.list_messages(res.conversation_id)
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert msgs[2].tool_call_id == "c1"

async def test_continue_conversation_replays_only_final_answers(db_session_factory, db_clean):
    m1 = FakeModel([
        AIMessage(content="我帮您查一下:", tool_calls=[
            {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}]),
        AIMessage(content="订单 1001 已揽件。")])
    cid = (await agent.run_agent_turn("u1", "订单1001到哪了", None, model=m1)).conversation_id
    m2 = FakeModel([AIMessage(content="还有什么可以帮您?")])
    await agent.run_agent_turn("u1", "谢谢", cid, model=m2)
    ai_contents = [m.content for m in m2.invoke_messages[0] if isinstance(m, AIMessage)]
    assert "订单 1001 已揽件。" in ai_contents            # 最终答回放
    assert "我帮您查一下:" not in ai_contents             # tool-calling preamble 不跨轮

async def test_unknown_conversation_raises(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="hi")])
    try:
        await agent.run_agent_turn("u1", "hi", 999999, model=model)
        assert False, "应抛 ConversationNotFound"
    except agent.ConversationNotFound:
        pass
```

流式出口另建 `tests/test_agent_stream.py`(复用上面 FakeModel):断言**有工具**分支事件序列 `tool`(query_logistics)→ 若干 `delta` → `done`(带 conversation_id)、落库 user→assistant→tool→assistant、`bind_calls==1`(astream 未 bind);**无工具**分支无 `tool` 事件、`delta`(turn1 文本)→`done`、`astream_messages==[]`、落库 user→assistant;新建会话回传新 id;非法 id 抛 `ConversationNotFound`。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_agent_orchestration.py -v`
Expected: FAIL(`ModuleNotFoundError: app.core.agent`)。

- [ ] **Step 3: 写 agent.py**

```python
# app/core/agent.py
import asyncio
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import settings
from app.core.llm import get_chat_model
from app.core.memory import trim_history
from app.core.prompts import AGENT_SYSTEM
from app.db import repository
from app.db.models import Message
from app.tools.infra import ToolRun, execute_tool_call
from app.tools.registry import get_all_tools

class ConversationNotFound(Exception):
    pass

@dataclass
class AgentResult:
    conversation_id: int
    answer: str
    tool_calls: list[dict]
    tool_runs: list[ToolRun]

def _text(msg: AIMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    # content 为分块 list 时拼接文本片段
    return "".join(p.get("text", "") for p in msg.content if isinstance(p, dict))

def _build_history(rows: list[Message]) -> list[BaseMessage]:
    """跨轮上下文:只回放 user 与「content 非空且非工具调用」的 assistant。
    带 tool_calls 的 assistant(常夹带 preamble)不跨轮回放,避免污染续接。"""
    out: list[BaseMessage] = [SystemMessage(AGENT_SYSTEM)]
    for m in rows:
        if m.role == "user":
            out.append(HumanMessage(m.content or ""))
        elif m.role == "assistant" and m.content and not m.tool_calls:
            out.append(AIMessage(m.content))
    return trim_history(out, max_tokens=settings.token_budget)

async def _prepare_turn(user_id, message, conversation_id, model):
    """共享编排前半:会话身份 → 落 user → 组装历史 → turn1 bind_tools 定工具 →
    落 assistant → 有工具则并行执行落 tool。返回 (cid, messages, ai, runs)。"""
    if conversation_id is None:
        conversation_id = await repository.create_conversation(user_id)
    elif await repository.get_conversation(conversation_id) is None:
        raise ConversationNotFound(conversation_id)

    await repository.append_message(conversation_id, "user", content=message)
    rows = await repository.list_messages(conversation_id)
    messages = _build_history(rows)

    ai: AIMessage = await model.bind_tools(get_all_tools()).ainvoke(messages)
    await repository.append_message(
        conversation_id, "assistant",
        content=_text(ai) or None, tool_calls=ai.tool_calls or None,
    )

    runs: list[ToolRun] = []
    if ai.tool_calls:
        runs = await asyncio.gather(
            *(execute_tool_call(tc, conversation_id) for tc in ai.tool_calls)
        )
        for r in runs:
            await repository.append_message(
                conversation_id, "tool",
                content=r.tool_message.content, tool_call_id=r.tool_call_id,
            )
    return conversation_id, messages, ai, runs

async def run_agent_turn(user_id, message, conversation_id, model=None) -> AgentResult:
    """非流式出口:一次性返回工具轨迹 + 最终回答(供 /api/agent、eval、单测)。"""
    model = model or get_chat_model()
    conversation_id, messages, ai, runs = await _prepare_turn(
        user_id, message, conversation_id, model)
    if not ai.tool_calls:
        return AgentResult(conversation_id, _text(ai), [], [])
    final: AIMessage = await model.ainvoke(         # 收敛:不 bind_tools
        [*messages, ai, *(r.tool_message for r in runs)])
    await repository.append_message(conversation_id, "assistant", content=_text(final))
    return AgentResult(conversation_id, _text(final), ai.tool_calls, runs)

async def stream_agent_turn(user_id, message, conversation_id, model=None):
    """流式出口:产出事件 dict 供 /api/chat 转 SSE。收敛用 astream 逐 token、不 bind。"""
    model = model or get_chat_model(streaming=True)
    conversation_id, messages, ai, runs = await _prepare_turn(
        user_id, message, conversation_id, model)

    if not ai.tool_calls:
        # 无工具:turn1 文本即答案(assistant 已落库),整块吐出
        yield {"type": "delta", "text": _text(ai)}
        yield {"type": "done", "conversation_id": conversation_id}
        return

    for tc in ai.tool_calls:
        yield {"type": "tool", "name": tc.get("name") or ""}

    chunks: list[str] = []
    async for chunk in model.astream([*messages, ai, *(r.tool_message for r in runs)]):
        text = chunk.content if isinstance(chunk.content, str) else _text(chunk)
        if not text:
            continue
        chunks.append(text)
        yield {"type": "delta", "text": text}

    await repository.append_message(conversation_id, "assistant", content="".join(chunks))
    yield {"type": "done", "conversation_id": conversation_id}
```

> 顶部另需 `from collections.abc import AsyncIterator`。**已知取舍**:无工具的回答整块吐、不打字机——turn1 必须非流式才能判断要不要调工具,判完发现无工具时答案已完整;有工具的最终答仍逐 token 流式。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_agent_orchestration.py tests/test_agent_stream.py -v`
Expected: PASS(非流式 4 项 + 流式 4 项)。

- [ ] **Step 5: Commit**

```bash
git add app/core/agent.py tests/test_agent_orchestration.py tests/test_agent_stream.py
git commit -m "feat(ch02): 编排共享核心 + 非流式/流式两出口(收敛不 bind,流式 astream)"
```

---

## Task 12: 接口层(/api/chat 流式主入口 + /api/agent 程序化出口)

**Files:**
- Modify: `app/schemas/chat.py`(`session_id` → `user_id` + `conversation_id`)
- Modify: `app/api/chat.py`(升级为带工具链的 SSE 流式)
- Create: `app/schemas/agent.py`、`app/api/agent.py`
- Modify: `app/main.py`(挂载 agent_router)
- Test: `tests/test_chat_api.py`(重写为新契约)、`tests/test_agent_api.py`

**Interfaces:**
- Consumes: `app.core.agent.stream_agent_turn`、`app.core.agent.run_agent_turn`、`app.core.agent.ConversationNotFound`
- Produces:
  - `POST /api/chat`(前端主入口,SSE 流式):请求 `ChatRequest(user_id, message, conversation_id?)`;帧 `{"event":"tool","name"}` / `{"delta"}` / `{"event":"done","conversation_id"}` / `event: error` / `[DONE]`
  - `POST /api/agent`(程序化/测试出口,非流式 JSON):请求 `AgentRequest(user_id, message, conversation_id?)`;响应 `AgentResponse(conversation_id, answer, tool_calls[], tool_results[])`

### 12A · /api/chat 升级为带工具链的流式主入口

- [ ] **A1: 改 ChatRequest**

```python
# app/schemas/chat.py
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, description="用户标识,用于建/归属会话")
    message: str = Field(min_length=1, description="用户本轮消息")
    conversation_id: int | None = Field(default=None, description="续接会话;为空则新建,由 done 帧回传")
```

- [ ] **A2: 重写 api/chat.py(消费 stream_agent_turn,事件转 SSE,错误分层转 error 帧)**

```python
# app/api/chat.py
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from sqlalchemy.exc import SQLAlchemyError

from app.core import agent
from app.core.llm import get_chat_model
from app.schemas.chat import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter()

def get_model() -> BaseChatModel:
    return get_chat_model(streaming=True)

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

def _sse_error(message: str) -> AsyncIterator[str]:
    yield "event: error\n"
    yield _sse({"message": message})

@router.post("/api/chat")
async def chat(req: ChatRequest, model: BaseChatModel = Depends(get_model)):
    async def event_stream() -> AsyncIterator[str]:
        try:
            async for ev in agent.stream_agent_turn(
                req.user_id, req.message, req.conversation_id, model=model
            ):
                if ev["type"] == "tool":
                    yield _sse({"event": "tool", "name": ev["name"]})
                elif ev["type"] == "delta":
                    yield _sse({"delta": ev["text"]})
                elif ev["type"] == "done":
                    yield _sse({"event": "done", "conversation_id": ev["conversation_id"]})
        except agent.ConversationNotFound:
            for f in _sse_error("会话不存在"):
                yield f
            return
        except SQLAlchemyError:
            logger.exception("数据库错误 user_id=%s", req.user_id)
            for f in _sse_error("数据库暂时不可用,请稍后重试"):
                yield f
            return
        except Exception:
            logger.exception("聊天工具编排失败 user_id=%s", req.user_id)
            for f in _sse_error("上游模型暂时不可用,请稍后重试"):
                yield f
            return
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

- [ ] **A3: 重写 tests/test_chat_api.py(旧 session_id/内存 store 用例按设计失效)**

用 `httpx.AsyncClient` + `ASGITransport` + `db_session_factory` fixture,`app.dependency_overrides[chat_api.get_model]` 注入 Task 11 的 FakeModel:
- 有工具路径:SSE 含 `{"event":"tool","name":"query_logistics"}` → 拼起来的 `delta` == 期望答 → `{"event":"done","conversation_id":int}` → `[DONE]`。
- 无工具路径:无 tool 帧;`delta` == turn1 文本;落库 user→assistant。
- 非法 conversation_id:出现 `event: error` + `{"message":"会话不存在"}`,且**无** `[DONE]`。
- 空 message:`POST /api/chat` → 422(未进流)。

- [ ] **A4: 跑 chat 测试**

Run: `uv run pytest tests/test_chat_api.py -v`
Expected: PASS(4 项)。

### 12B · /api/agent 非流式程序化/测试出口

- [ ] **Step 1: 写 API 测试(先红,依赖注入替换编排)**

```python
# tests/test_agent_api.py
from fastapi.testclient import TestClient

from app.core import agent
from app.core.agent import AgentResult
from app.main import app
from app.tools.infra import ToolRun
from langchain_core.messages import ToolMessage

def _fake_result():
    tm = ToolMessage(content='{"status":"运输中"}', tool_call_id="c1", name="query_logistics")
    return AgentResult(
        conversation_id=12,
        answer="订单 1001 正在运输中。",
        tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}],
        tool_runs=[ToolRun("c1", "query_logistics", True, tm)],
    )

def test_agent_endpoint_returns_tool_trace(monkeypatch):
    async def fake_run(user_id, message, conversation_id, model=None):
        return _fake_result()
    monkeypatch.setattr(agent, "run_agent_turn", fake_run)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "订单 1001 到哪了"})
    assert r.status_code == 200
    body = r.json()
    assert body["conversation_id"] == 12
    assert body["tool_calls"][0]["name"] == "query_logistics"
    assert body["tool_results"][0]["ok"] is True

def test_agent_endpoint_404_on_unknown_conversation(monkeypatch):
    async def fake_run(*a, **k):
        raise agent.ConversationNotFound(999)
    monkeypatch.setattr(agent, "run_agent_turn", fake_run)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "hi", "conversation_id": 999})
    assert r.status_code == 404

def test_agent_endpoint_422_on_missing_fields():
    client = TestClient(app)
    assert client.post("/api/agent", json={"message": "hi"}).status_code == 422
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_agent_api.py -v`
Expected: FAIL(`404` for `/api/agent` 未挂载 / import 错误)。

- [ ] **Step 3: 写 schemas/agent.py**

```python
# app/schemas/agent.py
from pydantic import BaseModel, Field

class AgentRequest(BaseModel):
    user_id: str = Field(min_length=1, description="用户标识")
    message: str = Field(min_length=1, description="用户本轮消息")
    conversation_id: int | None = Field(default=None, description="续接会话,首轮留空")

class ToolCallView(BaseModel):
    id: str
    name: str
    args: dict

class ToolResultView(BaseModel):
    tool_call_id: str
    name: str
    ok: bool
    content: str

class AgentResponse(BaseModel):
    conversation_id: int
    answer: str
    tool_calls: list[ToolCallView]
    tool_results: list[ToolResultView]
```

- [ ] **Step 4: 写 api/agent.py**

```python
# app/api/agent.py
import logging

from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from app.core import agent
from app.schemas.agent import AgentRequest, AgentResponse, ToolCallView, ToolResultView

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/api/agent", response_model=AgentResponse)
async def run_agent(req: AgentRequest) -> AgentResponse:
    try:
        result = await agent.run_agent_turn(req.user_id, req.message, req.conversation_id)
    except agent.ConversationNotFound:
        raise HTTPException(status_code=404, detail="会话不存在")
    except SQLAlchemyError:
        logger.exception("数据库错误 user_id=%s", req.user_id)
        raise HTTPException(status_code=503, detail="数据库暂时不可用,请稍后重试")
    except Exception:
        logger.exception("agent 编排失败 user_id=%s", req.user_id)
        raise HTTPException(status_code=502, detail="上游模型暂时不可用,请稍后重试")

    return AgentResponse(
        conversation_id=result.conversation_id,
        answer=result.answer,
        tool_calls=[ToolCallView(id=tc["id"], name=tc["name"], args=tc["args"])
                    for tc in result.tool_calls],
        tool_results=[ToolResultView(tool_call_id=r.tool_call_id, name=r.name, ok=r.ok,
                                     content=r.tool_message.content)
                      for r in result.tool_runs],
    )
```

> 注:`agent.run_agent_turn` 用 `agent.` 前缀调用(而非 `from ... import run_agent_turn`),使测试的 `monkeypatch.setattr(agent, "run_agent_turn", ...)` 生效。

- [ ] **Step 5: 挂载到 main.py**

`app/main.py` 修改(加 import 与 include_router,其余不动):
```python
from app.api.agent import router as agent_router
...
app.include_router(agent_router)
```

- [ ] **Step 6: 跑测试确认通过**

Run: `uv run pytest tests/test_agent_api.py -v`
Expected: PASS(3 项)。

- [ ] **Step 7: 全量回归**

Run: `uv run pytest -v`
Expected: ch01 全绿 + ch02 新增全绿,0 error。

- [ ] **Step 8: Commit**

```bash
git add app/schemas/chat.py app/api/chat.py app/schemas/agent.py app/api/agent.py app/main.py \
        tests/test_chat_api.py tests/test_agent_api.py
git commit -m "feat(ch02): /api/chat 流式带工具主入口 + /api/agent 非流式出口(错误分层)"
```

---

## Task 13: 前端聊天页(Vibe Coding)

**Files:**
- Modify: `app/static/index.html`

**说明**:按用户工作要求 1,聊天页改造走 **Vibe Coding**——不套 brainstorm/TDD/code review,直接改,浏览器实测。无单测。

**要点:**
- **请求体**:`streamChat` 的 body 从 `{session_id, message}` 改为 `{user_id, message, conversation_id}`;`user_id` 复用 localStorage 里的稳定标识,`conversation_id` 存 localStorage、`done` 帧回传后写回;「+ 新对话」清空 `conversation_id`(开新会话)。
- **SSE 解析**:新增处理 `{"event":"tool","name"}`(渲染工具轨迹徽章)、`{"event":"done","conversation_id"}`(存会话 id);`delta`/`[DONE]`/`event: error` 沿用。
- **工具轨迹徽章**:bot 气泡文本区上方渲染灰色虚线胶囊「🔧 调用了 <name>」(可多个)。
- **工具期间打字点**:出徽章后、首个 `delta` 到达前,文本区保留流动的三个点(不能清空,否则看着像卡住);首字到达才替换成正文。
- **markdown 渲染**:内联零依赖渲染器(先 HTML 转义防注入,再解析标题/加粗/斜体/行内码/代码块/有序无序列表/表格/引用/分隔线/仅 http(s) 链接),对累积的 raw 文本**逐帧**渲染进文本区;用户气泡仍纯文本。

- [ ] **Step 1: 改造 index.html**(请求体 + 新帧解析 + 徽章 + 打字点 + markdown,见上要点)

- [ ] **Step 2: 浏览器实测**

起服务(`docker compose up -d` / `make seed` / `make dev`),浏览器开 `http://localhost:8000`:
- 「订单 1001 的物流到哪了」→ 徽章「🔧 调用了 query_logistics」+ 工具期间点在流动 + 回答里 markdown(加粗/表格)正确渲染。
- 「你好」→ 无徽章,直接答(无工具整块吐,符合取舍)。
- 「+ 新对话」→ 清空会话;连续追问带同一 conversation_id 续接。

- [ ] **Step 3: Commit**

```bash
git add app/static/index.html
git commit -m "feat(ch02): 聊天页接工具链(工具徽章+markdown渲染+工具期间打字点+会话持久化)"
```

---

## Task 14: 标注样例评估 + 浏览器验收 + demo + dev-notes 留痕

**Files:**
- Create: `scripts/eval_agent.py`
- Create: `scripts/demo_agent.sh`
- Modify: `Makefile`(加 `eval-agent` target)
- Modify: `README.md`(加 ch02 启动/验收说明)
- Modify: `dev-notes/ch02.md`(记录 eval 结果与验收 3 漏召回)

**说明**:模型「选对工具」是模型行为,按用户工作要求 1 走标注样例评估(真实 `glm-5.2`),不做单测。需先 `make dev` 起应用、`docker compose up -d` 起 MySQL、`make seed` 灌 faq。

**Interfaces:**
- Consumes: `POST /api/agent`(真实链路)
- Produces: 每条样例「期望工具 vs 实际选中工具」对照 + 通过率;验收 3 漏召回如实记录

- [ ] **Step 1: 写标注样例 + 评估脚本**

```python
# scripts/eval_agent.py
"""标注样例评估:核对 glm-5.2 是否按预期选中工具。需服务运行中。"""
import asyncio
import json

import httpx

BASE = "http://localhost:8000"

# (用户问法, 期望命中的工具名集合之一;None 表示期望不调用工具)
SAMPLES = [
    ("订单 1001 的物流到哪了", {"query_logistics"}),
    ("退货政策是什么", {"query_faq"}),
    ("邮费是多少", {"query_faq"}),                 # 关注漏召回:答案在「运费怎么算」但字面 LIKE 对不上
    ("iPhone 还有货吗", {"query_product"}),
    ("订单 2002 多少钱", {"query_order"}),
    ("我要投诉,给我登记一下", {"create_ticket"}),
    ("今天天气怎么样", None),                       # 超范围,期望不调工具
]

async def main() -> None:
    passed = 0
    async with httpx.AsyncClient(timeout=60) as client:
        for msg, expect in SAMPLES:
            r = await client.post(f"{BASE}/api/agent", json={"user_id": "eval", "message": msg})
            body = r.json()
            names = {tc["name"] for tc in body["tool_calls"]}
            ok = (not names) if expect is None else bool(names & expect)
            passed += ok
            note = ""
            if msg == "邮费是多少":
                faq = [tr for tr in body["tool_results"] if tr["name"] == "query_faq"]
                miss = faq and '"hits": []' in faq[0]["content"]
                note = f" [漏召回={'是' if miss else '否'}]"
            print(f"{'✅' if ok else '❌'} {msg!r} -> {names or '(未调用)'} 期望={expect}{note}")
            print(f"    answer: {body['answer'][:60]}")
    print(f"\n通过 {passed}/{len(SAMPLES)}")

if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: 加 Makefile / demo 脚本**

`Makefile` 追加(`.PHONY` 补 `eval-agent`):
```makefile
eval-agent:
	uv run python scripts/eval_agent.py
```
`scripts/demo_agent.sh`:
```bash
#!/usr/bin/env bash
# ch02 验收演示:三条验收标准
set -euo pipefail
BASE=http://localhost:8000
echo "== 验收1:订单物流(应选中 query_logistics)=="
curl -s $BASE/api/agent -H 'Content-Type: application/json' \
  -d '{"user_id":"demo","message":"订单 1001 的物流到哪了"}' | python3 -m json.tool
echo "== 验收2:退货政策(query_faq 命中)=="
curl -s $BASE/api/agent -H 'Content-Type: application/json' \
  -d '{"user_id":"demo","message":"退货政策是什么"}' | python3 -m json.tool
echo "== 验收3:换说法(query_faq 漏召回,预期结果)=="
curl -s $BASE/api/agent -H 'Content-Type: application/json' \
  -d '{"user_id":"demo","message":"邮费是多少"}' | python3 -m json.tool
```
Run: `chmod +x scripts/demo_agent.sh`

- [ ] **Step 3: 起服务并跑评估**

Run:
```bash
docker compose up -d && make seed
make dev   # 另开终端:起 :8000 应用
make eval-agent
```
Expected:验收 1/2/6 选对工具;验收 3 打印 `[漏召回=是]`(或如实记录=否)。glm 抖动导致偶发失败时重试并记录(承接 ch01 经验)。

- [ ] **Step 4: 浏览器验收三条(spec 的验收口径)**

浏览器开 `http://localhost:8000`,逐条:
- 验收1「订单 1001 的物流到哪了」→ 气泡出现「🔧 调用了 query_logistics」徽章 + 工具期间点在流动 + 按结果作答(markdown 渲染)。
- 验收2「退货政策是什么」→ 徽章「🔧 调用了 query_faq」+ 答出 7 天无理由。
- 验收3「邮费是多少」→ 徽章 query_faq,但答不出邮费(字面 LIKE 对不上「运费怎么算」)——**预期漏召回**,留 ch03。

程序化 demo(可选,走 /api/agent 一眼看轨迹):`./scripts/demo_agent.sh`。

- [ ] **Step 5: 更新 README 与 dev-notes**

- `README.md` 加 ch02 段:`docker compose up -d` / `make seed` / `make dev` / `make eval-agent` / `./scripts/demo_agent.sh`。
- `dev-notes/ch02.md` 追加「阶段 3 · Task 14」:eval 通过率、验收 3 漏召回实测结论(留给 ch03 向量检索)、glm 抖动情况。

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_agent.py scripts/demo_agent.sh Makefile README.md dev-notes/ch02.md
git commit -m "feat(ch02): 标注样例评估 + 验收 demo(含验收3漏召回记录)"
```

---

## Self-Review(计划对照 spec)

**1. Spec 覆盖检查:**
- 基建(FastAPI+SQLAlchemy 分层、Docker MySQL、4 表建表灌数)→ Task 2/3/4/5 ✅
- 5 个 `@tool`(3 mock + query_faq + create_ticket)→ Task 6/7/8 ✅
- 工具基础设施(注册/校验/超时重试/错误回灌)→ Task 9 ✅
- 编排共享核心 + 单轮收敛 + 结果回灌 → Task 11 ✅
- **工具链长在前端聊天入口**:`/api/chat` 升级为带工具链的 SSE 流式主入口 → Task 12A + Task 13 ✅
- 流式最终答(astream 逐 token)+ 工具轨迹帧/徽章 → Task 11(stream_agent_turn)+ Task 13 ✅
- `/api/agent` 程序化/测试出口(非流式 JSON、工具轨迹)→ Task 12B ✅
- 错误分层(chat:error 帧;agent:404/422/502/503)→ Task 12 ✅
- 会话身份(user_id→conversation_id)+ 落库 → Task 11 ✅
- create_ticket 系统注入 conversation_id → Task 8(InjectedToolArg)✅
- 写类工具不重试 → Task 9 ✅
- glm-5.2 tool-calling 风险闸 → Task 1 ✅
- 验收 1/2/3(浏览器)+ eval → Task 14 ✅
- 测试切分:可单测走 TDD(含流式路径 test_agent_stream)、prompt/选工具走 eval、前端走 Vibe Coding+浏览器 ✅
- `/api/extract` 不动;`SessionStore`/`CUSTOMER_SERVICE_PROMPT` 弃用不删 → 无任务删除它们 ✅

**2. 占位符扫描:** 无 TBD/TODO;每个代码步骤含完整代码;Task 8 的 Fallback 是明确的二选一分支(非占位)。

**3. 类型一致性:** `execute_tool_call`→`ToolRun`(`tool_call_id`/`name`/`ok`/`tool_message`)贯穿 Task 9→11→12;`run_agent_turn`→`AgentResult`(`conversation_id`/`answer`/`tool_calls`/`tool_runs`)贯穿 Task 11→12;`tool_call` dict 键 `name`/`args`/`id` 与 Context7 核对结果一致;repository 函数签名 Task 4 定义、Task 7/8/11 消费一致。

**4. 已知延后点(非缺陷):** `InjectedToolArg` 的 schema 排除行为由 Task 8 Step 1 测试即时验证,附 Fallback,不留悬空。
