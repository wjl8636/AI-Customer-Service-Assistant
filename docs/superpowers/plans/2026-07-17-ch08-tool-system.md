# Ch08 即插即用工具系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task(用户已拍板 inline 模式). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把工具层从写死的内置清单升级为即插即用系统:统一注册中心(内置+MCP)、JSON Schema 校验、读写权限、统一执行引擎(超时/重试/分诊/格式化)、tool_audit_logs 审计、两台自建业务 MCP Server、create_ticket interrupt 确认流。

**Architecture:** 方案 A(spec §2):进程内 ToolSpec 注册表,内置工具 `app/tools/builtin/` 包扫描注册、MCP 工具每轮 `MultiServerMCPClient.get_tools()` 现拉合并;`engine.py` 六段管道(查工具→校验→权限→执行→分诊→格式化+审计);`agent_tools` 顶置 interrupt 做建工单确认。Spec:`docs/superpowers/specs/2026-07-17-ch08-tool-system-design.md`。

**Tech Stack:** MCP 官方 Python SDK(Streamable HTTP)、langchain-mcp-adapters `MultiServerMCPClient`、jsonschema(Draft 2020-12)、既有 FastAPI + SQLAlchemy 2.0 async + LangGraph。

## Global Constraints

- 技术选型定死(用户工作要求 4):MCP Server=官方 Python SDK Streamable HTTP;Client=langchain-mcp-adapters MultiServerMCPClient。走不通停下来问,不许换方案。
- 库 API 动手前先 Context7 核对(工作要求 3)。已核关键结论:新版 SDK 高层类为 `MCPServer`(`mcp.server`,FastMCP 已更名,transport 参数移到 `run()`);adapters 的 transport 键为 `"streamable_http"`;MCP 工具 `args_schema` 即 JSON Schema dict;`MultiServerMCPClient(handle_tool_errors=False)` 让工具错误抛 `ToolException` 由引擎统一分诊;get_tools/工具调用每次新 session(现问现拿天然成立)。**实装版本若与此不符,以实装为准并在 dev-notes 记录。**
- JSON 序列化一律 `ensure_ascii=False`;审计写失败不许反拦工具执行;写操作(create_ticket)恒不自动重试。
- 每任务完成即 commit + 追记 `dev-notes/ch08.md`(不许收尾补记);commit message 结尾带 Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>。
- 测试命令:`uv run pytest <path> -v`;全量 `make test`。测试连真 MySQL 测试库(docker mewhelp-mysql 需在跑)。
- ch05 投诉按钮建单路径(`complaint_reply` 节点 + `POST /api/actions/create-ticket`)一行不动。
- 前端(Task 7)走 Vibe Coding,不套 TDD;prompt 改动用标注样例跑验证(Task 8),不硬凑单测。

---

### Task 1: 审计地基(依赖 + settings + ToolAuditLog + repository + 测试库接线)

**Files:**
- Modify: `pyproject.toml`(uv add)
- Modify: `app/config.py:30-34`(ch07 块后追加 ch08 块)
- Modify: `app/db/models.py`(文件尾加 ToolAuditLog)
- Modify: `app/db/repository.py`(文件尾加 insert_tool_audit)
- Modify: `tests/conftest.py:12-20`(_DDL_FILES/_TABLES)
- Modify: `.env.example`(追加 ch08 条目)
- Test: `tests/db/test_tool_audit.py`(新建)

**Interfaces:**
- Produces: `repository.insert_tool_audit(conversation_id: int | None, tool_call_id: str | None, tool_name: str, tool_source: str, mcp_server: str | None, arguments: dict | None, result_summary: str | None, status: str, error_message: str | None, retry_count: int, duration_ms: int | None) -> None`(engine 在 Task 3 消费);settings 新字段 `mcp_logistics_url / mcp_aftersales_url / tool_default_timeout / mcp_tool_timeout / tool_max_retries / demo_ticket_delay_seconds`。

- [ ] **Step 1: 装依赖**

```bash
uv add mcp langchain-mcp-adapters jsonschema
uv run python -c "import mcp, langchain_mcp_adapters, jsonschema; print(mcp.__file__)"
uv run python -c "from mcp.server import MCPServer; print('MCPServer OK')" || uv run python -c "from mcp.server.fastmcp import FastMCP; print('FastMCP(旧API) OK')"
uv run python -c "from langchain_mcp_adapters.client import MultiServerMCPClient; import inspect; print('handle_tool_errors' in inspect.signature(MultiServerMCPClient.__init__).parameters)"
```
Expected: 三条 import 全过;第三条打出 True(0.3+)或 False(旧版,Task 5 相应去掉该参数并在 dev-notes 记录)。记下 MCPServer/FastMCP 哪个可用,Task 4 按它写。

- [ ] **Step 2: settings + .env.example**

`app/config.py` 在 `summary_model` 行后追加:

```python
    # ch08 工具系统(注册中心 + 执行引擎 + MCP 接入)
    mcp_logistics_url: str = "http://127.0.0.1:8101/mcp"    # 物流 MCP Server
    mcp_aftersales_url: str = "http://127.0.0.1:8102/mcp"   # 售后 MCP Server
    tool_default_timeout: float = 5.0    # 内置工具默认超时(秒)
    mcp_tool_timeout: float = 10.0       # MCP 工具默认超时(走 HTTP,放宽)
    tool_max_retries: int = 2            # 只读工具暂时性故障最大重试次数
    demo_ticket_delay_seconds: float = 0.0  # 验收 6:>0 时 create_ticket 人为变慢(写超时演示)
```

`.env.example` 尾部追加:

```
# ch08 工具系统
MCP_LOGISTICS_URL=http://127.0.0.1:8101/mcp
MCP_AFTERSALES_URL=http://127.0.0.1:8102/mcp
DEMO_TICKET_DELAY_SECONDS=0
```

- [ ] **Step 3: 写失败测试 tests/db/test_tool_audit.py**

```python
from sqlalchemy import select

from app.db import repository
from app.db.models import ToolAuditLog


async def test_insert_tool_audit_minimal(db_session_factory):
    await repository.insert_tool_audit(
        conversation_id=None, tool_call_id=None, tool_name="query_order",
        tool_source="builtin", mcp_server=None, arguments={"order_id": "1001"},
        result_summary="{}", status="成功", error_message=None,
        retry_count=0, duration_ms=12,
    )
    async with db_session_factory() as s:
        row = (await s.execute(select(ToolAuditLog))).scalars().one()
    assert row.tool_name == "query_order" and row.status == "成功"
    assert row.conversation_id is None          # 无会话上下文可空
    assert row.arguments == {"order_id": "1001"}


async def test_insert_tool_audit_all_statuses(db_session_factory):
    for st in ["成功", "失败", "超时", "校验拦下", "权限拒绝"]:
        await repository.insert_tool_audit(
            conversation_id=1, tool_call_id="tc-1", tool_name="create_ticket",
            tool_source="mcp", mcp_server="logistics", arguments=None,
            result_summary=None, status=st, error_message="原因",
            retry_count=2, duration_ms=None,
        )
    async with db_session_factory() as s:
        rows = (await s.execute(select(ToolAuditLog))).scalars().all()
    assert {r.status for r in rows} == {"成功", "失败", "超时", "校验拦下", "权限拒绝"}
```

- [ ] **Step 4: 跑测试确认失败**

Run: `uv run pytest tests/db/test_tool_audit.py -v`
Expected: FAIL(ImportError: ToolAuditLog / insert_tool_audit 不存在)

- [ ] **Step 5: 实现**

`tests/conftest.py` 改两处:

```python
_DDL_FILES = [
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch02-ddl.sql",
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch03-ddl.sql",
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch04-ddl.sql",
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch07-ddl.sql",
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch08-ddl.sql",
]
# 删除顺序:先子表后父表;knowledge_chunks 自引用 FK 靠 FOREIGN_KEY_CHECKS=0 兜
_TABLES = ["low_confidence_questions", "messages", "tickets", "tool_audit_logs",
           "conversations", "faq", "qa_extraction_staging", "knowledge_chunks"]
```

`app/db/models.py` 文件尾追加(对齐 sql/ch08-ddl.sql,无 FK):

```python
class ToolAuditLog(Base):
    __tablename__ = "tool_audit_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)  # 不挂 FK:审计不能被引用约束拦
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(128))
    tool_source: Mapped[str] = mapped_column(Enum("builtin", "mcp"))
    mcp_server: Mapped[str | None] = mapped_column(String(64), nullable=True)
    arguments: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Enum("成功", "失败", "超时", "校验拦下", "权限拒绝"))
    error_message: Mapped[str | None] = mapped_column(String(512), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, server_default="0")
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

`app/db/repository.py` 尾部追加:

```python
# ---- ch08 工具审计 ----


async def insert_tool_audit(
    conversation_id: int | None,
    tool_call_id: str | None,
    tool_name: str,
    tool_source: str,
    mcp_server: str | None,
    arguments: dict | None,
    result_summary: str | None,
    status: str,
    error_message: str | None,
    retry_count: int,
    duration_ms: int | None,
) -> None:
    """工具调用审计落一条。调用方(engine)自行 try/except——审计失败不许反拦工具执行。"""
    async with db.async_session() as s:
        s.add(ToolAuditLog(
            conversation_id=conversation_id, tool_call_id=tool_call_id,
            tool_name=tool_name, tool_source=tool_source, mcp_server=mcp_server,
            arguments=arguments, result_summary=result_summary, status=status,
            error_message=error_message, retry_count=retry_count, duration_ms=duration_ms,
        ))
        await s.commit()
```
(models import 行加 `ToolAuditLog`。)

- [ ] **Step 6: 跑测试确认过 + dev 库应用 DDL**

Run: `uv run pytest tests/db/test_tool_audit.py -v` → PASS
Run: `docker exec -i mewhelp-mysql mysql --default-character-set=utf8mb4 -uroot -proot mewhelp < sql/ch08-ddl.sql && docker exec -i mewhelp-mysql mysql -uroot -proot mewhelp -e "DESC tool_audit_logs;"`
Expected: 表结构 13 列。

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock app/config.py app/db/models.py app/db/repository.py tests/conftest.py tests/db/test_tool_audit.py .env.example
git commit -m "feat(ch08): 审计地基——ToolAuditLog 模型/insert_tool_audit + ch08 settings + 依赖"
```

---

### Task 2: ToolSpec 注册中心 + builtin/ 包(注册即定义,query_logistics 下线)

**Files:**
- Rewrite: `app/tools/registry.py`
- Create: `app/tools/builtin/__init__.py`、`app/tools/builtin/orders.py`、`app/tools/builtin/faq.py`、`app/tools/builtin/tickets.py`、`app/tools/builtin/refunds.py`
- Modify: `app/tools/business.py`(瘦身成 mock 纯函数模块)
- Modify: `app/main.py`(lifespan 里触发 registry 扫描)
- Modify: `tests/test_tools_mock.py`、`tests/test_tool_ticket.py`(import 路径)、`tests/test_tools_infra.py`(仅 test_registry_has_all_tools 改)
- Test: `tests/tools/test_registry.py`(新建)

**Interfaces:**
- Consumes: Task 1 settings(demo_ticket_delay_seconds)。
- Produces(Task 3/5/6 依赖):
  - `@dataclass ToolSpec(name: str, description: str, json_schema: dict, tool: BaseTool, permission: str, source: str, mcp_server: str | None = None, timeout: float | None = None, inject_conversation: bool = False, format_result: Callable[[dict], dict] | None = None)`
  - `registry.register(spec: ToolSpec) -> None`(重名丢弃+warning)
  - `registry.builtin_specs() -> list[ToolSpec]`(惰性触发扫描)
  - `registry.get_builtin_spec(name) -> ToolSpec | None`
  - `registry.permission_for(name: str) -> str`(WRITE_TOOLS 独裁)
  - `registry.spec_from_langchain_tool(tool, *, source, mcp_server=None, ...) -> ToolSpec`(schema 提取三样齐)
  - 过渡兼容(Task 5 删):`get_all_tools()`、`get_tool(name)`、`NO_RETRY`、`INJECT_CONVERSATION`、`TOOL_TIMEOUTS`(旧 infra.py/main_agent 仍在用,本任务不能断)

- [ ] **Step 1: 写失败测试 tests/tools/test_registry.py**

```python
import logging

from app.tools import registry


def test_builtin_scan_registers_five_tools_without_query_logistics():
    names = {s.name for s in registry.builtin_specs()}
    assert names == {"query_order", "query_product", "query_faq", "create_ticket", "submit_refund"}
    # ch08:物流查询由 MCP 接管,内置 query_logistics 下线,不留重名


def test_every_spec_has_three_essentials():
    for s in registry.builtin_specs():
        assert s.name and s.description                      # 名字 + 用途描述
        assert isinstance(s.json_schema, dict) and s.json_schema.get("properties") is not None


def test_permissions_only_trust_our_side():
    by = {s.name: s for s in registry.builtin_specs()}
    assert by["create_ticket"].permission == "write"
    assert all(s.permission == "read" for n, s in by.items() if n != "create_ticket")
    assert registry.permission_for("mcp_random_tool") == "read"   # 未登记 MCP 工具默认只读放行


def test_create_ticket_schema_excludes_injected_conversation_id():
    spec = registry.get_builtin_spec("create_ticket")
    assert spec.inject_conversation is True
    assert "conversation_id" not in spec.json_schema.get("properties", {})  # 模型不见注入参数


def test_duplicate_register_is_dropped_with_warning(caplog):
    spec = registry.builtin_specs()[0]
    dup = registry.ToolSpec(name=spec.name, description="dup", json_schema={"type": "object", "properties": {}},
                            tool=spec.tool, permission="read", source="builtin")
    with caplog.at_level(logging.WARNING):
        registry.register(dup)
    assert registry.get_builtin_spec(spec.name).description != "dup"   # 先到者保留
    assert any("重名" in r.message for r in caplog.records)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/tools/test_registry.py -v`
Expected: FAIL(builtin_specs 不存在)

- [ ] **Step 3: business.py 瘦身 + builtin/ 包实现**

`app/tools/business.py` 只留(其余全迁走;文件头注释改"mock 数据源(纯函数),工具实现见 app/tools/builtin/"):
- `order_snapshot(order_id)`、`list_user_orders(user_id)`(graph.fetch_order 在用,签名不动)

`app/tools/builtin/__init__.py`:空文件(包标记;扫描由 registry 做)。

`app/tools/builtin/orders.py`(query_order/query_product 原样迁入,含 OrderInput/ProductInput,`from app.tools.business import order_snapshot`;尾部注册):

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.tools import registry
from app.tools.business import order_snapshot


class OrderInput(BaseModel):
    order_id: str = Field(description="订单号,例如 1001")


class ProductInput(BaseModel):
    product_name: str = Field(description="商品名称或关键词,例如 猫粮")


@tool(args_schema=OrderInput)
async def query_order(order_id: str) -> dict:
    """查询订单的状态、金额、下单时间、商品名和物流单号(tracking_no)。用于用户询问某个订单情况时。
    要查物流轨迹,需先用本工具拿到订单的 tracking_no,再把它传给 query_logistics。"""
    return order_snapshot(order_id)


@tool(args_schema=ProductInput)
async def query_product(product_name: str) -> dict:
    """查询商品的价格、库存和规格。用于用户咨询某商品是否有货、多少钱时。"""
    import random
    rng = random.Random(f"product:{product_name}")
    return {"product_name": product_name, "price": rng.randint(20, 999),
            "stock": rng.randint(0, 500), "spec": rng.choice(["标准装", "家庭装", "试用装"])}


registry.register(registry.spec_from_langchain_tool(query_order, source="builtin"))
registry.register(registry.spec_from_langchain_tool(query_product, source="builtin"))
```

`app/tools/builtin/faq.py`:query_faq 及 FaqInput 原样迁入(imports: settings/query_understanding/retrieval/selfcheck),尾部:
```python
registry.register(registry.spec_from_langchain_tool(query_faq, source="builtin", timeout=30.0))
# query_faq 走 RAG 管线(改写+混合检索+重排+自评,多次上游往返),远慢于默认 5s
```

`app/tools/builtin/tickets.py`(create_ticket 迁入,加验收 6 写超时演示钩子,描述补 ch08 规矩):

```python
import asyncio
from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg, tool

from app.config import settings
from app.db import repository
from app.tools import registry


@tool
async def create_ticket(
    description: str,
    ticket_type: Literal["售后", "投诉", "咨询"],
    conversation_id: Annotated[int, InjectedToolArg],
) -> dict:
    """创建人工工单。仅当用户明确要求建工单/要求人工跟进时才调用;调用前必须确认 description
    (问题描述)已从用户处问清,信息不足时先向用户追问,严禁编造或用占位文本。
    ticket_type 从 售后/投诉/咨询 中选;工单关联的会话号由系统注入,你不要传。"""
    if settings.demo_ticket_delay_seconds > 0:      # 验收 6:环境变量注入写操作超时演示
        await asyncio.sleep(settings.demo_ticket_delay_seconds)
    ticket_no = await repository.create_ticket(conversation_id, description, ticket_type)
    return {"ticket_no": ticket_no, "status": "已转人工"}


registry.register(registry.spec_from_langchain_tool(
    create_ticket, source="builtin", inject_conversation=True))
```

`app/tools/builtin/refunds.py`:submit_refund + RefundInput 原样迁入,`registry.register(registry.spec_from_langchain_tool(submit_refund, source="builtin"))`。

- [ ] **Step 4: registry.py 重造**

```python
"""ch08 工具注册中心:内置(启动扫描 builtin/ 包)+ MCP(mcp_client 现拉)统一登记 ToolSpec。
三样必齐:工具名、用途描述、JSON Schema 参数定义。权限只认我们侧 WRITE_TOOLS,不看 Server 声明。"""
import importlib
import logging
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass, field

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# 权限我们侧独裁:写清单按名字定;未登记(含所有 MCP)工具一律按只读放行。
# 生产环境接不受信 Server 时应默认拒绝未知写操作;本章两台自建 Server 都是查询类,只读放行足够。
WRITE_TOOLS: set[str] = {"create_ticket"}


@dataclass
class ToolSpec:
    name: str
    description: str
    json_schema: dict            # 参数 JSON Schema(校验用;模型可见口径,不含注入参数)
    tool: BaseTool
    permission: str              # "read" | "write"
    source: str                  # "builtin" | "mcp"
    mcp_server: str | None = None
    timeout: float | None = None            # None → engine 按来源取默认
    inject_conversation: bool = False       # 执行前注入 conversation_id
    format_result: Callable[[dict], dict] | None = None  # 挑字段/枚举翻人话,None 透传


_BUILTIN: dict[str, ToolSpec] = {}
_scanned = False


def permission_for(name: str) -> str:
    return "write" if name in WRITE_TOOLS else "read"


def _json_schema_of(tool: BaseTool) -> dict:
    """三样齐的 schema 口径:MCP 工具 args_schema 本就是 dict;内置 pydantic 模型用
    tool_call_schema(排除 InjectedToolArg,模型可见口径)转 JSON Schema。"""
    raw = getattr(tool, "args_schema", None)
    if isinstance(raw, dict):
        return raw
    tcs = getattr(tool, "tool_call_schema", None) or raw
    return tcs.model_json_schema()


def spec_from_langchain_tool(tool: BaseTool, *, source: str, mcp_server: str | None = None,
                             timeout: float | None = None, inject_conversation: bool = False,
                             format_result: Callable | None = None) -> ToolSpec:
    return ToolSpec(name=tool.name, description=tool.description or "",
                    json_schema=_json_schema_of(tool), tool=tool,
                    permission=permission_for(tool.name), source=source, mcp_server=mcp_server,
                    timeout=timeout, inject_conversation=inject_conversation,
                    format_result=format_result)


def register(spec: ToolSpec) -> None:
    if spec.name in _BUILTIN:
        logger.warning("工具重名,丢弃后注册者 name=%s(先到者保留)", spec.name)
        return
    _BUILTIN[spec.name] = spec


def scan_builtin() -> None:
    """服务启动(lifespan)时调用:导入 builtin/ 包全部模块,模块 import 即注册。幂等。"""
    global _scanned
    if _scanned:
        return
    _scanned = True
    from app.tools import builtin as pkg
    for m in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"{pkg.__name__}.{m.name}")
    logger.info("内置工具注册完成:%s", sorted(_BUILTIN))


def builtin_specs() -> list[ToolSpec]:
    scan_builtin()
    return list(_BUILTIN.values())


def get_builtin_spec(name: str) -> ToolSpec | None:
    scan_builtin()
    return _BUILTIN.get(name)


# ---- 过渡兼容(旧 infra.py / main_agent 仍在用;Task 5 接线 MCP 后删除)----
NO_RETRY: set[str] = {"create_ticket", "query_faq", "submit_refund"}
INJECT_CONVERSATION: set[str] = {"create_ticket"}
TOOL_TIMEOUTS: dict[str, float] = {"query_faq": 30.0}


def get_all_tools() -> list[BaseTool]:
    return [s.tool for s in builtin_specs()]


def get_tool(name: str) -> BaseTool | None:
    spec = get_builtin_spec(name)
    return spec.tool if spec else None
```

`app/main.py` lifespan 里(`await runtime.init_graph()` 前一行)加:

```python
    from app.tools import registry as tool_registry
    tool_registry.scan_builtin()   # ch08:内置工具服务启动时登记
```

- [ ] **Step 5: 更新受影响旧测试**

- `tests/test_tools_infra.py::test_registry_has_all_tools`:期望集合去掉 `"query_logistics"`,断言注释改"ch08:物流由 MCP 接管"。
- `tests/test_tools_mock.py`:`from app.tools.business import ...` 中工具相关改 `from app.tools.builtin.orders import query_order, query_product`;涉及 query_logistics 的测试删除(该工具下线,MCP 侧 Task 4 另测)。
- `tests/test_tool_ticket.py`:`business.create_ticket` → `from app.tools.builtin.tickets import create_ticket`。
- 其他文件先 `grep -rn "tools import business\|tools.business\|query_logistics" app tests scripts` 逐一核(nodes.py 的 order_snapshot/list_user_orders 引用不动;`scripts/smoke_toolcall.py` 等按同样方式改 import 或注明 ch08 下线)。

- [ ] **Step 6: 跑测试**

Run: `uv run pytest tests/tools/test_registry.py tests/test_tools_infra.py tests/test_tools_mock.py tests/test_tool_ticket.py -v`
Expected: 全 PASS。再 `make test` 全量确认无回归(Milvus 不在跑或上游不可用导致的既有跳过/失败按 ch07 基线对照)。

- [ ] **Step 7: Commit**

```bash
git add -A app/tools tests app/main.py scripts
git commit -m "feat(ch08): ToolSpec 注册中心+builtin 包注册即定义;query_logistics 下线"
```

---

### Task 3: 统一执行引擎 engine.py(六段管道 + 审计)

**Files:**
- Create: `app/tools/engine.py`
- Test: `tests/tools/test_engine.py`(新建;旧 `tests/test_tools_infra.py` 的执行类用例迁入改造,infra.py 本任务保留、Task 5 删)

**Interfaces:**
- Consumes: Task 2 `ToolSpec`;Task 1 `repository.insert_tool_audit`、settings。
- Produces(Task 5/6 依赖):
  - `@dataclass ToolRun(tool_call_id: str, name: str, ok: bool, tool_message: ToolMessage, status: str, retry_count: int = 0, duration_ms: int = 0)`
  - `async engine.execute_tool_call(tool_call: dict, conversation_id: int, specs: dict[str, ToolSpec], *, confirmed: bool = False, deny_note: str | None = None) -> ToolRun`
  - `engine.validate_args(spec: ToolSpec, args: dict) -> str | None`(None=通过;agent_tools 判"要不要 interrupt"复用)

- [ ] **Step 1: 写失败测试 tests/tools/test_engine.py**

```python
import asyncio

import pytest

from app.tools import engine
from app.tools.registry import ToolSpec

SCHEMA = {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}


def _spec(name="query_order", *, permission="read", source="builtin", tool=None,
          schema=SCHEMA, timeout=None, inject=False, fmt=None):
    return ToolSpec(name=name, description="测试工具", json_schema=schema, tool=tool,
                    permission=permission, source=source, timeout=timeout,
                    inject_conversation=inject, format_result=fmt)


def _tool(fn, name="query_order"):
    return type("T", (), {"name": name, "ainvoke": staticmethod(fn)})()


@pytest.fixture()
def audits(monkeypatch):
    rows = []
    async def fake_audit(**kw):
        rows.append(kw)
    monkeypatch.setattr(engine.repository, "insert_tool_audit",
                        lambda *a, **kw: fake_audit(**kw) if kw else fake_audit(
                            **dict(zip(["conversation_id", "tool_call_id", "tool_name", "tool_source",
                                        "mcp_server", "arguments", "result_summary", "status",
                                        "error_message", "retry_count", "duration_ms"], a))))
    return rows


async def test_unknown_tool(audits):
    run = await engine.execute_tool_call({"name": "nope", "args": {}, "id": "c1"}, 1, {})
    assert run.ok is False and "未知工具" in run.tool_message.content
    assert audits[-1]["status"] == "失败"


async def test_validation_blocks_and_feeds_back(audits):
    async def boom(_): raise AssertionError("校验不过不许执行")
    spec = _spec(tool=_tool(boom))
    run = await engine.execute_tool_call({"name": "query_order", "args": {}, "id": "c1"}, 1,
                                         {"query_order": spec})
    assert run.ok is False and run.status == "校验拦下"
    assert "参数校验未通过" in run.tool_message.content and run.tool_message.status == "error"
    assert audits[-1]["status"] == "校验拦下" and audits[-1]["retry_count"] == 0


async def test_write_without_confirmation_denied(audits):
    async def boom(_): raise AssertionError("未确认不许执行")
    schema = {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]}
    spec = _spec("create_ticket", permission="write", tool=_tool(boom, "create_ticket"), schema=schema)
    run = await engine.execute_tool_call({"name": "create_ticket", "args": {"description": "x"}, "id": "c1"},
                                         1, {"create_ticket": spec})
    assert run.ok is False and run.status == "权限拒绝"
    assert audits[-1]["status"] == "权限拒绝"


async def test_write_confirmed_executes_and_not_retried(audits):
    calls = {"n": 0}
    async def flaky(_):
        calls["n"] += 1
        raise RuntimeError("boom")
    schema = {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]}
    spec = _spec("create_ticket", permission="write", tool=_tool(flaky, "create_ticket"), schema=schema)
    run = await engine.execute_tool_call({"name": "create_ticket", "args": {"description": "x"}, "id": "c1"},
                                         1, {"create_ticket": spec}, confirmed=True)
    assert run.ok is False and calls["n"] == 1 and run.retry_count == 0   # 写操作恒不重试


async def test_transient_timeout_retries_then_gives_up(audits):
    async def slow(_):
        await asyncio.sleep(1)
    spec = _spec(tool=_tool(slow), timeout=0.05)
    run = await engine.execute_tool_call({"name": "query_order", "args": {"order_id": "1"}, "id": "c1"},
                                         1, {"query_order": spec})
    assert run.ok is False and run.status == "超时"
    assert run.retry_count == 2                      # settings.tool_max_retries 默认 2
    assert audits[-1]["status"] == "超时" and audits[-1]["retry_count"] == 2
    assert audits[-1]["duration_ms"] is not None


async def test_business_error_not_retried(audits):
    calls = {"n": 0}
    async def fail(_):
        calls["n"] += 1
        raise ValueError("业务错")                    # 非暂时性 → 不重试
    spec = _spec(tool=_tool(fail))
    run = await engine.execute_tool_call({"name": "query_order", "args": {"order_id": "1"}, "id": "c1"},
                                         1, {"query_order": spec})
    assert run.ok is False and calls["n"] == 1 and "工具暂时不可用" in run.tool_message.content


async def test_retry_succeeds_second_attempt(audits):
    calls = {"n": 0}
    async def flaky(_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("网络抖动")
        return {"order_id": "1", "status": "已发货"}
    spec = _spec(tool=_tool(flaky))
    run = await engine.execute_tool_call({"name": "query_order", "args": {"order_id": "1"}, "id": "c1"},
                                         1, {"query_order": spec})
    assert run.ok is True and calls["n"] == 2 and run.retry_count == 1
    assert "已发货" in run.tool_message.content       # ensure_ascii=False,中文不转义
    assert audits[-1]["status"] == "成功" and audits[-1]["retry_count"] == 1


async def test_format_result_hook_translates_enum(audits):
    async def ok(_):
        return {"tracking_no": "SF1", "status_code": "IN_TRANSIT", "internal_ref": "x9"}
    def fmt(d):
        return {"tracking_no": d["tracking_no"],
                "status": {"IN_TRANSIT": "运输中"}.get(d.get("status_code"), d.get("status_code"))}
    spec = _spec("query_logistics", source="mcp", tool=_tool(ok, "query_logistics"),
                 schema={"type": "object", "properties": {"tracking_no": {"type": "string"}},
                         "required": ["tracking_no"]}, fmt=fmt)
    run = await engine.execute_tool_call({"name": "query_logistics", "args": {"tracking_no": "SF1"}, "id": "c1"},
                                         1, {"query_logistics": spec})
    assert "运输中" in run.tool_message.content and "internal_ref" not in run.tool_message.content
    assert audits[-1]["tool_source"] == "mcp"


async def test_audit_failure_never_blocks_execution(monkeypatch):
    async def audit_boom(*a, **kw):
        raise RuntimeError("审计库挂了")
    monkeypatch.setattr(engine.repository, "insert_tool_audit", audit_boom)
    async def ok(_):
        return {"order_id": "1"}
    spec = _spec(tool=_tool(ok))
    run = await engine.execute_tool_call({"name": "query_order", "args": {"order_id": "1"}, "id": "c1"},
                                         1, {"query_order": spec})
    assert run.ok is True                              # 审计失败不反拦


async def test_inject_conversation_after_validation(audits):
    seen = {}
    async def ok(args):
        seen.update(args)
        return {"ticket_no": "T1"}
    schema = {"type": "object", "properties": {"description": {"type": "string"}},
              "required": ["description"], "additionalProperties": False}
    spec = _spec("create_ticket", permission="write", tool=_tool(ok, "create_ticket"),
                 schema=schema, inject=True)
    run = await engine.execute_tool_call({"name": "create_ticket", "args": {"description": "x"}, "id": "c1"},
                                         7, {"create_ticket": spec}, confirmed=True)
    assert run.ok is True and seen.get("conversation_id") == 7   # 校验后注入,schema 不含它也不冲突
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/tools/test_engine.py -v` → FAIL(engine 不存在)

- [ ] **Step 3: 实现 app/tools/engine.py**

```python
"""ch08 统一执行引擎:所有工具调用的唯一通道。
管道:查工具 → JSON Schema 校验 → 权限门 → 执行(超时/重试)→ 分诊 → 格式化 + 审计。
坏消息如实回灌模型;审计写失败只 log,绝不反拦工具执行。"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass

import httpx
from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException

from app.config import settings
from app.db import repository
from app.tools.registry import ToolSpec

logger = logging.getLogger(__name__)

# 暂时性故障(值得重试):超时、网络抖动。业务异常/ToolException 不在列——重试不解决问题。
_TRANSIENT = (asyncio.TimeoutError, TimeoutError, ConnectionError, httpx.TransportError)
_SUMMARY_LIMIT = 500


@dataclass
class ToolRun:
    tool_call_id: str
    name: str
    ok: bool
    tool_message: ToolMessage
    status: str               # 成功 / 失败 / 超时 / 校验拦下 / 权限拒绝
    retry_count: int = 0
    duration_ms: int = 0


def validate_args(spec: ToolSpec, args: dict) -> str | None:
    """按 JSON Schema 校验模型给的参数;返回人类可读错误(None=通过)。agent_tools 复用它
    判「create_ticket 参数齐不齐、要不要弹确认卡」。"""
    err = best_match(Draft202012Validator(spec.json_schema).iter_errors(args))
    if err is None:
        return None
    where = f"(字段 {err.json_path})" if err.json_path != "$" else ""
    return f"{err.message}{where}"


def _timeout_of(spec: ToolSpec) -> float:
    if spec.timeout is not None:
        return spec.timeout
    return settings.mcp_tool_timeout if spec.source == "mcp" else settings.tool_default_timeout


def _summarize(content: str) -> str:
    return content if len(content) <= _SUMMARY_LIMIT else content[:_SUMMARY_LIMIT] + "…(截断)"


def _format_content(spec: ToolSpec, result) -> str:
    """结果格式化:MCP 工具常返回 JSON 文本,先解回 dict;format_result 钩子挑字段/枚举翻人话;
    统一 ensure_ascii=False 序列化(中文不转义)。"""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (json.JSONDecodeError, ValueError):
            return result                      # 纯文本结果原样回灌
    if spec.format_result is not None and isinstance(result, dict):
        result = spec.format_result(result)
    return json.dumps(result, ensure_ascii=False, default=str)


async def _audit(conversation_id, tool_call_id, name, spec: ToolSpec | None, args,
                 result_summary, status, error_message, retry_count, duration_ms) -> None:
    try:
        await repository.insert_tool_audit(
            conversation_id=conversation_id or None, tool_call_id=tool_call_id or None,
            tool_name=name,
            tool_source=(spec.source if spec else "builtin"),   # 未知工具无来源,按缺省 builtin 落
            mcp_server=(spec.mcp_server if spec else None),
            arguments=args or None, result_summary=result_summary, status=status,
            error_message=error_message, retry_count=retry_count, duration_ms=duration_ms,
        )
    except Exception:  # noqa: BLE001 —— 审计失败不许反拦工具执行
        logger.exception("审计写入失败(不影响工具执行) tool=%s status=%s", name, status)


async def execute_tool_call(
    tool_call: dict, conversation_id: int, specs: dict[str, ToolSpec],
    *, confirmed: bool = False, deny_note: str | None = None,
) -> ToolRun:
    name = tool_call.get("name") or ""
    tc_id = tool_call.get("id") or ""
    args = dict(tool_call.get("args") or {})
    started = time.monotonic()

    def _run(ok: bool, content: str, status: str, retry_count: int = 0, *, msg_status=None) -> ToolRun:
        return ToolRun(tool_call_id=tc_id, name=name, ok=ok, status=status, retry_count=retry_count,
                       duration_ms=int((time.monotonic() - started) * 1000),
                       tool_message=ToolMessage(content=content, tool_call_id=tc_id, name=name or "unknown",
                                                status=msg_status or ("success" if ok else "error")))

    spec = specs.get(name)
    # ① 查工具
    if spec is None:
        run = _run(False, f"工具执行失败:未知工具 {name}", "失败")
        await _audit(conversation_id, tc_id, name or "unknown", None, args, None,
                     "失败", f"未知工具 {name}", 0, run.duration_ms)
        return run

    # ② JSON Schema 校验:拦下不抛异常,错误说明回灌模型(让它追问用户或重组调用)
    verr = validate_args(spec, args)
    if verr is not None:
        run = _run(False, f"参数校验未通过:{verr}。请修正参数重新调用;缺少的信息请先向用户追问,不要编造。",
                   "校验拦下")
        await _audit(conversation_id, tc_id, name, spec, args, None, "校验拦下", verr, 0, run.duration_ms)
        return run

    # ③ 权限门:写操作必须带确认令牌(interrupt 确认后由 agent_tools 传入),模型绕不过
    if spec.permission == "write" and not confirmed:
        note = deny_note or "该写操作需要用户确认,未确认前拒绝执行。请勿再次发起,除非用户明确要求。"
        run = _run(False, f"建工单未执行:{note}", "权限拒绝")
        await _audit(conversation_id, tc_id, name, spec, args, None, "权限拒绝", note, 0, run.duration_ms)
        return run

    # ④ 执行:超时 + 暂时性故障重试(写操作恒不重试——超时未必没执行,重复执行比失败更糟)
    if spec.inject_conversation:
        args["conversation_id"] = conversation_id      # 校验后注入(schema 不含注入参数)
    retries = 0 if spec.permission == "write" else settings.tool_max_retries
    tool_timeout = _timeout_of(spec)
    attempt = 0
    while True:
        try:
            result = await asyncio.wait_for(spec.tool.ainvoke(args), timeout=tool_timeout)
            content = _format_content(spec, result)
            run = _run(True, content, "成功", attempt)
            await _audit(conversation_id, tc_id, name, spec, args, _summarize(content),
                         "成功", None, attempt, run.duration_ms)
            return run
        except _TRANSIENT as e:
            if attempt < retries:
                attempt += 1
                logger.warning("工具暂时性故障,第 %s 次重试 name=%s err=%s", attempt, name, type(e).__name__)
                await asyncio.sleep(0.2 * attempt)
                continue
            is_timeout = isinstance(e, (asyncio.TimeoutError, TimeoutError))
            status = "超时" if is_timeout else "失败"
            # ⑤ 分诊:真故障如实回灌,不装没事
            run = _run(False, f"工具暂时不可用:{'执行超时' if is_timeout else type(e).__name__},请稍后再试或如实告知用户。",
                       status, attempt)
            await _audit(conversation_id, tc_id, name, spec, args, None, status,
                         type(e).__name__, attempt, run.duration_ms)
            return run
        except (ToolException, Exception) as e:  # noqa: BLE001 业务/未知异常:不重试,如实回灌
            logger.exception("工具执行失败 name=%s", name)
            run = _run(False, f"工具暂时不可用:{type(e).__name__}。请如实告知用户,不要编造结果。", "失败", attempt)
            await _audit(conversation_id, tc_id, name, spec, args, None, "失败",
                         f"{type(e).__name__}: {e}"[:500], attempt, run.duration_ms)
            return run
```
(注:测试里 audits fixture 直接 monkeypatch `engine.repository.insert_tool_audit` 为 kwargs 版即可,实现按关键字传参,Step 1 的 fixture 简化为只收 kwargs——执行时以先写的测试为准微调,保持断言不变。)

- [ ] **Step 4: 跑测试**

Run: `uv run pytest tests/tools/test_engine.py -v` → 全 PASS
Run: `make test` → 无回归

- [ ] **Step 5: Commit**

```bash
git add app/tools/engine.py tests/tools/test_engine.py
git commit -m "feat(ch08): 统一执行引擎——校验/权限/超时重试/分诊/格式化/审计六段管道"
```

---

### Task 4: 两台业务 MCP Server(独立进程,Streamable HTTP)+ 起停脚本

**Files:**
- Create: `mcp_servers/__init__.py`(空)、`mcp_servers/logistics_server.py`、`mcp_servers/aftersales_server.py`
- Modify: `Makefile`(mcp-up/mcp-down)、`scripts/dev.sh`(uvicorn 前拉起 MCP)
- Test: `tests/tools/test_mcp_servers.py`(集成:子进程起 Server → 官方 client 列工具+真调)

**Interfaces:**
- Produces: `http://127.0.0.1:8101/mcp`(工具 query_logistics)、`http://127.0.0.1:8102/mcp`(query_warranty / query_return_status);mock 返回含内部枚举码字段 `status_code`(Task 5 formatter 翻人话);env `MOCK_DELAY_SECONDS` 注入延迟(验收 6)、`PORT` 覆盖端口。

- [ ] **Step 1: 动手前按 Task 1 Step 1 记下的 SDK API 写 Server(两代 API 以实装为准)**

`mcp_servers/logistics_server.py`:

```python
"""物流 MCP Server(ch08 自建,mock 数据,不接真实系统、不建表)。
独立进程:uv run python mcp_servers/logistics_server.py
验收 6:MOCK_DELAY_SECONDS=8 起进程 → 每次工具调用慢 8 秒,客服侧看超时/重试/审计。"""
import asyncio
import os
import random
from typing import Annotated

from pydantic import Field

try:                                    # 新版 SDK(FastMCP 已更名 MCPServer)
    from mcp.server import MCPServer
except ImportError:                     # 旧版兜底
    from mcp.server.fastmcp import FastMCP as MCPServer

mcp = MCPServer("logistics")

_DELAY = float(os.environ.get("MOCK_DELAY_SECONDS", "0"))
_STATUS_CODES = ["PICKED_UP", "IN_TRANSIT", "DELIVERING", "DELIVERED"]  # 内部枚举码,client 侧翻人话
_CITIES = ["深圳", "广州", "杭州", "上海", "成都"]


@mcp.tool()
async def query_logistics(
    tracking_no: Annotated[str, Field(description="物流单号(形如 SF 开头),需先用 query_order 查订单拿到该单号")],
) -> dict:
    """用物流单号(tracking_no)查询物流状态、当前位置和轨迹。用于用户询问物流/快递到哪了时。
    物流单号不是订单号,需先用 query_order 查订单拿到 tracking_no,再调用本工具。"""
    if _DELAY > 0:
        await asyncio.sleep(_DELAY)
    rng = random.Random(f"logistics:{tracking_no}")      # 种子固定 → 同单号稳定
    code = rng.choice(_STATUS_CODES)
    city = rng.choice(_CITIES)
    return {
        "tracking_no": tracking_no,
        "status_code": code,                              # 内部枚举码,不翻译,交 client 侧治理
        "current_city": city,
        "trace": [f"{city}分拨中心 已发出", f"内部状态码:{code}"],
        "carrier_code": "SF-EXP-01",                      # 内部承运商编码(回答用不上,client 侧应剔除)
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http",
            host="127.0.0.1", port=int(os.environ.get("PORT", "8101")))
```
(若实装为旧 FastMCP:host/port 移到构造器 `MCPServer("logistics", host=..., port=...)`,`run(transport="streamable-http")`;哪条路生效记 dev-notes。)

`mcp_servers/aftersales_server.py` 同构(:8102),两个工具:

```python
@mcp.tool()
async def query_warranty(
    order_id: Annotated[str, Field(description="订单号,例如 1001")],
) -> dict:
    """查询某订单商品是否在保修期内(在保状态、到期日)。用于用户问保修/在保时。"""
    if _DELAY > 0:
        await asyncio.sleep(_DELAY)
    rng = random.Random(f"warranty:{order_id}")
    code = rng.choice(["IN_WARRANTY", "EXPIRED"])
    return {"order_id": order_id, "warranty_code": code,
            "warranty_until": f"2026-{rng.randint(8, 12):02d}-{rng.randint(1, 28):02d}",
            "policy_ref": "AS-POLICY-07"}                 # 内部政策编号,回答用不上


@mcp.tool()
async def query_return_status(
    order_id: Annotated[str, Field(description="订单号,例如 1001")],
) -> dict:
    """查询某订单的退货进度(审核中/退货中/已退款/无退货记录)。用于用户问退货到哪一步了。"""
    if _DELAY > 0:
        await asyncio.sleep(_DELAY)
    rng = random.Random(f"return:{order_id}")
    code = rng.choice(["AUDITING", "RETURNING", "REFUNDED", "NONE"])
    return {"order_id": order_id, "return_code": code,
            "updated_at": f"2026-07-{rng.randint(1, 16):02d} 10:00"}
```

- [ ] **Step 2: Makefile + dev.sh**

Makefile 追加(.PHONY 行加 mcp-up mcp-down):

```makefile
# ch08: 两台业务 MCP Server 起停(独立进程,Streamable HTTP :8101/:8102)
mcp-up:
	@mkdir -p log data
	@nohup uv run python mcp_servers/logistics_server.py  > log/mcp-logistics.log 2>&1 & echo $$! > data/mcp-logistics.pid
	@nohup uv run python mcp_servers/aftersales_server.py > log/mcp-aftersales.log 2>&1 & echo $$! > data/mcp-aftersales.pid
	@sleep 1 && echo "MCP Servers 已拉起: logistics=:8101 aftersales=:8102(pid 见 data/*.pid)"

mcp-down:
	-@kill `cat data/mcp-logistics.pid 2>/dev/null` 2>/dev/null; rm -f data/mcp-logistics.pid
	-@kill `cat data/mcp-aftersales.pid 2>/dev/null` 2>/dev/null; rm -f data/mcp-aftersales.pid
	@echo "MCP Servers 已停"
```

`scripts/dev.sh` 在 `uv run uvicorn app.main:app --port 8000` 前加:

```bash
make mcp-up   # ch08: 物流/售后 MCP Server(独立进程)
```

- [ ] **Step 3: 写集成测试 tests/tools/test_mcp_servers.py(先写,伴随实现调通)**

```python
"""ch08 MCP Server 集成测试:真起子进程 → adapters client 列工具 + 真调。
连不上视为环境问题直接 fail(两台 Server 是本章交付物,不 skip)。"""
import os
import subprocess
import sys
import time

import pytest
import pytest_asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient


@pytest.fixture(scope="module")
def mcp_procs():
    env = {**os.environ, "PORT": "18101"}
    p1 = subprocess.Popen([sys.executable, "mcp_servers/logistics_server.py"], env=env)
    env2 = {**os.environ, "PORT": "18102"}
    p2 = subprocess.Popen([sys.executable, "mcp_servers/aftersales_server.py"], env=env2)
    time.sleep(2.0)                     # 等 HTTP 就绪(执行时如竞态,改轮询探活)
    yield
    p1.terminate(); p2.terminate()
    p1.wait(timeout=5); p2.wait(timeout=5)


def _client() -> MultiServerMCPClient:
    return MultiServerMCPClient({
        "logistics": {"transport": "streamable_http", "url": "http://127.0.0.1:18101/mcp"},
        "aftersales": {"transport": "streamable_http", "url": "http://127.0.0.1:18102/mcp"},
    }, handle_tool_errors=False)


async def test_list_tools_three_essentials(mcp_procs):
    tools = await _client().get_tools()
    by = {t.name: t for t in tools}
    assert set(by) == {"query_logistics", "query_warranty", "query_return_status"}
    for t in by.values():
        assert t.description                                   # 用途描述
        schema = t.args_schema if isinstance(t.args_schema, dict) else t.args_schema.model_json_schema()
        assert schema.get("properties")                        # JSON Schema 参数定义


async def test_invoke_logistics_stable_mock(mcp_procs):
    tools = {t.name: t for t in await _client().get_tools(server_name="logistics")}
    r1 = await tools["query_logistics"].ainvoke({"tracking_no": "SF123"})
    r2 = await tools["query_logistics"].ainvoke({"tracking_no": "SF123"})
    assert r1 == r2 and "status_code" in str(r1)               # 种子稳定 + 内部枚举码在
```

- [ ] **Step 4: 跑集成测试**

Run: `uv run pytest tests/tools/test_mcp_servers.py -v` → 2 PASS(若因启动竞态 flaky,把 sleep 换成对 `http://127.0.0.1:1810X/mcp` 的轮询探活)
Run: `make mcp-up && sleep 1 && make mcp-down` → 起停正常,log/ 下有两份日志。

- [ ] **Step 5: Commit**

```bash
git add mcp_servers Makefile scripts/dev.sh tests/tools/test_mcp_servers.py
git commit -m "feat(ch08): 物流/售后两台业务 MCP Server(Streamable HTTP 独立进程)+ 起停脚本"
```

---

### Task 5: MCP Client 接入 + 清单合并 + graph 接线(现问现拿)

**Files:**
- Create: `app/tools/mcp_client.py`
- Modify: `app/tools/registry.py`(加 `get_all_specs()`;删过渡兼容段)
- Delete: `app/tools/infra.py`
- Modify: `app/graph/nodes.py`(main_agent bind 全量、agent_tools 走 engine——本任务先不动 create_ticket 拦截语义)
- Modify: `tests/test_tools_infra.py` → 删除(执行类用例已被 test_engine 覆盖;registry 用例已迁 test_registry)
- Test: `tests/tools/test_mcp_client.py`(新建)+ `tests/graph/test_agent_nodes.py` 适配

**Interfaces:**
- Consumes: Task 2 registry、Task 3 engine、Task 4 两台 Server。
- Produces(Task 6 依赖):
  - `async registry.get_all_specs() -> list[ToolSpec]`(builtin + MCP 合并,重名 builtin 优先;MCP 单台不可达跳过)
  - `mcp_client.fetch_mcp_specs() -> list[ToolSpec]`、`mcp_client.FORMATTERS: dict[str, Callable]`
  - nodes 内部惯例:`specs = {s.name: s for s in await registry.get_all_specs()}` 后 `engine.execute_tool_call(tc, cid, specs)`

- [ ] **Step 1: 写失败测试 tests/tools/test_mcp_client.py**

```python
from app.tools import mcp_client, registry
from app.tools.registry import ToolSpec


class _FakeTool:
    def __init__(self, name):
        self.name = name
        self.description = f"{name} 描述"
        self.args_schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}


async def test_fetch_mcp_specs_marks_source_and_permission(monkeypatch):
    async def fake_get_tools(*, server_name=None):
        return [_FakeTool("query_logistics")] if server_name == "logistics" else [_FakeTool("query_warranty")]
    monkeypatch.setattr(mcp_client, "_get_tools_of", fake_get_tools)
    specs = await mcp_client.fetch_mcp_specs()
    by = {s.name: s for s in specs}
    assert by["query_logistics"].source == "mcp" and by["query_logistics"].mcp_server == "logistics"
    assert all(s.permission == "read" for s in specs)          # 我们侧规则:MCP 不在写清单→只读
    assert by["query_logistics"].format_result is mcp_client.FORMATTERS["query_logistics"]


async def test_one_server_down_degrades_gracefully(monkeypatch, caplog):
    async def fake_get_tools(*, server_name=None):
        if server_name == "logistics":
            raise ConnectionError("拒绝连接")
        return [_FakeTool("query_warranty")]
    monkeypatch.setattr(mcp_client, "_get_tools_of", fake_get_tools)
    specs = await mcp_client.fetch_mcp_specs()
    assert {s.name for s in specs} == {"query_warranty"}       # 单台挂了跳过,不拖垮
    assert any("不可达" in r.message for r in caplog.records)


async def test_get_all_specs_merges_builtin_wins(monkeypatch):
    async def fake_fetch():
        return [ToolSpec(name="query_order", description="MCP 冒名", json_schema={"type": "object", "properties": {}},
                         tool=_FakeTool("query_order"), permission="read", source="mcp", mcp_server="x"),
                ToolSpec(name="query_logistics", description="物流", json_schema={"type": "object", "properties": {}},
                         tool=_FakeTool("query_logistics"), permission="read", source="mcp", mcp_server="logistics")]
    monkeypatch.setattr(mcp_client, "fetch_mcp_specs", fake_fetch)
    specs = await registry.get_all_specs()
    by = {s.name: s for s in specs}
    assert by["query_order"].source == "builtin"               # 重名 builtin 优先,后到 MCP 丢弃
    assert by["query_logistics"].source == "mcp"               # 物流已下线内置,由 MCP 接管
    assert {"query_faq", "create_ticket", "submit_refund", "query_product"} <= set(by)


def test_logistics_formatter_translates_codes():
    out = mcp_client.FORMATTERS["query_logistics"](
        {"tracking_no": "SF1", "status_code": "IN_TRANSIT", "current_city": "深圳",
         "trace": ["深圳分拨中心 已发出"], "carrier_code": "SF-EXP-01"})
    assert out == {"tracking_no": "SF1", "status": "运输中", "current_city": "深圳",
                   "trace": ["深圳分拨中心 已发出"]}            # 挑字段 + 枚举翻人话,内部编码剔除
```

- [ ] **Step 2: 跑测试确认失败** → FAIL(mcp_client 不存在)

- [ ] **Step 3: 实现 app/tools/mcp_client.py**

```python
"""ch08 MCP Client:MultiServerMCPClient 多 Server 接入,现问现拿(每次现拉工具清单,
adapters 每次调用新建 session——Server 侧加工具,客服系统不重启即可见)。
权限/格式化只认我们侧:Server 自报的用途描述仅供模型参考,能不能调按 registry.WRITE_TOOLS。"""
import logging

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import settings
from app.tools import registry
from app.tools.registry import ToolSpec

logger = logging.getLogger(__name__)


def _translate(mapping: dict[str, str], code):
    return mapping.get(code, code)


def _fmt_logistics(d: dict) -> dict:
    return {"tracking_no": d.get("tracking_no"),
            "status": _translate({"PICKED_UP": "已揽件", "IN_TRANSIT": "运输中",
                                  "DELIVERING": "派送中", "DELIVERED": "已签收"}, d.get("status_code")),
            "current_city": d.get("current_city"), "trace": d.get("trace")}


def _fmt_warranty(d: dict) -> dict:
    return {"order_id": d.get("order_id"),
            "warranty": _translate({"IN_WARRANTY": "在保", "EXPIRED": "已过保"}, d.get("warranty_code")),
            "warranty_until": d.get("warranty_until")}


def _fmt_return(d: dict) -> dict:
    return {"order_id": d.get("order_id"),
            "return_status": _translate({"AUDITING": "审核中", "RETURNING": "退货中",
                                         "REFUNDED": "已退款", "NONE": "无退货记录"}, d.get("return_code")),
            "updated_at": d.get("updated_at")}


# 结果格式化我们侧登记(挑回答用得上的字段 + 内部枚举码翻人话);未登记的 MCP 工具透传
FORMATTERS = {"query_logistics": _fmt_logistics, "query_warranty": _fmt_warranty,
              "query_return_status": _fmt_return}

_client: MultiServerMCPClient | None = None


def _connections() -> dict:
    return {
        "logistics": {"transport": "streamable_http", "url": settings.mcp_logistics_url},
        "aftersales": {"transport": "streamable_http", "url": settings.mcp_aftersales_url},
    }


def get_client() -> MultiServerMCPClient:
    global _client
    if _client is None:
        # handle_tool_errors=False:工具错误抛 ToolException,由执行引擎统一分诊/回灌
        _client = MultiServerMCPClient(_connections(), handle_tool_errors=False)
    return _client


async def _get_tools_of(*, server_name: str):
    """薄壳:单测 monkeypatch 锚点。"""
    return await get_client().get_tools(server_name=server_name)


async def fetch_mcp_specs() -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    for server in _connections():
        try:
            tools = await _get_tools_of(server_name=server)
        except Exception as e:  # noqa: BLE001 单台不可达:告警+跳过,不拖垮本轮对话
            logger.warning("MCP Server「%s」不可达,本轮跳过其工具:%s", server, type(e).__name__)
            continue
        for t in tools:
            specs.append(registry.spec_from_langchain_tool(
                t, source="mcp", mcp_server=server, format_result=FORMATTERS.get(t.name)))
    return specs
```

`app/tools/registry.py`:删"过渡兼容"整段(get_all_tools/get_tool/NO_RETRY/INJECT_CONVERSATION/TOOL_TIMEOUTS),加:

```python
async def get_all_specs() -> list[ToolSpec]:
    """内置 + MCP 现拉合并;重名 builtin 优先、后到丢弃告警(本章无重名:内置 query_logistics 已下线)。"""
    from app.tools import mcp_client   # 延迟导入避免环
    merged: dict[str, ToolSpec] = {s.name: s for s in builtin_specs()}
    for s in await mcp_client.fetch_mcp_specs():
        if s.name in merged:
            logger.warning("MCP 工具与已注册工具重名,丢弃 name=%s server=%s", s.name, s.mcp_server)
            continue
        merged[s.name] = s
    return list(merged.values())
```

- [ ] **Step 4: nodes.py 接线**

`app/graph/nodes.py`:
- import 改:删 `from app.tools.infra import execute_tool_call`、`from app.tools.registry import get_all_tools`;加 `from app.tools import engine, registry`。
- `main_agent`:`.bind_tools(get_all_tools())` → 先 `specs = await registry.get_all_specs()` 再 `.bind_tools([s.tool for s in specs])`(现问现拿:MCP 新工具下一轮即可 bind)。
- `agent_tools`:开头加 `specs = {s.name: s for s in await registry.get_all_specs()}`;else 分支 `run = await execute_tool_call(tc, state.get("conversation_id", 0))` → `run = await engine.execute_tool_call(tc, state.get("conversation_id", 0), specs)`(create_ticket/submit_refund 拦截分支本任务保持原样,Task 6 改)。
- 删除 `app/tools/infra.py` 与 `tests/test_tools_infra.py`(用例已由 test_engine/test_registry 覆盖)。
- `tests/graph/test_agent_nodes.py` 等 monkeypatch `execute_tool_call` 的点位改为 monkeypatch `nodes.engine.execute_tool_call` 或 `nodes.registry.get_all_specs`(执行时读该文件对位适配,断言语义不变)。

- [ ] **Step 5: 跑测试(单测 + 真 MCP 全链路冒烟)**

Run: `uv run pytest tests/tools tests/graph -v` → 全 PASS;`make test` 无回归。
Run(真链路):
```bash
make mcp-up && sleep 1
uv run python - <<'EOF'
import asyncio
from app.tools import registry
async def main():
    specs = await registry.get_all_specs()
    names = sorted(s.name for s in specs)
    print("清单:", names)
    assert "query_logistics" in names and "query_warranty" in names
    by = {s.name: s for s in specs}
    from app.tools import engine
    run = await engine.execute_tool_call(
        {"name": "query_logistics", "args": {"tracking_no": "SF999"}, "id": "smoke"}, 0, by)
    print("结果:", run.tool_message.content, run.status)
    assert run.ok and "运输" in run.tool_message.content or run.ok
asyncio.run(main())
EOF
make mcp-down
```
Expected: 清单含内置 4 + MCP 3;调用成功且 content 是翻译后的中文字段。(此冒烟会向 mewhelp 库落审计——dev 库已建表,正好人肉 SELECT 验一条。)

- [ ] **Step 6: Commit**

```bash
git add -A app/tools app/graph/nodes.py tests
git commit -m "feat(ch08): MCP Client 现问现拿接入+清单合并;agent 全量 bind;infra.py 退役"
```

---

### Task 6: 建工单确认流(interrupt 预览 → resume 放行/权限拒绝)

**Files:**
- Modify: `app/graph/nodes.py`(agent_tools 重写 create_ticket 分支)
- Modify: `app/core/prompts.py:38-54`(AGENT_SYSTEM 工具使用原则)
- Modify: `app/graph/runtime.py:94-100,140-146`(interrupt payload 泛化)
- Modify: `app/api/chat.py`(interrupt 帧透传)、`app/api/actions.py:51-81`、`app/schemas/actions.py:28-30`(resume 泛化)
- Test: `tests/graph/test_ch08_confirm_ticket.py`(新建,graph 级 interrupt→resume 两分支)

**Interfaces:**
- Consumes: Task 3 engine(confirmed/deny_note、validate_args)、Task 5 get_all_specs。
- Produces: interrupt payload `{"type": "confirm_ticket", "preview": {"ticket_type": str, "description": str}}`;resume 值 `{"confirmed": bool}`;SSE interrupt 帧 `{"event":"interrupt","kind":"confirm_ticket","preview":{...},"conversation_id":N}`(select_order 帧保持 `orders` 字段兼容);`POST /api/actions/resume` body `{conversation_id, order_id?, confirmed?}`(前端 Task 7 消费)。

- [ ] **Step 1: 写失败测试 tests/graph/test_ch08_confirm_ticket.py**

(执行时参考 `tests/graph/test_ch06_nodes.py` 的既有 fixture 风格对位;骨架如下,断言语义不得减:)

```python
"""graph 级:建工单确认流。业务路由 → main_agent 发 create_ticket → agent_tools 顶置 interrupt
推预览 → resume confirmed 两分支(true 落库回工单号 / false 权限拒绝审计+不落库)。"""
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph import nodes
from app.graph.build import build_graph
from app.tools import engine, registry


class FakeModel:
    """第一次调用发 create_ticket tool_call,之后回普通文本收敛。"""
    def __init__(self):
        self.calls = 0
    def bind_tools(self, tools): return self
    def bind(self, **kw): return self
    async def ainvoke(self, msgs, config=None):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="", tool_calls=[{
                "name": "create_ticket", "id": "tc-1",
                "args": {"description": "猫砂盆漏电", "ticket_type": "售后"}}])
        return AIMessage(content="已为您创建工单")


@pytest.fixture()
def wired(monkeypatch):
    async def fake_coref(q, h): return q
    async def fake_intent(q, h): return {"intent": "售后", "confidence": 0.95}
    monkeypatch.setattr(nodes.coref, "resolve", fake_coref)
    monkeypatch.setattr(nodes.intent_mod, "classify", fake_intent)
    monkeypatch.setattr(nodes, "get_chat_model", lambda **kw: FakeModel())
    async def only_builtin():
        return [s for s in registry.builtin_specs()]
    monkeypatch.setattr(nodes.registry, "get_all_specs", only_builtin)

    created, audits = [], []
    async def fake_create(cid, desc, ttype):
        created.append((cid, desc, ttype)); return "T20260717001"
    async def fake_audit(**kw): audits.append(kw)
    monkeypatch.setattr("app.db.repository.create_ticket", fake_create)
    monkeypatch.setattr(engine.repository, "insert_tool_audit", fake_audit)
    async def no_msg(*a, **kw): return 1
    monkeypatch.setattr(nodes.repository, "append_message", no_msg)
    return created, audits


def _input(text):
    from app.graph.runtime import _graph_input
    return _graph_input("u1", text, 1, 1, "", 0)


async def test_confirm_true_creates_ticket(wired):
    created, audits = wired
    g = build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    st = await g.ainvoke(_input("帮我建个工单,猫砂盆漏电"), cfg)
    intr = st["__interrupt__"][0].value
    assert intr["type"] == "confirm_ticket"
    assert intr["preview"] == {"ticket_type": "售后", "description": "猫砂盆漏电"}
    assert created == []                                        # interrupt 时未执行

    st2 = await g.ainvoke(Command(resume={"confirmed": True}), cfg)
    assert created == [(1, "猫砂盆漏电", "售后")]
    assert any(a["status"] == "成功" for a in audits)
    assert "T20260717001" in str(st2["messages"])               # 工单号回灌到 ToolMessage


async def test_confirm_false_denied_and_audited(wired):
    created, audits = wired
    g = build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t2"}}
    await g.ainvoke(_input("帮我建个工单,猫砂盆漏电"), cfg)
    await g.ainvoke(Command(resume={"confirmed": False}), cfg)
    assert created == []                                        # 没建
    assert any(a["status"] == "权限拒绝" for a in audits)       # 审计落权限拒绝


async def test_missing_description_asks_instead_of_interrupt(wired, monkeypatch):
    """模型第一轮发缺 description 的 create_ticket → 校验拦下回灌(不 interrupt),
    第二轮模型收敛为追问文本。"""
    class NoDescModel(FakeModel):
        async def ainvoke(self, msgs, config=None):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="", tool_calls=[{
                    "name": "create_ticket", "id": "tc-1", "args": {"ticket_type": "咨询"}}])
            return AIMessage(content="请问您遇到的具体问题是什么呢?")
    monkeypatch.setattr(nodes, "get_chat_model", lambda **kw: NoDescModel())
    g = build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t3"}}
    st = await g.ainvoke(_input("帮我建个工单"), cfg)
    assert "__interrupt__" not in st                            # 没弹卡
    created, audits = wired
    assert created == [] and any(a["status"] == "校验拦下" for a in audits)
```

- [ ] **Step 2: 跑测试确认失败** → FAIL(agent_tools 仍走旧 suggested_actions 拦截,interrupt 不存在)

- [ ] **Step 3: 实现**

`app/graph/nodes.py` 的 `agent_tools` 整体替换:

```python
async def agent_tools(state) -> dict:
    """ReAct 行动步(ch08):一切工具经统一执行引擎。create_ticket(唯一写操作)走确认流——
    参数齐则节点顶部 interrupt 推工单预览(interrupt 前只做纯计算,resume 重跑安全,照 fetch_order 范式);
    参数缺则交引擎按「校验拦下」回灌,模型自然向用户追问。submit_refund 仍拦成前端退款表单(ch06)。"""
    last = state["messages"][-1]
    cid = state.get("conversation_id", 0)
    specs = {s.name: s for s in await registry.get_all_specs()}

    ticket_calls = [tc for tc in last.tool_calls if tc["name"] == "create_ticket"]
    decision = None
    tspec = specs.get("create_ticket")
    if ticket_calls and tspec is not None \
            and engine.validate_args(tspec, dict(ticket_calls[0].get("args") or {})) is None:
        first_args = ticket_calls[0]["args"]
        decision = interrupt({"type": "confirm_ticket",
                              "preview": {"ticket_type": first_args.get("ticket_type", "咨询"),
                                          "description": first_args.get("description", "")}})

    tool_msgs = []
    actions = list(state.get("suggested_actions", []))
    for tc in last.tool_calls:
        if tc["name"] == "submit_refund":
            actions.append({"type": "refund_form",
                            "draft": {"order_id": tc["args"].get("order_id", ""),
                                      "reason": tc["args"].get("reason")}})
            tool_msgs.append(ToolMessage(
                content="已把『提交退款工单』选项交给用户确认。请用一句话说明这一单可以退款并停止,不要再调用任何工具。",
                tool_call_id=tc["id"], name="submit_refund"))
        elif tc["name"] == "create_ticket" and decision is not None:
            if tc is ticket_calls[0]:
                if decision.get("confirmed"):
                    run = await engine.execute_tool_call(tc, cid, specs, confirmed=True)
                else:
                    run = await engine.execute_tool_call(
                        tc, cid, specs, confirmed=False,
                        deny_note="用户在工单预览卡片上点了取消,本次不建单。请勿再发起,除非用户再次明确要求。")
                tool_msgs.append(run.tool_message)
            else:
                tool_msgs.append(ToolMessage(content="一次只处理一个建工单请求,本次调用已忽略。",
                                             tool_call_id=tc["id"], name="create_ticket", status="error"))
        else:
            run = await engine.execute_tool_call(tc, cid, specs)
            tool_msgs.append(run.tool_message)
    out = {"messages": tool_msgs}
    if actions:
        out["suggested_actions"] = actions
    return out
```
(顶部 import 已在 Task 5 就位;`interrupt` 已 import。)

`app/core/prompts.py` AGENT_SYSTEM「工具使用原则」第三条替换 + 追加一条:

```
- 只有用户明确要求建工单/要求人工跟进时,才发起 create_ticket;发起前先核对问题描述等必填信息,
  缺什么就先向用户追问,严禁编造或用占位文本充数。发起后系统会把工单预览交给用户确认,
  用户取消时不要擅自重试。
- 工具清单可能动态变化(如物流、在保、退货进度等由外部服务提供),按各工具的用途描述选用;
  工具返回错误说明时,按说明修正参数或如实告知用户,不要编造结果。
```

`app/graph/runtime.py`:
- `_stream_events` interrupt 分支替换:

```python
            if "__interrupt__" in chunk:
                payload = chunk["__interrupt__"][0].value
                ev = {"type": "interrupt", "kind": payload.get("type", ""), "conversation_id": cid}
                for k in ("orders", "preview"):                 # 按中断类型透传负载
                    if payload.get(k) is not None:
                        ev[k] = payload[k]
                yield ev
                return
```
- `_interrupt_orders` 改名 `_interrupt_payload`(返回整个 `intr[0].value`,`run_turn`/`resume_turn` 的 `"interrupt"` 键值随之为 payload dict;`scripts/eval_ch06.py`/`smoke_interrupt.py` 若引用 orders 形状,执行时对位适配)。

`app/schemas/actions.py`:

```python
class ResumeRequest(BaseModel):
    conversation_id: int
    order_id: str | None = Field(default=None, min_length=1)   # ch06 订单选择器
    confirmed: bool | None = None                              # ch08 工单预览确认/取消
```

`app/api/actions.py` `resume_action` 开头加分派(其余流式分支不动):

```python
    if req.order_id is None and req.confirmed is None:
        raise HTTPException(status_code=400, detail="order_id 与 confirmed 至少传一个")
    resume_value = req.order_id if req.order_id is not None else {"confirmed": bool(req.confirmed)}
```
`stream_resume(req.conversation_id, req.order_id)` → `stream_resume(req.conversation_id, resume_value)`;interrupt 帧 SSE 组装改为透传:

```python
                elif ev["type"] == "interrupt":
                    yield _sse({"event": "interrupt",
                                **{k: v for k, v in ev.items() if k != "type"}})
```
`app/api/chat.py` interrupt 帧同样改透传。

- [ ] **Step 4: 跑测试**

Run: `uv run pytest tests/graph/test_ch08_confirm_ticket.py -v` → 3 PASS
Run: `make test` → 全量无回归(test_actions_api 若有 resume 用例按新 schema 补 confirmed 分支断言)。

- [ ] **Step 5: Commit**

```bash
git add app/graph app/core/prompts.py app/api app/schemas tests scripts
git commit -m "feat(ch08): create_ticket interrupt 确认流——预览/resume 放行/取消落权限拒绝"
```

---

### Task 7: 前端工单预览卡(Vibe Coding,不套 TDD)

**Files:**
- Modify: `app/static/index.html`(`renderOrderCards` 附近加 `renderTicketConfirm`;`streamInto` interrupt 回调按 kind 分发;样式对齐 `.order-cards`)

**Interfaces:**
- Consumes: SSE interrupt 帧 `{"event":"interrupt","kind":"confirm_ticket","preview":{ticket_type,description},"conversation_id":N}`;`POST /api/actions/resume {conversation_id, confirmed: true|false}`。

- [ ] **Step 1: 实现(要点,细节按用户描述迭代)**
  - `streamInto` 的 interrupt 处理:`kind === "select_order"` → 走现有 `renderOrderCards`;`kind === "confirm_ticket"` → `renderTicketConfirm(bubble, data.preview, data.conversation_id)`。
  - `renderTicketConfirm`:气泡内渲染预览卡(标题「工单预览」、两行字段:工单类型/问题描述)+「确认提交」「取消」两按钮;点任一按钮后禁用两按钮,回显用户选择,开新 bot 气泡 `streamInto(nb, () => fetch("/api/actions/resume", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({conversation_id, confirmed})}))`;样式复用 `.order-cards` 一族,卡片风格一致。
- [ ] **Step 2: 手测**:起全套服务浏览器走一遍(确认/取消两分支),按用户反馈改。
- [ ] **Step 3: Commit** `git commit -m "feat(ch08): 前端工单预览卡(确认提交/取消)"`

---

### Task 8: 验收配套(promotions 演示工具草稿 + prompt 样例验证 + 演示手册)

**Files:**
- Create: `scripts/eval_ch08.py`(prompt/确认流标注样例端到端验证;照 eval_ch06.py 惯例,需全服务)
- Create: `scripts/demo_ch08_promotions.py.txt`(验收 1 演示用:现场 `cp` 成 `app/tools/builtin/promotions.py` 重启即注册;不预先放进 builtin/,保证"演示时才丢文件"可信)
- Modify: `Makefile`(eval-ch08 目标)

**Interfaces:**
- Consumes: 全链路。
- Produces: `make eval-ch08`;验收 1/3 的演示物料。

- [ ] **Step 1: promotions 演示文件**(`scripts/demo_ch08_promotions.py.txt`,内容即最终 .py):

```python
"""ch08 验收 1 演示:新内置工具=往 builtin/ 丢本文件(cp 本文件为 app/tools/builtin/promotions.py),
重启服务即被扫描注册,核心代码零改动,Agent 对话里即可用。"""
import random

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.tools import registry


class PromotionInput(BaseModel):
    keyword: str = Field(description="想查的活动关键词或商品品类,例如 猫粮")


@tool(args_schema=PromotionInput)
async def query_promotions(keyword: str) -> dict:
    """查询当前进行中的优惠活动/促销信息。用于用户问「有什么活动/优惠/折扣」时。"""
    rng = random.Random(f"promo:{keyword}")
    return {"keyword": keyword,
            "promotions": [f"{keyword}品类满{rng.randint(99, 299)}减{rng.randint(10, 50)}",
                           f"新客专享 {rng.choice(['9折', '8.5折', '95折'])}"],
            "valid_until": "2026-07-31"}


registry.register(registry.spec_from_langchain_tool(query_promotions, source="builtin"))
```

- [ ] **Step 2: scripts/eval_ch08.py**(标注样例,直连 runtime 层,监听真库;样例集:①「帮我建个工单」缺描述→期望:无 confirm_ticket 中断、tickets 不落、答复含追问;②补齐描述→期望:中断出现且 preview 字段齐;③resume confirmed=True→tickets +1、答复含工单号;④resume False→无新工单、审计有权限拒绝;⑤「我的猫粮什么时候到,单号 SFxxxx」→ 走 MCP query_logistics,答复含翻译后的中文状态)。执行时按 eval_ch05/eval_ch06 的输出风格打 PASS/FAIL 汇总,退出码非 0 表示有 FAIL。Makefile:

```makefile
eval-ch08:  ## ch08 验收样例端到端(需 make dev 全服务 + mcp-up)
	uv run python scripts/eval_ch08.py
```

- [ ] **Step 3: 跑一遍** `make mcp-up && make eval-ch08`(dev 服务起着)→ 全 PASS;失败即修再跑。
- [ ] **Step 4: Commit** `git commit -m "feat(ch08): 验收配套——eval_ch08 样例验证 + promotions 演示物料"`

---

### Task 9: 端到端验收(浏览器)+ code review + finish

- [ ] **Step 1: 全套服务起**:`make dev`(dev.sh 已含 mcp-up)。dev 库确认已应用 ch08 DDL。
- [ ] **Step 2: 六条验收逐条过**(浏览器 + 数据库核对;用户要求端到端成功才交付):
  1. `cp scripts/demo_ch08_promotions.py.txt app/tools/builtin/promotions.py` → 重启 uvicorn → 聊天问「最近有什么优惠活动」→ Agent 调 query_promotions 回答。演示后删除该文件恢复原状(或保留,验收时问用户)。
  2. 聊天问「我的订单 1001 的物流到哪了」(Agent 先 query_order 拿单号再 MCP query_logistics)、「订单 1001 还在保吗」→ 都答出 mock 数据(中文状态)。
  3. 向 `mcp_servers/logistics_server.py` 临时加 `query_delivery_eta` 工具 → `make mcp-down && make mcp-up`(只重启 MCP)→ 客服系统不动 → 问「订单 1001 预计几号送到」→ 用上新工具;演示完回滚该临时工具。
  4. 「帮我建个工单」(不说问题)→ 追问;补「猫砂盆漏电」→ 预览卡 → 确认提交 → `SELECT * FROM tickets ORDER BY created_at DESC LIMIT 1` 有新单,回复带工单号。
  5. 再走一遍到预览卡 → 点取消 → tickets 无新增,`SELECT status FROM tool_audit_logs WHERE tool_name='create_ticket' ORDER BY id DESC LIMIT 1` = 权限拒绝。
  6. `MOCK_DELAY_SECONDS=12 PORT=8101 uv run python mcp_servers/logistics_server.py`(替换物流 Server)→ 问物流 → 兜底话术;审计该次调用 status=超时、retry_count=2、duration_ms≈3×10s 段位。恢复正常 Server;`DEMO_TICKET_DELAY_SECONDS=8` 重启主服务 → 建单确认后超时 → 审计 create_ticket status=超时、retry_count=0。恢复环境。
- [ ] **Step 3: code review**:invoke superpowers:requesting-code-review 对 ch08 全部 diff 审(前端除外按用户规矩);发现问题按 receiving-code-review 处理。
- [ ] **Step 4: finish**:全量 `make test` + dev-notes finish 段(交付清单:演示命令/测试结果/spec/plan/dev-notes 路径)+ 向用户交付验收演示手册。

---

## Self-Review 记录

- Spec 覆盖:§3 注册中心→Task 2/5;§4 引擎→Task 3;§5 确认流→Task 6;§6 MCP Server→Task 4;§7 配置/DB→Task 1;§8 前端→Task 7;§9 测试/验收→各任务内 + Task 8/9。六条验收标准全部落在 Task 9 Step 2 并有前置任务支撑。
- 类型一致性:ToolSpec/ToolRun/execute_tool_call/validate_args/get_all_specs 签名在 Task 2/3/5/6 间一致;resume payload 与前端消费一致。
- 已知留白(有意,执行时对位):test_ch06_nodes/test_agent_nodes 的具体 fixture 名、eval_ch08.py 全文、前端具体 JS——均标注了"执行时读对应文件对位",不属于 TBD 型占位。
