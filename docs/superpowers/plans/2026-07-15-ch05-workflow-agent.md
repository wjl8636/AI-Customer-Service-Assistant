# Ch05 Workflow + Agent 混合架构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 LangGraph 把客服后端重构成「确定性 Workflow 骨架 + 主力 Agent 核心节点」的混合架构,两个 API 入口都走图,前端加转人工/建工单两个按钮。

**Architecture:** `StateGraph` 编排六段骨架(指代消解→意图识别→分流→主力Agent→置信度兜底→日志);知识类强制先检索、证据弱生成前短路兜底;主力 Agent 是 `agent_llm ↔ agent_tools` 的 ReAct 环,复用 ch02 五工具;会话状态用 State 贯穿、AsyncSqliteSaver 持久化;转人工/建工单由后端产出可选项、前端按钮触发。

**Tech Stack:** LangGraph + langgraph-checkpoint-sqlite + aiosqlite;FastAPI(既有);LangChain ChatOpenAI 直连上游(既有);MySQL/SQLAlchemy(既有,存工单/低置信池/消息审计);原生 HTML/CSS/JS 前端(既有,加按钮)。

## Global Constraints

- 上游选型定死:LangGraph 图编排 + State + checkpointer;checkpointer = **AsyncSqliteSaver**(自带、持久);前端 = 原生 HTML/CSS/JS,仅加按钮。发现矛盾停下问用户,不自行换方案。
- 复用 ch02 五工具(query_order/query_product/query_logistics/query_faq/create_ticket),**本章不新写业务工具**。
- 意图识别单标签七类(物流/订单/商品咨询/退款退货/售后/投诉/闲聊),指代消解原样透传——两者正式版留 ch06。
- 转人工、建工单是两件事,分开;后端**不自动执行**,只产出可选项;建工单写库只在按钮端点、点了才写;转人工纯前端模拟。
- 涉及 LangGraph/FastAPI/SQLAlchemy 具体 API,动手前用 Context7 核实当前接口(版本对不上是返工重灾区)。
- 不可单测的产出(纯 Prompt/数据)用标注样例/评估集验证代替 TDD;其余走 TDD。
- 端到端**真实**验收必须跑通(非纸面):五条验收在浏览器/真实服务上复现。
- 每个任务末尾 commit;dev-notes/ch05.md 每完成一阶段追记。
- 模型非确定性:eval 抖动如实重跑记录。

## 权威来源
- spec:`docs/superpowers/specs/2026-07-15-ch05-workflow-agent-design.md`
- 章节:`mewhelp-course/ch05-workflow-agent/README.md`

---

## 文件结构

**新建:**
- `app/graph/__init__.py` — 包
- `app/graph/state.py` — `ConversationState` + reducer
- `app/graph/routing.py` — `route_by_intent` / `confidence_gate` / `should_continue`
- `app/graph/nodes.py` — 各节点函数
- `app/graph/build.py` — `build_graph()` 组装 StateGraph
- `app/graph/runtime.py` — checkpointer 生命周期 + 图单例 + `run_turn`/`stream_turn`
- `app/core/intent.py` — 意图分类器(prompt + 结构化输出)
- `app/api/actions.py` — `POST /api/actions/create-ticket`
- `scripts/bare_agent_loop.py` — 祛魅裸循环
- `scripts/eval_intent.py` — 意图分类标注评估
- `scripts/eval_ch05.py` — 五验收端到端评估
- `tests/graph/` — 图相关单测

**修改:**
- `pyproject.toml` — 加依赖
- `app/config.py` — 加 ch05 配置
- `app/core/prompts.py` — 加意图/兜底/安抚/闲聊话术
- `app/schemas/agent.py` — `AgentResponse` 加 `suggested_actions`
- `app/api/chat.py` — 驱动 `graph.astream`
- `app/api/agent.py` — 驱动 `graph.ainvoke`
- `app/main.py` — lifespan 起 checkpointer + 挂 actions 路由
- `app/static/index.html` — actions 帧渲染 + 两按钮
- `Makefile` — ch05 demo/eval 目标

**退役:** `app/core/agent.py` 的 `run_agent_turn`/`stream_agent_turn` 两次调用编排被图取代;其工具消息辅助函数(`_gen_tool_messages` 等)按需迁进 nodes。相关旧单测(test_agent_orchestration/test_agent_stream)改为测图。

---

## Task 1: 依赖 + 配置 + LangGraph/checkpointer/流式 fail-fast 冒烟(红线)

**目的:** 在写任何图代码前,先在**已安装版本**上验证 StateGraph + AsyncSqliteSaver + `astream(stream_mode=["messages","updates"])` 三条链路真能跑通,并**钉死流式输出的元组形状**(spec §12 R1/R2)。不通则停下按选型内解决或问用户。

**Files:**
- Modify: `pyproject.toml`
- Modify: `app/config.py`
- Create: `scripts/smoke_langgraph.py`

**Interfaces:**
- Produces: `settings.max_agent_steps:int`、`settings.checkpointer_db_path:str`

- [ ] **Step 1: 用 Context7 核实 AsyncSqliteSaver 与 astream 接口**

查 `/websites/langchain_oss_python_langgraph`:确认 `from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver`、`AsyncSqliteSaver.from_conn_string(path)` 为 async context manager、`await checkpointer.setup()`;确认 `graph.astream(inp, config, stream_mode=["messages","updates"])` 迭代产出 `(mode, chunk)` 元组(messages 模式 chunk=`(msg, metadata)`,updates 模式 chunk=`{node: state_delta}`)。把确认结论写进 dev-notes。

- [ ] **Step 2: 加依赖**

`pyproject.toml` 的 dependencies 增加:
```
"langgraph>=0.2",
"langgraph-checkpoint-sqlite>=2.0",
"aiosqlite>=0.20",
```
Run: `uv sync`
Expected: 安装成功,`uv pip show langgraph` 有版本。

- [ ] **Step 3: 加配置**

`app/config.py` 的 `Settings` 增加:
```python
    # ch05 图编排
    max_agent_steps: int = 6              # ReAct 环最大步数(封顶,超则兜底)
    # token 花销不在环里卡:那是成本控制,归 ch09 跟 Langfuse 的账一起看
    checkpointer_db_path: str = "data/ch05_checkpoints.sqlite"  # LangGraph checkpointer(data/*.db* 已 gitignore)
```

- [ ] **Step 4: 写冒烟脚本**

`scripts/smoke_langgraph.py`:
```python
"""ch05 红线冒烟:StateGraph + AsyncSqliteSaver + 多模式流式 是否在当前版本跑通,并打印流式形状。"""
import asyncio
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.core.llm import get_chat_model


class S(TypedDict):
    messages: Annotated[list, add_messages]


async def call_model(state: S):
    ai = await get_chat_model(streaming=True).ainvoke(state["messages"])
    return {"messages": [ai]}


async def main():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as cp:
        await cp.setup()
        b = StateGraph(S)
        b.add_node("call_model", call_model)
        b.add_edge(START, "call_model")
        b.add_edge("call_model", END)
        graph = b.compile(checkpointer=cp)
        config = {"configurable": {"thread_id": "smoke-1"}}
        seen_modes = set()
        async for mode, chunk in graph.astream(
            {"messages": [HumanMessage("用一句话介绍你自己")]},
            config, stream_mode=["messages", "updates"],
        ):
            seen_modes.add(mode)
            if mode == "messages":
                msg, meta = chunk
                print("MSG node=", meta.get("langgraph_node"), "text=", (msg.content or "")[:20])
            else:
                print("UPD", list(chunk.keys()))
        print("OK modes=", seen_modes)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 5: 跑冒烟(需上游可用)**

Run: `make dev` 另开;`.venv/bin/python -m scripts.smoke_langgraph`
Expected: 打印若干 `MSG node= call_model text=...`(证明 messages 模式按节点带 token)、`UPD ['call_model']`(updates 模式带节点名)、`OK modes= {'messages', 'updates'}`。
若元组形状与假设不符(如返回 StreamPart dict 需 `version="v2"`),**在此处记录真实形状并据此调整 Task 12/14 的解析代码**;若 checkpointer/图跑不通,停下问用户(选型定死)。

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock app/config.py scripts/smoke_langgraph.py
git commit -m "feat(ch05): 加 LangGraph 依赖+配置,红线冒烟钉死流式形状"
```

---

## Task 2: 祛魅热身——手写裸 Agent 循环(不用框架)

**目的:** req#1 祛魅。产出是教学脚本,非单测代码 → 用运行验证代替 TDD。

**Files:**
- Create: `scripts/bare_agent_loop.py`

- [ ] **Step 1: 写裸循环脚本**

`scripts/bare_agent_loop.py`(照章节 85–98 行范本,复用 ch02 工具与既有模型工厂,不引入 LangGraph):
```python
"""祛魅:不用任何框架,手写最裸的 Agent 循环——看清它就是个带工具清单的 for 循环。
用法:.venv/bin/python -m scripts.bare_agent_loop "订单1001的物流到哪了"
"""
import asyncio
import sys

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import get_chat_model
from app.core.prompts import AGENT_SYSTEM
from app.tools.infra import execute_tool_call
from app.tools.registry import get_all_tools


async def run_agent(query: str, max_turns: int = 6):
    model = get_chat_model().bind_tools(get_all_tools())
    messages = [SystemMessage(AGENT_SYSTEM), HumanMessage(query)]
    for step in range(1, max_turns + 1):
        ai = await model.ainvoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            print(f"[第{step}步] 无工具调用 → 收敛")
            return ai.content
        for tc in ai.tool_calls:
            print(f"[第{step}步] 调用工具 {tc['name']} args={tc['args']}")
            run = await execute_tool_call(tc, conversation_id=0)
            messages.append(run.tool_message)
    print(f"[封顶] max_turns={max_turns} 用尽仍未收敛,生产环境此处应走兜底话术")
    return "(未收敛)"


async def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "订单1001的物流到哪了"
    print("问题:", query)
    print("答复:", await run_agent(query))


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: 跑简单问 + 复杂问验证收敛(需上游+服务)**

Run:
```
.venv/bin/python -m scripts.bare_agent_loop "退货政策是什么"
.venv/bin/python -m scripts.bare_agent_loop "我手机尾号1001的订单发货了吗?到哪了?"
```
Expected: 简单问 1 步收敛;复杂问看到多步(先查订单再查物流)后收敛。抖动如实重跑记录。

- [ ] **Step 3: dev-notes 记祛魅结论 + Commit**

在 `dev-notes/ch05.md` 追记"阶段2:祛魅"一段(裸循环就是带工具清单的 for 循环,LangGraph 只是把它放进更大的图管理)。
```bash
git add scripts/bare_agent_loop.py && git add -f dev-notes/ch05.md
git commit -m "feat(ch05): 祛魅裸 Agent 循环脚本 + 收敛验证"
```

---

## Task 3: 会话 State 定义

**Files:**
- Create: `app/graph/__init__.py`(空)
- Create: `app/graph/state.py`
- Test: `tests/graph/__init__.py`(空)、`tests/graph/test_state.py`

**Interfaces:**
- Produces:
  - `ConversationState`(TypedDict)字段见下
  - `merge_dict(a: dict, b: dict) -> dict`(trace 的 reducer)

- [ ] **Step 1: 写失败测试**

`tests/graph/test_state.py`:
```python
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages

from app.graph.state import ConversationState, merge_dict


def test_merge_dict_accumulates():
    assert merge_dict({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert merge_dict(None, {"b": 2}) == {"b": 2}
    assert merge_dict({"a": 1}, {"a": 9}) == {"a": 9}


def test_add_messages_reducer_appends():
    merged = add_messages([HumanMessage("hi")], [AIMessage("yo")])
    assert [m.content for m in merged] == ["hi", "yo"]


def test_state_has_required_keys():
    keys = ConversationState.__annotations__
    for k in ["messages", "user_id", "conversation_id", "intent", "route",
              "evidence", "citations", "evidence_strong", "answer",
              "steps", "tokens_used", "suggested_actions", "trace"]:
        assert k in keys
```

- [ ] **Step 2: 跑验证失败**

Run: `.venv/bin/pytest tests/graph/test_state.py -v`
Expected: FAIL(`app.graph.state` 不存在)

- [ ] **Step 3: 实现 State**

`app/graph/state.py`:
```python
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def merge_dict(a: dict | None, b: dict | None) -> dict:
    """trace 累加 reducer:后写覆盖同键,其余合并。"""
    return {**(a or {}), **(b or {})}


class ConversationState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # 跨轮历史,checkpointer 续接
    user_id: str
    conversation_id: int
    intent: str            # 七类之一
    route: str             # knowledge | business | complaint | chitchat
    evidence: str          # 知识路编号证据文本
    citations: list        # 引用 chunk(前端可点)
    evidence_strong: bool  # 生成前证据闸信号
    answer: str            # 确定性节点产出的答复(agent 答复走流式,不落此字段)
    steps: int             # ReAct 步数(停止条件)
    tokens_used: int       # 逐步累加进 trace,供排障与 ch09 统计,不当停止条件
    suggested_actions: list  # [{"type": "transfer_human"} | {"type": "create_ticket", "draft": {...}}]
    trace: Annotated[dict, merge_dict]  # 留痕
```

- [ ] **Step 4: 跑验证通过**

Run: `.venv/bin/pytest tests/graph/test_state.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/graph/__init__.py app/graph/state.py tests/graph/
git commit -m "feat(ch05): ConversationState + trace reducer"
```

---

## Task 4: 意图分类器(Prompt/数据 → 评估集验证)

**目的:** req#6 单标签七类。产出是纯 Prompt → 用标注评估集代替 TDD。

**Files:**
- Modify: `app/core/prompts.py`(加 `INTENT_CLASSIFY_SYSTEM` / `INTENT_CLASSIFY_PROMPT`)
- Create: `app/core/intent.py`
- Create: `scripts/eval_intent.py`

**Interfaces:**
- Produces: `async def classify(query: str) -> str`(返回七类之一;异常/越界回退 "闲聊")

- [ ] **Step 1: 加分类 Prompt**

`app/core/prompts.py` 末尾追加:
```python
INTENT_CLASSIFY_SYSTEM = """你是电商客服的意图分类器。把用户这句话归到且仅归到下面七类之一,输出分类结果:
- 物流:问快递到哪了、发货没、物流进度(通常带订单号/单号)。
- 订单:问某订单的状态、金额、下单时间、买了什么。
- 商品咨询:问商品价格、库存、规格、功能、怎么用等通用商品/政策/FAQ 问题。
- 退款退货:想退款、退货、换货,或问退换货政策/流程。
- 售后:维修、保修、换新进度等售后处理(不含退款退货)。
- 投诉:表达强烈不满、要投诉、要说法(不夹带可自助解决的具体查询时)。
- 闲聊:问候、寒暄、与购物无关的话题。
判不准时选最接近的一类;严格从这七类里选,不要造新类。"""
```
并加:
```python
INTENT_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages(
    [("system", INTENT_CLASSIFY_SYSTEM), ("human", "用户这句话:{query}")]
)
```

- [ ] **Step 2: 实现分类器**

`app/core/intent.py`:
```python
from typing import Literal

from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import INTENT_CLASSIFY_PROMPT

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊")


class _Intent(BaseModel):
    intent: Literal["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊"] = Field(
        description="七类意图之一"
    )


async def classify(query: str) -> str:
    """单标签七类意图。扁平字段避开 glm 嵌套 502;越界/异常回退闲聊(最保守出口)。"""
    model = get_chat_model().with_structured_output(_Intent)
    try:
        r: _Intent = await (INTENT_CLASSIFY_PROMPT | model).ainvoke({"query": query})
    except Exception:
        return "闲聊"
    return r.intent if r.intent in INTENTS else "闲聊"
```

- [ ] **Step 3: 写标注评估脚本**

`scripts/eval_intent.py`:
```python
"""意图分类标注评估:核对 classify 七类判对率。需上游可用。
用法:.venv/bin/python -m scripts.eval_intent"""
import asyncio

from app.core.intent import classify

# (用户问法, 期望意图)——七类各若干条
SAMPLES = [
    ("订单1001的快递到哪了", "物流"),
    ("我买的东西发货了吗", "物流"),
    ("订单2002现在什么状态", "订单"),
    ("我上周下的单多少钱来着", "订单"),
    ("这款猫粮多少钱一包", "商品咨询"),
    ("智能猫砂盆怎么用啊", "商品咨询"),
    ("我想退货", "退款退货"),
    ("退款一般几天到账", "退款退货"),
    ("我的猫爬架坏了能保修吗", "售后"),
    ("换货进度到哪了", "售后"),
    ("你们这什么破服务,我要投诉", "投诉"),
    ("太差了给我个说法", "投诉"),
    ("你好呀", "闲聊"),
    ("今天天气不错", "闲聊"),
]


async def main():
    passed = 0
    for q, expect in SAMPLES:
        got = await classify(q)
        ok = got == expect
        passed += ok
        print(f"{'✅' if ok else '❌'} {q!r} -> {got} 期望={expect}")
    print(f"\n判对 {passed}/{len(SAMPLES)}(glm 非确定性,抖动如实重跑记录)")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: 跑评估(需上游可用)**

Run: `.venv/bin/python -m scripts.eval_intent`
Expected: 判对率高(容许个别边界抖动,如"售后↔退款退货");明显偏差则调 Prompt 重跑。把结果记 dev-notes。

- [ ] **Step 5: Commit**

```bash
git add app/core/prompts.py app/core/intent.py scripts/eval_intent.py
git commit -m "feat(ch05): 单标签七类意图分类器 + 标注评估"
```

---

## Task 5: 路由与停止条件(确定性 → TDD)

**Files:**
- Create: `app/graph/routing.py`
- Test: `tests/graph/test_routing.py`

**Interfaces:**
- Produces:
  - `INTENT_TO_ROUTE: dict[str, str]`
  - `route_by_intent(state) -> str`(返回 "knowledge"|"business"|"complaint"|"chitchat")
  - `confidence_gate(state) -> str`(返回 "strong"|"weak")
  - `should_continue(state) -> str`(返回 "continue"|"stop")

- [ ] **Step 1: 写失败测试**

`tests/graph/test_routing.py`:
```python
from langchain_core.messages import AIMessage

from app.config import settings
from app.graph.routing import (
    confidence_gate, route_by_intent, should_continue,
)


def test_route_maps_seven_to_four():
    cases = {
        "商品咨询": "knowledge", "退款退货": "knowledge",
        "物流": "business", "订单": "business", "售后": "business",
        "投诉": "complaint", "闲聊": "chitchat",
    }
    for intent, route in cases.items():
        assert route_by_intent({"intent": intent}) == route


def test_route_unknown_intent_defaults_business():
    assert route_by_intent({"intent": "火星语"}) == "business"


def test_confidence_gate():
    assert confidence_gate({"evidence_strong": True}) == "strong"
    assert confidence_gate({"evidence_strong": False}) == "weak"
    assert confidence_gate({}) == "weak"


def test_should_continue_stops_when_no_tool_calls():
    state = {"messages": [AIMessage("答完了")], "steps": 1, "tokens_used": 0}
    assert should_continue(state) == "stop"


def test_should_continue_continues_on_tool_calls_under_budget():
    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "1"}])
    state = {"messages": [ai], "steps": 1, "tokens_used": 0}
    assert should_continue(state) == "continue"


def test_should_continue_stops_on_step_cap():
    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "1"}])
    state = {"messages": [ai], "steps": settings.max_agent_steps, "tokens_used": 0}
    assert should_continue(state) == "stop"


def test_token_花销不再当停止条件():
    # 环里只按步数封顶。token 记账继续走(进 trace、给 ch09 统计),但不参与判断
    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "1"}])
    state = {"messages": [ai], "steps": 1, "tokens_used": 10_000_000}
    assert should_continue(state) == "continue"
```

- [ ] **Step 2: 跑验证失败**

Run: `.venv/bin/pytest tests/graph/test_routing.py -v`
Expected: FAIL(模块不存在)

- [ ] **Step 3: 实现路由**

`app/graph/routing.py`:
```python
from langchain_core.messages import AIMessage

from app.config import settings

# 七类意图 → 四出口(写死在代码里的分流规则,spec §3.1)
INTENT_TO_ROUTE: dict[str, str] = {
    "商品咨询": "knowledge",
    "退款退货": "knowledge",
    "物流": "business",
    "订单": "business",
    "售后": "business",
    "投诉": "complaint",
    "闲聊": "chitchat",
}


def route_by_intent(state) -> str:
    """按意图分流;未知意图保守归 business(让 Agent 自己应对)。"""
    return INTENT_TO_ROUTE.get(state.get("intent", ""), "business")


def confidence_gate(state) -> str:
    """知识路生成前证据闸:强放行、弱兜底。"""
    return "strong" if state.get("evidence_strong") else "weak"


def should_continue(state) -> str:
    """ReAct 停止条件:无 tool_calls 就收敛,步数封顶则强停,否则继续。

    防打转靠步数:一步一次模型调用,数得清、好解释,各家 Agent SDK 给的也是
    max_turns / max_iterations 这类步数上限。token 花销是另一件事,见下面那段。"""
    last = state["messages"][-1]
    has_tool_calls = isinstance(last, AIMessage) and bool(last.tool_calls)
    if not has_tool_calls:
        return "stop"
    if state.get("steps", 0) >= settings.max_agent_steps:
        return "stop"
    return "continue"
```

- [ ] **Step 4: 跑验证通过**

Run: `.venv/bin/pytest tests/graph/test_routing.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/graph/routing.py tests/graph/test_routing.py
git commit -m "feat(ch05): 分流规则 + ReAct 停止条件"
```

---

## Task 6: 确定性出口节点(chitchat / complaint / fallback + 话术)

**Files:**
- Modify: `app/core/prompts.py`(加话术常量)
- Create: `app/graph/nodes.py`(先放这三个节点 + 话术)
- Test: `tests/graph/test_reply_nodes.py`

**Interfaces:**
- Produces(nodes.py 内):
  - `CHITCHAT_REPLY: str`、`COMPLAINT_REPLY: str`、`FALLBACK_REPLY: str`
  - `async def chitchat_reply(state) -> dict`
  - `async def complaint_reply(state) -> dict`
  - `async def fallback_reply(state) -> dict`
  - `def _user_text(state) -> str`(取最后一条 human 文本)

- [ ] **Step 1: 写失败测试**

`tests/graph/test_reply_nodes.py`:
```python
import pytest
from langchain_core.messages import HumanMessage

from app.graph import nodes


@pytest.mark.asyncio
async def test_chitchat_reply_fixed_no_actions():
    out = await nodes.chitchat_reply({"messages": [HumanMessage("你好")]})
    assert out["answer"] == nodes.CHITCHAT_REPLY
    assert out["trace"]["route"] == "chitchat"
    assert not out.get("suggested_actions")


@pytest.mark.asyncio
async def test_complaint_reply_offers_two_actions():
    out = await nodes.complaint_reply({"messages": [HumanMessage("我要投诉你们")], "conversation_id": 7})
    assert out["answer"] == nodes.COMPLAINT_REPLY
    types = [a["type"] for a in out["suggested_actions"]]
    assert types == ["transfer_human", "create_ticket"]
    ticket = next(a for a in out["suggested_actions"] if a["type"] == "create_ticket")
    assert ticket["draft"]["ticket_type"] == "投诉"
    assert ticket["draft"]["description"] == "我要投诉你们"


@pytest.mark.asyncio
async def test_fallback_reply_records_low_confidence(monkeypatch):
    calls = {}

    async def fake_insert(conversation_id, raw_question, source, reason):
        calls.update(conversation_id=conversation_id, raw=raw_question, source=source)
        return 1

    monkeypatch.setattr(nodes.repository, "insert_low_confidence", fake_insert)
    out = await nodes.fallback_reply(
        {"messages": [HumanMessage("怎么注销账号")], "conversation_id": 3,
         "trace": {"evidence_top": 0.1}}
    )
    assert out["answer"] == nodes.FALLBACK_REPLY
    assert calls["source"] == "retrieval_low_conf"
    assert calls["conversation_id"] == 3
```

- [ ] **Step 2: 跑验证失败**

Run: `.venv/bin/pytest tests/graph/test_reply_nodes.py -v`
Expected: FAIL

- [ ] **Step 3: 加话术 + 实现节点**

`app/core/prompts.py` 追加:
```python
CHITCHAT_REPLY_TEXT = "你好呀~我是喵喵优选的智能客服小喵。商品、订单、物流、售后都可以问我哦,有什么能帮您的?"
COMPLAINT_REPLY_TEXT = "非常抱歉给您带来了不好的体验,我理解您的心情。您可以选择转接人工客服,或让我为您登记一张工单跟进处理。"
FALLBACK_REPLY_TEXT = "抱歉,这个问题我暂时没有查到确切信息,不敢乱答。建议您联系人工客服进一步确认,以免给您错误的指引。"
```

`app/graph/nodes.py`(新建,先放这部分):
```python
import logging

from langchain_core.messages import HumanMessage

from app.core.prompts import (
    CHITCHAT_REPLY_TEXT, COMPLAINT_REPLY_TEXT, FALLBACK_REPLY_TEXT,
)
from app.db import repository

logger = logging.getLogger(__name__)

CHITCHAT_REPLY = CHITCHAT_REPLY_TEXT
COMPLAINT_REPLY = COMPLAINT_REPLY_TEXT
FALLBACK_REPLY = FALLBACK_REPLY_TEXT


def _user_text(state) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


async def chitchat_reply(state) -> dict:
    """闲聊:固定话术,零模型调用。"""
    return {"answer": CHITCHAT_REPLY, "trace": {"route": "chitchat"}}


async def complaint_reply(state) -> dict:
    """投诉:安抚话术 + 转人工/建工单两个可选项(后端不执行,交前端自选)。"""
    actions = [
        {"type": "transfer_human"},
        {"type": "create_ticket",
         "draft": {"description": _user_text(state), "ticket_type": "投诉"}},
    ]
    return {"answer": COMPLAINT_REPLY, "suggested_actions": actions,
            "trace": {"route": "complaint"}}


async def fallback_reply(state) -> dict:
    """置信度兜底:证据弱回兜底话术,并把问题落低置信池留给数据飞轮。"""
    reason = f"检索证据不足(top={state.get('trace', {}).get('evidence_top', 0):.3f})"
    await repository.insert_low_confidence(
        state.get("conversation_id"), _user_text(state), "retrieval_low_conf", reason
    )
    return {"answer": FALLBACK_REPLY, "trace": {"route": "fallback"}}
```

- [ ] **Step 4: 跑验证通过**

Run: `.venv/bin/pytest tests/graph/test_reply_nodes.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/core/prompts.py app/graph/nodes.py tests/graph/test_reply_nodes.py
git commit -m "feat(ch05): 闲聊/投诉/兜底三出口节点 + 话术"
```

---

## Task 7: 知识路强制检索节点 forced_rag

**Files:**
- Modify: `app/graph/nodes.py`(加 `forced_rag` + `coref` + `classify_intent`)
- Test: `tests/graph/test_forced_rag.py`

**Interfaces:**
- Consumes: `retrieval.search_knowledge`、`retrieval.arrange_head_tail`、`selfcheck.check_sufficient`、`query_understanding.understand`(均既有)、`settings.rerank_min_score`
- Produces:
  - `async def coref(state) -> dict`
  - `async def classify_intent(state) -> dict`
  - `async def forced_rag(state) -> dict`(设 `evidence_strong` / `evidence` / `citations` / trace)
  - `async def confidence_check(state) -> dict`(生成前证据闸的实体节点:留痕门控决策 strong/weak,判断本身在其后的条件边 `confidence_gate`)

- [ ] **Step 1: 写失败测试(mock 检索,测组装与证据闸)**

`tests/graph/test_forced_rag.py`:
```python
import pytest
from langchain_core.messages import HumanMessage

from app.graph import nodes


@pytest.mark.asyncio
async def test_forced_rag_strong_builds_evidence(monkeypatch):
    hits = [{"id": 1, "question": "退货政策", "answer": "7天无理由", "rerank_score": 0.9,
             "section_path": "政策/退货", "content_type": "policy"}]
    monkeypatch.setattr(nodes.query_understanding, "understand",
                        _aval({"standard": "退货政策", "expanded": []}))
    monkeypatch.setattr(nodes.retrieval, "search_knowledge", _aval(hits))
    monkeypatch.setattr(nodes.retrieval, "arrange_head_tail", lambda h: h)
    monkeypatch.setattr(nodes.selfcheck, "check_sufficient", _aval({"useful": True, "reason": ""}))

    out = await nodes.forced_rag({"messages": [HumanMessage("退货政策是什么")]})
    assert out["evidence_strong"] is True
    assert "[1]" in out["evidence"]
    assert out["citations"][0]["n"] == 1
    assert out["trace"]["forced_rag"] is True


@pytest.mark.asyncio
async def test_forced_rag_weak_when_below_threshold(monkeypatch):
    monkeypatch.setattr(nodes.query_understanding, "understand",
                        _aval({"standard": "注销账号", "expanded": []}))
    monkeypatch.setattr(nodes.retrieval, "search_knowledge", _aval([]))
    out = await nodes.forced_rag({"messages": [HumanMessage("怎么注销账号")]})
    assert out["evidence_strong"] is False
    assert out["trace"]["forced_rag"] is True


@pytest.mark.asyncio
async def test_confidence_check_traces_decision():
    assert (await nodes.confidence_check({"evidence_strong": True}))["trace"]["confidence"] == "strong"
    assert (await nodes.confidence_check({"evidence_strong": False}))["trace"]["confidence"] == "weak"


def _aval(value):
    async def _f(*a, **k):
        return value
    return _f
```

- [ ] **Step 2: 跑验证失败**

Run: `.venv/bin/pytest tests/graph/test_forced_rag.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 coref / classify_intent / forced_rag**

`app/graph/nodes.py` 顶部 import 追加:
```python
from app.config import settings
from app.core import intent as intent_mod
from app.core import query_understanding, retrieval, selfcheck
```
追加节点:
```python
async def coref(state) -> dict:
    """指代消解:本章最简,原样透传(正式版留 ch06)。"""
    return {"trace": {"coref": "passthrough"}}


async def classify_intent(state) -> dict:
    """意图识别:单标签七类。"""
    intent = await intent_mod.classify(_user_text(state))
    return {"intent": intent, "trace": {"intent": intent}}


async def forced_rag(state) -> dict:
    """知识类强制检索(复用 ch03/04 检索器):产出编号证据 + 证据强弱信号。
    复刻 query_faq 的两道生成前证据闸(检索分闸 + 自评闸),但把结果落进 State。"""
    query_raw = _user_text(state)
    u = await query_understanding.understand(query_raw)
    query = u["standard"]
    bm25_text = query + (" " + " ".join(u["expanded"]) if u["expanded"] else "")

    hits = await retrieval.search_knowledge(
        query, strategy="hybrid_rerank", bm25_text=bm25_text)
    top = hits[0]["rerank_score"] if hits else 0.0

    # 机械闸:无召回 or 最高分低于阈值
    if not hits or top < settings.rerank_min_score:
        return {"evidence_strong": False,
                "trace": {"forced_rag": True, "evidence_top": top}}

    # 语义闸:生成前自评证据够不够
    ev_texts = [f"{h['question']} {h['answer']}" for h in hits]
    chk = await selfcheck.check_sufficient(query, ev_texts)
    if not chk["useful"]:
        return {"evidence_strong": False,
                "trace": {"forced_rag": True, "evidence_top": top, "self_check": chk["reason"]}}

    arranged = retrieval.arrange_head_tail(hits)
    citations = [
        {"n": i + 1, "id": h["id"], "section_path": h["section_path"],
         "question": h["question"], "answer": h["answer"], "content_type": h["content_type"]}
        for i, h in enumerate(arranged)
    ]
    evidence = "\n".join(f"[{c['n']}] {c['question']}: {c['answer']}" for c in citations)
    return {"evidence_strong": True, "evidence": evidence, "citations": citations,
            "trace": {"forced_rag": True, "evidence_top": top}}


async def confidence_check(state) -> dict:
    """置信度兜底(生成前证据闸)的实体节点:留痕门控决策供可观测;
    强/弱的实际分流在其后的条件边 confidence_gate(读 evidence_strong)。"""
    decision = "strong" if state.get("evidence_strong") else "weak"
    return {"trace": {"confidence": decision}}
```

- [ ] **Step 4: 跑验证通过**

Run: `.venv/bin/pytest tests/graph/test_forced_rag.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/graph/nodes.py tests/graph/test_forced_rag.py
git commit -m "feat(ch05): coref 透传 + 意图识别 + 知识路强制检索节点"
```

---

## Task 8: 主力 Agent 节点(agent_llm / agent_tools + create_ticket 拦截 + token 累加)

**Files:**
- Modify: `app/graph/nodes.py`(加 `agent_llm` / `agent_tools` / `_agent_messages`)
- Test: `tests/graph/test_agent_nodes.py`

**Interfaces:**
- Consumes: `get_chat_model`(既有)、`get_all_tools`、`execute_tool_call`、`AGENT_SYSTEM`
- Produces:
  - `def _agent_messages(state) -> list`(system[+证据] + 历史)
  - `async def agent_llm(state, config=None) -> dict`(累加 steps/tokens_used,追加 AIMessage)
  - `async def agent_tools(state) -> dict`(执行工具;拦截 create_ticket 为提议)

- [ ] **Step 1: 写失败测试**

`tests/graph/test_agent_nodes.py`:
```python
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph import nodes


def test_agent_messages_injects_evidence_on_knowledge():
    msgs = nodes._agent_messages(
        {"route": "knowledge", "evidence": "[1] 退货: 7天",
         "messages": [HumanMessage("能退吗")]})
    assert isinstance(msgs[0], SystemMessage)
    assert "[1] 退货: 7天" in msgs[0].content
    assert "query_faq" in msgs[0].content  # 指示别再检索


def test_agent_messages_no_evidence_on_business():
    msgs = nodes._agent_messages({"route": "business", "messages": [HumanMessage("订单1001")]})
    assert isinstance(msgs[0], SystemMessage)
    assert "已检索到的知识证据" not in msgs[0].content


@pytest.mark.asyncio
async def test_agent_llm_accumulates_steps_and_tokens(monkeypatch):
    ai = AIMessage("好的", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

    class FakeModel:
        def bind_tools(self, tools):
            return self
        async def ainvoke(self, msgs, config=None):
            return ai

    monkeypatch.setattr(nodes, "get_chat_model", lambda **k: FakeModel())
    out = await nodes.agent_llm({"messages": [HumanMessage("hi")], "steps": 1, "tokens_used": 100})
    assert out["steps"] == 2
    assert out["tokens_used"] == 115
    assert out["messages"][0] is ai


@pytest.mark.asyncio
async def test_agent_tools_executes_normal_tool(monkeypatch):
    from app.tools.infra import ToolRun
    from langchain_core.messages import ToolMessage

    async def fake_exec(tc, cid):
        return ToolRun(tool_call_id=tc["id"], name=tc["name"], ok=True,
                       tool_message=ToolMessage(content='{"status":"已发货"}', tool_call_id=tc["id"], name=tc["name"]))

    monkeypatch.setattr(nodes, "execute_tool_call", fake_exec)
    ai = AIMessage("", tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "t1"}])
    out = await nodes.agent_tools({"messages": [ai], "conversation_id": 5})
    assert out["messages"][0].name == "query_logistics"
    assert not out.get("suggested_actions")


@pytest.mark.asyncio
async def test_agent_tools_intercepts_create_ticket(monkeypatch):
    async def fake_exec(tc, cid):
        raise AssertionError("create_ticket 不应被执行(应拦截为提议)")

    monkeypatch.setattr(nodes, "execute_tool_call", fake_exec)
    ai = AIMessage("", tool_calls=[{"name": "create_ticket",
                    "args": {"description": "屏幕碎了", "ticket_type": "售后"}, "id": "t9"}])
    out = await nodes.agent_tools({"messages": [ai], "conversation_id": 5})
    act = out["suggested_actions"][0]
    assert act["type"] == "create_ticket"
    assert act["draft"]["description"] == "屏幕碎了"
    # 回一条合成 ToolMessage 让模型收敛(不再调工具)
    assert out["messages"][0].tool_call_id == "t9"
```

- [ ] **Step 2: 跑验证失败**

Run: `.venv/bin/pytest tests/graph/test_agent_nodes.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 Agent 节点**

`app/graph/nodes.py` import 追加:
```python
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from app.core.llm import get_chat_model
from app.core.prompts import AGENT_SYSTEM
from app.tools.infra import execute_tool_call
from app.tools.registry import get_all_tools
```
追加:
```python
_KNOWLEDGE_EVIDENCE_HINT = (
    "\n\n## 已检索到的知识证据(请据此作答,每个关键结论后标注来源编号如[1];"
    "证据已给,不要再调用 query_faq;仍可按需调用订单/物流等工具)\n"
)


def _agent_messages(state) -> list:
    """system(知识路拼证据) + 跨轮历史。"""
    sys = AGENT_SYSTEM
    if state.get("route") == "knowledge" and state.get("evidence"):
        sys = AGENT_SYSTEM + _KNOWLEDGE_EVIDENCE_HINT + state["evidence"]
    return [SystemMessage(sys), *state.get("messages", [])]


async def agent_llm(state, config=None) -> dict:
    """ReAct 推理步:调模型(带工具),累加 steps 与 token 消耗。"""
    model = get_chat_model(streaming=True).bind_tools(get_all_tools())
    ai: AIMessage = await model.ainvoke(_agent_messages(state), config)
    used = (ai.usage_metadata or {}).get("total_tokens", 0) if ai.usage_metadata else 0
    return {"messages": [ai],
            "steps": state.get("steps", 0) + 1,
            "tokens_used": state.get("tokens_used", 0) + used}


async def agent_tools(state) -> dict:
    """ReAct 行动步:执行工具并回灌结果。create_ticket 拦截为『提议』——不写库,
    转成前端可选项,并回一条合成 ToolMessage 让模型收敛(真正写库在按钮端点)。"""
    last = state["messages"][-1]
    tool_msgs = []
    actions = list(state.get("suggested_actions", []))
    for tc in last.tool_calls:
        if tc["name"] == "create_ticket":
            draft = {"description": tc["args"].get("description", ""),
                     "ticket_type": tc["args"].get("ticket_type", "咨询")}
            actions.append({"type": "create_ticket", "draft": draft})
            tool_msgs.append(ToolMessage(
                content="已把『建工单』选项交给用户自行确认。请用一句话简要说明并停止,不要再调用任何工具。",
                tool_call_id=tc["id"], name="create_ticket"))
        else:
            run = await execute_tool_call(tc, state.get("conversation_id", 0))
            tool_msgs.append(run.tool_message)
    out = {"messages": tool_msgs}
    if actions:
        out["suggested_actions"] = actions
    return out
```

- [ ] **Step 4: 跑验证通过**

Run: `.venv/bin/pytest tests/graph/test_agent_nodes.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/graph/nodes.py tests/graph/test_agent_nodes.py
git commit -m "feat(ch05): 主力 Agent 节点 + create_ticket 拦截 + token 累加"
```

---

## Task 9: 日志节点 log

**Files:**
- Modify: `app/graph/nodes.py`(加 `log_node` + `resolve_answer`)
- Test: `tests/graph/test_log_node.py`

**Interfaces:**
- Produces:
  - `def resolve_answer(state) -> str`(确定性节点 answer 优先,否则取最后 AIMessage 文本)
  - `async def log_node(state) -> dict`(记 trace 日志 + 落 MySQL assistant 消息)

- [ ] **Step 1: 写失败测试**

`tests/graph/test_log_node.py`:
```python
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph import nodes


def test_resolve_answer_prefers_explicit():
    assert nodes.resolve_answer({"answer": "固定话术", "messages": []}) == "固定话术"


def test_resolve_answer_falls_back_to_last_ai():
    state = {"messages": [HumanMessage("hi"), AIMessage("模型答复")]}
    assert nodes.resolve_answer(state) == "模型答复"


@pytest.mark.asyncio
async def test_log_node_persists_assistant(monkeypatch):
    saved = {}

    async def fake_append(cid, role, content=None, **k):
        saved.update(cid=cid, role=role, content=content)
        return 1

    monkeypatch.setattr(nodes.repository, "append_message", fake_append)
    await nodes.log_node({"conversation_id": 8, "intent": "订单", "route": "business",
                          "messages": [AIMessage("订单已发货")], "trace": {"forced_rag": False}})
    assert saved["cid"] == 8
    assert saved["role"] == "assistant"
    assert saved["content"] == "订单已发货"
```

- [ ] **Step 2: 跑验证失败**

Run: `.venv/bin/pytest tests/graph/test_log_node.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 log_node**

`app/graph/nodes.py` 追加:
```python
def resolve_answer(state) -> str:
    """最终答复:确定性节点写在 state['answer'];Agent 答复取最后一条 AIMessage 文本。"""
    if state.get("answer"):
        return state["answer"]
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage):
            content = m.content
            if isinstance(content, str):
                return content
            return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


async def log_node(state) -> dict:
    """日志记录:留痕 intent/route/trace(可观测地基),并落 MySQL 一条 assistant 消息(审计)。"""
    logger.info(
        "ch05 turn conv=%s intent=%s route=%s trace=%s",
        state.get("conversation_id"), state.get("intent"), state.get("route"),
        state.get("trace", {}),
    )
    answer = resolve_answer(state)
    if state.get("conversation_id"):
        await repository.append_message(state["conversation_id"], "assistant",
                                        content=answer or None)
    return {}
```

- [ ] **Step 4: 跑验证通过**

Run: `.venv/bin/pytest tests/graph/test_log_node.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/graph/nodes.py tests/graph/test_log_node.py
git commit -m "feat(ch05): 日志记录节点(留痕 + 消息审计)"
```

---

## Task 10: 图组装 build_graph

**Files:**
- Create: `app/graph/build.py`
- Test: `tests/graph/test_build.py`

**Interfaces:**
- Consumes: 全部节点 + routing
- Produces: `def build_graph(checkpointer=None)`(返回 compiled graph);`def _builder()`(返回未编译 StateGraph,供结构测试)

- [ ] **Step 1: 写失败测试(测拓扑结构,不打模型)**

`tests/graph/test_build.py`:
```python
from app.graph.build import _builder


def test_graph_has_all_nodes():
    g = _builder().compile()
    names = set(g.get_graph().nodes)
    for n in ["coref", "classify_intent", "forced_rag", "confidence_check",
              "agent_llm", "agent_tools", "complaint_reply", "chitchat_reply",
              "fallback_reply", "log"]:
        assert n in names, f"缺节点 {n}"


def test_graph_compiles_with_checkpointer():
    from langgraph.checkpoint.memory import InMemorySaver
    g = _builder().compile(checkpointer=InMemorySaver())
    assert g is not None
```

- [ ] **Step 2: 跑验证失败**

Run: `.venv/bin/pytest tests/graph/test_build.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 build_graph**

`app/graph/build.py`:
```python
from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.routing import confidence_gate, route_by_intent, should_continue
from app.graph.state import ConversationState


def _builder() -> StateGraph:
    b = StateGraph(ConversationState)
    # 节点
    b.add_node("coref", nodes.coref)
    b.add_node("classify_intent", nodes.classify_intent)
    b.add_node("forced_rag", nodes.forced_rag)
    b.add_node("confidence_check", nodes.confidence_check)  # 实体节点(trace 门控);判断在其后条件边
    b.add_node("agent_llm", nodes.agent_llm)
    b.add_node("agent_tools", nodes.agent_tools)
    b.add_node("complaint_reply", nodes.complaint_reply)
    b.add_node("chitchat_reply", nodes.chitchat_reply)
    b.add_node("fallback_reply", nodes.fallback_reply)
    b.add_node("log", nodes.log_node)

    # 骨架:消解 → 意图 → 分流
    b.add_edge(START, "coref")
    b.add_edge("coref", "classify_intent")
    b.add_conditional_edges("classify_intent", route_by_intent, {
        "knowledge": "forced_rag",
        "business": "agent_llm",
        "complaint": "complaint_reply",
        "chitchat": "chitchat_reply",
    })
    # 知识路:检索 → 生成前证据闸
    b.add_edge("forced_rag", "confidence_check")
    b.add_conditional_edges("confidence_check", confidence_gate, {
        "strong": "agent_llm",
        "weak": "fallback_reply",
    })
    # 主力 Agent ReAct 环
    b.add_conditional_edges("agent_llm", should_continue, {
        "continue": "agent_tools",
        "stop": "log",
    })
    b.add_edge("agent_tools", "agent_llm")
    # 确定性出口 → 日志 → END
    b.add_edge("complaint_reply", "log")
    b.add_edge("chitchat_reply", "log")
    b.add_edge("fallback_reply", "log")
    b.add_edge("log", END)
    return b


def build_graph(checkpointer=None):
    return _builder().compile(checkpointer=checkpointer)
```

- [ ] **Step 4: 跑验证通过**

Run: `.venv/bin/pytest tests/graph/test_build.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/graph/build.py tests/graph/test_build.py
git commit -m "feat(ch05): StateGraph 骨架组装(六段接力 + ReAct 环)"
```

---

## Task 11: 运行时——checkpointer 生命周期 + 图单例 + run_turn/stream_turn

**Files:**
- Create: `app/graph/runtime.py`
- Test: `tests/graph/test_runtime.py`

**Interfaces:**
- Consumes: `build_graph`、`repository`(会话生命周期)
- Produces:
  - `async def init_graph() -> None`(lifespan 起:开 AsyncSqliteSaver + setup + 编译图,存模块单例)
  - `async def close_graph() -> None`
  - `def get_graph()`(取单例;未初始化则报错)
  - `class ConversationNotFound(Exception)`
  - `async def run_turn(user_id, message, conversation_id) -> dict`(返回 `{"conversation_id":int, "state":dict}`)
  - `async def stream_turn(user_id, message, conversation_id) -> AsyncIterator[dict]`(产出事件 dict,见下)

  事件类型:`{"type":"tool","name":str}`、`{"type":"citations","items":list}`、`{"type":"delta","text":str}`、`{"type":"actions","items":list}`、`{"type":"done","conversation_id":int}`

- [ ] **Step 1: 写失败测试(用 InMemorySaver 注入,mock 会话与图)**

`tests/graph/test_runtime.py`:
```python
import pytest

from app.graph import runtime


@pytest.mark.asyncio
async def test_run_turn_creates_conversation_when_none(monkeypatch):
    async def fake_create(uid): return 42
    async def fake_append(cid, role, content=None, **k): return 1

    class FakeGraph:
        async def ainvoke(self, inp, config):
            return {"answer": "hi", "messages": [], "conversation_id": inp["conversation_id"]}

    monkeypatch.setattr(runtime.repository, "create_conversation", fake_create)
    monkeypatch.setattr(runtime.repository, "append_message", fake_append)
    monkeypatch.setattr(runtime, "get_graph", lambda: FakeGraph())

    out = await runtime.run_turn("u1", "你好", None)
    assert out["conversation_id"] == 42


@pytest.mark.asyncio
async def test_run_turn_raises_when_conversation_missing(monkeypatch):
    async def fake_get(cid): return None
    monkeypatch.setattr(runtime.repository, "get_conversation", fake_get)
    with pytest.raises(runtime.ConversationNotFound):
        await runtime.run_turn("u1", "hi", 999)


@pytest.mark.asyncio
async def test_stream_turn_maps_events(monkeypatch):
    async def fake_create(uid): return 7
    async def fake_append(cid, role, content=None, **k): return 1
    monkeypatch.setattr(runtime.repository, "create_conversation", fake_create)
    monkeypatch.setattr(runtime.repository, "append_message", fake_append)

    from langchain_core.messages import AIMessage, ToolMessage

    class FakeGraph:
        async def astream(self, inp, config, stream_mode=None):
            # messages 模式:agent_llm 的 token
            yield ("messages", (AIMessage("已"), {"langgraph_node": "agent_llm"}))
            yield ("messages", (AIMessage("发货"), {"langgraph_node": "agent_llm"}))
            # messages 模式:非答复节点 token 应被过滤
            yield ("messages", (AIMessage("订单"), {"langgraph_node": "classify_intent"}))
            # updates 模式:工具帧 + citations + actions
            yield ("updates", {"agent_tools": {"messages": [
                ToolMessage(content="{}", tool_call_id="1", name="query_logistics")]}})
            yield ("updates", {"forced_rag": {"citations": [{"n": 1}]}})
            yield ("updates", {"complaint_reply": {"answer": "抱歉",
                    "suggested_actions": [{"type": "transfer_human"}]}})

    monkeypatch.setattr(runtime, "get_graph", lambda: FakeGraph())
    events = [e async for e in runtime.stream_turn("u1", "订单1001物流", None)]
    kinds = [e["type"] for e in events]
    deltas = [e["text"] for e in events if e["type"] == "delta"]
    tools = [e["name"] for e in events if e["type"] == "tool"]
    assert "已" in deltas and "发货" in deltas
    assert "订单" not in deltas  # 非答复节点被过滤
    assert tools == ["query_logistics"]
    assert any(e["type"] == "citations" for e in events)
    assert any(e["type"] == "actions" for e in events)
    assert kinds[-1] == "done" and events[-1]["conversation_id"] == 7
```

- [ ] **Step 2: 跑验证失败**

Run: `.venv/bin/pytest tests/graph/test_runtime.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 runtime(流式形状以 Task 1 冒烟结论为准)**

`app/graph/runtime.py`:
```python
import logging
from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import settings
from app.db import repository
from app.graph.build import build_graph

logger = logging.getLogger(__name__)

# 会流式吐答复 token 的节点(其余节点的模型调用 token 不进 delta)
ANSWER_NODES = {"agent_llm"}
# 确定性节点把答复写在 state['answer'],整块作为 delta 吐出
DETERMINISTIC_ANSWER_NODES = {"chitchat_reply", "complaint_reply", "fallback_reply"}

_graph = None
_cm = None  # AsyncSqliteSaver context manager,持有以免被 GC


class ConversationNotFound(Exception):
    pass


async def init_graph() -> None:
    """lifespan 起:开持久 checkpointer + setup + 编译图。"""
    global _graph, _cm
    _cm = AsyncSqliteSaver.from_conn_string(settings.checkpointer_db_path)
    checkpointer = await _cm.__aenter__()
    await checkpointer.setup()
    _graph = build_graph(checkpointer=checkpointer)
    logger.info("ch05 图已编译,checkpointer=%s", settings.checkpointer_db_path)


async def close_graph() -> None:
    global _graph, _cm
    if _cm is not None:
        await _cm.__aexit__(None, None, None)
        _cm = None
    _graph = None


def get_graph():
    if _graph is None:
        raise RuntimeError("图未初始化,应在 FastAPI lifespan 调 init_graph()")
    return _graph


async def _ensure_conversation(user_id: str, conversation_id: int | None) -> int:
    if conversation_id is None:
        return await repository.create_conversation(user_id)
    if await repository.get_conversation(conversation_id) is None:
        raise ConversationNotFound(conversation_id)
    return conversation_id


def _graph_input(user_id: str, message: str, cid: int) -> dict:
    return {"messages": [HumanMessage(message)], "user_id": user_id,
            "conversation_id": cid, "steps": 0, "tokens_used": 0}


async def run_turn(user_id, message, conversation_id) -> dict:
    """非流式:落 user 消息 → ainvoke → 返回终态(供 /api/agent、eval)。"""
    cid = await _ensure_conversation(user_id, conversation_id)
    await repository.append_message(cid, "user", content=message)
    config = {"configurable": {"thread_id": str(cid)}}
    final = await get_graph().ainvoke(_graph_input(user_id, message, cid), config)
    return {"conversation_id": cid, "state": final}


async def stream_turn(user_id, message, conversation_id) -> AsyncIterator[dict]:
    """流式:落 user 消息 → astream 多模式 → 映射成事件 dict(供 /api/chat 转 SSE)。"""
    cid = await _ensure_conversation(user_id, conversation_id)
    await repository.append_message(cid, "user", content=message)
    config = {"configurable": {"thread_id": str(cid)}}

    actions: list = []
    async for mode, chunk in get_graph().astream(
        _graph_input(user_id, message, cid), config,
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            msg, meta = chunk
            if meta.get("langgraph_node") in ANSWER_NODES:
                text = msg.content if isinstance(msg.content, str) else ""
                if text:
                    yield {"type": "delta", "text": text}
        elif mode == "updates":
            for node, upd in chunk.items():
                if not isinstance(upd, dict):
                    continue
                if node in DETERMINISTIC_ANSWER_NODES and upd.get("answer"):
                    yield {"type": "delta", "text": upd["answer"]}
                if node == "forced_rag" and upd.get("citations"):
                    yield {"type": "citations", "items": upd["citations"]}
                if node == "agent_tools":
                    for m in upd.get("messages", []):
                        name = getattr(m, "name", None)
                        if name and name != "create_ticket":
                            yield {"type": "tool", "name": name}
                if upd.get("suggested_actions"):
                    actions = upd["suggested_actions"]
    if actions:
        yield {"type": "actions", "items": actions}
    yield {"type": "done", "conversation_id": cid}
```

- [ ] **Step 4: 跑验证通过**

Run: `.venv/bin/pytest tests/graph/test_runtime.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/graph/runtime.py tests/graph/test_runtime.py
git commit -m "feat(ch05): 图运行时(checkpointer 生命周期 + run_turn/stream_turn 事件映射)"
```

---

## Task 12: /api/agent 接图(非流式)+ schema 加 suggested_actions

**Files:**
- Modify: `app/schemas/agent.py`
- Modify: `app/api/agent.py`
- Test: `tests/test_agent_api.py`(改造)、`tests/graph/test_agent_view.py`

**Interfaces:**
- Consumes: `runtime.run_turn`
- Produces:
  - `AgentResponse.suggested_actions: list`(默认 `[]`)
  - `def _views_from_state(state) -> tuple[list[ToolCallView], list[ToolResultView]]`(从终态 messages 重建工具轨迹)

- [ ] **Step 1: 写失败测试**

`tests/graph/test_agent_view.py`:
```python
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.api.agent import _views_from_state


def test_views_rebuilt_from_messages():
    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {"order_id": "1"}, "id": "c1"}])
    tm = ToolMessage(content='{"status":"已发货"}', tool_call_id="c1", name="query_order")
    calls, results = _views_from_state({"messages": [HumanMessage("q"), ai, tm, AIMessage("答")]})
    assert calls[0].name == "query_order"
    assert results[0].tool_call_id == "c1"
    assert results[0].name == "query_order"
```

改 `tests/test_agent_api.py`:把断言基线从旧两次调用编排改为「经图 ainvoke,返回 answer + tool_calls」;闲聊问句断言 `tool_calls == []`。(具体断言随图行为定,保留原有 404/DB 错/上游错三条错误路径测试。)

- [ ] **Step 2: 跑验证失败**

Run: `.venv/bin/pytest tests/graph/test_agent_view.py -v`
Expected: FAIL

- [ ] **Step 3: 改 schema + api/agent**

`app/schemas/agent.py`:`AgentResponse` 加字段:
```python
    suggested_actions: list = Field(default_factory=list)
```
(确认文件顶部已 import `Field`;未 import 则补 `from pydantic import BaseModel, Field`。)

`app/api/agent.py` 重写:
```python
import logging

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, ToolMessage
from sqlalchemy.exc import SQLAlchemyError

from app.graph import runtime
from app.schemas.agent import AgentRequest, AgentResponse, ToolCallView, ToolResultView

logger = logging.getLogger(__name__)
router = APIRouter()


def _views_from_state(state):
    """从终态 messages 重建工具调用/结果轨迹(供评估/单测看模型选了什么工具)。"""
    calls, results = [], []
    for m in state.get("messages", []):
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                calls.append(ToolCallView(id=tc["id"], name=tc["name"], args=tc["args"]))
        elif isinstance(m, ToolMessage):
            results.append(ToolResultView(
                tool_call_id=m.tool_call_id, name=m.name or "", ok=(m.status != "error"),
                content=m.content))
    return calls, results


@router.post("/api/agent", response_model=AgentResponse)
async def run_agent(req: AgentRequest) -> AgentResponse:
    try:
        out = await runtime.run_turn(req.user_id, req.message, req.conversation_id)
    except runtime.ConversationNotFound:
        raise HTTPException(status_code=404, detail="会话不存在")
    except SQLAlchemyError:
        logger.exception("数据库错误 user_id=%s", req.user_id)
        raise HTTPException(status_code=503, detail="数据库暂时不可用,请稍后重试")
    except Exception:
        logger.exception("图编排失败 user_id=%s", req.user_id)
        raise HTTPException(status_code=502, detail="上游模型暂时不可用,请稍后重试")

    state = out["state"]
    calls, results = _views_from_state(state)
    from app.graph.nodes import resolve_answer
    return AgentResponse(
        conversation_id=out["conversation_id"],
        answer=resolve_answer(state),
        tool_calls=calls,
        tool_results=results,
        suggested_actions=state.get("suggested_actions", []),
    )
```

- [ ] **Step 4: 跑验证通过**

Run: `.venv/bin/pytest tests/graph/test_agent_view.py tests/test_agent_api.py -v`
Expected: PASS(test_agent_api 需图初始化——若测试未起 lifespan,用 fixture monkeypatch `runtime.get_graph` 注入 InMemorySaver 编译的图,或直接注入假图;沿用 conftest 既有 DB fixture)

- [ ] **Step 5: Commit**

```bash
git add app/schemas/agent.py app/api/agent.py tests/graph/test_agent_view.py tests/test_agent_api.py
git commit -m "feat(ch05): /api/agent 走图 ainvoke + suggested_actions"
```

---

## Task 13: /api/chat 接图(流式 SSE)+ actions 帧

**Files:**
- Modify: `app/api/chat.py`
- Test: `tests/test_chat_api.py`(改造)

**Interfaces:**
- Consumes: `runtime.stream_turn`
- Produces:SSE 帧多一种 `{"event":"actions","items":[...]}`

- [ ] **Step 1: 写失败测试**

改 `tests/test_chat_api.py`:monkeypatch `runtime.stream_turn` 产出一串事件(含 delta/tool/citations/actions/done),断言 HTTP 流里出现 `"event": "actions"`、`"delta"`、`[DONE]`,且 done 帧带 conversation_id。保留 error 帧路径测试。示例核心断言:
```python
def test_chat_stream_includes_actions(client, monkeypatch):
    async def fake_stream(user_id, message, conversation_id):
        yield {"type": "delta", "text": "抱歉"}
        yield {"type": "actions", "items": [{"type": "transfer_human"},
                                            {"type": "create_ticket", "draft": {}}]}
        yield {"type": "done", "conversation_id": 5}

    from app.graph import runtime
    monkeypatch.setattr(runtime, "stream_turn", fake_stream)
    r = client.post("/api/chat", json={"user_id": "u1", "message": "我要投诉"})
    body = r.text
    assert '"event": "actions"' in body
    assert '"delta": "\\u62b1\\u6b49"' in body or "抱歉" in body
    assert "[DONE]" in body
```

- [ ] **Step 2: 跑验证失败**

Run: `.venv/bin/pytest tests/test_chat_api.py -v`
Expected: FAIL

- [ ] **Step 3: 改 api/chat**

`app/api/chat.py` 重写事件循环(移除对旧 `agent` 模块的依赖,改调 `runtime.stream_turn`):
```python
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import SQLAlchemyError

from app.graph import runtime
from app.schemas.chat import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_error(message: str) -> AsyncIterator[str]:
    yield "event: error\n"
    yield _sse({"message": message})


@router.post("/api/chat")
async def chat(req: ChatRequest):
    async def event_stream() -> AsyncIterator[str]:
        try:
            async for ev in runtime.stream_turn(req.user_id, req.message, req.conversation_id):
                if ev["type"] == "tool":
                    yield _sse({"event": "tool", "name": ev["name"]})
                elif ev["type"] == "delta":
                    yield _sse({"delta": ev["text"]})
                elif ev["type"] == "citations":
                    yield _sse({"event": "citations", "items": ev["items"]})
                elif ev["type"] == "actions":
                    yield _sse({"event": "actions", "items": ev["items"]})
                elif ev["type"] == "done":
                    yield _sse({"event": "done", "conversation_id": ev["conversation_id"]})
        except runtime.ConversationNotFound:
            for f in _sse_error("会话不存在"):
                yield f
            return
        except SQLAlchemyError:
            logger.exception("数据库错误 user_id=%s", req.user_id)
            for f in _sse_error("数据库暂时不可用,请稍后重试"):
                yield f
            return
        except Exception:
            logger.exception("图编排失败 user_id=%s", req.user_id)
            for f in _sse_error("上游模型暂时不可用,请稍后重试"):
                yield f
            return
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

- [ ] **Step 4: 跑验证通过**

Run: `.venv/bin/pytest tests/test_chat_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/api/chat.py tests/test_chat_api.py
git commit -m "feat(ch05): /api/chat 走图 astream + actions SSE 帧"
```

---

## Task 14: 建工单按钮端点 /api/actions/create-ticket

**Files:**
- Create: `app/api/actions.py`
- Create: `app/schemas/actions.py`
- Test: `tests/test_actions_api.py`

**Interfaces:**
- Consumes: ch02 `create_ticket` 工具(经 `repository.create_ticket`)
- Produces:`POST /api/actions/create-ticket`,body `{conversation_id:int, description:str, ticket_type:str}`,返回 `{ticket_no:str, status:"已转人工"}`

- [ ] **Step 1: 写失败测试**

`tests/test_actions_api.py`:
```python
def test_create_ticket_action_writes(client, monkeypatch):
    async def fake_create(cid, desc, ttype):
        assert cid == 5 and ttype == "投诉"
        return "T20260715001"

    from app.api import actions
    monkeypatch.setattr(actions.repository, "create_ticket", fake_create)
    r = client.post("/api/actions/create-ticket", json={
        "conversation_id": 5, "description": "东西坏了", "ticket_type": "投诉"})
    assert r.status_code == 200
    assert r.json()["ticket_no"] == "T20260715001"


def test_create_ticket_action_rejects_bad_type(client):
    r = client.post("/api/actions/create-ticket", json={
        "conversation_id": 5, "description": "x", "ticket_type": "乱填"})
    assert r.status_code == 422
```

- [ ] **Step 2: 跑验证失败**

Run: `.venv/bin/pytest tests/test_actions_api.py -v`
Expected: FAIL

- [ ] **Step 3: 实现端点**

`app/schemas/actions.py`:
```python
from typing import Literal

from pydantic import BaseModel, Field


class CreateTicketRequest(BaseModel):
    conversation_id: int
    description: str = Field(min_length=1)
    ticket_type: Literal["售后", "投诉", "咨询"]


class CreateTicketResponse(BaseModel):
    ticket_no: str
    status: str = "已转人工"
```

`app/api/actions.py`:
```python
import logging

from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from app.db import repository
from app.schemas.actions import CreateTicketRequest, CreateTicketResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/actions/create-ticket", response_model=CreateTicketResponse)
async def create_ticket_action(req: CreateTicketRequest) -> CreateTicketResponse:
    """建工单按钮:用户点了才写 tickets 表(复用 ch02 工单能力)。转人工是另一件事,纯前端。"""
    try:
        ticket_no = await repository.create_ticket(
            req.conversation_id, req.description, req.ticket_type)
    except SQLAlchemyError:
        logger.exception("建工单失败 conv=%s", req.conversation_id)
        raise HTTPException(status_code=503, detail="工单系统暂时不可用,请稍后重试")
    return CreateTicketResponse(ticket_no=ticket_no)
```

- [ ] **Step 4: 跑验证通过**

Run: `.venv/bin/pytest tests/test_actions_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/api/actions.py app/schemas/actions.py tests/test_actions_api.py
git commit -m "feat(ch05): 建工单按钮端点 /api/actions/create-ticket"
```

---

## Task 15: main.py 接线(lifespan 起图 + 挂 actions 路由)

**Files:**
- Modify: `app/main.py`
- Test: `tests/test_static.py`(既有,回归)

**Interfaces:**
- Consumes: `runtime.init_graph`/`close_graph`、`actions.router`

- [ ] **Step 1: 改 main.py lifespan + 路由**

`app/main.py`:在 lifespan 里 Milvus 预热之外加图初始化;挂 actions 路由:
```python
from app.api.actions import router as actions_router
from app.graph import runtime
```
lifespan 内(预热 Milvus 后、`yield` 前)加:
```python
    await runtime.init_graph()
```
`yield` 后加:
```python
    await runtime.close_graph()
```
路由注册区加:
```python
app.include_router(actions_router)
```

- [ ] **Step 2: 跑回归 + 启动冒烟**

Run: `.venv/bin/pytest tests/test_static.py -v`;并 `make dev` 后 `curl -s localhost:8000/docs` 确认 `/api/actions/create-ticket` 在 schema 里、应用能起(lifespan 起图无异常)。
Expected: PASS + 应用正常起。

- [ ] **Step 3: Commit**

```bash
git add app/main.py
git commit -m "feat(ch05): lifespan 起图 checkpointer + 挂 actions 路由"
```

---

## Task 16: 前端——actions 帧渲染 + 转人工/建工单两按钮(浏览器验收)

建工单交互从「点按钮直接用后端草稿一键建单」改为「点按钮弹出工单创建表单弹窗」——反馈类别(下拉 售后/投诉/咨询,不预选、必选)+ 反馈描述(留空、必填),用户当场填写、前端校验后再调 `/api/actions/create-ticket`。后端端点零改动。详见 spec 6.3 与 dev-notes 阶段 7,浏览器已端到端验收(截图 `dev-notes/ch05-ticket-form*.png`)。

**目的:** req#8 前端。产出是 UI + 交互 → 浏览器手动验收代替单测(memory:功能必须落到前端入口)。

**Files:**
- Modify: `app/static/index.html`

**Interfaces:**
- Consumes:SSE `{"event":"actions","items":[...]}`;`POST /api/actions/create-ticket`

- [ ] **Step 1: streamChat 解析 actions 帧**

在 `streamChat` 的帧分发里(`data.event === "citations"` 分支旁)加:
```javascript
          else if (data.event === "actions") onActions(data.items || []);
```
并给 `streamChat` 增加 `onActions` 形参、`send` 里传入回调。

- [ ] **Step 2: 渲染两个独立按钮 + 交互**

在 `send()` 里加 `onActions` 回调:收到 actions 在当前 bot 气泡下渲染按钮容器;每个 action 一个独立按钮;点击行为如下(complaint 与 agent 提议共用):
```javascript
      function renderActions(bubble, items) {
        const bar = document.createElement("div");
        bar.className = "action-bar";
        for (const a of items) {
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "action-btn";
          if (a.type === "transfer_human") {
            btn.textContent = "转人工";
            btn.addEventListener("click", () => {
              btn.disabled = true;
              // 纯前端模拟:显示已转接 + 客服小猫问候(本章不接真人系统)
              const sys = addRow("bot"); sys.textContent = "已转接人工客服";
              const greet = addRow("bot");
              greet.textContent = "您好,我是客服小猫,请问有什么可以帮您的";
              messagesEl.scrollTop = messagesEl.scrollHeight;
            });
          } else if (a.type === "create_ticket") {
            btn.textContent = "建工单";
            btn.addEventListener("click", async () => {
              btn.disabled = true;
              try {
                const resp = await fetch("/api/actions/create-ticket", {
                  method: "POST", headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({
                    conversation_id: getConversationId(),
                    description: (a.draft && a.draft.description) || "用户申请工单",
                    ticket_type: (a.draft && a.draft.ticket_type) || "咨询",
                  }),
                });
                const data = await resp.json();
                const ok = addRow("bot");
                ok.textContent = resp.ok ? ("工单已创建:" + data.ticket_no) : "工单创建失败,请稍后重试";
              } catch (e) {
                addRow("bot").textContent = "工单创建失败,请稍后重试";
              }
              messagesEl.scrollTop = messagesEl.scrollHeight;
            });
          }
          bar.appendChild(btn);
        }
        bubble.appendChild(bar);
      }
```
在 `send` 的流结束处(`addFeedbackBar` 附近)按收到的 actions 调 `renderActions(bubble, actions)`;两按钮互不绑定,用户不点继续发消息正常走(不禁用输入框)。

- [ ] **Step 3: 加按钮样式**

在 `<style>` 里加 `.action-bar`(flex, gap)、`.action-btn`(次要按钮样式,与既有 chip 风格协调)、`:disabled` 灰态。

- [ ] **Step 4: 浏览器验收(需全服务起)**

起 `docker compose up -d` + `make milvus-up` + `make kb-build`(如需)+ `make dev`;浏览器开 `localhost:8000`:
- 问「我要投诉」→ 出安抚话术 + 「转人工」「建工单」两独立按钮;
- 点「转人工」→ 显示「已转接人工客服」+「您好,我是客服小猫…」;
- 重新问一次、点「建工单」→ 显示「工单已创建 T…」;`SELECT * FROM tickets ORDER BY created_at DESC LIMIT 1` 见新行;
- 都不点、继续发消息 → 正常对话。
截图存 `dev-notes/ch05-acceptance3-*.png`。

- [ ] **Step 5: Commit**

```bash
git add app/static/index.html && git add -f dev-notes/ch05-acceptance3-*.png
git commit -m "feat(ch05): 前端 actions 渲染 + 转人工/建工单两按钮交互"
```

---

## Task 17: 五验收端到端评估脚本 + 真实验收

**目的:** 端到端真实验收(非纸面)。5 条验收在真实服务上跑通并留痕。

**Files:**
- Create: `scripts/eval_ch05.py`
- Modify: `Makefile`(加 `eval-ch05` 目标)

- [ ] **Step 1: 写端到端评估脚本**

`scripts/eval_ch05.py`:走 `/api/agent`(非流式,看得到工具轨迹与最终态)+ 直连图跑五条验收,断言路径/节点:
```python
"""ch05 五验收端到端评估。需全服务起(mysql/milvus/app)。
用法:.venv/bin/python -m scripts.eval_ch05"""
import asyncio
import json

import httpx

BASE = "http://localhost:8000"


async def agent(client, msg, cid=None):
    r = await client.post(f"{BASE}/api/agent",
                          json={"user_id": "eval-ch05", "message": msg, "conversation_id": cid})
    return r.json()


async def main():
    async with httpx.AsyncClient(timeout=120) as c:
        results = []

        # 验收2:业务数据类,Agent 自调 query_logistics
        b = await agent(c, "订单1001的物流到哪了")
        names = {tc["name"] for tc in b["tool_calls"]}
        results.append(("验收2 物流自调工具", "query_logistics" in names, names))

        # 验收3:投诉出两可选项(转人工/建工单),后端不自动建单
        b = await agent(c, "我要投诉,你们太差了")
        types = {a["type"] for a in b.get("suggested_actions", [])}
        results.append(("验收3 投诉出两按钮", {"transfer_human", "create_ticket"} <= types, types))

        # 验收4:闲聊固定话术,零工具
        b = await agent(c, "你好呀")
        results.append(("验收4 闲聊固定话术", not b["tool_calls"] and bool(b["answer"]), b["answer"][:20]))

        # 验收5:复杂问 ReAct 多步(先查订单再查物流)
        b = await agent(c, "我手机尾号1001那个订单发货没?到哪了?")
        results.append(("验收5 ReAct 多步", len(b["tool_calls"]) >= 2, [tc["name"] for tc in b["tool_calls"]]))

        for name, ok, detail in results:
            print(f"{'✅' if ok else '❌'} {name} -> {detail}")

    print("\n验收1(强制检索节点被走到)看 app 日志:问『退货政策是什么』后应见 "
          "`ch05 turn ... route=knowledge trace={...forced_rag: True...}`")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: 加 Makefile 目标**

`Makefile` 加:
```makefile
eval-ch05:  ## ch05 五验收端到端评估(需全服务起)
	.venv/bin/python -m scripts.eval_ch05
```

- [ ] **Step 3: 真实端到端跑五验收**

起全服务(`docker compose up -d`、`make milvus-up`、建库、`make dev`),然后:
- Run: `make eval-ch05` → 验收 2/3/4/5 打勾;
- 验收1:问「退货政策是什么」后 `grep "route=knowledge" ` app 日志见 `forced_rag: True`;
- 验收3 前端部分见 Task 16。
把结果 + 抖动如实记 `dev-notes/ch05.md`(阶段:finish)。glm 抖动重跑记录。

- [ ] **Step 4: 全量回归**

Run: `make milvus-up && .venv/bin/pytest -q`
Expected: ch01–05 全绿(旧 agent 编排测试已改为测图)。

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_ch05.py Makefile && git add -f dev-notes/ch05.md
git commit -m "test(ch05): 五验收端到端评估脚本 + 真实验收留痕"
```

---

## Task 18: 收尾清理与 README/dev-notes

**Files:**
- Modify: `app/core/agent.py`(退役旧编排:删 `run_agent_turn`/`stream_agent_turn` 或留薄壳;迁走仍被复用的 helper)
- Modify: `README.md`(补 ch05 验收命令)
- Modify: `dev-notes/ch05.md`(finish 段)

- [ ] **Step 1: 处理旧 agent.py**

确认 `app/core/agent.py` 的 `run_agent_turn`/`stream_agent_turn` 已无引用(全被 runtime 取代):
Run: `grep -rn "run_agent_turn\|stream_agent_turn\|core.agent\|core import agent" app tests scripts`
- 若仅测试引用 → 删这两个函数及其私有 helper;`_faq_result`/`_gen_tool_messages` 若被 nodes 复用则迁进 `app/graph/nodes.py`,否则删。
- 相应删除/改写 `tests/test_agent_orchestration.py`、`tests/test_agent_stream.py`(改为测图节点/runtime,或删除已被 tests/graph 覆盖的用例)。

- [ ] **Step 2: 跑全量回归确认无断链**

Run: `make milvus-up && .venv/bin/pytest -q`
Expected: 全绿,无 import 错。

- [ ] **Step 3: README + dev-notes**

`README.md` 补 ch05 验收命令块(make eval-ch05、日志看 forced_rag、前端投诉两按钮、复杂问 ReAct 多步)。`dev-notes/ch05.md` 追 finish 段(测试结果、五验收结论、演示命令、翻车汇总)。

- [ ] **Step 4: Commit**

```bash
git add app/core/agent.py app/api tests README.md && git add -f dev-notes/ch05.md
git commit -m "chore(ch05): 退役旧两次调用编排 + README/dev-notes 收尾"
```

---

## Self-Review

**Spec 覆盖核对:**
- §1 目标 → Task 1–18 整体;§2 架构数据流 → Task 10 组装;§3 十节点 → Task 6/7/8/9/10;§3.1 7→4 分流 → Task 5;§4 ReAct 停止/token → Task 5(should_continue)+ Task 8(累加);§5 State+AsyncSqliteSaver → Task 3 + Task 11;§6 转人工/建工单动作机制 → Task 8(拦截)+ Task 14(端点)+ Task 16(前端);§7 两入口走图 → Task 12/13;§8 依赖与验证 → Task 1 + eval 脚本 Task 4/17;§9 数据模型无新表 → 复用;§10 文件布局 → 文件结构节;§11 决策 D1–D6 → 各任务;§12 风险闸 R1(流式过滤 Task 11 ANSWER_NODES + Task 1 冒烟)/R2(Task 1/11 checkpointer)/R3(Task 8 拦截 + Task 17 评估);§13 五验收 → Task 16/17。
- **无遗漏。** 置信度位置已按修订版(生成前证据闸,forced_rag→confidence_check→conditional)落在 Task 10 组装。

**占位符扫描:** 无 TBD/TODO;每个代码步给了完整代码,每个测试步给了真实断言。Task 12 test_agent_api 改造与 Task 18 旧测试处理写明了判断依据(grep 引用后决定删/改),非模糊占位。

**类型一致性:** `ConversationState` 字段(Task 3)↔ 各节点读写一致;`route_by_intent`/`confidence_gate`/`should_continue` 返回值(Task 5)↔ build.py 条件边映射键(Task 10)一致("knowledge/business/complaint/chitchat"、"strong/weak"、"continue/stop");`suggested_actions` 结构(`{type, draft?}`)在 Task 6/8/12/13/16 一致;`run_turn` 返回 `{conversation_id, state}`(Task 11)↔ Task 12 消费一致;事件 dict 类型(Task 11)↔ Task 13 SSE 映射一致。
