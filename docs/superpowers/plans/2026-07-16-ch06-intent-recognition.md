# Ch06 意图识别与对话管理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ch05 骨架里占位的「分流器前段」做成正式版:指代消解+Query 改写、意图四件套(8 类含「其他」+ confidence)、五出口路由、退款退货/售后的确定性子流程(interrupt 弹订单选择器 → Query 扩写强制检索政策 → 主力 Agent 只判能不能退 → submit_refund 承载退款表单)。

**Architecture:** 沿用 ch05 `StateGraph`。前段两站做实:`resolve_reference`(LLM coref+改写,写 `resolved_query`)→ `classify_intent`(四件套,写 `intent`/`intent_confidence`)。`route_by_intent` 由 4 出口扩成 5 出口(+`refund_flow`,+「其他」归 `fallback_script`)。新增确定性子流程 `fetch_order → retrieve_policy → main_agent`:缺订单号用 LangGraph `interrupt` 暂停等前端点选、`Command(resume)` 续跑;`retrieve_policy` 做 Query 扩写多查去重合并强制检索政策证据;主力 Agent 判「能不能退」,能退调 `submit_refund` 被 `agent_tools` 拦成前端退款表单(仿 ch05 create_ticket)。置信度闸位置、主力 Agent ReAct 环、两入口走图,全部沿用 ch05 不动。

**Tech Stack:** LangGraph(interrupt/Command 是自带 HITL 能力,**无新增依赖**)+ AsyncSqliteSaver(既有);LangChain ChatOpenAI 直连上游(既有);ch04 混合检索+重排检索器(既有);MySQL/SQLAlchemy(既有,tickets 表加 `退款` 枚举值);原生 HTML/CSS/JS 前端(既有,加订单卡片 + 退款表单)。

## Global Constraints

- 上游选型定死:LangGraph `interrupt`/`Command(resume)` 做订单选择器 HITL(**无新组件**);既有模型工厂;复用 ch04 检索器与 tickets 表。发现矛盾/死路停下问用户,不自行换方案。
- 意图识别默认走够格大模型(`settings.intent_model`,空则回落 `chat_model`)**先求准**;降级路(小模型判+置信度、低置信升级大模型)本章**只落配置种子,不实现运行时**,`intent_mode` 恒 `accuracy`。
- Query 扩写(严格 3 条 JSON `{"queries":[...]}`)**只在 refund_flow 的 `retrieve_policy`** 检索侧做;商品咨询 `knowledge` 路沿用 ch05 同义词轻扩;**库里知识只留一份**,不在入库侧拆存。
- 退款子流程只把「这一单能不能退」交主力 Agent;取单、检索政策是写死的确定性节点;**缺订单号不猜不编**,执行阶段 `interrupt` 弹订单选择器点选回填;退款原因提交时固定类目下拉。
- **前端例外走 Vibe Coding**(用户描述效果、我改):不套 brainstorm/TDD/code-review;功能必须落到前端在用入口,浏览器端到端验收。
- 不可单测的产出(纯 Prompt:coref/expand/意图四件套)用**标注评估集**代替 TDD;确定性代码走 **TDD**;`interrupt`/`resume`/`astream` 中断 surface 形状先用**红线冒烟**钉死(fail-fast 最前)。
- 涉及 LangGraph/FastAPI/SQLAlchemy 具体 API,动手前用 **Context7** 核实当前接口(版本对不上是返工重灾区)。
- 沿用 ch05:置信度闸在 Agent 之前(生成前证据闸,ch05 D3);节点名 `resolve_reference`/`retrieve_knowledge`/`main_agent` 不变;trace 观测键(`coref`/`forced_rag`)保留。
- 每个任务末尾 commit(分支 `ch06-intent-recognition`);`dev-notes/ch06.md` 每完成一阶段追记,**不许收尾时一次性补记**。
- 模型非确定性:eval 抖动如实重跑记录。

## 权威来源
- spec:`docs/superpowers/specs/2026-07-16-ch06-intent-recognition-design.md`
- 章节:`mewhelp-course/ch06-intent-recognition/README.md`(扩写 prompt L33-48;意图四件套 prompt L120-140;`route_by_intent` L208-220;缺信息执行阶段补 L168-182;分流出口 L184-224)

---

## 文件结构

**新建:**
- `app/core/coref.py` — 指代消解+改写 `resolve(query, history) -> str`
- `scripts/smoke_interrupt.py` — interrupt/resume/astream 中断 surface 红线冒烟(R1)
- `scripts/eval_coref.py` — 指代消解标注评估(含已完整透传用例)
- `scripts/eval_expand.py` — Query 扩写标注评估(核心场景出 3 条 / JSON 稳定)
- `scripts/eval_ch06.py` — 四验收端到端评估
- `sql/ch06-ticket-type.sql` — tickets.ticket_type 枚举加 `退款`
- `tests/graph/test_ch06_nodes.py` — fetch_order/retrieve_policy/submit_refund 拦截/_agent_messages 测试
- `tests/test_refund_api.py` — create-refund + resume 端点测试

**修改:**
- `app/graph/state.py` — `ConversationState` 加 `resolved_query`/`intent_confidence`/`order_id`/`order_data`
- `app/graph/routing.py` — `INTENT_TO_ROUTE` 8→5 出口;`route_by_intent`
- `app/graph/nodes.py` — `resolve_reference`/`classify_intent` 做实;新增 `fetch_order`/`retrieve_policy`/`script_reply`;`agent_tools` 加 submit_refund 拦截;`_agent_messages` refund 适配
- `app/graph/build.py` — 5 出口条件边 + refund_flow 链 + 退役 `chitchat_reply` 节点
- `app/graph/runtime.py` — `resume_turn`/`stream_resume` + interrupt 事件 + 节点名对齐(`script_reply`/`retrieve_policy` citations)
- `app/core/intent.py` — 七类单标签 → 八类 `{intent, confidence}` + 「其他」兜底
- `app/core/query_understanding.py` — 加 `expand_queries(query) -> list[str]`
- `app/core/prompts.py` — 加 `COREF_REWRITE_*`/`EXPAND_QUERIES_*`/重写 `INTENT_CLASSIFY_*`(八类四件套)/`SCRIPT_REPLY_*` 文案/`_REFUND_JUDGE_HINT`
- `app/core/llm.py` — `get_chat_model` 加可选 `model` 覆盖参
- `app/config.py` — 意图模型配置种子
- `app/tools/business.py` — 抽 `order_snapshot`;加 `list_user_orders`/`submit_refund`
- `app/tools/registry.py` — 注册 `submit_refund`
- `app/db/models.py` — `Ticket.ticket_type` 枚举加 `退款`
- `app/api/actions.py` — `POST /api/actions/create-refund` + `POST /api/actions/resume`(SSE)
- `app/api/chat.py` — `interrupt` 事件 → SSE `event: interrupt`
- `app/api/agent.py` — `AgentResponse` 带 `interrupt` 字段
- `app/schemas/actions.py` — `CreateRefundRequest/Response`、`ResumeRequest`
- `app/schemas/agent.py` — `AgentResponse` 加 `interrupt`
- `app/static/index.html` — 订单选择器卡片 + 退款表单(Vibe Coding)
- `Makefile` — `smoke-interrupt`/`eval-ch06` 目标
- `tests/graph/test_routing.py` — 路由断言 7→4 改 8→5(替换 `test_route_maps_seven_to_four`,Task 3)
- `tests/graph/test_reply_nodes.py` — `chitchat_reply` 测试改 `script_reply`(Task 12)
- `tests/graph/test_forced_rag.py` — `test_classify_intent_sets_route_in_state` 改 dict 返回 + refund_flow(Task 4)

**退役:** `nodes.chitchat_reply` 节点被 `script_reply` 取代(Task 12 删)。

---

## Task 1: interrupt/resume/astream 中断 surface 红线冒烟(fail-fast)

**目的:** spec §8 R1。动图代码前,在**已装 langgraph 1.2.9** 上钉死三件事:①`interrupt(payload)` 在节点里被直接调用(无 resume 上下文)抛的**异常类型与载荷取法**(供 Task 7 fetch_order 单测断言);②`ainvoke` 遇中断返回态的 `__interrupt__` 键形状 + `Command(resume=v)` 续跑(供 Task 13 run_turn/resume_turn);③`astream(stream_mode=["messages","updates"])` 遇中断时中断信息出现在**哪种 chunk**(updates 的 `__interrupt__` 键?还是需 `aget_state` 探 pending?供 Task 13 stream_turn)。不通则停下按选型内解决或问用户。

**Files:**
- Create: `scripts/smoke_interrupt.py`
- Modify: `Makefile`

**Interfaces:**
- Produces:钉死的中断 surface 结论(写进 `dev-notes/ch06.md`),后续 Task 7/13 据此写断言。

- [ ] **Step 1: 用 Context7 核实 interrupt/Command 接口**

查 `/websites/langchain_oss_python_langgraph`:确认 `from langgraph.types import interrupt, Command`、`from langgraph.errors import GraphInterrupt`;确认 `interrupt(value)` 在编译图内首次执行时抛出 `GraphInterrupt`、图暂停并 checkpoint;确认 `ainvoke` 返回值里 `__interrupt__` 的结构(`list[Interrupt]`,每项有 `.value`);确认 `graph.ainvoke(Command(resume=v), config)`(同 thread_id)从被中断节点**从头重跑**、`interrupt()` 处返回 `v`。把结论写进 dev-notes。

- [ ] **Step 2: 写冒烟脚本**

`scripts/smoke_interrupt.py`:
```python
"""ch06 红线冒烟:interrupt/Command(resume) 在当前 langgraph 上的中断 surface 形状。
需要打上游吗?不需要——只用一个纯 interrupt 节点。用法:.venv/bin/python -m scripts.smoke_interrupt"""
import asyncio
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class S(TypedDict):
    messages: Annotated[list, add_messages]
    picked: str


async def ask_order(state: S):
    picked = interrupt({"type": "select_order", "orders": [{"order_id": "1001"}, {"order_id": "2002"}]})
    return {"picked": picked}


async def confirm(state: S):
    return {"messages": [("ai", f"已选订单 {state['picked']}")]}


async def main():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as cp:
        await cp.setup()
        b = StateGraph(S)
        b.add_node("ask_order", ask_order)
        b.add_node("confirm", confirm)
        b.add_edge(START, "ask_order")
        b.add_edge("ask_order", "confirm")
        b.add_edge("confirm", END)
        graph = b.compile(checkpointer=cp)
        config = {"configurable": {"thread_id": "smoke-int-1"}}

        # A) 非流式:首跑应带 __interrupt__
        out = await graph.ainvoke({"messages": [("human", "我要退款")]}, config)
        print("A ainvoke keys=", list(out.keys()))
        print("A __interrupt__=", out.get("__interrupt__"))

        # B) 非流式 resume
        out2 = await graph.ainvoke(Command(resume="1001"), config)
        print("B resume picked=", out2.get("picked"), "msgs=", [m.content for m in out2["messages"]])

        # C) 流式:中断信息出现在哪种 chunk
        config2 = {"configurable": {"thread_id": "smoke-int-2"}}
        print("C astream chunks:")
        async for mode, chunk in graph.astream(
            {"messages": [("human", "我要退款")]}, config2, stream_mode=["messages", "updates"]):
            if mode == "updates":
                print("  UPD keys=", list(chunk.keys()), "val=", chunk)
        # C') 流式后探 pending(备用探测路径)
        snap = await graph.aget_state(config2)
        print("C' aget_state .next=", snap.next, "interrupts=", getattr(snap, "interrupts", None),
              "tasks_interrupts=", [t.interrupts for t in snap.tasks])
        # C'') 流式 resume
        print("C'' astream resume:")
        async for mode, chunk in graph.astream(
            Command(resume="2002"), config2, stream_mode=["messages", "updates"]):
            print("  ", mode, chunk if mode == "updates" else (chunk[0].content, chunk[1].get("langgraph_node")))
```
在 `main()` 末尾补一段直接调用节点观察异常形状(供 Task 7):
```python
        # D) 直接调用被中断节点(无 resume 上下文)——Task 7 fetch_order 单测要断言这个形状
        from langgraph.errors import GraphInterrupt
        try:
            await ask_order({"messages": []})
        except GraphInterrupt as e:
            print("D GraphInterrupt args=", e.args, "first.value=", e.args[0][0].value if e.args else None)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: 加 Makefile 目标 + 跑冒烟**

`Makefile` 加:
```makefile
smoke-interrupt:  ## ch06 interrupt/resume 中断 surface 红线冒烟
	.venv/bin/python -m scripts.smoke_interrupt
```
Run: `make smoke-interrupt`
Expected: 打印 A/B/C/C'/C''/D 各段真实形状。**记录结论**:
- A:`__interrupt__` 是否在返回 dict、取值路径(如 `out["__interrupt__"][0].value`);
- C:流式下中断是否作为 `{"__interrupt__": (...)}` 出现在某个 updates chunk;若否,C' 的 `aget_state` 哪个字段暴露 pending 中断(`snap.next` / `snap.tasks[*].interrupts`);
- D:`GraphInterrupt` 载荷取法(`e.args[0][0].value`)。
把三段确定的取法写进 `dev-notes/ch06.md`(阶段3 起始),Task 7/13 直接引用。若 `interrupt`/`Command` 在 1.2.9 跑不通或 API 不符,**停下问用户**(选型定死)。

- [ ] **Step 4: Commit**

```bash
git add scripts/smoke_interrupt.py Makefile
git commit -m "feat(ch06): interrupt/resume 中断 surface 红线冒烟钉死形状"
```

---

## Task 2: 会话 State 扩字段 + 入口清零

**目的:** spec §10。加 refund_flow 与前段所需的四个无 reducer 标量,并在 `_graph_input` 入口清零(防 ch05 C1 跨轮泄漏)。

**Files:**
- Modify: `app/graph/state.py`
- Modify: `app/graph/runtime.py:72-79`(`_graph_input`)
- Test: `tests/graph/test_runtime.py`(既有,扩断言)

**Interfaces:**
- Produces:`ConversationState` 加 `resolved_query:str`、`intent_confidence:float`、`order_id:str`、`order_data:dict`;`_graph_input` 把这四个连同 ch05 输出通道一起清零。

- [ ] **Step 1: 扩既有回归测试**

在 `tests/graph/test_runtime.py::test_graph_input_resets_per_turn_output_channels` 里追加断言:
```python
    assert inp["resolved_query"] == "" and inp["intent_confidence"] == 0.0
    assert inp["order_id"] == "" and inp["order_data"] == {}
```

- [ ] **Step 2: 跑验证失败**

Run: `.venv/bin/pytest tests/graph/test_runtime.py::test_graph_input_resets_per_turn_output_channels -v`
Expected: FAIL(KeyError,新键还没进 `_graph_input`)。

- [ ] **Step 3: 加 State 字段**

`app/graph/state.py` 的 `ConversationState` 里,在 `intent` 附近加:
```python
    resolved_query: str    # 指代消解+改写后的完整问句(下游检索/判意图都用它)
    intent_confidence: float  # 意图 JSON 的 confidence(0-1)
    order_id: str          # refund_flow:抽到/点选回填的订单号
    order_data: dict       # refund_flow:query_order 查到的订单数据
```

- [ ] **Step 4: `_graph_input` 清零新通道**

`app/graph/runtime.py` 的 `_graph_input` 返回 dict 里加:
```python
            "resolved_query": "", "intent_confidence": 0.0,
            "order_id": "", "order_data": {},
```
(注:`order_id` 在 refund_flow 内经 resume 回填;入口清零只影响**新一轮**,resume 走 `Command(resume=...)` 不经此函数,不冲突。)

- [ ] **Step 5: 跑验证通过**

Run: `.venv/bin/pytest tests/graph/test_runtime.py -v`
Expected: PASS(全部)。

- [ ] **Step 6: Commit**

```bash
git add app/graph/state.py app/graph/runtime.py tests/graph/test_runtime.py
git commit -m "feat(ch06): State 加 resolved_query/intent_confidence/order_id/order_data + 入口清零"
```

---

## Task 3: 路由 5 出口(INTENT_TO_ROUTE 8→5 + route_by_intent)

**目的:** spec §3.1 / README L208-220。八类意图 → 五出口,单一来源避免漂移。纯确定性 → TDD。

**Files:**
- Modify: `app/graph/routing.py:5-19`
- Modify: `tests/graph/test_routing.py:1-20`(替换 ch05 的 `test_route_maps_seven_to_four` + `test_route_unknown_intent_defaults_business`;`confidence_gate`/`should_continue` 测试保留不动——它们仍是 ch05 基线)

**Interfaces:**
- Consumes:`state["intent"]`(8 类之一,Task 4 产出)
- Produces:`INTENT_TO_ROUTE:dict[str,str]`(8 键→5 值:`escalate/fallback_script/knowledge/refund_flow/business`);`route_by_intent(state)->str`(未知意图保守归 `business`)。`confidence_gate`/`should_continue` 不动。

- [ ] **Step 1: 改失败测试(原地更新 ch05 路由测试)**

`tests/graph/test_routing.py`:顶部 import 加 `import pytest`,并把 `from app.graph.routing import (confidence_gate, route_by_intent, should_continue,)` 补上 `INTENT_TO_ROUTE`。把 `test_route_maps_seven_to_four` 与 `test_route_unknown_intent_defaults_business` 两个函数**整块替换**为(ch05 的 7→4 断言已过时):
```python
@pytest.mark.parametrize("intent,expect", [
    ("投诉", "escalate"),
    ("闲聊", "fallback_script"),
    ("其他", "fallback_script"),
    ("商品咨询", "knowledge"),
    ("退款退货", "refund_flow"),
    ("售后", "refund_flow"),
    ("物流", "business"),
    ("订单", "business"),
])
def test_route_by_intent_five_outlets(intent, expect):
    assert route_by_intent({"intent": intent}) == expect


def test_route_by_intent_unknown_defaults_business():
    assert route_by_intent({"intent": "火星语"}) == "business"
    assert route_by_intent({}) == "business"


def test_intent_to_route_covers_eight_classes():
    assert set(INTENT_TO_ROUTE) == {
        "投诉", "闲聊", "其他", "商品咨询", "退款退货", "售后", "物流", "订单"}
```

- [ ] **Step 2: 跑验证失败**

Run: `.venv/bin/pytest tests/graph/test_routing.py -v`
Expected: FAIL(其他/退款退货/售后/投诉/闲聊 的映射还是 ch05 旧值)。

- [ ] **Step 3: 改路由**

`app/graph/routing.py` 顶部 `INTENT_TO_ROUTE` 整体替换为(注释更新为 spec §3.1):
```python
# 八类意图 → 五出口(写死的分流规则,spec §3.1 / README route_by_intent;
# 返回值 = build.py 条件边映射键,单一来源不漂移)
INTENT_TO_ROUTE: dict[str, str] = {
    "投诉": "escalate",
    "闲聊": "fallback_script",
    "其他": "fallback_script",
    "商品咨询": "knowledge",
    "退款退货": "refund_flow",
    "售后": "refund_flow",
    "物流": "business",
    "订单": "business",
}
```
`route_by_intent` 保持 `return INTENT_TO_ROUTE.get(state.get("intent", ""), "business")` 不变。

- [ ] **Step 4: 跑验证通过**

Run: `.venv/bin/pytest tests/graph/test_routing.py -v`
Expected: PASS(5 出口新断言 + ch05 confidence_gate/should_continue 仍绿)。

- [ ] **Step 5: Commit**

```bash
git add app/graph/routing.py tests/graph/test_routing.py
git commit -m "feat(ch06): 路由扩成 5 出口(+refund_flow,其他归 fallback_script)"
```

---

## Task 4: 意图四件套(intent.py 重写 + 模型配置种子 + classify_intent 节点)

**目的:** spec §6 / README L114-152。八类枚举选择题 + 强制 JSON `{intent,confidence}` + 边界 few-shot + 「其他」兜底。纯 Prompt → 评估集;节点/配置确定性 → TDD。

**Files:**
- Modify: `app/core/prompts.py:128-140`(重写 `INTENT_CLASSIFY_*`)
- Modify: `app/core/intent.py`(全文)
- Modify: `app/core/llm.py:6-14`(加 `model` 参)
- Modify: `app/config.py:21-24`(加意图配置种子)
- Modify: `app/graph/nodes.py:61-66`(`classify_intent` 节点)
- Test: `tests/graph/test_ch06_nodes.py`(新建,含 classify_intent 节点 + llm override 用例)
- Modify: `scripts/eval_intent.py`(扩八类 + 其他 + 边界 + 多轮 + confidence)

**Interfaces:**
- Consumes:`state["resolved_query"]`(Task 5 产出;空则回落 `_user_text`)、`INTENT_TO_ROUTE`(Task 3)
- Produces:`intent.classify(query:str, history:str="") -> dict{"intent":str,"confidence":float}`(八类+其他,解析失败/越界→其他);`get_chat_model(streaming=False, model:str|None=None)`;`classify_intent` 节点写 `intent`/`intent_confidence`/`route` + trace(用 `intent_confidence` 键,**不占用** `confidence` 键——那是 confidence_check 的 strong/weak)。
- **config 种子:** `settings.intent_model:str=""`、`intent_small_model:str=""`、`intent_mode:str="accuracy"`、`intent_conf_threshold:float=0.6`。

- [ ] **Step 1: 写失败测试(节点 + llm override)**

`tests/graph/test_ch06_nodes.py`:
```python
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph import nodes


@pytest.mark.asyncio
async def test_classify_intent_writes_confidence_and_route(monkeypatch):
    async def fake_classify(query, history=""):
        return {"intent": "退款退货", "confidence": 0.83}
    monkeypatch.setattr(nodes.intent_mod, "classify", fake_classify)
    out = await nodes.classify_intent({"messages": [HumanMessage("这个能退吗")],
                                       "resolved_query": "蓝牙耳机还能申请退货吗"})
    assert out["intent"] == "退款退货"
    assert out["intent_confidence"] == 0.83
    assert out["route"] == "refund_flow"                 # 退款退货 → refund_flow
    assert out["trace"]["route"] == "refund_flow"
    assert out["trace"]["intent_confidence"] == 0.83     # 不覆盖 confidence_check 的 confidence 键


def test_get_chat_model_honors_model_override():
    from app.core.llm import get_chat_model
    m = get_chat_model(model="glm-4-flash")
    assert m.model_name == "glm-4-flash"
    d = get_chat_model()
    from app.config import settings
    assert d.model_name == settings.chat_model
```

- [ ] **Step 2: 跑验证失败**

Run: `.venv/bin/pytest tests/graph/test_ch06_nodes.py -v`
Expected: FAIL(classify 还返回 str;get_chat_model 无 model 参)。

- [ ] **Step 3: llm.py 加 model 覆盖**

`app/core/llm.py`:
```python
def get_chat_model(streaming: bool = False, model: str | None = None) -> ChatOpenAI:
    """直连聊天上游。地址、模型名、密钥全从 .env 来,换上游不动这里。
    model 显式覆盖模型名(意图识别可配更大模型求准),None 时回落 settings.chat_model。"""
    return ChatOpenAI(
        model=model or settings.chat_model,
        base_url=settings.chat_base_url,
        api_key=settings.chat_api_key,
        streaming=streaming,
        temperature=0.3,
    )
```

- [ ] **Step 4: config 加意图配置种子**

`app/config.py` 的 `Settings` 在 ch05 块后加:
```python
    # ch06 意图识别模型配置(先求准:默认大模型;降级路仅种子,本章不实现运行时)
    intent_model: str = ""            # 意图识别模型名;空=回落 chat_model
    intent_small_model: str = ""      # 降级路小模型(种子,未接运行时)
    intent_mode: str = "accuracy"     # accuracy=只用大模型;cost=小模型判→低置信升级大模型(未实现)
    intent_conf_threshold: float = 0.6  # cost 模式升级阈值(种子)
```

- [ ] **Step 5: 重写意图四件套 prompt**

`app/core/prompts.py` 把 `INTENT_CLASSIFY_SYSTEM` / `INTENT_CLASSIFY_PROMPT` 整块替换(八类 + 边界 + few-shot + 强制 JSON,照 README L120-152):
```python
INTENT_CLASSIFY_SYSTEM = """## 角色
你是电商客服的意图识别器,判断用户这句话属于哪个意图。输入的已是指代消解后的完整问题,你只管判意图、不要再改写。

## 判断标准(八选一)
1. 物流:查快递到哪了、发货没有(通常带订单号/单号)。
2. 订单:查某订单的状态、金额、下单时间、买了什么。
3. 商品咨询:商品参数、退换货政策、通用 FAQ 这类「查规则、查说明」的问题(还没锁定到某一单)。
4. 退款退货:要对某个已购订单退货或退款(得先查这一单能不能退)。
5. 售后:维修、保修、换新这类售后处理(不含退款退货)。
6. 投诉:对产品或服务不满、要追责、要说法。
7. 闲聊:寒暄、玩笑、与购物无关的话题。
8. 其他:拿不准、又不该硬塞进上面某类时选它(兜底,宁可归这里也别硬贴标签)。

## 边界样例(few-shot)
- 「这个还能退吗」→ 退款退货(要退某一单,不是商品咨询)。
- 「退货运费谁出」→ 商品咨询(问的是政策规则,还没锁定订单)。
- 「我的猫爬架坏了能保修吗」→ 售后(维修保修,不是退款退货)。
- 「你们这什么破服务」→ 投诉。
- 「在吗」→ 闲聊。
- 「帮我写首诗」→ 其他(与电商客服无关又不该塞进闲聊业务处理)。

## 输出要求
必须使用以下 JSON 格式返回,不得包含任何其他文本:
{"intent": "...", "confidence": 0.0-1.0}
intent 只能是上面八类中文标签之一;confidence 是你对该判断的把握(0-1)。"""

INTENT_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages(
    [("system", INTENT_CLASSIFY_SYSTEM),
     ("human", "最近对话(可空):\n{history}\n\n当前用户这句话:{query}")]
)
```

- [ ] **Step 6: 重写 intent.py**

`app/core/intent.py`(全文替换):
```python
from typing import Literal

from pydantic import BaseModel, Field

from app.config import settings
from app.core.llm import get_chat_model
from app.core.prompts import INTENT_CLASSIFY_PROMPT

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他")


class _Intent(BaseModel):
    intent: Literal["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他"] = Field(
        description="八类意图之一")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="判断把握 0-1")


async def classify(query: str, history: str = "") -> dict:
    """八类意图 + confidence。扁平字段避开 glm 嵌套 502;
    解析失败/越界 → 归「其他」(最保守兜底,替换 ch05 的回退闲聊)。
    模型默认走 settings.intent_model(空则回落 chat_model)——先求准。"""
    model = get_chat_model(model=settings.intent_model or None).with_structured_output(_Intent)
    try:
        r: _Intent = await (INTENT_CLASSIFY_PROMPT | model).ainvoke(
            {"query": query, "history": history or "(无)"})
    except Exception:
        return {"intent": "其他", "confidence": 0.0}
    intent = r.intent if r.intent in INTENTS else "其他"
    return {"intent": intent, "confidence": float(r.confidence)}
```

- [ ] **Step 7: 改 classify_intent 节点**

`app/graph/nodes.py` 的 `classify_intent` 替换为(读 `resolved_query` + 历史;写 confidence;trace 用 `intent_confidence` 键):
```python
async def classify_intent(state) -> dict:
    """意图四件套:八类 + confidence + 「其他」兜底。吃 resolved_query(空则回落原话)+ 最近历史
    判当前意图(应对物流→退款→物流漂移)。把归属出口 route 落进 State(供 _agent_messages 判注入、
    log 留痕;route_by_intent 条件边按同一 INTENT_TO_ROUTE 分流,单一来源不漂移)。"""
    query = state.get("resolved_query") or _user_text(state)
    r = await intent_mod.classify(query, _history_text(state))
    intent, conf = r["intent"], r["confidence"]
    route = INTENT_TO_ROUTE.get(intent, "business")
    return {"intent": intent, "intent_confidence": conf, "route": route,
            "trace": {"intent": intent, "intent_confidence": conf, "route": route}}
```
并在 `nodes.py` 的 helper 区(`_user_text` 附近)加 `_history_text`(Task 5 的 `resolve_reference` 也复用):
```python
def _history_text(state, max_turns: int = 6) -> str:
    """把最近若干轮 human/ai 消息(不含本轮最后一条 human)压成紧凑文本,供 coref/意图读上下文。"""
    msgs = state.get("messages", [])
    prior = msgs[:-1] if msgs else []
    lines = []
    for m in prior[-max_turns:]:
        role = "用户" if isinstance(m, HumanMessage) else "客服"
        text = m.content if isinstance(m.content, str) else ""
        if text:
            lines.append(f"{role}:{text}")
    return "\n".join(lines)
```

- [ ] **Step 8: 跑节点/llm 测试通过**

Run: `.venv/bin/pytest tests/graph/test_ch06_nodes.py -v`
Expected: PASS(classify_intent + get_chat_model override)。
并跑 ch05 既有 `tests/graph/test_forced_rag.py::test_classify_intent_sets_route_in_state` —— 它 monkeypatch 的 `fake_classify` 返回 str、且断言 `route=="knowledge"`,现已过时:改该用例的 `fake_classify` 返回 `{"intent":"退款退货","confidence":0.8}`、断言 `out["route"]=="refund_flow"`(退款退货 ch06 改走 refund_flow)。

- [ ] **Step 9: 扩意图评估集(纯 Prompt 用评估代替 TDD)**

`scripts/eval_intent.py` 的 `SAMPLES` 扩到八类(加「其他」+ 边界 + 怪问题),并把断言改成读 dict、校验 JSON 可解析与「其他」兜底:
```python
"""意图四件套标注评估:八类判对率 + confidence 可解析 + 怪问题落其他 + 多轮漂移。需上游可用。
用法:.venv/bin/python -m scripts.eval_intent"""
import asyncio

from app.core.intent import classify

SAMPLES = [
    ("订单1001的快递到哪了", "物流"), ("我买的东西发货了吗", "物流"),
    ("订单2002现在什么状态", "订单"), ("我上周下的单多少钱来着", "订单"),
    ("这款猫粮多少钱一包", "商品咨询"), ("退货运费一般谁承担", "商品咨询"),
    ("我要退货", "退款退货"), ("这个订单还能申请退款吗", "退款退货"),
    ("我的猫爬架坏了能保修吗", "售后"), ("换新进度到哪了", "售后"),
    ("你们这什么破服务,我要投诉", "投诉"), ("太差了给我个说法", "投诉"),
    ("你好呀", "闲聊"), ("今天天气不错", "闲聊"),
    ("帮我写一段 Python 代码", "其他"), ("阿斯顿发发", "其他"),
]


async def main():
    passed = bad_json = 0
    for q, expect in SAMPLES:
        r = await classify(q)
        got = r.get("intent")
        conf = r.get("confidence")
        ok = got == expect
        json_ok = got in ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他") \
            and isinstance(conf, float) and 0.0 <= conf <= 1.0
        bad_json += not json_ok
        passed += ok
        print(f"{'✅' if ok else '❌'} {q!r} -> {got}(conf={conf}) 期望={expect}")

    # 多轮漂移:物流→退款→物流,当前句意图应随上下文
    hist = "用户:订单1001到哪了\n客服:已发货,深圳分拨中心\n用户:那我想退了\n客服:好的,帮您看下退款\n"
    r = await classify("那它现在到哪了", hist)
    print(f"多轮漂移『那它现在到哪了』(退款后)-> {r['intent']}(期望 物流)")

    print(f"\n判对 {passed}/{len(SAMPLES)};JSON 越界 {bad_json} 条(glm 非确定性,抖动如实重跑记录)")


if __name__ == "__main__":
    asyncio.run(main())
```
Run(需上游可用):`.venv/bin/python -m scripts.eval_intent` —— 记录判对率与抖动到 dev-notes(阶段4)。边界个例 miss 可接受(如 ch05 售后↔退款退货边界),JSON 越界应为 0。

- [ ] **Step 10: Commit**

```bash
git add app/core/intent.py app/core/prompts.py app/core/llm.py app/config.py \
  app/graph/nodes.py tests/graph/test_ch06_nodes.py tests/graph/test_forced_rag.py scripts/eval_intent.py
git commit -m "feat(ch06): 意图四件套(8类+confidence+其他兜底)+ 模型配置种子 + classify_intent 节点"
```

---

## Task 5: 指代消解 + Query 改写(coref.resolve + resolve_reference 做实)

**目的:** spec §4 / README L11-21。一次 LLM 调用完成 coref + 口语归一,写 `resolved_query`;已完整/指代已明确的原样透传,不强改。纯 Prompt → 评估集。

**Files:**
- Create: `app/core/coref.py`
- Modify: `app/core/prompts.py`(加 `COREF_REWRITE_*`)
- Modify: `app/graph/nodes.py:56-58`(`resolve_reference` 做实)
- Create: `scripts/eval_coref.py`

**Interfaces:**
- Consumes:`state["messages"]`(本轮 + 历史)、`_history_text`(Task 4 加的 helper)
- Produces:`coref.resolve(query:str, history:str="") -> str`(完整问句;已完整则原样);`resolve_reference` 节点写 `resolved_query` + trace `coref` = `"rewrite"|"passthrough"`(保留 ch05 观测键 `coref`)。

- [ ] **Step 1: 加 coref prompt**

`app/core/prompts.py` 加:
```python
COREF_REWRITE_SYSTEM = """## 角色
你是电商客服在检索/判意图之前的「问题补全器」。把用户这一轮依赖上下文才看得懂的话,结合最近几轮对话,改写成一句脱离对话也能独立看懂的完整问题。

## 规则
1. 把「它/这个/那款/这一单」等指代,依据历史补成明确实体(如「这个能退吗」+上文蓝牙耳机 → 「蓝牙耳机还能申请退货吗」)。
   指代指向某笔订单时,改写必须把订单号带上,别只补商品名——同一件商品往往有好几单,
   只给商品名下游锁不到是哪一单(如「那它能退吗」+上文订单 1001 智能猫砂盆 → 「订单 1001 的智能猫砂盆能申请退货吗」)。
2. 口语、模糊、带情绪的问法归一成简洁标准问法,保留关键实体(型号、品类、订单号、政策词)。
3. 用户这句本身已完整、指代已明确时,原样返回,不要改写,更不要引入历史里没有的信息(硬改会越改越偏)。
   问平台通用规则的问题尤其如此:「你们支持花呗分期吗」「运费怎么算」这类句子里没有任何指代,
   哪怕上文刚聊过某笔订单,也不要把订单号或商品名塞进去——绑上之后检索反而查不到通用政策。
4. 只输出改写后的一句问题本身,不要解释、不要加引号。"""

COREF_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", COREF_REWRITE_SYSTEM),
     ("human", "最近对话(可空):\n{history}\n\n用户这句话:{query}\n\n补全后的完整问题:")]
)
```

- [ ] **Step 2: 写失败测试(节点透传 + 改写)**

在 `tests/graph/test_ch06_nodes.py` 追加:
```python
@pytest.mark.asyncio
async def test_resolve_reference_passthrough_when_complete(monkeypatch):
    async def fake_resolve(q, history=""):
        return q  # 已完整,原样
    monkeypatch.setattr(nodes.coref, "resolve", fake_resolve)
    out = await nodes.resolve_reference({"messages": [HumanMessage("蓝牙耳机的保修期多久")]})
    assert out["resolved_query"] == "蓝牙耳机的保修期多久"
    assert out["trace"]["coref"] == "passthrough"


@pytest.mark.asyncio
async def test_resolve_reference_rewrites_with_history(monkeypatch):
    async def fake_resolve(q, history=""):
        return "蓝牙耳机还能申请退货吗"
    monkeypatch.setattr(nodes.coref, "resolve", fake_resolve)
    out = await nodes.resolve_reference({"messages": [
        HumanMessage("蓝牙耳机什么时候到"), AIMessage("预计明天"), HumanMessage("这个能退吗")]})
    assert out["resolved_query"] == "蓝牙耳机还能申请退货吗"
    assert out["trace"]["coref"] == "rewrite"
```

- [ ] **Step 3: 跑验证失败**

Run: `.venv/bin/pytest tests/graph/test_ch06_nodes.py -k resolve_reference -v`
Expected: FAIL(`nodes.coref` 不存在;`resolve_reference` 还是透传占位)。

- [ ] **Step 4: 写 coref.py**

`app/core/coref.py`:
```python
from app.core.llm import get_chat_model
from app.core.prompts import COREF_REWRITE_PROMPT


async def resolve(query: str, history: str = "") -> str:
    """指代消解 + 口语归一。结合历史把半截话补成完整问句;已完整则原样。
    出错兜底:返回原句(不阻断下游)。"""
    try:
        model = get_chat_model()
        r = await (COREF_REWRITE_PROMPT | model).ainvoke(
            {"query": query, "history": history or "(无)"})
        text = (r.content if isinstance(r.content, str) else "").strip()
    except Exception:
        return query
    return text or query
```

- [ ] **Step 5: 做实 resolve_reference 节点**

`app/graph/nodes.py`:先在 import 区加 `from app.core import coref`;把 `resolve_reference` 替换为:
```python
async def resolve_reference(state) -> dict:
    """指代消解 + Query 改写(合一):吃最近几轮历史把半截话补成完整问句;已完整则透传。
    写 resolved_query,供 classify_intent 与 refund_flow 检索共用(意图识别不再改写)。"""
    query = _user_text(state)
    resolved = await coref.resolve(query, _history_text(state))
    mode = "rewrite" if resolved != query else "passthrough"
    return {"resolved_query": resolved, "trace": {"coref": mode}}
```

- [ ] **Step 6: 跑验证通过**

Run: `.venv/bin/pytest tests/graph/test_ch06_nodes.py -k resolve_reference -v`
Expected: PASS。

- [ ] **Step 7: 写 coref 评估集**

`scripts/eval_coref.py`:
```python
"""指代消解标注评估:多轮带指代补全 + 已完整透传。需上游可用。
用法:.venv/bin/python -m scripts.eval_coref"""
import asyncio

from app.core.coref import resolve

# (history, query, 期望:'rewrite' 补出实体 / 'passthrough' 原样)
CASES = [
    ("用户:蓝牙耳机什么时候到货\n客服:预计明天送达", "这个能退吗", "rewrite"),
    ("用户:订单1001买的猫粮\n客服:已发货", "它到哪了", "rewrite"),
    ("", "蓝牙耳机的保修期是多久", "passthrough"),
    ("", "退货运费谁承担", "passthrough"),
]


async def main():
    for hist, q, kind in CASES:
        got = await resolve(q, hist)
        changed = got != q
        ok = (changed and kind == "rewrite") or (not changed and kind == "passthrough")
        print(f"{'✅' if ok else '❌'} {q!r} -> {got!r} 期望={kind}")
    print("\n(glm 非确定性;透传类必须不改,补全类应含上文实体。抖动如实重跑记录)")


if __name__ == "__main__":
    asyncio.run(main())
```
Run(需上游可用):`.venv/bin/python -m scripts.eval_coref` —— 记录到 dev-notes(阶段4)。透传类**必须**不改写(硬改判 fail)。

- [ ] **Step 8: Commit**

```bash
git add app/core/coref.py app/core/prompts.py app/graph/nodes.py \
  tests/graph/test_ch06_nodes.py scripts/eval_coref.py
git commit -m "feat(ch06): 指代消解+改写做实(coref.resolve + resolve_reference 写 resolved_query)"
```

---

## Task 6: Query 扩写(expand_queries)

**目的:** spec §5 / README L33-58。一句问题泛化成严格 3 条检索友好查询(强制 JSON 单字段 `queries`)。只在 refund_flow 检索侧调。纯 Prompt → 评估集(去重合并逻辑在 Task 8 TDD)。

**Files:**
- Modify: `app/core/prompts.py`(加 `EXPAND_QUERIES_*`)
- Modify: `app/core/query_understanding.py`(加 `expand_queries`)
- Create: `scripts/eval_expand.py`

**Interfaces:**
- Produces:`query_understanding.expand_queries(query:str) -> list[str]`(严格取前 3 条非空;全空则回落 `[query]`)。

- [ ] **Step 1: 加扩写 prompt(照 README L33-48)**

`app/core/prompts.py` 加:
```python
EXPAND_QUERIES_SYSTEM = """## 角色
你是电商客服的查询优化助手,把用户问题泛化成多条检索友好的中文查询,用于知识库检索。

## 改写规则
1. 出现产品名、型号、平台这类关键实体时,改写要保持一致。
2. 不要引入原问题里没有的型号、参数、数值。
3. 每条查询尽量短、含关键词,彼此侧重点不同。

## 输出要求
严格 3 条,必须使用以下 JSON 格式返回,不得包含任何其他文本:
{"queries": ["蓝牙耳机退货政策", "蓝牙耳机无理由退换货条件", "耳机退货时间限制"]}"""

EXPAND_QUERIES_PROMPT = ChatPromptTemplate.from_messages(
    [("system", EXPAND_QUERIES_SYSTEM), ("human", "用户问题:{query}")]
)
```

- [ ] **Step 2: 加 expand_queries**

`app/core/query_understanding.py` 追加(不动既有 `understand`):
```python
from app.core.prompts import EXPAND_QUERIES_PROMPT  # 与既有 QUERY_REWRITE_PROMPT 并列导入


class _Expanded(BaseModel):
    queries: list[str] = Field(default_factory=list, description="严格3条检索友好查询")


async def expand_queries(query: str) -> list[str]:
    """把一句问题泛化成检索友好查询(强制 JSON 单字段,扁平 list[str] 避开 glm 502)。
    严格取前 3 条非空;模型异常/全空则回落 [query],保证 retrieve_policy 至少有一条可查。"""
    model = get_chat_model().with_structured_output(_Expanded)
    try:
        r: _Expanded = await (EXPAND_QUERIES_PROMPT | model).ainvoke({"query": query})
        qs = [q.strip() for q in (r.queries or []) if q and q.strip()]
    except Exception:
        qs = []
    return qs[:3] or [query]
```

- [ ] **Step 3: 写扩写评估集**

`scripts/eval_expand.py`:
```python
"""Query 扩写标注评估:核心场景出 3 条 + JSON 稳定 + 保持关键实体。需上游可用。
用法:.venv/bin/python -m scripts.eval_expand"""
import asyncio

from app.core.query_understanding import expand_queries

CASES = ["蓝牙耳机还能申请退货吗", "订单1001的猫粮质量问题能退款吗", "智能猫砂盆坏了怎么保修"]


async def main():
    for q in CASES:
        qs = await expand_queries(q)
        print(f"{'✅' if len(qs) == 3 else '⚠️'} {q!r} -> {qs}(条数={len(qs)})")
    print("\n核心场景应出 3 条、彼此侧重点不同、含原问题关键实体;JSON 不稳时条数会 <3。抖动如实重跑记录")


if __name__ == "__main__":
    asyncio.run(main())
```
Run(需上游可用):`.venv/bin/python -m scripts.eval_expand` —— 记录到 dev-notes(阶段4)。

- [ ] **Step 4: Commit**

```bash
git add app/core/prompts.py app/core/query_understanding.py scripts/eval_expand.py
git commit -m "feat(ch06): Query 扩写 expand_queries(严格3条 JSON,检索侧现查现用)"
```

---

## Task 7: fetch_order 节点 + list_user_orders 工具 + 缺单 interrupt

**目的:** spec §7 / README L168-182,L194。抽订单号;缺则 `interrupt` 弹订单选择器;拿到后查订单数据。确定性 → TDD(interrupt 载荷形状引用 Task 1 冒烟结论)。

**Files:**
- Modify: `app/tools/business.py`(抽 `order_snapshot`;加 `list_user_orders`)
- Modify: `app/graph/nodes.py`(加 `fetch_order` + `_extract_order_id`)
- Test: `tests/graph/test_ch06_nodes.py`(追加)

**Interfaces:**
- Consumes:`state["resolved_query"]`/`state["order_id"]`/`state["user_id"]`;`order_snapshot`(抽出的纯函数)
- Produces:`business.order_snapshot(order_id:str)->dict`(与 `query_order` 同源);`business.list_user_orders(user_id:str)->list[dict]`(每项 `{order_id,product,status,amount}`);`nodes._extract_order_id(text)->str|None`;`fetch_order(state)->dict`(写 `order_id`/`order_data`/trace;缺单 `interrupt({"type":"select_order","orders":[...]})`)。

- [ ] **Step 1: 写失败测试**

`tests/graph/test_ch06_nodes.py` 追加(**Task 1 冒烟 D 已钉死**:`interrupt()` 只能在编译图内跑,图外直接调节点会 `RuntimeError`(非 GraphInterrupt),故缺单 interrupt 路径经【最小编译图 + ainvoke】测真实中断 surface,取 `out["__interrupt__"][0].value`):
```python
def test_extract_order_id():
    assert nodes._extract_order_id("订单1001的物流") == "1001"
    assert nodes._extract_order_id("尾号 20260701 那单") == "20260701"
    assert nodes._extract_order_id("我要退货") is None


@pytest.mark.asyncio
async def test_fetch_order_uses_id_in_query(monkeypatch):
    out = await nodes.fetch_order({"resolved_query": "订单1001能退吗", "user_id": "u1"})
    assert out["order_id"] == "1001"
    assert out["order_data"]["order_id"] == "1001"          # order_snapshot 同源
    assert out["trace"]["fetch_order"]["order_id"] == "1001"


@pytest.mark.asyncio
async def test_fetch_order_interrupts_when_missing(monkeypatch):
    # interrupt() 只能在编译图内跑(Task 1 冒烟 D:图外直接调是 RuntimeError),
    # 故缺单路径经最小编译图 + InMemorySaver ainvoke 测真实中断 surface。
    monkeypatch.setattr(nodes.business, "list_user_orders",
                        lambda uid: [{"order_id": "1001", "product": "猫粮", "status": "已签收", "amount": 99}])
    from langgraph.graph import START, END, StateGraph
    from langgraph.checkpoint.memory import InMemorySaver
    from app.graph.state import ConversationState
    b = StateGraph(ConversationState)
    b.add_node("fetch_order", nodes.fetch_order)
    b.add_edge(START, "fetch_order")
    b.add_edge("fetch_order", END)
    graph = b.compile(checkpointer=InMemorySaver())
    out = await graph.ainvoke({"resolved_query": "我要退款", "user_id": "u1"},
                              {"configurable": {"thread_id": "t-fetch"}})
    payload = out["__interrupt__"][0].value      # Task 1 冒烟 A/C 钉死此取法
    assert payload["type"] == "select_order"
    assert payload["orders"][0]["order_id"] == "1001"


def test_list_user_orders_stable_and_queryable():
    from app.tools import business
    a = business.list_user_orders("u-42")
    b = business.list_user_orders("u-42")
    assert a == b and len(a) >= 2                            # 同 user 稳定
    # 选中即可 query_order(同源快照)
    snap = business.order_snapshot(a[0]["order_id"])
    assert snap["order_id"] == a[0]["order_id"] and snap["product"] == a[0]["product"]
```

- [ ] **Step 2: 跑验证失败**

Run: `.venv/bin/pytest tests/graph/test_ch06_nodes.py -k "fetch_order or extract_order or list_user" -v`
Expected: FAIL。

- [ ] **Step 3: business.py 抽 order_snapshot + 加 list_user_orders**

`app/tools/business.py`:把 `query_order` 的随机快照抽成模块函数,`query_order` 改为调用它,并加 `list_user_orders`:
```python
def order_snapshot(order_id: str) -> dict:
    """订单快照(纯函数,随机种子固定 → 同 order_id 稳定)。query_order 工具与 fetch_order 节点同源。"""
    rng = random.Random(f"order:{order_id}")
    return {
        "order_id": order_id,
        "status": rng.choice(["待付款", "已付款", "已发货", "已签收"]),
        "amount": rng.randint(50, 2000),
        "created_at": f"2026-07-{rng.randint(1, 12):02d} 10:00",
        "product": rng.choice(["智能猫砂盆", "猫粮 5kg", "猫爬架", "自动饮水机"]),
        "tracking_no": f"SF{rng.randint(10**11, 10**12 - 1)}",
    }


def list_user_orders(user_id: str) -> list[dict]:
    """按 user_id 稳定列出该用户的 2-4 笔订单(mock,不落库)。每笔用 order_snapshot 同源,
    前端选中后回填 order_id 即可 query_order。"""
    rng = random.Random(f"user_orders:{user_id}")
    ids = [str(rng.randint(1000, 9999)) for _ in range(rng.randint(2, 4))]
    out = []
    for oid in ids:
        s = order_snapshot(oid)
        out.append({"order_id": oid, "product": s["product"],
                    "status": s["status"], "amount": s["amount"]})
    return out
```
`query_order` 工具体改为:
```python
@tool(args_schema=OrderInput)
async def query_order(order_id: str) -> dict:
    """查询订单的状态、金额、下单时间、商品名和物流单号(tracking_no)。用于用户询问某个订单情况时。
    要查物流轨迹,需先用本工具拿到订单的 tracking_no,再把它传给 query_logistics。"""
    return order_snapshot(order_id)
```

- [ ] **Step 4: 加 fetch_order 节点**

`app/graph/nodes.py`:import 区加 `import re`、`from app.tools import business`、`from langgraph.types import interrupt`;加:
```python
_ORDER_RE = re.compile(r"\b(\d{4,})\b")   # 4+ 位连续数字视作订单号


def _extract_order_id(text: str) -> str | None:
    m = _ORDER_RE.search(text or "")
    return m.group(1) if m else None


async def fetch_order(state) -> dict:
    """退款子流程第一步:抽订单号;缺则 interrupt 弹订单选择器等前端点选(resume 回填);
    拿到后 order_snapshot 取订单数据。interrupt 之前只做只读(resume 时本节点从头重跑)。"""
    oid = state.get("order_id") or _extract_order_id(
        state.get("resolved_query") or _user_text(state))
    if not oid:
        orders = business.list_user_orders(state.get("user_id", ""))   # 只读,可安全重跑
        oid = interrupt({"type": "select_order", "orders": orders})    # resume 回填订单号
    data = business.order_snapshot(oid)
    return {"order_id": oid, "order_data": data,
            "trace": {"fetch_order": {"order_id": oid}}}
```

- [ ] **Step 5: 跑验证通过 + ch02 工具回归**

Run: `.venv/bin/pytest tests/graph/test_ch06_nodes.py -k "fetch_order or extract_order or list_user" tests/test_tools_mock.py -v`
Expected: PASS(fetch_order 三例 + `test_tools_mock.py` 里 query_order 既有断言仍绿——order_snapshot 抽取不改行为)。

- [ ] **Step 6: Commit**

```bash
git add app/tools/business.py app/graph/nodes.py tests/graph/test_ch06_nodes.py
git commit -m "feat(ch06): fetch_order 抽单/缺单 interrupt + list_user_orders mock"
```

---

## Task 8: retrieve_policy 节点(扩写多查 + 去重合并强制检索政策)

**目的:** spec §7 / README L194。退款子流程第二步:Query 扩写 3 条 + 多 query 强制检索政策 + 按 chunk id 去重合并 + 拼编号证据。确定性 → TDD。

**Files:**
- Modify: `app/graph/nodes.py`(加 `retrieve_policy`)
- Test: `tests/graph/test_ch06_nodes.py`(追加)

**Interfaces:**
- Consumes:`state["resolved_query"]`/`state["order_data"]`;`query_understanding.expand_queries`(Task 6);`retrieval.search_knowledge`/`arrange_head_tail`(ch04)
- Produces:`retrieve_policy(state)->dict`,写 `evidence:str`/`citations:list`(结构同 `retrieve_knowledge`)+ trace `retrieve_policy`。按 chunk `id` 去重、保留每 id 最高 `rerank_score`。

- [ ] **Step 1: 写失败测试**

`tests/graph/test_ch06_nodes.py` 追加(复用文件底部的 `_aval` helper 模式;若无则本文件内定义):
```python
@pytest.mark.asyncio
async def test_retrieve_policy_expands_dedups_merges(monkeypatch):
    async def fake_expand(q):
        return ["退货政策", "无理由退换货", "退货时限"]
    # 三条 query 各自的召回:id=1 在两条里出现(取高分),id=2 只在一条
    per_query = {
        "退货政策": [{"id": 1, "question": "退货", "answer": "7天无理由", "rerank_score": 0.7,
                     "section_path": "政策/退货", "content_type": "policy"}],
        "无理由退换货": [{"id": 1, "question": "退货", "answer": "7天无理由", "rerank_score": 0.9,
                       "section_path": "政策/退货", "content_type": "policy"},
                      {"id": 2, "question": "运费", "answer": "质量问题商家承担", "rerank_score": 0.6,
                       "section_path": "政策/运费", "content_type": "policy"}],
        "退货时限": [],
    }
    async def fake_search(q, **k):
        return per_query.get(q, [])
    monkeypatch.setattr(nodes.query_understanding, "expand_queries", fake_expand)
    monkeypatch.setattr(nodes.retrieval, "search_knowledge", fake_search)
    monkeypatch.setattr(nodes.retrieval, "arrange_head_tail", lambda h: h)

    out = await nodes.retrieve_policy({"resolved_query": "这个订单能退吗",
                                       "order_data": {"status": "已签收"}})
    # 去重:id=1 只保留一次(取 0.9),id=2 保留;共 2 条
    assert len(out["citations"]) == 2
    ids = [c["id"] for c in out["citations"]]
    assert ids == [1, 2]                              # 按 rerank_score 降序(0.9, 0.6)
    assert "[1]" in out["evidence"] and "[2]" in out["evidence"]
    assert out["trace"]["retrieve_policy"]["hits"] == 2
    assert out["trace"]["retrieve_policy"]["queries"] == ["退货政策", "无理由退换货", "退货时限"]
```

- [ ] **Step 2: 跑验证失败**

Run: `.venv/bin/pytest tests/graph/test_ch06_nodes.py -k retrieve_policy -v`
Expected: FAIL(`retrieve_policy` 未定义)。

- [ ] **Step 3: 加 retrieve_policy 节点**

`app/graph/nodes.py`:
```python
async def retrieve_policy(state) -> dict:
    """退款子流程强制检索政策:Query 扩写 3 条 → 多 query 各检索一次 → 按 chunk id 去重合并
    (保留每 id 最高分)→ 按分降序拼编号证据。产出注入 main_agent 作「能不能退」的判据(README L194:
    不让模型凭记忆答)。库里知识一份,扩写只在检索侧现查现用。"""
    base = state.get("resolved_query") or _user_text(state)
    od = state.get("order_data") or {}
    seed = f"{base} {od.get('status', '')}".strip()
    queries = await query_understanding.expand_queries(seed)

    merged: dict = {}                       # chunk id -> 最高分 hit
    for q in queries:
        for h in await retrieval.search_knowledge(q, strategy="hybrid_rerank", bm25_text=q):
            cur = merged.get(h["id"])
            if cur is None or h["rerank_score"] > cur["rerank_score"]:
                merged[h["id"]] = h
    ranked = sorted(merged.values(), key=lambda h: h["rerank_score"], reverse=True)
    arranged = retrieval.arrange_head_tail(ranked)
    citations = [
        {"n": i + 1, "id": h["id"], "section_path": h["section_path"],
         "question": h["question"], "answer": h["answer"], "content_type": h["content_type"]}
        for i, h in enumerate(arranged)
    ]
    evidence = "\n".join(f"[{c['n']}] {c['question']}: {c['answer']}" for c in citations)
    return {"evidence": evidence, "citations": citations,
            "trace": {"retrieve_policy": {"queries": queries, "hits": len(ranked)}}}
```

- [ ] **Step 4: 跑验证通过**

Run: `.venv/bin/pytest tests/graph/test_ch06_nodes.py -k retrieve_policy -v`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/graph/nodes.py tests/graph/test_ch06_nodes.py
git commit -m "feat(ch06): retrieve_policy 扩写多查+去重合并强制检索政策证据"
```

---

## Task 9: submit_refund 工具 + agent_tools 拦截 + _agent_messages refund 适配

**目的:** spec §7 D4 / README L196。给主力 Agent 一个 `submit_refund` 工具:判「能退」就调,被 `agent_tools` 拦成前端「提交退款工单」可选项(仿 ch05 create_ticket,不写库);`_agent_messages` 在 refund 路注入 order_data + 政策证据 + 判定指令。确定性 → TDD。

**Files:**
- Modify: `app/tools/business.py`(加 `submit_refund`)
- Modify: `app/tools/registry.py`(注册)
- Modify: `app/core/prompts.py`(加 `_REFUND_JUDGE_HINT`)
- Modify: `app/graph/nodes.py`(`agent_tools` 加拦截;`_agent_messages` 适配;import hint)
- Test: `tests/graph/test_ch06_nodes.py`(追加)

**Interfaces:**
- Consumes:`state["route"]=="refund_flow"`、`state["order_data"]`、`state["evidence"]`
- Produces:`business.submit_refund` 工具(args `order_id:str`,`reason:str|None`);`agent_tools` 把 `submit_refund` 调用拦成 `{"type":"refund_form","draft":{"order_id":...,"reason":...}}` + 合成 ToolMessage(不执行、不写库);`_agent_messages` 证据注入条件由 `route=="knowledge" and evidence` 放宽为**只要 `evidence` 存在**(知识路与退款路都注入),并在 `route=="refund_flow"` 追加 order_data + 判定指令。

- [ ] **Step 1: 写失败测试**

`tests/graph/test_ch06_nodes.py` 追加:
```python
from langchain_core.messages import SystemMessage


@pytest.mark.asyncio
async def test_agent_tools_intercepts_submit_refund():
    ai = AIMessage(content="", tool_calls=[
        {"id": "r1", "name": "submit_refund", "args": {"order_id": "1001", "reason": None}}])
    out = await nodes.agent_tools({"messages": [ai]})
    assert out["suggested_actions"] == [{"type": "refund_form", "draft": {"order_id": "1001", "reason": None}}]
    tm = out["messages"][0]
    assert tm.name == "submit_refund"                       # 合成 ToolMessage 促收敛
    assert "退款" in tm.content


def test_agent_messages_injects_order_and_policy_on_refund():
    msgs = nodes._agent_messages({
        "route": "refund_flow",
        "order_data": {"order_id": "1001", "status": "已签收", "product": "猫粮 5kg"},
        "evidence": "[1] 退货: 7天无理由",
        "messages": [HumanMessage("这单能退吗")]})
    sys = msgs[0]
    assert isinstance(sys, SystemMessage)
    assert "7天无理由" in sys.content                        # 政策证据注入
    assert "猫粮 5kg" in sys.content and "submit_refund" in sys.content  # 订单数据 + 判定指令


def test_agent_messages_knowledge_path_still_injects_evidence():
    msgs = nodes._agent_messages({
        "route": "knowledge", "evidence": "[1] 运费: 满99包邮",
        "messages": [HumanMessage("运费多少")]})
    assert "满99包邮" in msgs[0].content                     # 放宽条件后知识路不回归
```

- [ ] **Step 2: 跑验证失败**

Run: `.venv/bin/pytest tests/graph/test_ch06_nodes.py -k "submit_refund or agent_messages" -v`
Expected: FAIL。

- [ ] **Step 3: 加 submit_refund 工具 + 注册**

`app/tools/business.py` 末尾加:
```python
class RefundInput(BaseModel):
    order_id: str = Field(description="要退款的订单号")
    reason: str | None = Field(default=None, description="退款原因(可选,最终以前端固定类目下拉为准)")


@tool(args_schema=RefundInput)
async def submit_refund(order_id: str, reason: str | None = None) -> dict:
    """判定这一单可以退款后,调用本工具发起退款申请。实际提交由前端退款表单确认后落库,
    本工具只表示『这一单可以退,已把提交入口交给用户』。"""
    return {"status": "待用户确认", "order_id": order_id}
```
`app/tools/registry.py`:import 加 `submit_refund`;`_ALL` 末尾加 `submit_refund`;`NO_RETRY` 加 `"submit_refund"`(被拦截、从不真执行,列入无妨):
```python
from app.tools.business import (
    create_ticket, query_faq, query_logistics, query_order, query_product, submit_refund,
)
_ALL: list[BaseTool] = [query_order, query_product, query_logistics, query_faq, create_ticket, submit_refund]
...
NO_RETRY: set[str] = {"create_ticket", "query_faq", "submit_refund"}
```

- [ ] **Step 4: 加 refund 判定 hint**

`app/core/prompts.py` 加:
```python
REFUND_JUDGE_HINT = (
    "\n\n## 退款判定任务(仅本轮)\n"
    "下面给出该订单数据与检索到的退换货政策证据。请只判断这一单能不能退款:\n"
    "- 能退:调用 submit_refund 工具(传该订单号),并用一句话告诉用户这一单可以退、简述依据;\n"
    "- 不能退:不要调用任何工具,说清不能退的原因,并告知可联系人工客服进一步确认。\n"
    "依据政策证据判断,不要臆造条款;时效/金额表述以平台售后规则为准。\n"
    "## 订单数据\n"
)
```

- [ ] **Step 5: 改 agent_tools 拦截 + _agent_messages 适配**

`app/graph/nodes.py`:import 区 `from app.core.prompts import ...` 追加 `REFUND_JUDGE_HINT`,并加 `import json`。
`agent_tools` 的 for 循环里,在 `create_ticket` 分支后加 `submit_refund` 分支:
```python
        elif tc["name"] == "submit_refund":
            actions.append({"type": "refund_form",
                            "draft": {"order_id": tc["args"].get("order_id", ""),
                                      "reason": tc["args"].get("reason")}})
            tool_msgs.append(ToolMessage(
                content="已把『提交退款工单』选项交给用户确认。请用一句话说明这一单可以退款并停止,不要再调用任何工具。",
                tool_call_id=tc["id"], name="submit_refund"))
```
`_agent_messages` 替换为(放宽证据注入条件 + refund 追加):
```python
def _agent_messages(state) -> list:
    """system(有证据就拼,知识路/退款路通用;退款路再拼 order_data + 判定指令) + 跨轮历史。"""
    sys = AGENT_SYSTEM
    if state.get("evidence"):
        sys = sys + _KNOWLEDGE_EVIDENCE_HINT + state["evidence"]
    if state.get("route") == "refund_flow":
        sys = sys + REFUND_JUDGE_HINT + json.dumps(state.get("order_data", {}), ensure_ascii=False)
    return [SystemMessage(sys), *state.get("messages", [])]
```

- [ ] **Step 6: 跑验证通过**

Run: `.venv/bin/pytest tests/graph/test_ch06_nodes.py -k "submit_refund or agent_messages" tests/graph/test_agent_nodes.py -v`
Expected: PASS(新增 3 例 + ch05 agent 节点测试回归——create_ticket 拦截路径不受影响)。

- [ ] **Step 7: Commit**

```bash
git add app/tools/business.py app/tools/registry.py app/core/prompts.py \
  app/graph/nodes.py tests/graph/test_ch06_nodes.py
git commit -m "feat(ch06): submit_refund 工具 + agent_tools 拦成退款表单 + _agent_messages 注入订单/政策"
```

---

## Task 10: script_reply 节点(闲聊/其他 按意图分文案)

**目的:** spec §3 / README L200,L224。`fallback_script` 出口节点:按 `intent` 分文案(闲聊引回产品 / 其他请说具体些),零模型调用。新增函数(不删 `chitchat_reply`,Task 12 组装时才替换)。确定性 → TDD。

**Files:**
- Modify: `app/core/prompts.py`(加 `SCRIPT_REPLY_*`)
- Modify: `app/graph/nodes.py`(加 `script_reply`)
- Test: `tests/graph/test_ch06_nodes.py`(追加)

**Interfaces:**
- Consumes:`state["intent"]`
- Produces:`script_reply(state)->dict`,写 `answer`(闲聊/其他两套文案)+ trace `route="fallback_script"`。

- [ ] **Step 1: 写失败测试**

`tests/graph/test_ch06_nodes.py` 追加:
```python
@pytest.mark.asyncio
async def test_script_reply_by_intent():
    from app.core.prompts import SCRIPT_REPLY_CHITCHAT, SCRIPT_REPLY_OTHER
    chit = await nodes.script_reply({"intent": "闲聊"})
    assert chit["answer"] == SCRIPT_REPLY_CHITCHAT
    assert chit["trace"]["route"] == "fallback_script"
    other = await nodes.script_reply({"intent": "其他"})
    assert other["answer"] == SCRIPT_REPLY_OTHER
```

- [ ] **Step 2: 跑验证失败**

Run: `.venv/bin/pytest tests/graph/test_ch06_nodes.py -k script_reply -v`
Expected: FAIL。

- [ ] **Step 3: 加文案 + 节点**

`app/core/prompts.py` 加(闲聊照 README L200 把话题拉回产品;其他请说具体些):
```python
SCRIPT_REPLY_CHITCHAT = "我暂时还不会回答这个,请告诉我你对我们产品的任何咨询~商品、订单、物流、售后都可以问我哦。"
SCRIPT_REPLY_OTHER = "抱歉,我不太确定您的意思。您可以把问题说得更具体些吗?比如您想咨询的商品、某个订单,或退款/售后问题。"
```
`app/graph/nodes.py`:import 区 `from app.core.prompts import ...` 追加 `SCRIPT_REPLY_CHITCHAT, SCRIPT_REPLY_OTHER`;加节点:
```python
async def script_reply(state) -> dict:
    """闲聊/其他 兜底话术(零模型):按 intent 分文案——闲聊把话题引回产品,其他请用户说具体些。
    分流后立即命中、不进 Agent。(与 ch05 知识路证据弱的 fallback_reply 是两码事,勿混。)"""
    text = SCRIPT_REPLY_OTHER if state.get("intent") == "其他" else SCRIPT_REPLY_CHITCHAT
    return {"answer": text, "trace": {"route": "fallback_script"}}
```

- [ ] **Step 4: 跑验证通过**

Run: `.venv/bin/pytest tests/graph/test_ch06_nodes.py -k script_reply -v`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/core/prompts.py app/graph/nodes.py tests/graph/test_ch06_nodes.py
git commit -m "feat(ch06): script_reply 按意图分文案(闲聊引回产品 / 其他请说具体些)"
```

---

## Task 11: tickets 枚举加「退款」+ create-refund 端点

**目的:** spec §9 / Q2。退款单复用 tickets 表(`ticket_type='退款'`),不新建 refund 表。确定性 → TDD。

**Files:**
- Modify: `app/db/models.py:58`(枚举加 `退款`)
- Create: `sql/ch06-ticket-type.sql`
- Modify: `app/schemas/actions.py`(加 `CreateRefundRequest/Response`)
- Modify: `app/api/actions.py`(加 `create-refund` 端点)
- Test: `tests/test_refund_api.py`(新建)

**Interfaces:**
- Consumes:`repository.create_ticket(conversation_id, description, ticket_type)`(既有;`ticket_type='退款'` 需枚举支持)
- Produces:`POST /api/actions/create-refund` body `{conversation_id:int, order_id:str, reason:枚举}` → `{ticket_no:str, status:"退款申请已提交"}`;固定类目 `reason`:七天无理由/质量问题/发错货/不想要了/其他。

- [ ] **Step 1: 写失败测试**

`tests/test_refund_api.py`(参照 ch05 `tests/test_actions_api.py` 的 `client`/monkeypatch 模式):
```python
def test_create_refund_writes_ticket(client, monkeypatch):
    async def fake_create(cid, desc, ttype):
        assert cid == 7 and ttype == "退款" and "1001" in desc and "质量问题" in desc
        return "T20260716001"
    from app.api import actions
    monkeypatch.setattr(actions.repository, "create_ticket", fake_create)
    r = client.post("/api/actions/create-refund", json={
        "conversation_id": 7, "order_id": "1001", "reason": "质量问题"})
    assert r.status_code == 200
    assert r.json()["ticket_no"] == "T20260716001"
    assert r.json()["status"] == "退款申请已提交"


def test_create_refund_rejects_bad_reason(client):
    r = client.post("/api/actions/create-refund", json={
        "conversation_id": 7, "order_id": "1001", "reason": "乱填"})
    assert r.status_code == 422
```
(`client` fixture 来自共享的 `tests/conftest.py`,pytest 自动注入,无需导入;参照 ch05 `tests/test_actions_api.py` 的用法。)

- [ ] **Step 2: 跑验证失败**

Run: `.venv/bin/pytest tests/test_refund_api.py -v`
Expected: FAIL(端点不存在)。

- [ ] **Step 3: models 枚举 + SQL 迁移**

`app/db/models.py` 的 `Ticket.ticket_type`:
```python
    ticket_type: Mapped[str] = mapped_column(Enum("售后", "投诉", "咨询", "退款"))
```
`sql/ch06-ticket-type.sql`:
```sql
-- ch06: tickets.ticket_type 枚举加「退款」(加值不动旧行,存量安全)

-- 确保中文 ENUM 定义值按 utf8mb4 解析
-- (否则 latin1 默认的 mysql client 会把中文 double-encode,四个枚举值全存成乱码)
SET NAMES utf8mb4;

ALTER TABLE tickets
  MODIFY COLUMN ticket_type ENUM('售后','投诉','咨询','退款') NOT NULL;
```

- [ ] **Step 4: schema + 端点**

`app/schemas/actions.py` 追加:
```python
class CreateRefundRequest(BaseModel):
    conversation_id: int
    order_id: str = Field(min_length=1)
    reason: Literal["七天无理由", "质量问题", "发错货", "不想要了", "其他"]


class CreateRefundResponse(BaseModel):
    ticket_no: str
    status: str = "退款申请已提交"
```
`app/api/actions.py` 追加 import 与端点:
```python
from app.schemas.actions import (
    CreateRefundRequest, CreateRefundResponse, CreateTicketRequest, CreateTicketResponse,
)


@router.post("/api/actions/create-refund", response_model=CreateRefundResponse)
async def create_refund_action(req: CreateRefundRequest) -> CreateRefundResponse:
    """退款表单提交:写 tickets(ticket_type='退款'),描述带订单号 + 固定类目原因。复用 ch02 工单能力。"""
    desc = f"退款申请 订单号={req.order_id} 原因={req.reason}"
    try:
        ticket_no = await repository.create_ticket(req.conversation_id, desc, "退款")
    except SQLAlchemyError:
        logger.exception("退款单创建失败 conv=%s", req.conversation_id)
        raise HTTPException(status_code=503, detail="退款系统暂时不可用,请稍后重试")
    return CreateRefundResponse(ticket_no=ticket_no)
```

- [ ] **Step 5: 跑验证通过 + 应用迁移**

Run: `.venv/bin/pytest tests/test_refund_api.py -v`
Expected: PASS。
对真实库应用迁移(端到端验收前必须跑):
Run: `docker exec -i mewhelp-mysql mysql --default-character-set=utf8mb4 -uroot -proot mewhelp < sql/ch06-ticket-type.sql`

- [ ] **Step 6: Commit**

```bash
git add app/db/models.py sql/ch06-ticket-type.sql app/schemas/actions.py app/api/actions.py tests/test_refund_api.py
git commit -m "feat(ch06): tickets 枚举加退款 + POST /api/actions/create-refund"
```

---

## Task 12: 图组装——5 出口 + refund_flow 链 + 退役 chitchat_reply

**目的:** spec §2/§3。把 5 出口条件边、refund_flow 确定性链接进 `build_graph`;`script_reply` 取代 `chitchat_reply` 节点。确定性 → TDD。

**Files:**
- Modify: `app/graph/build.py`
- Modify: `app/graph/nodes.py`(删 `chitchat_reply`——被 `script_reply` 取代)
- Test: `tests/graph/test_build.py`(改断言)
- Test: `tests/graph/test_reply_nodes.py`(ch05 的 `test_chitchat_reply_*` 改测 `script_reply`)

**Interfaces:**
- Consumes:`nodes.{resolve_reference,classify_intent,fetch_order,retrieve_policy,retrieve_knowledge,confidence_check,main_agent,agent_tools,complaint_reply,script_reply,fallback_reply,log_node}`;`route_by_intent`(5 键)/`confidence_gate`/`should_continue`
- Produces:图节点集含 `fetch_order`/`retrieve_policy`/`script_reply`,不再含 `chitchat_reply`;边:`classify_intent`--5 出口-->{complaint_reply/script_reply/retrieve_knowledge/fetch_order/main_agent};`fetch_order→retrieve_policy→main_agent`;`script_reply→log`。

- [ ] **Step 1: 改图断言**

`tests/graph/test_build.py::test_graph_has_all_nodes` 的节点清单改为:
```python
    for n in ["resolve_reference", "classify_intent", "retrieve_knowledge", "confidence_check",
              "main_agent", "agent_tools", "complaint_reply", "script_reply",
              "fallback_reply", "fetch_order", "retrieve_policy", "log"]:
        assert n in names, f"缺节点 {n}"
    assert "chitchat_reply" not in names, "chitchat_reply 应已被 script_reply 取代"
```

- [ ] **Step 2: 跑验证失败**

Run: `.venv/bin/pytest tests/graph/test_build.py -v`
Expected: FAIL(缺 fetch_order/retrieve_policy/script_reply;chitchat_reply 还在)。

- [ ] **Step 3: 改 build.py**

`app/graph/build.py` 的 `_builder`:节点区把 `b.add_node("chitchat_reply", nodes.chitchat_reply)` 换成 `b.add_node("script_reply", nodes.script_reply)`,并加两节点:
```python
    b.add_node("fetch_order", nodes.fetch_order)
    b.add_node("retrieve_policy", nodes.retrieve_policy)
```
把 `classify_intent` 的条件边整块替换为 5 出口:
```python
    b.add_conditional_edges("classify_intent", route_by_intent, {
        "escalate": "complaint_reply",
        "fallback_script": "script_reply",
        "knowledge": "retrieve_knowledge",
        "refund_flow": "fetch_order",
        "business": "main_agent",
    })
    # 退款子流程确定性链:取单 → 检索政策 → 交主力 Agent 判能不能退
    b.add_edge("fetch_order", "retrieve_policy")
    b.add_edge("retrieve_policy", "main_agent")
```
把 `b.add_edge("chitchat_reply", "log")` 改为 `b.add_edge("script_reply", "log")`。其余(知识路 confidence 闸、ReAct 环、complaint/fallback→log→END)不动。

- [ ] **Step 4: 删 chitchat_reply 节点 + 改其 ch05 测试**

`app/graph/nodes.py`:删除 `chitchat_reply` 函数(及顶部 `CHITCHAT_REPLY = CHITCHAT_REPLY_TEXT` 别名——若已无引用)。`grep -rn "chitchat_reply\|CHITCHAT_REPLY" app` 确认只剩 prompts 里的 `CHITCHAT_REPLY_TEXT`(可留作历史,不强删)。
ch05 的 `tests/graph/test_reply_nodes.py::test_chitchat_reply_fixed_no_actions` 引用了已删的 `nodes.chitchat_reply`,把它整块替换为测 `script_reply`(complaint/fallback 两例不动):
```python
@pytest.mark.asyncio
async def test_script_reply_fixed_no_actions():
    from app.core.prompts import SCRIPT_REPLY_CHITCHAT
    out = await nodes.script_reply({"intent": "闲聊"})
    assert out["answer"] == SCRIPT_REPLY_CHITCHAT
    assert out["trace"]["route"] == "fallback_script"
    assert not out.get("suggested_actions")
```

- [ ] **Step 5: 跑验证通过**

Run: `.venv/bin/pytest tests/graph/test_build.py tests/graph/test_reply_nodes.py -v`
Expected: PASS。

- [ ] **Step 6: Commit**

```bash
git add app/graph/build.py app/graph/nodes.py tests/graph/test_build.py tests/graph/test_reply_nodes.py
git commit -m "feat(ch06): 图组装 5 出口 + refund_flow 链 + script_reply 退役 chitchat_reply"
```

---

## Task 13: 运行时——resume + interrupt 事件 + 节点名对齐

**目的:** spec §8/§11。`run_turn` surfaces 中断、`resume_turn`/`stream_resume` 续跑、`stream_turn` 遇中断吐 `interrupt` 事件;`runtime` 节点名对齐(`script_reply` 答复、`retrieve_policy` citations)。中断 surface 取法 = Task 1 冒烟结论。确定性 → TDD(FakeGraph)。

**Files:**
- Modify: `app/graph/runtime.py`
- Test: `tests/graph/test_runtime.py`(追加)

**Interfaces:**
- Consumes:`get_graph().ainvoke/astream`;`Command`(`from langgraph.types import Command`)
- Produces:`run_turn` 返回 `{conversation_id, state, interrupt}`(`interrupt`=orders 或 None);`resume_turn(cid, resume_value)->{conversation_id, state, interrupt}`;`stream_turn` 遇中断 yield `{"type":"interrupt","kind":str,"orders":list}`;`stream_resume(cid, resume_value)` 同 `stream_turn` 事件流;`DETERMINISTIC_ANSWER_NODES` 用 `script_reply`;citations 来自 `retrieve_knowledge`+`retrieve_policy`。

- [ ] **Step 1: 写失败测试(FakeGraph 驱动中断/续跑)**

`tests/graph/test_runtime.py` 追加(中断形状按 Task 1 冒烟:此处以 `updates` 里出现 `__interrupt__` 键为例;若冒烟测出需 `aget_state` 探,则改 FakeGraph 与 stream_turn 实现一致):
```python
@pytest.mark.asyncio
async def test_stream_turn_emits_interrupt_event(monkeypatch):
    async def fake_create(uid): return 9
    async def fake_append(cid, role, content=None, **k): return 1
    monkeypatch.setattr(runtime.repository, "create_conversation", fake_create)
    monkeypatch.setattr(runtime.repository, "append_message", fake_append)

    class Intr:
        def __init__(self, value): self.value = value

    class FakeGraph:
        async def astream(self, inp, config, stream_mode=None):
            yield ("updates", {"fetch_order": None})
            yield ("updates", {"__interrupt__": (Intr({"type": "select_order",
                     "orders": [{"order_id": "1001"}]}),)})
    monkeypatch.setattr(runtime, "get_graph", lambda: FakeGraph())
    events = [e async for e in runtime.stream_turn("u1", "我要退款", None)]
    intr = [e for e in events if e["type"] == "interrupt"]
    assert intr and intr[0]["kind"] == "select_order"
    assert intr[0]["orders"] == [{"order_id": "1001"}]


@pytest.mark.asyncio
async def test_resume_turn_drives_command(monkeypatch):
    from langgraph.types import Command
    seen = {}

    class FakeGraph:
        async def ainvoke(self, inp, config):
            seen["is_command"] = isinstance(inp, Command)
            seen["resume"] = getattr(inp, "resume", None)
            return {"answer": "这一单可以退款", "messages": []}
    async def fake_get(cid): return object()
    monkeypatch.setattr(runtime.repository, "get_conversation", fake_get)
    monkeypatch.setattr(runtime, "get_graph", lambda: FakeGraph())
    out = await runtime.resume_turn(5, "1001")
    assert seen["is_command"] and seen["resume"] == "1001"
    assert out["conversation_id"] == 5 and out["state"]["answer"] == "这一单可以退款"
```

- [ ] **Step 2: 跑验证失败**

Run: `.venv/bin/pytest tests/graph/test_runtime.py -k "interrupt_event or resume_turn" -v`
Expected: FAIL。

- [ ] **Step 3: 改 runtime.py**

顶部 import 加 `from langgraph.types import Command`。节点名集合更新:
```python
DETERMINISTIC_ANSWER_NODES = {"script_reply", "complaint_reply", "fallback_reply"}
CITATION_NODES = {"retrieve_knowledge", "retrieve_policy"}
```
把 `stream_turn` 里 `if node == "retrieve_knowledge" and upd.get("citations")` 改为 `if node in CITATION_NODES and upd.get("citations")`。
在 `stream_turn` 的 `elif mode == "updates":` 循环最前面加中断检测(形状按 Task 1 冒烟;下例为 `__interrupt__` 键路径):
```python
            if "__interrupt__" in chunk:
                payload = chunk["__interrupt__"][0].value
                yield {"type": "interrupt", "kind": payload.get("type", ""),
                       "orders": payload.get("orders", [])}
                return                    # 图已暂停,结束本次流(前端点选后走 /resume 续流)
```
把 `stream_turn` 主体抽成内部生成器 `_stream_events(cid, stream_source)`(参数为 `_graph_input(...)` 或 `Command(resume=...)`),`stream_turn` 与新 `stream_resume` 都调它:
```python
async def _stream_events(cid: int, stream_source) -> AsyncIterator[dict]:
    config = {"configurable": {"thread_id": str(cid)}}
    actions: list = []
    async for mode, chunk in get_graph().astream(
            stream_source, config, stream_mode=["messages", "updates"]):
        if mode == "messages":
            msg, meta = chunk
            if meta.get("langgraph_node") in ANSWER_NODES:
                text = msg.content if isinstance(msg.content, str) else ""
                if text:
                    yield {"type": "delta", "text": text}
        elif mode == "updates":
            if "__interrupt__" in chunk:
                payload = chunk["__interrupt__"][0].value
                yield {"type": "interrupt", "kind": payload.get("type", ""),
                       "orders": payload.get("orders", [])}
                return
            for node, upd in chunk.items():
                if not isinstance(upd, dict):
                    continue
                if node in DETERMINISTIC_ANSWER_NODES and upd.get("answer"):
                    yield {"type": "delta", "text": upd["answer"]}
                if node in CITATION_NODES and upd.get("citations"):
                    yield {"type": "citations", "items": upd["citations"]}
                if node == "agent_tools":
                    for m in upd.get("messages", []):
                        name = getattr(m, "name", None)
                        if name and name not in ("create_ticket", "submit_refund"):
                            yield {"type": "tool", "name": name}
                if upd.get("suggested_actions"):
                    actions.extend(upd["suggested_actions"])
    if actions:
        yield {"type": "actions", "items": dedup_actions(actions)}
    yield {"type": "done", "conversation_id": cid}


async def stream_turn(user_id, message, conversation_id) -> AsyncIterator[dict]:
    cid = await _ensure_conversation(user_id, conversation_id)
    await repository.append_message(cid, "user", content=message)
    async for ev in _stream_events(cid, _graph_input(user_id, message, cid)):
        yield ev


async def stream_resume(conversation_id: int, resume_value) -> AsyncIterator[dict]:
    """前端点选订单后续跑同一会话的暂停流。"""
    if await repository.get_conversation(conversation_id) is None:
        raise ConversationNotFound(conversation_id)
    async for ev in _stream_events(conversation_id, Command(resume=resume_value)):
        yield ev
```
`run_turn` 结尾把中断 surface 进返回(形状按 Task 1 冒烟):
```python
    final = await get_graph().ainvoke(_graph_input(user_id, message, cid), config)
    return {"conversation_id": cid, "state": final, "interrupt": _interrupt_orders(final)}
```
加 `resume_turn` + helper:
```python
def _interrupt_orders(state: dict):
    intr = state.get("__interrupt__")
    if not intr:
        return None
    return intr[0].value.get("orders")


async def resume_turn(conversation_id: int, resume_value) -> dict:
    """非流式续跑(供 /api/agent、eval):Command(resume) 回填后跑到下一个中断或结束。"""
    if await repository.get_conversation(conversation_id) is None:
        raise ConversationNotFound(conversation_id)
    config = {"configurable": {"thread_id": str(conversation_id)}}
    final = await get_graph().ainvoke(Command(resume=resume_value), config)
    return {"conversation_id": conversation_id, "state": final,
            "interrupt": _interrupt_orders(final)}
```
(注:`tests/graph/test_runtime.py::test_run_turn_creates_conversation_when_none` 的 FakeGraph.ainvoke 返回 dict 无 `__interrupt__`,`_interrupt_orders` 返回 None,该用例仍绿——但其断言只看 `conversation_id`,不受影响。)

- [ ] **Step 4: 跑验证通过(含 ch05 回归)**

Run: `.venv/bin/pytest tests/graph/test_runtime.py -v`
Expected: PASS(新 2 例 + ch05 既有全绿)。

- [ ] **Step 5: Commit**

```bash
git add app/graph/runtime.py tests/graph/test_runtime.py
git commit -m "feat(ch06): runtime resume_turn/stream_resume + interrupt 事件 + 节点名对齐"
```

---

## Task 14: 两入口接线——chat interrupt 帧 + /api/actions/resume + agent interrupt 字段

**目的:** spec §11。流式入口吐 `interrupt` SSE 帧、新增 SSE 续跑端点;非流式入口带 `interrupt` 字段(供 eval)。确定性 → TDD。

**Files:**
- Modify: `app/api/chat.py:30-39`(加 interrupt 帧)
- Modify: `app/api/actions.py`(加 `/api/actions/resume` SSE)
- Modify: `app/schemas/actions.py`(加 `ResumeRequest`)
- Modify: `app/api/agent.py:44-50`(带 interrupt)
- Modify: `app/schemas/agent.py:23-28`(`AgentResponse` 加 `interrupt`)
- Test: `tests/test_refund_api.py`(追加 resume 端点用例)

**Interfaces:**
- Consumes:`runtime.stream_turn`(现含 `interrupt` 事件)、`runtime.stream_resume`、`runtime.run_turn`(现含 `interrupt`)
- Produces:`/api/chat` SSE `{"event":"interrupt","kind":...,"orders":[...]}`;`POST /api/actions/resume` body `{conversation_id:int, order_id:str}` → SSE(同 chat 事件流);`AgentResponse.interrupt: list | None`。

- [ ] **Step 1: 写失败测试(resume SSE 端点)**

`tests/test_refund_api.py` 追加:
```python
def test_resume_endpoint_streams(client, monkeypatch):
    async def fake_stream_resume(cid, resume_value):
        assert cid == 3 and resume_value == "1001"
        yield {"type": "delta", "text": "这一单可以退款"}
        yield {"type": "actions", "items": [{"type": "refund_form", "draft": {"order_id": "1001"}}]}
        yield {"type": "done", "conversation_id": 3}
    from app.api import actions
    monkeypatch.setattr(actions.runtime, "stream_resume", fake_stream_resume)
    r = client.post("/api/actions/resume", json={"conversation_id": 3, "order_id": "1001"})
    assert r.status_code == 200
    body = r.text
    assert "这一单可以退款" in body
    assert "refund_form" in body and "[DONE]" in body
```

- [ ] **Step 2: 跑验证失败**

Run: `.venv/bin/pytest tests/test_refund_api.py -k resume -v`
Expected: FAIL(端点不存在)。

- [ ] **Step 3: chat.py 加 interrupt 帧**

`app/api/chat.py` 的事件分发里,在 `citations` 分支旁加:
```python
                elif ev["type"] == "interrupt":
                    yield _sse({"event": "interrupt", "kind": ev["kind"], "orders": ev["orders"]})
```

- [ ] **Step 4: actions.py 加 resume SSE 端点**

`app/api/actions.py`:import 加 `json`、`from collections.abc import AsyncIterator`、`from fastapi.responses import StreamingResponse`、`from app.graph import runtime`、`ResumeRequest`。加:
```python
def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/api/actions/resume")
async def resume_action(req: ResumeRequest):
    """订单选择器点选后续跑暂停的退款子流程(SSE,事件与 /api/chat 同构)。"""
    async def event_stream() -> AsyncIterator[str]:
        try:
            async for ev in runtime.stream_resume(req.conversation_id, req.order_id):
                if ev["type"] == "tool":
                    yield _sse({"event": "tool", "name": ev["name"]})
                elif ev["type"] == "delta":
                    yield _sse({"delta": ev["text"]})
                elif ev["type"] == "citations":
                    yield _sse({"event": "citations", "items": ev["items"]})
                elif ev["type"] == "actions":
                    yield _sse({"event": "actions", "items": ev["items"]})
                elif ev["type"] == "interrupt":
                    yield _sse({"event": "interrupt", "kind": ev["kind"], "orders": ev["orders"]})
                elif ev["type"] == "done":
                    yield _sse({"event": "done", "conversation_id": ev["conversation_id"]})
        except runtime.ConversationNotFound:
            yield "event: error\n"
            yield _sse({"message": "会话不存在"})
            return
        except Exception:
            logger.exception("续跑失败 conv=%s", req.conversation_id)
            yield "event: error\n"
            yield _sse({"message": "上游暂时不可用,请稍后重试"})
            return
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```
`app/schemas/actions.py` 加:
```python
class ResumeRequest(BaseModel):
    conversation_id: int
    order_id: str = Field(min_length=1)
```

- [ ] **Step 5: agent.py 带 interrupt 字段**

`app/schemas/agent.py` 的 `AgentResponse` 加:
```python
    interrupt: list | None = Field(default=None, description="待补槽位(如订单选择器 orders);无则 None")
```
`app/api/agent.py` 的 `AgentResponse(...)` 构造加 `interrupt=out.get("interrupt")`。

- [ ] **Step 6: 跑验证通过 + 回归**

Run: `.venv/bin/pytest tests/test_refund_api.py tests/test_actions_api.py tests/test_agent_api.py -v`
Expected: PASS(resume 新例 + ch05 建工单/agent 端点回归)。

- [ ] **Step 7: Commit**

```bash
git add app/api/chat.py app/api/actions.py app/api/agent.py app/schemas/actions.py app/schemas/agent.py tests/test_refund_api.py
git commit -m "feat(ch06): chat interrupt 帧 + /api/actions/resume SSE + agent interrupt 字段"
```

---

## Task 15: 前端——订单选择器卡片 + 退款表单(Vibe Coding + 浏览器验收)

**目的:** spec §12 / req#6。前端例外走 **Vibe Coding**(用户描述效果、我改,不套 TDD/review);功能落到前端在用入口,浏览器端到端验收。

**Files:**
- Modify: `app/static/index.html`

**Interfaces:**
- Consumes:SSE `{"event":"interrupt","kind":"select_order","orders":[...]}`(来自 /api/chat 与 /api/actions/resume);SSE `{"event":"actions","items":[{"type":"refund_form","draft":{"order_id":...}}]}`;`POST /api/actions/resume`、`POST /api/actions/create-refund`。

- [ ] **Step 1: 消费 interrupt 帧 → 渲染订单选择器卡片**

在 `streamChat` 帧分发加 `data.event === "interrupt"` 分支,回调 `onInterrupt(data)`;`send()` 里实现 `onInterrupt`:在当前 bot 气泡下渲染可点选订单卡片(订单号/商品/状态/金额),点选任一卡片 → 禁用整组 → `POST /api/actions/resume {conversation_id, order_id}` 并**用同一套 SSE 消费函数**继续把续流渲染到对话里(政策判定 → 退款按钮/说明)。参考 ch05 `renderActions` 的按钮容器与 `getConversationId()` 模式。

- [ ] **Step 2: 消费 refund_form 动作 → 退款表单弹窗**

在 `renderActions` 里加 `a.type === "refund_form"` 分支:渲染「提交退款工单」按钮 → 点开表单弹窗(订单号预填只读 = `a.draft.order_id`;**退款原因固定类目下拉**:七天无理由/质量问题/发错货/不想要了/其他,不预选、必选)→ 提交 `POST /api/actions/create-refund {conversation_id, order_id, reason}` → 成功显示「退款申请已提交:<ticket_no>」。复用 ch05 建工单弹窗表单的 DOM/样式。

- [ ] **Step 3: 样式**

复用 ch05 `.action-bar`/`.action-btn`/弹窗样式;订单卡片加 `.order-card`(可点、hover 高亮、选中禁用灰态),与既有 chip/气泡风格协调。

- [ ] **Step 4: 浏览器端到端验收(需全服务起)**

起 `docker compose up -d` + `make milvus-up` +(如需)`make kb-build` + `sql/ch06-ticket-type.sql` 已应用 + `make dev`;浏览器开 `localhost:8000`,按 spec §18 验收4 真跑:
- 不带订单号问「我要退款」→ 聊天流弹订单选择器卡片;
- 点一张卡片 → 续流:检索政策 → 主力 Agent 判「能不能退」;
- 能退 → 出「提交退款工单」按钮 → 点开表单(订单号只读 + 原因下拉)→ 提交 → 显示退款单号,`SELECT * FROM tickets WHERE ticket_type='退款' ORDER BY created_at DESC LIMIT 1` 见新行。
截图存 `dev-notes/ch06-acceptance4-*.png`。用户会描述期望效果,按 Vibe Coding 迭代到位。

- [ ] **Step 5: Commit**

```bash
git add app/static/index.html && git add -f dev-notes/ch06-acceptance4-*.png
git commit -m "feat(ch06): 前端订单选择器卡片 + 退款表单(interrupt/resume 全链路)"
```

---

## Task 16: 四验收端到端评估脚本 + 真实验收

**目的:** spec §18。四条验收在真实服务上跑通并留痕(非纸面)。

**Files:**
- Create: `scripts/eval_ch06.py`
- Modify: `Makefile`(加 `eval-ch06`)

**Interfaces:**
- Consumes:`/api/agent`(非流式,`run_turn`→带 tool_calls/interrupt)、`/api/actions/resume`(SSE)

- [ ] **Step 1: 写端到端评估脚本**

`scripts/eval_ch06.py`:
```python
"""ch06 四验收端到端评估。需全服务起(mysql/milvus/app)+ 已应用 sql/ch06-ticket-type.sql。
用法:.venv/bin/python -m scripts.eval_ch06"""
import asyncio

import httpx

BASE = "http://localhost:8000"


async def agent(client, msg, cid=None):
    r = await client.post(f"{BASE}/api/agent",
                          json={"user_id": "eval-ch06", "message": msg, "conversation_id": cid})
    return r.json()


async def main():
    async with httpx.AsyncClient(timeout=180) as c:
        results = []

        # 验收1:多轮 物流→退款→物流,每轮意图/指代随上下文(靠 app 日志看 intent/route/coref)
        r1 = await agent(c, "订单1001到哪了")
        cid = r1["conversation_id"]
        await agent(c, "那我想把它退了", cid)          # 指代「它」+ 意图漂移到退款退货
        r3 = await agent(c, "算了,它现在到哪了", cid)   # 又漂回物流
        results.append(("验收1 多轮意图漂移(看日志 intent/route)", True,
                        f"conv={cid};日志应见 route 物流→refund_flow→物流"))

        # 验收3:「这个能退吗」先补指代、再走退款子流程(带订单号,免 interrupt)
        b = await agent(c, "订单2002这个能退吗")
        # 退款子流程会因 submit_refund 拦截产出 refund_form,或说明不能退
        acts = {a["type"] for a in b.get("suggested_actions", [])}
        results.append(("验收3 退款子流程(refund_form 或说明)", bool(b["answer"]),
                        {"actions": list(acts), "answer": b["answer"][:30]}))

        # 验收4(非流式部分):不带订单号问退款 → interrupt 返回订单列表
        b = await agent(c, "我要退款")
        results.append(("验收4 缺单弹订单选择器(interrupt)", bool(b.get("interrupt")),
                        b.get("interrupt")))

        for name, ok, detail in results:
            print(f"{'✅' if ok else '❌'} {name} -> {detail}")

    print("\n验收2(意图 JSON 稳定/怪问题落其他):跑 `.venv/bin/python -m scripts.eval_intent`。")
    print("验收4 前端 interrupt→resume→退款按钮 全链路:见 Task 15 浏览器截图。")
    print("验收1/3 的 coref/intent/route 明细:grep app 日志 `ch05 turn ... intent=.. route=.. trace={..coref..}`。")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: 加 Makefile 目标**

`Makefile` 加(并把 `eval-ch06` 挂进 `.PHONY`):
```makefile
eval-ch06:  ## ch06 四验收端到端评估(需全服务起 + 已应用 sql/ch06-ticket-type.sql)
	.venv/bin/python -m scripts.eval_ch06
```

- [ ] **Step 3: 真实端到端跑四验收**

起全服务 + 应用 `sql/ch06-ticket-type.sql`,然后:
- Run: `make eval-ch06` → 验收1/3/4(非流式)打勾;
- Run: `.venv/bin/python -m scripts.eval_intent`(验收2:JSON 稳定 + 怪问题落其他)、`.venv/bin/python -m scripts.eval_coref`、`.venv/bin/python -m scripts.eval_expand`;
- 验收4 前端全链路见 Task 15。
把结果 + glm 抖动如实记 `dev-notes/ch06.md`(阶段5 真实验收)。

- [ ] **Step 4: 全量回归**

Run: `make milvus-up && .venv/bin/pytest -q`
Expected: ch01–06 全绿(ch05 图/路由/runtime 测试已按 ch06 更新)。

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_ch06.py Makefile && git add -f dev-notes/ch06.md
git commit -m "test(ch06): 四验收端到端评估脚本 + 真实验收留痕"
```

---

## Task 17: 收尾——README + dev-notes finish

**Files:**
- Modify: `README.md`(补 ch06 验收命令)
- Modify: `dev-notes/ch06.md`(finish 段)

- [ ] **Step 1: 检查无断链**

Run: `grep -rn "chitchat_reply" app tests scripts`
Expected: 仅 `prompts.CHITCHAT_REPLY_TEXT`(保留)或全无;无 `nodes.chitchat_reply` 残引用。
Run: `grep -rn "classify(" app tests scripts | grep -v "def classify"` 确认所有调用点都按 `classify(query, history) -> dict` 用法(无残留 `== "闲聊"` 的旧 str 假设)。

- [ ] **Step 2: README + dev-notes**

`README.md` 补 ch06 验收命令块:`make smoke-interrupt`、`make eval-ch06`、三个 prompt eval、日志看 `route=refund_flow`/`coref`、前端 interrupt→resume→退款表单、`sql/ch06-ticket-type.sql` 迁移。`dev-notes/ch06.md` 追 finish 段(测试结果、四验收结论、演示命令、翻车汇总)。

- [ ] **Step 3: 全量回归 + 应用迁移确认**

Run: `make milvus-up && .venv/bin/pytest -q`
Expected: 全绿。确认 `sql/ch06-ticket-type.sql` 已对目标库应用(端到端验收依赖)。

- [ ] **Step 4: Commit**

```bash
git add README.md && git add -f dev-notes/ch06.md
git commit -m "chore(ch06): README + dev-notes 收尾"
```

---

## Self-Review

**Spec 覆盖核对:**
- §1 目标 → Task 1–17 整体;§2/§3 5 出口图+节点表 → Task 3(路由)+ Task 12(组装);§3.1 route_by_intent → Task 3;§4 coref+改写 → Task 5;§5 Query 扩写 → Task 6;§6 意图四件套 → Task 4;§7 refund_flow(fetch_order/retrieve_policy/submit_refund/两出口)→ Task 7/8/9;§8 interrupt/resume 全链路 → Task 1(冒烟)+ Task 13(runtime)+ Task 14(入口)+ Task 15(前端);§9 退款单复用 tickets → Task 11;§10 State 新字段 → Task 2;§11 两入口+流式 → Task 13/14;§12 前端 → Task 15;§13 依赖与验证分层 → 各任务(评估集 Task 4/5/6、TDD Task 2/3/7/8/9/10/11/12/13/14、红线 Task 1、Vibe Task 15、端到端 Task 16);§14 数据模型(tickets 枚举 + list_user_orders mock)→ Task 11/7;§15 文件布局 → 文件结构节;§16 决策 D1–D8 → 各任务(D1 interrupt=Task1/7/13、D2 tickets=Task11、D3 配置种子=Task4、D4 submit_refund=Task9、D5 coref 合一=Task5、D6 扩写仅 refund=Task6/8、D7 5 出口+fallback_script=Task3/10、D8 置信度闸/节点名沿用=不动);§17 风险闸 R1(Task1 冒烟)/R2(Task7 interrupt 前只读 + 测试)/R3(Task4 其他兜底 + eval)/R4(Task11 ALTER 加值);§18 四验收 → Task 16 + Task 15。
- **无遗漏。** 「其他」新意图贯穿 Task 3(路由)/Task 4(分类+eval)/Task 10(script_reply 文案)。

**占位符扫描:** 无 TBD/TODO。每个代码步给完整代码,每个测试步给真实断言。唯二「按冒烟结论落地」的活扣是 **Task 1 → Task 7/13 的中断 surface 形状**:已写明默认取法(`e.args[0][0].value`、updates `__interrupt__` 键)并注明「若冒烟实测不同则按实测改这一行」——这是 fail-fast 红线的正确处理(先实测钉死,不硬编可能错的形状),非模糊占位。

**类型一致性:**
- `ConversationState` 新字段(Task 2)`resolved_query:str`/`intent_confidence:float`/`order_id:str`/`order_data:dict` ↔ 各节点读写一致(resolve_reference 写 resolved_query;classify_intent 写 intent_confidence;fetch_order 写 order_id/order_data)。
- `intent.classify` 返回 `dict{intent,confidence}`(Task 4)↔ classify_intent 节点消费(Task 4)↔ eval 消费(Task 4 Step 9)一致;**已同步改** ch05 `test_forced_rag.py::test_classify_intent_sets_route_in_state`(Task 4 Step 8)。
- `INTENT_TO_ROUTE` 5 个返回值(Task 3)↔ build.py 条件边映射键(Task 12)一致:`escalate/fallback_script/knowledge/refund_flow/business` → `complaint_reply/script_reply/retrieve_knowledge/fetch_order/main_agent`。
- `suggested_actions` 结构:`{type:"refund_form", draft:{order_id, reason}}`(Task 9 agent_tools)↔ 前端消费(Task 15)↔ runtime dedup(Task 13,按 type 去重)一致;沿用 ch05 `{type, draft?}` 形状。
- `order_snapshot`(Task 7)被 `query_order` 工具、`fetch_order` 节点、`list_user_orders` 三处同源调用,键一致(order_id/status/amount/product/tracking_no/created_at)。
- 事件 dict 类型 `{"type":"interrupt","kind","orders"}`(Task 13)↔ chat/resume SSE 映射(Task 14)一致;`interrupt` 字段 `list|None`(Task 13 run_turn/resume_turn)↔ AgentResponse.interrupt(Task 14)↔ eval_ch06 消费(Task 16)一致。
- trace 键不冲突:`coref`(Task5)、`intent`/`route`/`intent_confidence`(Task4,**避开** confidence_check 的 `confidence`)、`fetch_order`/`retrieve_policy`(Task7/8)、`forced_rag`(ch05 保留)、`confidence`=strong/weak(ch05 confidence_check 保留)。
