# Ch06 设计:意图识别与对话管理(占位分流器 → 正式版)

## 1. 背景与目标

ch05 骨架把分流器前段留成占位:`resolve_reference`(指代消解)原样透传、`classify_intent` 单标签七类无 confidence/无「其他」、`retrieve_knowledge` 只做同义词轻扩、`route_by_intent` 无退款子流程。本章按权威章节 `mewhelp-course/ch06-intent-recognition/README.md` 把这个"分流器"做成正式版。

**目标(需求 7 条):**
1. **指代消解 + Query 改写(合一)**:LLM 结合最近几轮历史,把「它能退吗」补全成脱离上下文也能看懂的完整问句;已完整/指代已明确的原样透传,不强改。
2. **Query 扩写**:一句问题泛化成多条侧重点不同的检索查询(强制 JSON,单字段 `queries`,严格 3 条),多 query 检索后去重合并;**只对退款退货/售后核心场景做**,简单 FAQ 不扩;**检索侧现查现用**,库里知识只留一份。
3. **意图识别四件套(LLM prompt 路线,不训模型)**:①八类枚举成选择题(物流/订单/商品咨询/退款退货/售后/投诉/闲聊/**其他**);②强制 JSON `{intent, confidence}`;③边界 few-shot;④「其他」兜底,拿不准归它。
4. **分流精化**:退款退货/售后走确定性子流程——先拿订单数据、再 Query 扩写并强制检索政策条款,最后只把「这一单能不能退」交主力 Agent 判。
5. **槽位处理**:缺订单号不让模型猜,执行阶段弹订单选择器点选回填;退款原因提交退款单时固定类目下拉;需求澄清留主力 Agent。
6. **前端配套**:订单选择器做成聊天流里可点选的订单卡片,点选自动回填、流程接着走;退款单是简单表单,退款原因固定类目下拉。
7. **模型配置**:意图识别默认配够格大模型(先求准);降级路(小模型判+置信度、低置信升级大模型)本章只落配置种子。

**非目标(本章不做):** 微调小模型/BERT 意图分类;跨会话记忆;意图降级路的运行时实现(仅配置种子);两级分类 / 检索式意图识别(README 明标"面试延伸、不在本课范围");ch06 图 21 SVG 暂不动(用户定)。

**权威来源:** `mewhelp-course/ch06-intent-recognition/README.md`(章节)+ 本 spec(本仓设计)。ch05 基线已对齐:节点名 `resolve_reference / retrieve_knowledge / main_agent`,置信度闸在 Agent 之前(生成前证据闸,ch05 D3,ch06 不动)。

---

## 2. 架构与数据流(ch05 骨架 → ch06 5 出口)

ch05 是 `resolve_reference → classify_intent → route_by_intent → {knowledge/business/complaint/chitchat}`。ch06 把前三站做实,并把出口补成 README 的五条:

```
START
 → resolve_reference   指代消解 + Query 改写(吃最近几轮历史;已完整则透传)→ state.resolved_query
 → classify_intent     四件套;→ state.intent(8 类)+ state.intent_confidence
 → route_by_intent  [条件边,五出口]
     投诉            → escalate        complaint_reply(ch05:安抚话术 + 转人工/建工单两按钮,后端不自动执行)  → log
     闲聊 / 其他      → fallback_script  script_reply(按意图分文案:闲聊引回产品 / 其他请说具体些)              → log
     商品咨询         → knowledge       retrieve_knowledge(ch05 强制检索)→ confidence_check →(强:main_agent / 弱:fallback_reply)
     物流 / 订单      → business        main_agent(ch05 ReAct,直接交)
     退款退货 / 售后   → refund_flow     【新增确定性子流程,见 §7】
```

`route_by_intent` 按 README 字面返回 `escalate / fallback_script / knowledge / refund_flow / business`(见 §3.1)。商品咨询/物流/订单/投诉/闲聊五条既有行为不动;新增 `refund_flow` 与「其他」归 `fallback_script`。

---

## 3. 图骨架变更(`app/graph/`)

| 节点 | 类型 | ch05 → ch06 |
|---|---|---|
| `resolve_reference` | Workflow | **做实**:占位透传 → coref+改写合一(吃历史;已完整透传)。写 `resolved_query` |
| `classify_intent` | Workflow | **重写**:七类单标签 → 四件套 8 类 + `{intent,confidence}`;吃 `resolved_query`;写 `intent`/`intent_confidence`/`route` |
| `route_by_intent`(条件边) | Workflow | **扩**:4 出口 → 5 出口(+refund_flow,+其他归 fallback_script) |
| `script_reply`(≈ch05 `chitchat_reply`) | Workflow | **改**:闲聊固定话术 → 按 `intent` 分文案(闲聊/其他);路由键 `fallback_script` |
| `complaint_reply` | Workflow | 不变;路由键改称 `escalate`(行为=ch05 安抚+两按钮) |
| `retrieve_knowledge` / `confidence_check` / `main_agent` / `agent_tools` / `fallback_reply` / `log` | — | **不变**(ch05 基线) |
| `fetch_order` | Workflow(新) | 抽订单号;缺则 `interrupt` 弹订单选择器;拿到后 `query_order` 取订单数据 |
| `retrieve_policy` | Workflow(新) | Query 扩写 3 条 + 多 query 强制检索政策 + 去重合并 → 注入证据 |

主力 Agent(`main_agent`↔`agent_tools`)全图共用一个;refund_flow 也复用它做「能不能退」判定。

### 3.1 八类意图 → 五出口(`route_by_intent`,照 README)

```python
def route_by_intent(state) -> str:
    intent = state["intent"]
    if intent == "投诉":
        return "escalate"
    if intent in ("闲聊", "其他"):
        return "fallback_script"
    if intent == "商品咨询":
        return "knowledge"
    if intent in ("退款退货", "售后"):
        return "refund_flow"       # 确定性子流程:取单、检索政策、只判能不能退
    return "business"              # 物流、订单:直接交主力 Agent
```

条件边映射(`build.py`):`escalate→complaint_reply`、`fallback_script→script_reply`、`knowledge→retrieve_knowledge`、`refund_flow→fetch_order`、`business→main_agent`。返回值与映射键单一来源(沿用 ch05 纪律,避免漂移)。

**⚠️ 两个"兜底"勿混**:`script_reply`(路由键 `fallback_script`,闲聊/其他的固定话术,分流后立即命中、不进 Agent)与 `fallback_reply`(ch05 既有,知识路证据弱的**生成前**兜底,在 `retrieve_knowledge→confidence_check` 之后)是**不同节点、不同触发点**,名字相近但不可合并。

---

## 4. 指代消解 + Query 改写(`resolve_reference` 做实)

一次 LLM 调用完成 coref + 口语归一。输入=本轮消息 + 最近 N 轮历史(从 `state["messages"]` 取,`trim_history` 控长);输出一句完整问句写进 `state["resolved_query"]`。

- **透传边界**:prompt 明确"已完整、指代已明确的原样返回,不引入历史里没有的信息"。README L19 的坑:硬改会越改越偏。
- **下游**:`classify_intent` 与 refund_flow 检索都用 `resolved_query`(意图识别不再改写,README L142)。
- **prompt**(新 `COREF_REWRITE_*`,`app/core/prompts.py`):角色=检索/判意图前的问题补全器;规则=保留关键实体、不臆造、已完整则原样。
- **验证**:纯 Prompt → 评估集 `scripts/eval_coref.py`(多轮带指代用例 + 已完整透传用例)。

---

## 5. Query 扩写(`expand_queries`,`app/core/query_understanding.py`)

README L31-50 的扩写 prompt:把一句问题泛化成 **严格 3 条** 检索友好查询,强制 JSON 单字段 `{"queries": [...]}`,不夹带解释文字;规则=保持产品名/型号一致、不引入原问题没有的参数、每条短且侧重点不同。

- **调用点**:只在 refund_flow 的 `retrieve_policy` 里调(核心场景)。商品咨询 `knowledge` 路沿用 ch05 同义词轻扩,不走 3-query 扩写。
- **检索侧现查现用**:扩出的 3 条各检索一次 → 结果按 chunk id 去重合并 → 拼编号证据。库里知识只留一份,不在入库侧拆存(填 RAG 进阶留的坑,README L58)。
- **结构化输出**:扁平 `list[str]` 单字段,避开 glm 嵌套 502(沿用 ch05 经验)。
- **验证**:纯 Prompt → 评估集 `scripts/eval_expand.py`(核心场景出 3 条且 JSON 稳定;非核心不扩)。

---

## 6. 意图识别四件套(`classify_intent` + `app/core/intent.py` 重写)

按 README L114-152 四件套:

1. **选择题**:八类枚举 + 每类边界("什么该归、什么不该")。八类:物流/订单/商品咨询/退款退货/售后/投诉/闲聊/**其他**。
2. **强制 JSON**:`{"intent": "...", "confidence": 0.0-1.0}`,`with_structured_output`;解析失败/越界 → 归「其他」(最保守,替换 ch05 的回退闲聊)。
3. **few-shot**:边界样例(如「这个还能退吗」→退款退货,不是商品咨询)。
4. **兜底类「其他」**:拿不准归它,不硬塞业务意图。

- **输入**:`resolved_query` + 最近几轮历史(README L86 强调多轮读整段轨迹判当前意图,应对物流→退款→物流的漂移)。
- **模型配置(§7 求准优先)**:默认走够格大模型(`settings.intent_model`,默认回落 `chat_model`);`confidence` 照常输出。
- **降级种子(Q3,不实现运行时)**:`config.py` 落 `intent_small_model / intent_mode='accuracy'|'cost' / intent_conf_threshold` 三个种子字段 + spec 注明"cost 模式=小模型判→confidence<阈值→大模型重判一次"的接入点;本章 `intent_mode` 恒 accuracy,不写升级逻辑。
- **验证**:纯 Prompt → 扩 `scripts/eval_intent.py`(8 类各若干 + 边界 few-shot 命中 + 多轮漂移 + JSON 稳定可解析 + 怪问题落「其他」)。

---

## 7. 退款退货/售后 确定性子流程(`refund_flow`)

按 README L190-198:先拿订单+政策,只把「能不能退」交主力 Agent。

```
route=refund_flow
 → fetch_order:
     order_id = 从 resolved_query/历史抽订单号(正则/模式匹配)
     缺 → interrupt({"type":"select_order","orders": list_user_orders(user_id)})  # 图暂停在 checkpointer
        resume 回填 → order_id = <点选值>
     order_data = query_order(order_id)                         # 复用 ch02 工具
 → retrieve_policy:
     queries = expand_queries(resolved_query + 订单状态)         # 3 条(此步才有料可扩,README L194)
     hits = 多 query 强制检索 → 按 id 去重合并 → 拼编号证据       # 复用 ch04 检索器/arrange_head_tail
 → main_agent(注入 order_data + 政策证据;工具含新增 submit_refund):
     判「能退」→ 调 submit_refund 工具 → agent_tools 拦成 {"type":"refund_form"} 前端可选项(仿 create_ticket,不写库)
     判「不能退」→ 说清原因 + 引导人工(不出按钮)
 → log → END
```

- **两出口用工具承载(D4)**:不额外加分类调用,给主力 Agent 一个 `submit_refund` 工具;能退它就调(被 `agent_tools` 拦成"提交退款工单"按钮),不能退直接答。与 ch05 `create_ticket` 拦截同构。
- **强制检索**:`retrieve_policy` 写死一趟(不让模型凭记忆答能不能退,README L194);与知识路 `retrieve_knowledge` 同理,但产出注入 main_agent 作证据。
- **`list_user_orders(user_id)`**:新增 mock(`app/tools/business.py`),与 `query_order` 随机种子一致(选中即可 `query_order`),按 user_id 稳定出几单。

---

## 8. 订单选择器 interrupt/resume 全链路(验收4;Context7 已核实)

`from langgraph.types import Command, interrupt`。`fetch_order` 缺订单号时 `interrupt(payload)`,图暂停在 `AsyncSqliteSaver`(ch05 已就绪)。

- **非流式(`/api/agent`、eval)**:`ainvoke` 返回的 state 带 `"__interrupt__"` 键 → `result["__interrupt__"][0].value` 取 orders;resume 用 `ainvoke(Command(resume=order_id), config)`(同 thread_id)。`runtime.run_turn` surfaces interrupt;新增 `runtime.resume_turn(cid, resume_value)`。
- **流式(`/api/chat`)**:astream 遇中断 → 吐 SSE `{"event":"interrupt","kind":"select_order","orders":[...]}`;前端渲染订单卡片;点选 → `POST /api/actions/resume {conversation_id, order_id}` → `resume_turn` 用 `Command(resume=order_id)` 续 astream,事件映射复用,子流程接着流式跑完(政策→判能不能退→退款按钮/说明)。
- **resume 语义坑(Context7)**:恢复时被中断节点**从头重跑**,`interrupt()` 返回 resume 值。故 `fetch_order` 中 `interrupt()` **之前**的代码须无副作用(`list_user_orders` 只读,OK);`query_order` 在 interrupt 之后只跑一次。
- **R1 红线**:`interrupt`/`Command(resume)` 在**已装 langgraph 1.2.9** 上、经 `astream(stream_mode=["messages","updates"])` 时中断的确切 surface 形状(`__interrupt__` 更新块 vs 需 `aget_state` 探 pending vs 需切 `stream_events` v3),动手前用 `scripts/smoke_interrupt.py` 实跑钉死(仿 ch05 Task 1)。

---

## 9. 退款工单(复用 tickets 表,Q2)

- **`submit_refund` 工具**(`app/tools/business.py` + 注册):args=order_id(+可选 reason);被 `agent_tools` 拦成 `{"type":"refund_form","draft":{"order_id":...}}` 动作,不写库;合成 ToolMessage 促收敛(同 create_ticket)。
- **前端退款表单(Vibe Coding)**:收到 `refund_form` 动作 → 渲染"提交退款工单"按钮 → 点开表单(订单号预填只读 + **退款原因固定类目下拉**:七天无理由/质量问题/发错货/不想要了/其他)→ 提交 → `POST /api/actions/create-refund`。
- **落库**:`POST /api/actions/create-refund` 写 `tickets` 表,`ticket_type='退款'`,`description` 带订单号+固定类目原因;复用 ch05 建单端点/前端弹窗模式。
- **数据模型**:`tickets.ticket_type` 枚举加 `'退款'`(小 SQL 迁移 + 模型 Enum 同步)。

---

## 10. 会话状态变更(`ConversationState`)

新增字段(无 reducer 标量,`_graph_input` 入口清零,防 ch05 C1 跨轮泄漏):
- `resolved_query: str` — coref+改写产出
- `intent_confidence: float` — 意图 JSON 的 confidence
- `order_id: str` — refund_flow 订单号
- `order_data: dict` — 已查订单数据

`evidence/citations` 复用(政策证据);`suggested_actions` 复用(refund_form)。`intent` 值域扩到 8 类(+其他)。

---

## 11. 两入口 + 流式改动(`runtime.py` / `api/`)

- `runtime.stream_turn`:astream 循环后检测未决中断 → 吐 `interrupt` 事件(R1 定探测方式);新增 `resume_turn(cid, resume_value)`。
- `runtime.run_turn`:`ainvoke` 结果含 `__interrupt__` 时,在返回里带 orders(供 /api/agent、eval)。
- `api/chat.py`:`interrupt` 事件 → SSE `event: interrupt` 帧。
- `api/actions.py`:新增 `POST /api/actions/resume`(SSE 续跑)、`POST /api/actions/create-refund`(写 tickets)。
- `api/agent.py`:`AgentResponse` 带 `interrupt` 字段(orders)供非流式/eval。
- `schemas/`:`ResumeRequest`、`CreateRefundRequest/Response`。

---

## 12. 前端配套(Vibe Coding,不套 TDD/review)

`app/static/index.html`,复用 ch05 `renderActions` + SSE 帧模式:
- **订单选择器**:收 `interrupt`(kind=select_order)帧 → 聊天流渲染可点选订单卡片(订单号/商品/状态/金额);点选 → `POST /api/actions/resume` → 继续消费续流。
- **退款表单**:收 `refund_form` 动作 → "提交退款工单"按钮 → 弹表单(订单号只读 + 退款原因固定下拉)→ 提交 `POST /api/actions/create-refund` → 显示退款单号。

按 memory:功能落到前端在用的入口,验收在浏览器跑。你描述效果我改,不套 brainstorm/TDD/code-review。

---

## 13. 依赖与验证策略(work-req#1)

**无新增组件**:延用 LangGraph(interrupt 是自带能力)、既有模型工厂、ch04 检索器、tickets 表。

| 产出 | 验证 |
|---|---|
| resolve_reference / expand_queries / classify_intent 四件套(纯 Prompt) | **评估集**替 TDD:`eval_coref.py`(含透传)、`eval_expand.py`、扩 `eval_intent.py`(8 类+其他+边界+多轮+JSON 稳定) |
| routing 5 出口 / State / fetch_order 抽单与 list_user_orders / retrieve_policy 去重组装 / submit_refund 拦截 / resume 管线 / create-refund 端点 | **TDD** |
| interrupt/resume/astream 中断 surface | **红线冒烟** `smoke_interrupt.py`(Context7 先行,已做) |
| 订单选择器卡片 / 退款表单(前端) | **Vibe Coding** + 浏览器验收 |
| 端到端四验收 | `eval_ch06.py` + 浏览器 chrome-devtools 真跑 + 截图 |

涉及 LangGraph/FastAPI/SQLAlchemy 具体 API,动手前 Context7 复核(work-req#3)。

---

## 14. 数据模型

- **`tickets.ticket_type` 枚举 +`'退款'`**:`app/db/models.py` Enum 同步 + `sql/` 小迁移(ALTER)。
- 无其它新表。`list_user_orders` 是 mock(不落库,与 `query_order` 一致的随机种子)。checkpointer sqlite 表由 AsyncSqliteSaver 自建。

---

## 15. 文件布局

**改:** `app/graph/{routing,nodes,build,state,runtime}.py`、`app/core/{intent,query_understanding,prompts}.py`、`app/api/{actions,agent,chat}.py`、`app/schemas/{actions,agent}.py`、`app/tools/{business,registry}.py`、`app/config.py`、`app/db/models.py`、`app/static/index.html`、`sql/`(枚举迁移)、`Makefile`(eval-ch06)。

**新增:** `scripts/{smoke_interrupt,eval_coref,eval_expand,eval_ch06}.py`。

---

## 16. 决策记录

- **D1** 订单选择器用 LangGraph `interrupt`/`Command(resume)` 做 HITL(用户 Q1),非两轮重入。checkpointer 已就绪。
- **D2** 退款工单复用 `tickets` 表(`ticket_type='退款'`),不新建 refund 表(用户 Q2)。
- **D3** 意图降级路(小→大)本章只落配置种子,默认走大模型求准(用户 Q3 + README"先求准")。
- **D4** 退款两出口用 `submit_refund` 工具承载(仿 ch05 create_ticket 拦截),不加额外分类调用。
- **D5** 指代消解 + Query 改写合一节点 `resolve_reference`;意图识别只判、不再改写。
- **D6** Query 扩写(3 条 JSON)只在 refund_flow 检索侧,商品咨询走 ch05 轻扩;库里知识一份。
- **D7** `route_by_intent` 按 README 字面五出口;闲聊+其他共用 `fallback_script`(script_reply 按意图分文案)。
- **D8** 沿用 ch05:置信度闸在 Agent 之前(生成前证据闸,ch05 D3),ch06 不动;节点名 resolve_reference/retrieve_knowledge/main_agent。

---

## 17. 风险闸(fail-fast)

- **R1** interrupt/resume/astream 中断 surface 形状(langgraph 1.2.9):`smoke_interrupt.py` 先钉死,不通停下按选型内解决或问用户。
- **R2** resume 时节点从头重跑:`fetch_order` 的 interrupt 前代码须只读(设计已保证);回归测试覆盖"缺单→interrupt→resume→query_order 只一次"。
- **R3** 意图 JSON 在 glm 上稳定性 + 「其他」兜底:结构化输出 + 解析失败归其他;eval 测 JSON 可解析率与怪问题落其他。
- **R4** `tickets` 枚举 ALTER 对存量数据:迁移用 `MODIFY COLUMN ... ENUM(...)` 加值,不动旧行。

---

## 18. 验收映射

1. 多轮(物流→退款→物流)每轮意图对 + 指代对 → `eval_ch06.py` 多轮用例 + 扩 `eval_intent.py`。
2. 意图 JSON 稳定可解析、怪问题落「其他」→ `eval_intent.py`。
3. 「这个能退吗」先补指代、再走退款子流程拿订单+政策 → trace 日志(`resolved_query` + `refund_flow` + 政策证据)+ eval。
4. 浏览器不带订单号问退款 → 弹订单选择器(interrupt)→ 点选(resume)→ 子流程走完 → chrome-devtools 真跑 + 截图。
