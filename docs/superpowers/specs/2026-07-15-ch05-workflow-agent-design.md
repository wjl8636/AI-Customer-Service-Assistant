# Ch05 设计:Workflow + Agent 混合架构(LangGraph 骨架)

> 状态:定稿(brainstorm 通过,待写实现计划)
> 章节权威来源:`mewhelp-course/ch05-workflow-agent/README.md`
> 上游选型定死:LangGraph(图编排 + State + checkpointer);前端沿用原生 HTML/CSS/JS,仅加两个动作按钮。

## 1. 背景与目标

前四章的客服后端是「单轮 tool-calling」:`app/core/agent.py` 走两次模型调用——turn1 `bind_tools` 定工具、执行、turn2 不带工具强制收敛。它接不住「手机号查物流」这类**依赖上一步结果**的多步业务(查订单拿到物流单号→再查物流),也没有确定性骨架把「知识类必先检索」「投诉不进 Agent」这些硬约束焊死。

本章把后端升级成**生产级混合架构**:用 LangGraph 编排一条**确定性骨架**(Workflow),把指代消解、意图识别、按意图分流、置信度兜底、日志这些硬约束嵌在路径节点上;骨架中心放一个**主力 Agent** 节点,用 ReAct 循环临场决定调哪个工具、走几步收敛。骨架管稳定可控可观测,主力 Agent 管组合多变的灵活业务。

**非目标(本章不做,留后)**:意图识别/指代消解正式版(ch06)、上下文管理升级(ch07)、MCP 接入(ch08)、飞轮入库与正式置信度检查(ch09)。

## 2. 架构与数据流

```
START → coref(指代消解:透传) → classify_intent(意图识别:单标签七类JSON)
      → [route_by_intent 条件边,写死 7→4]
          ├ chitchat  → chitchat_reply(固定话术,零模型调用) → log → END
          ├ complaint → complaint_reply(安抚话术 + 抛出「转人工」「建工单」可选项) → log → END
          ├ knowledge → forced_rag(强制检索) → confidence_check(生成前证据闸)
          │                 ├ 证据弱 → fallback_reply(兜底话术 + 落低置信池) → log → END
          │                 └ 证据强 → main_agent ↘
          └ business  → ───────────────────────────────────────────────→ main_agent
   main_agent(ReAct 环:agent_llm ↔ agent_tools) → log → END
```

- 前三步(coref / classify_intent / route)+ 末步(log)走确定性 Workflow 路径;第四步 main_agent 是 Agent 临场判断;confidence_check 是知识路检索后、进 Agent 前的确定性证据闸。
- **置信度闸位置(重要,偏离章节示意图)**:章节 181 行把 confidence_check 画在 main_agent 之后,那是"先看清骨架"的示意(连闲聊/投诉边都省了)。本章最简置信度=**生成前证据闸**,必须放在 forced_rag 之后、agent 之前,原因:① req#7 的判据是 ch04 检索时的证据分/覆盖度,检索后即可知;② 主力 Agent 答案逐 token 流式吐给前端,弱证据必须在生成前短路,否则幻觉答案已流出;③ 与 ch04「生成前先判证据够不够」一致。业务路无检索证据,不设此闸,直进 Agent。
- 知识类:检索结果作为上下文随问题一并交给 main_agent,模型不用「记得」检索——写死在流程里绕不过去。
- 投诉:不进 Agent,回安抚话术并把动作可选项交前端,**后端不自动执行**。

## 3. 图骨架节点(`app/graph/`)

| 节点 | 类型 | 职责 | 本章实现 |
|---|---|---|---|
| `coref` | Workflow | 指代消解 | 最简:原样透传(正式版留 ch06) |
| `classify_intent` | Workflow | 意图识别 | 简单 prompt + 结构化输出,判七类,输出 JSON |
| `route_by_intent` | 条件边函数 | 7→4 分流 | 写死 dict 映射 |
| `forced_rag` | Workflow | 知识类强制检索 | 复用 `retrieval.search_knowledge` + `selfcheck`,产出编号证据/citations/证据强弱信号 |
| `agent_llm` | Agent | ReAct 推理步 | `bind_tools` 调模型;streaming |
| `agent_tools` | Agent | ReAct 行动步 | 复用 `tools/infra.execute_tool_call`;含 create_ticket 拦截 |
| `confidence_check` | 条件边+节点 | 置信度兜底(生成前证据闸) | 知识路:证据弱→路由到 fallback_reply;证据强→放行进 agent_llm。仅知识路,业务路不经此节点 |
| `fallback_reply` | Workflow | 兜底出口 | 兜底话术 + `insert_low_confidence` 落池 |
| `complaint_reply` | Workflow | 投诉出口 | 安抚话术 + `suggested_actions=[转人工, 建工单]` |
| `chitchat_reply` | Workflow | 闲聊出口 | 固定话术,不调模型 |
| `log` | Workflow | 日志记录 | 留痕 trace + 落 MySQL 一条 assistant 消息 |

### 3.1 七类意图 → 四出口(`route_by_intent` 写死)

| 出口 | 归入意图 | 进 Agent | 预检索 |
|---|---|:--:|:--:|
| knowledge | 商品咨询、退款退货 | 是 | **强制先检索** |
| business | 物流、订单、售后 | 是 | 否(Agent 自调工具) |
| complaint | 投诉 | 否 | 否 |
| chitchat | 闲聊 | 否 | 否 |

- 退款退货虽属售后,但归 **knowledge**:先检索退货政策再交 Agent,Agent 仍可自调订单工具。其余售后归 business。
- classify_intent 输出单标签(本章最简);多意图(如「又投诉又问 business」)留 ch06 正式版。

## 4. 主力 Agent:ReAct 循环(req#4)

`agent_llm ↔ agent_tools` 两节点 + `should_continue` 条件边构成环:

```
agent_llm → [should_continue]
    ├ 有 tool_calls 且未触停 → agent_tools → agent_llm
    └ 无 tool_calls / 触停 → confidence_check
```

**停止条件 & token 控制**:
1. 无 tool_calls → 收敛(简单问一步即出);
2. `steps ≥ max_agent_steps`(config,默认 6)→ 强停,走兜底话术(不能把工具结果当答复吐出);
3. `tokens_used` 由每次响应 `usage_metadata` 累加进 State,只供排障与 ch09 成本统计,不当停止条件;
4. 模型判信息不足 → 产出一句追问、结束本轮等用户。

知识路进 agent_llm 时上下文已带编号证据,系统提示指示「证据已给,别再调 query_faq;可自调订单等工具」——对上退款退货场景。

## 5. 会话状态与持久化(req#5)

`app/graph/state.py` 定义贯穿全图的 `ConversationState`(TypedDict):

| 字段 | 类型 | 说明 |
|---|---|---|
| `messages` | `Annotated[list, add_messages]` | 对话历史,跨轮由 checkpointer 续接 |
| `user_id` / `conversation_id` | str / int | 身份;conversation_id 建在 MySQL |
| `intent` / `route` | str | 七类 / 四出口 |
| `evidence` / `citations` / `evidence_strong` | str / list / bool | 知识路检索产物 + 强弱信号 |
| `answer` | str | 最终回答 |
| `steps` / `tokens_used` | int | ReAct 停止与预算 |
| `suggested_actions` | list | 转人工/建工单可选项 |
| `trace` | dict | 留痕:intent/route/是否检索/步数/token/是否兜底 |

**持久化:AsyncSqliteSaver**(`langgraph-checkpoint-sqlite`),独立 `.sqlite` 文件,`thread_id = str(conversation_id)`;跨轮历史由 checkpointer + `add_messages` 自动续接。MySQL 仍存工单、低置信池、消息审计(log 节点写),两者职责分明。

> 选型说明:LangGraph 自带 checkpointer 只有 InMemorySaver(不持久)、SqliteSaver/AsyncSqliteSaver、PostgresSaver;**无官方 MySQL 版**。项目业务库虽是 MySQL,但「自带 checkpointer」+「生产级持久化」二者取 AsyncSqliteSaver(用户拍板)。

## 6. 转人工 / 建工单动作机制(req#8)

转人工与建工单是**两件事**,都由用户在前端点按钮触发,**后端不自动执行**。

### 6.1 后端产出可选项
- 新增 SSE 事件 `actions`,载荷:`{"event":"actions","items":[{"type":"transfer_human"},{"type":"create_ticket","draft":{"description":..,"ticket_type":..}}]}`。可只给一个。
- **complaint_reply**(确定性):安抚话术 + 两个可选项;建工单草稿 description=用户原话、ticket_type=投诉。
- **main_agent(拦截 create_ticket 为「提议」)**:Agent 用「调 create_ticket」表达判断;`agent_tools` 拦截该调用——**不写库**,把 args 转成「建工单」可选项存进 `suggested_actions`,并回一条合成 ToolMessage(「工单选项已呈现给用户,请简要说明并停止」)让模型收尾收敛。真正写库只在按钮端点发生。不新写业务工具(create_ticket 仍是 ch02 那个)。

### 6.2 后端按钮端点
- **建工单**:`POST /api/actions/create-ticket`(body: `conversation_id` + `ticket_type` + `description`,由前端表单收集)→ 调 ch02 `create_ticket` 写 tickets 表 → 返回 `ticket_no`。
- **转人工**:纯前端,无后端动作(本章不接真人系统)。

### 6.3 前端(现有 `app/static/index.html` 上加)
- 收到 `actions` 事件 → 在该气泡下渲染独立按钮(转人工 / 建工单,给几个渲染几个)。
- 点**转人工**:前端直接显示「已转接人工客服」,再蹦一条 bot 气泡「您好,我是客服小猫,请问有什么可以帮您的」(前端模拟)。
- 点**建工单**:弹出**工单创建表单弹窗**——「反馈类别」(下拉,固定 售后/投诉/咨询,不预选、必选)+「反馈描述」(文本域,留空、必填)。二者由用户当场填写,**不预填后端草稿**(`draft` 仅是后端提议语义,表单不消费);提交前前端校验必填,通过后带 `ticket_type`+`description` 调 `/api/actions/create-ticket`,成功显示「工单已创建 T…」并置灰该气泡的建工单按钮;点取消或遮罩/Esc 关闭、不建单、按钮可重开。
- 两按钮互不绑定;用户哪个都不点、继续发消息 → 当普通对话正常走下去(按钮可保留或置灰,不阻断输入)。

## 7. 两个 API 入口都走图(定死:单一骨架)

- `/api/chat`(前端 SSE,契约扩一个 actions 帧):驱动 `graph.astream(input, config, stream_mode=["messages","updates"])`——`agent_llm`/`chitchat_reply`/`complaint_reply` 的 token 按 `langgraph_node` 元数据过滤成 `delta`;`agent_tools` 出 `tool`;`forced_rag` 出 `citations`;动作出 `actions`;末尾 `done`。
- `/api/agent`(评估/单测,非流式):`graph.ainvoke` 后从终态组装 `AgentResponse`(answer + 收集的 tool_calls/tool_results + suggested_actions)。
- 旧 `app/core/agent.py` 两次调用编排被图取代;同步改 `scripts/eval_agent.py` 与相关单测(test_agent_orchestration / test_agent_api / test_agent_stream / test_chat_api)。

## 8. 依赖与验证策略

### 8.1 新增依赖(pyproject)
`langgraph`、`langgraph-checkpoint-sqlite`、`aiosqlite`。动手前用 Context7 核实 StateGraph / add_conditional_edges / AsyncSqliteSaver / astream(stream_mode="messages") 的当前接口(已初核)。

### 8.2 验证分层(work-req #1:不可单测的换评估集)
- **确定性代码走 TDD**:route_by_intent 映射、should_continue 停止条件、token 累加、SSE 事件映射、create_ticket 拦截逻辑、`/api/actions/create-ticket` 端点、State reducer 接线。
- **Prompt/数据走标注评估集**:
  - `scripts/eval_intent.py`:七类意图各若干条标注样例,核对 classify_intent 判对率;
  - `scripts/eval_ch05.py`:5 条验收场景端到端跑,断言走对路径/节点(读 trace)。

## 9. 数据模型

无新表。复用:`conversations` / `messages`(log 节点写)、`tickets`(建工单端点写)、`low_confidence_questions`(confidence_check 落池)。checkpointer 的 sqlite 表由 `AsyncSqliteSaver.setup()` 自建,不入 MySQL。

## 10. 文件布局(预估)

```
app/graph/
  __init__.py
  state.py           # ConversationState
  nodes.py           # 各节点函数
  routing.py         # route_by_intent / should_continue
  build.py           # StateGraph 组装 + compile(checkpointer)
  runtime.py         # 图的单例 + astream/ainvoke 封装(供两入口调)
app/api/
  chat.py            # 改:驱动 graph.astream
  agent.py           # 改:驱动 graph.ainvoke
  actions.py         # 新:POST /api/actions/create-ticket
app/core/prompts.py  # 加:INTENT_CLASSIFY / 兜底话术 / 安抚话术 / 闲聊话术
app/static/index.html # 改:actions 帧渲染 + 两按钮交互
scripts/bare_agent_loop.py  # 新:祛魅裸循环
scripts/eval_intent.py      # 新:意图分类评估
scripts/eval_ch05.py        # 新:5 验收端到端评估
```

## 11. 决策记录

- **D1 checkpointer=AsyncSqliteSaver**:LangGraph 无官方 MySQL checkpointer;取自带且持久的 sqlite 版,与 MySQL 业务库解耦。
- **D2 两入口都走图**:单一骨架,/api/agent 走 ainvoke、/api/chat 走 astream;旧两次调用编排退役。
- **D3 confidence_check 在 forced_rag 之后、agent 之前**(生成前证据闸,仅知识路)。偏离章节 181 行"agent 之后"的示意图,理由:req#7 判据是检索时证据分、流式下弱证据须生成前短路、与 ch04「生成前判证据」一致(详见 §2 说明)。业务路不设此闸。
- **D4 create_ticket 拦截为提议**:Agent 判断合适时不写库,转成前端可选项;写库只在按钮端点。不新写业务工具。
- **D5 投诉不进 Agent**:硬约束焊进轨道(章节核心),complaint_reply 确定性给安抚话术 + 可选项。
- **D6 单标签意图**:多意图留 ch06。
- **D7 query_logistics 造真实依赖(执行中用户拍板)**:为让验收5「先查订单再查物流、ReAct 走不止一步」真实成立(否则两工具都吃 order_id、gpt-5.5 会并行一步调俩、非顺序多步),把 `query_logistics` 的入参从 `order_id` 改为 `tracking_no`,而 `tracking_no` 只能从 `query_order` 的返回里取到——形成真实数据依赖,贴合章节「手机号→订单→物流单号→物流」原例。属**修改现有工具**(非新写业务工具),波及 query_order(增 tracking_no 字段)+ test_tools_mock + 编排测试里的假 args。链式冒烟证实:query_order(1001)→query_logistics(SF…)→收敛。

## 12. 风险闸(fail-fast)

- **R1 LangGraph 流式与前端 SSE 对齐**:`stream_mode="messages"` 按 `langgraph_node` 过滤只吐 agent_llm/reply 节点的 token,别把 classify_intent 等中间模型调用的 token 漏进 delta。第一批实现即端到端冒烟。
- **R2 AsyncSqliteSaver 在 async server 下的并发**:确认 async 版 + `.setup()` 时机,thread_id 映射稳定。
- **R3 create_ticket 拦截后循环收敛**:合成 ToolMessage 后模型要能停,不再反复调 create_ticket——评估集里放投诉夹带业务问题的样例验证。

## 13. 验收映射

| # | 验收 | 落点 |
|---|---|---|
| 1 | 政策类问题日志见强制检索节点 | log 节点 trace 含 forced_rag |
| 2 | 「订单1001物流到哪」Agent 自调工具 | business 路 → agent 调 query_logistics |
| 3 | 「我要投诉」出两独立按钮;转人工前端显示已转接+问候,建工单才写 tickets,分开,都不点正常对话 | complaint_reply + actions 事件 + 前端交互 + 建工单端点 |
| 4 | 闲聊拿固定话术 | chitchat_reply |
| 5 | 复杂问 ReAct 走不止一步 | agent_llm↔agent_tools,trace steps>1 |
