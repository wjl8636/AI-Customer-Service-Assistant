# MewHelp Ch02 设计:Function Calling 工具链(工具链长在客服聊天里)

## 目标

给 ch01 的纯对话客服装上「查数据」的能力,而且**能力要长在客服聊天本身**:用户在聊天页(ch01 那个 SSE 流式对话页)问一句,后端就用 Function Calling 让 `glm-5.2` **自己决定调哪个工具**,执行工具、把结果回灌给模型,组织最终回答;最终回答**仍逐 token 流式吐出**,工具执行那一段先推个状态帧;聊天记录(含工具调用与结果)落 `conversations`/`messages` 表;聊天页在气泡里显示这轮调了哪个工具(工具轨迹小徽章)。本章只做**单轮**工具调用(模型调一次工具就收敛),不做多轮 Agent Loop、不做向量检索/RAG。

### 验收标准(在浏览器聊天页上验)

1. 浏览器打开聊天页,问「订单 1001 的物流到哪了」,能看到模型**选中工具**(`query_logistics`,气泡带工具徽章)并按返回结果作答。
2. 问「退货政策是什么」,`query_faq` 关键词查 `faq` 表**查得到**并作答。
3. 换个说法问「邮费是多少」,确认关键词查表**查不出来**(答案其实在「运费怎么算」条目里,但字面 LIKE 对不上)——这个漏召回是预期结果,如实记录,留给 ch03 向量检索升级。

## 技术栈(定死,不可自行更换)

- Python 3.12 + FastAPI + LangChain(`@tool` 装饰器)
- SQLAlchemy 2.0 异步 + MySQL(Docker 起)
- LLM 链路沿用 ch01:应用层 `ChatOpenAI` 说 OpenAI 协议,直连 `CHAT_BASE_URL` 指定的上游
- 前端:ch01 的静态聊天页(`app/static/index.html`,原生 JS + SSE),改造走 **Vibe Coding**(用户描述效果直接改,不套 brainstorm/TDD/code review)
- 依赖管理:uv
- **不使用 Redis**(用户在 brainstorm 中主动撤销,原「工具结果缓存」作废)

> 具体库/框架/API 的签名与用法,在 writing-plans 阶段先用 Context7 核对最新官方文档再落计划。核心新增面已核对:`model.astream([...])` 流式吐 `AIMessageChunk` token(与 ch01 `/api/chat` 现用法一致);第一轮为拿完整 `tool_calls` 用 `ainvoke`(非流式)。

## 已知最大风险(开发早期先验证)

上游模型的 tool calling(`bind_tools`)支持度未知。这是整个工具链的地基:**第一个开发任务就做一次真实 tool-calling 冒烟**(绑一个最简单工具,确认上游能返回结构化 `tool_calls`),不通就停下问用户,不自行换方案。

## 架构:工具编排核心 + 两个出口

```
[ 浏览器聊天页 ] app/static/index.html ──SSE──> FastAPI :8000
[ LLM 链路 ]     FastAPI ──OpenAI协议──> CHAT_BASE_URL 指定的上游
[ 数据 ]         FastAPI ──SQLAlchemy async(asyncmy)──> MySQL :3306

编排核心(app/core/agent.py)
  ├─ 共享 _prepare_turn:建/取会话 → 落 user → turn1 bind_tools 定工具 → 落 assistant → 执行工具 → 落 tool
  ├─ stream_agent_turn() 流式,产出 SSE 事件流       → 供 /api/chat(前端主入口)
  └─ run_agent_turn()    非流式,一次性返回轨迹+答案 → 供 /api/agent(程序化/测试出口)
```

**两个出口共用一个编排核心**:把「会话身份 + 落 user + turn1 定工具 + 落 assistant + 执行工具 + 落 tool」抽成共享逻辑 `_prepare_turn`,`stream_agent_turn`(流式)和 `run_agent_turn`(非流式)都调它,不重复实现。差别只在最后收敛那一步:流式用 `astream` 逐 token 吐 + 发 SSE 帧,非流式用 `ainvoke` 拿完整答案一次性返回。

**前端主入口是 `/api/chat`**(SSE 流式、带工具链、会话落 DB);**`/api/agent` 是非流式 JSON 的程序化/测试出口**(供 eval、脚本、单测一眼看工具轨迹)。`/api/extract`、`core/llm.py` 不动。

## 目录结构(本章在 ch01 的 app/ 上新增/改动)

```
app/
  db/                  # 本章新建:base.py(async engine/session)/ models.py(4 表映射)/ repository.py(CRUD)
  tools/               # 本章新建:business.py(5 工具)/ registry.py(注册表)/ infra.py(执行基建)
  core/
    agent.py           # 本章新建:_prepare_turn 共享核心 + stream_agent_turn(流式)+ run_agent_turn(非流式)
    prompts.py         # 加 AGENT_SYSTEM(客服+工具统一 system prompt)
    memory.py          # 沿用 ch01 trim_history;SessionStore 弃用保留(见下)
    llm.py             # 沿用 ch01,不动
  api/
    chat.py            # 升级 /api/chat 为带工具链的 SSE 流式;请求体加 user_id/conversation_id
    agent.py           # 本章新建:POST /api/agent(程序化/测试出口)
    extract.py         # 沿用 ch01,不动
  schemas/
    chat.py            # ChatRequest 改 user_id / conversation_id
    agent.py           # 本章新建
  static/index.html    # 聊天页接工具链(Vibe Coding):请求体、SSE 新帧解析、工具徽章、conversation_id 持久化
sql/                   # 本章新建:ch02-ddl.sql / ch02-seed.sql(均自带 SET NAMES utf8mb4)
scripts/               # eval_agent.py / demo_agent.sh
```

**弃用但保留(用户拍板)**:内存 `SessionStore` 和 ch01 的 `CUSTOMER_SERVICE_PROMPT` 不被 `/api/chat` 引用(会话改落 DB、prompt 改用 `AGENT_SYSTEM`),但**文件保留不删**——删除是额外风险,留着无害。

## 数据模型(4 张表)

以 [sql/ch02-ddl.sql](sql/ch02-ddl.sql) 为权威;ORM 只映射,不用 `create_all`。DDL/seed 均自带 `SET NAMES utf8mb4`——否则 docker mysql client 默认 latin1 会把中文 ENUM 值/数据 double-encode 存成乱码。

| 表 | 主键 | 关键字段 | 说明 |
|---|---|---|---|
| `conversations` | `id` BIGINT 自增 | `user_id`、`status` ENUM(进行中/已转人工/已结束)、时间戳 | 会话壳 |
| `messages` | `id` BIGINT 自增 | `conversation_id` FK、`role` ENUM(user/assistant/tool)、`content`、`tool_calls` JSON、`tool_call_id` | 消息流水 |
| `faq` | `id` BIGINT 自增 | `question`、`answer`、`category` | `query_faq` 数据源 |
| `tickets` | `ticket_no` VARCHAR | `conversation_id` FK、`description`、`ticket_type`、`status` | `create_ticket` 落地 |

## 工具层:5 个 `@tool`

| 工具 | 对模型暴露参数 | 行为 | 数据源 |
|---|---|---|---|
| `query_order` | `order_id: str` | 订单状态/金额/时间/商品名 | mock(以入参为随机种子,可复现) |
| `query_product` | `product_name: str` | 价格/库存/规格 | mock(可复现) |
| `query_logistics` | `order_id: str` | 物流状态/位置/时间线 | mock(可复现) |
| `query_faq` | `keyword: str` | `question LIKE %keyword%`,命中空返回「未找到」 | 真实查 `faq` |
| `create_ticket` | `description`、`ticket_type` | 生成 `ticket_no` 写 `tickets`,会话置「已转人工」 | 真实写 `tickets` |

`create_ticket.conversation_id` 用 `InjectedToolArg` 系统注入,不暴露给模型(自增主键模型不该编造);`create_ticket` 不自动重试(无幂等设施,避免重复工单)。工具基础设施(`infra.py`):注册/校验/超时/重试/错误回灌,异常不冒泡。

## 编排核心(app/core/agent.py:共享 + 两出口)

### 共享 `_prepare_turn`(两出口都走)

```
1. 会话身份:无 conversation_id → 建会话拿自增 id;有 → 校验存在(不存在抛 ConversationNotFound)
2. 落库:messages.append(role=user, content=message)
3. 组装上下文:读会话历史 → 转 LangChain 消息 → trim
   [AGENT_SYSTEM] + [跨轮历史:user 与「content 非空且非工具调用」的 assistant] + [本轮 user]
4. turn1(非流式,必须拿完整 tool_calls):model.bind_tools(get_all_tools()).ainvoke(messages) → AIMessage
5. 落库:messages.append(role=assistant, content=ai.content or None, tool_calls=ai.tool_calls or None)
6. 无 tool_calls → 到收敛点 A;有 tool_calls → 并行执行(asyncio.gather,每调用独立 session)→ 落 tool → 收敛点 B
```

### 出口差异(唯一区别在收敛)

- **`stream_agent_turn`(流式,async generator 产出事件 dict)**:
  - 收敛点 A(无工具):把 turn1 的 `ai.content` 作为 `delta` 帧吐出(整块一帧),再 `done`。
  - 收敛点 B(有工具):对每个 tool_call 先发 `tool` 帧(工具名),再 `model.astream([...messages, ai, ...tool_messages])`(不 bind_tools)**逐 token 发 `delta` 帧**;累积文本落 assistant;最后 `done`。
- **`run_agent_turn`(非流式)**:收敛点 B 用 `model.ainvoke([...messages, ai, ...tool_messages])`(不 bind_tools)拿完整 `final`,落 assistant,返回 `AgentResult`(轨迹 + 答案);收敛点 A 直接返回 ai.content。

**单轮边界(硬约束)**:收敛那一步**不 `bind_tools`**,模型即使想再调工具也无从发起,天然收敛,不出现第二轮工具执行。两个出口同此约束。

**跨轮上下文**:只回放 `user` 与「`content` 非空且**无** `tool_calls`」的 `assistant`(每轮提问与最终回答);带 `tool_calls` 的 assistant(常夹带 preamble)与 `tool` 消息不跨轮回放,避免污染续接;但全部消息仍完整落库以备追溯。

**已知取舍**:无工具的回答整块吐、不打字机——turn1 必须非流式才能判断要不要调工具,判完发现无工具时答案已完整;有工具的最终答仍逐 token 流式。

## 接口契约

### POST /api/chat(SSE 流式,前端主入口,带工具链)

请求(在 ch01 的 `{session_id, message}` 基础上改为会话身份 + 续接):

```json
{"user_id": "u1", "message": "订单 1001 的物流到哪了", "conversation_id": null}
```

- `conversation_id` 为空/缺省 → 新建会话,由 `done` 帧回传新建 id。
- 前端把 `user_id`(localStorage 里的稳定标识)与 `conversation_id` 存本地,续接带上。

SSE 事件帧(在 ch01 的 `delta`/`error`/`[DONE]` 基础上新增 `tool`、`done`):

| 帧 | 格式 | 含义 |
|---|---|---|
| `tool` | `data: {"event":"tool","name":"query_logistics"}` | 这轮调了某工具(前端渲染徽章);可多帧 |
| `delta` | `data: {"delta":"文本片段"}` | 最终回答的 token 片段(逐字打字机) |
| `done` | `data: {"event":"done","conversation_id":12}` | 结束,回传会话 id |
| `error` | `event: error\ndata: {"message":"…"}` | 出错(流已发 200,错误只能走帧) |
| `[DONE]` | `data: [DONE]` | 流终止哨兵(沿用 ch01) |

### POST /api/agent(非流式 JSON,程序化/测试出口)

契约:`user_id`/`message`/`conversation_id` → `answer`/`tool_calls`/`tool_results`,一次性返回完整工具轨迹 + 最终回答,供 eval、脚本、单测使用(`curl` 一眼看到模型选了哪个工具)。

## 错误处理分层

| 场景 | `/api/chat`(流式) | `/api/agent`(JSON) |
|---|---|---|
| 工具执行失败 | 基础设施捕获 → 回灌 → 模型正常作答,`tool` 帧照发 | 200,`tool_results[].ok=false` |
| `conversation_id` 不存在 | `error` 帧(流已发 200) | 404 |
| 请求体校验失败 | 422(未进流) | 422 |
| 上游 LLM 失败 | `error` 帧 | 502 |
| DB 失败 | `error` 帧 | 503 |

## 前端聊天页改造(Vibe Coding)

`app/static/index.html`:
- `streamChat` 请求体:`{session_id, message}` → `{user_id, message, conversation_id}`;`user_id` 用现有 localStorage 稳定标识,`conversation_id` 存 localStorage、`done` 帧回传后写回。
- SSE 解析:新增处理 `event:tool`(在 bot 气泡文本区上方渲染灰字徽章「🔧 调用了 <name>」)、`event:done`(存 `conversation_id`)。
- 工具执行、等首字期间文本区保留流动的三个点(不清空,免得看着像卡住);首字到达才替换成正文。
- markdown 渲染:内联零依赖渲染器(先转义防注入,支持标题/加粗/列表/表格/代码块/引用/分隔线/http 链接),逐帧渲染最终回答。
- 「+ 新对话」清空本地 `conversation_id`(开新会话);打字机、错误红字沿用。

## 测试与验证

- **可单测代码走 TDD**(pytest,不打真实模型):`repository`(会话/消息/faq/工单 CRUD)、工具基础设施(超时/重试/错误回灌/写类不重试)、5 个工具、编排两出口。
- **`run_agent_turn` 单测**:无工具直答、有工具执行收敛、落库顺序、单轮硬约束(收敛不 bind)、续接只回放最终答、非法会话抛异常。
- **`stream_agent_turn` 流式单测**(FakeModel 支持 `astream`):有工具分支事件序列 `tool`→`delta`→`done`(带 conversation_id)+ 落库顺序;无工具分支无 `tool` 帧;收敛不 bind;新建/非法会话。
- **`/api/chat` API 测试**(async httpx + DB fixture + FakeModel):SSE 帧格式、error 帧(会话不存在)、422 校验。
- **前端无单测**:Vibe Coding + 浏览器手动验收(三条验收标准直接在聊天页跑)。
- **标注样例评估**(`scripts/eval_agent.py`):真实 glm-5.2 跑「问法→期望工具」,如实记录选中对照与抖动。

## 验收标准逐条映射(浏览器)

| 验收 | 机制 | 判定 |
|---|---|---|
| 1. 订单 1001 物流 | 聊天页 SSE 收到 `tool:query_logistics` 帧 + 基于结果的流式回答 | 气泡见徽章 + 物流答复 |
| 2. 退货政策 | `query_faq` LIKE 命中 seed「退货政策」条目 | 气泡答出 7 天无理由 |
| 3. 邮费是多少 | `query_faq` 字面 LIKE 对不上「运费怎么算」→ 漏召回 | 如实记录,写 dev-notes 留 ch03 |

**faq seed 设计**:FAQ 用书面语,question 不含具体品类词——使验收 2「退货政策」能命中;验收 3「邮费是多少」的答案落在「运费怎么算」条目里,但字面 LIKE 对不上,暴露关键词检索的语义鸿沟(ch03 向量检索的正例)。

## 本章不做

- 多轮自动循环 Agent Loop(收敛不 bind_tools)
- 向量检索、RAG(验收 3 漏召回是 ch03 引子)
- Redis(用户撤销)
- 鉴权、限流、多副本、工单幂等
- 改造 `/api/extract`(保持 ch01 原样);删除 `SessionStore`/`CUSTOMER_SERVICE_PROMPT`(弃用保留)

## 关键决策记录

| 决策 | 选择 | 理由/备选 |
|---|---|---|
| 工具链落点 | 长在前端在用的 `/api/chat` | 能力要落在用户实际使用的聊天入口上;只做独立程序化接口,用户视角里等于没交付 |
| 编排结构 | 抽共享核心 `_prepare_turn` + `stream_agent_turn`(流式)/ `run_agent_turn`(非流式)两出口 | 两出口只在收敛那步不同,不重复实现 |
| 流式与工具 | turn1 `ainvoke` 拿完整 tool_calls,收敛用 `astream` 逐 token | 工具调用需完整参数,不能边流边判;最终答案可流式(Context7 已核对) |
| 工具轨迹呈现 | SSE `tool` 帧 → 气泡灰字徽章 | 用户选「显示小徽章」,课程演示肉眼可见工具被调 |
| 会话持久化 | `/api/chat` 落 DB(conversations/messages),不用内存 SessionStore | 与工具轨迹落库统一;跨轮上下文有源 |
| system prompt | 客服 chat 统一用 `AGENT_SYSTEM`(带工具原则) | 既懂工具又保客服人设 |
| `/api/agent` | 保留为非流式程序化/测试出口 | 供 eval/脚本/单测一眼看轨迹,不与前端流式重复 |
| 前端改造 | Vibe Coding | 用户工作要求 1:聊天页改造是 Superpowers 流程例外 |
| SessionStore/旧 prompt | 弃用但不删文件 | 删除是额外风险,留着无害(用户拍板) |
| 建表字符集 | DDL/seed 自带 `SET NAMES utf8mb4` | docker mysql client 默认 latin1 会 double-encode 中文 |
