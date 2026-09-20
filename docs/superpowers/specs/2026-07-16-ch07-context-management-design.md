# Ch07 设计:会话上下文管理(滑动窗口 + 异步滚动摘要)

## 1. 背景与目标

ch05/ch06 的 `main_agent` 把 checkpoint 里的**全量历史**原样递给模型(`_agent_messages` = system + `state["messages"]` 整条),轮数一多 token 必爆;ch02 遗留的 `trim_history`(`app/core/memory.py`)已经用上 `trim_messages` 但**没接进图**。本章把"第一版简单裁剪"升级成正经的三层上下文管理（近处原文、中间半压、远处摘要）。**只管当前会话,不跨会话、不存用户画像。**

**目标(需求 5 条):**
1. **三层结构**:最近几轮留原文(层 1 保真);稍远的半压(层 2,用户原话不动、客服回复和工具结果压缩);更早轮次异步压成摘要接在上下文前头(层 3)。摘要落 `conversations.summary`,`summary_upto_msg_id` 标出覆盖到哪条消息、滑窗从其后接原文(DDL 已就位:`sql/ch07-ddl.sql`)。
2. **摘要后台异步**:距上次摘要新增满 N 条消息(起步 30)就后台起一次摘要任务,滚动更新 summary,全程不阻塞当轮回复。摘要只提炼事实与诉求(问过哪款商品、订单号手机号、明确诉求、未决问题),不编造、去寒暄,几十到一两百字,严格 JSON `{"summary": "string"}`。
3. **拼装顺序固定**:system 人设红线 → 历史摘要(system 消息)→ 滑窗内最近几轮原文(起步 5-10 轮,一字不压)→ 用户当前这句。
4. **token 预算**:滑窗用 LangChain `trim_messages` 按 token 上限兜底;写入/裁剪/丢弃策略可控。
5. **双轨并行**:State.messages(add_messages reducer)+ checkpoint 存全量历史;调模型前另拼裁过带摘要的精简版,各走各的互不影响。对话历史照旧落 conversations/messages 两表。
6. **可观测与验收配套**:每次调模型把实际发出的上下文(摘要全文 + 滑窗逐条)打进 log/app.log,验收 `tail -f` 直接看;前端加会话侧栏支持多会话切换与历史回载(配两个只读接口),长对话验收不依赖 localStorage 单会话。

**非目标(本章不做):** 语义检索捞历史;主题重要度留关键事实;跨会话长期记忆;用户画像。

**用户澄清决策(brainstorm 定稿):**
- 摘要**三处注入**:main_agent、指代消解(coref)、意图识别都能看到摘要(跨滑窗指代如「最开始那个订单」才能被改写/判对)。
- 边界对齐走**方案 B**:State 带 DB id 锚点(见 §3),精确对齐、纯函数、不加库读。
- 最终验收**全浏览器手动**:用户在前端亲手聊 20+ 轮验收(不写自动灌话脚本进验收环节)。

---

## 2. 架构与数据流(双轨并行)

```
用户消息 → run_turn/stream_turn(runtime)
  ├─ repository.append_message → msg_id
  ├─ 读 conversations.summary / summary_upto_msg_id(每轮一次库读)
  ├─ _graph_input:HumanMessage(text, id=f"db-{msg_id}") + summary 字段进 State
  ↓
图(checkpoint 存全量 State.messages,一条不删 ←—— 事实源)
  ├─ resolve_reference / classify_intent:摘要行 + 滑窗内最近 6 条 ——精简版①
  ├─ main_agent:[Sys(人设红线), *滑窗原文, Human(摘要+本轮材料)] ——精简版②
  │    └─ 调模型前 model_ctx 日志:摘要全文 + 滑窗逐条 + tokens 估算(验收可观测)
  └─ log:落 assistant 消息 → MySQL
  ↓
轮结束(runtime 层):count(id > summary_upto_msg_id) ≥ N 且无在跑任务
  → asyncio.create_task(摘要任务)  ——不 await,当轮回复已返回
      读 MySQL 消息 → 保最近 K 轮,更早的+旧摘要 → LLM 滚动重写
      → UPDATE conversations SET summary, summary_upto_msg_id
```

State 全量与调模型精简版**各走各的**:摘要更新只影响下一轮拼装,State.messages 永不回删。

---

## 3. 边界对齐:State 带 DB id 锚点(方案 B)

难点:`summary_upto_msg_id` 是 MySQL 消息 id,滑窗要从 LangGraph `State.messages` 切,两套体系互不认识。

**做法**:入口落库用户消息拿到 `msg_id` 后,构造 `HumanMessage(text, id=f"db-{msg_id}")` 入图。切窗 = 在 `State.messages` 里找**第一条** `db-` id 数值 `> summary_upto_msg_id` 的用户消息,从它起切到尾,再 `trim_messages` token 上限兜底。

- `add_messages` 按消息 id 追踪、同 id 会更新覆盖(Context7 已核):MySQL 自增主键唯一,`db-{msg_id}` 不会撞。
- 历史轮次的 AIMessage(tool_calls)/ToolMessage 结构保留在窗内;`trim_messages(start_on="human")` 保证裁剪后从用户消息起步,不产生孤儿 tool 消息。
- **退化路径**:summary 为空 / 锚点找不到(如 ch07 上线前的旧消息无 `db-` id)→ 纯 token 裁剪,不崩。

## 4. 上下文拼装(`app/core/memory.py` 升级 + 三处消费点)

`memory.py` 从 ch02 遗留升级为本章核心模块(`SessionStore` 是 ch02 教学产物、仅测试引用,保留不动;`trim_history` 演进为 `build_window` 的兜底段):

```python
def build_window(messages, summary_upto_msg_id, max_tokens) -> list[BaseMessage]:
    """锚点切窗(第一条 db-id > 边界的 human 起)→ trim_messages 兜底;无锚点退化纯裁剪。"""

def summary_line(summary) -> str:      # coref/意图用:「(早前对话摘要:...)」单行,空摘要返回空串
def summary_system(summary) -> SystemMessage | None   # main_agent 用
```

**三处消费点**:
| 消费点 | 现状 | ch07 |
|---|---|---|
| `_agent_messages`(main_agent) | system + 全量历史 | `[Sys(人设+红线), *build_window(...), Human(摘要+本轮材料)]`,全列表只此一条 system;摘要与本轮材料合并成一条,插在窗内最后一条 human 之后,都没有则不插 |
| `_history_text`(coref/意图共用) | 全量历史取最近 6 条 | 摘要行 + **滑窗内**最近 6 条(先切窗再取尾) |
| State 新字段 | — | `summary: str`、`summary_upto_msg_id: int`(入口每轮加载,普通覆盖通道) |

拼装顺序固定:人设红线打头(恒定,不拼任何可变内容),摘要以 system 紧跟,再接滑窗原文,本轮材料紧跟窗内最后一条用户消息。

**上下文日志**(`nodes._log_model_context`):main_agent 每次调模型前,把实际发出的上下文打一条 `model_ctx` 日志——摘要全文(本就几十到两百字)、滑窗每条消息的角色+前 40 字、窗口条数、`count_tokens_approximately` 估算值。这是「滑窗和摘要真的按设计进了模型」的直接证据,验收与排障都靠它,不用猜。

## 5. 后台摘要任务(runtime 层,图外)

- **触发**:每轮结束后(含 interrupt 续跑结束),`count(messages.id > summary_upto_msg_id)`(边界空则数全量)≥ `summary_trigger_messages` 且该会话无在跑任务 → `asyncio.create_task`,不 await。
- **任务体**:
  1. 读该会话全部 user/assistant 消息(MySQL 只有这两类 + tool 不落库,源数据干净);
  2. 新边界 = 倒数第 K 轮用户消息的**前一条**消息 id(K=`context_window_turns`;不足 K 轮、或新边界未超过旧边界 → 直接放弃本次);
  3. 待压段 = `summary_upto_msg_id <` id `≤ 新边界` 的消息 + 旧摘要 → 摘要模型滚动重写;
  4. 严格 JSON `{"summary": "..."}` 解析成功才 `UPDATE conversations SET summary=?, summary_upto_msg_id=新边界`(原子,一起更)。
- **提示词红线**:只提炼事实与诉求(商品、订单号、手机号、明确诉求、未决问题);对话里没出现的一个字不许编;寒暄闲聊不留;几十到一两百字。
- **可观测**(验收标准 3):触发(conv、待压条数)、完成(覆盖到哪条 id、耗时 ms、摘要长度)、失败均 `logger.info/exception` 进 log/app.log。
- **失败处理**:异常/解析失败只 log 不重试——计数条件仍满足,下一轮自然重触发;summary 字段只在成功后更新,不会写坏。
- **并发防抖**:模块级 `dict[cid -> asyncio.Task]`,在跑不重复起,done 后移除。

## 6. 配置(settings,全部可调)

| 配置 | 起步值 | 含义 |
|---|---|---|
| `summary_trigger_messages` | 30 | 距上次摘要新增满多少条触发后台任务 |
| `context_window_turns` | 8 | 摘要后保留原文的最近轮数(需求 5-10 取中) |
| `context_window_max_tokens` | 3000 | 滑窗 token 上限(`trim_messages` 兜底) |
| `summary_model` | ""(空) | 摘要模型,空回落 `chat_model` |

窗口实际在 K 到 K+N/2 轮间浮动(触发一次压回 K 轮),token 上限兜住极端长消息。

## 7. 数据库与 repository

- `models.py`:`Conversation` 加 `summary: Text | None`、`summary_upto_msg_id: BigInteger | None`(对应 `sql/ch07-ddl.sql`,测试库建表自动带上)。
- `repository.py` 新增:
  - `get_conversation_summary(cid) -> (summary, upto_id)`(或复用 `get_conversation` 返回的 ORM 对象);
  - `count_messages_after(cid, after_id) -> int`;
  - `list_dialog_messages(cid) -> list[Message]`(user/assistant,按 id 序,供摘要任务与历史回载共用);
  - `update_conversation_summary(cid, summary, upto_msg_id) -> None`;
  - `list_conversations(user_id) -> list[dict]`(新在前,带首问预览 + 有无摘要标记,供会话侧栏)。

## 8. 验收入口:前端会话侧栏 + 只读接口

长对话验收要能开多个会话对照、随时切回旧会话续聊,现有前端 localStorage 只存单个 conversation_id、「新对话」即丢旧会话,撑不起来。配套做三件事:

- **两个只读接口**(`app/api/conversations.py`,不碰写路径):
  - `GET /api/conversations?user_id=` → 会话列表(新在前、首问预览、`has_summary` 标记);
  - `GET /api/conversations/{id}/messages` → user/assistant 历史(会话不存在 404)。
- **前端会话侧栏**(沿用像素风):列表 + active 高亮 + 「已摘要」徽标;点击切换会话并回载历史原文继续聊;每轮结束后自动刷新;「新对话」只是不带 conversation_id 开新会话,旧会话仍在侧栏可切回。窄屏(<780px)隐藏侧栏,不影响手机聊天。
- **边界**:侧栏/历史加载失败只静默降级,不影响聊天主链路。

## 9. 错误处理与边界情况

- 摘要 JSON 解析失败 / 模型异常 → log + 放弃本次,下轮重触发;不影响用户链路。
- interrupt 续跑:`Command(resume)` 不新增用户消息,State 沿用上轮加载的 summary 字段;续跑结束照做触发检查。
- 旧会话消息无 `db-` 锚点:切窗跳过非锚点消息,找不到锚点退化纯 token 裁剪。
- 摘要任务与下一轮并发读:任务只写 conversations 两字段,入口读到旧值最多晚一轮生效,无一致性风险。
- 服务重启:在跑任务丢失无妨,触发条件仍在,下轮重起。

## 10. 测试与验证

- **代码走 TDD**(测试库 `mewhelp_ch02_test` 惯例):
  - `build_window`:锚点命中/锚点缺失退化/token 兜底/摘要边界后无消息等分支;
  - 摘要边界计算(倒数第 K 轮前一条)、触发计数、防抖(含并发 gather 双触发只起一个、触发检查异常不外抛进回复路径);
  - repository 新函数(真库):摘要读写、计数、对话读取、会话列表;
  - `_agent_messages`/`_history_text` 拼装顺序(有/无摘要两态)、`_log_model_context` 日志内容(caplog);
  - 会话两接口走路由层打桩测试(本仓惯例:API 层 mock、repository 层真库——TestClient 的 anyio portal 与 pytest-asyncio 事件循环不共库连接)。
- **摘要 prompt 换标注样例验证**(不可单测,按工作要求以 eval 代 TDD):`scripts/eval_ch07.py` 构造带订单号/手机号/寒暄的对话样例,跑真模型断言:严格 JSON、关键事实齐(订单号/诉求在)、无编造实体、长度在几十到两百字、寒暄不留。
- **最终验收(用户,全浏览器手动)**:前端连聊 20+ 轮(侧栏可开多会话对照、切回旧会话续聊)→ 问「最开始那个订单后来怎么说」→ 答复含早期订单号与诉求;同时 `grep model_ctx log/app.log` 看每轮实际发给模型的摘要+滑窗,`grep summary` 看后台任务触发/完成痕迹与耗时,确认当轮回复未被阻塞、token 不爆系统不崩。

## 11. 权威来源与 API 核实(Context7)

- `trim_messages(strategy="last", token_counter=count_tokens_approximately, max_tokens, start_on="human")`:与现 `memory.py` 用法一致,官方推荐在调模型前于节点内裁剪(不改 State)。
- `add_messages` 按消息 id 追踪,同 id 更新覆盖 → 自定义 `db-{msg_id}` 必须唯一(自增主键,成立)。
- 实现期涉及 SQLAlchemy 列新增、LangChain JSON 输出解析等,动手前逐一再查。
