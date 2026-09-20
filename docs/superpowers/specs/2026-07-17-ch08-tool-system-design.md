# Ch08 设计:即插即用工具系统(注册中心 + 执行引擎 + MCP 接入 + 建工单确认流)

## 1. 背景与目标

ch02-ch07 的工具层是"几个写死的内置工具":`registry.py` 是静态列表 + 三个特例集合(NO_RETRY / INJECT_CONVERSATION / TOOL_TIMEOUTS),`infra.py` 只管超时重试,没有参数校验、没有权限分层、没有审计;create_ticket 在 `agent_tools` 里被硬编码拦成前端按钮。本章升级成**即插即用的工具系统**:统一注册、统一校验、统一权限、统一执行、统一审计,并以 MCP 打通外部工具生态。

**目标(需求 7 条):**

1. **工具注册中心**:内置和 MCP 工具一律带齐三样——工具名、用途描述、JSON Schema 参数定义——登记进同一份注册表;内置工具服务启动时登记,MCP 工具连上 Server 动态发现、现问现拿;新工具注册进来就能被主力 Agent 用上,不改核心代码(内置:丢文件即注册;MCP:不重启客服系统)。
2. **参数校验**:执行前统一按 JSON Schema 校验,类型不对、必填缺失、取值越界的拦下;校验错误包成工具结果回灌模型(不抛异常),让它追问用户或重新组织调用。
3. **权限控制**:工具分只读/写两类,写操作由执行引擎把门;本系统写操作只有 create_ticket——只有客户明确要求建工单才发起,真正执行要等前端确认(interrupt→resume)回来,没确认的调用引擎直接拒绝;外部 MCP 工具的用途声明不可信,权限只认我们侧规则,不看 Server 声明、不让模型临场判断。
4. **执行引擎**:所有工具调用走同一处,统一超时、重试、错误处理、结果格式化;重试只给暂时性故障(超时/网络),业务空结果不重试,写操作默认不自动重试(超时未必没执行,重复执行比失败更糟);错误分三类分诊(参数不合法 / 查询落空 / 真故障),坏消息如实回灌;结果挑回答用得上的字段、内部枚举码翻人话、JSON 中文不转义。
5. **审计留痕**:`tool_audit_logs` 表每次调用落一条(含被权限拒、被校验拦的);不挂外键;审计写失败不许反拦工具执行。
6. **MCP 接入**:自建物流(查轨迹)、售后(查在保/查退货进度)两台业务 MCP Server,独立进程、Streamable HTTP、mock 数据不接真实系统不建表;客服系统作为 MCP Client 接入,工具与内置清单合成一份,一视同仁;内置 query_logistics 下线,物流查询由 MCP 接管,不留重名。
7. **建工单确认流(带前端)**:客户明确要求才走;Agent 先核对信息、缺必填主动追问不许编;凑齐后发起 create_ticket,执行引擎不直接执行,照 ch06 订单选择器用 LangGraph interrupt 推工单预览(类型+描述)给前端;前端预览卡带「确认提交」「取消」;确认→resume 放行落 tickets 表、回复带工单号;取消→不执行、审计落「权限拒绝」;ch05 投诉流程前端按钮建单路径保持不动(点按钮即用户确认)。

**非目标(本章不做):** Skill 机制(只是生态概念);仓储等更多外部系统;MCP Server 侧鉴权;注册表落库/跨进程注册中心(方案 B 已否);工具清单缓存与失效策略(现问现拿,真实系统再谈)。

**技术栈(定死):** MCP Server 用官方 Python SDK(Streamable HTTP);MCP Client 用 langchain-mcp-adapters 的 `MultiServerMCPClient`。

**用户澄清决策(brainstorm 定稿):**
- 架构选**方案 A**:进程内动态注册表 + 四段式执行管道,registry/infra 原位升级;否掉工具表落 MySQL(与现问现拿矛盾)和 LangGraph ToolNode 定制(黑盒不利教学)。
- 验收 6 超时注入走**环境变量**:物流 MCP Server 读 `MOCK_DELAY_SECONDS` 延迟返回;主服务 `DEMO_TICKET_DELAY_SECONDS` 让 create_ticket 变慢——不埋魔法参数后门。

**实现期用户拍板(Task 6 路由矛盾,实测分类器证据后决策):**
- 现行路由把「售后」送 refund_flow、「其他」送兜底话术,实测「订单1001还在保吗」→售后(0.95)、「帮我建个工单」→其他(0.55)、「猫砂盆漏电了,帮我建个工单跟进」→售后(0.92)——验收 2 的在保、验收 4 的建单话术都到不了主力 Agent。
- 用户裁决:**验收 2 改口径**——用物流轨迹演示 MCP 即可,不做「问在保」对话验收(售后→refund_flow 路由不动;在保/退货进度工具仍在清单里,由注册中心/集成测试证明可用);**验收 4 从意图识别解**——意图新增第九类「人工」(明确要求建工单/转人工跟进)→ 路由 business,主力 Agent 走 create_ticket 确认流。意图 prompt/`INTENTS`/路由表/评估集四处同步,eval_intent 18/18(原 16 例零回归 + 人工 2 例判对)。

---

## 2. 架构与数据流

```
                     ┌ app/tools/builtin/  启动时 pkgutil 扫描,注册即定义
                     │   orders.py / faq.py / tickets.py / refunds.py / (promotions.py=验收1)
 registry.py ────────┤
 (ToolSpec 注册中心)  └ mcp_client.py  每次现拉:MultiServerMCPClient.get_tools(server_name=...)
      │                    ├ logistics_server  :8101 (mcp_servers/,独立进程)
      │                    └ aftersales_server :8102 (mcp_servers/,独立进程)
      ↓ get_all_specs()  = builtin + mcp 合并,重名 builtin 优先
 main_agent  ── bind_tools([spec.tool]) ── 模型发起 tool_calls
      ↓
 agent_tools ── 含 create_ticket?→ 先纯校验 → interrupt(confirm_ticket 预览) ⇄ 前端卡片
      ↓                                        resume {"confirmed": bool}
 engine.py 统一管道:查工具 → jsonschema 校验 → 权限门 → 执行(超时/重试) → 分诊 → 格式化
      │                                                                    │
      └────────────── 每次调用(含校验拦下/权限拒绝)──→ tool_audit_logs(审计,失败不反拦)
```

## 3. 工具注册中心(`app/tools/registry.py` 重造)

```python
@dataclass
class ToolSpec:
    name: str                 # 工具名(唯一键)
    description: str          # 用途描述
    json_schema: dict         # 参数 JSON Schema(内置:args_schema.model_json_schema();MCP:adapter 转回的 input schema)
    tool: BaseTool            # 底层可调用对象
    permission: str           # "read" | "write" —— 只认我们侧规则
    source: str               # "builtin" | "mcp"
    mcp_server: str | None    # 来源 Server 名,内置为 None
    timeout: float | None     # 超时覆盖(None 用默认:内置 5s / MCP 10s)
    inject_conversation: bool # 是否注入 conversation_id
    format_result: Callable | None  # 结果格式化钩子(挑字段/枚举翻人话),None 透传
```

- **内置注册即定义**:`app/tools/builtin/` 包,每模块 = @tool 实现 + `registry.register(ToolSpec(...))`;服务启动(lifespan)时 registry pkgutil 扫描导入整包完成登记(对齐需求"内置工具服务启动时登记")。现有工具从 `business.py` 迁入:`orders.py`(query_order/query_product)、`faq.py`(query_faq,30s 超时)、`tickets.py`(create_ticket,write)、`refunds.py`(submit_refund);**query_logistics 下线不迁**。`business.py` 瘦身成 mock 数据纯函数(order_snapshot / list_user_orders,`graph.fetch_order` 继续用)。
- **验收 1 路径**:新工具 = 往 builtin/ 丢一个文件(实现+register),重启服务即被扫描注册,核心代码零改动。演示用 `promotions.py`(查优惠活动 mock)。
- **MCP 现问现拿**:`app/tools/mcp_client.py` 持一个 `MultiServerMCPClient`(两台 Server 的 URL 进 settings);`fetch_mcp_specs()` 逐 Server `get_tools(server_name=...)` 包 try/except——单台连不上告警+跳过,不拖垮对话。adapters 文档已核:get_tools 与工具调用均每次新建 session,Server 侧加工具即时可见(验收 3 的机制保证)。
- **合并与重名**:`get_all_specs()`(async)= 内置 + MCP;重名后到者丢弃并 warning(本章靠下线内置 query_logistics 避免撞名,MCP 物流工具沿用 `query_logistics` 名,query_order 描述里"先查订单拿 tracking_no 再查物流"的引导语义不变)。
- **权限我们侧独裁**:`WRITE_TOOLS = {"create_ticket"}` 按名字定权限;MCP 工具不看 Server 声明,不在写清单一律按只读放行(代码注释点明:生产环境接不受信 Server 应默认拒绝未知写操作,本章两台 Server 都是查询类)。
- **拉取代价**:main_agent(bind)与 agent_tools(执行)各自现拉,一步 ReAct ≈ 2 次本地 HTTP 清单拉取,毫秒级;换来零缓存失效逻辑,符合"现问现拿"。

## 4. 统一执行引擎(`app/tools/engine.py`,替代 infra.py)

管道:**查工具 → jsonschema 校验 → 权限门 → 执行(超时/重试)→ 分诊 → 格式化 + 审计**。对外仍是 `execute_tool_call(tool_call, conversation_id, *, confirmed=False) -> ToolRun`,`agent_tools` 是唯一调用方。

| 段 | 规则 | 审计 status |
|---|---|---|
| 查工具 | 名字查 spec,查不到→"未知工具"回灌 | 失败 |
| 校验 | `jsonschema.validate(args, spec.json_schema)`;错误包成 ToolMessage 回灌:"参数校验未通过:<具体错误>,请修正参数或向用户追问",不抛异常 | 校验拦下 |
| 权限门 | `permission=="write"` 且 `confirmed!=True` → 直接拒绝(模型绕不过);确认令牌只能由 agent_tools 在 interrupt 确认后传入 | 权限拒绝 |
| 执行 | `asyncio.wait_for`;超时默认内置 5s / MCP 10s,spec.timeout 覆盖(query_faq 30s);**重试只给 TimeoutError + 网络类异常(ConnectionError/httpx 传输错)**,指数退避 0.2s*attempt,默认 2 次;write 恒 0 次;业务空结果是正常返回,天然不重试 | 成功/失败/超时 |
| 分诊 | 参数不合法(校验层拦)/ 查询落空(工具返回语义表达,如 query_faq sufficient=False,原样回灌)/ 真故障(异常→"工具暂时不可用:<类名>"如实回灌,不装没事) | — |
| 格式化 | `json.dumps(..., ensure_ascii=False)`;`spec.format_result` 挑回答用得上的字段、内部枚举码翻人话(MCP 物流返回故意带 `status_code: "TRANSIT"` 类内部码,client 侧翻译——"不信 Server、我们侧治理"的教学点) | — |

**审计**:每次调用(含被拦的)构造记录 → `repository.insert_tool_audit(conversation_id, tool_call_id, tool_name, tool_source, mcp_server, arguments, result_summary(截500字), status, error_message, retry_count(实际值), duration_ms)`;**try/except 包死,写失败只 log 不拦执行**;表无外键(DDL 已定稿 `sql/ch08-ddl.sql`)。

## 5. 建工单确认流(graph 改造)

- **`agent_tools` 重写**(照 fetch_order 的 interrupt 范式:interrupt 前只做纯计算,resume 重跑安全):
  1. 收集 tool_calls;含 create_ticket → 先**纯校验**参数(不落审计不执行):缺 description 等 → 走引擎校验拦截路径回灌(审计「校验拦下」),模型自然向用户追问,**不 interrupt**;
  2. 校验通过 → **节点顶部、执行任何工具之前** `interrupt({"type": "confirm_ticket", "preview": {"ticket_type", "description"}})`;一轮多个 create_ticket 只理第一个,其余回灌"一次只处理一个建工单请求";
  3. resume 值 `{"confirmed": bool}`:true → 引擎带 `confirmed=True` 真执行(落 tickets 表),ToolMessage 带工单号,模型答复带给用户;false → 引擎按「权限拒绝」落审计,回灌"用户取消了本次建工单,请不要再发起";
  4. 其余工具在 interrupt 之后照常走引擎。
- **删除**旧的 create_ticket→suggested_actions 拦截;**保留** submit_refund→refund_form 拦截原样;**不动** ch05 `complaint_reply` 按钮建单路径(`POST /api/actions/create-ticket`,点按钮即确认)。
- **prompts**:AGENT_SYSTEM 补 create_ticket 规矩——仅用户明确要求才发起;必填缺失先追问不许编;create_ticket 工具描述同步改。
- **runtime.py**:`_stream_events` 的 interrupt 帧泛化为透传 payload(帧含 `kind`,前端按 kind 分发;select_order 保持兼容);`/api/actions/resume` 的 payload 扩成 `{conversation_id, order_id?, confirmed?}`,按非空字段构造 `Command(resume=...)`。

## 6. 两台业务 MCP Server(顶层新目录 `mcp_servers/`)

| Server | 端口 | 工具 | 说明 |
|---|---|---|---|
| logistics_server.py | 8101 | `query_logistics(tracking_no)` | mock 轨迹(随机种子稳定),返回含内部枚举码 `status_code`;读 `MOCK_DELAY_SECONDS` 注入延迟(验收 6) |
| aftersales_server.py | 8102 | `query_warranty(order_id)`、`query_return_status(order_id)` | 查在保/查退货进度,mock 同上 |

- 官方 SDK,`run(transport="streamable-http")`,独立进程,不接 DB 不建表;SDK 新老 API(FastMCP → MCPServer 迁移中)以实装版本为准,动手前用 Context7 按版本核对。
- Makefile:`mcp-up` / `mcp-down`(nohup + pid 文件落 log/),`make dev`(dev.sh)一并拉起。
- 验收 3 演示:现场往 logistics_server.py 加 `query_delivery_eta` 工具,只重启 8101,客服系统代码不动服务不重启,下一轮对话即可用。

## 7. 配置 / 依赖 / DB

- settings 新增:`mcp_logistics_url`(默认 `http://127.0.0.1:8101/mcp`)、`mcp_aftersales_url`(8102)、`tool_default_timeout=5.0`、`mcp_tool_timeout=10.0`、`tool_max_retries=2`、`demo_ticket_delay_seconds=0`(验收 6 写超时演示);.env.example 同步。
- 依赖:`uv add mcp langchain-mcp-adapters jsonschema`。
- DB:models.py 加 `ToolAuditLog`(对齐 `sql/ch08-ddl.sql`,无 FK);repository 加 `insert_tool_audit()`;dev 库照惯例 docker exec 应用 DDL;tests/conftest `_DDL_FILES` += ch08-ddl.sql、`_TABLES` += tool_audit_logs。

## 8. 前端(Vibe Coding,不走 brainstorm/TDD/review)

- interrupt 帧按 `kind` 分发:`select_order` → 现有订单卡不动;`confirm_ticket` → 新预览卡(工单类型 + 问题描述 + 「确认提交」「取消」两按钮),样式对齐现有 order-cards;
- 确认 → `POST /api/actions/resume {conversation_id, confirmed: true}` 续流,回复带工单号;取消 → `confirmed: false`,气泡显示已取消;
- 刷新丢预览卡与 ch06 订单选择器同待遇(interrupt 瞬态,不做历史回载)。

## 9. 测试与验证

- **TDD 覆盖**:engine 管道(校验拦下/权限拒绝/超时重试/写不重试/分诊/审计字段,monkeypatch repository)、registry(扫描注册/重名丢弃/MCP 降级)、审计 repository(真测试库)、agent_tools 确认流(graph 级 interrupt→resume 两分支断言,参考 tests/graph/test_ch06_nodes)。
- **MCP 集成测试**:pytest fixture 子进程拉起两台 Server → client 拉清单断言三要素齐 → 真调一次工具拿 mock 数据。
- **Prompt 改动**(建单追问规矩)不可单测:按工作要求用标注对话样例跑验证(明确要求但缺描述 → 应追问不应编)。
- **前端**:Vibe Coding 手测。
- **验收对照**:6 条验收各有落点——①promotions.py 丢文件;②问物流/在保走 MCP;③Server 侧加工具只重启该 Server;④缺信息追问→预览卡→确认→tickets 落表回工单号;⑤取消→审计「权限拒绝」;⑥`MOCK_DELAY_SECONDS` 看读工具重试+审计三字段,`DEMO_TICKET_DELAY_SECONDS` 看写超时 retry_count=0。
