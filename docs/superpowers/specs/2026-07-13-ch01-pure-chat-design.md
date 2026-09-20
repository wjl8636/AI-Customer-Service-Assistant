# MewHelp Ch01 设计:纯对话客服(SSE 流式 + 结构化提取)

## 目标

电商智能客服系统第一章:跑通纯对话。不做工具调用、不做 Agent 循环。

验收标准:
1. curl 调对话接口能看到流式回复(逐 token)。
2. 连续问两轮,第二轮能接住第一轮的上下文。
3. 发一段售后描述,能拿到结构化 JSON(订单号/诉求类型/期望方案)。

## 技术栈(定死,不可自行更换)

- Python 3.12 + FastAPI + LangChain
- 模型接入直连上游,应用侧统一说 OpenAI 协议;地址、模型名、密钥全在 `.env`,分 `CHAT_*` / `EMBED_*` / `RERANK_*` 三组
- 依赖管理:uv
- 验收模型:`claude-opus-4-8`,Anthropic 兼容端点 `http://8.219.192.36:9000`(token 放 `.env`,不进 git)

## 架构:单进程

一条命令(`make dev` 或 `scripts/dev.sh`)拉起 FastAPI 应用(:8000)。LangChain 的
`ChatOpenAI` 直接指向 `settings.chat_base_url`,换上游改 `.env` 里两行,不动代码。

换模型时注意 `CHAT_MODEL` 要写上游认的真实模型名。这一步没有别名可用,写错了不会在
启动时报错,而是第一次调模型时回一句 `Model does not exist`。

## 目录结构

```
app/
  main.py              # FastAPI 入口
  config.py            # pydantic-settings 读 .env
  api/chat.py          # POST /api/chat (SSE)
  api/extract.py       # POST /api/extract
  core/llm.py          # ChatOpenAI 工厂
  core/prompts.py      # 所有 PromptTemplate 集中在此
  core/memory.py       # 内存会话存储 + token 预算裁剪
  schemas/             # pydantic 请求/响应/结构化提取模型
scripts/dev.sh
tests/
dev-notes/ch01.md
.env / .env.example
```

## 接口契约

### POST /api/chat(SSE 流式对话)

请求:

```json
{"session_id": "abc", "message": "你们家猫粮发什么快递?"}
```

响应:`text/event-stream`,逐 token 推送:

```
data: {"delta": "我"}
data: {"delta": "们"}
...
data: [DONE]
```

- 历史按 `session_id` 存服务端内存 dict(本章不做持久化)。
- 每次调用前用 LangChain `trim_messages` 按 token 预算裁剪(默认 2000 token,可配),策略:保 system prompt + 从最近往前保留完整轮次。
- 上游 LLM 出错:SSE 推 `event: error` + 错误信息后关流。

### POST /api/extract(结构化提取)

请求:

```json
{"text": "我上周买的猫爬架,订单号 MH20260701123,到货就散架了,我要退款"}
```

响应(`with_structured_output` 绑定 pydantic 模型 `AfterSalesTicket`):

```json
{"order_id": "MH20260701123", "request_type": "退款", "expected_solution": "商品到货损坏,要求退款"}
```

- `request_type` 枚举:退款 / 换货 / 维修 / 投诉 / 其他。
- 订单号缺失返回 `null`,不臆造。
- 上游失败返回 502 JSON。

## Prompt 管理

`core/prompts.py` 集中管理两个 `ChatPromptTemplate`:

1. **客服对话 prompt**:system 写清角色设定(电商客服)与行为约束——不臆造订单/物流信息、超范围问题引导回客服话题、语气规范。
2. **提取任务 prompt**:面向售后描述的字段提取指令。

## 测试与验证

按"可单测代码走 TDD、纯 Prompt 产出走标注样例验证"拆两类:

**TDD(pytest,先红后绿):**
- `core/memory.py`:会话存储、token 预算裁剪逻辑
- SSE 组帧(用 FakeLLM / mock,不打真实模型)
- schema 校验(AfterSalesTicket 枚举、可空字段)

**标注样例验证(真实模型):**
- 5 条标注好的售后描述样例跑 `/api/extract` 核对字段,覆盖 edge case:订单号缺失、诉求模糊。
- 脚本连发两轮对话,验证第二轮接住第一轮上下文。

## 本章不做

- 工具调用、Agent 循环
- 会话持久化(重启丢历史,接受)
- 鉴权、限流、多副本
- 对话中自动触发提取(提取是独立接口)

## 关键决策记录

| 决策 | 选择 | 备选与理由 |
|---|---|---|
| 上游接入形态 | 应用直连,配置分三组 | 中间垫网关只为归一 rerank 格式一件实事,却要多一个进程、多一份配置、换模型改两处 |
| 会话管理 | 服务端 session_id | 客户端全量 history 对 curl 验收不友好 |
| 提取形态 | 独立 /extract 接口 | 融合进对话需意图判断,超出本章"纯对话"范围 |
| 依赖管理 | uv | 用户确认 |
