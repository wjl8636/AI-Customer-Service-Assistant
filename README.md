<div align="center">

# 🛒 电商智能客服 Agent

一个从零到一完整落地的 **AI 电商客服 Agent**：能查订单物流、答政策 FAQ、走退款子流程、从对话里挖知识补库，还能微调一个主题分类器做旁路归类的整套生产级客服系统。

《AI Agent 智能客服实战》配套源码，代码随课程从 ch01 纯对话起步一路长到 ch10 收尾，**不分支、不演示碎片**——就是一个能跑起来的完整系统。

[快速开始](#-快速开始) · [架构](#-总体架构) · [功能演进](#-十章功能演进) · [目录结构](#-目录结构) · [项目部署](DEPLOY.md)

<sub>Python 3.12 · FastAPI · LangGraph · Milvus · MySQL</sub>

</div>

---

## ✨ 特性一览

- **多上游直连**：聊天 / 嵌入 / 重排三组上游各自直连，没有网关那一层。换供应商、换模型只改 `.env`，不动代码。
- **知识库（RAG）**：文档切块 → MySQL / Milvus 双写 → 混合检索（Dense + BM25 + RRF）→ 重排 → 忠实度滤地下的证据引用。
- **可视化工作流（LangGraph）**：分流器、意图识别、指代消解、ReAct 环、退款子流程的 interrupt/resume，全程可观测。
- **上下文管理**：滑窗 + 异步摘要 + 前缀缓存 + 上下文预算，窗口装不下自动降级而不是崩掉。
- **工具系统**：内置 `@tool`、MCP Server 动态发现、统一执行引擎、审计日志。
- **数据飞轮（ch09）**：从低置信度/转人工对话里挖知识回填知识库，配合 Langfuse 自部署观测与成本账闭环。
- **主题分类器（ch10）**：语料流水线 → RoBERTa-wwm-ext 微调 → 阈值扫描 → 导出 ONNX 轻量推理服务，17 类目旁路归类。
- **可运行即可验收**：每一章都有对应的评估脚本与前端验收页，`make test` 跑全部单测，不打真实模型。

---

## 🧱 总体架构

```
                 ┌────────────────────────────────────────────────┐
  用户浏览器      │                FastAPI 应用 (:8000)            │
  /  /kb /review │                                                │
  /observability │   API 层 (app/api)     静态页面 (app/static)    │
  /topics ... ───┼───────┬───────────────────────────────────────┘
                 │       │
                 │  LangGraph 工作流 (app/graph) — 分流/意图/ReAct/子流程
                 │       │
                 │  core 单点能力 (app/core)                        MCP Servers
                 │  检索·重排·摘要·置信度·飞轮·预算                  (独立进程)
                 │       │                                    ┌──────────────┐
                 │  tool 系统 (app/tools) MCP 客户端 ────────▶│ 物流 :8101    │
                 │  内置@tool/注册表/执行引擎                  │ 售后 :8102    │
                 │                                            └──────────────┘
                 └───────┬───────────────┬───────────────┬───────────────
                 MySQL :3306 │    Milvus :19530 │   Langfuse :3000 (自部署)
                 业务数据/知识切块 │   向量/完整检索    │   观测·成本账 (可选)
                             数据库  │   (etc+minio)   │
```

三组上游（聊天 `CHAT_*` / 嵌入 `EMBED_*` / 重排 `RERANK_*`）在 `.env` 中配置，各自独立直连。

### 核心技术栈

| 领域 | 选型 |
| --- | --- |
| Web 框架 | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| Agent 编排 | [LangGraph](https://www.langchain.com/langgraph) + LangChain |
| 向量 / 混合检索 | [Milvus](https://milvus.io/) Standalone（原生 BM25）+ Dense + RRF + bge-reranker |
| 数据库 | SQLAlchemy (async) + MySQL 8 / SQLite (测试) |
| 嵌入 / 重排 | bge-m3 / bge-reranker-v2-m3（硅基流动） |
| 工具协议 | Model Context Protocol (MCP) + `langchain-mcp-adapters` |
| 可观测 | Langfuse 自部署（可选） |
| 主题分类 | RoBERTa-wwm-ext 微调 → ONNX Runtime 推理 |
| 工程化 | uv · pytest · pydantic-settings · Makefile |

---

## 🚀 快速开始

> 完整的分步部署任务书（含镜像加速、端口冲突、故障排查）见 [`DEPLOY.md`](DEPLOY.md)，交给 AI 助手或自读皆可。下面是能跑起来的最短路径。

### 0. 前置条件

| 项 | 说明 |
| --- | --- |
| [uv](https://docs.astral.sh/uv/) | Python 环境管理，自动下载 Python 3.12 |
| [Docker](https://www.docker.com/) | 跑 MySQL、Milvus、etcd、MinIO |
| make | 项目大量用 `make` 目标 |
| 模型密钥 | 聊天 + 嵌入 + 重排 三组（见第 2 步） |

内存 ≥ 8G，磁盘 ≥ 10G。

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置 `.env`

```bash
cp .env.example .env
```

编辑 `.env`，至少填这三项（密钥**必须是你自己的**，不要用占位符）：

```bash
# 聊天（示例用 DeepSeek，可按需换任意 OpenAI 兼容上游）
CHAT_BASE_URL=https://api.deepseek.com/v1
CHAT_MODEL=deepseek-v4-flash
CHAT_API_KEY=<你的聊天密钥>

# 嵌入 & 重排：bge-m3 / bge-reranker-v2-m3，走硅基流动（这些在那边免费），两个 KEY 填同一个值
EMBED_API_KEY=<你的硅基流动密钥>
RERANK_API_KEY=<同一个硅基流动密钥>
```

> 嵌入 / 重排的 `BASE_URL` 有代码默认值，通常不用填。聊天上游可选任意 OpenAI 兼容端点。

### 3. 起依赖容器 + 灌数据 + 建知识库

```bash
docker compose up -d              # MySQL + Milvus(Standalone) + etcd + MinIO
make seed                         # FAQ 业务数据（容器首启已自动跑，重复执行幂等）
make kb-build && make kb-vectorize   # 知识切块落 MySQL → 向量化写 Milvus
```

### 4. 启动

```bash
make dev    # 拉依赖容器 → 等 Milvus 就绪 → 拉起两台 MCP(8101/8102) → 应用 :8000
```

> `make dev` 是前台常驻进程，放后台跑用 `make dev-down` 停。

浏览器打开 **<http://localhost:8000>**，问问「订单 1001 的物流到哪了」，看到回答上方的两个工具调用标记（`query_order` + `query_logistics`）即部署成功。

---

## ⚙️ 配置说明（.env）

`.env.example` 内每一项都有详细注释，这里说明最关键的三组：

| 组 | 环境变量 | 作用 |
| --- | --- | --- |
| **聊天** | `CHAT_MODEL` / `CHAT_BASE_URL` / `CHAT_API_KEY` | 主对话模型。模型名必须是上游真实名（无网关别名） |
| **嵌入** | `EMBED_API_KEY` / `EMBED_BASE_URL` | bge-m3，默认硅基流动地址 |
| **重排** | `RERANK_API_KEY` / `RERANK_BASE_URL` | bge-reranker-v2-m3 |
| **可选槽位** | `INTENT_*` / `SUMMARY_*` | 意图识别、摘要默认继承聊天上游；要换更便宜的模型才填 |
| **可观测** | `LANGFUSE_*` | 三者齐全才挂回调，缺省时系统照常跑 |

> 思考链（`CHAT_THINKING` / `CHAT_REASONING_EFFORT` / `CHAT_REASONING_SPLIT`）按上游能力配置，详见 `.env.example`。

---

## 📱 页面与端口

### 前端页面

| 路径 | 功能 |
| --- | --- |
| `/` | 聊天页 |
| `/admin` | 后台管理首页（各模块入口汇总） |
| `/kb` | 知识库录入（贴文档切块入库、向量化、当场检索自测） |
| `/review` | ch09 飞轮待审队列 |
| `/observability` | 意图成本账、评估趋势、置信度阈值校准 |
| `/topics` | ch10 主题分布（17 类目问题量）；`/topics/questions` 单类问题列表 |
| `/acceptance` | ch10 验收总览（九项实证闸门）+ `/acceptance/eval` `/data` `/errors` |

### 端口

| 端口 | 用途 |
| --- | --- |
| 8000 | FastAPI 应用 |
| 8101 / 8102 | 业务 MCP Server（物流 / 售后） |
| 8110 | ch10 主题分类器推理服务（`make classifier-up` 后） |
| 3000 | Langfuse（`make langfuse-up` 后） |
| 19530 | Milvus |

---

## 📚 十章功能演进

代码随课程逐章生长，`make test` 跑全部单测（不打真实模型）；下列验收命令需真实服务在跑。`make help` 可查看带说明的完整目标清单。

| 章 | 长出了什么 | 验收命令 |
| --- | --- | --- |
| ch01 | 流式对话、结构化提取 | `make eval` |
| ch02 | 5 个 `@tool` 业务工具，单轮 Function Calling | `make eval-agent` |
| ch03 | 切块、嵌入、MySQL 与 Milvus 双写、对话挖知识 | `make kb-build` `make kb-vectorize` `make eval-retrieval` |
| ch04 | 混合检索、RRF、重排、Query 改写、四策略评估 | `make smoke-rag` `make eval-rag` |
| ch05 | LangGraph workflow 骨架 + 主力 Agent 的 ReAct 环 | `make eval-ch05` |
| ch06 | 分流器、指代消解、退款子流程 interrupt/resume | `make smoke-interrupt` `make eval-ch06` |
| ch07 | 上下文管理：滑窗、摘要、前缀缓存 | `make eval-ch07` |
| ch08 | 工具系统：MCP 动态发现、统一执行引擎、审计日志 | `make eval-ch08` |
| ch09 | Langfuse 自部署、数据飞轮、成本账 | `make langfuse-up` `make flywheel` `make eval-flywheel` `make cost-report` |
| ch10 | 主题分类器：语料、微调、阈值扫描、ONNX 推理 | `make ch10-corpus` `make ch10-train` `make ch10-eval` |

### ch10 主题分类器（进阶）

```bash
make ch10-corpus    # 语料流水线：捞池→清洗→预标→补足→人工抽审
make ch10-dataset   # 80/10/10 分层划分 + 训练集增强
make ch10-train     # RoBERTa-wwm-ext 全参微调（MPS/CUDA/CPU 自适应，重依赖走 uv ml 组）
make ch10-eval      # 每类 P/R/F1 + 混淆矩阵 + 容错红线
make ch10-export    # 导出 ONNX 并校验与 torch 预测一致
make classifier-up  # ONNX 推理服务 :8110
make classify-pool  # 旁路批量归类，写 topic_classifications
```

---

## 🗂 目录结构

```
app/
├── api/            # HTTP 入口：聊天、Agent、知识库录入、复核、验收页、成本看板
├── graph/          # LangGraph 图：state 状态、nodes 节点、routing 分流、build 组装
├── core/           # 单点能力：上游客户端、检索、重排、意图、指代、摘要、置信度、飞轮、可观测
├── kb/             # 知识入库：切块、嵌入、双写 MySQL 与 Milvus、去重、从对话挖问答对
├── tools/          # 工具系统：内置 @tool、MCP 客户端、注册表、统一执行引擎
├── db/             # 表模型与仓储
├── schemas/        # Pydantic 请求/响应模型
└── static/         # 前端页面（聊天、录入、飞轮待审、观测、主题、验收）
mcp_servers/        # 两台业务 MCP Server（物流/售后，独立进程）
sql/                # 各章建表与迁移，容器首启按文件名顺序自动执行
scripts/            # 建库、评估、微调等离线活 + ch10/ 分类器全流程
docs/superpowers/   # 各章的 spec 与 plan（工程方法论范本）
primer/             # 前置知识配套例子（独立于主项目），与项目共用 .env
data/               # 知识库文档、评估报告产物（运行时生成）
tests/              # pytest 单测（api / core / graph / kb / tools / db）
```

- 内置业务工具：`query_order` / `query_product`（查订单）、`query_faq`（政策问答，RAG 混合检索）、`create_refund`（退款子流程）、`create_ticket`（人工工单）
- 观众主题 17 类目（退货物流 / 尺码发票 / 质量运费 / 优惠价保 / 支付订单 / 库存商品 / 保修账号 / 会员评价 / 其他 等），见 `app/core/taxonomy.py`
- 订单/物流数据为 `app/tools/business.py` 按 `user_id` 稳定生成的模拟数据，任何账号名下有 1001 / 2002 两笔演示单，数据库中不存在订单表属正常

---

## 🧪 测试

```bash
make test    # uv run pytest -v，全部单测不打真实模型
```

测试覆盖：网络节点图行为、混合检索/重排、工具引擎与 MCP 客户端、知识库入库/去重、上下文预算、置信度闸、飞轮数据表、主题仓储与推理库等。

---

## 📖 其他文档

- [`DEPLOY.md`](DEPLOY.md) — 给 AI 编程助手执行的分步部署任务书（含国内镜像加速、端口冲突、故障排查）
- [`Makefile`](Makefile) — `make help` 查看全部带说明的目标
- [`docs/superpowers/plans/`](docs/superpowers/plans) / [`specs/`](docs/superpowers/specs) — 各章设计与计划
- [`primer/README.md`](primer/README.md) — 前置知识配套说明

---

## 🤝 License

本项目源码用于教学用途。

<sub>⚠️ 运行需要合法的上游 API 密钥，密钥请自行保管、勿提交到仓库（`.env.example` 已含 gitignore 规则）。</sub>