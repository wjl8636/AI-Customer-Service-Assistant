# Ch09 设计:可观测性与数据飞轮(Langfuse 接入 + Cost Control + 三入口问题池 + 标准化查重待审 + 人工审核写回 + 评估趋势)

> 对齐课程文档 `mewhelp-course/ch09-observability-flywheel/README.md`:前半装可观测性(Langfuse 调用树、按意图算 token 账、评估趋势线),后半装数据飞轮(三入口低置信度问题池 → 标准化查重 → 人工审核三道闸 → 写回知识库闭环)。

## 1. 背景与目标

ch02-ch08 把主力 Agent 的能力搭齐了,但线上是个黑盒:每一步想了啥、调了哪个工具、花了多少 token,请求跑完就丢。答不上来的问题也散了:兜底一句糊弄过去,下回同样的问题照样答不上。本章两件事:**把黑盒照亮**(Langfuse trace 树 + 按意图 token 账 + 评估趋势线),**把答不上的问题变成改进燃料**(数据飞轮闭环)。

**目标(需求 6 条):**

1. **接 Langfuse**:设环境变量、图编译时挂一次回调;每条请求的完整 trace 树(每个节点的 prompt、工具调用、检索结果、token 消耗和耗时)在界面上铺开可看。
2. **Cost Control**:意图识别结果打进 trace 元数据,按意图维度统计 token 花销,知道钱花在哪类问题上。
3. **数据飞轮三入口**,都汇进 ch04 的 `low_confidence_questions` 问题池,各自标好 `source`;落池时把当时的召回片段快照(Top 几条原文和得分)存进新列 `retrieved_chunks`,给人工审核当依据,没走检索的入口空着:
   - **证据置信度低**:把 ch05 最简版置信度闸升级成正式的 `evidence_confidence`(精排 Top1 相关性分、有效证据数、Top1-Top2 分差、有没有命中关键条款),阈值拿 ch04 评估集校准;闸的位置不动(知识类检索之后、进 Agent 之前),拦下回兜底话术并落池,`source=retrieval_low_conf`。
   - **生成阶段模型自评不够答**(README 的 useful 判断,代码里是 `selfcheck`):ch04 已有判断逻辑,本章把它的落池 source 打对(`self_check`),接进飞轮。
   - **用户反馈没解决**:👎 从纯前端采集接到后端,点了就把该轮用户问题落池,`source=user_feedback`。
4. **飞轮流水线**(定时批处理):标准化(口语原话 → FAQ 式标准问题 + 示例答案)与查重(和待审队列已有问题判同义,命中就归并累加)由模型**一次输出**;入待审队列 `review_queue`(一行一个去重缺口),归并落点记回 `low_confidence_questions.matched_review_id`;人工在后台审核,通过的走 ch03 落库流程写入知识库。
5. **自动化评估流水线**:复用 ch04 评估集与指标(检索段 Recall@K、MRR,生成段 Faithfulness)定期跑,每轮分数落 `eval_runs` 表连成趋势,哪个指标在下滑一眼看出。
6. **前端配套**:待审队列后台管理页(列标准化问题、出现次数、示例答案,可通过/驳回;行内详情看归并原话 + 召回片段快照,判断知识库是真缺还是有但没检到);观测与成本页(意图成本账、评估趋势、置信度阈值校准三张报表一页看齐,也能在页上重跑);聊天页 👎 接后端。

**非目标(本章不做):**
- 低置信度问题按主题归类的微调分类器(下一章正题)。
- LangSmith(课程只作对比提及,数据出境,不接)。
- Langfuse 里配自定义模型单价算钱(统计以 token 数为准,单价配置留给运营在界面自己做)。
- 真·进程内定时器(APScheduler 等):飞轮与评估都是脚本 + make 驱动,交付物里给 cron 示例,不引调度框架。
- 👍 落任何库(需求只要求 👎 落池;👍 后端收到只记日志)。
- MCP 工具内部(`query_faq` 等)的落池改造之外的行为变更。

**技术栈(定死):** Langfuse 开源自部署(v3,官方 docker compose 栈),链路数据不出自家服务器;回调用 `langfuse.langchain.CallbackHandler`(与课程 README 代码一致)。库/框架 API 用法(FastAPI、SQLAlchemy、LangChain、LangGraph、Langfuse)实现前一律 Context7 查最新文档,不凭记忆写。

**用户澄清决策(brainstorm 定稿):**
- **Langfuse 部署**:独立 `docker-compose.langfuse.yml` + `make langfuse-up/down`,不并入现有 compose,不用 v2 精简版。
- **飞轮触发**:定时批处理脚本(否掉"落池后异步近实时"),验收时手动 `make flywheel` 触发一轮。
- **查重范围**:和 `review_queue` **全部行**比对;命中已驳回/已通过的行只累加 `occurrence_count`、**状态不变**(驳回即终审,不复活、不新建重复行)。
- **👎 快照**:尽力回捞——从该会话 checkpointer state 取最近一轮召回快照,对不上就空着。
- **整体架构**:方案 A"回调为主、脚本为辅"——观测靠编译时挂一次 CallbackHandler 吃全图,不做逐节点 @observe 插桩;飞轮/评估是批处理脚本;审核页是静态单页 + REST。
- spec 直接写完整版对齐课程 README,不逐批确认。

## 2. 架构与数据流

```
 ┌─ 可观测(前半)──────────────────────────────────────────────────────────┐
 │  docker-compose.langfuse.yml (web:3000 + worker + pg + clickhouse + redis + minio)│
 │        ▲ trace 上报(全留自家服务器)                                       │
 │  runtime.init_graph():compile().with_config({"callbacks":[CallbackHandler()]})   │
 │        │ 编译时挂一次,全图零侵入;缺 key 时不挂,系统照常跑                  │
 │  classify_intent 节点 → 把 intent 写进当前 trace 的 metadata/tag            │
 │  get_chat_model() 统一 bind include_usage → 所有 LLM 调用 token 全被采集     │
 │  scripts/cost_by_intent.py → 调 Langfuse API 按 intent 聚合 token 账         │
 │  scripts/eval_flywheel.py → 复用 ch04 评估集/指标 → eval_runs 表 → 趋势对比   │
 └──────────────────────────────────────────────────────────────────────┘
 ┌─ 数据飞轮(后半)────────────────────────────────────────────────────────┐
 │ 入口1 evidence_confidence 低 ┐ (app/core/confidence.py,阈值评估集校准)      │
 │ 入口2 selfcheck 不够答      ├→ fallback_reply 落池(source + 快照)           │
 │ 入口3 前端👎 → POST /api/feedback ┘ (尽力回捞快照)                          │
 │        ↓ low_confidence_questions (+retrieved_chunks +matched_review_id)   │
 │ make flywheel → scripts/flywheel_pipeline.py → app/core/flywheel.py         │
 │        │ 扫 matched_review_id IS NULL → LLM 一次输出标准化+查重+示例答案       │
 │        ↓ review_queue(命中→occurrence_count+1;未命中→新建待审行)            │
 │ /review 审核页(垃圾/时效/频次三道闸) → POST approve/reject                  │
 │        │ approve:核准答案 → dualwrite.write_pending → vectorize_pending      │
 │        ↓ 知识库(MySQL + Milvus) → 下次同类问题检索命中 → 闭环合上            │
 └──────────────────────────────────────────────────────────────────────┘
```

## 3. Langfuse 部署与接入

### 3.1 部署(`docker-compose.langfuse.yml`)

- 以 Langfuse 官方 v3 自部署 compose 为底:`langfuse-web`(暴露 `localhost:3000`)、`langfuse-worker`、`postgres`、`clickhouse`、`redis`、`minio`。与现有 `docker-compose.yml`(mysql + milvus)解耦,不用时整体停掉省资源。
- Makefile:`langfuse-up` / `langfuse-down`。
- 首次启动后在界面建组织/项目,拿 public/secret key 写进 `.env`。README 补一段初始化步骤。

### 3.2 配置与挂载

- `app/config.py` 新增:`langfuse_public_key` / `langfuse_secret_key` / `langfuse_base_url`(均默认空字符串)。环境变量名与课程 README 及 SDK 文档一致:`LANGFUSE_BASE_URL`(已 Context7 核实);`Langfuse(...)` 构造参名(host/base_url)实装时再核。
- `runtime.init_graph()`:编译完图后,**三个配置齐全才** `.with_config({"callbacks": [CallbackHandler()]})`,否则原样返回——观测是可选增强,不成为启动硬依赖,测试环境不需要 Langfuse。
- 会话归属:每次 invoke/astream 的 config 里把 `conversation_id` 传成 Langfuse 的 session(v3 用 `metadata={"langfuse_session_id": ...}` 之类的键,准确键名 Context7 定),同一会话多轮在界面串成一组。
- 依赖:`pyproject.toml` 加 `langfuse`。

### 3.3 token usage 收口

现状只有 `main_agent` bind 了 `stream_options={"include_usage": True}`,意图/自评/摘要的调用不回传 usage,按意图统计会漏账。本章把这个 bind 收进 `app/core/llm.py` 的 `get_chat_model()` 统一处理,`main_agent` 里的重复 bind 移除;state 的 `tokens_used` 累加逻辑不动。

### 3.4 意图打进 trace 元数据

`classify_intent` 节点得出意图后,把 `intent`(及 `intent_confidence`)写进当前 trace 的 metadata,并把 intent 打成 trace tag 便于界面过滤。写法用 Langfuse SDK 的"更新当前 trace"能力(v3 的 `langfuse.update_current_trace(...)` 或等价 API,Context7 定);Langfuse 未启用时该调用必须静默跳过(包一层 guard,不许影响主流程)。

## 4. Cost Control:按意图算 token 账

- `scripts/cost_by_intent.py` + `make cost-report`:调 Langfuse 公开 API(优先 Metrics API 按 metadata.intent 分组;若分组维度不支持则退化为拉 traces 列表自行聚合——两条路 plan 阶段查文档后定一条)。
  **实装勘误(2026-07-17)**:两条路都不通——metadata 不可作分组维度,且自部署 v3.218 的 v1 Metrics 端点只开放 observations/scores 视图(traces 视图与 v4 typed client 均需 server v4);最终走 `api.legacy.metrics_v1` 按 observations 视图以 tags+traceId 分组、客户端聚合(意图打成 `intent:xxx` tag)。输出:

```
意图         请求数   总tokens   平均tokens   占比
商品咨询       12     58,320      4,860      41%   ← 最烧钱
退款退货        8     31,200      3,900      22%
...
```

- 支持 `--days N` 时间窗参数;结果落 `data/ch09/reports/cost_by_intent.txt` 与同名 `.json`(json 给「观测与成本」页读,见 10.2)。平均 token 与占比由脚本算好写进产物,页面不再算一遍。

## 5. 置信度闸升级(入口 1)

### 5.1 `app/core/confidence.py`(新)

```python
@dataclass
class EvidenceConfidence:
    score: float          # 0-1,加权组合后的总分
    signals: dict         # 各原始信号,落池 reason 与 trace 留痕用

def compute_evidence_confidence(hits: list[dict]) -> EvidenceConfidence:
    # 信号(README 点名的四个):
    #   top1_score      精排 Top1 的 rerank_score(0-1)
    #   valid_count     有效证据数:rerank_score >= 有效线的条数(归一化封顶)
    #   margin          Top1 - Top2 分差(证据聚焦度;只有一条时取 top1 自身)
    #   key_clause_hit  Top 命中里有没有关键条款标记(ch03 build_chunks 打过标)
    # 组合:固定权重线性加权归一到 0-1;权重放模块常量,不进 settings(校准的是阈值不是权重)
```

- 纯函数、不碰网络,可 TDD。
- 阈值 `settings.evidence_confidence_threshold`,默认值由校准脚本给出后写死进 config 默认值。

### 5.2 阈值校准(`scripts/calibrate_confidence.py` + `make calibrate-confidence`)

- 拿 `tests/data/eval_ch04.jsonl`:A/B/C 桶(应可答)与 D 桶(应拒答)逐条跑 hybrid_rerank 检索,算各自的 `evidence_confidence` 分布。
- 扫候选阈值,选可答/应拒分离最优的点(最大化 Youden J = 真阳率 - 假阳率),输出分布表 + 推荐阈值。
- 校准报告落 `data/ch09/reports/confidence_calibration.txt` 与同名 `.json`(json 带分布、整条扫描曲线、推荐阈值,给「观测与成本」页画图);人把推荐值写进 `config.py` 默认值——**不拍脑袋定**,这一步是验收演示的一部分。

### 5.3 接入位置(闸位不动)

`retrieve_knowledge`(nodes.py)里改造,顺序保持"检索 → 置信度闸 → 语义自评闸 → 进 Agent 或兜底":

1. 现有机械闸(`top < settings.rerank_min_score`)替换为 `compute_evidence_confidence(hits).score < threshold` → 不过则 `evidence_strong=False`,`fallback_source="retrieval_low_conf"`,reason 带信号值(如 `top1=0.21, valid=1, margin=0.05, key_clause=false`)。
2. 置信度过了才跑 `selfcheck.check_sufficient`(省一次模型调用的顺序不变);selfcheck 说不够 → `evidence_strong=False`,`fallback_source="self_check"`(**修正现状:fallback 落池 source 此前硬编码 retrieval_low_conf,self_check 从未被写入**)。
3. **只要走了检索就把快照存进 state**(不只闸不过时):`retrieved_snapshot = [{"question":..., "answer":..., "rerank_score":..., "section_path":...}, ...]`(精排后前 3)。闸不过时它随落池写进 `retrieved_chunks`;闸都过、正常作答的轮次它留在 state 里,供用户事后点 👎 时的快照回捞(6.1)——踩的往往正是"检到了但答砸"的轮次。

state 新字段(`app/graph/state.py`):`evidence_confidence: float`、`fallback_source: str`、`retrieved_snapshot: list`。

### 5.4 落池(`fallback_reply` 改造)

- 兜底话术与转人工按钮行为不变。
- `insert_low_confidence` 调用改为:`source=state.fallback_source`(不再硬编码)、`reason` 带信号值、`retrieved_chunks=state.retrieved_snapshot`(JSON 列)。

## 6. 用户反馈入口(入口 3)

### 6.1 后端 `POST /api/feedback`(`app/api/feedback.py` 新)

请求体:`{"conversation_id": int, "rating": "up"|"down", "question": str}`(question = 被评价那轮的用户原话,前端从 DOM 里带上来)。

- `up`:记一行应用日志,返回 200,不落库。
- `down`:落 `low_confidence_questions`,`source="user_feedback"`,`reason="用户反馈未解决"`;
  - **快照尽力回捞**:从 checkpointer 取该 `conversation_id` 的图 state(`graph.aget_state`),若 state 里最近一轮的用户问题与请求体 `question` 一致且有 `citations`/`retrieved_snapshot`,把它作为 `retrieved_chunks` 存下;对不上或该轮没走检索则存 NULL。
- 幂等:同一轮重复 👎 前端已禁用;后端不做强幂等(教学项目,重复落池由飞轮查重兜住)。

### 6.2 前端(`app/static/index.html`,vibe coding)

- `addFeedbackBar()` 的 👎 点击处理加一段 `fetch("/api/feedback", ...)`,带上该轮用户问题文本;失败静默(反馈条 UI 行为不变)。👍 也发(后端只记日志),保持埋点对称。

## 7. 飞轮流水线(定时批处理)

### 7.1 ORM(`app/db/models.py`)

- 新增 `ReviewQueue` 映射 `review_queue`(注意 `review_status` 是中文值 ENUM `'待审'/'通过'/'驳回'`,SQLAlchemy 用原生 Enum 映射)。
- 新增 `EvalRun` 映射 `eval_runs`。
- `LowConfidenceQuestion` 加 `retrieved_chunks`(JSON)、`matched_review_id`(FK)两列映射。
- DDL 已定稿于 `sql/ch09-ddl.sql`(用户手写),conftest 解析 CREATE/ALTER 的既有机制天然覆盖。

### 7.2 标准化 + 查重(`app/core/flywheel.py` 新)

- Prompt(`app/core/prompts.py` 加 `FLYWHEEL_NORMALIZE_PROMPT`):输入用户原话 + 候选问题列表(`review_queue` 全部行的 `id + normalized_question`,按 `updated_at` 倒序截 200 条防 token 爆),模型**一次输出**(与课程 README 的 JSON 形状一致):

```json
{
  "normalized_question": "商品出现质量问题(如开胶)能否退货",
  "matched_question_id": 128,        // 命中候选就填其 id,没有同类为 null
  "ai_suggested_answer": "..."       // 示例答案备查
}
```

- `process_pending(limit)` 主流程:
  1. 扫 `low_confidence_questions` where `matched_review_id IS NULL`(处理游标,天然幂等,失败重跑不重复);
  2. 逐条调模型标准化+查重;
  3. 命中(`matched_question_id` 非空且存在):`review_queue.occurrence_count + 1`,**状态不动**(命中已驳回/已通过的行也只累加——驳回即终审);
  4. 未命中:新建 `review_queue` 行(status=待审,occurrence_count=1);
  5. 回写 `low_confidence_questions.matched_review_id`;
  6. 模型输出解析失败/幻觉 id(候选里不存在):该条跳过并告警日志,游标留在原地下轮重试。
- **串行逐条处理**(不并发):同批内两条同义新问题若并发会各自建行,串行则第二条能命中第一条刚建的行。

### 7.3 驱动(`scripts/flywheel_pipeline.py` + `make flywheel`)

- 打印本轮处理条数、新建缺口数、归并数;交付物给 cron 示例(如 `*/30 * * * * cd ... && make flywheel`),不引进程内调度器。

### 7.4 repository 新方法(`app/db/repository.py`)

`fetch_unmatched_low_conf(limit)` / `list_review_candidates(limit)` / `insert_review_item(...)` / `increment_occurrence(id)` / `set_matched_review(lcq_id, review_id)` / `list_review_queue(status)` / `get_review_detail(id)`(含归并原话列表) / `update_review_status(id, status, approved_answer)` / `insert_eval_run(...)` / `list_eval_runs(limit)`。

## 8. 审核 API 与写回知识库

### 8.1 REST(`app/api/review.py` 新)

| 接口 | 行为 |
|---|---|
| `GET /api/review/queue?status=待审` | 列表:标准化问题、出现次数、示例答案、状态;按 `occurrence_count` 降序(频次=优先级) |
| `GET /api/review/{id}` | 详情:队列行 + 归并进来的原话列表(`raw_question`、`source`、`created_at`、`retrieved_chunks` 快照) |
| `POST /api/review/{id}/approve` | 体:`{"approved_answer": str}`;置状态=通过、存核准答案,并**同步写回知识库** |
| `POST /api/review/{id}/reject` | 置状态=驳回 |

- 仅"待审"状态可 approve/reject,其余 409。教学项目不做登录鉴权(与现有接口一致)。

### 8.2 写回(ch03 落库流程复用)

approve 时:以 `normalized_question` + `approved_answer` 构造一条 QA 知识 chunk(形状对齐 ch03 挖掘写回的 faq 类 chunk),走 `app/kb/dualwrite.py` 的 `write_pending` → `vectorize_pending`(**同步执行**,需嵌入上游在线)。向量化成功才提交"通过"状态;失败则整体回滚返回 5xx,可重试——保证"点了通过 ≙ 下次能检到"(验收 3)。

## 9. 自动化评估流水线

- `scripts/eval_flywheel.py` + `make eval-flywheel`:复用 `scripts/eval_ch04.py` 的评估集与指标实现(抽公共函数,不复制粘贴):
  - 检索段:hybrid_rerank 策略的 Recall@10、MRR(可答桶);
  - 生成段:Faithfulness(LLM 判)+ D 桶拒答率;
- 每轮跑完落 `eval_runs`(`triggered_by`:CLI 参数 `--triggered-by 手动|定时`,默认手动;`dataset_size`;`metrics` JSON)。
- 输出趋势对比:读最近 N 轮,打印每个指标相对上一轮的涨跌(`↑ / ↓ / →`),**下滑的指标标 ⚠**;同时把趋势表落 `data/ch09/reports/eval_trend.txt`。趋势本身的权威源是 `eval_runs` 表,页面直接读表,不读这份文本——同一条趋势不留两个出处。
- 定期 = make + cron 示例;验收要求至少两轮有对比。

## 10. 前端(vibe coding,不走 TDD/review 流程)

- **聊天页**:👎 接 `/api/feedback`(见 6.2)。
- **审核页** `app/static/review.html`,挂 `/review`(main.py 加一条静态路由):
  - 列表:标准化问题、出现次数(降序)、示例答案摘要、状态筛选(默认待审);
  - 行操作:通过(弹框填核准答案,预填 `ai_suggested_answer`)/ 驳回;
  - 行详情展开:归并进来的用户原话列表(带入池入口 source 标签、时间)+ 每条的召回片段快照卡(原文 + 得分)——审核人对着快照判断"知识库真缺这块,还是有但没检到";
  - 页顶放审核三道闸提示文案(垃圾过滤 / 时效 / 频次,课程 README 的审核要点);
  - 样式复用 index.html 的猫咪 CSS 变量与卡片风格。
  - 具体交互效果按 vibe coding:先出一版,用户描述效果再改。
- **审核页属于后台管理的一块**:与知识库录入页(`/kb`)共用 ch03 搭好的那套外壳——一份共用导航(`app/static/admin.js` 的 `mountAdminNav`)+ 聚合首页 `/admin` 上占一张卡(待审 / 通过 / 驳回三个数 + 一句结论)。审核页仍在 `/review` 这个原路径上——导航只是把入口收到一处,不改各页地址。

### 10.2 观测与成本页(三张终端报表搬进后台)

本章的三样东西原来只在终端里:意图成本账、评估趋势、置信度阈值校准。它们全是「跑一次看结论」的报表,归到一页 `/observability`(`app/static/observability.html`),报表在页面上看,重跑也在页面上按。

- **只读 API** `app/api/observability.py`(`GET /api/observability/overview`),三块各带自己的状态与 job:
  | 块 | 权威源 | 缺产物时 |
  |---|---|---|
  | 意图成本账 | `data/ch09/reports/cost_by_intent.json` | `present=false` + 提示先聊几句攒 trace 再重跑 |
  | 评估趋势 | `eval_runs` 表最近 10 轮 | `present=false` + 提示按一次就是趋势的第一个点 |
  | 置信度阈值校准 | `data/ch09/reports/confidence_calibration.json` + `settings.evidence_confidence_threshold` | `present=false` + 提示去扫一遍 |
- **不许出现第二个真相**:占比、单均、Youden J、推荐阈值都由脚本算好落在产物里,API 一个都不重算;涨跌箭头由页面拿同一批 `eval_runs` 行两两相减画,不是另一份数据。
- **三块互不连坐**:Langfuse 没起只是成本账没产物,mysql 挂了只是趋势那块显示读数失败,剩下的照样端出去。
- **重跑按钮**走 ch03 那套白名单作业运行器(`app/core/jobs.py` 注册 `cost-report` / `eval-flywheel` / `calibrate-confidence`,后两个是分钟级重活,页面二次确认);前端只传作业名,传不进任何命令片段。
- **推荐阈值与在用阈值不一致**时页面明说「该回填 `app/config.py`」,后台首页那张卡也跟着标要干活——校准跑完不回填,等于没校准。
- 与其他后台页同一套外壳:导航加一格「观测与成本」,`/admin` 首页多一张卡。

### 10.3 图底下那句「读图」交给模型写(`app/core/read_notes.py`)

每张图下面有一句「读图」小注,原来是写死在页面里的。数换了、结论还是那句话:单均最高的意图换成一路不走多步工具的,写死的「ReAct 那类多步工具链」半句就成了错的。所以这句话改由模型看着**这一轮的数**生成。

- **生成时机在产物落盘那一刻**,不在页面渲染时。三条理由:看板不该依赖上游活着(依赖不齐的时候恰恰最该打开它);同一份产物每次打开都得是同一句话,刷新一次换一种说法不像结论;重跑时数和注一起换,不会新数配旧注。
- **两道闸挡幻觉**:prompt 只喂这一轮的数并明说不许自己算;落盘前机械校验,注里出现的每个数字都得在给它的数据里对得上(含四舍五入、百分比、千分位逗号,以及比例的余数——0.85 的通过率说成「15% 被误拦」是同一个数的另一面)。对不上、太长、超时,一律丢掉整条,页面回落自己那句兜底话。宁可显示旧口径的句子,也不能把编出来的数印在页面上。标识符里的数字不算数字(`p25`、`Recall@10`、`bge-m3`),不然真话也会被误杀。
- **payload 给中文标签**:喂字段名进去,注里就会冒出 `C_colloquial`、`avg_tokens` 这种只有代码里才有的说法。
- **注跟着产物走**:`rag_eval.json` / `cost_by_intent.json` / `confidence_calibration.json` 各自多一个 `read_notes` 字段;趋势的权威源是 `eval_runs` 表不是文件,所以那句注旁挂 `eval_trend_note.json` 并记下它描述的轮次 id,API 只在这个 id 还是最新一轮时才端出去。
- 前端一个共用函数 `readNoteHtml(note, fallbackHtml)`:有注就用注(转义后把数字自动加粗),没有才用页面自己那句。这一层同时管着 `/rag-eval` 的四张图。

## 11. 测试与验证策略

- **TDD 单测**(可单测代码):`compute_evidence_confidence` 纯函数(各信号边界:空召回、单条、同分);flywheel 归并逻辑(mock LLM 返回:命中/未命中/幻觉 id/坏 JSON);`/api/feedback`(up 不落库、down 落库、快照回捞对得上/对不上);`/api/review` 四个接口(approve 走 mock 的 dualwrite、失败回滚);repository 新方法;`fallback_reply` 落池 source/快照;`init_graph` 缺 key 不挂回调。
- **标注样例验证代替 TDD**(纯 Prompt/数据类):
  - 标准化+查重 prompt:造 `tests/data/flywheel_samples.json`(约 10-15 条:口语原话 × 候选队列 × 期望命中/不命中),脚本跑通过率,阈值 ≥ 80% 视为可用;
  - 置信度阈值:校准脚本本身就是验证(评估集分布 + 分离度);
  - 评估流水线:跑两轮真实评估集,趋势对比表出得来即验证。
- **前端**:vibe coding 例外,浏览器里人工验收。
- 现有测试零回归:`make test`。

## 12. 验收标准映射

| # | 验收 | 演示 |
|---|---|---|
| 1 | Langfuse 里点开任意请求看到完整链路 | `make langfuse-up` + 起服务发一条消息,localhost:3000 点开 trace 树 |
| 2 | 问知识库没有的问题 → 兜底话术 + 进待审队列,详情能看原话与快照 | 聊天页问 D 桶类问题 → `make flywheel` → /review 页点开详情 |
| 3 | 审核通过后同一问题再问能答对(飞轮整圈) | /review 点通过填核准答案 → 聊天页再问 → 答对 |
| 4 | 聊天页点 👎 → 落池 → 标准化查重后出现在待审队列 | 点 👎 → `make flywheel` → /review 出现该问题 |
| 5 | 按意图汇总的 token 花销统计 | `make cost-report` 输出意图 × token 表,同一份数在 `/observability` 上看得到 |
| 6 | 评估流水线至少两轮、指标趋势对比可见 | `make eval-flywheel` 跑两轮 → 趋势表(含 ↑↓/⚠),`/observability` 的趋势那块同源 |
| 7 | 三张报表在浏览器里看得到、也按得动 | `/observability` 三块都有数,按「重跑 意图成本账」现场跑一轮,日志尾回显在页面上 |

## 13. 风险与开放问题

- **Langfuse v3 compose 较重**(6 容器):独立 compose 已隔离;机器吃紧时验收 1/5 单独起。
- **SDK API 形态**(CallbackHandler 构造、session/metadata 键名、update_current_trace、Metrics API 分组能力)一律以 Context7 查到的 v3 文档为准,README 示例仅作对齐参照;若查实与课程 README 写法冲突,停下来向用户确认(工作要求 4)。
- **ENUM 中文值**(`'待审'` 等)的 ORM 映射与 charset:DDL 已 utf8mb4,SQLAlchemy 侧用字符串值 Enum;测试库重建走 conftest 既有 DDL 解析,ch09-ddl.sql 的 CREATE+ALTER 符合其范式。
- **approve 同步向量化依赖嵌入上游在线**:失败回滚 + 明确报错,审核页提示重试;不做异步补偿队列(YAGNI)。
- **查重候选截断**(200 条):教学规模够用;截断时日志说明,不静默。
