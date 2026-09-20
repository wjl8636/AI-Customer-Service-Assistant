# Ch04 设计:混合检索 + 重排 + 评估体系

状态:已定稿(2026-07-14 brainstorm 通过)。
前置:ch03 已交付 dense 单路语义检索(`query_faq` 内部走 Milvus Lite dense Top-K)。本章把检索质量做上一个台阶。

## §0 目标与不做

**做**:混合检索(Milvus 原生 BM25 + dense,hybrid_search RRF 融合)、重排(bge-reranker-v2-m3 精排 Top-10)、元数据过滤、Query 理解(改写 + 同义词扩展)、生成质量控制(带引用 / 拒答 / 自评落池 / 负面知识)、评估体系(四策略对比:检索排序 + 证据覆盖度 + 端到端答案覆盖度,分桶报告 + 后台里的 RAG 评估页)、前端可点引用与每段回答满意度反馈(👍/👎)。

**不做**(本章边界):指代消解、多轮改写。

**固定选型(定死,不自换)**:Milvus 2.5+ 原生 BM25 全文检索 + `hybrid_search` RRF;重排模型 `bge-reranker-v2-m3`。嵌入沿用 ch03 的 `BAAI/bge-m3`(直连硅基流动)。

## §1 架构总览与数据流

### 1.1 基础设施变更:Milvus Lite → Standalone

ch03 用的 Milvus Lite **不支持**原生 BM25 全文检索(analyzer + BM25 Function),这是 Milvus 官方文档明确的当前限制(full-text search 仅 Standalone / Distributed / Zilliz Cloud 支持,Lite 计划未来实现)。要落地固定选型,Milvus 部署形态升到 **Standalone**(docker-compose 加 `milvus-standalone` + `etcd` + `minio`),`settings.milvus_uri` 改为 `http://localhost:19530`。

附带收益:ch03 记录的「Milvus Lite 独占锁 → 在线 app 与离线建库不能同时持库、只能单 worker」遗留问题,升 Standalone 后自然消失(建库与在线可并发)。

### 1.2 Milvus 集合 schema(重建)

集合 `knowledge` 重建为多字段 + 双向量 + BM25 Function:

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | INT64(主键, auto_id=False) | = MySQL `knowledge_chunks.id`,重跑 upsert 幂等(延续 ch03) |
| `dense` | FLOAT_VECTOR(dim=1024, COSINE) | bge-m3 稠密向量 |
| `text` | VARCHAR(enable_analyzer=True, analyzer=chinese) | BM25 Function 输入;入库内容 = `category + questions + answer` 拼接(与 dense 同源) |
| `sparse` | SPARSE_FLOAT_VECTOR(SPARSE_INVERTED_INDEX / BM25) | BM25 Function 输出,系统自动生成 |
| `question` / `answer` | VARCHAR | 原文,检索一次取回(延续 ch03「milvus 里取就行」) |
| `section_path` | VARCHAR | 章节路径,引用溯源用 |
| `content_type` | VARCHAR | faq/policy/manual |
| `category` | VARCHAR | 品类/上级标题,元数据过滤用 |

BM25 Function(`FunctionType.BM25`,input=`text`,output=`sparse`)必须在 `create_collection` 前 `schema.add_function` 挂上;索引 `dense`→AUTOINDEX/COSINE、`sparse`→SPARSE_INVERTED_INDEX/BM25。

> dense 向量维度沿用 ch03 冒烟实测的 1024。text 与 dense 同源(都由 `category+questions+answer` 得来),保证两路召回的是同一批 chunk、id 对得上。

### 1.3 在线主链路(单一 agent 入口,`query_faq` 升级为 RAG 工具)

保持 ch01–03 的单一聊天入口与 tool-calling agent 编排不变;事务型工具(query_order / query_product / query_logistics / create_ticket)原样保留。`query_faq` 内部从「dense Top-K」升级为完整 RAG 检索管线:

```
聊天页 → /api/chat → agent(bind_tools) → 命中 query_faq
  query_faq 内部(RAG 检索管线):
    ① Query 理解      口语→标准问法改写 + 同义词扩展(仅检索侧,不改入库)
    ② 元数据过滤      按 category 等字段先过滤(Milvus filter 表达式)
    ③ 混合召回        dense Top-50 + BM25 Top-50 → hybrid_search([dense_req, bm25_req], RRFRanker())
    ④ 重排            bge-reranker-v2-m3(直连上游 /rerank)精排出 Top-10
    ⑤ 检索证据低闸    最高 rerank 分 < 阈值或无命中 → 判证据不足(见下)
    ⑥ 生成前自评闸    结构化自评:模型判召回证据够不够答这个问题(useful?)→ 不够判证据不足
    ⑦ 首尾组装        证据足才组装:最相关放首、次相关放尾,其余居中(缓解 lost-in-the-middle)
    ⑧ 返回            带编号证据块 + "证据足,请据此作答" 给模型;
                      citations 元数据(编号→id/section_path/原文)暂存到 ToolRun
  → agent 收敛生成:证据足 → 带引用作答(system prompt 含引用规则 + 负面知识);
                    证据不足 → 据工具信号显式拒答
  → SSE 多推一个 citations 事件,把 section_path/原文/编号下发前端
```

**证据不足分支(两道闸,先后判,任一触发即拒答)**:
- ⑤ 机械闸:管线 ④ 后最高 rerank 分 < 阈值或无命中 → 落 `low_confidence_questions(source=retrieval_low_conf)`。
- ⑥ 语义闸(生成前自评):证据召回了但模型自评「不足以回答」→ 落 `low_confidence_questions(source=self_check)`。

两闸都**在生成之前**完成:query_faq 工具判定证据不足时,不再组装证据、直接返回「未检索到足以回答的知识,请向用户拒答」信号并同步落池(落池是确定性的,不依赖模型),agent 收敛步据此拒答;两闸互补——机械闸挡「没召回到/相似度全低」,语义闸挡「召回了同主题但没答案」。

### 1.4 技术取舍

- 不引入 LangGraph / Langfuse,沿用 ch03 的纯 async 函数编排(与既有 `agent.py` 一致)。
- 重排走 **上游 `/rerank`**(SiliconFlow),它是 Jina / Cohere 那套形状、不是 OpenAI 协议,所以手写请求、不用 openai 客户端。可行性排为 fail-fast 冒烟任务(见 §6.3)。

## §2 检索管线组件(确定性代码,TDD)

| 组件 | 职责 | 测试要点 |
|---|---|---|
| `milvus_client`(扩展) | 新 schema + BM25 Function;`hybrid_search([dense_req, bm25_req], RRFRanker())`;`dense_search` / `bm25_search` 单路(评估用);`filter` 元数据过滤参数 | 真实临时 Standalone 集合:两路各召回、RRF 融合有序、filter 生效、BM25 命中型号词 |
| `rerank`(新) | 直连上游 /rerank 调 bge-reranker-v2-m3,输入 query + docs → 重排序 + 截断 Top-10 | mock 上游:按 relevance 重排、Top-10 截断、空输入安全 |
| `retrieval.search_knowledge`(重写) | 编排 ①②③④⑤;`strategy` 参数切换四策略(在线与评估复用同一函数) | mock 子步:管线顺序、首尾组装排列正确、strategy 分支正确 |
| `query_understanding`(新,Prompt) | 口语→标准问法改写 + 同义词扩展(检索侧) | 纯 Prompt → 标注样例验证(非 TDD) |

**`strategy` 入参**(`vector` / `bm25` / `hybrid` / `hybrid_rerank`)是关键设计:在线默认 `hybrid_rerank`;评估脚本用同一 `search_knowledge` 跑四策略,保证"评估的就是线上的"。同义词扩展只作用于 BM25 检索文本,不改 dense query、不改入库。

## §3 生成质量控制(Prompt + 少量确定性代码)

- **带引用生成**:system prompt 要求答案在句/段末标 `[n]`,`n` 对应 ⑧ 的证据编号。编号→`section_path`/原文的映射由前端消费 SSE `citations` 事件完成。
- **生成前结构化自评(证据够不够,先判后生成)**:重排出 Top-10 后、组装进生成 prompt 之前,做一次结构化自评调用,输入 query + 召回证据,输出**扁平字段** `useful: bool` + `reason: str`(避开 ch03 记录的 glm-5.2 对嵌套对象数组 502 的坑,不用 list[对象])。`useful=false` → 不进入生成,query_faq 返回拒答信号 + 落 `low_confidence_questions(source=self_check, reason=reason)`。`useful=true` → 组装证据、进生成。
- **检索证据低**:见 §1.3 ⑤机械闸,`source=retrieval_low_conf`,同样在生成前触发。
- **负面知识**:RAG 生成 system prompt 明列禁止承诺项(不承诺到账时间 / 到货时间 / 赔偿时效等),统一「以平台售后规则为准」。沿用并扩充 ch03 `prompts.py` 的行为约束。
- `low_confidence_questions` 写入走确定性 repository 方法 → TDD;各 prompt → 标注样例验证。
- `user_feedback` 入口(枚举第三值)本章不接线,留 ch09 数据飞轮;本章只做 `retrieval_low_conf` 与 `self_check` 两入口(与 ch04-ddl.sql 注释一致)。

## §4 评估体系(数据 + 脚本,TDD 换成跑评估集)

- **评估集** `tests/data/eval_ch04.jsonl`:五桶 300 题(每桶 60,含跨文档桶 `E_multi`——一问要两三块不同小节的知识),带难度梯度;每题标注 `expect_section`(预期命中章节关键词,算 Recall/MRR)与 `expect_points`(标准答案要点,A/B/C 用,算覆盖度):
  - A 政策/流程通用类(20)
  - B 型号/具体名词类(20,验收2;含**易混型号族**——多款饮水机/猫砂盆/加热垫/喂食器,考验「精确匹配」对「语义混淆」的区分)
  - C 口语模糊改写类(20,query 理解)
  - D 库里没有→应拒答类(20,验收4)
- **检索段指标(确定性,可复现)**:Recall@5、MRR(A/B/C/E 四个可作答桶,按 `expect_section` 命中 `section_path` 计;跨文档题用 `expect_sections_all` 按组给部分分;D 桶不计)。
- **证据覆盖度(确定性 · 跨策略)**:四策略各自召回的 Top-K 证据里,机械匹配盖住 `expect_points` 的比例——衡量「检索有没有把答题所需事实捞上来」,零 LLM、可复现。
- **生成段指标(glm)**:
  - **四策略答案覆盖度**:四策略各自把召回证据交给 glm 生成答案,再由裁判数「答案盖住了几个 `expect_points`」——端到端证明「检索越好、答案越全」;
  - **Faithfulness**:hybrid_rerank 线上管线答案的忠实度(主张是否被证据支撑;A/B/C 桶);
  - **拒答率**:D 桶(库外问题应全部拒答)。
  - 裁判沿用 glm-5.2(项目上游唯一模型),自评自判偏袒风险为已知取舍、可接受(用户拍板)。
- **四策略对比**:`vector` / `bm25` / `hybrid` / `hybrid_rerank`,复用 `search_knowledge(strategy=...)`,「评估的就是线上的」。
- **裁判本身也要校准**:两个裁判跑 `temperature=0`(0.3 下同一份输入重放会给出不同判断,那点抖动会直接写进分数);忠实度提示词写明七类豁免 + 两类真编造 + 六条边界样例;`make judge-check` 拿台账里已处置的个案原样重放,量裁判与人工的一致率,不一致就退出码 1。
- **健壮性**:生成段每次 glm 调用带超时 + 失败跳过 + 低并发,单点故障不拖垮整轮;检索段与证据覆盖度不依赖 glm、始终产出。
- **产出**:`scripts/eval_ch04.py` → 策略 × 桶 × 指标分桶报告,跑完落 `data/ch04/reports/` 两份产物,`rag_eval.txt` 是这轮的运行日志给人读,`rag_eval.json` 给页面读。入口 `make eval-rag`。

## §4.1 RAG 评估页(报告在后台里看,也在后台里跑)

评估这条链原本只在终端:`make eval-rag` 跑完,读 stdout 或者打开生成出来的那份 html 文件。报告落成后台的一页 `/rag-eval`:

| 页面上的块 | 内容 | 数据来源 |
|---|---|---|
| 报告头 | 评估集题数、知识库块数、嵌入与重排模型、裁判模型、上次跑的时间 | `rag_eval.json` 的 `meta` |
| 四项 KPI | 最佳整体 MRR、口语桶 MRR 提升、答案覆盖度、库外拒答率 | 同一份产物,不另算 |
| 检索质量 | 四策略 × 四个可作答桶分组柱状图,MRR / Recall@5 / 证据覆盖度三个指标切着看,每张图配一句读图 | `retrieval` + `evidence_coverage` |
| 生成质量 | 四策略答案覆盖度柱状图、上线管线四桶忠实度、库外拒答率圆环 | `generation` |
| 完整数据 | 分桶 MRR 与两项覆盖度汇总表,总体 MRR 最佳那一行带底色;编造个案默认收起,展开逐条看问题、生成答案、裁判理由 | `generation.faithfulness_cases` |
| 编造个案台账 | 顶上两个口径的幻觉率(本轮裁判判出率 / 本轮人工确认幻觉率,分子只数本轮报告判出的题号,台账累计单独一行);跨轮累计的编造个案:状态页签 + 分页,每条带这一轮的 Top-K 证据全集(标出答案引用了哪几条)与「已解决 / 无需解决」按钮,处置要填一句说明 | `faith_cases` 表(`/api/rag-eval/faith-cases`) |
| 就地重跑 | 「重跑 RAG 评估」按钮 + 日志窗口,跑完自动重新取数 | `/api/jobs`(作业名 `eval-rag`) |

**不变量**:

- **页面只读产物,不重算**。API 端出去的每个指标都来自 `rag_eval.json`,MRR 不在 web 层再算一遍。页面上的数与终端 `make eval-rag` 跑出来的必须是同一份,不许有第二个真相。
- **重跑走共用的白名单作业运行器**(`app/core/jobs.py` + `/api/jobs`,ch03 建库与 ch10 验收按的也是它)。前端只传作业名 `eval-rag`,传不进任何命令片段。分钟级重活,页面二次确认才发起。
- **产物缺失是状态**。没跑过就回 `present=false` + 该按哪个作业,页面长出「去跑一次」而不是白屏;产物写坏了(半个 json)也按没跑过处理。
- **半份结果照样端出来**。裁判上游挂掉那一轮 `generation` 是 null,检索段与证据覆盖度照常呈现,生成段单独标「本次未完成」,补跑一次就齐。
- 页面挂进后台共用导航(`app/static/admin.js`),并在聚合首页 `/admin` 上占一张卡(最佳 MRR、评估集题数、库外拒答率)。

## §5 前端(Vibe Coding,不套 brainstorm/TDD/review)

在现有 `app/static/index.html` 聊天页:回答文本里的 `[n]` 渲染成可点角标,点开浮层/侧栏显示该 citation 的 `section_path` + 来源 chunk 原文。数据来自 SSE 新增的 `citations` 事件。效果由用户描述、迭代式直接改。

另在每段助手回答的左下角挂一组满意度反馈(👍 / 👎):点击即点亮所选、另一个淡出、显示「已反馈」,一次性锁定,纯前端交互。这条信号将来入库进问题池(`low_confidence_questions.source = user_feedback`,§7 schema 已预留该入口)以驱动数据飞轮,属后续数据飞轮章的衔接,本章只做前端采集、不接线落库。

## §6 测试策略、冒烟与验收映射

### 6.1 TDD(确定性代码,红→绿→提交)
milvus 混合检索 / filter / RRF、rerank 客户端、`search_knowledge` 编排与首尾组装、citation 解析与暂存、`low_confidence_questions` 写入。真实临时 Milvus Standalone 集合进单测(嵌入 / rerank / LLM mock)。

### 6.2 标注样例验证(替代 TDD 的纯 Prompt / 数据任务)
query 改写、结构化自评、faithfulness judge、评估集本身。

### 6.3 Fail-fast 冒烟(排最前,red line)
- SiliconFlow 的 `/rerank` → `BAAI/bge-reranker-v2-m3` 跑通(确认上游提供该模型、返回形状是 results + relevance_score);
- Milvus Standalone 原生 BM25 全文检索 + `hybrid_search` 跑通(analyzer=chinese 中文分词命中)。

任一不通 → 停下问用户,只在固定选型内换法(不自行换方案)。

### 6.4 验收映射

| 验收 | 覆盖 |
|---|---|
| 1 四策略对比报告能跑出数字 | §4 `make eval-rag` 报告 + §4.1 RAG 评估页 |
| 2 型号具体问题 BM25 那路能命中 | §1.3③ BM25 路 + B 桶评估 + 浏览器实测 |
| 3 引用编号定位原文、聊天页点引用看来源 | §3 引用 + §5 前端 + 浏览器实测 |
| 4 库里没有→明确拒答且进低置信池 | §3 拒答 + D 桶 + 浏览器实测落库 |
| 5 每段回答可点 👍/👎、点亮并显示已反馈 | §5 前端 + 浏览器实测 |

## §7 数据模型

- 新表 `low_confidence_questions`(见 `sql/ch04-ddl.sql`):`id / conversation_id(FK) / raw_question / source(retrieval_low_conf|self_check|user_feedback) / reason / created_at`。本章写入 `retrieval_low_conf` 与 `self_check` 两来源。
- 新表 `faith_cases`(同一份 `sql/ch04-ddl.sql`):`id / eval_id(唯一,一题一行) / bucket / query / strategy / answer / reason / citations(这一轮喂给模型的 Top-K 证据全集快照 JSON,答案里的角标 [n] 即其序号) / judge_model / status(未解决|已解决|无需解决) / resolution(处置说明,标已解决/无需解决时必填) / seen_count / first_seen_at / last_seen_at / resolved_at`。报告是产物、重跑即覆盖,个案与其处置状态要跨轮留存;同一题再判编造只更新该行并累加次数,已处置又被判出即退回未解决(复发)。
- `knowledge_chunks`(ch03 已建)复用其 `section_path` / `content_type` / `category` 字段,建库时一并写入 Milvus 新 schema 的对应标量字段。
- 知识库内容(`data/kb/*.md`)覆盖四类(faq/policy/manual/spec)44 块(ch09 飞轮审核写回后库里为 51 块);`spec` 收录多款**易混型号族**(饮水机 W20/W40/W60、猫砂盆 LP100/LP50/LP200、加热垫 HP12/HP20、喂食器 FD10/FD30 等),使四策略评估在「精确匹配 vs 语义混淆」上具区分度。

## §8 决策记录

1. Milvus Lite → Standalone(docker):Lite 不支持原生 BM25 全文检索,是落地固定选型的必要基础设施变更(非换选型);顺带解决 ch03 独占锁遗留。(用户拍板)
2. RAG 管线接入方式:升级 `query_faq` 为 RAG 工具 + 引用回传,保持单一 agent 入口(非独立 RAG 路由端点)。(用户拍板)
3. 评估集:4 桶 80 题(每桶 20),标注 `expect_section` + `expect_points`,一一对应四条验收与四个功能点。(用户拍板)
4. 重排接入:直连上游 /rerank,手写请求(它不是 OpenAI 协议)。(默认,冒烟验证)
5. 不引入 LangGraph/Langfuse,沿用纯 async 编排。(默认,与 ch03 一致)
6. 自评结构化输出用扁平字段(`useful`/`reason`),避开 glm-5.2 嵌套对象数组 502。(ch03 教训)
7. text 与 dense 同源(`category+questions+answer`),两路召回同批 chunk。(用户确认「三个一起」)
8. 自评为**生成前**的证据充分性闸(先判够不够、不够即拒答不进生成),非生成后自评覆盖。(用户纠偏)
9. Faithfulness 裁判维持 glm-5.2,不换裁判模型。(用户拍板)
10. 评估分三层指标:检索排序(Recall/MRR,确定性)+ 证据覆盖度(机械匹配 · 跨策略 · 确定性)+ 端到端答案覆盖度(四策略生成 → glm 判分),兼顾可复现与端到端说服力;知识库配易混型号族使策略可区分。
11. 评估报告随 `make eval-rag` 落盘:`rag_eval.txt` 是运行日志,`rag_eval.json` 给 RAG 评估页读,数字始终与本次结果一致。生成段带超时 / 跳过 / 低并发保障健壮性。

## §9 风险闸

- **最大风险**:上游 /rerank + Milvus Standalone BM25 两条新链路能否跑通 → 排为第一个 fail-fast 冒烟任务(§6.3),不通即停下问。
- **前置依赖**:docker 起 Milvus Standalone 三件套(milvus/etcd/minio),资源占用比 Lite 大;`SILICONFLOW_API_KEY` 已在 `.env`(ch03)。
- Milvus 集合重建需重跑建库 + 向量化(ch03 幂等三段式复用)。

## §10 文件布局(预计)

- 改:`app/kb/milvus_client.py`(新 schema + hybrid_search + 单路 + filter)、`app/core/retrieval.py`(重写编排 + strategy)、`app/tools/business.py`(query_faq 返回带编号证据 + citations 暂存)、`app/core/agent.py` + `app/api/chat.py`(citations SSE 事件、自评拒答)、`app/core/prompts.py`(RAG 生成/自评/改写 prompt + 负面知识)、`app/config.py`(milvus_uri、rerank_model、top-50/top-10 等)、`app/static/index.html`(可点引用)、`docker-compose` / `Makefile`。
- 新:`app/core/rerank.py`、`app/core/query_understanding.py`、`app/db/repository.py` 加 low_conf 写入、`scripts/eval_ch04.py`、`scripts/smoke_rerank.py`、`tests/data/eval_ch04.jsonl`、相关单测。
