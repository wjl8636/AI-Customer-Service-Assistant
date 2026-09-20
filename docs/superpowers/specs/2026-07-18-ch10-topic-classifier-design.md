# ch10 · 模型微调:多标签主题分类器(RoBERTa-wwm-ext 全参微调)设计

> 对齐课程 mewhelp-course/ch10-fine-tuning/README.md,不漏功能点。
> 目标:把飞轮攒下的低置信度问题按 17 类主题批量归类,喂给飞轮后台看各主题分布,决定先补哪块知识。

## 1. 背景与目标

ch09 数据飞轮把 bot 没答好的问题攒进 `low_confidence_questions`(现有 70 条,重复较多),并经标准化查重归并到 `review_queue`。本章训一台小分类器(哈工大讯飞 RoBERTa-wwm-ext,全参微调),挂旁路按批量把池里的问题归进 17 类主题,结果落 `topic_classifications` 表(DDL 已由用户定稿于 sql/ch10-ddl.sql),飞轮后台新增主题分布页展示各类目问题量。

**为什么微调而不是提示词(课程 README 的三道坎)**:说法无穷无尽 few-shot 举不完;方言、错别字、多诉求句容易判错且答案飘忽;海量逐条喂大模型又贵又慢。量大、类目固定、要便宜要快 → 微调小分类器。

**实时对话主链路不调它**;意图识别(ch06,会话维度)不动,主题归类是单句维度的独立任务。

## 2. 权威类目表(全系统唯一,17 类)

落 `app/core/taxonomy.py`,数据处理、预标、训练、推理、评测、前端全部 import 它,不许各自抄一份。类目名与边界说明采用课程 README 的归并术语表原文,说法示例按猫用品电商(喵喵优选)场景微调:

| # | 类目 | 什么算这一类(边界说明) | 用户的说法示例 |
|---|------|----------------------|--------------|
| 1 | 退换货 | 退货、换货、退款怎么办;修归保修维修,退归这里 | 退货、退款、退钱、想退了、七天无理由还能退不 |
| 2 | 物流 | 货走到哪了、什么时候送到;运费的钱事归运费 | 快递、发货、到哪了、怎么还不动、海外直邮 |
| 3 | 尺码 | 大小、码数合不合适 | 猫窝买大了、猫别墅尺寸、项圈偏码、适合几斤的猫 |
| 4 | 发票 | 开票、抬头、报销凭证 | 开发票、发票抬头开错了、能开增值税发票吗、合并开票 |
| 5 | 质量问题 | 商品本身的毛病 | 开胶、破了个洞、有瑕疵、猫砂盆电机坏了 |
| 6 | 运费 | 运费谁出、运费险理赔;管的是钱,货走到哪了归物流 | 包邮吗、退货运费谁承担、运费险怎么赔 |
| 7 | 优惠活动 | 券和活动怎么用、能不能叠 | 优惠券、满减、活动价、能叠加用吗、双十一有活动吗 |
| 8 | 价保 | 买完降价了补不补差价 | 刚买就降价了、能补差价吗、保价期多久 |
| 9 | 支付 | 付款环节出的问题 | 付不了款、花呗分期、扣了两次钱、货到付款、数字人民币 |
| 10 | 订单修改 | 下单之后改信息、取消订单 | 改地址、改电话号码、订单还能取消吗 |
| 11 | 库存补货 | 有没有货、什么时候补 | 有货吗、断货了、什么时候补货、有现货吗 |
| 12 | 商品信息 | 材质、功能、用法 | 什么材质、怎么洗、冻干怎么保存、废砂盒多久倒、猫粮怎么选 |
| 13 | 保修维修 | 保修期限、维修换新;修归这里,退归退换货 | 保修多久、坏了能修吗、能换新吗 |
| 14 | 账号 | 登录、绑定、账号安全 | 登录不上、忘了密码、换绑手机号、注销账号 |
| 15 | 会员积分 | 会员权益、积分怎么用 | 积分怎么用、会员几级、积分能抵钱吗 |
| 16 | 评价 | 评价、晒单的规则 | 评价怎么改、追评在哪写、晒单有奖励吗 |
| 17 | 其他 | 上面都对不上的,先兜底 | 闲聊、转人工、客服几点上班 |

**标注铁律(字面规则)**:一句话字面提到几个诉求就打几个标签,一个不多一个不少。「买大了想退」→ 尺码+退换货;「这鞋买大了一码」→ 只标尺码,不许脑补退换货。类目间关联留给业务,标注不揽这个活。

`taxonomy.py` 内容:`TOPIC_CLASSES`(17 个类目名的有序元组,顺序即 label id)、每类 `boundary`(边界说明)与 `examples`(说法示例)、`LABEL2ID/ID2LABEL`。

## 3. 数据处理流水线(四步,scripts/ch10/)

语料从池里捞真实问题,量不够由大模型照术语表造模拟问题补足。目标规模:**每类约 100 条、总量约 1700**(以标签计;多标签样本给每个命中类目都记数)。造数与预标全部走仓库现有的聊天上游(`CHAT_BASE_URL` + `CHAT_MODEL`),接口用法先查 Context7 再写。

### 3.1 捞取与文本选择

- **分类旁路只处理已归并的问题**:从 `low_confidence_questions` 捞有 `matched_review_id` 的行,文本取 `review_queue.normalized_question`(归并阶段产出的标准化单句)。未归并的不进分类,留到下一轮归并后再归——保证分类器只吃语义完整的问法、绝不喂原话(对齐 README:上游指代消解+标准化已把问题处理成语义完整单句)。
- 训练语料捞取同样优先标准化问法,未归并时退回 `raw_question`,保留 `question_id` 溯源(建训练集为一次性全量、产出的模型已冻结,故保留原话兜底)。

### 3.2 清洗

- **脱敏**:正则抹掉手机号、订单号、微信号/QQ 等敏感串,替换为占位符。
- **去重**:完全相同文本合并(池里「MH-LP100 废砂盒」重复 20+ 条,去重后真实独特问题约 35 条)。
- **修错别字与格式**:大模型批量清洗(只修字面错误与乱格式,不改语义、不改口语风格),输出与输入逐条对照落盘可复查。

### 3.3 归并术语表 + 预标 + 人工抽审(折中路线)

- 标签获取:大模型照 `taxonomy.py` 的 17 类术语表(含边界说明)**预标**,结构化输出 `{"text":..., "labels":[...]}`;prompt 里写死字面铁律与近邻边界(修/退、运费/物流、价保/优惠)。
- **预标质量闸(替代 TDD 的验证步)**:手工标 30 条黄金样例(覆盖多标签、方言、错别字、近邻类目),预标 prompt 在黄金样例上通过率 ≥ 80% 才放行批量预标(先例:ch09 validate_flywheel_samples)。
- **人工抽审**:预标完按类目分层抽样导出 `data/ch10/sample_review.md`(句子+预标标签,真实池样本全量进抽审),用户在对话里指出错标,改完再进下一步。

### 3.4 模拟语料补足

- 每类不足 100 条的缺口由大模型照术语表造模拟问题,**生成时自带标签**(同样过字面铁律)。
- 刁钻样本按比例掺入:方言口语约 15%(「俺买的那玩意儿咋还没到俺这疙瘩」)、错别字约 10%(「退活」)、多诉求多标签句约 15%(「催下快递,对了这个有色差还能换不」)。
- 模拟语料同样进人工抽审的分层抽样。

### 3.5 分层抽样划分

- 80/10/10 切训练/验证/测试三份;**按标签组合分层**(组合样本数 <10 的按其第一标签归层),保证每个类目按比例出现在三份里。
- 划分脚本可单测:比例、无重复文本跨集泄漏、每类目三份中都有样本。

### 3.6 数据增强(只扩训练集)

- 同义词替换、句式微调(陈述句改「我想问一下……」口气),大模型批量生成,不改原意、标签继承。
- 每条训练样本扩 1 个变体(训练集约翻倍);**验证集、测试集保持原样**(考题不许照练习题变)。

产物落盘:`data/ch10/corpus_raw.jsonl` → `corpus_clean.jsonl` → `corpus_labeled.jsonl` → `dataset/{train,val,test}.jsonl`(格式 `{"text": "...", "labels": ["尺码","退换货"]}`)+ `sample_review.md` + `golden_samples.jsonl`。

## 4. 训练(全参微调,不用 LoRA)

- 基座:`hfl/chinese-roberta-wwm-ext`(HuggingFace;网络不通时 `HF_ENDPOINT=https://hf-mirror.com` 兜底)。
- 方式:transformers `Trainer`,`problem_type="multi_label_classification"`(BCEWithLogitsLoss),17 维 sigmoid 输出,全参更新。
- 设备自适应:cuda → mps → cpu(本机 Apple Silicon 走 MPS)。
- 防过拟合:`weight_decay=0.01`(正则化)+ `EarlyStoppingCallback`(盯验证集 micro-F1,patience=2)+ 训练中每 epoch 评验证集。
- 超参基线:lr=2e-5,batch=16,max_length=128,epochs≤8(早停收手)。
- 判定阈值:默认 0.5;训完在验证集上扫全局最优阈值(0.3~0.7 步进 0.05),存进模型目录 `threshold.json`;**全类无一过线时取分数最高的一类兜底**(不产出空标签)。
- 产物:`data/ch10/model/`(权重+tokenizer+threshold),训练日志含每 epoch 验证分数。

## 5. 评测

- 留出测试集上按类目算精确率、召回率、F1,外加 micro/macro 汇总;每类目一张 2×2 混淆矩阵(sklearn `multilabel_confusion_matrix`)。
- **容错红线按档位设闸**,不是每类一条线:17 类按「归错了会不会带偏补知识的优先级」分档——**严**(退换货、物流、尺码、发票、质量问题)红线 F1 ≥ 0.9,**中**(运费、优惠活动、价保、支付、订单修改、库存补货、商品信息、保修维修)≥ 0.8,**宽**(账号、会员积分、评价、其他)归错影响小、不设线。不达标标红提示「先回头搞数据」;宽档那一列画 — 而不是 ✅,免得读者以为它也过了某条线。
- **判错样本导出人工复核**:测试集上预测≠标准答案的样本落 `data/ch10/reports/error_samples.md`(句子/标准标签/预测标签),人工复核错在哪一类、是不是标注本身有毛病。同时按错误方向归类——**漏打**(该打的没打,只记 1 笔放跑)/**多打**(不该打的多打一个,只记 1 笔冤枉)/**错位**(该打的没打、反而打了别的类,一次记 2 笔:被抢的类放跑 +1、抢标签的类冤枉 +1),错例条数与混淆矩阵笔数的差额全在错位上。
- **判定阈值可复算**:`scan_threshold_replay.py` 重演训练时的阈值扫描——验证集经 :8110 打分一次、分数表固定,九个候选线(0.30~0.70 步进 0.05)套同一张表各算一遍 micro-F1,当选线须与 `threshold.json` 在用的那条一致,证明阈值是扫出来的不是拍的。
- **报告一式两份**:`.md` 给人读、`.json` 给验收页读(§7.2),同一次跑写同一份数——页面只读不重算,不许在接口里算出第二个真相。落 `data/ch10/reports/`:`eval_report.{md,json}`、`error_samples.md`、`golden_report.json`、`threshold_scan.json`、`export_report.json`、`classify_run.json`。评测集只用自家场景数据,不引公开榜单。

## 6. 部署:ONNX + 独立推理服务 + 旁路批处理

### 6.1 ONNX 导出

- `torch.onnx.export` 导出(dynamic axes 支持变长 batch),导出后跑一批样本校验 ONNX 与 torch 预测一致(logits 容差内、标签完全一致)才算导出成功。
- 产物:`data/ch10/onnx/model.onnx` + tokenizer 文件。

### 6.2 推理服务(独立进程 :8110)

- FastAPI + onnxruntime + tokenizers,**不依赖 torch/transformers**(轻运行时);仓库惯例 nohup + pid 文件,make `classifier-up` / `classifier-down` 起停。
- 接口:`POST /classify` 入 `{"texts": ["...", ...]}`,出逐条 `{"labels": [...], "scores": {类目: 分数}}`(阈值用训练产物 threshold.json);`GET /healthz` 探活。

### 6.3 旁路批处理(主链路不碰)

- `scripts/ch10/classify_pool.py`:捞 `low_confidence_questions` 中尚未归类的行(LEFT JOIN `topic_classifications` 为空),按 3.1 规则取文本,批量调 :8110,结果写 `topic_classifications`(question_id / labels JSON / classified_at)。
- 「攒够一批归一次」:默认 `--min-batch 10`,不足提示不跑,`--force` 可强跑;make `classify-pool` 触发(与 ch09 flywheel 定时批处理同款手动/定时两相宜)。
- 重跑幂等:已归类的行不重复归(如需重归先清表对应行)。

## 7. 后端 API + 飞轮后台前端

本章的实证不留在终端:数据血缘、评测报告、阈值扫描、混淆矩阵、错例复核这些东西一律有页面,验收在浏览器上完成。

### 7.1 主题分布页

- ORM:`TopicClassification` 进 `app/db/models.py`;建表用 sql/ch10-ddl.sql(用户已定稿,不改)。
- API:`app/api/topics.py`,`GET /api/topics/distribution` → 17 类各自问题量(labels JSON 在 Python 侧聚合,量级小)+ 每类若干条样例问题 + 总量/最近归类时间;挂进 FastAPI app。
- 前端:`app/static/topics.html` 独立页,与 review.html 同款像素风;17 类横向条形图(纯 HTML/CSS,不引外部图表库,与仓库单文件无依赖惯例一致)+ 计数,哪类堆得多一眼看出。
- 每类只给三条样例,且**样例按文本去重**:归并后同一句标准化问法对应池里好几行,三条样例都一样就白给了(计数照旧按条算,去重只作用于样例列表)。

### 7.1.1 类目问题列表(点进某一类看全部)

分布图只回答「哪类堆得多」,补知识还得看这一类具体堆的是什么问题。点类目名进 `/topics/questions?label=<类目>`(`app/static/topic-questions.html`),分页列出这一类下归类到的全部问题。

- API:`GET /api/topics/questions?label=&page=&size=` → `{label, total, page, pages, size, items[]}`。每条给标准化问法(没归并的退回原话,并标出 `normalized=false`)、用户原话、命中的全部类目、落池入口 source、同义合并条数、审核状态、落池与归类时间。
- 类目名不在权威表里回 400 说清楚,不静静回一页空列表——那会被当成「这类没有问题」。
- 筛选与分页在 Python 侧做:`labels` 是 JSON 数组,归类总量是问题池规模(百级),一次全捞换来不依赖 MySQL 的 JSON 函数,测试库上跑得到同样结果。
- 多标签问题在它命中的每个类目下都出现,与分布图的计数口径一致。
- 类目名与页码都在地址栏里:这一页是能贴给别人的链接,刷新回到同一页。

### 7.2 验收页(四页)

九项实证按关注点拆四页,各自独立路由,在后台共用导航里归一个模块、底下挂一排子页互切:

| 路由 | 看什么 |
|------|--------|
| `/acceptance` | 九项实证各一张闸门卡,顶部总闸 N/9 与分类器在线状态;每张卡带该项的重跑按钮 |
| `/acceptance/eval` | micro vs macro 对照、每类 P/R/F1/support/红线、阈值扫描九候选线、17 类混淆矩阵、单句试分类 |
| `/acceptance/data` | 语料血缘四段、文件盘点、三份考卷分布与泄漏自检、训练三件套、ONNX 产物 |
| `/acceptance/errors` | 逐条标准/预测对照、错误方向记账、边界摩擦配对 |

- API:`app/api/acceptance.py`,前缀 `/api/acceptance`。只读端点 `overview` / `eval` / `data` / `errors` 读 §5 的 json 产物与文件 stat;`service` 现场探 :8110;`classify` 代理 :8110 做单句试分类(演示「17 类各自独立过线,过几个打几个」)。产物缺失不报错,回 `present=false` + 该跑哪个 make 目标,页面据此长出重跑按钮而不是白屏。
- **闸门三态**:`pass`(跑过且过线)/ `fail`(跑过没过线)/ `missing`(还没跑过)。missing 不显示失败态结论文案——没跑过不等于没过线,那话会把读者引到错误的下一步动作。
- **三份考卷泄漏自检是硬闸**:训练∩验证、训练∩测试、验证∩测试的文本重叠必须为 0。考题一旦被训练集见过,后面所有分数都不作数,所以这栏是闸不是提示。
- **就地重跑**:`app/core/jobs.py` 白名单作业运行器 + `app/api/jobs.py`(前缀 `/api/jobs`,发起 / 查状态与日志尾 / 停止)。注册表里每个 make 目标的 argv 全部写死在模块里,前端只传作业名,传不进任何命令片段、参数或路径——页面有重跑按钮,但没有 shell。这套运行器不专属本章:知识库建库(ch03 录入页)按的也是它,所以端点放在中立前缀下,不挂在验收 API 里面。命令一律走 make:配方是仓库里那一份,页面按的和终端敲的必须是同一条命令,不许分叉。日志落 `log/acceptance/<job>.log` 供前端轮询回显;分钟级重活(语料流水线、数据集、训练、导出)标 heavy,页面二次确认才发起;`start_new_session=True` 起独立会话,停止时 killpg 连 make → uv → python 整条链一起收,不留孤儿进程;同名作业未结束时拒绝重入,避免两份进程抢同一批产物。
- 共享外壳 `app/static/acceptance.css` + `acceptance.js`(取数、提示、重跑按钮与日志窗口那一套)四页共用,故主应用挂 `/static` 静态目录;页间导航另在 `app/static/admin.js`,后台各页共用一份;条形图、混淆矩阵、分数条仍是纯 HTML/CSS,不引外部图表库。
- 这五页(含主题分布)是后台管理的一部分:与知识库录入页、飞轮待审页共用同一份导航与配色,并在聚合首页 `/admin` 上占一张卡(过闸数 N/9 + 分类器在线状态)。**前端走 Vibe Coding:用户描述效果、直接改,不套 brainstorm/TDD/code review。**

## 8. 依赖与落仓

- `pyproject.toml` 加 `[dependency-groups] ml`:torch、transformers、accelerate、scikit-learn、onnx、onnxruntime、tokenizers、numpy;主 dependencies 零污染,`uv run --group ml` 按需装。
- 代码:`scripts/ch10/`(build_corpus.py、build_dataset.py、corpus_lib.py、prelabel.py、validate_golden.py、train.py、evaluate.py、export_onnx.py、scan_threshold_replay.py、serve.py、inference_lib.py、classify_pool.py)+ `app/core/taxonomy.py` + `app/core/jobs.py` + `app/api/jobs.py` + `app/api/topics.py` + `app/api/acceptance.py` + `app/static/`(topics.html、topic-questions.html、acceptance.html、acceptance-eval.html、acceptance-data.html、acceptance-errors.html、acceptance.css、acceptance.js、admin.js)。
- 产物:`data/ch10/`(语料/数据集/模型/onnx/报告),模型与语料不进 git(.gitignore),黄金样例与报告进 git;作业日志落 `log/acceptance/`(随 log/ 一并 gitignore)。
- Make 目标:`ch10-golden`(黄金样例验证)、`ch10-corpus`、`ch10-dataset`、`ch10-train`、`ch10-eval`、`ch10-export`、`ch10-threshold-scan`(阈值扫描重演)、`classifier-up`、`classifier-down`、`classify-pool`(`FORCE=1` 不足一批也跑)。这份清单进作业运行器的白名单(与 ch03 的建库目标同一张注册表)。

## 9. 测试与验证策略

- **可单测(TDD)**:taxonomy 模块(17 类唯一、边界说明齐全)、脱敏函数、分层划分(比例/无泄漏/类目覆盖)、阈值应用与兜底逻辑、topics API(test DB)、classify_pool 写表幂等、作业运行器(白名单外的作业名拒绝、同名作业未结束拒绝重入)、验收 API 的产物缺失回 `present=false` 不抛错。
- **验证跑替代 TDD**(纯 Prompt/数据/训练类):预标 prompt → 黄金样例 ≥80%;清洗 → 抽样对照;模拟造数 → 进人工抽审;训练 → 验证集分数曲线 + 早停触发记录;ONNX → 与 torch 预测一致性校验。
- 库/框架 API(transformers Trainer、torch.onnx、onnxruntime、FastAPI、SQLAlchemy)一律先 Context7 查最新文档再写。

## 10. 验收标准(用户定稿)

1. 测试集各类目 F1 和混淆矩阵报告能跑出来,并在 `/acceptance/eval` 上看到每类的红线达标情况。
2. 拿一批攒下的真实问题跑归类,后台主题分布页能看到各类目统计;点某一类的名字能进到这一类的问题列表,分页看全,每条带来源、同义合并条数与审核状态。
3. 「买大了想退」这类多诉求句子,能同时命中多个类目(尺码+退换货)——在 `/acceptance/eval` 的单句试分类里现场打分,17 类各自过线情况一目了然。
4. 九项实证(语料与数据集、黄金样例闸、训练产物、ONNX 导出与服务、测试集评测、阈值扫描、混淆矩阵、错例复核、旁路批量归类)在验收页上全部可看、可重跑,验收过程不必回终端。

## 11. 本章不做

- LoRA / QLoRA(大模型才用的 PEFT 手段,本章 102M 全参足够)。
- Embedding 微调(README 定位:最后手段,先做切分/混合检索/重排)、意图识别微调。
- 实时对话主链路接入分类器。
- 类目表治理后台(术语表以 taxonomy.py 代码为准)。
- 验收页的鉴权、多用户与操作审计(本地开发后台单人使用;作业运行器的安全边界靠白名单,不靠权限模型)。
- 作业排队与并发调度(同名作业拒绝重入即够,不做队列)。

## 12. 风险与对策

- **模拟语料占比高(约 96%)**:真实池仅 70 条(去重后约 35 条),分布代表性有限;对策——真实样本全量进人工抽审、评测报告注明语料构成;后续真实问题攒多了可增量重训(效果不行,先搞数据)。
- **MPS 训练兼容性**:个别算子回退 CPU 属正常;训练脚本设备自适应,速度可接受(千级样本、102M 参数)。
- **HF 模型下载**:网络不通走 hf-mirror 镜像兜底。
- **标签噪声**:同一说法标出多种组合是第一版翻车主因;黄金样例闸 + 字面铁律写死进预标 prompt + 人工抽审三道手段压住。
