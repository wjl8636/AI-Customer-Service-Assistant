# ch10 多标签主题分类器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 训练 RoBERTa-wwm-ext 全参微调的 17 类多标签主题分类器,旁路批量归类飞轮低置信度问题,结果落 topic_classifications 表;飞轮后台出主题分布页与四页验收页,九项实证在浏览器上看、在浏览器上重跑。

**Architecture:** 数据流水线(捞池→清洗→预标→模拟补足→抽审→划分→增强)全走脚本落 `scripts/ch10/` + `data/ch10/`;权威类目表唯一落 `app/core/taxonomy.py`;训练/评测/ONNX 导出走 uv 依赖组 ml;推理是独立 FastAPI 进程 :8110(仅 onnxruntime 轻运行时);旁路批处理脚本调服务写表。各脚本的结论一式两份落盘(`.md` 给人读、`.json` 给页面读),主应用加只读的分布 API 与验收 API(只读 json 产物不重算)+ 白名单作业运行器(把 make 目标搬到页面按钮上)+ 五页静态前端。

**Tech Stack:** transformers v5(Trainer, `problem_type="multi_label_classification"`)、torch(MPS)、scikit-learn、torch.onnx(`dynamo=False` + dynamic_axes)、onnxruntime、tokenizers、FastAPI、SQLAlchemy 2 async、LangChain(造数/预标走聊天上游)。

## Global Constraints

- 对齐 spec:`docs/superpowers/specs/2026-07-18-ch10-topic-classifier-design.md`;对齐课程 `mewhelp-course/ch10-fine-tuning/README.md`,不漏功能点。
- 技术选型定死:RoBERTa-wwm-ext(`hfl/chinese-roberta-wwm-ext`)全参微调、不用 LoRA/QLoRA;ONNX + 独立 FastAPI 服务;实现走不通**停下来问用户,不许自行换方案**。
- 涉及库 API(transformers/torch/onnxruntime/FastAPI/SQLAlchemy/LangChain)先 Context7 查最新文档再动手。已查证并写死进本计划:TrainingArguments 用 `eval_strategy`(不是 evaluation_strategy);Trainer 用 `processing_class=`(不是 tokenizer=);`load_best_model_at_end=True` 要求 save/eval strategy 一致;EarlyStoppingCallback 要求设 `metric_for_best_model`;torch 2.9+ `torch.onnx.export` 默认 `dynamo=True`,导 HF 模型走稳定路线**必须显式 `dynamo=False`** 配 `dynamic_axes`。
- 每完成一个任务:在 `dev-notes/ch10.md` 追记一段(用户关键原话/关键产出/纠偏/翻车返工),**不许收尾时一次性补记**;dev-notes 不进 git(仓库 .gitignore 惯例)。
- 纯 Prompt/数据/训练类任务用「验证跑」替代 TDD(黄金样例闸、分布表、一致性校验);可单测的照常 TDD。
- 前端(Task 13、Task 16)走 Vibe Coding:用户描述效果直接改,不套 TDD/code review。
- 页面上的数一律读脚本落盘的 json 产物,**不许在 API 里重算一遍**——同一个指标出现两个来源就会出现两个真相。
- 验收页的重跑按钮只能触发白名单里的 make 目标,argv 写死在 `app/core/jobs.py`;前端传作业名,不传命令片段。页面按的和终端敲的必须是同一条 make 配方。
- 直接在 main 上开发提交(仓库历代章节惯例,无 feature 分支)。
- 训练/推理产物不进 git:`.gitignore` 加 `data/ch10/*` + `!data/ch10/reports`;黄金样例落 `scripts/ch10/golden_samples.jsonl` 进 git。
- 类目表全系统唯一:任何脚本/服务/前端要类目一律 `from app.core.taxonomy import ...`,不许抄第二份。
- 测试命令:`PYTHONPATH=. uv run pytest tests/... -v`(需本地 mysql,test 库 `mewhelp_ch02_test` 由 conftest 自建);LLM 相关脚本需上游可用(`.env` 里填好 `CHAT_*` 三项)。

---

### Task 1: 依赖组 ml + 权威类目表 taxonomy.py(TDD)

**Files:**
- Modify: `pyproject.toml`(加 [dependency-groups] ml)
- Modify: `.gitignore`(data/ch10 产物忽略)
- Create: `app/core/taxonomy.py`
- Test: `tests/core/test_taxonomy.py`

**Interfaces:**
- Produces: `TOPIC_CLASSES: tuple[TopicClass, ...]`(TopicClass: name/boundary/examples/severity)、`TOPIC_NAMES: tuple[str, ...]`、`LABEL2ID: dict[str,int]`、`ID2LABEL: dict[int,str]`、`NUM_CLASSES == 17`、`terminology_table() -> str`。后续所有任务只从这里取类目。

- [ ] **Step 1: pyproject.toml 的 `[dependency-groups]` 里 dev 组后追加 ml 组**

```toml
ml = [
    "torch>=2.6",
    "transformers>=4.46",
    "accelerate>=1.0",
    "scikit-learn>=1.5",
    "onnx>=1.17",
    "onnxruntime>=1.20",
    "tokenizers>=0.20",
]
```

- [ ] **Step 2: .gitignore 末尾追加**

```
data/ch10/*
!data/ch10/reports
```

- [ ] **Step 3: 写失败测试 `tests/core/test_taxonomy.py`**

```python
from app.core.taxonomy import (ID2LABEL, LABEL2ID, NUM_CLASSES, TOPIC_CLASSES,
                               TOPIC_NAMES, terminology_table)


def test_seventeen_unique_classes():
    assert NUM_CLASSES == 17
    assert len(set(TOPIC_NAMES)) == 17


def test_four_leads_first():
    assert TOPIC_NAMES[:4] == ("退换货", "物流", "尺码", "发票")


def test_every_class_has_boundary_examples_severity():
    for c in TOPIC_CLASSES:
        assert c.boundary.strip()
        assert len(c.examples) >= 3 or c.name == "其他"
        assert c.severity in ("严", "中", "宽")


def test_label_id_roundtrip():
    for name, i in LABEL2ID.items():
        assert ID2LABEL[i] == name
    assert LABEL2ID["退换货"] == 0 and LABEL2ID["其他"] == 16


def test_terminology_table_covers_all():
    table = terminology_table()
    for name in TOPIC_NAMES:
        assert name in table
```

- [ ] **Step 4: 跑测试确认失败**

Run: `PYTHONPATH=. uv run pytest tests/core/test_taxonomy.py -v`
Expected: FAIL(ModuleNotFoundError: app.core.taxonomy)

- [ ] **Step 5: 写 `app/core/taxonomy.py`**

```python
"""ch10 权威归并术语表:全系统唯一一份,17 类主题类目。
数据处理、预标、训练、推理、评测、前端全部 import 这里,不许各自抄一份。
类目名与边界说明采用课程 README 归并术语表原文;元组顺序即 label id,训练/推理共用。
severity 是容错档位:归错会带偏补知识优先级的类目从严。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TopicClass:
    name: str
    boundary: str                 # 一句边界说明:什么算这一类,近邻类目靠它划开
    examples: tuple[str, ...]     # 用户的说法示例
    severity: str                 # 容错档位:严 / 中 / 宽


TOPIC_CLASSES: tuple[TopicClass, ...] = (
    TopicClass("退换货", "退货、换货、退款怎么办;修归保修维修,退归这里",
               ("退货", "退款", "退钱", "想退了", "七天无理由还能退不"), "严"),
    TopicClass("物流", "货走到哪了、什么时候送到;运费的钱事归运费",
               ("快递", "发货", "到哪了", "怎么还不动", "海外直邮"), "严"),
    TopicClass("尺码", "大小、码数合不合适",
               ("猫窝买大了", "猫别墅尺寸", "项圈偏码", "适合几斤的猫"), "严"),
    TopicClass("发票", "开票、抬头、报销凭证",
               ("开发票", "发票抬头开错了", "能开增值税发票吗", "合并开票"), "严"),
    TopicClass("质量问题", "商品本身的毛病",
               ("开胶", "破了个洞", "有瑕疵", "猫砂盆电机坏了"), "严"),
    TopicClass("运费", "运费谁出、运费险理赔;管的是钱,货走到哪了归物流",
               ("包邮吗", "退货运费谁承担", "运费险怎么赔"), "中"),
    TopicClass("优惠活动", "券和活动怎么用、能不能叠",
               ("优惠券", "满减", "活动价", "能叠加用吗", "双十一有活动吗"), "中"),
    TopicClass("价保", "买完降价了补不补差价",
               ("刚买就降价了", "能补差价吗", "保价期多久"), "中"),
    TopicClass("支付", "付款环节出的问题",
               ("付不了款", "花呗分期", "扣了两次钱", "货到付款", "数字人民币"), "中"),
    TopicClass("订单修改", "下单之后改信息、取消订单",
               ("改地址", "改电话号码", "订单还能取消吗"), "中"),
    TopicClass("库存补货", "有没有货、什么时候补",
               ("有货吗", "断货了", "什么时候补货", "有现货吗"), "中"),
    TopicClass("商品信息", "材质、功能、用法",
               ("什么材质", "怎么洗", "冻干怎么保存", "废砂盒多久倒", "猫粮怎么选"), "中"),
    TopicClass("保修维修", "保修期限、维修换新;修归这里,退归退换货",
               ("保修多久", "坏了能修吗", "能换新吗"), "中"),
    TopicClass("账号", "登录、绑定、账号安全",
               ("登录不上", "忘了密码", "换绑手机号", "注销账号"), "宽"),
    TopicClass("会员积分", "会员权益、积分怎么用",
               ("积分怎么用", "会员几级", "积分能抵钱吗"), "宽"),
    TopicClass("评价", "评价、晒单的规则",
               ("评价怎么改", "追评在哪写", "晒单有奖励吗"), "宽"),
    TopicClass("其他", "上面都对不上的,先兜底",
               ("闲聊", "转人工", "客服几点上班"), "宽"),
)

TOPIC_NAMES: tuple[str, ...] = tuple(c.name for c in TOPIC_CLASSES)
LABEL2ID: dict[str, int] = {name: i for i, name in enumerate(TOPIC_NAMES)}
ID2LABEL: dict[int, str] = {i: name for i, name in enumerate(TOPIC_NAMES)}
NUM_CLASSES = len(TOPIC_CLASSES)
SEVERITY: dict[str, str] = {c.name: c.severity for c in TOPIC_CLASSES}


def terminology_table() -> str:
    """预标/造数 prompt 用的术语表文本:类目:边界说明(示例)。"""
    return "\n".join(
        f"- {c.name}:{c.boundary}(示例:{'、'.join(c.examples)})" for c in TOPIC_CLASSES
    )
```

- [ ] **Step 6: 跑测试确认通过**

Run: `PYTHONPATH=. uv run pytest tests/core/test_taxonomy.py -v`
Expected: 5 passed

- [ ] **Step 7: 装 ml 组验证可解析(不训练,只装)**

Run: `uv sync --group ml`
Expected: 解析成功装上 torch/transformers 等(首次下载较久)

- [ ] **Step 8: Commit + dev-notes 追记**

```bash
git add pyproject.toml .gitignore uv.lock app/core/taxonomy.py tests/core/test_taxonomy.py
git commit -m "feat(ch10): 权威类目表 taxonomy.py(17 类,课程 README 原表)+ ml 依赖组"
```

---

### Task 2: 建表 + ORM + repository(TDD)

**Files:**
- Modify: `app/db/models.py`(文件尾追加 TopicClassification)
- Modify: `app/db/repository.py`(尾部追加 4 个函数)
- Modify: `tests/conftest.py`(_DDL_FILES 加 ch10-ddl.sql;_TABLES 头部加 topic_classifications)
- Test: `tests/db/test_topic_repository.py`

**Interfaces:**
- Consumes: `app.core.taxonomy.TOPIC_NAMES`
- Produces:
  - `TopicClassification`(ORM:id/question_id/labels: list/classified_at)
  - `repository.list_pool_texts() -> list[dict]`(全量池,dict: question_id/text,text 优先 normalized_question)
  - `repository.list_unclassified_questions(limit=500) -> list[dict]`(排除已归类,且只取已归并的问题——未归并的不进分类)
  - `repository.insert_topic_classifications(rows: list[dict]) -> int`(rows: [{question_id, labels}])
  - `repository.topic_distribution(samples_per_class=3) -> dict`({total, latest, classes: [{label, count, samples}]},17 类全出、顺序同 TOPIC_NAMES)

- [ ] **Step 1: 开发库应用 DDL**

Run: `mysql -uroot -proot -h127.0.0.1 mewhelp < sql/ch10-ddl.sql && mysql -uroot -proot -h127.0.0.1 mewhelp -e "SHOW CREATE TABLE topic_classifications\G" | head -5`
Expected: 表结构输出

- [ ] **Step 2: conftest.py 注册 DDL 与清表顺序**

`_DDL_FILES` 列表末尾加一行:

```python
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch10-ddl.sql",
]
```

`_TABLES` 最前面插入 `"topic_classifications"`(它 FK 指向 low_confidence_questions,必须先删):

```python
_TABLES = ["topic_classifications", "low_confidence_questions", "review_queue", "eval_runs",
           "messages", "tickets", "tool_audit_logs", "conversations", "faq",
           "qa_extraction_staging", "knowledge_chunks"]
```

- [ ] **Step 3: 写失败测试 `tests/db/test_topic_repository.py`**

```python
import pytest

from app.db import repository
from app.db.models import LowConfidenceQuestion, ReviewQueue


async def _seed_question(session_maker, raw="猫窝买大了想退", norm=None):
    async with session_maker() as s:
        review_id = None
        if norm:
            rq = ReviewQueue(normalized_question=norm)
            s.add(rq)
            await s.flush()
            review_id = rq.id
        q = LowConfidenceQuestion(raw_question=raw, source="retrieval_low_conf",
                                  matched_review_id=review_id)
        s.add(q)
        await s.commit()
        return q.id


async def test_insert_and_exclude_classified(db_session_maker):
    qid = await _seed_question(db_session_maker, norm="猫窝买大了能不能退货")  # 已归并才进分类
    rows = await repository.list_unclassified_questions()
    assert any(r["question_id"] == qid for r in rows)

    n = await repository.insert_topic_classifications(
        [{"question_id": qid, "labels": ["尺码", "退换货"]}])
    assert n == 1
    rows = await repository.list_unclassified_questions()
    assert not any(r["question_id"] == qid for r in rows)


async def test_unmerged_excluded_from_classify(db_session_maker):
    qid = await _seed_question(db_session_maker, raw="猫抓板回收吗")  # 无 norm=未归并
    rows = await repository.list_unclassified_questions()
    assert not any(r["question_id"] == qid for r in rows)


async def test_text_prefers_normalized(db_session_maker):
    qid = await _seed_question(db_session_maker, raw="这个还能退吗",
                               norm="智能猫砂盆能否七天无理由退货")
    rows = await repository.list_pool_texts()
    row = next(r for r in rows if r["question_id"] == qid)
    assert row["text"] == "智能猫砂盆能否七天无理由退货"


async def test_topic_distribution_counts_all_17(db_session_maker):
    qid = await _seed_question(db_session_maker)
    await repository.insert_topic_classifications(
        [{"question_id": qid, "labels": ["尺码", "退换货"]}])
    dist = await repository.topic_distribution()
    assert len(dist["classes"]) == 17
    by_label = {c["label"]: c for c in dist["classes"]}
    assert by_label["尺码"]["count"] >= 1 and by_label["退换货"]["count"] >= 1
    assert by_label["尺码"]["samples"]
```

注:`db_session_maker` 为 conftest 既有 fixture(若名字不同,以 tests/db 现有用法为准,先读 tests/db/ 里任一测试对齐写法)。

- [ ] **Step 4: 跑测试确认失败**

Run: `PYTHONPATH=. uv run pytest tests/db/test_topic_repository.py -v`
Expected: FAIL(TopicClassification / repository 函数不存在)

- [ ] **Step 5: models.py 文件尾追加 ORM**

```python
class TopicClassification(Base):
    """ch10 主题分类结果:旁路批量归类,一行 = 一条问题的一次归类;labels 存命中类目名数组。"""
    __tablename__ = "topic_classifications"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    question_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("low_confidence_questions.id"))
    labels: Mapped[list] = mapped_column(JSON)
    classified_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

- [ ] **Step 6: repository.py 尾部追加 4 个函数**

```python
# ---------- ch10 主题分类 ----------

def _pool_text_stmt():
    """池问题取文本的公共查询:优先归并后的标准化问法,没归并退回原话。"""
    return (
        select(LowConfidenceQuestion.id, LowConfidenceQuestion.raw_question,
               ReviewQueue.normalized_question)
        .outerjoin(ReviewQueue, LowConfidenceQuestion.matched_review_id == ReviewQueue.id)
        .order_by(LowConfidenceQuestion.id)
    )


async def list_pool_texts() -> list[dict]:
    """ch10 训练语料捞取:全量低置信度问题。"""
    async with async_session() as s:
        rows = (await s.execute(_pool_text_stmt())).all()
    return [{"question_id": qid, "text": norm or raw} for qid, raw, norm in rows]


async def list_unclassified_questions(limit: int = 500) -> list[dict]:
    """ch10 旁路批处理捞取:尚未归类、且已归并的低置信度问题。
    只归有 matched_review_id 的问题——分类器只吃归并阶段产出的标准化问法,
    未归并的留到下一轮归并后再归,不喂原话。"""
    stmt = (
        _pool_text_stmt()
        .outerjoin(TopicClassification,
                   TopicClassification.question_id == LowConfidenceQuestion.id)
        .where(TopicClassification.id.is_(None))
        .where(LowConfidenceQuestion.matched_review_id.is_not(None))
        .limit(limit)
    )
    async with async_session() as s:
        rows = (await s.execute(stmt)).all()
    return [{"question_id": qid, "text": norm or raw} for qid, raw, norm in rows]


async def insert_topic_classifications(rows: list[dict]) -> int:
    """批量写归类结果;rows: [{question_id, labels}]。"""
    async with async_session() as s:
        s.add_all([TopicClassification(question_id=r["question_id"], labels=r["labels"])
                   for r in rows])
        await s.commit()
    return len(rows)


async def topic_distribution(samples_per_class: int = 3) -> dict:
    """主题分布:17 类各自问题量 + 每类样例;labels JSON 在 Python 侧聚合(量级小)。"""
    from app.core.taxonomy import TOPIC_NAMES

    stmt = (
        select(TopicClassification.labels, LowConfidenceQuestion.raw_question,
               ReviewQueue.normalized_question, TopicClassification.classified_at)
        .join(LowConfidenceQuestion,
              TopicClassification.question_id == LowConfidenceQuestion.id)
        .outerjoin(ReviewQueue, LowConfidenceQuestion.matched_review_id == ReviewQueue.id)
    )
    async with async_session() as s:
        rows = (await s.execute(stmt)).all()
    counts = {name: 0 for name in TOPIC_NAMES}
    samples: dict[str, list[str]] = {name: [] for name in TOPIC_NAMES}
    latest = None
    for labels, raw, norm, ts in rows:
        text = norm or raw
        latest = ts if latest is None or ts > latest else latest
        for lb in labels or []:
            if lb in counts:
                counts[lb] += 1
                if len(samples[lb]) < samples_per_class:
                    samples[lb].append(text)
    return {
        "total": len(rows),
        "latest": latest.isoformat() if latest else None,
        "classes": [{"label": n, "count": counts[n], "samples": samples[n]}
                    for n in TOPIC_NAMES],
    }
```

同文件头部 import 区把 `TopicClassification` 加进 `from app.db.models import ...`。

- [ ] **Step 7: 跑测试确认通过**

Run: `PYTHONPATH=. uv run pytest tests/db/test_topic_repository.py tests/db -v`
Expected: 新增 3 个测试 passed,既有 db 测试不回归

- [ ] **Step 8: Commit + dev-notes 追记**

```bash
git add app/db/models.py app/db/repository.py tests/conftest.py tests/db/test_topic_repository.py
git commit -m "feat(ch10): topic_classifications ORM + 池捞取/写表/分布聚合 repository"
```

---

### Task 3: corpus_lib 纯函数:脱敏 / 去重 / 分层划分(TDD)

**Files:**
- Create: `scripts/ch10/__init__.py`(空文件)
- Create: `scripts/ch10/corpus_lib.py`
- Test: `tests/test_ch10_corpus_lib.py`

**Interfaces:**
- Consumes: `app.core.taxonomy.LABEL2ID`
- Produces:
  - `desensitize(text: str) -> str`
  - `dedupe(samples: list[dict]) -> list[dict]`(按 text 精确去重,保序取首见)
  - `split_dataset(samples: list[dict], seed=42) -> tuple[list, list, list]`(train/val/test ≈ 80/10/10,按标签组合分层,组合 <10 条的按 label id 最小的标签归层)

- [ ] **Step 1: 写失败测试 `tests/test_ch10_corpus_lib.py`**

```python
from scripts.ch10.corpus_lib import dedupe, desensitize, split_dataset


def test_desensitize_masks_sensitive_keeps_model_no():
    assert desensitize("我手机13812345678帮我查下") == "我手机[手机号]帮我查下"
    assert desensitize("订单202601180001234567还没发") == "订单[单号]还没发"
    assert desensitize("加我微信 cat_lover2026 聊") == "加我微信[账号]聊"
    # 商品型号不许误伤
    assert "MH-LP100" in desensitize("MH-LP100 猫砂盆废砂盒多久倒一次")


def test_dedupe_keeps_first():
    out = dedupe([{"text": "a", "origin": "pool"}, {"text": "a", "origin": "simulated"},
                  {"text": "b", "origin": "pool"}])
    assert [s["text"] for s in out] == ["a", "b"]
    assert out[0]["origin"] == "pool"


def _make_samples():
    samples = []
    for cls, n in (("退换货", 40), ("物流", 40), ("尺码", 30)):
        samples += [{"text": f"{cls}问题{i}", "labels": [cls]} for i in range(n)]
    samples += [{"text": f"多诉求{i}", "labels": ["尺码", "退换货"]} for i in range(12)]
    samples += [{"text": f"小组合{i}", "labels": ["物流", "退换货"]} for i in range(3)]
    return samples


def test_split_ratio_and_no_leakage():
    train, val, test = split_dataset(_make_samples())
    total = len(train) + len(val) + len(test)
    assert total == 125
    assert 0.72 <= len(train) / total <= 0.88
    texts = [s["text"] for s in train + val + test]
    assert len(texts) == len(set(texts))


def test_split_every_class_in_every_split():
    train, val, test = split_dataset(_make_samples())
    for split in (train, val, test):
        found = {lb for s in split for lb in s["labels"]}
        assert {"退换货", "物流", "尺码"} <= found


def test_split_deterministic():
    a = split_dataset(_make_samples(), seed=42)
    b = split_dataset(_make_samples(), seed=42)
    assert a == b
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. uv run pytest tests/test_ch10_corpus_lib.py -v`
Expected: FAIL(scripts.ch10 不存在)

- [ ] **Step 3: 写 `scripts/ch10/corpus_lib.py`(以及空 `scripts/ch10/__init__.py`)**

```python
"""ch10 数据流水线纯函数:脱敏、去重、分层划分。不碰网络与 DB,可单测。"""
import random
import re

from app.core.taxonomy import LABEL2ID

_PHONE = re.compile(r"1[3-9]\d{9}")
_LONG_DIGITS = re.compile(r"\d{10,}")            # 订单号/运单号一类长数字串
_WX_QQ = re.compile(r"(微信|weixin|wx|QQ|qq)[号:: ]*[A-Za-z0-9_-]{5,}")


def desensitize(text: str) -> str:
    """脱敏:手机号 → [手机号],长数字单号 → [单号],微信/QQ 号 → 前缀+[账号]。
    商品型号(如 MH-LP100,字母开头短串)不匹配上述模式,不受影响。"""
    text = _PHONE.sub("[手机号]", text)
    text = _LONG_DIGITS.sub("[单号]", text)
    text = _WX_QQ.sub(lambda m: m.group(1) + "[账号]", text)
    return text


def dedupe(samples: list[dict]) -> list[dict]:
    """按 text 精确去重,保序取首见;顺带 strip 掉首尾空白。"""
    seen: set[str] = set()
    out: list[dict] = []
    for s in samples:
        t = s["text"].strip()
        if t and t not in seen:
            seen.add(t)
            out.append({**s, "text": t})
    return out


def split_dataset(samples: list[dict], seed: int = 42) -> tuple[list, list, list]:
    """80/10/10 分层划分:按标签组合分层;组合样本 <10 条的并进其 label id 最小
    标签的单标签层,保证小类目不至于在验证/测试里绝迹。层内 <3 条全给训练集。"""
    rng = random.Random(seed)
    combos: dict[tuple, list[dict]] = {}
    for s in samples:
        key = tuple(sorted(s["labels"], key=LABEL2ID.get))
        combos.setdefault(key, []).append(s)
    strata: dict[tuple, list[dict]] = {}
    for key, items in combos.items():
        target = key if len(items) >= 10 else (min(key, key=LABEL2ID.get),)
        strata.setdefault(target, []).extend(items)
    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for key in sorted(strata):            # 排序保证确定性
        items = strata[key]
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * 0.1)) if n >= 3 else 0
        n_val = max(1, round(n * 0.1)) if n >= 3 else 0
        test.extend(items[:n_test])
        val.extend(items[n_test:n_test + n_val])
        train.extend(items[n_test + n_val:])
    return train, val, test
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. uv run pytest tests/test_ch10_corpus_lib.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit + dev-notes 追记**

```bash
git add scripts/ch10/__init__.py scripts/ch10/corpus_lib.py tests/test_ch10_corpus_lib.py
git commit -m "feat(ch10): 语料纯函数——脱敏/去重/标签组合分层划分"
```

---

### Task 4: 预标模块 + 黄金样例验证闸(验证跑替代 TDD)

**Files:**
- Create: `scripts/ch10/prelabel.py`
- Create: `scripts/ch10/golden_samples.jsonl`(30 条,进 git)
- Create: `scripts/ch10/validate_golden.py`
- Modify: `Makefile`(加 ch10-golden)

**Interfaces:**
- Consumes: `app.core.llm.get_chat_model`、`taxonomy.terminology_table/LABEL2ID`
- Produces: `prelabel_batch(texts: list[str]) -> list[list[str]]`(逐条标签数组,非法标签过滤、空结果兜底 ["其他"])

- [ ] **Step 1: 写 `scripts/ch10/prelabel.py`**

```python
"""ch10 预标:大模型照权威术语表给问题打多标签。折中路线的前半——预标,后半人工抽审。"""
import asyncio

from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.taxonomy import LABEL2ID, terminology_table


class _Labeled(BaseModel):
    labels: list[str] = Field(description="命中的类目名数组,取术语表类目名原文")


PRELABEL_PROMPT = """你是电商客服主题标注员。对照下面这份 17 类权威归并术语表,给用户问题打主题标签。

术语表(类目:边界说明(示例)):
{terminology}

标注铁律:
1. 字面提到几个诉求就打几个标签,一个不多一个不少。「买大了想退」→ 尺码+退换货;\
「这鞋买大了一码」→ 只标尺码,不许因为「可能要退」就脑补退换货。
2. 近邻边界:修归保修维修、退归退换货;运费管钱、物流管货;价保是补差价、优惠活动是券和满减。
3. 方言、口语、错别字要看穿字面认诉求:「俺买的那玩意儿咋还没到俺这疙瘩」是物流;「退活」是「退货」的错字。
4. 都对不上就标「其他」;标签必须取术语表的类目名原文。

用户问题:{text}"""


async def prelabel_one(text: str) -> list[str]:
    model = get_chat_model().with_structured_output(_Labeled)
    try:
        r: _Labeled = await model.ainvoke(
            PRELABEL_PROMPT.format(terminology=terminology_table(), text=text))
    except Exception:
        return ["其他"]
    labels = [lb for lb in r.labels if lb in LABEL2ID]
    return labels or ["其他"]


async def prelabel_batch(texts: list[str], concurrency: int = 8) -> list[list[str]]:
    sem = asyncio.Semaphore(concurrency)

    async def one(t: str) -> list[str]:
        async with sem:
            return await prelabel_one(t)

    return list(await asyncio.gather(*[one(t) for t in texts]))
```

- [ ] **Step 2: 写黄金样例 `scripts/ch10/golden_samples.jsonl`(30 条,覆盖 17 类 + 方言/错别字/多诉求/近邻边界)**

```jsonl
{"text": "七天无理由还能退不", "labels": ["退换货"]}
{"text": "我的快递到哪了怎么还不动", "labels": ["物流"]}
{"text": "猫窝买大了想换个小一号的", "labels": ["尺码", "退换货"]}
{"text": "能开增值税发票吗", "labels": ["发票"]}
{"text": "猫砂盆到手就是坏的电机不转", "labels": ["质量问题"]}
{"text": "退货运费谁承担", "labels": ["运费"]}
{"text": "优惠券能叠加用吗", "labels": ["优惠活动"]}
{"text": "刚买就降价了能补差价吗", "labels": ["价保"]}
{"text": "付款的时候扣了两次钱", "labels": ["支付"]}
{"text": "订单还能取消吗", "labels": ["订单修改"]}
{"text": "智能猫砂盆什么时候补货", "labels": ["库存补货"]}
{"text": "冻干猫粮怎么保存", "labels": ["商品信息"]}
{"text": "猫爬架保修多久", "labels": ["保修维修"]}
{"text": "登录不上了忘了密码", "labels": ["账号"]}
{"text": "积分能抵钱吗", "labels": ["会员积分"]}
{"text": "追评在哪写", "labels": ["评价"]}
{"text": "你们客服几点上班", "labels": ["其他"]}
{"text": "俺买的那玩意儿咋还没到俺这疙瘩", "labels": ["物流"]}
{"text": "这个猫抓板我想退活", "labels": ["退换货"]}
{"text": "催下快递对了这个有色差还能换不", "labels": ["物流", "退换货"]}
{"text": "这猫窝买大了一码", "labels": ["尺码"]}
{"text": "猫别墅围栏断了能修吗", "labels": ["保修维修"]}
{"text": "想退了但退货要自己出邮费吗", "labels": ["退换货", "运费"]}
{"text": "发票抬头开错了能重开吗", "labels": ["发票"]}
{"text": "支持货到付款吗", "labels": ["支付"]}
{"text": "帮我改下收货地址", "labels": ["订单修改"]}
{"text": "猫碗是陶瓷的还是不锈钢的", "labels": ["商品信息"]}
{"text": "双十一有活动吗", "labels": ["优惠活动"]}
{"text": "怎么注销我的账号", "labels": ["账号"]}
{"text": "买的猫粮里有虫子必须退钱而且邮费你们出", "labels": ["质量问题", "退换货", "运费"]}
```

- [ ] **Step 3: 写 `scripts/ch10/validate_golden.py`**

```python
"""ch10 预标质量闸:黄金样例上标签集合完全一致率 ≥ 80% 才放行批量预标。
运行:make ch10-golden(需上游可用)。不过线就改 prompt,不改黄金样例来凑分。"""
import asyncio
import json
import pathlib
import sys

from scripts.ch10.prelabel import prelabel_batch

GOLDEN = pathlib.Path(__file__).parent / "golden_samples.jsonl"
PASS_RATE = 0.8


async def main() -> int:
    samples = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    predicted = await prelabel_batch([s["text"] for s in samples])
    hits = 0
    for s, pred in zip(samples, predicted):
        ok = set(pred) == set(s["labels"])
        hits += ok
        if not ok:
            print(f"✗ {s['text']}\n    标准: {s['labels']}  预标: {pred}")
    rate = hits / len(samples)
    print(f"\n黄金样例 {len(samples)} 条,集合全对 {hits} 条,通过率 {rate:.0%}(闸线 {PASS_RATE:.0%})")
    return 0 if rate >= PASS_RATE else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 4: Makefile 加目标(放在 ch09 目标块之后)**

```make
# ch10: 主题分类器(数据→训练→评测→ONNX→旁路批量归类)
ch10-golden:  ## 预标 prompt 黄金样例验证(通过率 ≥ 80% 才放行批量预标)
	PYTHONPATH=. uv run python scripts/ch10/validate_golden.py
```

- [ ] **Step 5: 验证跑(需上游可用;先确认 `.env` 里 `CHAT_*` 三项填好)**

Run: `make ch10-golden`
Expected: 通过率 ≥ 80%,exit 0。不过线:按错例改 PRELABEL_PROMPT(通常是近邻边界表述),重跑;**不许改黄金样例凑分**。连改三轮仍不过 → 停下问用户。

- [ ] **Step 6: Commit + dev-notes 追记(记录首轮通过率与错例)**

```bash
git add scripts/ch10/prelabel.py scripts/ch10/golden_samples.jsonl scripts/ch10/validate_golden.py Makefile
git commit -m "feat(ch10): LLM 预标模块 + 30 条黄金样例验证闸(≥80%)"
```

---

### Task 5: build_corpus 语料流水线 + 人工抽审(检查点)

**Files:**
- Create: `scripts/ch10/build_corpus.py`
- Modify: `Makefile`(加 ch10-corpus)

**Interfaces:**
- Consumes: `repository.list_pool_texts`、`corpus_lib.desensitize/dedupe`、`prelabel.prelabel_batch`、`taxonomy`
- Produces: `data/ch10/corpus_raw.jsonl`、`corpus_clean.jsonl`、`corpus_labeled.jsonl`(样本 dict: text/labels/origin[pool|simulated])、`data/ch10/sample_review.md`

- [ ] **Step 1: 写 `scripts/ch10/build_corpus.py`**

```python
"""ch10 语料流水线:捞池 → 脱敏去重 → LLM 修错别字 → LLM 预标 → 模拟补足 → 导出人工抽审。
运行:make ch10-corpus(需 mysql + 上游可用)。产物落 data/ch10/,抽审文件给用户对话里审。"""
import asyncio
import json
import pathlib
import random

from pydantic import BaseModel

from app.core.llm import get_chat_model
from app.core.taxonomy import LABEL2ID, TOPIC_CLASSES, terminology_table
from app.db import repository
from scripts.ch10.corpus_lib import dedupe, desensitize
from scripts.ch10.prelabel import prelabel_batch

OUT = pathlib.Path("data/ch10")
TARGET_PER_CLASS = 100
SIM_BATCH = 20     # 单次造数条数,小批多次控质量

CLEAN_PROMPT = """修正下面这句用户问题里的错别字和乱格式:不改语义、不改口语风格、不增删诉求,
没有错误就原样返回。只输出句子本身。

{text}"""


class _SimItem(BaseModel):
    text: str
    labels: list[str]


class _SimBatch(BaseModel):
    items: list[_SimItem]


SIMULATE_PROMPT = """你是电商客服语料造数员。为猫用品电商「喵喵优选」(卖猫粮、冻干、猫零食、猫砂盆、\
猫抓板、猫窝、猫爬架、猫碗、逗猫棒、项圈等)生成 {n} 条模拟用户问题,全部命中主题类目「{name}」。

17 类权威术语表:
{terminology}

要求:
1. 每条一句独立、语义完整的口语化用户问题,长短语气多样,贴近真实客服提问,彼此不重复。
2. 大约 15% 用方言土话(如「俺买的那玩意儿咋还没到俺这疙瘩」),10% 故意带错别字(如「退活」=退货),\
15% 是多诉求句——除「{name}」外再字面提到一个其他类目的诉求,labels 把两个类都打上。
3. 其余条目只含「{name}」一个诉求,labels 只有它。
4. 标签铁律:字面提到几个诉求打几个标签,一个不多一个不少;标签取术语表类目名原文。"""


def _dump(path: pathlib.Path, samples: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in samples),
                    encoding="utf-8")


async def clean_texts(texts: list[str], concurrency: int = 8) -> list[str]:
    model = get_chat_model()
    sem = asyncio.Semaphore(concurrency)

    async def one(t: str) -> str:
        async with sem:
            try:
                r = await model.ainvoke(CLEAN_PROMPT.format(text=t))
                out = r.content.strip()
                return out or t
            except Exception:
                return t

    return list(await asyncio.gather(*[one(t) for t in texts]))


async def simulate(name: str, need: int) -> list[dict]:
    model = get_chat_model().with_structured_output(_SimBatch)
    out: list[dict] = []
    misses = 0
    while len(out) < need and misses < 5:
        n = min(SIM_BATCH, need - len(out))
        try:
            r: _SimBatch = await model.ainvoke(SIMULATE_PROMPT.format(
                n=n, name=name, terminology=terminology_table()))
        except Exception:
            misses += 1
            continue
        got = 0
        for it in r.items:
            labels = [lb for lb in it.labels if lb in LABEL2ID]
            if name not in labels:
                labels = [name] + labels
            if it.text.strip():
                out.append({"text": it.text.strip(), "labels": labels, "origin": "simulated"})
                got += 1
        misses = misses + 1 if got == 0 else 0
    return out[:need]


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # 1) 捞池(标准化问法优先)
    pool = await repository.list_pool_texts()
    raw = [{"text": p["text"], "labels": [], "origin": "pool"} for p in pool]
    _dump(OUT / "corpus_raw.jsonl", raw)
    print(f"捞池 {len(raw)} 条")
    # 2) 清洗:脱敏 → 去重 → LLM 修错别字 → 再去重
    cleaned = dedupe([{**s, "text": desensitize(s["text"])} for s in raw])
    fixed = await clean_texts([s["text"] for s in cleaned])
    cleaned = dedupe([{**s, "text": t} for s, t in zip(cleaned, fixed)])
    _dump(OUT / "corpus_clean.jsonl", cleaned)
    print(f"清洗后 {len(cleaned)} 条")
    # 3) 预标真实问题
    labels = await prelabel_batch([s["text"] for s in cleaned])
    labeled = [{**s, "labels": lb} for s, lb in zip(cleaned, labels)]
    # 4) 模拟补足:按标签计数补到每类 TARGET_PER_CLASS(多诉求句给命中的每类都记数)
    counts = {c.name: 0 for c in TOPIC_CLASSES}
    for s in labeled:
        for lb in s["labels"]:
            counts[lb] += 1
    for c in TOPIC_CLASSES:
        need = TARGET_PER_CLASS - counts[c.name]
        if need <= 0:
            continue
        sims = await simulate(c.name, need)
        labeled.extend(sims)
        for s in sims:
            for lb in s["labels"]:
                counts[lb] += 1
        print(f"{c.name}: 补造 {len(sims)} 条(当前 {counts[c.name]})")
    labeled = dedupe(labeled)
    _dump(OUT / "corpus_labeled.jsonl", labeled)
    print(f"语料总量 {len(labeled)} 条;各类:{counts}")
    # 5) 抽审导出:真实池全量 + 每类模拟抽 5
    rng = random.Random(42)
    lines = ["# ch10 语料人工抽审(预标 + 模拟)", "",
             "> 格式:问题 → 标签。看到错标直接指出行号或原句。", "",
             "## 真实池问题(全量)", ""]
    for s in labeled:
        if s["origin"] == "pool":
            lines.append(f"- {s['text']} → {'、'.join(s['labels'])}")
    lines += ["", "## 模拟问题(每类抽 5)", ""]
    for c in TOPIC_CLASSES:
        sims = [s for s in labeled if s["origin"] == "simulated" and c.name in s["labels"]]
        lines.append(f"### {c.name}")
        lines += [f"- {s['text']} → {'、'.join(s['labels'])}"
                  for s in rng.sample(sims, min(5, len(sims)))]
        lines.append("")
    (OUT / "sample_review.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"抽审文件已导出:{OUT / 'sample_review.md'}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Makefile 加目标**

```make
ch10-corpus:  ## 语料流水线:捞池→清洗→预标→模拟补足→导出人工抽审(需 mysql + 上游可用)
	PYTHONPATH=. uv run python scripts/ch10/build_corpus.py
```

- [ ] **Step 3: 跑流水线**

Run: `make ch10-corpus`
Expected: 逐类打印补造进度;`data/ch10/corpus_labeled.jsonl` 约 1700+ 条;`sample_review.md` 生成

- [ ] **Step 4: 【检查点·人工抽审】把 sample_review.md 内容贴给用户在对话里审,等改标意见**

用户指出错标的:直接改 `corpus_labeled.jsonl` 对应行(以及黄金样例若同错),记录进 dev-notes。**没过这个检查点不许进 Task 6。**

- [ ] **Step 5: Commit + dev-notes 追记(记录语料构成:真实 N 条/模拟 M 条、抽审改了几条)**

```bash
git add scripts/ch10/build_corpus.py Makefile
git commit -m "feat(ch10): 语料流水线——捞池/清洗/预标/模拟补足/抽审导出"
```

---

### Task 6: build_dataset 分层划分 + 训练集增强

**Files:**
- Create: `scripts/ch10/build_dataset.py`
- Modify: `Makefile`(加 ch10-dataset)

**Interfaces:**
- Consumes: `corpus_lib.split_dataset`、`data/ch10/corpus_labeled.jsonl`
- Produces: `data/ch10/dataset/{train,val,test}.jsonl`(每行 `{"text","labels"}`;train 含 origin=augmented 变体,val/test 保持原样)

- [ ] **Step 1: 写 `scripts/ch10/build_dataset.py`**

```python
"""ch10 数据集:分层划分 80/10/10 + 训练集增强(同义词替换/句式微调)。
增强只扩训练集——验证/测试是考题,不许照练习题变。运行:make ch10-dataset(需上游可用)。"""
import asyncio
import json
import pathlib

from app.core.llm import get_chat_model
from app.core.taxonomy import TOPIC_NAMES
from scripts.ch10.corpus_lib import split_dataset

SRC = pathlib.Path("data/ch10/corpus_labeled.jsonl")
OUT = pathlib.Path("data/ch10/dataset")

AUGMENT_PROMPT = """把下面这句电商客服用户问题改写一个变体:换同义词、微调句式(比如改成「我想问一下……」的口气),
不改原意、不增删诉求。只输出改写后的句子。

{text}"""


async def augment(samples: list[dict], concurrency: int = 8) -> list[dict]:
    model = get_chat_model()
    sem = asyncio.Semaphore(concurrency)

    async def one(s: dict) -> dict | None:
        async with sem:
            try:
                r = await model.ainvoke(AUGMENT_PROMPT.format(text=s["text"]))
                t = r.content.strip()
            except Exception:
                return None
        return {"text": t, "labels": s["labels"], "origin": "augmented"} if t else None

    outs = await asyncio.gather(*[one(s) for s in samples])
    return [o for o in outs if o]


def _dump(path: pathlib.Path, samples: list[dict]) -> None:
    path.write_text("\n".join(
        json.dumps({"text": s["text"], "labels": s["labels"]}, ensure_ascii=False)
        for s in samples), encoding="utf-8")


def _dist(name: str, samples: list[dict]) -> None:
    counts = {n: 0 for n in TOPIC_NAMES}
    for s in samples:
        for lb in s["labels"]:
            counts[lb] += 1
    print(f"{name}({len(samples)} 条): " + " ".join(f"{k}={v}" for k, v in counts.items()))


async def main() -> None:
    samples = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    train, val, test = split_dataset(samples)
    aug = await augment(train)
    seen = {s["text"] for s in samples}          # 变体撞上任何原句(含考题)就丢弃
    train = train + [a for a in aug if a["text"] not in seen]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, ds in (("train", train), ("val", val), ("test", test)):
        _dump(OUT / f"{name}.jsonl", ds)
        _dist(name, ds)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Makefile 加目标**

```make
ch10-dataset:  ## 分层划分 80/10/10 + 训练集增强(需上游可用)
	PYTHONPATH=. uv run python scripts/ch10/build_dataset.py
```

- [ ] **Step 3: 跑并检查分布表(验证跑)**

Run: `make ch10-dataset`
Expected: 三份分布表打印;每类目在 train/val/test 都非零;train 条数约为划分后 2 倍(增强翻倍)

- [ ] **Step 4: Commit + dev-notes 追记(附三份分布数字)**

```bash
git add scripts/ch10/build_dataset.py Makefile
git commit -m "feat(ch10): 数据集划分 + 训练集同义句增强"
```

---

### Task 7: train.py 全参微调(验证跑:早停 + 验证集分数)

**Files:**
- Create: `scripts/ch10/train.py`
- Modify: `Makefile`(加 ch10-train)

**Interfaces:**
- Consumes: `data/ch10/dataset/{train,val}.jsonl`、taxonomy
- Produces: `data/ch10/model/`(HF 格式权重 + tokenizer + `threshold.json` = `{"threshold": float, "val_micro_f1": float}`)

- [ ] **Step 1: 写 `scripts/ch10/train.py`**

```python
"""ch10 训练:RoBERTa-wwm-ext 全参微调,17 类多标签(BCEWithLogitsLoss)。
运行:make ch10-train。设备自适应 cuda→mps→cpu;正则化 weight_decay + 早停盯验证集 micro-F1。
HF 下载不通时:HF_ENDPOINT=https://hf-mirror.com make ch10-train。"""
import json
import pathlib

import numpy as np
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, Trainer, TrainingArguments,
                          default_data_collator)

from app.core.taxonomy import ID2LABEL, LABEL2ID, NUM_CLASSES

BASE = "hfl/chinese-roberta-wwm-ext"
DATA = pathlib.Path("data/ch10/dataset")
OUT = pathlib.Path("data/ch10/model")


def load_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def encode(samples: list[dict], tokenizer) -> list[dict]:
    enc = tokenizer([s["text"] for s in samples], truncation=True,
                    padding="max_length", max_length=128)
    items = []
    for i, s in enumerate(samples):
        vec = [0.0] * NUM_CLASSES                 # float 向量:BCEWithLogitsLoss 要求
        for lb in s["labels"]:
            vec[LABEL2ID[lb]] = 1.0
        items.append({"input_ids": enc["input_ids"][i],
                      "attention_mask": enc["attention_mask"][i],
                      "token_type_ids": enc["token_type_ids"][i],
                      "labels": vec})
    return items


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (1 / (1 + np.exp(-logits)) >= 0.5).astype(int)
    return {"micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0)}


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=NUM_CLASSES, problem_type="multi_label_classification",
        id2label=ID2LABEL, label2id=LABEL2ID)
    train_ds = encode(load_jsonl(DATA / "train.jsonl"), tokenizer)
    val_ds = encode(load_jsonl(DATA / "val.jsonl"), tokenizer)
    args = TrainingArguments(
        output_dir="data/ch10/checkpoints",
        eval_strategy="epoch",                    # v5 参数名,不是 evaluation_strategy
        save_strategy="epoch",                    # load_best_model_at_end 要求与 eval 一致
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=8,
        weight_decay=0.01,                        # 正则化防过拟合
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=20,
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tokenizer,               # v5:tokenizer= 已改名
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    # 验证集上扫全局最优阈值(0.30~0.70 步进 0.05)
    logits = trainer.predict(val_ds).predictions
    probs = 1 / (1 + np.exp(-logits))
    gold = np.array([d["labels"] for d in val_ds])
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.30, 0.71, 0.05):
        f1 = f1_score(gold, (probs >= t).astype(int), average="micro", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = round(float(t), 2), float(f1)
    OUT.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUT)
    tokenizer.save_pretrained(OUT)
    (OUT / "threshold.json").write_text(
        json.dumps({"threshold": best_t, "val_micro_f1": best_f1}))
    print(f"最优阈值 {best_t},验证集 micro-F1 {best_f1:.4f};模型已存 {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Makefile 加目标**

```make
ch10-train:  ## RoBERTa-wwm-ext 全参微调(MPS/CUDA/CPU 自适应,重依赖走 ml 组)
	PYTHONPATH=. uv run --group ml python scripts/ch10/train.py
```

- [ ] **Step 3: 跑训练(验证跑)**

Run: `make ch10-train`
Expected: 每 epoch 打印验证集 micro_f1/macro_f1;早停或跑满 8 epoch 收手;末尾打印最优阈值与验证 F1;`data/ch10/model/threshold.json` 存在。若 MPS 报个别算子不支持:`PYTORCH_ENABLE_MPS_FALLBACK=1` 重跑(算子回退 CPU 属正常);若下载 hfl 模型超时:`HF_ENDPOINT=https://hf-mirror.com` 重跑。

- [ ] **Step 4: Commit + dev-notes 追记(记录训练时长、早停轮次、验证 F1、阈值)**

```bash
git add scripts/ch10/train.py Makefile
git commit -m "feat(ch10): RoBERTa-wwm-ext 全参微调训练脚本(正则化+早停+阈值扫描)"
```

---

### Task 8: evaluate.py 评测报告(验收标准 1)

**Files:**
- Create: `scripts/ch10/evaluate.py`
- Modify: `Makefile`(加 ch10-eval)

**Interfaces:**
- Consumes: `data/ch10/model/`、`data/ch10/dataset/test.jsonl`、`taxonomy.SEVERITY/TOPIC_NAMES`
- Produces: `data/ch10/reports/eval_report.md`(每类 P/R/F1+support、micro/macro、每类混淆矩阵、容错红线)、`data/ch10/reports/error_samples.md`(判错样本供人工复核)

- [ ] **Step 1: 写 `scripts/ch10/evaluate.py`**

```python
"""ch10 评测:留出测试集上每类 P/R/F1 + 每类混淆矩阵 + 容错红线 + 判错样本导出。
运行:make ch10-eval。评测集扎在自家电商场景(dataset/test.jsonl),不引公开榜单。"""
import json
import pathlib

import numpy as np
import torch
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.taxonomy import LABEL2ID, NUM_CLASSES, SEVERITY, TOPIC_NAMES

MODEL_DIR = pathlib.Path("data/ch10/model")
TEST = pathlib.Path("data/ch10/dataset/test.jsonl")
REPORTS = pathlib.Path("data/ch10/reports")
RED_LINE = 0.8   # 严档类目 F1 红线


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def predict(model, tokenizer, texts: list[str], threshold: float, device: str) -> np.ndarray:
    model.eval()
    probs_all = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tokenizer(texts[i:i + 32], truncation=True, padding=True,
                            max_length=128, return_tensors="pt").to(device)
            probs_all.append(torch.sigmoid(model(**enc).logits).cpu().numpy())
    probs = np.concatenate(probs_all)
    preds = (probs >= threshold).astype(int)
    for r in range(len(preds)):                    # 全不过线取最高分兜底,不产出空标签
        if preds[r].sum() == 0:
            preds[r][probs[r].argmax()] = 1
    return preds


def main() -> None:
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    threshold = json.loads((MODEL_DIR / "threshold.json").read_text())["threshold"]
    samples = [json.loads(l) for l in TEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = [s["text"] for s in samples]
    gold = np.zeros((len(samples), NUM_CLASSES), dtype=int)
    for i, s in enumerate(samples):
        for lb in s["labels"]:
            gold[i][LABEL2ID[lb]] = 1
    preds = predict(model, tokenizer, texts, threshold, device)

    p, r, f1, support = precision_recall_fscore_support(gold, preds, zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        gold, preds, average="micro", zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        gold, preds, average="macro", zero_division=0)
    cms = multilabel_confusion_matrix(gold, preds)

    REPORTS.mkdir(parents=True, exist_ok=True)
    lines = ["# ch10 分类器评测报告(留出测试集)", "",
             f"测试集 {len(samples)} 条;判定阈值 {threshold}(验证集扫描所得)。", "",
             f"**micro**: P={micro_p:.3f} R={micro_r:.3f} F1={micro_f1:.3f}  |  "
             f"**macro**: P={macro_p:.3f} R={macro_r:.3f} F1={macro_f1:.3f}", "",
             "## 每类指标(容错档位:严档 F1 红线 0.8)", "",
             "| 类目 | 档位 | P | R | F1 | support | 红线 |",
             "|---|---|---|---|---|---|---|"]
    for i, name in enumerate(TOPIC_NAMES):
        sev = SEVERITY[name]
        flag = "🔴 不达标,先回头搞数据" if sev == "严" and f1[i] < RED_LINE else "✅"
        lines.append(f"| {name} | {sev} | {p[i]:.3f} | {r[i]:.3f} | {f1[i]:.3f} "
                     f"| {int(support[i])} | {flag} |")
    lines += ["", "## 每类混淆矩阵(TN FP / FN TP)", ""]
    for i, name in enumerate(TOPIC_NAMES):
        tn, fp = cms[i][0]
        fn, tp = cms[i][1]
        lines.append(f"- **{name}**: TN={tn} FP={fp} FN={fn} TP={tp}")
    (REPORTS / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")

    err = ["# ch10 判错样本(人工复核:错在哪一类?标注本身有没有毛病?)", ""]
    for i, s in enumerate(samples):
        pred_labels = [TOPIC_NAMES[j] for j in range(NUM_CLASSES) if preds[i][j]]
        if set(pred_labels) != set(s["labels"]):
            err.append(f"- {s['text']}\n  标准: {s['labels']}  预测: {pred_labels}")
    (REPORTS / "error_samples.md").write_text("\n".join(err), encoding="utf-8")
    print(f"micro-F1 {micro_f1:.4f} / macro-F1 {macro_f1:.4f};"
          f"报告与判错样本已落 {REPORTS}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Makefile 加目标**

```make
ch10-eval:  ## 测试集评测:每类 P/R/F1 + 混淆矩阵 + 容错红线 + 判错样本导出
	PYTHONPATH=. uv run --group ml python scripts/ch10/evaluate.py
```

- [ ] **Step 3: 跑评测(验证跑 + 人工复核判错样本)**

Run: `make ch10-eval`
Expected: 报告两文件生成。把 error_samples.md 里判错样本过一遍:若集中在某类且是标注毛病 → 回 Task 5 改标重训(README:效果不行先搞数据);严档类目 F1 低于 0.8 同样回数据。结论记 dev-notes。

- [ ] **Step 4: Commit + dev-notes 追记(附 micro/macro F1 和红线达标情况)**

```bash
git add scripts/ch10/evaluate.py Makefile data/ch10/reports/
git commit -m "feat(ch10): 评测脚本——每类 PRF1/混淆矩阵/容错红线/判错样本导出"
```

---

### Task 9: export_onnx.py 导出 + 一致性校验

**Files:**
- Create: `scripts/ch10/export_onnx.py`
- Modify: `Makefile`(加 ch10-export)

**Interfaces:**
- Consumes: `data/ch10/model/`、`data/ch10/dataset/test.jsonl`
- Produces: `data/ch10/onnx/model.onnx` + `tokenizer.json` + `threshold.json`(服务只吃这个目录)

- [ ] **Step 1: 写 `scripts/ch10/export_onnx.py`**

```python
"""ch10 ONNX 导出:torch 模型 → onnx(变长 batch/seq),导出后跑测试集校验与 torch 预测完全一致。
注意:torch 2.9+ 的 torch.onnx.export 默认 dynamo=True,导 HF 模型此处显式 dynamo=False
走 TorchScript 导出器 + dynamic_axes 稳定路线(Context7 查证)。运行:make ch10-export。"""
import json
import pathlib
import shutil

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = pathlib.Path("data/ch10/model")
OUT = pathlib.Path("data/ch10/onnx")
TEST = pathlib.Path("data/ch10/dataset/test.jsonl")


class LogitsWrapper(torch.nn.Module):
    """HF 模型输出是 ModelOutput dict,导出前包一层只回 logits 张量。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).logits


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    wrapper = LogitsWrapper(model)
    sample = tokenizer(["买大了想退", "快递到哪了"], padding=True, return_tensors="pt")
    dyn = {0: "batch", 1: "seq"}
    torch.onnx.export(
        wrapper,
        (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]),
        str(OUT / "model.onnx"),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": dyn, "attention_mask": dyn,
                      "token_type_ids": dyn, "logits": {0: "batch"}},
        dynamo=False,
        opset_version=17,
    )
    tokenizer.save_pretrained(OUT)          # 带出 tokenizer.json 给轻运行时用
    shutil.copy(MODEL_DIR / "threshold.json", OUT / "threshold.json")

    # 一致性校验:测试集全量,ONNX 与 torch 的过线标签必须完全一致
    threshold = json.loads((OUT / "threshold.json").read_text())["threshold"]
    texts = [json.loads(l)["text"]
             for l in TEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    sess = ort.InferenceSession(str(OUT / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    mismatch = 0
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tokenizer(texts[i:i + 32], truncation=True, padding=True,
                            max_length=128, return_tensors="pt")
            t_logits = wrapper(**enc).numpy()
            (o_logits,) = sess.run(["logits"], {k: v.numpy() for k, v in enc.items()})
            assert np.allclose(t_logits, o_logits, atol=1e-3), "logits 超容差"
            t_pred = (1 / (1 + np.exp(-t_logits)) >= threshold)
            o_pred = (1 / (1 + np.exp(-o_logits)) >= threshold)
            mismatch += int((t_pred != o_pred).any(axis=1).sum())
    if mismatch:
        raise SystemExit(f"ONNX 与 torch 预测不一致 {mismatch} 条,导出失败")
    print(f"ONNX 导出并校验通过({len(texts)} 条预测完全一致):{OUT}/model.onnx")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Makefile 加目标**

```make
ch10-export:  ## 导出 ONNX 并校验与 torch 预测完全一致
	PYTHONPATH=. uv run --group ml python scripts/ch10/export_onnx.py
```

- [ ] **Step 3: 跑导出(验证跑)**

Run: `make ch10-export`
Expected: 「ONNX 导出并校验通过」;`data/ch10/onnx/` 下有 model.onnx / tokenizer.json / threshold.json

- [ ] **Step 4: Commit + dev-notes 追记**

```bash
git add scripts/ch10/export_onnx.py Makefile
git commit -m "feat(ch10): ONNX 导出 + torch 预测一致性校验"
```

---

### Task 10: serve.py 推理服务 :8110 + make 起停

**Files:**
- Create: `scripts/ch10/serve.py`
- Modify: `Makefile`(加 classifier-up / classifier-down)

**Interfaces:**
- Consumes: `data/ch10/onnx/`、`taxonomy.TOPIC_NAMES`
- Produces: HTTP `POST /classify` `{"texts": [...]}` → `{"results": [{"labels": [...], "scores": {类目: 分数}}]}`;`GET /healthz` → `{"ok": true}`

- [ ] **Step 1: 写 `scripts/ch10/serve.py`**

```python
"""ch10 推理服务:ONNX + FastAPI 独立进程 :8110,轻运行时(onnxruntime + tokenizers,不背 torch)。
起停:make classifier-up / classifier-down(仓库惯例 nohup + pid 文件)。"""
import json
import pathlib

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from tokenizers import Tokenizer

from app.core.taxonomy import TOPIC_NAMES

DIR = pathlib.Path("data/ch10/onnx")

app = FastAPI(title="mewhelp ch10 topic classifier")
_sess = ort.InferenceSession(str(DIR / "model.onnx"), providers=["CPUExecutionProvider"])
_tok = Tokenizer.from_file(str(DIR / "tokenizer.json"))
_tok.enable_truncation(max_length=128)
_tok.enable_padding(pad_id=0, pad_token="[PAD]")
_threshold: float = json.loads((DIR / "threshold.json").read_text())["threshold"]


class ClassifyIn(BaseModel):
    texts: list[str]


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/classify")
def classify(body: ClassifyIn):
    if not body.texts:
        return {"results": []}
    encs = _tok.encode_batch(body.texts)
    feed = {
        "input_ids": np.array([e.ids for e in encs], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in encs], dtype=np.int64),
        "token_type_ids": np.array([e.type_ids for e in encs], dtype=np.int64),
    }
    (logits,) = _sess.run(["logits"], feed)
    probs = 1 / (1 + np.exp(-logits))
    results = []
    for row in probs:
        labels = [TOPIC_NAMES[i] for i, p in enumerate(row) if p >= _threshold]
        if not labels:                       # 全不过线取最高分兜底,不产出空标签
            labels = [TOPIC_NAMES[int(row.argmax())]]
        results.append({"labels": labels,
                        "scores": {TOPIC_NAMES[i]: round(float(p), 4)
                                   for i, p in enumerate(row)}})
    return {"results": results}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8110)
```

- [ ] **Step 2: Makefile 加起停目标(照 mcp-up/down 惯例)**

```make
classifier-up:  ## ch10 推理服务 :8110(ONNX 轻运行时)
	@mkdir -p log data
	@nohup uv run --group ml python scripts/ch10/serve.py > log/classifier.log 2>&1 & echo $$! > data/classifier.pid
	@sleep 2 && curl -sf http://127.0.0.1:8110/healthz >/dev/null && echo "分类器服务已拉起: :8110(pid 见 data/classifier.pid)" || echo "启动失败,看 log/classifier.log"

classifier-down:
	-@kill `cat data/classifier.pid 2>/dev/null` 2>/dev/null; rm -f data/classifier.pid
	@echo "分类器服务已停"
```

注:PYTHONPATH 由脚本内相对 data/ 路径决定必须在仓库根跑;serve 只 import `app.core.taxonomy`(无 settings 依赖),`uv run` 自动带上仓库根为工作目录,nohup 行首加 `PYTHONPATH=.` 保险:`@PYTHONPATH=. nohup uv run --group ml python scripts/ch10/serve.py ...`。

- [ ] **Step 3: 起服务并验收多标签单句(验收标准 3 预演)**

Run:
```bash
make classifier-up
curl -s -X POST http://127.0.0.1:8110/classify -H 'Content-Type: application/json' \
  -d '{"texts": ["买大了想退", "我的快递到哪了", "能开发票吗顺便问下退货运费谁出"]}' | python3 -m json.tool
```
Expected: 第 1 条 labels 同时含「尺码」「退换货」;第 2 条「物流」;第 3 条含「发票」「运费」(±退换货看字面)。不对 → 回 Task 8 看该类指标,通常是数据问题。

- [ ] **Step 4: Commit + dev-notes 追记(附 curl 实测输出)**

```bash
git add scripts/ch10/serve.py Makefile
git commit -m "feat(ch10): ONNX 推理服务 :8110(/classify 批量 + /healthz)"
```

---

### Task 11: classify_pool.py 旁路批处理(验收标准 2 前半)

**Files:**
- Create: `scripts/ch10/classify_pool.py`
- Modify: `Makefile`(加 classify-pool)

**Interfaces:**
- Consumes: `repository.list_unclassified_questions/insert_topic_classifications`、服务 :8110
- Produces: `topic_classifications` 表数据;CLI 输出归类统计

- [ ] **Step 1: 写 `scripts/ch10/classify_pool.py`**

```python
"""ch10 旁路批处理:低置信度问题攒够一批,整批喂分类器归一次类,结果写 topic_classifications。
实时对话主链路不调它。运行:make classify-pool(需 mysql + 分类器服务 :8110)。
幂等:已归类的行(LEFT JOIN 命中)不重复归。定时跑 cron 示例:
  0 3 * * * cd /path/to/mewhelp && make classify-pool >> log/classify-pool.log 2>&1"""
import argparse
import asyncio

import httpx

from app.db import repository

SERVICE = "http://127.0.0.1:8110"


async def main(min_batch: int, force: bool) -> None:
    rows = await repository.list_unclassified_questions(limit=500)
    if not rows:
        print("池里没有待归类问题")
        return
    if len(rows) < min_batch and not force:
        print(f"待归类 {len(rows)} 条,不足一批({min_batch} 条);--force 可强跑")
        return
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{SERVICE}/classify",
                              json={"texts": [x["text"] for x in rows]})
        r.raise_for_status()
        results = r.json()["results"]
    n = await repository.insert_topic_classifications(
        [{"question_id": x["question_id"], "labels": res["labels"]}
         for x, res in zip(rows, results)])
    counts: dict[str, int] = {}
    for res in results:
        for lb in res["labels"]:
            counts[lb] = counts.get(lb, 0) + 1
    print(f"归类完成:{n} 条写入 topic_classifications;"
          + " ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-batch", type=int, default=10, help="攒够多少条才归一次")
    ap.add_argument("--force", action="store_true", help="不足一批也强跑")
    a = ap.parse_args()
    asyncio.run(main(a.min_batch, a.force))
```

- [ ] **Step 2: Makefile 加目标**

```make
classify-pool:  ## ch10 旁路批量归类:攒够一批归一次,写 topic_classifications(需 :8110)
	PYTHONPATH=. uv run python scripts/ch10/classify_pool.py
```

- [ ] **Step 3: 真实池跑一遍 + 幂等验证**

Run:
```bash
make classify-pool
mysql -uroot -proot -h127.0.0.1 mewhelp -e "SELECT COUNT(*) FROM topic_classifications;"
make classify-pool   # 第二遍:应报「池里没有待归类问题」,行数不变
```
Expected: 首遍写入约 70 条(全池);二遍幂等不重写

- [ ] **Step 4: Commit + dev-notes 追记(附真实池归类分布输出)**

```bash
git add scripts/ch10/classify_pool.py Makefile
git commit -m "feat(ch10): 旁路批处理——攒批归类写 topic_classifications,幂等"
```

---

### Task 12: 主题分布与类目问题列表 API(TDD)

**Files:**
- Create: `app/api/topics.py`
- Modify: `app/main.py`(import + include_router,照 review_router 惯例)、`app/db/repository.py`(加 `topic_questions`)
- Test: `tests/test_topics_api.py`

**Interfaces:**
- Consumes: `repository.topic_distribution`、`repository.topic_questions`
- Produces: `GET /api/topics/distribution` → `{"total": int, "latest": str|null, "classes": [{"label","count","samples"}] }`(17 类全出,顺序同 TOPIC_NAMES);`GET /api/topics/questions?label=&page=&size=` → `{label,total,page,pages,size,items[]}`

- [ ] **Step 1: 写失败测试 `tests/test_topics_api.py`(照 tests/test_review_api.py 的 client/种子写法对齐)**

```python
from app.db import repository
from app.db.models import LowConfidenceQuestion


async def test_distribution_empty_returns_all_17(client):
    resp = await client.get("/api/topics/distribution")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert len(body["classes"]) == 17


async def test_distribution_counts(client, db_session_maker):
    async with db_session_maker() as s:
        q = LowConfidenceQuestion(raw_question="猫窝买大了想退", source="retrieval_low_conf")
        s.add(q)
        await s.commit()
        qid = q.id
    await repository.insert_topic_classifications(
        [{"question_id": qid, "labels": ["尺码", "退换货"]}])
    resp = await client.get("/api/topics/distribution")
    body = resp.json()
    by_label = {c["label"]: c for c in body["classes"]}
    assert body["total"] == 1
    assert by_label["尺码"]["count"] == 1 and by_label["退换货"]["count"] == 1
    assert by_label["物流"]["count"] == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. uv run pytest tests/test_topics_api.py -v`
Expected: FAIL(404,路由不存在)

- [ ] **Step 3: 写 `app/api/topics.py` 并注册**

```python
"""ch10 主题分布 API:飞轮后台看各类目问题量,决定先补哪块知识。只读。"""
from fastapi import APIRouter

from app.db import repository

router = APIRouter()


@router.get("/api/topics/distribution")
async def distribution():
    return await repository.topic_distribution()
```

`app/main.py` 照既有 import 惯例加:

```python
from app.api.topics import router as topics_router
...
app.include_router(topics_router)
```

- [ ] **Step 4: 类目问题列表端点(先测后写)**

分布图只回答「哪类堆得多」,补知识还得看这一类具体堆的是什么。RED 先加四条测试:①一类下分页,`total`/`pages` 按这一类算而不是全表条数;②多标签问题在它命中的每个类目下都出现且带上同伴类目;③没归并的显示原话、`normalized=false`、审核状态为空;④归并过的显示标准化问法、另给一份原话、审核状态跟着 `review_queue`;⑤类目名不在权威表里回 400。另补一条:分布的每类样例按文本去重(归并后同一句问法对应池里好几行,三条样例都一样就白给了,计数不受影响)。

GREEN 在 `repository.topic_questions(label, page, size)` 里实现:一次捞出归类结果连同池行与队列行,`labels` 命中判断、分页都在 Python 侧做(归类总量是问题池规模,百级;换来的是不依赖 MySQL 的 JSON 函数,测试库上结果一致)。`app/api/topics.py` 加 `questions` 端点,`label` 先对 `TOPIC_NAMES` 校验。

- [ ] **Step 5: 跑测试确认通过 + 全量回归**

Run: `PYTHONPATH=. uv run pytest tests/test_topics_api.py tests -x -q`
Expected: 新增 8 passed,全量不回归

- [ ] **Step 6: Commit + dev-notes 追记**

```bash
git add app/api/topics.py app/db/repository.py app/main.py tests/test_topics_api.py
git commit -m "feat(ch10): 主题分布 API + 类目问题列表分页 API"
```

---

### Task 13: 主题分布页与类目问题列表页(Vibe Coding)+ 挂进后台导航

**Files:**
- Create: `app/static/topics.html`、`app/static/topic-questions.html`
- Modify: `app/static/admin.js`(导航注册表加「主题分布」一行)、`app/main.py`(两条静态路由)

**流程例外:** 本任务走 Vibe Coding——先出第一版,起服务截图给用户看,用户描述效果直接改,不套 TDD/code review。

- [ ] **Step 1: 写第一版 `app/static/topics.html`**

与 review.html 同款像素风(复用其 :root 配色变量、边框阴影、topbar 结构;顶栏标题「喵喵优选 · 主题分布」)。主体:
- 顶部统计条:总归类问题数、最近归类时间;
- 17 类横向条形图:纯 HTML/CSS(每行:类目名 + 按 count/max 百分比宽度的色条 + 计数徽章),count 为 0 的类目淡显;按 count 降序排;
- 每行可点开(details/summary)露出该类 samples 样例问题;
- 类目名做成链接,点它进 `/topics/questions?label=<类目>`;展开的样例框底下再放一个「查看全部 N 条」入口;
- 数据来自 `fetch('/api/topics/distribution')`,无外部图表库、无外部依赖。

同页系另写一版 `app/static/topic-questions.html`(路由 `/topics/questions`):一行一条问题,标准化问法在上、用户原话在下,旁边摆命中的类目标签、来源、同义合并条数、审核状态与归类时间;顶部一排类目按问题量排开,当前类高亮,点别的类直接换;底部上一页/下一页。类目与页码都走地址栏,这一页贴给别人也能打开,刷新回到同一页。

跨页入口不在本页手写:引 `app/static/admin.js`、`mountAdminNav('/topics')`,共用导航自己摆在顶栏底下,注册表里加上本页一行就行。往后每加一块后台,加的也是这一行,不必回头改各页顶栏。

- [ ] **Step 2: 起主应用看效果并截图给用户**

Run: `make dev`(或按仓库惯例起 uvicorn),浏览器开 `http://localhost:8000/static/topics.html`,用 chrome-devtools 截紧凑图贴给用户(用户偏好:紧凑截图)。

- [ ] **Step 3: 【检查点·Vibe 循环】按用户描述改到满意为止**

- [ ] **Step 4: Commit + dev-notes 追记(记录用户改了几轮、都改了什么)**

```bash
git add app/static/topics.html app/static/topic-questions.html app/static/admin.js app/main.py
git commit -m "feat(ch10): 飞轮后台主题分布页(像素风条形图)+ 类目问题列表页 + 挂进后台导航"
```

---

### Task 14: 结构化验收产物(各脚本 .md + .json 一式两份)

**Files:**
- Modify: `scripts/ch10/validate_golden.py`、`scripts/ch10/evaluate.py`、`scripts/ch10/export_onnx.py`、`scripts/ch10/classify_pool.py`
- Create: `scripts/ch10/scan_threshold_replay.py`
- Modify: `Makefile`(加 `ch10-threshold-scan`;`classify-pool` 支持 `FORCE=1`)

**Interfaces:**
- Produces(全部落 `data/ch10/reports/`,验收 API 的唯一数据来源):
  - `golden_report.json` → `{ran_at, total, hits, rate, pass_line, passed, failures:[{text,gold,pred}]}`
  - `eval_report.json` → `{ran_at, test_size, threshold, red_lines, micro:{p,r,f1}, macro:{...}, classes:[{name,severity,p,r,f1,support,red_line,passed,tn,fp,fn,tp}], errors:[{text,gold,pred,missed,extra,kind,matrix_entries}], total_cells, total_fp, total_fn, red_line_passed}`
  - `threshold_scan.json` → `{ran_at, val_size, scan:[{threshold,micro_f1,tp,fp,fn}], best_threshold, best_micro_f1, in_use_threshold, consistent}`
  - `export_report.json` → `{ran_at, checked, mismatch, passed, onnx_path, onnx_bytes, opset}`
  - `classify_run.json` → `{ran_at, status: done|empty|below_batch, pending, written, counts}`

**口径铁律:** 页面上的数与终端 make 跑出来的是同一份产物,API 只读不重算。

- [ ] **Step 1: evaluate.py 在写 .md 之外落 eval_report.json**

判错样本按错误方向归类,`missed`/`extra` 相互组合决定 `kind` 与矩阵笔数——错例条数与矩阵笔数的差额全在错位上,这笔账要摊开给页面用:

```python
missed = [lb for lb in s["labels"] if lb not in pred_labels]
extra = [lb for lb in pred_labels if lb not in s["labels"]]
kind = "错位" if missed and extra else ("漏打" if missed else "多打")
errors.append({"text": s["text"], "gold": list(s["labels"]), "pred": pred_labels,
               "missed": missed, "extra": extra, "kind": kind,
               "matrix_entries": len(missed) + len(extra)})
```

每类的 `passed` 三值:严/中档给 bool,宽档给 `None`(页面画 —,不画 ✅);`red_line_passed = all(c["passed"] is not False ...)`。

- [ ] **Step 2: validate_golden.py / export_onnx.py / classify_pool.py 各落自己那份 json**

- export 的报告**失败也要写**(`mismatch != 0` 时先写报告再 `SystemExit`),否则验收页看不到「上次导出没过」,只看到一片空白;
- classify_pool 的 `status` 区分 `done` / `empty` / `below_batch`——第二遍跑出 `empty` 就是幂等的实证。

- [ ] **Step 3: 写 `scripts/ch10/scan_threshold_replay.py`(阈值扫描重演)**

验证集经 :8110 打分一次、分数表固定,九个候选线套同一张表各算一遍 micro-F1;`consistent` 记「重演当选线 == threshold.json 在用线」,证明阈值可复算。打平时先到的留任(`if f1 > best_f1` 用严格大于),所以 0.45 压住 0.50。

- [ ] **Step 4: Makefile 加目标**

```makefile
ch10-threshold-scan:  ## ch10 阈值扫描重演:九候选线各算一遍 micro-F1(需 :8110)
	PYTHONPATH=. uv run python scripts/ch10/scan_threshold_replay.py
```

`classify-pool` 配方尾部加 `$(if $(FORCE),--force,)`,让「不足一批强跑」也走同一条 make 配方,不在别处复制命令。

- [ ] **Step 5: 各跑一遍确认 json 落地(验证跑)**

Run: `make ch10-eval && make ch10-export && make ch10-threshold-scan && make ch10-golden && make classify-pool`
Expected: `data/ch10/reports/` 下五份 json 齐全;`threshold_scan.json` 的 `consistent` 为 true;第二遍 `classify-pool` 的 `status` 为 `empty`

- [ ] **Step 6: Commit + dev-notes 追记**

```bash
git add scripts/ch10 Makefile
git commit -m "feat(ch10): 验收产物一式两份——.md 给人读,.json 给验收页读"
```

---

### Task 15: 验收 API + 白名单作业运行器(TDD)

**Files:**
- Create: `app/core/jobs.py`、`app/api/jobs.py`、`app/api/acceptance.py`
- Modify: `app/main.py`(include_router + 挂 `/static` 静态目录)
- Test: `tests/test_acceptance_api.py`、`tests/test_jobs_api.py`、`tests/core/test_jobs.py`

**Interfaces:**
- Consumes: Task 14 的五份 json + `data/ch10/dataset/*.jsonl` + `data/ch10/model|onnx/` 文件 stat + `repository.topic_distribution` + :8110
- Produces:
  - `GET /api/acceptance/overview` → `{blocks:[{key,no,title,page,status,headline,note,jobs}], passed, total, all_pass, classifier, jobs}`
  - `GET /api/acceptance/eval` → `{eval, scan, threshold_in_use, severity, classifier}`
  - `GET /api/acceptance/data` → `{lineage, dataset:{splits,leaks,clean}, sample_review, model, onnx, topic_names}`
  - `GET /api/acceptance/errors` → `{eval, errors, kinds, matrix_entries, total_fp, total_fn, pairs, recipes}`
  - `GET /api/acceptance/service`、`POST /api/acceptance/classify`
  - `GET|POST /api/jobs[/{name}[/stop]]`(中立前缀:同一套运行器 ch03 建库也在用)

- [ ] **Step 1: 写失败测试**

`tests/core/test_jobs.py`:白名单外的作业名不在 `JOBS` 里;`status()` 对未跑过的作业回 `idle`;同名作业 `status == "running"` 时 `start()` 抛 `RuntimeError`。
`tests/test_acceptance_api.py`:reports 目录为空时 `/api/acceptance/overview` 仍 200,对应 block 的 `status == "missing"`、`note` 不含失败态文案;`/api/acceptance/errors` 在产物缺失时回空 errors 不抛错。
`tests/test_jobs_api.py`:白名单外的作业名 POST 回 404;同名作业未结束时再发起回 409;注册表里每个作业的 `cmd` 都以 `make ` 开头,且目标真在 Makefile 里(防止注册表写了个不存在的目标,页面按钮按下去才发现)。

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. uv run pytest tests/core/test_jobs.py tests/test_acceptance_api.py -v`
Expected: FAIL(模块不存在)

- [ ] **Step 3: 写 `app/core/jobs.py`**

安全边界写在模块头注释里:只能跑 `JOBS` 注册表里的目标,argv 全部写死;前端只传作业名,传不进任何命令片段。

```python
@dataclass(frozen=True)
class JobSpec:
    name: str
    title: str
    argv: tuple[str, ...]
    needs: str        # 前置条件,页面按钮上直接提示
    heavy: bool = False   # 分钟级重活:页面二次确认才发起
```

注册 §8 的 11 个 make 目标(`classify-pool-force` 走 `("make","classify-pool","FORCE=1")`)。运行要点:
- 日志覆盖写 `log/acceptance/<name>.log`(每次只看本次),头部写入 argv 与发起时间;
- `start_new_session=True`——停止时 `killpg` 能连 make → uv → python 整条链一起收,不留孤儿;
- 人为 kill 的退出码也是非 0,`status == "stopped"` 时不改写成 `failed`,免得冤枉它;
- 内存状态重启归零,但日志 mtime 与产物 `ran_at` 都在盘上,页面照样能说出「上次什么时候跑的」。

- [ ] **Step 4: 写 `app/api/jobs.py` + `app/api/acceptance.py`**

作业端点自己一个路由(`/api/jobs`:发起 / 查状态与日志尾 / 停止),不挂在验收 API 底下——同一张注册表后面还会被别的后台页按,发起作业这件事不属于哪一章的验收:

- `POST /api/jobs/{name}` 发起(未注册 404、同名未结束 409)、`GET /api/jobs/{name}` 回状态 + 日志尾、`POST /api/jobs/{name}/stop` 停;
- 验收 API 只在 `overview` 里带上「这块该按哪个作业」的作业名,不自己再开一套发起端点——同一件事只许有一处入口。

`app/api/acceptance.py` 这边:

- `_stat()` 只对 `.jsonl` / `.md` 算行数——`config.json`、`tokenizer.json` 算行数是噪声,摆在「条数」列会被当成记录数误读;
- `_load_report()` 缺产物回 `{present: False, make: <target>, hint: ...}`;
- `_split_stats()` 顺手算三份考卷两两文本重叠,`clean = 重叠总数 == 0`(硬闸);
- `gate()` 三态 + `note()`:产物没跑过时不许显示失败态文案(「没跑过」≠「没过线」);
- 字节数一律 1024 进制,与前端 `fmtBytes` 同口径——否则同一个权重文件会显示成两个数;
- mysql 读不到时 `topic_distribution` 那一项单独降级,不连坐整页。

`app/main.py` 除 include_router 外挂静态目录(四页共享外壳要有文件可取):

```python
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
```

- [ ] **Step 5: 跑测试确认通过 + 全量回归**

Run: `PYTHONPATH=. uv run pytest tests -x -q`
Expected: 新增测试 passed,全量不回归

- [ ] **Step 6: Commit + dev-notes 追记**

```bash
git add app/core/jobs.py app/api/jobs.py app/api/acceptance.py app/main.py tests/core/test_jobs.py tests/test_jobs_api.py tests/test_acceptance_api.py
git commit -m "feat(ch10): 验收 API + 白名单作业运行器(make 目标搬到页面按钮)"
```

---

### Task 16: 验收四页(Vibe Coding)

**Files:**
- Create: `app/static/acceptance.css`、`app/static/acceptance.js`、`app/static/acceptance.html`、`app/static/acceptance-eval.html`、`app/static/acceptance-data.html`、`app/static/acceptance-errors.html`
- Modify: `app/main.py`(四条页面路由)、`app/static/admin.js`(导航注册表加本章这几页)

**流程例外:** 与 Task 13 同,走 Vibe Coding——先出第一版,起服务截图给用户看,用户描述效果直接改,不套 TDD/code review。

- [ ] **Step 1: 写共享外壳 `acceptance.css` + `acceptance.js`**

配色与手绘硬阴影沿用 review.html / topics.html 同一套(`:root` 变量、4px 边框、`box-shadow` 硬投影、等宽字体)。`acceptance.js` 提供四页公用件:`api()` 取数、`toast()`、`scoreCell()` 分数配横条、`jobButton()/jobRow()` 重跑按钮 + 日志窗口(POST 发起 → 每 1.2s 轮询 → 终态停轮询并回调页面重取数;heavy 作业先 `confirm()`)、`missingBox()` 产物缺失占位。

日志窗口有个坑要一并躲掉:作业跑完要回调页面重取数,重取数又会把作业那一行连 `<pre>` 一起重建,日志就在跑完那一刻空了。所以按作业名把日志尾缓存在内存里,重建后贴回去。

页间导航不写在这里:`mountAdminNav(active)` 在 `app/static/admin.js`,后台各页共用同一份(自带样式注入,挂在样式内联的旧页上也不打架)。本章这几页往它的注册表里加一行即可——验收四页归一个模块、底下再挂一排子页(总览 / 评测详情 / 数据产物 / 错例复核),主题分布自己一格。

- [ ] **Step 2: 写四个页面**

- `acceptance.html`:九张闸门卡(编号 + 标题 + 三态药丸 + 一句结论 + 详情链接 + 该项的重跑按钮),顶部总闸 N/9、:8110 状态、评测结论;
- `acceptance-eval.html`:micro/macro 两块对照(附「为什么要看两个分」的解释)、每类指标表(不达标行标红、宽档红线列画 —)、阈值扫描九候选线(九条分数挤在小数第三位,按 min~max 拉伸才看得出高低)、17 类混淆矩阵宫格、单句试分类(17 类分数条 + 阈值竖线 + 兜底提示);
- `acceptance-data.html`:语料血缘四段落差 + 文件盘点表 + 三份考卷分布(类目为行、三卷为列)与泄漏自检硬闸 + 训练三件套 + ONNX 产物;
- `acceptance-errors.html`:记账口径(错例条数 ≠ 矩阵笔数)+ 边界摩擦配对表(同一对出现 ≥2 次标红)+ 逐条错例(放跑标红、冤枉标橙、对上的标绿)。

页面标题用产品口径(「分类器验收总览」等),不写章节编号——课程编号不该出现在产品后台。

`app/main.py` 加四条路由;新页面不成孤岛靠的是共用导航,不是各页互相手写链接。

- [ ] **Step 3: 起服务看效果并截图给用户**

Run: `make dev` + `make classifier-up`,浏览器开 `http://localhost:8000/acceptance`,chrome-devtools 按区块截紧凑图(隐藏其余面板 + 按内容高度裁窗口)贴给用户。

- [ ] **Step 4: 【检查点·Vibe 循环】按用户描述改到满意为止**

- [ ] **Step 5: Commit + dev-notes 追记(记录用户改了几轮、都改了什么)**

```bash
git add app/static/acceptance* app/main.py app/static/admin.js
git commit -m "feat(ch10): 验收四页——九项实证在浏览器上看、在浏览器上重跑"
```

---

### Task 17: 端到端验收 + code review + finish

**Files:** 无新增(演示 + 修复)

- [ ] **Step 1: 四条验收标准逐条演示并留证——全程在浏览器上跑**

起服务后从后台首页 `http://localhost:8000/admin` 进(各块的状态与入口都在这儿),开 `/acceptance`,用页面按钮依次跑:ONNX 导出 → 拉起分类器 → 测试集评测 → 阈值扫描 → 黄金样例闸 → 旁路批量归类,看总闸走到 9/9。逐条留证:

1. `/acceptance/eval` 每类 F1、support、红线达标与 17 类混淆矩阵(截图);
2. `/topics` 各类目统计与 `/acceptance` 第 9 张卡的归类结论,再点类目名进 `/topics/questions` 翻两页看这一类下的问题(截图);
3. `/acceptance/eval` 单句试分类喂「买大了想退」→ 尺码与退换货同时过线(截图);
4. 九项闸门全绿,且每项都能就地重跑(截图)。

- [ ] **Step 2: 全量测试回归**

Run: `PYTHONPATH=. uv run pytest tests -q`
Expected: 全 passed

- [ ] **Step 3: 邀请 code review(superpowers:requesting-code-review 技能;前端 Task 13、Task 16 除外)**

对 ch10 全部后端/脚本 diff 跑 code review,发现的问题修完回归测试,结论记 dev-notes。

- [ ] **Step 4: dev-notes 补 finish 段 + 收尾(superpowers:finishing-a-development-branch 技能)**

交付清单:后台各页地址(聚合首页 `/admin` 起,含 `/acceptance`、`/acceptance/eval`、`/acceptance/data`、`/acceptance/errors`、`/topics`、`/topics/questions`)、兜底的 make 命令清单、测试结果、dev-notes/ch10.md 路径。

---

## Self-Review 结论(已执行)

- **Spec 覆盖**:类目表→T1;输入文本规则→T2(_pool_text_stmt);清洗/预标/抽审/模拟(含刁钻掺比)→T4/T5;黄金样例闸→T4;分层划分/增强→T3/T6;全参微调+正则+早停+阈值→T7;每类 PRF1/混淆矩阵/容错红线/判错人工复核→T8;ONNX+一致性校验→T9;独立服务→T10;旁路批处理+攒批+幂等→T11;分布与类目问题列表 API→T12;主题分布页与类目问题列表页→T13;结构化产物与阈值可复算→T14;验收 API + 白名单作业运行器→T15;验收四页→T16;验收四条→T17。无缺口。
- **占位符扫描**:无 TBD/TODO。脚本与后端步骤含完整代码;两个 Vibe Coding 任务(T13、T16)按流程例外只给接口契约、页面构成与口径约束,终稿由用户反馈迭代出来,不预写死。
- **类型一致性**:`list_pool_texts`/`list_unclassified_questions` 返回 `[{question_id, text}]` 在 T2 定义、T5/T11 消费一致;`insert_topic_classifications(rows=[{question_id, labels}])` T2/T11/T12 一致;`topic_distribution` 返回结构 T2 定义、T12 API 透传、T13 前端消费一致;`topic_questions` 返回 `{label,total,page,pages,size,items[]}` T12 定义、T13 列表页消费一致;样本 dict `{text, labels, origin}` 贯穿 T3-T6;服务出参 `{results: [{labels, scores}]}` T10 定义、T11/T15 消费一致;五份 json 产物 schema T14 定义、T15 只读透传、T16 前端消费一致(`eval_report.json` 的 `classes[]` 同时喂每类指标表与混淆矩阵宫格,`errors[]` 同时喂错例卡与配对统计)。
- **口径一致性**:字节数 1024 进制在 T15 的 headline 与 T16 的 `fmtBytes` 两处统一;条数只对 `.jsonl`/`.md` 计;闸门三态 `pass`/`fail`/`missing` 在 T15 判定、T16 渲染,`missing` 两侧都不显示失败态文案。
