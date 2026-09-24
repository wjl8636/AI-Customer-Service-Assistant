"""ch10 验收 API:把原本只能在终端看的九项实证搬上页面。只读 + 作业发起,不改任何产物。

数据来源分两类:
  1. 落盘产物(reports/*.json、dataset/*.jsonl、model/):由各 make 目标产出,这里只读不算
     ——页面上的数和终端跑出来的必须是同一份,不许在 API 里重算一遍造出第二个真相。
  2. 现场探活(:8110 healthz、单句试分类、文件 stat):秒级,请求时现取。
产物缺失不报错,回 present=false + 该跑哪个 make 目标,页面据此长出「重跑」按钮。
"""
import json
import pathlib
from datetime import datetime

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core import jobs
from app.core.taxonomy import SEVERITY, TOPIC_NAMES
from app.db import repository

router = APIRouter(prefix="/api/acceptance")

CH10 = pathlib.Path("data/ch10")
REPORTS = CH10 / "reports"
DATASET = CH10 / "dataset"
MODEL = CH10 / "model"
ONNX = CH10 / "onnx"
CLASSIFIER = "http://127.0.0.1:8110"

# 语料血缘的四段落差:捞池 → 脱敏去重 → 预标+模拟补足 → 分层切卷。
# 每段配一句「这一步在干什么」,页面直接显示,读者不必回去翻脚本。
LINEAGE = (
    ("corpus_raw.jsonl", "捞池", "低置信度问题池原始问法(标准化问法优先)", "ch10-corpus"),
    ("corpus_clean.jsonl", "脱敏去重", "去手机号/单号 → 去重 → LLM 修错别字 → 再去重", "ch10-corpus"),
    ("corpus_labeled.jsonl", "预标 + 模拟补足", "LLM 预标真实问题,再按类补造到每类 100 条", "ch10-corpus"),
)
SPLITS = (("train", "训练集(含增强与定向补数)"), ("val", "验证集(挑轮次 + 定阈值)"),
          ("test", "测试集(留出考卷,只在评测用一次)"))
MODEL_TRIO = ("model.safetensors", "tokenizer.json", "threshold.json")


def _stat(path: pathlib.Path) -> dict:
    """文件盘点的统一口径:存在性 + 字节 + 行数 + mtime。行数只对 jsonl/md 算。"""
    if not path.exists():
        return {"path": str(path), "present": False}
    st = path.stat()
    out = {"path": str(path), "present": True, "bytes": st.st_size,
           "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")}
    # 只有 jsonl/md 的行数才等于「条数」;config.json、tokenizer.json 之类算行数是噪声,
    # 摆在「条数」列会被当成记录数误读,所以不算
    if path.suffix in (".jsonl", ".md"):
        out["lines"] = sum(1 for line in path.read_text(encoding="utf-8").splitlines()
                           if line.strip())
    return out


def _load_report(name: str, make_target: str) -> dict:
    """读 reports/*.json;缺了就回 present=false 与该跑的 make 目标(页面据此提示重跑)。"""
    path = REPORTS / name
    if not path.exists():
        return {"present": False, "make": make_target,
                "hint": f"产物还没生成,先跑 make {make_target}"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {"present": False, "make": make_target, "hint": f"产物解析失败:{e}"}
    data["present"] = True
    data["make"] = make_target
    return data


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _split_stats() -> dict:
    """三份考卷的规模 + 各类标签计数 + 「考题有没有漏进练习题」的重叠自检。
    重叠必须是 0:验证/测试是考卷,一旦被训练集见过,后面所有分数都不作数。"""
    texts: dict[str, set[str]] = {}
    out: dict[str, dict] = {}
    for key, desc in SPLITS:
        rows = _read_jsonl(DATASET / f"{key}.jsonl")
        counts = {n: 0 for n in TOPIC_NAMES}
        multi = 0
        for r in rows:
            labels = r.get("labels") or []
            multi += len(labels) > 1
            for lb in labels:
                if lb in counts:
                    counts[lb] += 1
        texts[key] = {r["text"] for r in rows}
        out[key] = {"desc": desc, "size": len(rows), "multi_label": multi,
                    "counts": counts, "file": _stat(DATASET / f"{key}.jsonl")}
    leaks = {
        "train_val": len(texts.get("train", set()) & texts.get("val", set())),
        "train_test": len(texts.get("train", set()) & texts.get("test", set())),
        "val_test": len(texts.get("val", set()) & texts.get("test", set())),
    }
    return {"splits": out, "leaks": leaks, "clean": sum(leaks.values()) == 0}


async def _probe_classifier() -> dict:
    """:8110 探活。连不上不是错误,是一种状态——页面显示离线 + 一个「拉起服务」按钮。"""
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get(f"{CLASSIFIER}/healthz")
            r.raise_for_status()
            return {"online": True, "detail": r.json()}
    except Exception as e:
        return {"online": False, "detail": f"{type(e).__name__}: {e}"}


def _threshold_in_use() -> float | None:
    path = MODEL / "threshold.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("threshold")


@router.get("/overview")
async def overview() -> dict:
    """验收总览:九个区块各给一句结论 + 一个闸门状态(pass / fail / missing)。
    状态口径:missing = 产物还没有;fail = 跑过但没过线;pass = 跑过且过线。"""
    golden = _load_report("golden_report.json", "ch10-golden")
    evaluation = _load_report("eval_report.json", "ch10-eval")
    scan = _load_report("threshold_scan.json", "ch10-threshold-scan")
    export = _load_report("export_report.json", "ch10-export")
    classify = _load_report("classify_run.json", "classify-pool")
    ds = _split_stats()
    health = await _probe_classifier()
    trio = {n: _stat(MODEL / n) for n in MODEL_TRIO}
    try:
        dist = await repository.topic_distribution()
        dist_total, dist_hit = dist["total"], sum(1 for c in dist["classes"] if c["count"])
        dist_err = None
    except Exception as e:                    # mysql 没起时不连坐整页
        dist_total = dist_hit = None
        dist_err = f"{type(e).__name__}: {e}"

    def gate(present: bool, ok: bool | None) -> str:
        if not present:
            return "missing"
        return "pass" if ok else "fail"

    def note(present: bool, ok: bool | None, yes: str, no: str) -> str:
        """产物没跑过时不许显示失败态的结论——「没跑过」不等于「没过线」,
        直接把 no 文案摆出来会把读者引到错误的下一步动作。"""
        if not present:
            return "还没跑过,点右下按钮现场跑一遍"
        return yes if ok else no

    corpus_lines = {f: _stat(CH10 / f).get("lines") for f, *_ in LINEAGE}
    blocks = [
        {"key": "data", "no": 1, "title": "语料与数据集", "page": "/acceptance/data",
         "status": gate(bool(ds["splits"]["test"]["size"]), ds["clean"]),
         "headline": (f"{corpus_lines.get('corpus_raw.jsonl') or 0} 条捞池 → "
                      f"{corpus_lines.get('corpus_clean.jsonl') or 0} 条清洗 → "
                      f"{corpus_lines.get('corpus_labeled.jsonl') or 0} 条语料 → "
                      f"{ds['splits']['train']['size']}/{ds['splits']['val']['size']}"
                      f"/{ds['splits']['test']['size']} 训练/验证/测试"),
         "note": "考卷与练习题零重叠" if ds["clean"] else "训练集与考卷有重叠,分数不作数",
         "jobs": ["ch10-corpus", "ch10-dataset"]},
        {"key": "golden", "no": 2, "title": "黄金样例闸", "page": None,
         "status": gate(golden.get("present", False), golden.get("passed")),
         "headline": (f"通过率 {golden['rate']:.0%}(闸线 {golden['pass_line']:.0%}),"
                      f"{golden['hits']}/{golden['total']} 条集合全对"
                      if golden.get("present") else golden.get("hint", "")),
         "note": (f"{len(golden.get('failures', []))} 条错例;不过线就改 prompt,"
                  "不许改黄金样例来凑分" if golden.get("present")
                  else "还没跑过,点右下按钮现场跑一遍"),
         "jobs": ["ch10-golden"]},
        {"key": "train", "no": 3, "title": "训练产物", "page": "/acceptance/data",
         "status": gate(all(v["present"] for v in trio.values()),
                        all(v["present"] for v in trio.values())),
         # 与数据产物页的 fmtBytes 同口径(1024 进制),否则同一个权重文件会显示成两个数
         "headline": (f"权重 {trio['model.safetensors'].get('bytes', 0) / 1024 / 1024:.0f}MB · "
                      f"tokenizer · threshold {_threshold_in_use()}"),
         "note": "三件套齐全,评测与服务都读 threshold.json"
                 if all(v["present"] for v in trio.values()) else "权重三件套不全",
         "jobs": ["ch10-train"]},
        {"key": "export", "no": 4, "title": "ONNX 导出与服务", "page": None,
         "status": gate(export.get("present", False),
                        export.get("passed") and health["online"]),
         "headline": (f"{export['checked']} 条预测与 torch 一致"
                      f"(不一致 {export['mismatch']} 条) · :8110 "
                      f"{'在线' if health['online'] else '离线'}"
                      if export.get("present") else export.get("hint", "")),
         "note": "导出后必须逐条对齐 torch 才放行",
         "jobs": ["ch10-export", "classifier-up", "classifier-down"]},
        {"key": "eval", "no": 5, "title": "测试集评测", "page": "/acceptance/eval",
         "status": gate(evaluation.get("present", False), evaluation.get("red_line_passed")),
         "headline": (f"micro-F1 {evaluation['micro']['f1']:.3f} · "
                      f"macro-F1 {evaluation['macro']['f1']:.3f} · "
                      f"测试集 {evaluation['test_size']} 条"
                      if evaluation.get("present") else evaluation.get("hint", "")),
         "note": note(evaluation.get("present", False), evaluation.get("red_line_passed"),
                      "严档 F1 ≥ 0.9、中档 ≥ 0.8 全部达标", "有类目跌破容错红线,回头搞数据"),
         "jobs": ["ch10-eval"]},
        {"key": "threshold", "no": 6, "title": "阈值扫描", "page": "/acceptance/eval",
         "status": gate(scan.get("present", False), scan.get("consistent")),
         "headline": (f"九候选线重演,当选 {scan['best_threshold']}"
                      f"(验证集 micro-F1 {scan['best_micro_f1']:.4f})"
                      if scan.get("present") else scan.get("hint", "")),
         "note": note(scan.get("present", False), scan.get("consistent"),
                      f"与 threshold.json 在用的 {scan.get('in_use_threshold')} 一致",
                      "重演结果与在用阈值不一致"),
         "jobs": ["ch10-threshold-scan"]},
        {"key": "matrix", "no": 7, "title": "混淆矩阵", "page": "/acceptance/eval",
         "status": gate(evaluation.get("present", False), True),
         "headline": (f"{evaluation['total_cells']} 道是非题错 "
                      f"{evaluation['total_fp'] + evaluation['total_fn']} 道:"
                      f"冤枉 {evaluation['total_fp']} · 放跑 {evaluation['total_fn']}"
                      if evaluation.get("present") else evaluation.get("hint", "")),
         "note": "要紧的不是错几个,是错的方向",
         "jobs": ["ch10-eval"]},
        {"key": "errors", "no": 8, "title": "错例复核", "page": "/acceptance/errors",
         "status": gate(evaluation.get("present", False), True),
         "headline": (f"{len(evaluation.get('errors', []))} 条错例,"
                      f"记 {sum(e['matrix_entries'] for e in evaluation.get('errors', []))} 笔矩阵账"
                      if evaluation.get("present") else evaluation.get("hint", "")),
         "note": "错例条数 ≠ 矩阵笔数:错位一条记两笔",
         "jobs": ["ch10-eval"]},
        {"key": "classify", "no": 9, "title": "旁路批量归类", "page": "/topics",
         "status": gate(classify.get("present", False), classify.get("status") != "failed"),
         "headline": ((f"上次归类写入 {classify['written']} 条"
                       if classify.get("status") == "done"
                       else "上次跑批:池里没有待归类问题(幂等)"
                       if classify.get("status") == "empty"
                       else f"待归类 {classify.get('pending')} 条,不足一批")
                      if classify.get("present") else classify.get("hint", "")),
         "note": (f"topic_classifications 共 {dist_total} 条,命中 {dist_hit}/17 类"
                  if dist_total is not None else f"库里读数失败:{dist_err}"),
         "jobs": ["classify-pool", "classify-pool-force"]},
    ]
    passed = sum(b["status"] == "pass" for b in blocks)
    return {"blocks": blocks, "passed": passed, "total": len(blocks),
            "all_pass": passed == len(blocks), "classifier": health,
            "jobs": jobs.status_all()}


@router.get("/eval")
async def eval_detail() -> dict:
    """评测详情页数据:每类 P/R/F1/support/红线、混淆矩阵、micro vs macro、阈值扫描九候选线。"""
    evaluation = _load_report("eval_report.json", "ch10-eval")
    scan = _load_report("threshold_scan.json", "ch10-threshold-scan")
    return {"eval": evaluation, "scan": scan,
            "threshold_in_use": _threshold_in_use(),
            "severity": SEVERITY, "classifier": await _probe_classifier()}


@router.get("/data")
async def data_detail() -> dict:
    """数据产物页数据:语料血缘四段 + 文件盘点 + 三份考卷分布与重叠自检 + 训练/ONNX 产物。"""
    lineage = [{"file": f, "stage": stage, "desc": desc, "make": target,
                **_stat(CH10 / f)} for f, stage, desc, target in LINEAGE]
    return {
        "lineage": lineage,
        "dataset": _split_stats(),
        "sample_review": _stat(CH10 / "sample_review.md"),
        "model": {"files": [_stat(p) for p in sorted(MODEL.glob("*")) if p.is_file()],
                  "threshold": _threshold_in_use(),
                  "threshold_file": _stat(MODEL / "threshold.json"),
                  "trio_ok": all((MODEL / n).exists() for n in MODEL_TRIO)},
        "onnx": {"files": [_stat(p) for p in sorted(ONNX.glob("*")) if p.is_file()],
                 "report": _load_report("export_report.json", "ch10-export")},
        "topic_names": list(TOPIC_NAMES),
    }


@router.get("/errors")
async def errors_detail() -> dict:
    """错例复核页数据:逐条标准/预测对照 + 错误方向(漏打/多打/错位)+ 边界摩擦配对统计。

    配对统计是机器算的:把「该打没打的类 ← 反而打了的类」按对计数,同一对反复出现
    就是两类之间有边界摩擦,补对照句要成对补。补哪些句子是人的判断,页面不替人编。"""
    evaluation = _load_report("eval_report.json", "ch10-eval")
    if not evaluation.get("present"):
        return {"eval": evaluation, "errors": [], "kinds": {}, "pairs": []}
    errors = evaluation.get("errors", [])
    kinds: dict[str, int] = {}
    pairs: dict[tuple[str, str], int] = {}
    for e in errors:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        for miss in e["missed"]:
            for extra in e["extra"]:
                pairs[(miss, extra)] = pairs.get((miss, extra), 0) + 1
    return {
        "eval": {k: evaluation[k] for k in ("ran_at", "test_size", "threshold", "present")},
        "errors": errors,
        "kinds": kinds,
        "matrix_entries": sum(e["matrix_entries"] for e in errors),
        "total_fp": evaluation["total_fp"], "total_fn": evaluation["total_fn"],
        "pairs": [{"missed": m, "grabbed": g, "count": c, "severity": SEVERITY.get(m)}
                  for (m, g), c in sorted(pairs.items(), key=lambda x: -x[1])],
        # 三种错各自的修法配方(课程结论,不是每条错例的具体补句)
        "recipes": {
            "漏打": "次要诉求被主旋律淹没 → 补「主诉求 + 顺带诉求」的双标签句",
            "错位": "某个词横跨两类 → 成对补对照句,两边同时喂才学得会看语境",
            "多打": "边界过宽把邻类也扫进来 → 补该类的反例(近似但不属于它的句子)",
        },
    }


@router.get("/service")
async def service() -> dict:
    return {**await _probe_classifier(), "threshold": _threshold_in_use(),
            "onnx_present": (ONNX / "model.onnx").exists()}


class ClassifyIn(BaseModel):
    text: str


@router.post("/classify")
async def classify(body: ClassifyIn) -> dict:
    """单句试分类:把一句话喂 :8110,回 17 类分数 + 过线标签。
    页面拿它演示「17 类各自独立过线,过几个打几个」——多标签机制的现场证据。"""
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="请输入一句话")
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(f"{CLASSIFIER}/classify", json={"texts": [text]})
            r.raise_for_status()
            result = r.json()["results"][0]
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"分类器服务不可用({type(e).__name__}),先拉起 :8110") from e
    threshold = _threshold_in_use()
    scores = sorted(result["scores"].items(), key=lambda x: -x[1])
    return {"text": text, "threshold": threshold, "labels": result["labels"],
            "scores": [{"label": k, "score": v, "hit": k in result["labels"]}
                       for k, v in scores],
            # 全类不过线时取最高分兜底:这条兜底走没走,页面直接标出来
            "fallback": bool(threshold is not None and scores
                             and scores[0][1] < threshold)}


# 「重跑」按钮的发起 / 状态 / 停止在 app/api/jobs.py(前缀 /api/jobs):
# 知识库建库与分类器验收共用同一个作业运行器,端点就只该有一处。
