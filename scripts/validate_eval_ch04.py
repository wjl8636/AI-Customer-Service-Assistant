# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""评估集自检(ch04):300 题手写的 ground truth 靠这个脚本兜住,不靠人眼。

查五件事:
  1) id 与 query 不重复,五个桶题数一致;
  2) 可作答桶的 expect_section 在知识库里至少能命中一个小节(命中 0 个 = 这题永远算漏);
  3) expect_points 必须逐字(去空白后)出现在目标小节的正文里——证据覆盖度是机械子串匹配,
     要点写得再对、只要库里不是这么写的,分数就永远上不去;
     跨文档桶(E_multi)写 expect_sections_all:每一组都必须能命中小节,要点在这几组的并集里查;
  4) D 桶不能带 ground truth,should_refuse 与桶必须自洽;
  5) 提示 expect_section 判得过松的题(一个关键词命中一堆小节,recall 会虚高)。

需 Milvus Standalone + 已建库(知识库正文从 Milvus 读,和评估时同一份)。
运行:make eval-check
"""
import asyncio
import collections
import json
import pathlib
import sys

from app.kb import milvus_client

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_EVALSET = _ROOT / "tests/data/eval_ch04.jsonl"
GRADED_BUCKETS = ("A_policy", "B_model", "C_colloquial", "E_multi")
LOOSE_LIMIT = 4          # 一个 expect_section 命中超过这么多小节就提示判松


def _norm(s: str) -> str:
    return "".join((s or "").split())


async def _kb_text_by_section() -> dict[str, str]:
    def work():
        c = milvus_client.get_client()
        milvus_client.ensure_collection(c)
        return c.query(collection_name=milvus_client.COLLECTION, filter="",
                       output_fields=["section_path", "question", "answer"],
                       limit=2000)
    rows = await milvus_client.acall(work)
    out: dict[str, str] = collections.defaultdict(str)
    for r in rows:
        out[r.get("section_path", "") or ""] += \
            f"{r.get('question', '')}:{r.get('answer', '')}\n"
    return out


async def main() -> int:
    rows = [json.loads(ln) for ln in open(_EVALSET, encoding="utf-8")]
    text_by_section = await _kb_text_by_section()
    all_text = _norm("".join(text_by_section.values()))
    errs: list[str] = []
    warns: list[str] = []

    for label, values in (("id", [r["id"] for r in rows]),
                          ("query", [r["query"] for r in rows])):
        dup = [v for v, n in collections.Counter(values).items() if n > 1]
        if dup:
            errs.append(f"{label} 重复:{dup}")
    per_bucket = collections.Counter(r["bucket"] for r in rows)
    if len(set(per_bucket.values())) != 1:
        errs.append(f"五桶题数不一致:{dict(per_bucket)}")

    cand_hist: collections.Counter = collections.Counter()
    for r in rows:
        rid = r["id"]
        if r["bucket"] not in GRADED_BUCKETS:
            if r["expect_section"] or r["expect_points"]:
                errs.append(f"{rid}:D 桶不该带 ground truth")
            if not r["should_refuse"]:
                errs.append(f"{rid}:D 桶 should_refuse 必须为 true")
            continue
        if r["should_refuse"]:
            errs.append(f"{rid}:可作答桶 should_refuse 必须为 false")
        if not r["expect_section"] or not r["expect_points"]:
            errs.append(f"{rid}:缺 expect_section 或 expect_points")
            continue
        groups = r.get("expect_sections_all") or [r["expect_section"]]
        matched, bad = [], False
        for gi, g in enumerate(groups, 1):
            g = g if isinstance(g, list) else [g]
            hit = [p for p in text_by_section if any(w in p for w in g)]
            if not hit:
                errs.append(f"{rid}:第 {gi} 组 {g} 命中 0 个小节")
                bad = True
            elif len(hit) > LOOSE_LIMIT:
                warns.append(f"{rid}:第 {gi} 组 {g} 命中 {len(hit)} 个小节,判定偏松")
            matched += hit
        if bad:
            continue
        matched = sorted(set(matched))
        cand_hist[len(groups)] += 1
        target = _norm("".join(text_by_section[p] for p in matched))
        for pt in r["expect_points"]:
            if _norm(pt) not in target:
                where = "在库里但不在目标小节" if _norm(pt) in all_text else "全库查不到"
                errs.append(f"{rid}:要点「{pt}」{where}")

    print(f"评估集 {len(rows)} 题 · 每桶 {dict(per_bucket)} · 知识库 {len(text_by_section)} 个小节")
    print(f"可作答题的「需要几组证据」分布:{dict(sorted(cand_hist.items()))}")
    for e in errs:
        print("  ✗", e)
    for w in warns:
        print("  !", w)
    print(f"\n错误 {len(errs)} 条 · 提示 {len(warns)} 条"
          + (" —— 自检通过" if not errs else " —— 自检不通过"))
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
