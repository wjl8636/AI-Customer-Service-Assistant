# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch10 旁路批处理:低置信度问题攒够一批,整批喂分类器归一次类,结果写 topic_classifications。
实时对话主链路不调它。运行:make classify-pool(需 mysql + 分类器服务 :8110)。
幂等:已归类的行(LEFT JOIN 命中)不重复归。定时跑 cron 示例:
  0 3 * * * cd /path/to/mewhelp && make classify-pool >> log/classify-pool.log 2>&1"""
import argparse
import asyncio
import datetime as dt
import json
import pathlib

import httpx

from app.db import repository

SERVICE = "http://127.0.0.1:8110"
REPORTS = pathlib.Path("data/ch10/reports")


def _write_report(status: str, pending: int, written: int, counts: dict) -> None:
    """每次跑批留一份结论给验收页(/acceptance):第二遍 status=empty 就是幂等的实证。"""
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "classify_run.json").write_text(json.dumps({
        "ran_at": dt.datetime.now().isoformat(timespec="seconds"),
        "status": status, "pending": pending, "written": written,
        "counts": dict(sorted(counts.items(), key=lambda x: -x[1])),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


async def main(min_batch: int, force: bool) -> None:
    rows = await repository.list_unclassified_questions(limit=500)
    if not rows:
        print("池里没有待归类问题")
        _write_report("empty", 0, 0, {})
        return
    if len(rows) < min_batch and not force:
        print(f"待归类 {len(rows)} 条,不足一批({min_batch} 条);--force 可强跑")
        _write_report("below_batch", len(rows), 0, {})
        return
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{SERVICE}/classify",
                              json={"texts": [x["text"] for x in rows]})
        r.raise_for_status()
        results = r.json()["results"]
    # strict:question_id ↔ labels 全靠按序等长对应,服务端少回/多回一条都必须炸响,
    # 否则错位标签写库后会被幂等机制永久固化
    n = await repository.insert_topic_classifications(
        [{"question_id": x["question_id"], "labels": res["labels"]}
         for x, res in zip(rows, results, strict=True)])
    counts: dict[str, int] = {}
    for res in results:
        for lb in res["labels"]:
            counts[lb] = counts.get(lb, 0) + 1
    print(f"归类完成:{n} 条写入 topic_classifications;"
          + " ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])))
    _write_report("done", len(rows), n, counts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-batch", type=int, default=10, help="攒够多少条才归一次")
    ap.add_argument("--force", action="store_true", help="不足一批也强跑")
    a = ap.parse_args()
    asyncio.run(main(a.min_batch, a.force))
