# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch09 Cost Control:按意图统计 token 花销(README:「按意图把 token 分堆一算,
哪类意图最烧钱立马现形」)。数据源 = Langfuse Metrics API(意图在 classify_intent 打成 intent:xxx tag)。
运行:make cost-report(DAYS=N 窗口天数,默认 7;需 Langfuse 在跑且 .env 配好三变量)。
产物:data/ch09/reports/cost_by_intent.{txt,json};「观测与成本」页只读这份 json,不另算一份。

实测口径(自部署 Langfuse v3.218,SDK 走 api.legacy.metrics_v1 的 v1 端点;
v4 typed client 的 metrics 需 server v4 write mode):v1 只开放 observations/scores 视图
(traces 视图不可用),故按 observations 视图以 tags+traceId 分组拉明细,
客户端聚合出「请求数(trace 数)× 总 tokens」;未打 intent tag 的老 trace 不计入。
说明:自定义模型名(glm-5.2 等)在 Langfuse 无内置单价,统计以 token 数为准;
要看钱在 Langfuse 界面配模型单价即可,不在本脚本范围。
"""
import argparse
import asyncio
import json
import pathlib
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from app.core import read_notes
from app.core.observability import get_langfuse

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OUT_DIR = _ROOT / "data/ch09/reports"
_OUT = _OUT_DIR / "cost_by_intent.txt"
_OUT_JSON = _OUT_DIR / "cost_by_intent.json"


def _window(days: int) -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return (now - timedelta(days=days)).strftime(fmt), now.strftime(fmt)


def _rows_of(resp) -> list[dict]:
    data = getattr(resp, "data", None)
    if data is None and isinstance(resp, dict):
        data = resp.get("data")
    return [row if isinstance(row, dict) else row.dict() for row in (data or [])]


def _query(client, frm: str, to: str) -> list[dict]:
    """observations 视图按 (tags, traceId) 分组拉 token 明细,一行≈一条 trace 在某 tag 下的用量。"""
    resp = client.api.legacy.metrics_v1.metrics(query=json.dumps({
        "view": "observations",
        "metrics": [{"measure": "totalTokens", "aggregation": "sum"}],
        "dimensions": [{"field": "tags"}, {"field": "traceId"}],
        "filters": [],
        "fromTimestamp": frm, "toTimestamp": to,
    }))
    return _rows_of(resp)


def _aggregate(rows: list[dict]) -> list[dict]:
    """按 intent 聚合:请求数 = 去重 trace 数,tokens = 求和。tags 是数组,取 intent: 前缀项。"""
    acc: dict[str, dict] = defaultdict(lambda: {"tokens": 0, "traces": set()})
    for row in rows:
        tags = row.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        intents = [t.removeprefix("intent:") for t in tags if str(t).startswith("intent:")]
        if not intents:
            continue   # 未打 intent tag 的老 trace 不计入
        intent = intents[0]
        acc[intent]["tokens"] += int(float(row.get("sum_totalTokens") or 0))
        acc[intent]["traces"].add(row.get("traceId"))
    return sorted(
        ({"intent": k, "tokens": v["tokens"], "count": len(v["traces"])} for k, v in acc.items()),
        key=lambda r: r["tokens"], reverse=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()

    client = get_langfuse()
    if client is None:
        print("Langfuse 未配置(.env 三变量),无账可查。")
        return 1
    frm, to = _window(args.days)
    rows = _aggregate(_query(client, frm, to))

    total = sum(r["tokens"] for r in rows) or 1
    # 平均与占比在这里算一次,页面直接用:同一个数不许在前端再算一遍
    for r in rows:
        r["avg_tokens"] = r["tokens"] // max(r["count"], 1)
        r["share"] = round(r["tokens"] / total, 4)

    lines = [f"=== 按意图 token 花销(近 {args.days} 天,数据源 Langfuse)===",
             f"{'意图':6s} {'请求数':>8s} {'总tokens':>12s} {'平均tokens':>12s} {'占比':>7s}"]
    for i, r in enumerate(rows):
        mark = "  ← 最烧钱" if i == 0 else ""
        lines.append(f"{r['intent']:6s} {r['count']:>8d} {r['tokens']:>12,d} "
                     f"{r['avg_tokens']:>12,d} {r['share']:>6.0%}{mark}")
    if not rows:
        lines.append("(窗口内没有带 intent tag 的 trace——先聊几句再来)")
    out = "\n".join(lines)
    print(out)

    # 读图小注和这一轮的数一起落盘:页面不在渲染时找模型,同一份产物每次打开都是同一句话。
    # 字段名给中文,免得注里冒出 avg_tokens 这种只有代码里才有的说法
    payload = {"窗口天数": args.days,
               "各意图": [{"意图": r["intent"], "请求数": r["count"], "总 token": r["tokens"],
                           "单均 token": r["avg_tokens"], "占比": r["share"]} for r in rows],
               "总 token": sum(r["tokens"] for r in rows),
               "总请求数": sum(r["count"] for r in rows)}
    note = asyncio.run(read_notes.generate("cost_by_intent", payload)) if rows else None
    print("\n读图小注:" + (note if note else "本轮没有(页面用兜底句)"))

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(out + "\n", encoding="utf-8")
    _OUT_JSON.write_text(json.dumps({
        "meta": {"days": args.days, "source": "Langfuse",
                 "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "window_from": frm, "window_to": to},
        "rows": rows,
        "total_tokens": payload["总 token"],
        "total_requests": payload["总请求数"],
        "read_notes": {"cost_by_intent": note} if note else {},
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n报告已落 {_OUT.relative_to(_ROOT)} 与 {_OUT_JSON.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
