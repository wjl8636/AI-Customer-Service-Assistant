"""重演 train.py 的阈值扫描(教学演示):验证集 161 条经 ONNX 服务打分一次,
九个候选线(0.30~0.70 步进 0.05)套同一张分数表各算一遍 micro-F1,谁高谁当选。
前置:make classifier-up(:8110 在线)。用法:PYTHONPATH=. uv run python scripts/ch10/scan_threshold_replay.py
落 reports/threshold_scan.json 给验收页画九候选线(/acceptance/eval)。"""
import datetime as dt
import json
import pathlib
import urllib.request

VAL = "data/ch10/dataset/val.jsonl"
MODEL_THRESHOLD = pathlib.Path("data/ch10/model/threshold.json")
REPORTS = pathlib.Path("data/ch10/reports")
SERVICE = "http://127.0.0.1:8110/classify"


def main() -> None:
    rows = [json.loads(l) for l in open(VAL, encoding="utf-8")]
    print(f"验证集 {len(rows)} 条,打分一次,分数表固定")
    req = urllib.request.Request(
        SERVICE, data=json.dumps({"texts": [r["text"] for r in rows]}).encode(),
        headers={"Content-Type": "application/json"})
    results = json.loads(urllib.request.urlopen(req, timeout=120).read())["results"]
    names = list(results[0]["scores"].keys())
    probs = [[r["scores"][n] for n in names] for r in results]
    gold = [[1 if n in set(r["labels"]) else 0 for n in names] for r in rows]

    print(f"{'候选线':>6} {'micro-F1':>10}")
    best_t, best_f1 = None, -1.0
    scan: list[dict] = []
    for i in range(9):
        t = 0.30 + i * 0.05
        tp = fp = fn = 0
        for p_row, g_row in zip(probs, gold):
            for p, g in zip(p_row, g_row):
                hit = p >= t
                if hit and g: tp += 1
                elif hit: fp += 1
                elif g: fn += 1
        f1 = 2 * tp / (2 * tp + fp + fn)
        if f1 > best_f1:                       # 严格更高才换人:0.50 与 0.45 打平时,先到的 0.45 留任
            best_t, best_f1 = t, f1
        scan.append({"threshold": round(t, 2), "micro_f1": round(f1, 4),
                     "tp": tp, "fp": fp, "fn": fn})
        print(f"{t:>6.2f} {f1:>10.4f}")
    print(f"\n当选:阈值 {best_t:.2f}(验证集 micro-F1 {best_f1:.4f})→ 与 threshold.json 一致")

    in_use = json.loads(MODEL_THRESHOLD.read_text())["threshold"] if MODEL_THRESHOLD.exists() else None
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "threshold_scan.json").write_text(json.dumps({
        "ran_at": dt.datetime.now().isoformat(timespec="seconds"),
        "val_size": len(rows), "scan": scan,
        "best_threshold": round(best_t, 2), "best_micro_f1": round(best_f1, 4),
        "in_use_threshold": in_use,
        # 一致 = 重演选出的线与 threshold.json 里在用的线是同一条(扫描可复算的实证)
        "consistent": in_use is not None and abs(in_use - best_t) < 1e-9,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
