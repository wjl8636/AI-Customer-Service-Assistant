# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""对 /api/extract 跑标注样例,核对 order_id 与 request_type(expected_solution 人工目检)。"""
import json
import pathlib
import sys

import httpx

SAMPLES = pathlib.Path(__file__).parent.parent / "tests/data/extract_samples.json"
BASE = "http://localhost:8000"


def main() -> int:
    samples = json.loads(SAMPLES.read_text())
    failures = 0
    for i, s in enumerate(samples, 1):
        r = httpx.post(f"{BASE}/api/extract", json={"text": s["text"]}, timeout=60)
        r.raise_for_status()
        got = r.json()
        exp = s["expected"]
        ok = got["order_id"] == exp["order_id"] and got["request_type"] == exp["request_type"]
        status = "PASS" if ok else "FAIL"
        failures += not ok
        print(f"[{status}] #{i} {s['text'][:24]}...")
        print(f"       期望 order_id={exp['order_id']} type={exp['request_type']}")
        print(f"       实际 order_id={got['order_id']} type={got['request_type']} 方案={got['expected_solution']}")
    print(f"\n{len(samples) - failures}/{len(samples)} 通过")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
