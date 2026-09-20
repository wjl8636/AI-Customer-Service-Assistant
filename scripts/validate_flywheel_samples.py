# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch09 Prompt 标注样例验证(工作要求:纯 Prompt 任务拿标注样例跑一遍代替 TDD)。
判分:matched_question_id 与 expect_match 一致 = 查重对;normalized_question 含全部
expect_keywords = 标准化对(仅新建样例查关键词)。两项都对才算过,通过率 ≥ 80% 视为可用。
运行:make flywheel-samples(需聊天上游在线)。
"""
import asyncio
import json
import sys

from app.core.flywheel import normalize_and_match

THRESHOLD = 0.8


async def main():
    samples = json.load(open("tests/data/flywheel_samples.json"))
    passed = 0
    for s in samples:
        try:
            r = await normalize_and_match(s["raw_question"], s["candidates"])
        except Exception as e:
            print(f"✗ {s['raw_question'][:24]}… 调用失败 {type(e).__name__}")
            continue
        match_ok = r.matched_question_id == s["expect_match"]
        kw_ok = all(k in r.normalized_question for k in s["expect_keywords"])
        ok = match_ok and kw_ok
        passed += ok
        mark = "✓" if ok else "✗"
        print(f"{mark} {s['raw_question'][:24]}… → {r.normalized_question} "
              f"match={r.matched_question_id}(期望 {s['expect_match']})")
    rate = passed / len(samples)
    print(f"\n通过 {passed}/{len(samples)} = {rate:.0%}(线 {THRESHOLD:.0%})")
    return 0 if rate >= THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
