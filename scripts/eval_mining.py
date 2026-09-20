# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""挖知识 eval:对样例对话跑抽取,核对该抽的抽到、不该抽的(纯个案)不硬抽。
需聊天上游可调通(真实 glm-5.2)。运行:PYTHONPATH=. uv run python scripts/eval_mining.py"""
import asyncio
import json
import pathlib
import sys

from app.kb import mining

SAMPLES = pathlib.Path(__file__).resolve().parent.parent / "tests/data/mining_samples.json"


async def main() -> int:
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    failures = 0
    for s in samples:
        pairs = await mining.extract_qa([s["conversation"]])
        if s.get("expect_empty"):
            ok = len(pairs) == 0
        else:
            ok = any(s["expect_question_contains"] in p.question for p in pairs)
        failures += not ok
        print(f"{'✅' if ok else '❌'} 抽到 {len(pairs)} 对:{[p.question for p in pairs]}")
    print(f"\n抽取符合预期 {len(samples) - failures}/{len(samples)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
