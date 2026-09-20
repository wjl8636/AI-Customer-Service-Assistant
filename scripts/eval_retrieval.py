# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""验收1 eval:对换说法的问题跑向量检索,核对是否召回期望内容(纯测向量召回,不掺 LLM)。
需已建库并向量化(make kb-build && make kb-vectorize)+ 嵌入上游可调通(真实嵌入)。
运行:PYTHONPATH=. uv run python scripts/eval_retrieval.py"""
import asyncio
import json
import pathlib
import sys

from app.core import retrieval

SAMPLES = pathlib.Path(__file__).resolve().parent.parent / "tests/data/retrieval_samples.json"


async def main() -> int:
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    failures = 0
    for s in samples:
        hits = await retrieval.search_knowledge(s["query"], strategy="vector")  # ch03 纯向量召回验证
        top = hits[0] if hits else None
        ok = bool(top) and s["expect_answer_contains"] in top["answer"]
        failures += not ok
        detail = f"{top['question']} | {top['answer'][:30]}" if top else "(空)"
        print(f"{'✅' if ok else '❌'} {s['query']!r} -> {detail}")
    print(f"\n召回正确 {len(samples) - failures}/{len(samples)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
