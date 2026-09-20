# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""CLI(可重跑 job):从历史对话挖 QA → 暂存表 → 去重 → 写 knowledge_chunks(pending)。
之后跑 scripts/vectorize_kb.py 向量化。运行:PYTHONPATH=. uv run python scripts/mine_knowledge.py"""
import asyncio

from app.kb import mining


async def main() -> None:
    stats = await mining.mine()
    print(f"✅ 挖知识:{stats}")


if __name__ == "__main__":
    asyncio.run(main())
