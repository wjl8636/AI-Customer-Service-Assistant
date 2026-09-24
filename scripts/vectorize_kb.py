"""CLI:把 knowledge_chunks 里 pending 的块向量化写入 Milvus(幂等可重跑)。
运行:PYTHONPATH=. uv run python scripts/vectorize_kb.py"""
import asyncio

from app.db.base import engine
from app.kb import dualwrite, milvus_client


async def main() -> None:
    client = milvus_client.get_client()
    milvus_client.ensure_collection(client)
    n = await dualwrite.vectorize_pending(client)
    print(f"✅ 本次向量化 {n} 块;Milvus 现有 {milvus_client.count(client)} 条")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
