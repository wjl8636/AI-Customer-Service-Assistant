"""冒烟:直连 SiliconFlow 调 BGE-M3,验证连通与维度。
需 .env 里 EMBED_BASE_URL / EMBED_API_KEY 填好。
运行:PYTHONPATH=. uv run python scripts/smoke_embed.py"""
import asyncio

from openai import AsyncOpenAI

from app.config import settings


async def main() -> None:
    client = AsyncOpenAI(base_url=settings.embed_base_url, api_key=settings.embed_api_key)
    resp = await client.embeddings.create(model=settings.embed_model, input=["邮费是多少", "运费怎么算"])
    dims = [len(d.embedding) for d in resp.data]
    print(f"返回 {len(resp.data)} 条向量,维度={dims}")
    assert dims and dims[0] == 1024, f"期望维度 1024,实际 {dims}"
    print("✅ 冒烟通过:直连 SiliconFlow BGE-M3,维度 1024")


if __name__ == "__main__":
    asyncio.run(main())
