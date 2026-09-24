"""CLI:把 data/kb/*.md 切块写入 knowledge_chunks(pending)。之后跑 vectorize_kb.py。
运行:PYTHONPATH=. uv run python scripts/build_kb.py(或在录入页 /kb 上按「离线建库」)"""
import asyncio

from app.db import repository
from app.db.base import engine
from app.kb import documents, dualwrite
from app.kb.sources import KB_DIR, SOURCE_TYPES as DOCS


async def main() -> None:
    # 幂等守卫:文档块非幂等插入,重跑会重复。已存在则跳过,重建走 make kb-reset。
    existing = await repository.count_chunks_by_content_types(set(DOCS.values()))
    if existing:
        print(f"⚠️ 已存在 {existing} 条文档块,跳过以防重复插入。重建请先: make kb-reset")
        return
    total = 0
    for fname, ctype in DOCS.items():
        md = (KB_DIR / fname).read_text(encoding="utf-8")
        chunks = documents.build_chunks(md, content_type=ctype)
        ids = await dualwrite.write_pending(chunks)
        total += len(ids)
        print(f"  {fname}: {len(ids)} 块")
    print(f"✅ 建库(pending):共 {total} 块。下一步:scripts/vectorize_kb.py")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
