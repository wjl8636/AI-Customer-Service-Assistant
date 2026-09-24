from sqlalchemy import select

from app.db.models import KnowledgeChunk, QaExtractionStaging


async def test_knowledge_chunk_roundtrip(db_session_factory):
    async with db_session_factory() as s:
        row = KnowledgeChunk(category="售后政策", questions="运费说明", answer="满99包邮")
        s.add(row)
        await s.commit()
        await s.refresh(row)
        assert row.id is not None
        assert row.vectorize_status == "pending"
        assert row.is_key_clause == 0


async def test_staging_roundtrip(db_session_factory):
    async with db_session_factory() as s:
        row = QaExtractionStaging(batch_no="b1", question="邮费多少", answer="满99包邮")
        s.add(row)
        await s.commit()
        got = (await s.execute(select(QaExtractionStaging))).scalars().all()
        assert len(got) == 1
        assert got[0].status == "extracted"
