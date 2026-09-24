from sqlalchemy import text


async def test_test_db_reachable_and_tables_exist(_test_engine, db_clean):
    async with _test_engine.connect() as conn:
        rows = (await conn.execute(text("SHOW TABLES"))).scalars().all()
    assert {"conversations", "messages", "faq", "tickets"} <= set(rows)
