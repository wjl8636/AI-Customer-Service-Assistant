from app.kb import chunking

TABLE = "\n".join([
    "| 商品 | 价格 |",
    "| --- | --- |",
    "| A | 1 |",
    "| B | 2 |",
    "| C | 3 |",
])


def test_is_table_block():
    assert chunking.is_table_block(TABLE)
    assert not chunking.is_table_block("普通段落文字。")


def test_split_table_replicates_header():
    out = chunking.split_table_rows(TABLE, max_rows=2)
    assert len(out) == 2
    assert out[0] == "| 商品 | 价格 |\n| --- | --- |\n| A | 1 |\n| B | 2 |"
    assert out[1] == "| 商品 | 价格 |\n| --- | --- |\n| C | 3 |"
    for block in out:
        assert block.startswith("| 商品 | 价格 |\n| --- | --- |")


def test_split_table_small_stays_whole():
    assert chunking.split_table_rows(TABLE, max_rows=10) == [TABLE]


def test_table_with_leading_prose_detected_and_split():
    block = "说明:下表为处理时限。\n| 类型 | 时限 |\n| --- | --- |\n| A | 1 |\n| B | 2 |\n| C | 3 |"
    assert chunking.is_table_block(block)  # 前置散文也应识别出表格
    out = chunking.split_table_rows(block, max_rows=2)
    assert len(out) == 2
    assert out[0].startswith("说明:下表为处理时限。")  # 前置散文保留在首块
    assert "| 类型 | 时限 |" in out[0]
    assert out[1] == "| 类型 | 时限 |\n| --- | --- |\n| C | 3 |"  # 后续块只复制表头
