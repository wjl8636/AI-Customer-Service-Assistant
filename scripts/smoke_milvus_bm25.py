# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""Milvus Standalone 原生 BM25 全文检索 + hybrid_search 冒烟。不通即红线停。"""
from pymilvus import (
    AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker,
)

URI = "http://localhost:19530"
COLL = "smoke_bm25"


def main() -> None:
    client = MilvusClient(uri=URI)
    if client.has_collection(COLL):
        client.drop_collection(COLL)

    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("text", DataType.VARCHAR, max_length=2048,
                     enable_analyzer=True, analyzer_params={"type": "chinese"})
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(
        name="text_bm25", input_field_names=["text"],
        output_field_names=["sparse"], function_type=FunctionType.BM25,
    ))
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
    index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    client.create_collection(COLL, schema=schema, index_params=index_params)

    client.insert(COLL, [
        {"id": 1, "dense": [0.1, 0.2, 0.3, 0.4], "text": "满99元包邮,不满收取10元运费"},
        {"id": 2, "dense": [0.2, 0.1, 0.4, 0.3], "text": "智能猫砂盆 Pro 型号支持自动清理和除臭"},
    ])
    client.flush(COLL)          # 刷盘,新写入数据才可被 BM25 检索到
    client.load_collection(COLL)
    import time
    time.sleep(2)               # 等 sparse 索引对新段生效

    # 纯 BM25:问型号应命中第 2 条
    res = client.search(COLL, data=["猫砂盆 Pro 型号"], anns_field="sparse",
                        limit=2, output_fields=["text"], search_params={"metric_type": "BM25"})
    print("BM25:", [(h["id"], round(float(h["distance"]), 3)) for h in res[0]])
    assert res[0][0]["id"] == 2, "型号词 BM25 应命中第 2 条"

    # hybrid RRF
    dense_req = AnnSearchRequest(data=[[0.1, 0.2, 0.3, 0.4]], anns_field="dense",
                                 param={"metric_type": "COSINE"}, limit=2)
    sparse_req = AnnSearchRequest(data=["运费"], anns_field="sparse",
                                  param={"metric_type": "BM25"}, limit=2)
    hres = client.hybrid_search(COLL, reqs=[dense_req, sparse_req], ranker=RRFRanker(),
                                limit=2, output_fields=["text"])
    print("hybrid:", [(h["id"], round(float(h["distance"]), 3)) for h in hres[0]])
    assert len(hres[0]) >= 1, "hybrid_search 应有结果"

    client.drop_collection(COLL)
    print("GO: Milvus Standalone BM25 + hybrid_search 通")


if __name__ == "__main__":
    main()
