# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import re

from app.config import settings
from app.core import embeddings, rerank
from app.kb import milvus_client

_CLAUSE = re.compile(r"[,，;；?？。]")
_MIN_CLAUSE = 4          # 太短的碎片(「怎么办」)当不了子查询


def split_clauses(query: str) -> list[str]:
    """把一问多意图的问题按标点拆成子句;拆不出两条就原样返回一条。

    「我在新疆下单 80 块钱,运费怎么算,会员的免运费能不能抵」这种一句话问三件事的,
    整句去检索时向量会被主语义带跑,只有一个意图能挤进前排。拆开各查一遍再轮转合并,
    每个意图都能占到名额。注意这招得配精排:子句各自的第一名要够准,合并才有意义,
    没有精排的单路检索拆完反而更吵(评估集实测:重排 +0.02、其余三路 -0.04 到 -0.06)。
    """
    parts = [p.strip() for p in _CLAUSE.split(query) if len(p.strip()) >= _MIN_CLAUSE]
    return parts if len(parts) >= 2 else [query]


def _merge_round_robin(lists: list[list[dict]]) -> list[dict]:
    """各子句轮流出一条,按 chunk id 去重;同一小节只留最靠前的那条,重复的挪到后面。"""
    out, seen_id, seen_sec, tail = [], set(), set(), []
    for i in range(max((len(x) for x in lists), default=0)):
        for lst in lists:
            if i >= len(lst):
                continue
            h = lst[i]
            key = h.get("id") or (h.get("section_path", ""), h.get("question", ""))
            if key in seen_id:
                continue
            seen_id.add(key)
            sec = h.get("section_path") or ""
            if sec in seen_sec:
                tail.append(h)
            else:
                seen_sec.add(sec)
                out.append(h)
    return out + tail


def arrange_head_tail(items: list) -> list:
    """最相关放首、次相关放尾,其余按序居中(缓解 lost-in-the-middle)。"""
    if len(items) <= 2:
        return items
    return [items[0], *items[2:], items[1]]


async def search_knowledge(
    query: str, strategy: str = "hybrid_rerank", top_k: int | None = None,
    category: str | None = None, client=None, collection: str = "knowledge",
    bm25_text: str | None = None, split: bool = True,
) -> list[dict]:
    """strategy: vector | bm25 | hybrid | hybrid_rerank。
    返回最终排序的 hit(hybrid_rerank 含 rerank_score);不做首尾组装(调用侧按需)。

    query:dense 向量化与重排用的标准问法。bm25_text:BM25/sparse 检索文本(默认同 query);
    调用侧传入「标准问法 + 同义词扩展」只作用于 BM25 一侧,不污染 dense query(spec §2)。

    所有 Milvus 同步调用(建集合/检索)经 milvus_client.acall 固定到专用单线程执行,
    客户端也在该线程惰性创建 —— 规避 gRPC 客户端跨线程共享在 async server 下的间歇空返回。
    """
    top_k = top_k or settings.rerank_top_k
    bt = bm25_text or query  # BM25 检索文本;dense 始终用干净的 query

    # 一问多意图:按子句各检索一遍再轮转合并,免得一个意图把名额占满(见 split_clauses)
    if split and settings.subquery_split:
        clauses = split_clauses(query)
        if len(clauses) >= 2:
            per = [await search_knowledge(c, strategy=strategy, top_k=top_k,
                                          category=category, client=client,
                                          collection=collection, split=False)
                   for c in clauses]
            return _merge_round_robin(per)[:top_k]

    if strategy == "bm25":
        def work():
            c = client or milvus_client.get_client()
            milvus_client.ensure_collection(c, collection=collection)
            return milvus_client.bm25_search(c, bt, top_k, category, collection)
        return await milvus_client.acall(work)

    if strategy == "vector":
        vec = await embeddings.embed_query(query)

        def work():
            c = client or milvus_client.get_client()
            milvus_client.ensure_collection(c, collection=collection)
            return milvus_client.dense_search(c, vec, top_k, category, collection)
        return await milvus_client.acall(work)

    # hybrid / hybrid_rerank:dense 用 query 向量、BM25 用 bt,先混合召回 recall_top_k
    vec = await embeddings.embed_query(query)

    def work():
        c = client or milvus_client.get_client()
        milvus_client.ensure_collection(c, collection=collection)
        return milvus_client.hybrid_search(
            c, vec, bt, settings.recall_top_k, settings.recall_top_k, category, collection)
    hits = await milvus_client.acall(work)

    if strategy == "hybrid":
        return hits[:top_k]

    # hybrid_rerank:对召回结果精排(rerank 走上游 async,不进 Milvus 线程)
    docs = [f"{h['question']} {h['answer']}" for h in hits]
    ranked = await rerank.rerank(query, docs, top_n=top_k)
    out = []
    for idx, score in ranked:
        h = dict(hits[idx]); h["rerank_score"] = score
        out.append(h)
    return out
