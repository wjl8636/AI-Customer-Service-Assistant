"""后台首页聚合 API:知识库、RAG 评估、飞轮待审、观测与成本、主题分布、分类器验收各给一张卡。

各块依赖不同(mysql / Milvus / 嵌入与聊天上游 / :8110 分类器)。任何一块的依赖没起,
只让它自己那张卡显示读数失败,不连坐整页——后台首页的用处正是在依赖不齐的时候
也能一眼看出「哪一块现在是活的」。

每张卡的数都从各模块自己的读数函数来,这里只做汇总,不另算一份。
"""
from fastapi import APIRouter

from app.api import acceptance, kb, observability, rageval
from app.db import repository

router = APIRouter(prefix="/api/admin")

REVIEW_STATES = ("待审", "通过", "驳回")
RAG_LABEL = {"vector": "纯向量", "bm25": "纯 BM25", "hybrid": "混合",
             "hybrid_rerank": "混合 + 重排"}


def _card(key: str, title: str, page: str, lede: str) -> dict:
    """页面上不标章节号:后台是一个系统的几块功能,读者看的是「哪块要干活」,不是它出自第几章。"""
    return {"key": key, "title": title, "page": page, "lede": lede,
            "status": "error", "headline": "读数失败", "metrics": [], "note": None}


async def _kb_card() -> dict:
    """知识库:块数、待向量化、Milvus 条数。双写一致 = 没有块卡 pending 且两边条数对得上。"""
    card = _card("kb", "知识库", "/kb",
                 "文档与对话挖出来的知识 → 切块 → 双写 MySQL 与 Milvus")
    try:
        stats = await repository.knowledge_stats()
    except Exception as e:
        card["note"] = f"{type(e).__name__}: {e}"
        return card
    milvus = await kb.milvus_state()
    card["metrics"] = [
        {"label": "知识块", "value": stats["total"]},
        {"label": "待向量化", "value": stats["pending"]},
        {"label": "Milvus", "value": milvus["count"] if milvus["online"] else "离线"},
        {"label": "关键条款", "value": stats["key_clause"]},
    ]
    if not stats["total"]:
        card["status"], card["headline"] = "missing", "库是空的,先录入或跑一次离线建库"
    elif not milvus["online"]:
        card["status"], card["headline"] = "attention", f"MySQL 有 {stats['total']} 块,Milvus 离线"
    elif stats["pending"] or stats["done"] != milvus["count"]:
        card["status"] = "attention"
        card["headline"] = (f"{stats['pending']} 块待向量化,"
                            f"MySQL 已向量化 {stats['done']} 对 Milvus {milvus['count']}")
    else:
        card["status"] = "ok"
        card["headline"] = f"{stats['total']} 块双写一致,可被语义检索"
    card["note"] = "类型分布 " + " ".join(f"{k}={v}" for k, v in stats["by_content_type"].items())
    return card


async def _rageval_card() -> dict:
    """RAG 评估:四策略对照的那份报告。产物没跑过就是「没数据」,不是故障。"""
    card = _card("rageval", "RAG 评估", "/rag-eval",
                 "四策略对照:检索排得准不准 → 证据够不够 → 答案全不全")
    try:
        ov = await rageval.overview()
    except Exception as e:
        card["note"] = f"{type(e).__name__}: {e}"
        return card
    if not ov["present"]:
        card["status"], card["headline"] = "missing", "还没跑过评估,进去按一次「重跑 RAG 评估」"
        card["note"] = "四策略 × 四桶,分钟级;需 Milvus + 已建库 + 聊天上游"
        return card
    best, gen = ov["best"], ov["generation"]
    card["metrics"] = [
        {"label": "最佳 MRR", "value": f"{best['mrr']:.3f}" if best["mrr"] is not None else "—"},
        {"label": "评估集", "value": f"{ov['meta'].get('n_samples', '—')} 题"},
        {"label": "库外拒答",
         "value": f"{round(gen['refusal']['rate'] * 100)}%" if gen else "—"},
    ]
    if not ov["generation_done"]:
        card["status"] = "attention"
        card["headline"] = "生成段没跑完,只有检索段的数,补跑一次就齐"
    else:
        card["status"] = "ok"
        card["headline"] = f"{RAG_LABEL.get(best['strategy'], best['strategy'])} 领先,总体 MRR {best['mrr']:.3f}"
    card["note"] = f"上次跑于 {ov['meta'].get('generated_at') or '—'};页面只读产物,不重算"
    return card


async def _review_card() -> dict:
    """飞轮待审:答不上的问题标准化查重后攒在队列里,等人审。待审有货就是要干活。"""
    card = _card("review", "飞轮待审队列", "/review",
                 "答不上的问题 → 标准化查重 → 人工审核 → 写回知识库")
    try:
        counts = {st: len(await repository.list_review_queue(st)) for st in REVIEW_STATES}
    except Exception as e:
        card["note"] = f"{type(e).__name__}: {e}"
        return card
    card["metrics"] = [{"label": st, "value": counts[st]} for st in REVIEW_STATES]
    total = sum(counts.values())
    if not total:
        card["status"], card["headline"] = "missing", "队列还没有货,先在聊天页问几个答不上的问题"
    elif counts["待审"]:
        card["status"], card["headline"] = "attention", f"{counts['待审']} 条等着审"
    else:
        card["status"], card["headline"] = "ok", f"待审清零,累计处理 {total} 条"
    card["note"] = "审核通过即写回知识库并即时向量化,下一轮就能召回"
    return card


async def _observability_card() -> dict:
    """观测与成本:意图成本账、评估趋势、置信度阈值校准三块,哪块缺产物就说哪块。"""
    card = _card("observability", "观测与成本", "/observability",
                 "钱花在哪类问题上 · 指标有没有劣化 · 兜底阈值怎么定的")
    try:
        ov = await observability.overview()
    except Exception as e:
        card["note"] = f"{type(e).__name__}: {e}"
        return card
    cost, trend, calib = ov["cost"], ov["trend"], ov["calibration"]
    latest = (trend["runs"][0]["metrics"] if trend.get("runs") else {}) or {}
    top = cost.get("top")
    card["metrics"] = [
        # 卡上一格只放一个数,四格才排得齐一行;哪条意图最烧钱写在结论句里
        {"label": "最烧钱占比", "value": f"{round(top['share'] * 100)}%" if top else "—"},
        {"label": "评估轮次", "value": len(trend.get("runs") or [])},
        {"label": "忠实度",
         "value": f"{latest['faithfulness']:.3f}" if latest.get("faithfulness") is not None else "—"},
        {"label": "在用阈值", "value": f"{calib['in_use']:.2f}"},
    ]
    missing = [name for name, blk in (("意图成本账", cost), ("评估趋势", trend),
                                      ("阈值校准", calib)) if blk["status"] != "ok"]
    if trend["status"] == "error":
        card["note"] = trend.get("note")
        return card
    if len(missing) == 3:
        card["status"], card["headline"] = "missing", "三块都还没跑过,进去按一次就有数"
    elif missing:
        card["status"], card["headline"] = "attention", "缺 " + "、".join(missing)
    elif calib["in_sync"] == "aggressive":
        card["status"] = "attention"
        card["headline"] = (f"在用阈值 {calib['in_use']} 低于推荐 "
                            f"{calib['recommended'].get('threshold')},应拒可能漏进来")
    else:
        card["status"] = "ok"
        faith = latest.get("faithfulness")
        card["headline"] = ((f"{top['intent']}最烧钱," if top else "")
                            + "最近一轮忠实度 "
                            + (f"{faith:.3f}" if faith is not None else "—"))
    card["note"] = card["note"] or "报表都是 make 落的产物,页面只读不重算"
    return card


async def _topics_card() -> dict:
    """主题分布:分类器旁路归类的结果,看哪类堆得多。"""
    card = _card("topics", "主题分布", "/topics",
                 "低置信度问题 → 分类器旁路归类 → 哪类堆得多,先补哪块知识")
    try:
        dist = await repository.topic_distribution()
    except Exception as e:
        card["note"] = f"{type(e).__name__}: {e}"
        return card
    hit = sum(1 for c in dist["classes"] if c["count"])
    # classes 是权威类目表的固定顺序,不是排行榜;要说「占前三」得自己按问题量排
    top = sorted((c for c in dist["classes"] if c["count"]),
                 key=lambda c: c["count"], reverse=True)[:3]
    card["metrics"] = [{"label": "已归类问题", "value": dist["total"]},
                       {"label": "命中类目", "value": f"{hit}/{len(dist['classes'])}"}]
    if not dist["total"]:
        card["status"], card["headline"] = "missing", "还没归类过,去分类器验收页跑一次旁路批量归类"
    else:
        card["status"] = "ok"
        card["headline"] = "占前三:" + "、".join(f"{c['label']} {c['count']}" for c in top)
    return card


async def _classifier_card() -> dict:
    """分类器验收:九项实证的闸门数直接取验收总览那一份,不在这里重算。"""
    card = _card("classifier", "分类器验收", "/acceptance",
                 "语料 → 微调 → 评测 → 导出 → 旁路归类,九项实证逐项过闸")
    try:
        ov = await acceptance.overview()
    except Exception as e:
        card["note"] = f"{type(e).__name__}: {e}"
        return card
    online = ov["classifier"]["online"]
    card["metrics"] = [{"label": "过闸", "value": f"{ov['passed']}/{ov['total']}"},
                       {"label": ":8110", "value": "在线" if online else "离线"}]
    if ov["all_pass"]:
        card["status"], card["headline"] = "ok", "九项全过闸"
    else:
        missing = [b["title"] for b in ov["blocks"] if b["status"] == "missing"]
        failed = [b["title"] for b in ov["blocks"] if b["status"] == "fail"]
        card["status"] = "attention" if failed else "missing"
        card["headline"] = "、".join(
            ([f"{len(failed)} 项不达标:" + "/".join(failed)] if failed else [])
            + ([f"{len(missing)} 项缺产物:" + "/".join(missing)] if missing else [])
        ) or "还没跑过"
    card["note"] = "页面上的数与终端 make 跑出来的是同一份产物"
    return card


@router.get("/overview")
async def overview() -> dict:
    return {"modules": [await _kb_card(), await _rageval_card(), await _review_card(),
                        await _observability_card(), await _topics_card(),
                        await _classifier_card()]}
