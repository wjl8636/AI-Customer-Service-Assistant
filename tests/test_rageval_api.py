"""RAG 评估 API:只读 make eval-rag 落下的那份报告,不重算;产物缺失是状态不是错误。"""
import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api import rageval

REPORT = {
    "meta": {"n_samples": 80, "kb_chunks": 51, "chat_model": "gpt-5.5",
             "generated_at": "2026-07-29 01:20"},
    "retrieval": {
        "vector": {"C_colloquial": {"recall": 1.0, "mrr": 0.887},
                   "overall": {"recall": 1.0, "mrr": 0.943}},
        "bm25": {"C_colloquial": {"recall": 0.75, "mrr": 0.557},
                 "overall": {"recall": 0.917, "mrr": 0.824}},
        "hybrid": {"overall": {"recall": 0.967, "mrr": 0.873}},
        "hybrid_rerank": {"overall": {"recall": 1.0, "mrr": 0.972}},
    },
    "evidence_coverage": {"vector": {"overall": 1.0}, "bm25": {"overall": 0.95},
                          "hybrid": {"overall": 0.983}, "hybrid_rerank": {"overall": 1.0}},
    "generation": {"answer_coverage": {"hybrid_rerank": {"overall": 1.0}},
                   "faithfulness": {"A_policy": {"v": 1.0, "answered": 20}},
                   "faithfulness_cases": [], "refusal": {"rate": 1.0, "total": 20, "correct": 20},
                   "skipped": 0},
}


@pytest.fixture
def app():
    a = FastAPI()
    a.include_router(rageval.router)
    return a


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        yield c


@pytest.fixture
def report(tmp_path, monkeypatch):
    """把产物路径指到临时目录:测试不读也不写仓库里那份真报告。"""
    p = tmp_path / "rag_eval.json"
    monkeypatch.setattr(rageval, "REPORT", p)
    monkeypatch.setattr(rageval, "TEXT_LOG", p.with_name("rag_eval.txt"))
    return p


async def test_missing_report_is_a_state_not_an_error(client, report):
    """还没跑过评估:200 + present=false + 该按哪个作业。白屏或者 500 都会让人以为坏了。"""
    body = (await client.get("/api/rag-eval/overview")).json()
    assert body["present"] is False
    assert body["make"] == "make eval-rag"
    assert body["job"]["specs"][0]["name"] == "eval-rag"
    assert body["job"]["artifacts"]["json"]["present"] is False


async def test_overview_reads_the_artifact_verbatim(client, report):
    """页面上的数就是产物里的数:API 一个指标都不重算。"""
    report.write_text(json.dumps(REPORT, ensure_ascii=False), encoding="utf-8")

    body = (await client.get("/api/rag-eval/overview")).json()
    assert body["present"] is True
    assert body["retrieval"] == REPORT["retrieval"]
    assert body["evidence_coverage"] == REPORT["evidence_coverage"]
    assert body["meta"]["n_samples"] == 80
    assert body["job"]["artifacts"]["json"]["present"] is True


async def test_best_strategy_is_the_top_overall_mrr(client, report):
    """四策略里挑总体 MRR 最高那一路:字典顺序、注册顺序都不算。"""
    data = json.loads(json.dumps(REPORT))
    data["retrieval"]["vector"]["overall"]["mrr"] = 0.99   # 让向量反超,结论必须跟着换
    report.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    body = (await client.get("/api/rag-eval/overview")).json()
    assert body["best"] == {"strategy": "vector", "mrr": 0.99}


async def test_read_notes_pass_through_and_default_to_empty(client, report):
    """读图小注是产物的一部分:有就原样端出去,没有就是空字典,页面回落自己那句兜底话。"""
    data = json.loads(json.dumps(REPORT))
    data["read_notes"] = {"rag_mrr": "型号桶纯向量只有 0.55,认错了型号"}
    report.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    body = (await client.get("/api/rag-eval/overview")).json()
    assert body["read_notes"]["rag_mrr"] == "型号桶纯向量只有 0.55,认错了型号"

    report.write_text(json.dumps(REPORT, ensure_ascii=False), encoding="utf-8")
    assert (await client.get("/api/rag-eval/overview")).json()["read_notes"] == {}


async def test_generation_missing_keeps_retrieval_half(client, report):
    """裁判上游挂了那一轮,generation 是 null:检索段照样端出去,页面只标生成段未完成。"""
    data = json.loads(json.dumps(REPORT))
    data["generation"] = None
    report.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    body = (await client.get("/api/rag-eval/overview")).json()
    assert body["present"] is True
    assert body["generation_done"] is False
    assert body["retrieval"]["hybrid_rerank"]["overall"]["mrr"] == 0.972


async def test_broken_report_reads_as_missing(client, report):
    """产物写坏了(半个 json)也不许 500:按「没跑过」处理,让人重跑一次。"""
    report.write_text("{半个 json", encoding="utf-8")

    body = (await client.get("/api/rag-eval/overview")).json()
    assert body["present"] is False
