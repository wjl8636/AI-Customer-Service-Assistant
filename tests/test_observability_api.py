"""观测与成本 API:三块报表各读自己的权威源,缺产物是状态不是错误,一块挂了不连坐另两块。"""
import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api import observability
from app.db import repository

COST = {
    "meta": {"days": 7, "source": "Langfuse", "generated_at": "2026-07-29 10:00"},
    "rows": [{"intent": "商品咨询", "count": 3, "tokens": 8821, "avg_tokens": 2940, "share": 0.4712},
             {"intent": "物流", "count": 1, "tokens": 7942, "avg_tokens": 7942, "share": 0.4243}],
    "total_tokens": 16763, "total_requests": 4,
}
CALIB = {
    "meta": {"generated_at": "2026-07-29 10:20", "n_answerable": 60, "n_absent": 20},
    "distribution": {"answerable": {"n": 60, "min": 0.034, "p25": 0.489, "p50": 0.715,
                                    "p75": 0.781, "max": 0.856},
                     "absent": {"n": 20, "min": 0.002, "p25": 0.101, "p50": 0.105,
                                "p75": 0.132, "max": 0.251}},
    "scan": [{"t": 0.25, "pass_rate": 0.85, "leak_rate": 0.05, "youden_j": 0.8},
             {"t": 0.26, "pass_rate": 0.85, "leak_rate": 0.0, "youden_j": 0.85}],
    "recommended": {"threshold": 0.26, "youden_j": 0.85, "pass_rate": 0.85, "leak_rate": 0.0},
}


@pytest.fixture
def app():
    a = FastAPI()
    a.include_router(observability.router)
    return a


@pytest.fixture
async def client(app, db_session_factory):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        yield c


@pytest.fixture
def reports(tmp_path, monkeypatch):
    """产物路径指到临时目录:测试不读也不写仓库里那几份真报告。"""
    cost, calib = tmp_path / "cost.json", tmp_path / "calib.json"
    note = tmp_path / "trend_note.json"
    monkeypatch.setattr(observability, "COST", cost)
    monkeypatch.setattr(observability, "CALIB", calib)
    monkeypatch.setattr(observability, "TREND_NOTE", note)
    return cost, calib, note


async def test_nothing_run_yet_is_a_state_not_an_error(client, reports):
    """三块都没跑过:200 + 各自 present=false + 该按哪个作业。白屏或 500 会让人以为坏了。"""
    body = (await client.get("/api/observability/overview")).json()
    assert [b["present"] for b in (body["cost"], body["trend"], body["calibration"])] \
        == [False, False, False]
    assert body["cost"]["job"]["name"] == "cost-report"
    assert body["trend"]["make"] == "make eval-flywheel"
    assert body["calibration"]["job"]["name"] == "calibrate-confidence"


async def test_cost_is_passed_through_verbatim(client, reports):
    """页面上的占比与单均就是产物里的数:API 一个都不重算,最烧钱那路取产物第一行。"""
    reports[0].write_text(json.dumps(COST, ensure_ascii=False), encoding="utf-8")

    cost = (await client.get("/api/observability/overview")).json()["cost"]
    assert cost["present"] is True and cost["status"] == "ok"
    assert cost["rows"] == COST["rows"]
    assert cost["top"]["intent"] == "商品咨询"
    assert cost["total_tokens"] == 16763


async def test_cost_with_no_trace_in_window_reads_as_missing(client, reports):
    """跑过但窗口里一条带 intent 的 trace 都没有:页面该说「先聊几句」,不是显示空表。"""
    reports[0].write_text(json.dumps({**COST, "rows": [], "total_tokens": 0},
                                     ensure_ascii=False), encoding="utf-8")

    cost = (await client.get("/api/observability/overview")).json()["cost"]
    assert cost["status"] == "missing" and cost["top"] is None
    assert "问几句" in cost["hint"]


async def test_trend_reads_eval_runs_newest_first(client, reports):
    """趋势的权威源是 eval_runs 表:新在上,页面按这个顺序和上一轮比涨跌。"""
    await repository.insert_eval_run("手动", 80, {"recall_at_10": 1.0, "mrr": 0.972,
                                                  "faithfulness": 1.0, "refusal_rate": 1.0})
    await repository.insert_eval_run("定时", 80, {"recall_at_10": 1.0, "mrr": 0.964,
                                                  "faithfulness": 0.95, "refusal_rate": 1.0})

    trend = (await client.get("/api/observability/overview")).json()["trend"]
    assert trend["present"] is True
    assert [r["triggered_by"] for r in trend["runs"]] == ["定时", "手动"]
    assert trend["runs"][0]["metrics"]["faithfulness"] == 0.95


async def test_calibration_flags_threshold_out_of_sync(client, reports, monkeypatch):
    """推荐值和线上在用的不一致就得说出来:校准的意义正是回填那一行配置。"""
    reports[1].write_text(json.dumps(CALIB, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(observability.settings, "evidence_confidence_threshold", 0.5)

    calib = (await client.get("/api/observability/overview")).json()["calibration"]
    assert calib["present"] is True
    assert calib["recommended"]["threshold"] == 0.26 and calib["in_use"] == 0.5
    assert calib["in_sync"] is False
    assert calib["weights"]["top1"] == 0.5          # 四信号权重是代码常量,页面照抄


async def test_broken_report_does_not_take_down_the_page(client, reports):
    """成本产物写坏了半个 json:按「没跑过」处理,趋势与校准照样端出去。"""
    reports[0].write_text("{半个 json", encoding="utf-8")
    reports[1].write_text(json.dumps(CALIB, ensure_ascii=False), encoding="utf-8")

    body = (await client.get("/api/observability/overview")).json()
    assert body["cost"]["present"] is False
    assert body["calibration"]["present"] is True


async def test_read_notes_come_from_the_artifacts(client, reports):
    """读图小注也是产物的一部分:脚本落盘时就校过数,API 原样端出去,不在这儿重写。"""
    reports[0].write_text(json.dumps(
        {**COST, "read_notes": {"cost_by_intent": "商品咨询占 47%,先给它瘦 prompt"}},
        ensure_ascii=False), encoding="utf-8")
    reports[1].write_text(json.dumps(
        {**CALIB, "read_notes": {"confidence_calibration": "线定在 0.26,库外一条都进不来"}},
        ensure_ascii=False), encoding="utf-8")

    body = (await client.get("/api/observability/overview")).json()
    assert body["cost"]["read_note"] == "商品咨询占 47%,先给它瘦 prompt"
    assert body["calibration"]["read_note"] == "线定在 0.26,库外一条都进不来"


async def test_missing_read_note_is_null_not_an_error(client, reports):
    """上游不稳那一轮没生成注:产物照样有数,注为空,页面回落自己那句兜底话。"""
    reports[0].write_text(json.dumps(COST, ensure_ascii=False), encoding="utf-8")

    cost = (await client.get("/api/observability/overview")).json()["cost"]
    assert cost["status"] == "ok" and cost["read_note"] is None


async def test_trend_note_only_counts_for_the_run_it_describes(client, reports):
    """注旁挂在文件里,趋势本身在表里:注记的轮次不是最新一轮就不端,免得新数配旧注。"""
    await repository.insert_eval_run("手动", 80, {"recall_at_10": 1.0, "mrr": 0.972,
                                                 "faithfulness": 1.0, "refusal_rate": 1.0})
    latest_id = (await repository.list_eval_runs(limit=1))[0].id

    reports[2].write_text(json.dumps({"run_id": latest_id, "note": "四项都没退步"},
                                     ensure_ascii=False), encoding="utf-8")
    trend = (await client.get("/api/observability/overview")).json()["trend"]
    assert trend["read_note"] == "四项都没退步"

    reports[2].write_text(json.dumps({"run_id": latest_id - 1, "note": "上一轮的话"},
                                     ensure_ascii=False), encoding="utf-8")
    trend = (await client.get("/api/observability/overview")).json()["trend"]
    assert trend["read_note"] is None


async def test_mysql_down_only_breaks_the_trend_block(client, reports, monkeypatch):
    """mysql 挂了只让趋势那一块显示读数失败,成本与校准不连坐——依赖不齐时最该看这一页。"""
    reports[0].write_text(json.dumps(COST, ensure_ascii=False), encoding="utf-8")

    async def boom(*a, **kw):
        raise RuntimeError("mysql 没起")
    monkeypatch.setattr("app.db.repository.list_eval_runs", boom)

    body = (await client.get("/api/observability/overview")).json()
    assert body["trend"]["status"] == "error" and "mysql 没起" in body["trend"]["note"]
    assert body["cost"]["status"] == "ok"
