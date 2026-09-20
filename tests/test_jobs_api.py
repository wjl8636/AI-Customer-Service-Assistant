# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""作业运行器的安全边界:名字不在白名单里就 404,同名作业没跑完不许重入。

这两条是「页面上有重跑按钮但没有 shell」的实证。另加一条注册表自检:每个作业都得是
仓库 Makefile 里真有的目标,免得页面按钮指向一个不存在的配方。
"""
import re
import pathlib

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api import jobs as jobs_api
from app.core import jobs


@pytest.fixture
def app():
    a = FastAPI()
    a.include_router(jobs_api.router)
    return a


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        yield c


async def test_unknown_job_is_rejected(client):
    """前端只传作业名。名字对不上就到此为止——命令片段根本没有地方可以传进来。"""
    for path, method in (("/api/jobs/rm -rf", "post"), ("/api/jobs/rm -rf", "get"),
                         ("/api/jobs/未注册/stop", "post")):
        resp = await getattr(client, method)(path)
        assert resp.status_code == 404, path
        assert "未注册的作业" in resp.json()["detail"]


async def test_running_job_refuses_reentry(client, monkeypatch):
    """同名作业在跑就拒绝再发起,否则两份进程抢同一批产物。"""
    monkeypatch.setitem(jobs._runs, "kb-build",
                        jobs.JobRun(status="running", pid=4242))
    resp = await client.post("/api/jobs/kb-build")
    assert resp.status_code == 409 and "正在运行中" in resp.json()["detail"]


async def test_status_all_covers_registry(client):
    body = (await client.get("/api/jobs")).json()
    assert {j["name"] for j in body["jobs"]} == set(jobs.JOBS)
    for j in body["jobs"]:
        assert j["cmd"].startswith("make ")     # 一律走 make,页面按的和终端敲的是同一条命令


def test_every_job_target_exists_in_makefile():
    text = (pathlib.Path(__file__).resolve().parents[1] / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"^([a-zA-Z0-9_.-]+):", text, re.MULTILINE))
    missing = [spec.argv[1] for spec in jobs.JOBS.values() if spec.argv[1] not in targets]
    assert not missing, f"作业指向的 make 目标不存在:{missing}"
