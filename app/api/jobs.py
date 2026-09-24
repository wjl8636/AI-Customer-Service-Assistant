"""作业 API:后台管理页上那些「重跑」按钮的唯一入口。发起 / 查状态与日志尾 / 停止。

白名单在 app/core/jobs.py,argv 写死在那边。这里只负责把作业名换成一次运行——
名字不在注册表里就 404,连命令都拼不出来。知识库建库(ch03)与分类器验收(ch10)共用这一套。
"""
from fastapi import APIRouter, HTTPException

from app.core import jobs

router = APIRouter(prefix="/api/jobs")


def _known(name: str) -> None:
    if name not in jobs.JOBS:
        raise HTTPException(status_code=404, detail=f"未注册的作业:{name}")


@router.get("")
async def jobs_list() -> dict:
    return {"jobs": jobs.status_all()}


@router.post("/{name}")
async def job_start(name: str) -> dict:
    _known(name)
    try:
        await jobs.start(name)
    except RuntimeError as e:                  # 同名作业还在跑:拒绝重入,别让两份进程抢产物
        raise HTTPException(status_code=409, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"命令起不来:{e}") from e
    return jobs.status(name, with_log=True)


@router.get("/{name}")
async def job_status(name: str) -> dict:
    _known(name)
    return jobs.status(name, with_log=True)


@router.post("/{name}/stop")
async def job_stop(name: str) -> dict:
    _known(name)
    try:
        await jobs.stop(name)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return jobs.status(name, with_log=True)
