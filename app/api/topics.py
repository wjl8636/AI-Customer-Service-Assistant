# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch10 主题分布 API:飞轮后台看各类目问题量,决定先补哪块知识。只读。

分布只给每类三条样例(一眼看出这类装的是什么);要看全,点类目名进问题列表,
那边按类目分页列出全部归类结果——同一批 topic_classifications,两种粒度。
"""
from fastapi import APIRouter, HTTPException, Query

from app.core.taxonomy import TOPIC_NAMES
from app.db import repository

router = APIRouter()


@router.get("/api/topics/distribution")
async def distribution():
    return await repository.topic_distribution()


@router.get("/api/topics/questions")
async def questions(
    label: str = Query(description="权威类目名,须在 17 类之内"),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
):
    # 类目名不在权威表里就是笔误或旧链接:400 说清楚,别静静回一页空列表
    if label not in TOPIC_NAMES:
        raise HTTPException(400, f"未知类目「{label}」,权威类目共 {len(TOPIC_NAMES)} 类")
    return await repository.topic_questions(label, page=page, size=size)
