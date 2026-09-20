# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.acceptance import router as acceptance_router
from app.api.actions import router as actions_router
from app.api.admin import router as admin_router
from app.api.agent import router as agent_router
from app.api.chat import router as chat_router
from app.api.conversations import router as conversations_router
from app.api.extract import router as extract_router
from app.api.feedback import router as feedback_router
from app.api.jobs import router as jobs_router
from app.api.kb import router as kb_router
from app.api.observability import router as observability_router
from app.api.rageval import router as rageval_router
from app.api.review import router as review_router
from app.api.topics import router as topics_router
from app.graph import runtime

# 让 app.* 的 INFO 日志可见(uvicorn 默认不给 app 记录器配 INFO handler,
# 否则 ch05 log 节点的 turn trace——验收1 靠它观测「强制检索节点被走到」——不会输出)
_applog = logging.getLogger("app")
if not _applog.handlers:
    _fmt = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    _h = logging.StreamHandler()
    _h.setFormatter(_fmt)
    _applog.addHandler(_h)
    # 同时落项目内 log/app.log:稳定的 tail 目标(app.* 决策日志,不含 uvicorn HTTP 噪声,
    # 因 propagate=False + uvicorn 日志不经 app 记录器)。目录随启动自建,已 gitignore。
    _log_dir = Path(__file__).resolve().parent.parent / "log"
    _log_dir.mkdir(parents=True, exist_ok=True)
    _fh = logging.FileHandler(_log_dir / "app.log", encoding="utf-8")
    _fh.setFormatter(_fmt)
    _applog.addHandler(_fh)
_applog.setLevel(logging.INFO)
_applog.propagate = False

logger = logging.getLogger(__name__)
_STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动预热 Milvus:加载集合并轮询直到可检索,避免首个用户请求撞上 Standalone 的加载竞态
    (集合 load 在后台异步完成,未就绪时检索会静默返回空)。预热失败不阻塞启动。"""
    try:
        from app.core import retrieval
        for _ in range(15):
            hits = await retrieval.search_knowledge("退货运费", strategy="bm25", top_k=1)
            if hits:
                logger.info("Milvus 预热完成,集合可检索")
                break
            await asyncio.sleep(1)
        else:
            logger.warning("Milvus 预热超时(15s)未返回结果,继续启动")
    except Exception:
        logger.warning("Milvus 预热异常,继续启动", exc_info=True)
    from app.tools import registry as tool_registry
    tool_registry.scan_builtin()   # ch08:内置工具服务启动时登记(扫描 builtin/ 包,import 即注册)
    _check_context_budget()
    await runtime.init_graph()
    yield
    await runtime.close_graph()


def _check_context_budget() -> None:
    """启动自检:算一遍上下文预算,装不下一轮就报警,别等运行时撞墙。"""
    from app.core import budget
    b = budget.compute()
    if b.healthy:
        logger.info("上下文预算 %s", budget.describe(b))
    else:
        logger.error("上下文预算不足:窗口 %s 装不下 固定开销 %s + 单轮 ReAct 峰值 %s。"
                     "换窗口更大的模型,或调小 MAX_AGENT_STEPS / TOOL_RESULT_MAX_TOKENS / "
                     "MAX_OUTPUT_TOKENS / RERANK_TOP_K。%s",
                     b.window, b.fixed, b.peak, budget.describe(b))


app = FastAPI(title="MewHelp", version="0.1.0", lifespan=lifespan)
app.include_router(actions_router)
app.include_router(chat_router)
app.include_router(extract_router)
app.include_router(agent_router)
app.include_router(conversations_router)
app.include_router(feedback_router)
app.include_router(review_router)
app.include_router(topics_router)
app.include_router(acceptance_router)
app.include_router(kb_router)
app.include_router(rageval_router)
app.include_router(observability_router)
app.include_router(admin_router)
app.include_router(jobs_router)

# 后台各页共用外壳(样式 + 取数/重跑脚本 + 导航),抽成文件放静态目录
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def chat_page() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/admin", include_in_schema=False)
async def admin_page() -> FileResponse:
    """后台管理首页:知识库、RAG 评估、飞轮待审、观测与成本、主题分布、分类器验收各一张卡,一处进出。
    各模块页面还是各自原本的路径,首页只把入口收到一起,文档里贴过的链接照样能用。"""
    return FileResponse(_STATIC_DIR / "admin.html")


@app.get("/kb", include_in_schema=False)
async def kb_page() -> FileResponse:
    """ch03 知识库录入页:贴文档就能切块入库、向量化、当场检索自测。"""
    return FileResponse(_STATIC_DIR / "kb.html")


@app.get("/rag-eval", include_in_schema=False)
async def rageval_page() -> FileResponse:
    """ch04 RAG 评估页:四策略对照的那份报告,页面上看,也在页面上重跑。"""
    return FileResponse(_STATIC_DIR / "rageval.html")


@app.get("/review", include_in_schema=False)
async def review_page() -> FileResponse:
    """ch09 飞轮待审队列后台管理页。"""
    return FileResponse(_STATIC_DIR / "review.html")


@app.get("/observability", include_in_schema=False)
async def observability_page() -> FileResponse:
    """ch09 观测与成本页:意图成本账、评估趋势、置信度阈值校准三张报表,页上看也页上重跑。"""
    return FileResponse(_STATIC_DIR / "observability.html")


@app.get("/topics", include_in_schema=False)
async def topics_page() -> FileResponse:
    """ch10 主题分布页:分类器旁路归类结果,按 17 类看问题量。"""
    return FileResponse(_STATIC_DIR / "topics.html")


@app.get("/topics/questions", include_in_schema=False)
async def topic_questions_page() -> FileResponse:
    """ch10 类目问题列表:从分布图点进某一类,分页看这类下归类到的全部问题。"""
    return FileResponse(_STATIC_DIR / "topic-questions.html")


@app.get("/acceptance", include_in_schema=False)
async def acceptance_page() -> FileResponse:
    """ch10 验收总览:九项实证各一张闸门卡,产物缺失就地重跑。"""
    return FileResponse(_STATIC_DIR / "acceptance.html")


@app.get("/acceptance/eval", include_in_schema=False)
async def acceptance_eval_page() -> FileResponse:
    """ch10 评测详情:每类 P/R/F1 与容错红线、micro vs macro、阈值扫描、混淆矩阵、单句试分类。"""
    return FileResponse(_STATIC_DIR / "acceptance-eval.html")


@app.get("/acceptance/data", include_in_schema=False)
async def acceptance_data_page() -> FileResponse:
    """ch10 数据产物:语料血缘、三份考卷与泄漏自检、训练与 ONNX 产物盘点。"""
    return FileResponse(_STATIC_DIR / "acceptance-data.html")


@app.get("/acceptance/errors", include_in_schema=False)
async def acceptance_errors_page() -> FileResponse:
    """ch10 错例复核:逐条标准/预测对照、错误方向记账、边界摩擦配对。"""
    return FileResponse(_STATIC_DIR / "acceptance-errors.html")
