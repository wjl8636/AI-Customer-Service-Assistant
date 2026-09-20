# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_serves_chat_page():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    # 页面自包含且对接 /api/chat 的 SSE
    assert "/api/chat" in resp.text
    assert "session_id" in resp.text
    # 聊天页是客户看的页面,不留后台入口;后台各页自己带「聊天页 →」
    assert 'href="/admin"' not in resp.text


def test_admin_pages_are_reachable():
    """后台各页都能直接开;共享外壳(样式 + 脚本 + 导航)也在静态目录里。"""
    for path, marker in (
        ("/admin", "/api/admin/overview"),
        ("/kb", "/api/kb/overview"),
        ("/rag-eval", "/api/rag-eval/overview"),
        ("/review", "/api/review/queue"),
        ("/observability", "/api/observability/overview"),
        ("/topics", "/api/topics/distribution"),
        ("/topics/questions", "/api/topics/questions"),
        ("/acceptance", "/api/acceptance/overview"),
        ("/acceptance/eval", "/api/acceptance/eval"),
        ("/acceptance/data", "/api/acceptance/data"),
        ("/acceptance/errors", "/api/acceptance/errors"),
    ):
        resp = client.get(path)
        assert resp.status_code == 200, path
        assert marker in resp.text, path
        assert "/static/admin.js" in resp.text, path    # 后台导航一份共用
    for asset in ("/static/admin.js", "/static/acceptance.js", "/static/acceptance.css"):
        assert client.get(asset).status_code == 200, asset
