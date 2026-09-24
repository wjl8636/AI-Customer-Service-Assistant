"""挖知识的人工闸:kb-mine 自己不许写知识库,采纳才写。

挖出来的问答对是模型从聊天记录里归纳的,质量参差:只对单笔订单成立的、夹带订单号
和收货地址的、把「稍等我帮您看看」当成答案的,都有。这种东西进了库,之后每次检索
都可能被捞出来当依据。所以 mine() 挖完只落暂存表,入库口只有一个:
/api/kb/staging/approve。

飞轮那条路(low_confidence_questions → review_queue → /review)本来就有闸,不在这里测。
"""
import pytest

from app.kb import dualwrite, mining


class _FakeRepo:
    """够用的假仓储:记住状态流转和插过哪些行。"""

    def __init__(self, pairs):
        self.rows = []
        self.pairs = pairs
        self.status = {}

    async def insert_staging(self, batch_no, source_ref, question, answer):
        rid = len(self.rows) + 1
        self.rows.append({"id": rid, "batch_no": batch_no, "question": question,
                          "answer": answer})
        self.status[rid] = "extracted"
        return rid

    async def list_staging_by_status(self, status):
        out = []
        for r in self.rows:
            if self.status[r["id"]] == status:
                out.append(type("Row", (), {**r, "batch_no": r["batch_no"],
                                            "question": r["question"],
                                            "answer": r["answer"]})())
        return out

    async def set_staging_status(self, ids, status):
        for i in ids:
            self.status[i] = status

    async def list_all_questions(self):
        return []

    async def list_conversations_with_messages(self):
        return [(1, [type("M", (), {"role": "user", "content": "运费怎么算"})(),
                     type("M", (), {"role": "assistant", "content": "满99包邮"})()])]


@pytest.fixture
def fake_mine(monkeypatch):
    repo = _FakeRepo(None)
    monkeypatch.setattr(mining, "repository", repo)

    async def fake_extract(texts, model=None):
        return [mining.QaPair(question="运费怎么算", answer="满99包邮")]

    monkeypatch.setattr(mining, "extract_qa", fake_extract)
    return repo


async def test_mine_不再往知识库写一个字(fake_mine, monkeypatch):
    """整条 mine() 里不许出现 write_pending —— 那是这道闸的全部意义。

    守卫打在 dualwrite 模块本身上,不是打在 mining 的引用上:将来谁绕个圈子
    (换个 import 写法、从别处调进来)想写库,照样在这里炸。
    """
    async def _boom(chunks):
        raise AssertionError("mine() 不许写知识库,入库只能走 /api/kb/staging/approve")

    monkeypatch.setattr(dualwrite, "write_pending", _boom)
    stats = await mining.mine()

    assert stats["kept"] == 1
    # 挖出来的那条停在待审,等人处理
    assert [s for s in fake_mine.status.values() if s == "kept"] == ["kept"]


async def test_挖完的条目状态是待审而不是已入库(fake_mine):
    await mining.mine()
    assert set(fake_mine.status.values()) == {"kept"}
    assert "approved" not in fake_mine.status.values()   # 入库得人点
