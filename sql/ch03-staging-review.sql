-- =============================================================
-- 挖知识加人工闸:暂存表补两个终态
-- =============================================================
-- 原来 kb-mine 抽完去重就直接写 knowledge_chunks,中间没人看。可那些问答对是模型从
-- 聊天记录里归纳的,客服当时那句回答可能只对那一单成立,可能带着具体订单号和收货地址,
-- 也可能只是一句「稍等我帮您看看」。进了库,之后每次检索都可能被捞出来当依据。
--
-- 改法不新开队列:挖知识本来就有这张暂存表和 /kb 页面上那张卡片,把 kept 从「去重保留、
-- 马上入库」改成「去重保留、等人审」,再补两个终态记人工结论。飞轮那条路(low_confidence
-- _questions → review_queue → /review)本来就有闸,不受影响。
--
-- kept 的语义变了但值没变:老数据里的 kept 是「已入库」,新数据里是「待审」。历史数据
-- 一次性归到 approved,免得建库时挖的那批被当成待办堆在页面上。
-- =============================================================

SET NAMES utf8mb4;

ALTER TABLE qa_extraction_staging
  MODIFY COLUMN status ENUM('extracted','kept','discarded','approved','rejected')
    NOT NULL DEFAULT 'extracted'
    COMMENT '已抽出待去重 / 去重保留待人审 / 去重丢弃 / 人工采纳已入库 / 人工弃用';

-- 本次迁移之前的 kept 都是已经写进 knowledge_chunks 的,直接标成 approved。
--
-- created_at 这个界限是给「手动重跑」兜底的,不能省。这个脚本正常只在 MySQL 容器
-- 首次初始化时跑一遍,但老数据卷升级上来的人得手动 apply,一不小心跑第二遍的话,
-- 不带界限的 UPDATE 会把当时**正在待审**的那批 kept 全标成 approved——它们一条都
-- 没进过知识库,却再也不会出现在待审列表里,知识就这么静默丢了。
UPDATE qa_extraction_staging
   SET status = 'approved'
 WHERE status = 'kept' AND created_at < '2026-08-16';
