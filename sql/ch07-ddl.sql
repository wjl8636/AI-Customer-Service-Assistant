-- =============================================================
-- ch07 · 会话上下文管理 · 建表 DDL(第一步,共两步)
-- 这一步给 ch02 的 conversations 表加两列:一段投影用的摘要,和摘要覆盖到哪条的锚点。
-- 分层要用的另一个锚点和分段摘要表在 ch07-layers.sql,两个文件都要 apply。
-- =============================================================

-- 确保中文 COMMENT 按 utf8mb4 解析(latin1 默认的 mysql client 会把中文 double-encode)
SET NAMES utf8mb4;

ALTER TABLE conversations
  ADD COLUMN summary             TEXT            NULL COMMENT '最近几段梗概拼成的投影,拼装时跟证据一起挂在用户那句之后' AFTER status,
  ADD COLUMN summary_upto_msg_id BIGINT UNSIGNED NULL COMMENT '摘要已覆盖到哪条消息,滑窗从其后接原文' AFTER summary;
