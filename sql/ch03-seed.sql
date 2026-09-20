-- =============================================================
-- ch03 · 合成历史客服对话(喂挖知识 job)。三段式 + SET NAMES utf8mb4 + 幂等。
-- =============================================================
SET NAMES utf8mb4;

-- 段1 查询:执行前 seed 会话数
SELECT COUNT(*) AS before_conv FROM conversations WHERE user_id LIKE 'seed-%';

-- 段2 写入:先幂等清理 seed-% 会话及其消息,再插入
DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id LIKE 'seed-%');
DELETE FROM conversations WHERE user_id LIKE 'seed-%';

INSERT INTO conversations (user_id, status) VALUES ('seed-u1', '已结束'), ('seed-u2', '已结束');
SET @c1 = (SELECT id FROM conversations WHERE user_id = 'seed-u1' ORDER BY id DESC LIMIT 1);
SET @c2 = (SELECT id FROM conversations WHERE user_id = 'seed-u2' ORDER BY id DESC LIMIT 1);

INSERT INTO messages (conversation_id, role, content) VALUES
  (@c1, 'user',      '你们发货一般多久啊'),
  (@c1, 'assistant', '现货商品付款后 48 小时内发货,预售以商品详情页标注时间为准。'),
  (@c2, 'user',      '满多少包邮'),
  (@c2, 'assistant', '单笔订单满 99 元包邮,未满收取 10 元运费,偏远地区另计。');

-- 段3 验证:seed 会话应为 2,消息应为 4
SELECT COUNT(*) AS after_conv FROM conversations WHERE user_id LIKE 'seed-%';
SELECT COUNT(*) AS after_msg FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id LIKE 'seed-%');
