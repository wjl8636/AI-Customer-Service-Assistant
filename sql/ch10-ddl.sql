-- =============================================================
-- ch10 · 模型微调 · 建表 DDL
-- 本章新建:topic_classifications(微调分类器旁路批量归类结果)
-- 低置信度问题攒够一批,分类器归一次类,结果落这张表喂给飞轮后台看主题分布
-- 实时对话主链路不写这张表
-- =============================================================

-- 确保中文 COMMENT 按 utf8mb4 解析(latin1 默认的 mysql client 会把中文 double-encode)
SET NAMES utf8mb4;

CREATE TABLE topic_classifications (
  id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  question_id   BIGINT UNSIGNED NOT NULL                COMMENT '归类的问题,指向 low_confidence_questions.id',
  labels        JSON            NOT NULL                COMMENT '多标签,17 类权威类目里命中的若干个',
  classified_at DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '归类时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_question_id (question_id),
  CONSTRAINT fk_topic_question FOREIGN KEY (question_id) REFERENCES low_confidence_questions (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='主题分类结果';
