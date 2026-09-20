-- ch06: tickets.ticket_type 枚举加「退款」(加值不动旧行,存量安全)

-- 确保中文 ENUM 定义值按 utf8mb4 解析
-- (否则 latin1 默认的 mysql client 会把中文 double-encode,四个枚举值全存成乱码)
SET NAMES utf8mb4;

ALTER TABLE tickets
  MODIFY COLUMN ticket_type ENUM('售后','投诉','咨询','退款') NOT NULL;
