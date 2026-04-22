-- Energy QA System - MySQL Database Schema

CREATE DATABASE IF NOT EXISTS energy_qa CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE energy_qa;

-- 会话表：每次对话为一个 session
CREATE TABLE IF NOT EXISTS sessions (
    id          VARCHAR(36)  NOT NULL PRIMARY KEY,   
    title       VARCHAR(255) NOT NULL DEFAULT '新对话',
    created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    message_count INT        NOT NULL DEFAULT 0,
    INDEX idx_sessions_updated (updated_at DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 消息表：每条用户提问或系统回答
CREATE TABLE IF NOT EXISTS messages (
    id              BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
    session_id      VARCHAR(36)  NOT NULL,
    role            ENUM('user','assistant') NOT NULL,
    content         LONGTEXT     NOT NULL,
    intent_summary  TEXT         NULL,                
    rag_sources     JSON         NULL,          
    elapsed_seconds FLOAT        NULL,               
    created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    INDEX idx_messages_session (session_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 评价表：用户对回答的点赞/踩
CREATE TABLE IF NOT EXISTS ratings (
    id          BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
    message_id  BIGINT       NOT NULL UNIQUE,         
    rating      TINYINT      NOT NULL,                
    feedback    TEXT         NULL,                   
    created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 子任务详情表：存储每次回答的三维分析子任务结果
CREATE TABLE IF NOT EXISTS subtask_results (
    id              BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
    message_id      BIGINT       NOT NULL,
    subtask_id      TINYINT      NOT NULL,
    dimension       VARCHAR(50)  NOT NULL,
    focus           VARCHAR(100) NOT NULL,
    query           TEXT         NOT NULL,
    answer          LONGTEXT     NOT NULL,
    knowledge_source ENUM('rag','web','none') NOT NULL DEFAULT 'none',
    sources         JSON         NULL,
    status          VARCHAR(20)  NOT NULL DEFAULT 'success',
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
    INDEX idx_subtasks_message (message_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 性能统计视图
CREATE OR REPLACE VIEW v_stats AS
SELECT
    COUNT(DISTINCT s.id)                                    AS total_sessions,
    COUNT(DISTINCT CASE WHEN m.role='user' THEN m.id END)   AS total_questions,
    ROUND(AVG(CASE WHEN m.role='assistant' THEN m.elapsed_seconds END), 2) AS avg_response_time,
    SUM(CASE WHEN r.rating = 1  THEN 1 ELSE 0 END)          AS total_likes,
    SUM(CASE WHEN r.rating = -1 THEN 1 ELSE 0 END)          AS total_dislikes
FROM sessions s
LEFT JOIN messages m  ON m.session_id = s.id
LEFT JOIN ratings  r  ON r.message_id = m.id;
