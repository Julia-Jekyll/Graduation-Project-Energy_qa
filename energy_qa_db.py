import json
import uuid
import time
from datetime import datetime
from typing import Optional
import pymysql
import pymysql.cursors
from config_loader import load_config

# 数据库配置
_config = load_config()
_db_cfg = _config.get("database", {})

DB_HOST     = _db_cfg.get("host",     "localhost")
DB_PORT     = int(_db_cfg.get("port", 3306))
DB_USER     = _db_cfg.get("user",     "root")
DB_PASSWORD = _db_cfg.get("password", "")
DB_NAME     = _db_cfg.get("name",     "energy_qa")


# 数据库连接管理
def _get_connection() -> pymysql.connections.Connection:
    """创建并返回一个 PyMySQL的数据库连接对象"""
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


# 会话管理
def create_session(title: str = "新对话") -> str:
    """
    创建新会话，返回 session_id。

    参数:
        title: 会话标题，默认 "新对话"，也可传入问题前20字

    返回:
        session_id: str
    """
    session_id = str(uuid.uuid4())
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (id, title) VALUES (%s, %s)",
                (session_id, title[:255]),
            )
        conn.commit()
        return session_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_session(session_id: str) -> Optional[dict]:
    """
    按 session_id 查询会话信息。

    返回:
        dict(id, title, created_at, updated_at, message_count) 或 None
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM sessions WHERE id = %s", (session_id,))
            return cur.fetchone()
    finally:
        conn.close()


def list_sessions(limit: int = 30) -> list[dict]:
    """
    按最近更新时间列出会话（最多 limit 条）。

    返回:
        list of dict(id, title, created_at, updated_at, message_count)
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT %s",
                (limit,),
            )
            return cur.fetchall()
    finally:
        conn.close()


def _update_session_title_if_needed(conn, session_id: str, first_question: str):
    """如果会话标题仍为默认值，则用第一个问题的前20字更新标题"""
    with conn.cursor() as cur:
        cur.execute("SELECT title, message_count FROM sessions WHERE id = %s", (session_id,))
        row = cur.fetchone()
        if row and row["title"] == "新对话" and row["message_count"] == 0:
            new_title = first_question[:20] + ("..." if len(first_question) > 20 else "")
            cur.execute(
                "UPDATE sessions SET title = %s WHERE id = %s",
                (new_title, session_id),
            )


# 消息持久化
def save_user_message(session_id: str, content: str) -> int:
    """
    保存用户消息，返回 message_id。
    同时更新会话的 message_count 与标题（首次提问时）。
    参数:
        session_id: 所属会话 ID
        content:    用户问题文本
    返回:
        message_id: int
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            _update_session_title_if_needed(conn, session_id, content)
            cur.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (%s, 'user', %s)",
                (session_id, content),
            )
            message_id = cur.lastrowid
            cur.execute(
                "UPDATE sessions SET message_count = message_count + 1 WHERE id = %s",
                (session_id,),
            )
        conn.commit()
        return message_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def save_assistant_message(
    session_id: str,
    content: str,
    intent_summary: str = None,
    rag_sources: list = None,
    elapsed_seconds: float = None,
) -> int:
    """
    保存助手（系统）回答，返回 message_id。

    参数:
        session_id:      所属会话 ID
        content:         最终综合答案文本
        intent_summary:  Step1 提取的意图摘要
        rag_sources:     知识库来源文件列表
        elapsed_seconds: 本次回答总耗时

    返回:
        message_id: int
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO messages
                    (session_id, role, content, intent_summary, rag_sources, elapsed_seconds)
                VALUES (%s, 'assistant', %s, %s, %s, %s)
                """,
                (
                    session_id,
                    content,
                    intent_summary,
                    json.dumps(rag_sources or [], ensure_ascii=False),
                    elapsed_seconds,
                ),
            )
            message_id = cur.lastrowid
            cur.execute(
                "UPDATE sessions SET message_count = message_count + 1 WHERE id = %s",
                (session_id,),
            )
        conn.commit()
        return message_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def save_subtask_results(message_id: int, subtask_results: list[dict]):
    """
    批量保存子任务详情到 subtask_results 表。

    参数:
        message_id:      对应助手消息的 ID
        subtask_results: step2_concurrent_subtasks 返回的子任务结果列表
                         每项包含 subtask_id, focus, query, answer,
                                  knowledge_source, sources, status
    """
    if not subtask_results:
        return

    rows = []
    for r in subtask_results:
        rows.append((
            message_id,
            r.get("subtask_id"),
            r.get("focus", ""),
            r.get("focus", ""),         
            r.get("query", ""),
            r.get("answer", ""),
            r.get("knowledge_source", "none"),
            json.dumps(r.get("sources", []), ensure_ascii=False),
            r.get("status", "success"),
        ))

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO subtask_results
                    (message_id, subtask_id, dimension, focus, query,
                     answer, knowledge_source, sources, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_session_messages(session_id: str) -> list[dict]:
    """
    按时间顺序查询某会话的全部消息（不含子任务详情）。

    返回:
        list of dict，每项包含 messages 表所有字段
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM messages WHERE session_id = %s ORDER BY created_at ASC",
                (session_id,),
            )
            rows = cur.fetchall()
            for row in rows:
                if row.get("rag_sources") and isinstance(row["rag_sources"], str):
                    try:
                        row["rag_sources"] = json.loads(row["rag_sources"])
                    except json.JSONDecodeError:
                        row["rag_sources"] = []
            return rows
    finally:
        conn.close()


# 用户评价（点赞 / 踩）
def rate_message(
    message_id: int,
    rating: int,
    feedback: str = None,
) -> dict:
    """
    对某条助手消息进行点赞（1）或踩（-1），支持更新已有评价。

    参数:
        message_id: 要评价的消息 ID（必须是 role='assistant' 的消息）
        rating:     1 = 点赞 👍，-1 = 踩 👎
        feedback:   可选的文字反馈

    返回:
        dict(message_id, rating, feedback, action)
        action: "created" | "updated"

    异常:
        ValueError: rating 不是 1 或 -1
        LookupError: message_id 不存在或不是助手消息
    """
    if rating not in (1, -1):
        raise ValueError(f"rating 只能是 1（点赞）或 -1（踩），收到：{rating}")

    conn = _get_connection()
    try:
        # 验证消息存在且为助手消息 
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, role FROM messages WHERE id = %s",
                (message_id,),
            )
            row = cur.fetchone()
            if not row:
                raise LookupError(f"消息 ID {message_id} 不存在")
            if row["role"] != "assistant":
                raise LookupError(f"消息 ID {message_id} 是用户消息，只能对助手消息评价")

        #  插入或更新评价
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM ratings WHERE message_id = %s",
                (message_id,),
            )
            existing = cur.fetchone()

            if existing:
                cur.execute(
                    "UPDATE ratings SET rating = %s, feedback = %s WHERE message_id = %s",
                    (rating, feedback, message_id),
                )
                action = "updated"
            else:
                cur.execute(
                    "INSERT INTO ratings (message_id, rating, feedback) VALUES (%s, %s, %s)",
                    (message_id, rating, feedback),
                )
                action = "created"

        conn.commit()
        return {
            "message_id": message_id,
            "rating": rating,
            "feedback": feedback,
            "action": action,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_message_rating(message_id: int) -> Optional[dict]:
    """
    查询某条消息的评价记录。

    返回:
        dict(id, message_id, rating, feedback, created_at, updated_at) 或 None
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM ratings WHERE message_id = %s",
                (message_id,),
            )
            return cur.fetchone()
    finally:
        conn.close()


# 统计视图
def get_stats() -> dict:
    """
    从 v_stats 视图查询系统整体统计数据。

    返回:
        dict(total_sessions, total_questions, avg_response_time,
             total_likes, total_dislikes)
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM v_stats")
            row = cur.fetchone()
            return row or {}
    finally:
        conn.close()


def get_recent_ratings(limit: int = 50) -> list[dict]:
    """
    查询最近的评价记录，用于质量回顾/优化分析。

    返回:
        list of dict，含 message_id, rating, feedback, content, created_at
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.id, r.message_id, r.rating, r.feedback,
                       m.content, m.intent_summary, r.created_at
                FROM ratings r
                JOIN messages m ON m.id = r.message_id
                ORDER BY r.created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            return cur.fetchall()
    finally:
        conn.close()


# 完整问答流程持久化封装
def persist_qa_turn(
    session_id: str,
    user_question: str,
    pipeline_result: dict,
) -> dict:
    """
    将一次完整的问答轮次持久化到数据库（用户消息 + 助手消息 + 子任务详情）。

    参数:
        session_id:      会话 ID（由 create_session() 返回）
        user_question:   用户原始问题
        pipeline_result: MultiStepPipelineClient.ask() 的返回值
                         需含 final_answer, intent_summary, rag_sources,
                               elapsed_seconds, subtask_results

    返回:
        dict(user_message_id, assistant_message_id)
    """
    user_msg_id = save_user_message(session_id, user_question)

    assistant_msg_id = save_assistant_message(
        session_id=session_id,
        content=pipeline_result["final_answer"],
        intent_summary=pipeline_result.get("intent_summary"),
        rag_sources=pipeline_result.get("rag_sources", []),
        elapsed_seconds=pipeline_result.get("elapsed_seconds"),
    )

    save_subtask_results(assistant_msg_id, pipeline_result.get("subtask_results", []))

    return {
        "user_message_id": user_msg_id,
        "assistant_message_id": assistant_msg_id,
    }


# CLI 评价工具
def prompt_user_rating(message_id: int) -> Optional[dict]:
    """
    在命令行中请求用户对回答进行评价，返回评价结果。如果用户跳过则返回 None。
    参数:
        message_id: 要评价的助手消息 ID
    返回:
        rate_message() 的返回值，或 None
    """
    print("\n" + "─" * 40)
    print(" 请对本次回答进行评价（可选）：")
    print("   👍 输入 1 = 赞一个")
    print("   👎 输入 -1 = 需改进")
    print("   直接回车 = 跳过评价")
    print("─" * 40)

    try:
        raw = input("你的评价：").strip()
        if not raw:
            print("   （已跳过评价）")
            return None

        if raw not in ("1", "-1"):
            print("     输入无效，请输入 1 或 -1，已跳过评价")
            return None

        rating_val = int(raw)
        feedback_raw = input("文字反馈（可选，直接回车跳过）：").strip()
        feedback = feedback_raw if feedback_raw else None

        result = rate_message(message_id, rating_val, feedback)
        emoji = "👍" if rating_val == 1 else "👎"
        print(f"   {emoji} 评价已记录（ID: {message_id}）")
        return result

    except (KeyboardInterrupt, EOFError):
        print("\n   （评价已跳过）")
        return None
    except Exception as e:
        print(f"     评价保存失败：{e}")
        return None


# 删除操作
def delete_session(session_id: str) -> bool:
    """
    删除整个会话及其所有关联数据（消息、子任务、评价）。
    依赖数据库 ON DELETE CASCADE 外键约束完成级联删除。

    参数:
        session_id: 要删除的会话 ID

    返回:
        True = 删除成功，False = 会话不存在
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            affected = cur.execute(
                "DELETE FROM sessions WHERE id = %s", (session_id,)
            )
        conn.commit()
        return affected > 0
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def delete_message(message_id: int) -> bool:
    """
    删除单条消息及其关联的子任务详情和评价。
    同时将所属会话的 message_count 减 1。
    依赖数据库 ON DELETE CASCADE 外键约束完成级联删除。

    参数:
        message_id: 要删除的消息 ID

    返回:
        True = 删除成功，False = 消息不存在
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            # 先取 session_id，方便更新计数
            cur.execute("SELECT session_id FROM messages WHERE id = %s", (message_id,))
            row = cur.fetchone()
            if not row:
                return False

            session_id = row["session_id"]
            cur.execute("DELETE FROM messages WHERE id = %s", (message_id,))
            # 同步 message_count（不低于 0）
            cur.execute(
                "UPDATE sessions SET message_count = GREATEST(0, message_count - 1) WHERE id = %s",
                (session_id,),
            )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def clear_session_messages(session_id: str) -> int:
    """
    清空某会话的全部消息（保留会话本身）。

    参数:
        session_id: 目标会话 ID

    返回:
        删除的消息条数
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            affected = cur.execute(
                "DELETE FROM messages WHERE session_id = %s", (session_id,)
            )
            cur.execute(
                "UPDATE sessions SET message_count = 0 WHERE id = %s", (session_id,)
            )
        conn.commit()
        return affected
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
