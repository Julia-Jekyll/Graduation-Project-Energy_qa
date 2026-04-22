"""
app.py - 能源管理在线问答系统 Web 服务
基于 Flask，提供 REST API 供前端调用
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import threading

#  导入已有模块
from energy_qa_db import (
    create_session,
    get_session,
    list_sessions,
    get_session_messages,
    persist_qa_turn,
    rate_message,
    get_message_rating,
    get_stats,
    get_recent_ratings,
    delete_session,
    delete_message,
    clear_session_messages,
)
from energy_qa_main import MultiStepPipelineClient

# Flask 应用初始化
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# 全局 pipeline 客户端（单例，避免重复初始化）
_pipeline_client = None
_pipeline_lock = threading.Lock()


def get_pipeline_client() -> MultiStepPipelineClient:
    global _pipeline_client
    if _pipeline_client is None:
        with _pipeline_lock:
            if _pipeline_client is None:
                _pipeline_client = MultiStepPipelineClient()
    return _pipeline_client


# 前端页面
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/avatar.jpg")
def avatar():
    """Serve the AI avatar image from the project directory"""
    return send_from_directory(".", "头像.jpg")


# API：会话管理
@app.route("/api/sessions", methods=["GET"])
def api_list_sessions():
    """列出最近会话"""
    try:
        limit = int(request.args.get("limit", 30))
        sessions = list_sessions(limit=limit)
        # datetime 转字符串
        for s in sessions:
            for k in ("created_at", "updated_at"):
                if s.get(k):
                    s[k] = str(s[k])
        return jsonify({"ok": True, "sessions": sessions})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/sessions", methods=["POST"])
def api_create_session():
    """创建新会话"""
    try:
        data = request.get_json(silent=True) or {}
        title = data.get("title", "新对话")
        session_id = create_session(title=title)
        return jsonify({"ok": True, "session_id": session_id})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/sessions/<session_id>", methods=["GET"])
def api_get_session(session_id):
    """获取单个会话信息"""
    try:
        session = get_session(session_id)
        if not session:
            return jsonify({"ok": False, "error": "会话不存在"}), 404
        for k in ("created_at", "updated_at"):
            if session.get(k):
                session[k] = str(session[k])
        return jsonify({"ok": True, "session": session})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/sessions/<session_id>/messages", methods=["GET"])
def api_get_messages(session_id):
    """获取会话的全部消息"""
    try:
        messages = get_session_messages(session_id)
        for m in messages:
            if m.get("created_at"):
                m["created_at"] = str(m["created_at"])
        return jsonify({"ok": True, "messages": messages})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# API：问答
@app.route("/api/ask", methods=["POST"])
def api_ask():
    """
    执行完整三步推理流水线并持久化结果
    请求体: { "session_id": str, "question": str }
    返回:   { "ok": bool, "answer": str, "assistant_message_id": int,
              "rag_sources": list, "elapsed_seconds": float,
              "subtask_results": list }
    """
    try:
        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id", "").strip()
        question   = data.get("question", "").strip()

        if not question:
            return jsonify({"ok": False, "error": "问题不能为空"}), 400
        if not session_id:
            return jsonify({"ok": False, "error": "session_id 不能为空"}), 400

        # 验证会话存在
        if not get_session(session_id):
            return jsonify({"ok": False, "error": "会话不存在，请先创建会话"}), 404

        client = get_pipeline_client()
        result = client.ask(question)

        ids = persist_qa_turn(session_id, question, result)

        return jsonify({
            "ok": True,
            "answer": result["final_answer"],
            "assistant_message_id": ids["assistant_message_id"],
            "rag_sources": result.get("rag_sources", []),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "subtask_results": [
                {
                    "id": r["subtask_id"],
                    "focus": r["focus"],
                    "knowledge_source": r.get("knowledge_source", "none"),
                    "sources": r.get("sources", []),
                    "status": r.get("status", "success"),
                }
                for r in result.get("subtask_results", [])
            ],
        })

    except Exception as e:
        app.logger.exception("API /ask error")
        return jsonify({"ok": False, "error": str(e)}), 500


# API：评价
@app.route("/api/rate", methods=["POST"])
def api_rate():
    """
    对助手消息评价
    请求体: { "message_id": int, "rating": 1 | -1, "feedback": str | null }
    """
    try:
        data = request.get_json(silent=True) or {}
        message_id = data.get("message_id")
        rating     = data.get("rating")
        feedback   = data.get("feedback") or None

        if message_id is None or rating not in (1, -1):
            return jsonify({"ok": False, "error": "参数错误：需要 message_id 和 rating (1 或 -1)"}), 400

        result = rate_message(int(message_id), int(rating), feedback)
        return jsonify({"ok": True, **result})

    except LookupError as e:
        return jsonify({"ok": False, "error": str(e)}), 404
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/rate/<int:message_id>", methods=["GET"])
def api_get_rating(message_id):
    """查询某条消息的评价"""
    try:
        r = get_message_rating(message_id)
        if r:
            r = {k: str(v) if hasattr(v, 'isoformat') else v for k, v in r.items()}
        return jsonify({"ok": True, "rating": r})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# API：历史记录管理（查看 / 删除）
@app.route("/api/history", methods=["GET"])
def api_history():
    """
    分页查询历史会话列表，每条附带首条用户消息预览。
    查询参数: limit (default 50), offset (default 0)
    """
    try:
        limit  = max(1, min(int(request.args.get("limit",  50)), 200))
        offset = max(0, int(request.args.get("offset", 0)))

        from energy_qa_db import _get_connection
        conn = _get_connection()
        try:
            import pymysql.cursors
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        s.id, s.title, s.created_at, s.updated_at, s.message_count,
                        (
                            SELECT m.content FROM messages m
                            WHERE m.session_id = s.id AND m.role = 'user'
                            ORDER BY m.created_at ASC LIMIT 1
                        ) AS first_question,
                        (
                            SELECT COUNT(*) FROM ratings r
                            JOIN messages m2 ON r.message_id = m2.id
                            WHERE m2.session_id = s.id AND r.rating = 1
                        ) AS like_count,
                        (
                            SELECT COUNT(*) FROM ratings r
                            JOIN messages m2 ON r.message_id = m2.id
                            WHERE m2.session_id = s.id AND r.rating = -1
                        ) AS dislike_count
                    FROM sessions s
                    ORDER BY s.updated_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (limit, offset),
                )
                rows = cur.fetchall()

                cur.execute("SELECT COUNT(*) AS total FROM sessions")
                total = cur.fetchone()["total"]
        finally:
            conn.close()

        for r in rows:
            for k in ("created_at", "updated_at"):
                if r.get(k):
                    r[k] = str(r[k])
            if r.get("first_question") and len(r["first_question"]) > 80:
                r["first_question"] = r["first_question"][:80] + "..."

        return jsonify({
            "ok": True,
            "total": total,
            "limit": limit,
            "offset": offset,
            "sessions": rows,
        })
    except Exception as e:
        app.logger.exception("API /history error")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def api_delete_session(session_id):
    """
    删除整个会话（含其所有消息、子任务、评价）。
    返回: { "ok": bool, "deleted": bool }
    """
    try:
        deleted = delete_session(session_id)
        if not deleted:
            return jsonify({"ok": False, "error": "会话不存在"}), 404
        return jsonify({"ok": True, "deleted": True, "session_id": session_id})
    except Exception as e:
        app.logger.exception("DELETE /sessions/<id> error")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/sessions/<session_id>/messages", methods=["DELETE"])
def api_clear_session_messages(session_id):
    """
    清空某会话的全部消息（保留会话本身）。
    返回: { "ok": bool, "deleted_count": int }
    """
    try:
        if not get_session(session_id):
            return jsonify({"ok": False, "error": "会话不存在"}), 404
        count = clear_session_messages(session_id)
        return jsonify({"ok": True, "deleted_count": count, "session_id": session_id})
    except Exception as e:
        app.logger.exception("DELETE /sessions/<id>/messages error")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/messages/<int:message_id>", methods=["DELETE"])
def api_delete_message(message_id):
    """
    删除单条消息（及其子任务详情和评价）。
    返回: { "ok": bool, "deleted": bool }
    """
    try:
        deleted = delete_message(message_id)
        if not deleted:
            return jsonify({"ok": False, "error": "消息不存在"}), 404
        return jsonify({"ok": True, "deleted": True, "message_id": message_id})
    except Exception as e:
        app.logger.exception("DELETE /messages/<id> error")
        return jsonify({"ok": False, "error": str(e)}), 500




@app.route("/api/stats", methods=["GET"])
def api_stats():
    """返回系统统计数据"""
    try:
        stats = get_stats()
        return jsonify({"ok": True, "stats": stats or {}})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/ratings/recent", methods=["GET"])
def api_recent_ratings():
    """返回最近评价列表"""
    try:
        limit = int(request.args.get("limit", 20))
        rows = get_recent_ratings(limit=limit)
        for r in rows:
            if r.get("created_at"):
                r["created_at"] = str(r["created_at"])
        return jsonify({"ok": True, "ratings": rows})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500



# 启动
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    print("\n能源管理在线问答系统启动中...")
    print(f"访问地址：http://0.0.0.0:{port}")
    print("按 Ctrl+C 停止服务\n")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )