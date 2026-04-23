"""
energy_qa_main.py - 能源管理在线问答系统（直接调用 GLM）
架构：GLM 直接回答用户问题，无多步骤流水线
"""

import requests
import json
import os
import time
from config_loader import load_config
from energy_qa_db import (
    create_session,
    get_session,
    list_sessions,
    persist_qa_turn,
    prompt_user_rating,
    get_stats,
)

# 全局配置
config = load_config()

# 重试配置
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 1


# ── 直接问答系统提示词 ──────────────────────────────────────────────────────────
DIRECT_QA_SYSTEM_PROMPT = """你是一位资深能源行业专家，拥有丰富的政策、技术与市场知识。

【回答规范】
1. 结构清晰：根据问题复杂程度，合理分段，必要时使用小标题
2. 专业准确：涉及标准时给出编号（如 GB/T、IEC），涉及政策时给出文件全称
3. 数据规范：引用数据须注明来源年份，估算须明确标注
4. 诚实可靠：无法确认的信息明确说明，不捏造数字或文件名
5. 语言简洁：避免冗余表述，直接给出有价值的分析和结论"""


# ── 基础 HTTP 调用层 ────────────────────────────────────────────────────────────
class BaseAPIClient:
    """底层 HTTP 调用，含重试逻辑"""

    def __init__(self, api_key: str, api_url: str, default_model: str, timeout: int):
        self.api_key = api_key
        self.api_url = api_url
        self.default_model = default_model
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def call(
        self,
        messages: list,
        model: str = None,
        temperature: float = 0.7,
        top_p: float = 0.8,
        max_tokens: int = 2048,
        attempt: int = 0,
    ) -> str:
        body = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            resp = requests.post(
                url=self.api_url,
                headers=self.headers,
                json=body,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = resp.json()
            choices = result.get("choices", [])
            if not choices:
                raise RuntimeError(f"返回无 choices：{json.dumps(result, ensure_ascii=False)[:300]}")
            return choices[0]["message"]["content"]

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < MAX_RETRY_ATTEMPTS - 1:
                wait = RETRY_BACKOFF_BASE ** (attempt + 1)
                print(f"   警告：网络异常，{wait:.1f}s 后重试（第 {attempt+1}/{MAX_RETRY_ATTEMPTS-1} 次）：{e}")
                time.sleep(wait)
                return self.call(messages, model, temperature, top_p, max_tokens, attempt + 1)
            raise ConnectionError(f"调用 {self.api_url} 失败（已重试 {MAX_RETRY_ATTEMPTS} 次）：{e}")

        except requests.exceptions.HTTPError as e:
            code = resp.status_code
            detail = resp.text[:200]
            suffix = "（API Key 无效/过期）" if code == 401 else ""
            if code in (429, 500, 502, 503) and attempt < MAX_RETRY_ATTEMPTS - 1:
                wait = RETRY_BACKOFF_BASE ** (attempt + 1)
                print(f"   警告：HTTP {code}，{wait:.1f}s 后重试（第 {attempt+1}/{MAX_RETRY_ATTEMPTS-1} 次）")
                time.sleep(wait)
                return self.call(messages, model, temperature, top_p, max_tokens, attempt + 1)
            raise RuntimeError(f"HTTP {code}{suffix}：{detail}")

        except (json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(f"响应解析失败：{e}")


def _build_glm_client() -> BaseAPIClient:
    cfg = config["glm"]
    return BaseAPIClient(
        api_key=cfg["api_key"],
        api_url=cfg["api_base_url"].rstrip("/") + "/chat/completions",
        default_model=cfg["default_model"],
        timeout=config["model_params"]["timeout"],
    )


# ── 主客户端 ────────────────────────────────────────────────────────────────────
class MultiStepPipelineClient:
    """
    直接调用 GLM 回答问题（保留原类名以兼容 app.py）
    """

    def __init__(self):
        self.glm = _build_glm_client()
        print(f"\nQA Client 初始化完成")
        print(f"   GLM 模型：{self.glm.default_model}")

    def ask(self, user_question: str) -> dict:
        """
        直接调用 GLM 回答，返回与原接口兼容的结果字典：
        {
          "final_answer": str,
          "intent_summary": str,
          "subtask_results": list,
          "all_sources": list,
          "rag_sources": list,
          "elapsed_seconds": float,
        }
        """
        t0 = time.time()
        print(f"\n【GLM】正在回答：{user_question[:50]}...")

        messages = [
            {"role": "system", "content": DIRECT_QA_SYSTEM_PROMPT},
            {"role": "user", "content": user_question},
        ]

        answer = self.glm.call(
            messages,
            temperature=config["model_params"]["temperature"],
            top_p=config["model_params"]["top_p"],
            max_tokens=config["model_params"]["max_tokens"],
        )

        elapsed = round(time.time() - t0, 1)
        print(f"   回答完成，耗时 {elapsed}s")

        return {
            "final_answer": answer,
            "intent_summary": user_question,
            "subtask_results": [],
            "all_sources": [],
            "rag_sources": [],
            "elapsed_seconds": elapsed,
        }


# ── 交互式命令行界面 ─────────────────────────────────────────────────────────────
def _print_divider(char: str = "─", width: int = 60):
    print(char * width)


def _show_stats():
    try:
        stats = get_stats()
        if stats:
            print("\n系统统计：")
            print(f"   总会话数：{stats.get('total_sessions', 0)}")
            print(f"   总提问数：{stats.get('total_questions', 0)}")
            avg_rt = stats.get('avg_response_time')
            print(f"   平均响应时间：{avg_rt:.1f}s" if avg_rt else "   平均响应时间：N/A")
            print(f"   点赞：{stats.get('total_likes', 0)}  踩：{stats.get('total_dislikes', 0)}")
    except Exception as e:
        print(f"   警告：统计查询失败：{e}")


def interactive_chat():
    _print_divider("═")
    print("欢迎使用能源问答系统（GLM 直接问答版）")
    print("退出：输入 q / quit / 退出")
    print("统计：输入 stats 查看使用统计")
    _print_divider("═")

    try:
        client = MultiStepPipelineClient()
    except Exception as e:
        print(f"\n错误：系统初始化失败：{e}")
        return

    try:
        session_id = create_session()
        print(f"新会话已创建（ID: {session_id[:8]}...）")
    except Exception as e:
        print(f"警告：数据库连接失败，将以无持久化模式运行：{e}")
        session_id = None

    while True:
        _print_divider()
        user_question = input("\n请输入你的问题：").strip()

        if user_question.lower() in ["q", "quit", "退出"]:
            print("已退出系统，再见！")
            break

        if user_question.lower() == "stats":
            _show_stats()
            continue

        if not user_question:
            print("警告：问题不能为空，请重新输入！")
            continue

        try:
            result = client.ask(user_question)

            print("\n回答：")
            _print_divider("─", 40)
            print(result["final_answer"])
            _print_divider("─", 40)
            print(f"\n总耗时：{result['elapsed_seconds']}s")

            assistant_message_id = None
            if session_id:
                try:
                    ids = persist_qa_turn(session_id, user_question, result)
                    assistant_message_id = ids["assistant_message_id"]
                    print(f"已保存到数据库（消息 ID: {assistant_message_id}）")
                except Exception as db_err:
                    print(f"警告：数据库保存失败（不影响问答）：{db_err}")

            if assistant_message_id:
                prompt_user_rating(assistant_message_id)

        except Exception as e:
            print(f"\n错误：问答流程出错：{type(e).__name__} - {e}")


# 入口
if __name__ == "__main__":
    interactive_chat()