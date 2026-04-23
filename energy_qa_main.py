"""
energy_qa_main.py 
架构流程：
1: GLM 对用户问题进行全局解析与任务分解（输出结构化 JSON 子任务列表）
2: LongCat 并发执行各子任务深度推理（每个子任务独立 RAG 检索 + 推理）
3: GLM 综合所有子任务结果与原始问题，生成最终优化答案（含来源标注）

"""

import requests
import json
import os
import sys
import time
import concurrent.futures
from config_loader import load_config
from energy_prompts import (
    DECOMPOSE_SYSTEM_PROMPT,
    SUBTASK_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
)

from energy_qa_db import (
    create_session,
    get_session,
    list_sessions,
    persist_qa_turn,
    prompt_user_rating,
    get_stats,
)

# 向量数据库已禁用（云端纯 API 模式）
EnergyVectorDB = None

# 全局配置 
config = load_config()

# 并发与重试配置 
MAX_CONCURRENT_SUBTASKS = 5   # 最大并发子任务数
MAX_RETRY_ATTEMPTS = 3        # 单个子任务最大重试次数
RETRY_BACKOFF_BASE = 1      # 退避基数（秒），实际等待 base^attempt 秒
SUBTASK_TIMEOUT = 60          # 子任务超时（秒）


# 基础 HTTP 调用层（同步）
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
        """发起一次同步 API 调用，失败时根据 attempt 决定是否重试"""
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


def _build_longcat_client() -> BaseAPIClient:
    cfg = config["longcat"]
    return BaseAPIClient(
        api_key=cfg["api_key"],
        api_url=cfg["api_url"],
        default_model=cfg["default_model"],
        timeout=config["model_params"]["timeout"],
    )


# RAG 检索层（云端纯 API 模式：向量库已禁用）
class RAGRetriever:
    """封装向量库检索，返回带来源标注的知识片段"""

    """云端模式下的 RAG 占位符，始终返回空结果，由网络搜索兜底"""

    def __init__(self):
        self.enabled = False
        print("   RAG 检索：已禁用（云端纯 API 模式，无需本地向量库）")

    def retrieve(self, query: str, top_k: int = None) -> tuple[str, list[dict]]:
        return "（RAG 已禁用）", []


# 网络搜索兜底 
def web_search_fallback(query: str) -> tuple[str, str]:
    """
    当向量库无结果时调用网络搜索作为兜底。
    使用 DuckDuckGo 即时答案 API。
    返回 (展示摘要文本, 用于推理的内容文本)
    如果网络搜索也失败，返回空字符串。
    """
    print(f"      向量库无结果，触发网络搜索：{query[:40]}...")
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": "1", "no_html": "1"},
            timeout=10,
            headers={"User-Agent": "EnergyQA/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()

        snippets = []
        if data.get("AbstractText"):
            snippets.append(f"摘要：{data['AbstractText']}")
        for topic in data.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and topic.get("Text"):
                snippets.append(f"相关：{topic['Text']}")

        if snippets:
            content = "\n".join(snippets)
            display = f"网络搜索结果（查询：{query}）：\n{content[:500]}"
            print(f"      网络搜索成功，获取 {len(snippets)} 条摘要")
            return display, content
        else:
            print(f"      警告：网络搜索未返回有效摘要")
            return "", ""

    except Exception as e:
        print(f"      警告：网络搜索失败：{e}")
        return "", ""


# Step 1：GLM 任务分解
def step1_decompose(user_question: str, glm: BaseAPIClient) -> dict:
    """Step 1：调用 GLM 将用户问题分解为结构化子任务列表"""
    print("\n【Step 1】GLM 正在解析问题并分解子任务...")
    messages = [
        {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
        {"role": "user", "content": f"请分解以下问题：\n\n{user_question}"},
    ]
    raw = glm.call(messages, temperature=0.3, max_tokens=1024)

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    try:
        parsed = json.loads(cleaned)
        subtasks = parsed.get("subtasks", [])
        intent = parsed.get("intent_summary", "未提取到意图")
        print(f"   意图：{intent}")
        print(f"   分解为 {len(subtasks)} 个子任务：")
        for st in subtasks:
            print(f"      [{st['id']}] {st['focus']}：{st['query']}")
        return parsed
    except json.JSONDecodeError:
        print(f"   警告：JSON 解析失败，降级为3个默认子任务。原始输出：{raw[:200]}")
        return {
            "intent_summary": user_question,
            "subtasks": [
                {"id": 1, "focus": "概念与定义", "query": f"请解释以下问题涉及的核心概念和基本原理：{user_question}", "reasoning_hint": "定义、原理"},
                {"id": 2, "focus": "现状与数据", "query": f"请介绍以下问题相关的当前现状、数据和案例：{user_question}", "reasoning_hint": "现状、数据、案例"},
                {"id": 3, "focus": "影响与展望", "query": f"请分析以下问题的影响、挑战和未来展望：{user_question}", "reasoning_hint": "影响、挑战、趋势"},
            ],
        }


# Step 2：LongCat 并发执行子任务
def _build_subtask_prompt(subtask: dict, knowledge_chunks: list[dict], web_content: str) -> str:
    """将子任务、知识库内容和网络搜索内容组合为 LongCat 的提示词"""
    if knowledge_chunks:
        kb_text = "\n\n".join(
            f"【来源文件：{c['source']}】\n{c['text']}"
            for c in knowledge_chunks
        )
        knowledge_section = (
            "===== 知识库内容（请引用时标注「来源：文件名」）=====\n" + kb_text
        )
    elif web_content:
        knowledge_section = (
            "===== 网络搜索补充内容（知识库无相关结果，以下为网络搜索摘要）=====\n"
            + web_content
        )
    else:
        knowledge_section = (
            "===== 参考内容 =====\n"
            "（知识库与网络搜索均未获取到有效内容，请基于通用能源领域知识回答）"
        )

    hint = subtask.get("reasoning_hint", "")
    hint_section = f"\n推理重点：{hint}" if hint else ""

    return (
        f"子任务方向：{subtask['focus']}\n"
        f"具体问题：{subtask['query']}"
        f"{hint_section}\n\n"
        f"{knowledge_section}\n\n"
        "请基于以上内容，对子任务进行深度推理，给出详细分析。"
    )


def _execute_single_subtask(
    subtask: dict,
    longcat: BaseAPIClient,
    retriever: RAGRetriever,   # 保留参数签名兼容性，云端模式下不实际调用
    semaphore,
) -> dict:
    """
    执行单个子任务（云端纯 API 模式：网络搜索兜底 → LongCat 推理），线程安全。
    knowledge_source: "web" | "none"
    """
    task_id = subtask["id"]
    query = subtask["query"]

    with semaphore:
        # 云端模式：跳过向量库，直接使用网络搜索
        knowledge_chunks = []
        print(f"\n   [子任务 {task_id}] 网络搜索：{query[:40]}...")
        _, web_content = web_search_fallback(query)
        knowledge_source = "web" if web_content else "none"
        sources = []

        prompt = _build_subtask_prompt(subtask, knowledge_chunks, web_content)
        messages = [
            {"role": "system", "content": SUBTASK_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        print(f"   [子任务 {task_id}] LongCat 推理中（知识来源：{knowledge_source}）...")
        try:
            answer = longcat.call(messages, temperature=0.5, max_tokens=1500)
            print(f"   [子任务 {task_id}] 完成")
            return {
                "subtask_id": task_id,
                "focus": subtask["focus"],
                "query": query,
                "answer": answer,
                "sources": sources,
                "knowledge_source": knowledge_source,
                "status": "success",
            }
        except Exception as e:
            print(f"   [子任务 {task_id}] 推理失败：{e}")
            return {
                "subtask_id": task_id,
                "focus": subtask["focus"],
                "query": query,
                "answer": f"（推理失败：{e}）",
                "sources": [],
                "knowledge_source": "none",
                "status": "error",
            }


def step2_concurrent_subtasks(
    subtasks: list[dict],
    longcat: BaseAPIClient,
    retriever: RAGRetriever,
) -> list[dict]:
    """Step 2：使用线程池并发执行所有子任务，受信号量控制并发数"""
    import threading
    print(f"\n【Step 2】LongCat 并发执行 {len(subtasks)} 个子任务（最大并发：{MAX_CONCURRENT_SUBTASKS}）...")

    semaphore = threading.Semaphore(MAX_CONCURRENT_SUBTASKS)
    results = [None] * len(subtasks)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SUBTASKS) as executor:
        future_to_idx = {
            executor.submit(
                _execute_single_subtask, st, longcat, retriever, semaphore
            ): i
            for i, st in enumerate(subtasks)
        }

        for future in concurrent.futures.as_completed(future_to_idx, timeout=SUBTASK_TIMEOUT * len(subtasks)):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result(timeout=SUBTASK_TIMEOUT)
            except concurrent.futures.TimeoutError:
                st = subtasks[idx]
                print(f"   [子任务 {st['id']}] 超时")
                results[idx] = {
                    "subtask_id": st["id"],
                    "focus": st["focus"],
                    "query": st["query"],
                    "answer": "（子任务超时）",
                    "sources": [],
                    "knowledge_source": "none",
                    "status": "timeout",
                }
            except Exception as e:
                st = subtasks[idx]
                print(f"   [子任务 {st['id']}] 意外错误：{e}")
                results[idx] = {
                    "subtask_id": st["id"],
                    "focus": st["focus"],
                    "query": st["query"],
                    "answer": f"（意外错误：{e}）",
                    "sources": [],
                    "knowledge_source": "none",
                    "status": "error",
                }

    return [r for r in results if r is not None]


# Step 3：GLM 综合整合
def _build_synthesis_prompt(
    user_question: str,
    intent_summary: str,
    subtask_results: list[dict],
) -> str:
    """构建 Step 3 的综合提示词，按 knowledge_source 区分来源类型"""
    _order = {"rag": 0, "web": 1, "none": 2}
    subtask_results = sorted(subtask_results, key=lambda r: _order.get(r.get("knowledge_source", "none"), 2))
    results_text = ""
    for r in subtask_results:
        ks = r.get("knowledge_source", "none")
        if ks == "rag":
            src_label = f"知识库文件（{', '.join(r['sources'])}）"
        elif ks == "web":
            src_label = "网络搜索（无需标注来源）"
        else:
            src_label = "通用知识（无需标注来源）"

        status_tag = "" if r["status"] == "success" else f"（状态：{r['status']}）"
        results_text += (
            f"--- 子任务 {r['subtask_id']}：{r['focus']} ---\n"
            f"子问题：{r['query']}\n"
            f"知识来源：{src_label}\n"
            f"推理结果{status_tag}：\n{r['answer']}\n\n"
        )

    rag_sources = sorted({s for r in subtask_results if r.get("knowledge_source") == "rag" for s in r["sources"]})
    if rag_sources:
        sources_instruction = f"本次知识库引用文件（答案末尾需列出）：{', '.join(rag_sources)}"
    else:
        sources_instruction = "本次无知识库命中，答案末尾不需要参考来源清单。"

    return (
        f"原始问题：{user_question}\n"
        f"问题意图：{intent_summary}\n\n"
        f"===== 各子任务推理结果 =====\n"
        f"{results_text}"
        f"===== 综合要求 =====\n"
        f"{sources_instruction}\n"
        f"请将以上子任务结果整合为一份完整、高质量的最终答案，遵守整合规则。"
    )


def step3_synthesize(
    user_question: str,
    intent_summary: str,
    subtask_results: list[dict],
    glm: BaseAPIClient,
) -> str:
    """Step 3：调用 GLM 综合所有子任务结果，生成最终优化答案"""
    print("\n【Step 3】GLM 正在综合整合子任务结果...")
    prompt = _build_synthesis_prompt(user_question, intent_summary, subtask_results)
    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    answer = glm.call(messages, temperature=0.4, max_tokens=2048)
    print("   最终答案生成完成")
    return answer


# 主编排器：三步流水线
class MultiStepPipelineClient:
    """
    三步协同推理客户端
      Step 1: GLM  → 任务分解（JSON 子任务列表）
      Step 2: LongCat（并发）→ 子任务深度推理 + RAG 检索
      Step 3: GLM  → 综合整合 + 来源标注
    """

    def __init__(self):
        self.glm = _build_glm_client()
        self.longcat = _build_longcat_client()
        self.retriever = RAGRetriever()
        print(f"\nMultiStepPipelineClient 初始化完成（云端纯 API 模式）")
        print(f"   GLM 模型：{self.glm.default_model}")
        print(f"   LongCat 模型：{self.longcat.default_model}")
        print(f"   RAG 检索：已禁用（网络搜索兜底）")

    def ask(self, user_question: str) -> dict:
        """
        执行完整三步流水线，返回结果字典：
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

        decompose_result = step1_decompose(user_question, self.glm)
        subtasks = decompose_result.get("subtasks", [])
        intent_summary = decompose_result.get("intent_summary", user_question)

        if not subtasks:
            subtasks = [{"id": 1, "focus": "完整问题", "query": user_question, "reasoning_hint": ""}]

        subtask_results = step2_concurrent_subtasks(subtasks, self.longcat, self.retriever)
        final_answer = step3_synthesize(user_question, intent_summary, subtask_results, self.glm)

        all_sources = sorted({s for r in subtask_results for s in r["sources"]})
        rag_sources = sorted({s for r in subtask_results if r.get("knowledge_source") == "rag" for s in r["sources"]})
        elapsed = round(time.time() - t0, 1)
        web_count = sum(1 for r in subtask_results if r.get("knowledge_source") == "web")
        print(f"\n   总耗时：{elapsed}s | 知识库来源：{rag_sources or ['无']} | 网络搜索子任务：{web_count}个")

        return {
            "final_answer": final_answer,
            "intent_summary": intent_summary,
            "subtask_results": subtask_results,
            "all_sources": all_sources,
            "rag_sources": rag_sources,
            "elapsed_seconds": elapsed,
        }


# 交互式对话界面（含 DB 持久化 + 评价）
def _print_divider(char: str = "─", width: int = 60):
    print(char * width)


def _show_stats():
    """打印系统使用统计"""
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
    """
    交互式多轮问答
    - 使用三步协同推理架构
    - 每次问答自动持久化到数据库
    - 回答后提示用户评价（可跳过）
    """
    _print_divider("═")
    print("欢迎使用能源问答系统（多步骤协同推理版 + 数据库记录）")
    print("架构：GLM 任务分解 → LongCat 并发推理 → GLM 综合整合")
    print("退出：输入 q / quit / 退出")
    print("统计：输入 stats 查看使用统计")
    _print_divider("═")

    try:
        client = MultiStepPipelineClient()
    except Exception as e:
        print(f"\n错误：系统初始化失败：{e}")
        return

    #  创建新会话 
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

            #  输出子任务摘要 
            print("\n子任务推理摘要：")
            _print_divider("─", 40)
            for r in result["subtask_results"]:
                status_icon = "[OK]" if r["status"] == "success" else "[FAIL]"
                ks = r.get("knowledge_source", "none")
                if ks == "rag":
                    src_tag = f"知识库（{'、'.join(r['sources'])}）"
                elif ks == "web":
                    src_tag = "网络搜索"
                else:
                    src_tag = "通用知识"
                print(f"  {status_icon} [{r['subtask_id']}] {r['focus']} · {src_tag}")

            #  输出最终答案 
            print("\n最终综合答案：")
            _print_divider("─", 40)
            print(result["final_answer"])
            _print_divider("─", 40)

            #  仅当有知识库命中时显示来源汇总 
            if result["rag_sources"]:
                print(f"\n本次引用知识库文件：")
                for src in result["rag_sources"]:
                    print(f"   • {src}")

            print(f"\n总耗时：{result['elapsed_seconds']}s")

            #  持久化到数据库 
            assistant_message_id = None
            if session_id:
                try:
                    ids = persist_qa_turn(session_id, user_question, result)
                    assistant_message_id = ids["assistant_message_id"]
                    print(f"已保存到数据库（消息 ID: {assistant_message_id}）")
                except Exception as db_err:
                    print(f"警告：数据库保存失败（不影响问答）：{db_err}")

            #  用户评价 
            if assistant_message_id:
                prompt_user_rating(assistant_message_id)

        except Exception as e:
            print(f"\n错误：问答流程出错：{type(e).__name__} - {e}")


# 入口
if __name__ == "__main__":
    interactive_chat()