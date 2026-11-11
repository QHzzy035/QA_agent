"""
文件名：web_search.py
介绍：联网搜索工具。把 Tavily 的调用与「搜索结果 -> 提示词文本」的格式化拆开，
      既让工具输出保留标题和 URL（模型才能标注来源），也让格式化逻辑可以脱离
      API Key 单独测试。

      客户端采用惰性创建：模块导入时不要求 TAVILY_API_KEY，只有真正调用
      web_search 时才实例化 TavilySearch。这样无 Key 环境下仍可导入本模块、
      跑纯逻辑单测。
"""
# 依赖库导入
from langchain_community.tools import tool
from langchain_tavily import TavilySearch

# 依赖文件导入
from tools.config_loader import tavily_conf

# 单条搜索结果最多保留多少字符，避免把整个网页塞进上下文
MAX_CONTENT_CHARS = 800

_client = None


def _tavily() -> TavilySearch:
    """返回 Tavily 客户端（首次调用时创建）。"""
    global _client
    if _client is None:
        _client = TavilySearch(
            max_results=tavily_conf.get("max_result", 3),
            topic="general",
        )
    return _client


def format_search_results(result) -> str:
    """把 Tavily 返回值格式化成带标题和 URL 的文本。

    输出形如：
        [1] 标题
        URL: https://...
        正文摘要

    保留 URL 是重点：只拼接正文的话，模型无法在回答里标注来源，
    用户也无从追溯。无法解析或没有结果时返回可读提示，而不是抛异常。
    """
    if not isinstance(result, dict):
        return f"联网搜索返回了无法解析的结果：{result!r}"

    items = result.get("results") or []
    if not isinstance(items, list) or not items:
        return "联网搜索没有返回结果。"

    parts = []
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        content = str(item.get("content") or "").strip()
        if not content and not url:
            continue
        if len(content) > MAX_CONTENT_CHARS:
            content = content[:MAX_CONTENT_CHARS] + "…"
        header = f"[{i}] {title}".rstrip()
        lines = [header]
        if url:
            lines.append(f"URL: {url}")
        if content:
            lines.append(content)
        parts.append("\n".join(lines))

    return "\n\n".join(parts) if parts else "联网搜索没有返回可用内容。"


@tool(description="联网搜索工具 web_search，在无法回答用户问题或用户要求联网搜索时使用")
def web_search(query: str) -> str:
    """联网搜索，返回带标题与 URL 的结果文本；失败时返回错误说明。"""
    try:
        result = _tavily().invoke({"query": query})
    except Exception as e:
        # 不把异常抛给 Agent：联网失败时让模型看到原因并据此回答，
        # 比整轮对话崩掉更可控。
        return f"联网搜索失败：{type(e).__name__}: {e}"
    return format_search_results(result)
