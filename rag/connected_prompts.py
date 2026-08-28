"""
文件名：connected_prompts.py
介绍：拼接用户问题和检索资料，用于给后续的智能体使用
"""
# 依赖库导入
from pathlib import Path
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
# 依赖文件导入
from rag.retriever import retriever
from tools.config_loader import LLM_conf

# 查询改写模型（复用主对话模型）
rewrite_model = init_chat_model(model=LLM_conf["chat_model_name"])

# 明显不需要检索文档库的问题（用简单规则识别，避免 LLM 判断把"通用知识"误判为不需要）
NO_RETRIEVAL_PATTERNS = [
    # 身份 / 能力
    "你是谁", "你叫什么", "你能做什么", "你能干什么", "介绍一下你", "你会什么",
    # 实时信息
    "现在几点", "现在几点了", "今天几号", "今天星期几", "今天日期", "现在时间",
    "今天天气", "明天天气", "今天多少度",
    # 闲聊问候
    "你好", "您好", "谢谢", "再见", "早上好", "晚上好", "晚安", "嗨",
]


def is_no_retrieval(query: str) -> bool:
    """判断问题是否明显不需要检索文档库（身份/实时信息/闲聊问候）。"""
    return any(p in query for p in NO_RETRIEVAL_PATTERNS)


def rewrite_query(query: str, history: list) -> str:
    """结合对话历史，把用户问题里的指代/省略改写成独立完整的检索查询。"""
    history_text = ""
    for msg in history:
        if isinstance(msg, HumanMessage):
            history_text += f"用户：{msg.content}\n"
        else:
            history_text += f"助手：{msg.content}\n"

    prompt = f"""你是一个查询改写助手。用户正在和一个文档问答系统进行多轮对话，系统需要先检索文档库再回答用户的问题。

请把用户当前的问题，结合对话历史，改写成一个独立、完整、不含指代的查询，方便直接用于文档检索。

规则：
- 如果当前问题已经独立完整（不依赖历史），直接原样返回；
- 如果当前问题包含指代（如"它""那个""这个""这"）或省略了主语，请结合历史补全；
- 只输出改写后的查询文本，不要输出任何解释。

对话历史：
{history_text}

用户当前问题：{query}

改写后的查询："""

    result = rewrite_model.invoke([HumanMessage(prompt)])
    return result.content.strip()


# 拼接新的提示词
def new_prompt(query: str, history: list = None):
    # 规则识别明显不需要检索的问题（你是谁、现在几点、闲聊等）
    if is_no_retrieval(query):
        return query, []

    # 查询改写：多轮对话时把指代/省略改写成独立查询，提升检索准确性
    search_query = query
    if history:
        search_query = rewrite_query(query, history)

    user_prompts = ""
    add_context = ""
    collected_msg = []

    # 接收检索器返回的结果（用改写后的查询检索）
    for content, metadata in retriever(search_query):
        file_name = Path(metadata.get("source", "未知来源")).name
        add_context += f"[来源: {file_name}]\n{content}\n\n"
        collected_msg.append({
            "source": file_name,
            "preview": content[:100] + "..."
        })

    # 如果没有找到相关的资料
    if not add_context:
        return user_prompts + f"用户问题：{query}以下是问题相关的信息：无", []

    prompt = user_prompts + f"""用户问题：{query}\n以下是问题相关的信息：\n{add_context}
        请基于以上资料回答问题。如果资料不足，请说明情况并使用 web_search 工具进行联网搜索。
    """
    return prompt, collected_msg


# 测试
if __name__ == '__main__':
    p, info = new_prompt("给我读一段三体这本书的内容")
    print(p)
    print(info)