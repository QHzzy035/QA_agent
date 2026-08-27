"""
文件名：connected_prompts.py
介绍：拼接用户问题和检索资料，用于给后续的智能体使用
"""
# 依赖库导入
from pathlib import Path
# 依赖文件导入
from rag.retriever import retriever


# 拼接新的提示词
def new_prompt(query: str):
    user_prompts = ""
    add_context = ""
    collected_msg = []

    # 接收检索器返回的结果
    for content, metadata in retriever(query):
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
