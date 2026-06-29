"""
文件名：connected_prompts.py
介绍：拼接用户问题和检索资料，用于给后续的智能体使用
"""
# 依赖库导入
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage
# 依赖文件导入
from rag.retriever import retriever
from config import settings

# 判断模型
def need_retriever(query: str) -> bool:
    model = ChatTongyi(model=settings.need_retriever_model)
    result = model.invoke(
        [HumanMessage(f"""你是一个查询判断助手。用户有一个私有文档库，包含个人收藏的文章、书籍节选等资料。
  判断用户的问题是否需要查询这个文档库才能回答。

  需要查询的情况：
  - 询问特定书籍/文章的内容（如"三体讲了什么"）
  - 询问文档中的具体信息（如"文档里关于XX的描述"）
  - 需要引用或朗读文档原文

  不需要查询的情况：
  - 日常问候（如"你好"、"谢谢"）
  - 通用知识问题（如"什么是机器学习"，模型本身一般都知道）
  - 闲聊

  只回复"是"或"否"。

  用户问题：{query}""")]
    )
    return "是" in result.content

# 拼接新的提示词
def new_prompt(query: str):
    # 分界点：用模型判断是否需要借助文档库回答问题。
    if not need_retriever(query):
        return query, []
    else:
        # 读取基本提示词
        user_prompts = ""
        add_context = ""
        collected_msg = []

        # 接收检索器返回的结果
        for content, metadata in retriever(query):
            file_name = metadata.get("filename", "未知来源")
            add_context += f"[来源: {file_name}]\n{content}\n\n"
            collected_msg.append({
                "source": file_name,
                "preview":content[:100] + "..."
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
    # p = new_prompt("给我读一段三体这本书的内容。")
    # print(p)
    print(need_retriever("你好"))
