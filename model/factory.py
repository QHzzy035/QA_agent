"""
文件名：factory.py
介绍：模型工厂，集中创建各角色的模型（聊天 / 历史总结 / 查询改写 / 嵌入）。
      之前模型散落在 agent.py / connected_prompts.py / streamlit_file.py / retriever.py 四处，
      且混用 create_agent(字符串) 与 init_chat_model 两种方式，换模型要改多处。
      工厂把模型创建集中在一点，换模型只改 config 里的 YAML。
"""
# 依赖库导入
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

# 依赖文件导入
from tools.config_loader import LLM_conf, rag_conf


class ChatModelFactory:
    """聊天模型工厂：按角色创建并缓存模型实例。"""

    _instances: dict[str, BaseChatModel] = {}

    @classmethod
    def get(cls, role: str) -> BaseChatModel:
        """按角色返回模型实例；config 里没有该角色配置时回退到 chat_model_name。"""
        if role not in cls._instances:
            model_name = LLM_conf.get(f"{role}_model_name", LLM_conf["chat_model_name"])
            cls._instances[role] = init_chat_model(model=model_name)
        return cls._instances[role]


class EmbeddingsFactory:
    """嵌入模型工厂：单例。"""

    _instance: Embeddings | None = None

    @classmethod
    def get(cls) -> Embeddings:
        if cls._instance is None:
            cls._instance = DashScopeEmbeddings(
                model=rag_conf["vector_database"]["embedding_model_name"]
            )
        return cls._instance


# 各角色模型实例（供其他模块直接导入）
chat_model = ChatModelFactory.get("chat")
summarize_model = ChatModelFactory.get("summarize")
rewrite_model = ChatModelFactory.get("rewrite")
embed_model = EmbeddingsFactory.get()