"""
文件名：retriever.py
介绍：定义chroma数据库，并定义返回MMR检索结果的方法

注意：数据库实例采用惰性创建（get_chroma）。原先在模块顶层直接实例化 Chroma，
      而 Chroma 会立即构造嵌入模型，导致任何 import 本模块的代码都要求
      DASHSCOPE_API_KEY。改为惰性后，纯逻辑模块可以正常导入，
      向量库依赖也能在测试中通过参数注入替换。
      `from rag.retriever import chroma` 的写法保持不变。
"""
# 依赖库导入
# 依赖文件导入
from langchain_chroma import Chroma
# 导入配置
from tools.config_loader import persist_directory, rag_conf

_chroma = None


def get_chroma() -> Chroma:
    """返回全局 Chroma 实例（首次调用时创建）。"""
    global _chroma
    if _chroma is None:
        from model.factory import embed_model

        _chroma = Chroma(
            collection_name=rag_conf["vector_database"]["collection_name"],
            embedding_function=embed_model,
            persist_directory=persist_directory,
        )
    return _chroma


# MMR检索方法
def retriever(query: str, k: int = None, fetch_k: int = None) -> list:
    if k is None:
        k = rag_conf["k"]
    if fetch_k is None:
        fetch_k = rag_conf["fetch_k"]
    results = get_chroma().max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
    return [(doc.page_content, doc.metadata) for doc in results]


def __getattr__(name: str):
    """PEP 562 模块级惰性属性：兼容旧的 `from rag.retriever import chroma`。"""
    if name == "chroma":
        return get_chroma()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# 测试
if __name__ == '__main__':
    res = retriever("给我读一段《三体》这本书的内容")
    print(res)
