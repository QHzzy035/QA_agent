"""
文件名：retriever.py
介绍：定义chroma数据库，并定义返回MMR检索结果的方法
"""
# 依赖库导入
from langchain_community.embeddings import DashScopeEmbeddings
# 依赖文件导入
from rag.splitter import doc_splitter
from langchain_chroma import Chroma
# 导入配置
from tools.config_loader import persist_directory, rag_conf

# 定义数据库
chroma = Chroma(
    collection_name=rag_conf["vector_database"]["collection_name"],
    embedding_function= DashScopeEmbeddings(model=rag_conf["vector_database"]["embedding_model_name"]),
    persist_directory=persist_directory,
)

# 将文档数据上传到数据库
chroma.add_documents(doc_splitter)

# MMR检索方法
def retriever(query: str) -> list:
    results = chroma.max_marginal_relevance_search(query, k=rag_conf["k"], fetch_k=rag_conf["fetch_k"])
    return [(doc.page_content, doc.metadata) for doc in results]

# 测试
if __name__ == '__main__':
    res = retriever("给我读一段《三体》这本书的内容")
    print(res)