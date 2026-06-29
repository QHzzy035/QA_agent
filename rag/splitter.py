"""
文件名：splitter.py
介绍：文档切割器，从加载器中读取文档内容并处理
"""
# 依赖库导入
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 依赖文件导入
from rag.loader import documents
# 导入配置
from config import settings

splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=[".",",","!","?","。","，","！","？","\n","\n\n"," ",""],
    length_function=len
)

doc_splitter = splitter.split_documents(documents)