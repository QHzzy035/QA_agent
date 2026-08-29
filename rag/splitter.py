"""
文件名：splitter.py
介绍：文档切割器，从加载器中读取文档内容并处理
"""
# 依赖库导入
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 导入配置
from tools.config_loader import rag_conf

splitter = RecursiveCharacterTextSplitter(
    chunk_size=rag_conf["splitter"]["chunk_size"],
    chunk_overlap=rag_conf["splitter"]["chunk_overlap"],
    separators=[".",",","!","?","。","，","！","？","\n","\n\n"," ",""],
    length_function=len
)