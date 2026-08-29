"""
文件名：loader.py
介绍：文档加载器，从提示词文件夹中加载提示词
"""
# 导入依赖
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from tools import config_loader

loader = DirectoryLoader(
    path=config_loader.BASE_DIR / "test_data",
    glob="**/*.*",
    loader_cls=UnstructuredFileLoader,
    loader_kwargs={"encoding": "utf-8"},
)