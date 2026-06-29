import os
from pathlib import Path

# 文件根目录
BASE_DIR = Path(__file__).parent.parent.resolve()

# LLM配置
chat_model_name = "deepseek-chat"
summarize_model_name = "deepseek-chat"
need_retriever_model = "qwen-turbo"

# 文本切割器配置
chunk_size = 500
chunk_overlap = 50

# 向量数据库配置
embedding_model_name = "text-embedding-v4"
collection_name = "data"
persist_directory = str(BASE_DIR/"chroma_data")

# 相似度匹配配置
k = 3
fetch_k = 10

# tavily配置
max_result = 3

# 确保重要文件都存在
os.makedirs(persist_directory, exist_ok=True)
os.makedirs(BASE_DIR/"test_data", exist_ok=True)
os.makedirs(BASE_DIR/"prompts", exist_ok=True)