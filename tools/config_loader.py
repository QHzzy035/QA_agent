"""
文件名：config_loader.py
介绍：配置加载器，为config文件夹下的所有yaml文件制作加载函数，用于在其他文件中使用。
其中，BASE_DIR代表根目录
"""
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

# 文件根目录
BASE_DIR = Path(__file__).parent.parent.resolve()

# 加载 .env 文件中的环境变量（若 .env 不存在则静默跳过，不覆盖已有的系统环境变量）
load_dotenv(BASE_DIR / ".env")

# 加载配置文件
def load_LLM_settings(config_path: str = str(BASE_DIR/"config"/"LLM.yaml"), encoding = "UTF-8"):
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_rag_settings(config_path: str = str(BASE_DIR/"config"/"rag.yaml"), encoding = "UTF-8"):
    with open(config_path,"r",encoding=encoding)as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_tavily_settings(config_path: str = str(BASE_DIR/"config"/"tavily.yaml"), encoding = "UTF-8"):
    with open(config_path,"r",encoding=encoding)as f:
        return yaml.load(f, Loader=yaml.FullLoader)

LLM_conf = load_LLM_settings()
rag_conf = load_rag_settings()
tavily_conf = load_tavily_settings()

# 拼接数据库的绝对路径
persist_directory = str(BASE_DIR/ rag_conf["vector_database"]["persist_directory"].lstrip("./"))

# 确保重要文件都存在
os.makedirs(persist_directory, exist_ok=True)
os.makedirs(BASE_DIR/"test_data", exist_ok=True)
os.makedirs(BASE_DIR/"prompts", exist_ok=True)