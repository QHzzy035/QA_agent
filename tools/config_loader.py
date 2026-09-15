"""
文件名：config_loader.py
介绍：配置加载器，为config文件夹下的所有yaml文件制作加载函数，用于在其他文件中使用。
其中，BASE_DIR代表根目录
"""
import os
import sys
from pathlib import Path
import yaml
from dotenv import load_dotenv

# 文件根目录
BASE_DIR = Path(__file__).parent.parent.resolve()

# 需要从外部注入的 API Key
API_KEYS = ("DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY", "TAVILY_API_KEY")


def _bridge_streamlit_secrets():
    """把 Streamlit secrets 里的 Key 桥接成环境变量。

    部署到 Streamlit Community Cloud 时没有 .env 文件，Key 配在平台的
    secrets 里。桥接之后，本地(.env) 与云端(secrets) 走的是同一套读取逻辑，
    业务代码不需要区分环境。

    只在 streamlit 已被导入时才尝试：避免给脚本、单元测试等非 Web 场景
    引入 streamlit 这个重依赖。streamlit_file.py 中 `import streamlit`
    位于所有业务 import 之前，所以本模块加载时它已在 sys.modules 中。
    """
    st = sys.modules.get("streamlit")
    if st is None:
        return
    try:
        for key in API_KEYS:
            if not os.getenv(key) and key in st.secrets:
                os.environ[key] = str(st.secrets[key])
    except Exception:
        # 没有 secrets.toml、或不在 Streamlit 运行时中访问 secrets 都会抛异常。
        # 属于正常情况（本地用 .env、测试两者都没有），静默跳过。
        pass


# 加载 .env 文件中的环境变量（若 .env 不存在则静默跳过，不覆盖已有的系统环境变量）
load_dotenv(BASE_DIR / ".env")
# 再尝试从 Streamlit secrets 补齐 .env 里没有的 Key（仅云端部署时生效）
_bridge_streamlit_secrets()


def resolve_data_dir() -> Path:
    """返回文档目录：优先 test_data，为空时回退到 demo_data。

    test_data 被 .gitignore 排除，所以新克隆的仓库和云端部署都没有它；
    demo_data 随仓库提供，保证「clone 下来就能问出东西」。
    本地已有测试语料时不受演示文件干扰。
    """
    test_data = BASE_DIR / "test_data"
    if test_data.is_dir() and any(p.is_file() for p in test_data.iterdir()):
        return test_data
    return BASE_DIR / "demo_data"

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
os.makedirs(BASE_DIR/"demo_data", exist_ok=True)
os.makedirs(BASE_DIR/"prompts", exist_ok=True)