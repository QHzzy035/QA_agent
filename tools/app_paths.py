"""
文件名：app_paths.py
介绍：应用路径解析。区分「只读资源目录」与「可写数据目录」，
      兼容源码运行和 PyInstaller 打包后的目录结构。

设计约束：
- 模块导入时不创建目录，避免测试/CI 出现副作用；
- 数据目录可用 QA_AGENT_DATA_DIR 覆盖，便于测试和便携模式；
- 源码运行默认 <项目根>/data；打包后默认用户数据目录。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "QAAgent"
DATA_DIR_ENV = "QA_AGENT_DATA_DIR"


def is_frozen() -> bool:
    """是否运行在 PyInstaller 等打包环境中。"""
    return bool(getattr(sys, "frozen", False))


def resource_dir() -> Path:
    """只读资源根目录：prompts / config / demo_data 等。"""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def default_data_dir() -> Path:
    """可写数据的默认根目录（不创建目录）。"""
    if not is_frozen():
        # 开发模式放在项目内，便于查看和清理；data/ 已在 .gitignore
        return resource_dir() / "data"

    if sys.platform == "win32":
        base = Path(os.getenv("LOCALAPPDATA") or Path.home() / "AppData/Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Application Support"
    else:
        base = Path(os.getenv("XDG_DATA_HOME") or Path.home() / ".local/share")
    return base / APP_NAME


def data_dir() -> Path:
    """返回可写数据根目录并确保存在。

    QA_AGENT_DATA_DIR 优先，便于测试、便携模式和未来桌面版切换目录。
    """
    override = os.getenv(DATA_DIR_ENV)
    root = Path(override).expanduser() if override else default_data_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def libraries_dir() -> Path:
    """虚拟知识库的数据目录。"""
    root = data_dir() / "libraries"
    root.mkdir(parents=True, exist_ok=True)
    return root


def settings_file() -> Path:
    """用户设置文件（不存放 API Key）。"""
    return data_dir() / "settings.json"


def logs_dir() -> Path:
    """运行日志目录。"""
    root = data_dir() / "logs"
    root.mkdir(parents=True, exist_ok=True)
    return root
