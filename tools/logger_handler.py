"""
文件名：logger_handler.py
介绍：结构化日志，控制台输出 INFO 及以上，文件记录 DEBUG 及以上，按天分割。
"""
# 依赖库导入
import logging
import os
from datetime import datetime

# 依赖文件导入
from tools.config_loader import BASE_DIR

# 日志根目录
LOG_ROOT = str(BASE_DIR / "logs")
os.makedirs(LOG_ROOT, exist_ok=True)

# 日志格式
DEFAULT_LOG_FORMAT = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)


def get_logger(
        name: str = "qa_agent",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file: str | None = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)

    # 文件 handler（按天分割）
    if log_file is None:
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(LOG_ROOT, f"{name}_{date_str}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)

    return logger


# 全局 logger 实例
logger = get_logger()
