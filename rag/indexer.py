"""
文件名：indexer.py
介绍：文档增量索引管理器。通过 MD5 记录每个文件的索引状态，
      只对新增/变更的文档做向量化，避免每次全量重建浪费 embedding 费用。
"""
# 依赖库导入
import hashlib
import json
import os
from pathlib import Path

# 依赖文件导入
from tools.config_loader import BASE_DIR
from tools.document_service import SUPPORTED_EXTENSIONS, extract_documents
from tools.logger_handler import logger

# 索引记录文件：{文件绝对路径(normpath): md5}
INDEX_FILE = BASE_DIR / "index_record.json"
# 默认文档目录
DATA_DIR = BASE_DIR / "test_data"


def get_file_md5_hex(file_path: str) -> str:
    """计算文件 MD5。"""
    chunk_size = 4096
    md5_obj = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5_obj.update(chunk)
    return md5_obj.hexdigest()


def _load_record(index_file: Path) -> tuple[dict, bool]:
    """读取索引记录，返回 (记录, 是否损坏)。损坏时不抛异常，交由调用方决定重建。"""
    if not index_file.exists():
        return {}, False
    try:
        with open(index_file, "r", encoding="utf-8") as f:
            return json.load(f), False
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"[indexer] 索引记录文件读取失败，将全量重建：{e}")
        return {}, True


def _save_record(record: dict, index_file: Path):
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def _list_data_files(data_dir: Path) -> list[str]:
    """返回数据目录下「受支持格式」文件的 normpath 绝对路径。

    只收支持的类型：否则 .zip / .xlsx 之类会被当纯文本强行解码，
    往向量库里灌一堆乱码 chunk。与 document_service.get_document_list 的过滤保持一致。
    """
    return [
        os.path.normpath(str(p.resolve()))
        for p in sorted(data_dir.glob("*.*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def _index_file(path: str, chroma, splitter) -> int:
    """加载单个文件、分片并写入向量库，返回写入的 chunk 数。

    抽取走 tools.document_service.extract_documents，与界面预览共用同一套实现。
    """
    documents = extract_documents(path)
    if not documents:
        return 0
    chunks = splitter.split_documents(documents)
    if not chunks:
        return 0
    chroma.add_documents(chunks)
    return len(chunks)


def incremental_index(chroma=None, splitter=None, index_file=None, data_dir=None) -> dict:
    """
    增量索引文档目录：
    - 新增文件 -> 向量化入库
    - 变更文件 -> 删除旧 chunk 后重新入库
    - 删除文件 -> 从向量库清理对应 chunk
    返回统计信息 {added, updated, removed, skipped, failed}。

    单个文件解析失败不会中断整轮索引：计入 failed 并在下一轮自动重试。
    chroma / splitter / index_file / data_dir 仅为测试注入预留，默认走真实实现。
    """
    # 延迟导入，避免模块加载时引入重依赖（也让无 Key 环境下可以导入本模块做单测）
    if splitter is None:
        from rag.splitter import splitter
    if chroma is None:
        from rag.retriever import get_chroma
        chroma = get_chroma()

    index_file = Path(index_file) if index_file is not None else INDEX_FILE
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR

    record, corrupt = _load_record(index_file)

    # 首次使用增量索引（记录文件不存在，说明向量库是旧的全量方式建的），
    # 或记录文件损坏无法与向量库对齐时，先清空重建：
    # 否则残留 chunk 会在重新入库时变成重复数据。
    if not index_file.exists() or corrupt:
        chroma.reset_collection()

    current_files = set(_list_data_files(data_dir))

    stats = {"added": [], "updated": [], "removed": [], "skipped": 0, "failed": []}

    # 1. 删除：记录里有、磁盘上已不存在的文件
    for path in list(record.keys()):
        if path not in current_files:
            chroma.delete(where={"source": path})
            del record[path]
            stats["removed"].append(path)

    # 2. 新增 / 变更
    for path in sorted(current_files):
        md5 = get_file_md5_hex(path)
        if path in record and record[path] == md5:
            stats["skipped"] += 1
            continue

        is_update = path in record
        try:
            # 变更：先删旧 chunk，再重新索引
            if is_update:
                chroma.delete(where={"source": path})
            _index_file(path, chroma, splitter)
        except Exception as e:
            # 单文件失败不影响其他文件；不写入 record，下一轮会自动重试
            logger.error(f"[indexer] 索引文件失败，已跳过：{path}，原因：{e}")
            stats["failed"].append(path)
            continue

        record[path] = md5
        stats["updated" if is_update else "added"].append(path)

    _save_record(record, index_file)
    return stats
