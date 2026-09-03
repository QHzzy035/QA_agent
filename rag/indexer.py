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

from langchain_community.document_loaders import UnstructuredFileLoader

# 依赖文件导入
from tools.config_loader import BASE_DIR

# 索引记录文件：{文件绝对路径(normpath): md5}
INDEX_FILE = BASE_DIR / "index_record.json"


def get_file_md5_hex(file_path: str) -> str:
    """计算文件 MD5。"""
    chunk_size = 4096
    md5_obj = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5_obj.update(chunk)
    return md5_obj.hexdigest()


def _load_record() -> dict:
    if not INDEX_FILE.exists():
        return {}
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_record(record: dict):
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def _list_data_files() -> list[str]:
    """返回 test_data 目录下所有文件的 normpath 绝对路径。"""
    data_dir = BASE_DIR / "test_data"
    return [
        os.path.normpath(str(p.resolve()))
        for p in data_dir.glob("*.*")
        if p.is_file()
    ]


def _index_file(path: str, chroma, splitter) -> int:
    """加载单个文件、分片并写入向量库，返回写入的 chunk 数。"""
    documents = UnstructuredFileLoader(path, encoding="utf-8").load()
    if not documents:
        return 0
    chunks = splitter.split_documents(documents)
    if not chunks:
        return 0
    chroma.add_documents(chunks)
    return len(chunks)


def incremental_index() -> dict:
    """
    增量索引 test_data 目录：
    - 新增文件 -> 向量化入库
    - 变更文件 -> 删除旧 chunk 后重新入库
    - 删除文件 -> 从向量库清理对应 chunk
    返回统计信息 {added, updated, removed, skipped}。
    """
    # 延迟导入，避免模块加载时引入重依赖
    from rag.splitter import splitter
    from rag.retriever import chroma

    # 首次使用增量索引：记录文件不存在，说明向量库是旧的全量方式建的，
    # 先清空重建，保证 source 路径格式统一，后续才能正确增量。
    first_run = not INDEX_FILE.exists()
    if first_run:
        chroma.reset_collection()

    record = _load_record()
    current_files = set(_list_data_files())

    stats = {"added": [], "updated": [], "removed": [], "skipped": 0}

    # 1. 删除：记录里有、磁盘上已不存在的文件
    for path in list(record.keys()):
        if path not in current_files:
            chroma.delete(where={"source": path})
            del record[path]
            stats["removed"].append(path)

    # 2. 新增 / 变更
    for path in current_files:
        md5 = get_file_md5_hex(path)
        if path not in record:
            _index_file(path, chroma, splitter)
            record[path] = md5
            stats["added"].append(path)
        elif record[path] != md5:
            # 变更：先删旧 chunk，再重新索引
            chroma.delete(where={"source": path})
            _index_file(path, chroma, splitter)
            record[path] = md5
            stats["updated"].append(path)
        else:
            stats["skipped"] += 1

    _save_record(record)
    return stats