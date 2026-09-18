"""
文件名：indexer.py
介绍：文档增量索引管理器。通过 MD5 记录每个文件的索引状态，
      只对新增/变更的文档做向量化，避免每次全量重建浪费 embedding 费用。

      记录里同时保存**建库时使用的嵌入模型名**。这一项不是装饰：
      模型一换，新旧向量就不在同一个空间里，直接增量索引会得到
      「库里有向量、检索却完全不准」的静默错误——不报错，只是结果变差。
"""
# 依赖库导入
import hashlib
import json
import os
from pathlib import Path

# 依赖文件导入
from tools.config_loader import BASE_DIR, rag_conf, resolve_data_dir
from tools.document_service import SUPPORTED_EXTENSIONS, extract_documents
from tools.logger_handler import logger

# 索引记录文件，格式见 _save_record
INDEX_FILE = BASE_DIR / "index_record.json"

# 记录格式版本：1 = 旧的扁平 {路径: md5}；2 = 带 _meta 的新格式
INDEX_FORMAT_VERSION = 2


def get_file_md5_hex(file_path: str) -> str:
    """计算文件 MD5。"""
    chunk_size = 4096
    md5_obj = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5_obj.update(chunk)
    return md5_obj.hexdigest()


def _load_record(index_file: Path, embed_model: str) -> tuple[dict, str | None]:
    """读取索引记录，返回 (文件记录, 需要全量重建的原因)。

    返回原因字符串表示应当清空重建，共三种情况：

    1. 记录文件不存在 —— 首次索引
    2. 记录损坏（JSON 解析失败）
    3. **记录里的嵌入模型与当前配置不一致** —— 换了嵌入模型，旧向量与新的
       查询向量不在同一空间，继续增量索引会得到静默的错误结果
    4. 旧版扁平格式（不含模型信息）—— 无从判断当初的向量空间，只能重建
    """
    if not index_file.exists():
        return {}, "索引记录不存在（首次索引）"

    try:
        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"[indexer] 索引记录文件读取失败，将全量重建：{e}")
        return {}, "索引记录损坏"

    # 旧格式是扁平的 {路径: md5}，没有 _meta —— 判不出嵌入模型，只能重建
    if not isinstance(data, dict) or "files" not in data:
        return {}, "索引记录是旧格式（不含嵌入模型信息），无法确认向量空间一致"

    recorded_model = (data.get("_meta") or {}).get("embed_model")
    if recorded_model != embed_model:
        return {}, (f"嵌入模型已变更：记录里为 {recorded_model!r}，"
                    f"当前配置为 {embed_model!r}")

    return data["files"], None


def _save_record(files: dict, index_file: Path, embed_model: str):
    """写入索引记录。

    格式：{"_meta": {"version": 2, "embed_model": "..."}, "files": {路径: md5}}
    把版本与嵌入模型放在 _meta 下，是为了让 files 保持「纯路径 → md5」，
    遍历删除时不会把元数据误当成文件路径。
    """
    payload = {
        "_meta": {"version": INDEX_FORMAT_VERSION, "embed_model": embed_model},
        "files": files,
    }
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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


def incremental_index(chroma=None, splitter=None, index_file=None, data_dir=None,
                      embed_model: str | None = None) -> dict:
    """
    增量索引文档目录：
    - 新增文件 -> 向量化入库
    - 变更文件 -> 删除旧 chunk 后重新入库
    - 删除文件 -> 从向量库清理对应 chunk
    - 换了嵌入模型 -> 全量重建（旧向量与新查询不在同一空间）
    返回统计信息 {added, updated, removed, skipped, failed}。

    单个文件解析失败不会中断整轮索引：计入 failed 并在下一轮自动重试。
    chroma / splitter / index_file / data_dir / embed_model 仅为注入预留，
    默认走真实实现。
    """
    # 延迟导入，避免模块加载时引入重依赖（也让无 Key 环境下可以导入本模块做单测）
    if splitter is None:
        from rag.splitter import splitter
    if chroma is None:
        from rag.retriever import get_chroma
        chroma = get_chroma()

    index_file = Path(index_file) if index_file is not None else INDEX_FILE
    data_dir = Path(data_dir) if data_dir is not None else resolve_data_dir()
    if embed_model is None:
        embed_model = rag_conf["vector_database"]["embedding_model_name"]

    record, rebuild_reason = _load_record(index_file, embed_model)

    # 需要清空重建：首次索引 / 记录损坏 / 换了嵌入模型 / 记录是旧格式。
    # 必须重建而不是接着增量——否则残留的旧 chunk 会与新数据混在同一个库里，
    # 既产生重复，又因向量空间不一致让检索结果完全失真。
    if rebuild_reason:
        logger.info(f"[indexer] 全量重建：{rebuild_reason}")
        chroma.reset_collection()
        record = {}

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

    _save_record(record, index_file, embed_model)
    return stats
