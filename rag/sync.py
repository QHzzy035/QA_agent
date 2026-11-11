"""
文件名：sync.py
介绍：虚拟知识库的同步引擎与文档状态推导。

同步只做增量：
- 新增文件 -> 抽取、切分、写入 Chroma；
- 内容变化 -> 删旧 chunk 再写新 chunk；
- 磁盘已无 -> 删除 chunk 与记录；
- 已移出（排除清单）-> 跳过，并清理可能残留的 chunk；
- 来源离线/禁用 -> 保留记录，不误删；
- 嵌入模型或记录版本变化 -> 全量重建。

状态推导供 UI 使用，不依赖 Streamlit。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from rag.scanner import ScanConfig, scan_source
from tools.library import Library

logger = logging.getLogger(__name__)

INDEX_FORMAT_VERSION = 3
EXCLUDED_FORMAT_VERSION = 1

STATE_INDEXED = "indexed"
STATE_STALE = "stale"
STATE_UNINDEXED = "unindexed"
STATE_EXCLUDED = "excluded"
STATE_MISSING = "missing"
STATE_OFFLINE = "offline"
STATE_DISABLED = "disabled"
STATE_UNKNOWN = "unknown"

STATE_LABELS = {
    STATE_INDEXED: "已索引",
    STATE_STALE: "待更新",
    STATE_UNINDEXED: "未索引",
    STATE_EXCLUDED: "已移出",
    STATE_MISSING: "文件缺失",
    STATE_OFFLINE: "来源离线",
    STATE_DISABLED: "来源已禁用",
    STATE_UNKNOWN: "状态未知",
}


class SyncError(Exception):
    """同步/记录层的基类异常。"""


class ScanLimitExceeded(SyncError):
    """来源文件数超过上限，拒绝部分索引。"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SyncError(f"JSON 格式不正确：{path}")
    return data


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _file_md5(path: Path) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(4096):
            md5.update(chunk)
    return md5.hexdigest()


def _split_key(doc_key: str) -> tuple[str, str]:
    source_id, sep, rel = str(doc_key).partition("/")
    if not sep or not source_id or not rel:
        raise SyncError(f"文档键格式不正确：{doc_key}")
    return source_id, rel


def _rel_from_key(doc_key: str, source_id: str) -> Path | None:
    prefix = source_id + "/"
    if not doc_key.startswith(prefix):
        return None
    rel = doc_key[len(prefix):]
    if not rel:
        return None
    return Path(rel)


@dataclass(frozen=True)
class DocRecord:
    md5: str
    size: int
    mtime: float
    chunks: int = 0
    indexed_at: str = ""

    def to_dict(self) -> dict:
        return {
            "md5": self.md5,
            "size": self.size,
            "mtime": self.mtime,
            "chunks": self.chunks,
            "indexed_at": self.indexed_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DocRecord":
        try:
            return cls(
                md5=str(data["md5"]),
                size=int(data.get("size", 0)),
                mtime=float(data.get("mtime", 0)),
                chunks=int(data.get("chunks", 0)),
                indexed_at=str(data.get("indexed_at") or ""),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise SyncError(f"索引记录字段无效：{data!r}") from e


@dataclass
class IndexRecordState:
    files: dict[str, DocRecord] = field(default_factory=dict)
    rebuild_reason: str | None = None


def load_index_record(
    library: Library,
    embed_model: str | None = None,
) -> IndexRecordState:
    """读取索引记录；返回 (files, 需要重建的原因)。"""
    path = library.index_file
    if not path.exists():
        return IndexRecordState({}, "first")

    try:
        data = _read_json(path)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("[sync] 索引记录读取失败：%s", e)
        return IndexRecordState({}, "corrupt")

    if "files" not in data:
        return IndexRecordState({}, "old_format")

    meta = data.get("_meta")
    if not isinstance(meta, dict):
        return IndexRecordState({}, "old_format")

    try:
        version = int(meta.get("version", 0))
    except (TypeError, ValueError):
        return IndexRecordState({}, "old_format")
    if version != INDEX_FORMAT_VERSION:
        return IndexRecordState({}, "old_format")

    recorded_library = meta.get("library_id")
    if recorded_library not in (None, library.id):
        return IndexRecordState({}, "library_mismatch")

    if embed_model is not None and meta.get("embed_model") != embed_model:
        return IndexRecordState({}, "embedding_model_changed")

    files: dict[str, DocRecord] = {}
    raw_files = data.get("files") or {}
    if not isinstance(raw_files, dict):
        return IndexRecordState({}, "corrupt")

    for key, value in raw_files.items():
        if not isinstance(key, str) or "/" not in key or not isinstance(value, dict):
            continue
        try:
            files[key] = DocRecord.from_dict(value)
        except SyncError:
            continue

    return IndexRecordState(files, None)


def save_index_record(
    library: Library,
    files: dict[str, DocRecord],
    embed_model: str,
) -> None:
    payload = {
        "_meta": {
            "version": INDEX_FORMAT_VERSION,
            "embed_model": embed_model,
            "library_id": library.id,
        },
        "files": {
            key: files[key].to_dict()
            for key in sorted(files)
        },
    }
    _atomic_write_json(library.index_file, payload)


def load_excluded(library: Library) -> dict[str, dict]:
    path = library.excluded_file
    if not path.exists():
        return {}
    try:
        data = _read_json(path)
    except (json.JSONDecodeError, OSError) as e:
        raise SyncError(f"排除记录读取失败：{e}") from e

    items = data.get("items")
    if not isinstance(items, dict):
        raise SyncError("excluded.json 的 items 必须是对象")
    return {
        str(key): dict(value)
        for key, value in items.items()
        if isinstance(value, dict)
    }


def save_excluded(library: Library, items: dict[str, dict]) -> None:
    payload = {
        "version": EXCLUDED_FORMAT_VERSION,
        "items": dict(sorted(items.items())),
    }
    _atomic_write_json(library.excluded_file, payload)


def exclude_document(
    library: Library,
    source_id: str,
    rel_path,
    reason: str = "manual",
) -> str:
    """把文档加入排除清单；文件本身不动。调用方随后应执行同步或删向量。"""
    key = library.doc_key(source_id, rel_path)
    items = load_excluded(library)
    items[key] = {"excluded_at": _now(), "reason": reason or "manual"}
    save_excluded(library, items)
    return key


def restore_document(library: Library, source_id: str, rel_path) -> bool:
    """从排除清单移除；返回是否真的移除了一个条目。"""
    key = library.doc_key(source_id, rel_path)
    items = load_excluded(library)
    existed = key in items
    if existed:
        del items[key]
        save_excluded(library, items)
    return existed


def is_excluded(library: Library, doc_key: str) -> bool:
    return doc_key in load_excluded(library)


def list_excluded(library: Library) -> list[dict]:
    items = load_excluded(library)
    result = []
    for key in sorted(items):
        source_id, sep, rel = str(key).partition("/")
        if not sep or not source_id or not rel:
            continue
        result.append({
            "doc_key": key,
            "source_id": source_id,
            "rel_path": rel,
            "excluded_at": items[key].get("excluded_at", ""),
            "reason": items[key].get("reason", ""),
        })
    return result


@dataclass
class SyncStats:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    skipped: int = 0
    excluded: int = 0
    failed: list[dict] = field(default_factory=list)
    offline_sources: list[str] = field(default_factory=list)
    disabled_sources: list[str] = field(default_factory=list)
    scan_errors: list[dict] = field(default_factory=list)
    scanned: int = 0
    rebuild_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "added": list(self.added),
            "updated": list(self.updated),
            "removed": list(self.removed),
            "skipped": self.skipped,
            "excluded": self.excluded,
            "failed": list(self.failed),
            "offline_sources": list(self.offline_sources),
            "disabled_sources": list(self.disabled_sources),
            "scan_errors": list(self.scan_errors),
            "scanned": self.scanned,
            "rebuild_reason": self.rebuild_reason,
        }


@dataclass(frozen=True)
class DocumentInfo:
    doc_key: str
    source_id: str
    source_label: str
    rel_path: Path
    name: str
    state: str
    exists: bool
    size: int | None = None
    mtime: float | None = None
    chunks: int = 0

    def to_dict(self) -> dict:
        return {
            "doc_key": self.doc_key,
            "source_id": self.source_id,
            "source_label": self.source_label,
            "rel_path": self.rel_path.as_posix(),
            "name": self.name,
            "state": self.state,
            "state_label": STATE_LABELS.get(self.state, self.state),
            "exists": self.exists,
            "size": self.size,
            "mtime": self.mtime,
            "chunks": self.chunks,
        }


def _delete_chunks(chroma, key: str) -> None:
    chroma.delete(where={"doc_key": key})


def _safe_delete(chroma, key: str) -> None:
    try:
        _delete_chunks(chroma, key)
    except Exception as e:
        logger.warning("[sync] 删除 chunk 失败 %s：%s", key, e)


def sync_library(
    library: Library,
    chroma=None,
    splitter=None,
    extract=None,
    embed_model: str | None = None,
    scan_config: ScanConfig | None = None,
) -> SyncStats:
    """同步一个知识库，返回统计信息。

    chroma / splitter / extract / embed_model / scan_config 均可注入，便于单测。
    """
    if embed_model is None:
        from tools.config_loader import rag_conf
        embed_model = rag_conf["vector_database"]["embedding_model_name"]

    if chroma is None:
        from rag.retriever import get_library_chroma
        chroma = get_library_chroma(library)

    if splitter is None:
        from rag.splitter import splitter as _splitter
        splitter = _splitter

    if extract is None:
        from tools.document_service import extract_documents
        extract = extract_documents

    cfg = scan_config or ScanConfig()
    stats = SyncStats()

    # 先扫描完所有在线来源；任何来源超限都在写操作前失败，避免部分索引。
    scans = {}
    incomplete_sources: set[str] = set()
    for source in library.sources:
        if not source.enabled:
            stats.disabled_sources.append(source.id)
            continue
        if not source.path.is_dir():
            stats.offline_sources.append(source.id)
            continue

        source_cfg = replace(cfg, recursive=source.recursive)
        try:
            result = scan_source(source.path, source_cfg)
        except NotADirectoryError:
            stats.offline_sources.append(source.id)
            continue
        if result.truncated:
            raise ScanLimitExceeded(
                f"来源「{source.label}」文件数超过上限 {source_cfg.max_files}，"
                "请缩小范围或提高上限"
            )
        if result.errors:
            incomplete_sources.add(source.id)
            for error in result.errors:
                stats.scan_errors.append({"source_id": source.id, **error})
        scans[source.id] = result
        stats.scanned += len(result.files)

    state = load_index_record(library, embed_model)
    files = state.files
    if state.rebuild_reason:
        chroma.reset_collection()
        files = {}
        stats.rebuild_reason = state.rebuild_reason

    excluded = load_excluded(library)
    stats.excluded = len(excluded)

    # 已移出的文档：清理残留 chunk 与索引记录
    for key in list(excluded):
        if key in files:
            _safe_delete(chroma, key)
            del files[key]
            stats.removed.append(key)

    # 来源被移除后遗留的记录
    valid_source_ids = {source.id for source in library.sources}
    for key in list(files):
        try:
            source_id, _ = _split_key(key)
        except SyncError:
            _safe_delete(chroma, key)
            del files[key]
            stats.removed.append(key)
            continue
        if source_id not in valid_source_ids:
            _safe_delete(chroma, key)
            del files[key]
            stats.removed.append(key)

    for source in library.sources:
        scan = scans.get(source.id)
        if scan is None:
            continue

        seen: set[str] = set()
        for scanned in scan.files:
            key = library.doc_key(source.id, scanned.rel_path)
            seen.add(key)

            if key in excluded:
                if key in files:
                    _safe_delete(chroma, key)
                    del files[key]
                    stats.removed.append(key)
                continue

            try:
                md5 = _file_md5(scanned.path)
            except Exception as e:
                stats.failed.append({
                    "doc_key": key,
                    "source_id": source.id,
                    "rel_path": scanned.rel_path.as_posix(),
                    "error": f"{type(e).__name__}: {e}",
                })
                continue

            old = files.get(key)
            if old is not None and old.md5 == md5:
                stats.skipped += 1
                continue

            # 先抽取/切分，成功后再删旧 chunk；抽取失败时保留旧索引。
            try:
                documents = extract(str(scanned.path))
                chunks = splitter.split_documents(documents)
            except Exception as e:
                stats.failed.append({
                    "doc_key": key,
                    "source_id": source.id,
                    "rel_path": scanned.rel_path.as_posix(),
                    "error": f"{type(e).__name__}: {e}",
                })
                continue

            metadata = {
                "doc_key": key,
                "source_id": source.id,
                "rel_path": scanned.rel_path.as_posix(),
            }
            for chunk in chunks:
                chunk.metadata = {**(chunk.metadata or {}), **metadata}

            try:
                if old is not None:
                    _delete_chunks(chroma, key)
                if chunks:
                    chroma.add_documents(chunks)
                files[key] = DocRecord(
                    md5=md5,
                    size=scanned.size,
                    mtime=scanned.mtime,
                    chunks=len(chunks),
                    indexed_at=_now(),
                )
                if old is None:
                    stats.added.append(key)
                else:
                    stats.updated.append(key)
            except Exception as e:
                # 清理可能写入一半的新 chunk；旧记录已不可信，移除后下轮重试
                try:
                    _delete_chunks(chroma, key)
                except Exception:
                    pass
                files.pop(key, None)
                stats.failed.append({
                    "doc_key": key,
                    "source_id": source.id,
                    "rel_path": scanned.rel_path.as_posix(),
                    "error": f"{type(e).__name__}: {e}",
                })

        # 来源在线且扫描完整时，才删除磁盘上已不存在的记录；
        # 扫描出错时保守保留，避免权限问题导致误删索引。
        if source.id not in incomplete_sources:
            prefix = source.id + "/"
            for key in list(files):
                if key.startswith(prefix) and key not in seen:
                    _safe_delete(chroma, key)
                    del files[key]
                    stats.removed.append(key)

    save_index_record(library, files, embed_model)
    return stats


def collect_documents(
    library: Library,
    scan_config: ScanConfig | None = None,
    record_files: dict[str, DocRecord] | None = None,
    excluded: dict[str, dict] | None = None,
) -> list[DocumentInfo]:
    """扫描当前来源并推导每篇文档的状态，供 UI 展示。"""
    cfg = scan_config or ScanConfig()
    if record_files is None:
        record_files = load_index_record(library).files
    if excluded is None:
        excluded = load_excluded(library)

    source_order = {source.id: index for index, source in enumerate(library.sources)}
    docs: list[DocumentInfo] = []
    seen: set[str] = set()

    def add_record(source, key: str, rec: DocRecord, state: str, exists: bool) -> None:
        rel = _rel_from_key(key, source.id)
        if rel is None:
            return
        docs.append(DocumentInfo(
            doc_key=key,
            source_id=source.id,
            source_label=source.label,
            rel_path=rel,
            name=rel.name,
            state=state,
            exists=exists,
            size=rec.size,
            mtime=rec.mtime,
            chunks=rec.chunks,
        ))
        seen.add(key)

    def add_excluded(source, key: str, meta: dict, exists: bool) -> None:
        rel = _rel_from_key(key, source.id)
        if rel is None:
            return
        docs.append(DocumentInfo(
            doc_key=key,
            source_id=source.id,
            source_label=source.label,
            rel_path=rel,
            name=rel.name,
            state=STATE_EXCLUDED,
            exists=exists,
        ))
        seen.add(key)

    for source in library.sources:
        source_records = {
            key: rec for key, rec in record_files.items()
            if key.startswith(source.id + "/")
        }
        source_excluded = {
            key: meta for key, meta in excluded.items()
            if key.startswith(source.id + "/")
        }

        if not source.enabled:
            for key, meta in source_excluded.items():
                if key in seen:
                    continue
                add_excluded(source, key, meta, exists=False)
            for key, rec in source_records.items():
                if key in seen:
                    continue
                add_record(source, key, rec, STATE_DISABLED, exists=False)
            continue

        if not source.path.is_dir():
            for key, meta in source_excluded.items():
                if key in seen:
                    continue
                add_excluded(source, key, meta, exists=False)
            for key, rec in source_records.items():
                if key in seen:
                    continue
                add_record(source, key, rec, STATE_OFFLINE, exists=False)
            continue

        source_cfg = replace(cfg, recursive=source.recursive)
        result = scan_source(source.path, source_cfg)
        if result.truncated:
            raise ScanLimitExceeded(
                f"来源「{source.label}」文件数超过上限 {source_cfg.max_files}"
            )

        for scanned in result.files:
            key = library.doc_key(source.id, scanned.rel_path)
            if key in seen:
                continue
            if key in excluded:
                add_excluded(source, key, excluded[key], exists=True)
                continue

            rec = record_files.get(key)
            if rec is None:
                state = STATE_UNINDEXED
                chunks = 0
            elif rec.size == scanned.size and rec.mtime == scanned.mtime:
                state = STATE_INDEXED
                chunks = rec.chunks
            else:
                state = STATE_STALE
                chunks = rec.chunks

            docs.append(DocumentInfo(
                doc_key=key,
                source_id=source.id,
                source_label=source.label,
                rel_path=scanned.rel_path,
                name=scanned.rel_path.name,
                state=state,
                exists=True,
                size=scanned.size,
                mtime=scanned.mtime,
                chunks=chunks,
            ))
            seen.add(key)

        missing_state = STATE_UNKNOWN if result.errors else STATE_MISSING
        for key, rec in source_records.items():
            if key in seen or key in excluded:
                continue
            add_record(source, key, rec, missing_state, exists=False)

        for key, meta in source_excluded.items():
            if key in seen:
                continue
            add_excluded(source, key, meta, exists=False)

    docs.sort(key=lambda item: (
        source_order.get(item.source_id, 9999),
        item.rel_path.as_posix().lower(),
    ))
    return docs
