"""
文件名：scanner.py
介绍：文件夹扫描器。只做「找文件 + 读元数据」，不解析内容、不做嵌入。

特点：
- 默认递归，可关闭；
- 跳过隐藏目录、常见工程目录、临时文件、符号链接/重解析点；
- 限制单文件大小与单来源文件数，避免误选整个磁盘导致失控；
- 返回结构化统计，供 UI 做「范围预览」；
- 纯标准库，便于单测和无 Key 环境导入。
"""
from __future__ import annotations

import math
import os
import stat
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

DEFAULT_IGNORED_DIRS = frozenset({
    ".git", ".svn", ".hg",
    "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", "wheels", ".egg-info",
    ".idea", ".vscode", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "chroma_data",
})

DEFAULT_IGNORED_NAMES = frozenset({
    ".DS_Store", "Thumbs.db", "desktop.ini",
})

DEFAULT_IGNORED_NAME_PATTERNS = ("~$*",)

DEFAULT_MAX_FILES = 5000
DEFAULT_MAX_FILE_SIZE = 50 * 1024 * 1024
DEFAULT_ESTIMATED_CHUNK_SIZE = 500

# PDF/DOCX 的压缩体积不等于可抽取文本量，预览时用更粗的二进制估算
BINARY_EXTENSIONS = frozenset({".pdf", ".docx"})
BINARY_ESTIMATED_CHUNK_SIZE = 20_000


@dataclass(frozen=True)
class ScanConfig:
    recursive: bool = True
    max_files: int = DEFAULT_MAX_FILES
    max_file_size: int = DEFAULT_MAX_FILE_SIZE
    ignored_dirs: frozenset[str] = DEFAULT_IGNORED_DIRS
    ignored_names: frozenset[str] = DEFAULT_IGNORED_NAMES
    ignored_name_patterns: tuple[str, ...] = DEFAULT_IGNORED_NAME_PATTERNS
    supported_extensions: tuple[str, ...] | None = None
    follow_symlinks: bool = False
    estimated_chunk_size: int = DEFAULT_ESTIMATED_CHUNK_SIZE


@dataclass(frozen=True)
class ScanFile:
    path: Path
    rel_path: Path
    size: int
    mtime: float


@dataclass
class ScanResult:
    root: Path
    files: list[ScanFile] = field(default_factory=list)
    skipped_unsupported: int = 0
    skipped_too_large: int = 0
    ignored_entries: int = 0
    errors: list[dict] = field(default_factory=list)
    truncated: bool = False


def _supported_extensions(config: ScanConfig) -> tuple[str, ...]:
    if config.supported_extensions is not None:
        return tuple(ext.lower() for ext in config.supported_extensions)

    # 延迟导入，避免 scanner 模块加载时带上 langchain 依赖
    from tools.document_service import SUPPORTED_EXTENSIONS

    return tuple(ext.lower() for ext in SUPPORTED_EXTENSIONS)


def _is_hidden(name: str) -> bool:
    return name.startswith(".") and name not in (".", "..")


def _is_ignored_dir(name: str, config: ScanConfig) -> bool:
    if _is_hidden(name):
        return True
    lowered = name.lower()
    return any(lowered == item.lower() for item in config.ignored_dirs)


def _is_ignored_file(name: str, config: ScanConfig) -> bool:
    if _is_hidden(name):
        return True
    lowered = name.lower()
    if any(lowered == item.lower() for item in config.ignored_names):
        return True
    return any(
        fnmatch(lowered, pattern.lower())
        for pattern in config.ignored_name_patterns
    )


def _is_reparse_point(path: Path) -> bool:
    """Windows 的 junction/符号链接；避免 os.walk 走进去。"""
    if not hasattr(os, "lstat"):
        return False
    try:
        st = os.lstat(path)
    except OSError:
        return False
    attrs = getattr(st, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attrs & flag)


def _should_skip_link(path: Path, config: ScanConfig) -> bool:
    if config.follow_symlinks:
        return False
    return path.is_symlink() or _is_reparse_point(path)


def scan_source(root, config: ScanConfig | None = None) -> ScanResult:
    """扫描一个来源目录，返回受支持文件的元数据列表。"""
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"来源不是文件夹：{root_path}")

    cfg = config or ScanConfig()
    extensions = _supported_extensions(cfg)
    result = ScanResult(root=root_path)

    if cfg.max_files <= 0:
        result.truncated = True
        return result

    def consider(path: Path) -> bool:
        """返回 True 表示已达到文件数上限，应停止扫描。"""
        name = path.name
        if _is_ignored_file(name, cfg):
            result.ignored_entries += 1
            return False
        if _should_skip_link(path, cfg):
            result.ignored_entries += 1
            return False
        if path.suffix.lower() not in extensions:
            result.skipped_unsupported += 1
            return False

        try:
            st = path.stat()
        except OSError as e:
            result.errors.append({"path": str(path), "error": str(e)})
            return False
        if not stat.S_ISREG(st.st_mode):
            return False
        if st.st_size > cfg.max_file_size:
            result.skipped_too_large += 1
            return False

        try:
            rel = path.relative_to(root_path)
        except ValueError:
            return False

        if len(result.files) >= cfg.max_files:
            result.truncated = True
            return True

        result.files.append(ScanFile(
            path=path,
            rel_path=rel,
            size=st.st_size,
            mtime=st.st_mtime,
        ))
        return False

    def on_error(error: OSError) -> None:
        result.errors.append({
            "path": str(getattr(error, "filename", "") or ""),
            "error": str(error),
        })

    if not cfg.recursive:
        try:
            entries = sorted(root_path.iterdir(), key=lambda p: p.name.lower())
        except OSError as e:
            result.errors.append({"path": str(root_path), "error": str(e)})
            return result

        for entry in entries:
            if entry.is_dir():
                result.ignored_entries += 1
                continue
            if consider(entry):
                break
        return result

    for dirpath, dirnames, filenames in os.walk(
        root_path, followlinks=cfg.follow_symlinks, onerror=on_error
    ):
        current = Path(dirpath)
        kept_dirs = []
        for name in sorted(dirnames, key=str.lower):
            candidate = current / name
            if _is_ignored_dir(name, cfg):
                result.ignored_entries += 1
                continue
            if _should_skip_link(candidate, cfg):
                result.ignored_entries += 1
                continue
            kept_dirs.append(name)
        dirnames[:] = kept_dirs

        for name in sorted(filenames, key=str.lower):
            if consider(current / name):
                return result

    return result


def preview_scan(root, config: ScanConfig | None = None) -> dict:
    """只统计、不解析，供 UI 在正式索引前展示范围与粗略成本。"""
    cfg = config or ScanConfig()
    result = scan_source(root, cfg)

    by_extension: dict[str, int] = {}
    total_bytes = 0
    estimated_chunks = 0
    for item in result.files:
        ext = item.path.suffix.lower() or "(无扩展名)"
        by_extension[ext] = by_extension.get(ext, 0) + 1
        total_bytes += item.size
        if ext in BINARY_EXTENSIONS:
            estimated_chunks += max(
                1, math.ceil(item.size / BINARY_ESTIMATED_CHUNK_SIZE)
            )
        else:
            estimated_chunks += max(
                1, math.ceil(item.size / cfg.estimated_chunk_size)
            )

    return {
        "root": str(result.root),
        "files": len(result.files),
        "by_extension": by_extension,
        "total_bytes": total_bytes,
        "estimated_chunks": estimated_chunks,
        "skipped_unsupported": result.skipped_unsupported,
        "skipped_too_large": result.skipped_too_large,
        "ignored_entries": result.ignored_entries,
        "errors": result.errors,
        "truncated": result.truncated,
        "recursive": cfg.recursive,
    }
