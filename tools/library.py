"""
文件名：library.py
介绍：虚拟知识库的数据模型。

一个 Library（知识库）可以挂多个 Source（文件夹来源），文件原地索引，
不复制、不移动。每个知识库在 data/libraries/<id>/ 下拥有独立的：

    library.json        # 名称、来源列表、时间戳
    index_record.json   # 索引记录（M1 使用）
    excluded.json       # 已移出文档的排除记录（M1 使用）
    chroma/             # 该库独立的向量库

本模块只依赖标准库，便于单测和无 Key 环境导入。
"""
from __future__ import annotations

import json
import os
import shutil
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from tools.app_paths import libraries_dir

LIBRARY_FORMAT_VERSION = 1
REGISTRY_FORMAT_VERSION = 1

META_FILE = "library.json"
INDEX_FILE = "index_record.json"
EXCLUDED_FILE = "excluded.json"
CHROMA_DIR = "chroma"
REGISTRY_FILE = "libraries.json"


class LibraryError(Exception):
    """知识库模型层的基类异常。"""


class LibraryNotFound(LibraryError):
    pass


class SourceNotFound(LibraryError):
    pass


class InvalidSourcePath(LibraryError):
    pass


class DuplicateSource(LibraryError):
    pass


class SourceOverlap(LibraryError):
    pass


class PathOutsideSource(LibraryError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict) -> None:
    """原子写入 JSON：先写临时文件，再替换，避免进程中断留下半截文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise LibraryError(f"JSON 格式不正确：{path}")
    return data


def _norm(path: Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def _same_path(a: Path, b: Path) -> bool:
    return _norm(a) == _norm(b)


def _is_inside(root: Path, target: Path) -> bool:
    """判断 target 是否在 root 内（Windows 下大小写不敏感）。"""
    root_p = Path(root).resolve()
    target_p = Path(target).resolve()
    try:
        target_p.relative_to(root_p)
        return True
    except ValueError:
        root_s = os.path.normcase(str(root_p))
        target_s = os.path.normcase(str(target_p))
        return target_s == root_s or target_s.startswith(root_s + os.sep)


def _paths_overlap(a: Path, b: Path) -> bool:
    """两个目录相同、或一个包含另一个。"""
    return _is_inside(a, b) or _is_inside(b, a)


def _normalize_source_path(path) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise InvalidSourcePath(f"来源文件夹不存在或不是目录：{p}")
    return p


def _normalize_rel_path(rel_path) -> Path:
    rel = Path(rel_path)
    if rel.is_absolute() or rel.drive or rel.root:
        raise PathOutsideSource(f"不允许绝对路径：{rel}")
    if any(part == ".." for part in rel.parts):
        raise PathOutsideSource(f"不允许包含 .. 的路径：{rel}")
    if not rel.parts or str(rel) == ".":
        raise PathOutsideSource("相对路径不能为空")
    return rel


@dataclass(frozen=True)
class Source:
    """知识库的一个文件夹来源。"""

    id: str
    path: Path
    label: str
    recursive: bool = True
    enabled: bool = True
    added_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "path": str(self.path),
            "label": self.label,
            "recursive": self.recursive,
            "enabled": self.enabled,
            "added_at": self.added_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Source":
        try:
            path = Path(data["path"]).expanduser().resolve()
            source_id = str(data["id"])
        except (KeyError, TypeError) as e:
            raise LibraryError(f"来源配置缺少必要字段：{e}") from e
        label = str(data.get("label") or "").strip() or path.name or str(path)
        return cls(
            id=source_id,
            path=path,
            label=label,
            recursive=bool(data.get("recursive", True)),
            enabled=bool(data.get("enabled", True)),
            added_at=str(data.get("added_at") or ""),
        )


@dataclass
class Library:
    """一个虚拟知识库。"""

    id: str
    name: str
    base_dir: Path
    sources: list[Source] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    version: int = LIBRARY_FORMAT_VERSION

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir).resolve()
        if not self.created_at:
            self.created_at = _now()
        if not self.updated_at:
            self.updated_at = self.created_at

    # ---- 路径 ----

    @property
    def meta_file(self) -> Path:
        return self.base_dir / META_FILE

    @property
    def index_file(self) -> Path:
        return self.base_dir / INDEX_FILE

    @property
    def excluded_file(self) -> Path:
        return self.base_dir / EXCLUDED_FILE

    @property
    def chroma_dir(self) -> Path:
        return self.base_dir / CHROMA_DIR

    def ensure_dirs(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

    # ---- 持久化 ----

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sources": [s.to_dict() for s in self.sources],
        }

    def save(self) -> None:
        self.ensure_dirs()
        self.updated_at = _now()
        _write_json(self.meta_file, self.to_dict())

    @classmethod
    def load(cls, base_dir) -> "Library":
        base = Path(base_dir).resolve()
        meta_file = base / META_FILE
        if not meta_file.exists():
            raise LibraryNotFound(f"知识库不存在：{base}")
        try:
            data = _read_json(meta_file)
        except (json.JSONDecodeError, OSError) as e:
            raise LibraryError(f"知识库配置读取失败：{meta_file}：{e}") from e

        try:
            version = int(data.get("version", 1))
        except (TypeError, ValueError) as e:
            raise LibraryError(f"library.json 的 version 字段无效：{data.get('version')!r}") from e
        if version > LIBRARY_FORMAT_VERSION:
            raise LibraryError(
                f"知识库版本 {version} 高于当前支持版本 {LIBRARY_FORMAT_VERSION}"
            )

        sources_data = data.get("sources") or []
        if not isinstance(sources_data, list):
            raise LibraryError("library.json 的 sources 必须是数组")

        return cls(
            id=str(data.get("id") or base.name),
            name=str(data.get("name") or "未命名知识库"),
            base_dir=base,
            sources=[Source.from_dict(item) for item in sources_data],
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
            version=version,
        )

    # ---- 来源管理 ----

    def source(self, source_id: str) -> Source:
        for source in self.sources:
            if source.id == source_id:
                return source
        raise SourceNotFound(f"来源不存在：{source_id}")

    def add_source(
        self,
        path,
        label: str | None = None,
        recursive: bool = True,
    ) -> Source:
        normalized = _normalize_source_path(path)

        for existing in self.sources:
            if _same_path(existing.path, normalized):
                raise DuplicateSource(f"该文件夹已添加：{normalized}")
            if _paths_overlap(existing.path, normalized):
                raise SourceOverlap(
                    f"与已有来源重叠：{existing.label}（{existing.path}）"
                )

        source = Source(
            id=uuid4().hex,
            path=normalized,
            label=(label or "").strip() or normalized.name or str(normalized),
            recursive=bool(recursive),
            enabled=True,
            added_at=_now(),
        )
        self.sources.append(source)
        self.updated_at = _now()
        return source

    def remove_source(self, source_id: str) -> None:
        source = self.source(source_id)
        self.sources = [s for s in self.sources if s.id != source.id]
        self.updated_at = _now()

    def set_source_enabled(self, source_id: str, enabled: bool) -> None:
        source = self.source(source_id)
        self.sources = [
            replace(s, enabled=bool(enabled)) if s.id == source.id else s
            for s in self.sources
        ]
        self.updated_at = _now()

    def rename_source(self, source_id: str, label: str) -> None:
        source = self.source(source_id)
        new_label = (label or "").strip()
        if not new_label:
            raise LibraryError("来源别名不能为空")
        self.sources = [
            replace(s, label=new_label) if s.id == source.id else s
            for s in self.sources
        ]
        self.updated_at = _now()

    def rename(self, name: str) -> None:
        new_name = (name or "").strip()
        if not new_name:
            raise LibraryError("知识库名称不能为空")
        self.name = new_name
        self.updated_at = _now()

    # ---- 文档键与路径 ----

    def doc_key(self, source_id: str, rel_path) -> str:
        """生成稳定的文档键：source_id/相对路径（统一用 /）。"""
        self.source(source_id)  # 校验来源存在
        rel = _normalize_rel_path(rel_path)
        return f"{source_id}/{rel.as_posix()}"

    def split_doc_key(self, doc_key: str) -> tuple[str, Path]:
        source_id, sep, rel = str(doc_key).partition("/")
        if not sep or not source_id or not rel:
            raise LibraryError(f"文档键格式不正确：{doc_key}")
        self.source(source_id)
        return source_id, _normalize_rel_path(rel)

    def relative_path(self, source_id: str, file_path) -> Path:
        """把来源内的绝对文件路径转成相对路径。"""
        source = self.source(source_id)
        target = Path(file_path).expanduser().resolve()
        if not _is_inside(source.path, target):
            raise PathOutsideSource(f"文件不在来源内：{target}")
        try:
            return target.relative_to(source.path)
        except ValueError:
            # Windows 大小写差异走这里
            return Path(os.path.relpath(target, source.path))

    def resolve_doc_path(self, source_id: str, rel_path) -> Path:
        """把来源内相对路径还原成绝对路径，并防止越界。"""
        source = self.source(source_id)
        rel = _normalize_rel_path(rel_path)
        target = (source.path / rel).resolve()
        if not _is_inside(source.path, target):
            raise PathOutsideSource(f"路径越出来源：{rel}")
        return target


def _has_supported_files(path: Path) -> bool:
    """来源目录顶层是否有受支持的文档，用于首次启动时选择默认来源。"""
    from tools.document_service import SUPPORTED_EXTENSIONS

    try:
        return any(
            p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            for p in path.iterdir()
        )
    except OSError:
        return False


def bootstrap_default_library(
    registry: "LibraryRegistry",
    candidates: Iterable[Path] = (),
) -> tuple[Library, Source | None]:
    """首次启动时创建默认知识库，并按顺序挂载第一个非空的候选目录。

    候选通常传 [<项目>/test_data, <项目>/demo_data]。
    已有知识库时直接返回默认库，不重复添加来源。
    """
    if registry.list_libraries():
        return registry.default_library(), None

    library = registry.create_library()
    source = None
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_dir():
            continue
        if not _has_supported_files(path):
            continue
        source = registry.add_source(library.id, path)
        break

    return registry.get_library(library.id), source


class LibraryRegistry:
    """知识库注册表。

    libraries.json 只保存轻量元数据；每个知识库的完整来源列表在
    <id>/library.json 中，避免两处配置重复。

    写操作会重新加载、修改并保存 Library；调用方之前持有的 Library
    实例是快照，不会自动刷新，需要 get_library() 重新读取。
    """

    def __init__(self, root=None):
        self.root = Path(root).resolve() if root else libraries_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.root / REGISTRY_FILE
        self._lock = threading.RLock()

    # ---- 索引文件 ----

    def _load_index(self) -> dict:
        if not self.registry_file.exists():
            return {"version": REGISTRY_FORMAT_VERSION, "libraries": []}
        try:
            data = _read_json(self.registry_file)
        except (json.JSONDecodeError, OSError) as e:
            raise LibraryError(f"知识库注册表读取失败：{e}") from e

        libraries = data.get("libraries") or []
        if not isinstance(libraries, list):
            raise LibraryError("libraries.json 的 libraries 必须是数组")
        try:
            version = int(data.get("version", REGISTRY_FORMAT_VERSION))
        except (TypeError, ValueError) as e:
            raise LibraryError(
                f"libraries.json 的 version 字段无效：{data.get('version')!r}"
            ) from e
        return {
            "version": version,
            "libraries": [
                item for item in libraries
                if isinstance(item, dict) and item.get("id")
            ],
        }

    def _save_index(self, index: dict) -> None:
        _write_json(self.registry_file, index)

    @staticmethod
    def _find_meta(index: dict, library_id: str) -> dict | None:
        for meta in index["libraries"]:
            if meta.get("id") == library_id:
                return meta
        return None

    @staticmethod
    def _meta_for(library: Library, last_opened_at: str) -> dict:
        return {
            "id": library.id,
            "name": library.name,
            "created_at": library.created_at,
            "last_opened_at": last_opened_at,
        }

    # ---- 查询 ----

    def _base_for(self, library_id: str) -> Path:
        """解析知识库目录并阻止 id 中的路径穿越。"""
        base = (self.root / str(library_id)).resolve()
        if not _is_inside(self.root, base):
            raise LibraryError(f"知识库路径越界：{library_id}")
        return base

    def list_libraries(self) -> list[dict]:
        """返回按最近打开时间倒序的知识库元数据。"""
        with self._lock:
            index = self._load_index()
            result = []
            for meta in index["libraries"]:
                base = self.root / str(meta["id"])
                try:
                    library = Library.load(base)
                except (LibraryError, OSError):
                    # 目录被手动删掉或配置损坏时跳过，不让整个列表崩掉
                    continue
                result.append({
                    **meta,
                    "name": library.name,
                    "source_count": len(library.sources),
                })
            result.sort(
                key=lambda item: item.get("last_opened_at") or "",
                reverse=True,
            )
            return result

    def get_library(self, library_id: str) -> Library:
        with self._lock:
            return Library.load(self._base_for(library_id))

    def default_library(self) -> Library:
        """最近打开的知识库；一个都没有时创建默认库。"""
        with self._lock:
            items = self.list_libraries()
            if not items:
                return self.create_library()
            library = self.get_library(str(items[0]["id"]))
            self.touch_library(library.id)
            return library

    # ---- 创建/更新 ----

    def create_library(self, name: str = "我的资料") -> Library:
        with self._lock:
            library_id = uuid4().hex
            library = Library(
                id=library_id,
                name=(name or "").strip() or "我的资料",
                base_dir=self.root / library_id,
            )
            library.ensure_dirs()
            library.save()

            index = self._load_index()
            index["libraries"].append(
                self._meta_for(library, last_opened_at=_now())
            )
            self._save_index(index)
            return library

    def touch_library(self, library_id: str) -> None:
        with self._lock:
            index = self._load_index()
            meta = self._find_meta(index, str(library_id))
            if meta is None:
                raise LibraryNotFound(f"知识库不在注册表里：{library_id}")
            meta["last_opened_at"] = _now()
            self._save_index(index)

    def rename_library(self, library_id: str, name: str) -> Library:
        with self._lock:
            library = self.get_library(library_id)
            library.rename(name)
            library.save()

            index = self._load_index()
            meta = self._find_meta(index, library.id)
            if meta is not None:
                meta["name"] = library.name
                self._save_index(index)
            return library

    def delete_library(self, library_id: str, delete_data: bool = False) -> None:
        """从注册表移除知识库；delete_data=True 时只删应用数据，不动来源文件。"""
        with self._lock:
            base = self._base_for(library_id)

            if delete_data and base.exists():
                shutil.rmtree(base, ignore_errors=False)

            index = self._load_index()
            index["libraries"] = [
                meta for meta in index["libraries"]
                if meta.get("id") != str(library_id)
            ]
            self._save_index(index)

    # ---- 来源操作 ----

    def add_source(
        self,
        library_id: str,
        path,
        label: str | None = None,
        recursive: bool = True,
    ) -> Source:
        with self._lock:
            library = self.get_library(library_id)
            source = library.add_source(path, label=label, recursive=recursive)
            library.save()
            self.touch_library(library.id)
            return source

    def remove_source(self, library_id: str, source_id: str) -> None:
        with self._lock:
            library = self.get_library(library_id)
            library.remove_source(source_id)
            library.save()
            self.touch_library(library.id)

    def set_source_enabled(
        self,
        library_id: str,
        source_id: str,
        enabled: bool,
    ) -> None:
        with self._lock:
            library = self.get_library(library_id)
            library.set_source_enabled(source_id, enabled)
            library.save()
            self.touch_library(library.id)

    def rename_source(self, library_id: str, source_id: str, label: str) -> None:
        with self._lock:
            library = self.get_library(library_id)
            library.rename_source(source_id, label)
            library.save()
            self.touch_library(library.id)
