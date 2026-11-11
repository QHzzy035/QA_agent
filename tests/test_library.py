"""
文件名：test_library.py
介绍：虚拟知识库数据模型测试。

覆盖：
- 知识库创建/加载/目录结构；
- 来源添加、重名、父子目录重叠、禁用、移除；
- doc_key 与相对路径的越界校验；
- 首次启动 bootstrap 的候选目录选择。
"""
from pathlib import Path

import pytest

from tools.library import (
    DuplicateSource,
    LibraryError,
    LibraryNotFound,
    LibraryRegistry,
    PathOutsideSource,
    SourceOverlap,
    bootstrap_default_library,
)


@pytest.fixture
def registry(tmp_path):
    return LibraryRegistry(tmp_path / "libraries")


@pytest.fixture
def docs(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.txt").write_text("hello", encoding="utf-8")
    return root


class TestLibrary:
    def test_create_and_reload(self, registry):
        library = registry.create_library("我的资料")

        assert library.id
        assert (library.base_dir / "library.json").is_file()
        assert (library.base_dir / "chroma").is_dir()

        loaded = registry.get_library(library.id)
        assert loaded.name == "我的资料"
        assert loaded.sources == []
        assert loaded.index_file.name == "index_record.json"
        assert loaded.excluded_file.name == "excluded.json"

    def test_missing_library_raises(self, registry):
        with pytest.raises(LibraryNotFound):
            registry.get_library("not-a-real-id")

    def test_path_traversal_id_rejected(self, registry):
        with pytest.raises(LibraryError):
            registry.get_library("..")

    def test_rename_library(self, registry):
        library = registry.create_library("旧名字")

        renamed = registry.rename_library(library.id, "新名字")

        assert renamed.name == "新名字"
        assert registry.get_library(library.id).name == "新名字"
        assert registry.list_libraries()[0]["name"] == "新名字"


class TestSources:
    def test_add_source_persists(self, registry, docs):
        library = registry.create_library()

        source = registry.add_source(
            library.id, docs, label="工作", recursive=False
        )

        assert source.path == docs.resolve()
        assert source.label == "工作"
        assert source.recursive is False
        assert source.enabled is True

        reloaded = registry.get_library(library.id)
        assert len(reloaded.sources) == 1
        assert reloaded.sources[0].id == source.id
        assert reloaded.sources[0].label == "工作"

    def test_duplicate_source_rejected(self, registry, docs):
        library = registry.create_library()
        registry.add_source(library.id, docs)

        with pytest.raises(DuplicateSource):
            registry.add_source(library.id, docs)

    def test_overlap_rejected_both_directions(self, registry, tmp_path):
        parent = tmp_path / "parent"
        child = parent / "child"
        child.mkdir(parents=True)
        (child / "a.txt").write_text("x", encoding="utf-8")

        first = registry.create_library("先父后子")
        registry.add_source(first.id, parent)
        with pytest.raises(SourceOverlap):
            registry.add_source(first.id, child)

        second = registry.create_library("先子后父")
        registry.add_source(second.id, child)
        with pytest.raises(SourceOverlap):
            registry.add_source(second.id, parent)

    def test_missing_source_path_rejected(self, registry, tmp_path):
        library = registry.create_library()

        with pytest.raises(LibraryError):
            registry.add_source(library.id, tmp_path / "does-not-exist")

    def test_disable_and_remove_source_keeps_files(self, registry, docs):
        library = registry.create_library()
        source = registry.add_source(library.id, docs)

        registry.set_source_enabled(library.id, source.id, False)
        assert registry.get_library(library.id).source(source.id).enabled is False

        registry.remove_source(library.id, source.id)
        assert registry.get_library(library.id).sources == []
        assert (docs / "a.txt").is_file()

    def test_rename_source(self, registry, docs):
        library = registry.create_library()
        source = registry.add_source(library.id, docs, label="旧别名")

        registry.rename_source(library.id, source.id, "新别名")

        assert registry.get_library(library.id).source(source.id).label == "新别名"

    def test_list_libraries_reports_source_count(self, registry, docs):
        library = registry.create_library()
        registry.add_source(library.id, docs)

        items = registry.list_libraries()

        assert len(items) == 1
        assert items[0]["source_count"] == 1


class TestDocKeys:
    def test_doc_key_and_resolve(self, registry, docs):
        sub = docs / "sub"
        sub.mkdir()
        target = sub / "b.txt"
        target.write_text("x", encoding="utf-8")

        library = registry.create_library()
        source = registry.add_source(library.id, docs)
        library = registry.get_library(library.id)

        rel = library.relative_path(source.id, target)
        assert rel == Path("sub") / "b.txt"

        key = library.doc_key(source.id, rel)
        assert key == f"{source.id}/sub/b.txt"
        assert library.split_doc_key(key) == (source.id, Path("sub/b.txt"))
        assert library.resolve_doc_path(source.id, rel) == target.resolve()

    def test_outside_path_rejected(self, registry, docs, tmp_path):
        library = registry.create_library()
        source = registry.add_source(library.id, docs)
        library = registry.get_library(library.id)
        outside = tmp_path / "outside.txt"
        outside.write_text("x", encoding="utf-8")

        with pytest.raises(PathOutsideSource):
            library.relative_path(source.id, outside)

        with pytest.raises(PathOutsideSource):
            library.resolve_doc_path(source.id, "../outside.txt")

    def test_absolute_rel_path_rejected(self, registry, docs):
        library = registry.create_library()
        source = registry.add_source(library.id, docs)
        library = registry.get_library(library.id)

        with pytest.raises(PathOutsideSource):
            library.resolve_doc_path(source.id, docs / "a.txt")

    def test_unknown_source_raises(self, registry):
        library = registry.create_library()

        with pytest.raises(LibraryError):
            library.source("missing")


class TestBootstrap:
    def test_prefers_non_empty_test_data(self, registry, tmp_path):
        test_data = tmp_path / "test_data"
        test_data.mkdir()
        (test_data / "a.txt").write_text("x", encoding="utf-8")

        demo_data = tmp_path / "demo_data"
        demo_data.mkdir()
        (demo_data / "b.md").write_text("x", encoding="utf-8")

        library, source = bootstrap_default_library(
            registry, [test_data, demo_data]
        )

        assert source is not None
        assert source.path == test_data.resolve()
        assert len(library.sources) == 1

    def test_falls_back_to_demo_data(self, registry, tmp_path):
        test_data = tmp_path / "test_data"
        test_data.mkdir()  # 空目录

        demo_data = tmp_path / "demo_data"
        demo_data.mkdir()
        (demo_data / "b.md").write_text("x", encoding="utf-8")

        _, source = bootstrap_default_library(registry, [test_data, demo_data])

        assert source is not None
        assert source.path == demo_data.resolve()

    def test_existing_library_is_not_touched(self, registry, tmp_path):
        candidate = tmp_path / "docs"
        candidate.mkdir()
        (candidate / "a.txt").write_text("x", encoding="utf-8")

        first_library, first_source = bootstrap_default_library(
            registry, [candidate]
        )
        second_library, second_source = bootstrap_default_library(
            registry, [candidate]
        )

        assert first_source is not None
        assert second_source is None
        assert second_library.id == first_library.id
        assert len(registry.list_libraries()) == 1


class TestDelete:
    def test_delete_data_keeps_source_files(self, registry, docs):
        library = registry.create_library()
        registry.add_source(library.id, docs)

        registry.delete_library(library.id, delete_data=True)

        assert not library.base_dir.exists()
        assert (docs / "a.txt").is_file()
        assert registry.list_libraries() == []
