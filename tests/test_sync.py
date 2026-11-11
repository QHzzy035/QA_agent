"""
文件名：test_sync.py
介绍：同步引擎与状态推导测试。

用内存 FakeChroma + FakeSplitter + 假抽取函数，不依赖 API Key、网络或真实向量库。
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from rag.scanner import ScanConfig
from rag.sync import (
    ScanLimitExceeded,
    STATE_DISABLED,
    STATE_EXCLUDED,
    STATE_INDEXED,
    STATE_MISSING,
    STATE_OFFLINE,
    STATE_STALE,
    STATE_UNINDEXED,
    collect_documents,
    exclude_document,
    list_excluded,
    load_index_record,
    restore_document,
    sync_library,
)
from tools.library import LibraryRegistry

SCAN = ScanConfig(supported_extensions=(".txt",))


class FakeChroma:
    def __init__(self):
        self.docs = []
        self.reset_count = 0
        self.delete_calls = []

    def reset_collection(self):
        self.reset_count += 1
        self.docs.clear()

    def add_documents(self, chunks):
        for chunk in chunks:
            self.docs.append({
                "content": chunk.page_content,
                "metadata": dict(chunk.metadata),
            })

    def delete(self, where=None):
        self.delete_calls.append(where)
        if not where:
            return
        key = where.get("doc_key")
        if key is not None:
            self.docs = [
                item for item in self.docs
                if item["metadata"].get("doc_key") != key
            ]

    def keys(self):
        return sorted({item["metadata"]["doc_key"] for item in self.docs})

    def contents_for(self, key):
        return [
            item["content"] for item in self.docs
            if item["metadata"].get("doc_key") == key
        ]


class FakeSplitter:
    def split_documents(self, documents):
        chunks = []
        for document in documents:
            lines = [
                line for line in document.page_content.splitlines()
                if line.strip()
            ]
            if not lines:
                lines = [""]
            for line in lines:
                chunks.append(Document(
                    page_content=line,
                    metadata=dict(document.metadata),
                ))
        return chunks


def fake_extract(path):
    p = Path(path)
    return [Document(
        page_content=p.read_text(encoding="utf-8"),
        metadata={"source": str(p)},
    )]


def boom_extract(_path):
    raise ValueError("模拟解析失败")


@pytest.fixture
def env(tmp_path):
    registry = LibraryRegistry(tmp_path / "libraries")
    library = registry.create_library("测试库")
    docs = tmp_path / "docs"
    docs.mkdir()
    registry.add_source(library.id, docs)
    library = registry.get_library(library.id)
    return SimpleNamespace(
        registry=registry,
        library=library,
        docs=docs,
        source=library.sources[0],
        tmp_path=tmp_path,
    )


def run_sync(env, chroma, extract=None, model="test-embed", scan_config=SCAN):
    return sync_library(
        env.library,
        chroma=chroma,
        splitter=FakeSplitter(),
        extract=extract or fake_extract,
        embed_model=model,
        scan_config=scan_config,
    )


def key_for(env, rel: str) -> str:
    return env.library.doc_key(env.source.id, Path(rel))


class TestSyncBasics:
    def test_first_sync_indexes_files(self, env):
        (env.docs / "a.txt").write_text("line1\nline2", encoding="utf-8")
        chroma = FakeChroma()

        stats = run_sync(env, chroma)

        key = key_for(env, "a.txt")
        assert stats.added == [key]
        assert stats.scanned == 1
        assert chroma.keys() == [key]
        assert chroma.contents_for(key) == ["line1", "line2"]

        record = load_index_record(env.library, "test-embed")
        assert record.rebuild_reason is None
        assert record.files[key].chunks == 2

        meta = chroma.docs[0]["metadata"]
        assert meta["source_id"] == env.source.id
        assert meta["rel_path"] == "a.txt"
        assert meta["source"] == str(env.docs / "a.txt")

    def test_second_sync_skips_unchanged(self, env):
        (env.docs / "a.txt").write_text("a", encoding="utf-8")
        chroma = FakeChroma()
        run_sync(env, chroma)
        before = len(chroma.docs)

        stats = run_sync(env, chroma)

        assert stats.skipped == 1
        assert stats.added == []
        assert len(chroma.docs) == before

    def test_changed_file_updates_without_duplicate(self, env):
        path = env.docs / "a.txt"
        path.write_text("old", encoding="utf-8")
        chroma = FakeChroma()
        run_sync(env, chroma)

        path.write_text("new", encoding="utf-8")
        stats = run_sync(env, chroma)

        key = key_for(env, "a.txt")
        assert stats.updated == [key]
        assert chroma.contents_for(key) == ["new"]
        assert len(chroma.docs) == 1

    def test_deleted_file_removed(self, env):
        path = env.docs / "a.txt"
        path.write_text("a", encoding="utf-8")
        chroma = FakeChroma()
        run_sync(env, chroma)

        path.unlink()
        stats = run_sync(env, chroma)

        key = key_for(env, "a.txt")
        assert stats.removed == [key]
        assert chroma.keys() == []
        assert load_index_record(env.library, "test-embed").files == {}


class TestExclusions:
    def test_excluded_before_first_sync_is_skipped(self, env):
        (env.docs / "a.txt").write_text("a", encoding="utf-8")
        (env.docs / "b.txt").write_text("b", encoding="utf-8")
        exclude_document(env.library, env.source.id, Path("a.txt"))
        chroma = FakeChroma()

        stats = run_sync(env, chroma)

        assert stats.excluded == 1
        assert chroma.keys() == [key_for(env, "b.txt")]

    def test_exclude_after_sync_removes_chunks(self, env):
        (env.docs / "a.txt").write_text("a", encoding="utf-8")
        chroma = FakeChroma()
        run_sync(env, chroma)
        key = key_for(env, "a.txt")

        exclude_document(env.library, env.source.id, Path("a.txt"))
        stats = run_sync(env, chroma)

        assert key in stats.removed
        assert chroma.keys() == []
        assert key not in load_index_record(env.library, "test-embed").files

    def test_restore_reindexes(self, env):
        (env.docs / "a.txt").write_text("a", encoding="utf-8")
        exclude_document(env.library, env.source.id, Path("a.txt"))
        chroma = FakeChroma()
        run_sync(env, chroma)

        assert restore_document(env.library, env.source.id, Path("a.txt")) is True
        stats = run_sync(env, chroma)

        assert stats.added == [key_for(env, "a.txt")]
        assert chroma.keys() == [key_for(env, "a.txt")]

    def test_list_excluded_reports_fields(self, env):
        (env.docs / "a.txt").write_text("a", encoding="utf-8")
        exclude_document(env.library, env.source.id, Path("a.txt"), reason="test")

        items = list_excluded(env.library)

        assert len(items) == 1
        assert items[0]["doc_key"] == key_for(env, "a.txt")
        assert items[0]["rel_path"] == "a.txt"
        assert items[0]["reason"] == "test"
        assert items[0]["excluded_at"]


class TestFailures:
    def test_extract_failure_keeps_old_index(self, env):
        path = env.docs / "a.txt"
        path.write_text("old", encoding="utf-8")
        chroma = FakeChroma()
        run_sync(env, chroma)
        key = key_for(env, "a.txt")

        path.write_text("new", encoding="utf-8")
        stats = run_sync(env, chroma, extract=boom_extract)

        assert len(stats.failed) == 1
        assert stats.failed[0]["doc_key"] == key
        assert chroma.contents_for(key) == ["old"]
        assert key in load_index_record(env.library, "test-embed").files

    def test_new_file_failure_is_not_recorded(self, env):
        (env.docs / "a.txt").write_text("a", encoding="utf-8")
        chroma = FakeChroma()

        stats = run_sync(env, chroma, extract=boom_extract)

        assert len(stats.failed) == 1
        assert chroma.keys() == []
        assert load_index_record(env.library, "test-embed").files == {}

    def test_scan_limit_raises_before_writing(self, env):
        (env.docs / "a.txt").write_text("a", encoding="utf-8")
        (env.docs / "b.txt").write_text("b", encoding="utf-8")
        chroma = FakeChroma()

        with pytest.raises(ScanLimitExceeded):
            run_sync(
                env,
                chroma,
                scan_config=ScanConfig(
                    supported_extensions=(".txt",),
                    max_files=1,
                ),
            )

        assert chroma.docs == []
        assert not env.library.index_file.exists()


class TestSourceStates:
    def test_offline_source_keeps_records(self, env):
        (env.docs / "a.txt").write_text("a", encoding="utf-8")
        chroma = FakeChroma()
        run_sync(env, chroma)
        before = list(chroma.keys())

        offline = env.tmp_path / "docs_offline"
        env.docs.rename(offline)
        stats = run_sync(env, chroma)

        assert stats.offline_sources == [env.source.id]
        assert chroma.keys() == before
        assert key_for(env, "a.txt") in load_index_record(env.library, "test-embed").files

    def test_disabled_source_keeps_records(self, env):
        (env.docs / "a.txt").write_text("a", encoding="utf-8")
        chroma = FakeChroma()
        run_sync(env, chroma)
        before = list(chroma.keys())

        env.registry.set_source_enabled(env.library.id, env.source.id, False)
        env.library = env.registry.get_library(env.library.id)
        stats = run_sync(env, chroma)

        assert stats.disabled_sources == [env.source.id]
        assert chroma.keys() == before
        assert key_for(env, "a.txt") in load_index_record(env.library, "test-embed").files

    def test_removed_source_records_are_cleaned(self, env):
        (env.docs / "a.txt").write_text("a", encoding="utf-8")
        chroma = FakeChroma()
        run_sync(env, chroma)

        env.registry.remove_source(env.library.id, env.source.id)
        env.library = env.registry.get_library(env.library.id)
        stats = run_sync(env, chroma)

        assert stats.removed
        assert chroma.keys() == []
        assert load_index_record(env.library, "test-embed").files == {}


class TestRebuild:
    def test_embedding_model_change_rebuilds(self, env):
        (env.docs / "a.txt").write_text("a", encoding="utf-8")
        chroma = FakeChroma()
        run_sync(env, chroma, model="model-a")
        resets_before = chroma.reset_count

        stats = run_sync(env, chroma, model="model-b")

        assert stats.rebuild_reason == "embedding_model_changed"
        assert chroma.reset_count == resets_before + 1
        assert stats.added == [key_for(env, "a.txt")]

    def test_old_format_rebuilds(self, env):
        env.library.index_file.write_text(
            json.dumps({"old": {"md5": "x"}}),
            encoding="utf-8",
        )
        (env.docs / "a.txt").write_text("a", encoding="utf-8")
        chroma = FakeChroma()

        stats = run_sync(env, chroma)

        assert stats.rebuild_reason == "old_format"
        assert chroma.reset_count == 1
        assert stats.added == [key_for(env, "a.txt")]


class TestCollectDocuments:
    def test_states_are_derived(self, env):
        a = env.docs / "a.txt"
        b = env.docs / "b.txt"
        a.write_text("a", encoding="utf-8")
        b.write_text("b", encoding="utf-8")
        chroma = FakeChroma()
        run_sync(env, chroma)

        # c 已移出、b 磁盘删除、e 新增未索引
        (env.docs / "c.txt").write_text("c", encoding="utf-8")
        exclude_document(env.library, env.source.id, Path("c.txt"))
        b.unlink()
        (env.docs / "e.txt").write_text("e", encoding="utf-8")

        docs = collect_documents(env.library, scan_config=SCAN)
        by_name = {doc.name: doc for doc in docs}

        assert by_name["a.txt"].state == STATE_INDEXED
        assert by_name["b.txt"].state == STATE_MISSING
        assert by_name["c.txt"].state == STATE_EXCLUDED
        assert by_name["e.txt"].state == STATE_UNINDEXED
        assert by_name["a.txt"].source_label == "docs"

    def test_stale_when_size_or_mtime_changes(self, env):
        path = env.docs / "a.txt"
        path.write_text("old", encoding="utf-8")
        chroma = FakeChroma()
        run_sync(env, chroma)

        path.write_text("new-content", encoding="utf-8")
        docs = collect_documents(env.library, scan_config=SCAN)

        assert docs[0].state == STATE_STALE

    def test_offline_and_disabled_states(self, env):
        (env.docs / "a.txt").write_text("a", encoding="utf-8")
        chroma = FakeChroma()
        run_sync(env, chroma)

        env.registry.set_source_enabled(env.library.id, env.source.id, False)
        env.library = env.registry.get_library(env.library.id)
        docs = collect_documents(env.library, scan_config=SCAN)
        assert docs[0].state == STATE_DISABLED

        env.registry.set_source_enabled(env.library.id, env.source.id, True)
        env.library = env.registry.get_library(env.library.id)
        env.docs.rename(env.tmp_path / "docs_offline")
        docs = collect_documents(env.library, scan_config=SCAN)
        assert docs[0].state == STATE_OFFLINE
