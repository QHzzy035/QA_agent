"""
文件名：test_indexer.py
介绍：增量索引测试。这是项目里最容易悄悄出错、又最烧钱（重复调 embedding）的一块，
      重点覆盖「新增/变更/删除/跳过」四类判定，以及失败文件不拖垮整批的容错分支。
"""
from rag.indexer import get_file_md5_hex


class TestFileMd5:
    def test_same_content_same_hash(self, tmp_path):
        a, b = tmp_path / "a.txt", tmp_path / "b.txt"
        a.write_text("同样的内容", encoding="utf-8")
        b.write_text("同样的内容", encoding="utf-8")
        assert get_file_md5_hex(str(a)) == get_file_md5_hex(str(b))

    def test_content_change_changes_hash(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("内容 A", encoding="utf-8")
        before = get_file_md5_hex(str(f))
        f.write_text("内容 B", encoding="utf-8")
        assert get_file_md5_hex(str(f)) != before

    def test_filename_does_not_affect_hash(self, tmp_path):
        a, b = tmp_path / "a.txt", tmp_path / "b.txt"
        a.write_text("同一份内容", encoding="utf-8")
        b.write_text("同一份内容", encoding="utf-8")
        assert get_file_md5_hex(str(a)) == get_file_md5_hex(str(b))

    def test_reads_binary_in_chunks(self, tmp_path):
        """跨 4096 字节分块边界的文件也要算对（分块读取是这里唯一的实现细节）。"""
        f = tmp_path / "big.bin"
        payload = bytes(range(256)) * 100  # 25600 字节，远超分块大小
        f.write_bytes(payload)
        import hashlib
        assert get_file_md5_hex(str(f)) == hashlib.md5(payload).hexdigest()


class TestIncrementalIndex:
    def test_first_run_indexes_everything_and_resets(self, index_env, fake_loader):
        fake_loader()
        index_env.write("a.txt", "第一行\n第二行")
        index_env.write("b.txt", "只有一行")

        stats = index_env.run()

        assert len(stats["added"]) == 2
        assert stats["skipped"] == 0
        assert stats["failed"] == []
        # 记录文件不存在 = 向量库可能是旧的全量数据，必须先清空
        assert index_env.chroma.reset_count == 1
        assert len(index_env.chroma.docs) == 3

    def test_second_run_skips_unchanged_files(self, index_env, fake_loader):
        fake_loader()
        index_env.write("a.txt", "第一行\n第二行")
        index_env.run()

        before = len(index_env.chroma.docs)
        stats = index_env.run()

        assert stats["added"] == [] and stats["updated"] == []
        assert stats["skipped"] == 1
        # 关键：没有重复写入向量库，也就没有重复的 embedding 花费
        assert len(index_env.chroma.docs) == before
        # 已经不是首次运行，不应再清空向量库
        assert index_env.chroma.reset_count == 1

    def test_modified_file_is_reindexed_not_duplicated(self, index_env, fake_loader):
        fake_loader()
        path = index_env.write("a.txt", "第一行\n第二行")
        index_env.run()

        index_env.write("a.txt", "改过的第一行")
        stats = index_env.run()

        assert stats["updated"] == [path]
        assert stats["added"] == []
        # 旧 chunk 必须先删掉，否则新旧内容会同时被检索到
        assert {"source": path} in index_env.chroma.delete_calls
        assert [d["content"] for d in index_env.chroma.docs_of(path)] == ["改过的第一行"]

    def test_deleted_file_is_removed_from_store_and_record(self, index_env, fake_loader):
        fake_loader()
        path = index_env.write("a.txt", "会被删掉")
        index_env.write("keep.txt", "留下来的")
        index_env.run()

        (index_env.data / "a.txt").unlink()
        stats = index_env.run()

        assert stats["removed"] == [path]
        assert index_env.chroma.docs_of(path) == []
        assert path not in index_env.record.read_text(encoding="utf-8")

    def test_failed_file_does_not_abort_the_batch(self, index_env, fake_loader):
        """一个坏文件不能拖垮整批索引，也不能被写进记录（否则永远不再重试）。"""
        fake_loader(fail_on={"broken.txt"})
        index_env.write("ok1.txt", "好的内容一")
        index_env.write("broken.txt", "坏文件")
        index_env.write("ok2.txt", "好的内容二")

        stats = index_env.run()

        assert len(stats["added"]) == 2
        assert len(stats["failed"]) == 1
        assert stats["failed"][0].endswith("broken.txt")
        assert "broken.txt" not in index_env.record.read_text(encoding="utf-8")
        # 两个好文件仍然入库了
        assert len(index_env.chroma.docs) == 2

    def test_failed_file_is_retried_on_next_run(self, index_env, fake_loader, monkeypatch):
        fake_loader(fail_on={"flaky.txt"})
        index_env.write("flaky.txt", "内容")
        assert len(index_env.run()["failed"]) == 1

        # 换成能解析的加载器，模拟文件被修复
        fake_loader()
        stats = index_env.run()
        assert len(stats["added"]) == 1
        assert stats["failed"] == []

    def test_empty_file_is_recorded_without_chunks(self, index_env, fake_loader):
        """空文件不算失败（记录进索引表，避免每轮重复尝试），但不产生 chunk。"""
        fake_loader(empty={"empty.txt"})
        index_env.write("empty.txt", "")

        stats = index_env.run()

        assert len(stats["added"]) == 1
        assert stats["failed"] == []
        assert index_env.chroma.docs == []

    def test_corrupt_record_triggers_rebuild(self, index_env, fake_loader):
        """索引记录损坏时应自动全量重建，而不是让整个索引流程挂掉。"""
        fake_loader()
        index_env.record.write_text("{ 这不是合法 JSON", encoding="utf-8")
        index_env.write("a.txt", "内容")

        stats = index_env.run()

        assert len(stats["added"]) == 1
        # 记录不可信 => 向量库内容也不可信，必须清空重建
        assert index_env.chroma.reset_count == 1
        assert index_env.record.read_text(encoding="utf-8").strip().startswith("{")

    def test_unsupported_files_are_ignored(self, index_env, fake_loader):
        """非支持格式不能被当纯文本硬读后灌进向量库。"""
        fake_loader()
        index_env.write("a.txt", "正常内容")
        (index_env.data / "archive.zip").write_bytes(b"PK\x03\x04")
        (index_env.data / "table.xlsx").write_bytes(b"PK\x03\x04")

        stats = index_env.run()

        assert len(stats["added"]) == 1
        assert stats["failed"] == []
        assert len(index_env.chroma.docs) == 1

    def test_record_uses_normalized_paths(self, index_env, fake_loader):
        """记录 key 与向量库 metadata 的 source 必须完全一致，否则删除/增量都会失效。"""
        fake_loader()
        path = index_env.write("a.txt", "内容")
        index_env.run()

        import json
        record = json.loads(index_env.record.read_text(encoding="utf-8"))
        assert list(record["files"].keys()) == [path]
        assert index_env.chroma.docs[0]["source"] == path

    def test_embedding_model_change_triggers_rebuild(self, index_env, fake_loader):
        """换了嵌入模型必须全量重建。

        这是本项目真实踩过的坑：增量索引只看文件 MD5，换了模型后 MD5 一个都没变，
        于是全部文件被跳过 —— 库里留着旧模型的向量，查询却用新模型编码，
        两个向量空间混在一起，检索不报错但结果完全失真。
        """
        fake_loader()
        path = index_env.write("a.txt", "内容")
        index_env.run(embed_model="model-a")
        assert index_env.chroma.reset_count == 1

        # 文件内容一个字节没动，只有嵌入模型变了
        stats = index_env.run(embed_model="model-b")

        assert index_env.chroma.reset_count == 2, "换模型必须触发全量重建"
        assert stats["added"] == [path], "应重新索引，而不是被跳过"
        assert stats["skipped"] == 0
        assert len(index_env.chroma.docs_of(path)) == 1, "重建后不应有重复"

    def test_same_embedding_model_still_skips(self, index_env, fake_loader):
        """对照组：模型没变时仍然走增量，别把重建做成每次都触发。"""
        fake_loader()
        index_env.write("a.txt", "内容")
        index_env.run(embed_model="model-a")
        stats = index_env.run(embed_model="model-a")

        assert index_env.chroma.reset_count == 1
        assert stats["skipped"] == 1
        assert stats["added"] == []

    def test_legacy_flat_record_triggers_rebuild(self, index_env, fake_loader):
        """旧版扁平记录（不含嵌入模型信息）同样要重建 —— 无从判断向量空间。"""
        import json
        fake_loader()
        path = index_env.write("a.txt", "内容")
        index_env.record.write_text(json.dumps({path: "deadbeef"}), encoding="utf-8")

        stats = index_env.run(embed_model="model-a")

        assert index_env.chroma.reset_count == 1
        assert stats["added"] == [path]
