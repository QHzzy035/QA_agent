"""
文件名：test_scanner.py
介绍：来源扫描器测试。

重点：
- 递归/非递归；
- 忽略目录、隐藏文件、临时文件；
- 不支持的扩展名、超大文件；
- 文件数上限与范围预览。
"""
import pytest
from rag.scanner import ScanConfig, preview_scan, scan_source

EXTS = (".txt", ".md", ".pdf")


def config(**kwargs) -> ScanConfig:
    return ScanConfig(supported_extensions=EXTS, **kwargs)


class TestScan:
    def test_recursive_includes_nested(self, tmp_path):
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.md").write_text("b", encoding="utf-8")

        result = scan_source(tmp_path, config())

        rels = {item.rel_path.as_posix() for item in result.files}
        assert rels == {"a.txt", "sub/b.md"}
        assert result.truncated is False

    def test_non_recursive_only_top_level(self, tmp_path):
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.md").write_text("b", encoding="utf-8")

        result = scan_source(tmp_path, config(recursive=False))

        assert [item.rel_path.as_posix() for item in result.files] == ["a.txt"]
        assert result.ignored_entries >= 1

    def test_ignores_common_dirs_and_hidden(self, tmp_path):
        for name in (".git", "node_modules", ".hidden"):
            folder = tmp_path / name
            folder.mkdir()
            (folder / "ignored.txt").write_text("x", encoding="utf-8")
        (tmp_path / "ok.txt").write_text("x", encoding="utf-8")

        result = scan_source(tmp_path, config())

        assert [item.rel_path.as_posix() for item in result.files] == ["ok.txt"]
        assert result.ignored_entries >= 3

    def test_ignores_temp_and_os_files(self, tmp_path):
        (tmp_path / "~$draft.docx").write_text("x", encoding="utf-8")
        (tmp_path / "Thumbs.db").write_text("x", encoding="utf-8")
        (tmp_path / ".DS_Store").write_text("x", encoding="utf-8")
        (tmp_path / "ok.txt").write_text("x", encoding="utf-8")

        result = scan_source(tmp_path, config())

        assert [item.rel_path.as_posix() for item in result.files] == ["ok.txt"]

    def test_unsupported_extension_skipped(self, tmp_path):
        (tmp_path / "a.zip").write_text("x", encoding="utf-8")
        (tmp_path / "b.txt").write_text("x", encoding="utf-8")

        result = scan_source(tmp_path, config())

        assert [item.rel_path.as_posix() for item in result.files] == ["b.txt"]
        assert result.skipped_unsupported == 1

    def test_large_file_skipped(self, tmp_path):
        (tmp_path / "big.txt").write_text("0123456789", encoding="utf-8")

        result = scan_source(tmp_path, config(max_file_size=5))

        assert result.files == []
        assert result.skipped_too_large == 1

    def test_max_files_truncates(self, tmp_path):
        for i in range(3):
            (tmp_path / f"{i}.txt").write_text("x", encoding="utf-8")

        result = scan_source(tmp_path, config(max_files=2))

        assert len(result.files) == 2
        assert result.truncated is True

    def test_exact_max_files_is_not_truncated(self, tmp_path):
        for i in range(2):
            (tmp_path / f"{i}.txt").write_text("x", encoding="utf-8")

        result = scan_source(tmp_path, config(max_files=2))

        assert len(result.files) == 2
        assert result.truncated is False

    def test_extension_match_is_case_insensitive(self, tmp_path):
        (tmp_path / "A.TXT").write_text("x", encoding="utf-8")

        result = scan_source(tmp_path, config())

        assert [item.rel_path.as_posix() for item in result.files] == ["A.TXT"]

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            scan_source(tmp_path / "missing", config())

    def test_sorted_output_is_deterministic(self, tmp_path):
        for name in ("b.txt", "a.txt", "c.txt"):
            (tmp_path / name).write_text("x", encoding="utf-8")

        result = scan_source(tmp_path, config())

        assert [item.rel_path.name for item in result.files] == [
            "a.txt", "b.txt", "c.txt"
        ]


class TestPreview:
    def test_counts_and_estimates(self, tmp_path):
        (tmp_path / "a.txt").write_text("a" * 10, encoding="utf-8")
        (tmp_path / "b.pdf").write_bytes(b"x" * 1000)
        (tmp_path / "skip.zip").write_text("x", encoding="utf-8")

        summary = preview_scan(tmp_path, config(estimated_chunk_size=500))

        assert summary["files"] == 2
        assert summary["by_extension"][".txt"] == 1
        assert summary["by_extension"][".pdf"] == 1
        assert summary["total_bytes"] == 1010
        assert summary["estimated_chunks"] == 2  # txt 按 500 字符估 1，pdf 按二进制估 1
        assert summary["skipped_unsupported"] == 1
        assert summary["truncated"] is False

    def test_truncated_is_reported(self, tmp_path):
        for i in range(3):
            (tmp_path / f"{i}.txt").write_text("x", encoding="utf-8")

        summary = preview_scan(tmp_path, config(max_files=1))

        assert summary["files"] == 1
        assert summary["truncated"] is True
