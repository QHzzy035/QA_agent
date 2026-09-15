"""
文件名：test_pdf_writer.py
介绍：纯标准库 PDF 生成器测试。

这段代码手写 TrueType 字体内嵌、cmap 解析与 xref 表，属于「平时看不出来、
坏了才发现文件打不开」的那类逻辑，值得用测试固定住输出结构。
"""
from pathlib import Path

import pytest

from tools import pdf_writer
from tools.pdf_writer import find_cjk_font, write_pdf, _wrap_line

requires_font = pytest.mark.skipif(
    find_cjk_font() is None, reason="系统中没有可用的中文字体"
)


class TestWrapLine:
    def test_short_line_is_untouched(self):
        assert _wrap_line("短句") == ["短句"]

    def test_long_line_is_split_into_equal_widths(self):
        text = "字" * 100
        parts = _wrap_line(text, width=42)
        assert [len(p) for p in parts] == [42, 42, 16]
        assert "".join(parts) == text

    def test_exact_multiple_does_not_emit_trailing_empty(self):
        assert _wrap_line("字" * 84, width=42) == ["字" * 42, "字" * 42]

    def test_empty_line_yields_nothing(self):
        assert _wrap_line("") == []


class TestFindCjkFont:
    def test_returns_existing_path_or_none(self):
        font = find_cjk_font()
        if font is not None:
            assert Path(font).exists()


class TestWritePdf:
    @requires_font
    def test_produces_structurally_valid_pdf(self, tmp_path):
        out = tmp_path / "out.pdf"
        write_pdf(out, "标题\n正文内容")

        data = out.read_bytes()
        assert data.startswith(b"%PDF-")
        assert data.rstrip().endswith(b"%%EOF")
        assert b"xref" in data and b"/Type /Catalog" in data
        assert b"/ToUnicode" in data, "缺少 ToUnicode CMap，PDF 里的中文将无法被复制或检索"

    @requires_font
    def test_xref_offsets_point_at_real_objects(self, tmp_path):
        """xref 偏移写错的话，PDF 阅读器会直接判定文件损坏。"""
        out = tmp_path / "out.pdf"
        write_pdf(out, "内容")
        data = out.read_bytes()

        start = data.rindex(b"startxref")
        xref_pos = int(data[start:].split(b"\n")[1])
        assert data[xref_pos:xref_pos + 4] == b"xref"

        # 每个 n 条目的偏移处应当是一个 "N 0 obj" 开头
        for line in data[xref_pos:].split(b"\n")[3:8]:
            if not line.rstrip().endswith(b" n"):
                continue
            off = int(line.split()[0])
            assert data[off:off + 20].split(b"\n")[0].endswith(b" obj"), \
                f"偏移 {off} 未指向对象"

    @requires_font
    def test_writes_truetype_font_stream(self, tmp_path):
        out = tmp_path / "out.pdf"
        write_pdf(out, "中文测试")
        assert b"/FontFile2" in out.read_bytes()

    @requires_font
    def test_long_text_does_not_crash_and_stays_valid(self, tmp_path):
        """文本超出单页高度时应截断而不是写出坏文件。"""
        out = tmp_path / "out.pdf"
        write_pdf(out, "\n".join(f"第{i}段内容" for i in range(200)))

        assert out.read_bytes().rstrip().endswith(b"%%EOF")

    def test_raises_when_no_font_available(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pdf_writer, "find_cjk_font", lambda: None)
        with pytest.raises(FileNotFoundError):
            write_pdf(tmp_path / "out.pdf", "内容")

    @requires_font
    def test_uses_explicit_font_path(self, tmp_path):
        out = tmp_path / "out.pdf"
        write_pdf(out, "内容", font_path=find_cjk_font())
        assert out.exists()
