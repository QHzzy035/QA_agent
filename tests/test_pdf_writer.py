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


def _wrap_as_ttc(ttf: bytes) -> bytes:
    """把一个单字体 .ttf 包装成只含一个字体的 .ttc（TrueType Collection）。

    .ttc 布局：'ttcf' + 版本 + 字体数 + 各字体的表目录偏移，
    其后才是字体数据。因此表目录里每个表的偏移量都要整体后移一个 header 的长度。
    """
    import struct

    num_tables = struct.unpack(">H", ttf[4:6])[0]
    header_size = 16
    dir_end = 12 + num_tables * 16

    shifted = bytearray(ttf[:dir_end])
    for i in range(num_tables):
        off = 12 + i * 16
        old = struct.unpack(">I", shifted[off + 8:off + 12])[0]
        shifted[off + 8:off + 12] = struct.pack(">I", old + header_size)

    header = (b"ttcf" + struct.pack(">I", 0x00010000)
              + struct.pack(">I", 1) + struct.pack(">I", header_size))
    return bytes(header) + bytes(shifted) + ttf[dir_end:]


class TestFontCollectionSupport:
    """Linux / macOS 上的中文字体大多是 .ttc（字体集合），与单字体 .ttf 布局不同。

    只按 .ttf 的布局解析会把 .ttc 的版本号当成表数量读出垃圾数据，
    随后在查找 cmap 表时抛异常 —— 而候选字体列表里恰好列了 .ttc 路径。
    """

    def test_sfnt_base_distinguishes_ttf_from_ttc(self):
        import struct

        ttf_like = b"\x00\x01\x00\x00" + b"\x00" * 20
        assert pdf_writer._sfnt_base(ttf_like) == 0

        ttc_like = (b"ttcf" + struct.pack(">I", 0x00010000)
                    + struct.pack(">I", 1) + struct.pack(">I", 16) + b"\x00" * 4)
        assert pdf_writer._sfnt_base(ttc_like) == 16

    @requires_font
    def test_ttc_collection_is_parsed_like_the_original_ttf(self):
        font = find_cjk_font()
        raw = Path(font).read_bytes()
        if raw[:4] == b"ttcf":
            pytest.skip("系统字体本身就是 .ttc，无法做「包装前后对照」")

        ttc = _wrap_as_ttc(raw)

        # cmap 是解析正确性的关键证据：28522 个映射全都依赖从正确偏移读取
        assert pdf_writer._parse_cmap(ttc) == pdf_writer._parse_cmap(raw)
        assert pdf_writer._font_metrics(ttc) == pdf_writer._font_metrics(raw)

    @requires_font
    def test_write_pdf_accepts_a_ttc_font(self, tmp_path):
        font = find_cjk_font()
        raw = Path(font).read_bytes()
        if raw[:4] == b"ttcf":
            ttc_path = Path(font)
        else:
            ttc_path = tmp_path / "wrapped.ttc"
            ttc_path.write_bytes(_wrap_as_ttc(raw))

        out = tmp_path / "out.pdf"
        write_pdf(out, "字体集合测试内容", font_path=str(ttc_path))

        from pypdf import PdfReader
        text = "\n".join(p.extract_text() or "" for p in PdfReader(str(out)).pages)
        assert "字体集合测试内容" in text


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
