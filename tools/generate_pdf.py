"""
文件名：generate_pdf.py
介绍：批量调用 LLM 生成中文长文档并输出为 PDF 格式，用于扩充 test_data 测试语料。
用法：在项目根目录执行 python -m tools.generate_pdf

说明：PDF 使用标准库 + 嵌入系统中文字体（simhei.ttf 等）生成，不引入额外依赖。
"""
import struct
import time
import zlib
from pathlib import Path

from model.factory import chat_model
from tools.config_loader import BASE_DIR

TEST_DATA_DIR = BASE_DIR / "test_data"

# 候选中文字体（TrueType .ttf，按优先级）
CJK_FONT_CANDIDATES = [
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsunb.ttf",
    "C:/Windows/Fonts/simfang.ttf",
    "C:/Windows/Fonts/simkai.ttf",
    "C:/Windows/Fonts/Deng.ttf",
    "C:/Windows/Fonts/NotoSansSC-VF.ttf",
]

# 主题列表：(英文文件名, 中文主题, 内容侧重引导)
TOPICS = [
    ("large_language_models", "大语言模型基础", "Transformer 架构、注意力机制、预训练与微调"),
    ("prompt_engineering", "提示词工程技巧", "提示词结构、角色设定、思维链、few-shot 示例"),
    ("computer_vision", "计算机视觉入门", "图像分类、目标检测、卷积神经网络、常见应用"),
    ("recommendation_system", "推荐系统原理", "协同过滤、基于内容的推荐、深度学习推荐、冷启动问题"),
    ("blockchain_basics", "区块链技术基础", "分布式账本、共识机制、智能合约、典型应用场景"),
    ("cloud_computing", "云计算与虚拟化", "IaaS/PaaS/SaaS、虚拟机、容器、主流云平台对比"),
    ("internet_of_things", "物联网概述", "传感器、通信协议、边缘计算、智慧家居应用"),
    ("robotics_intro", "机器人技术入门", "机械结构、传感器、运动控制、ROS 框架、应用"),
    ("bioinformatics", "生物信息学基础", "基因测序、序列比对、常用数据库、基因组学应用"),
    ("renewable_energy", "可再生能源", "太阳能、风能、储能技术、碳中和、政策趋势"),
]

PROMPT_TEMPLATE = """你是一名专业的中文科普/技术内容作者。请围绕主题「{topic}」撰写一篇结构清晰、内容详实的文章。

要求：
1. 字数在 800～1200 字之间。
2. 第一行是文章标题（纯文字，不要用 # 或 markdown 标记）。
3. 正文包含 3～5 个小节，每节有简短的小标题。
4. 内容要具体，包含具体的数字、年份、专有名词、例子或数据，方便信息检索验证。
5. 语气专业、客观，像百科词条或教程文档，不要出现"作为 AI"之类的表述。
6. 只使用常见汉字、数字和标点，不要使用 emoji、生僻字或特殊符号。

内容侧重：{guide}

直接输出文章正文，不要输出任何解释性文字。"""

model = chat_model


def generate_doc(topic: str, guide: str) -> str:
    prompt = PROMPT_TEMPLATE.format(topic=topic, guide=guide)
    result = model.invoke(prompt)
    return result.content.strip()


# ---------------------------------------------------------------------------
# PDF 生成（标准库，嵌入 TrueType 中文字体）
# ---------------------------------------------------------------------------

def _u16(data: bytes, off: int) -> int:
    return struct.unpack(">H", data[off:off + 2])[0]


def _i16(data: bytes, off: int) -> int:
    return struct.unpack(">h", data[off:off + 2])[0]


def _u32(data: bytes, off: int) -> int:
    return struct.unpack(">I", data[off:off + 4])[0]


def _sfnt_tables(ttf: bytes) -> dict[bytes, tuple[int, int]]:
    num_tables = _u16(ttf, 4)
    tables = {}
    for i in range(num_tables):
        off = 12 + i * 16
        tag = ttf[off:off + 4]
        tables[tag] = (_u32(ttf, off + 8), _u32(ttf, off + 12))
    return tables


def _parse_cmap(ttf: bytes) -> dict[int, int]:
    """解析 TrueType cmap 表，返回 {unicode 码点: glyph id}。"""
    cmap_off, _ = _sfnt_tables(ttf)[b"cmap"]
    num_sub = _u16(ttf, cmap_off + 2)
    subs = []
    for i in range(num_sub):
        rec = cmap_off + 4 + i * 8
        subs.append((_u16(ttf, rec), _u16(ttf, rec + 2), _u32(ttf, rec + 4)))
    # 优先：Windows(3) Unicode full(10) → Windows(3) BMP(1) → Unicode(0) full(4)/BMP(3)
    order = [(3, 10), (3, 1), (0, 4), (0, 3), (0, 6), (0, 0)]
    for plat, enc in order:
        for p, e, off in subs:
            if (p, e) != (plat, enc):
                continue
            fmt = _u16(ttf, cmap_off + off)
            if fmt == 4:
                return _cmap_format4(ttf, cmap_off + off)
            if fmt == 12:
                return _cmap_format12(ttf, cmap_off + off)
            if fmt == 0:
                return {c: ttf[cmap_off + off + 6 + c] for c in range(256)
                        if ttf[cmap_off + off + 6 + c] != 0}
    return {}


def _cmap_format4(ttf: bytes, off: int) -> dict[int, int]:
    seg_count = _u16(ttf, off + 6) // 2
    end_codes = struct.unpack(f">{seg_count}H", ttf[off + 14:off + 14 + seg_count * 2])
    start_codes = struct.unpack(
        f">{seg_count}H", ttf[off + 16 + seg_count * 2:off + 16 + seg_count * 4]
    )
    id_delta = struct.unpack(
        f">{seg_count}h", ttf[off + 16 + seg_count * 4:off + 16 + seg_count * 6]
    )
    id_range_off = off + 16 + seg_count * 6
    id_range = struct.unpack(
        f">{seg_count}H", ttf[id_range_off:id_range_off + seg_count * 2]
    )
    result = {}
    for i in range(seg_count):
        if end_codes[i] == 0xFFFF:
            continue
        for c in range(start_codes[i], end_codes[i] + 1):
            if id_range[i] == 0:
                gid = (c + id_delta[i]) & 0xFFFF
            else:
                addr = id_range_off + i * 2 + id_range[i] + (c - start_codes[i]) * 2
                gid = _u16(ttf, addr)
                if gid != 0:
                    gid = (gid + id_delta[i]) & 0xFFFF
            result[c] = gid
    return result


def _cmap_format12(ttf: bytes, off: int) -> dict[int, int]:
    n_groups = _u32(ttf, off + 12)
    result = {}
    p = off + 16
    for _ in range(n_groups):
        start_char, end_char, start_glyph = struct.unpack(">III", ttf[p:p + 12])
        p += 12
        for c in range(start_char, end_char + 1):
            result[c] = start_glyph + (c - start_char)
    return result


def _to_unicode_cmap(unique: dict[int, int]) -> bytes:
    """按 glyph id -> unicode 码点构建 ToUnicode CMap 内容。"""
    entries = [f"<{gid:04x}> <{cp:04x}>" for gid, cp in unique.items() if cp <= 0xFFFF]
    bfchar = "\n".join(entries)
    return (
        "/CIDInit /ProcSet findresource begin\n"
        "12 dict begin\n"
        "begincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        "/CMapName /Adobe-Identity-UCS def\n"
        "/CMapType 2 def\n"
        "1 begincodespacerange\n"
        "<0000> <FFFF>\n"
        "endcodespacerange\n"
        f"{len(entries)} beginbfchar\n"
        f"{bfchar}\n"
        "endbfchar\n"
        "endcmap\n"
        "CMapName currentdict /CMap defineresource pop\n"
        "end\n"
        "end\n"
    ).encode("ascii")


def _font_metrics(ttf: bytes) -> tuple[list[float], int, int]:
    tables = _sfnt_tables(ttf)
    head_off, _ = tables[b"head"]
    hhea_off, _ = tables[b"hhea"]
    units_per_em = _u16(ttf, head_off + 18)
    scale = 1000.0 / units_per_em
    bbox = [
        round(_i16(ttf, head_off + 36) * scale),
        round(_i16(ttf, head_off + 38) * scale),
        round(_i16(ttf, head_off + 40) * scale),
        round(_i16(ttf, head_off + 42) * scale),
    ]
    ascent = round(_i16(ttf, hhea_off + 4) * scale)
    descent = round(_i16(ttf, hhea_off + 6) * scale)
    return bbox, ascent, descent


def _wrap_line(line: str, width: int = 42) -> list[str]:
    """按字符宽度折行，避免超出页面。"""
    parts = []
    while len(line) > width:
        parts.append(line[:width])
        line = line[width:]
    if line:
        parts.append(line)
    return parts


def write_pdf(save_path: Path, text: str):
    font_path = next((p for p in CJK_FONT_CANDIDATES if Path(p).exists()), None)
    if font_path is None:
        raise FileNotFoundError("未找到可用的中文字体（.ttf），无法生成 PDF")

    ttf = Path(font_path).read_bytes()
    cmap = _parse_cmap(ttf)

    # 收集用到的字符 -> glyph id，用于构建 ToUnicode CMap
    unique = {}
    for ch in set(text):
        gid = cmap.get(ord(ch), 0)
        if gid > 0:
            unique[gid] = ord(ch)

    bbox, ascent, descent = _font_metrics(ttf)

    # 内容流：按段落折行、逐行绘制
    content_parts = ["BT", "/F1 12 Tf"]
    y = 750
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            y -= 8
            continue
        for line in _wrap_line(para):
            if y < 50:
                break
            gids = [cmap.get(ord(ch), 0) for ch in line]
            hex_str = "".join(f"{g:04x}" for g in gids)
            content_parts.append(f"50 {y} Td")
            content_parts.append(f"<{hex_str}> Tj")
            y -= 16
    content_parts.append("ET")
    content_stream = "\n".join(content_parts).encode("ascii")

    to_unicode = _to_unicode_cmap(unique)
    font_compressed = zlib.compress(ttf, 9)

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type0 /BaseFont /CJKFont "
        b"/Encoding /Identity-H /DescendantFonts [7 0 R] /ToUnicode 6 0 R >>",
        b"<< /Length " + str(len(content_stream)).encode() + b" >>\nstream\n"
        + content_stream + b"\nendstream",
        b"<< /Length " + str(len(to_unicode)).encode() + b" >>\nstream\n"
        + to_unicode + b"\nendstream",
        b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /CJKFont "
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
        b"/FontDescriptor 8 0 R /DW 1000 /CIDToGIDMap /Identity >>",
        b"<< /Type /FontDescriptor /FontName /CJKFont /Flags 4 "
        + f"/FontBBox [{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}] ".encode()
        + f"/ItalicAngle 0 /Ascent {ascent} /Descent {descent} "
          f"/CapHeight {ascent} /StemV 80 /FontFile2 9 0 R >>".encode(),
        b"<< /Length " + str(len(font_compressed)).encode()
        + b" /Filter /FlateDecode /Length1 " + str(len(ttf)).encode() + b" >>\nstream\n"
        + font_compressed + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos = len(out)
    count = len(objects) + 1
    out += f"xref\n0 {count}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    ).encode()

    save_path.write_bytes(bytes(out))


def main():
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    success, skipped, failed = 0, 0, 0

    for slug, topic, guide in TOPICS:
        save_path = TEST_DATA_DIR / f"{slug}.pdf"
        if save_path.exists():
            print(f"[跳过] {slug}.pdf 已存在")
            skipped += 1
            continue

        try:
            content = generate_doc(topic, guide)
            write_pdf(save_path, content)
            print(f"[生成] {slug}.pdf  ({len(content)} 字)")
            success += 1
        except Exception as e:
            print(f"[失败] {slug}.pdf  -> {e}")
            failed += 1

        time.sleep(0.5)  # 简单限流，避免触发 API 频率限制

    print(f"\n完成：成功 {success} 篇，跳过 {skipped} 篇，失败 {failed} 篇")


if __name__ == "__main__":
    main()
