"""
文件名：document_service.py
介绍：文档读写服务层。原先这些函数直接写在 streamlit_file.py 里，
      与 Streamlit 的 UI 代码混在一起，导致无法脱离界面单独测试
      （import streamlit_file 会直接执行整个页面脚本）。
      抽到本模块后，UI 只负责渲染，文档逻辑可以独立单测。

      支持的格式：txt / md / docx / pdf，全部用轻量库实现（pypdf + python-docx），
      不再依赖 unstructured（它会拖入 torch / transformers / spacy，体积 2GB 且从不被用到）。

      本模块同时是「索引」和「界面预览」的唯一抽取入口：
      - extract_documents()  给索引用，产出带 source 元数据的 Document 列表
      - read_document()      给界面预览用，失败时返回提示文本而非抛异常
      两者共用同一套底层抽取，避免「预览看到的」和「实际入库的」不一致。
"""
# 依赖库导入
import os
from pathlib import Path

from langchain_core.documents import Document

# 依赖文件导入
from tools.config_loader import BASE_DIR, resolve_data_dir

# 支持的文件扩展名（索引与上传校验共用）
SUPPORTED_EXTENSIONS = (".txt", ".md", ".docx", ".pdf")

def _normalize_newlines(text: str) -> str:
    """统一换行符为 \\n。

    Windows 下的 CRLF 必须归一化：切分器的分隔符里有 "\\n\\n"，
    而 "\\r\\n\\r\\n" 匹配不上它，段落边界会被忽略、chunk 质量下降。
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _read_text_file(path: Path) -> str:
    """读取文本类文件，自动适配常见中文编码。全部失败时用 gb18030 强制解码兜底。"""
    raw = path.read_bytes()

    # 带 BOM 的 UTF-16 用 BOM 精确判定，不参与下面的猜测——
    # utf-16 解码对任意偶数字节流都可能「成功」，猜错会得到满篇乱码
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return _normalize_newlines(raw.decode("utf-16"))

    # utf-8-sig 能同时处理「纯 UTF-8」和「带 BOM 的 UTF-8」，且会剥掉 BOM
    try:
        return _normalize_newlines(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, UnicodeError):
        pass

    # 国内导出的中文 txt 大量是 GBK/GB18030
    try:
        return _normalize_newlines(raw.decode("gb18030"))
    except (UnicodeDecodeError, UnicodeError):
        pass

    # 兜底：宁可有个别替换字符，也不让整个文件读不进来
    return _normalize_newlines(raw.decode("gb18030", errors="replace"))


def _read_docx(path: Path) -> str:
    """抽取 .docx 正文。除段落外还包含表格内容——只看 paragraphs 会丢掉整张表格。"""
    import docx

    document = docx.Document(str(path))
    parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def _read_pdf(path: Path) -> list[tuple[str, int]]:
    """抽取 PDF 文本，返回 [(页文本, 页码)]，页码从 1 开始。"""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [
        (text, page_no)
        for page_no, page in enumerate(reader.pages, start=1)
        if (text := _normalize_newlines(page.extract_text() or "")).strip()
    ]


def extract_documents(file_path: str) -> list[Document]:
    """
    把文件抽取成 Document 列表（索引用）。

    每个 Document 的 metadata["source"] 严格等于传入的 file_path 字符串——
    增量索引的比对与删除都依赖这个字段完全一致，不能改写路径格式。
    抽取失败会抛异常（由调用方决定是记入 failed 还是回退）。
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    # 白名单校验：不做「未知格式当纯文本读」的兜底，
    # 否则 .xlsx / .zip 之类会被强行解码成乱码 chunk 灌进向量库。
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"不支持的文件格式：{ext or '(无扩展名)'}，"
            f"当前支持 {'/'.join(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".pdf":
        # PDF 按页产出，保留页码信息，也避免整本拼成一个大字符串
        return [
            Document(page_content=text, metadata={"source": file_path, "page": page_no})
            for text, page_no in _read_pdf(path)
        ]

    if ext == ".docx":
        text = _read_docx(path)
    else:
        # txt / md 等文本类格式。md 不做渲染：保留原始标记更利于检索
        text = _read_text_file(path)

    return [Document(page_content=text, metadata={"source": file_path})] if text.strip() else []


def read_document(file_path: str) -> str:
    """读取文档内容用于界面预览。读取失败时返回提示文本而非抛异常。"""
    try:
        documents = extract_documents(file_path)
    except Exception as e:
        return f"（读取失败：{e}）"

    if not documents:
        return "（文件为空或未抽取到文本内容）"

    parts = []
    for doc in documents:
        text = doc.page_content
        # PDF 预览专用：markdown 会折叠单换行，补两个空格保留原始断行。
        # 只在预览路径做，避免污染真正入库的文本。
        if "page" in doc.metadata:
            text = text.replace("\n", "  \n")
        parts.append(text)
    return "\n\n".join(parts)


def get_document_list(chroma=None, data_dir=None) -> list[dict]:
    """返回文档列表，含文件名、索引 chunk 数与索引状态。"""
    if chroma is None:
        from rag.retriever import get_chroma
        chroma = get_chroma()
    data_dir = Path(data_dir) if data_dir is not None else resolve_data_dir()

    # 只列出支持的文件类型，避免把 .DS_Store 之类的东西显示出来
    files = sorted(
        p for p in data_dir.glob("*.*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    # 从向量库统计每个文档（按 source 路径）的 chunk 数
    source_counts: dict[str, int] = {}
    metadatas = chroma.get(include=["metadatas"]).get("metadatas") or []
    for meta in metadatas:
        # 向量库里可能存在没有 source 的历史遗留条目，跳过而不是让文档列表整个崩掉
        source = (meta or {}).get("source")
        if not source:
            continue
        source = os.path.normpath(source)
        source_counts[source] = source_counts.get(source, 0) + 1

    doc_list = []
    for f in files:
        chunks = source_counts.get(os.path.normpath(str(f)), 0)
        doc_list.append({
            "name": f.name,
            "chunks": chunks,
            "indexed": chunks > 0,
            "path": str(f),
        })
    return doc_list


def delete_document(file_path: str, chroma=None) -> None:
    """删除文档：从向量库移除对应 chunk，并从磁盘删除文件。"""
    if chroma is None:
        from rag.retriever import get_chroma
        chroma = get_chroma()

    # 规范化路径，与向量库 metadata 中的 source 保持一致
    normalized = os.path.normpath(file_path)
    # 1. 从向量库删除该文档的所有 chunk（按 source 路径）
    chroma.delete(where={"source": normalized})
    # 2. 从磁盘删除文件
    Path(file_path).unlink(missing_ok=True)
