"""
文件名：generate_docx.py
介绍：批量调用 LLM 生成中文长文档并封装为 Word(.docx) 格式，用于扩充 test_data 测试语料。
用法：在项目根目录执行 python -m tools.generate_docx

说明：.docx 仅依赖 Python 标准库（zipfile + WordprocessingML）生成，不引入额外依赖。
"""
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from langchain.chat_models import init_chat_model

from tools.config_loader import BASE_DIR, LLM_conf

TEST_DATA_DIR = BASE_DIR / "test_data"

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

内容侧重：{guide}

直接输出文章正文，不要输出任何解释性文字。"""

model = init_chat_model(model=LLM_conf["chat_model_name"])


def generate_doc(topic: str, guide: str) -> str:
    prompt = PROMPT_TEMPLATE.format(topic=topic, guide=guide)
    result = model.invoke(prompt)
    return result.content.strip()


# ---------------------------------------------------------------------------
# DOCX 封装（标准库生成最小可用的 Word 文档）
# ---------------------------------------------------------------------------

_CONTENT_TYPES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

_RELS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _docx_document_xml(paragraphs: list[str]) -> str:
    body = "".join(
        f'<w:p><w:r><w:t xml:space="preserve">{_xml_escape(p)}</w:t></w:r></w:p>'
        for p in paragraphs
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f'<w:body>{body}</w:body></w:document>'
    )


def write_docx(save_path: Path, text: str):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    with ZipFile(save_path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _CONTENT_TYPES_XML)
        zf.writestr("_rels/.rels", _RELS_XML)
        zf.writestr("word/document.xml", _docx_document_xml(paragraphs))


def main():
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    success, skipped, failed = 0, 0, 0

    for slug, topic, guide in TOPICS:
        save_path = TEST_DATA_DIR / f"{slug}.docx"
        if save_path.exists():
            print(f"[跳过] {slug}.docx 已存在")
            skipped += 1
            continue

        try:
            content = generate_doc(topic, guide)
            write_docx(save_path, content)
            print(f"[生成] {slug}.docx  ({len(content)} 字)")
            success += 1
        except Exception as e:
            print(f"[失败] {slug}.docx  -> {e}")
            failed += 1

        time.sleep(0.5)  # 简单限流，避免触发 API 频率限制

    print(f"\n完成：成功 {success} 篇，跳过 {skipped} 篇，失败 {failed} 篇")


if __name__ == "__main__":
    main()
