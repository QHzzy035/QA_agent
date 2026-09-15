"""
文件名：generate_md.py
介绍：批量调用 LLM 生成中文 Markdown 文档，用于扩充 test_data 测试语料。
用法：在项目根目录执行 python -m tools.generate_md
"""
import time
from pathlib import Path

from tools.config_loader import BASE_DIR

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

PROMPT_TEMPLATE = """你是一名专业的中文科普/技术内容作者。请围绕主题「{topic}」撰写一篇结构清晰、内容详实的 Markdown 文档。

要求：
1. 字数在 800～1200 字之间。
2. 使用 Markdown 语法组织内容：一级标题用 #，小节标题用 ##，适当使用无序列表、加粗、以及一个代码块或表格。
3. 内容要具体，包含具体的数字、年份、专有名词、例子或数据，方便信息检索验证。
4. 语气专业、客观，像百科词条或教程文档，不要出现"作为 AI"之类的表述。

内容侧重：{guide}

直接输出 Markdown 正文，不要输出任何解释性文字。"""



def generate_doc(topic: str, guide: str) -> str:
    # 延迟导入模型：避免本模块被导入时就要求 DEEPSEEK_API_KEY
    from model.factory import chat_model

    prompt = PROMPT_TEMPLATE.format(topic=topic, guide=guide)
    result = chat_model.invoke(prompt)
    return result.content.strip()


def main():
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    success, skipped, failed = 0, 0, 0

    for slug, topic, guide in TOPICS:
        save_path = TEST_DATA_DIR / f"{slug}.md"
        if save_path.exists():
            print(f"[跳过] {slug}.md 已存在")
            skipped += 1
            continue

        try:
            content = generate_doc(topic, guide)
            save_path.write_text(content, encoding="utf-8")
            print(f"[生成] {slug}.md  ({len(content)} 字)")
            success += 1
        except Exception as e:
            print(f"[失败] {slug}.md  -> {e}")
            failed += 1

        time.sleep(0.5)  # 简单限流，避免触发 API 频率限制

    print(f"\n完成：成功 {success} 篇，跳过 {skipped} 篇，失败 {failed} 篇")


if __name__ == "__main__":
    main()
