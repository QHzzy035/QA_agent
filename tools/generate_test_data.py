"""
文件名：generate_test_data.py
介绍：批量调用 LLM 生成中文长文档，用于扩充 test_data 测试语料。
用法：python generate_test_data.py
"""
import time
from pathlib import Path

from langchain.chat_models import init_chat_model

from tools.config_loader import BASE_DIR, LLM_conf

TEST_DATA_DIR = BASE_DIR / "test_data"

# 主题列表：(英文文件名, 中文主题, 内容侧重引导)
TOPICS = [
    # 技术类
    ("python_advanced", "Python 高级特性", "生成器、装饰器、上下文管理器、异步编程"),
    ("machine_learning_algorithms", "机器学习常用算法", "决策树、随机森林、支持向量机、神经网络的原理与对比"),
    ("sql_database", "关系型数据库与 SQL", "表结构设计、索引、事务、常用 SQL 语句"),
    ("docker_container", "Docker 容器化技术", "镜像、容器、Dockerfile、数据卷与网络"),
    ("linux_commands", "Linux 常用命令", "文件操作、进程管理、权限、管道与重定向"),
    ("javascript_frontend", "前端 JavaScript 基础", "变量、函数、DOM 操作、事件、异步请求"),
    ("data_structure_algorithm", "数据结构与算法", "数组、链表、栈、队列、树、图与常见排序查找"),
    ("network_security", "网络安全基础", "常见攻击方式、加密、认证、防护措施"),
    ("git_version_control", "Git 版本控制", "分支、合并、提交、回滚、协作流程"),
    ("web_framework", "后端 Web 框架对比", "Django、Flask、FastAPI 的适用场景与差异"),
    # 科学类
    ("quantum_computing", "量子计算入门", "量子比特、叠加态、纠缠、量子门与潜在应用"),
    ("black_hole", "黑洞与广义相对论", "事件视界、奇点、引力波、观测证据"),
    ("gene_editing_crispr", "基因编辑技术 CRISPR", "原理、应用、伦理争议、发展历程"),
    ("climate_change", "全球气候变化", "温室效应、碳排放、极端天气、应对措施"),
    ("ai_history", "人工智能发展简史", "图灵测试、专家系统、深度学习、大模型时代"),
    # 人文类
    ("modern_chinese_history", "中国近代史概述", "鸦片战争、洋务运动、辛亥革命、五四运动"),
    ("world_geography", "世界地理常识", "七大洲、主要国家、气候带、自然资源"),
    ("classical_poetry", "中国古典诗词赏析", "唐诗宋词、格律、代表诗人与名篇"),
    ("western_philosophy", "西方哲学流派", "古希腊哲学、理性主义、经验主义、存在主义"),
    ("economics_principles", "经济学基本原理", "供需关系、机会成本、通货膨胀、货币政策"),
    ("psychology_intro", "心理学入门", "认知、情绪、记忆、人格、常见心理效应"),
    ("legal_common_sense", "日常生活法律常识", "合同、劳动权益、消费者保护、知识产权"),
    ("traditional_chinese_medicine", "中医学基础理论", "阴阳五行、脏腑、经络、辨证论治"),
    # 生活类
    ("healthy_diet", "健康饮食指南", "营养均衡、膳食结构、常见误区、食谱建议"),
    ("fitness_exercise", "运动健身入门", "有氧与无氧、力量训练、拉伸、训练计划"),
    ("travel_guide", "旅行攻略技巧", "行程规划、预算、签证、安全、小众目的地"),
    ("workplace_communication", "职场沟通技巧", "向上汇报、跨部门协作、邮件写作、冲突处理"),
    ("personal_finance", "个人理财规划", "储蓄、基金、股票、保险、资产配置"),
    ("coffee_knowledge", "咖啡知识科普", "豆种、烘焙、冲煮方式、风味、咖啡因"),
    ("pet_care", "宠物饲养指南", "常见宠物、喂养、疫苗、训练、常见疾病"),
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


def main():
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    success, skipped, failed = 0, 0, 0

    for slug, topic, guide in TOPICS:
        save_path = TEST_DATA_DIR / f"{slug}.txt"
        if save_path.exists():
            print(f"[跳过] {slug}.txt 已存在")
            skipped += 1
            continue

        try:
            content = generate_doc(topic, guide)
            save_path.write_text(content, encoding="utf-8")
            print(f"[生成] {slug}.txt  ({len(content)} 字)")
            success += 1
        except Exception as e:
            print(f"[失败] {slug}.txt  -> {e}")
            failed += 1

        time.sleep(0.5)  # 简单限流，避免触发 API 频率限制

    print(f"\n完成：成功 {success} 篇，跳过 {skipped} 篇，失败 {failed} 篇")


if __name__ == "__main__":
    main()