"""
文件名：agent.py
介绍：定义智能体，包括提示词注入，工具定义，中间件装配。
"""
# 依赖库导入
from langchain.agents import create_agent
from langchain_community.tools import tool
# 依赖文件导入
from tools.web_search import web_search
from middleware import monitor_tool, log_before_model, mode_switch, MODE_PROMPTS
from model.factory import chat_model

# 定义工具：模式切换标记（空工具，仅用于触发中间件动态切换提示词）
@tool(description="当用户明确要求『总结/精读/概括』某篇文档或某个主题时，调用此工具切换到文档精读总结模式")
def switch_to_summary_mode():
    return "switch_to_summary_mode 已调用"

@tool(description="当用户明确要求『生成报告/写一份报告』时，调用此工具切换到报告生成模式")
def switch_to_report_mode():
    return "switch_to_report_mode 已调用"

# 定义智能体
agent = create_agent(
    model=chat_model,
    system_prompt=MODE_PROMPTS["normal"],
    tools=[web_search, switch_to_summary_mode, switch_to_report_mode],
    middleware=[monitor_tool, log_before_model, mode_switch],
)