"""
文件名：agent.py
介绍：定义智能体，包括提示词注入，工具定义，中间件装配。
"""
# 依赖库导入
from langchain.agents import create_agent
from langchain_community.tools import tool
from langchain_tavily import TavilySearch
# 依赖文件导入
from tools.config_loader import tavily_conf
from middleware import monitor_tool, log_before_model, mode_switch, MODE_PROMPTS
from model.factory import chat_model

# 定义tavily对象
tavily_obj = TavilySearch(
    max_results = tavily_conf["max_result"],
    topic = "general",
)

# 定义工具：联网搜索
@tool(description="联网搜索工具 web_search，在无法回答用户问题或用户要求联网搜索时使用")
def web_search(query: str) -> str:
    search_result = tavily_obj.invoke({"query": query})
    all_context = ""
    for f in search_result["results"]:
        all_context += f["content"]
    return all_context

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