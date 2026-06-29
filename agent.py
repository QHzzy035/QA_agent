# 依赖库导入
from datetime import datetime
from langchain.agents import create_agent
from langchain_community.tools import tool
from langchain_tavily import TavilySearch
from langchain_core.globals import set_debug
# 依赖文件导入
from config import settings

# 开启调试模式，在终端打印完整的思考过程
# set_debug(True)

# 读取系统提示词，并注入当前日期
with open(settings.BASE_DIR/"prompts"/"system_prompts.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()
system_prompt += f"\n\n# 当前时间\n现在的日期是 {datetime.now().strftime('%Y年%m月%d日')}。"

# 定义tavily对象
tavily_obj = TavilySearch(
    max_results = settings.max_result,
    topic = "general",
)

# 定义工具
@tool(description="联网搜索工具 web_search，在无法回答用户问题或用户要求联网搜索时使用")
def web_search(query: str) -> str:
    search_result = tavily_obj.invoke({"query": query})
    all_context = ""
    for f in search_result["results"]:
        all_context += f["content"]
    return all_context

# 定义智能体
agent = create_agent(
    model=settings.chat_model_name,
    system_prompt = system_prompt,
    tools=[web_search],
)