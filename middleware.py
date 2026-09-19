"""
文件名：middleware.py
介绍：智能体中间件，包含工具监控、模型日志、动态提示词切换三件套。
      实现"普通问答 / 文档精读总结 / 报告生成"三种模式的运行时切换。
"""
# 依赖库导入
from typing import Callable
from datetime import datetime

from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, ModelRequest, dynamic_prompt
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

# 依赖文件导入
from tools.config_loader import BASE_DIR
from tools.logger_handler import logger


# 提示词加载：三种模式对应的系统提示词
def _load_prompt(filename: str) -> str:
    with open(BASE_DIR / "prompts" / filename, "r", encoding="utf-8") as f:
        return f.read()


# 注入当前日期（普通问答模式需要，其他模式也一并带上）
_DATE_SUFFIX = f"\n\n# 当前时间\n现在的日期是 {datetime.now().strftime('%Y年%m月%d日')}。"

MODE_PROMPTS = {
    "normal": _load_prompt("system_prompts.txt") + _DATE_SUFFIX,
    "summary": _load_prompt("summary_prompt.txt") + _DATE_SUFFIX,
    "report": _load_prompt("report_prompt.txt") + _DATE_SUFFIX,
}

# 标记工具名 -> 目标模式 的映射
MODE_SWITCH_TOOLS = {
    "switch_to_summary_mode": "summary",
    "switch_to_report_mode": "report",
}


# 中间件一：工具监控 + 捕获模式切换标记
@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    tool_name = request.tool_call["name"]
    logger.info(f"[tool_monitor] 执行工具：{tool_name}")
    logger.info(f"[tool_monitor] 传入参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[tool_monitor] 工具 {tool_name} 调用成功")

        # 捕获模式切换标记工具，写入 runtime context，供动态提示词切换读取
        if tool_name in MODE_SWITCH_TOOLS:
            request.runtime.context["mode"] = MODE_SWITCH_TOOLS[tool_name]
            logger.info(f"[tool_monitor] 已切换模式 -> {MODE_SWITCH_TOOLS[tool_name]}")

        return result
    except Exception as e:
        logger.error(f"[tool_monitor] 工具 {tool_name} 调用失败，原因：{e}")
        raise


# 中间件二：模型调用前日志
@before_model
def log_before_model(state: AgentState, runtime: Runtime):
    # 防御性读取 context：invoke / stream 都可能不传 context，
    # 此时 runtime.context 是 None，直接 .get 会抛 AttributeError。
    # （与下面 mode_switch 的处理保持一致）
    ctx = getattr(runtime, "context", None) or {}
    logger.info(
        f"[log_before_model] 即将调用模型，带有 {len(state['messages'])} 条信息，"
        f"当前模式：{ctx.get('mode', 'normal')}"
    )
    last = state['messages'][-1]
    last_content = last.content if isinstance(last.content, str) else str(last.content)
    logger.debug(
        f"[log_before_model] 最后一条消息：{type(last).__name__} | {last_content.strip()}"
    )
    return None


# 中间件三：动态提示词切换
@dynamic_prompt
def mode_switch(request: ModelRequest):
    # 防御性读取 context（invoke 等路径可能未显式传入 context）
    ctx = getattr(request.runtime, "context", None) or {}
    mode = ctx.get("mode", "normal")
    return MODE_PROMPTS.get(mode, MODE_PROMPTS["normal"])
