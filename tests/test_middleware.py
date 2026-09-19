"""
文件名：test_middleware.py
介绍：中间件对「未显式传 context」的容错。

背景：langgraph 的 invoke / stream 都可能不传 context，此时 `runtime.context`
是 **None**。中间件里若直接写 `runtime.context.get(...)`，就会抛
`AttributeError: 'NoneType' object has no attribute 'get'`。

这个坑真实发生过：`streamlit_file.py` 的回退路径
（「stream 失败则 invoke」）没传 context，于是一调就崩 ——
**那条路径本意是兜底，结果自己必崩，等于没有兜底。**

`mode_switch` 早先已做防御性读取，但 `log_before_model` 漏了。
本文件把两条都锁住，防止以后有人把防御删掉。
"""
import sys
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _runtime(context):
    """最小可用的 runtime 替身 —— 这两个中间件只访问 .context。"""
    return SimpleNamespace(context=context)


def _state():
    return {"messages": [HumanMessage("测试消息")]}


class TestNoneContextTolerance:
    """context 为 None 时必须能正常返回，而不是抛异常。"""

    def test_log_before_model_tolerates_none_context(self):
        from middleware import log_before_model

        assert log_before_model.before_model(_state(), _runtime(None)) is None

    def test_log_before_model_works_with_context(self):
        from middleware import log_before_model

        assert log_before_model.before_model(
            _state(), _runtime({"mode": "report"})
        ) is None

    def test_mode_switch_tolerates_none_context(self):
        from middleware import mode_switch

        assert mode_switch.before_model(_state(), _runtime(None)) is None
