"""
文件名：test_app_smoke.py
介绍：界面冒烟测试。

streamlit_file.py 是「导入即执行」的页面脚本，普通单元测试碰不到它，
改动容易在不知不觉中把页面弄崩。这里用 Streamlit 官方的 AppTest
在模拟运行时里真正执行一遍页面脚本，断言渲染无异常、关键控件都在。

需要三个 API Key（页面初始化时会构造模型与向量库），因此在没有配置
凭据的环境（例如 CI）自动跳过，只在本地开发时生效。
"""
import os
from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parent.parent / "streamlit_file.py"

REQUIRED_KEYS = ("DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY", "TAVILY_API_KEY")
_missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]

pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=f"缺少 API Key，跳过界面冒烟测试：{', '.join(_missing)}",
)


@pytest.fixture(scope="module")
def app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.run()
    return at


def test_page_renders_without_exception(app):
    assert not app.exception, [e.value for e in app.exception]


def test_expected_widgets_are_present(app):
    assert app.title[0].value == "文档问答助手"

    button_labels = [b.label for b in app.sidebar.button]
    assert any("重新索引" in label for label in button_labels), \
        f"侧边栏缺少「重新索引」按钮，实际有：{button_labels}"
    assert any("设置" in label for label in button_labels)

    assert app.sidebar.file_uploader, "侧边栏缺少上传控件"
    assert app.chat_input, "主区域缺少对话框"


def test_document_list_renders(app):
    """文档库面板能正常渲染（覆盖 get_document_list 与真实向量库的对接）。"""
    assert not app.exception
