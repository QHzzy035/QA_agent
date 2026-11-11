"""
文件名：test_app_no_key_smoke.py
介绍：无 API Key 的界面冒烟测试。

本项目之前的界面测试需要三个 Key，CI 上一直跳过。现在 Agent 改为惰性创建，
页面在无 Key 时也应能渲染、浏览文档、切换预览。本测试用临时数据目录，
不写入项目 data/，也不依赖网络。
"""
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent.parent / "streamlit_file.py"


def test_page_renders_and_preview_without_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("QA_AGENT_DATA_DIR", str(tmp_path / "data"))
    for key in ("DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == "文档问答助手"
    assert at.chat_input, "对话视图缺少输入框"

    # 切到预览，应显示空状态
    at.radio[0].set_value("预览").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(
        "点击文档旁边的预览按钮" in info.value
        for info in at.info
    )

    # 回到对话，点击第一个文档的预览按钮
    at.radio[0].set_value("对话").run()
    preview_buttons = [b for b in at.sidebar.button if b.label == "👁️"]
    assert preview_buttons, "侧边栏没有预览按钮"
    preview_buttons[0].click().run()

    assert not at.exception, [e.value for e in at.exception]
    assert at.radio[0].value == "预览"
