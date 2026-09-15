"""
文件名：test_prompts.py
介绍：提示词与模式配置测试。

三种模式的提示词在模块导入时就被读取，任何一次误删/改名都会让应用直接起不来；
另外「标记工具 -> 目标模式」的映射如果没有对应的提示词，运行时会静默回退到普通问答，
用户看到的是「切了模式但没生效」——这类问题很难排查，用测试挡住。
"""
from datetime import datetime

import pytest

from middleware import MODE_PROMPTS, MODE_SWITCH_TOOLS
from tools.config_loader import BASE_DIR

EXPECTED_MODES = {"normal", "summary", "report"}


class TestModePrompts:
    def test_all_modes_are_registered(self):
        assert set(MODE_PROMPTS) == EXPECTED_MODES

    @pytest.mark.parametrize("mode", sorted(EXPECTED_MODES))
    def test_prompt_is_not_empty(self, mode):
        assert MODE_PROMPTS[mode].strip(), f"{mode} 模式的提示词为空"

    @pytest.mark.parametrize("mode", sorted(EXPECTED_MODES))
    def test_prompt_carries_current_date(self, mode):
        """普通问答需要知道「今天」是什么时候，否则回答时间类问题会出错。"""
        today = datetime.now().strftime("%Y年%m月%d日")
        assert today in MODE_PROMPTS[mode]

    def test_prompt_files_exist_on_disk(self):
        for filename in ("system_prompts.txt", "summary_prompt.txt", "report_prompt.txt"):
            path = BASE_DIR / "prompts" / filename
            assert path.exists(), f"缺少提示词文件：{path}"
            assert path.read_text(encoding="utf-8").strip()


class TestModeSwitchMapping:
    def test_every_switch_target_has_a_prompt(self):
        """标记工具指向的模式必须真的存在，否则会静默回退到普通问答。"""
        unknown = set(MODE_SWITCH_TOOLS.values()) - set(MODE_PROMPTS)
        assert not unknown, f"这些模式没有对应的提示词：{unknown}"

    def test_mapping_is_not_empty(self):
        assert MODE_SWITCH_TOOLS
