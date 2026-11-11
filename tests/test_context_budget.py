"""
文件名：test_context_budget.py
介绍：上下文保护 v0 测试。

验证裁剪顺序：单条截断 → 丢最旧历史 → 最后截断提示词，
系统提示词不参与裁剪。
"""
from langchain_core.messages import AIMessage, HumanMessage

from tools.context_budget import (
    BudgetConfig,
    cap_history,
    finalize_prompt,
    is_context_error,
    message_text,
    truncate_text,
)


class TestCapHistory:
    def test_truncates_long_message(self):
        config = BudgetConfig(max_message_chars=5, max_history_chars=100)
        history = [HumanMessage(content="abcdefgh")]

        capped, report = cap_history(history, config)

        assert len(message_text(capped[0])) <= 5
        assert report.truncated_messages == 1
        assert report.trimmed_messages == 0

    def test_drops_oldest_messages(self):
        config = BudgetConfig(max_message_chars=100, max_history_chars=20)
        history = [
            HumanMessage(content="a" * 10),
            AIMessage(content="b" * 10),
            HumanMessage(content="c" * 10),
        ]

        capped, report = cap_history(history, config)

        assert len(capped) == 2
        assert message_text(capped[0]) == "b" * 10
        assert report.trimmed_messages == 1
        assert report.removed_chars == 10

    def test_keeps_recent_message_types(self):
        config = BudgetConfig(max_message_chars=100, max_history_chars=20)
        history = [
            HumanMessage(content="a" * 10),
            AIMessage(content="b" * 10),
        ]

        capped, _ = cap_history(history, config)

        assert isinstance(capped[-1], AIMessage)


class TestFinalizePrompt:
    def test_drops_history_before_truncating_prompt(self):
        config = BudgetConfig(max_prompt_chars=35)
        history = [
            HumanMessage(content="a" * 10),
            HumanMessage(content="b" * 10),
        ]

        fitted, prompt, report = finalize_prompt(
            system_prompt="s" * 10,
            history=history,
            prompt="p" * 10,
            config=config,
        )

        assert len(fitted) == 1
        assert prompt == "p" * 10
        assert report.trimmed_messages == 1
        assert report.truncated_prompt is False

    def test_truncates_prompt_as_last_resort(self):
        config = BudgetConfig(max_prompt_chars=1000)
        system = "s" * 100
        prompt = "x" * 2000

        fitted, result, report = finalize_prompt(
            system_prompt=system,
            history=[],
            prompt=prompt,
            config=config,
        )

        assert fitted == []
        assert len(result) <= 1000 - len(system)
        assert report.truncated_prompt is True

    def test_system_prompt_is_never_trimmed(self):
        config = BudgetConfig(max_prompt_chars=100)
        system = "s" * 500

        _, prompt, report = finalize_prompt(
            system_prompt=system,
            history=[],
            prompt="short",
            config=config,
        )

        assert report.system_chars == 500
        assert prompt == "short"


class TestHelpers:
    def test_truncate_text(self):
        assert truncate_text("abcdef", 10) == "abcdef"
        assert truncate_text("abcdef", 4) == "abc…"
        assert truncate_text("abcdef", 0) == ""

    def test_from_settings(self):
        config = BudgetConfig.from_settings({
            "max_prompt_chars": 1234,
            "max_message_chars": 567,
            "max_history_chars": 890,
            "max_retrieval_chars": 12,
        })

        assert config.max_prompt_chars == 1234
        assert config.max_message_chars == 567
        assert config.max_history_chars == 890
        assert config.max_retrieval_chars == 12

    def test_halved(self):
        config = BudgetConfig(
            max_prompt_chars=48000,
            max_history_chars=24000,
            max_retrieval_chars=6000,
        )

        halved = config.halved()

        assert halved.max_prompt_chars == 24000
        assert halved.max_history_chars == 12000
        assert halved.max_retrieval_chars == 3000

    def test_context_error_detection(self):
        assert is_context_error(RuntimeError("maximum context length exceeded"))
        assert is_context_error(RuntimeError("context_length_exceeded"))
        assert is_context_error(RuntimeError("too many tokens"))
        assert not is_context_error(RuntimeError("connection timeout"))
