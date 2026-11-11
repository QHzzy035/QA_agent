"""
文件名：context_budget.py
介绍：上下文保护 v0。

按字符数做保守预算，顺序是：
1. 截断单条过长的消息；
2. 丢弃最旧的历史消息；
3. 减少检索资料（在 new_prompt 中执行）；
4. 最后才截断当前提示词；
5. 系统提示词永远不裁。

这是防止上下文溢出的兜底，不是精确 token 计费。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

ELLIPSIS = "…"


@dataclass(frozen=True)
class BudgetConfig:
    max_prompt_chars: int = 48000
    max_message_chars: int = 4000
    max_history_chars: int = 24000
    max_retrieval_chars: int = 6000

    @classmethod
    def from_settings(cls, settings: dict) -> "BudgetConfig":
        return cls(
            max_prompt_chars=int(settings.get("max_prompt_chars", 48000)),
            max_message_chars=int(settings.get("max_message_chars", 4000)),
            max_history_chars=int(settings.get("max_history_chars", 24000)),
            max_retrieval_chars=int(settings.get("max_retrieval_chars", 6000)),
        )

    def halved(self) -> "BudgetConfig":
        """上下文仍然超限时，用更保守的预算重试一次。"""
        return replace(
            self,
            max_prompt_chars=max(2000, self.max_prompt_chars // 2),
            max_history_chars=max(500, self.max_history_chars // 2),
            max_retrieval_chars=max(200, self.max_retrieval_chars // 2),
        )


@dataclass
class TrimReport:
    trimmed_messages: int = 0
    truncated_messages: int = 0
    removed_chars: int = 0
    history_chars: int = 0
    prompt_chars: int = 0
    system_chars: int = 0
    truncated_prompt: bool = False

    def changed(self) -> bool:
        return (
            self.trimmed_messages > 0
            or self.truncated_messages > 0
            or self.truncated_prompt
        )


def message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return str(content)


def truncate_text(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit == 1:
        return ELLIPSIS
    return text[:limit - 1] + ELLIPSIS


def _truncate_message(message, limit: int):
    text = message_text(message)
    shortened = truncate_text(text, limit)
    if shortened == text:
        return message

    try:
        return message.model_copy(update={"content": shortened})
    except Exception:
        try:
            return type(message)(content=shortened)
        except Exception:
            return message


def cap_history(history: list, config: BudgetConfig) -> tuple[list, TrimReport]:
    """截断单条消息并丢弃最旧历史，使历史总量不超过预算。"""
    report = TrimReport()
    limit = min(config.max_message_chars, config.max_history_chars)
    capped = []

    for message in history:
        text = message_text(message)
        if len(text) > limit:
            message = _truncate_message(message, limit)
            report.truncated_messages += 1
        capped.append(message)

    total = sum(len(message_text(message)) for message in capped)
    while total > config.max_history_chars and capped:
        removed = capped.pop(0)
        removed_len = len(message_text(removed))
        total -= removed_len
        report.removed_chars += removed_len
        report.trimmed_messages += 1

    report.history_chars = total
    return capped, report


def finalize_prompt(
    system_prompt: str,
    history: list,
    prompt: str,
    config: BudgetConfig,
) -> tuple[list, str, TrimReport]:
    """总预算兜底：先丢最旧历史，最后才截断当前提示词。"""
    report = TrimReport(
        system_chars=len(system_prompt),
        history_chars=sum(len(message_text(message)) for message in history),
        prompt_chars=len(prompt),
    )
    history = list(history)

    def total_chars() -> int:
        return (
            len(system_prompt)
            + sum(len(message_text(message)) for message in history)
            + len(prompt)
        )

    while total_chars() > config.max_prompt_chars and history:
        removed = history.pop(0)
        removed_len = len(message_text(removed))
        report.trimmed_messages += 1
        report.removed_chars += removed_len

    if total_chars() > config.max_prompt_chars:
        available = max(100, config.max_prompt_chars - len(system_prompt))
        if len(prompt) > available:
            prompt = truncate_text(prompt, available)
            report.truncated_prompt = True

    report.history_chars = sum(len(message_text(message)) for message in history)
    report.prompt_chars = len(prompt)
    return history, prompt, report


def is_context_error(error: Exception) -> bool:
    text = str(error).lower()
    needles = (
        "context length",
        "maximum context",
        "context_length_exceeded",
        "too many tokens",
        "reduce the length",
        "prompt is too long",
        "token limit",
        "context window",
    )
    if any(needle in text for needle in needles):
        return True
    return "token" in text and ("exceed" in text or "limit" in text)
