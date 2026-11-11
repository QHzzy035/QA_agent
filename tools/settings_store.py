"""
文件名：settings_store.py
介绍：用户设置持久化。

只保存非敏感设置（检索条数、历史阈值、主题、上下文预算）；
API Key 仍然来自 .env / 环境变量，不写入 settings.json。

设置文件位于 app_paths.settings_file()，通常是 data/settings.json。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from tools.app_paths import settings_file

DEFAULT_SETTINGS = {
    "k": 3,
    "history_limit": 10,
    "theme": "system",
    "max_prompt_chars": 48000,
    "max_message_chars": 4000,
    "max_history_chars": 24000,
    "max_retrieval_chars": 6000,
}

VALID_THEMES = {"system", "light", "dark"}


def _clamp_int(value, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def normalize_settings(raw) -> dict:
    """把任意输入规整成合法设置，非法字段回退默认值。"""
    if not isinstance(raw, dict):
        raw = {}

    return {
        "k": _clamp_int(raw.get("k"), DEFAULT_SETTINGS["k"], 1, 10),
        "history_limit": _clamp_int(
            raw.get("history_limit"),
            DEFAULT_SETTINGS["history_limit"],
            2,
            50,
        ),
        "theme": (
            str(raw.get("theme"))
            if str(raw.get("theme")) in VALID_THEMES
            else DEFAULT_SETTINGS["theme"]
        ),
        "max_prompt_chars": _clamp_int(
            raw.get("max_prompt_chars"),
            DEFAULT_SETTINGS["max_prompt_chars"],
            4000,
            200000,
        ),
        "max_message_chars": _clamp_int(
            raw.get("max_message_chars"),
            DEFAULT_SETTINGS["max_message_chars"],
            500,
            20000,
        ),
        "max_history_chars": _clamp_int(
            raw.get("max_history_chars"),
            DEFAULT_SETTINGS["max_history_chars"],
            1000,
            100000,
        ),
        "max_retrieval_chars": _clamp_int(
            raw.get("max_retrieval_chars"),
            DEFAULT_SETTINGS["max_retrieval_chars"],
            500,
            20000,
        ),
    }


def load_settings() -> dict:
    path = settings_file()
    if not path.exists():
        return dict(DEFAULT_SETTINGS)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_SETTINGS)

    return normalize_settings(data)


def save_settings(settings: dict) -> dict:
    normalized = normalize_settings(settings)
    path = settings_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return normalized


def reset_settings() -> dict:
    return save_settings(DEFAULT_SETTINGS)
