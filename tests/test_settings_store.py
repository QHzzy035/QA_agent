"""
文件名：test_settings_store.py
介绍：设置持久化测试。

设置文件放在 QA_AGENT_DATA_DIR 下；测试不触碰项目 data/。
"""
import json

import pytest

import tools.settings_store as store


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("QA_AGENT_DATA_DIR", str(tmp_path / "data"))


class TestDefaults:
    def test_missing_file_returns_defaults(self):
        assert store.load_settings() == store.DEFAULT_SETTINGS

    def test_reset_writes_defaults(self):
        store.save_settings({"k": 9})
        restored = store.reset_settings()

        assert restored == store.DEFAULT_SETTINGS
        assert store.load_settings() == store.DEFAULT_SETTINGS


class TestPersistence:
    def test_save_and_reload(self):
        saved = store.save_settings({
            "k": 5,
            "history_limit": 20,
            "theme": "dark",
            "max_prompt_chars": 60000,
            "max_message_chars": 6000,
            "max_history_chars": 30000,
            "max_retrieval_chars": 8000,
        })

        loaded = store.load_settings()

        assert loaded == saved
        assert loaded["k"] == 5
        assert loaded["theme"] == "dark"
        assert loaded["max_prompt_chars"] == 60000

    def test_corrupt_file_falls_back_to_defaults(self):
        path = store.settings_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")

        assert store.load_settings() == store.DEFAULT_SETTINGS

    def test_settings_file_contains_no_api_keys(self):
        store.save_settings({"k": 4})
        text = store.settings_file().read_text(encoding="utf-8")

        assert "API_KEY" not in text
        assert "DEEPSEEK" not in text


class TestNormalization:
    def test_out_of_range_values_are_clamped(self):
        normalized = store.normalize_settings({
            "k": 99,
            "history_limit": 1,
            "theme": "rainbow",
            "max_prompt_chars": 999999,
            "max_message_chars": 1,
            "max_history_chars": 1,
            "max_retrieval_chars": 1,
        })

        assert normalized["k"] == 10
        assert normalized["history_limit"] == 2
        assert normalized["theme"] == "system"
        assert normalized["max_prompt_chars"] <= 200000
        assert normalized["max_message_chars"] >= 500
        assert normalized["max_history_chars"] >= 1000
        assert normalized["max_retrieval_chars"] >= 500

    def test_invalid_types_fall_back(self):
        normalized = store.normalize_settings({
            "k": "abc",
            "history_limit": None,
            "theme": None,
        })

        assert normalized["k"] == store.DEFAULT_SETTINGS["k"]
        assert normalized["history_limit"] == store.DEFAULT_SETTINGS["history_limit"]
        assert normalized["theme"] == store.DEFAULT_SETTINGS["theme"]

    def test_non_dict_input(self):
        assert store.normalize_settings("bad") == store.DEFAULT_SETTINGS
        assert store.normalize_settings(None) == store.DEFAULT_SETTINGS
