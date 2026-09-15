"""
文件名：test_config_loader.py
介绍：配置加载测试。

覆盖两处「静默出错」的逻辑：
- resolve_data_dir：文档目录的选取。选错了不会报错，只会让应用读到空目录，
  表现成「上传了文件但问不出东西」。
- _bridge_streamlit_secrets：云端部署的密钥桥接。写错了本地一切正常，
  一部署到 Streamlit Cloud 就因为拿不到 Key 而崩。
"""
import os
import sys
from types import SimpleNamespace

import pytest

import tools.config_loader as cl


class TestResolveDataDir:
    def test_prefers_test_data_when_it_has_files(self, tmp_path, monkeypatch):
        (tmp_path / "test_data").mkdir()
        (tmp_path / "test_data" / "a.txt").write_text("x", encoding="utf-8")
        (tmp_path / "demo_data").mkdir()
        monkeypatch.setattr(cl, "BASE_DIR", tmp_path)

        assert cl.resolve_data_dir().name == "test_data"

    def test_falls_back_when_test_data_is_empty(self, tmp_path, monkeypatch):
        """新克隆的仓库就是这种情况：test_data 被 gitignore，目录存在但为空。"""
        (tmp_path / "test_data").mkdir()
        (tmp_path / "demo_data").mkdir()
        monkeypatch.setattr(cl, "BASE_DIR", tmp_path)

        assert cl.resolve_data_dir().name == "demo_data"

    def test_falls_back_when_test_data_is_missing(self, tmp_path, monkeypatch):
        """云端部署：test_data 根本不存在。"""
        (tmp_path / "demo_data").mkdir()
        monkeypatch.setattr(cl, "BASE_DIR", tmp_path)

        assert cl.resolve_data_dir().name == "demo_data"

    def test_ignores_subdirectories(self, tmp_path, monkeypatch):
        """test_data 里只有子目录、没有文件时仍应回退，不能把子目录当成语料。"""
        (tmp_path / "test_data" / "nested").mkdir(parents=True)
        (tmp_path / "demo_data").mkdir()
        monkeypatch.setattr(cl, "BASE_DIR", tmp_path)

        assert cl.resolve_data_dir().name == "demo_data"


class TestStreamlitSecretsBridge:
    def test_does_nothing_when_streamlit_not_loaded(self, monkeypatch):
        """脚本、单元测试等场景没有 streamlit，不应因此报错。"""
        monkeypatch.delitem(sys.modules, "streamlit", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

        cl._bridge_streamlit_secrets()

        assert "DEEPSEEK_API_KEY" not in os.environ

    def test_injects_keys_missing_from_environment(self, monkeypatch):
        fake_st = SimpleNamespace(secrets={"DEEPSEEK_API_KEY": "from-secrets"})
        monkeypatch.setitem(sys.modules, "streamlit", fake_st)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

        cl._bridge_streamlit_secrets()

        assert os.environ["DEEPSEEK_API_KEY"] == "from-secrets"

    def test_existing_environment_variable_wins(self, monkeypatch):
        """本地 .env 已配好的 Key 不应被云端 secrets 覆盖。"""
        fake_st = SimpleNamespace(secrets={"DEEPSEEK_API_KEY": "from-secrets"})
        monkeypatch.setitem(sys.modules, "streamlit", fake_st)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")

        cl._bridge_streamlit_secrets()

        assert os.environ["DEEPSEEK_API_KEY"] == "from-env"

    def test_secrets_access_failure_is_swallowed(self, monkeypatch):
        """没有 secrets.toml 时访问 st.secrets 会抛异常，不能让它冒出来打断启动。"""

        class Boom:
            @property
            def secrets(self):
                raise RuntimeError("No secrets found")

        monkeypatch.setitem(sys.modules, "streamlit", Boom())

        cl._bridge_streamlit_secrets()  # 不应抛异常

    def test_all_three_keys_are_bridged(self, monkeypatch):
        fake_st = SimpleNamespace(secrets={
            "DASHSCOPE_API_KEY": "d",
            "DEEPSEEK_API_KEY": "s",
            "TAVILY_API_KEY": "t",
        })
        monkeypatch.setitem(sys.modules, "streamlit", fake_st)
        for key in cl.API_KEYS:
            monkeypatch.delenv(key, raising=False)

        cl._bridge_streamlit_secrets()

        for key in cl.API_KEYS:
            assert os.environ[key] == fake_st.secrets[key]


class TestConfigFiles:
    def test_yaml_configs_are_loaded(self):
        assert cl.LLM_conf["chat_model_name"]
        assert cl.rag_conf["vector_database"]["collection_name"]
        assert cl.tavily_conf["max_result"] > 0

    def test_persist_directory_is_absolute(self):
        """相对路径在不同工作目录下会指向不同位置，必须是绝对路径。"""
        assert os.path.isabs(cl.persist_directory)

    def test_splitter_settings_are_sane(self):
        """chunk_overlap 必须小于 chunk_size，否则文本切分会陷入死循环或异常。"""
        s = cl.rag_conf["splitter"]
        assert 0 < s["chunk_overlap"] < s["chunk_size"]

    def test_fetch_k_is_not_smaller_than_k(self):
        """MMR 先从 fetch_k 个候选里挑 k 个，候选池小于结果数没有意义。"""
        assert cl.rag_conf["fetch_k"] >= cl.rag_conf["k"]
