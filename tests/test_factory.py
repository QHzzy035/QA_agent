"""
文件名：test_factory.py
介绍：模型工厂测试。

工厂同时承担两件事：按角色选模型名（配置驱动），以及缓存实例（避免每次调用都重建客户端）。
另外覆盖「惰性创建」这个改造点：模块导入本身不应该触发任何模型实例化。
"""
from types import SimpleNamespace

import pytest

from model.factory import ChatModelFactory, EmbeddingsFactory


@pytest.fixture(autouse=True)
def reset_factories():
    """工厂是类级缓存，测试之间必须清空，否则会互相污染。"""
    import model.factory as factory_module

    def _reset():
        factory_module.ChatModelFactory._instances.clear()
        factory_module.EmbeddingsFactory._instance = None

    _reset()
    yield
    _reset()


@pytest.fixture
def fake_chat_model(monkeypatch):
    """替换底层模型构造，记录被创建过哪些模型名。"""
    created: list[str] = []

    def _init_chat_model(model=None, **kwargs):
        created.append(model)
        return SimpleNamespace(model_name=model)

    monkeypatch.setattr("model.factory.init_chat_model", _init_chat_model)
    return created


class TestChatModelFactory:
    def test_uses_role_specific_model_name(self, monkeypatch, fake_chat_model):
        monkeypatch.setattr("model.factory.LLM_conf", {
            "chat_model_name": "chat-model",
            "summarize_model_name": "summarize-model",
            "rewrite_model_name": "rewrite-model",
        })

        assert ChatModelFactory.get("summarize").model_name == "summarize-model"
        assert ChatModelFactory.get("rewrite").model_name == "rewrite-model"

    def test_falls_back_to_chat_model_for_unknown_role(self, monkeypatch, fake_chat_model):
        """配置里没写的角色要回退到主模型，而不是 KeyError 崩掉。"""
        monkeypatch.setattr("model.factory.LLM_conf", {"chat_model_name": "chat-model"})

        assert ChatModelFactory.get("某个不存在的角色").model_name == "chat-model"

    def test_passes_configured_temperature(self, monkeypatch):
        """温度必须显式传给模型。

        不传的话走 provider 默认值（偏高）—— 实测同一问题每次答案都不同、
        长度能差 3 倍，既让用户觉得不可靠，也让生成质量评测无法复现。
        """
        seen = {}

        def _init_chat_model(model=None, **kwargs):
            seen["model"] = model
            seen.update(kwargs)
            return SimpleNamespace(model_name=model)

        monkeypatch.setattr("model.factory.init_chat_model", _init_chat_model)
        monkeypatch.setattr("model.factory.LLM_conf",
                            {"chat_model_name": "chat-model", "temperature": 0.2})

        ChatModelFactory.get("chat")

        assert seen["temperature"] == 0.2

    def test_falls_back_to_default_temperature(self, monkeypatch):
        """配置里没写 temperature 时用默认值，而不是 KeyError 崩掉。"""
        seen = {}

        def _init_chat_model(model=None, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(model_name=model)

        monkeypatch.setattr("model.factory.init_chat_model", _init_chat_model)
        monkeypatch.setattr("model.factory.LLM_conf", {"chat_model_name": "chat-model"})

        ChatModelFactory.get("chat")

        assert seen["temperature"] == ChatModelFactory.DEFAULT_TEMPERATURE

    def test_caches_instance_per_role(self, monkeypatch, fake_chat_model):
        monkeypatch.setattr("model.factory.LLM_conf", {"chat_model_name": "chat-model"})

        first = ChatModelFactory.get("chat")
        second = ChatModelFactory.get("chat")

        assert first is second
        assert fake_chat_model == ["chat-model"], "同一个角色只应创建一次模型"

    def test_different_roles_are_cached_separately(self, monkeypatch, fake_chat_model):
        monkeypatch.setattr("model.factory.LLM_conf", {
            "chat_model_name": "chat-model",
            "summarize_model_name": "summarize-model",
        })

        assert ChatModelFactory.get("chat") is not ChatModelFactory.get("summarize")
        assert len(fake_chat_model) == 2


class TestEmbeddingsFactory:
    def test_is_singleton(self, monkeypatch):
        created = []

        def _fake_embeddings(model=None, **kwargs):
            created.append(model)
            return SimpleNamespace(model_name=model)

        monkeypatch.setattr("model.factory.DashScopeEmbeddings", _fake_embeddings)

        assert EmbeddingsFactory.get() is EmbeddingsFactory.get()
        assert len(created) == 1

    def test_uses_configured_embedding_model(self, monkeypatch):
        """断言「工厂用的就是配置里的模型」，而不是硬编码某个模型名。

        硬编码的话，每次换嵌入模型这个测试都要跟着改 —— 那它测的就不是
        配置是否生效，而是「配置恰好没变过」。
        """
        seen = {}

        def _fake_embeddings(model=None, **kwargs):
            seen["model"] = model
            return SimpleNamespace(model_name=model)

        monkeypatch.setattr("model.factory.DashScopeEmbeddings", _fake_embeddings)

        from tools.config_loader import rag_conf
        EmbeddingsFactory.get()

        assert seen["model"] == rag_conf["vector_database"]["embedding_model_name"]


class TestLazyCreation:
    """惰性创建是这次重构的核心：没有 Key 也要能 import，用到才要 Key。"""

    def test_instances_are_not_created_at_import(self):
        import model.factory as factory_module

        for name in ("chat_model", "summarize_model", "rewrite_model", "embed_model"):
            assert name not in vars(factory_module), (
                f"{name} 不应在模块导入时创建，否则任何 import 都会要求 API Key"
            )

    def test_attribute_access_creates_model_on_demand(self, monkeypatch, fake_chat_model):
        monkeypatch.setattr("model.factory.LLM_conf", {"chat_model_name": "chat-model"})

        import model.factory as factory_module

        assert fake_chat_model == [], "访问之前不应创建"
        assert factory_module.chat_model.model_name == "chat-model"
        assert fake_chat_model == ["chat-model"]

    def test_unknown_attribute_raises_attribute_error(self):
        import model.factory as factory_module

        with pytest.raises(AttributeError):
            factory_module.根本没有这个属性
