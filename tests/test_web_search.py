"""
文件名：test_web_search.py
介绍：联网搜索结果格式化测试。

重点不是 Tavily 本身，而是「工具输出有没有把 URL 带回来」以及
「空结果 / 异常 / 超长内容会不会把整轮对话弄崩」。
"""
from tools.web_search import MAX_CONTENT_CHARS, format_search_results, web_search


class TestFormatSearchResults:
    def test_includes_title_url_and_content(self):
        result = {
            "results": [{
                "title": "RAG 入门",
                "url": "https://example.com/rag",
                "content": "检索增强生成的基本流程。",
            }]
        }

        text = format_search_results(result)

        assert "RAG 入门" in text
        assert "https://example.com/rag" in text
        assert "检索增强生成的基本流程" in text

    def test_multiple_results_are_numbered(self):
        result = {"results": [
            {"title": "A", "url": "https://a.example", "content": "a"},
            {"title": "B", "url": "https://b.example", "content": "b"},
        ]}

        text = format_search_results(result)

        assert "[1]" in text and "[2]" in text
        assert "https://a.example" in text and "https://b.example" in text

    def test_empty_results_returns_readable_message(self):
        assert "没有返回" in format_search_results({"results": []})

    def test_missing_fields_do_not_crash(self):
        text = format_search_results({"results": [{"content": "只有正文"}]})

        assert "只有正文" in text

    def test_long_content_is_truncated(self):
        result = {"results": [{
            "url": "https://example.com/long",
            "content": "x" * (MAX_CONTENT_CHARS + 200),
        }]}

        text = format_search_results(result)

        assert "…" in text
        assert "x" * MAX_CONTENT_CHARS in text
        assert "x" * (MAX_CONTENT_CHARS + 1) not in text

    def test_unexpected_payload_returns_message(self):
        assert "无法解析" in format_search_results(None)
        assert "无法解析" in format_search_results("not a dict")

    def test_items_without_url_or_content_are_skipped(self):
        result = {"results": [{}, {"title": "空"}]}

        assert "没有返回可用内容" in format_search_results(result)


class TestWebSearchErrorHandling:
    def test_web_search_output_contains_url(self, monkeypatch):
        import tools.web_search as mod

        class FakeTavily:
            def invoke(self, _):
                return {"results": [{
                    "title": "示例",
                    "url": "https://example.com/doc",
                    "content": "正文",
                }]}

        monkeypatch.setattr(mod, "_tavily", lambda: FakeTavily())

        text = mod.web_search.func("查询")

        assert "https://example.com/doc" in text
        assert "正文" in text

    def test_search_failure_returns_message_instead_of_raising(self, monkeypatch):
        import tools.web_search as mod

        def boom():
            raise RuntimeError("TAVILY_API_KEY 未配置")

        monkeypatch.setattr(mod, "_tavily", boom)

        text = mod.web_search.func("测试查询")

        assert "联网搜索失败" in text
        assert "TAVILY_API_KEY" in text

    def test_module_import_does_not_instantiate_client(self):
        """惰性创建：重新导入模块后，客户端应仍未创建。"""
        import importlib

        import tools.web_search as mod

        importlib.reload(mod)

        assert mod._client is None
