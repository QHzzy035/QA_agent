"""
文件名：test_query_rules.py
介绍：检索前置规则测试。

is_no_retrieval 决定一个问题「跳过文档检索直接回答」还是「先去知识库查」。
规则写宽了会让正常提问检索不到资料，写窄了会让闲聊也去查库、白白烧一次检索；
而且它是个纯函数，正好适合用测试把行为钉死。
"""
import pytest

from rag.connected_prompts import is_no_retrieval


class TestNoRetrievalRules:
    @pytest.mark.parametrize("query", [
        "你好",
        "你是谁",
        "你能做什么",
        "介绍一下你自己",
        "现在几点了",
        "今天星期几",
        "今天天气怎么样",
        "谢谢",
        "再见",
    ])
    def test_skips_retrieval(self, query):
        assert is_no_retrieval(query) is True

    @pytest.mark.parametrize("query", [
        "量子计算是什么",
        "帮我总结一下区块链那篇文档",
        "CRISPR 基因编辑有什么伦理争议",
        "黑洞的事件视界是怎么定义的",
        "生成一份关于气候变化的报告",
        "咖啡豆的烘焙程度有哪些",
    ])
    def test_goes_through_retrieval(self, query):
        assert is_no_retrieval(query) is False

    def test_empty_query_is_not_skipped(self):
        assert is_no_retrieval("") is False

    def test_matches_as_substring_not_exact(self):
        """规则是包含匹配，带前后缀的问候语同样应被识别。"""
        assert is_no_retrieval("你好呀，今天过得怎么样") is True
