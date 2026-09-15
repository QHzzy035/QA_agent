"""
文件名：test_eval_metrics.py
介绍：检索指标测试。

指标算错的话，评测报告会「看起来有数字」但完全不可信——而且这种错误
不会报错、不会崩，只会给出一个漂亮的假数字。所以边界情况要钉死。
"""
import pytest

from eval.metrics import (
    average_precision, dedupe, evaluate, hit_at_k,
    precision_at_k, recall_at_k, reciprocal_rank, to_topics,
)


class TestToTopics:
    def test_strips_directory_and_extension(self):
        assert to_topics([r"D:\data\a\blockchain_basics.md"]) == ["blockchain_basics"]

    def test_handles_forward_slashes(self):
        assert to_topics(["/home/u/test_data/black_hole.txt"]) == ["black_hole"]

    def test_same_topic_in_different_formats_collapses_to_one(self):
        """语料里同一主题有 md/docx/pdf 三份，命中任意一种都算同一个主题。"""
        assert to_topics([
            "blockchain_basics.pdf", "blockchain_basics.docx", "blockchain_basics.md",
        ]) == ["blockchain_basics"]

    def test_preserves_first_occurrence_order(self):
        assert to_topics(["b.txt", "a.txt", "b.txt", "c.txt"]) == ["b", "a", "c"]

    def test_empty_input(self):
        assert to_topics([]) == []


class TestDedupe:
    def test_preserves_order(self):
        assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]

    def test_empty(self):
        assert dedupe([]) == []


class TestHitAtK:
    def test_hit_within_k(self):
        assert hit_at_k(["a"], ["x", "a", "b"], k=2) == 1.0

    def test_hit_outside_k(self):
        assert hit_at_k(["a"], ["x", "y", "a"], k=2) == 0.0

    def test_no_hit(self):
        assert hit_at_k(["a"], ["x", "y"], k=5) == 0.0

    def test_k_larger_than_results(self):
        """召回结果少于 K 时不应越界，也不应误判为命中。"""
        assert hit_at_k(["a"], ["x"], k=10) == 0.0
        assert hit_at_k(["x"], ["x"], k=10) == 1.0


class TestRecallAtK:
    def test_partial_recall_with_multiple_expected(self):
        assert recall_at_k(["a", "b"], ["a", "x"], k=2) == pytest.approx(0.5)

    def test_full_recall(self):
        assert recall_at_k(["a", "b"], ["a", "b"], k=2) == pytest.approx(1.0)

    def test_empty_expected_is_zero_not_error(self):
        """没有期望主题时返回 0，而不是 ZeroDivisionError。"""
        assert recall_at_k([], ["a"], k=1) == 0.0


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b"], ["a", "b"], k=2) == pytest.approx(1.0)

    def test_half_relevant(self):
        assert precision_at_k(["a"], ["a", "x"], k=2) == pytest.approx(0.5)

    def test_k_zero_does_not_divide_by_zero(self):
        assert precision_at_k(["a"], ["a"], k=0) == 0.0

    def test_duplicates_do_not_inflate(self):
        """同一主题重复出现只算一次，否则精密度会被虚高。"""
        assert precision_at_k(["a"], ["a", "a", "a"], k=3) == pytest.approx(1 / 3)


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a"], ["a", "b"]) == pytest.approx(1.0)

    def test_third_position(self):
        assert reciprocal_rank(["a"], ["x", "y", "a"]) == pytest.approx(1 / 3)

    def test_not_found(self):
        assert reciprocal_rank(["a"], ["x", "y"]) == 0.0

    def test_takes_best_position_not_average(self):
        assert reciprocal_rank(["a"], ["x", "a", "a"]) == pytest.approx(0.5)


class TestAveragePrecision:
    def test_relevant_first_and_second(self):
        # 命中位置 1 和 2：(1/1 + 2/2) / 2 = 1.0
        assert average_precision(["a", "b"], ["a", "b"]) == pytest.approx(1.0)

    def test_relevant_second_and_fourth(self):
        # (1/2 + 2/4) / 2 = 0.5
        assert average_precision(["a", "b"], ["x", "a", "y", "b"]) == pytest.approx(0.5)

    def test_penalizes_missing_expected(self):
        """漏掉一个期望主题时，AP 应低于全部命中的情况。"""
        both = average_precision(["a", "b"], ["a", "b"])
        only_one = average_precision(["a", "b"], ["a", "x"])
        assert only_one < both

    def test_empty_expected(self):
        assert average_precision([], ["a"]) == 0.0


class TestEvaluate:
    def _records(self):
        return [
            {"query": "q1", "expected": ["a"], "retrieved": ["a", "b"]},   # 命中第 1 位
            {"query": "q2", "expected": ["b"], "retrieved": ["x", "b"]},   # 命中第 2 位
            {"query": "q3", "expected": ["c"], "retrieved": ["x", "y"]},   # 未命中
        ]

    def test_aggregates_means(self):
        s = evaluate(self._records(), k=2)
        assert s["count"] == 3
        assert s["hit_at_k"] == pytest.approx(2 / 3)
        assert s["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)

    def test_lists_misses_for_diagnosis(self):
        s = evaluate(self._records(), k=2)
        assert [m["query"] for m in s["misses"]] == ["q3"]
        assert s["misses"][0]["expected"] == ["c"]
        assert s["misses"][0]["retrieved"] == ["x", "y"]

    def test_same_records_give_different_scores_at_different_k(self):
        """排在 K 之外的命中，在大 K 下才算数。"""
        records = [{"query": "q", "expected": ["a"], "retrieved": ["x", "y", "a"]}]
        assert evaluate(records, k=2)["hit_at_k"] == 0.0
        assert evaluate(records, k=3)["hit_at_k"] == 1.0

    def test_empty_records(self):
        assert evaluate([], k=3) == {"count": 0, "k": 3}


class TestRenderMarkdown:
    """报告是给人看的，格式错了不会报错，只会让人读出一个错误的结论。"""

    def _records(self):
        return [
            {"query": "q1", "expected": ["a"], "retrieved": ["a", "b"]},
            {"query": "q2", "expected": ["c"], "retrieved": ["x", "y"]},
        ]

    def test_contains_one_row_per_k(self):
        from eval.run_eval import render_markdown

        text = render_markdown([evaluate(self._records(), k) for k in (1, 3, 5)])

        for k in (1, 3, 5):
            assert f"| {k} |" in text

    def test_mrr_and_map_are_outside_the_table(self):
        """MRR/MAP 不随 K 截断，放进 @K 的表里每行都一样，会被误读成 bug。"""
        from eval.run_eval import render_markdown

        text = render_markdown([evaluate(self._records(), k) for k in (1, 3)])
        lines = text.splitlines()

        header = next(l for l in lines if l.startswith("| K |"))
        # 注意不能只判断 set(l) <= set("|-")：空行的字符集是空集，任何集合都包含它
        sep = next(i for i, l in enumerate(lines)
                   if l.strip() and set(l) <= set("|-"))
        rows = []
        for line in lines[sep + 1:]:
            if not line.startswith("|"):
                break
            rows.append(line)

        assert rows, "表格应当有数据行"
        assert "MRR" not in header, "MRR 不应出现在表头"
        assert all("MRR" not in r for r in rows), "MRR 不应出现在表格行里"
        assert "MRR" in text, "但 MRR 必须仍然被报告出来"

    def test_reports_miss_count(self):
        from eval.run_eval import render_markdown

        text = render_markdown([evaluate(self._records(), k=2)])

        assert "| 2 | 0.500 | 0.500 | 0.250 | 1 |" in text
