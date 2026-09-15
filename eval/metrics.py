"""
文件名：metrics.py
介绍：检索质量指标（Recall@K / MRR / MAP 等）。

      全部是纯函数，不依赖向量库、模型或网络，可以直接单测。
      输入统一为「主题名」而非文件名：语料里同一主题存在多种格式
      （blockchain_basics 同时有 .md / .docx / .pdf），命中任意一种都算对。
"""
from collections.abc import Iterable, Sequence


def dedupe(seq: Iterable) -> list:
    """按首次出现顺序去重，保留顺序（用于文档级指标）。

    同一篇文档会有多个 chunk 被召回，计算文档级指标时必须去重，
    否则一篇文档会挤占多个排名位。
    """
    seen = set()
    out = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def to_topics(sources: Iterable[str]) -> list[str]:
    """把文件名列表转成去重后的主题名列表（去扩展名，保留顺序）。

    这里刻意用主题名而非文件名做匹配单位：语料中同一主题有多种格式，
    要求命中特定格式没有意义。
    """
    topics = []
    for src in sources:
        name = str(src).replace("\\", "/").rsplit("/", 1)[-1]
        topics.append(name.rsplit(".", 1)[0] if "." in name else name)
    return dedupe(topics)


def hit_at_k(expected: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    """Top-K 中是否至少命中一个期望主题。"""
    return 1.0 if set(expected) & set(retrieved[:k]) else 0.0


def recall_at_k(expected: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    """Top-K 覆盖了期望主题的比例。单期望主题时等价于 hit_at_k。"""
    expected = set(expected)
    if not expected:
        return 0.0
    return len(expected & set(retrieved[:k])) / len(expected)


def precision_at_k(expected: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    """Top-K 里有多少比例是期望主题。

    注意：K 越大这个值天然越低，跨 K 比较没有意义，只用于同一 K 下的横向对比。
    """
    if k <= 0:
        return 0.0
    return len(set(expected) & set(retrieved[:k])) / k


def reciprocal_rank(expected: Sequence[str], retrieved: Sequence[str]) -> float:
    """第一个命中主题的排名倒数；未命中记 0。

    对「把正确文档排到第几位」敏感，是 MRR 的单个查询分量。
    """
    expected = set(expected)
    for rank, topic in enumerate(retrieved, start=1):
        if topic in expected:
            return 1.0 / rank
    return 0.0


def average_precision(expected: Sequence[str], retrieved: Sequence[str]) -> float:
    """平均精度：同时惩罚「排得靠后」和「漏掉部分期望主题」。"""
    expected = set(expected)
    if not expected:
        return 0.0
    hits = 0
    total = 0.0
    for rank, topic in enumerate(retrieved, start=1):
        if topic in expected:
            hits += 1
            total += hits / rank
    return total / len(expected)


def evaluate(records: Sequence[dict], k: int) -> dict:
    """汇总一批检索结果。

    records: [{"query": str, "expected": [主题], "retrieved": [主题]}, ...]
    返回各指标的均值，以及未命中（hit@k == 0）的查询明细，便于定位问题。
    """
    if not records:
        return {"count": 0, "k": k}

    n = len(records)
    summary = {
        "count": n,
        "k": k,
        "recall_at_k": sum(recall_at_k(r["expected"], r["retrieved"], k) for r in records) / n,
        "hit_at_k": sum(hit_at_k(r["expected"], r["retrieved"], k) for r in records) / n,
        "precision_at_k": sum(precision_at_k(r["expected"], r["retrieved"], k) for r in records) / n,
        "mrr": sum(reciprocal_rank(r["expected"], r["retrieved"]) for r in records) / n,
        "map": sum(average_precision(r["expected"], r["retrieved"]) for r in records) / n,
        "misses": [
            {"query": r["query"], "expected": list(r["expected"]),
             "retrieved": list(r["retrieved"][:k])}
            for r in records
            if hit_at_k(r["expected"], r["retrieved"], k) == 0
        ],
    }
    return summary
