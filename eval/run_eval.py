"""
文件名：run_eval.py
介绍：RAG 检索质量评测。按 golden set 逐条检索并计算 Recall@K / MRR / MAP。

用法：
    # 只校验 golden set（不调用任何 API，零成本）
    python -m eval.run_eval --validate-only

    # 正式评测（每条查询会调用一次嵌入 API）
    python -m eval.run_eval --k 1 3 5

    # 结果写成 markdown 表格
    python -m eval.run_eval --k 3 --out eval/report.md

关于成本：
    - 每条查询调用一次嵌入 API（查询很短，费用极低）
    - 无论评估几个 K，检索只跑一次（MMR 是贪心选择，前 k 个结果具有前缀稳定性），
      所以 --k 1 3 5 和 --k 5 的调用次数相同

关于度量口径：
    K 指的是「召回的 chunk 条数」，与线上行为一致（应用就是取 top-k 个 chunk）。
    计算文档级指标前会按主题去重，避免同一篇文档的多个 chunk 挤占多个排名位。
    评测走的是纯检索路径（rag.retriever.retriever），不经过查询改写——
    golden set 里的问题都是自包含的，不依赖多轮上下文。
"""
import argparse
import json
import sys
from pathlib import Path

# 允许以 python eval/run_eval.py 或 python -m eval.run_eval 两种方式运行
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from eval.metrics import evaluate, to_topics  # noqa: E402
from tools.document_service import SUPPORTED_EXTENSIONS  # noqa: E402

DEFAULT_GOLDEN = BASE_DIR / "eval" / "golden_set.json"
DEFAULT_DATA_DIR = BASE_DIR / "test_data"


def load_golden(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["queries"]


def corpus_topics(data_dir: Path = DEFAULT_DATA_DIR) -> set[str]:
    return {
        p.stem for p in data_dir.glob("*.*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    }


def validate(golden: list[dict], corpus: set[str]) -> list[str]:
    """校验 golden set 自身的问题：期望主题不存在 / 重复查询 / 字段缺失。"""
    problems = []
    seen_queries = {}

    for i, item in enumerate(golden, start=1):
        query = item.get("query")
        expected = item.get("expected")

        if not query:
            problems.append(f"第 {i} 条缺少 query")
            continue
        if not expected:
            problems.append(f"第 {i} 条 ({query}) 缺少 expected")
            continue

        if query in seen_queries:
            problems.append(f"第 {i} 条与第 {seen_queries[query]} 条查询重复：{query}")
        seen_queries[query] = i

        for topic in expected:
            if topic not in corpus:
                problems.append(f"第 {i} 条 ({query}) 的期望主题不存在于语料中：{topic}")

    return problems


def retrieve_topics(queries: list[dict], k_max: int) -> list[dict]:
    """逐条检索，返回 [{query, expected, retrieved}]。

    retrieved 是「按 chunk 顺序、按主题去重后」的完整列表；
    评估不同 K 时直接切片即可，无需重复检索。
    """
    from rag.retriever import retriever

    records = []
    total = len(queries)
    for i, item in enumerate(queries, start=1):
        results = retriever(item["query"], k=k_max)
        sources = [meta.get("source", "") for _, meta in results]
        records.append({
            "query": item["query"],
            "expected": item["expected"],
            "retrieved": to_topics(sources),
        })
        print(f"\r  检索中 {i}/{total} ...", end="", flush=True)
    print("\r" + " " * 30 + "\r", end="")
    return records


def render_markdown(summaries: list[dict]) -> str:
    lines = [
        "## 检索质量评测结果",
        "",
        f"评测集：{summaries[0]['count']} 条查询　|　度量口径：召回 top-k 个 chunk，按文档去重后计算",
        "",
        "| K | Recall@K | Hit@K | Precision@K | MRR | MAP | 未命中数 |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['k']} | {s['recall_at_k']:.3f} | {s['hit_at_k']:.3f} "
            f"| {s['precision_at_k']:.3f} | {s['mrr']:.3f} | {s['map']:.3f} "
            f"| {len(s['misses'])} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="RAG 检索质量评测")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN, help="golden set 路径")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="语料目录")
    parser.add_argument("--k", type=int, nargs="+", default=[3], help="要评估的 K（可多个）")
    parser.add_argument("--validate-only", action="store_true",
                        help="只校验 golden set，不调用任何 API")
    parser.add_argument("--out", type=Path, help="把结果写成 markdown 文件")
    parser.add_argument("--json-out", type=Path, help="把原始结果写成 JSON 文件")
    args = parser.parse_args()

    golden = load_golden(args.golden)
    corpus = corpus_topics(args.data_dir)

    problems = validate(golden, corpus)
    if problems:
        print(f"golden set 校验失败（{len(problems)} 个问题）：")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    print(f"golden set 校验通过：{len(golden)} 条查询，语料 {len(corpus)} 个主题")

    if args.validate_only:
        return

    ks = sorted(set(args.k))
    k_max = max(ks)

    print(f"即将调用嵌入 API：{len(golden)} 次（每条查询一次，与评估几个 K 无关）")
    records = retrieve_topics(golden, k_max)

    summaries = [evaluate(records, k) for k in ks]
    report = render_markdown(summaries)
    print()
    print(report)

    for s in summaries:
        if s["misses"]:
            print(f"\n未命中的查询（K={s['k']}）：")
            for m in s["misses"]:
                print(f"  - {m['query']}")
                print(f"      期望 {m['expected']}　实际召回 {m['retrieved']}")

    if args.out:
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"\n已写入 {args.out}")

    if args.json_out:
        args.json_out.write_text(
            json.dumps({"records": records, "summaries": summaries},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"已写入 {args.json_out}")


if __name__ == "__main__":
    main()
