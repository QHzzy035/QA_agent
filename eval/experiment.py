"""
文件名：experiment.py
用途：检索质量对照实验 —— 用评测集回答「这个参数/模型该怎么选」。

支持变换两个变量（也可交叉）：

  1. chunk_size      —— 回答「为什么 chunk_size 选 500」
  2. embedding_model —— 回答「要不要换更新的嵌入模型」

为什么必须重建索引：这两个变量任何一个变了，**整库向量都得重算**。
chunk_size 变了分块就变了；嵌入模型变了向量空间就变了。

**隔离设计（重要）**：每轮建一个独立的临时向量库（默认 D:\\qa_chunk_exp\\ 下），
绝不触碰项目的 chroma_data/ 与 index_record.json。

  - 索引：incremental_index(chroma=..., splitter=..., index_file=...) 注入
    （这三个参数本来就是为可注入预留的）
  - 检索：改 rag.retriever 的模块级全局 `_chroma`，只在实验进程内生效

成本（2026-09 实测）：语料 134,108 字符 ≈ 0.62 token/字符，一轮约 9 万 token。
text-embedding-v4 与 qwen3.7-text-embedding 同为 ¥0.0005/千 token，一轮约 ¥0.05。

用法：
    python -m eval.experiment                                   # 按当前配置跑一轮（自检）
    python -m eval.experiment --sizes 300 500 800               # 比 chunk_size
    python -m eval.experiment --embeds text-embedding-v4 qwen3.7-text-embedding
    python -m eval.experiment --sizes 500 800 --embeds a b      # 交叉对比（4 轮）
    python -m eval.experiment --keep                            # 保留临时向量库
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# 允许 `python eval/experiment.py` 直接跑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import rag.retriever as R
from rag.indexer import incremental_index
from tools.config_loader import rag_conf

from eval.metrics import evaluate
from eval.run_eval import load_golden, render_markdown, retrieve_topics

# 与 rag/splitter.py 保持一致的分隔符
SEPARATORS = [".", ",", "!", "?", "。", "，", "！", "？", "\n\n", "\n", " ", ""]

# 检索候选池：固定不变，只让被测变量变
FETCH_K = 10
K_MAX = 10
KS = [1, 3, 5, 10]

DEFAULT_SIZES = [500]
DEFAULT_WORK = Path("D:/qa_chunk_exp")

# 单价（元/千 token）——两家相同，用于估算花费
PRICE_PER_1K = 0.0005
TOKENS_PER_RUN = 83_000 * 1.10  # 语料全量嵌入一次 + 分块重叠


def build_splitter(chunk_size: int) -> RecursiveCharacterTextSplitter:
    """按 chunk_size 建切割器。overlap 固定取 10%，随 chunk_size 等比缩放。"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 10,
        separators=SEPARATORS,
        length_function=len,
    )


def build_embedding(model_name: str) -> DashScopeEmbeddings:
    """按模型名构造嵌入函数（与 model.factory.EmbeddingsFactory 同一实现）。"""
    return DashScopeEmbeddings(model=model_name)


def _retry(fn, attempts: int = 3, delay: float = 8.0, label: str = ""):
    """对网络抖动做重试。

    实测遇到过 dashscope.aliyuncs.com 的瞬时 DNS 失败（getaddrinfo failed），
    整轮实验就此中断。这类错误重试一次基本就好了。
    """
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if i == attempts - 1:
                raise
            print(f"  {label} 第 {i + 1} 次失败（{type(exc).__name__}），"
                  f"{delay:.0f}s 后重试……", flush=True)
            time.sleep(delay)


def run_one(size: int, embed_name: str, work: Path, golden: list[dict]) -> dict:
    """跑一组配置：重建索引 → 检索 → 算指标。"""
    root = work / f"cs{size}__{embed_name}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}\nchunk_size = {size} (overlap {size // 10})   "
          f"embed = {embed_name}\n{'=' * 60}")

    chroma = Chroma(
        collection_name=rag_conf["vector_database"]["collection_name"],
        embedding_function=build_embedding(embed_name),
        persist_directory=str(root / "chroma"),
    )
    splitter = build_splitter(size)

    t0 = time.perf_counter()
    index_file = root / "index_record.json"
    stats = incremental_index(chroma=chroma, splitter=splitter, index_file=index_file)

    # 单文件失败（多为瞬时网络错误）靠增量索引自愈：
    # 索引记录已写好，再跑一轮只会重试失败的那几个文件，成功的会被 skip。
    for _ in range(2):
        if not stats["failed"]:
            break
        print(f"  ⚠️ {len(stats['failed'])} 个文件失败，重试……", flush=True)
        time.sleep(5)
        stats = incremental_index(chroma=chroma, splitter=splitter, index_file=index_file)
    index_seconds = time.perf_counter() - t0

    n_chunks = chroma._collection.count()
    print(f"  索引完成：{len(stats['added'])} 个文件 / {n_chunks} 个 chunk / "
          f"{len(stats['failed'])} 个失败 / {index_seconds:.1f}s")
    for f in stats["failed"]:
        print(f"    ⚠️ 失败: {f}")

    # 注入给 retriever —— get_chroma() 检查 `if _chroma is None`，设置后即生效
    R._chroma = chroma
    try:
        records = _retry(lambda: retrieve_topics(golden, K_MAX), label="检索")
    finally:
        R._chroma = None  # 恢复，避免影响同进程内的后续调用

    return {
        "chunk_size": size,
        "chunk_overlap": size // 10,
        "embed_model": embed_name,
        "n_chunks": n_chunks,
        "index_seconds": round(index_seconds, 2),
        "failed": stats["failed"],
        "summaries": [evaluate(records, k) for k in KS],
        "records": records,
    }


def comparison_table(results: list[dict]) -> str:
    """把各轮结果并成一张对照表。两个变量都列出来，不做省略 —— 免得看混。"""
    lines = [
        "## 检索质量对照",
        "",
        f"固定条件：MMR 检索 / fetch_k={FETCH_K} / 同一份 golden set / 同一批语料文件",
        "",
        "| chunk_size | 嵌入模型 | chunk 数 | "
        + " | ".join(f"Hit@{k}" for k in KS) + " | MRR | MAP |",
        "|" + "---|" * (3 + len(KS) + 2),
    ]
    for r in results:
        by_k = {s["k"]: s for s in r["summaries"]}
        hits = " | ".join(f"{by_k[k]['hit_at_k']:.3f}" for k in KS)
        lines.append(
            f"| {r['chunk_size']} | {r['embed_model']} | {r['n_chunks']} | {hits} "
            f"| {by_k[KS[0]]['mrr']:.3f} | {by_k[KS[0]]['map']:.3f} |"
        )

    lines += ["", "Recall@K / Precision@K 明细：", ""]
    for r in results:
        lines.append(f"**chunk_size={r['chunk_size']} · {r['embed_model']}**"
                     f"（{r['n_chunks']} chunk）")
        lines.append("")
        lines.append("| K | Recall@K | Hit@K | Precision@K |")
        lines.append("|---|---|---|---|")
        for s in r["summaries"]:
            lines.append(
                f"| {s['k']} | {s['recall_at_k']:.3f} | {s['hit_at_k']:.3f} "
                f"| {s['precision_at_k']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="检索质量对照实验")
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
                    help="chunk_size 取值（默认 [500]）")
    ap.add_argument("--embeds", nargs="+", default=None,
                    help="嵌入模型名（默认取 config/rag.yaml 里的那个）")
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK,
                    help="临时向量库目录（默认 D:/qa_chunk_exp）")
    ap.add_argument("--golden", type=Path,
                    default=Path(__file__).resolve().parent / "golden_set.json")
    ap.add_argument("--out", type=Path, default=None, help="结果 Markdown 输出路径")
    ap.add_argument("--keep", action="store_true", help="保留临时向量库（默认跑完删除）")
    args = ap.parse_args()

    default_embed = rag_conf["vector_database"]["embedding_model_name"]
    embeds = args.embeds or [default_embed]
    golden = load_golden(args.golden)

    combos = [(s, e) for s in args.sizes for e in embeds]
    print(f"将跑 {len(combos)} 组配置：")
    for s, e in combos:
        print(f"    chunk_size={s}  embed={e}")
    print(f"golden set {len(golden)} 条")
    print(f"预计嵌入消耗约 {TOKENS_PER_RUN * len(combos):,.0f} token "
          f"≈ ¥{TOKENS_PER_RUN * len(combos) / 1000 * PRICE_PER_1K:.3f}"
          f"（¥{PRICE_PER_1K}/千 token）")
    print(f"临时向量库目录: {args.work}")

    results = [run_one(s, e, args.work, golden) for s, e in combos]

    md = comparison_table(results)
    print("\n\n" + "=" * 60)
    print(md)
    print("=" * 60)

    if args.out:
        args.out.write_text(md + "\n", encoding="utf-8")
        print(f"\n已写入 {args.out}")

    raw = args.work / "results.json"
    raw.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"原始数据: {raw}")

    if not args.keep:
        print("\n清理临时向量库……（--keep 可保留）")
        for s, e in combos:
            shutil.rmtree(args.work / f"cs{s}__{e}", ignore_errors=True)
        print("已清理")


if __name__ == "__main__":
    main()
