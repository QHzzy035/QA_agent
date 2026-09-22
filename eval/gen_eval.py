"""
文件名：gen_eval.py
用途：生成质量评测 —— 回答「模型会不会胡说」。

分三层，逐层收窄：

    第 1 层  groundedness（资料忠实度）
             答案里哪些内容**跑出了检索到的资料**？不判真假，只判「在不在资料里」。

    第 2 层  事实核查
             对上一步挑出的陈述**逐条判真假**：正确 / 错误 / 不确定。
             用推理模型（qwen3.8-flash），因为它需要「知道自己不知道」——
             调用方明确要求「没把握就说不确定，不要猜」。

    第 3 层  联网核实
             用 Tavily 搜这条陈述，把搜索结果交给模型重判一次。
             LLM 单点判断只算「一个意见」，加上独立来源才叫核实。

最终指标是**三个数**，不是一个分数：
    编造数 / 正确补充数 / 无法确定数

⚠️ 关于「跑出资料 ≠ 胡说」：
    第 1 层的产出**不等于**编造。实测中大量「不被资料支持」的陈述是
    **事实正确、只是不在检索到的片段里**的内容（模型在补充自己的知识）。
    所以才有第 2 层 —— 别把 groundedness 说成幻觉率。

⚠️ 裁判模型不能是 DeepSeek（生成方），否则有自评偏差。

成本（2026-09 实测）：qwen3.8-flash 输入 ¥0.8/M、输出 ¥2.7/M。
单条核查约 143 输入 / 927 输出 ≈ ¥0.0026。全量约 ¥0.2–0.5。

⚠️ **每条查询默认跑 3 轮**：DeepSeek 侧存在 provider 层非确定性，
   温度设成 0 也不能保证输出一致（实测同问题仍有约 6% 波动）。
   单次分数不可复现，所以报的是均值与范围，不是一个数。

⚠️ **评测算的是应用的完整行为**：agent 与应用完全一致（含 web_search）。
   联网搜索的结果会被**捕获并并入判分材料** —— 否则「基于联网结果的正确回答」
   会被误判成「跑出资料」。

⚠️ **第 1 层会剔除「关于资料的元陈述」**（如「资料中未提及 X」）。
   这类句子在说"材料里有没有"，不是"世界上有没有"，无法用事实核查判断。
   第一版没剔除，结果 7 条「编造」里 6 条是误判 —— 见 notes/工作日志.md。

用法：
    python -m eval.gen_eval                        # 5 条 × 3 轮（默认）
    python -m eval.gen_eval --limit 0              # 全量 44 条 × 3 轮
    python -m eval.gen_eval --repeat 1             # 只跑一轮（快，但分数不可复现）
    python -m eval.gen_eval --no-verify            # 跳过第 3 层（省钱、快）
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# DashScope 的 OpenAI 兼容端点。qwen3.8-flash 走原生 SDK 会报「模型不存在」，
# 兼容端点才认。
COMPAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

GROUNDEDNESS_MODEL = "qwen-plus"     # 第 1 层：对着文本比对，不需要重推理
FACTCHECK_MODEL = "qwen3.8-flash"    # 第 2、3 层：判真假，需要推理

DEFAULT_GOLDEN = Path(__file__).resolve().parent / "golden_set.json"

# ---------------------------------------------------------------- 第 1 层

GROUNDEDNESS_PROMPT = """你是严格的答案质量评测员。下面给你【资料】【问题】【回答】。

请判断两件事：

1. **忠实度（faithfulness）**：回答中的每一条事实性陈述，能否由【资料】支持？
   资料里没有的、或与资料矛盾的内容都算「不被支持」。
   全部有据 = 1.0；有明显编造 = 0.0。
2. **相关性（relevance）**：回答有没有正面回应用户的问题？
   完全答非所问 = 0.0；准确切题 = 1.0。

注意：**回答里出现"资料不足/无法回答"并说明原因是加分的**，不算不忠实。

**unsupported 列表的硬性要求**（不满足的宁可不列）：

- 每条必须是**自包含、能独立判断真假的完整陈述**，不要摘录片段。
  反例：「第三波浪潮」—— 太短，不知道指什么，根本没法核查。
  正例：「AI 的第三波浪潮始于 2010 年前后，由深度学习推动」。
- 🔴 **绝对不要列「关于资料本身的陈述」**。例如：
    「Thread 在资料中未给出任何技术参数」
    「Wi-Fi 在表格中被列为家居通信协议，但资料中未出现相关描述」
  这类句子的**主语是「资料」**，在说"材料里有没有"，**不是"世界上有没有"**。
  它们根本无法用事实核查判断 —— 列进来只会制造误判。
  只列**关于客观世界的事实性陈述**（人名、年份、数字、因果）。
- 最多列 8 条，挑**最值得核查**的：含具体人名、年份、数字、因果断言的优先。

【资料】
{context}

【问题】
{question}

【回答】
{answer}

只输出一个 JSON 对象，不要解释文字、不要 markdown 代码块：
{{"faithfulness": <0到1的小数>, "relevance": <0到1的小数>,
"unsupported": ["回答中无法由资料支持的具体陈述", "..."], "reason": "一句话理由"}}"""


# ---------------------------------------------------------------- 第 2 层

FACTCHECK_PROMPT = """判断下面这条陈述在**客观事实上**是否正确。

陈述：{claim}

要求：
- 如果你没有把握，**必须回答"不确定"，不要猜**。宁可说不确定，也不要给一个错的判断。
- 只依据你确实掌握的知识，不要为了给出结论而编造。

⚠️ **如果这条陈述在说「资料/文档/材料里有什么」**（例如「资料中未提及 X」
「文档把 Y 列为 Z」），那就**不是事实问题** —— 要判断它得去核对那份资料，
而你手上没有。这种情况**必须回答"不确定"**，不要拿你对外部世界的了解去判它。

只输出 JSON，不要解释文字、不要 markdown 代码块：
{{"verdict": "正确" 或 "错误" 或 "不确定", "reason": "一句话依据"}}"""


# ---------------------------------------------------------------- 第 3 层

REJUDGE_PROMPT = """下面是一条待核实的陈述，以及一份联网搜索的结果摘要。

【陈述】
{claim}

【搜索引擎返回的摘要】
{evidence}

**第一步：先判断相关性 —— 这一步最容易出错，务必认真。**

这些结果和陈述讲的是**同一件事**吗？

如果结果讲的是**另一个领域**的同名概念，就是**不相关**。实例：
陈述讲的是 AI 的「第三波浪潮」，而搜索结果全是政治学的
「第三波民主化浪潮」（亨廷顿）—— 这是**不相关**，不是支持。

**第二步：只在相关的前提下判真假。**

- 摘要明确支持 → "正确"
- 摘要明确矛盾 → "错误"
- 摘要没提到、说法不一、或**不相关** → "不确定"

⚠️ 不相关时 `relevant` 必须为 false，`verdict` 必须为「不确定」——
**绝对不要拿别的领域的材料硬套到这条陈述上。**

只输出 JSON，不要解释文字、不要 markdown 代码块：
{{"relevant": true 或 false, "verdict": "正确" 或 "错误" 或 "不确定",
"reason": "一句话依据"}}"""


def _llm(prompt: str, model: str, timeout: int = 240) -> tuple[str, dict]:
    """调 DashScope 的 OpenAI 兼容端点，返回 (正文, usage)。"""
    import requests

    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("环境变量里没有 DASHSCOPE_API_KEY")

    resp = requests.post(
        COMPAT_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    return (msg.get("content") or "").strip(), (data.get("usage") or {})


def _parse_json(raw: str) -> dict | None:
    """从模型输出里抠出第一个 JSON 对象。模型偶尔会裹代码块或加一句话。"""
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _retry(fn, attempts: int = 3, delay: float = 8.0, label: str = ""):
    """网络抖动重试 —— 实测遇到过 dashscope 的瞬时 DNS 失败。"""
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if i == attempts - 1:
                raise
            print(f"      {label} 第 {i+1} 次失败（{type(exc).__name__}），"
                  f"{delay:.0f}s 后重试……", flush=True)
            time.sleep(delay)


# ---------------------------------------------------------------- 三层实现

_eval_agent = None

# 本轮生成期间的工具调用记录（每轮开始前清空）
_tool_log: list[dict] = []


def _make_capture_middleware():
    """构造一个「记录工具调用与返回」的中间件。

    为什么需要捕获：应用的 agent 会**自行联网搜索**，而搜索结果不在检索语料里。
    判分若只看语料，就会把「基于联网结果的正确回答」判成「跑出资料」。

    （第一版曾把 web_search 直接从评测 agent 里去掉 —— 那是错的：
      应用的系统提示词明写着「你拥有联网搜索工具」，工具一抽掉，
      模型就改口说「抱歉，搜索工具暂时不可用」，行为与应用完全脱节。）
    """
    from langchain.agents.middleware import wrap_tool_call

    @wrap_tool_call
    def capture_tool(request, handler):
        result = handler(request)
        content = getattr(result, "content", result)
        _tool_log.append({
            "name": request.tool_call["name"],
            "args": request.tool_call.get("args", {}),
            "result": str(content)[:3000],
        })
        return result

    return capture_tool


def _get_eval_agent():
    """评测用 agent —— 与应用的 agent **完全一致**，只多一个工具记录中间件。

    工具集、系统提示词、其余中间件都照搬 agent.py，保证测的是应用的真实行为。
    """
    global _eval_agent
    if _eval_agent is None:
        from langchain.agents import create_agent

        from agent import switch_to_report_mode, switch_to_summary_mode, web_search
        from middleware import (MODE_PROMPTS, log_before_model, mode_switch,
                                monitor_tool)
        from model.factory import chat_model

        _eval_agent = create_agent(
            model=chat_model,
            system_prompt=MODE_PROMPTS["normal"],
            tools=[web_search, switch_to_summary_mode, switch_to_report_mode],
            middleware=[monitor_tool, _make_capture_middleware(),
                        log_before_model, mode_switch],
        )
    return _eval_agent


def generate_answer(query: str) -> tuple[str, str, list[str]]:
    """用应用的生成链路生成答案。返回 (答案, 提示词全文, 来源文件名)。"""
    from langchain_core.messages import HumanMessage

    from rag.connected_prompts import new_prompt

    prompt_text, sources = new_prompt(query)
    # 必须传 context —— 中间件要读 runtime.context["mode"]，不传会拿到 None
    response = _get_eval_agent().invoke(
        {"messages": [HumanMessage(prompt_text)]},
        context={"mode": "normal"},
    )
    return response["messages"][-1].content, prompt_text, [s["source"] for s in sources]


# 「引用语料」的陈述特征。
#
# 实测教训：第一轮跑出 7 条「编造」，逐条验证后发现全是误判 —— 模型说的是
# 「资料里没有 X」「这个数字出自资料中对 Y 的描述」，**主语是「资料」**；
# 而裁判拿网页去核实，发现 X 在世界上存在 → 判「错误」。**逻辑完全错位。**
#
# 为什么不是只挡「否定式」：一开始只挡了「资料中未…」，结果「…出自资料中的描述」
# 这类肯定式照样误判。凡是**引用语料的陈述**都排除 —— 它们要用那份语料来判断，
# 不是事实问题，而事实核查层手上没有语料。
#
# 代价：少数「资料提到 X 导致 Y」这种夹带语料引用的世界性陈述会被一并丢弃。
# 宁可漏检，也不要产出站不住的假阳性。
_CORPUS_REF_MARKERS = (
    "资料", "文档", "文中", "材料", "原文", "知识库", "检索到", "给出的信息",
)


def _is_meta_statement(claim: str) -> bool:
    """判断一条陈述是否在**引用语料**（而非陈述客观世界的事实）。

    这类陈述无法用事实核查判断 —— 要判它得去核对那份语料。
    """
    return any(m in claim for m in _CORPUS_REF_MARKERS)


def judge_groundedness(context: str, question: str, answer: str,
                       model: str) -> dict:
    prompt = GROUNDEDNESS_PROMPT.format(
        context=context[:6000], question=question, answer=answer[:3000])
    raw, usage = _retry(lambda: _llm(prompt, model), label="groundedness")
    verdict = _parse_json(raw) or {}
    verdict["_usage"] = usage
    verdict["_raw"] = raw
    return verdict


def fact_check(claim: str, model: str) -> dict:
    """第 2 层：只凭模型自己的知识判真假。"""
    raw, usage = _retry(lambda: _llm(FACTCHECK_PROMPT.format(claim=claim), model),
                        label="fact_check")
    verdict = _parse_json(raw) or {"verdict": "不确定", "reason": f"输出无法解析：{raw[:80]}"}
    verdict["_usage"] = usage
    return verdict


_tavily = None


def _get_tavily():
    """Tavily 客户端（惰性单例，避免每次核查都重建）。"""
    global _tavily
    if _tavily is None:
        from langchain_tavily import TavilySearch
        _tavily = TavilySearch(max_results=5, topic="general")
    return _tavily


def verify_online(claim: str, query: str = "") -> list[dict]:
    """第 3 层：联网搜索，返回 [{content, url}, ...]。

    **搜索词要带上原问题**。实测踩过坑：只用陈述去搜「第三波浪潮」，
    搜回来全是政治学的「第三波民主化浪潮」，裁判拿它当证据判了「正确」。
    带上问题（「人工智能发展史… 第三波浪潮」）才能限定到正确的领域。
    """
    search_query = f"{query} {claim}".strip() if query else claim
    result = _retry(lambda: _get_tavily().invoke({"query": search_query}), label="tavily")
    return [
        {"content": (r.get("content") or "")[:600], "url": r.get("url", "")}
        for r in result.get("results", [])
    ]


def rejudge_with_evidence(claim: str, evidence: list[dict], model: str) -> dict:
    """第 3 层：把搜索结果交给模型重判。"""
    if not evidence:
        return {}
    block = "\n\n".join(f"[{i}] {e['content']}" for i, e in enumerate(evidence, 1))
    raw, usage = _retry(
        lambda: _llm(REJUDGE_PROMPT.format(claim=claim, evidence=block[:6000]), model),
        label="rejudge")
    verdict = _parse_json(raw) or {"verdict": "不确定", "reason": "输出无法解析"}
    verdict["_usage"] = usage
    return verdict


# ---------------------------------------------------------------- 编排

def run_query(query: str, repeat: int, ground_model: str, fact_model: str,
              verify: bool) -> dict:
    """同一条查询跑 repeat 轮，返回分布。

    为什么要跑多轮：DeepSeek 侧存在 **provider 层的非确定性**（MoE 的专家路由
    随批次组成变化），**温度设成 0 也不能保证输出一致** —— 实测同一问题仍有
    约 6% 的长度波动。单次分数不可复现，只能跑多轮看分布。
    """
    runs = []
    for i in range(repeat):
        print(f"    第 {i+1}/{repeat} 轮……", flush=True)
        runs.append(run_once(query, ground_model, fact_model, verify))

    fs = [r["faithfulness"] for r in runs if r["faithfulness"] is not None]
    return {
        "query": query,
        "runs": runs,
        "faithfulness_mean": round(sum(fs) / len(fs), 3) if fs else None,
        "faithfulness_min": round(min(fs), 3) if fs else None,
        "faithfulness_max": round(max(fs), 3) if fs else None,
    }


def run_once(query: str, ground_model: str, fact_model: str,
             verify: bool) -> dict:
    _tool_log.clear()
    t0 = time.perf_counter()
    answer, prompt_text, sources = generate_answer(query)
    gen_seconds = time.perf_counter() - t0

    # 把联网搜索结果并入判分材料：模型看得到，裁判也必须看得到，
    # 否则「基于联网结果的正确回答」会被误判成「跑出资料」。
    web = [t for t in _tool_log if t["name"] == "web_search"]
    judge_context = prompt_text
    if web:
        judge_context += ("\n\n===== 联网搜索结果（模型当时也看到了这部分）=====\n"
                          + "\n\n".join(t["result"] for t in web))

    ground = judge_groundedness(judge_context, query, answer, ground_model)

    # 两道兜底过滤（提示词里都要求过了，但模型不一定照做）：
    #   ① 过短的片段无法独立核查（例如光一个「第三波浪潮」）
    #   ② 关于资料的元陈述（「资料中未提及 X」）无法用事实核查判断
    raw_claims = ground.get("unsupported") or []
    claims = [c for c in raw_claims
              if len(str(c).strip()) >= 10 and not _is_meta_statement(str(c))]
    dropped_meta = sum(1 for c in raw_claims if _is_meta_statement(str(c)))

    checks = []
    for claim in claims:
        fc = fact_check(claim, fact_model)
        item = {
            "claim": claim,
            "verdict": fc.get("verdict", "不确定"),
            "reason": fc.get("reason", ""),
            "evidence": [],
            "rejudged": None,
            "relevant": None,
        }
        if verify:
            item["evidence"] = verify_online(claim, query)
            rj = rejudge_with_evidence(claim, item["evidence"], fact_model)
            item["rejudged"] = rj.get("verdict")
            item["relevant"] = rj.get("relevant")
            # **只有证据「相关」时才允许改判** —— 不相关还改，就等于拿
            # 别的领域的材料覆盖原判，实测会制造假阳性。
            if rj.get("relevant") and rj.get("verdict"):
                item["final"] = rj["verdict"]
                item["final_reason"] = rj.get("reason", "")
        item.setdefault("final", item["verdict"])
        item.setdefault("final_reason", item["reason"])
        checks.append(item)

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "used_web_search": bool(web),
        "dropped_meta_claims": dropped_meta,
        "gen_seconds": round(gen_seconds, 2),
        "faithfulness": ground.get("faithfulness"),
        "relevance": ground.get("relevance"),
        "claims": checks,
    }


def _counts(results: list[dict]) -> tuple[dict, int, int]:
    """**跨所有轮次**统计每条陈述的最终判定。

    另返回两个健康指标：
      flipped    —— 联网核实后**真的改判了**的条数（只算证据相关的）
      irrelevant —— 搜索证据被判为**与陈述无关**的条数
                    这个数高说明搜索词质量差，是评测本身的告警，不是模型的问题
    """
    tally = {"编造": 0, "正确补充": 0, "无法确定": 0}
    flipped = 0
    irrelevant = 0
    for r in results:
        for run in r.get("runs", []):
            for c in run["claims"]:
                if c.get("relevant") is False:
                    irrelevant += 1
                elif c.get("rejudged") and c["rejudged"] != c["verdict"]:
                    flipped += 1
                v = c["final"]
                if v == "错误":
                    tally["编造"] += 1
                elif v == "正确":
                    tally["正确补充"] += 1
                else:
                    tally["无法确定"] += 1
    return tally, flipped, irrelevant


def summarize(results: list[dict], verified: bool, repeat: int) -> str:
    all_fs = [run["faithfulness"] for r in results for run in r["runs"]
              if run.get("faithfulness") is not None]
    all_rel = [run["relevance"] for r in results for run in r["runs"]
               if run.get("relevance") is not None]
    tally, flipped, irrelevant = _counts(results)
    total_claims = sum(tally.values())

    lines = [
        "## 生成质量评测结果",
        "",
        f"{len(results)} 条查询 × **{repeat} 轮** = {len(results) * repeat} 次生成，"
        f"共 {total_claims} 条待核查陈述",
        "",
        f"> **为什么要跑 {repeat} 轮**：DeepSeek 侧存在 provider 层非确定性"
        "（MoE 专家路由随批次变化），**温度设成 0 也不能保证输出一致** —— "
        "实测同一问题仍有约 6% 的长度波动。单次分数不可复现，只能看分布。",
        "",
        "### 第 1 层：资料忠实度",
        "",
    ]
    if all_fs:
        lines += [
            "| 指标 | 均值 | 最低 | 最高 |",
            "|---|---|---|---|",
            f"| 资料忠实度 | **{sum(all_fs)/len(all_fs):.3f}** "
            f"| {min(all_fs):.3f} | {max(all_fs):.3f} |",
            f"| 相关性 | **{sum(all_rel)/len(all_rel):.3f}** "
            f"| {min(all_rel):.3f} | {max(all_rel):.3f} |",
            "",
            "> 忠实度低**不等于胡说** —— 实测中大量「不被资料支持」的内容是"
            "**事实正确、只是不在检索片段里**的。真假要看第 2、3 层。",
            "",
        ]

    all_runs = [run for r in results for run in r["runs"]]
    n_web = sum(1 for run in all_runs if run.get("used_web_search"))
    n_meta = sum(run.get("dropped_meta_claims", 0) for run in all_runs)

    lines += [
        "### 第 2、3 层：跑出资料的部分，逐条核查",
        "",
        f"联网核实：{'已开启' if verified else '**未开启**（--no-verify）'}　|　"
        f"触发过联网搜索的轮次：**{n_web}/{len(all_runs)}**　|　"
        f"被剔除的「关于资料的元陈述」：**{n_meta}** 条",
        "",
        "| 分类 | 条数 |",
        "|---|---|",
        f"| 🔴 **编造**（判定为错误） | **{tally['编造']}** |",
        f"| ✅ 正确补充 | {tally['正确补充']} |",
        f"| ⚠️ 无法确定 | {tally['无法确定']} |",
    ]
    if verified:
        lines += [
            f"| （其中证据**相关**、并因此改判的） | {flipped} |",
            f"| （其中证据被判**不相关**、维持原判的） | {irrelevant} |",
        ]
    lines.append("")
    if verified and total_claims and irrelevant / total_claims > 0.3:
        lines += [
            f"> ⚠️ **{irrelevant}/{total_claims} 条的证据被判不相关** —— 比例偏高，",
            "> 说明搜索词质量有问题，**这个评测自身需要先修**，别拿结论下判断。",
            "",
        ]

    if total_claims:
        lines += [
            f"> 跨 {repeat} 轮共 {total_claims} 条补充陈述，"
            f"**{tally['编造']} 条判为编造**（{tally['编造']/total_claims:.0%}）。",
            "",
        ]

    lines += ["### 逐条明细", ""]
    for r in results:
        mean = r.get("faithfulness_mean")
        rng = ""
        if r.get("faithfulness_min") is not None and r["faithfulness_max"] != r["faithfulness_min"]:
            rng = f"，范围 {r['faithfulness_min']}–{r['faithfulness_max']}"
        lines.append(f"**「{r['query']}」**（忠实度 {mean}{rng}）")
        lines.append("")
        for i, run in enumerate(r["runs"], 1):
            if not run["claims"]:
                lines.append(f"- 第 {i} 轮：无跑出资料的陈述")
                continue
            lines.append(f"- **第 {i} 轮**（忠实度 {run['faithfulness']}）")
            for c in run["claims"]:
                mark = {"错误": "🔴", "正确": "✅", "不确定": "⚠️"}.get(c["final"], "❔")
                chg = f"（初判 {c['verdict']}）" if c.get("rejudged") else ""
                lines.append(f"  - {mark} {c['claim'][:78]}  {chg}")
                if c.get("final_reason"):
                    lines.append(f"    - 依据：{c['final_reason'][:100]}")
        lines.append("")
    return "\n".join(lines)


def _checkpoint(results: list[dict], json_out: Path | None, md_out: Path | None,
                verified: bool, repeat: int) -> None:
    """把当前进度写盘。

    实测教训：这个评测跑一轮约 50 分钟，中途被系统的内存回收器杀掉过一次。
    而当时输出**只在全部跑完后才写** —— 前面 40% 的进度一点不剩。
    改成每跑完一条就落盘，再被杀最多丢一条。
    """
    if json_out:
        json_out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    if md_out:
        md_out.write_text(summarize(results, verified, repeat) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="生成质量评测（三层）")
    ap.add_argument("--limit", type=int, default=5, help="只跑前 N 条（默认 5，0 为全量）")
    ap.add_argument("--repeat", type=int, default=3,
                    help="每条查询跑几轮（默认 3）。单次分数不可复现，见文件头说明")
    ap.add_argument("--ground-model", default=GROUNDEDNESS_MODEL)
    ap.add_argument("--judge-model", default=FACTCHECK_MODEL, help="事实核查模型")
    ap.add_argument("--no-verify", action="store_true", help="跳过第 3 层联网核实")
    ap.add_argument("--offset", type=int, default=0,
                    help="从第 N 条开始（配合 --limit 分批跑）")
    ap.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--from-json", type=Path,
                    help="不跑评测，直接从已保存的原始结果重新渲染报告")
    args = ap.parse_args()

    verify = not args.no_verify

    # 只重新渲染报告（改报告格式时不必再调一遍 API）
    if args.from_json:
        results = json.loads(args.from_json.read_text(encoding="utf-8"))
        report = summarize(results, verify, args.repeat)
        print(report)
        if args.out:
            args.out.write_text(report + "\n", encoding="utf-8")
            print(f"\n已写入 {args.out}")
        return

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))["queries"]
    queries = [g["query"] for g in golden]
    if args.offset:
        queries = queries[args.offset:]
    if args.limit:
        queries = queries[: args.limit]
    print(f"第1层 groundedness : {args.ground_model}")
    print(f"第2/3层 事实核查   : {args.judge_model}（推理型）")
    print(f"第3层 联网核实     : {'Tavily' if verify else '关闭'}")
    print(f"样本: {len(queries)} 条\n")

    results = []
    for i, q in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {q}", flush=True)
        try:
            r = run_query(q, args.repeat, args.ground_model, args.judge_model, verify)
            results.append(r)
            n_claims = sum(len(run["claims"]) for run in r["runs"])
            n_bad = sum(1 for run in r["runs"] for c in run["claims"] if c["final"] == "错误")
            print(f"    → 忠实度均值 {r['faithfulness_mean']}"
                  f"（{r['faithfulness_min']}–{r['faithfulness_max']}）"
                  f"  跑出资料 {n_claims} 条  判为编造 {n_bad} 条", flush=True)
        except Exception as exc:
            import traceback
            print(f"    ❌ {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            print("    " + traceback.format_exc()[-800:].replace("\n", "\n    "), flush=True)
            results.append({"query": q, "runs": [], "faithfulness_mean": None,
                            "faithfulness_min": None, "faithfulness_max": None})

        # 每跑完一条就落盘 —— 中途被杀最多丢当前这一条
        _checkpoint(results, args.json_out, args.out, verify, args.repeat)

    report = summarize(results, verify, args.repeat)
    print("\n" + "=" * 62)
    print(report)
    print("=" * 62)
    print(f"\n原始数据: {args.json_out}" if args.json_out else "")
    print(f"报告: {args.out}" if args.out else "")


if __name__ == "__main__":
    main()
