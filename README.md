# QA Agent — 你的私人知识库问答助手

[![CI](https://github.com/QHzzy035/QA_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/QHzzy035/QA_agent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)

把你收藏的文章、笔记、文档，变成一个可以**直接对话的虚拟知识库**。一个知识库可以挂多个文件夹来源，文件**原地索引、不复制不移动**；提问时跨来源检索、带出处回答，右侧可以立即预览原文；查不到还能联网搜索，并支持文档精读总结、报告生成模式。

## 工程要点

不只是把模型接起来跑通。几处专门做了工程处理：

**依赖瘦身：1.8 GB → 616 MB。** 项目原本通过 `unstructured` 引入了
torch / transformers / spacy，而代码实际只用到它的文件加载器。追溯依赖树
确认这些包从未被调用后，改用 `pypdf` + `python-docx` 直接实现——
索引模块的导入耗时从 8.1 秒降到 0.45 秒，安装体积从 208 个包降到 140 个。

**251 个测试 + GitHub Actions CI。** 除 3 个需要真实 Key 的界面冒烟外，全部
**不依赖 API Key、网络与真实向量库**（模型 / 向量库 / 文档加载器均以假对象
替换），CI 上直接运行；另有 1 个无 Key 界面冒烟，验证页面可渲染和预览切换。
这些测试不是凑覆盖率的摆设——开发过程中它们抓出了
4 个真实 Bug，其中一个（Windows 换行符未归一化导致文本切分的段落边界被
静默忽略）不报错、不崩溃，只是让检索质量悄悄变差。

**44 条标注的检索评测集，Hit@1 93.2%。** 用 Recall@K / MRR / MAP 量化检索质量，
让调参有依据而不是凭感觉改 `k` 和 `chunk_size`。标注刻意**不照抄文档原句**——
字面完全重合的话，连最笨的检索器都能命中，指标会虚高。三条未命中的查询逐个做了
归因（一条栽在「数字」与「向量」的口语化词汇鸿沟上，一条被「虚拟机」的字面重叠带偏），
**没有为了让数字好看而把难例删掉**。

参数也是**测出来**的，不是拍的：对照实验显示 `chunk_size` 在 300–800 区间
**不敏感**（差异仅 1 条查询，落在噪声内），由此判断不该在此处继续调参；嵌入模型
对比后换用 `qwen3.7-text-embedding`（Hit@1 持平，Hit@3 与 MRR/MAP 一致更好）。
见 [`eval/`](eval/)。

**生成质量也评了——不只是检索。** 回答是严格依据检索到的资料，还是自己补充了外部知识？
补充的部分对不对？三层设计：先挑出「跑出资料」的内容 → 逐条判真假 → 用独立来源联网核实。

44 条 × 3 轮的结果：**忠实度 0.973**，171 条待核查陈述里**验证到 3 条属实错误（约 1.8%）**。
而且**没有一条是凭空捏造**——全是「记错年份」「记错数字」「话说太满」，
**是精度问题，不是幻觉问题**。

> **已知局限（写在前面）**：裁判模型本身也会判错（实测 5 条里有 1 条误判），
> 也区分不了「事实陈述」与「观点/建议」。所以那个 1.8% 是**带误差的估计，不是精确值**。
> 另外评测只覆盖了 DeepSeek 一个生成模型。

**纯标准库手写的 PDF 生成器。** TrueType 字体内嵌、cmap 子表解析、ToUnicode
CMap、xref 交叉引用表全部自己实现，不引入 reportlab 等依赖。支持 `.ttf` 与
`.ttc` 字体集合，生成的文件用 pypdf 能正确抽回中文。见
[`tools/pdf_writer.py`](tools/pdf_writer.py)。

**模块导入无副作用。** 模型、向量库与联网客户端改为惰性创建，任何模块都能在
没有 API Key 的环境下导入——这是让上面那 251 个测试得以存在的前提。CI 里有一步
专门断言这件事，防止以后有人把模块级副作用加回来。

## 功能

- 📚 **虚拟知识库**：一个库挂多个文件夹来源，跨盘也可以；文件原地索引，不复制、不移动
- 📄 **多格式文档支持**：txt / pdf / docx / md
- 🗂️ **统一文档列表**：已索引 / 待更新 / 未索引 / 已移出 / 文件缺失 / 来源离线，支持筛选
- ⚡ **增量同步**：基于 MD5 只处理新增/变更/删除，避免全量重建；同步前展示范围预览
- 🗑️ **安全移出**：默认只移出索引并保留文件，可恢复；物理删除走系统回收站
- 👁️ **右侧预览**：对话 / 预览切换，不打断聊天；回答里的来源可点击跳转
- 📤 **多文件上传**：选择目标来源，重名自动改名，上传后自动同步
- 🎭 **三模式问答**：普通问答 / 文档精读总结 / 报告生成，智能体根据用户需求自动切换
- 🔍 **MMR 多样化检索**：兼顾相关性和多样性
- 🌐 **智能联网搜索**：文档库查不到时自动联网搜索，结果保留标题与来源 URL
- 💬 **多轮对话**：上下文理解 + 指代消解（查询改写）+ 来源记忆，历史过长自动摘要压缩
- 🧠 **思考过程可视化**：侧边栏展示检索来源和工具调用
- 📂 **打开文件夹**：在系统文件管理器中定位来源或当前文档
- ⚙️ **持久化设置**：参考片段数、历史阈值、主题（跟随系统/浅色/深色）、上下文预算
- 🧱 **上下文保护**：字符预算裁剪 + 上下文超限自动缩短重试
- 📋 **结构化日志**：控制台 + 文件双输出，按天分割，便于排查

## MVP 边界

- **单知识库、多来源**：不做多个独立知识库同时检索；
- **文本预览**：PDF/DOCX 抽取文本展示，不做 PDF 原样渲染或 OCR；
- **需要外部 API**：DeepSeek 对话、DashScope 嵌入、Tavily 搜索，均需网络与 Key；
- **手动同步**：不做文件监听，外部改文件后点「同步文档库」或重新上传；
- **单用户本地**：公开部署需自行加访问控制与额度限制；
- **回收站删除**：优先 `send2trash`；Windows 无该依赖时用系统 Shell 回收站，其他平台未安装时会返回失败提示。

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/QHzzy035/QA_agent.git
cd QA_agent
```

### 2. 安装依赖

项目使用 [uv](https://github.com/astral-sh/uv) 管理依赖，需要 Python >= 3.11。先安装 uv，然后在项目目录执行：

```bash
uv sync
```

### 3. 配置 API Key

复制 `.env.example` 为 `.env`，然后填入你自己的 Key：

```bash
# Windows（PowerShell）
Copy-Item .env.example .env
# macOS / Linux
cp .env.example .env
```

编辑 `.env`，填入三个 Key：

```env
DASHSCOPE_API_KEY=你的阿里云APIKey
TAVILY_API_KEY=你的TavilyAPIKey
DEEPSEEK_API_KEY=你的DeepSeekAPIKey
```

> ⚠️ **提醒**：`.env` 已被 `.gitignore` 排除，**不会提交到仓库**，请保管好自己的 Key 不要外泄。如果你已经把 Key 配在系统环境变量里了，也可以跳过这步（两种方式效果相同）。

### 4. 放入文档

**这一步可以跳过**：仓库自带的 `demo_data/` 里有 6 篇演示文档，首次启动会自动成为虚拟知识库的第一个来源；点左侧「🔄 同步文档库」即可建立索引。

想用自己的文档：

- **添加文件夹（推荐）**：左侧「➕ 添加文件夹」，选择已有文件夹。文件**原地索引，不复制、不移动**；可以添加多个来源，跨盘也可以。
- **上传文件**：左侧「📤 上传文件」，选择目标来源后上传。上传会**复制**到该来源文件夹，不会自动新增来源。
- **test_data**：如果 `test_data/` 存在且有文件，首次启动会优先把它作为默认来源。

也可以生成测试文档（txt 30 篇，md / docx / pdf 各 10 篇）：

  ```bash
  python -m tools.generate_txt    # 生成 30 篇 .txt
  python -m tools.generate_md     # 生成 10 篇 .md
  python -m tools.generate_docx   # 生成 10 篇 .docx
  python -m tools.generate_pdf    # 生成 10 篇 .pdf
  ```

  > 生成会调用 DeepSeek API 产生费用，建议安排在空闲时段运行。

### 5. 启动

```bash
streamlit run streamlit_file.py
```

### 6. 使用

- **添加来源**：左侧「➕ 添加文件夹」，可以添加多个文件夹；
- **同步文档库**：左侧「🔄 同步文档库」，先看扫描预览，再建立/更新索引；
- **普通问答**：直接提问，AI 跨当前知识库的所有启用来源回答；
- **预览原文**：点文档旁的 👁️，或点回答下方的来源，右侧切到预览模式；
- **移出/恢复**：文档「⋯ → 移出知识库」只移除索引，文件保留；「已移出」筛选里可恢复；
- **物理删除**：文档「⋯ → 删除文件（回收站）」，文件进系统回收站；
- **文档精读总结**：说「总结 / 精读 / 概括」某篇文档，自动切换到总结模式；
- **报告生成**：说「生成报告 / 写一份报告」，自动切换到报告模式。

## 项目结构

```
QA Agent/
├── agent.py                     # 智能体定义
├── middleware.py                # 智能体中间件（工具监控/日志/动态提示词）
├── streamlit_file.py            # Web 界面
├── pyproject.toml               # 项目依赖配置（uv 管理）
├── uv.lock                      # 依赖锁文件
├── requirements.txt             # 由 uv.lock 导出，供 Streamlit Cloud 部署使用
├── .python-version              # Python 版本要求
├── .github/
│   └── workflows/ci.yml         # CI：跑测试 + 校验模块可在无 Key 环境导入
├── tests/                       # 单元测试（不依赖 API Key / 网络 / 真实向量库）
│   ├── conftest.py              #   公共假对象（向量库 / 切分器 / 文档加载）
│   ├── test_library.py          #   虚拟库、来源、路径越界
│   ├── test_scanner.py          #   递归扫描、忽略规则、范围预览
│   ├── test_sync.py             #   增量同步、排除清单、状态推导
│   ├── test_settings_store.py   #   设置持久化与校验
│   ├── test_context_budget.py   #   上下文裁剪顺序
│   ├── test_trash.py            #   回收站辅助
│   ├── test_indexer.py          #   旧版单目录增量索引
│   ├── test_document_service.py #   多格式抽取、编码自适应、CRLF
│   ├── test_query_rules.py      #   检索前置规则
│   ├── test_factory.py          #   模型工厂选型与缓存
│   ├── test_pdf_writer.py       #   PDF 生成器结构有效性
│   ├── test_prompts.py          #   提示词、模式映射、按库来源键
│   ├── test_eval_metrics.py     #   检索指标计算
│   ├── test_app_no_key_smoke.py #   无 Key 界面渲染与预览切换
│   └── test_app_smoke.py        #   界面冒烟（需 API Key，无则跳过）
├── eval/                        # 检索质量评测
│   ├── README.md                #   度量口径、成本说明、局限
│   ├── golden_set.json          #   44 条标注查询，覆盖全部语料主题
│   ├── metrics.py               #   Recall@K / MRR / MAP（纯函数）
│   └── run_eval.py              #   评测入口（支持 --validate-only 零成本校验）
├── assets/
│   ├── doc_manage_demo.gif      # 文档库管理演示（动图）
│   └── dialog_demo.gif          # 知识库对话演示（动图）
├── .streamlit/
│   ├── config.toml              # Streamlit 配置
│   └── secrets.toml.example     # 云端部署的密钥模板（真正的 secrets.toml 已被 gitignore）
├── config/                      # 配置目录（YAML 格式）
│   ├── LLM.yaml                 #   大模型配置
│   ├── rag.yaml                 #   RAG 检索配置
│   └── tavily.yaml              #   联网搜索配置
├── model/
│   └── factory.py               # 模型工厂（聊天/总结/改写/嵌入统一创建，惰性实例化）
├── tools/
│   ├── app_paths.py             # 资源目录 / 数据目录（开发与打包兼容）
│   ├── library.py               # 虚拟知识库、多来源、路径校验
│   ├── settings_store.py        # 设置持久化（不含 API Key）
│   ├── context_budget.py        # 上下文预算与裁剪
│   ├── trash.py                 # 移到系统回收站
│   ├── config_loader.py         # 配置加载工具
│   ├── web_search.py            # 联网搜索工具（Tavily 封装 + 结果格式化，带 URL）
│   ├── logger_handler.py        # 结构化日志
│   ├── document_service.py      # 文档抽取/预览（旧单目录接口仍保留）
│   ├── pdf_writer.py            # 纯标准库 PDF 生成（中文字体内嵌）
│   ├── generate_txt.py          # 测试数据生成脚本（txt）
│   ├── generate_md.py           # 测试数据生成脚本（md）
│   ├── generate_docx.py         # 测试数据生成脚本（docx）
│   └── generate_pdf.py          # 测试数据生成脚本（pdf）
├── prompts/
│   ├── system_prompts.txt       # 系统提示词（普通问答模式）
│   ├── summary_prompt.txt       # 系统提示词（精读总结模式）
│   └── report_prompt.txt        # 系统提示词（报告生成模式）
├── rag/
│   ├── scanner.py               # 来源扫描（递归/忽略/范围预览）
│   ├── sync.py                  # 增量同步、排除清单、文档状态
│   ├── indexer.py               # 旧版单目录增量索引（兼容保留）
│   ├── splitter.py              # 文档切割
│   ├── retriever.py             # 向量检索（含按库检索）
│   └── connected_prompts.py     # 提示词拼接
├── data/                        # 运行时数据（已 gitignore）：libraries/settings/logs
├── demo_data/                   # 演示语料（随仓库提供，首次启动作为默认来源）
└── test_data/                   # 本地文档目录（已 gitignore；存在时优先作为默认来源）
```

> **虚拟库数据目录**：默认 `<项目>/data`（可用 `QA_AGENT_DATA_DIR` 覆盖），
> 每个库在 `data/libraries/<id>/` 下保存 `library.json`、`index_record.json`、
> `excluded.json` 和独立的 `chroma/`；`data/` 已被 `.gitignore` 排除。
>
> **首次启动**：自动创建默认知识库，按 `test_data`（非空）→ `demo_data`
> 的顺序挂载第一个来源。**不会自动建立索引**，需要点「🔄 同步文档库」。
>
> **上传方向**：上传会复制到用户选中的**已有来源**，不会自动新增来源；
> `demo_data` 不会被上传覆盖或删除。想原地索引已有文件夹，请用「添加文件夹」。
>
> **移出与删除**：「移出知识库」只移除索引并保留文件，可在「已移出」中恢复；
> 「删除文件（回收站）」才会移动磁盘文件。

## 开发

```bash
uv sync              # 安装依赖（含开发依赖）
uv run pytest        # 运行测试（无需 API Key 也可运行）
```

测试覆盖虚拟库/多来源、扫描与同步、移出与状态推导、设置持久化、
上下文裁剪、多格式抽取、模型工厂与提示词一致性等，不依赖 API Key、
网络与真实向量库；另有 1 个无 Key 界面冒烟测试验证页面渲染与预览切换。

检索质量评测见 [`eval/README.md`](eval/README.md)：

```bash
python -m eval.run_eval --validate-only   # 零成本校验标注
python -m eval.run_eval --k 1 3 5         # 正式评测（会调用嵌入 API）
```

## 部署到 Streamlit Community Cloud

1. 把仓库推到 GitHub（本地提交完成后执行 `git push`）

2. 到 <https://share.streamlit.io> 新建应用，选择本仓库，
   主文件路径填 `streamlit_file.py`

3. 在应用的 **Settings → Secrets** 里按 `.streamlit/secrets.toml.example`
   的格式填入三个 Key，保存后应用会自动重启

4. 依赖由仓库根目录的 `requirements.txt` 提供（从 `uv.lock` 导出，版本全锁定）。
   `test_data/` 与 `data/` 都被 gitignore，云端不存在，应用会自动使用 `demo_data/`
   作为默认来源；首次打开后点一次「同步文档库」即可。

> **⚠️ 公开部署前请注意**：应用会用**你自己**的 API Key 处理**任何人**的提问。
> 侧边栏的添加来源、上传与「同步文档库」对访客开放，意味着任何人都能添加/上传文档、
> 触发嵌入调用，消耗你的额度。正式公开前建议至少做其一：
> 关闭上传/添加来源入口、给应用加访问密码（Streamlit 支持在 Secrets 里配
> `password` 字段并用 `st.secrets` 校验），或设置用量告警。
