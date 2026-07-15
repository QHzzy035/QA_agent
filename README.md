# QA Agent — 文档问答助手

基于 RAG + 智能体的文档问答系统，支持多格式文档检索和联网搜索。

## 功能

- 📄 **多格式文档支持**：txt / pdf / docx / md
- 🔍 **MMR 多样化检索**：兼顾相关性和多样性
- 🌐 **智能联网搜索**：文档库查不到时自动联网搜索
- 💬 **多轮对话**：支持上下文对话 + 历史摘要压缩
- 🧠 **思考过程可视化**：侧边栏展示检索来源和工具调用
- 📤 **文件上传**：Streamlit 界面内直接上传文档

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/QHzzy035/QA-agent.git
cd QA-agent
```

### 2. 安装依赖

```bash
pip install -e .
```

### 3. 配置 API Key

在项目根目录创建 `.env` 文件：

```env
DASHSCOPE_API_KEY=你的阿里云APIKey
TAVILY_API_KEY=你的TavilyAPIKey
DEEPSEEK_API_KEY=你的DeepSeekAPIKey
```

### 4. 放入文档

将需要检索的文档放入 `test_data/` 文件夹。

### 5. 启动

```bash
streamlit run streamlit_file.py
```

## 项目结构

```
QA Agent/
├── agent.py                 # 智能体定义
├── streamlit_file.py        # Web 界面
├── config/                  # 配置目录（YAML 格式）
│   ├── LLM.yaml             #   大模型配置
│   ├── rag.yaml             #   RAG 检索配置
│   └── tavily.yaml          #   联网搜索配置
├── tools/
│   └── config_loader.py     # 配置加载工具
├── prompts/
│   └── system_prompts.txt   # 系统提示词
├── rag/
│   ├── loader.py            # 文档加载
│   ├── splitter.py          # 文档切割
│   ├── retriever.py         # 向量检索
│   └── connected_prompts.py # 提示词拼接
└── test_data/               # 文档存放目录
```