"""
文件名：streamlit_file.py
介绍：前端部分，使用streamlit进行编写
"""
# 依赖库导入
import streamlit as st
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessageChunk, AIMessage, SystemMessage
# 依赖文件导入
from agent import agent
from tools.config_loader import LLM_conf, rag_conf, resolve_data_dir
from tools.document_service import (
    read_document, get_document_list, delete_document,
    is_store_empty, open_in_file_manager, SUPPORTED_EXTENSIONS,
)
from rag.connected_prompts import new_prompt
from rag.indexer import incremental_index

# 方法定义：历史消息过多时，从最早的几次历史消息中进行总结（阈值可从 LLM.yaml 的 history_summarize_length 配置）
def summarize_history(history: list, summarize_length: int = None):
    if summarize_length is None:
        summarize_length = st.session_state.get("summarize_length_override", LLM_conf.get("history_summarize_length", 10))
    if len(history) <= summarize_length:
        return history

    from model.factory import summarize_model

    old_msg = ""
    old_history = history[:-summarize_length]
    recent_history = history[-summarize_length:]

    # 将旧的对话进行拼接
    for msg in old_history:
        if isinstance(msg, HumanMessage):
            old_msg += f"用户：{msg.content}\n"
        else:
            old_msg += f"智能助手：{msg.content}\n"

    # 总结旧的对话
    summarize = summarize_model.invoke(
        [
            SystemMessage("对以下对话进行简短的总结"),
            HumanMessage(old_msg)
        ]
    )

    return [summarize] + recent_history


@st.dialog("文档内容")
def show_document(name: str, content: str):
    """弹窗展示文档原文。"""
    st.caption(name)
    st.markdown(content)


# 设置弹窗：配置检索条数和历史总结阈值（仅本次会话生效，重启恢复默认）
@st.dialog("设置")
def settings_dialog():
    current_k = st.session_state.get("k_override", rag_conf["k"])
    current_summarize = st.session_state.get("summarize_length_override", LLM_conf.get("history_summarize_length", 10))

    new_k = st.slider("每次检索返回多少条", 1, 10, current_k)
    new_summarize = st.slider("历史多少条后压缩", 2, 50, current_summarize)

    st.caption("改动仅本次会话生效，重启后恢复默认值")

    col_save, col_reset = st.columns(2)
    with col_save:
        if st.button("保存", use_container_width=True):
            st.session_state["k_override"] = new_k
            st.session_state["summarize_length_override"] = new_summarize
            st.rerun()
    with col_reset:
        if st.button("恢复默认", use_container_width=True):
            st.session_state.pop("k_override", None)
            st.session_state.pop("summarize_length_override", None)
            st.rerun()


def auto_index_on_first_run():
    """首次运行时自动建一次索引。

    新克隆的仓库和云端部署都从空向量库开始。若不自动处理，使用者打开界面
    会发现问什么都查不到，还得先去侧边栏点一次「重新索引」再等进度条——
    演示时这一步很尴尬。只在「库为空」且「文档目录里确实有文件」时才触发，
    已有索引的环境完全不受影响。
    """
    data_dir = resolve_data_dir()
    has_files = any(
        p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        for p in data_dir.glob("*.*")
    )
    if not has_files or not is_store_empty():
        return

    with st.spinner(f"首次运行，正在为 {data_dir.name} 中的文档建立索引..."):
        stats = incremental_index()
    st.toast(f"索引完成：新增 {len(stats['added'])} 个文档", icon="✅")


st.title("文档问答助手")

# 首次运行自动建索引，让「clone 下来直接跑」无需任何手动步骤
auto_index_on_first_run()

# 上传文件
with st.sidebar:
    # 限制侧边栏最小宽度，防止被压缩到太窄导致内容错位
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        min-width: 340px !important;
    }
    /* 文档名不换行，超出省略，避免窄屏下被掰断 */
    .doc-name {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        display: block;
    }
    </style>
    """, unsafe_allow_html=True)

    # 设置按钮（放在侧边栏顶部，固定不随主内容滚动）
    if st.button("⚙️ 设置", use_container_width=True):
        settings_dialog()

    # 上传器 key 含重置计数：删除文档时自增，换 key 清空上传器，
    # 避免被删除文件的上传残留状态在 rerun 时重新写回
    if "uploader_reset" not in st.session_state:
        st.session_state.uploader_reset = 0

    st.header("文档管理")

    # 文档目录：显示位置 + 一键在系统文件管理器中打开
    # 打开动作是由运行 Streamlit 的机器执行的（浏览器不允许网页碰本地文件系统），
    # 所以本地跑时开的是你自己的文件夹，云端部署时服务器没有图形界面会失败——
    # 失败时把完整路径显示出来，让人能自己去找。
    data_dir = resolve_data_dir()
    col_dir, col_open = st.columns([5, 1], vertical_alignment="center")
    with col_dir:
        st.markdown(
            f"<span class='doc-name' title='{data_dir}'>📁 {data_dir}</span>",
            unsafe_allow_html=True,
        )
    with col_open:
        if st.button("📂", help="在文件管理器中打开文档目录", use_container_width=True):
            opened, detail = open_in_file_manager(data_dir)
            if opened:
                st.toast("已在文件管理器中打开", icon="📂")
            else:
                st.warning(f"无法打开文件管理器（{detail}）\n\n文档目录：`{data_dir}`")

    uploaded_files = st.file_uploader(
        "上传文档",
        type=["txt", "pdf", "docx", "md"],  # 支持的文件类型
        accept_multiple_files=True,  # 允许一次传多个
        key=f"uploader_{st.session_state.uploader_reset}",
    )

    # 保存文件到文档目录（已存在的静默跳过，避免每次 rerun 重复提示）
    for file in (uploaded_files or []):
        save_path = resolve_data_dir() / file.name
        if not save_path.exists():
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
            st.toast(f"已上传: {file.name}")

    st.divider()
    # 重新加载（增量索引）
    if st.button("🔄 重新索引文档库"):
        with st.spinner("正在处理文档..."):
            stats = incremental_index()
            msg = (f"文档库已更新：新增 {len(stats['added'])} 个，"
                   f"变更 {len(stats['updated'])} 个，删除 {len(stats['removed'])} 个，"
                   f"跳过 {stats['skipped']} 个")
            st.success(msg)
            if stats["failed"]:
                names = "、".join(Path(p).name for p in stats["failed"])
                st.warning(f"以下 {len(stats['failed'])} 个文档解析失败，已跳过：{names}")

    st.divider()
    # 文档库列表：展示已入库的文档及其索引状态，支持删除（默认折叠，点击展开）
    with st.expander("📚 文档库"):
        doc_list = get_document_list()
        if doc_list:
            for doc in doc_list:
                col_name, col_view, col_del = st.columns([3, 1, 1], vertical_alignment="center")
                with col_name:
                    status = f"已索引 · {doc['chunks']} 段" if doc["indexed"] else "未索引"
                    st.markdown(f"<span class='doc-name'>📄 {doc['name']}</span>", unsafe_allow_html=True)
                    st.caption(status)
                with col_view:
                    if st.button("👁️", key=f"view_{doc['name']}", help="浏览原文"):
                        content = read_document(doc["path"])
                        show_document(doc["name"], content)
                with col_del:
                    with st.popover("🗑️", key=f"pop_{doc['name']}"):
                        st.write(f"删除「{doc['name']}」？")
                        if st.button("确认删除", key=f"confirm_del_{doc['name']}"):
                            delete_document(doc["path"])
                            # 清空上传器，避免被删文件的上传残留状态重新写回
                            st.session_state.uploader_reset += 1
                            st.rerun()
        else:
            st.caption("暂无文档，可上传或运行生成脚本")

# 初始化会话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 已提示过的工具调用集合，用于避免同一个工具调用在流式输出里被重复提醒
if "shown_tool_calls" not in st.session_state:
    st.session_state.shown_tool_calls = set()

# 每轮检索到的来源，用于让 agent 在多轮对话中记住之前引用过的文档
if "retrieved_sources" not in st.session_state:
    st.session_state.retrieved_sources = []

# 显示历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
user_input = st.chat_input("请输入你的问题...")

if user_input:
    # 定义历史消息
    history = []
    assistant_count = 0
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            content = msg["content"]
            # 把该轮检索到的来源附加到助手消息末尾，让 agent 多轮时记住引用了哪些文档
            if assistant_count < len(st.session_state.retrieved_sources):
                sources = st.session_state.retrieved_sources[assistant_count]
                if sources:
                    content += f"\n\n[本轮检索来源: {sources}]"
            assistant_count += 1
            history.append(AIMessage(content=content))

    # 旧消息的总结
    history = summarize_history(history)

    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    prompt_text, retrieved_info = new_prompt(user_input, history, k=st.session_state.get("k_override"))

    # 保存本轮检索来源（供下一轮 agent 理解指代）
    if retrieved_info:
        sources_text = "；".join(
            f"{info['source']}（{info['preview'][:30]}…）" for info in retrieved_info
        )
    else:
        sources_text = ""
    st.session_state.retrieved_sources.append(sources_text)

    current_message = HumanMessage(prompt_text)

    # 侧边栏：展示思考过程
    with st.sidebar:
        st.divider()
        st.subheader("💭 思考过程")
        if retrieved_info:
            for info in retrieved_info:
                st.caption(f"📄 来源: {info['source']}")
                st.text(info['preview'])
        else:
            st.caption("未检索文档库（直接回答）")

    # AI 回复（流式输出）
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        with st.spinner("思考中..."):
            try:
                for chunk in agent.stream(
                    {"messages": history + [current_message]},
                    stream_mode="messages",
                    context={"mode": "normal"},
                ):
                    if isinstance(chunk, tuple):
                        msg, metadata = chunk
                    else:
                        msg = chunk
                        metadata = {}

                    # 检测工具调用（流式输出时 tool_calls 会被拆成多个 chunk，
                    # 需按 tool_call 的 id 去重，且等名称就绪后再提示，避免重复或空提醒）
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                            tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                            # 名称还没流到（部分 chunk 里 name 为空），跳过避免显示空提醒
                            if not tc_name:
                                continue
                            # 同一个工具调用只提示一次
                            if tc_id not in st.session_state.shown_tool_calls:
                                st.session_state.shown_tool_calls.add(tc_id)
                                with st.sidebar:
                                    st.info(f"🔧 正在调用工具: {tc_name}")

                    # 只取 AI 消息的内容片段
                    if isinstance(msg, AIMessageChunk) and msg.content:
                        full_response += msg.content
                        placeholder.markdown(full_response + "▌")

            except Exception as e:
                # 如果 stream 失败，回退到 invoke
                prompt_text, _ = new_prompt(user_input)
                response = agent.invoke({
                    "messages": [HumanMessage(prompt_text)],
                })
                full_response = response["messages"][-1].content

        # 最终显示完整回复
        if full_response:
            placeholder.markdown(full_response)
        else:
            placeholder.markdown("抱歉，我没能生成回答，请重试。")

    # 保存到历史
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
