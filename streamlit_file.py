"""
文件名：streamlit_file.py
介绍：前端部分，使用streamlit进行编写
"""
# 依赖库导入
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessageChunk, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model
# 依赖文件导入
from agent import agent
from tools.config_loader import LLM_conf,BASE_DIR
from rag.connected_prompts import new_prompt
from rag.loader import loader
from rag.splitter import splitter
from rag.retriever import chroma

summarize_model = init_chat_model(model=LLM_conf["chat_model_name"])

# 方法定义：历史消息过多时，从最早的几次历史消息中进行总结
def summarize_history(history: list, summarize_length: int = 6):
    if len(history) <= summarize_length:
        return history

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

st.title("文档问答助手")

# 上传文件
with st.sidebar:
    st.header("文档管理")
    uploaded_files = st.file_uploader(
        "上传文档",
        type=["txt", "pdf", "docx", "md"],  # 支持的文件类型
        accept_multiple_files=True  # 允许一次传多个
    )

    # 保存文件到test_data里
    if uploaded_files:
        for file in uploaded_files:
            save_path = BASE_DIR / "test_data" / file.name
            # 避免重复上传
            if not save_path.exists():
                with open(save_path, "wb") as f:
                    f.write(file.getbuffer())
                st.success(f"已上传: {file.name}")
            else:
                st.warning(f"文件已存在: {file.name}")

    st.divider()
    # 重新加载
    if st.button("🔄 重新索引文档库"):
        with st.spinner("正在处理文档..."):
            # 清空旧数据，重新加载
            documents = loader.load()
            doc_chunks = splitter.split_documents(documents)
            chroma.reset_collection()
            chroma.add_documents(doc_chunks)
            st.success("文档库已更新！")

# 初始化会话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 已提示过的工具调用集合，用于避免同一个工具调用在流式输出里被重复提醒
if "shown_tool_calls" not in st.session_state:
    st.session_state.shown_tool_calls = set()

# 显示历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
user_input = st.chat_input("请输入你的问题...")

if user_input:
    # 定义历史消息
    history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))

    # 旧消息的总结
    history = summarize_history(history)

    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    prompt_text, retrieved_info = new_prompt(user_input)
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
                response = agent.invoke({
                    "messages": [HumanMessage(new_prompt(user_input))],
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
