"""
文件名：streamlit_file.py
介绍：虚拟知识库的 Streamlit 页面。

左栏：来源、统一文档列表、上传、同步、设置；
右栏：对话 / 预览切换。

页面只做展示与交互编排，检索、同步、状态推导都在 rag/ 与 tools/ 中。
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

from agent import get_agent
from rag.connected_prompts import new_prompt
from rag.retriever import retriever_for_library
from rag.scanner import ScanConfig, preview_scan
from rag.sync import (
    STATE_DISABLED,
    STATE_EXCLUDED,
    STATE_INDEXED,
    STATE_LABELS,
    STATE_MISSING,
    STATE_OFFLINE,
    STATE_STALE,
    STATE_UNINDEXED,
    STATE_UNKNOWN,
    collect_documents,
    exclude_document,
    restore_document,
    sync_library,
)
from tools.app_paths import resource_dir
from tools.config_loader import LLM_conf, rag_conf
from tools.document_service import (
    SUPPORTED_EXTENSIONS,
    open_in_file_manager,
    read_document,
)
from tools.library import (
    DuplicateSource,
    LibraryError,
    LibraryRegistry,
    SourceOverlap,
    bootstrap_default_library,
)
from tools.trash import move_to_trash

st.set_page_config(page_title="QA Agent", page_icon="📚", layout="wide")

REQUIRED_KEYS = ("DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY", "TAVILY_API_KEY")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

FILTER_STATES = {
    "全部": None,
    "已索引": {STATE_INDEXED, STATE_STALE},
    "未索引": {STATE_UNINDEXED},
    "已移出": {STATE_EXCLUDED},
    "文件缺失": {STATE_MISSING, STATE_UNKNOWN},
    "离线/禁用": {STATE_OFFLINE, STATE_DISABLED},
}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _key_id(value: str) -> str:
    return hashlib.md5(str(value).encode("utf-8")).hexdigest()[:10]


def _human_size(size: int | None) -> str:
    if size is None:
        return "-"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem} ({index}){path.suffix}")
        if not candidate.exists():
            return candidate
    raise OSError(f"无法为重名文件生成新名称：{path.name}")


def set_flash(message: str, icon: str = "✅") -> None:
    st.session_state["_flash"] = (message, icon)


def show_flash() -> None:
    item = st.session_state.pop("_flash", None)
    if item:
        st.toast(item[0], icon=item[1])


def bump_refresh() -> None:
    st.session_state["refresh_token"] = st.session_state.get("refresh_token", 0) + 1


def apply_pending_view() -> None:
    pending = st.session_state.pop("_pending_view", None)
    if pending:
        st.session_state.view_mode = pending


def goto_preview(doc_key: str) -> None:
    st.session_state.preview_doc_key = doc_key
    st.session_state["_pending_view"] = "预览"
    st.rerun()


# ---------------------------------------------------------------------------
# 知识库与数据读取
# ---------------------------------------------------------------------------

@st.cache_resource
def get_registry() -> LibraryRegistry:
    return LibraryRegistry()


def ensure_library(registry: LibraryRegistry):
    candidates = [
        resource_dir() / "test_data",
        resource_dir() / "demo_data",
    ]
    if "library_id" not in st.session_state:
        library, _ = bootstrap_default_library(registry, candidates)
        st.session_state.library_id = library.id
    st.session_state.setdefault("refresh_token", 0)

    try:
        return registry.get_library(st.session_state.library_id)
    except LibraryError:
        library, _ = bootstrap_default_library(registry, candidates)
        st.session_state.library_id = library.id
        return registry.get_library(library.id)


@st.cache_data(show_spinner=False, ttl=30)
def load_documents(library_id: str, refresh_token: int):
    registry = LibraryRegistry()
    library = registry.get_library(library_id)
    return [doc.to_dict() for doc in collect_documents(library)]


@st.cache_data(show_spinner=False, max_entries=20)
def load_preview(path: str, mtime: float):
    return read_document(path)


# ---------------------------------------------------------------------------
# 设置与历史
# ---------------------------------------------------------------------------

def summarize_history(history: list, summarize_length: int | None = None):
    if summarize_length is None:
        summarize_length = st.session_state.get(
            "summarize_length_override",
            LLM_conf.get("history_summarize_length", 10),
        )
    if len(history) <= summarize_length:
        return history

    from model.factory import summarize_model

    old_history = history[:-summarize_length]
    recent_history = history[-summarize_length:]

    old_msg = ""
    for msg in old_history:
        if isinstance(msg, HumanMessage):
            old_msg += f"用户：{msg.content}\n"
        else:
            old_msg += f"智能助手：{msg.content}\n"

    summary = summarize_model.invoke([
        SystemMessage("对以下对话进行简短的总结"),
        HumanMessage(old_msg),
    ])
    return [summary] + recent_history


@st.dialog("设置")
def settings_dialog():
    current_k = st.session_state.get("k_override", rag_conf["k"])
    current_summarize = st.session_state.get(
        "summarize_length_override",
        LLM_conf.get("history_summarize_length", 10),
    )

    new_k = st.slider("每次检索参考片段数", 1, 10, current_k)
    new_summarize = st.slider("历史多少条后压缩", 2, 50, current_summarize)

    st.caption("API 状态")
    for key in REQUIRED_KEYS:
        st.caption(f"{'✅' if os.getenv(key) else '❌'} {key}")

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


# ---------------------------------------------------------------------------
# 文件夹选择与添加来源
# ---------------------------------------------------------------------------

def pick_folder() -> str | None:
    """在运行 Streamlit 的机器上弹出原生目录选择框。"""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        st.warning(f"当前环境无法打开系统文件对话框：{e}")
        return None

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            return filedialog.askdirectory(title="选择知识库文件夹") or None
        finally:
            root.destroy()
    except Exception as e:
        st.warning(f"打开目录选择框失败：{e}")
        return None


@st.dialog("添加文件夹")
def add_source_dialog(library_id: str):
    pending = st.session_state.get("pending_source_path", "")
    path = st.text_input("文件夹路径", value=pending)

    if st.button("浏览..."):
        picked = pick_folder()
        if picked:
            st.session_state.pending_source_path = picked
            path = picked
            st.caption(f"已选择：{picked}")

    recursive = st.checkbox("包含子文件夹", value=True)

    summary = None
    raw = (path or "").strip()
    if raw:
        try:
            summary = preview_scan(raw, ScanConfig(recursive=recursive))
        except Exception as e:
            st.error(f"无法扫描：{e}")

    if summary:
        st.caption(
            f"{summary['files']} 个文件 · 预计 {summary['estimated_chunks']} 个片段 · "
            f"{_human_size(summary['total_bytes'])}"
        )
        if summary["skipped_unsupported"]:
            st.caption(f"跳过 {summary['skipped_unsupported']} 个不支持的文件")
        if summary["skipped_too_large"]:
            st.caption(f"跳过 {summary['skipped_too_large']} 个超大文件")
        if summary["truncated"]:
            st.error("文件数超过上限，请缩小范围或提高上限")

    disabled = not raw or summary is None or summary["truncated"]
    if st.button("添加来源", type="primary", disabled=disabled):
        registry = get_registry()
        try:
            registry.add_source(library_id, raw, recursive=recursive)
        except (SourceOverlap, DuplicateSource, LibraryError) as e:
            st.error(f"添加失败：{e}")
        else:
            st.session_state.pop("pending_source_path", None)
            set_flash("已添加来源；点“同步文档库”开始索引", "📁")
            bump_refresh()
            st.rerun()


# ---------------------------------------------------------------------------
# 同步与删除
# ---------------------------------------------------------------------------

@st.dialog("同步文档库")
def sync_dialog(library):
    rows = []
    for source in library.sources:
        if not source.enabled or not source.path.is_dir():
            continue
        try:
            summary = preview_scan(
                source.path,
                ScanConfig(recursive=source.recursive),
            )
        except Exception as e:
            st.error(f"扫描「{source.label}」失败：{e}")
            return
        rows.append((source, summary))

    if not rows:
        st.warning("没有在线且启用的来源，请先添加文件夹")
        return

    truncated = False
    for source, summary in rows:
        st.markdown(f"**{source.label}**")
        st.caption(
            f"{summary['files']} 个文件 · 预计 {summary['estimated_chunks']} 个片段 · "
            f"{_human_size(summary['total_bytes'])}"
        )
        if summary["skipped_too_large"]:
            st.caption(f"跳过 {summary['skipped_too_large']} 个超大文件")
        if summary["truncated"]:
            st.error(f"「{source.label}」文件数超过上限")
            truncated = True

    if st.button("开始同步", type="primary", disabled=truncated):
        with st.spinner("正在扫描、抽取并建立索引..."):
            try:
                stats = sync_library(library)
            except Exception as e:
                st.error(f"同步失败：{type(e).__name__}: {e}")
                return

        message = (
            f"同步完成：新增 {len(stats.added)}，变更 {len(stats.updated)}，"
            f"移除 {len(stats.removed)}，跳过 {stats.skipped}"
        )
        if stats.failed:
            message += f"，失败 {len(stats.failed)}"
        if stats.offline_sources:
            message += f"，离线来源 {len(stats.offline_sources)}"
        if stats.rebuild_reason:
            message += "（已全量重建）"
        set_flash(message, "✅" if not stats.failed else "⚠️")
        bump_refresh()
        st.rerun()


def _target_for(library, doc: dict) -> dict:
    path = None
    try:
        path = library.resolve_doc_path(
            doc["source_id"], Path(doc["rel_path"])
        )
    except Exception:
        path = None
    return {
        "doc_key": doc["doc_key"],
        "source_id": doc["source_id"],
        "rel_path": doc["rel_path"],
        "name": doc["name"],
        "exists": bool(doc["exists"]),
        "path": str(path) if path else "",
    }


@st.dialog("确认删除")
def delete_dialog(library, target: dict, preselect: bool = False):
    st.write(f"文档：**{target['name']}**")
    st.caption(target["rel_path"])

    delete_file = st.checkbox(
        "同时删除磁盘文件（移到回收站）",
        value=preselect,
        disabled=not target["exists"],
    )
    if delete_file:
        st.warning(f"文件将从磁盘移到回收站：{target['path']}")

    col_cancel, col_confirm = st.columns(2)
    with col_cancel:
        if st.button("取消", use_container_width=True):
            st.rerun()
    with col_confirm:
        if st.button("确认", type="primary", use_container_width=True):
            if delete_file:
                ok, message = move_to_trash(target["path"])
                if not ok:
                    st.error(message)
                    return
                try:
                    restore_document(
                        library,
                        target["source_id"],
                        Path(target["rel_path"]),
                    )
                except Exception:
                    pass
            else:
                try:
                    exclude_document(
                        library,
                        target["source_id"],
                        Path(target["rel_path"]),
                    )
                except Exception as e:
                    st.error(f"移出失败：{e}")
                    return

            try:
                sync_library(library)
                set_flash(
                    "已删除文件并清理索引"
                    if delete_file
                    else "已移出知识库，文件仍保留在原文件夹",
                    "🗑️",
                )
            except Exception as e:
                set_flash(f"记录已更新，但同步失败：{e}", "⚠️")
            bump_refresh()
            st.rerun()


def _restore_document(library, doc: dict) -> None:
    try:
        restore_document(
            library,
            doc["source_id"],
            Path(doc["rel_path"]),
        )
        sync_library(library)
        set_flash("已恢复并重新索引", "✅")
    except Exception as e:
        set_flash(f"恢复失败：{e}", "⚠️")
    bump_refresh()
    st.rerun()


def _exclude_document(library, doc: dict) -> None:
    try:
        exclude_document(
            library,
            doc["source_id"],
            Path(doc["rel_path"]),
        )
        try:
            sync_library(library)
            set_flash("已移出知识库，文件仍保留在原文件夹", "🗑️")
        except Exception as e:
            set_flash(f"已移出，但向量清理失败：{e}", "⚠️")
    except Exception as e:
        set_flash(f"移出失败：{e}", "⚠️")
    bump_refresh()
    st.rerun()


def _clean_missing(library) -> None:
    try:
        sync_library(library)
        set_flash("已清理缺失记录", "✅")
    except Exception as e:
        set_flash(f"清理失败：{e}", "⚠️")
    bump_refresh()
    st.rerun()


def render_doc_actions(library, doc: dict) -> None:
    ident = _key_id(doc["doc_key"])
    target = _target_for(library, doc)

    with st.popover("⋯", key=f"more_{ident}"):
        if doc["state"] == STATE_EXCLUDED:
            if st.button("恢复", key=f"restore_{ident}"):
                _restore_document(library, doc)
            if target["exists"] and st.button(
                "删除文件（回收站）", key=f"trash_{ident}"
            ):
                delete_dialog(library, target, preselect=True)
            return

        if target["exists"]:
            if st.button("移出知识库", key=f"exclude_{ident}"):
                delete_dialog(library, target, preselect=False)
            if st.button("删除文件（回收站）", key=f"trash_{ident}"):
                delete_dialog(library, target, preselect=True)
        elif doc["state"] in (STATE_OFFLINE, STATE_DISABLED):
            if st.button("移出知识库", key=f"exclude_{ident}"):
                _exclude_document(library, doc)
        else:
            if st.button("清理记录", key=f"clean_{ident}"):
                _clean_missing(library)


# ---------------------------------------------------------------------------
# 左栏
# ---------------------------------------------------------------------------

def render_sources(library, docs: list[dict]) -> None:
    with st.expander("📁 来源", expanded=True):
        if not library.sources:
            st.caption("还没有文件夹来源")

        for source in library.sources:
            count = sum(1 for doc in docs if doc["source_id"] == source.id)
            offline = not source.path.is_dir()
            status = "离线" if offline else ("已启用" if source.enabled else "已禁用")

            col_name, col_toggle, col_more = st.columns([5, 1, 1])
            with col_name:
                st.markdown(f"**{source.label}**")
                st.caption(f"{count} 篇 · {status}")
            with col_toggle:
                enabled = st.toggle(
                    "启用",
                    value=source.enabled,
                    key=f"enable_{source.id}",
                    label_visibility="collapsed",
                )
                if enabled != source.enabled:
                    get_registry().set_source_enabled(
                        library.id, source.id, enabled
                    )
                    bump_refresh()
                    st.rerun()
            with col_more:
                with st.popover("⋯", key=f"src_more_{source.id}"):
                    if st.button("打开文件夹", key=f"open_src_{source.id}"):
                        ok, detail = open_in_file_manager(source.path)
                        if not ok:
                            st.warning(detail)
                    if st.button(
                        "移除来源（不删文件）",
                        key=f"remove_src_{source.id}",
                    ):
                        get_registry().remove_source(library.id, source.id)
                        bump_refresh()
                        st.rerun()

        if st.button("➕ 添加文件夹", use_container_width=True):
            add_source_dialog(library.id)


def render_doc_list(library, docs: list[dict]) -> None:
    with st.expander(f"📚 文档库（{len(docs)}）", expanded=True):
        selected = st.selectbox(
            "筛选",
            list(FILTER_STATES),
            key="doc_filter",
            label_visibility="collapsed",
        )
        states = FILTER_STATES[selected]
        filtered = [
            doc for doc in docs
            if states is None or doc["state"] in states
        ]
        st.caption(f"显示 {len(filtered)} 篇")

        if not filtered:
            st.caption("没有符合条件的文档")

        for doc in filtered:
            ident = _key_id(doc["doc_key"])
            col_name, col_preview, col_more = st.columns([5, 1, 1])
            with col_name:
                st.markdown(f"**{doc['name']}**")
                st.caption(f"{doc['source_label']} · {doc['state_label']}")
            with col_preview:
                if st.button(
                    "👁️",
                    key=f"preview_{ident}",
                    help="预览",
                ):
                    goto_preview(doc["doc_key"])
            with col_more:
                render_doc_actions(library, doc)


def render_upload(library) -> None:
    with st.expander("📤 上传文件", expanded=False):
        sources = [
            source for source in library.sources
            if source.enabled and source.path.is_dir()
        ]
        if not sources:
            st.caption("请先添加一个在线来源")
            return

        with st.form("upload_form", clear_on_submit=True):
            files = st.file_uploader(
                "选择文件",
                type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
                accept_multiple_files=True,
            )
            target_id = st.selectbox(
                "保存到",
                options=[source.id for source in sources],
                format_func=lambda source_id: next(
                    source.label for source in sources
                    if source.id == source_id
                ),
            )
            submitted = st.form_submit_button(
                "上传并同步", use_container_width=True
            )

        if not submitted:
            return
        if not files:
            st.warning("没有选择文件")
            return

        source = next(item for item in sources if item.id == target_id)
        saved = []
        skipped = 0
        notes = []

        for file in files:
            name = Path(file.name).name
            if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                notes.append(f"{name}：不支持的格式")
                continue
            if file.size and file.size > MAX_UPLOAD_BYTES:
                notes.append(f"{name}：超过 50 MB")
                continue

            data = bytes(file.getbuffer())
            dest = source.path / name
            if dest.exists():
                try:
                    same = (
                        hashlib.md5(dest.read_bytes()).hexdigest()
                        == hashlib.md5(data).hexdigest()
                    )
                except OSError:
                    same = False
                if same:
                    skipped += 1
                    continue
                dest = _unique_path(dest)

            try:
                dest.write_bytes(data)
                saved.append(dest.name)
            except OSError as e:
                notes.append(f"{name}：写入失败 {e}")

        if saved:
            try:
                stats = sync_library(library)
                set_flash(
                    f"上传完成 {len(saved)} 个文件；"
                    f"同步新增 {len(stats.added)} 个",
                    "📤",
                )
            except Exception as e:
                set_flash(f"文件已保存，但同步失败：{e}", "⚠️")
        elif skipped:
            set_flash(f"没有新增文件（{skipped} 个内容相同）", "✅")
        else:
            set_flash("没有文件被保存", "⚠️")

        if notes:
            set_flash(f"上传完成，{len(notes)} 个文件未处理", "⚠️")
        bump_refresh()
        st.rerun()


def render_sidebar(library, docs: list[dict]) -> None:
    with st.sidebar:
        missing = [key for key in REQUIRED_KEYS if not os.getenv(key)]
        if missing:
            st.warning(
                "未配置 API Key："
                + "、".join(missing)
                + "。同步和问答会失败，请写入 .env 后重启应用。"
            )

        if st.button("⚙️ 设置", use_container_width=True):
            settings_dialog()

        st.header("知识库")
        st.caption(library.name)

        render_sources(library, docs)
        render_doc_list(library, docs)

        st.divider()
        if st.button("🔄 同步文档库", use_container_width=True):
            sync_dialog(library)

        render_upload(library)

        st.divider()
        st.caption(f"数据目录：{library.base_dir}")


# ---------------------------------------------------------------------------
# 右栏：预览
# ---------------------------------------------------------------------------

def render_preview(library) -> None:
    doc_key = st.session_state.get("preview_doc_key")
    if not doc_key:
        st.info("点击文档旁边的预览按钮，即可开始预览")
        return

    try:
        source_id, rel_path = library.split_doc_key(doc_key)
        source = library.source(source_id)
        path = library.resolve_doc_path(source_id, rel_path)
    except Exception as e:
        st.warning(f"无法打开文档：{e}")
        return

    col_back, col_folder, _ = st.columns([1, 1, 4])
    with col_back:
        if st.button("← 返回对话"):
            st.session_state["_pending_view"] = "对话"
            st.rerun()
    with col_folder:
        if st.button("📂 所在文件夹"):
            ok, detail = open_in_file_manager(path.parent)
            if not ok:
                st.warning(detail)

    st.subheader(rel_path.name)
    st.caption(f"{source.label} · {path}")

    if not path.is_file():
        st.warning("文件不存在，可能已被移动或删除")
        return

    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0
    content = load_preview(str(path), mtime)
    with st.container(height=620):
        st.markdown(content)


# ---------------------------------------------------------------------------
# 右栏：对话
# ---------------------------------------------------------------------------

def _build_history() -> list:
    history = []
    assistant_count = 0
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
            continue

        content = msg["content"]
        if assistant_count < len(st.session_state.retrieved_sources):
            sources = st.session_state.retrieved_sources[assistant_count]
            if sources:
                content += f"\n\n[本轮检索来源: {sources}]"
        assistant_count += 1
        history.append(AIMessage(content=content))
    return history


def render_chat(library) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "shown_tool_calls" not in st.session_state:
        st.session_state.shown_tool_calls = set()
    if "retrieved_sources" not in st.session_state:
        st.session_state.retrieved_sources = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("请输入你的问题...")
    if not user_input:
        return

    history = _build_history()
    try:
        history = summarize_history(history)
    except Exception as e:
        st.warning(f"历史压缩失败，使用完整历史：{e}")

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    turn_id = len(st.session_state.messages)

    try:
        prompt_text, retrieved_info = new_prompt(
            user_input,
            history,
            k=st.session_state.get("k_override"),
            retriever_fn=lambda q, k=None, fetch_k=None: retriever_for_library(
                q, library, k, fetch_k
            ),
        )
    except Exception as e:
        st.warning(f"检索失败，直接回答：{e}")
        prompt_text = user_input
        retrieved_info = []

    if retrieved_info:
        sources_text = "；".join(
            f"{info['source']}（{info['preview'][:30]}…）"
            for info in retrieved_info
        )
    else:
        sources_text = ""
    st.session_state.retrieved_sources.append(sources_text)

    current_message = HumanMessage(prompt_text)

    with st.sidebar:
        st.divider()
        st.subheader("💭 思考过程")
        if retrieved_info:
            for info in retrieved_info:
                st.caption(f"📄 来源: {info['source']}")
                st.text(info["preview"])
        else:
            st.caption("未检索文档库（直接回答）")

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        with st.spinner("思考中..."):
            try:
                agent = get_agent()
                for chunk in agent.stream(
                    {"messages": history + [current_message]},
                    stream_mode="messages",
                    context={"mode": "normal"},
                ):
                    if isinstance(chunk, tuple):
                        msg, _metadata = chunk
                    else:
                        msg = chunk

                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            call_id = (
                                tool_call.get("id")
                                if isinstance(tool_call, dict)
                                else getattr(tool_call, "id", None)
                            )
                            call_name = (
                                tool_call.get("name")
                                if isinstance(tool_call, dict)
                                else getattr(tool_call, "name", None)
                            )
                            if not call_name:
                                continue
                            if call_id not in st.session_state.shown_tool_calls:
                                st.session_state.shown_tool_calls.add(call_id)
                                with st.sidebar:
                                    st.info(f"🔧 正在调用工具: {call_name}")

                    if isinstance(msg, AIMessageChunk) and msg.content:
                        full_response += msg.content
                        placeholder.markdown(full_response + "▌")
            except Exception as e:
                st.warning(f"流式输出失败，回退到普通调用：{e}")
                try:
                    agent = get_agent()
                    response = agent.invoke(
                        {"messages": [HumanMessage(prompt_text)]},
                        context={"mode": "normal"},
                    )
                    full_response = response["messages"][-1].content
                except Exception as e2:
                    full_response = f"抱歉，当前无法调用模型：{e2}"

        if full_response:
            placeholder.markdown(full_response)
        else:
            placeholder.markdown("抱歉，我没能生成回答，请重试。")

    if full_response:
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
        })

    if retrieved_info:
        st.caption("参考来源")
        columns = st.columns(min(3, len(retrieved_info)))
        for index, info in enumerate(retrieved_info[:6]):
            doc_key = info.get("doc_key")
            if not doc_key:
                continue
            with columns[index % len(columns)]:
                if st.button(
                    f"📄 {info['source']}",
                    key=f"src_{turn_id}_{index}",
                    use_container_width=True,
                ):
                    goto_preview(doc_key)


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------

st.title("文档问答助手")
show_flash()

registry = get_registry()
library = ensure_library(registry)
apply_pending_view()

try:
    documents = load_documents(library.id, st.session_state.refresh_token)
except Exception as e:
    st.error(f"读取文档库失败：{e}")
    documents = []

render_sidebar(library, documents)

view_mode = st.radio(
    "视图",
    ["对话", "预览"],
    horizontal=True,
    key="view_mode",
    label_visibility="collapsed",
)

if view_mode == "预览":
    render_preview(library)
else:
    render_chat(library)
