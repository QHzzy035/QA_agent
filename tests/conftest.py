"""
文件名：conftest.py
介绍：公共测试夹具。

本测试套件不依赖真实 API Key、不访问网络、不读写真实向量库：
- 模型实例：不触发（相关模块已做惰性创建）
- 向量库：用 FakeChroma 内存实现替换
- 文档加载：用 FakeLoader 替换 UnstructuredFileLoader（顺带避免引入 torch）
因此 CI 上可以直接跑，也不需要配置任何 secrets。
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

# 保证可以直接 import 项目根目录下的模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeChroma:
    """内存版向量库，只实现 incremental_index / document_service 用到的最小接口。"""

    def __init__(self):
        self.docs: list[dict] = []
        self.reset_count = 0
        self.delete_calls: list[dict | None] = []

    # --- 写 ---
    def reset_collection(self):
        self.reset_count += 1
        self.docs.clear()

    def add_documents(self, chunks):
        for c in chunks:
            self.docs.append({
                "content": c.page_content,
                "source": c.metadata.get("source"),
            })

    def delete(self, where=None):
        self.delete_calls.append(where)
        if where and "source" in where:
            self.docs = [d for d in self.docs if d["source"] != where["source"]]

    # --- 读 ---
    def get(self, include=None):
        return {"metadatas": [{"source": d["source"]} for d in self.docs]}

    # --- 断言辅助 ---
    def docs_of(self, path: str) -> list[dict]:
        return [d for d in self.docs if d["source"] == path]


class FakeSplitter:
    """按行切分，便于精确断言 chunk 数量。"""

    def split_documents(self, documents):
        out = []
        for doc in documents:
            for line in doc.page_content.splitlines():
                line = line.strip()
                if line:
                    out.append(Document(page_content=line, metadata=dict(doc.metadata)))
        return out


@pytest.fixture
def fake_chroma():
    return FakeChroma()


@pytest.fixture
def fake_splitter():
    return FakeSplitter()


@pytest.fixture
def fake_loader(monkeypatch):
    """替换 rag.indexer 里用到的文档抽取函数。

    返回一个安装函数，可指定哪些文件模拟「解析失败」或「解析为空」，
    用于覆盖错误处理分支。默认实现保持与真实抽取一致的行为：
    metadata["source"] 严格等于传入的路径。
    """

    def _install(fail_on=(), empty=()):
        fail_on, empty = set(fail_on), set(empty)

        def _extract(path):
            name = Path(path).name
            if name in fail_on:
                raise ValueError(f"模拟解析失败: {name}")
            if name in empty:
                return []
            return [Document(
                page_content=Path(path).read_text(encoding="utf-8"),
                metadata={"source": path},
            )]

        monkeypatch.setattr("rag.indexer.extract_documents", _extract)
        return _extract

    return _install


@pytest.fixture
def index_env(tmp_path, fake_chroma, fake_splitter):
    """一套隔离的增量索引环境（临时数据目录 + 临时记录文件 + 假向量库）。"""
    data = tmp_path / "data"
    data.mkdir()
    record = tmp_path / "index_record.json"

    def run():
        from rag.indexer import incremental_index
        return incremental_index(
            chroma=fake_chroma,
            splitter=fake_splitter,
            index_file=record,
            data_dir=data,
        )

    def write(name: str, text: str) -> str:
        """在数据目录写一个文件，返回其 normpath 绝对路径（与索引记录 key 一致）。"""
        import os
        p = data / name
        p.write_text(text, encoding="utf-8")
        return os.path.normpath(str(p.resolve()))

    return SimpleNamespace(
        data=data, record=record, chroma=fake_chroma, run=run, write=write
    )
