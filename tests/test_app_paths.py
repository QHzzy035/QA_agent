"""
文件名：test_app_paths.py
介绍：路径解析测试。

重点：
- 源码运行与打包运行使用不同的资源/数据目录；
- QA_AGENT_DATA_DIR 覆盖生效；
- 导入模块本身不创建目录。
"""
import sys
from pathlib import Path

import tools.app_paths as ap


class TestResourceDir:
    def test_source_run_uses_project_root(self):
        assert ap.resource_dir() == Path(ap.__file__).resolve().parent.parent

    def test_frozen_run_uses_meipass(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ap, "is_frozen", lambda: True)
        monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

        assert ap.resource_dir() == tmp_path


class TestDataDir:
    def test_override_is_created_and_resolved(self, tmp_path, monkeypatch):
        target = tmp_path / "custom-data"
        monkeypatch.setenv(ap.DATA_DIR_ENV, str(target))

        assert ap.data_dir() == target.resolve()
        assert target.is_dir()

    def test_dev_default_is_project_data_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ap, "resource_dir", lambda: tmp_path)
        monkeypatch.setattr(ap, "is_frozen", lambda: False)
        monkeypatch.delenv(ap.DATA_DIR_ENV, raising=False)

        assert ap.default_data_dir() == tmp_path / "data"

    def test_frozen_windows_uses_localappdata(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ap, "is_frozen", lambda: True)
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

        assert ap.default_data_dir() == tmp_path / ap.APP_NAME

    def test_subdirectories_are_created(self, tmp_path, monkeypatch):
        root = tmp_path / "root"
        monkeypatch.setenv(ap.DATA_DIR_ENV, str(root))

        assert ap.libraries_dir() == (root / "libraries").resolve()
        assert ap.logs_dir() == (root / "logs").resolve()
        assert ap.settings_file() == (root / "settings.json").resolve()
        assert (root / "libraries").is_dir()
        assert (root / "logs").is_dir()
