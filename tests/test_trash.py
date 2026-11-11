"""
文件名：test_trash.py
介绍：回收站辅助函数的基础测试。

不真正删除文件到回收站（避免测试产生系统副作用），只验证
不存在的路径和目录会被明确拒绝。
"""
from tools.trash import move_to_trash


class TestMoveToTrash:
    def test_missing_file_is_rejected(self, tmp_path):
        ok, message = move_to_trash(tmp_path / "missing.txt")

        assert ok is False
        assert "不存在" in message

    def test_directory_is_rejected(self, tmp_path):
        folder = tmp_path / "folder"
        folder.mkdir()

        ok, message = move_to_trash(folder)

        assert ok is False
        assert "目录" in message
