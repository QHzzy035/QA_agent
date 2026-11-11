"""
文件名：trash.py
介绍：把文件移到系统回收站。

优先使用 send2trash（跨平台）；未安装时在 Windows 上回退到
Shell 的 SHFileOperationW（FOF_ALLOWUNDO），避免直接永久删除。
其他平台未安装 send2trash 时返回失败，由调用方决定是否提示用户。
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

_FO_DELETE = 3
_FOF_SILENT = 0x0004
_FOF_NOCONFIRMATION = 0x0010
_FOF_ALLOWUNDO = 0x0040
_FOF_NOERRORUI = 0x0400


class _SHFILEOPSTRUCTW(ctypes.Structure):
    _fields_ = [
        ("hwnd", ctypes.c_void_p),
        ("wFunc", ctypes.c_uint),
        ("pFrom", ctypes.c_void_p),
        ("pTo", ctypes.c_void_p),
        ("fFlags", ctypes.c_ushort),
        ("fAnyOperationsAborted", ctypes.c_int),
        ("hNameMappings", ctypes.c_void_p),
        ("lpszProgressTitle", ctypes.c_void_p),
    ]


def _windows_recycle(path: Path) -> None:
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    shell32.SHFileOperationW.argtypes = [ctypes.POINTER(_SHFILEOPSTRUCTW)]
    shell32.SHFileOperationW.restype = ctypes.c_int

    operation = _SHFILEOPSTRUCTW()
    operation.wFunc = _FO_DELETE
    text = str(path.resolve()) + "\0"
    buffer = ctypes.create_unicode_buffer(text, len(text) + 1)
    operation.pFrom = ctypes.cast(buffer, ctypes.c_void_p)
    operation.fFlags = (
        _FOF_ALLOWUNDO | _FOF_NOCONFIRMATION | _FOF_SILENT | _FOF_NOERRORUI
    )

    result = shell32.SHFileOperationW(ctypes.byref(operation))
    if result != 0:
        raise OSError(f"SHFileOperationW 返回错误码 {result}")
    if operation.fAnyOperationsAborted:
        raise OSError("删除操作被取消")
    if path.exists():
        raise OSError("文件仍在原位置")


def move_to_trash(path) -> tuple[bool, str]:
    """把单个文件移到回收站，返回 (是否成功, 说明)。"""
    target = Path(path).expanduser()
    if not target.exists():
        return False, f"文件不存在：{target}"
    if target.is_dir():
        return False, "当前只支持删除文件，不支持目录"

    try:
        from send2trash import send2trash

        send2trash(str(target))
        return True, "已移到回收站"
    except ImportError:
        pass
    except Exception as e:
        return False, f"移到回收站失败：{type(e).__name__}: {e}"

    if sys.platform == "win32":
        try:
            _windows_recycle(target)
            return True, "已移到回收站"
        except Exception as e:
            return False, f"移到回收站失败：{type(e).__name__}: {e}"

    return False, "当前平台未安装 send2trash，无法移到回收站"
