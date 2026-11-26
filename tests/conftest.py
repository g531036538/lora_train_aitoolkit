"""测试夹具。

样本图集不要提交进仓库，运行时临时生成。

默认的 pytest `tmp_path` 在本机不可用（系统临时目录落在沙箱可写范围外，
会报 `could not create numbered dir with prefix pytest-`），所以统一改用
仓库内的 `.scratch/`：可写、可控、进 .gitignore。
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

SCRATCH_ROOT = Path(__file__).resolve().parent.parent / ".scratch"


@pytest.fixture
def scratch() -> Path:
    """一个空目录，用完删除。替代 tmp_path。"""
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="t-", dir=SCRATCH_ROOT))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def images(scratch: Path) -> list[Path]:
    """三张假图片文件（只有文件名，不是真图）。"""
    paths = []
    for name in ("frame_01.png", "frame_02.png", "frame_03.png"):
        p = scratch / name
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
        paths.append(p)
    return paths
