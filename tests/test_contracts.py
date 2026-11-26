"""共享契约的自检：假后端必须确定性、必须能按文件名失败。

放在这里而不是各自的 test_* 里，是因为界面和打标两边都依赖它，
它一旦漂移，两边的测试会同时变成假绿。
"""

from pathlib import Path

import pytest

from loraforge.testing import FakeBackend


def test_caption_is_deterministic_for_same_file(scratch: Path) -> None:
    backend = FakeBackend()
    img = scratch / "frame_01.png"
    first = backend.caption(img, ctx=None)  # 假后端不读 ctx，允许传 None
    second = backend.caption(img, ctx=None)
    assert first == second
    assert "frame_01" in first


def test_different_files_give_different_captions(scratch: Path) -> None:
    backend = FakeBackend()
    a = backend.caption(scratch / "a.png", ctx=None)
    b = backend.caption(scratch / "b.png", ctx=None)
    assert a != b


def test_fail_on_raises_instead_of_returning_empty(scratch: Path) -> None:
    backend = FakeBackend(fail_on={"broken"})
    with pytest.raises(RuntimeError):
        backend.caption(scratch / "broken.png", ctx=None)


def test_caps_declares_no_gpu() -> None:
    caps = FakeBackend().caps()
    assert caps.needs_gpu is False
    assert "en" in caps.languages
