"""确定性假后端。

界面测试和打标流水线测试共用这一份，避免各自复制一份后互相漂移。
它不读图，只按文件名派生输出，所以断言是稳定的。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 只在类型检查时需要，运行时不必依赖 caption 包
    from loraforge.caption.base import BackendCaps, CaptionContext


class FakeBackend:
    """满足 CaptionBackend 协议的最小实现。

    seed_text: 每次输出的固定前缀，用来验证 styler 有没有把后端文本吃掉。
    fail_on:   命中的文件名（不含扩展名）会抛 RuntimeError，用来测「单张失败不中断整批」。
    """

    def __init__(
        self,
        *,
        seed_text: str = "a photo of a person",
        fail_on: set[str] | None = None,
    ) -> None:
        self.name = "fake"
        self.seed_text = seed_text
        self.fail_on = set(fail_on or ())

    def caps(self) -> BackendCaps:
        # 延迟 import：caption 包还没落地时，import 这个模块不应该失败
        from loraforge.caption.base import BackendCaps

        return BackendCaps(
            needs_gpu=False,
            supports_tags=True,
            supports_prose=True,
            languages=("en",),
        )

    def caption(self, image: Path, ctx: CaptionContext) -> str:
        stem = Path(image).stem
        if stem in self.fail_on:
            raise RuntimeError(f"fake backend 故意失败：{stem}")
        return f"{self.seed_text}, subject {stem}"

    def __repr__(self) -> str:  # 便于 pytest 失败时看清用了什么
        return f"FakeBackend(seed_text={self.seed_text!r}, fail_on={sorted(self.fail_on)!r})"
