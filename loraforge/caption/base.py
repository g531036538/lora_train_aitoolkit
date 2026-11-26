"""打标契约：后端与 styler 的接口定义。

这个文件由主管冻结。实现（后端、styler、pipeline、checks）放在旁边，不要改这里。
要改接口 → 开 Issue 说明理由，主管定。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from loraforge.models import DatasetKind, TargetModel


@dataclass(frozen=True)
class CaptionContext:
    """打标时的上下文。所有后端都读它决定「怎么描述这张图」。"""

    target: TargetModel
    kind: DatasetKind
    trigger: str = ""  # 空字符串表示不注入
    language: Literal["en", "zh"] = "en"
    extra: dict | None = None  # 后端专属参数，不要往里塞跨后端的东西


@dataclass(frozen=True)
class BackendCaps:
    """后端能力声明。界面据此决定哪些选项可点。"""

    needs_gpu: bool
    supports_tags: bool  # 能直接产 Danbooru 式标签
    supports_prose: bool  # 能产自然语言句子
    languages: tuple[str, ...]


class CaptionBackend(Protocol):
    """图 → 原始描述。只负责「看懂图」，不负责套模板。"""

    name: str

    def caps(self) -> BackendCaps: ...

    def caption(self, image: Path, ctx: CaptionContext) -> str:
        """失败必须抛异常，不许返回空串——空串会被下游当成「写完了」。"""
        ...


class CaptionStyler(Protocol):
    """原始描述 → 目标模型的最终打标词。纯函数，不读文件、不碰网络。"""

    name: str

    def render(self, raw: str, ctx: CaptionContext, tags: list[str] | None = None) -> str: ...


__all__ = [
    "BackendCaps",
    "CaptionBackend",
    "CaptionContext",
    "CaptionStyler",
]
