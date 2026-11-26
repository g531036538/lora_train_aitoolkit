"""三个目标模型的默认 styler，集中一处映射，避免 pipeline/界面各写一份。"""

from __future__ import annotations

from loraforge.caption.base import CaptionStyler
from loraforge.models import TargetModel

from .krea2 import Krea2ProseStyler
from .moss import MossPreciseStyler
from .sdxl import SdxlTagsStyler

STYLERS: dict[str, type[CaptionStyler]] = {
    TargetModel.SDXL.value: SdxlTagsStyler,
    TargetModel.KREA2.value: Krea2ProseStyler,
    TargetModel.MOSS.value: MossPreciseStyler,
}

_ALIASES: dict[str, str] = {
    "sdxl": TargetModel.SDXL.value,
    "krea2": TargetModel.KREA2.value,
    "krea-2": TargetModel.KREA2.value,
    "moss": TargetModel.MOSS.value,
    "moss_image": TargetModel.MOSS.value,
    "moss-image": TargetModel.MOSS.value,
}


def default_styler(target: str) -> CaptionStyler:
    """按目标模型取默认 styler 实例；未知目标抛 KeyError。"""
    key = getattr(target, "value", target)
    key = str(key).strip().lower()
    key = _ALIASES.get(key, key)
    try:
        cls = STYLERS[key]
    except KeyError:
        raise KeyError(f"没有为 {target!r} 配置默认 styler；可选：{', '.join(sorted(STYLERS))}") from None
    return cls()


__all__ = [
    "STYLERS",
    "default_styler",
    "Krea2ProseStyler",
    "MossPreciseStyler",
    "SdxlTagsStyler",
]
