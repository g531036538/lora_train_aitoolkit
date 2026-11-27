"""整批体检：只报「整批才看得出来」的问题，每种 code 最多一条。

逐图问题归 ``checks.check_captions()``；这里的规则是把同一件事从
「86 张图刷 86 条黄字」收成一条：数量写进 ``message``，``files`` 只给样例。

判定依据见 ``docs/ARCHITECTURE.md`` §7.3。
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

from loraforge.caption.base import CaptionContext
from loraforge.caption.checks import Severity
from loraforge.models import PROFILES

DatasetCode = Literal["alpha_channel", "ubiquitous_tag", "small_resolution"]

# 低于训练桶下限会被放大，细节追不回来。
SMALL_SIDE = 512
# 小图集不报同质：5 张以下样本量不够，容易误伤。
MIN_BATCH_FOR_UBIQUITY = 5

_TAG_SPLIT_RE = re.compile(r"[,;\n]+")
_WEIGHT_RE = re.compile(r"\(([^():]+):\s*[0-9.]+\)")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class DatasetIssue:
    code: DatasetCode
    severity: Severity
    message: str
    files: tuple[str, ...]


def _normalize(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def _tags(text: str) -> list[str]:
    """与 ``checks._tags`` 同一套切分：逗号/分号/换行、去权重、下划线转空格。"""
    tags: list[str] = []
    for piece in _TAG_SPLIT_RE.split(text):
        tag = _WEIGHT_RE.sub(r"\1", piece).replace("_", " ").strip().lower()
        if tag:
            tags.append(_normalize(tag))
    return tags


def _style_of(ctx: CaptionContext) -> str:
    profile = PROFILES.get(ctx.target)
    return getattr(profile, "caption_style", "tags")


def _analyze_images(active: list[tuple[Path, str]]) -> tuple[list[str], list[str]]:
    """返回 (带真透明通道的图, 短边 < 512 的图)，都只收文件名。"""
    transparent: list[str] = []
    small: list[str] = []
    for image, _caption in active:
        try:
            with Image.open(image) as handle:
                alpha_min = handle.convert("RGBA").getchannel("A").getextrema()[0]
                width, height = handle.size
        except OSError:
            # 读不了就跳过：本模块不负责报坏文件，别因为一张坏图整批失败。
            continue
        if alpha_min == 0:
            transparent.append(image.name)
        if min(width, height) < SMALL_SIDE:
            small.append(image.name)
    return transparent, small


def analyze_dataset(
    pairs: Sequence[tuple[Path, str]],
    ctx: CaptionContext,
    *,
    threshold: float = 0.8,
    sample: int = 20,
) -> list[DatasetIssue]:
    """整批体检；每种 code 最多一条。空打标词的图不参与任何统计。"""
    entries = [(Path(image), str(caption or "")) for image, caption in pairs]
    active = [(image, caption) for image, caption in entries if _normalize(caption)]
    if not active:
        return []

    total = len(active)
    issues: list[DatasetIssue] = []

    transparent, small = _analyze_images(active)
    if transparent:
        issues.append(
            DatasetIssue(
                code="alpha_channel",
                severity="warn",
                message=(
                    f"{len(transparent)} 张图带真正的透明通道（共 {total} 张有效图）；"
                    "训练读取时透明区可能被当成黑底或白底，建议先合成到纯色底再导出"
                ),
                files=tuple(transparent[:sample]),
            )
        )

    if _style_of(ctx) == "tags" and total >= MIN_BATCH_FOR_UBIQUITY:
        trigger_tag = _normalize(ctx.trigger.replace("_", " ").lower())
        tag_counts: Counter[str] = Counter()
        for _image, caption in active:
            tag_counts.update(set(_tags(caption)))
        hot = [
            (tag, count)
            for tag, count in tag_counts.items()
            if count > threshold * total and tag != trigger_tag
        ]
        if hot:
            hot.sort(key=lambda item: (-item[1], item[0]))
            shown = "、".join(f'"{tag}"（{count} 张）' for tag, count in hot[:10])
            more = f" 等 {len(hot)} 个" if len(hot) > 10 else ""
            hot_tags = {tag for tag, _count in hot}
            affected = [
                image.name
                for image, caption in active
                if hot_tags.intersection(_tags(caption))
            ]
            issues.append(
                DatasetIssue(
                    code="ubiquitous_tag",
                    severity="warn",
                    message=(
                        f"{shown}{more} 出现在超过 {int(threshold * 100)}% 的打标词里"
                        f"（共 {total} 张有效图）；"
                        "如果它们不是角色本体，建议写进采样 prompt 而不是打标词"
                    ),
                    files=tuple(affected[:sample]),
                )
            )

    if small:
        issues.append(
            DatasetIssue(
                code="small_resolution",
                severity="warn",
                message=(
                    f"{len(small)} 张图短边小于 {SMALL_SIDE} px（共 {total} 张有效图）；"
                    "低于训练桶下限会被放大，细节不可追，建议换更高分辨率的原图"
                ),
                files=tuple(small[:sample]),
            )
        )

    return issues
