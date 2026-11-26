"""打标质量闸门：给界面用的结构化结果，不是 print。

每个 code 对应一种可操作的问题：
- ``empty``           空打标词（标红，必须重跑）
- ``missing_trigger`` 触发词缺失（标红，身份/风格学不进去）
- ``too_short``       过短 / 标签过少（标黄，信息不够）
- ``duplicate``       多图完全一致，或某标签出现在 >80% 的图上（标黄，泄漏风险）

同一张图有多个问题时合并成一条 ``CheckResult``，message 里用中文分号分开；
code 按 empty > missing_trigger > too_short > duplicate 取最严重的那个。
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from loraforge.caption.base import CaptionContext
from loraforge.models import PROFILES

CheckCode = Literal["ok", "empty", "too_short", "missing_trigger", "duplicate"]
Severity = Literal["info", "warn", "error"]

_STRUCTURAL_TAGS = frozenset(
    {
        "1girl",
        "1boy",
        "1other",
        "2girls",
        "multiple girls",
        "solo",
        "solo focus",
        "no humans",
    }
)

_TAG_SPLIT_RE = re.compile(r"[,;\n]+")
_WEIGHT_RE = re.compile(r"\(([^():]+):\s*[0-9.]+\)")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class CheckResult:
    image: Path
    code: CheckCode
    severity: Severity
    message: str


def _normalize(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def _tags(text: str) -> list[str]:
    tags: list[str] = []
    for piece in _TAG_SPLIT_RE.split(text):
        tag = _WEIGHT_RE.sub(r"\1", piece).replace("_", " ").strip().lower()
        if tag:
            tags.append(_normalize(tag))
    return tags


def _has_trigger(text: str, trigger: str) -> bool:
    trigger = trigger.strip()
    if not trigger:
        return True
    pattern = r"(?<!\w)" + re.escape(trigger) + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _style_of(ctx: CaptionContext) -> str:
    profile = PROFILES.get(ctx.target)
    return getattr(profile, "caption_style", "tags")


def _duplicate_message(current: str, names: list[str]) -> str:
    others = [name for name in names if name != current]
    shown = "、".join(others[:3])
    suffix = " 等" if len(others) > 3 else ""
    return (
        f"与其他 {len(others)} 张图（{shown}{suffix}）打标完全一致；"
        "检查后端是否复制，或这些图是否过于相似"
    )


def check_captions(
    pairs: Sequence[tuple[Path, str]],
    ctx: CaptionContext,
) -> list[CheckResult]:
    """每张图恰好一条结果；全部正常也给 ``code="ok", severity="info"``。"""
    entries = [(Path(image), str(caption or "")) for image, caption in pairs]
    total = len(entries)
    if total == 0:
        return []

    style = _style_of(ctx)
    trigger = ctx.trigger.strip()
    trigger_tag = _normalize(trigger.replace("_", " ").lower())
    normalized = [_normalize(caption) for _, caption in entries]
    counts = Counter(text for text in normalized if text)

    tag_counts: Counter[str] = Counter()
    per_image_tags: list[list[str]] = []
    for _, raw in entries:
        tags = _tags(raw) if style == "tags" else []
        per_image_tags.append(tags)
        tag_counts.update(set(tags))

    high_freq: list[str] = []
    if style == "tags" and total >= 3:
        threshold = total * 0.8
        high_freq = sorted(
            tag
            for tag, count in tag_counts.items()
            if count > threshold and tag not in _STRUCTURAL_TAGS and tag != trigger_tag
        )

    results: list[CheckResult] = []
    for position, (image, caption) in enumerate(entries, start=1):
        text = normalized[position - 1]
        if not text:
            results.append(
                CheckResult(
                    image=image,
                    code="empty",
                    severity="error",
                    message="打标词为空；空打标会静默毁掉训练，必须重新生成",
                )
            )
            continue

        problems: list[str] = []
        tags = per_image_tags[position - 1]

        if not _has_trigger(text, trigger):
            problems.append(f'缺少触发词 "{trigger}"；身份/风格会学不进去')

        too_short = len(text) < 10 or (style == "tags" and len(tags) < 3)
        if too_short:
            problems.append(f"打标词过短（{len(text)} 字符 / {len(tags)} 个标签），LoRA 学不到足够信息")

        same_names = [
            entries[other][0].name
            for other, other_text in enumerate(normalized)
            if other_text and other_text == text
        ]
        if len(same_names) > 1:
            problems.append(_duplicate_message(image.name, same_names))

        tag_set = set(tags)
        present = [tag for tag in high_freq if tag in tag_set]
        if present:
            shown = "、".join(f'"{tag}"' for tag in present[:5])
            more = " 等" if len(present) > 5 else ""
            problems.append(
                f"标签 {shown}{more} 出现在 >80% 的图上，LoRA 可能把服装/道具学成角色；"
                "考虑改写进变化词或加入丢弃列表"
            )

        if not problems:
            tag_note = f"（{len(tags)} 个标签）" if style == "tags" else ""
            results.append(
                CheckResult(image=image, code="ok", severity="info", message=f"打标正常{tag_note}")
            )
            continue

        if not _has_trigger(text, trigger):
            code: CheckCode = "missing_trigger"
            severity: Severity = "error"
        elif too_short:
            code = "too_short"
            severity = "warn"
        else:
            code = "duplicate"
            severity = "warn"

        results.append(
            CheckResult(image=image, code=code, severity=severity, message="；".join(problems))
        )

    return results


__all__ = ["CheckCode", "CheckResult", "Severity", "check_captions"]
