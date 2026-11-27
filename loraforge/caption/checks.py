"""打标质量闸门：给界面用的结构化结果，不是 print。

每个 code 对应一种可操作的问题：
- ``empty``           空打标词（标红，必须重跑）
- ``missing_trigger`` 触发词缺失（标红，身份/风格学不进去）
- ``too_short``       过短 / 标签过少（标黄，信息不够）
- ``identity_removed`` Krea2 已从原文剥掉身份描述（标黄，提醒输入被改）
- ``unquoted_text``   MOSS 图内文字没有加引号（标黄，格式提醒）
- ``token_overflow``   SDXL 打标词超过近似上限（标黄，可能被 CLIP 静默截断）
- ``duplicate``       多图完全一致，或某标签出现在 >80% 的图上（标黄，泄漏风险）

同一张图有多个问题时合并成一条 ``CheckResult``，message 里用中文分号分开；
code 按 empty > missing_trigger > token_overflow > too_short >
identity_removed > unquoted_text > duplicate 取最严重的那个。

``raw`` 是可选的原图描述映射；不传时行为与旧版完全一致。传了才能发现
styler 改过用户的输入（Krea2 剥身份、MOSS 文字没加引号）。
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from loraforge.caption.base import CaptionContext
from loraforge.caption.stylers._text import identity_hits
from loraforge.models import PROFILES, TargetModel

CheckCode = Literal[
    "ok",
    "empty",
    "too_short",
    "missing_trigger",
    "identity_removed",
    "unquoted_text",
    "token_overflow",
    "duplicate",
]
Severity = Literal["info", "warn", "error"]

SDXL_SOFT_TAG_LIMIT = 75
SDXL_SOFT_WORD_LIMIT = 60

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
_ENGLISH_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")

_IDENTITY_LABELS = {
    "face": "脸型",
    "hair": "发型",
    "skin": "肤色",
    "eyes": "眼睛",
    "age": "年龄",
}

_MOSS_TEXT_NOUN = r"(?:text|word|words|lettering|title|header|footer|caption|label|logo|watermark)"
_MOSS_TEXT_INTRO_RE = re.compile(
    rf"(?:\b(?:says?|reads?|written|printed)\b|\b{_MOSS_TEXT_NOUN}\b\s*[:：])",
    re.IGNORECASE,
)
_MOSS_QUOTE_CHARS = '"“”'


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


def _identity_message(raw_caption: str, final_caption: str) -> str | None:
    """Krea2 把身份描述剥掉后，告诉用户剥掉了几项、分别是什么。"""
    hits = identity_hits(raw_caption)
    if not hits:
        return None

    final_lower = final_caption.lower()
    removed = [
        (category, fragment)
        for category, fragment in hits
        if fragment.lower() not in final_lower
    ]
    if not removed:
        return None

    counts = Counter(category for category, _ in removed)
    parts = [
        f"{_IDENTITY_LABELS[category]} {counts[category]}"
        for category in ("face", "hair", "skin", "eyes", "age")
        if counts[category]
    ]
    return (
        f"已移除 {len(removed)} 项身份描述（{'、'.join(parts)}）；"
        "身份交给触发词，换装时脸不会飘"
    )


def _moss_unquoted_lines(raw_caption: str) -> list[int]:
    """MOSS 原文里哪些行的图内文字没有加引号。"""
    bad: list[int] = []
    for line_number, line in enumerate(raw_caption.splitlines(), start=1):
        for match in _MOSS_TEXT_INTRO_RE.finditer(line):
            rest = line[match.end() :].lstrip(" \t:：-—")
            if not rest:
                continue
            if rest[0] in _MOSS_QUOTE_CHARS:
                continue
            bad.append(line_number)
            break
    return bad


def _unquoted_message(line_numbers: list[int]) -> str:
    shown = "、".join(str(number) for number in line_numbers)
    return f"第 {shown} 行图内文字未加引号；MOSS 要求一行一条，并用引号包住"


def _unique_tags(tags: Sequence[str]) -> list[str]:
    """按首次出现顺序去重；重复标签不该重复占用提示词预算。"""
    return list(dict.fromkeys(tags))


def _token_overflow_message(tags: list[str]) -> str | None:
    """SDXL 近似 token 上限提示；本机没有 transformers，不精确计数。"""
    unique = _unique_tags(tags)
    tag_count = len(unique)
    word_count = sum(len(_ENGLISH_WORD_RE.findall(tag)) for tag in unique)
    over_tags = tag_count > SDXL_SOFT_TAG_LIMIT
    # 标签数已经超过词数软上限时，词数几乎全是标签本身；此时以标签数为准，
    # 避免「74 个单字标签」被词数误报（74 个标签仍是 74 个词）。
    over_words = tag_count <= SDXL_SOFT_WORD_LIMIT and word_count > SDXL_SOFT_WORD_LIMIT
    if not over_tags and not over_words:
        return None

    counts: list[str] = []
    if over_tags:
        counts.append(f"标签数 {tag_count}（软上限 {SDXL_SOFT_TAG_LIMIT}）")
    if over_words:
        counts.append(f"英文词数 {word_count}（软上限 {SDXL_SOFT_WORD_LIMIT}）")
    dedupe_note = "，重复标签已按一次计" if len(tags) != tag_count else ""
    return (
        f"{'、'.join(counts)} 超过 SDXL 近似上限，CLIP 可能在 77 token 处静默截断；"
        "请把标签压到 75 个、正文压到 60 词以内，删掉重复的通用标签，"
        "或把固定描述挪进采样 prompt。"
        f"这是近似估算，不是精确 tokenizer 计数{dedupe_note}"
    )


def check_captions(
    pairs: Sequence[tuple[Path, str]],
    ctx: CaptionContext,
    *,
    raw: Mapping[Path, str] | None = None,
) -> list[CheckResult]:
    """每张图恰好一条结果；全部正常也给 ``code="ok", severity="info"``。

    ``raw`` 为原图描述（styler 渲染前的文本）。不传时保持旧行为；
    传了以后，Krea2 的身份剥离和 MOSS 的未加引号文字会各出一条 warn。
    """
    entries = [(Path(image), str(caption or "")) for image, caption in pairs]
    total = len(entries)
    if total == 0:
        return []

    raw_map = (
        {Path(image): str(text or "") for image, text in raw.items()} if raw else {}
    )

    style = _style_of(ctx)
    trigger = ctx.trigger.strip()
    trigger_tag = _normalize(trigger.replace("_", " ").lower())
    normalized = [_normalize(caption) for _, caption in entries]

    tag_counts: Counter[str] = Counter()
    per_image_tags: list[list[str]] = []
    for _, caption in entries:
        tags = _tags(caption) if style == "tags" else []
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
        raw_caption = raw_map.get(image, "")

        if not _has_trigger(text, trigger):
            problems.append(f'缺少触发词 "{trigger}"；身份/风格会学不进去')

        token_note = (
            _token_overflow_message(tags)
            if ctx.target == TargetModel.SDXL
            else None
        )
        if token_note:
            problems.append(token_note)

        too_short_notes: list[str] = []
        if len(text) < 10:
            too_short_notes.append(f"打标词过短（{len(text)} 字符），LoRA 学不到足够信息")
        if style == "tags" and len(tags) < 3:
            too_short_notes.append(
                f"标签数只有 {len(tags)} 个（建议 ≥3），信息密度可能不够"
            )
        problems.extend(too_short_notes)

        identity_note = (
            _identity_message(raw_caption, text)
            if raw_caption and style == "prose"
            else None
        )
        if identity_note:
            problems.append(identity_note)

        unquoted_lines = (
            _moss_unquoted_lines(raw_caption)
            if raw_caption and style == "prose_precise"
            else []
        )
        if unquoted_lines:
            problems.append(_unquoted_message(unquoted_lines))

        has_duplicate = False
        same_names = [
            entries[other][0].name
            for other, other_text in enumerate(normalized)
            if other_text and other_text == text
        ]
        if len(same_names) > 1:
            problems.append(_duplicate_message(image.name, same_names))
            has_duplicate = True

        tag_set = set(tags)
        present = [tag for tag in high_freq if tag in tag_set]
        if present:
            shown = "、".join(f'"{tag}"' for tag in present[:5])
            more = " 等" if len(present) > 5 else ""
            problems.append(
                f"标签 {shown}{more} 出现在 >80% 的图上，LoRA 可能把服装/道具学成角色；"
                "考虑改写进变化词或加入丢弃列表"
            )
            has_duplicate = True

        if not problems:
            tag_note = ""
            if style == "tags":
                unique_count = len(_unique_tags(tags))
                if unique_count == len(tags):
                    tag_note = f"（{unique_count} 个标签）"
                else:
                    tag_note = f"（{unique_count} 个标签，重复标签已按一次计，原 {len(tags)} 个）"
            results.append(
                CheckResult(image=image, code="ok", severity="info", message=f"打标正常{tag_note}")
            )
            continue

        if not _has_trigger(text, trigger):
            code: CheckCode = "missing_trigger"
            severity: Severity = "error"
        elif token_note:
            code = "token_overflow"
            severity = "warn"
        elif too_short_notes:
            code = "too_short"
            severity = "warn"
        elif identity_note:
            code = "identity_removed"
            severity = "warn"
        elif unquoted_lines:
            code = "unquoted_text"
            severity = "warn"
        elif has_duplicate:
            code = "duplicate"
            severity = "warn"
        else:  # 理论上不会到这里；保留兜底，避免返回无 code 的结果
            code = "ok"
            severity = "info"

        results.append(
            CheckResult(image=image, code=code, severity=severity, message="；".join(problems))
        )

    return results


__all__ = [
    "CheckCode",
    "CheckResult",
    "SDXL_SOFT_TAG_LIMIT",
    "SDXL_SOFT_WORD_LIMIT",
    "Severity",
    "check_captions",
]
