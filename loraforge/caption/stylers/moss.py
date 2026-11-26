"""MOSS-Image 打标：四段式精确描述。

固定四段：对象 / 图内文字 / 外观 / 构图。
图内文字逐条用引号、一行一条，并标 role（header / footer / ...）。
需要精确位置时输出 JSON layout：bbox 为 ``[y_min, x_min, y_max, x_max]``，
原点在画布中心，边长必须是 16 的倍数。
"""

from __future__ import annotations

import json
import re
from typing import Any

from loraforge.caption.base import CaptionContext

from ._text import (
    classify_clause,
    extract_quoted_texts,
    extract_subject,
    join_sentence,
    normalize_space,
    remove_trigger,
    split_clauses,
    strip_identity,
    strip_identity_clauses,
)

_LABELS: dict[str, str] = {
    "object": "Object",
    "text": "In-image text",
    "appearance": "Appearance",
    "composition": "Composition",
    "none": "none",
    "missing": "not described",
}

_ROLE_ALIASES: dict[str, str] = {
    "标题": "title",
    "页眉": "header",
    "页脚": "footer",
    "正文": "body",
    "标签": "label",
    "标志": "logo",
    "标识": "logo",
    "水印": "watermark",
    "招牌": "sign",
    "说明": "caption",
    "副标题": "subtitle",
}

_ALLOWED_ROLES: frozenset[str] = frozenset(
    {"header", "footer", "title", "subtitle", "body", "label", "logo", "sign", "watermark", "caption"}
)

_TEXT_CLAUSE_RE = re.compile(
    r"\b(?:text|word|words|lettering|sign|signage|title|header|footer|caption|label|"
    r"logo|watermark|reads?|says?|written|printed)\b",
    re.IGNORECASE,
)


def normalize_role(role: Any) -> str:
    if not isinstance(role, str):
        return "body"
    key = normalize_space(role).lower()
    if key in _ALLOWED_ROLES:
        return key
    return _ROLE_ALIASES.get(key, "body")


def validate_bbox(value: Any) -> list[int]:
    """校验 ``[y_min, x_min, y_max, x_max]``，不合法直接抛 ValueError。

    坐标原点在画布中心，所以可以为负；边长（y_max-y_min、x_max-x_min）必须是 16 的倍数。
    不在这里偷偷交换或取整——精确布局写错比报错更糟。
    """
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"MOSS layout bbox 必须是 4 个数 [y_min,x_min,y_max,x_max]，收到：{value!r}")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in value):
        raise ValueError(f"MOSS layout bbox 必须是整数，收到：{value!r}")

    y_min, x_min, y_max, x_max = value
    if y_min >= y_max or x_min >= x_max:
        raise ValueError(
            "MOSS layout bbox 必须满足 y_min < y_max 且 x_min < x_max，"
            f"收到：{value!r}"
        )
    height = y_max - y_min
    width = x_max - x_min
    if height % 16 or width % 16:
        raise ValueError(
            f"MOSS layout bbox 边长必须是 16 的倍数，收到高 {height}、宽 {width}"
        )
    return [y_min, x_min, y_max, x_max]


def _quote(text: str) -> str:
    return '"' + text.replace('"', r"\"") + '"'


def _maybe_json(text: str) -> Any | None:
    stripped = text.lstrip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _first_str(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _clean_section(text: str) -> str:
    """对对象/外观做身份过滤，再压成一行。"""
    if not text.strip():
        return ""
    clauses = strip_identity_clauses(split_clauses(text))
    return normalize_space(", ".join(clauses))


def _drop_text_clauses(clauses: list[str], has_quotes: bool) -> list[str]:
    if not has_quotes:
        return clauses
    kept = []
    for clause in clauses:
        if _TEXT_CLAUSE_RE.search(clause) and re.search(r'[“"]', clause):
            continue
        kept.append(clause)
    return kept


def _collect_texts(raw: str, data: Any | None) -> tuple[list[dict[str, Any]], bool]:
    """返回 (文字条目, 是否带 bbox)。"""
    texts: list[dict[str, Any]] = []
    has_bbox = False

    if isinstance(data, dict):
        candidates = (
            data.get("texts")
            or data.get("in_image_text")
            or data.get("layout")
            or data.get("text_elements")
            or []
        )
    elif isinstance(data, list):
        candidates = data
    else:
        candidates = []

    if candidates:
        for item in candidates:
            if isinstance(item, str):
                text = normalize_space(item)
                role = "body"
                bbox = None
            elif isinstance(item, dict):
                text = _first_str(item, "text", "content", "value", "word")
                role = normalize_role(item.get("role"))
                bbox = item.get("bbox")
                if bbox is not None:
                    bbox = validate_bbox(bbox)
                    has_bbox = True
            else:
                continue
            if not text:
                continue
            entry: dict[str, Any] = {"text": text, "role": role}
            if bbox is not None:
                entry["bbox"] = bbox
            texts.append(entry)
        return texts, has_bbox

    texts = extract_quoted_texts(raw)
    return texts, False


def _split_sections(
    raw: str,
    ctx: CaptionContext,
    *,
    allow_quoted_text: bool,
) -> tuple[str, list[dict[str, Any]], str, str, bool]:
    text = raw.strip()
    if not text:
        raise ValueError("MOSS styler 收到空描述")

    text = remove_trigger(text, ctx.trigger)
    quoted = extract_quoted_texts(text) if allow_quoted_text else []
    clauses = split_clauses(text)
    clauses = _drop_text_clauses(clauses, has_quotes=bool(quoted))
    clauses = strip_identity_clauses(clauses)

    subject, rest = extract_subject(clauses)
    slots: dict[str, list[str]] = {
        "framing": [],
        "angle": [],
        "expression": [],
        "hair": [],
        "clothing": [],
        "light": [],
        "background": [],
        "texture": [],
        "object": [],
    }
    for clause in rest:
        slots[classify_clause(clause)].append(clause)

    object_bits = ([subject] if subject else []) + slots["object"]
    appearance_bits = (
        slots["expression"] + slots["hair"] + slots["clothing"] + slots["light"] + slots["texture"]
    )
    composition_bits = slots["framing"] + slots["angle"] + slots["background"]

    object_text = normalize_space(", ".join(object_bits))
    appearance_text = normalize_space(", ".join(appearance_bits))
    composition_text = normalize_space(", ".join(composition_bits))
    return object_text, quoted, appearance_text, composition_text, False


def _apply_trigger(object_text: str, trigger: str, missing: str) -> str:
    trigger = trigger.strip()
    if trigger:
        if object_text:
            return f"{trigger}, {object_text}"
        return trigger
    return object_text or missing


def _format_text(
    object_text: str,
    texts: list[dict[str, Any]],
    appearance_text: str,
    composition_text: str,
) -> str:
    lines = [f"{_LABELS['object']}: {object_text}", f"{_LABELS['text']}:"]
    if texts:
        for item in texts:
            lines.append(f"{_quote(item['text'])} (role: {item['role']})")
    else:
        lines[-1] = f"{_LABELS['text']}: {_LABELS['none']}"
    lines.append(f"{_LABELS['appearance']}: {appearance_text}")
    lines.append(f"{_LABELS['composition']}: {composition_text}")
    return "\n".join(lines)


def _from_json(data: Any, ctx: CaptionContext) -> tuple[str, list[dict[str, Any]], str, str, bool]:
    if isinstance(data, list):
        data = {"texts": data}
    if not isinstance(data, dict):
        raise ValueError("MOSS styler 收到 JSON，但不是对象或数组")

    object_text = _first_str(data, "object", "subject", "caption", "description")
    appearance_text = _first_str(data, "appearance", "look", "outfit", "style")
    composition_text = _first_str(data, "composition", "layout_description", "composition_notes")
    texts, has_bbox = _collect_texts("", data)

    object_text = _clean_section(object_text)
    appearance_text = _clean_section(appearance_text)
    composition_text = normalize_space(composition_text)
    return object_text, texts, appearance_text, composition_text, has_bbox


class MossPreciseStyler:
    """MOSS-Image 的默认 styler：``prose_precise``。"""

    name = "moss-prose-precise"

    def render(
        self,
        raw: str,
        ctx: CaptionContext,
        tags: list[str] | None = None,
    ) -> str:
        text = raw.strip()
        if tags:
            text = ", ".join(t.strip() for t in tags if t.strip())
        if not text:
            raise ValueError("MOSS styler 收到空描述")

        data = _maybe_json(text)
        if data is not None:
            object_text, texts, appearance_text, composition_text, has_bbox = _from_json(data, ctx)
        else:
            object_text, texts, appearance_text, composition_text, has_bbox = _split_sections(
                text, ctx, allow_quoted_text=True
            )

        missing = _LABELS["missing"]
        object_text = _apply_trigger(object_text, ctx.trigger, missing)
        appearance_text = appearance_text or missing
        composition_text = composition_text or missing

        if has_bbox:
            payload = {
                "object": object_text,
                "texts": texts,
                "appearance": appearance_text,
                "composition": composition_text,
            }
            return json.dumps(payload, ensure_ascii=False, indent=2)

        return _format_text(
            object_text,
            texts,
            appearance_text,
            composition_text,
        )


__all__ = ["MossPreciseStyler", "normalize_role", "validate_bbox"]
