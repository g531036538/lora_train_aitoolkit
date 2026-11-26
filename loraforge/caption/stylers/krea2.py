"""Krea 2 打标：一句话自然语言，只写「这次变化的」。

身份交给触发词。脸型、发型/发长/发色、眼睛颜色、肤色一律不进打标词，
否则换装时模型会把身份和衣服绑在一起，脸会跟着漂。
"""

from __future__ import annotations

from loraforge.caption.base import CaptionContext

from ._text import (
    classify_clause,
    extract_subject,
    has_person,
    join_sentence,
    remove_trigger,
    split_clauses,
    strip_identity_clauses,
)

# 模板顺序：单人 → 取景 → 角度 → 表情 → 头发（只留状态）→ 衣服 → 光 → 背景 → 质感。
_SLOT_ORDER: tuple[str, ...] = (
    "framing",
    "angle",
    "expression",
    "hair",
    "clothing",
    "light",
    "background",
    "object",
    "texture",
)

_PERSON_EVIDENCE_SLOTS = frozenset({"expression", "hair", "clothing"})


class Krea2ProseStyler:
    """Krea 2 的默认 styler：``prose``。"""

    name = "krea2-prose"

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
            raise ValueError("Krea2 styler 收到空描述")

        text = remove_trigger(text, ctx.trigger)
        clauses = strip_identity_clauses(split_clauses(text))
        if not clauses:
            raise ValueError(
                "Krea2 styler 过滤身份特征后没有可写内容；"
                "请让后端描述取景、表情、衣服、光或背景，而不是只描述长相"
            )

        subject, rest = extract_subject(clauses)
        slots: dict[str, list[str]] = {key: [] for key in _SLOT_ORDER}
        for clause in rest:
            slots[classify_clause(clause)].append(clause)

        parts: list[str] = []
        if subject:
            parts.append(subject)
        elif any(slots[key] for key in _PERSON_EVIDENCE_SLOTS) or any(
            has_person(clause) for clause in rest
        ):
            # 有表情/衣服/头发状态，说明画面里有人；模板补「单人」但不编造长相。
            parts.append("a single person")

        for key in _SLOT_ORDER:
            parts.extend(slots[key])

        body = join_sentence(parts)
        if not body:
            raise ValueError("Krea2 styler 没有拼出可用的句子")

        trigger = ctx.trigger.strip()
        if trigger:
            body = body[0].lower() + body[1:] if body else body
            return f"{trigger}, {body}"
        return body


__all__ = ["Krea2ProseStyler"]
