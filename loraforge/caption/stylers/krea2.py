"""Krea 2 打标：一句话自然语言，只写「这次变化的」。

身份交给触发词。脸型、发型/发长/发色、眼睛颜色、肤色一律不进打标词，
否则换装时模型会把身份和衣服绑在一起，脸会跟着漂。

保留后端给的子句顺序：只删身份子句、去掉完全重复的子句，不重排、不补主语、
不改写代词——后端写「她穿着」就还是「她穿着」，styler 不替它换措辞。
"""

from __future__ import annotations

from loraforge.caption.base import CaptionContext

from ._text import (
    join_sentence,
    remove_trigger,
    split_clauses,
    strip_identity_clauses,
)


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

        body = join_sentence(clauses)
        if not body:
            raise ValueError("Krea2 styler 没有拼出可用的句子")

        trigger = ctx.trigger.strip()
        if trigger:
            body = body[0].lower() + body[1:] if body else body
            return f"{trigger}, {body}"
        return body


__all__ = ["Krea2ProseStyler"]
