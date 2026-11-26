"""SDXL 打标：Danbooru 风格的逗号分隔小写标签。

SDXL 的 CLIP-L/G 文本编码器对短标签更熟；触发词放最前。
身份特征在这里不主动剔除——SDXL 侧的约定是「触发词承载身份，其余描述」，
但短标签本身就是描述，Krea2 的严格身份过滤规则不搬到这里。
"""

from __future__ import annotations

import re
from typing import Iterable

from loraforge.caption.base import CaptionContext

_TAG_SPLIT_RE = re.compile(r"[,\n;]+")
_WEIGHT_RE = re.compile(r"[\(\[]\s*([^()\[\]:]+?)\s*:\s*[0-9.]+(?:e[-+]?\d+)?\s*[\)\]]", re.IGNORECASE)
_LEADING_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)
_LEADING_AND_RE = re.compile(r"^(?:and|plus|with)\s+", re.IGNORECASE)

_FILLER_PREFIXES: tuple[str, ...] = (
    "a photo of",
    "an image of",
    "a picture of",
    "photo of",
    "image of",
    "picture of",
    "this is",
    "there is",
)

# 这些词不是 Danbooru 标签，写进 LoRA 会把「画质口号」也学进去。
_DROP_TAGS: frozenset[str] = frozenset(
    {
        "masterpiece",
        "best quality",
        "high quality",
        "high resolution",
        "highres",
        "8k",
        "4k",
        "ultra detailed",
        "extremely detailed",
        "award winning",
        "professional photography",
    }
)

# 常见写法的规范词。只收确定等价的，不做会丢信息的改写。
_CANONICAL: dict[str, str] = {
    "1 girl": "1girl",
    "one girl": "1girl",
    "a girl": "1girl",
    "single girl": "1girl",
    "girl": "1girl",
    "1 boy": "1boy",
    "one boy": "1boy",
    "a boy": "1boy",
    "single boy": "1boy",
    "boy": "1boy",
    "looking at the camera": "looking at viewer",
    "looking at camera": "looking at viewer",
    "facing the camera": "looking at viewer",
    "long haired": "long hair",
    "short haired": "short hair",
    "blue eyed": "blue eyes",
    "green eyed": "green eyes",
    "brown eyed": "brown eyes",
}


def _strip_filler(tag: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in _FILLER_PREFIXES:
            if tag.startswith(prefix + " "):
                tag = tag[len(prefix) + 1 :].strip()
                changed = True
    return tag


def _clean_tag(tag: str) -> str:
    tag = tag.strip()
    if not tag:
        return ""
    tag = _WEIGHT_RE.sub(r"\1", tag)
    tag = tag.lower().replace("_", " ")
    tag = re.sub(r"\s+", " ", tag).strip(" ,;:.!?、，；：。！？-")
    tag = _LEADING_AND_RE.sub("", tag)
    tag = _strip_filler(tag)
    tag = _LEADING_ARTICLE_RE.sub("", tag)
    tag = _strip_filler(tag)
    tag = re.sub(r"\s+", " ", tag).strip(" ,;:.!?、，；：。！？-")
    tag = _CANONICAL.get(tag, tag)
    return tag


def _iter_tags(raw: str, tags: list[str] | None) -> Iterable[str]:
    if tags:
        for item in tags:
            yield item
        return
    for item in _TAG_SPLIT_RE.split(raw):
        yield item


class SdxlTagsStyler:
    """SDXL 的默认 styler：``tags``。"""

    name = "sdxl-tags"

    def render(
        self,
        raw: str,
        ctx: CaptionContext,
        tags: list[str] | None = None,
    ) -> str:
        cleaned: list[str] = []
        seen: set[str] = set()

        for item in _iter_tags(raw, tags):
            tag = _clean_tag(item)
            if not tag or tag in _DROP_TAGS:
                continue
            if tag in seen:
                continue
            seen.add(tag)
            cleaned.append(tag)

        trigger = ctx.trigger.strip()
        if trigger:
            # 触发词最前；正文里重复出现的同词去掉，避免两次身份信号。
            trigger_low = trigger.lower()
            cleaned = [t for t in cleaned if t.lower() != trigger_low]
            cleaned.insert(0, trigger)

        if not cleaned:
            raise ValueError("SDXL styler 没有得到任何可用标签（原始描述为空或全是画质词）")

        return ", ".join(cleaned)


__all__ = ["SdxlTagsStyler"]
