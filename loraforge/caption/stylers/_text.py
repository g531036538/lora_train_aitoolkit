"""Styler 共用的纯文本工具。

这里只做字符串处理：分句、剔身份特征、给句子归槽位。
不读文件、不碰网络、不依赖任何模型。
"""

from __future__ import annotations

import re
from typing import Iterable

# ---------------------------------------------------------------------------
# 基础清洗
# ---------------------------------------------------------------------------

_SPACE_RE = re.compile(r"\s+")
_TRAIL_PUNCT_RE = re.compile(r"[\s,;:.!?、，；：。！？]+$")
_LEAD_PUNCT_RE = re.compile(r"^[\s,;:.!?、，；：。！？]+")

_WORD_L = r"(?<![0-9A-Za-z_])"
_WORD_R = r"(?![0-9A-Za-z_])"


def normalize_space(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def strip_punct(text: str) -> str:
    return _TRAIL_PUNCT_RE.sub("", _LEAD_PUNCT_RE.sub("", text)).strip()


def split_clauses(raw: str) -> list[str]:
    """按逗号/分号/句号/换行分句，引号内的分隔符不切。

    图内文字经常带逗号，例如 ``the sign says "Open, 9-5"``；
    先按引号状态跳过，避免把文字内容切碎。
    """
    clauses: list[str] = []
    buf: list[str] = []
    closing: str | None = None

    for ch in raw:
        if closing is not None:
            buf.append(ch)
            if ch == closing:
                closing = None
            continue

        if ch == '"':
            closing = '"'
            buf.append(ch)
            continue
        if ch == "“":
            closing = "”"
            buf.append(ch)
            continue
        if ch == "”":
            # 孤立的右引号，当普通字符
            buf.append(ch)
            continue

        if ch in ",;.\n":
            part = normalize_space("".join(buf))
            if part:
                clauses.append(part)
            buf = []
            continue

        buf.append(ch)

    part = normalize_space("".join(buf))
    if part:
        clauses.append(part)
    return clauses


def remove_trigger(text: str, trigger: str) -> str:
    """去掉文本里已经出现的触发词（大小写不敏感，按词边界）。"""
    trigger = trigger.strip()
    if not trigger:
        return text
    pat = re.compile(_WORD_L + re.escape(trigger) + _WORD_R, re.IGNORECASE)
    return strip_punct(normalize_space(pat.sub(" ", text)))


# ---------------------------------------------------------------------------
# 身份特征
# ---------------------------------------------------------------------------
# 原则：脸型、发型/发长/发色、眼睛颜色、肤色都属于「身份」，交给触发词。
# 注意只删「身份描述」，不删「头发状态」：wet hair / hair blowing in the wind
# 是这次画面的变化，可以留。

_IDENTITY_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 年龄（同样属于身份，不写进 Krea2/MOSS）
    re.compile(r"\b\d+\s*[- ]?year[- ]old\b", re.IGNORECASE),
    # 发型 / 发长 / 发色（允许 long black hair 这种多形容词）
    re.compile(
        r"\b(?:long|short|medium|shoulder[- ]length|waist[- ]length|straight|curly|"
        r"wavy|coily|kinky|blonde|blond|black|brown|red|ginger|auburn|pink|blue|"
        r"green|purple|silver|white|gray|grey|dyed)"
        r"(?:\s+(?:and\s+)?(?:long|short|medium|shoulder[- ]length|waist[- ]length|"
        r"straight|curly|wavy|coily|kinky|blonde|blond|black|brown|red|ginger|auburn|"
        r"pink|blue|green|purple|silver|white|gray|grey|dyed))*"
        r"\s+(?:hair|bangs|fringe)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:hair|bangs|fringe)\s+(?:color|colour|length|style|texture)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:hairstyle|haircut|bob cut|pixie cut|ponytail|braid|braids|bun|updo|"
        r"dreadlocks|mohawk)\b",
        re.IGNORECASE,
    ),
    # 眼睛颜色
    re.compile(
        r"\b(?:blue|green|brown|hazel|black|red|amber|gray|grey|violet|purple)\s+eyes?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\beye\s+(?:color|colour)\b", re.IGNORECASE),
    # 肤色
    re.compile(
        r"\b(?:pale|fair|light|dark|tan|tanned|olive|brown|white|black|bronze|"
        r"golden|ebony|porcelain)\s+(?:skin|complexion)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:skin\s+(?:tone|color|colour)|complexion)\b", re.IGNORECASE),
    # 脸型 / 面部结构
    re.compile(
        r"\b(?:round|oval|square|heart[- ]shaped|diamond[- ]shaped|triangular|"
        r"long|angular|chiseled|soft|broad|narrow)\s+face\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bface\s+shape\b", re.IGNORECASE),
    re.compile(r"\b(?:high|prominent|sharp|defined)\s+cheekbones?\b", re.IGNORECASE),
    re.compile(r"\b(?:sharp|defined|strong|soft)\s+jawline\b", re.IGNORECASE),
)

_IDENTITY_CATEGORY_BY_INDEX: tuple[str, ...] = (
    "age",
    "hair",
    "hair",
    "hair",
    "eyes",
    "eyes",
    "skin",
    "skin",
    "face",
    "face",
    "face",
    "face",
)

_CONNECTOR = r"(?:with|and|having|featuring|of|has|have|had|got)"
_DANGLING_RE = re.compile(
    rf"(?:\b{_CONNECTOR}\b\s*)+(?:(?:a|an|the)\s*)?$", re.IGNORECASE
)
_LEADING_CONN_RE = re.compile(rf"^(?:{_CONNECTOR}\b\s*)+", re.IGNORECASE)
_REPEATED_CONN_RE = re.compile(rf"(?:\b{_CONNECTOR}\b\s*){{2,}}", re.IGNORECASE)


def strip_identity(clause: str) -> str:
    """删掉一句里的身份描述，返回剩下的部分；删空则返回空串。

    例：``a woman with long black hair`` → ``a woman``
        ``long black hair`` → ````
        ``she has blue eyes`` → ````
    """
    cleaned = clause
    matched = False
    for pat in _IDENTITY_PATTERNS:
        cleaned, count = pat.subn(" ", cleaned)
        if count:
            matched = True
    if not matched:
        return clause

    cleaned = normalize_space(cleaned)
    cleaned = cleaned.strip(" ,;:-")
    cleaned = _DANGLING_RE.sub("", cleaned)
    cleaned = _LEADING_CONN_RE.sub("", cleaned)
    cleaned = _DANGLING_RE.sub("", cleaned)
    cleaned = _REPEATED_CONN_RE.sub(" ", cleaned)
    cleaned = normalize_space(cleaned).strip(" ,;:-")

    # 删完只剩虚词/单字（she、a woman 之外的空壳）就视为没有内容
    words = re.findall(r"[0-9A-Za-z\u4e00-\u9fff]+", cleaned)
    if len(words) < 2:
        return ""
    return cleaned


def strip_identity_clauses(clauses: Iterable[str]) -> list[str]:
    out: list[str] = []
    for clause in clauses:
        cleaned = strip_identity(clause)
        if cleaned:
            out.append(cleaned)
    return out


def identity_hits(text: str) -> list[tuple[str, str]]:
    """返回文本里命中的身份描述 ``(类别, 原文片段)``，按出现位置排序。

    类别是 ``age`` / ``hair`` / ``eyes`` / ``skin`` / ``face``；
    重叠的匹配只保留最长的一条，避免 ``long black hair`` 被数两次。
    """
    matches: list[tuple[int, int, str, str]] = []
    for category, pattern in zip(_IDENTITY_CATEGORY_BY_INDEX, _IDENTITY_PATTERNS):
        for match in pattern.finditer(text):
            fragment = normalize_space(match.group(0))
            if fragment:
                matches.append((match.start(), match.end(), category, fragment))

    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected: list[tuple[int, int, str, str]] = []
    for start, end, category, fragment in matches:
        if any(start < other_end and end > other_start for other_start, other_end, _, _ in selected):
            continue
        selected.append((start, end, category, fragment))

    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for _, _, category, fragment in selected:
        key = (category, fragment.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((category, fragment))
    return out


# ---------------------------------------------------------------------------
# 句子槽位
# ---------------------------------------------------------------------------

_PERSON_RE = re.compile(
    r"\b(?:woman|women|man|men|person|people|girl|girls|boy|boys|child|children|"
    r"guy|lady|gentleman|figure|subject|character|model|she|he|they)\b|"
    r"\b(?:1girl|1boy|2girls|2boys|multiple girls|multiple boys|solo)\b",
    re.IGNORECASE,
)

_BOILERPLATE_RE = re.compile(
    r"^(?:(?:a|an|the)\s+)?(?:photo|photograph|image|picture|painting|illustration|"
    r"render|shot|frame)\s+of\s+",
    re.IGNORECASE,
)
_THIS_IS_RE = re.compile(r"^(?:this is|there is)\s+", re.IGNORECASE)
_AGE_RE = re.compile(r"\b\d+\s*[- ]?year[- ]old\b", re.IGNORECASE)
_TEXT_OBJECT_RE = re.compile(
    r"^(?:(?:a|an|the)\s+)?(?:poster|sign|signage|billboard|banner|book|magazine|"
    r"newspaper|flyer|menu|card|label|package|packaging|screen|display|plaque|"
    r"sticker|mural|graffiti|t-shirt|shirt)\b",
    re.IGNORECASE,
)
_PLACE_PREP_RE = re.compile(r"\b(?:in|on|at|inside|outside)\b", re.IGNORECASE)

_SLOT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "framing",
        re.compile(
            r"\b(?:close[- ]?up|medium shot|wide shot|full[- ]body|upper[- ]body|"
            r"cowboy shot|headshot|portrait|bust shot|long shot|establishing shot)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "angle",
        re.compile(
            r"\b(?:low angle|high angle|from above|from below|eye[- ]level|from behind|"
            r"side view|three[- ]quarter|front view|overhead|dutch angle|tilted angle|"
            r"profile view)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "expression",
        re.compile(
            r"\b(?:smil(?:e|es|ing)|laugh(?:s|ing)?|serious|neutral|frown(?:s|ing)?|"
            r"surprised|angry|sad|pensive|thoughtful|looking (?:at|away|toward)|gaze|"
            r"expression|eyes (?:closed|open)|winking)\b",
            re.IGNORECASE,
        ),
    ),
    ("hair", re.compile(r"\b(?:hair|bangs|fringe)\b", re.IGNORECASE)),
    (
        "clothing",
        re.compile(
            r"\b(?:wearing|dressed|outfit|clothes|clothing|dress|shirt|t-shirt|blouse|"
            r"jacket|coat|suit|uniform|kimono|hoodie|sweater|skirt|pants|trousers|"
            r"jeans|shorts|hat|cap|glasses|sunglasses|scarf|gloves|boots|shoes|"
            r"sneakers|jewelry|necklace|earrings|ring|watch|bag|backpack)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "light",
        re.compile(
            r"\b(?:lighting|lit|backlit|sunlight|sunlit|neon|golden hour|soft light|"
            r"rim light|shadow|shadows|glow|dim|bright|warm light|cool light|"
            r"moonlight|candlelight|flash|overexposed|underexposed)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "background",
        re.compile(
            r"\b(?:background|in a|on a|at a|inside|outside|indoors|outdoors|street|"
            r"room|cafe|restaurant|forest|beach|studio|city|park|bedroom|kitchen|"
            r"office|sky|wall|field|mountain|river|lake|garden|alley|station|shop|store)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "texture",
        re.compile(
            r"\b(?:film grain|grain|35mm|analog|soft focus|bokeh|shallow depth of field|"
            r"matte|glossy|textured|texture|sharp focus|crisp|blurry|motion blur|"
            r"vignette|high contrast)\b",
            re.IGNORECASE,
        ),
    ),
)


def classify_clause(clause: str) -> str:
    if _TEXT_OBJECT_RE.search(clause):
        return "object"
    if _PLACE_PREP_RE.search(clause):
        for slot, pat in _SLOT_PATTERNS:
            if slot == "background" and pat.search(clause):
                return "background"
    for slot, pat in _SLOT_PATTERNS:
        if pat.search(clause):
            return slot
    return "object"


def has_person(text: str) -> bool:
    return bool(_PERSON_RE.search(text))


def _ensure_single(subject: str) -> str:
    """把主语写成「单人」口径；明确多人时不动，避免说谎。"""
    low = subject.lower().strip()
    if re.search(r"\b(?:two|three|four|five|several|many|multiple|couple|group)\b", low):
        return subject
    if " and " in f" {low} " or " with " in f" {low} ":
        return subject
    if re.search(r"\bsingle\b", low):
        return subject

    rewrites = {
        "1girl": "a single young woman",
        "1boy": "a single young man",
        "solo": "a single person",
        "woman": "a single woman",
        "man": "a single man",
        "girl": "a single girl",
        "boy": "a single boy",
        "person": "a single person",
        "lady": "a single woman",
        "guy": "a single man",
    }
    if low in rewrites:
        return rewrites[low]

    m = re.match(r"^(a|an|the)\s+(.+)$", subject.strip(), re.IGNORECASE)
    if m and not re.match(r"^(?:single|one)\b", m.group(2), re.IGNORECASE):
        return f"{m.group(1)} single {m.group(2)}"
    return subject


def extract_subject(clauses: list[str]) -> tuple[str, list[str]]:
    """找主语短语，并把它从原句里拆出来。

    返回 ``(主语, 剩余从句)``。找不到主语时返回空串和原从句。
    """
    for idx, clause in enumerate(clauses):
        probe = _THIS_IS_RE.sub("", _BOILERPLATE_RE.sub("", clause))
        m = _PERSON_RE.search(probe)
        if not m:
            continue

        # 主语只取到人/角色词为止，后面的动作/穿着留给槽位归类；
        # 年龄同属身份信息，不写进 Krea2/MOSS 的打标词。
        subject = normalize_space(probe[: m.end()]).strip(" ,;:-")
        subject = normalize_space(_AGE_RE.sub(" ", subject)).strip(" ,;:-")
        rest = normalize_space(probe[m.end() :]).strip(" ,;:-")
        remaining = clauses[:idx] + ([rest] if rest else []) + clauses[idx + 1 :]
        return _ensure_single(subject), remaining
    return "", clauses


def join_sentence(parts: Iterable[str]) -> str:
    """把槽位内容拼成一句话：逗号分隔，句号收尾。"""
    seen: list[str] = []
    seen_low: set[str] = set()
    for part in parts:
        part = strip_punct(normalize_space(part))
        if not part:
            continue
        low = part.lower()
        if low in seen_low:
            continue
        # 去掉被更完整句子包含的重复主语，例如 "a woman" 与 "a young woman"
        if any(low != other and low in other for other in seen_low):
            continue
        seen = [s for s in seen if not (low != s.lower() and s.lower() in low)]
        seen_low = {s.lower() for s in seen}
        seen.append(part)
        seen_low.add(low)
    if not seen:
        return ""
    sentence = ", ".join(seen)
    if sentence and sentence[0].islower():
        sentence = sentence[0].upper() + sentence[1:]
    return sentence + "."


# ---------------------------------------------------------------------------
# 图内文字
# ---------------------------------------------------------------------------

_QUOTED_RE = re.compile(r'[“"]([^”"\n]{1,300})[”"]')


def _infer_role(raw: str, start: int, end: int) -> str:
    # 先看引号后面的词（"at the bottom"），再看前面的词；按距离近的优先，
    # 避免第一段文字里的 "top" 把第二段的 "bottom" 抢走。
    after = raw[end : min(len(raw), end + 60)].lower()
    before = raw[max(0, start - 60) : start].lower()
    checks = (
        ("subtitle", ("subtitle", "subheading")),
        ("title", ("title", "headline")),
        ("header", ("header", "top", "above", "upper")),
        ("footer", ("footer", "bottom", "below", "lower")),
        ("logo", ("logo", "brand mark")),
        ("label", ("label", "tag", "badge")),
        ("sign", ("sign", "signage", "billboard")),
        ("watermark", ("watermark",)),
        ("caption", ("caption", "cutline")),
    )
    for scope in (after, before):
        for role, words in checks:
            if any(re.search(r"\b" + re.escape(word) + r"\b", scope) for word in words):
                return role
    return "body"


def extract_quoted_texts(raw: str) -> list[dict[str, str]]:
    """从原始描述里抽图内文字：每个词条一条，并推断 role。"""
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for m in _QUOTED_RE.finditer(raw):
        text = normalize_space(m.group(1))
        if not text:
            continue
        role = _infer_role(raw, m.start(), m.end())
        key = (text, role)
        if key in seen:
            continue
        seen.add(key)
        out.append({"text": text, "role": role})
    return out


__all__ = [
    "classify_clause",
    "extract_quoted_texts",
    "extract_subject",
    "has_person",
    "identity_hits",
    "join_sentence",
    "normalize_space",
    "remove_trigger",
    "split_clauses",
    "strip_identity",
    "strip_identity_clauses",
    "strip_punct",
]
