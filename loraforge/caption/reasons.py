"""``BackendInfo.reason`` 的稳定前缀与解析。

界面按前缀分组、配图标，不能去猜中文；所以后缀是给人看的说明，前缀必须
来自固定词表。``missing_dependency`` 例外：它后面必须跟一个非空的依赖名，
例如 ``missing_dependency:httpx``。
"""

from __future__ import annotations

REASON_PREFIXES: tuple[str, ...] = (
    "ok",
    "missing_dependency",
    "model_weights_missing",
    "requires_gpu",
    "requires_network",
    "requires_credentials",
    "unsupported_target",
)

_MISSING_DEPENDENCY = "missing_dependency"


def parse_reason(reason: str) -> tuple[str, str]:
    """把 reason 拆成 ``(稳定前缀, 给人看的细节)``。

    ``missing_dependency:<name>`` 的依赖名属于前缀，例如
    ``parse_reason("missing_dependency:httpx") == ("missing_dependency:httpx", "")``；
    依赖名为空时前缀退化为 ``missing_dependency``，方便调用方判为非法。
    """
    text = str(reason or "").strip()
    head, sep, tail = text.partition(":")
    head = head.strip()

    if head == _MISSING_DEPENDENCY:
        name, name_sep, detail = tail.partition(":")
        name = name.strip()
        if not name:
            return _MISSING_DEPENDENCY, tail.strip()
        prefix = f"{_MISSING_DEPENDENCY}:{name}"
        return prefix, detail.strip() if name_sep else ""

    return head, tail.strip() if sep else ""


def is_valid_reason(reason: str) -> bool:
    """判断 reason 是否满足「固定前缀 + 可选细节」契约。"""
    prefix, _ = parse_reason(reason)
    if prefix == _MISSING_DEPENDENCY:
        return False
    if prefix.startswith(f"{_MISSING_DEPENDENCY}:"):
        return bool(prefix.partition(":")[2].strip())
    return prefix in REASON_PREFIXES


def human_reason(reason: str) -> str:
    """去掉前缀，返回可以放进异常/CLI 的中文说明。"""
    prefix, detail = parse_reason(reason)
    if prefix == "missing_dependency:httpx":
        return "缺少 httpx；无法调用 OpenAI 兼容端点"
    if prefix == "model_weights_missing":
        return "缺少模型权重：把 LORAFORGE_WD_MODEL_DIR 指到含 model.onnx + selected_tags.csv 的目录"
    if prefix == _MISSING_DEPENDENCY:
        return detail or "缺少依赖"
    if prefix.startswith(f"{_MISSING_DEPENDENCY}:"):
        return detail or f"缺少依赖：{prefix.partition(':')[2]}"
    return detail or prefix


__all__ = [
    "REASON_PREFIXES",
    "human_reason",
    "is_valid_reason",
    "parse_reason",
]
