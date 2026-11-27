"""``BackendInfo.reason`` 的前缀契约。"""

from __future__ import annotations

import pytest

from loraforge.caption.backends import openai_http
from loraforge.caption.pipeline import list_backends
from loraforge.caption.reasons import (
    REASON_PREFIXES,
    human_reason,
    is_valid_reason,
    parse_reason,
)


def test_list_backends_reasons_use_known_prefixes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(openai_http.ENV_URL, raising=False)
    monkeypatch.delenv(openai_http.ENV_MODEL, raising=False)
    monkeypatch.setenv("LORAFORGE_ENABLE_FAKE", "1")

    infos = {info.key: info for info in list_backends()}
    assert infos["fake"].reason == "ok"
    assert infos["openai-http"].available is False

    for info in infos.values():
        assert is_valid_reason(info.reason), f"{info.key}: {info.reason!r}"

    prefix, _ = parse_reason(infos["openai-http"].reason)
    assert prefix in {
        "missing_dependency:httpx",
        "requires_credentials",
    }


def test_parse_reason_keeps_dependency_name_in_prefix() -> None:
    assert parse_reason("ok") == ("ok", "")
    assert parse_reason("missing_dependency:httpx") == ("missing_dependency:httpx", "")
    assert parse_reason("missing_dependency:httpx: 缺少库") == (
        "missing_dependency:httpx",
        "缺少库",
    )
    assert parse_reason("requires_credentials:未配置 LORAFORGE_VLM_URL") == (
        "requires_credentials",
        "未配置 LORAFORGE_VLM_URL",
    )


def test_reason_prefix_table_is_the_seven_contract_categories() -> None:
    assert REASON_PREFIXES == (
        "ok",
        "missing_dependency",
        "model_weights_missing",
        "requires_gpu",
        "requires_network",
        "requires_credentials",
        "unsupported_target",
    )


@pytest.mark.parametrize(
    "reason",
    [
        "ok",
        "missing_dependency:httpx",
        "missing_dependency:openai-http: 后端模块不可用：boom",
        "model_weights_missing",
        "requires_gpu",
        "requires_network",
        "requires_credentials:未配置 LORAFORGE_VLM_URL",
        "unsupported_target",
    ],
)
def test_is_valid_reason_accepts_contract_shapes(reason: str) -> None:
    assert is_valid_reason(reason)


@pytest.mark.parametrize(
    "reason",
    ["", "未配置 LORAFORGE_VLM_URL", "banana", "missing_dependency:", "missing_dependency:  "],
)
def test_is_valid_reason_rejects_unknown_or_empty_prefixes(reason: str) -> None:
    assert not is_valid_reason(reason)


def test_human_reason_strips_prefix_for_developers() -> None:
    assert human_reason("requires_credentials:未配置 LORAFORGE_VLM_URL") == (
        "未配置 LORAFORGE_VLM_URL"
    )
    assert human_reason("missing_dependency:httpx") == "缺少 httpx；无法调用 OpenAI 兼容端点"
    assert human_reason("missing_dependency:openai-http: 后端模块不可用：boom") == (
        "后端模块不可用：boom"
    )
