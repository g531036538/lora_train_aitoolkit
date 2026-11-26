"""``openai_http --check`` 的自检，全部用 MockTransport，不联网。"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from loraforge.caption.backends import openai_http
from loraforge.caption.backends.openai_http import main


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        openai_http.ENV_URL,
        openai_http.ENV_MODEL,
        openai_http.ENV_KEY,
        openai_http.ENV_TIMEOUT,
        openai_http.ENV_MAX_TOKENS,
    ):
        monkeypatch.delenv(name, raising=False)


def configure(monkeypatch: pytest.MonkeyPatch, *, key: str = "sk-super-secret") -> None:
    monkeypatch.setenv(openai_http.ENV_URL, "http://vlm.test/v1")
    monkeypatch.setenv(openai_http.ENV_MODEL, "test-vision")
    monkeypatch.setenv(openai_http.ENV_KEY, key)


def ok_handler(seen: dict[str, Any] | None = None) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("authorization")
            seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "echo-vision",
                "choices": [{"message": {"content": "OK"}}],
            },
            request=request,
        )

    return handler


def test_cli_success_reports_reachable_model_and_elapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(monkeypatch)
    out, err = io.StringIO(), io.StringIO()
    code = openai_http.check_endpoint(
        transport=httpx.MockTransport(ok_handler()),
        out=out,
        err=err,
    )
    assert code == 0
    text = out.getvalue()
    assert "端点可达：是" in text
    assert "模型名回显：echo-vision" in text
    assert "耗时：" in text and " ms" in text
    assert "sk-super-secret" not in text + err.getvalue()


def test_main_check_prints_result_to_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure(monkeypatch)
    code = openai_http.main(["--check"], transport=httpx.MockTransport(ok_handler()))
    captured = capsys.readouterr()
    assert code == 0
    assert "端点可达：是" in captured.out
    assert "模型名回显：echo-vision" in captured.out
    assert "sk-super-secret" not in captured.out + captured.err


def test_cli_sends_minimal_vision_request(monkeypatch: pytest.MonkeyPatch) -> None:
    configure(monkeypatch)
    seen: dict[str, Any] = {}
    out, err = io.StringIO(), io.StringIO()
    code = openai_http.check_endpoint(
        transport=httpx.MockTransport(ok_handler(seen)),
        out=out,
        err=err,
    )
    assert code == 0
    assert seen["url"] == "http://vlm.test/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-super-secret"
    body = seen["body"]
    assert body["model"] == "test-vision"
    assert body["max_tokens"] == 8
    parts = body["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "Reply with OK if you can see this image."}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_cli_http_error_includes_status_and_redacts_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            text="upstream said sk-super-secret is invalid",
            request=request,
        )

    out, err = io.StringIO(), io.StringIO()
    code = openai_http.check_endpoint(
        transport=httpx.MockTransport(handler),
        out=out,
        err=err,
    )
    assert code == 1
    assert "端点可达：否" in err.getvalue()
    assert "HTTP 503" in err.getvalue()
    assert "sk-super-secret" not in out.getvalue() + err.getvalue()
    assert "[redacted]" in err.getvalue()


def test_cli_network_error_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    configure(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    out, err = io.StringIO(), io.StringIO()
    code = openai_http.check_endpoint(
        transport=httpx.MockTransport(handler),
        out=out,
        err=err,
    )
    assert code == 1
    assert "端点可达：否" in err.getvalue()
    assert "网络" in err.getvalue()
    assert "sk-super-secret" not in out.getvalue() + err.getvalue()


def test_cli_missing_url_is_a_clear_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, err = io.StringIO(), io.StringIO()
    code = openai_http.check_endpoint(out=out, err=err)
    assert code == 2
    assert "LORAFORGE_VLM_URL" in err.getvalue()
    assert "端点可达：否" in err.getvalue()


def test_cli_missing_model_is_a_clear_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(openai_http.ENV_URL, "http://vlm.test/v1")
    out, err = io.StringIO(), io.StringIO()
    code = openai_http.check_endpoint(out=out, err=err)
    assert code == 2
    assert "LORAFORGE_VLM_MODEL" in err.getvalue()


def test_cli_without_check_prints_help_and_returns_two(capsys: pytest.CaptureFixture[str]) -> None:
    code = openai_http.main([], transport=httpx.MockTransport(ok_handler()))
    captured = capsys.readouterr()
    assert code == 2
    assert "usage:" in captured.out


def test_cli_redacts_key_even_in_model_echo(monkeypatch: pytest.MonkeyPatch) -> None:
    configure(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "echo sk-super-secret",
                "choices": [{"message": {"content": "OK"}}],
            },
            request=request,
        )

    out, err = io.StringIO(), io.StringIO()
    code = openai_http.check_endpoint(
        transport=httpx.MockTransport(handler),
        out=out,
        err=err,
    )
    assert code == 0
    assert "sk-super-secret" not in out.getvalue() + err.getvalue()
    assert "[redacted]" in out.getvalue()


def test_cli_uses_configured_model_when_response_omits_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "OK"}}]},
            request=request,
        )

    out, err = io.StringIO(), io.StringIO()
    code = openai_http.check_endpoint(
        transport=httpx.MockTransport(handler),
        out=out,
        err=err,
    )
    assert code == 0
    assert "模型名回显：（响应未带 model，配置为 test-vision）" in out.getvalue()
