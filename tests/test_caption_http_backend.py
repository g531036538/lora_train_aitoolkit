from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from loraforge.caption.backends import openai_http
from loraforge.caption.backends.openai_http import OpenAIVisionBackend
from loraforge.caption.base import CaptionContext
from loraforge.caption.stylers import SdxlTagsStyler
from loraforge.models import DatasetKind, TargetModel


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


@pytest.fixture
def image(scratch: Path) -> Path:
    path = scratch / "frame.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return path


def ctx(
    target: TargetModel = TargetModel.SDXL,
    kind: DatasetKind = DatasetKind.CHARACTER,
) -> CaptionContext:
    return CaptionContext(target=target, kind=kind, trigger="my_tok")


def backend(handler: Callable[[httpx.Request], httpx.Response]) -> OpenAIVisionBackend:
    return OpenAIVisionBackend(
        url="http://vlm.test/v1",
        model="test-vision",
        api_key="sk-test",
        transport=httpx.MockTransport(handler),
    )


def ok_response(text: str) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": text}}]},
            request=request,
        )

    return handler


def test_availability_reports_missing_url() -> None:
    assert openai_http.availability() == (
        False,
        "requires_credentials:未配置 LORAFORGE_VLM_URL",
    )


def test_availability_reports_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(openai_http.ENV_URL, "http://vlm.test/v1")
    assert openai_http.availability() == (
        False,
        "requires_credentials:未配置 LORAFORGE_VLM_MODEL",
    )


def test_availability_is_true_when_url_and_model_are_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(openai_http.ENV_URL, "http://vlm.test/v1")
    monkeypatch.setenv(openai_http.ENV_MODEL, "test-vision")
    assert openai_http.availability() == (True, "ok")


def test_constructor_requires_url() -> None:
    with pytest.raises(RuntimeError, match="LORAFORGE_VLM_URL"):
        OpenAIVisionBackend(model="test-vision")


def test_constructor_requires_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(openai_http.ENV_URL, "http://vlm.test/v1")
    with pytest.raises(RuntimeError, match="LORAFORGE_VLM_MODEL"):
        OpenAIVisionBackend()


def test_constructor_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(openai_http.ENV_URL, "http://vlm.test/v1/")
    monkeypatch.setenv(openai_http.ENV_MODEL, "env-vision")
    monkeypatch.setenv(openai_http.ENV_KEY, "env-key")
    client = OpenAIVisionBackend()
    assert client.url == "http://vlm.test/v1/"
    assert client.model == "env-vision"
    assert client.api_key == "env-key"
    assert client.endpoint == "http://vlm.test/v1/chat/completions"


def test_endpoint_does_not_double_append() -> None:
    client = OpenAIVisionBackend(
        url="http://vlm.test/v1/chat/completions",
        model="test-vision",
    )
    assert client.endpoint == "http://vlm.test/v1/chat/completions"


def test_caption_posts_expected_request_and_returns_text(image: Path) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": " my_tok, 1girl, red dress "}}]},
            request=request,
        )

    client = backend(handler)
    assert client.caption(image, ctx()) == "my_tok, 1girl, red dress"

    assert seen["url"] == "http://vlm.test/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    body = seen["body"]
    assert body["model"] == "test-vision"
    assert body["messages"][0]["role"] == "system"
    user_parts = body["messages"][1]["content"]
    assert user_parts[0]["type"] == "text"
    assert "comma-separated" in user_parts[0]["text"]
    assert user_parts[1]["type"] == "image_url"
    assert user_parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_caption_omits_authorization_without_key(image: Path) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return ok_response("my_tok, 1girl, red dress")(request)

    client = OpenAIVisionBackend(
        url="http://vlm.test/v1",
        model="test-vision",
        api_key="",
        transport=httpx.MockTransport(handler),
    )
    client.caption(image, ctx())
    assert seen["auth"] is None


def test_caption_accepts_content_parts(image: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "my_tok, 1girl,"},
                                {"type": "text", "text": " red dress"},
                            ]
                        }
                    }
                ]
            },
            request=request,
        )

    assert backend(handler).caption(image, ctx()) == "my_tok, 1girl,\n red dress"


def test_caption_raises_with_http_status(image: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="model overloaded", request=request)

    with pytest.raises(RuntimeError, match="HTTP 503") as excinfo:
        backend(handler).caption(image, ctx())
    assert "model overloaded" in str(excinfo.value)


def test_caption_raises_on_network_error(image: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(RuntimeError, match="网络") as excinfo:
        backend(handler).caption(image, ctx())
    assert "connection refused" in str(excinfo.value)


def test_caption_raises_on_non_json_response(image: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>oops</html>", request=request)

    with pytest.raises(RuntimeError, match="不是 JSON"):
        backend(handler).caption(image, ctx())


def test_caption_raises_on_empty_content(image: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "   "}}]},
            request=request,
        )

    with pytest.raises(RuntimeError, match="空打标词"):
        backend(handler).caption(image, ctx())


def test_caption_raises_on_missing_choices(image: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []}, request=request)

    with pytest.raises(RuntimeError, match="choices"):
        backend(handler).caption(image, ctx())


def test_caption_rejects_missing_image(image: Path) -> None:
    client = backend(ok_response("my_tok"))
    with pytest.raises(RuntimeError, match="图片不存在"):
        client.caption(image.parent / "missing.png", ctx())


def test_caption_rejects_unknown_suffix(image: Path) -> None:
    bad = image.with_suffix(".gif")
    bad.write_bytes(b"GIF89a")
    try:
        with pytest.raises(RuntimeError, match="不支持的图片格式"):
            backend(ok_response("my_tok")).caption(bad, ctx())
    finally:
        bad.unlink(missing_ok=True)


def test_prompt_for_sdxl_asks_for_tags(image: Path) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["prompt"] = json.loads(request.content)["messages"][1]["content"][0]["text"]
        return ok_response("my_tok, 1girl")(request)

    backend(handler).caption(image, ctx(TargetModel.SDXL))
    assert "Danbooru-style tags" in seen["prompt"]
    assert "Do not write a sentence" in seen["prompt"]
    assert "Do not explain" in seen["prompt"]


def test_prompt_for_krea2_asks_for_prose(image: Path) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["prompt"] = json.loads(request.content)["messages"][1]["content"][0]["text"]
        return ok_response("my_tok, a woman in a red dress")(request)

    backend(handler).caption(image, ctx(TargetModel.KREA2))
    assert "one natural-language sentence" in seen["prompt"]


def test_want_tags_forces_tag_prompt_for_krea2(image: Path) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["prompt"] = json.loads(request.content)["messages"][1]["content"][0]["text"]
        return ok_response("my_tok, 1girl")(request)

    context = CaptionContext(
        target=TargetModel.KREA2,
        kind=DatasetKind.CHARACTER,
        trigger="my_tok",
        extra={"want_tags": True},
    )
    backend(handler).caption(image, context)
    assert "Danbooru-style tags" in seen["prompt"]
    assert "one natural-language sentence" not in seen["prompt"]


def test_sdxl_sentence_response_is_cleaned_by_styler(image: Path) -> None:
    sentence = (
        "This is a photo of a woman. She is wearing a red dress. "
        "She is standing in a park."
    )
    raw = backend(ok_response(sentence)).caption(image, ctx(TargetModel.SDXL))
    rendered = SdxlTagsStyler().render(raw, ctx(TargetModel.SDXL))
    assert rendered == "my_tok, 1girl, red dress, park"


def test_prompt_for_moss_text_layout_asks_for_json_bbox(image: Path) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["prompt"] = json.loads(request.content)["messages"][1]["content"][0]["text"]
        return ok_response('{"object": "poster"}')(request)

    backend(handler).caption(image, ctx(TargetModel.MOSS, DatasetKind.TEXT_LAYOUT))
    assert "single JSON object" in seen["prompt"]
    assert "bbox" in seen["prompt"]
    assert "multiple of 16" in seen["prompt"]


def test_caps_support_tags_and_prose_and_cpu() -> None:
    caps = backend(ok_response("my_tok")).caps()
    assert caps.needs_gpu is False
    assert caps.supports_tags is True
    assert caps.supports_prose is True
    assert "en" in caps.languages and "zh" in caps.languages
