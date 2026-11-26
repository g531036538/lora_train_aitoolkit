"""OpenAI 兼容的 Vision API 后端。

配置只从环境变量读，绝不硬编码、不写进仓库：

- ``LORAFORGE_VLM_URL``    端点根地址，例如 ``http://127.0.0.1:8000/v1``
- ``LORAFORGE_VLM_MODEL``  模型名
- ``LORAFORGE_VLM_KEY``    可选；没有鉴权的本地端点可以不给

本机装不了 onnxruntime/transformers，所以这是唯一可用的真实后端。
任何网络/解析失败都抛异常并带出 HTTP 状态码，绝不返回空串。
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from loraforge.caption.base import BackendCaps, CaptionContext
from loraforge.models import DatasetKind, TargetModel

try:  # httpx 预装在机器上；缺失时也不让 import 本模块失败
    import httpx
except ImportError:  # pragma: no cover - 只在缺依赖的环境发生
    httpx = None  # type: ignore[assignment]

ENV_URL = "LORAFORGE_VLM_URL"
ENV_MODEL = "LORAFORGE_VLM_MODEL"
ENV_KEY = "LORAFORGE_VLM_KEY"
ENV_TIMEOUT = "LORAFORGE_VLM_TIMEOUT"
ENV_MAX_TOKENS = "LORAFORGE_VLM_MAX_TOKENS"

DEPENDENCY_ERROR = "" if httpx is not None else "缺少 httpx；无法调用 OpenAI 兼容端点"

CAPS = BackendCaps(
    needs_gpu=False,
    supports_tags=False,
    supports_prose=True,
    languages=("en", "zh"),
)

_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

# 1x1 透明 PNG，只用于 --check 的最小 vision 请求；不读本地文件。
_CHECK_IMAGE = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def availability() -> tuple[bool, str]:
    """返回 (是否可用, 不可用原因)。不抛异常，供 list_backends 调用。"""
    if DEPENDENCY_ERROR:
        return False, DEPENDENCY_ERROR
    if not os.getenv(ENV_URL, "").strip():
        return False, f"未配置 {ENV_URL}"
    if not os.getenv(ENV_MODEL, "").strip():
        return False, f"未配置 {ENV_MODEL}"
    return True, ""


def _prompt(ctx: CaptionContext) -> str:
    if ctx.target == TargetModel.SDXL:
        return (
            "Caption this image for SDXL LoRA training. Reply with a comma-separated list of "
            "lowercase Danbooru-style tags, most important first. Use canonical tags "
            "(1girl, solo, long hair, school uniform, outdoors, sunlight, ...). "
            "Do not write a sentence. Do not add quality tags like masterpiece or 8k."
        )

    if ctx.target == TargetModel.MOSS and ctx.kind == DatasetKind.TEXT_LAYOUT:
        return (
            "Describe this image for MOSS-Image text-layout training. Reply with a single JSON "
            "object with keys: object, texts, appearance, composition. texts is an array of "
            '{"text": exact in-image text, "role": "header|footer|title|subtitle|body|label|'
            'logo|sign|watermark", "bbox": [y_min, x_min, y_max, x_max]}. bbox origin is the '
            "canvas center and every side length must be a multiple of 16. Quote text exactly. "
            "Do not describe face shape, hairstyle, hair length or color, eye color, or skin tone."
        )

    extra = ""
    if ctx.target == TargetModel.MOSS:
        extra = (
            " If the image contains text, quote each string exactly and state its role "
            "(header, footer, title, body, label, logo, sign, watermark)."
        )
    return (
        "Describe this image for LoRA training in one natural-language sentence: subject, "
        "framing, camera angle, expression, clothing, lighting, background, and material. "
        "Do not describe face shape, hairstyle, hair length or color, eye color, or skin tone; "
        "identity is handled by the trigger word." + extra
    )


def _extract_content(body: Any) -> str:
    if not isinstance(body, dict):
        raise RuntimeError("VLM 返回的 JSON 不是对象")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError('VLM 响应缺少 choices[0].message.content')
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                pieces.append(part["text"])
        return "\n".join(pieces).strip()
    raise RuntimeError('VLM 响应缺少 choices[0].message.content')


class OpenAIVisionBackend:
    """``CaptionBackend`` 的 HTTP 实现。"""

    name = "openai-http"

    def __init__(
        self,
        *,
        transport: Any = None,
        url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        if httpx is None:
            raise RuntimeError(DEPENDENCY_ERROR)
        self.url = (url if url is not None else os.getenv(ENV_URL, "")).strip()
        self.model = (model if model is not None else os.getenv(ENV_MODEL, "")).strip()
        self.api_key = (api_key if api_key is not None else os.getenv(ENV_KEY, "")).strip()
        if not self.url:
            raise RuntimeError(f"未配置 {ENV_URL}")
        if not self.model:
            raise RuntimeError(f"未配置 {ENV_MODEL}")
        if timeout is None:
            raw = os.getenv(ENV_TIMEOUT, "120").strip()
            try:
                timeout = float(raw)
            except ValueError:
                timeout = 120.0
        self.timeout = timeout
        self._transport = transport

    def caps(self) -> BackendCaps:
        return CAPS

    @property
    def endpoint(self) -> str:
        base = self.url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    def _post(self, payload: dict[str, Any]) -> Any:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                response = client.post(self.endpoint, headers=headers, json=payload)
        except httpx.RequestError as exc:
            raise RuntimeError(f"VLM 请求失败（网络）：{exc}") from exc

        if response.status_code >= 400:
            detail = response.text[:300].strip()
            raise RuntimeError(f"VLM 请求失败：HTTP {response.status_code} {detail}".rstrip())

        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(f"VLM 返回的不是 JSON：HTTP {response.status_code}") from exc

    def caption(self, image: Path, ctx: CaptionContext) -> str:
        image = Path(image)
        if not image.exists():
            raise RuntimeError(f"图片不存在：{image}")
        suffix = image.suffix.lower()
        if suffix not in _MIME:
            raise RuntimeError(f"不支持的图片格式：{suffix or '(无扩展名)'}")

        try:
            payload_bytes = image.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"读取图片失败：{exc}") from exc

        data_uri = f"data:{_MIME[suffix]};base64,{base64.b64encode(payload_bytes).decode('ascii')}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a precise image captioner."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _prompt(ctx)},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
            "temperature": 0.2,
            "max_tokens": _max_tokens(),
        }
        body = self._post(payload)
        text = _extract_content(body)
        if not text:
            raise RuntimeError("VLM 返回了空打标词")
        return text

    def probe(self) -> tuple[str | None, str]:
        """发一次最小 vision 请求，返回 (响应里的 model, 回复文本)。"""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Reply with OK if you can see this image.",
                        },
                        {"type": "image_url", "image_url": {"url": _CHECK_IMAGE}},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 8,
        }
        body = self._post(payload)
        text = _extract_content(body)
        model = body.get("model") if isinstance(body, dict) else None
        echoed = model.strip() if isinstance(model, str) and model.strip() else None
        return echoed, text


def _max_tokens() -> int:
    raw = os.getenv(ENV_MAX_TOKENS, "512").strip()
    try:
        value = int(raw)
    except ValueError:
        return 512
    return max(1, value)


def _redact(text: str, secret: str) -> str:
    """把密钥值从任何要打印的文本里去掉，连前几位也不留。"""
    if secret:
        return text.replace(secret, "[redacted]")
    return text


def check_endpoint(
    *,
    transport: Any = None,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    """``--check`` 的实现。返回进程退出码；不抛异常，不打印密钥。"""
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err

    available, reason = availability()
    if not available:
        print("端点可达：否", file=err)
        print(f"原因：{reason}", file=err)
        return 2

    try:
        backend = OpenAIVisionBackend(transport=transport)
    except Exception as exc:
        print("端点可达：否", file=err)
        print(f"原因：{_redact(str(exc), os.getenv(ENV_KEY, ''))}", file=err)
        return 2

    started = time.perf_counter()
    try:
        echoed, _ = backend.probe()
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        print("端点可达：否", file=err)
        print(f"错误：{_redact(str(exc), backend.api_key)}", file=err)
        print(f"耗时：{elapsed:.1f} ms", file=err)
        return 1

    elapsed = (time.perf_counter() - started) * 1000
    print("端点可达：是", file=out)
    if echoed:
        print(f"模型名回显：{_redact(echoed, backend.api_key)}", file=out)
    else:
        print(
            f"模型名回显：（响应未带 model，配置为 {_redact(backend.model, backend.api_key)}）",
            file=out,
        )
    print(f"耗时：{elapsed:.1f} ms", file=out)
    return 0


def main(argv: Sequence[str] | None = None, *, transport: Any = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m loraforge.caption.backends.openai_http",
        description="OpenAI 兼容 Vision 后端自检。",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="读取环境变量并发一次最小请求，检查端点是否可达",
    )
    args = parser.parse_args(argv)
    if not args.check:
        parser.print_help()
        return 2
    return check_endpoint(transport=transport)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAPS",
    "DEPENDENCY_ERROR",
    "ENV_KEY",
    "ENV_MODEL",
    "ENV_URL",
    "OpenAIVisionBackend",
    "availability",
    "check_endpoint",
    "main",
]
