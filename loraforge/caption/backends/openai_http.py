"""OpenAI 兼容的 Vision API 后端。

配置只从环境变量读，绝不硬编码、不写进仓库：

- ``LORAFORGE_VLM_URL``    端点根地址，例如 ``http://127.0.0.1:8000/v1``
- ``LORAFORGE_VLM_MODEL``  模型名
- ``LORAFORGE_VLM_KEY``    可选；没有鉴权的本地端点可以不给

本机装不了 onnxruntime/transformers，所以这是唯一可用的真实后端。
任何网络/解析失败都抛异常并带出 HTTP 状态码，绝不返回空串。
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

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
            body = response.json()
        except ValueError as exc:
            raise RuntimeError(f"VLM 返回的不是 JSON：HTTP {response.status_code}") from exc

        text = _extract_content(body)
        if not text:
            raise RuntimeError("VLM 返回了空打标词")
        return text


def _max_tokens() -> int:
    raw = os.getenv(ENV_MAX_TOKENS, "512").strip()
    try:
        value = int(raw)
    except ValueError:
        return 512
    return max(1, value)


__all__ = [
    "CAPS",
    "DEPENDENCY_ERROR",
    "ENV_KEY",
    "ENV_MODEL",
    "ENV_URL",
    "OpenAIVisionBackend",
    "availability",
]
