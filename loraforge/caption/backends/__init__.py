"""Caption 后端实现。

本机不装 onnxruntime/transformers，所以只提供 OpenAI 兼容的 HTTP 后端；
WD/本地 VLM 后端在没有依赖时不会假装存在，见 ``pipeline.list_backends()``。
"""


def __getattr__(name: str):
    if name == "OpenAIVisionBackend":
        from .openai_http import OpenAIVisionBackend

        return OpenAIVisionBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["OpenAIVisionBackend"]
