"""Caption 后端实现。

本机已装 onnxruntime，可本地跑 WD tagger；OpenAI 兼容的 HTTP 后端作为云端选项。
WD/本地 VLM 后端在没有依赖时不会假装存在，见 ``pipeline.list_backends()``。
"""


def __getattr__(name: str):
    if name == "OpenAIVisionBackend":
        from .openai_http import OpenAIVisionBackend

        return OpenAIVisionBackend
    if name == "WdTaggerBackend":
        from .wd_tagger import WdTaggerBackend

        return WdTaggerBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["OpenAIVisionBackend", "WdTaggerBackend"]
