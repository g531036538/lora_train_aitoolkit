"""真实后端适配器。

本机装不了 onnxruntime/transformers，所以目前只提供 OpenAI 兼容的 HTTP 后端；
它的配置全部来自环境变量，见 ``openai_http``。
"""

from .openai_http import OpenAIVisionBackend

__all__ = ["OpenAIVisionBackend"]
