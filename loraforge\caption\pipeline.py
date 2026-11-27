"""批量打标流水线：遍历图集 → 后端识别 → styler 排版 → 写 ``<图名>.txt``。

这里只做编排：单张失败不中断整批，失败原因留在 ``BatchItem.error`` 里给界面显示。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from loraforge.caption.base import BackendCaps, CaptionBackend, CaptionContext, CaptionStyler
from loraforge.caption.reasons import human_reason
from loraforge.caption.stylers import default_styler

ProgressHook = Callable[[int, int, Path, str | None, str | None], None]
# (done, total, image, caption_or_None, error_or_None)


@dataclass(frozen=True)
class BackendInfo:
    key: str
    label: str
    caps: BackendCaps
    available: bool
    reason: str = ""


@dataclass(frozen=True)
class BatchItem:
    image: Path
    caption: str | None
    error: str | None


@dataclass(frozen=True)
class BatchReport:
    items: list[BatchItem]
    written: int
    failed: int
    skipped: int

    @property
    def ok(self) -> bool:
        return self.failed == 0


def _http_backend_info() -> BackendInfo:
    """HTTP 后端的信息；缺依赖或没配置都只体现在 available/reason 里。"""
    try:
        from loraforge.caption.backends.openai_http import (
            CAPS as HTTP_CAPS,
            availability,
        )

        available, reason = availability()
        return BackendInfo(
            key="openai-http",
            label="OpenAI 兼容 Vision API",
            caps=HTTP_CAPS,
            available=available,
            reason=reason,
        )
    except Exception as exc:  # 连模块都 import 不了：httpx 缺失等
        return BackendInfo(
            key="openai-http",
            label="OpenAI 兼容 Vision API",
            caps=BackendCaps(
                needs_gpu=False,
                supports_tags=False,
                supports_prose=True,
                languages=("en", "zh"),
            ),
            available=False,
            reason=f"missing_dependency:openai-http: 后端模块不可用：{exc}",
        )


def list_backends() -> list[BackendInfo]:
    """列出全部已知后端。不可用也列，reason 是「稳定前缀 + 可选细节」。"""
    from loraforge.testing import FakeBackend

    fake_caps = FakeBackend().caps()
    return [
        BackendInfo(
            key="fake",
            label="Fake（测试用，不读图）",
            caps=fake_caps,
            available=True,
            reason="ok",
        ),
        _http_backend_info(),
    ]


def get_backend(key: str) -> CaptionBackend:
    """按 key 取后端实例。未知 key 抛 KeyError，依赖/配置缺失抛 RuntimeError。"""
    normalized = str(key).strip().lower()
    if normalized == "fake":
        from loraforge.testing import FakeBackend

        return FakeBackend()

    if normalized in {"openai-http", "openai", "http"}:
        try:
            from loraforge.caption.backends.openai_http import (
                OpenAIVisionBackend,
                availability,
            )
        except Exception as exc:
            raise RuntimeError(f"OpenAI 兼容后端依赖缺失：{exc}") from exc

        available, reason = availability()
        if not available:
            raise RuntimeError(human_reason(reason))
        return OpenAIVisionBackend()

    raise KeyError(f"未知后端：{key!r}")


def _read_existing(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def run_batch(
    images: Sequence[Path],
    ctx: CaptionContext,
    backend: CaptionBackend,
    *,
    styler: CaptionStyler | None = None,
    overwrite: bool = True,
    on_progress: ProgressHook | None = None,
) -> BatchReport:
    """跑一批图。单张失败不中断，最后用 BatchReport 汇总。"""
    paths = [Path(image) for image in images]
    total = len(paths)
    active_styler = styler if styler is not None else default_styler(ctx.target)

    items: list[BatchItem] = []
    written = 0
    failed = 0
    skipped = 0

    for index, image in enumerate(paths, start=1):
        out_path = image.with_suffix(".txt")
        caption: str | None = None
        error: str | None = None
        try:
            existing = ""
            if not overwrite and out_path.exists():
                existing = _read_existing(out_path)
            if existing:
                skipped += 1
                caption = existing
            else:
                raw = backend.caption(image, ctx)
                if not raw or not raw.strip():
                    raise RuntimeError("后端返回了空打标词（空打标会静默毁掉训练）")

                caption = active_styler.render(raw, ctx)
                if not caption or not caption.strip():
                    raise RuntimeError("styler 生成了空打标词")

                out_path.write_text(caption.rstrip() + "\n", encoding="utf-8")
                written += 1
        except Exception as exc:
            failed += 1
            caption = None
            error = str(exc).strip() or type(exc).__name__

        items.append(BatchItem(image=image, caption=caption, error=error))
        if on_progress is not None:
            on_progress(index, total, image, caption, error)

    return BatchReport(items=items, written=written, failed=failed, skipped=skipped)


__all__ = [
    "BackendInfo",
    "BatchItem",
    "BatchReport",
    "ProgressHook",
    "get_backend",
    "list_backends",
    "run_batch",
]
