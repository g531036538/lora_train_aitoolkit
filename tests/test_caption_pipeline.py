"""批量流水线的单测：用 FakeBackend，不联网、不读真图内容。"""

from __future__ import annotations

from pathlib import Path

import pytest

from loraforge.caption.base import BackendCaps, CaptionContext
from loraforge.caption.pipeline import (
    BatchItem,
    BatchReport,
    get_backend,
    list_backends,
    run_batch,
)
from loraforge.models import DatasetKind, TargetModel
from loraforge.testing import FakeBackend


def ctx(*, target: TargetModel = TargetModel.SDXL, trigger: str = "tok") -> CaptionContext:
    return CaptionContext(target=target, kind=DatasetKind.CHARACTER, trigger=trigger)


class EmptyBackend:
    name = "empty"

    def caps(self) -> BackendCaps:
        return BackendCaps(False, False, True, ("en",))

    def caption(self, image: Path, ctx: CaptionContext) -> str:
        return "   "


class PassthroughStyler:
    name = "passthrough"

    def render(self, raw: str, ctx: CaptionContext, tags: list[str] | None = None) -> str:
        return f"{ctx.trigger}|{raw}"


def test_run_batch_writes_txt_and_reports(images: list[Path]) -> None:
    seen: list[tuple[int, int, Path, str | None, str | None]] = []
    report = run_batch(
        images,
        ctx(trigger="MyTok"),
        FakeBackend(),
        on_progress=lambda *args: seen.append(args),
    )

    assert report.written == 3
    assert report.failed == 0
    assert report.skipped == 0
    assert report.ok
    assert len(report.items) == 3

    for image, item in zip(images, report.items):
        assert item.image == image
        assert item.error is None
        assert item.caption is not None
        assert image.with_suffix(".txt").read_text(encoding="utf-8").strip() == item.caption
        assert item.caption.startswith("MyTok, ")

    assert [row[0] for row in seen] == [1, 2, 3]
    assert all(row[1] == 3 for row in seen)
    assert all(row[4] is None for row in seen)
    assert [row[2] for row in seen] == images


def test_overwrite_false_skips_existing_and_true_rewrites(images: list[Path]) -> None:
    run_batch(images, ctx(), FakeBackend())

    skipped = run_batch(images, ctx(), FakeBackend(), overwrite=False)
    assert skipped.written == 0
    assert skipped.skipped == 3
    assert skipped.failed == 0
    assert skipped.ok
    assert all(item.caption and item.error is None for item in skipped.items)

    rewritten = run_batch(images, ctx(), FakeBackend(), overwrite=True)
    assert rewritten.written == 3
    assert rewritten.skipped == 0


def test_empty_existing_txt_is_regenerated_not_skipped(images: list[Path]) -> None:
    empty = images[0].with_suffix(".txt")
    empty.write_text("   \n", encoding="utf-8")

    report = run_batch(images, ctx(), FakeBackend(), overwrite=False)
    assert report.written == 3
    assert report.skipped == 0
    assert empty.read_text(encoding="utf-8").strip().startswith("tok, ")


def test_single_failure_does_not_stop_batch(images: list[Path]) -> None:
    seen: list[tuple[int, int, Path, str | None, str | None]] = []
    backend = FakeBackend(fail_on={"frame_02"})
    report = run_batch(images, ctx(), backend, on_progress=lambda *args: seen.append(args))

    assert report.failed == 1
    assert report.written == 2
    assert not report.ok
    assert report.items[1].caption is None
    assert report.items[1].error
    assert not images[1].with_suffix(".txt").exists()
    assert seen[1][3] is None
    assert seen[1][4] == report.items[1].error


def test_empty_backend_output_is_error_and_writes_nothing(images: list[Path]) -> None:
    report = run_batch(images, ctx(), EmptyBackend())
    assert report.failed == 3
    assert report.written == 0
    assert not report.ok
    assert all("空打标" in (item.error or "") for item in report.items)
    assert not any(image.with_suffix(".txt").exists() for image in images)


def test_custom_styler_is_used(images: list[Path]) -> None:
    report = run_batch(images, ctx(trigger="x"), FakeBackend(), styler=PassthroughStyler())
    assert report.written == 3
    for image, item in zip(images, report.items):
        expected = f"x|a photo of a person, subject {image.stem}"
        assert item.caption == expected
        assert image.with_suffix(".txt").read_text(encoding="utf-8").strip() == expected


def test_empty_batch_is_ok() -> None:
    report = run_batch([], ctx(), FakeBackend())
    assert report.items == []
    assert report.written == report.failed == report.skipped == 0
    assert report.ok


def test_batch_report_ok_property() -> None:
    assert BatchReport(items=[], written=0, failed=0, skipped=0).ok
    assert not BatchReport(
        items=[BatchItem(Path("a.png"), None, "boom")], written=0, failed=1, skipped=0
    ).ok


def test_list_backends_includes_unavailable_with_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LORAFORGE_VLM_URL", raising=False)
    monkeypatch.delenv("LORAFORGE_VLM_MODEL", raising=False)
    infos = {info.key: info for info in list_backends()}
    assert infos["fake"].available
    assert infos["fake"].reason == ""
    http = infos["openai-http"]
    assert not http.available
    assert "LORAFORGE_VLM_URL" in http.reason


def test_list_backends_http_available_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LORAFORGE_VLM_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("LORAFORGE_VLM_MODEL", "test-model")
    infos = {info.key: info for info in list_backends()}
    assert infos["openai-http"].available
    assert infos["openai-http"].reason == ""


def test_get_backend_fake_and_unknown() -> None:
    assert get_backend("fake").name == "fake"
    with pytest.raises(KeyError):
        get_backend("nope")


def test_get_backend_http_requires_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LORAFORGE_VLM_URL", raising=False)
    with pytest.raises(RuntimeError, match="LORAFORGE_VLM_URL"):
        get_backend("openai-http")


def test_get_backend_http_returns_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LORAFORGE_VLM_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("LORAFORGE_VLM_MODEL", "test-model")
    backend = get_backend("openai-http")
    assert backend.name == "openai-http"
