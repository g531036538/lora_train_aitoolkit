"""WD tagger 后端测试：全部用假会话，不加载真 onnx，不发网络请求。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from loraforge.caption.backends import wd_tagger
from loraforge.caption.backends.wd_tagger import WdTaggerBackend
from loraforge.caption.base import BackendCaps, CaptionContext
from loraforge.models import DatasetKind, TargetModel

TAGS_CSV = """tag_id,name,category,count
0,1girl,0,100
1,long hair,0,90
2,hatsune miku,4,80
3,explicit,9,70
4,blue sky,0,60
5,rare tag,0,10
"""

# 1girl 0.90, long hair 0.80, hatsune miku 0.90（character）, explicit 0.99（rating）,
# blue sky 0.60, rare tag 0.10
DEFAULT_SCORES = [0.90, 0.80, 0.90, 0.99, 0.60, 0.10]


class FakeSession:
    def __init__(self, scores: list[float]) -> None:
        self.scores = np.asarray([scores], dtype=np.float32)
        self.feeds: list[dict[str, np.ndarray]] = []

    def run(self, output_names, feed):
        self.feeds.append(feed)
        return [self.scores]


def _write_png(path: Path, size: tuple[int, int], color=(255, 0, 0)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def _ctx(extra: dict | None = None) -> CaptionContext:
    return CaptionContext(
        target=TargetModel.SDXL,
        kind=DatasetKind.CHARACTER,
        trigger="my_tok",
        extra=extra,
    )


@pytest.fixture(autouse=True)
def _clear_model_dir(monkeypatch):
    monkeypatch.delenv(wd_tagger.ENV_MODEL_DIR, raising=False)


@pytest.fixture
def model_dir(scratch: Path, monkeypatch) -> Path:
    d = scratch / "wd-model"
    d.mkdir()
    (d / wd_tagger.TAGS_FILENAME).write_text(TAGS_CSV, encoding="utf-8")
    monkeypatch.setenv(wd_tagger.ENV_MODEL_DIR, str(d))
    return d


def _backend(model_dir: Path, scores: list[float] | None = None):
    session = FakeSession(DEFAULT_SCORES if scores is None else scores)
    backend = WdTaggerBackend(session_factory=lambda path: session)
    return backend, session


def test_caps_are_cpu_tags_only_english():
    caps = wd_tagger.CAPS
    assert caps == BackendCaps(
        needs_gpu=False,
        supports_tags=True,
        supports_prose=False,
        languages=("en",),
    )


def test_availability_reports_missing_onnxruntime(monkeypatch):
    monkeypatch.setattr(wd_tagger, "ort", None)
    assert wd_tagger.availability() == (False, "missing_dependency:onnxruntime")


def test_availability_reports_missing_model_dir(monkeypatch):
    monkeypatch.setattr(wd_tagger, "ort", object())
    assert wd_tagger.availability() == (False, "model_weights_missing")


def test_availability_requires_both_files(model_dir: Path, monkeypatch):
    monkeypatch.setattr(wd_tagger, "ort", object())
    assert wd_tagger.availability() == (False, "model_weights_missing")

    (model_dir / wd_tagger.MODEL_FILENAME).write_bytes(b"not really onnx")
    assert wd_tagger.availability() == (True, "ok")


def test_wide_image_is_centered_on_white(model_dir: Path, scratch: Path):
    backend, session = _backend(model_dir)
    image = _write_png(scratch / "wide.png", (40, 20))

    assert backend.caption(image, _ctx()) == "1girl, long hair, hatsune miku, blue sky"

    batch = session.feeds[-1]["input"]
    assert batch.shape == (1, 448, 448, 3)
    assert batch.dtype == np.float32
    # 40x20 补成 40x40：上下各 10px 白边，中心仍是红色。
    assert batch[0, 0, 224].min() > 250
    assert batch[0, 447, 224].min() > 250
    assert batch[0, 224, 224, 2] > 230


def test_tall_image_is_centered_on_white(model_dir: Path, scratch: Path):
    backend, session = _backend(model_dir)
    image = _write_png(scratch / "tall.png", (20, 40))

    backend.caption(image, _ctx())

    batch = session.feeds[-1]["input"]
    assert batch[0, 224, 0].min() > 250
    assert batch[0, 224, 447].min() > 250
    assert batch[0, 224, 224, 2] > 230


def test_channel_order_is_bgr(model_dir: Path, scratch: Path):
    backend, session = _backend(model_dir)
    image = _write_png(scratch / "red.png", (20, 20), color=(255, 0, 0))

    backend.caption(image, _ctx())

    pixel = session.feeds[-1]["input"][0, 224, 224]
    # 红图进模型时必须是 BGR：蓝通道 0、红通道 255。
    assert pixel[0] < 20
    assert pixel[2] > 230


def test_rating_is_dropped(model_dir: Path, scratch: Path):
    backend, _ = _backend(model_dir)
    image = _write_png(scratch / "rating.png", (20, 20))

    caption = backend.caption(image, _ctx())
    assert "explicit" not in caption


def test_order_follows_csv_not_score(model_dir: Path, scratch: Path):
    backend, _ = _backend(model_dir)
    image = _write_png(scratch / "order.png", (20, 20))

    assert backend.caption(image, _ctx()) == "1girl, long hair, hatsune miku, blue sky"


def test_character_and_general_have_separate_thresholds(model_dir: Path, scratch: Path):
    backend, _ = _backend(model_dir)
    image = _write_png(scratch / "thresholds.png", (20, 20))

    caption = backend.caption(
        image,
        _ctx({"general_threshold": 0.75, "character_threshold": 0.65}),
    )
    assert caption == "1girl, long hair, hatsune miku"


def test_empty_result_raises(model_dir: Path, scratch: Path):
    backend, _ = _backend(model_dir, [0.1] * 6)
    image = _write_png(scratch / "empty.png", (20, 20))

    with pytest.raises(RuntimeError, match="没有超过阈值"):
        backend.caption(image, _ctx())


def test_input_is_float32_nhwc_0_to_255(model_dir: Path, scratch: Path):
    backend, session = _backend(model_dir)
    image = _write_png(scratch / "input.png", (20, 20))

    backend.caption(image, _ctx())

    batch = session.feeds[-1]["input"]
    assert batch.shape == (1, 448, 448, 3)
    assert batch.dtype == np.float32
    assert 0.0 <= float(batch.min()) and float(batch.max()) <= 255.0


def test_pipeline_lists_wd_tagger_when_ready(model_dir: Path, monkeypatch):
    from loraforge.caption.pipeline import list_backends

    monkeypatch.setattr(wd_tagger, "ort", object())
    (model_dir / wd_tagger.MODEL_FILENAME).write_bytes(b"not really onnx")

    infos = {info.key: info for info in list_backends()}
    assert "wd_tagger" in infos
    info = infos["wd_tagger"]
    assert info.available is True
    assert info.reason == "ok"
    assert info.caps.supports_tags is True
    assert info.caps.supports_prose is False


def test_get_backend_reports_missing_weights(model_dir: Path, monkeypatch):
    from loraforge.caption.pipeline import get_backend

    monkeypatch.setattr(wd_tagger, "ort", object())
    # 异常信息走 human_reason：给用户可执行的路径提示，不再是裸前缀。
    with pytest.raises(RuntimeError, match="LORAFORGE_WD_MODEL_DIR"):
        get_backend("wd_tagger")


def test_get_backend_reuses_instance_and_invalidates_on_mtime(
    model_dir: Path, monkeypatch
):
    """同一模型两次 get_backend 必须同一实例；权重 mtime 变了要重建。"""
    from loraforge.caption.pipeline import get_backend

    model_path = model_dir / wd_tagger.MODEL_FILENAME
    model_path.write_bytes(b"not really onnx")
    calls: list[str] = []

    class FakeOrt:
        @staticmethod
        def InferenceSession(path, providers):
            calls.append(path)
            return FakeSession(DEFAULT_SCORES)

    monkeypatch.setattr(wd_tagger, "ort", FakeOrt)
    monkeypatch.setattr(wd_tagger, "_SESSION_CACHE", {})
    monkeypatch.setattr(wd_tagger, "_BACKEND_CACHE", {})

    first = get_backend("wd_tagger")
    second = get_backend("wd_tagger")
    assert first is second
    assert len(calls) == 1

    stat = model_path.stat()
    os.utime(model_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    third = get_backend("wd_tagger")
    assert third is not first
    assert len(calls) == 2
