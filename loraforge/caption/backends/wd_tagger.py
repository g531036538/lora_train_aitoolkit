"""本地 WD tagger v3（ONNX）后端。

模型目录只从环境变量 ``LORAFORGE_WD_MODEL_DIR`` 读，目录里需要：

- ``model.onnx``
- ``selected_tags.csv``（WD tagger v3 的标签表，列含 ``name`` / ``category``）

推理口径按 WD tagger v3 官方 Space（``app.py``）：输入名 ``input``，NHWC
``[1, 448, 448, 3]``，float32、值域 0–255；读图后先压白底、居中补成正方形、
BICUBIC resize 到 448，最后 RGB→BGR。输出分数与 ``selected_tags.csv`` 的行序
一一对应，只保留 general（0）与 character（4），rating（9）一律丢弃。
general 与 character 使用不同阈值，不能共用一个。
"""

from __future__ import annotations

import csv
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from loraforge.caption.base import BackendCaps, CaptionContext

try:  # 预装在机器上；缺失时让 availability 报 missing_dependency
    import onnxruntime as ort
except ImportError:  # pragma: no cover - 只在缺依赖的环境发生
    ort = None  # type: ignore[assignment]

ENV_MODEL_DIR = "LORAFORGE_WD_MODEL_DIR"
MODEL_FILENAME = "model.onnx"
TAGS_FILENAME = "selected_tags.csv"
INPUT_NAME = "input"
INPUT_SIZE = 448
DEFAULT_GENERAL_THRESHOLD = 0.35
DEFAULT_CHARACTER_THRESHOLD = 0.85

CATEGORY_GENERAL = 0
CATEGORY_CHARACTER = 4
CATEGORY_RATING = 9
_KEEP_CATEGORIES = frozenset({CATEGORY_GENERAL, CATEGORY_CHARACTER})

CAPS = BackendCaps(
    needs_gpu=False,
    supports_tags=True,
    supports_prose=False,
    languages=("en",),
)

SessionFactory = Callable[[str], Any]


def availability() -> tuple[bool, str]:
    """返回 (是否可用, 带稳定前缀的 reason)。不抛异常，供 list_backends 调用。"""
    if ort is None:
        return False, "missing_dependency:onnxruntime"

    raw = os.getenv(ENV_MODEL_DIR, "").strip()
    if not raw:
        return False, "model_weights_missing"

    model_dir = Path(raw)
    if not model_dir.is_dir():
        return False, "model_weights_missing"
    if not (model_dir / MODEL_FILENAME).is_file():
        return False, "model_weights_missing"
    if not (model_dir / TAGS_FILENAME).is_file():
        return False, "model_weights_missing"
    return True, "ok"


def _load_tags(path: Path) -> list[tuple[str, int]]:
    """按 CSV 行序读 (name, category)，顺序不能改。"""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise RuntimeError(f"标签表没有表头：{path}")
            rows: list[tuple[str, int]] = []
            for line_no, row in enumerate(reader, start=2):
                name = str(row.get("name") or "").strip()
                if not name:
                    raise RuntimeError(f"标签表第 {line_no} 行缺少 name：{path}")
                raw_category = str(row.get("category") or "").strip()
                try:
                    category = int(raw_category)
                except ValueError as exc:
                    raise RuntimeError(
                        f"标签表第 {line_no} 行 category 不是整数：{raw_category!r}"
                    ) from exc
                rows.append((name, category))
    except OSError as exc:
        raise RuntimeError(f"读取标签表失败：{exc}") from exc

    if not rows:
        raise RuntimeError(f"标签表为空：{path}")
    return rows


def _prepare_image(image: Path) -> np.ndarray:
    """读图并按 WD v3 官方口径预处理，返回 [1, 448, 448, 3] float32。

    顺序固定为：透明图压白底 → RGB → 居中补白边成正方形 → BICUBIC 448 →
    float32 0–255 → RGB→BGR → 加 batch 维。
    """
    try:
        with Image.open(image) as opened:
            rgba = opened.convert("RGBA")
    except OSError as exc:
        raise RuntimeError(f"读取图片失败：{exc}") from exc

    canvas = Image.new("RGBA", rgba.size, (255, 255, 255))
    canvas.alpha_composite(rgba)
    rgb = canvas.convert("RGB")

    width, height = rgb.size
    side = max(width, height)
    pad_left = (side - width) // 2
    pad_top = (side - height) // 2
    padded = Image.new("RGB", (side, side), (255, 255, 255))
    padded.paste(rgb, (pad_left, pad_top))

    resized = padded.resize((INPUT_SIZE, INPUT_SIZE), Image.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32)
    arr = arr[:, :, ::-1]  # RGB → BGR，官方模型要 BGR
    batch = np.expand_dims(arr, axis=0)
    return np.ascontiguousarray(batch)


class WdTaggerBackend:
    """``CaptionBackend`` 的本地 ONNX 实现。"""

    name = "wd_tagger"

    def __init__(self, session_factory: SessionFactory | None = None) -> None:
        raw = os.getenv(ENV_MODEL_DIR, "").strip()
        if not raw:
            raise RuntimeError(f"未配置 {ENV_MODEL_DIR}")

        self.model_dir = Path(raw)
        tags_path = self.model_dir / TAGS_FILENAME
        if not tags_path.is_file():
            raise RuntimeError(f"缺少标签表：{tags_path}")
        self._tags = _load_tags(tags_path)

        if session_factory is not None:
            self.session = session_factory(str(self.model_dir / MODEL_FILENAME))
        else:
            if ort is None:
                raise RuntimeError("缺少 onnxruntime；无法运行 WD tagger")
            model_path = self.model_dir / MODEL_FILENAME
            if not model_path.is_file():
                raise RuntimeError(f"缺少模型文件：{model_path}")
            self.session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )

    def caps(self) -> BackendCaps:
        return CAPS

    def _threshold(self, ctx: CaptionContext, key: str, default: float) -> float:
        raw = (ctx.extra or {}).get(key, default)
        try:
            threshold = float(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{key} 必须是数字：{raw!r}") from exc
        if not 0.0 <= threshold <= 1.0:
            raise RuntimeError(f"{key} 必须在 0–1 之间：{threshold!r}")
        return threshold

    def _thresholds(self, ctx: CaptionContext) -> tuple[float, float]:
        general = self._threshold(
            ctx, "general_threshold", DEFAULT_GENERAL_THRESHOLD
        )
        character = self._threshold(
            ctx, "character_threshold", DEFAULT_CHARACTER_THRESHOLD
        )
        return general, character

    def caption(self, image: Path, ctx: CaptionContext) -> str:
        image = Path(image)
        if not image.exists():
            raise RuntimeError(f"图片不存在：{image}")

        batch = _prepare_image(image)
        outputs = self.session.run(None, {INPUT_NAME: batch})
        if not outputs:
            raise RuntimeError("WD tagger 没有返回输出")

        scores = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if scores.size < len(self._tags):
            raise RuntimeError(
                f"WD tagger 输出分数 {scores.size} 个，少于标签表 {len(self._tags)} 行"
            )

        general_threshold, character_threshold = self._thresholds(ctx)
        picked: list[str] = []
        for (name, category), score in zip(self._tags, scores):
            if category not in _KEEP_CATEGORIES:
                continue
            threshold = (
                general_threshold
                if category == CATEGORY_GENERAL
                else character_threshold
            )
            if float(score) >= threshold:
                picked.append(name)

        if not picked:
            raise RuntimeError("WD tagger 没有超过阈值的标签")
        return ", ".join(picked)


__all__ = [
    "CAPS",
    "DEFAULT_CHARACTER_THRESHOLD",
    "DEFAULT_GENERAL_THRESHOLD",
    "ENV_MODEL_DIR",
    "INPUT_NAME",
    "INPUT_SIZE",
    "MODEL_FILENAME",
    "TAGS_FILENAME",
    "WdTaggerBackend",
    "availability",
]
