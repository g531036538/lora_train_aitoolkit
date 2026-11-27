"""透明底合成：把带 alpha 的图铺到纯色底上，输出 RGB PNG（新目录，原图不动）。

真图集里一半以上的 PNG 带真透明通道。ai-toolkit 读到 RGBA 会自己转 RGB，
透明区变黑还是变白不确定——训练底色不该交给运气。契约见 ``docs/ARCHITECTURE.md`` §4.1。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from loraforge.export.core import CAPTION_SUFFIX

#: 界面给的预设底色；纯色默认白。
BACKGROUNDS: dict[str, tuple[int, int, int]] = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "gray": (128, 128, 128),
}

Color = tuple[int, int, int]


@dataclass
class FlattenReport:
    """一次合成的结果。``items`` 每项是 ``(源, 产物, 中文错误原因)``。"""

    items: list[tuple[Path, Path | None, str | None]]
    written: int
    failed: int

    @property
    def ok(self) -> bool:
        return self.failed == 0

    @property
    def failures(self) -> list[tuple[Path, str]]:
        return [(source, reason or "") for source, product, reason in self.items if product is None]


class FlattenError(Exception):
    """单张图失败的原因；消息一律中文，直接给人看。"""


def _check_background(background: Sequence[int]) -> Color:
    if len(tuple(background)) != 3:
        raise ValueError("底色必须是三个 0-255 的整数，例如 (255, 255, 255)")
    values: list[int] = []
    for channel in background:
        if not isinstance(channel, int) or not 0 <= channel <= 255:
            raise ValueError(f"底色分量必须是 0-255 的整数：{channel!r}")
        values.append(channel)
    return (values[0], values[1], values[2])


def _has_alpha(image: Image.Image) -> bool:
    if image.mode in {"RGBA", "LA", "PA"}:
        return True
    return image.mode == "P" and "transparency" in image.info


def _flatten_one(source: Path, target: Path, background: Color) -> None:
    """读源图、铺底色、写 RGB PNG。任何问题抛 :class:`FlattenError`。"""
    if not source.is_file():
        raise FlattenError(f"源图不存在：{source}")
    try:
        with Image.open(source) as image:
            image.load()
            if _has_alpha(image):
                rgba = image.convert("RGBA")
                canvas = Image.new("RGB", rgba.size, background)
                canvas.paste(rgba, mask=rgba.getchannel("A"))
                result = canvas
            else:
                result = image.convert("RGB")
            if result.mode != "RGB":  # 防御：契约要求无 alpha 的 RGB PNG
                raise FlattenError(f"合成后不是 RGB：{result.mode}")
            result.save(target, format="PNG")
    except UnidentifiedImageError as exc:
        raise FlattenError("不是可识别的图片文件") from exc
    except FlattenError:
        raise
    except Exception as exc:  # Pillow 的异常类型比想象中杂，统一转中文
        raise FlattenError(f"合成失败：{exc}") from exc


def flatten_alpha(
    images: Sequence[Path],
    out_dir: Path,
    *,
    background: Color = BACKGROUNDS["white"],
    copy_sidecar: bool = True,
) -> FlattenReport:
    """把 ``images`` 逐张铺到 ``background`` 上，写进 ``out_dir``（新目录）。

    - 目标目录已有同名文件（或同名 ``.txt``）时**报错跳过**，绝不覆盖，更不碰原图；
    - 输出一律 RGB PNG；``copy_sidecar=True`` 时把同名 ``.txt`` 逐字节复制，保持图词配对；
    - 单张失败不中断整批，原因写进 ``items`` 第三项。
    """
    color = _check_background(background)
    out = Path(out_dir).expanduser()
    if out.exists() and not out.is_dir():
        raise ValueError(f"输出路径已存在且不是目录：{out}")
    out.mkdir(parents=True, exist_ok=True)

    items: list[tuple[Path, Path | None, str | None]] = []
    written = 0
    failed = 0

    for item in images:
        source = Path(item).expanduser()
        target = out / source.name
        sidecar_source = source.with_suffix(CAPTION_SUFFIX)
        sidecar_target = out / sidecar_source.name
        try:
            if target.exists():
                raise FlattenError(f"目标目录已有同名文件，不覆盖：{target.name}")
            if copy_sidecar and sidecar_source.is_file() and sidecar_target.exists():
                raise FlattenError(f"目标目录已有同名打标词，不覆盖：{sidecar_target.name}")
            _flatten_one(source, target, color)
            if copy_sidecar and sidecar_source.is_file():
                sidecar_target.write_bytes(sidecar_source.read_bytes())
        except FlattenError as exc:
            failed += 1
            items.append((source, None, str(exc)))
            continue
        except Exception as exc:  # 兜底：任何一张出事都不许拖垮整批
            failed += 1
            items.append((source, None, f"合成失败：{exc}"))
            continue
        written += 1
        items.append((source, target, None))

    return FlattenReport(items=items, written=written, failed=failed)


__all__ = ["BACKGROUNDS", "FlattenError", "FlattenReport", "flatten_alpha"]
