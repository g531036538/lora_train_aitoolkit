"""打标前的图集清点：哪些是训练图、哪些文件被跳过、哪张已有 .txt。

和 ``core.scan_dataset`` 的区别：那个是**打包口径**（认 webp/bmp，只关心图与词的配对），
这个是**打标口径** —— 只认 ai-toolkit 能直接吃进去的 ``.jpg/.jpeg/.png``，
并且把「不是图、被跳过」的文件原样列出来，让人一眼看到自己目录里混进了什么。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loraforge.export.core import CAPTION_SUFFIX, ExportError

#: 打标阶段认作训练图的扩展名（小写比较）。ai-toolkit 的图集只吃这几种。
CAPTIONING_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


@dataclass(frozen=True)
class ScanEntry:
    """目录里的一张训练图。"""

    name: str
    path: Path
    size: int
    has_caption: bool
    caption_bytes: int


@dataclass(frozen=True)
class ScanResult:
    """一次清点的结果。``skipped`` 是**文件名**，不是静默计数。"""

    root: Path
    images: list[ScanEntry]
    skipped: list[str]

    @property
    def with_caption(self) -> list[ScanEntry]:
        return [entry for entry in self.images if entry.has_caption]

    @property
    def without_caption(self) -> list[ScanEntry]:
        return [entry for entry in self.images if not entry.has_caption]


def scan_images(dataset_dir: Path | str) -> ScanResult:
    """扫描图集目录**顶层**（与 ai-toolkit 的 ``folder_path`` 一致，不递归）。

    不是训练图的文件（``.txt``、``.json``、隐藏文件、子目录）一律进 ``skipped``。
    """
    root = Path(dataset_dir).expanduser()
    if not root.is_dir():
        raise ExportError(f"图集目录不存在或不是目录：{root}")

    entries = sorted(root.iterdir(), key=lambda item: item.name.lower())
    captions: dict[str, Path] = {}
    for entry in entries:
        if entry.is_file() and entry.suffix.lower() == CAPTION_SUFFIX:
            captions.setdefault(entry.stem.lower(), entry)

    images: list[ScanEntry] = []
    skipped: list[str] = []
    for entry in entries:
        if entry.is_dir():
            skipped.append(entry.name)
            continue
        if entry.suffix.lower() not in CAPTIONING_SUFFIXES:
            skipped.append(entry.name)
            continue
        caption = captions.get(entry.stem.lower())
        images.append(
            ScanEntry(
                name=entry.name,
                path=entry,
                size=entry.stat().st_size,
                has_caption=caption is not None,
                caption_bytes=caption.stat().st_size if caption is not None else 0,
            )
        )

    return ScanResult(root=root, images=images, skipped=skipped)


__all__ = [
    "CAPTIONING_SUFFIXES",
    "ScanEntry",
    "ScanResult",
    "scan_images",
]
