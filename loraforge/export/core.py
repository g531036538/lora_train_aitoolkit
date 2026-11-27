"""把「图 + 同名 .txt + 训练配置」打成一个能直接传云的 zip。

契约见 ``docs/ARCHITECTURE.md`` §4：``package(...) -> ExportReport``。
这一层只负责**如实报出**图与打标词的对应关系，不替用户决定要不要导出：
缺打标词、孤儿 .txt、同名冲突全部进报告，由界面决定是否拦。
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

#: 认作训练图的扩展名。比较时统一小写。
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})

#: ai-toolkit 的 ``caption_ext``，见 config_modules.py:953。
CAPTION_SUFFIX = ".txt"


class ExportError(ValueError):
    """导出前置条件不满足（目录/配置不存在、没有图、打包失败等）。"""


@dataclass(frozen=True)
class ExportReport:
    """一次导出的结果。字段顺序按 §4 冻结，新增字段必须带默认值。"""

    zip_path: Path
    images: int
    captions: int
    missing_caption: list[str]
    orphan_caption: list[str]
    #: 同名冲突（``a.png`` 与 ``a.jpg`` 抢同一份 ``a.txt``）。契约外新增，带默认值。
    ambiguous: list[str] = field(default_factory=list)
    #: 附带的非图文件（目前是验收清单 ``验收.md``），按写入顺序记名字。
    extra_files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """无缺标、无孤儿、无冲突才算干净。"""
        return not self.missing_caption and not self.orphan_caption and not self.ambiguous


@dataclass(frozen=True)
class DatasetScan:
    """目录扫描结果。图与词的配对用 ``stem.lower()``，大小写不敏感。"""

    root: Path
    images: list[Path]
    #: ``stem.lower() -> 打标文件``。同一 stem 有多份 .txt 时取排序第一个（大小写敏感
    #: 文件系统上才可能出现），图侧撞名由 ``ambiguous`` 报。
    captions: dict[str, Path]

    @property
    def image_keys(self) -> set[str]:
        return {image.stem.lower() for image in self.images}

    @property
    def missing_caption(self) -> list[str]:
        """有图没词的图名（排序稳定，报告可直接对比）。"""
        captions = self.captions
        return sorted(
            image.name for image in self.images if image.stem.lower() not in captions
        )

    @property
    def orphan_caption(self) -> list[str]:
        """有词没图的 txt 名。"""
        keys = self.image_keys
        return sorted(
            caption.name
            for stem, caption in self.captions.items()
            if stem not in keys
        )

    @property
    def ambiguous(self) -> list[str]:
        """图或词在同名键上撞了，配对结果不确定。"""
        problems: list[str] = []
        by_key: dict[str, list[str]] = {}
        for image in self.images:
            by_key.setdefault(image.stem.lower(), []).append(image.name)
        for stem, names in by_key.items():
            if len(names) > 1:
                problems.append(
                    f"{' 与 '.join(sorted(names))} 共用同一份 {stem}{CAPTION_SUFFIX}，"
                    "无法判断谁该配哪一句"
                )
        return problems


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def scan_dataset(dataset_dir: Path | str) -> DatasetScan:
    """扫描图集目录（只看顶层，不递归 —— ai-toolkit 的 folder_path 也只看一层）。"""
    root = Path(dataset_dir)
    if not root.is_dir():
        raise ExportError(f"图集目录不存在或不是目录：{root}")

    images: list[Path] = []
    caption_files: list[Path] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_file():
            continue
        if _is_image(entry):
            images.append(entry)
        elif entry.suffix.lower() == CAPTION_SUFFIX:
            caption_files.append(entry)

    captions: dict[str, Path] = {}
    for caption in caption_files:
        key = caption.stem.lower()
        # 同名 .txt 里取排序第一个；冲突本身由 ambiguous 报出来。
        captions.setdefault(key, caption)

    return DatasetScan(root=root, images=images, captions=captions)


def _arcname(prefix: str, name: str) -> str:
    """zip 内部路径。只允许纯文件名，防止把目录外的路径塞进来。"""
    safe = Path(name).name
    if not safe or safe in {".", ".."}:
        raise ExportError(f"文件名不可用于打包：{name!r}")
    return f"{prefix}/{safe}"


def package(
    dataset_dir: Path | str,
    config_path: Path | str,
    out_zip: Path | str,
    *,
    include_images: bool = True,
    extra_files: Mapping[str, bytes] | None = None,
) -> ExportReport:
    """打包图集 → ``out_zip``。

    zip 结构（§4）：``<图集目录名>/`` 下平铺 ``图 + 同名 .txt + 配置 yaml``。

    * ``include_images=False`` 只打标注与配置（云端已有图时用）。
    * 缺打标词的图**仍然打进去**，但会出现在 ``missing_caption`` 里；
      是否拦截由调用方决定（界面在缺标时禁止导出）。
    * 孤儿 .txt 不进包，只出现在 ``orphan_caption`` 里。
    * ``extra_files`` 是附带文件（名字 → 字节），目前只有验收清单 ``验收.md``。
      写入前把 CRLF/CR 统一成 LF，保证包里的文本换行口径一致；与已有条目撞名直接报错，
      不静默覆盖。
    """
    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        raise ExportError(f"训练配置不存在：{cfg_path}")

    scan = scan_dataset(dataset_dir)
    if not scan.images:
        raise ExportError(f"图集目录里没有可训练的图：{scan.root}")

    out = Path(out_zip)
    out.parent.mkdir(parents=True, exist_ok=True)
    prefix = scan.root.resolve().name

    missing = scan.missing_caption
    ambiguous = scan.ambiguous
    orphan = scan.orphan_caption

    extra_written: list[str] = []
    part = out.with_name(out.name + ".part")
    try:
        with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(cfg_path, _arcname(prefix, cfg_path.name))
            written: set[str] = set()
            for image in scan.images:
                if include_images:
                    arcname = _arcname(prefix, image.name)
                    if arcname not in written:
                        archive.write(image, arcname)
                        written.add(arcname)
                caption = scan.captions.get(image.stem.lower())
                if caption is not None:
                    # 统一用图的 stem 命名，保证云端 `caption_ext: txt` 能配到。
                    arcname = _arcname(prefix, f"{image.stem}{CAPTION_SUFFIX}")
                    if arcname in written:
                        # a.png 与 a.jpg 共用 a.txt：只写一份，冲突交给 ambiguous 报。
                        continue
                    archive.write(caption, arcname)
                    written.add(arcname)
            for name, payload in (extra_files or {}).items():
                arcname = _arcname(prefix, name)
                if arcname in written:
                    raise ExportError(f"附带文件与包里已有条目撞名：{name!r}")
                # 换行统一成 LF：与 .txt / yaml 的内网口径保持一致，避免 Windows 写出 CRLF。
                blob = bytes(payload).replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                archive.writestr(arcname, blob)
                written.add(arcname)
                extra_written.append(name)
        os.replace(part, out)
    except Exception:
        part.unlink(missing_ok=True)
        raise

    captions = len(scan.images) - len(missing)
    return ExportReport(
        zip_path=out.resolve(),
        images=len(scan.images),
        captions=captions,
        missing_caption=missing,
        orphan_caption=orphan,
        ambiguous=ambiguous,
        extra_files=extra_written,
    )


__all__ = [
    "CAPTION_SUFFIX",
    "IMAGE_SUFFIXES",
    "DatasetScan",
    "ExportError",
    "ExportReport",
    "package",
    "scan_dataset",
]
