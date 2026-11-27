"""导出打包：图集 + 同名 .txt + 训练配置 → 一个 zip。

对外只有 ``package`` 与 ``ExportReport``（契约见 ``docs/ARCHITECTURE.md`` §4）；
另有只读的 ``scan_dataset`` 给界面预览配对情况，
``scan_images`` 给打标前的图集清点（只认 jpg/jpeg/png，跳过的文件明示列出）。
"""

from __future__ import annotations

from loraforge.export.core import (
    CAPTION_SUFFIX,
    IMAGE_SUFFIXES,
    DatasetScan,
    ExportError,
    ExportReport,
    package,
    scan_dataset,
)
from loraforge.export.flatten import BACKGROUNDS, FlattenError, FlattenReport, flatten_alpha
from loraforge.export.scan import (
    CAPTIONING_SUFFIXES,
    ScanEntry,
    ScanResult,
    scan_images,
)

__all__ = [
    "BACKGROUNDS",
    "CAPTIONING_SUFFIXES",
    "CAPTION_SUFFIX",
    "IMAGE_SUFFIXES",
    "DatasetScan",
    "ExportError",
    "ExportReport",
    "FlattenError",
    "FlattenReport",
    "ScanEntry",
    "ScanResult",
    "package",
    "flatten_alpha",
    "scan_dataset",
    "scan_images",
]
