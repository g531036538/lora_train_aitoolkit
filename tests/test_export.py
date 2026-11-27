"""导出打包的契约测试：zip 结构、缺标与孤儿必须显式报出。"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from loraforge.export import ExportError, package, scan_dataset

CN = "mytoken, a photo of a person, standing in a garden"


def _make_dataset(scratch: Path) -> Path:
    """两张图，只有第一张有打标词。"""
    dataset = scratch / "my_dataset"
    dataset.mkdir()
    for name in ("frame_01.png", "frame_02.png"):
        (dataset / name).write_bytes(b"\x89PNG\r\n\x1a\n")
    (dataset / "frame_01.txt").write_text(CN + "\n", encoding="utf-8")
    return dataset


def _make_config(scratch: Path) -> Path:
    config = scratch / "train.yaml"
    config.write_text("job: extension\nconfig:\n  name: sdxl-lora-v1\n", encoding="utf-8")
    return config


def _names(report) -> set[str]:
    with zipfile.ZipFile(report.zip_path) as archive:
        return set(archive.namelist())


def test_package_layout_and_report(scratch: Path) -> None:
    dataset = _make_dataset(scratch)
    report = package(dataset, _make_config(scratch), scratch / "out.zip")

    assert report.images == 2
    assert report.captions == 1
    assert report.missing_caption == ["frame_02.png"]
    assert report.orphan_caption == []
    assert report.ambiguous == []
    assert report.ok is False  # 缺了标就不算干净

    names = _names(report)
    assert {"my_dataset/train.yaml", "my_dataset/frame_01.png", "my_dataset/frame_01.txt",
            "my_dataset/frame_02.png"} <= names
    # 缺标的图不带 txt 进去，避免云端把图配到别人的标注上
    assert "my_dataset/frame_02.txt" not in names


def test_orphan_caption_reported_not_packed(scratch: Path) -> None:
    dataset = _make_dataset(scratch)
    (dataset / "ghost.txt").write_text("mytoken, leftover\n", encoding="utf-8")

    report = package(dataset, _make_config(scratch), scratch / "out.zip")

    assert report.orphan_caption == ["ghost.txt"]
    assert report.ok is False
    assert "my_dataset/ghost.txt" not in _names(report)


def test_caption_is_renamed_to_image_stem(scratch: Path) -> None:
    """`frame_01.PNG` 配 `frame_01.txt`：zip 里统一叫 `frame_01.txt`，云端 caption_ext 才对得上。"""
    dataset = scratch / "ds"
    dataset.mkdir()
    (dataset / "frame_01.PNG").write_bytes(b"\x89PNG\r\n\x1a\n")
    (dataset / "frame_01.txt").write_text(CN + "\n", encoding="utf-8")

    report = package(dataset, _make_config(scratch), scratch / "out.zip")

    assert report.missing_caption == []
    assert "ds/frame_01.txt" in _names(report)
    assert "ds/frame_01.PNG" in _names(report)


def test_two_images_sharing_one_caption_is_ambiguous(scratch: Path) -> None:
    """`a.png` + `a.jpg` 抢同一份 `a.txt`：能打包，但必须报出来让人自己选。

    大小写撞名（A.PNG / a.png）在 Windows 上根本落不了盘，只有扩展名不同才可能同 stem。
    """
    dataset = scratch / "ds"
    dataset.mkdir()
    (dataset / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (dataset / "a.jpg").write_bytes(b"\xff\xd8\xff\xe0")
    (dataset / "a.txt").write_text(CN + "\n", encoding="utf-8")

    report = package(dataset, _make_config(scratch), scratch / "out.zip")

    assert report.ambiguous and "a.jpg" in report.ambiguous[0]
    assert report.ok is False


def test_include_images_false_keeps_only_captions_and_config(scratch: Path) -> None:
    dataset = _make_dataset(scratch)
    report = package(dataset, _make_config(scratch), scratch / "out.zip", include_images=False)

    names = _names(report)
    assert not [name for name in names if name.endswith(".png")]
    assert {"my_dataset/frame_01.txt", "my_dataset/train.yaml"} <= names


def test_empty_dataset_raises(scratch: Path) -> None:
    empty = scratch / "empty"
    empty.mkdir()
    with pytest.raises(ExportError, match="没有可训练的图"):
        package(empty, _make_config(scratch), scratch / "out.zip")


def test_missing_config_raises(scratch: Path) -> None:
    dataset = _make_dataset(scratch)
    with pytest.raises(ExportError, match="训练配置不存在"):
        package(dataset, scratch / "nope.yaml", scratch / "out.zip")


def test_scan_dataset_pairs_case_insensitively(scratch: Path) -> None:
    dataset = _make_dataset(scratch)
    scan = scan_dataset(dataset)
    assert scan.missing_caption == ["frame_02.png"]
    assert scan.orphan_caption == []
    assert scan.captions["frame_01"].name == "frame_01.txt"
