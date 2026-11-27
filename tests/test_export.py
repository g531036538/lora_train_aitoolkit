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
    assert {"train.yaml", "my_dataset/frame_01.png", "my_dataset/frame_01.txt",
            "my_dataset/frame_02.png"} <= names
    # 配置在根，不混进数据集目录
    assert "my_dataset/train.yaml" not in names
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
    assert {"my_dataset/frame_01.txt", "train.yaml"} <= names


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


def test_extra_files_are_normalized_to_lf(scratch: Path) -> None:
    """验收.md 这类附带文件进包前统一成 LF，Windows 的 CRLF 不能混进去。"""
    dataset = _make_dataset(scratch)
    report = package(
        dataset,
        _make_config(scratch),
        scratch / "out.zip",
        extra_files={"验收.md": "# 标题\r\n第一行\r第二行\n".encode("utf-8")},
    )

    assert report.extra_files == ["验收.md"]
    with zipfile.ZipFile(report.zip_path) as archive:
        blob = archive.read("验收.md")
    assert blob == "# 标题\n第一行\n第二行\n".encode("utf-8")
    assert b"\r" not in blob


def test_extra_file_collision_raises(scratch: Path) -> None:
    """附带文件与同一层已有条目撞名时不许静默覆盖。

    配置在根目录，所以撞名只可能发生在根：叫 ``train.yaml`` 的附带文件必须报错。
    数据集目录里的同名条目不算撞名 —— 那是不同层，各自合法。
    """
    dataset = _make_dataset(scratch)
    with pytest.raises(ExportError, match="撞名"):
        package(
            dataset,
            _make_config(scratch),
            scratch / "out.zip",
            extra_files={"train.yaml": b"x"},
        )


def test_config_and_extras_live_at_zip_root(scratch: Path) -> None:
    """Issue #72：根目录 = train.yaml + 验收.md；数据集目录里只有图与同名 .txt。"""
    dataset = _make_dataset(scratch)
    report = package(
        dataset,
        _make_config(scratch),
        scratch / "out.zip",
        extra_files={"验收.md": "清单\n".encode("utf-8")},
    )

    names = set(_names(report))
    assert {"train.yaml", "验收.md"} <= names
    assert "my_dataset/train.yaml" not in names
    assert "my_dataset/验收.md" not in names
    assert {name for name in names if name.startswith("my_dataset/")} == {
        "my_dataset/frame_01.png",
        "my_dataset/frame_01.txt",
        "my_dataset/frame_02.png",
    }


def test_extra_file_named_like_dataset_entry_is_allowed(scratch: Path) -> None:
    """根目录的 frame_01.txt 与数据集里的 frame_01.txt 是两层，不算撞名。"""
    dataset = _make_dataset(scratch)
    report = package(
        dataset,
        _make_config(scratch),
        scratch / "out.zip",
        extra_files={"frame_01.txt": b"root level\n"},
    )

    names = set(_names(report))
    assert "frame_01.txt" in names
    assert "my_dataset/frame_01.txt" in names
    with zipfile.ZipFile(report.zip_path) as archive:
        assert archive.read("frame_01.txt") == b"root level\n"
        assert archive.read("my_dataset/frame_01.txt") != b"root level\n"


def test_no_extra_files_by_default(scratch: Path) -> None:
    dataset = _make_dataset(scratch)
    report = package(dataset, _make_config(scratch), scratch / "out.zip")
    assert report.extra_files == []
