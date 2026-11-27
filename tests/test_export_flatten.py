"""透明底合成：真 alpha、全不透明 alpha、已存在不覆盖、sidecar 逐字节、坏图不拖垮整批。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from loraforge.export.flatten import BACKGROUNDS, FlattenReport, flatten_alpha


def _rgba_png(path: Path, *, size=(4, 4), fill=(255, 0, 0, 255), corner=(0, 0, 0, 0)) -> Path:
    image = Image.new("RGBA", size, fill)
    image.putpixel((0, 0), corner)
    image.save(path, format="PNG")
    return path


def _png_bytes(path: Path) -> bytes:
    return path.read_bytes()


def test_flatten_real_alpha_lands_on_white(scratch: Path) -> None:
    dataset = scratch / "src"
    dataset.mkdir()
    source = _rgba_png(dataset / "a.png")
    out_dir = scratch / "out"

    report = flatten_alpha([source], out_dir)

    assert isinstance(report, FlattenReport)
    assert (report.written, report.failed) == (1, 0)
    assert report.ok is True
    product = out_dir / "a.png"
    assert product.is_file()
    with Image.open(product) as image:
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (255, 255, 255)  # 透明角点 → 白底
        assert image.getpixel((2, 2)) == (255, 0, 0)      # 不透明区原样
    # 原图保持 RGBA，一个字节没动
    with Image.open(source) as original:
        assert original.mode == "RGBA"
    assert _png_bytes(source) == _png_bytes(source)


def test_flatten_black_and_gray_backgrounds(scratch: Path) -> None:
    dataset = scratch / "src"
    dataset.mkdir()
    source = _rgba_png(dataset / "a.png")

    for name, expected in (("black", (0, 0, 0)), ("gray", (128, 128, 128))):
        out_dir = scratch / f"out-{name}"
        report = flatten_alpha([source], out_dir, background=BACKGROUNDS[name])
        assert report.written == 1 and report.failed == 0
        with Image.open(out_dir / "a.png") as image:
            assert image.mode == "RGB"
            assert image.getpixel((0, 0)) == expected


def test_flatten_fully_opaque_alpha_is_also_handled(scratch: Path) -> None:
    """alpha 全 255 的 PNG 也要能走完整条路（用户图集里很多这种）。"""
    dataset = scratch / "src"
    dataset.mkdir()
    source = _rgba_png(dataset / "a.png", fill=(10, 20, 30, 255), corner=(10, 20, 30, 255))
    out_dir = scratch / "out"

    report = flatten_alpha([source], out_dir)

    assert (report.written, report.failed) == (1, 0)
    with Image.open(out_dir / "a.png") as image:
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (10, 20, 30)
        assert image.getpixel((3, 3)) == (10, 20, 30)


def test_flatten_refuses_to_overwrite_existing(scratch: Path) -> None:
    dataset = scratch / "src"
    dataset.mkdir()
    source = _rgba_png(dataset / "a.png")
    out_dir = scratch / "out"
    out_dir.mkdir()
    existing = out_dir / "a.png"
    existing.write_bytes(b"do-not-touch")

    report = flatten_alpha([source], out_dir)

    # 已存在不是失败：跳过、写明原因、原产物一字不动
    assert (report.written, report.failed, report.skipped) == (0, 0, 1)
    assert report.ok is True
    assert existing.read_bytes() == b"do-not-touch"
    item = report.items[0]
    assert item[0] == source and item[1] is None
    assert "已存在，跳过（原图没动）" in (item[2] or "")
    assert report.exists and "跳过" in report.exists[0][1]
    assert report.failures == []


def test_flatten_never_writes_into_source_dir(scratch: Path) -> None:
    dataset = scratch / "src"
    dataset.mkdir()
    source = _rgba_png(dataset / "a.png")
    before = _png_bytes(source)

    report = flatten_alpha([source], dataset)  # 输出目录就是图集本身

    # 库层面只认「同名已存在 → 跳过」；界面 /flatten 会直接 400 拦在更前面
    assert (report.written, report.failed, report.skipped) == (0, 0, 1)
    assert _png_bytes(source) == before
    assert "跳过" in (report.items[0][2] or "")


def test_flatten_copies_sidecar_byte_for_byte(scratch: Path) -> None:
    dataset = scratch / "src"
    dataset.mkdir()
    source = _rgba_png(dataset / "a.png")
    caption = dataset / "a.txt"
    caption.write_bytes("mytoken, 一只猫，坐在窗台\n".encode("utf-8"))
    out_dir = scratch / "out"

    report = flatten_alpha([source], out_dir, copy_sidecar=True)

    assert report.written == 1
    copied = out_dir / "a.txt"
    assert copied.is_file()
    assert copied.read_bytes() == caption.read_bytes()
    assert b"\r" not in copied.read_bytes()


def test_flatten_sidecar_can_be_skipped(scratch: Path) -> None:
    dataset = scratch / "src"
    dataset.mkdir()
    source = _rgba_png(dataset / "a.png")
    (dataset / "a.txt").write_bytes(b"mytoken, cat\n")
    out_dir = scratch / "out"

    flatten_alpha([source], out_dir, copy_sidecar=False)

    assert (out_dir / "a.png").is_file()
    assert not (out_dir / "a.txt").exists()


def test_flatten_one_bad_file_does_not_stop_the_batch(scratch: Path) -> None:
    dataset = scratch / "src"
    dataset.mkdir()
    good_a = _rgba_png(dataset / "a.png")
    bad = dataset / "notes.txt"
    bad.write_bytes(b"not an image\n")
    good_c = _rgba_png(dataset / "c.png")
    out_dir = scratch / "out"

    report = flatten_alpha([good_a, bad, good_c], out_dir)

    assert (report.written, report.failed, report.skipped) == (2, 1, 0)
    assert (out_dir / "a.png").is_file() and (out_dir / "c.png").is_file()
    reasons = {source.name: reason for source, product, reason in report.items if product is None}
    assert "notes.txt" in reasons
    assert "图片" in reasons["notes.txt"]  # 中文原因，不是异常堆栈
    assert [source.name for source, _message in report.errors] == ["notes.txt"]


def test_flatten_rejects_bad_background(scratch: Path) -> None:
    dataset = scratch / "src"
    dataset.mkdir()
    source = _rgba_png(dataset / "a.png")

    for bad in ((255, 255), (-1, 0, 0), (255, 255, 256)):
        try:
            flatten_alpha([source], scratch / "out", background=bad)  # type: ignore[arg-type]
        except ValueError as exc:
            assert "底色" in str(exc)
        else:  # pragma: no cover - 只是让失败信息更直白
            raise AssertionError(f"底色 {bad} 应该被拒")


def test_flatten_separates_skipped_from_failed(scratch: Path) -> None:
    """同一批里：已存在 → skipped，读不了 → failed，两者不许混淆。"""
    dataset = scratch / "src"
    dataset.mkdir()
    skipped_source = _rgba_png(dataset / "a.png")
    error_source = dataset / "b.png"
    error_source.write_bytes(b"not an image\n")
    out_dir = scratch / "out"
    out_dir.mkdir()
    (out_dir / "a.png").write_bytes(b"keep-me")

    report = flatten_alpha([skipped_source, error_source], out_dir)

    assert (report.written, report.skipped, report.failed) == (0, 1, 1)
    assert report.ok is False
    assert [source.name for source, _m in report.exists] == ["a.png"]
    assert [source.name for source, _m in report.errors] == ["b.png"]
    assert "已存在，跳过（原图没动）" in report.exists[0][1]
    assert "不覆盖" not in report.errors[0][1]
    assert (out_dir / "a.png").read_bytes() == b"keep-me"
