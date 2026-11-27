"""整批体检 ``analyze_dataset`` 的单测：不联网、不依赖真实图集，图在临时目录里生成。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from loraforge.caption.base import CaptionContext
from loraforge.caption.dataset import analyze_dataset
from loraforge.models import DatasetKind, TargetModel

HOMOGENEOUS = "my_tok, mecha, robot, science fiction, standing"


def ctx(
    target: TargetModel = TargetModel.SDXL,
    trigger: str = "my_tok",
    kind: DatasetKind = DatasetKind.CHARACTER,
) -> CaptionContext:
    return CaptionContext(target=target, kind=kind, trigger=trigger)


def make_image(
    path: Path, *, size: tuple[int, int] = (640, 640), alpha: int | None = None
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if alpha is None:
        Image.new("RGB", size, (90, 110, 130)).save(path)
    else:
        Image.new("RGBA", size, (90, 110, 130, alpha)).save(path)
    return path


def make_batch(
    tmp_path: Path,
    count: int,
    *,
    caption: str = HOMOGENEOUS,
    size: tuple[int, int] = (640, 640),
    alpha: int | None = None,
) -> list[tuple[Path, str]]:
    return [
        (make_image(tmp_path / f"{index:03d}.png", size=size, alpha=alpha), caption)
        for index in range(count)
    ]


def codes(issues: list) -> list[str]:
    return [issue.code for issue in issues]


def test_homogeneous_batch_reports_one_issue() -> None:
    # 不需要真实文件：同质判定只读打标词，读不到的图会被跳过（不是坏文件检查）。
    pairs = [(Path(f"{index:03d}.png"), HOMOGENEOUS) for index in range(86)]
    issues = analyze_dataset(pairs, ctx())
    assert codes(issues) == ["ubiquitous_tag"]
    issue = issues[0]
    assert issue.severity == "warn"
    assert "86" in issue.message
    assert '"mecha"' in issue.message and '"robot"' in issue.message
    # files 只给样例，最多 sample 个
    assert len(issue.files) == 20


def test_sample_caps_files() -> None:
    pairs = [(Path(f"{index:03d}.png"), HOMOGENEOUS) for index in range(30)]
    issue = analyze_dataset(pairs, ctx(), sample=3)[0]
    assert len(issue.files) == 3
    assert "30" in issue.message


def test_ubiquitous_tag_needs_five_images() -> None:
    pairs = [(Path(f"{index}.png"), HOMOGENEOUS) for index in range(4)]
    assert analyze_dataset(pairs, ctx()) == []


def test_tag_at_exactly_threshold_is_not_flagged() -> None:
    pairs = [(Path(f"{index}.png"), "my_tok, red dress, unique handbag") for index in range(4)]
    pairs.append((Path("4.png"), "my_tok, blue coat"))
    assert codes(analyze_dataset(pairs, ctx())) == []


def test_all_tags_at_full_coverage_are_flagged() -> None:
    pairs = [(Path(f"{index}.png"), "my_tok, red dress") for index in range(5)]
    issues = analyze_dataset(pairs, ctx())
    assert codes(issues) == ["ubiquitous_tag"]
    assert '"red dress"' in issues[0].message
    assert '"my_tok"' not in issues[0].message  # 触发词是故意的，不报


def test_prose_target_does_not_report_ubiquitous_tag() -> None:
    text = "my_tok, a woman wearing a red dress in a cafe."
    pairs = [(Path(f"{index}.png"), text) for index in range(6)]
    issues = analyze_dataset(pairs, ctx(TargetModel.KREA2))
    assert issues == []


def test_truly_transparent_png_is_flagged_once(tmp_path: Path) -> None:
    pairs = [
        (make_image(tmp_path / "0.png", alpha=0), "my_tok, red dress"),
        (make_image(tmp_path / "1.png"), "my_tok, blue coat"),
        (make_image(tmp_path / "2.png"), "my_tok, green hat"),
    ]
    issues = analyze_dataset(pairs, ctx())
    assert codes(issues) == ["alpha_channel"]
    issue = issues[0]
    assert issue.files == ("0.png",)
    assert "1 张" in issue.message
    assert "3 张图" in issue.message


def test_opaque_alpha_channel_is_not_flagged(tmp_path: Path) -> None:
    """PNG 带 alpha 通道但全 255，不是问题。"""
    pairs = [
        (make_image(tmp_path / f"{index}.png", alpha=255), HOMOGENEOUS)
        for index in range(3)
    ]
    assert analyze_dataset(pairs, ctx()) == []


def test_small_resolution_is_flagged_one_time(tmp_path: Path) -> None:
    pairs = [
        (make_image(tmp_path / f"{index}.png", size=(400, 640)), f"my_tok, tag {index}")
        for index in range(3)
    ]
    issues = analyze_dataset(pairs, ctx())
    assert codes(issues) == ["small_resolution"]
    assert "3 张" in issues[0].message
    assert "512" in issues[0].message


def test_max_one_issue_per_code(tmp_path: Path) -> None:
    pairs = [
        (make_image(tmp_path / f"{index}.png", size=(400, 400), alpha=0 if index < 2 else None), HOMOGENEOUS)
        for index in range(6)
    ]
    issues = analyze_dataset(pairs, ctx(), sample=2)
    assert sorted(codes(issues)) == ["alpha_channel", "small_resolution", "ubiquitous_tag"]
    assert all(len(issue.files) <= 2 for issue in issues)


def test_alpha_channel_survives_empty_captions(tmp_path: Path) -> None:
    # 打标词全空时，透明通道仍然要报：判的是图片本身，不是打标词。
    pairs = [
        (make_image(tmp_path / f"{index}.png", alpha=0), "")
        for index in range(3)
    ] + [
        (make_image(tmp_path / f"{index}.png"), "")
        for index in range(3, 6)
    ]
    issues = analyze_dataset(pairs, ctx())
    assert codes(issues) == ["alpha_channel"]
    assert issues[0].files == ("0.png", "1.png", "2.png")
    assert "3 张图" in issues[0].message


def test_small_resolution_survives_empty_captions(tmp_path: Path) -> None:
    pairs = [
        (make_image(tmp_path / f"{index}.png", size=(300, 640)), "")
        for index in range(6)
    ]
    issues = analyze_dataset(pairs, ctx())
    assert codes(issues) == ["small_resolution"]
    assert "6 张图" in issues[0].message


def test_empty_captions_do_not_report_ubiquitous_tag(tmp_path: Path) -> None:
    # 空词不进同质统计分母，所以即便整批全空也不报 ubiquitous_tag。
    pairs = [
        (make_image(tmp_path / f"{index}.png"), "")
        for index in range(6)
    ]
    assert analyze_dataset(pairs, ctx()) == []


def test_empty_batch_returns_empty_list() -> None:
    assert analyze_dataset([], ctx()) == []
