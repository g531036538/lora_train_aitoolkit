from __future__ import annotations

from pathlib import Path

from loraforge.caption.base import CaptionContext
from loraforge.caption.checks import CheckResult, check_captions
from loraforge.models import DatasetKind, TargetModel


def ctx(target: TargetModel = TargetModel.SDXL, trigger: str = "my_tok") -> CaptionContext:
    return CaptionContext(target=target, kind=DatasetKind.CHARACTER, trigger=trigger)


def by_name(results: list[CheckResult]) -> dict[str, CheckResult]:
    return {r.image.name: r for r in results}


def test_every_image_exactly_one_result() -> None:
    pairs = [
        (Path("a.png"), "my_tok, 1girl, red dress"),
        (Path("b.png"), ""),
        (Path("c.png"), "my_tok, 1girl, blue hat"),
    ]
    results = check_captions(pairs, ctx())
    assert len(results) == len(pairs)
    assert {r.image.name for r in results} == {"a.png", "b.png", "c.png"}


def test_empty_caption_is_error() -> None:
    result = check_captions([(Path("empty.png"), "   ")], ctx())[0]
    assert result.code == "empty"
    assert result.severity == "error"
    assert "空打标" in result.message
    assert "必须重新生成" in result.message


def test_missing_trigger_is_error() -> None:
    result = check_captions([(Path("a.png"), "1girl, solo, red dress")], ctx())[0]
    assert result.code == "missing_trigger"
    assert result.severity == "error"
    assert '缺少触发词 "my_tok"' in result.message


def test_trigger_must_match_whole_word() -> None:
    result = check_captions([(Path("a.png"), "my_tokyo, 1girl, red dress")], ctx())[0]
    assert result.code == "missing_trigger"


def test_trigger_match_is_case_insensitive_and_ignores_weight() -> None:
    result = check_captions([(Path("a.png"), "(MY_TOK:1.2), 1girl, red dress")], ctx())[0]
    assert result.code == "ok"
    assert result.severity == "info"


def test_short_tags_are_warn() -> None:
    result = check_captions([(Path("a.png"), "my_tok, hat")], ctx())[0]
    assert result.code == "too_short"
    assert result.severity == "warn"
    assert "标签数只有 2 个" in result.message
    assert "信息密度" in result.message


def test_short_prose_is_warn() -> None:
    result = check_captions(
        [(Path("a.png"), "my_tok")],
        ctx(TargetModel.KREA2, trigger="my_tok"),
    )[0]
    assert result.code == "too_short"
    assert result.severity == "warn"
    assert "9 字符" in result.message or "8 字符" in result.message or "字符" in result.message


def test_ok_has_info_severity_and_tag_count() -> None:
    result = check_captions([(Path("a.png"), "my_tok, 1girl, solo, red dress")], ctx())[0]
    assert result.code == "ok"
    assert result.severity == "info"
    assert "打标正常" in result.message
    assert "4 个标签" in result.message


def test_ok_tag_count_uses_deduplicated_tags() -> None:
    """40 个 Tag_A + 40 个 tag_a 去重后只有 2 个，不该报「81 个标签」。"""
    text = "my_tok, " + ", ".join(["Tag_A"] * 40) + ", " + ", ".join(["tag_a"] * 40)
    result = check_captions([(Path("a.png"), text)], ctx())[0]
    assert result.code == "ok"
    assert "2 个标签" in result.message
    assert "重复标签已按一次计" in result.message
    assert "（81 个标签" not in result.message


def test_identical_captions_are_warned_on_every_image() -> None:
    text = "my_tok, 1girl, solo, red dress"
    results = check_captions(
        [(Path("a.png"), text), (Path("b.png"), text), (Path("c.png"), text)],
        ctx(),
    )
    assert all(r.code == "duplicate" and r.severity == "warn" for r in results)
    messages = {r.image.name: r.message for r in results}
    assert "b.png" in messages["a.png"] and "c.png" in messages["a.png"]
    assert "完全一致" in messages["a.png"]


def test_identical_prose_is_also_warned() -> None:
    text = "my_tok, a woman wearing a red dress in a cafe."
    results = check_captions(
        [(Path("a.png"), text), (Path("b.png"), text)],
        ctx(TargetModel.KREA2, trigger="my_tok"),
    )
    assert all(r.code == "duplicate" and r.severity == "warn" for r in results)


def test_high_frequency_tag_is_warned_with_removal_hint() -> None:
    pairs = [
        (Path(f"{i}.png"), f"my_tok, 1girl, solo, red dress, unique{i} hat")
        for i in range(4)
    ]
    results = check_captions(pairs, ctx())
    assert all(r.code == "duplicate" and r.severity == "warn" for r in results)
    assert all('"red dress"' in r.message for r in results)
    assert all(">80%" in r.message for r in results)
    assert all("丢弃列表" in r.message for r in results)


def test_tag_at_exactly_80_percent_is_not_flagged() -> None:
    pairs = [
        (Path(f"{i}.png"), f"my_tok, 1girl, solo, red dress, unique{i} hat")
        for i in range(4)
    ]
    pairs.append((Path("4.png"), "my_tok, 1girl, solo, blue coat, unique4 scarf"))
    results = check_captions(pairs, ctx())
    assert all("red dress" not in r.message for r in results)


def test_structural_tags_are_not_flagged_as_leaks() -> None:
    pairs = [
        (Path(f"{i}.png"), f"my_tok, 1girl, solo, unique{i} hat")
        for i in range(4)
    ]
    results = check_captions(pairs, ctx())
    assert all(r.code == "ok" for r in results)


def test_trigger_is_not_flagged_as_leak() -> None:
    pairs = [
        (Path(f"{i}.png"), f"my_tok, 1girl, solo, unique{i} hat")
        for i in range(4)
    ]
    results = check_captions(pairs, ctx())
    assert all('"my_tok"' not in r.message for r in results)


def test_single_problem_uses_its_own_code() -> None:
    # 缺触发词 + 过短：code 取更严重的 missing_trigger，message 两个问题都在。
    result = check_captions([(Path("a.png"), "red dress")], ctx())[0]
    assert result.code == "missing_trigger"
    assert result.severity == "error"
    assert "缺少触发词" in result.message
    assert "过短" in result.message
    assert "；" in result.message


def test_newline_separated_tags_are_counted() -> None:
    result = check_captions([(Path("a.png"), "my_tok\n1girl\nred dress")], ctx())[0]
    assert result.code == "ok"
    assert "3 个标签" in result.message


def test_empty_batch_returns_empty_list() -> None:
    assert check_captions([], ctx()) == []
