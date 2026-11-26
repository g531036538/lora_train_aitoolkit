from __future__ import annotations

from pathlib import Path

from loraforge.caption.base import CaptionContext
from loraforge.caption.checks import CheckResult, check_captions
from loraforge.caption.stylers import default_styler
from loraforge.models import DatasetKind, TargetModel


def ctx(target: TargetModel, trigger: str = "my_tok") -> CaptionContext:
    return CaptionContext(target=target, kind=DatasetKind.CHARACTER, trigger=trigger)


def by_name(results: list[CheckResult]) -> dict[str, CheckResult]:
    return {result.image.name: result for result in results}


KREA_RAW = (
    "my_tok, a 25 year old woman with long black hair and pale skin, "
    "blue eyes, round face, wearing a red dress, smiling in a cafe"
)


def test_krea2_identity_removed_is_warn_with_category_counts() -> None:
    context = ctx(TargetModel.KREA2)
    final = default_styler(TargetModel.KREA2).render(KREA_RAW, context)
    result = check_captions(
        [(Path("a.png"), final)],
        context,
        raw={Path("a.png"): KREA_RAW},
    )[0]

    assert result.code == "identity_removed"
    assert result.severity == "warn"
    assert "已移除 5 项身份描述" in result.message
    assert "脸型 1" in result.message
    assert "发型 1" in result.message
    assert "肤色 1" in result.message
    assert "眼睛 1" in result.message
    assert "年龄 1" in result.message
    assert "身份交给触发词" in result.message


def test_krea2_identity_still_present_is_not_reported() -> None:
    context = ctx(TargetModel.KREA2)
    final = "my_tok, a single woman with blue eyes, wearing a red dress, smiling in a cafe"
    result = check_captions(
        [(Path("a.png"), final)],
        context,
        raw={Path("a.png"): KREA_RAW},
    )[0]

    assert result.code == "identity_removed"
    assert "眼睛 1" not in result.message
    assert "脸型 1" in result.message


def test_krea2_identity_warn_outranks_duplicate_but_stays_one_result() -> None:
    context = ctx(TargetModel.KREA2)
    final = default_styler(TargetModel.KREA2).render(KREA_RAW, context)
    pairs = [(Path("a.png"), final), (Path("b.png"), final)]
    results = check_captions(
        pairs,
        context,
        raw={Path("a.png"): KREA_RAW, Path("b.png"): KREA_RAW},
    )

    assert len(results) == 2
    assert [result.code for result in results] == ["identity_removed", "identity_removed"]
    assert all("打标完全一致" in result.message for result in results)


def test_krea2_identity_check_does_not_fire_without_raw() -> None:
    context = ctx(TargetModel.KREA2)
    final = default_styler(TargetModel.KREA2).render(KREA_RAW, context)
    result = check_captions([(Path("a.png"), final)], context)[0]
    assert result.code == "ok"

    pairs = [(Path("a.png"), final), (Path("b.png"), final)]
    duplicated = check_captions(pairs, context)
    assert [result.code for result in duplicated] == ["duplicate", "duplicate"]


def test_moss_unquoted_text_reports_line_numbers_and_is_warn() -> None:
    context = ctx(TargetModel.MOSS)
    raw = (
        "a poster that says GRAND OPENING\n"
        'header: "Saturday 10am"\n'
        "a sign that says WELCOME"
    )
    final = default_styler(TargetModel.MOSS).render(raw, context)
    result = check_captions(
        [(Path("poster.png"), final)],
        context,
        raw={Path("poster.png"): raw},
    )[0]

    assert result.code == "unquoted_text"
    assert result.severity == "warn"
    assert "第 1、3 行" in result.message
    assert "未加引号" in result.message
    assert "引号包住" in result.message


def test_moss_quoted_text_does_not_warn() -> None:
    context = ctx(TargetModel.MOSS)
    raw = (
        'header: "GRAND OPENING"\n'
        'footer: "Saturday 10am"\n'
        'the sign says: "WELCOME"'
    )
    final = default_styler(TargetModel.MOSS).render(raw, context)
    result = check_captions(
        [(Path("poster.png"), final)],
        context,
        raw={Path("poster.png"): raw},
    )[0]

    assert result.code == "ok"
    assert result.severity == "info"


def test_moss_unquoted_warn_outranks_duplicate_but_stays_one_result() -> None:
    context = ctx(TargetModel.MOSS)
    raw = "a poster that says GRAND OPENING"
    final = default_styler(TargetModel.MOSS).render(raw, context)
    pairs = [(Path("a.png"), final), (Path("b.png"), final)]
    results = check_captions(
        pairs,
        context,
        raw={Path("a.png"): raw, Path("b.png"): raw},
    )

    assert len(results) == 2
    assert [result.code for result in results] == ["unquoted_text", "unquoted_text"]
    assert all("打标完全一致" in result.message for result in results)


def test_default_raw_none_keeps_existing_ok_behavior() -> None:
    context = ctx(TargetModel.MOSS)
    final = default_styler(TargetModel.MOSS).render('header: "GRAND OPENING"', context)
    result = check_captions([(Path("a.png"), final)], context)[0]
    assert result.code == "ok"


def test_empty_and_missing_trigger_keep_their_severity_with_raw() -> None:
    context = ctx(TargetModel.KREA2)
    raw = {Path("a.png"): KREA_RAW}

    empty = check_captions([(Path("a.png"), "   ")], context, raw=raw)[0]
    assert (empty.code, empty.severity) == ("empty", "error")

    missing = check_captions(
        [(Path("a.png"), "a single woman, smiling, wearing a red dress in a cafe")],
        context,
        raw=raw,
    )[0]
    assert (missing.code, missing.severity) == ("missing_trigger", "error")


def test_raw_accepts_string_keys_and_only_matching_image_gets_new_check() -> None:
    context = ctx(TargetModel.KREA2)
    final = default_styler(TargetModel.KREA2).render(KREA_RAW, context)
    results = check_captions(
        [(Path("a.png"), final), (Path("b.png"), final)],
        context,
        raw={"a.png": KREA_RAW},
    )

    assert len(results) == 2
    by_image = by_name(results)
    assert by_image["a.png"].code == "identity_removed"
    assert by_image["b.png"].code == "duplicate"
