"""SDXL token_overflow 近似检查：边界、跨模型、与 duplicate 的优先级。"""

from __future__ import annotations

from pathlib import Path

from loraforge.caption.checks import (
    SDXL_SOFT_TAG_LIMIT,
    SDXL_SOFT_WORD_LIMIT,
    check_captions,
)
from loraforge.caption.base import CaptionContext
from loraforge.models import DatasetKind, TargetModel


def _ctx(target: TargetModel = TargetModel.SDXL, trigger: str = "my_tok"):
    return CaptionContext(target=target, kind=DatasetKind.CHARACTER, trigger=trigger)


def _caption_with_tags(total_tags: int) -> str:
    """构造恰好 total_tags 个逗号分隔标签（触发词占一个）。"""
    parts = ["my_tok"] + [f"t{i}" for i in range(total_tags - 1)]
    return ", ".join(parts)


def _caption_with_words(total_words: int, tag_count: int = 3) -> str:
    """构造 tag_count 个标签、刚好 total_words 个英文词。"""
    words = [f"w{i}" for i in range(total_words)]
    chunks = [words[i::tag_count] for i in range(tag_count)]
    return ", ".join(" ".join(chunk) for chunk in chunks)


def _one(caption: str, ctx: CaptionContext, image: Path = Path("a.png")):
    results = check_captions([(image, caption)], ctx)
    assert len(results) == 1
    return results[0]


def test_74_tags_do_not_report():
    result = _one(_caption_with_tags(74), _ctx())
    assert result.code == "ok"


def test_75_tags_do_not_report():
    result = _one(_caption_with_tags(75), _ctx())
    assert result.code == "ok"


def test_76_tags_report_token_overflow():
    result = _one(_caption_with_tags(76), _ctx())
    assert result.code == "token_overflow"
    assert result.severity == "warn"
    assert "76" in result.message
    assert str(SDXL_SOFT_TAG_LIMIT) in result.message
    assert "近似" in result.message


def test_60_words_do_not_report():
    result = _one(_caption_with_words(60), _ctx(trigger=""))
    assert result.code == "ok"


def test_61_words_report_token_overflow():
    result = _one(_caption_with_words(61), _ctx(trigger=""))
    assert result.code == "token_overflow"
    assert result.severity == "warn"
    assert "61" in result.message
    assert str(SDXL_SOFT_WORD_LIMIT) in result.message


def test_krea2_long_caption_does_not_report():
    result = _one(_caption_with_tags(76), _ctx(TargetModel.KREA2))
    assert result.code != "token_overflow"


def test_moss_long_caption_does_not_report():
    result = _one(_caption_with_tags(76), _ctx(TargetModel.MOSS))
    assert result.code != "token_overflow"


def test_overflow_beats_duplicate_but_still_one_result():
    caption = _caption_with_tags(76)
    results = check_captions([(Path("a.png"), caption), (Path("b.png"), caption)], _ctx())
    assert len(results) == 2
    for result in results:
        assert result.code == "token_overflow"
        assert result.severity == "warn"
        assert "76" in result.message
        assert "打标词完全相同" in result.message


def test_76_identical_tags_dedupe_to_no_report():
    """主管实测组：76 个 same_tag + 触发词，去重后只剩 2 个，不该报溢出。"""
    caption = ", ".join(["my_tok"] + ["same_tag"] * 76)
    result = _one(caption, _ctx())
    assert result.code == "ok"
    assert "超过" not in result.message


def test_dedupe_still_reports_when_unique_tags_exceed_limit():
    """去重只折叠重复项，不放过真正超限的 76 个不同标签。"""
    unique = [f"t{i}" for i in range(76)]
    caption = ", ".join(unique + unique)
    result = _one(caption, _ctx(trigger=""))
    assert result.code == "token_overflow"
    assert "76" in result.message
    assert "重复标签已按一次计" in result.message
