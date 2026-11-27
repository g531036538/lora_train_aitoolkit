"""三个 styler 的纯函数单测：不读图、不联网、不碰模型。"""

from __future__ import annotations

import json

import pytest

from loraforge.caption.base import CaptionContext
from loraforge.caption.stylers import (
    Krea2ProseStyler,
    MossPreciseStyler,
    SdxlTagsStyler,
    default_styler,
)
from loraforge.caption.stylers.moss import validate_bbox
from loraforge.models import DatasetKind, TargetModel


def ctx(target: TargetModel, *, trigger: str = "", kind: DatasetKind = DatasetKind.CHARACTER, language: str = "en") -> CaptionContext:
    return CaptionContext(target=target, kind=kind, trigger=trigger, language=language)


# ---------------------------------------------------------------------------
# SDXL：tags
# ---------------------------------------------------------------------------


def test_sdxl_tags_trigger_first_and_canonical() -> None:
    raw = "A photo of a Girl, (long black hair:1.2), blue_eyes, masterpiece, best quality, 8k, girl, outdoors"
    out = SdxlTagsStyler().render(raw, ctx(TargetModel.SDXL, trigger="MyTok"))
    tags = out.split(", ")
    assert tags[0] == "MyTok"
    assert "1girl" in tags
    assert "blue eyes" in tags
    assert "long black hair" in tags  # SDXL 不删身份描述
    assert "masterpiece" not in tags
    assert "8k" not in tags
    assert all(t == t.lower() for t in tags[1:])
    assert len(tags) == len(set(tags))


def test_sdxl_tags_accepts_tags_argument_and_empty_is_error() -> None:
    out = SdxlTagsStyler().render("", ctx(TargetModel.SDXL, trigger="tok"), tags=["A", "B"])
    assert out == "tok, a, b"
    with pytest.raises(ValueError):
        SdxlTagsStyler().render("masterpiece, best quality", ctx(TargetModel.SDXL))


def test_sdxl_drops_absurdres_like_other_quality_tags() -> None:
    out = SdxlTagsStyler().render("1girl, absurdres, huge filesize, smile", ctx(TargetModel.SDXL, trigger="tok"))
    assert out == "tok, 1girl, huge filesize, smile"


# ---------------------------------------------------------------------------
# Krea2：prose
# ---------------------------------------------------------------------------


def test_krea2_one_sentence_and_identity_stripped() -> None:
    raw = (
        "a photo of a 30 year old woman with long black hair and blue eyes and pale skin "
        "and a round face, close-up, smiling, wearing a red dress, golden hour, in a cafe"
    )
    out = Krea2ProseStyler().render(raw, ctx(TargetModel.KREA2, trigger="my_tok"))
    assert out.startswith("my_tok, ")
    assert out.endswith(".")
    assert "a photo of a woman" in out
    assert "close-up" in out
    assert "smiling" in out
    assert "red dress" in out
    assert "cafe" in out
    for forbidden in ("black hair", "blue eyes", "pale skin", "round face", "30 year old"):
        assert forbidden not in out
    # 子句顺序保持后端给的顺序，不再按模板重排
    assert out.index("close-up") < out.index("smiling") < out.index("red dress") < out.index("cafe")


def test_krea2_trigger_case_is_preserved() -> None:
    out = Krea2ProseStyler().render("smiling, wearing a red dress", ctx(TargetModel.KREA2, trigger="MyTok"))
    assert out.startswith("MyTok, ")


def test_krea2_does_not_insert_subject() -> None:
    """有人物线索也不补主语：后端没写主语，styler 就不该替它编一个。"""
    out = Krea2ProseStyler().render("smiling, wearing a red dress", ctx(TargetModel.KREA2))
    assert out == "Smiling, wearing a red dress."
    assert "a single" not in out


def test_krea2_keeps_pronoun_and_clause_order() -> None:
    raw = "She is wearing a white hoodie, sitting on a sofa, in a bright living room."
    out = Krea2ProseStyler().render(raw, ctx(TargetModel.KREA2, trigger="tok"))
    assert out == "tok, she is wearing a white hoodie, sitting on a sofa, in a bright living room."


def test_krea2_dedupes_repeated_clause_without_reordering() -> None:
    raw = "white hoodie and blue jeans, white hoodie and blue jeans, sitting on a sofa, sky"
    out = Krea2ProseStyler().render(raw, ctx(TargetModel.KREA2, trigger="tok"))
    assert out == "tok, white hoodie and blue jeans, sitting on a sofa, sky."


def test_krea2_only_identity_raises() -> None:
    with pytest.raises(ValueError, match="身份"):
        Krea2ProseStyler().render("long black hair, blue eyes, pale skin", ctx(TargetModel.KREA2))


def test_krea2_strips_identity_in_underscore_form() -> None:
    """下划线写法（Danbooru 形状）同样算身份，不能被漏进 .txt。"""
    raw = "long_hair, blue_eyes, pale_skin, red coat, rain"
    out = Krea2ProseStyler().render(raw, ctx(TargetModel.KREA2, trigger="sks_person"))
    assert out == "sks_person, red coat, rain."


def test_krea2_keeps_composition_words() -> None:
    """full body 里的 body 是构图，不是身份：整句必须保留。"""
    raw = "full body, long hair, red dress"
    out = Krea2ProseStyler().render(raw, ctx(TargetModel.KREA2, trigger="sks_person"))
    assert out == "sks_person, full body, red dress."


def test_krea2_keeps_count_and_crowd_tags() -> None:
    """1girl / solo / mature female 描述构图与人群，换装时不该漂，保留。"""
    raw = "1girl, solo, mature female, red dress"
    out = Krea2ProseStyler().render(raw, ctx(TargetModel.KREA2, trigger="sks_person"))
    assert out == "sks_person, 1girl, solo, mature female, red dress."


def test_krea2_empty_raises() -> None:
    with pytest.raises(ValueError):
        Krea2ProseStyler().render("   ", ctx(TargetModel.KREA2))


# ---------------------------------------------------------------------------
# MOSS：prose_precise
# ---------------------------------------------------------------------------


def test_moss_four_sections_and_quoted_text_roles() -> None:
    raw = (
        'a poster on a brick wall, with the text "GRAND OPENING" at the top '
        'and "Saturday 10am" at the bottom'
    )
    out = MossPreciseStyler().render(raw, ctx(TargetModel.MOSS, trigger="my_tok"))
    lines = out.splitlines()
    assert lines[0].startswith("Object: ")
    assert lines[1] == "In-image text:"
    assert lines[2] == '"GRAND OPENING" (role: header)'
    assert lines[3] == '"Saturday 10am" (role: footer)'
    assert any(line.startswith("Appearance: ") for line in lines)
    assert any(line.startswith("Composition: ") for line in lines)


def test_moss_identity_stripped_but_quoted_text_untouched() -> None:
    raw = 'a 30 year old woman with long black hair, holding a sign that says "long black hair"'
    out = MossPreciseStyler().render(raw, ctx(TargetModel.MOSS, trigger="tok"))
    assert '"long black hair" (role: ' in out
    assert "single woman" in out
    assert "30 year old" not in out
    assert "with long black hair" not in out


def test_moss_json_bbox_and_layout() -> None:
    raw = json.dumps(
        {
            "object": "a 30 year old woman with long black hair, holding a sign",
            "texts": [{"text": "SALE", "role": "header", "bbox": [-64, -128, 0, 128]}],
            "appearance": "wearing a red dress",
            "composition": "close-up, centered",
        }
    )
    out = MossPreciseStyler().render(raw, ctx(TargetModel.MOSS, trigger="tok"))
    data = json.loads(out)
    assert data["object"] == "tok, a woman, holding a sign"
    assert data["texts"] == [{"text": "SALE", "role": "header", "bbox": [-64, -128, 0, 128]}]
    assert data["appearance"] == "wearing a red dress"


def test_moss_labels_stay_english() -> None:
    out = MossPreciseStyler().render(
        "a red apple", ctx(TargetModel.MOSS, trigger="tok", language="zh")
    )
    assert out.startswith("Object: tok, a red apple")
    assert "In-image text: none" in out
    assert "Appearance: not described" in out
    assert "Composition: not described" in out


def test_moss_empty_raises() -> None:
    with pytest.raises(ValueError):
        MossPreciseStyler().render("", ctx(TargetModel.MOSS))


def test_moss_invalid_json_fallback_is_cleaned() -> None:
    """以 ``{`` 开头但解析不了的 JSON：清掉括号与段名前缀，残骸不许进打标词。"""
    raw = (
        '{Object: sks_person, a red poster, In-image text: the word "SALE" at the top, '
        "Appearance: soft lighting, Composition: yellow background"
    )
    out = MossPreciseStyler().render(
        raw, ctx(TargetModel.MOSS, trigger="sks_person", kind=DatasetKind.TEXT_LAYOUT)
    )
    assert out.splitlines() == [
        "Object: sks_person, a red poster",
        "In-image text:",
        '"SALE" (role: header)',
        "Appearance: soft lighting",
        "Composition: yellow background",
    ]
    assert "{" not in out and "}" not in out
    assert "Appearance: Appearance" not in out


def test_moss_unparseable_json_without_content_raises() -> None:
    with pytest.raises(ValueError):
        MossPreciseStyler().render("{[", ctx(TargetModel.MOSS))


@pytest.mark.parametrize(
    "bbox",
    [
        [-64, -128, 0],  # 长度不对
        [0, 0, 16.0, 16],  # 不是整数
        [16, 0, 0, 16],  # y_min >= y_max
        [0, 16, 16, 0],  # x_min >= x_max
        [0, 0, 10, 16],  # 边长不是 16 的倍数
    ],
)
def test_validate_bbox_rejects_bad_values(bbox: list[object]) -> None:
    with pytest.raises(ValueError):
        validate_bbox(bbox)


def test_validate_bbox_accepts_negative_center_origin() -> None:
    assert validate_bbox([-64, -128, 0, 128]) == [-64, -128, 0, 128]


# ---------------------------------------------------------------------------
# 默认映射
# ---------------------------------------------------------------------------


def test_default_styler_mapping() -> None:
    assert isinstance(default_styler(TargetModel.SDXL), SdxlTagsStyler)
    assert isinstance(default_styler(TargetModel.KREA2), Krea2ProseStyler)
    assert isinstance(default_styler(TargetModel.MOSS), MossPreciseStyler)
    assert isinstance(default_styler("moss"), MossPreciseStyler)


def test_default_styler_unknown_target() -> None:
    with pytest.raises(KeyError):
        default_styler("not-a-model")
