"""打标输出黄金基线：styler 是纯函数，输出逐字节钉住。

每个用例两个文件，放在 ``tests/golden/caption/``：

- ``<名字>.in.txt``：输入（后端返回的原始打标词 / 描述）
- ``<名字>.out.txt``：期望输出（``render()`` 的返回值 + 末尾单个换行）

``<名字>`` 以目标模型开头（``sdxl-`` / ``krea2-`` / ``moss-``），见 ``_NAME_TARGETS``。
比对是逐字节的，不做任何换行归一——换行本身就是被测项之一。

输出变了只有两种可能：有人改了 styler（要在这里显式同意），或者有人改错了黄金文件。
两种都应该在失败信息里一眼看见「期望」和「实际」两段。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loraforge.caption.base import CaptionContext
from loraforge.caption.stylers import default_styler
from loraforge.models import DatasetKind, TargetModel

GOLDEN_DIR = Path(__file__).parent / "golden" / "caption"
IN_SUFFIX = ".in.txt"
OUT_SUFFIX = ".out.txt"

# 三个目标都用同一个触发词，避免每个用例自己发明身份串。
TRIGGER = "sks_person"

# 文件名前缀 → 目标模型。用例改名时把这里一起想清楚，别让用例跑错目标。
_NAME_TARGETS: dict[str, TargetModel] = {
    "sdxl": TargetModel.SDXL,
    "krea2": TargetModel.KREA2,
    "moss": TargetModel.MOSS,
}

CASES: dict[str, CaptionContext] = {
    "sdxl-character-basic": CaptionContext(
        target=TargetModel.SDXL, kind=DatasetKind.CHARACTER, trigger=TRIGGER
    ),
    "sdxl-character-weights": CaptionContext(
        target=TargetModel.SDXL, kind=DatasetKind.CHARACTER, trigger=TRIGGER
    ),
    "sdxl-concept-long": CaptionContext(
        target=TargetModel.SDXL, kind=DatasetKind.CONCEPT, trigger=TRIGGER
    ),
    "krea2-style-outfit": CaptionContext(
        target=TargetModel.KREA2, kind=DatasetKind.STYLE, trigger=TRIGGER
    ),
    "moss-text-layout-json": CaptionContext(
        target=TargetModel.MOSS, kind=DatasetKind.TEXT_LAYOUT, trigger=TRIGGER
    ),
}


def _names(suffix: str) -> set[str]:
    return {path.name[: -len(suffix)] for path in GOLDEN_DIR.glob("*" + suffix)}


def _golden_path(name: str, suffix: str) -> Path:
    return GOLDEN_DIR / f"{name}{suffix}"


def _first_difference(expected: bytes, actual: bytes) -> str:
    limit = min(len(expected), len(actual))
    for index in range(limit):
        if expected[index] != actual[index]:
            return f"第 {index} 字节：期望 {expected[index:index + 1]!r}，实际 {actual[index:index + 1]!r}"
    if len(expected) != len(actual):
        longer = "期望" if len(expected) > len(actual) else "实际"
        return f"前 {limit} 字节相同，但{longer}多出 {abs(len(expected) - len(actual))} 字节"
    return "两段完全相同（不应出现在失败分支）"


def test_golden_pairs_are_complete() -> None:
    inputs = _names(IN_SUFFIX)
    outputs = _names(OUT_SUFFIX)
    assert inputs, f"{GOLDEN_DIR} 下一个黄金用例都没有"
    assert inputs == outputs, (
        f"黄金文件没有配对：只有 .in.txt 的 {sorted(inputs - outputs)}；"
        f"只有 .out.txt 的 {sorted(outputs - inputs)}"
    )


def test_golden_cases_cover_every_target_and_match_file_names() -> None:
    names = _names(IN_SUFFIX)
    assert names == set(CASES), (
        f"目录里的用例和 CASES 登记不一致："
        f"只在目录 {sorted(names - set(CASES))}；只在 CASES {sorted(set(CASES) - names)}"
    )
    for name, ctx in CASES.items():
        prefix = name.split("-", 1)[0]
        expected_target = _NAME_TARGETS.get(prefix)
        assert expected_target is not None, f"{name} 的前缀 {prefix!r} 不在 _NAME_TARGETS 里"
        assert ctx.target is expected_target, f"{name} 的目标是 {ctx.target}，文件名却写的是 {prefix}"
        assert ctx.trigger == TRIGGER, f"{name} 的触发词应统一为 {TRIGGER!r}"
    covered = {ctx.target for ctx in CASES.values()}
    all_targets = set(_NAME_TARGETS.values())
    assert covered == all_targets, f"还有目标模型没有黄金用例：{sorted(all_targets - covered)}"


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_output_matches(name: str) -> None:
    ctx = CASES[name]
    # 文件末尾的单换行是文本文件约定，不属于 caption 本身。
    raw = _golden_path(name, IN_SUFFIX).read_text(encoding="utf-8").rstrip("\n")
    expected = _golden_path(name, OUT_SUFFIX).read_bytes()
    actual = default_styler(ctx.target).render(raw, ctx).encode("utf-8") + b"\n"

    if actual != expected:
        pytest.fail(
            f"黄金基线不一致：{name}\n"
            f"文件：{_golden_path(name, OUT_SUFFIX)}\n"
            f"\n--- 期望 ---\n{expected.decode('utf-8')}"
            f"\n--- 实际 ---\n{actual.decode('utf-8')}"
            f"\n--- 首个差异 ---\n{_first_difference(expected, actual)}"
        )


@pytest.mark.parametrize(
    "path", sorted(GOLDEN_DIR.glob("*")), ids=lambda path: path.name
)
def test_golden_files_are_lf_with_single_trailing_newline(path: Path) -> None:
    data = path.read_bytes()
    assert b"\r" not in data, f"{path.name} 含 CR，黄金文件必须是 LF"
    assert data.endswith(b"\n"), f"{path.name} 末尾缺少换行"
    assert not data.endswith(b"\n\n"), f"{path.name} 末尾多于一个换行"
