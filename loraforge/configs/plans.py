"""配置侧读口：显存档位与默认方案。

接口见 ``docs/ARCHITECTURE.md`` §8（字段名冻结）。界面不许硬编码默认数值，
一律从这里读；数值要改就改这里的数据，不要动字段名。

出处约定
--------
* 「这个模型用哪个 checkpoint」来自 ``loraforge/models.py`` 的 ``PROFILES``，
  再追溯到 ai-toolkit ``toolkit/models/registry.py``。
* 显存档位是**工程档位**（量化 / offload / 降分辨率三件套的组合），
  没有逐档实测数据，``note`` 里会写清楚；涉及 ai-toolkit 能力的部分标了源码行号。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loraforge.configs.base import TrainingPlan
from loraforge.models import PROFILES, TargetModel, profile


@dataclass(frozen=True)
class VramTier:
    """一个显存档位。字段名按 §8 冻结。"""

    key: str  # "8" / "12" / "16" / "24" / "48"，单位 GB
    label: str
    quantize: str | None  # None = 不量化；否则是 ai-toolkit 的 qtype
    batch_size: int
    max_resolution: int
    note: str


#: 显存档位表。24 GB 是默认档（用户主力机器）。
#:
#: 量化档位的依据：ai-toolkit 的 ``quantize`` / ``qtype`` 默认 ``qfloat8``
#: （toolkit/config_modules.py:695-703），MOSS 官方用 ``convrot8``
#: （toolkit/models/registry.py:59）；``layer_offloading`` 是官方推荐的降显存手段
#: （toolkit/config_modules.py:723-727），SDXL 另有 ``low_vram``
#: （toolkit/config_modules.py:699）。**逐档实测数据还没有，属于工程建议。**
VRAM_TIERS: dict[str, VramTier] = {
    "8": VramTier(
        key="8",
        label="8 GB",
        quantize="qfloat8",
        batch_size=1,
        max_resolution=512,
        note="工程建议，未实测：只跑小图；SDXL 需 low_vram，Krea2/MOSS 大概率跑不动",
    ),
    "12": VramTier(
        key="12",
        label="12 GB",
        quantize="qfloat8",
        batch_size=1,
        max_resolution=768,
        note="工程建议，未实测：单图 768；SDXL 需 low_vram，Krea2/MOSS 需开量化",
    ),
    "16": VramTier(
        key="16",
        label="16 GB",
        quantize="qfloat8",
        batch_size=1,
        max_resolution=1024,
        note="工程建议，未实测：单图 1024；Krea2/MOSS 开量化，SDXL 需 low_vram",
    ),
    "24": VramTier(
        key="24",
        label="24 GB",
        quantize="qfloat8",
        batch_size=1,
        max_resolution=1024,
        note="默认档：Krea2/MOSS 按官方 registry 开 qfloat8，SDXL 不量化",
    ),
    "48": VramTier(
        key="48",
        label="48 GB",
        quantize=None,
        batch_size=2,
        max_resolution=1024,
        note="工程建议，未实测：不量化 + batch 2；再大没有官方出处，不往上猜",
    ),
}


#: 三款模型在 20 张小图集上的初始训练步数。
#: 2500 来自 ``docs/ARCHITECTURE.md`` §3 的契约默认值；模型特定最优值 UNVERIFIED。
DEFAULT_STEPS = 2500

#: LoRA rank / alpha 的初始值。契约默认 32/32；单模型调优值 UNVERIFIED。
DEFAULT_RANK = 32
DEFAULT_ALPHA = 32

#: 学习率初始值。契约默认 1e-4；单模型调优值 UNVERIFIED。
DEFAULT_LR = 1e-4


def _tier(tier_key: str) -> VramTier:
    try:
        return VRAM_TIERS[str(tier_key)]
    except KeyError:
        allowed = ", ".join(sorted(VRAM_TIERS, key=int))
        raise ValueError(f"未知显存档位 {tier_key!r}，可用：{allowed}") from None


def plan_for_tier(
    target: TargetModel | str,
    tier_key: str = "24",
    **overrides: Any,
) -> TrainingPlan:
    """按档位给一份方案。``overrides`` 透传给 ``TrainingPlan``。"""
    t = TargetModel(target)
    tier = _tier(tier_key)

    # 分辨率：从 512 到档位上限，按 256 的整数倍铺开，bucket 好切。
    resolutions = [res for res in (512, 768, 1024) if res <= tier.max_resolution] or [512]

    params: dict[str, Any] = {
        # 有真实 checkpoint 才算「能直接用」；占位路径会让 ai-toolkit 直接失败。
        "model_path": profile(t).train_base,
        "steps": DEFAULT_STEPS,
        "rank": DEFAULT_RANK,
        "alpha": DEFAULT_ALPHA,
        "lr": DEFAULT_LR,
        "batch_size": tier.batch_size,
        "resolutions": resolutions,
        "vram_gb": int(tier.key),
    }
    params.update(overrides)
    return TrainingPlan(**params)


def default_plan(target: TargetModel | str) -> TrainingPlan:
    """按模型给一份能直接用的初始方案（默认 24 GB 档）。"""
    return plan_for_tier(target, "24")


def tier_for_vram(vram_gb: int) -> VramTier:
    """把显存数字映射到档位：取不超过该显存的最高档，低于 8 按 8 处理。"""
    keys = sorted(VRAM_TIERS, key=int)
    chosen = keys[0]
    for key in keys:
        if int(key) <= int(vram_gb):
            chosen = key
    return VRAM_TIERS[chosen]


__all__ = [
    "VramTier",
    "VRAM_TIERS",
    "DEFAULT_STEPS",
    "DEFAULT_RANK",
    "DEFAULT_ALPHA",
    "DEFAULT_LR",
    "default_plan",
    "plan_for_tier",
    "tier_for_vram",
]
