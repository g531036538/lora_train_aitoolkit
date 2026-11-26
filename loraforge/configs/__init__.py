"""ai-toolkit 配置生成与校验。接口见 docs/ARCHITECTURE.md。

* :func:`build_config` / :func:`validate_config` —— §3 冻结签名
* :func:`validate_config_detailed` / :class:`ConfigProblem` —— 带 error/warn 分级
* :func:`default_plan` / :data:`VRAM_TIERS` —— §8 读口
* :func:`acceptance_plan` —— 训练后验收（采样参数 + 三提示法）
"""

from __future__ import annotations

from loraforge.configs.acceptance import (
    AcceptancePlan,
    AcceptancePrompt,
    SamplingParams,
    acceptance_dict,
    acceptance_plan,
    krea2_mu,
)
from loraforge.configs.base import TrainingPlan, build_config, validate_config
from loraforge.configs.plans import (
    DEFAULT_ALPHA,
    DEFAULT_LR,
    DEFAULT_RANK,
    DEFAULT_STEPS,
    VRAM_TIERS,
    VramTier,
    default_plan,
    plan_for_tier,
    tier_for_vram,
)
from loraforge.configs.problems import (
    ConfigProblem,
    known_codes,
    validate_config_detailed,
)

__all__ = [
    "TrainingPlan",
    "build_config",
    "validate_config",
    "ConfigProblem",
    "validate_config_detailed",
    "known_codes",
    "VramTier",
    "VRAM_TIERS",
    "DEFAULT_STEPS",
    "DEFAULT_RANK",
    "DEFAULT_ALPHA",
    "DEFAULT_LR",
    "default_plan",
    "plan_for_tier",
    "tier_for_vram",
    "SamplingParams",
    "AcceptancePrompt",
    "AcceptancePlan",
    "acceptance_plan",
    "acceptance_dict",
    "krea2_mu",
]
