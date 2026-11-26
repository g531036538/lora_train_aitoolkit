"""训练后验收方案：采样参数 + 三提示法。

界面要读结构化数据，不要散落的字符串。这里给出：

* ``SamplingParams``：验收出图用哪个模型、采样器、步数、CFG、mu / shift、负向提示。
* ``AcceptancePrompt``：三提示法——分布内 / 泛化 / 去触发词对照。
* ``acceptance_plan(target)``：一次拿全。

出处
----
* SDXL：``toolkit/models/registry.py:50``（1024 / 25 步 / CFG 6.0）。
* Krea 2：``toolkit/models/registry.py:26``（Turbo，8 步 / CFG 1.0）；
  mu 来自 ``extensions_built_in/diffusion_models/krea2/krea2.py:81-99`` 的
  动态 shift 端点（base_shift 0.5 @ seq 256 → max_shift 1.15 @ seq 6400），
  公式见 ``toolkit/samplers/custom_flowmatch_sampler.py:10-20``。
* MOSS-Image：``toolkit/models/registry.py:59``（30 步 / CFG 3.0），
  shift 5.0 来自 ``extensions_built_in/diffusion_models/moss_image/src/pipeline.py:17``。

负向提示：三款模型的官方 entry 都没给负向词，统一留空字符串；
CFG > 1 时（SDXL 6.0、MOSS 3.0）负向才生效，Krea 2 Turbo 是 CFG 1.0，负向不参与。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loraforge.models import TargetModel, profile

# Krea 2 的动态 shift 端点（krea2.py:81-99），用于把 mu 从公式算出来。
KREA2_BASE_SEQ_LEN = 256
KREA2_MAX_SEQ_LEN = 6400
KREA2_BASE_SHIFT = 0.5
KREA2_MAX_SHIFT = 1.15
# Krea 2 的 latent 下采样：VAE 8 × patch 2（krea2.py:204-206）。
KREA2_LATENT_DIVISOR = 16

# MOSS-Image 的固定 shift（src/pipeline.py:17）。
MOSS_SHIFT = 5.0


def krea2_mu(image_seq_len: int) -> float:
    """按 ai-toolkit 的 calculate_shift 算 Krea 2 的 mu。

    ``toolkit/samplers/custom_flowmatch_sampler.py:10-20``::

        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b

    1024×1024 时 ``image_seq_len = (1024 // 16) ** 2 = 4096``，mu = 0.90625。
    """
    m = (KREA2_MAX_SHIFT - KREA2_BASE_SHIFT) / (KREA2_MAX_SEQ_LEN - KREA2_BASE_SEQ_LEN)
    b = KREA2_BASE_SHIFT - m * KREA2_BASE_SEQ_LEN
    return round(image_seq_len * m + b, 5)


@dataclass(frozen=True)
class SamplingParams:
    """验收出图参数。字段给界面直接渲染。"""

    width: int
    height: int
    sampler: str
    steps: int
    guidance_scale: float
    mu: float | None  # Krea 2 动态 shift；其它模型为 None
    shift: float | None  # MOSS 固定 shift；其它模型为 None
    negative: str
    source: str  # 出处，便于界面悬停显示


@dataclass(frozen=True)
class AcceptancePrompt:
    """三提示法里的一条。"""

    key: str  # "in_distribution" / "generalization" / "no_trigger"
    label: str
    purpose: str
    template: str  # 含 [trigger] 占位符，由 ai-toolkit 换成 trigger_word


@dataclass(frozen=True)
class AcceptancePlan:
    """一次训练的验收方案。"""

    target: TargetModel
    verify_model: str  # 出图用哪个 checkpoint（≠ 训练底模）
    sampling: SamplingParams
    prompts: tuple[AcceptancePrompt, ...] = field(default_factory=tuple)
    checklist: tuple[str, ...] = field(default_factory=tuple)


_THREE_PROMPTS: tuple[AcceptancePrompt, ...] = (
    AcceptancePrompt(
        key="in_distribution",
        label="分布内",
        purpose="和训练图同构图、同背景：看像不像本人/本物，有没有过拟合到某一张",
        template="[trigger], portrait photo, plain background, soft lighting",
    ),
    AcceptancePrompt(
        key="generalization",
        label="泛化",
        purpose="训练图里没出现过的场景与动作：看 LoRA 学的是身份还是背景",
        template="[trigger] walking through a rainy city street at night, neon signs, full body",
    ),
    AcceptancePrompt(
        key="no_trigger",
        label="去触发词对照",
        purpose="同 seed 去掉触发词：必须明显不像，否则说明触发词没吃进去、污染了底模",
        template="portrait photo, plain background, soft lighting",
    ),
)

_CHECKLIST: tuple[str, ...] = (
    "三条提示用同一个 seed、walk_seed 关闭，才能逐张对比",
    "去触发词那组如果不加 LoRA 也一样像，说明 LoRA 没学到身份",
    "同一提示跑 3 个 seed，看是稳定像还是抽卡",
    "LoRA 强度 0.8 / 1.0 各出一组，强度 1.0 过曝就往 0.8 走",
)


_ACCEPTANCE: dict[TargetModel, AcceptancePlan] = {
    TargetModel.SDXL: AcceptancePlan(
        target=TargetModel.SDXL,
        verify_model=profile(TargetModel.SDXL).verify_base,
        sampling=SamplingParams(
            width=1024,
            height=1024,
            sampler="ddpm",
            steps=25,
            guidance_scale=6.0,
            mu=None,
            shift=None,
            negative="",
            source="toolkit/models/registry.py:50",
        ),
        prompts=_THREE_PROMPTS,
        checklist=_CHECKLIST,
    ),
    TargetModel.KREA2: AcceptancePlan(
        target=TargetModel.KREA2,
        # 训练用 Raw，出图用 Turbo（loraforge/models.py:54-55）
        verify_model=profile(TargetModel.KREA2).verify_base,
        sampling=SamplingParams(
            width=1024,
            height=1024,
            sampler="flowmatch",
            steps=8,
            guidance_scale=1.0,
            mu=krea2_mu((1024 // KREA2_LATENT_DIVISOR) ** 2),
            shift=None,
            negative="",
            # 出图验收走 Turbo：8 步 / guidance 1.0（toolkit/models/registry.py:26）。
            # Raw 上的训练期采样是 52 步 / 4.5，见 docs/models/krea2.md:224（UNVERIFIED）。
            source="toolkit/models/registry.py:26 + docs/models/krea2.md:224",
        ),
        prompts=_THREE_PROMPTS,
        checklist=_CHECKLIST,
    ),
    TargetModel.MOSS: AcceptancePlan(
        target=TargetModel.MOSS,
        # 训练用 Base，出图用 SFT（loraforge/models.py:66-67）
        verify_model=profile(TargetModel.MOSS).verify_base,
        sampling=SamplingParams(
            width=1024,
            height=1024,
            sampler="flowmatch",
            # docs/models/moss_image.md:193：官方 sample_steps ≥50
            steps=50,
            guidance_scale=3.0,
            mu=None,
            shift=MOSS_SHIFT,
            negative="",
            source="docs/models/moss_image.md:193 + moss_image/src/pipeline.py:17",
        ),
        prompts=_THREE_PROMPTS,
        checklist=_CHECKLIST,
    ),
}


def acceptance_plan(target: TargetModel | str) -> AcceptancePlan:
    """拿一个模型的验收方案。"""
    return _ACCEPTANCE[TargetModel(target)]


def acceptance_dict(target: TargetModel | str) -> dict[str, Any]:
    """给界面用的纯 dict 版本（可以直接 json 化）。"""
    plan = acceptance_plan(target)
    return {
        "target": plan.target.value,
        "verify_model": plan.verify_model,
        "sampling": {
            "width": plan.sampling.width,
            "height": plan.sampling.height,
            "sampler": plan.sampling.sampler,
            "steps": plan.sampling.steps,
            "guidance_scale": plan.sampling.guidance_scale,
            "mu": plan.sampling.mu,
            "shift": plan.sampling.shift,
            "negative": plan.sampling.negative,
            "source": plan.sampling.source,
        },
        "prompts": [
            {
                "key": p.key,
                "label": p.label,
                "purpose": p.purpose,
                "template": p.template,
            }
            for p in plan.prompts
        ],
        "checklist": list(plan.checklist),
    }


__all__ = [
    "SamplingParams",
    "AcceptancePrompt",
    "AcceptancePlan",
    "acceptance_plan",
    "acceptance_dict",
    "krea2_mu",
]
