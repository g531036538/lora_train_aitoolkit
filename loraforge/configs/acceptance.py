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

from loraforge.configs.base import TrainingPlan, build_config
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
            # Raw 上的训练期采样是 52 步、生效 4.5（yaml 里写 5.5），见 docs/models/krea2.md:224（UNVERIFIED）。
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


_MODEL_LABEL: dict[TargetModel, str] = {
    TargetModel.SDXL: "SDXL",
    TargetModel.KREA2: "Krea 2",
    TargetModel.MOSS: "MOSS-Image",
}

#: 每条验收提示「怎么算过」。写在渲染器里，不塞进 AcceptancePrompt：
#: prompt 的 ``purpose`` 说「看什么」，判读说「看到什么程度算过」，两者都要留在产物里。
_JUDGEMENT: dict[str, str] = {
    "in_distribution": (
        "脸/轮廓能对上原图、背景没有被固定成训练图里的某一张，即通过；"
        "五官糊掉或姿势被锁死，说明过拟合"
    ),
    "generalization": (
        "换到训练图里没出现过的场景后仍能一眼认出是同一人/同一物，即通过；"
        "人物跟着背景一起变形（脸漂移、衣服被重画），算没过"
    ),
    "no_trigger": (
        "同 seed 去掉触发词后与「分布内」那张明显不同、且接近底座原样，即通过；"
        "两张几乎一样，说明触发词没吃进去（LoRA 在模仿背景而不是身份）"
    ),
}

#: 采样表里「没写」的键统一写「不适用（原因）」。只写三个字，用户会以为漏配。
_NA = "不适用（{reason}）"


def _sampling_rows(
    target: TargetModel,
    data: dict[str, Any],
    cfg: dict[str, Any],
    size: int,
) -> list[tuple[str, str, str, str]]:
    """构造采样参数表：参数 / yaml 里的值 / 实际生效值 / 说明。"""
    proc = cfg["config"]["process"][0]
    sample = proc.get("sample", {})
    train = proc.get("train", {})
    sampling = data["sampling"]
    yaml_steps = sample.get("sample_steps", sampling["steps"])
    yaml_cfg = sample.get("guidance_scale", sampling["guidance_scale"])
    rows: list[tuple[str, str, str, str]] = []

    rows.append(
        (
            "尺寸",
            f"{sample.get('width', size)} × {sample.get('height', size)}",
            f"{size} × {size}",
            "yaml 里是示例值；验收跟训练的最大分辨率桶走，避免「尺度外推」被当成训练问题",
        )
    )

    if target is TargetModel.KREA2:
        rows.append(
            (
                "采样器",
                str(sampling["sampler"]),
                str(sampling["sampler"]),
                "Krea 2 出图不读 sample.sampler（krea2.py:411-467），"
                "但训练侧必须与 train.noise_scheduler 一致",
            )
        )
        rows.append(
            (
                "步数",
                f"{yaml_steps}（训练期采样）；{sampling['steps']}（验收出图）",
                f"{yaml_steps}；{sampling['steps']}",
                "训练期采样看趋势，验收走 Turbo 的 8 步（registry.py:26）",
            )
        )
        rows.append(
            (
                "CFG",
                f"{yaml_cfg}（训练期采样）；{sampling['guidance_scale']}（验收出图）",
                f"{max(0.0, yaml_cfg - 1.0)}（训练期采样）；"
                f"{max(0.0, sampling['guidance_scale'] - 1.0)}（验收出图）",
                "Krea 2 生效值 = 写的值 − 1（krea2.py:455）：写 5.5 生效 4.5，"
                "写 4.5 会变成 3.5；Turbo 写 1.0 生效 0，即没有分类器引导",
            )
        )
    else:
        rows.append(
            (
                "采样器",
                str(sampling["sampler"]),
                str(sampling["sampler"]),
                "必须与 train.noise_scheduler 一致"
                if target is TargetModel.SDXL
                else "MOSS 出图不读 sample.sampler（moss_image.py:395-411），"
                "但训练侧必须与 train.noise_scheduler 一致",
            )
        )
        rows.append(("步数", str(yaml_steps), str(yaml_steps), "yaml 与验收同值"))
        rows.append(
            (
                "CFG",
                f"{yaml_cfg}",
                f"{yaml_cfg}",
                "SDXL 不做换算" if target is TargetModel.SDXL else
                "MOSS 不做 −1 换算（moss_image.py:403），和 Krea 2 不同",
            )
        )

    timestep = train.get("timestep_type")
    if target is TargetModel.SDXL:
        rows.append(
            (
                "timestep_type",
                _NA.format(reason="SDXL 不是 flow matching，没有 timestep 类型"),
                _NA.format(reason="不是漏配，SDXL 走 ddpm 离散步"),
                "写 sigmoid/linear 都没有意义",
            )
        )
    else:
        rows.append(
            (
                "timestep_type",
                str(timestep),
                str(timestep),
                "训练侧 timestep 分布；验收出图不写这个键",
            )
        )

    if target is TargetModel.KREA2:
        rows.append(
            (
                "mu",
                _NA.format(reason="mu 不是 yaml 键，由调度器按分辨率算"),
                f"{krea2_mu((size // KREA2_LATENT_DIVISOR) ** 2)}（{size} 分辨率）",
                "动态 shift：1024 与 512 的 mu 不同，换尺寸要重算",
            )
        )
        rows.append(
            (
                "shift",
                _NA.format(reason="Krea 2 用动态 shift（mu），不用固定值"),
                "—",
                "验收出图不额外传 shift",
            )
        )
    elif target is TargetModel.MOSS:
        rows.append(
            (
                "mu",
                _NA.format(reason="只有 Krea 2 用动态 shift（mu）"),
                "—",
                "MOSS 用固定 shift，不按分辨率变",
            )
        )
        rows.append(
            (
                "shift",
                f"{data['sampling']['shift']}",
                f"{MOSS_SHIFT}",
                "moss_image/src/pipeline.py:17",
            )
        )
    else:
        rows.append(
            (
                "mu",
                _NA.format(reason="只有 Krea 2 用动态 shift（mu）"),
                "—",
                "SDXL 不是 flow matching",
            )
        )
        rows.append(
            (
                "shift",
                _NA.format(reason="SDXL 不是 flow matching，没有 shift"),
                "—",
                "SDXL 走 ddpm 离散步",
            )
        )

    negative = data["sampling"]["negative"]
    cfg_effective = max(0.0, yaml_cfg - 1.0) if target is TargetModel.KREA2 else yaml_cfg
    negative_cell = f"`{negative}`" if negative else "（空）"
    rows.append(
        (
            "负向提示",
            negative_cell,
            negative_cell,
            "CFG > 1 时负向才生效，官方没给词所以留空"
            if cfg_effective > 1.0
            else "生效 CFG ≤ 1，负向不参与计算",
        )
    )
    return rows


def render_acceptance_markdown(target: TargetModel | str, plan: TrainingPlan) -> str:
    """把验收方案渲染成一份 Markdown 清单（纯函数，不读文件、不联网）。

    用户拿到导出 zip 后照着这份清单在云上自查；落盘由导出层负责，这里只返回字符串。
    数值全部来自 ``acceptance_dict(target)`` 与 ``plan``，另用 ``build_config`` 取
    「yaml 里写的值」，这样表里两列永远跟着生成器走，不会各说一套。
    """
    t = TargetModel(target)
    data = acceptance_dict(t)
    cfg = build_config(t, plan)
    label = _MODEL_LABEL[t]
    train_base = profile(t).train_base
    size = max(plan.resolutions) if plan.resolutions else 1024
    trigger = plan.trigger or "mytoken"

    lines: list[str] = [
        f"# {label} LoRA 验收清单 —— 触发词 `{trigger}`",
        "",
        "这份清单是给你在云上跑完训练后自查用的；训练不在这里跑，这里只管「怎么判断练成了」。",
        "",
        "## 0. 出图用哪个模型",
        "",
        "| 用途 | 模型 |",
        "| --- | --- |",
        f"| 训练底模（你刚训的） | `{train_base}` |",
        f"| 验收出图（加载 LoRA） | `{data['verify_model']}` |",
        "",
    ]
    if data["verify_model"] != train_base:
        lines.append(
            "**别用训练底座出图**：出图侧是蒸馏版/对齐版，步数与 CFG 口径都不一样，"
            "拿训练底座验收会把正常的 LoRA 判成坏的。"
        )
    else:
        lines.append(
            "SDXL 的验收就用同一个底模加载 LoRA——注意「加载了 LoRA」和「没加载」"
            "要各出一张对照，否则分不清像的是 LoRA 还是底模本来就像。"
        )
    lines += [
        "",
        "## 1. 采样参数",
        "",
        "| 参数 | yaml 里的值 | 实际生效值 | 说明 |",
        "| --- | --- | --- | --- |",
    ]
    for name, yaml_value, effective, note in _sampling_rows(t, data, cfg, size):
        lines.append(f"| {name} | {yaml_value} | {effective} | {note} |")

    lines += [
        "",
        f"> 验收尺寸取训练分辨率的最大档（{size}）。"
        "若该值大于 1024，另用 1024 复测一张作对照；"
        "在该尺寸崩而 1024 正常，说明是尺度外推而不是训练问题。"
        if size > 1024
        else f"> 验收尺寸取训练分辨率的最大档（{size}）：训练见过的最大尺度，才有可比性。",
        "",
        "## 2. 三提示法",
        "",
        "三条都用同一个 seed，`walk_seed` 关闭；每条给的是可以直接抄的模板，"
        f"`{trigger}` 就是你的触发词。",
        "",
    ]

    for index, prompt in enumerate(data["prompts"], start=1):
        template = prompt["template"].replace("[trigger]", trigger)
        lines += [
            f"### 2.{index} {prompt['label']}",
            "",
            f"`{template}`",
            "",
            f"- 看什么：{prompt['purpose']}",
            f"- 怎么判读：{_JUDGEMENT.get(prompt['key'], '')}",
            "",
        ]

    lines += ["## 3. 判定口径", ""]
    for index, item in enumerate(data["checklist"], start=1):
        lines.append(f"{index}. {item}")

    lines += [
        "",
        "## 4. 参数出处",
        "",
        f"- 采样与 mu/shift：{data['sampling']['source']}",
        "- 三提示法与判定口径：`loraforge/configs/acceptance.py`"
        "（改这里会同时改掉所有模型的 `验收.md`）",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "SamplingParams",
    "AcceptancePrompt",
    "AcceptancePlan",
    "acceptance_plan",
    "acceptance_dict",
    "render_acceptance_markdown",
    "krea2_mu",
]
