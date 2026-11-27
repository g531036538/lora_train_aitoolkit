"""ai-toolkit 配置的生成与校验。

接口见 ``docs/ARCHITECTURE.md`` §3（签名冻结）与 §8（读口）。

出处约定
--------
每个默认值都在注释里标出来源，格式 ``文件:行号`` 或 URL。参数取值一律以阿研的
调研文档 ``docs/models/<target>.md`` 为准（该文档同时给出 ai-toolkit 源码行号）；
文档没覆盖的字段直接指向 ``ostris/ai-toolkit`` 主线源码（本地镜像：
``.scratch/src/ai-toolkit-main``，从
https://codeload.github.com/ostris/ai-toolkit/tar.gz/refs/heads/main 下载）。
两处都找不到出处的值一律在注释里标 ``UNVERIFIED``，不编造。

三款模型的差异（照抄 FLUX 会训坏）
----------------------------------
* SDXL：走 legacy ``StableDiffusion`` 路径，``arch: sdxl``，``noise_scheduler``
  用 ``ddpm``；``quantize`` 只对 transformer 系开放，SDXL 不能开。
* Krea 2：``flowmatch`` + ``timestep_type: linear``。**禁止**把 FLUX 的
  ``sigmoid`` 搬过来。
* MOSS-Image：训练必须用 Base 变体，且 ``model_kwargs.uncond_skip_layers: [22]``；
  ``arch`` 只能写 ``moss_image``（``moss_image:base`` 里的 ``:`` 之后会被
  ai-toolkit 截断，见 ``toolkit/config_modules.py:776-778``）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loraforge.models import PROFILES, TargetModel, profile
from loraforge.configs.problems import validate_config_detailed

# ---------------------------------------------------------------------------
# 训练方案
# ---------------------------------------------------------------------------


@dataclass
class TrainingPlan:
    """一次 LoRA 训练的全部入参。

    默认值来自 ``docs/ARCHITECTURE.md`` §3（契约冻结），不是模型调研值。
    模型相关的默认值走 :func:`loraforge.configs.plans.default_plan`。
    """

    model_path: str = "/path/to/model"  # 云端占位，导出前填
    dataset_dir: str = "/path/to/dataset"
    output_dir: str = "/path/to/output"
    trigger: str = "mytoken"
    steps: int = 2500
    rank: int = 32
    alpha: int = 32
    lr: float = 1e-4
    batch_size: int = 1
    resolutions: list[int] = field(default_factory=lambda: [512, 768, 1024])
    vram_gb: int = 24  # 决定量化 / offload 档位
    sample_prompts: list[str] = field(default_factory=list)


# 训练总时长的存档 / 采样间隔：按总步数的 1/10 取，2500 步时正好 250，
# 与官方示例的 save_every: 250 一致（config/examples/train_lora_flux_24gb.yaml:26）。
_INTERVAL_DIVISOR = 10

# 数据集的 bucket 尺寸必须能被这个数整除。ai-toolkit 里 SDXL 与两套新架构
# 都落在 16（toolkit/stable_diffusion_model.py:364-372 的 vae 下采样 ×2；
# extensions_built_in/diffusion_models/krea2/krea2.py:204-206 与
# .../moss_image/moss_image.py:207-208 的 8×patch）。
_BUCKET_DIVISIBILITY = 16


def _interval(steps: int) -> int:
    """存档 / 采样间隔。"""
    return max(1, int(steps) // _INTERVAL_DIVISOR)


def _target(target: TargetModel | str) -> TargetModel:
    return TargetModel(target)


def _prompts(plan: TrainingPlan) -> list[str]:
    """训练期采样提示词。

    空列表时给两条最朴素的，``[trigger]`` 由 ai-toolkit 替换成 trigger_word
    （``config/examples/train_lora_flux_24gb.yaml:70`` 注释）。
    """
    if plan.sample_prompts:
        return list(plan.sample_prompts)
    return ["[trigger], portrait photo", "[trigger], full body photo"]


def _common_sections(target: TargetModel, plan: TrainingPlan) -> dict[str, Any]:
    """三款模型共用的骨架。字段名以 ai-toolkit 主线为准。"""
    t = _target(target)
    steps = int(plan.steps)
    interval = _interval(steps)

    return {
        "type": "sd_trainer",  # jobs/process 的 process 类型
        "training_folder": plan.output_dir,
        "device": "cuda:0",
        # 会把 trigger 追加到没有它的 caption 上（toolkit/job.py 的 sd_trainer 分支）
        "trigger_word": plan.trigger,
        "network": {
            "type": "lora",
            "linear": int(plan.rank),
            "linear_alpha": int(plan.alpha),
        },
        "save": {
            "dtype": "float16",
            "save_every": interval,
            "max_step_saves_to_keep": 4,
            "push_to_hub": False,
        },
        "datasets": [
            {
                "folder_path": plan.dataset_dir,
                "caption_ext": "txt",  # toolkit/config_modules.py:953
                "caption_dropout_rate": 0.05,  # 示例默认 config/examples/train_lora_flux_24gb.yaml:34
                "shuffle_tokens": False,  # toolkit/config_modules.py:967
                "cache_latents_to_disk": True,  # toolkit/config_modules.py:1018
                "buckets": True,  # toolkit/config_modules.py:961
                "resolution": _resolutions(t, plan),
            }
        ],
        "train": {
            "batch_size": int(plan.batch_size),
            "steps": steps,
            "gradient_accumulation_steps": 1,  # toolkit/config_modules.py:461
            "train_unet": True,
            "train_text_encoder": False,  # toolkit/config_modules.py:407
            "gradient_checkpointing": True,  # toolkit/config_modules.py:419
            "optimizer": "adamw8bit",  # 示例默认 config/examples/train_lora_flux_24gb.yaml:48
            "lr": float(plan.lr),
            "dtype": "bf16",
            "ema_config": {  # toolkit/config_modules.py:547-555
                "use_ema": True,
                "ema_decay": 0.99,
            },
        },
        "sample": {
            "sample_every": interval,
            "sample_start_step": 0,
            "width": _sample_size(t, plan),
            "height": _sample_size(t, plan),
            "prompts": _prompts(plan),
            "neg": "",
            "seed": 42,
            "walk_seed": False,
        },
        "logging": {"log_every": 10, "use_wandb": False, "verbose": False},
    }


def _resolutions(target: TargetModel, plan: TrainingPlan) -> list[int]:
    """有效训练分辨率：被档位上限截断，再按 bucket 整除性对齐。

    小显存不许硬塞 1024（``max_resolution`` 来自 ``VRAM_TIERS``）。
    """
    from loraforge.configs.plans import VRAM_TIERS

    tier = VRAM_TIERS.get(str(plan.vram_gb))
    cap = tier.max_resolution if tier is not None else max(int(r) for r in plan.resolutions)
    out: list[int] = []
    for res in plan.resolutions:
        res = min(int(res), cap)
        res -= res % _BUCKET_DIVISIBILITY
        res = max(_BUCKET_DIVISIBILITY, res)
        if res not in out:
            out.append(res)
    return out or [cap]


def _sample_size(target: TargetModel, plan: TrainingPlan) -> int:
    """采样出图边长：不超过档位上限，也不超过训练集最大分辨率。"""
    res = max(_resolutions(target, plan))
    return min(1024, res)


# ---------------------------------------------------------------------------
# 三款模型各自的参数
# ---------------------------------------------------------------------------


def _build_sdxl(proc: dict[str, Any], plan: TrainingPlan) -> None:
    """SDXL：legacy StableDiffusion 路径。

    * ``arch: sdxl`` → ``is_xl``（toolkit/config_modules.py:790-792）；
      legacy 架构集合见 toolkit/util/get_model.py:46-58。
    * ``noise_scheduler`` 保持 ``ddpm``（默认值 toolkit/config_modules.py:383）；
      SDXL 不是 flow matching，配 flowmatch 会训坏（docs/models/sdxl.md:234）。
    * 不写 ``min_snr_gamma``：docs/models/sdxl.md:228 的起点是"不设"；
      该值须 >1e-6 才生效（SDTrainer.py:1059-1070），没校准就不加。
    * ``vae_dtype: float32``（docs/models/sdxl.md:233）：官方 VAE 用 fp32，
      低精度偶发 NaN，源码无保护（config_modules.py:690 默认跟随 model dtype）。
    * SDXL 不开 ``quantize``：该开关目前只对 flux 系开放
      （toolkit/config_modules.py:697-698），小显存改用 ``low_vram``。
    """
    proc["train"].update(
        {
            "noise_scheduler": "ddpm",
            "train_text_encoder": False,
        }
    )
    model: dict[str, Any] = {
        "name_or_path": plan.model_path,
        "arch": "sdxl",
        # 与 train.dtype 保持一致：jobs/process/BaseSDTrainProcess.py:117 会用
        # train.dtype 覆盖 model.dtype（ModelConfig 见 config_modules.py:655），
        # 这里写另一个值只会让 yaml 读起来自相矛盾，不改变实际行为。
        "dtype": "bf16",
        "vae_dtype": "float32",  # docs/models/sdxl.md:233
    }
    if int(plan.vram_gb) <= 16:
        # toolkit/config_modules.py:699；显存够时不加，省得拖慢
        model["low_vram"] = True
    proc["model"] = model
    proc["sample"].update(
        {
            "sampler": "ddpm",  # 必须与 train.noise_scheduler 一致
            "sample_steps": 25,  # toolkit/models/registry.py:50
            "guidance_scale": 6.0,  # toolkit/models/registry.py:50
        }
    )


def _build_krea2(proc: dict[str, Any], plan: TrainingPlan) -> None:
    """Krea 2：单流 DiT + flow matching。

    * 训练底模 ``krea/Krea-2-Raw``，出图用 Turbo（README.md:39-40）。
    * ``arch: krea2``（extensions_built_in/diffusion_models/krea2/krea2.py:144-145）。
    * ``noise_scheduler: flowmatch`` + ``timestep_type: linear``（krea2.py:159 is_flow_matching；
      训练调度器 krea2.py:201-202）。**不许**用 FLUX 的 ``sigmoid``。
    * ``quantize``/``quantize_te`` 与 ``qfloat8``（toolkit/models/registry.py:26；
      qtype 默认 qfloat8，toolkit/config_modules.py:695-703）。
    * ``diff_output_preservation`` 从 ``false`` 起步（docs/models/krea2.md:222）：
      种子建议开 ``true``，但需要 trigger_word 且不能开 TE，先跑通再加。
    * ``vae_dtype: float32``（docs/models/krea2.md:218，预防性设置，可选项）。
    * 采样 **52 步 / guidance 5.5**（docs/models/krea2.md:15,224）：训练在 Raw
      （非蒸馏）上采样，不是 Turbo 的 8 步 / 1.0。
    * guidance 要换算：ai-toolkit 生效值 = ``max(0, yaml 值 − 1.0)``
      （krea2.py:455），所以要生效 4.5 就得写 5.5。registry 里 Turbo 写 1.0、
      生效 0，正好自洽这条读法。换算口径仍 **UNVERIFIED**——若实拍证明
      ai-toolkit 不减 1，只改 ``_build_krea2`` 里那一行，回到 4.5。
    """
    proc["train"].update(
        {
            "noise_scheduler": "flowmatch",
            "timestep_type": "linear",
            "train_text_encoder": False,
            "diff_output_preservation": False,  # docs/models/krea2.md:222
        }
    )
    proc["model"] = {
        "name_or_path": plan.model_path,
        "arch": "krea2",
        "dtype": "bf16",
        "vae_dtype": "float32",  # docs/models/krea2.md:218
        "quantize": True,
        "qtype": "qfloat8",
        "quantize_te": True,
        "qtype_te": "qfloat8",
    }
    proc["sample"].update(
        {
            "sampler": "flowmatch",  # 必须与 train.noise_scheduler 一致
            "sample_steps": 52,  # docs/models/krea2.md:224（Raw 口径）
            # docs/models/krea2.md:224 要的是生效 4.5；ai-toolkit 生效值 = 本值 − 1
            # （krea2.py:455），所以写 5.5。UNVERIFIED：实拍证明不减 1 就改回 4.5。
            "guidance_scale": 5.5,
        }
    )


def _build_moss(proc: dict[str, Any], plan: TrainingPlan) -> None:
    """MOSS-Image：训练用 Base，出图用 SFT。

    * 训练底模 ``OpenMOSS-Team/MOSS-Image-Base``（toolkit/models/registry.py:60）。
    * ``arch: moss_image``（extensions_built_in/diffusion_models/moss_image/moss_image.py:159-160）。
      不能写 ``moss_image:base``——``:`` 之后会被截断（toolkit/config_modules.py:776-778）。
    * ``uncond_skip_layers: [22]`` 是**推理侧** SLG 口径（docs/models/moss_image.md:195；
      registry.py:60 的 Base 变体、src/pipeline.py:20）；训练侧源码未见读取，
      写上是给出图对照组留的，不是训练必需。
    * 采样 **50 步 / CFG 3.0**（docs/models/moss_image.md:193：官方 ≥50、CFG 3.0–6.0），
      shift 5（src/pipeline.py:17 DEFAULT_SHIFT=5.0，经 ``sample_shift`` 透传，
      moss_image.py:184）。MOSS 不像 Krea2 那样减 1（moss_image.py:403 直接传）。
    * ``vae_dtype: float32``（docs/models/moss_image.md:264，官方 VAE fp32 口径）。
    * ``timestep_type: linear``：与 MOSS 自家训练网格对齐。训练调度器是
      ``shift=5.0`` + ``use_dynamic_shifting=False``（moss_image.py:72-75,202-205；
      src/pipeline.py:17 DEFAULT_SHIFT=5.0），而 ``set_train_timesteps`` 的 shift 分支
      是在 diffusers ``__init__`` **已经 shift 过的端点上再 shift 一次**
      （custom_flowmatch_sampler.py:147-152,172；diffusers@c9438378
      scheduling_flow_match_euler_discrete.py ``__init__``），等效 shift 应用两次：
      1024² 下 σ 中位 0.9619，``linear`` 只有 0.5005。推理 ``moss_sigmas()`` 吸附的
      ``grid(total, training_shift=1.0)`` 网格正是未 shift 的 linear 网格
      （src/pipeline.py:27,40,137），所以取 ``linear``。复核记录见 Issue #23。
    """
    proc["train"].update(
        {
            "noise_scheduler": "flowmatch",
            "train_text_encoder": False,
            # 取 linear：MOSS 的 shift 分支等效 shift×2（σ 中位 0.9619 vs 0.5005），
            # 且推理吸附的 training_shift=1.0 网格就是 linear 网格。依据见上方
            # docstring 与 Issue #23 复核（custom_flowmatch_sampler.py:147-152,172）。
            "timestep_type": "linear",
        }
    )
    proc["model"] = {
        "name_or_path": plan.model_path,
        "arch": "moss_image",
        "dtype": "bf16",
        "vae_dtype": "float32",  # docs/models/moss_image.md:264
        "quantize": True,
        "qtype": "convrot8",  # toolkit/models/registry.py:59
        "quantize_te": True,
        "qtype_te": "convrot8",
        "model_kwargs": {
            "uncond_skip_layers": [22],  # 推理侧 SLG 口径，见上
            "sample_shift": 5.0,  # src/pipeline.py:17
        },
    }
    proc["sample"].update(
        {
            "sampler": "flowmatch",  # 必须与 train.noise_scheduler 一致
            "sample_steps": 50,  # docs/models/moss_image.md:193
            "guidance_scale": 3.0,  # docs/models/moss_image.md:193
        }
    )


_BUILDERS = {
    TargetModel.SDXL: _build_sdxl,
    TargetModel.KREA2: _build_krea2,
    TargetModel.MOSS: _build_moss,
}


def build_config(target: TargetModel | str, plan: TrainingPlan) -> dict:
    """生成可直接 ``yaml.safe_dump`` 的 ai-toolkit 配置 dict。

    结构对齐 ``config/examples/train_lora_flux_24gb.yaml``：
    ``job: extension`` → ``config.name`` → ``config.process[0]`` → ``meta``。
    """
    t = _target(target)
    proc = _common_sections(t, plan)
    _BUILDERS[t](proc, plan)

    return {
        "job": "extension",  # toolkit/job.py:6-20 要求顶层有 job
        "config": {
            "name": f"{t.value}-lora-v1",
            "process": [proc],
        },
        "meta": {"name": "[name]", "version": "1.0"},
    }


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

# 逐条检查与 error/warn 分级在 problems.py。这里保持 §3 冻结的签名：
# 只返回硬错（severity == "error"），界面据此禁用导出；
# 需要警告时调 validate_config_detailed。


def validate_config(cfg: dict, target: TargetModel | str) -> list[str]:
    """返回**硬错**列表，空列表 = 可以导出。

    签名与语义与 ``docs/ARCHITECTURE.md`` §3 冻结版本一致：只说"会让人白跑
    或直接报错"的问题。分级细节与 warn 见
    :func:`loraforge.configs.problems.validate_config_detailed`。
    """
    return [
        problem.message
        for problem in validate_config_detailed(cfg, target)
        if problem.severity == "error"
    ]
