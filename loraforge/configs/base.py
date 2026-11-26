"""ai-toolkit 配置的生成与校验。

接口见 ``docs/ARCHITECTURE.md`` §3（签名冻结）与 §8（读口）。

出处约定
--------
每个默认值都在注释里标出来源，格式 ``文件:行号`` 或 URL。出处一律指向
``ostris/ai-toolkit`` 主线源码（本地镜像：``.scratch/src/ai-toolkit-main``，
从 https://codeload.github.com/ostris/ai-toolkit/tar.gz/refs/heads/main 下载）。

``docs/models/<target>.md`` 目前尚未落地（main 上 404），因此所有引用暂时
直接写 ai-toolkit 源码行号；等阿研的调研文档合入后再回填成 ``docs/models/*.md``。
找不到出处的值一律在注释里标 ``UNVERIFIED``，不编造。

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
      SDXL 不是 flow matching，配 flowmatch 会训坏。
    * ``min_snr_gamma: 5.0``（extensions_built_in/sd_trainer/config/train.example.yaml:43），
      字段定义 toolkit/config_modules.py:411。
    * ``vae_dtype`` 显式 float16（默认跟随 model dtype，toolkit/config_modules.py:690）。
    * SDXL 不开 ``quantize``：该开关目前只对 flux 系开放
      （toolkit/config_modules.py:697-698），小显存改用 ``low_vram``。
    """
    proc["train"].update(
        {
            "noise_scheduler": "ddpm",
            "min_snr_gamma": 5.0,
            "train_text_encoder": False,
        }
    )
    model: dict[str, Any] = {
        "name_or_path": plan.model_path,
        "arch": "sdxl",
        "dtype": "float16",
        "vae_dtype": "float16",
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
    * ``diff_output_preservation: true``（toolkit/config_modules.py:495；
      由 jobs/process/BaseSDTrainProcess.py:159-162 下发到 dataset）。
    * 采样 8 步 / CFG 1.0 来自 toolkit/models/registry.py:26，但那是 **Turbo**
      的推理口径；训练在 Raw 上采样，这两项 UNVERIFIED，先按 registry 填。
    """
    proc["train"].update(
        {
            "noise_scheduler": "flowmatch",
            "timestep_type": "linear",
            "train_text_encoder": False,
            "diff_output_preservation": True,
        }
    )
    proc["model"] = {
        "name_or_path": plan.model_path,
        "arch": "krea2",
        "dtype": "bf16",
        "quantize": True,
        "qtype": "qfloat8",
        "quantize_te": True,
        "qtype_te": "qfloat8",
    }
    proc["sample"].update(
        {
            "sampler": "flowmatch",  # 必须与 train.noise_scheduler 一致
            "sample_steps": 8,  # toolkit/models/registry.py:26；Turbo 口径
            "guidance_scale": 1.0,  # toolkit/models/registry.py:26；Turbo 口径
        }
    )


def _build_moss(proc: dict[str, Any], plan: TrainingPlan) -> None:
    """MOSS-Image：训练用 Base，出图用 SFT。

    * 训练底模 ``OpenMOSS-Team/MOSS-Image-Base``（toolkit/models/registry.py:60）。
    * ``arch: moss_image``（extensions_built_in/diffusion_models/moss_image/moss_image.py:159-160）。
      不能写 ``moss_image:base``——``:`` 之后会被截断（toolkit/config_modules.py:776-778）。
    * Base 必须带 ``uncond_skip_layers: [22]``（moss_image.py:185-187；
      src/pipeline.py:20）。
    * 采样 30 步 / CFG 3.0（toolkit/models/registry.py:59），shift 5
      （src/pipeline.py:17 DEFAULT_SHIFT=5.0，经 ``sample_shift`` 透传，moss_image.py:184）。
    * ``timestep_type`` 不写：调研文档没有给出口径，走 ai-toolkit 默认
      （toolkit/config_modules.py:560，默认 ``sigmoid``），避免再搬错一次。
    """
    proc["train"].update(
        {
            "noise_scheduler": "flowmatch",
            "train_text_encoder": False,
        }
    )
    proc["model"] = {
        "name_or_path": plan.model_path,
        "arch": "moss_image",
        "dtype": "bf16",
        "quantize": True,
        "qtype": "convrot8",  # toolkit/models/registry.py:59
        "quantize_te": True,
        "qtype_te": "convrot8",
        "model_kwargs": {
            "uncond_skip_layers": [22],  # 训练必须显式给，见上
            "sample_shift": 5.0,  # src/pipeline.py:17
        },
    }
    proc["sample"].update(
        {
            "sampler": "flowmatch",  # 必须与 train.noise_scheduler 一致
            "sample_steps": 30,  # toolkit/models/registry.py:59
            "guidance_scale": 3.0,  # toolkit/models/registry.py:59
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


def _resolve_arch_hint(target: TargetModel, model: dict[str, Any]) -> bool:
    """模型路径是否属于该 target 的家族。"""
    name = str(model.get("name_or_path") or "").lower()
    if not name:
        return False
    if target is TargetModel.SDXL:
        return "stable-diffusion-xl" in name or "sdxl" in name
    if target is TargetModel.KREA2:
        return name.startswith("krea/") or "krea-2" in name
    return "moss-image" in name


def validate_config(cfg: dict, target: TargetModel | str) -> list[str]:
    """返回问题列表，空列表 = 通过。

    宁可在这里说人话，也不要让训练跑到一半炸。检查项：
    必填项、模型路径与 target 是否匹配、互斥 / 危险组合、rank 与 alpha、
    采样器与训练调度器是否一致、数据集的 bucket 尺寸。
    """
    problems: list[str] = []

    try:
        t: TargetModel | None = _target(target)
    except ValueError:
        return [f"未知目标模型 {target!r}，只支持 sdxl / krea2 / moss_image"]

    if not isinstance(cfg, dict):
        return ["配置必须是 dict（yaml.safe_load 的结果）"]

    if cfg.get("job") != "extension":
        problems.append('顶层缺少 job: extension（ai-toolkit 要求，见 toolkit/job.py:6-20）')

    config = cfg.get("config")
    if not isinstance(config, dict):
        problems.append("缺少 config 段")
        return problems
    if not config.get("name"):
        problems.append("config.name 为空")

    process = config.get("process")
    if not isinstance(process, list) or not process:
        problems.append("config.process 必须是非空列表")
        return problems

    proc = process[0]
    if not isinstance(proc, dict):
        problems.append("config.process[0] 必须是 dict")
        return problems
    if proc.get("type") != "sd_trainer":
        problems.append(f"process[0].type 应为 sd_trainer，实际 {proc.get('type')!r}")

    # ---- 模型段 -----------------------------------------------------------
    model = proc.get("model")
    if not isinstance(model, dict):
        problems.append("缺少 model 段")
        model = {}
    if not model.get("name_or_path"):
        problems.append("model.name_or_path 为空，ai-toolkit 会直接报错")
    elif not _resolve_arch_hint(t, model):
        problems.append(
            f"model.name_or_path 与 target={t.value} 不匹配："
            f"{model.get('name_or_path')!r}，应使用 {PROFILES[t].train_base}"
        )

    arch = str(model.get("arch") or "")
    expected_arch = {TargetModel.SDXL: "sdxl", TargetModel.KREA2: "krea2", TargetModel.MOSS: "moss_image"}[t]
    if arch != expected_arch:
        problems.append(f"model.arch 应为 {expected_arch!r}，实际 {arch!r}")
    if arch == "moss_image:base":
        problems.append(
            "model.arch 不能写 moss_image:base——冒号后会被截断成 moss_image"
            "（toolkit/config_modules.py:776-778）；Base 变体靠 name_or_path + "
            "model_kwargs.uncond_skip_layers 表达"
        )

    if t is TargetModel.SDXL and model.get("quantize"):
        problems.append(
            "SDXL 不能开模型量化：quantize 目前只对 transformer 系生效"
            "（toolkit/config_modules.py:697-698），小显存请用 low_vram"
        )

    kwargs = model.get("model_kwargs") or {}
    if t is TargetModel.MOSS:
        skip = kwargs.get("uncond_skip_layers")
        if not isinstance(skip, (list, tuple)) or 22 not in [int(x) for x in skip]:
            problems.append(
                "MOSS-Image 训练用 Base 时必须带 model.model_kwargs.uncond_skip_layers: [22]"
                "（toolkit/models/registry.py:60），否则采样发散"
            )

    # ---- 训练段 -----------------------------------------------------------
    train = proc.get("train")
    if not isinstance(train, dict):
        problems.append("缺少 train 段")
        train = {}

    for key in ("steps", "lr", "batch_size", "noise_scheduler"):
        if train.get(key) in (None, ""):
            problems.append(f"train.{key} 为空")

    noise = train.get("noise_scheduler")
    if t is TargetModel.SDXL:
        if noise == "flowmatch":
            problems.append(
                "SDXL 不能用 noise_scheduler: flowmatch：SDXL 不是 flow matching 模型，"
                "请用 ddpm（toolkit/config_modules.py:383）"
            )
        elif noise not in (None, "", "ddpm"):
            problems.append(f"SDXL 的 noise_scheduler 建议 ddpm，实际 {noise!r}")
    else:
        if noise != "flowmatch":
            problems.append(
                f"{t.value} 必须用 noise_scheduler: flowmatch，实际 {noise!r}"
                "（Krea2 见 krea2.py:201-202，MOSS 见 moss_image.py:202-205）"
            )

    if t is TargetModel.KREA2:
        ttype = train.get("timestep_type")
        if ttype == "sigmoid":
            problems.append(
                "Krea2 的 timestep_type 不能是 sigmoid——那是 FLUX 的口径，"
                "会训坏；必须 linear"
            )
        elif ttype != "linear":
            problems.append(f"Krea2 的 timestep_type 应为 linear，实际 {ttype!r}")
        if not train.get("diff_output_preservation"):
            problems.append(
                "Krea2 需要 train.diff_output_preservation: true（toolkit/config_modules.py:495）"
            )

    if t in (TargetModel.KREA2, TargetModel.MOSS) and train.get("train_text_encoder"):
        problems.append(f"{t.value} 的文本编码器不参与训练，train_text_encoder 应为 false")

    steps = train.get("steps")
    if isinstance(steps, (int, float)) and steps <= 0:
        problems.append(f"train.steps 必须为正数，实际 {steps}")
    lr = train.get("lr")
    if isinstance(lr, (int, float)):
        if lr <= 0:
            problems.append(f"train.lr 必须为正数，实际 {lr}")
        elif lr > 1e-2:
            problems.append(f"train.lr={lr} 过大，LoRA 通常 1e-4 量级")

    # ---- LoRA 维度 ---------------------------------------------------------
    network = proc.get("network") or {}
    rank = network.get("linear")
    alpha = network.get("linear_alpha")
    if not isinstance(rank, (int, float)) or rank <= 0:
        problems.append(f"network.linear (rank) 必须为正数，实际 {rank!r}")
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        problems.append(f"network.linear_alpha 必须为正数，实际 {alpha!r}")
    if isinstance(rank, (int, float)) and isinstance(alpha, (int, float)) and rank > 0 and alpha > 0:
        ratio = alpha / rank
        if ratio < 0.5 or ratio > 2.0:
            problems.append(
                f"rank={rank} 与 alpha={alpha} 严重不匹配（alpha/rank={ratio:.2f}），"
                "常见做法是 alpha 等于 rank"
            )

    # ---- 数据集 -----------------------------------------------------------
    datasets = proc.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        problems.append("缺少 datasets，训练没有图可读")
        datasets = []
    for i, ds in enumerate(datasets):
        if not isinstance(ds, dict):
            problems.append(f"datasets[{i}] 必须是 dict")
            continue
        if not ds.get("folder_path"):
            problems.append(f"datasets[{i}].folder_path 为空")
        res = ds.get("resolution")
        if not isinstance(res, (list, tuple)) or not res:
            problems.append(f"datasets[{i}].resolution 必须是非空列表")
            continue
        for value in res:
            if not isinstance(value, int) or value <= 0:
                problems.append(f"datasets[{i}].resolution 含非法值 {value!r}")
            elif value % _BUCKET_DIVISIBILITY != 0:
                problems.append(
                    f"datasets[{i}].resolution={value} 不能被 {_BUCKET_DIVISIBILITY} 整除，"
                    "bucket 会算不出来"
                )

    # ---- 采样段 -----------------------------------------------------------
    sample = proc.get("sample")
    if not isinstance(sample, dict):
        problems.append("缺少 sample 段")
        return problems
    if sample.get("sampler") != noise:
        problems.append(
            f"sample.sampler={sample.get('sampler')!r} 必须与 train.noise_scheduler={noise!r} 一致"
            "（config/examples/train_lora_flux_24gb.yaml:74）"
        )
    if not sample.get("prompts"):
        problems.append("sample.prompts 为空，训练期不会有采样图")
    for key in ("width", "height", "sample_steps"):
        value = sample.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            problems.append(f"sample.{key} 必须为正数，实际 {value!r}")
    for key in ("width", "height"):
        value = sample.get(key)
        if isinstance(value, int) and value % 8 != 0:
            problems.append(f"sample.{key}={value} 需要能被 8 整除")

    return problems
