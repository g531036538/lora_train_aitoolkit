"""校验问题的严重级别层：``error`` 拦导出，``warn`` 只提示。

为什么分两层
------------
``validate_config`` 返回的是一串"不能跑"的硬错，界面据此禁用导出。但有些问题
只是效果打折（分辨率没对齐、caption dropout 偏高、采样器口径不一致），把它算成
硬错会让用户白白卡在一个能跑的配置上。所以：

* ``error``：**会让人白跑几小时**，或者 ai-toolkit 直接抛异常 / 静默走错架构；
* ``warn``：只是效果打折、成本翻倍、可观测性下降，配置本身能跑。

接口（``docs/ARCHITECTURE.md`` §3 的 ``validate_config`` 签名不变）：

* :class:`ConfigProblem` —— ``code`` / ``severity`` / ``message``，界面按 ``code``
  定位，不解析文案；
* :func:`validate_config_detailed` —— 返回全部问题（含 warn）；
* ``validate_config`` 是它的薄包装，只留 ``error`` 的 message。

判级理由写在下方 ``_REGISTRY`` 里，紧挨着 code；出处指向
``ai-toolkit`` 源码（本地镜像 ``.scratch/src/ai-toolkit-main``）与阿研的
``docs/models/<target>.md``。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from loraforge.models import PROFILES, TargetModel

Severity = Literal["error", "warn"]

# ---------------------------------------------------------------------------
# 阈值
# ---------------------------------------------------------------------------

# 数据集 bucket 尺寸的整除量。必须与 ``base._BUCKET_DIVISIBILITY`` 一致，
# ``tests/test_configs.py`` 里有断言守着两处不漂移。
# 出处：toolkit/stable_diffusion_model.py:364-372（SDXL，vae 8 × 2）；
#       krea2.py:204-206、moss_image.py:207-208（8 × patch 2）。
_TARGET_DIVISIBILITY = 16

# 实际对齐时的推荐量（bucket_tolerance 的整数倍）
# 出处：docs/models/krea2.md:220（``bucket_tolerance: 64``）
_RECOMMENDED_ALIGNMENT = 64

# caption dropout 超过这个值就开始明显削弱文本对齐，但仍能训
# 出处：官方示例 0.05（config/examples/train_lora_flux_24gb.yaml:34）
_CAPTION_DROPOUT_MAX = 0.15

# MOSS 文本长度硬上限
# 出处：extensions_built_in/diffusion_models/moss_image/moss_image.py:70
_MOSS_MAX_TEXT_LENGTH = 4096

# LoRA 的 lr 量级上限：再高基本是把权重往坏里推
# 出处：docs/models/sdxl.md:212-213 契约默认 1e-4；无源码硬约束，属工程判断
_LR_MAX = 1e-2

# alpha/rank 的合理区间（有效学习率 ≈ lr × alpha/rank）
# 出处：docs/models/sdxl.md:213（契约默认 alpha == rank）
_RANK_ALPHA_MIN = 0.5
_RANK_ALPHA_MAX = 2.0

# timestep_type 白名单
# 出处：toolkit/samplers/custom_flowmatch_sampler.py:127-216
_TIMESTEP_TYPES = ("sigmoid", "linear", "lognorm_blend", "next_sample", "weighted", "one_step")

# 已知 qtype 家族
# 出处：toolkit/util/ostris_quant.py:185-211 + convrot_quant.py:54-56、
#       float8_quant.py:16、nvfp4_quant.py:34、orbit_quant.py:39、
#       orbit_vq_quant.py:54-58、uintx_quant.py:42；qfloat8 是默认值，
#       见 toolkit/config_modules.py:697-698。
_KNOWN_QTYPE = re.compile(
    r"^(?:qfloat8|float8_e4m3fn|nvfp4|convrot(?:4|8|bitnet|comfyw4a4|w4a8|int[2-8])"
    r"|orbit[2-4]|orbitvq[2-4]|uint[2-8])$"
)

# 采样尺寸 / 分辨率都得能被 8 整除（VAE 下采样）
# 出处：toolkit/stable_diffusion_model.py:364-372 等，bucket 的公共因子
_SIZE_MULTIPLE = 8


# ---------------------------------------------------------------------------
# 判级登记表：每个 code 一处，理由就在旁边
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, tuple[Severity, str]] = {
    # -- 顶层 / 骨架 -------------------------------------------------------
    "target.unknown": ("error", "未知 target 没法生成配置，直接拒绝"),
    "cfg.not-dict": ("error", "不是 dict 就没法校验下游任何字段"),
    "job.missing": ("error", "toolkit/job.py:6-20 要求顶层 job，缺了读不进"),
    "config.missing": ("error", "toolkit/config.py:38-51 要校验 job/config/config.name，缺段读不进"),
    "config.name": ("error", "同上，name 为空直接拒绝"),
    "config.process": ("error", "extension job 走 config.process[]，空列表等于没任务"),
    "config.process-item": ("error", "process[0] 不是 dict 无法解析"),
    "process.type": ("error", "type 错了不会进 sd_trainer 分支"),
    # -- 模型段 -----------------------------------------------------------
    "model.missing": ("error", "没有 model 段连底座都加载不了"),
    "model.name-or-path": ("error", "toolkit/config_modules.py:675-676 直接拒绝空路径"),
    "model.family-mismatch": (
        "error",
        "路径与 target 不属于同一家族 = 拿错底座，几小时训练全白跑",
    ),
    "model.arch": (
        "error",
        "toolkit/config_modules.py:809-831：arch 不匹配会落回 sd1，整条训练路径全错",
    ),
    "model.arch-suffix": (
        "error",
        "toolkit/config_modules.py:776-778：描号后面会被 split 掉，等于静默丢变体",
    ),
    "model.moss-uncond-skip-layers": (
        "warn",
        "docs/models/moss_image.md:195：uncond_skip_layers 是推理侧 SLG 口径，"
        "训练侧源码未见读取，只影响采样预览出图；判 error 会拦住合法训练配置",
    ),
    "model.quantize-on-sdxl": (
        "warn",
        "toolkit/config_modules.py:697-698 注明量化只对 transformer 系生效，"
        "SDXL 写了等于没写，不会失败",
    ),
    "model.qtype-unknown": (
        "warn",
        "toolkit/util/ostris_quant.py:185-211：家族外的 qtype 会退回默认实现，只是不是你要的格式",
    ),
    "model.vae-dtype": (
        "warn",
        "docs/models/sdxl.md:233、docs/models/moss_image.md:264：官方 VAE 用 fp32，"
        "低精度偶发 NaN；源码无保护（config_modules.py:690），属预防项",
    ),
    "model.max-text-length": (
        "warn",
        "moss_image.py:70 是硬上限，超了会被截断，不报错",
    ),
    # -- 训练段 -----------------------------------------------------------
    "train.missing": ("error", "没有 train 段训不了"),
    "train.required-empty": ("error", "steps/lr/batch_size/noise_scheduler 缺一个就跑不起来"),
    "train.noise-scheduler-not-flowmatch": (
        "error",
        "BaseSDTrainProcess.py:1173-1210：只有 flowmatch 才走 set_train_timesteps，"
        "否则 timestep_type 静默失效（docs/models/krea2.md:163）",
    ),
    "train.noise-scheduler-flowmatch-on-sdxl": (
        "error",
        "SDXL 不是 flow matching 模型，调度器不匹配直接训坏",
    ),
    "train.noise-scheduler-not-ddpm": (
        "warn",
        "SDXL 用其它采样器也能跑，只是偏离默认口径（config_modules.py:383）",
    ),
    "train.timestep-type-sigmoid": (
        "error",
        "FLUX 的 sigmoid 口径搬到 Krea2 上必坏（krea2.py:159 is_flow_matching）",
    ),
    "train.timestep-type-unknown": (
        "error",
        "custom_flowmatch_sampler.py:127-216：白名单外会被当成未知分支，等于没设",
    ),
    "train.timestep-type-not-linear": (
        "error",
        "docs/models/krea2.md:163：白名单内也行，但 Krea2 训练口径是 linear，"
        "sigmoid 以外的分支属未验证路线",
    ),
    "train.diff-output-preservation-without-trigger": (
        "error",
        "SDTrainer.py:95-98：开了 preservation 却没有 trigger_word 会直接抛",
    ),
    "train.diff-output-preservation-with-te": (
        "error",
        "docs/models/sdxl.md 检查表第 7 条：preservation 与训 TE 语义冲突",
    ),
    "train.text-encoder": (
        "error",
        "Krea2 是静默空操作、MOSS 官方要求冻结（lora_special.py:284,678-680），"
        "开了要么白搭要么违反官方口径",
    ),
    "train.steps": ("error", "步数非正数，循环根本不成立"),
    "train.lr": ("error", "学习率非正数训不动"),
    "train.lr-too-large": (
        "error",
        "lr ≥ 1e-2 基本直接把 LoRA 推废，几小时白跑（阈值属工程判断，见 _LR_MAX）",
    ),
    # -- LoRA 维度 ---------------------------------------------------------
    "network.rank": ("error", "rank 非正数，LoRA 层建不出来"),
    "network.alpha": ("error", "alpha 非正数，缩放系数非法"),
    "network.rank-alpha-ratio": (
        "error",
        "alpha/rank 失配 = 有效学习率错一个量级，训出来的东西不可用",
    ),
    # -- 数据集 -----------------------------------------------------------
    "datasets.missing": ("error", "没有数据集就没图可训"),
    "datasets.item": ("error", "数据集条目不是 dict，解析器会抛"),
    "datasets.folder-path": ("error", "folder_path 为空读不到图"),
    "datasets.resolution": ("error", "resolution 必须是非空列表（config_modules.py:899）"),
    "datasets.resolution-value": ("error", "非正整数分辨率算不出 bucket"),
    "datasets.resolution-alignment": (
        "warn",
        "不对齐只是多 padding / 成本上升，bucket 仍能算；属实提示",
    ),
    "datasets.resolution-parallel": (
        "warn",
        "config_modules.py:1116-1152：resolution 是列表会把数据集拆成 ×N 份，成本翻倍。"
        "永久报、不按档位数降噪——2 档就已经踩坑，默认配置自己就命中，正该响",
    ),
    "datasets.caption-dropout-high": (
        "warn",
        "dropout 过高只是文本对齐变弱，仍能训（官方示例 0.05）",
    ),
    # -- 采样段 -----------------------------------------------------------
    "sample.missing": ("error", "没有 sample 段训练进程初始化时会抛"),
    "sample.sampler-mismatch": (
        "warn",
        "出图口径与训练调度器不一致而已；Krea2 甚至不读 sample.sampler（krea2.py:175）",
    ),
    "sample.prompts": ("warn", "没有采样提示词 = 看不到拐点，但训练能跑"),
    "sample.size": ("error", "width/height/sample_steps 非法会让采样直接失败"),
    "sample.size-alignment": ("error", "尺寸不能被 8 整除，VAE 下采样对不上"),
}


# ---------------------------------------------------------------------------
# 数据类型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigProblem:
    """一条校验结果。界面按 ``code`` 定位，不解析 ``message``。"""

    code: str
    severity: Severity
    message: str

    def __str__(self) -> str:
        return self.message


def _problem(code: str, message: str) -> ConfigProblem:
    """按登记表给 code 配上级别，避免每个出错点重复写一遍判级。"""
    try:
        severity, _ = _REGISTRY[code]
    except KeyError:  # pragma: no cover - 只有加新 code 忘了登记才会走到
        raise KeyError(f"未登记的校验 code: {code!r}") from None
    return ConfigProblem(code=code, severity=severity, message=message)


def known_codes() -> dict[str, Severity]:
    """所有已登记的 code 与级别（界面 / 文档用）。"""
    return {code: severity for code, (severity, _) in _REGISTRY.items()}


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _target(target: TargetModel | str) -> TargetModel:
    return TargetModel(target)


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


def _positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


# ---------------------------------------------------------------------------
# 详细校验
# ---------------------------------------------------------------------------


def validate_config_detailed(cfg: dict, target: TargetModel | str) -> list[ConfigProblem]:
    """返回全部问题（``error`` + ``warn``），空列表 = 完全干净。"""
    problems: list[ConfigProblem] = []

    try:
        t = _target(target)
    except ValueError:
        return [
            _problem(
                "target.unknown",
                f"未知目标模型 {target!r}，只支持 sdxl / krea2 / moss_image",
            )
        ]

    if not isinstance(cfg, dict):
        return [_problem("cfg.not-dict", "配置必须是 dict（yaml.safe_load 的结果）")]

    # ---- 顶层 -------------------------------------------------------------
    if cfg.get("job") != "extension":
        problems.append(
            _problem("job.missing", "顶层缺少 job: extension（ai-toolkit 要求，见 toolkit/job.py:6-20）")
        )

    config = cfg.get("config")
    if not isinstance(config, dict):
        problems.append(_problem("config.missing", "缺少 config 段"))
        return problems
    if not config.get("name"):
        problems.append(_problem("config.name", "config.name 为空"))

    process = config.get("process")
    if not isinstance(process, list) or not process:
        problems.append(_problem("config.process", "config.process 必须是非空列表"))
        return problems

    proc = process[0]
    if not isinstance(proc, dict):
        problems.append(_problem("config.process-item", "config.process[0] 必须是 dict"))
        return problems
    if proc.get("type") != "sd_trainer":
        problems.append(
            _problem("process.type", f"process[0].type 应为 sd_trainer，实际 {proc.get('type')!r}")
        )

    # ---- 模型段 -----------------------------------------------------------
    model = proc.get("model")
    if not isinstance(model, dict):
        problems.append(_problem("model.missing", "缺少 model 段"))
        model = {}
    if not model.get("name_or_path"):
        problems.append(
            _problem("model.name-or-path", "model.name_or_path 为空，ai-toolkit 会直接报错")
        )
    elif not _resolve_arch_hint(t, model):
        problems.append(
            _problem(
                "model.family-mismatch",
                f"model.name_or_path 与 target={t.value} 不匹配："
                f"{model.get('name_or_path')!r}，应使用 {PROFILES[t].train_base}",
            )
        )

    arch = str(model.get("arch") or "")
    expected_arch = {
        TargetModel.SDXL: "sdxl",
        TargetModel.KREA2: "krea2",
        TargetModel.MOSS: "moss_image",
    }[t]
    if arch != expected_arch:
        problems.append(_problem("model.arch", f"model.arch 应为 {expected_arch!r}，实际 {arch!r}"))
    if arch == "moss_image:base":
        problems.append(
            _problem(
                "model.arch-suffix",
                "model.arch 不能写 moss_image:base——冒号后会被截断成 moss_image"
                "（toolkit/config_modules.py:776-778）；Base 变体靠 name_or_path + "
                "model_kwargs.uncond_skip_layers 表达",
            )
        )

    if t is TargetModel.SDXL and model.get("quantize"):
        problems.append(
            _problem(
                "model.quantize-on-sdxl",
                "SDXL 不能开模型量化：quantize 目前只对 transformer 系生效"
                "（toolkit/config_modules.py:697-698），小显存请用 low_vram",
            )
        )

    qtype = model.get("qtype")
    if qtype is not None and not _KNOWN_QTYPE.match(str(qtype)):
        problems.append(
            _problem(
                "model.qtype-unknown",
                f"model.qtype={qtype!r} 不在已知量化家族里"
                "（toolkit/util/ostris_quant.py:185-211），会退回默认实现",
            )
        )

    vae_dtype = model.get("vae_dtype")
    if vae_dtype is not None and str(vae_dtype) != "float32":
        problems.append(
            _problem(
                "model.vae-dtype",
                f"model.vae_dtype={vae_dtype!r}：官方口径是 float32"
                "（config_modules.py:690 默认跟随 model dtype，无保护）",
            )
        )

    kwargs = model.get("model_kwargs") or {}
    if t is TargetModel.MOSS:
        if not isinstance(kwargs, dict):
            kwargs = {}
        max_len = kwargs.get("max_text_length")
        if isinstance(max_len, (int, float)) and max_len > _MOSS_MAX_TEXT_LENGTH:
            problems.append(
                _problem(
                    "model.max-text-length",
                    f"model.model_kwargs.max_text_length={max_len} 超过硬上限 "
                    f"{_MOSS_MAX_TEXT_LENGTH}（moss_image.py:70），会被截断",
                )
            )
        skip = kwargs.get("uncond_skip_layers")
        skip_list: list[int] = []
        if isinstance(skip, (list, tuple)):
            skip_list = [
                int(x) for x in skip if isinstance(x, (int, float)) and not isinstance(x, bool)
            ]
        if 22 not in skip_list:
            problems.append(
                _problem(
                    "model.moss-uncond-skip-layers",
                    "MOSS-Image 训练用 Base 建议带 model.model_kwargs.uncond_skip_layers: [22]"
                    "（官方 SLG 口径，docs/models/moss_image.md:195；推理侧生效）。"
                    "仅影响采样预览出图，不影响训练；缺了只是出图对照组不齐",
                )
            )

    # ---- 训练段 -----------------------------------------------------------
    train = proc.get("train")
    if not isinstance(train, dict):
        problems.append(_problem("train.missing", "缺少 train 段"))
        train = {}

    for key in ("steps", "lr", "batch_size", "noise_scheduler"):
        if train.get(key) in (None, ""):
            problems.append(_problem("train.required-empty", f"train.{key} 为空"))

    noise = train.get("noise_scheduler")
    if t is TargetModel.SDXL:
        if noise == "flowmatch":
            problems.append(
                _problem(
                    "train.noise-scheduler-flowmatch-on-sdxl",
                    "SDXL 不能用 noise_scheduler: flowmatch：SDXL 不是 flow matching 模型，"
                    "请用 ddpm（toolkit/config_modules.py:383）",
                )
            )
        elif noise not in (None, "", "ddpm"):
            problems.append(
                _problem(
                    "train.noise-scheduler-not-ddpm",
                    f"SDXL 的 noise_scheduler 建议 ddpm，实际 {noise!r}",
                )
            )
    elif noise != "flowmatch":
        problems.append(
            _problem(
                "train.noise-scheduler-not-flowmatch",
                f"{t.value} 必须用 noise_scheduler: flowmatch，实际 {noise!r}"
                "（Krea2 见 krea2.py:201-202，MOSS 见 moss_image.py:202-205）",
            )
        )

    if t is TargetModel.KREA2:
        ttype = train.get("timestep_type")
        if ttype == "sigmoid":
            problems.append(
                _problem(
                    "train.timestep-type-sigmoid",
                    "Krea2 的 timestep_type 不能是 sigmoid——那是 FLUX 的口径，"
                    "会训坏；必须 linear",
                )
            )
        elif ttype not in _TIMESTEP_TYPES:
            problems.append(
                _problem(
                    "train.timestep-type-unknown",
                    f"Krea2 的 timestep_type={ttype!r} 不在白名单 {_TIMESTEP_TYPES} 里",
                )
            )
        elif ttype != "linear":
            # 白名单内但不是 linear：能跑，口径偏了
            problems.append(
                _problem(
                    "train.timestep-type-not-linear",
                    f"Krea2 的 timestep_type 应为 linear，实际 {ttype!r}",
                )
            )

    preservation = train.get("diff_output_preservation")
    if preservation:
        if not proc.get("trigger_word"):
            problems.append(
                _problem(
                    "train.diff-output-preservation-without-trigger",
                    "开了 train.diff_output_preservation 却没有 process.trigger_word，"
                    "SDTrainer.py:95-98 会直接抛",
                )
            )
        if train.get("train_text_encoder"):
            problems.append(
                _problem(
                    "train.diff-output-preservation-with-te",
                    "train.diff_output_preservation 不能与 train_text_encoder 同开",
                )
            )

    if t in (TargetModel.KREA2, TargetModel.MOSS) and train.get("train_text_encoder"):
        problems.append(
            _problem(
                "train.text-encoder",
                f"{t.value} 的文本编码器不参与训练，train_text_encoder 应为 false",
            )
        )

    steps = train.get("steps")
    if isinstance(steps, (int, float)) and not isinstance(steps, bool) and steps <= 0:
        problems.append(_problem("train.steps", f"train.steps 必须为正数，实际 {steps}"))
    lr = train.get("lr")
    if isinstance(lr, (int, float)) and not isinstance(lr, bool):
        if lr <= 0:
            problems.append(_problem("train.lr", f"train.lr 必须为正数，实际 {lr}"))
        elif lr > _LR_MAX:
            problems.append(
                _problem("train.lr-too-large", f"train.lr={lr} 过大，LoRA 通常 1e-4 量级")
            )

    # ---- LoRA 维度 ---------------------------------------------------------
    network = proc.get("network") or {}
    if not isinstance(network, dict):
        network = {}
    rank = network.get("linear")
    alpha = network.get("linear_alpha")
    if not _positive(rank):
        problems.append(
            _problem("network.rank", f"network.linear (rank) 必须为正数，实际 {rank!r}")
        )
    if not _positive(alpha):
        problems.append(
            _problem("network.alpha", f"network.linear_alpha 必须为正数，实际 {alpha!r}")
        )
    if _positive(rank) and _positive(alpha):
        ratio = alpha / rank
        if ratio < _RANK_ALPHA_MIN or ratio > _RANK_ALPHA_MAX:
            problems.append(
                _problem(
                    "network.rank-alpha-ratio",
                    f"rank={rank} 与 alpha={alpha} 严重不匹配（alpha/rank={ratio:.2f}），"
                    "常见做法是 alpha 等于 rank",
                )
            )

    # ---- 数据集 -----------------------------------------------------------
    datasets = proc.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        problems.append(_problem("datasets.missing", "缺少 datasets，训练没有图可读"))
        datasets = []
    for i, ds in enumerate(datasets):
        if not isinstance(ds, dict):
            problems.append(_problem("datasets.item", f"datasets[{i}] 必须是 dict"))
            continue
        if not ds.get("folder_path"):
            problems.append(
                _problem("datasets.folder-path", f"datasets[{i}].folder_path 为空")
            )
        dropout = ds.get("caption_dropout_rate")
        if (
            isinstance(dropout, (int, float))
            and not isinstance(dropout, bool)
            and dropout > _CAPTION_DROPOUT_MAX
        ):
            problems.append(
                _problem(
                    "datasets.caption-dropout-high",
                    f"datasets[{i}].caption_dropout_rate={dropout} 偏高"
                    f"（官方示例 0.05，超过 {_CAPTION_DROPOUT_MAX} 会明显削弱文本对齐）",
                )
            )
        res = ds.get("resolution")
        if not isinstance(res, (list, tuple)) or not res:
            problems.append(
                _problem("datasets.resolution", f"datasets[{i}].resolution 必须是非空列表")
            )
            continue
        if len(res) > 1:
            problems.append(
                _problem(
                    "datasets.resolution-parallel",
                    f"datasets[{i}].resolution 给了 {len(res)} 档 = 数据集复制成 "
                    f"{len(res)} 份（config_modules.py:1116-1152），"
                    f"steps ≈ 张数 × num_repeats × {len(res)}；只覆盖更多尺寸，"
                    f"不会让训练更快，训练量与耗时同倍增加",
                )
            )
        for value in res:
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                problems.append(
                    _problem(
                        "datasets.resolution-value",
                        f"datasets[{i}].resolution 含非法值 {value!r}",
                    )
                )
            elif value % _RECOMMENDED_ALIGNMENT != 0:
                problems.append(
                    _problem(
                        "datasets.resolution-alignment",
                        f"datasets[{i}].resolution={value} 不是 {_RECOMMENDED_ALIGNMENT} 的倍数"
                        f"（最低要求 {_TARGET_DIVISIBILITY} 的倍数），会多 padding",
                    )
                )

    # ---- 采样段 -----------------------------------------------------------
    sample = proc.get("sample")
    if not isinstance(sample, dict):
        problems.append(_problem("sample.missing", "缺少 sample 段"))
        return problems
    if sample.get("sampler") != noise:
        problems.append(
            _problem(
                "sample.sampler-mismatch",
                f"sample.sampler={sample.get('sampler')!r} 与 train.noise_scheduler={noise!r} "
                "不一致（config/examples/train_lora_flux_24gb.yaml:74），出图口径会对不上",
            )
        )
    if not sample.get("prompts"):
        problems.append(_problem("sample.prompts", "sample.prompts 为空，训练期不会有采样图"))
    for key in ("width", "height", "sample_steps"):
        value = sample.get(key)
        if not _positive(value):
            problems.append(
                _problem("sample.size", f"sample.{key} 必须为正数，实际 {value!r}")
            )
    for key in ("width", "height"):
        value = sample.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value % _SIZE_MULTIPLE != 0:
            problems.append(
                _problem(
                    "sample.size-alignment",
                    f"sample.{key}={value} 需要能被 {_SIZE_MULTIPLE} 整除",
                )
            )

    return problems
