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

判级经过 ``docs/models/_review-severity.md``（阿研，2026-11-27）逐条审核：
48 条里 34 条同意，14 条按审核意见改过；改的那 14 条在 code 旁边标了
``[review #n]``，方便回溯。
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
# 出处：docs/models/sdxl.md:212-213 契约默认 1e-4；无源码硬约束，属工程判断。
# [review #18] 阈值压到 5e-3：1e-2 时「正好等于 1e-2」不响，边界很难解释。
_LR_MAX = 5e-3

# alpha/rank 的合理区间（有效学习率 ≈ lr × alpha/rank）
# 出处：docs/models/sdxl.md:213（契约默认 alpha == rank）；kohya 官方 SDXL 示例 32/16 = 0.5。
# [review #1] 原区间 [0.5, 2.0] 会把 ai-toolkit 自己的默认值（rank 4 / alpha 1.0 = 0.25）
# 判成 error（config_modules.py:187-191），属假阳性。放宽到 [0.25, 4.0] 并降为 warn：
# ratio 偏小只是有效学习率变小、训练偏保守，不是「不可用」。
_RANK_ALPHA_MIN = 0.25
_RANK_ALPHA_MAX = 4.0

# timestep_type 白名单
# 出处：toolkit/samplers/custom_flowmatch_sampler.py:127-216（linear / weighted / sigmoid /
#       shift / flux_shift / lumina2_shift，见 :145）；
#       jobs/process/BaseSDTrainProcess.py:1181-1261（one_step / two_step / four_step /
#       eight_step / next_sample）。
# [review #2] 原表漏了 shift / flux_shift / lumina2_shift 三个源码明确处理的合法分支——
# 我们自己文档给的 MOSS `timestep_type: shift`（docs/models/moss_image.md:179）会被判
# 「未知」。next_sample / one_step 保留，它们确实有分支。
_TIMESTEP_TYPES = (
    "sigmoid",
    "linear",
    "shift",
    "flux_shift",
    "lumina2_shift",
    "lognorm_blend",
    "next_sample",
    "weighted",
    "one_step",
    "two_step",
    "four_step",
    "eight_step",
)

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

# flow 系（krea2 / moss）的采样尺寸要按 bucket 整除量判
# [review #19] krea2.py:204-206、moss_image.py:207-208 的 get_bucket_divisibility() 都是 16，
# 只查 8 的倍数放过了一半会被 padding 的尺寸。
_FLOW_SIZE_MULTIPLE = 16


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
        "warn",
        # [review #11] 判定靠 name_or_path 子串（config_modules.py:267-271），本地权重
        # `sd_xl_base_1.0.safetensors`（含 sd_xl 不含 sdxl）、`moss_image_base.safetensors`
        # 都会被误杀。子串启发式不该有拦截权，降为 warn。
        "路径与 target 看起来不是同一家族，可能拿错底座；但子串启发式会误杀本地"
        "权重名（如 sd_xl_base_1.0.safetensors），只提示不拦",
    ),
    "model.arch": (
        "error",
        # [review #6] 条件放宽：config_modules.py:809-831 允许由 is_xl 推出 sdxl，
        # 官方 flux 示例只写 is_flux；SDXL 同理。arch 与家族标志都缺才 error（会落 sd1）。
        "toolkit/config_modules.py:809-831：arch 不匹配会落回 sd1，整条训练路径全错；"
        "但 is_xl: true 可以推出 sdxl，两者都缺才拦",
    ),
    "model.arch-suffix": (
        "warn",
        # [review #12] config_modules.py:776-778 是通用逻辑，krea2:raw、sdxl:foo 的 tag
        # 一样被丢弃；丢 tag 不改变训练路径（变体由 name_or_path + model_kwargs 决定）。
        "toolkit/config_modules.py:776-778：冒号后面会被 split 掉，等于静默丢变体；"
        "属配置卫生问题，不改变训练路径",
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
    "train.required-empty": (
        "warn",
        # [review #7] 四个键全有默认值（noise_scheduler ddpm :383、steps 1000 :386、
        # lr 1e-6 :387、batch_size 1 :399），省略即走默认、配置能跑；只拦显式空值/非法值。
        "steps/lr/batch_size/noise_scheduler 写成了空值或 None；省略会走默认，"
        "显式空值才拦（config_modules.py:383,386,387,399）",
    ),
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
        "warn",
        # [review #3] custom_flowmatch_sampler.py:131-144 只是 timestep 分布不同（中心 t≈500），
        # kohya 的 MOSS 文档也只说「与预训练分布差很多」，仍能训；拦成 error 属假阳性。
        "timestep_type: sigmoid 是 FLUX 口径，flow 系（krea2/moss）推荐 linear 或 shift；"
        "能训，但 timestep 分布偏中心，效果打折",
    ),
    "train.timestep-type-unknown": (
        "error",
        "custom_flowmatch_sampler.py:127-216 + BaseSDTrainProcess.py:1181-1261："
        "白名单外会被当成未知分支，等于没设",
    ),
    "train.timestep-type-not-linear": (
        "warn",
        # [review #4] 实现注释自己写「白名单内但不是 linear：能跑，口径偏了」，登记表给
        # error 属自相矛盾（docs/models/krea2.md:163 只是推荐起点）。
        "Krea2 的推荐口径是 linear，当前取值属未验证路线；能跑，只是口径偏了",
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
        "warn",
        # [review #5] lora_special.py:284,678-680：krea2/moss 的 Qwen TE 建出 0 个 LoRA，
        # 只打印 skipped、不抛错；MOSS 官方要求冻结。属「白搭」不属「白跑」。
        "krea2/moss 开了 train_text_encoder 也不会生效（TE 侧建 0 个 LoRA，"
        "只打印 skipped，不抛错）；MOSS 官方要求 TE 冻结，请保持 false",
    ),
    "train.steps": ("error", "步数非正数，循环根本不成立"),
    "train.lr": ("error", "学习率非正数训不动"),
    "train.lr-too-large": (
        "error",
        # [review #18] 阈值属工程判断，从 1e-2 压到 5e-3（1e-2 时边界不响）。
        "lr ≥ 5e-3 基本直接把 LoRA 推废，几小时白跑（经验值，见 _LR_MAX）",
    ),
    # -- LoRA 维度 ---------------------------------------------------------
    "network.rank": ("error", "rank 非正数，LoRA 层建不出来"),
    "network.alpha": ("error", "alpha 非正数，缩放系数非法"),
    "network.rank-alpha-ratio": (
        "warn",
        # [review #1] alpha/rank 只影响有效学习率（lr × alpha/rank）；ai-toolkit 默认
        # rank 4 / alpha 1.0 = 0.25，判 error 是假阳性。
        "alpha/rank 偏离常见区间：有效学习率 = lr × alpha/rank，偏离只是训练偏保守或偏激进，"
        "不是不可用",
    ),
    # -- 数据集 -----------------------------------------------------------
    "datasets.missing": ("error", "没有数据集就没图可训"),
    "datasets.item": ("error", "数据集条目不是 dict，解析器会抛"),
    "datasets.folder-path": (
        "error",
        # [review #10] config_modules.py:935 另定义 dataset_path，:1127 把两者一起拆分；
        # 只认 folder_path 会拦掉合法写法。
        "folder_path（或别名 dataset_path）为空读不到图（config_modules.py:935,1127）",
    ),
    "datasets.resolution": (
        "error",
        # [review #9] 原出处 :899 写错；config_modules.py:959 明确 resolution: int 默认 512，
        # 单个 int 是合法值，:1122-1152 只在 list 时复制数据集。
        "resolution 必须是正整数或非空列表（config_modules.py:959 默认为 int）",
    ),
    "datasets.resolution-value": ("error", "非正整数分辨率算不出 bucket"),
    "datasets.resolution-alignment": (
        "warn",
        "不对齐只是多 padding / 成本上升，bucket 仍能算；属实提示",
    ),
    "datasets.resolution-parallel": (
        "warn",
        # [review #15] 原文案「训练量与耗时同倍增加」不准确：steps 固定时训练时长不变，
        # 变的是每张图的曝光次数（等效 epochs 除以 N）。
        "config_modules.py:1116-1152：resolution 是列表会把数据集拆成 ×N 份。"
        "同样步数下每张图被看到的次数变少；要等价效果需把 steps ×N。"
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
        # [review #13] krea2.py:411-467、moss_image.py:371-411 的 generate_single_image
        # 不读 gen_config.sampler，且 flow 系 train.noise_scheduler 必为 flowmatch →
        # 恒响的噪音。只在 SDXL 保留（sample.sampler 会真的影响出图口径）。
        "SDXL 的 sample.sampler 与 train.noise_scheduler 不一致，出图口径会对不上；"
        "krea2/moss 不读该键，不校验",
    ),
    "sample.prompts": ("warn", "没有采样提示词 = 看不到拐点，但训练能跑"),
    "sample.size": (
        "error",
        # [review #8] SampleConfig 的 width/height 默认 512、sample_steps 默认 20
        # （config_modules.py:82-125），省略是合法写法。
        "width/height/sample_steps 写了非正数，会让采样直接失败（省略则走默认）",
    ),
    "sample.size-alignment": (
        "error",
        # [review #19] flow 系按 get_bucket_divisibility()=16 判，SDXL 按 8 判。
        "尺寸不能被 bucket 整除量（flow 系 16 / SDXL 8）整除，VAE 下采样对不上",
    ),
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
    # ai-toolkit 读 arch 时会先按冒号截断（config_modules.py:776-778），所以
    # ``moss_image:base`` 实际落到 ``moss_image``，主干是对的；只由下面的
    # model.arch-suffix 提示 tag 丢失。
    arch_base = arch.split(":", 1)[0]
    expected_arch = {
        TargetModel.SDXL: "sdxl",
        TargetModel.KREA2: "krea2",
        TargetModel.MOSS: "moss_image",
    }[t]
    if arch_base != expected_arch:
        # [review #6] config_modules.py:809-831 允许由 is_xl 推出 sdxl（官方 flux 示例
        # 只写 is_flux: true）；家族标志正确时不报，两者都缺才拦（会落 sd1）。
        implied_by_flag = arch == "" and t is TargetModel.SDXL and bool(model.get("is_xl"))
        if not implied_by_flag:
            if arch:
                hint = ""
            elif t is TargetModel.SDXL:
                hint = "；也可以省掉 arch 但显式写 is_xl: true"
            else:
                hint = "；缺失会落回 sd1，训练路径全错"
            problems.append(
                _problem(
                    "model.arch",
                    f"model.arch 应为 {expected_arch!r}，实际 {arch!r}{hint}",
                )
            )
    if ":" in arch:
        # [review #12] config_modules.py:776-778 对全 target 都是截断逻辑，不改变训练路径。
        problems.append(
            _problem(
                "model.arch-suffix",
                f"model.arch={arch!r} 带冒号：冒号后会被截断成 {arch.split(':')[0]!r}"
                "（toolkit/config_modules.py:776-778），变体 tag 静默丢失；"
                "变体应由 name_or_path + model_kwargs 表达",
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
    # [review #17] 量化关着的时候查 qtype 家族是噪音，只在真正会用到时校验。
    if (model.get("quantize") or model.get("quantize_te")) and (
        qtype is not None and not _KNOWN_QTYPE.match(str(qtype))
    ):
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
        # [review #16] 只在键存在且不含 22 时提示；缺省不算错（训练侧源码根本没读它）。
        if "uncond_skip_layers" in kwargs and 22 not in skip_list:
            problems.append(
                _problem(
                    "model.moss-uncond-skip-layers",
                    f"MOSS-Image 的 model.model_kwargs.uncond_skip_layers={skip!r} 不含 22："
                    "官方 SLG 口径建议 22（22/24/26 均可，docs/models/moss_image.md:195）。"
                    "这是推理侧参数，不影响训练，只是采样预览的对照组会不齐",
                )
            )

    # ---- 训练段 -----------------------------------------------------------
    train = proc.get("train")
    if not isinstance(train, dict):
        problems.append(_problem("train.missing", "缺少 train 段"))
        train = {}

    # [review #7] 四个键都有默认值，省略合法；只有显式写空/None 才提示。
    for key in ("steps", "lr", "batch_size", "noise_scheduler"):
        if key in train and train.get(key) in (None, ""):
            problems.append(_problem("train.required-empty", f"train.{key} 显式写成了空值"))

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

    # timestep_type：flow 系（krea2 / moss）才走 set_train_timesteps，两个模型都要查。
    # [review #14] 原实现只在 KREA2 分支里跑，MOSS 完全没校验——MOSS 恰恰最需要它
    # （预训练 logit-normal，docs/models/moss_image.md:179）；非法值会在
    # custom_flowmatch_sampler.py:230 抛 ValueError，属真硬错。
    if t in (TargetModel.KREA2, TargetModel.MOSS):
        ttype = train.get("timestep_type")
        if ttype is not None and ttype not in _TIMESTEP_TYPES:
            problems.append(
                _problem(
                    "train.timestep-type-unknown",
                    f"{t.value} 的 timestep_type={ttype!r} 不在白名单 "
                    f"{_TIMESTEP_TYPES} 里（custom_flowmatch_sampler.py:230 会抛 ValueError）",
                )
            )
        elif ttype == "sigmoid":
            problems.append(
                _problem(
                    "train.timestep-type-sigmoid",
                    f"{t.value} 的 timestep_type 是 sigmoid（FLUX 口径）：能训，但 timestep "
                    "分布偏中心，与预训练分布差得多，效果打折",
                )
            )
        elif t is TargetModel.KREA2 and ttype is not None and ttype != "linear":
            # 白名单内但不是 linear：能跑，口径偏了（实现注释自己也这么写）
            problems.append(
                _problem(
                    "train.timestep-type-not-linear",
                    f"Krea2 的 timestep_type 推荐 linear，实际 {ttype!r}（能跑，口径偏了）",
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
        # [review #10] config_modules.py:935 定义 dataset_path，:1127 两个键都会被拆分。
        if not (ds.get("folder_path") or ds.get("dataset_path")):
            problems.append(
                _problem(
                    "datasets.folder-path",
                    f"datasets[{i}].folder_path（或 dataset_path）为空",
                )
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
        # [review #9] config_modules.py:959 默认就是 int，单个 int 合法；:1122-1152 只在
        # list 时复制数据集。
        res = ds.get("resolution")
        if isinstance(res, int) and not isinstance(res, bool):
            res_values: list[Any] = [res]
        elif isinstance(res, (list, tuple)) and res:
            res_values = list(res)
        else:
            problems.append(
                _problem(
                    "datasets.resolution",
                    f"datasets[{i}].resolution 必须是正整数或非空列表，实际 {res!r}",
                )
            )
            continue
        if len(res_values) > 1:
            problems.append(
                _problem(
                    "datasets.resolution-parallel",
                    f"datasets[{i}].resolution 给了 {len(res_values)} 档 = 数据集复制成 "
                    f"{len(res_values)} 份（config_modules.py:1116-1152）：同样步数下每张图"
                    f"被看到的次数变少；要等价效果需把 steps ×{len(res_values)}",
                )
            )
        for value in res_values:
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
    # [review #13] krea2/moss 的 generate_single_image 不读 sample.sampler
    # （krea2.py:411-467、moss_image.py:371-411），且 train.noise_scheduler 必为 flowmatch
    # → 每条 flow 系配置都恒响一次，属噪音。SDXL 的 sample.sampler 会真的影响出图口径。
    if t is TargetModel.SDXL and sample.get("sampler") != noise:
        problems.append(
            _problem(
                "sample.sampler-mismatch",
                f"sample.sampler={sample.get('sampler')!r} 与 train.noise_scheduler={noise!r} "
                "不一致（config/examples/train_lora_flux_24gb.yaml:74），出图口径会对不上",
            )
        )
    if not sample.get("prompts"):
        problems.append(_problem("sample.prompts", "sample.prompts 为空，训练期不会有采样图"))
    # [review #8] SampleConfig 的 width/height 默认 512、sample_steps 默认 20
    # （config_modules.py:82-125），省略合法；只在键存在时校验。
    for key in ("width", "height", "sample_steps"):
        if key not in sample:
            continue
        value = sample.get(key)
        if not _positive(value):
            problems.append(
                _problem("sample.size", f"sample.{key} 必须为正数，实际 {value!r}")
            )
    # [review #19] flow 系按 bucket 整除量 16 判（krea2.py:204-206、moss_image.py:207-208），
    # SDXL 保持 8。
    size_multiple = _FLOW_SIZE_MULTIPLE if t in (TargetModel.KREA2, TargetModel.MOSS) else _SIZE_MULTIPLE
    for key in ("width", "height"):
        value = sample.get(key)
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value % size_multiple != 0
        ):
            problems.append(
                _problem(
                    "sample.size-alignment",
                    f"sample.{key}={value} 需要能被 {size_multiple} 整除"
                    + ("（flow 系的 bucket 整除量）" if size_multiple != _SIZE_MULTIPLE else ""),
                )
            )

    return problems
