"""配置层测试：三模型的必填字段、危险组合、error/warn 分级、yaml round-trip。

夹具用 ``scratch``（仓库内可写目录），不要用 ``tmp_path``——本机系统临时目录
在沙箱外，见 ``tests/conftest.py``。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from loraforge.configs import (
    VRAM_TIERS,
    ConfigProblem,
    acceptance_dict,
    build_config,
    default_plan,
    known_codes,
    krea2_mu,
    plan_for_tier,
    tier_for_vram,
    validate_config,
    validate_config_detailed,
)
from loraforge.models import TargetModel

TARGETS = [TargetModel.SDXL, TargetModel.KREA2, TargetModel.MOSS]


def _proc(cfg: dict) -> dict:
    return cfg["config"]["process"][0]


@pytest.fixture
def cfgs() -> dict[TargetModel, dict]:
    return {t: build_config(t, default_plan(t)) for t in TARGETS}


def _codes(items: list[ConfigProblem]) -> set[str]:
    return {p.code for p in items}


def _warn_codes(cfg: dict, target: TargetModel) -> set[str]:
    return _codes([p for p in validate_config_detailed(cfg, target) if p.severity == "warn"])


def _error_codes(cfg: dict, target: TargetModel) -> set[str]:
    return _codes([p for p in validate_config_detailed(cfg, target) if p.severity == "error"])


# ---------------------------------------------------------------------------
# 必填字段
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", TARGETS)
def test_build_config_has_required_fields(target: TargetModel) -> None:
    plan = default_plan(target)
    cfg = build_config(target, plan)

    assert cfg["job"] == "extension"
    assert cfg["config"]["name"]
    assert isinstance(cfg["config"]["process"], list) and cfg["config"]["process"]

    proc = _proc(cfg)
    assert proc["type"] == "sd_trainer"
    assert proc["training_folder"] == plan.output_dir
    assert proc["trigger_word"] == plan.trigger

    net = proc["network"]
    assert net["type"] == "lora"
    assert net["linear"] == plan.rank
    assert net["linear_alpha"] == plan.alpha

    model = proc["model"]
    assert model["name_or_path"] == plan.model_path
    assert model["arch"] in {"sdxl", "krea2", "moss_image"}
    assert model["dtype"] in {"float16", "bf16"}

    train = proc["train"]
    assert train["steps"] == plan.steps
    assert train["batch_size"] == plan.batch_size
    assert train["lr"] == plan.lr
    assert train["noise_scheduler"]
    assert train["train_text_encoder"] is False

    dataset = proc["datasets"][0]
    assert dataset["folder_path"] == plan.dataset_dir
    assert dataset["caption_ext"] == "txt"
    assert dataset["buckets"] is True
    assert dataset["resolution"] == plan.resolutions

    sample = proc["sample"]
    assert sample["sampler"] == train["noise_scheduler"]
    assert sample["prompts"]
    assert sample["width"] == sample["height"] == 1024


def test_sdxl_is_legacy_path_without_quantize(cfgs: dict[TargetModel, dict]) -> None:
    proc = _proc(cfgs[TargetModel.SDXL])
    assert proc["model"]["arch"] == "sdxl"
    # SDXL 不是 flow matching，不能配 flowmatch（docs/models 里反复强调的坑）
    assert proc["train"]["noise_scheduler"] == "ddpm"
    # docs/models/sdxl.md:228 的起点是「不设」min_snr_gamma
    assert "min_snr_gamma" not in proc["train"]
    # docs/models/sdxl.md:233：官方 VAE 用 fp32
    assert proc["model"]["vae_dtype"] == "float32"
    # quantize 只对 transformer 系开放
    assert "quantize" not in proc["model"] or not proc["model"]["quantize"]
    assert proc["sample"]["sampler"] == "ddpm"
    assert proc["sample"]["sample_steps"] == 25
    assert proc["sample"]["guidance_scale"] == 6.0


def test_krea2_uses_flowmatch_linear_not_sigmoid(cfgs: dict[TargetModel, dict]) -> None:
    proc = _proc(cfgs[TargetModel.KREA2])
    assert proc["model"]["name_or_path"] == "krea/Krea-2-Raw"
    assert proc["model"]["arch"] == "krea2"
    assert proc["train"]["noise_scheduler"] == "flowmatch"
    assert proc["train"]["timestep_type"] == "linear"
    # 这是社区反复踩的坑：FLUX 的 sigmoid 不许搬过来
    assert proc["train"]["timestep_type"] != "sigmoid"
    # docs/models/krea2.md:222：先跑通再开 diff_output_preservation
    assert proc["train"]["diff_output_preservation"] is False
    assert proc["model"]["quantize"] is True
    assert proc["model"]["qtype"] == "qfloat8"
    # docs/models/krea2.md:224：训练在 Raw 上采样，52 步、要生效 4.5
    assert proc["sample"]["sampler"] == "flowmatch"
    assert proc["sample"]["sample_steps"] == 52
    # ai-toolkit 生效值 = yaml 值 − 1（krea2.py:455），所以写 5.5
    assert proc["sample"]["guidance_scale"] == 5.5


def test_moss_uses_base_with_uncond_skip_layers(cfgs: dict[TargetModel, dict]) -> None:
    proc = _proc(cfgs[TargetModel.MOSS])
    assert proc["model"]["name_or_path"] == "OpenMOSS-Team/MOSS-Image-Base"
    # arch 只能是 moss_image，冒号后面的 tag 会被 ai-toolkit 截断
    assert proc["model"]["arch"] == "moss_image"
    kwargs = proc["model"]["model_kwargs"]
    assert kwargs["uncond_skip_layers"] == [22]
    assert kwargs["sample_shift"] == 5.0
    assert proc["train"]["noise_scheduler"] == "flowmatch"
    assert proc["sample"]["sampler"] == "flowmatch"
    # docs/models/moss_image.md:193：官方 sample_steps ≥50
    assert proc["sample"]["sample_steps"] == 50
    assert proc["sample"]["guidance_scale"] == 3.0


# ---------------------------------------------------------------------------
# 校验：默认方案必须干净，错误配置必须被抓出来
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", TARGETS)
def test_default_configs_validate_clean(cfgs: dict[TargetModel, dict], target: TargetModel) -> None:
    assert validate_config(cfgs[target], target) == []
    # 默认方案只允许出现「分辨率档位多 → 数据集被拆成多份」这条提示性告警
    assert _error_codes(cfgs[target], target) == set()
    assert _warn_codes(cfgs[target], target) == {"datasets.resolution-parallel"}


def test_validate_rejects_missing_job_and_process(cfgs: dict[TargetModel, dict]) -> None:
    target = TargetModel.SDXL
    cfg = copy.deepcopy(cfgs[target])

    del cfg["job"]
    cfg["config"]["process"] = []
    problems = validate_config(cfg, target)
    assert problems
    text = " ".join(problems)
    assert "job" in text
    assert "process" in text


def test_model_path_mismatch_is_only_a_warning(cfgs: dict[TargetModel, dict]) -> None:
    """[review #11] 家族判定靠 name_or_path 子串，会误杀本地权重名，只提示不拦。"""
    target = TargetModel.KREA2
    cfg = copy.deepcopy(cfgs[target])
    _proc(cfg)["model"]["name_or_path"] = "black-forest-labs/FLUX.1-dev"

    detailed = validate_config_detailed(cfg, target)
    hits = [p for p in detailed if p.code == "model.family-mismatch"]
    assert hits
    assert hits[0].severity == "warn"
    assert "不匹配" in hits[0].message
    assert validate_config(cfg, target) == []


def test_krea2_sigmoid_is_only_a_warning(cfgs: dict[TargetModel, dict]) -> None:
    """[review #3] sigmoid 只是 timestep 分布不同，kohya 也说仍能训。"""
    target = TargetModel.KREA2
    cfg = copy.deepcopy(cfgs[target])
    _proc(cfg)["train"]["timestep_type"] = "sigmoid"

    detailed = validate_config_detailed(cfg, target)
    hits = [p for p in detailed if p.code == "train.timestep-type-sigmoid"]
    assert hits
    assert hits[0].severity == "warn"
    assert "sigmoid" in hits[0].message
    assert validate_config(cfg, target) == []


def test_validate_rejects_sdxl_flowmatch(cfgs: dict[TargetModel, dict]) -> None:
    target = TargetModel.SDXL
    cfg = copy.deepcopy(cfgs[target])
    _proc(cfg)["train"]["noise_scheduler"] = "flowmatch"

    problems = validate_config(cfg, target)
    assert problems
    assert any("flowmatch" in p for p in problems)


def test_moss_uncond_skip_layers_is_only_a_warning(cfgs: dict[TargetModel, dict]) -> None:
    """docs/models/moss_image.md:195：uncond_skip_layers 是推理侧 SLG 口径。

    训练侧源码没读它，缺了只是出图对照组不齐，不该拦导出；[review #16] 只在键
    存在且不含 22 时提示，完全省略不算错。
    """
    target = TargetModel.MOSS
    cfg = copy.deepcopy(cfgs[target])
    _proc(cfg)["model"]["model_kwargs"]["uncond_skip_layers"] = [20]

    detailed = validate_config_detailed(cfg, target)
    hits = [p for p in detailed if p.code == "model.moss-uncond-skip-layers"]
    assert hits, [p.code for p in detailed]
    assert hits[0].severity == "warn"
    assert "uncond_skip_layers" in hits[0].message
    assert "不影响训练" in hits[0].message
    assert validate_config(cfg, target) == []

    # 键整个省略时不该响（缺省本来就不是 [22]）
    omitted = copy.deepcopy(cfgs[target])
    _proc(omitted)["model"]["model_kwargs"].pop("uncond_skip_layers")
    assert "model.moss-uncond-skip-layers" not in _codes(validate_config_detailed(omitted, target))


def test_moss_arch_suffix_is_only_a_warning(cfgs: dict[TargetModel, dict]) -> None:
    """[review #12] 冒号后被截断是全 target 通用问题，丢 tag 不改变训练路径。"""
    target = TargetModel.MOSS
    cfg = copy.deepcopy(cfgs[target])
    _proc(cfg)["model"]["arch"] = "moss_image:base"

    detailed = validate_config_detailed(cfg, target)
    hits = [p for p in detailed if p.code == "model.arch-suffix"]
    assert hits
    assert hits[0].severity == "warn"
    assert "moss_image" in hits[0].message
    assert validate_config(cfg, target) == []

    # krea2:raw 同样会被截断，不该只查 moss
    kcfg = copy.deepcopy(cfgs[TargetModel.KREA2])
    _proc(kcfg)["model"]["arch"] = "krea2:raw"
    assert "model.arch-suffix" in _codes(validate_config_detailed(kcfg, TargetModel.KREA2))


def test_rank_alpha_ratio_is_only_a_warning(cfgs: dict[TargetModel, dict]) -> None:
    """[review #1] alpha/rank 只影响有效学习率 = lr × alpha/rank；默认值 4/1.0 是 0.25。"""
    target = TargetModel.SDXL
    cfg = copy.deepcopy(cfgs[target])
    proc = _proc(cfg)
    proc["network"]["linear"] = 8
    proc["network"]["linear_alpha"] = 64  # ratio 8.0，明显偏大

    detailed = validate_config_detailed(cfg, target)
    hits = [p for p in detailed if p.code == "network.rank-alpha-ratio"]
    assert hits
    assert hits[0].severity == "warn"
    assert "alpha" in hits[0].message
    assert validate_config(cfg, target) == []

    # ai-toolkit 自己的默认值 4 / 1.0 = 0.25 不许被判成问题
    default_cfg = copy.deepcopy(cfgs[target])
    default_cfg["config"]["process"][0]["network"].update(linear=4, linear_alpha=1.0)
    assert "network.rank-alpha-ratio" not in _codes(
        validate_config_detailed(default_cfg, target)
    )


def test_sampler_mismatch_is_only_a_warning_for_sdxl(cfgs: dict[TargetModel, dict]) -> None:
    """[review #13] flow 系不读 sample.sampler，恒响是噪音；只在 SDXL 提示。"""
    target = TargetModel.SDXL
    cfg = copy.deepcopy(cfgs[target])
    _proc(cfg)["sample"]["sampler"] = "euler"

    assert "sample.sampler-mismatch" in _warn_codes(cfg, target)
    assert validate_config(cfg, target) == []

    for flow_target in (TargetModel.KREA2, TargetModel.MOSS):
        flow_cfg = copy.deepcopy(cfgs[flow_target])
        _proc(flow_cfg)["sample"]["sampler"] = "ddpm"
        assert "sample.sampler-mismatch" not in _codes(
            validate_config_detailed(flow_cfg, flow_target)
        )


def test_resolution_not_aligned_is_only_a_warning(cfgs: dict[TargetModel, dict]) -> None:
    """分辨率没对齐 64 只是多 padding（docs/models/krea2.md:220），不拦导出。"""
    target = TargetModel.SDXL
    cfg = copy.deepcopy(cfgs[target])
    _proc(cfg)["datasets"][0]["resolution"] = [1000]

    assert "datasets.resolution-alignment" in _warn_codes(cfg, target)
    assert validate_config(cfg, target) == []


def test_resolution_parallel_explains_the_multiplier(cfgs: dict[TargetModel, dict]) -> None:
    """N 档分辨率 = 数据集复制 N 份；文案要把乘数讲明白，且 2 档就要报。"""
    target = TargetModel.SDXL
    cfg = copy.deepcopy(cfgs[target])
    _proc(cfg)["datasets"][0]["resolution"] = [768, 1024]

    hits = [p for p in validate_config_detailed(cfg, target) if p.code == "datasets.resolution-parallel"]
    assert len(hits) == 1
    message = hits[0].message
    assert "×" in message or "份" in message
    # [review #15] 文案改成「曝光次数 / steps ×N」，不再说「耗时同倍增加」
    assert "步数" in message
    assert "2" in message  # 把 N 这个实际档位数写进去
    assert "耗时" not in message
    # 默认三档也要报，不许只在 >2 档时才报
    default_hits = [
        p
        for p in validate_config_detailed(cfgs[target], target)
        if p.code == "datasets.resolution-parallel"
    ]
    assert len(default_hits) == 1


def test_caption_dropout_high_is_only_a_warning(cfgs: dict[TargetModel, dict]) -> None:
    target = TargetModel.SDXL
    cfg = copy.deepcopy(cfgs[target])
    _proc(cfg)["datasets"][0]["caption_dropout_rate"] = 0.4

    assert "datasets.caption-dropout-high" in _warn_codes(cfg, target)
    assert validate_config(cfg, target) == []


def test_sdxl_quantize_is_only_a_warning(cfgs: dict[TargetModel, dict]) -> None:
    """config_modules.py:697-698：量化只对 transformer 生效，SDXL 写了等于没写。"""
    target = TargetModel.SDXL
    cfg = copy.deepcopy(cfgs[target])
    _proc(cfg)["model"]["quantize"] = True
    _proc(cfg)["model"]["qtype"] = "qfloat8"

    assert "model.quantize-on-sdxl" in _warn_codes(cfg, target)
    assert validate_config(cfg, target) == []


def test_validate_reports_at_least_five_distinct_errors(cfgs: dict[TargetModel, dict]) -> None:
    """任务要求：至少能抓 5 种不同的错误配置。"""
    cases = [
        # (target, 变异函数, 期望关键词)
        (
            TargetModel.KREA2,
            lambda c: _proc(c)["train"].update(timestep_type="bogus"),
            "bogus",
        ),
        (
            TargetModel.MOSS,
            lambda c: _proc(c)["train"].update(timestep_type="bogus"),
            "bogus",
        ),
        (TargetModel.SDXL, lambda c: _proc(c)["train"].update(noise_scheduler="flowmatch"), "flowmatch"),
        (
            TargetModel.SDXL,
            lambda c: _proc(c)["network"].update(linear=None),
            "rank",
        ),
        (
            TargetModel.KREA2,
            lambda c: _proc(c)["model"].update(name_or_path=""),
            "name_or_path",
        ),
        (TargetModel.MOSS, lambda c: _proc(c)["sample"].update(width=-64), "width"),
    ]
    caught = 0
    for target, mutate, keyword in cases:
        cfg = copy.deepcopy(cfgs[target])
        mutate(cfg)
        problems = validate_config(cfg, target)
        assert any(keyword in p for p in problems), (target, problems)
        caught += 1
    assert caught >= 5


def test_moss_timestep_type_is_validated(cfgs: dict[TargetModel, dict]) -> None:
    """[review #14] 原实现只查 KREA2，MOSS 的 timestep_type 完全没校验。

    MOSS 取 ``linear``：shift 分支等效 shift×2（σ 中位 0.9619 vs 0.5005），
    推理吸附的 training_shift=1.0 网格就是 linear 网格（Issue #23 复核）。
    """
    target = TargetModel.MOSS
    assert _proc(cfgs[target])["train"]["timestep_type"] == "linear"

    cfg = copy.deepcopy(cfgs[target])
    _proc(cfg)["train"]["timestep_type"] = "nonsense"
    hits = [p for p in validate_config_detailed(cfg, target) if p.code == "train.timestep-type-unknown"]
    assert hits
    assert hits[0].severity == "error"
    assert "nonsense" in hits[0].message


def test_timestep_whitelist_accepts_shift_family(cfgs: dict[TargetModel, dict]) -> None:
    """[review #2] shift / flux_shift / lumina2_shift 是源码明确处理的合法分支。"""
    for value in ("shift", "flux_shift", "lumina2_shift"):
        cfg = copy.deepcopy(cfgs[TargetModel.KREA2])
        _proc(cfg)["train"]["timestep_type"] = value
        codes = _codes(validate_config_detailed(cfg, TargetModel.KREA2))
        assert "train.timestep-type-unknown" not in codes, value


def test_sdxl_arch_may_be_implied_by_is_xl(cfgs: dict[TargetModel, dict]) -> None:
    """[review #6] config_modules.py:809-831 允许由 is_xl 推出 sdxl。"""
    target = TargetModel.SDXL
    cfg = copy.deepcopy(cfgs[target])
    proc = _proc(cfg)
    proc["model"].pop("arch")
    assert "model.arch" in _codes(validate_config_detailed(cfg, target))

    proc["model"]["is_xl"] = True
    assert "model.arch" not in _codes(validate_config_detailed(cfg, target))


def test_datasets_accept_scalar_resolution_and_dataset_path_alias(
    cfgs: dict[TargetModel, dict],
) -> None:
    """[review #9/#10] resolution 允许单个 int；folder_path 允许 dataset_path 别名。"""
    target = TargetModel.SDXL
    cfg = copy.deepcopy(cfgs[target])
    ds = _proc(cfg)["datasets"][0]
    ds["resolution"] = 1024
    ds.pop("folder_path")
    ds["dataset_path"] = "./datasets/my_character"

    codes = _codes(validate_config_detailed(cfg, target))
    assert "datasets.resolution" not in codes
    assert "datasets.folder-path" not in codes
    assert "datasets.resolution-parallel" not in codes


def test_sample_size_keys_may_be_omitted(cfgs: dict[TargetModel, dict]) -> None:
    """[review #8] width/height/sample_steps 都有默认值，省略不该报错。"""
    target = TargetModel.SDXL
    cfg = copy.deepcopy(cfgs[target])
    sample = _proc(cfg)["sample"]
    for key in ("width", "height", "sample_steps"):
        sample.pop(key)

    codes = _codes(validate_config_detailed(cfg, target))
    assert "sample.size" not in codes
    assert "sample.size-alignment" not in codes


def test_flow_sample_size_must_be_divisible_by_16(cfgs: dict[TargetModel, dict]) -> None:
    """[review #19] krea2/moss 的 bucket 整除量是 16，不是 8。"""
    for target in (TargetModel.KREA2, TargetModel.MOSS):
        cfg = copy.deepcopy(cfgs[target])
        _proc(cfg)["sample"]["width"] = 1032  # 8 的倍数、不是 16 的倍数
        hits = [
            p
            for p in validate_config_detailed(cfg, target)
            if p.code == "sample.size-alignment"
        ]
        assert hits, target
        assert "16" in hits[0].message

    sdxl_cfg = copy.deepcopy(cfgs[TargetModel.SDXL])
    _proc(sdxl_cfg)["sample"]["width"] = 1032
    assert "sample.size-alignment" not in _codes(
        validate_config_detailed(sdxl_cfg, TargetModel.SDXL)
    )


def test_qtype_is_only_checked_when_quantization_is_on(
    cfgs: dict[TargetModel, dict],
) -> None:
    """[review #17] quantize 关着的时候查 qtype 家族是噪音。"""
    target = TargetModel.SDXL  # SDXL 默认不开量化
    cfg = copy.deepcopy(cfgs[target])
    _proc(cfg)["model"]["qtype"] = "w8a8-unknown"
    assert "model.qtype-unknown" not in _codes(validate_config_detailed(cfg, target))

    _proc(cfg)["model"]["quantize"] = True
    assert "model.qtype-unknown" in _codes(validate_config_detailed(cfg, target))


def test_lr_boundary_and_required_empty_semantics(cfgs: dict[TargetModel, dict]) -> None:
    """[review #7/#18] 空值只拦显式写法；lr 阈值是 5e-3。"""
    target = TargetModel.SDXL
    omitted = copy.deepcopy(cfgs[target])
    _proc(omitted)["train"].pop("batch_size")
    assert "train.required-empty" not in _codes(validate_config_detailed(omitted, target))

    explicit = copy.deepcopy(cfgs[target])
    _proc(explicit)["train"]["batch_size"] = None
    hits = [
        p
        for p in validate_config_detailed(explicit, target)
        if p.code == "train.required-empty"
    ]
    assert hits
    assert hits[0].severity == "warn"

    high = copy.deepcopy(cfgs[target])
    _proc(high)["train"]["lr"] = 8e-3
    assert "train.lr-too-large" in _error_codes(high, target)


def test_validate_accepts_raw_dict_with_missing_sections() -> None:
    """脏输入不能抛异常，只能返回问题列表。"""
    assert validate_config({}, TargetModel.SDXL)
    assert validate_config({"job": "extension", "config": {}}, TargetModel.SDXL)
    assert validate_config({"job": "extension", "config": {"process": [{}]}}, TargetModel.MOSS)
    assert validate_config({"job": "extension"}, "not-a-model")


# ---------------------------------------------------------------------------
# 分级层：error 拦导出，warn 只提示
# ---------------------------------------------------------------------------


def test_error_and_warn_coexist_but_validate_keeps_only_errors(
    cfgs: dict[TargetModel, dict],
) -> None:
    """同一份配置既有硬错又有告警时，validate_config 只吐硬错。"""
    target = TargetModel.SDXL
    cfg = copy.deepcopy(cfgs[target])
    _proc(cfg)["train"]["noise_scheduler"] = "flowmatch"  # error
    _proc(cfg)["datasets"][0]["resolution"] = [1000]  # warn

    detailed = validate_config_detailed(cfg, target)
    codes = _codes(detailed)
    assert "train.noise-scheduler-flowmatch-on-sdxl" in codes
    assert "datasets.resolution-alignment" in codes

    hard = validate_config(cfg, target)
    assert hard
    # 只能剩 error 的 message，warn 不许漏进去
    warn_messages = {p.message for p in detailed if p.severity == "warn"}
    assert not (set(hard) & warn_messages)
    assert {p.message for p in detailed if p.severity == "error"} == set(hard)


def test_known_codes_covers_every_emitted_code(cfgs: dict[TargetModel, dict]) -> None:
    """发出的每个 code 都必须在登记表里（否则 _problem 会 KeyError）。"""
    emitted: set[str] = set()
    for target in TARGETS:
        emitted |= _codes(validate_config_detailed(cfgs[target], target))

    mutations = [
        (TargetModel.SDXL, lambda c: c.pop("job")),
        (TargetModel.SDXL, lambda c: _proc(c)["train"].update(noise_scheduler="flowmatch")),
        (TargetModel.SDXL, lambda c: _proc(c)["model"].update(quantize=True)),
        (TargetModel.KREA2, lambda c: _proc(c)["train"].update(timestep_type="sigmoid")),
        (TargetModel.KREA2, lambda c: _proc(c)["sample"].update(sampler="ddpm")),
        (TargetModel.MOSS, lambda c: _proc(c)["model"]["model_kwargs"].update(uncond_skip_layers=[20])),
        (TargetModel.MOSS, lambda c: _proc(c)["train"].update(train_text_encoder=True)),
        (TargetModel.SDXL, lambda c: _proc(c)["datasets"][0].update(caption_dropout_rate=0.5)),
        (TargetModel.MOSS, lambda c: _proc(c)["model"].update(qtype="w8a8-unknown")),
        (TargetModel.MOSS, lambda c: _proc(c)["model"].update(vae_dtype="float16")),
        (TargetModel.SDXL, lambda c: _proc(c)["datasets"][0].update(resolution=[1000])),
    ]
    for target, mutate in mutations:
        cfg = copy.deepcopy(cfgs[target])
        mutate(cfg)
        emitted |= _codes(validate_config_detailed(cfg, target))

    registry = known_codes()
    assert emitted <= set(registry)
    assert {registry[c] for c in emitted} <= {"error", "warn"}
    # 两侧都得有，否则分级没意义
    assert "error" in {registry[c] for c in emitted}
    assert "warn" in {registry[c] for c in emitted}


def test_problem_codes_are_stable_across_calls(cfgs: dict[TargetModel, dict]) -> None:
    target = TargetModel.KREA2
    cfg = copy.deepcopy(cfgs[target])
    _proc(cfg)["train"].update(timestep_type="sigmoid")
    _proc(cfg)["sample"].update(sampler="ddpm")

    first = validate_config_detailed(cfg, target)
    second = validate_config_detailed(copy.deepcopy(cfg), target)
    assert [p.code for p in first] == [p.code for p in second]
    assert [p.severity for p in first] == [p.severity for p in second]
    assert all(isinstance(p, ConfigProblem) for p in first)


# ---------------------------------------------------------------------------
# yaml round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", TARGETS)
def test_yaml_roundtrip_is_stable(
    cfgs: dict[TargetModel, dict], scratch: Path, target: TargetModel
) -> None:
    cfg = cfgs[target]
    path = scratch / f"{target.value}.yaml"

    first = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    path.write_text(first, encoding="utf-8")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    second = yaml.safe_dump(loaded, sort_keys=False, allow_unicode=True)

    assert loaded == cfg
    assert first == second


#: 示例 yaml 使用的路径覆盖（与 .scratch/work/gen_examples.py 保持一致）。
EXAMPLE_PLAN = {
    "dataset_dir": "./datasets/my_character",
    "output_dir": "./output",
    "trigger": "mychar",
    "steps": 2500,
    "sample_prompts": [
        "[trigger], portrait photo, plain background, soft lighting",
        "[trigger] walking through a rainy city street at night, neon signs, full body",
        "portrait photo, plain background, soft lighting",
    ],
}


def test_examples_are_built_by_the_builder() -> None:
    """configs/examples/ 里的三份 yaml 必须和 build_config 的输出一致。"""
    examples = Path(__file__).resolve().parent.parent / "configs" / "examples"
    for target in TARGETS:
        path = examples / f"{target.value}.yaml"
        assert path.exists(), f"缺少示例 {path}"
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        expected = build_config(target, plan_for_tier(target, "24", **EXAMPLE_PLAN))
        assert loaded == expected, path
        # 示例本身也要过校验，不能出现「示例自己都过不了」
        assert validate_config(loaded, target) == [], path


# ---------------------------------------------------------------------------
# 显存档位与验收数据
# ---------------------------------------------------------------------------


def test_vram_tiers_cover_required_keys() -> None:
    assert set(VRAM_TIERS) == {"8", "12", "16", "24", "48"}
    for key, tier in VRAM_TIERS.items():
        assert tier.key == key
        assert tier.label
        assert tier.note
        assert tier.batch_size >= 1
        assert tier.max_resolution > 0
        assert tier.max_resolution % 16 == 0
        assert tier.quantize is None or isinstance(tier.quantize, str)


def test_tier_for_vram_picks_largest_fitting() -> None:
    assert tier_for_vram(24).key == "24"
    assert tier_for_vram(40).key == "24"
    assert tier_for_vram(12).key == "12"
    assert tier_for_vram(6).key == "8"


def test_default_plan_is_runnable_and_24gb() -> None:
    for target in TARGETS:
        plan = default_plan(target)
        assert plan.vram_gb == 24
        assert plan.model_path != "/path/to/model"  # 占位路径 ai-toolkit 读不了
        assert plan.resolutions and all(r % 16 == 0 for r in plan.resolutions)


def test_small_vram_tier_caps_resolution() -> None:
    plan = plan_for_tier(TargetModel.MOSS, "8")
    cfg = build_config(TargetModel.MOSS, plan)
    resolution = _proc(cfg)["datasets"][0]["resolution"]
    assert max(resolution) <= VRAM_TIERS["8"].max_resolution
    assert _proc(cfg)["sample"]["width"] <= VRAM_TIERS["8"].max_resolution


def test_plan_for_tier_rejects_unknown_tier() -> None:
    with pytest.raises(ValueError):
        plan_for_tier(TargetModel.SDXL, "6")


def test_acceptance_data_is_structured() -> None:
    payload = acceptance_dict(TargetModel.KREA2)
    assert payload["target"] == "krea2"
    assert payload["verify_model"] == "krea/Krea-2-Turbo"
    assert payload["sampling"]["sampler"] == "flowmatch"
    assert payload["sampling"]["steps"] == 8
    assert payload["sampling"]["mu"] == 0.90625
    assert {p["key"] for p in payload["prompts"]} == {
        "in_distribution",
        "generalization",
        "no_trigger",
    }
    assert payload["checklist"]

    moss = acceptance_dict(TargetModel.MOSS)
    assert moss["verify_model"] == "OpenMOSS-Team/MOSS-Image-SFT"
    assert moss["sampling"]["steps"] == 50
    assert moss["sampling"]["shift"] == 5.0

    sdxl = acceptance_dict(TargetModel.SDXL)
    assert sdxl["sampling"]["sampler"] == "ddpm"
    assert sdxl["sampling"]["mu"] is None
    assert sdxl["sampling"]["shift"] is None

    # 结构化数据必须能 json 化（界面走 HTTP）
    json.dumps(payload, ensure_ascii=False)


def test_krea2_mu_follows_the_scheduler_formula() -> None:
    assert krea2_mu((1024 // 16) ** 2) == 0.90625
    assert krea2_mu((512 // 16) ** 2) == 0.58125
