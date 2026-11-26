"""配置层测试：三模型的必填字段、危险组合、yaml round-trip。

夹具用 ``scratch``（仓库内可写目录），不要用 ``tmp_path``——本机系统临时目录
在沙箱外，见 ``tests/conftest.py``。
"""

from __future__ import annotations

import copy
import json
import shutil
import uuid
from pathlib import Path

import pytest
import yaml

from loraforge.configs import (
    VRAM_TIERS,
    acceptance_dict,
    build_config,
    default_plan,
    krea2_mu,
    plan_for_tier,
    tier_for_vram,
    validate_config,
)
from loraforge.models import TargetModel

TARGETS = [TargetModel.SDXL, TargetModel.KREA2, TargetModel.MOSS]


def _proc(cfg: dict) -> dict:
    return cfg["config"]["process"][0]


@pytest.fixture
def cfgs() -> dict[TargetModel, dict]:
    return {t: build_config(t, default_plan(t)) for t in TARGETS}


@pytest.fixture
def writable_scratch(scratch: Path) -> Path:
    """可写的工作目录，锚在 ``scratch`` 的父目录（仓库内 ``.scratch/``）。

    ``scratch`` 本身由 ``tempfile.mkdtemp(mode=0o700)`` 创建，本机 Windows 上
    这个 mode 会变成「只给 owner」的 ACL，当前进程连 ``chmod`` 都拿不到
    ``WRITE_DAC``，写文件直接 ``PermissionError``。所以不往它里面写，
    而在同一个 ``.scratch/`` 下用默认权限另建一个，用完删掉。
    """
    base: Path = scratch.parent
    path = base / f"w-{uuid.uuid4().hex[:8]}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


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
    assert proc["train"]["min_snr_gamma"] == 5.0
    assert proc["model"]["vae_dtype"] == "float16"
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
    assert proc["train"]["diff_output_preservation"] is True
    assert proc["model"]["quantize"] is True
    assert proc["model"]["qtype"] == "qfloat8"
    assert proc["sample"]["sampler"] == "flowmatch"


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
    assert proc["sample"]["sample_steps"] == 30
    assert proc["sample"]["guidance_scale"] == 3.0


# ---------------------------------------------------------------------------
# 校验：默认方案必须干净，错误配置必须被抓出来
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", TARGETS)
def test_default_configs_validate_clean(cfgs: dict[TargetModel, dict], target: TargetModel) -> None:
    assert validate_config(cfgs[target], target) == []


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


def test_validate_rejects_model_path_mismatch(cfgs: dict[TargetModel, dict]) -> None:
    target = TargetModel.KREA2
    cfg = copy.deepcopy(cfgs[target])
    _proc(cfg)["model"]["name_or_path"] = "black-forest-labs/FLUX.1-dev"

    problems = validate_config(cfg, target)
    assert problems
    assert any("不匹配" in p for p in problems)


def test_validate_rejects_krea2_sigmoid(cfgs: dict[TargetModel, dict]) -> None:
    target = TargetModel.KREA2
    cfg = copy.deepcopy(cfgs[target])
    _proc(cfg)["train"]["timestep_type"] = "sigmoid"

    problems = validate_config(cfg, target)
    assert problems
    assert any("sigmoid" in p for p in problems)


def test_validate_rejects_sdxl_flowmatch(cfgs: dict[TargetModel, dict]) -> None:
    target = TargetModel.SDXL
    cfg = copy.deepcopy(cfgs[target])
    _proc(cfg)["train"]["noise_scheduler"] = "flowmatch"

    problems = validate_config(cfg, target)
    assert problems
    assert any("flowmatch" in p for p in problems)


def test_validate_rejects_moss_without_uncond_skip_layers(cfgs: dict[TargetModel, dict]) -> None:
    target = TargetModel.MOSS
    cfg = copy.deepcopy(cfgs[target])
    _proc(cfg)["model"]["model_kwargs"].pop("uncond_skip_layers")

    problems = validate_config(cfg, target)
    assert problems
    assert any("uncond_skip_layers" in p for p in problems)


def test_validate_rejects_moss_arch_suffix(cfgs: dict[TargetModel, dict]) -> None:
    target = TargetModel.MOSS
    cfg = copy.deepcopy(cfgs[target])
    _proc(cfg)["model"]["arch"] = "moss_image:base"

    problems = validate_config(cfg, target)
    assert problems
    assert any("moss_image" in p for p in problems)


def test_validate_rejects_rank_alpha_mismatch(cfgs: dict[TargetModel, dict]) -> None:
    target = TargetModel.SDXL
    cfg = copy.deepcopy(cfgs[target])
    proc = _proc(cfg)
    proc["network"]["linear"] = 8
    proc["network"]["linear_alpha"] = 64

    problems = validate_config(cfg, target)
    assert problems
    assert any("alpha" in p for p in problems)


def test_validate_rejects_sampler_mismatch(cfgs: dict[TargetModel, dict]) -> None:
    target = TargetModel.KREA2
    cfg = copy.deepcopy(cfgs[target])
    _proc(cfg)["sample"]["sampler"] = "ddpm"

    problems = validate_config(cfg, target)
    assert problems
    assert any("sampler" in p for p in problems)


def test_validate_rejects_bad_resolution(cfgs: dict[TargetModel, dict]) -> None:
    target = TargetModel.SDXL
    cfg = copy.deepcopy(cfgs[target])
    _proc(cfg)["datasets"][0]["resolution"] = [1000]

    problems = validate_config(cfg, target)
    assert problems
    assert any("整除" in p for p in problems)


def test_validate_rejects_sdxl_quantize(cfgs: dict[TargetModel, dict]) -> None:
    target = TargetModel.SDXL
    cfg = copy.deepcopy(cfgs[target])
    _proc(cfg)["model"]["quantize"] = True
    _proc(cfg)["model"]["qtype"] = "qfloat8"

    problems = validate_config(cfg, target)
    assert problems
    assert any("量化" in p for p in problems)


def test_validate_reports_at_least_five_distinct_errors(cfgs: dict[TargetModel, dict]) -> None:
    """任务要求：至少能抓 5 种不同的错误配置。"""
    cases = [
        # (target, 变异函数, 期望关键词)
        (TargetModel.KREA2, lambda c: _proc(c)["train"].update(timestep_type="sigmoid"), "sigmoid"),
        (TargetModel.SDXL, lambda c: _proc(c)["train"].update(noise_scheduler="flowmatch"), "flowmatch"),
        (
            TargetModel.MOSS,
            lambda c: _proc(c)["model"]["model_kwargs"].pop("uncond_skip_layers"),
            "uncond_skip_layers",
        ),
        (
            TargetModel.SDXL,
            lambda c: _proc(c)["network"].update(linear=8, linear_alpha=64),
            "alpha",
        ),
        (
            TargetModel.KREA2,
            lambda c: _proc(c)["model"].update(name_or_path="stabilityai/stable-diffusion-xl-base-1.0"),
            "不匹配",
        ),
    ]
    caught = 0
    for target, mutate, keyword in cases:
        cfg = copy.deepcopy(cfgs[target])
        mutate(cfg)
        problems = validate_config(cfg, target)
        assert any(keyword in p for p in problems), (target, problems)
        caught += 1
    assert caught >= 5


def test_validate_accepts_raw_dict_with_missing_sections() -> None:
    """脏输入不能抛异常，只能返回问题列表。"""
    assert validate_config({}, TargetModel.SDXL)
    assert validate_config({"job": "extension", "config": {}}, TargetModel.SDXL)
    assert validate_config({"job": "extension", "config": {"process": [{}]}}, TargetModel.MOSS)
    assert validate_config({"job": "extension"}, "not-a-model")


# ---------------------------------------------------------------------------
# yaml round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", TARGETS)
def test_yaml_roundtrip_is_stable(
    cfgs: dict[TargetModel, dict], writable_scratch: Path, target: TargetModel
) -> None:
    cfg = cfgs[target]
    path = writable_scratch / f"{target.value}.yaml"

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
    assert moss["sampling"]["steps"] == 30
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
