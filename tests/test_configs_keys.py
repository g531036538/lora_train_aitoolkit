"""键名对账回归：``build_config`` 发出的每个键，都必须能在 ai-toolkit 里找到读取点。

每行是「键路径 → 出处（文件:行号)」，出处来自对 ``ostris/ai-toolkit`` main（VERSION 0.13.43）
的逐条核对，对账报告见 Issue #23。改键名、删键、加键都要先改这张表并重新对账，否则
``test_emitted_keys_match_table`` 会红——这是故意的：拼错的键在 ai-toolkit 里是静默忽略，
不报错、照跑、结果不对。

三档分类：

* ``KEY_SOURCES``：真有读取点的键；
* ``FREE_FORM``：``meta`` 段，ai-toolkit 整段收下但不解析内容（jobs/BaseJob.py:19-20）；
* ``NOOP_KEYS``：发了但 sd_trainer 路径不读的键，保留只为贴近上游示例。
"""
from __future__ import annotations

import re

import pytest

from loraforge.configs import build_config, default_plan
from loraforge.models import TargetModel

_LOCATION = re.compile(r"^[A-Za-z0-9_./\-]+\.py:\d+$")

# ---------------------------------------------------------------------------
# 公共键（三模型都发；SDXL 的键集合就是这张表的全部）
# ---------------------------------------------------------------------------
COMMON_KEYS = frozenset({
    'config',
    'config.name',
    'config.process',
    'config.process[].datasets',
    'config.process[].datasets[].buckets',
    'config.process[].datasets[].cache_latents_to_disk',
    'config.process[].datasets[].caption_dropout_rate',
    'config.process[].datasets[].caption_ext',
    'config.process[].datasets[].folder_path',
    'config.process[].datasets[].resolution',
    'config.process[].datasets[].shuffle_tokens',
    'config.process[].device',
    'config.process[].logging',
    'config.process[].logging.log_every',
    'config.process[].logging.use_wandb',
    'config.process[].logging.verbose',
    'config.process[].model',
    'config.process[].model.arch',
    'config.process[].model.dtype',
    'config.process[].model.name_or_path',
    'config.process[].model.vae_dtype',
    'config.process[].network',
    'config.process[].network.linear',
    'config.process[].network.linear_alpha',
    'config.process[].network.type',
    'config.process[].sample',
    'config.process[].sample.guidance_scale',
    'config.process[].sample.height',
    'config.process[].sample.neg',
    'config.process[].sample.prompts',
    'config.process[].sample.sample_every',
    'config.process[].sample.sample_start_step',
    'config.process[].sample.sample_steps',
    'config.process[].sample.sampler',
    'config.process[].sample.seed',
    'config.process[].sample.walk_seed',
    'config.process[].sample.width',
    'config.process[].save',
    'config.process[].save.dtype',
    'config.process[].save.max_step_saves_to_keep',
    'config.process[].save.push_to_hub',
    'config.process[].save.save_every',
    'config.process[].train',
    'config.process[].train.batch_size',
    'config.process[].train.dtype',
    'config.process[].train.ema_config',
    'config.process[].train.ema_config.ema_decay',
    'config.process[].train.ema_config.use_ema',
    'config.process[].train.gradient_accumulation_steps',
    'config.process[].train.gradient_checkpointing',
    'config.process[].train.lr',
    'config.process[].train.noise_scheduler',
    'config.process[].train.optimizer',
    'config.process[].train.steps',
    'config.process[].train.train_text_encoder',
    'config.process[].train.train_unet',
    'config.process[].training_folder',
    'config.process[].trigger_word',
    'config.process[].type',
    'job',
    'meta',
    'meta.name',
    'meta.version',
})

# 各模型在公共键之外额外发的键
KREA2_EXTRA_KEYS = frozenset({
    'config.process[].model.qtype',
    'config.process[].model.qtype_te',
    'config.process[].model.quantize',
    'config.process[].model.quantize_te',
    'config.process[].train.diff_output_preservation',
    'config.process[].train.timestep_type',
})

MOSS_EXTRA_KEYS = frozenset({
    'config.process[].model.model_kwargs',
    'config.process[].model.model_kwargs.sample_shift',
    'config.process[].model.model_kwargs.uncond_skip_layers',
    'config.process[].model.qtype',
    'config.process[].model.qtype_te',
    'config.process[].model.quantize',
    'config.process[].model.quantize_te',
    'config.process[].train.timestep_type',
})

EXPECTED_KEYS = {
    TargetModel.SDXL: COMMON_KEYS,
    TargetModel.KREA2: COMMON_KEYS | KREA2_EXTRA_KEYS,
    TargetModel.MOSS: COMMON_KEYS | MOSS_EXTRA_KEYS,
}

NOOP_KEYS = {
    "config.process[].device": (
        "jobs/ExtensionJob.py:11 只读 config 级 device；sd_trainer 用 get_accelerator()"
        "（jobs/process/BaseSDTrainProcess.py:85,105-106），process 级 device 无人读。"
        "上游示例 config/examples/train_lora_flux_24gb.yaml:12 也照写，保留但视为 no-op。"
    ),
}

FREE_FORM = {
    "meta": "jobs/BaseJob.py:19-20：整段收下，不解析内容",
    "meta.name": "自由字段；[name] 由 toolkit/config.py:45-49 全局替换",
    "meta.version": "自由字段，内容不参与训练",
}

# 键 → 出处（文件:行号）。文件路径相对 ai-toolkit 仓库根。
KEY_SOURCES = {
    'config': 'jobs/BaseJob.py:15',
    'config.name': 'jobs/BaseJob.py:18',
    'config.process': 'jobs/BaseJob.py:54',
    'config.process[].datasets': 'jobs/process/BaseSDTrainProcess.py:144',
    'config.process[].datasets[].buckets': 'toolkit/config_modules.py:961',
    'config.process[].datasets[].cache_latents_to_disk': 'toolkit/config_modules.py:1018',
    'config.process[].datasets[].caption_dropout_rate': 'toolkit/config_modules.py:968',
    'config.process[].datasets[].caption_ext': 'toolkit/config_modules.py:953',
    'config.process[].datasets[].folder_path': 'toolkit/config_modules.py:933',
    'config.process[].datasets[].resolution': 'toolkit/config_modules.py:959',
    'config.process[].datasets[].shuffle_tokens': 'toolkit/config_modules.py:967',
    'config.process[].logging': 'jobs/process/BaseSDTrainProcess.py:129',
    'config.process[].logging.log_every': 'toolkit/config_modules.py:37',
    'config.process[].logging.use_wandb': 'toolkit/config_modules.py:39',
    'config.process[].logging.verbose': 'toolkit/config_modules.py:38',
    'config.process[].model': 'jobs/process/BaseSDTrainProcess.py:113',
    'config.process[].model.arch': 'toolkit/config_modules.py:716',
    'config.process[].model.dtype': 'toolkit/config_modules.py:655',
    'config.process[].model.model_kwargs': 'toolkit/config_modules.py:767',
    'config.process[].model.model_kwargs.sample_shift': 'extensions_built_in/diffusion_models/moss_image/moss_image.py:184',
    'config.process[].model.model_kwargs.uncond_skip_layers': 'extensions_built_in/diffusion_models/moss_image/moss_image.py:186',
    'config.process[].model.name_or_path': 'toolkit/config_modules.py:638',
    'config.process[].model.qtype': 'toolkit/config_modules.py:697',
    'config.process[].model.qtype_te': 'toolkit/config_modules.py:698',
    'config.process[].model.quantize': 'toolkit/config_modules.py:695',
    'config.process[].model.quantize_te': 'toolkit/config_modules.py:696',
    'config.process[].model.vae_dtype': 'toolkit/config_modules.py:690',
    'config.process[].network': 'jobs/process/BaseSDTrainProcess.py:107',
    'config.process[].network.linear': 'toolkit/config_modules.py:179',
    'config.process[].network.linear_alpha': 'toolkit/config_modules.py:191',
    'config.process[].network.type': 'toolkit/config_modules.py:177',
    'config.process[].sample': 'jobs/process/BaseSDTrainProcess.py:121',
    'config.process[].sample.guidance_scale': 'toolkit/config_modules.py:92',
    'config.process[].sample.height': 'toolkit/config_modules.py:88',
    'config.process[].sample.neg': 'toolkit/config_modules.py:89',
    'config.process[].sample.prompts': 'toolkit/config_modules.py:110',
    'config.process[].sample.sample_every': 'toolkit/config_modules.py:85',
    'config.process[].sample.sample_start_step': 'toolkit/config_modules.py:86',
    'config.process[].sample.sample_steps': 'toolkit/config_modules.py:93',
    'config.process[].sample.sampler': 'toolkit/config_modules.py:84',
    'config.process[].sample.seed': 'toolkit/config_modules.py:90',
    'config.process[].sample.walk_seed': 'toolkit/config_modules.py:91',
    'config.process[].sample.width': 'toolkit/config_modules.py:87',
    'config.process[].save': 'jobs/process/BaseSDTrainProcess.py:120',
    'config.process[].save.dtype': 'toolkit/config_modules.py:26',
    'config.process[].save.max_step_saves_to_keep': 'toolkit/config_modules.py:27',
    'config.process[].save.push_to_hub': 'toolkit/config_modules.py:31',
    'config.process[].save.save_every': 'toolkit/config_modules.py:25',
    'config.process[].train': 'jobs/process/BaseSDTrainProcess.py:112',
    'config.process[].train.batch_size': 'toolkit/config_modules.py:399',
    'config.process[].train.diff_output_preservation': 'toolkit/config_modules.py:495',
    'config.process[].train.dtype': 'toolkit/config_modules.py:401',
    'config.process[].train.ema_config': 'toolkit/config_modules.py:547',
    'config.process[].train.ema_config.ema_decay': 'toolkit/config_modules.py:838',
    'config.process[].train.ema_config.use_ema': 'toolkit/config_modules.py:837',
    'config.process[].train.gradient_accumulation_steps': 'toolkit/config_modules.py:461',
    'config.process[].train.gradient_checkpointing': 'toolkit/config_modules.py:419',
    'config.process[].train.lr': 'toolkit/config_modules.py:387',
    'config.process[].train.noise_scheduler': 'toolkit/config_modules.py:383',
    'config.process[].train.optimizer': 'toolkit/config_modules.py:393',
    'config.process[].train.steps': 'toolkit/config_modules.py:386',
    'config.process[].train.timestep_type': 'toolkit/config_modules.py:560',
    'config.process[].train.train_text_encoder': 'toolkit/config_modules.py:407',
    'config.process[].train.train_unet': 'toolkit/config_modules.py:406',
    'config.process[].training_folder': 'jobs/process/BaseTrainProcess.py:43',
    'config.process[].trigger_word': 'jobs/process/BaseSDTrainProcess.py:135',
    'config.process[].type': 'jobs/BaseJob.py:58',
    'job': 'toolkit/job.py:9',
}


def _key_paths(node, prefix=""):
    """把一个配置 dict 摊平成 ``a.b[]`` 形式的键路径集合。"""
    out = set()
    if isinstance(node, dict):
        for key, value in node.items():
            path = "{}.{}".format(prefix, key) if prefix else str(key)
            out.add(path)
            out |= _key_paths(value, path)
    elif isinstance(node, list):
        for item in node:
            out |= _key_paths(item, prefix + "[]")
    return out


def _proc(cfg):
    return cfg["config"]["process"][0]


def emitted_keys(target):
    return _key_paths(build_config(target, default_plan(target)))


@pytest.mark.parametrize("target", list(TargetModel))
def test_emitted_keys_match_table(target):
    emitted = emitted_keys(target)
    expected = set(EXPECTED_KEYS[target])
    missing = sorted(expected - emitted)
    unexpected = sorted(emitted - expected)
    assert not missing, "{} 少了这些已登记键: {}".format(target.value, missing)
    assert not unexpected, (
        "{} 多出未登记键: {}；新增键请先在 KEY_SOURCES 里写清 ai-toolkit 的读取点".format(
            target.value, unexpected
        )
    )


def test_every_pinned_key_has_a_location():
    bad = {k: v for k, v in KEY_SOURCES.items() if not _LOCATION.match(v)}
    assert not bad, "这些键的出处不是「文件:行号」: {}".format(bad)


def test_key_table_covers_every_emitted_key():
    union = set().union(*(EXPECTED_KEYS[t] for t in TargetModel))
    assert union == set(KEY_SOURCES) | set(NOOP_KEYS) | set(FREE_FORM)


def test_noop_keys_are_not_pinned_as_read():
    for key, why in NOOP_KEYS.items():
        assert key not in KEY_SOURCES, "{} 同时被登记成必读和 no-op".format(key)
        assert "accelerator" in why or "无人读" in why
        assert key in emitted_keys(TargetModel.SDXL)


def test_meta_is_top_level_not_inside_config():
    # jobs/BaseJob.py:19-20 读的是文件顶层的 config['meta']，不是 config.config.meta
    for target in TargetModel:
        cfg = build_config(target, default_plan(target))
        assert "meta" in cfg, "{} 缺顶层 meta".format(target.value)
        assert "meta" not in cfg["config"], "{} 把 meta 放进了 config 段".format(target.value)


def test_model_dtype_equals_train_dtype():
    # jobs/process/BaseSDTrainProcess.py:117 会用 train.dtype 覆盖 model.dtype，
    # 两者写不一致只会让 yaml 自相矛盾；这里钉住"写一样"。
    for target in TargetModel:
        proc = _proc(build_config(target, default_plan(target)))
        assert proc["model"]["dtype"] == proc["train"]["dtype"], target.value


def test_flow_models_sampler_matches_noise_scheduler():
    # sample.sampler 会被 SampleConfig 读出（config_modules.py:84），但 krea2/moss 的
    # generate_single_image（krea2.py:411-467、moss_image.py:395-411）不消费它；
    # 保持与 noise_scheduler 一致，避免用户以为它在这两个模型上有语义。
    for target in (TargetModel.KREA2, TargetModel.MOSS):
        proc = _proc(build_config(target, default_plan(target)))
        assert proc["train"]["noise_scheduler"] == "flowmatch"
        assert proc["sample"]["sampler"] == "flowmatch"


def test_dataset_keys_are_pinned():
    dataset_keys = {k for k in KEY_SOURCES if "datasets[]." in k}
    assert dataset_keys == {
        "config.process[].datasets[].buckets",
        "config.process[].datasets[].cache_latents_to_disk",
        "config.process[].datasets[].caption_dropout_rate",
        "config.process[].datasets[].caption_ext",
        "config.process[].datasets[].folder_path",
        "config.process[].datasets[].resolution",
        "config.process[].datasets[].shuffle_tokens",
    }
