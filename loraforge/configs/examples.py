"""``configs/examples/`` 三份示例 yaml 的生成器。

示例是**生成物**，不是手写稿：交付前由 :func:`write_examples` 写出，测试里由
:func:`render_example` 逐字节比对。手改示例会被漂移测试抓住——否则示例和
``build_config`` 迟早分叉，而示例恰恰是用户第一眼看到的东西。

路径一律用相对目录（``./datasets/...``、``./output``），用户 clone 下来改两行就能跑。
"""

from __future__ import annotations

import pathlib
from typing import Final

from loraforge.configs.plans import plan_for_tier
from loraforge.configs.render import render_train_config
from loraforge.models import TargetModel

#: 示例文件名（不带扩展名）。与 ``loraforge.models.TargetModel`` 的取值对应。
BASE_NAME: Final[dict[TargetModel, str]] = {
    TargetModel.SDXL: "sdxl",
    TargetModel.KREA2: "krea2",
    TargetModel.MOSS: "moss_image",
}

#: 示例统一用 24 GB 档（用户主力档位），其余字段取 20 张图集的最小可用值。
EXAMPLE_PLAN: Final[dict[str, object]] = {
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

#: 每份示例顶部的说明块。数值出处写在注释里，改数值时一起改。
HEADERS: Final[dict[TargetModel, str]] = {
    TargetModel.SDXL: """\
# SDXL LoRA / ai-toolkit —— 20 张图集示例
# 训练底模：stabilityai/stable-diffusion-xl-base-1.0（legacy StableDiffusion 路径）
#   arch: sdxl            toolkit/config_modules.py:790-792；LEGACY_ARCHS 见 util/get_model.py:46-58
#   noise_scheduler ddpm   SDXL 不是 flow matching，配 flowmatch 会训坏（config_modules.py:383）
#   min_snr_gamma 不设     docs/models/sdxl.md:228（须 >1e-6 才生效）
#   vae_dtype float32      docs/models/sdxl.md:233（fp16 VAE 偶发 NaN）
#   不开 quantize          该开关只对 transformer 系生效（config_modules.py:697-698），小显存用 low_vram
#   bucket 必须能被 16 整除 toolkit/stable_diffusion_model.py:364-372
# 出图：同底模，sample.sampler 必须与 train.noise_scheduler 一致
#       采样 25 步 / CFG 6.0 来自 toolkit/models/registry.py:50
# 改这两行就能跑：datasets[0].folder_path、training_folder
""",
    TargetModel.KREA2: """\
# Krea 2 LoRA / ai-toolkit —— 20 张图集示例
# 训练底模：krea/Krea-2-Raw    出图：krea/Krea-2-Turbo（loraforge/models.py:54-55）
#   arch: krea2                  extensions_built_in/diffusion_models/krea2/krea2.py:144-145
#   noise_scheduler flowmatch    krea2.py:201-202
#   timestep_type linear         **绝不能写 sigmoid**，那是 FLUX 的口径，社区反复踩的坑
#   quantize qfloat8             toolkit/models/registry.py:26；qtype 默认 qfloat8（config_modules.py:695-703）
#   diff_output_preservation false  docs/models/krea2.md:222（先跑通再开；开了必须给 trigger_word 且不能开 TE）
#   train_text_encoder false     docs/models/krea2.md:166（Krea2 的 TE 是静默空操作）
#   bucket 能被 16 整除          krea2.py:204-206（VAE 8 × patch 2）
# 采样：Raw 上 52 步、要生效 4.5（docs/models/krea2.md:224）。
#       ai-toolkit 生效值 = yaml 值 − 1（krea2.py:455），所以写 5.5；UNVERIFIED，
#       若实拍证明不减 1 就改回 4.5（只动 _build_krea2 那一行）。
#       出图验收走 Turbo：8 步 / guidance 1.0（toolkit/models/registry.py:26）
""",
    TargetModel.MOSS: """\
# MOSS-Image LoRA / ai-toolkit —— 20 张图集示例
# 训练底模：OpenMOSS-Team/MOSS-Image-Base    出图：OpenMOSS-Team/MOSS-Image-SFT
#   arch: moss_image              extensions_built_in/diffusion_models/moss_image/moss_image.py:159-160
#     **不能写 moss_image:base** —— 冒号后会被截断（toolkit/config_modules.py:776-778）
#   model_kwargs.uncond_skip_layers: [22]  官方 SLG 推荐值，属推理侧口径（docs/models/moss_image.md:195）
#   model_kwargs.sample_shift: 5.0          src/pipeline.py:17（DEFAULT_SHIFT）
#   noise_scheduler flowmatch     moss_image.py:202-205
#   vae_dtype float32             docs/models/moss_image.md:264（官方 VAE fp32）
#   quantize convrot8             toolkit/models/registry.py:59
#   timestep_type linear           与厂商训练网格对齐：shift 分支等效 shift×2，
#                                  1024² σ 中位 0.9619 vs linear 0.5005；
#                                  推理吸附的 training_shift=1.0 网格即 linear（Issue #23 复核）
#   bucket 能被 16 整除           moss_image.py:207-208
# 采样 50 步 / CFG 3.0 / shift 5.0（docs/models/moss_image.md:193）
""",
}


def example_path(directory: str | pathlib.Path, target: TargetModel) -> pathlib.Path:
    """某个 target 的示例文件路径。"""
    return pathlib.Path(directory) / f"{BASE_NAME[target]}.yaml"


def render_example(target: TargetModel) -> str:
    """渲染一份示例 yaml 的完整文本（含头部说明）。

    与 :func:`write_examples` 写出的字节完全一致，测试靠它做漂移比对。
    """
    plan = plan_for_tier(target, "24", **EXAMPLE_PLAN)
    # 示例的头部是手写的长说明，所以拿不带默认头部的本体；渲染口径统一在
    # render_train_config 里（纯 LF、\n 结尾、键序与 build_config 一致）。
    # portable=False：示例里的 "./datasets/my_character" 是给读者看的相对路径，
    # README 也是这么讲的；可移植命名是导出 zip 才需要的事（Issue #69）。
    body = render_train_config(target, plan, header=False, portable=False)
    return HEADERS[target] + body.decode("utf-8")


def write_examples(directory: str | pathlib.Path) -> list[pathlib.Path]:
    """把三份示例写到 ``directory``，返回写出的路径。"""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []
    for target in TargetModel:
        path = example_path(directory, target)
        # newline="\n"：gh.py 直传字节，不经 git，Windows 默认 CRLF 会原样进仓库
        path.write_text(render_example(target), encoding="utf-8", newline="\n")
        written.append(path)
    return written
