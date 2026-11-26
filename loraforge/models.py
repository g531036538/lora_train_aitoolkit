"""目标模型与数据集类型的唯一枚举定义。跨模块引用这些，不要另行定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class TargetModel(str, Enum):
    """要训练的底模。"""

    SDXL = "sdxl"
    KREA2 = "krea2"
    MOSS = "moss_image"


class DatasetKind(str, Enum):
    """这批图要让 LoRA 学到什么。决定打标时「哪些该写、哪些不能写」。"""

    CHARACTER = "character"  # 角色/人脸：身份交给触发词，打标词只写这次的变化
    STYLE = "style"  # 风格：打标词写画面内容，不写风格词
    CONCEPT = "concept"  # 物件/概念：打标词写语境与关系
    TEXT_LAYOUT = "text_layout"  # 海报/包装/排版：图内文字必须精确


@dataclass(frozen=True)
class ModelProfile:
    """一个底模的静态事实。不包含任何运行期参数。

    这里只放「这个模型是什么」，参数默认值放 loraforge/configs/。
    """

    key: TargetModel
    label: str
    train_base: str  # ai-toolkit 训练用的 checkpoint
    verify_base: str  # 验收时用哪个出图
    caption_style: Literal["tags", "prose", "prose_precise"]
    notes: list[str] = field(default_factory=list)


PROFILES: dict[TargetModel, ModelProfile] = {
    TargetModel.SDXL: ModelProfile(
        key=TargetModel.SDXL,
        label="SDXL",
        train_base="stabilityai/stable-diffusion-xl-base-1.0",
        verify_base="stabilityai/stable-diffusion-xl-base-1.0",
        caption_style="tags",
        notes=["CLIP-L/G 文本编码器，对逗号分隔的短标签更熟", "触发词放最前"],
    ),
    TargetModel.KREA2: ModelProfile(
        key=TargetModel.KREA2,
        label="Krea 2",
        train_base="krea/Krea-2-Raw",
        verify_base="krea/Krea-2-Turbo",
        caption_style="prose",
        notes=[
            "12.9B 单流 DiT + Qwen3-VL-4B 文本编码器",
            "官方口径：TRAIN on Raw, RUN on Turbo",
            "timestep_type 必须 linear，禁止从 FLUX 搬 sigmoid",
        ],
    ),
    TargetModel.MOSS: ModelProfile(
        key=TargetModel.MOSS,
        label="MOSS-Image",
        train_base="OpenMOSS-Team/MOSS-Image-Base",
        verify_base="OpenMOSS-Team/MOSS-Image-SFT",
        caption_style="prose_precise",
        notes=[
            "OpenMOSS 开源，Apache-2.0，Base 用于训练 LoRA、SFT 用于出图",
            "图内文字用引号逐行给出，位置要求可用 JSON layout",
            "采样 30 步 / CFG 3.0 / shift 5；Base 需 uncond_skip_layers=[22]",
        ],
    ),
}


def profile(target: TargetModel) -> ModelProfile:
    return PROFILES[TargetModel(target)]
