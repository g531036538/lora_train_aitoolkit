"""把一份训练配置渲染成**可以落盘、可以直接进 zip** 的字节。

这是「生成 → 落盘 → 打包」这条链上唯一的渲染点：``build_config`` 只给 dict，
到这里才决定字节长什么样（缩进、键序、换行、头部注释）。导出层
（``loraforge/export/core.py``）只原样拷贝文件，不该再有自己的渲染口径——
两处渲染迟早分叉，而分叉的表现是「用户拿到的 yaml 和示例不一样」。

约定（``docs/ARCHITECTURE.md`` §4 的导出契约）：

* 纯 LF，不出现 ``\\r\\n``（``gh.py`` 直传字节，Windows 的默认 CRLF 会原样进仓库）；
* 以 ``\\n`` 结尾（最后一行不是半行）；
* ``sort_keys=False``：字段顺序就是 ``build_config`` 的书写顺序，读起来像文档；
* ``allow_unicode=True``：注释与触发词里可能有中文。
"""

from __future__ import annotations

from typing import Final

import yaml

from loraforge import __version__
from loraforge.configs.base import TrainingPlan, build_config
from loraforge.configs.plans import tier_for_vram
from loraforge.models import PROFILES, TargetModel

#: 头部注释的行数与内容固定：用户在云上打开 yaml 第一眼要能确认「这份是谁生成的」。
_HEADER_TEMPLATE: Final[str] = (
    "# 由 LoRAForge {version} 生成（build_config）；要改参数请重跑生成，别手改——"
    "手改会在导出校验时报出来\n"
    "# 目标模型：{label}（{model_path}），显存档位：{tier}\n"
    "# 触发词：{trigger}\n"
)


def render_train_config(
    target: TargetModel | str,
    plan: TrainingPlan,
    *,
    header: bool = True,
) -> bytes:
    """渲染训练配置。

    ``header=True`` 时在 yaml 前加三行注释（生成器与版本 / 目标模型与档位 / 触发词），
    ``header=False`` 只返回 yaml 本体——示例生成器自己有一段更长的说明，用的是这一档。
    """
    model = TargetModel(target)
    body = yaml.safe_dump(
        build_config(model, plan), sort_keys=False, allow_unicode=True
    )
    if header:
        profile = PROFILES[model]
        body = _HEADER_TEMPLATE.format(
            version=__version__,
            label=profile.label,
            model_path=profile.train_base,
            tier=tier_for_vram(plan.vram_gb).label,
            trigger=plan.trigger,
        ) + body
    text = body.replace("\r\n", "\n")
    if not text.endswith("\n"):
        text += "\n"
    return text.encode("utf-8")


__all__ = ["render_train_config"]
