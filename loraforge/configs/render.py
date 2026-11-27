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

#: 头部注释的行数与内容固定：用户在云上打开 yaml 第一眼要能确认「这份是谁生成的」，
#: 以及「数据集目录要不要改」。行数固定四行，第四行的文案随 ``portable`` 变——
#: 可移植档不能写「绝对路径」，非可移植档也不能写「相对当前目录」，两种都是假话。
_HEADER_TEMPLATE: Final[str] = (
    "# 由 LoRAForge {version} 生成（build_config）；要改参数请重跑生成，别手改——"
    "手改会在导出校验时报出来\n"
    "# 目标模型：{label}（{model_path}），显存档位：{tier}\n"
    "# 触发词：{trigger}\n"
    "# 数据集目录：{dataset}\n"
)

#: 可移植档下第四行的说明。导出 zip 是给云端机器训的，那边没有生成机的盘符。
_PORTABLE_NOTE: Final[str] = (
    "{name}（相对于训练机上的当前目录；解压后请把 folder_path 改成数据所在目录的绝对路径）"
)

#: 非可移植档下第四行的说明：明说只能本机训，别让人以为能直接搬走。
_ABSOLUTE_NOTE: Final[str] = "{path}（绝对路径；这份配置只能在本机直接训练）"


def _portable_folder_name(dataset_dir: str) -> str:
    """取数据集目录名，Windows 与 POSIX 两种写法都认。

    ``pathlib.Path`` 在 Linux 上不会把 ``C:\\x\\flat`` 切开，所以先统一分隔符
    再取最后一段——导出包可能在任何机器上生成、又要在另一台机器上解压使用。
    """
    normalized = dataset_dir.replace("\\", "/").rstrip("/")
    name = normalized.rsplit("/", 1)[-1]
    return name or "."


def render_train_config(
    target: TargetModel | str,
    plan: TrainingPlan,
    *,
    header: bool = True,
    portable: bool = True,
) -> bytes:
    """渲染训练配置。

    ``header=True`` 时在 yaml 前加四行注释（生成器与版本 / 目标模型与档位 / 触发词 /
    数据集目录说明），``header=False`` 只返回 yaml 本体——示例生成器自己有一段更长的
    说明，用的是这一档。

    ``portable=True``（默认）把 ``datasets[0].folder_path`` 写成数据集**目录名**：
    导出的 zip 是要传去云端 GPU 机器训练的，生成机的盘符在那边不存在（Issue #69）。
    ``portable=False`` 保留 ``plan.dataset_dir`` 原值，供本机直接训练。
    """
    model = TargetModel(target)
    cfg = build_config(model, plan)
    dataset_name = _portable_folder_name(plan.dataset_dir)
    if portable:
        datasets = cfg["config"]["process"][0].get("datasets") or []
        if datasets:
            datasets[0]["folder_path"] = dataset_name
    body = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    if header:
        profile = PROFILES[model]
        dataset_note = (
            _PORTABLE_NOTE.format(name=dataset_name)
            if portable
            else _ABSOLUTE_NOTE.format(path=plan.dataset_dir)
        )
        body = _HEADER_TEMPLATE.format(
            version=__version__,
            label=profile.label,
            model_path=profile.train_base,
            tier=tier_for_vram(plan.vram_gb).label,
            trigger=plan.trigger,
            dataset=dataset_note,
        ) + body
    text = body.replace("\r\n", "\n")
    if not text.endswith("\n"):
        text += "\n"
    return text.encode("utf-8")


__all__ = ["render_train_config"]
