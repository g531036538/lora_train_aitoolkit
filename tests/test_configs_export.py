"""导出 zip 里那份 ``train.yaml`` 是不是真能拿去训练。见 Issue #54。

三条断言，按「证据强度」从弱到强：

1. 逐字节：zip 里取出来的 ``train.yaml`` 与写进去的字节完全一致（顺带守住 CRLF
   不被改写——``gh.py`` 直传字节，这类事故真发生过）。
2. 语义 + 校验：解出来的 yaml 与 ``build_config`` 的 dict 相等，且
   ``validate_config`` 返回空列表。
3. 能被 ai-toolkit 吃：走它自己的 ``toolkit.config.get_config`` 与
   ``sd_trainer`` 扩展分派，断言 ``model.name_or_path`` / ``model.arch`` /
   ``network.type`` 与登记值一致。依赖缺失时 skip 并打原因，不让没装依赖的机器红。

这里不碰 ``loraforge/export/core.py``（小界的写集）：``package()`` 只负责原样
拷贝调用方给的配置文件，测试负责把「调用方给的」定义成 ``build_config`` 的产出。
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys
import zipfile

import pytest
import yaml

from loraforge.configs import (
    build_config,
    default_plan,
    render_train_config,
    validate_config,
    validate_config_detailed,
)
from loraforge.export.core import package
from loraforge.models import TargetModel

#: zip 内部的图集目录名，与 ``package()`` 用 ``scan.root.resolve().name`` 拼出来的一致。
WORK = "my_character"

#: ai-toolkit 读出来的五个字段，与 tests/test_configs_loadable.py 的登记同口径。
EXPECTED = {
    TargetModel.SDXL: ("stabilityai/stable-diffusion-xl-base-1.0", "sdxl"),
    TargetModel.KREA2: ("krea/Krea-2-Raw", "krea2"),
    TargetModel.MOSS: ("OpenMOSS-Team/MOSS-Image-Base", "moss_image"),
}

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = pathlib.Path(
    __import__("os").environ.get(
        "LORAFORGE_AITOOLKIT_SRC", REPO_ROOT / ".scratch" / "src" / "ai-toolkit-main"
    )
)
DEPS = pathlib.Path(
    __import__("os").environ.get("LORAFORGE_AITOOLKIT_DEPS", REPO_ROOT / ".scratch" / "deps")
)


def _make_dataset(root: pathlib.Path) -> pathlib.Path:
    """三张假图 + 同名 .txt（只有文件名与字节，不是真图）。"""
    dataset = root / WORK
    dataset.mkdir()
    for index in (1, 2, 3):
        stem = f"frame_{index:02d}"
        (dataset / f"{stem}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (dataset / f"{stem}.txt").write_text(
            f"a photo of mychar, pose {index}\n", encoding="utf-8", newline="\n"
        )
    return dataset


def _load_ai_toolkit():
    """把 ai-toolkit 的加载器挂进来；装不上就 skip 并说明原因。"""
    if not SRC.exists():
        pytest.skip(f"ai-toolkit 源码树不在 {SRC}（设 LORAFORGE_AITOOLKIT_SRC 或跳过）")
    for entry in (str(DEPS), str(SRC)):
        if pathlib.Path(entry).exists() and entry not in sys.path:
            sys.path.insert(0, entry)
    try:
        from jobs import BaseJob  # noqa: PLC0415
        from toolkit.config import get_config  # noqa: PLC0415
        from toolkit.extension import get_all_extensions  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - 缺包就是 skip
        pytest.skip(f"ai-toolkit 未安装依赖：{type(exc).__name__}: {exc}")
    return get_config, BaseJob, get_all_extensions


@pytest.mark.parametrize("target", list(TargetModel))
def test_exported_train_yaml_is_trainable(target: TargetModel, scratch: pathlib.Path) -> None:
    dataset = _make_dataset(scratch)
    plan = dataclasses.replace(
        default_plan(target),
        dataset_dir=str(dataset),
        output_dir=str(scratch / "output"),
    )
    cfg = build_config(target, plan)
    payload = render_train_config(target, plan)
    config_path = scratch / "train.yaml"
    config_path.write_bytes(payload)

    report = package(dataset, config_path, scratch / "lora.zip")

    assert report.ok, (report.missing_caption, report.orphan_caption, report.ambiguous)
    assert (report.images, report.captions) == (3, 3)

    with zipfile.ZipFile(report.zip_path) as archive:
        member = f"{WORK}/train.yaml"
        assert member in archive.namelist(), archive.namelist()
        inside = archive.read(member)

    # 1. 逐字节；并钉住渲染口径（头部三行、纯 LF、以换行结尾）
    assert inside == payload
    assert b"\r\n" not in inside
    assert inside.endswith(b"\n")
    head = inside.decode("utf-8").splitlines()[:3]
    assert head[0].startswith("# 由 LoRAForge ") and "生成" in head[0]
    assert head[1].startswith("# 目标模型：") and "显存档位：" in head[1]
    assert head[2] == f"# 触发词：{plan.trigger}"
    # header=False 只剩 yaml 本体，与 build_config 的 dict 逐键一致
    body = render_train_config(target, plan, header=False)
    assert yaml.safe_load(body) == cfg
    assert not body.startswith(b"#")

    # 2. 语义与校验
    loaded = yaml.safe_load(inside)
    assert loaded == cfg
    assert validate_config(loaded, target) == []

    # 3. 能被 ai-toolkit 吃
    model_path, arch = EXPECTED[target]
    extracted = scratch / f"train-{target.value}.yaml"
    extracted.write_bytes(inside)
    get_config, BaseJob, get_all_extensions = _load_ai_toolkit()
    parsed = get_config(str(extracted))
    process_cfg = parsed["config"]["process"][0]
    assert parsed["job"] == "extension"
    assert process_cfg["type"] == "sd_trainer"
    assert process_cfg["model"]["name_or_path"] == model_path
    assert process_cfg["model"]["arch"] == arch
    assert process_cfg["network"]["type"] == "lora"

    extension = {item.uid: item for item in get_all_extensions()}["sd_trainer"]
    process_class = extension.get_process()
    assert f"{process_class.__module__}.{process_class.__name__}" == (
        "extensions_built_in.sd_trainer.SDTrainer.SDTrainer"
    )
    process_cfg["training_folder"] = str(scratch / "training")
    process = process_class(0, BaseJob(parsed), process_cfg)
    assert process.model_config.arch == arch
    assert process.network_config.type == "lora"


def test_tampered_config_inside_zip_is_caught(scratch: pathlib.Path) -> None:
    """反例：手改 zip 里的 train.yaml，校验层必须报出来（证明断言不是摆设）。

    两处改动分别落在 error 与 warn 两级：
    * 清空 ``model.name_or_path`` → error（会直接抛 ValueError，见 config_modules.py:676）；
    * ``arch: krea2:raw`` → warn（``model.arch-suffix``，冒号后缀会被截断，见
      config_modules.py:776-778 与 _review-severity.md 第 12 条）。
    """
    dataset = _make_dataset(scratch)
    plan = dataclasses.replace(
        default_plan(TargetModel.KREA2),
        dataset_dir=str(dataset),
        output_dir=str(scratch / "output"),
    )
    cfg = build_config(TargetModel.KREA2, plan)
    cfg["config"]["process"][0]["model"]["name_or_path"] = ""
    cfg["config"]["process"][0]["model"]["arch"] = "krea2:raw"
    config_path = scratch / "train.yaml"
    # 这里故意模拟「用户手改过的 yaml」，所以不走 render_train_config——
    # 渲染器保证的是它自己的产出，手改本来就不该经过它。
    config_path.write_bytes(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True).encode("utf-8")
    )
    report = package(dataset, config_path, scratch / "lora.zip")

    with zipfile.ZipFile(report.zip_path) as archive:
        loaded = yaml.safe_load(archive.read(f"{WORK}/train.yaml"))

    assert validate_config(loaded, TargetModel.KREA2), "空 name_or_path 必须被当 error"
    codes = {problem.code for problem in validate_config_detailed(loaded, TargetModel.KREA2)}
    assert "model.arch-suffix" in codes
