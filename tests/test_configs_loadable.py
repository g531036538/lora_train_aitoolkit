"""让 ai-toolkit 自己的加载器读我们的示例 yaml。

这不是「我们的校验器说没问题」，是「它的 `get_config` 能把这几个文件读出来，
并且读到的值就是我们登记的值」。装了 ai-toolkit 依赖的机器上会真跑；
没装的机器上**跳过并说明原因**，不能让没装依赖的人看到红。

准备方式（不进仓库）：

    python -c "import shutil,pathlib; shutil.rmtree('.scratch/deps', ignore_errors=True)"   # 可选，清干净
    # oyaml/diffusers/... 装进 .scratch/deps：
    #   python .scratch/work/pip_patched.py install --no-cache-dir --no-deps --target .scratch/deps <pkg>

源码树路径可用环境变量覆盖：`LORAFORGE_AITOOLKIT_SRC`、`LORAFORGE_AITOOLKIT_DEPS`。
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = pathlib.Path(
    os.environ.get("LORAFORGE_AITOOLKIT_SRC", REPO_ROOT / ".scratch" / "src" / "ai-toolkit-main")
)
DEPS = pathlib.Path(os.environ.get("LORAFORGE_AITOOLKIT_DEPS", REPO_ROOT / ".scratch" / "deps"))
EXAMPLES = REPO_ROOT / "configs" / "examples"

# 五个字段的登记值：改这里等于改「我们声称 ai-toolkit 会读到什么」。
EXPECTED = {
    "sdxl.yaml": {
        "job": "extension",
        "process_type": "sd_trainer",
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
        "arch": "sdxl",
        "network": "lora",
    },
    "krea2.yaml": {
        "job": "extension",
        "process_type": "sd_trainer",
        "model": "krea/Krea-2-Raw",
        "arch": "krea2",
        "network": "lora",
    },
    "moss_image.yaml": {
        "job": "extension",
        "process_type": "sd_trainer",
        "model": "OpenMOSS-Team/MOSS-Image-Base",
        "arch": "moss_image",
        "network": "lora",
    },
}


def _load_get_config():
    """把 ai-toolkit 的 get_config 挂进来；装不了就 skip，不装成通过。"""
    if not SRC.exists():
        pytest.skip(f"ai-toolkit 源码树不在 {SRC}（设 LORAFORGE_AITOOLKIT_SRC 或跳过）")
    for entry in (str(DEPS), str(SRC)):
        if pathlib.Path(entry).exists() and entry not in sys.path:
            sys.path.insert(0, entry)
    try:
        from toolkit.config import get_config  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - 缺包就是 skip
        pytest.skip(f"ai-toolkit 未安装依赖：{type(exc).__name__}: {exc}")
    return get_config


@pytest.mark.parametrize("filename", sorted(EXPECTED))
def test_example_is_readable_by_ai_toolkit(filename: str) -> None:
    get_config = _load_get_config()
    path = EXAMPLES / filename
    assert path.exists(), f"示例文件不在：{path}"

    cfg = get_config(str(path))
    process = cfg["config"]["process"][0]
    expected = EXPECTED[filename]

    assert cfg["job"] == expected["job"]
    assert process["type"] == expected["process_type"]
    assert process["model"]["name_or_path"] == expected["model"]
    assert process["model"]["arch"] == expected["arch"]
    assert process["network"]["type"] == expected["network"]

# 第二个层次：光「读得出来」还不够，得能分派到真正的 process 类。
# `get_job()` 在这台机器上被无关扩展链卡住（见 Issue #47），所以这里只走
# 「configuration -> BaseJob -> sd_trainer 扩展 -> SDTrainer 类实例化」这一段：
# 用的还是上游自己的类，没有 mock，也没有改它的源码。
EXPECTED_PROCESS = {
    "sdxl.yaml": ("sd_trainer", "extensions_built_in.sd_trainer.SDTrainer.SDTrainer", "sdxl"),
    "krea2.yaml": ("sd_trainer", "extensions_built_in.sd_trainer.SDTrainer.SDTrainer", "krea2"),
    "moss_image.yaml": ("sd_trainer", "extensions_built_in.sd_trainer.SDTrainer.SDTrainer", "moss_image"),
}


def _load_dispatch():
    """拿到 get_config / BaseJob / get_all_extensions；拿不到就 skip 并说原因。"""
    get_config = _load_get_config()
    try:
        from jobs import BaseJob  # noqa: PLC0415
        from toolkit.extension import get_all_extensions  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - 缺包就 skip
        pytest.skip(f"ai-toolkit 依赖不全：{type(exc).__name__}: {exc}")
    return get_config, BaseJob, get_all_extensions


@pytest.mark.parametrize("filename", sorted(EXPECTED_PROCESS))
def test_example_dispatches_to_real_process_class(filename: str, scratch: pathlib.Path) -> None:
    get_config, BaseJob, get_all_extensions = _load_dispatch()
    cfg = get_config(str(EXAMPLES / filename))
    proc_cfg = cfg["config"]["process"][0]
    proc_cfg["training_folder"] = str(scratch / "training")

    uid, class_name, arch = EXPECTED_PROCESS[filename]
    assert proc_cfg["type"] == uid

    extension = {item.uid: item for item in get_all_extensions()}[uid]
    process_class = extension.get_process()
    assert f"{process_class.__module__}.{process_class.__name__}" == class_name

    job = BaseJob(cfg)
    process = process_class(0, job, proc_cfg)
    assert type(process) is process_class
    assert process.model_config.arch == arch
    assert process.network_config.type == "lora"

