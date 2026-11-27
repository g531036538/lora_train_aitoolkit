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
