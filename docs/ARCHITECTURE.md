# 架构与接口契约

> 这份文件里的接口签名是跨模块的唯一约定。改动必须先经主管确认。

## 分层

```
app/            ← 操作界面，只调 loraforge 的公开函数，不写业务逻辑
  ↓
loraforge/      ← 核心库，纯 Python，无 UI 依赖，可被 CLI/测试直接调用
  ├── caption/  ← 图 → 打标词
  ├── configs/  ← 参数 → ai-toolkit YAML
  └── export/   ← 文件夹 → zip
  ↓
文件系统        ← images/*.png + *.txt + train.yaml
```

核心库不允许 import gradio / fastapi。界面不允许直接拼 YAML。

---

## 1. 目标模型枚举

```python
# loraforge/models.py
from enum import Enum

class TargetModel(str, Enum):
    SDXL  = "sdxl"
    KREA2 = "krea2"
    MOSS  = "moss_image"

class DatasetKind(str, Enum):
    CHARACTER = "character"   # 角色/人脸：身份进触发词，打标词只写变化
    STYLE     = "style"       # 风格：打标词写内容，不写风格
    CONCEPT   = "concept"     # 物件/概念：打标词写语境
    TEXT_LAYOUT = "text_layout"  # 海报/包装/排版：MOSS 强项，文字要精确
```

---

## 2. 打标契约（`loraforge/caption/`）

```python
# loraforge/caption/base.py
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Literal

@dataclass(frozen=True)
class CaptionContext:
    target: TargetModel
    kind: DatasetKind
    trigger: str                 # 触发词，空字符串表示不注入
    language: Literal["en", "zh"] = "en"
    extra: dict | None = None    # 后端专属参数

@dataclass(frozen=True)
class BackendCaps:
    needs_gpu: bool
    supports_tags: bool          # 能否直接产标签（Danbooru 式）
    supports_prose: bool         # 能否产自然语言段落
    languages: tuple[str, ...]

class CaptionBackend(Protocol):
    name: str
    def caps(self) -> BackendCaps: ...
    def caption(self, image: Path, ctx: CaptionContext) -> str:
        """返回单张图的原始描述文本（未套模板）。失败必须抛异常，不许返回空串。"""
        ...

class CaptionStyler(Protocol):
    name: str
    def render(self, raw: str, ctx: CaptionContext, tags: list[str] | None = None) -> str:
        """把后端原始输出套成目标模型要的最终打标词。纯函数，不读文件。"""
        ...
```

**流水线**：`backend.caption()` → `styler.render()` → 写 `<image>.txt`。

- 后端负责「看懂图」，styler 负责「说对话」。两者解耦，才能三模型复用同一批后端。
- 每个 `TargetModel` 至少配一个 styler，实现放在 `loraforge/caption/stylers/`。
- 缺省 styler 的选择逻辑集中在一处，不要散落在界面里。

---

## 3. 配置契约（`loraforge/configs/`）

```python
# loraforge/configs/base.py
from dataclasses import dataclass, field

@dataclass
class TrainingPlan:
    model_path: str = "/path/to/model"      # 云端占位，导出前填
    dataset_dir: str = "/path/to/dataset"
    output_dir: str = "/path/to/output"
    trigger: str = "mytoken"
    steps: int = 2500
    rank: int = 32
    alpha: int = 32
    lr: float = 1e-4
    batch_size: int = 1
    resolutions: list[int] = field(default_factory=lambda: [512, 768, 1024])
    vram_gb: int = 24                       # 决定量化/offload 档位
    sample_prompts: list[str] = field(default_factory=list)


def build_config(target: TargetModel, plan: TrainingPlan) -> dict:
    """返回可直接 yaml.safe_dump 的 dict。必须是 ai-toolkit 能读的结构。"""

def validate_config(cfg: dict, target: TargetModel) -> list[str]:
    """返回问题列表，空列表 = 通过。检查项见 docs/models/<target>.md。"""
```

**硬约束**：生成的配置必须能被 ai-toolkit 的 `run.py` 直接读取。字段名以 ai-toolkit 主线源码为准，不以博客文章为准。

---

## 4. 导出契约（`loraforge/export/`）

```python
def package(dataset_dir: Path, config_path: Path, out_zip: Path,
            *, include_images: bool = True) -> ExportReport

@dataclass
class ExportReport:
    zip_path: Path
    images: int
    captions: int
    missing_caption: list[str]   # 有图无标的文件名
    orphan_caption: list[str]    # 有标无图的文件名
```

导出前必须做一次配对自检，把「有图无标」和「有标无图」都报出来。

---

## 5. 质量闸门（`loraforge/caption/checks.py`）

打标完成后必须跑一次体检，界面里给结果，不阻止流程但必须显示：

| 检查项 | 阈值 |
|---|---|
| 空打标词 | 直接标红 |
| 过短（< 10 字符或 < 3 个标签） | 标黄 |
| 触发词缺失 | 标红 |
| 多图打标词完全一致 | 标黄（说明后端在偷懒或图太像） |
| 某标签出现在 > 80% 图里 | 提示可加入丢弃列表 |

---

## 6. 界面契约（`app/`）

界面只做四件事，每件都调核心库：

1. 选文件夹 → 列出图片缩略图
2. 选模型/类型/后端 → 调 `caption` 流水线 → 显示进度
3. 逐图校对（左图右文本框）→ 保存回 `.txt`
4. 填参数 → 调 `build_config` → 预览 YAML → 调 `package` 导出

界面不实现任何打标或 YAML 逻辑。
