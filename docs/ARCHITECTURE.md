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
    # language 定案：训练用打标词一律英文，.txt 内容不随它变；zh 只影响给人看的
    # 说明文字（界面预览 / checks message / 错误说明）。要中文训练就新增 target，
    # 不要复用 language。触发词是拉丁串，语言开关不碰它。
    # 推论（硬约束）：任何 styler 的 render() 输出必须是纯 ASCII。段名、占位词、
    # 分隔符都算输出的一部分；汉字进 .txt 会被 BPE 切成多个碎片，污染 token 分布。
    # 中文预览是界面层的翻译，不落盘。

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

### 4.1 透明底合成（`loraforge/export/flatten.py`）

真图集里 55/86 张 PNG 带真透明通道。ai-toolkit 读到 RGBA 会自己转 RGB，
透明区变黑还是变白不确定——**训练前必须消掉**。原图不许动。

```python
def flatten_alpha(
    images: Sequence[Path], out_dir: Path,
    *, background: tuple[int, int, int] = (255, 255, 255),
    copy_sidecar: bool = True,
) -> FlattenReport

@dataclass
class FlattenReport:
    items: list[tuple[Path, Path | None, str | None]]  # 源、产物、错误原因
    written: int
    failed: int
```

规则：

1. 产物写进**新目录** `out_dir`，`out_dir` 已有同名文件就报错跳过，**绝不覆盖**原图。
2. 一律输出 RGB PNG（无 alpha）；`copy_sidecar=True` 时把同名 `.txt` 一并复制过去，
   内容逐字节不变——打标词与图必须继续配对。
3. 单张失败不中断整批，记进 `items` 的第三个字段（中文原因）。
4. 界面提供「合成到纯色底（生成新目录）」动作，纯色默认白，可选黑/灰，
   显示 `written/failed`，并提示「原图没有改动」。

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

**技术栈：FastAPI + Jinja2 + 原生 JS，不引入任何需要新装的东西。** 本机没有任何可用的
pip（`python -m pip` 不存在，`ensurepip` 因沙箱拒写临时目录而失败），所以 gradio 这类
框架装不上，也不许为了它去改环境。

- 入口：`python -m app` → uvicorn 起在 `127.0.0.1:7861`
- 静态资源：`app/static/`（CSS/JS 直接写，不用打包工具）
- 模板：`app/templates/`（Jinja2）
- 前端只发 HTTP 请求，后端路由层负责调 `loraforge`，业务逻辑一行都不许写在 JS 里

界面只做四件事，每件都调核心库：

1. 选文件夹 → 列出图片缩略图
2. 选模型/类型/后端 → 调 `caption` 流水线 → 显示进度
3. 逐图校对（左图右文本框）→ 保存回 `.txt`
4. 填参数 → 调 `build_config` → 预览 YAML → 调 `package` 导出

界面不实现任何打标或 YAML 逻辑。

---

## 7. 批处理与体检：界面唯一入口（`loraforge/caption/pipeline.py`、`checks.py`）

界面**只认下面这些名字**。这几个签名不实现就无法出「原因文字」，所以在这里定死。

```python
# loraforge/caption/pipeline.py
@dataclass(frozen=True)
class BackendInfo:
    key: str            # 界面下拉用这个值，如 "fake"、"wd-eva02"
    label: str          # 给人看的名字
    caps: BackendCaps
    available: bool     # 依赖/权重是否就绪
    reason: str         # 前缀式，见下方词表；available=True 时写 "ok"，不是空串

def list_backends() -> list[BackendInfo]:
    """列出全部已知后端。不得抛异常：不可用的也要列出来并说明原因。

    reason 必须是「稳定前缀 + 可选细节」的形式，前缀取自固定词表：
      ok / missing_dependency:<name> / model_weights_missing /
      requires_gpu / requires_network / requires_credentials / unsupported_target
    界面按前缀渲染图标与分组；前缀之后的文字随便写，给人看。
    新增前缀不算破坏契约，但必须同时更新本词表，否则界面只能按「未知」处理。

    「测试用假后端」（key="fake"）默认不列：它是拿假标签冒充真结果的演示品，
    出现在下拉框里会让用户以为打标成功。仅当环境变量 LORAFORGE_ENABLE_FAKE=1
    时列出，此时它的 label 必须带「测试用，不读图」字样。
    测试代码直接构造 FakeBackend()，不依赖 list_backends() 把它列出来。
    """

def get_backend(key: str) -> CaptionBackend:
    """按 key 取实例。未知 key → KeyError；依赖缺失 → RuntimeError(缺什么)。"""

@dataclass(frozen=True)
class BatchItem:
    image: Path
    caption: str | None      # 成功时为最终文本（已套 styler）
    error: str | None        # 失败时为原因；与 caption 互斥

@dataclass(frozen=True)
class BatchReport:
    items: list[BatchItem]
    written: int
    failed: int
    skipped: int             # 已有 .txt 且 overwrite=False
    @property
    def ok(self) -> bool: ...

ProgressHook = Callable[[int, int, Path, str | None, str | None], None]
# (done, total, image, caption_or_None, error_or_None)

def run_batch(
    images: Sequence[Path],
    ctx: CaptionContext,
    backend: CaptionBackend,
    *,
    styler: CaptionStyler | None = None,   # None → 按 ctx.target 取默认 styler
    overwrite: bool = True,
    on_progress: ProgressHook | None = None,
) -> BatchReport:
    """单张失败不中断整批：记进 items 的 error，继续下一张。"""
```

```python
# loraforge/caption/checks.py
CheckCode = Literal["ok", "empty", "too_short", "missing_trigger", "duplicate",
                    "identity_removed", "unquoted_text", "token_overflow"]
Severity  = Literal["info", "warn", "error"]

@dataclass(frozen=True)
class CheckResult:
    image: Path
    code: CheckCode
    severity: Severity
    message: str    # 给人看的中文原因，界面直接显示，不要再二次解释

def check_captions(
    pairs: Sequence[tuple[Path, str]], ctx: CaptionContext,
    *, raw: Mapping[Path, str] | None = None,
) -> list[CheckResult]:
    """每张图一条。全部正常也给一条 code="ok", severity="info"。

    raw 是「后端原文」，默认 None 时行为与不带 raw 完全一致。
    给了 raw 才能做两条 warn：
      - identity_removed：Krea2 剥掉了身份描述（脸型/发型/肤色等）
      - token_overflow：仅 SDXL。打标词超过 CLIP 的 77 token 上限，会被
        `truncate=True` 静默截断（`toolkit/train_tools.py:402-422`），触发词和尾部标签直接丢失。
        本机没有 transformers，**只能用近似口径**：逗号分隔标签数 > `SDXL_SOFT_TAG_LIMIT`(75)
        或英文词数 > `SDXL_SOFT_WORD_LIMIT`(60) 即报 warn。message 必须说明是近似估算，
        并给出可执行动作（压到 60 以内 / 把固定描述挪进采样 prompt）。不做精确 tokenizer 计数。
    新检查优先级高于 duplicate，但**仍然每张图只出一条**。
    """
```

`CheckCode` 是**可扩展的字面量**：以后还会加，界面**必须按 `severity` 上色、不要按 code 白名单**，
否则新增一个 code 就渲染不出来。

标红/标黄对应 `severity`：`error` → 红，`warn` → 黄，`info` → 正常。

### 7.3 整批体检（`loraforge/caption/dataset.py`）

`check_captions()` 回答「这一张有什么问题」；**整批才看得出的问题不许按张重复报**。
86 张图整批同一个黄字刷 86 遍，用户会学会无视黄字——这是已验证的真实事故。

```python
# loraforge/caption/dataset.py
DatasetCode = Literal["alpha_channel", "ubiquitous_tag", "small_resolution"]

@dataclass(frozen=True)
class DatasetIssue:
    code: DatasetCode
    severity: Severity
    message: str                 # 中文，含数量与样例文件名
    files: tuple[str, ...]       # 受影响文件名，最多 sample 个

def analyze_dataset(
    pairs: Sequence[tuple[Path, str]], ctx: CaptionContext,
    *, threshold: float = 0.8, sample: int = 20,
) -> list[DatasetIssue]:
```

- `alpha_channel`（warn）：存在带真透明通道的图（`alpha.getextrema()[0] == 0`）。
  ai-toolkit 会把 RGBA 读成 RGB 并自己猜底色，透明区可能变黑也可能变白；建议先合成到纯色底。
- `ubiquitous_tag`（warn）：某个标签出现在 **> threshold** 比例的**非空**打标词里，且张数 ≥ 5。
  一条 issue 列出全部这类标签。语义是「这批图太同质」或「这个词该进采样 prompt 而不是标签」。
- `small_resolution`（warn）：短边 < 512 的图。低于训练桶下限会被放大，细节不可追。

规则：

1. **每种 code 最多一条**，不许按图展开；`files` 给样例，数量写进 `message`。
2. `check_captions()` 的 `duplicate` 从此**只表示「这些图的打标词一字不差」**，
   不再承担「某个词出现在太多图上」——那是 `ubiquitous_tag` 的事。
3. 界面把 `analyze_dataset()` 的结果**单独一栏显示一次**（在逐图列表上方），
   不进逐图列表。逐图列表只显示 `check_captions()` 的结果。
4. 空打标词只从 `ubiquitous_tag` 的统计里排除；`alpha_channel` 与 `small_resolution`
   判的是**图片本身**，与有没有打标词无关——否则还没打标时就看不到「一半图带透明通道」。
5. `ubiquitous_tag` 不统计触发词：触发词本来就该 100% 出现，报它等于每批必响。

---

## 8. 配置侧的读口（`loraforge/configs/plans.py`）

界面不许硬编码任何默认数值，一律走这两个读口：

```python
def default_plan(target: TargetModel) -> TrainingPlan:
    """按模型给一份能直接用的初始方案。数值出处：docs/models/<target>.md。"""

@dataclass(frozen=True)
class VramTier:
    key: str            # "8" / "12" / "16" / "24" / "48"，单位 GB
    label: str
    quantize: str | None
    batch_size: int
    max_resolution: int
    note: str

VRAM_TIERS: dict[str, VramTier]
```

字段名固定；**取值由配置侧按调研文档填**，填错了改数据不改接口。

---

## 9. 测试用的假后端（`loraforge/testing/fakes.py`）

界面测试与打标测试共用同一份假后端，避免两处漂移：

```python
class FakeBackend:
    name: str                                  # 默认 "fake"
    def __init__(self, *, seed_text: str = "a photo of a person", fail_on: set[str] | None = None): ...
    def caps(self) -> BackendCaps: ...
    def caption(self, image: Path, ctx: CaptionContext) -> str:
        """确定性输出：由文件名派生，不含真实识图。fail_on 命中文件名则抛 RuntimeError。"""
```

这个模块由主管维护，其他人可以 import，不要改。
