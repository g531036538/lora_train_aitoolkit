# 后端落地清单（实情，不是设计空间）

> 对应文档：[`_caption-backend-matrix.md`](./_caption-backend-matrix.md)（那是设计空间：3 目标 × 4 类型该由哪类后端产标签）。
> 本文只写**代码里真实存在、能选到的后端**，以及每个后端的 `reason` 现在到底会吐什么。
> 依据：`g531036538/lora_train_aitoolkit@e590223a`（2026-11-27 读取，GitHub contents API 原始字节）。
> 结论先行：**现在只有两个后端——`fake` 与 `openai-http`。T1 tagger（WD/Z3D）与 T2 本地 VLM（JoyCaption 等）不是「不可用」，是「没有实现」。**

## 1. 清单

| key | label | 代码 `name` | `caps()` | 依赖 | `available` / `reason` |
|---|---|---|---|---|---|
| `fake` | Fake（测试用，不读图） | `"fake"`（`loraforge/testing/fakes.py:29`） | `needs_gpu=False, supports_tags=True, supports_prose=True, languages=("en",)`（`fakes.py:33-45`） | 无 | 恒 `available=True`，`reason=""`（`loraforge/caption/pipeline.py:83-90`）。**契约已定（`9e31b659`，ARCHITECTURE §7）：默认不列，仅 `LORAFORGE_ENABLE_FAKE=1` 时列出**；main `e590223a` 仍恒列，差这一条待实现 |
| `openai-http` | OpenAI 兼容 Vision API | `"openai-http"`（`backends/openai_http.py:130`） | `needs_gpu=False, supports_tags=False, supports_prose=True, languages=("en","zh")`（`openai_http.py:40-45`） | `httpx` + 一个 OpenAI 兼容端点 | 见 §2 |

- `list_backends()` 恒返回这两条，**不可用也列**（`pipeline.py:78-95`，docstring 原话「不可用也列，并在 reason 里写清楚缺什么」）。
- `get_backend()` 只认 `fake` 与 `openai-http` / `openai` / `http` 三个别名（`pipeline.py:96-116`）；未知 key 抛 `KeyError`，不可用抛 `RuntimeError(reason)`。
- `fake` 的 `caption()` 不读图，返回 `f"{seed_text}, subject {stem}"`（`fakes.py:48`）——它只能验证流水线通不通，不能产真标签。

## 2. `openai-http` 的可用性判定（`openai_http.py:61-69`）

按顺序检查，先命中先返回：

| 顺序 | 条件 | `available` | 现在吐的 `reason` 字符串 |
|---|---|---|---|
| 1 | `import httpx` 失败（`DEPENDENCY_ERROR`，`:37`） | False | `缺少 httpx；无法调用 OpenAI 兼容端点` |
| 2 | 未设 `LORAFORGE_VLM_URL` | False | `未配置 LORAFORGE_VLM_URL` |
| 3 | 未设 `LORAFORGE_VLM_MODEL` | False | `未配置 LORAFORGE_VLM_MODEL` |
| 4 | 都配好 | True | `""` |

另一个入口：模块 import 就炸时，`pipeline.py:66-75` 兜底返回 `reason=f"后端模块不可用：{exc}"`，`caps` 用硬编码的同值。

环境变量（`openai_http.py:32-36`）：

| 变量 | 必填 | 用途 |
|---|---|---|
| `LORAFORGE_VLM_URL` | 是 | 端点根地址，如 `http://127.0.0.1:8000/v1` |
| `LORAFORGE_VLM_MODEL` | 是 | 模型名 |
| `LORAFORGE_VLM_KEY` | 否 | 无鉴权的本地端点可不给 |
| `LORAFORGE_VLM_TIMEOUT` / `LORAFORGE_VLM_MAX_TOKENS` | 否 | 请求超时 / 输出上限 |

## 3. `reason` 契约 vs 代码现状（差距表）

契约（`docs/ARCHITECTURE.md` §7，`85503a69` 起）：`reason` = **稳定前缀 + 可选细节**，前缀固定为
`ok` / `missing_dependency:<name>` / `model_weights_missing` / `requires_gpu` / `requires_network` / `requires_credentials` / `unsupported_target`。

**但 main 上的实现还是自由文本**（`e590223a`，小标在改）：

| 现状字符串 | 按契约应为 | 说明 |
|---|---|---|
| `缺少 httpx；无法调用 OpenAI 兼容端点` | `missing_dependency:httpx` | 细节可保留在冒号后，供 tooltip |
| `未配置 LORAFORGE_VLM_URL` | `requires_credentials:LORAFORGE_VLM_URL` | 缺配置归 `requires_credentials`，不是缺依赖 |
| `未配置 LORAFORGE_VLM_MODEL` | `requires_credentials:LORAFORGE_VLM_MODEL` | 同上 |
| `后端模块不可用：{exc}` | `missing_dependency:<name>` | `<name>` 需从异常里提出来，不能整句塞 |
| `""`（fake / 都配好） | `ok` | 空串与 `ok` 需二选一，界面按前缀分组时不能两种并存 |

在 `85503a69` 的实现合并前，**界面不能按前缀分组**——现在是自由文本，分组会全落进「其它」。这条是小标的活，本文只记录差距。

## 4. 对设计空间矩阵的映射（实情版）

| 组合 | 有没有合规后端 | 说明 |
|---|---|---|
| SDXL × 4 种 kind | **待补：`openai-http` tag 模式** | SDXL 要规范标签，而 `openai-http` 现在 `supports_tags=False`。小标正在加 tag 模式（对 SDXL 目标要求输出 Danbooru 风格标签，`supports_tags` 随之为 True）；落地前只能人工写。**不写「无后端」**——这条路是待补，不是封死 |
| Krea 2 × 4 种 kind | 有，`openai-http`（配好端点时） | `supports_prose=True`、`languages=("en","zh")`，与 Krea 2 的散文口径一致 |
| MOSS × character/style/concept | 有，`openai-http` | 同上；长文本 4096 token 上限是模型的，不是后端的 |
| MOSS × text_layout | 勉强有 | `openai-http` 未声明结构化输出能力，JSON 布局只能靠 prompt 约束，**未测** |

界面规则：**未实现的 T1/T2 不放进下拉框**；`fake` 默认不列（`LORAFORGE_ENABLE_FAKE=1` 才列，`9e31b659`）；`openai-http` 这类「存在但不可用」的要出现、置灰、显示 `reason`。

## 5. 开放问题（不擅自决定）

1. **已定（`9e31b659`）**：`fake` 默认不列，仅 `LORAFORGE_ENABLE_FAKE=1` 时列出。理由——它拿假标签冒充真结果，用户选中会得到「看起来成功」的假打标词，不报错、结果不可用，是最坏的一类失败。main `e590223a` 还恒列，需按契约补一条过滤。
2. `openai-http` 的结构化输出能力没测：MOSS 的 `text_layout` 需要 JSON，`caps` 里没有对应字段，是否要加 `supports_json` 得先有实测。
3. `reason` 与界面分组要对齐，但实现未合并（§3）。本文写法会随小标的改动更新。

## 6. 依据文件指纹（从 main 取的原始字节）

| 文件 | bytes | lines | sha256 |
|---|---|---|---|
| `loraforge/caption/pipeline.py` | 5,634 | 185 | `61b47d1b3de28cc5…` |
| `loraforge/caption/backends/openai_http.py` | 12,369 | 343 | `a3666b21cf5c00ee…` |
| `loraforge/testing/fakes.py` | 1,772 | 51 | `09ee70a9384f61c2…` |
| `loraforge/caption/base.py` | 1,762 | 62 | `d55cc0c2ced3e317…` |
