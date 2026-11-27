# 打标后端能力矩阵：目标模型 × 数据集类型

> 本文回答一个问题：**每个 (target, kind) 组合，该由哪类后端产标签，以及我们本机能不能跑。**
> 只讨论「标签由谁产」，不讨论模板前后缀、丢弃策略、rating（那些是 G2/G4，已明确进 backlog，见 `_competitor-gaps.md`）。
> 证据都是 2026-11-27 抓的 `_src/` 本地副本，出处写成「本地路径:行号」；对应公网 URL 与可信度登记在 `docs/sources.md` §9/§10。

## 0. 一句话结论

- **SDXL 走规范标签（Danbooru tagger）**，Krea 2 与 MOSS 走自然语言（VLM）。这不是「谁描述得更好」，而是两种文本体系：tagger 输出的是模型词表里的规范标签，VLM 输出的是自由句子。
- **本机跑不了真正的 tagger 与本地 VLM**：缺 `onnxruntime`（tagger 运行时）与 `transformers`（本地 VLM 运行时）。当前只剩两条路——远程 HTTP VLM、或人工/模板兜底。
- 所以「可用 / 不可用」这一列，指的是**当前这台开发机**；用户机器装不装可选依赖，我们无法替他们验证，只能在界面上给出 `reason`。

- 哪些后端**真的存在**、每个的 `reason` 现在怎么写，见 [`_backend-inventory.md`](./_backend-inventory.md)。设计空间（本文）与已实现清单不要混写。

## 1. 两种文本体系，不要混成「质量高低」

| | 规范标签（Tagger） | 自然语言（VLM） |
|---|---|---|
| 输出形态 | 逗号分隔的受控词表词条，如 `1girl, blue hair, school uniform` | 完整句子/段落 |
| 谁吃它 | SDXL 及在其标签语料上微调的底模（Illustrious / NoobAI / Pony 系） | Qwen3-VL / Qwen3.5 这类 LLM 文本编码器（Krea 2、MOSS） |
| 给错模型的后果 | 喂给 Krea 2/MOSS：标签串不是它们被训练的表达形式 | 喂给 SDXL：两种**不同**的失败，见 §1.1——分布偏移（词不在词表里）与信息损失（77 token 截断），不是一个问题 |

一句话：**VLM 给 SDXL 打标的问题不是「标签难看」，是「标签不在模型的词表分布里」。** 这条口径按主管 2026-11-27 的纠正写，不是我们的实测。

### 1.1 两个不同的问题，别合并

| 问题 | 机制 | 后果 | 能不能靠写法缓解 |
|---|---|---|---|
| **分布偏移** | VLM 产的是自由语言，用词不在 CLIP 训练分布里；tagger 产的是受控词表 | 模型「认不出」这些词，条件信号弱 | **不能**。只能换后端（用 tagger），精简句子没用 |
| **信息损失** | CLIP-L/G 上限 77 token（`_src/ai-toolkit/toolkit/train_tools.py:402-422` 取 `tokenizer.model_max_length`、`truncate=True` 默认） | 长句被静默截断，**丢掉描述内容** | **能**。人工精简/截短、或只保留变化最强的部分 |

主管 2026-11-27 指出：这两条并列写容易读成一回事。换后端解决第一个，改写法只能缓解第二个。

## 2. 后端分四类（与 `BackendCaps` 对齐）

契约见 `docs/ARCHITECTURE.md:51-77`：`BackendCaps{needs_gpu, supports_tags, supports_prose, languages}`。

| 代号 | 后端类型 | 典型实现 | caps 预期 | 备注 |
|---|---|---|---|---|
| T1 | 规范标签 tagger | WD EVA02（Danbooru）、Z3D e621（furry/Pony） | `supports_tags=True, supports_prose=False` | 需 ONNX 运行时，可 CPU |
| T2 | 本地 VLM | JoyCaption、Florence-2、Qwen2.5-VL | `supports_prose=True`；支持结构化输出的才有 `supports_tags` 之外的 JSON 能力 | 需 transformers + 权重 + 显存（JoyCaption bf16 约 17GB，见 `_src/joycaption-README.md`） |
| T3 | 远程 VLM（HTTP） | OpenAI 兼容端点、任意自建 HTTP | 取决于模型：`supports_prose`，部分可约束 JSON | 需出网 + 用户自备 key/endpoint |
| T4 | 无后端（人工/模板） | 触发词 + kind 模板、逐图手写 | 恒可用 | 没有「看懂图」的能力，只保证流水线不断 |

## 3. 矩阵

列：**推荐后端类型 / 为什么 / 本机可用性 / 不可用时给用户的替代路径**

### 3.1 SDXL（`caption_style="tags"`，`loraforge/models.py` PROFILES）

| kind | 推荐后端 | 为什么 | 本机可用性 | 替代路径 |
|---|---|---|---|---|
| character | **T1 tagger** | 身份交给触发词，标签只写这次的变化（服装/动作/场景）；SDXL 族吃规范标签（`_src/deviser-README.md:178-179`「Danbooru tags … SDXL, Illustrious, NoobAI … which learn poorly from prose」） | ❌ `missing_dependency: onnxruntime` | 远程 VLM 出句子 → 标签化 styler 压成逗号标签 + 人工校对；或 T4 逐图手写 |
| style | **T1 tagger** | 规则是「写画面内容、不写风格词」（`_src/deviser-README.md:176-180`）：内容标签正是 tagger 的强项 | ❌ 同上 | 同上 |
| concept | **T1 tagger** | 概念 LoRA 要的是「触发词 + 语境标签」；关系型描述用标签也能表达 | ❌ 同上 | 同上 |
| text_layout | **T3 远程 VLM**（首选）；T1 不够 | tagger 不保序、不保留原文字符串；SDXL 不吃 JSON，只能「引号内原样文字 + 版式标签」，且要压进 77 token | ⚠️ T3 可用但需 `requires_network` + `requires_credentials`；T1 ❌ | T4 人工抄字（图内文字少时最稳） |

### 3.2 Krea 2（`caption_style="prose"`）

共同的硬依据：`_src/ai-toolkit/extensions_built_in/diffusion_models/krea2/src/text_encoder.py:28-33` 的固定 system 指令要求模型按 *color, shape, size, texture, quantity, text, spatial relationships* 描述画面；`encode_krea_prompt` 的 `max_length` 默认 512（`:45`）。所以 Krea 2 的 caption 应该是「描述这些属性」的句子，标签串不是它的表达形式。

| kind | 推荐后端 | 为什么 | 本机可用性 | 替代路径 |
|---|---|---|---|---|
| character | **T2 本地 VLM**（首选）/ T3 | 句子只写变化，身份不进 caption（`README.md:18-23` 差异表） | ❌ `missing_dependency: transformers`（另需权重与显存）；⚠️ T3 可用 | T3（推荐当前唯一可行）；或 T4 手写句子 |
| style | **T2 / T3** | 写画面内容、不写风格词；句子才装得下内容描述 | ❌ / ⚠️ 同上 | 同上 |
| concept | **T2 / T3** | 概念要写语境与关系，句子比标签更自然 | ❌ / ⚠️ 同上 | 同上 |
| text_layout | **T3**（需模型会原样抄文字） | 指令里明确要求描述 `text` 与 `spatial relationships`；引号保文字、句子写位置 | ❌ / ⚠️ | T4 手写（并逐字校对文字） |

### 3.3 MOSS-Image（`caption_style="prose_precise"`）

硬依据：TE 是 Qwen3.5-9B、`max_text_length=4096`（`_src/.../moss_image/moss_image.py:70,183-195`）；官方打标输出是 `{caption:{long,medium,short}, watermark:[…], borders:[…]}`，长 caption 要含全部可见文字且原样保留（`_src/moss-repo-USERGUIDE.md` 打标章节）；布局用官方 JSON schema（`bbox_format: yxyx` / `bbox_coord_space: abs_centralized`）。

| kind | 推荐后端 | 为什么 | 本机可用性 | 替代路径 |
|---|---|---|---|---|
| character | **T2 / T3**（要长句 + 可给多候选） | 官方训练用 LLM 写的长自然语言，且一图多 caption 候选、训练时随机取一（kohya `docs/moss_image_train_network.md`） | ❌ `missing_dependency: transformers`；⚠️ T3 可用 | T3；或 T4 手写长句 + 多候选 |
| style | **T2 / T3** | 同上；内容描述越长越占便宜（4096 预算） | ❌ / ⚠️ | 同上 |
| concept | **T2 / T3** | 同上 | ❌ / ⚠️ | 同上 |
| text_layout | **T3（结构化输出）** 首选 | 只有 JSON 能表达 bbox 与文字块；tagger 完全不行 | ⚠️ T3 可用但**要端点支持 JSON/结构化输出**；❌ T2 | T4 按官方 draft 格式手写，再用 `compile_prompt.py` 编译 |

## 4. 本机可用性实测（2026-11-27，`importlib.util.find_spec`）

| 依赖 | 状态 | 影响 |
|---|---|---|
| `onnxruntime` | **MISSING** | T1 tagger 跑不了 |
| `transformers` | **MISSING** | T2 本地 VLM 跑不了 |
| `open_clip` / `timm` | **MISSING** | 部分 tagger/VLM 实现的预处理依赖 |
| `torch` / `huggingface_hub` / `numpy` / `PIL` | present | 只够做张量/IPC/解码，不够跑模型 |

结论：**本机唯一可用的是 T3（远程 HTTP）与 T4（人工/模板）。** 这是「当前开发机」的事实，不是产品约束；但既然本机就缺，界面必须默认按「可能没有本地运行时」设计。

## 5. `list_backends()` 的 reason 口径（提案，无现有实现）

仓库里 `grep -rn "list_backends\|reason"` 0 命中（2026-11-27），以下词汇是本文提案，实现时以契约为准：

| reason | 含义 | 何时出现 |
|---|---|---|
| `ok` | 可用 | 包/权重/key 都就绪 |
| `missing_dependency: onnxruntime` | 缺运行时 | T1 |
| `missing_dependency: transformers` | 缺运行时 | T2 |
| `model_weights_missing` | 有运行时、没权重 | T1/T2 未下载模型 |
| `requires_gpu` | 需要 GPU 且当前不可用 | T2 显存不足 |
| `requires_network` | 需出网 | T3 离线 |
| `requires_credentials` | 缺 endpoint/key | T3 未配置 |
| `unsupported_target` | 输出体系与目标模型不匹配（**只提示，不阻断**） | 例如「用 tagger 给 MOSS 打标」 |

`unsupported_target` 与其它 reason 的区别：其它是「跑不起来」，它是「跑得起来但会产出错体系的东西」。矩阵里所有 ❌/⚠️ 的组合都应按这一类给出提示，而不是把后端从列表里删掉。

## 6. 界面默认值建议

| target | 默认后端 | 不可用时的降级顺序 |
|---|---|---|
| SDXL | T1（装了 `onnxruntime` 时） | T3 + 标签化 styler → T4 手写 |
| Krea 2 | T3（本机没有 `transformers` 时） | T2 本地 VLM（若装）→ T4 手写 |
| MOSS | T3（结构化输出） | T2 本地 VLM（若支持 JSON）→ T4 手写 |

## 7. 边界与未查实

- **未实测**：T1 标签器精度、T3 各模型对 Krea/MOSS 的产标质量、MOSS 结构化输出在常见端点上的成功率——都是「按体系推的」，不是跑出来的。
- **未覆盖**：多语言（`language="zh"`）时 tagger 一律不可用（Danbooru 词表是英文），只有 T2/T3 能出中文；本轮没测中文产标质量。
- **不实现**：不自动下载模型权重、不在 `validate_config` 里因后端不可用而报错（只在界面提示）。
- 本文是 G1 的唯一产出；G2（styler 模板参数）、G4（`drop_candidates`）按主管 2026-11-27 决定进 backlog，不在本文范围。
