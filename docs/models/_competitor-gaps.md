# 竞品对照：我们缺什么（只看缺口，不抄功能）

> 2026-11-27 调研。对象：`EnragedAntelope/dataset-deviser`（最接近我们）、`jhc13/taggui`、`fpgaminer/joycaption`。
> 规则：只记「我们的交付物里缺的东西」；竞品有、但我们产品定位不需要的（比如生成训练图、接云 API 出图），一律写「不做」，不算缺口。
> 我们的定位（`README.md`）：只产出 `.txt` 打标词 + ai-toolkit YAML + 导出包，不训练、不装 ai-toolkit、不做数据集管理。

---

## 0. 结论先行

1. **方向是对的**：deviser 的三种数据集类型语义与我们的 `DatasetKind` 几乎逐字对应（char描述变化 / style描述内容 / concept描述语境），说明契约不是拍脑袋。
2. **真正的缺口在「打标词从哪来」和「打标结果怎么变成动作」**，不在参数默认值：我们写了「SDXL 用逗号短标签」，但没写标签由谁产出、怎么保证是规范标签而不是 VLM 编的句子。
3. **可复现性缺口**：joycaption 把每种打标模式落成一段固定 prompt 并回显所用 prompt；我们的 styler 规则散在 docs 里，没有「模板文本 + 版本」，同一批图两次打标无法回归对比。
4. 竞品的「数据集生成 / 云端出图 / 一键训」是它们的商业面，**明确不做**，不要被功能表带着跑。

---

## 1. dataset-deviser（9★，最接近）

### 做对了什么（有证据的部分）

| 做法 | 出处 | 对我们的意义 |
|---|---|---|
| 三种类型各自规定「caption 描述什么」：character→写变化；style→写内容、不写风格；concept→写语境、不写概念固定形态 | README「Dataset types」表 | 与 `DatasetKind` 语义一致；但我们的文档只按**模型**写规则，没按**类型**写，实现者只能猜 |
| 触发词与 caption 的分工写成一句硬规则：「The trigger word is what the LoRA learns; captions describe everything *except* it」 | README | 我们三份文档都隐含这条，但没有一句话把它立起来 |
| caption 风格按目标模型切：Prose（Flux/Krea/Qwen/SDXL3）/ Danbooru tags（SDXL、Illustrious、NoobAI、Pony-booru）/ e621 tags（Pony、furry） | README「Caption style」 | **我们缺这张矩阵**：SDXL 到底给哪些 checkpoint 用标签串、哪些用 prose，文档里没分 |
| 标签不由 VLM「编」，而是用专用 tagger（WD EVA02 / Z3D e621）直接读图产规范标签；并说明 VLM 产标签质量更差 | README「For canonical tags…」 | **我们缺**：`BackendCaps.supports_tags` 有这个能力位，但没有任何文档说 SDXL 的标签应由 tagger 产出 |
| Tag options：固定 `prefix/suffix`（如 Pony `score_9, score_8_up`）、丢弃噪声标签（`watermark`/`signature`）、追加 rating、保留下划线、tagger 阈值、跳过已打标 | README「Tag options」 | **我们缺模板级参数**（前缀/后缀/丢弃/rating/下划线），`CaptionStyler` 接口也没有对应入参位 |
| 一键 health check：空、过短、触发词缺失、**完全相同**、以及「几乎每张都出现的标签」→ 直接喂给丢弃列表 | README「health check」 | 闸门项与我们 `:143-149` 几乎一一对应；差别是他们把结果**接到动作**（丢弃列表），我们只提示 |
| 断点续跑：云端失败后「Skip images that already have a caption」重跑，不重复计费 | README | **我们缺**：`caption` 流水线没有「已有 .txt 则跳过」的语义，批量失败要重头跑 |
| 导出 flat `NN.png` + `NN.txt` + `metadata.json/jsonl` | README | 我们的 `package()` 有配对自检，但没有 metadata 清单；非必须，记为可选 |
| 「training is never launched for you」 | README | 与我们定位一致，可引用作为「不训练」的旁证 |

### 他们自己的坑（可直接变成我们的打标规则）

- 角色手里拿着的东西要**在生成时就命名**（"cup, plate"），否则每张都被重画 → 反过来就是打标规则：character 类里**道具/场景要写出「变」的部分**，否则会被学成角色的一部分。
- 大量放大的低清源图会让生成模型把瑕疵带进每一张 → 我们的数据集检查（低清硬放大）应进 caption/数据集建议。
- 全局只有一个 name + trigger，所有阶段共用 → 避免「不同阶段用不同触发词」的漂移；我们的 `TrainingPlan.trigger` 也是单值，方向一致。

---

## 2. taggui（1368★）

### 做对了什么

| 做法 | 出处 | 对我们的意义 |
|---|---|---|
| 键盘优先的逐图打标：图与文本框同屏、N/M 进度、保存并下一张 | README Features/Load Directory | 与我们界面契约第 3 条一致；他们把它做成主交互，不是附属页 |
| 标签自动补全来自**用户自己最常用的标签**（不是通用词表） | README Features | 打标一致性的低成本手段；我们闸门只看「>80% 标签」，没有「收口到用户词表」 |
| 内置 token 计数（面向 Stable Diffusion） | README Features | **我们缺**：SDXL CLIP 77 token 静默截断（`train_tools.py:402-422`），闸门只数「字符/标签数」，标签下划线多时字符数会严重低估 token |
| 批量操作：重命名、删除、排序标签；图片列表过滤 | README Features | 校对阶段的批量清洗；我们界面契约未提，属产品面 |
| 自动打标内置多模型（含 JoyCaption），采样参数（prompt/长度）可调 | README Auto-Captioning | 与我们 `CaptionBackend` 抽象同构，说明后端可插拔是对的 |

### 我们不需要的

- 桌面打包、跨平台发布、热键体系——不是文档该管的。

---

## 3. joycaption（1305★）

### 做对了什么

| 做法 | 出处 | 对我们的意义 |
|---|---|---|
| 每种打标模式是一段**固定、可复制的 prompt 原文**（Descriptive / Straightforward / SD Prompt / Danbooru / e621 …），并说明哪种稳、哪种不稳 | README「How to Prompt JoyCaption」 | **我们缺**：styler 规则应落成「模板字符串 + 参数 + 版本号」，否则不可复现、不可回归 |
| 界面**回显所用 prompt** | README 同上 | 打标结果可追溯；我们文档里写了规则，但没写「这次用的是哪版模板」 |
| `Straightforward` 的硬约束值得直接并入我们三模型：以「主体 + 媒介」开头；只写具体可见（颜色/形状/材质/空间关系）；**不写画面里没有的**；**不写分辨率/不可观测项**；文字原样引号；提到水印/签名/压缩伪影 | README 第 2 条 | 这几条与我们已有的「别写身份细节/别写套话」互补，尤其「不写不存在的」和「水印」两条 |
| 明确标注哪些模式不稳定（SD Prompt 约 3% 抽风、MidJourney 会复读） | README 备注 | 提醒我们：VLM 打标要假设会抽风，闸门必须能兜住（我们已经用「多图完全一致」兜） |
| 显存与量化（bf16 约 17GB，8/4-bit 可压） | README VRAM | 部署信息，非文档缺口 |

---

## 4. 缺口清单（按优先级）

| # | 缺口 | 影响 | 建议补在哪 |
|---|---|---|---|
| G1 | 无「目标模型 × 数据集类型 → 后端能力」矩阵；SDXL 的标签由 tagger 还是 VLM 产出没有规定 | 用 VLM 产标签串会得到「像标签的句子」，SDXL LoRA 学不到规范标签 | `docs/models/sdxl.md` ④ + 一份 `docs/models/_caption-backends.md`（新增，待主管确认写集） |
| G2 | `CaptionStyler` 无模板级参数位：前后缀、丢弃标签、rating、下划线、tagger 阈值 | Pony/Illustrious 用户必须手改每一张 | 接口归主管改；文档先写「需要哪些参数」 |
| G3 | 无断点续跑语义（已有 `.txt` 跳过） | 批量失败重跑会重复计费 | `app/` + `caption/` 实现面，文档提一句 |
| G4 | 闸门只提示，不产出可执行动作（丢弃列表候选） | 「>80% 标签」用户还得自己删 | 闸门输出加 `drop_candidates`（接口归主管） |
| G5 | SDXL token 数不检查（只查字符数/标签数） | 77 token 静默截断，用户看不出 | `docs/models/sdxl.md` ④/⑤ + 闸门 |
| G6 | styler 规则没有「模板文本 + 版本」，不可复现 | 两次打标不一致、无法回归测试 | 三份模型文档的 ④ 改成「模板原文 + 参数表」 |
| G7 | `DatasetKind` 有枚举，但「每类 caption 写什么」没有成文规则（只有逐模型规则） | 实现者按模型猜，风格类会被写成内容类 | 三份模型文档 ④ 加「按类型的写法」一小节 |

**不做（明确排除）**：生成训练图、接云端出图/打标计费、一键训练、数据集管理器、桌面打包。竞品有这些是因为它们卖的是「数据集生成」，我们卖的是「打标 + 配置」，越界会拖垮主线。

---

## 5. 与「二手线索」的边界

主管给的种子线索里有一条 `dataset-deviser` 的观感（同一套衣服每张都出现→服装被学成角色）。本文件**没有**把它当实测结论：它在本仓库的链条是「主管 → 竞品 README → 作者笔记」，我只在 §1 引用了 README 里**能直接读到**的规则（命名道具、caption 描述变化），不再加一层转述。真正的竞品判断以上面表格的原文为准。
