# 来源清单

> 规则：每条外部来源都记 **URL + 访问日期 + 可信度**；访问失败的也要记，标「访问失败」并保留原始错误。可信度分四档：
>
> - **一手官方源码** — 仓库源码逐行可查，本项目的最高证据等级
> - **官方文档** — 模型/框架作者自己写的文档、README、用户指南
> - **第三方官方（部署实测）** — 别的框架/平台官方文档里对同一模型的部署记录，可信但非模型作者口径
> - **二手转述** — 博客、论坛、二手仓库、主管种子文件；只作线索，不作默认值
>
> 所有访问日期均为 **2026-11-27**（Asia/Singapore）。

---

## 1. ai-toolkit 源码（一手官方源码）

- 仓库：`https://github.com/ostris/ai-toolkit`
- 版本：`VERSION = "0.13.43"`（`version.py:1`）；本地 `_src/ai-toolkit` 抓取日 2026-11-27
- commit：`73c84b091dd0f5647fc6a4b59122a2efc1c8147b`（GitHub API `list_commits` 查得 main HEAD，committer date 2026-11-18T19:26:49Z；tarball 无 `.git`，故用 API 反查）
- 可信度：**一手官方源码**
- 本文件引用的路径：`toolkit/config.py`、`toolkit/config_modules.py`、`toolkit/models/registry.py`、`toolkit/sampler.py`、`toolkit/samplers/custom_flowmatch_sampler.py`、`toolkit/lora_special.py`、`toolkit/buckets.py`、`toolkit/dataloader_mixins.py`、`toolkit/optimizer.py`、`toolkit/scheduler.py`、`toolkit/train_tools.py`、`toolkit/util/ostris_quant.py`、`jobs/process/BaseSDTrainProcess.py`、`extensions_built_in/sd_trainer/SDTrainer.py`、`extensions_built_in/diffusion_models/krea2/krea2.py`、`extensions_built_in/diffusion_models/krea2/src/text_encoder.py`、`extensions_built_in/diffusion_models/moss_image/moss_image.py`、`extensions_built_in/diffusion_models/moss_image/src/transformer.py`、`extensions_built_in/diffusion_models/moss_image/src/pipeline.py`、`extensions_built_in/diffusion_models/moss_image/src/checkpoints.py`、`config/examples/*.yaml`

## 2. MOSS-Image 官方仓库文档（官方文档）

- `https://raw.githubusercontent.com/OpenMOSS/MOSS-Image/main/README.md` — 访问成功
- `https://raw.githubusercontent.com/OpenMOSS/MOSS-Image/main/USERGUIDE.md` — 访问成功
- 主仓：`https://github.com/OpenMOSS/MOSS-Image`
- 可信度：**官方文档**（模型作者自述）
- 采用的关键口径：
  - Base 未做 CFG 等推理后训练/蒸馏，目的就是让社区训 LoRA；SFT 是微调版
  - 5B backbone；42-layer diffusion transformer + Qwen3.5-9B TE + FLUX.2 VAE
  - 推理：1M/1.5M/2M 总像素（0.25M–4M）、宽高 16 的倍数、steps ≥50、CFG 3.0（3.0–6.0）、shift 5.0、Euler、SLG 跳过层 22（22/24/26 均可）
  - 加载 dtype：`{"default": bfloat16, "vae": float32}`
  - LoRA 训练：**keep the text encoder frozen. Do not fine-tune it.**
  - 布局 JSON 正式 schema：`summary{meta,canvas,caption}` + `items[{label,caption,text?,bbox}]` + `bbox_coord_space: abs_centralized` + `bbox_format: yxyx`（`[y_min,x_min,y_max,x_max]`，中心原点，1024 画布 −512..+512）
  - 打标：官方 recaptioner prompt（`caption.long/medium/short` + `watermark[].box_2d` 归一化 0–1000 + `borders[]`）；文字原样保留不翻译；推荐 Gemma 4 / Qwen 3.5 27B / Qwen 3.5 35B-A3B 并开 thinking
  - 负向词建议：`nsfw, worst quality, bad anatomy, watermark, jpeg artifact`
  - 标签序列顺序：Quality→Score→Meta→Year→Character→Series→Artist→General
  - 显存：1024×1024 / 50 步峰值 GPU 3.65/3.85 GiB（含 offload）、系统内存 14 GiB
  - License：pipeline Apache 2.0；技术报告 arXiv 2610.23374（2026-11-27 已读 HTML 全文，见 §10.1）

## 3. SGLang Krea 2 部署 cookbook（第三方官方，部署实测）

- `https://raw.githubusercontent.com/sgl-project/sglang/71de97b264b04dcd514cf904003028aefe9775c8/docs/cookbook/diffusion/Krea/Krea-2.mdx` — 访问成功
- 仓库：`https://github.com/sgl-project/sglang`（commit `71de97b264b04dcd514cf904003028aefe9775c8`）
- 可信度：**第三方官方（部署实测）**——不是 Krea 官方口径，但记录了两卡真实跑起来的参数与显存
- 采用的关键口径：
  - 两变体同 backbone，只差采样配方：Turbo 蒸馏 8 步 / guidance 1.0（无 CFG）；Raw 非蒸馏 base / ~52 步 / guidance ≈4.5
  - 单流 MMDiT + Qwen3-VL TE + Qwen-Image VAE，diffusers 布局；48 query heads / 12 KV heads / 20 text heads
  - DiT bf16 约 **24 GB**；2×H200 峰值 38 GB → tp=2 后 27 GB/卡
- **口径警告**：SGLang 的 guidance 是 diffusers 语义（1.0 = 关），ai-toolkit 的 `krea2.py:455` 会把 yaml 值减 1 再传给 pipeline。两者不能直接互换，见 `docs/models/krea2.md` ③ 的说明。

## 4. 访问失败的来源（保留原始错误）

| URL | 目标 | 结果 |
|---|---|---|
| `https://huggingface.co/krea/Krea-2-Raw` | Krea 2 Raw 官方 model card | 网页抓取通道 **失败**（`TypeError: fetch failed`）；改 Python `urllib` + User-Agent 后返回 **HTTP 401**（`gated: auto`，需登录并接受 License）——未绕过，见 §8 |
| `https://huggingface.co/krea/Krea-2-Raw/raw/main/README.md` | 同上（raw README） | 同 401（urllib 取得状态码，正文未取得） |
| `https://www.krea.ai/krea-2-open-source` | Krea 官方开源页 | **访问失败**：`TypeError: fetch failed`（本轮未再试 urllib 通道） |
| `https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base` | MOSS Base 官方 model card | 网页通道失败，**`urllib` 直连成功** → README 全文 + `api.json` 已取得，见 §8 |
| `https://www.reddit.com/r/StableDiffusion/search.json?q=SDXL+LoRA+dataset+size` | r/StableDiffusion 检索（JSON） | **访问失败**：`TypeError: fetch failed` |
| `https://old.reddit.com/r/StableDiffusion/search?q=SDXL+LoRA+dataset+size&restrict_sr=on` | 同上（old 站，绕开前端 JS） | **访问失败**：`TypeError: fetch failed` |
| modsearch `read_page` → firecrawl | 同上（第三方渲染通道） | **403（firecrawl 拒绝）+ local fetch 失败** |

> 影响：本轮 **MOSS Base 的 model card 已取得**（§8），MOSS 能引一手口径；**Krea 2 Raw 仍缺 model card**（gated，需登录接受许可），其参数量/权重细节仍是 `UNVERIFIED`。Reddit 三条路径全失败（见下），论坛口径本轮没有来源。
>
> 复现方式：本机 `node fetch` 与 modsearch 通道对 HF 均失败，Python `urllib.request` + `User-Agent` 头可通（2026-11-27 由主管鱼鱼验证并取回 §8 的三个文件）。

## 5. 二手转述与未访问来源（只作线索）

| 来源 | 类型 | 可信度 | 本轮处理 |
|---|---|---|---|
| 主管种子笔记（原 `docs/models/_seed-from-lead.md`；main 上已移至 `docs/research/lead-seed-notes.md`） | 内部二手线索 | **二手转述**，未核实 | 2026-11-27 已逐条对照源码/官方文档；本仓库里的旧副本已删除。核对结果见三份模型文档「未查实」表；本文件不再把它当引用来源 |
| `chengyansen-ai/krea2-lora-training`（MIT，约 6★） | 第三方训练仓库 | **二手汇编**（2026-11-27 已直接读取 README 与 `references/`，见 §9.1） | 其 Krea2 参数（32/32、1e-4、2500 步、flowmatch、linear、桶 512/768/1024、caption_dropout 0.05、DOP）与源码机制不冲突，但作者未公开日志，**没有一条经我们复现**；只作起点参考与踩坑索引 |
| `EnragedAntelope/dataset-deviser` | 竞品仓库 | 已直接读取 README（2026-11-27，见 §9.3） | README 里可直接读到的规则（trigger 学身份 / caption 写变化 / 道具先命名）作为**竞品规则**引用；「同一套衣服每张都出现→服装被学成角色」这类观感仍属二手转述，见 `_competitor-gaps.md` §5 |
| `jhc13/taggui`、`fpgaminer/joycaption` | 竞品仓库 | 已直接读取 README（2026-11-27，见 §9.3） | 结论只记「我们缺什么」，见 `_competitor-gaps.md` |
| Reddit / Civitai / Discord 社区经验 | 社区 | 传闻 | Reddit 全线不可达（JSON 接口、old 站、firecrawl 三条路都失败，见 §4）；本轮改用 HF discussions、GitHub issue 与官方文档里的作者实测（§9），论坛口径仍缺 |

---

## 6. Hugging Face 官方博客：SDXL LoRA 进阶脚本（官方文档 + 作者实测）

- `https://raw.githubusercontent.com/huggingface/blog/8000ba4622f89f41f52151e67b666a521061a2f6/sdxl_lora_advanced_script.md` — 访问成功
- 作者：Linoy Tsaban、multimodalart（HF 官方博客），commit `8000ba4`
- 可信度：**官方文档 + 作者实测**——教程里的数字来自作者自己跑的对照，不是纯推荐值
- 采用的关键数字：
  - 人脸 LoRA：6–10 张；步数 **120×张数**（6 张→720）；rank **32** 默认、rank 64 皮肤「磨皮」；UNet lr `1e-4`、TE lr `5e-5`（普通微调）/`3e-4`（pivotal tuning）；batch 4、1024、bf16；VAE 用 `madebyollin/sdxl-vae-fp16-fix`
  - 风格 LoRA 27 张：同参数 2500 步**明显过拟合**；1000 步（batch 1–2）的 v4–v6 最平衡
  - `min_snr_gamma` 推荐 5.0，但**小数据集上效果不明显**
  - 数据集：质量>数量；训脸时不要出现别的脸；特写+全身都要；避免人物太远；背景/服装/姿势/光照多样
  - prior preservation 用**真实** class 图优于模型生成图

## 7. kohya-ss/sd-scripts 文档（官方文档）

- `https://github.com/kohya-ss/sd-scripts` README（v0.13.1，2026-11-23）— 访问成功
- `docs/sdxl_train_network.md` — 访问成功
- `docs/moss_image_train_network.md` — 访问成功
- 可信度：**官方文档**（训练框架作者文档，参数即框架在用的默认/推荐值）
- SDXL 采用的关键口径：示例 `--network_dim=32 --network_alpha=16 --unet_lr=1e-4 --text_encoder_lr1/2=1e-5`（TE lr 比 UNet 小）；`--no_half_vae`「SDXL VAE 在 float16 下不稳定，fp16 时必须开」；`cache_text_encoder_outputs` 需配 `network_train_unet_only`
- MOSS 采用的关键口径：
  - 预训练 timestep = **logit-normal(mean −1.09, std 0.95)**、1000 步网格无 shift、截断 0.95、混 1% 均匀；复现用 `truncated_logit_normal`，简洁近似 `shift` + `discrete_flow_shift 3.0`；**shift 5.0 只在推理**；默认 `sigmoid` 中心 t=500 与预训练差很多
  - ~1MP/batch1/bf16/dim16：全量重算 **13.5GB/1.29 s**、不重算 55GB、ConvRot INT8+compile **10.0GB/0.84 s**；RTX 4090 bf16 14.3GB/1.88s vs INT8 10.0GB/1.00s（1.9×）
  - 权重 bf16：DiT ≈10.2GB、TE ≈15.9GB；VAE 恒 fp32、TE 恒 bf16
  - 官方训练 caption 是「一图多候选、每条随机选一」；`--moss_max_token_length` 上限 4096
  - TE LoRA 在 kohya 里可训（Qwen3_5Attention/GatedDeltaNet/MLP 白名单），但官方仍建议冻结
  - ConvRot INT8 与 bf16 的 LoRA loss 曲线一致；不支持 fp8，`fp16` 不推荐

---

## 8. Hugging Face 模型卡（官方，一手）

> 抓取方式：网页抓取通道（`web_fetch` / modsearch）对 HF 全线 `TypeError: fetch failed`；改用 Python `urllib.request` + `User-Agent` 头直连成功（2026-11-27）。文件由主管鱼鱼取回，落在 `_src/`，指纹用本机 `sha256` 复测。

### 8.1 `OpenMOSS-Team/MOSS-Image-Base`（README 全文已取得）

- URL：`https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base` — 访问日期 2026-11-27
- 本地副本：`_src/hf-moss-image-base-README.md`（9,026 B / 197 行 / sha256 `f9a207bae9ce68953a4b63a846a4bce858a498b354304e11180c89272c9fcfa1`）
- 元数据副本：`_src/hf-moss-image-base-api.json`（3,969 B / 1 行紧凑 JSON / sha256 `e27781fab576b0267ccf89a89965d176825ae95bf65081c69b4a1fd875e8280a`）
- 可信度：**官方**（发布方 model card）
- 元数据：`gated: false`、`license: apache-2.0`、`language: en, zh`、`pipeline_tag: text-to-image`、`diffusers:MOSSImagePipeline`、8,535 downloads、714 likes、37 个文件、`arxiv: 2610.23374`
- 采用的口径：
  - 输入三种：**自然语言 prompt / 标签序列 / JSON layout**
  - 依赖：Python 3.10+、PyTorch 2.9+、`diffusers>=0.38`、`transformers>=5.3`、`accelerate>=1.10`、`safetensors>=0.6`、`Pillow>=10`
  - dtype 原文：「The transformer and text encoder use bfloat16; the VAE and initial noise use float32.」→ `vae_dtype: float32` 的官方依据
  - Sampling 表：Flow-matching Euler / Steps 50 / CFG 3 / Shift 5 / Layer skipping = negative branch layer index 22（zero-based）/ Negative prompt 空；「default guidance uses two transformer forwards per step」
  - Python API 尺寸必须是 **16 的正整数倍**
  - Layout：`summary.caption` + `items`，`bbox_format: yxyx`、`bbox_coord_space: abs_centralized`；draft → 编译用 `skills/moss-image-prompt-enhancer/scripts/compile_prompt.py`；`json.dumps(layout, ensure_ascii=False)`
  - Citation：《MOSS-Image Technical Report》MOSI. AI 2026，`arXiv:2610.23374`（**比社区指南更硬，遇分歧以论文为准**）；权重与打包推理代码 Apache 2.0

### 8.2 `krea/Krea-2-Raw`（仅元数据取得，正文 HTTP 401）

- URL：`https://huggingface.co/krea/Krea-2-Raw` — 访问日期 2026-11-27
- 元数据副本：`_src/hf-krea2-raw-api.json`（37,877 B / 1 行紧凑 JSON / sha256 `9c493dab5566b8afe8b5e45f9a6463727299367b53614d7a4446122ad6aa9252`）
- 可信度：元数据 **官方**；model card 正文**未取得**（`gated: auto`，README 返回 **HTTP 401**，需登录并接受 License，本轮未绕过）
- 取得的元数据：105,924 downloads、1,482 likes、`diffusers:Krea2Pipeline`、`license: other`、55 个文件、`gated: auto`
- 仍未取得：参数量、权重文件名、`config.json`、官方推荐采样配方原文 → `docs/models/krea2.md` 相应条目保持 `UNVERIFIED`

---

## 9. 社区实战与竞品（2026-11-27 一手抓取）

> 抓取方式：GitHub raw / GitHub API 均可用；HF 社区讨论用 `https://huggingface.co/api/models/<repo>/discussions` 与 `/discussions/<n>`（`urllib` + User-Agent）。本地副本在 `_src/`，指纹为本机 `sha256` 前 16 位。
> 可信度四档沿用本文件开头：**一手实测**（作者跑过并给数字/日志）/ **官方文档** / **第三方官方** / **二手汇编或用户帖**（未复现、单帖、无方法学）。

### 9.1 Krea 2 社区证据

| 来源 | 内容 | 可信度 | 本地副本与指纹 |
|---|---|---|---|
| `https://github.com/chengyansen-ai/krea2-lora-training`（MIT，README 与 `references/`）— 访问 2026-11-27 | 社区校准参数（`linear`+`flowmatch`、32/32、1e-4、2500 步、qfloat8、DOP、桶 512/768/1024）；步数实测区间；踩坑全集（头大身小、采样器漂脸、霓虹染发、概念泄漏、DOP 双加载、32767 环境变量） | **二手汇编**（作者称做过多轮 cysgirl 实跑，未公开日志；引用的 r/StableDiffusion 帖本机不可达） | `_src/krea2-lora-training-README.md` 11,495 B/198 行/`b2df8c87cf82363f…`；`_src/krea2skill-references-krea2-community-best-practices.md` 10,899 B/202 行/`0a0fb8887bb72afc…`；`_src/krea2skill-references-踩坑全集.md` 8,767 B/170 行/`7f14a469ff0a8fe6…` |
| `https://github.com/ostris/ai-toolkit/issues/1057`（2026-09-21 开） | 「Krea 2 LoRA + layer offloading = LoRA 无效果」，INT8、100% offloading，作者称可复现 | **用户帖**（单用户，未复现） | 通过 GitHub API 读取 |
| `https://github.com/ostris/ai-toolkit/issues/1070`（2026-10-01 开） | feature request：训练用 Raw、采样用 Turbo | **用户帖**（说明该需求真实存在且未实现） | 同上 |
| `https://github.com/ostris/ai-toolkit/issues/968`（2026-07-21） | 启动报 `No module named 'dotenv'` 的环境问题 | 用户帖（与训练无关） | 同上 |
| `https://huggingface.co/api/models/krea/Krea-2-Raw/discussions` 及 `/10` `/1` — 访问 2026-11-27 | 6 条讨论：<4GB 显存跑（第三方 WeeLLM）、ComfyUI 工作流、bbox 控制询问、1-click 安装 | 用户帖；无训练参数信息 | `_src/hf-krea2-discussions.json` 3,783 B/`921696f747899719…` |
| kohya `README.md`（2026-11-27 抓取） | 全文 **0 处** 提及 Krea → kohya 不支持 Krea2 | 一手官方文档（缺席本身是证据） | `_src/kohya-README.md` 33,691 B |

### 9.2 MOSS-Image 社区证据

| 来源 | 内容 | 可信度 | 本地副本与指纹 |
|---|---|---|---|
| `https://github.com/sorryhyun/anima_lora`（commit `780028ac994467aef1ecb94bffc6c8760da5bd0a`，2026-10-18）`project/moss_lora/plan.md`、`docs/moss-findings/*` | 16GB 卡 LoRA 方案（6 线性层×42、fp32 master+bf16、缓存 TE/latent、DiT 常驻 9.7 GiB、目标 <1h/<15GiB）；caption 长度步时代价（+24% linear / +53% attention）；chat template 即 attention sink（0.24–0.34 图像 query 质量）；Base vs SFT 观感；2MP fp32 VAE 解码需先卸 DiT | **一手实测**（作者给了探针、数字、环境；非官方、单作者） | `_src/anima-project-moss_lora-plan.md` 10,412 B/169 行/`67923657edffba7f…`；`_src/anima-docs-moss-findings-attention_sink_chat_template.md` 7,594 B/130 行/`59f2afc0b9590f55…` |
| `https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/discussions/6` | i2i 上色 LoRA：`qwen(image, caption)` + `vae(downscale 2)`，5070 Ti 约 1 小时 | 用户帖（附权重/工作流/训练代码链接） | `_src/moss-6.json` 4,955 B/`07e023560d8df72f…` |
| `…/discussions/9` | 「BLUR everywhere」：背景糊、主体身体也糊；同 prompt 在 Krea2/Z-Image/Chroma/Flux 正常 | 用户帖（机制为发帖人猜测） | `_src/moss-9.json` 3,036 B/`133ebf3fc5bdada3…` |
| `…/discussions/12` | 「All the problems」：背景糊 + 风格 LoRA 不像 + 形体崩 | 用户帖（无配置） | `_src/moss-12.json` 3,388 B/`249a1c020f53a2e1…` |
| `…/discussions/14` | 1024² 无布局框，footer 小字网格逐条引号写出，**全部拼对**、栅格对齐 | 用户帖（带图与 recipe） | `_src/moss-14.json` 3,897 B/`7fc9ffa71dfdc1ff…` |
| `https://huggingface.co/api/models/OpenMOSS-Team/MOSS-Image-Base/discussions` — 访问 2026-11-27 | 共 16 条讨论（含 2 条 PR），其余为安装/Colab/口味类 | 用户帖 | `_src/hf-moss-discussions.json` 10,452 B/`bc3e1b122950e184…` |

### 9.3 竞品（README 一手，2026-11-27）

| 来源 | 内容 | 可信度 | 本地副本与指纹 |
|---|---|---|---|
| `https://github.com/EnragedAntelope/dataset-deviser`（MIT，9★） | 三种数据集类型的 caption 语义；Prose/Danbooru/e621 三种 caption 风格按目标模型切；规范标签走 WD/Z3D tagger 而非 VLM；tag options（前后缀/丢弃/rating/下划线/阈值/跳过已打标）；health check 含「几乎每张都出现的标签」→ 丢弃列表；失败可断点续跑 | **官方 README**（作者自述，未复现其流程） | `_src/deviser-README.md` 20,709 B/325 行/`b319ea577705875a…` |
| `https://github.com/jhc13/taggui`（1368★） | 键盘优先逐图打标、N/M 进度、用户词表自动补全、SD token 计数、批量标签操作、列表过滤、内置多 VLM 打标 | **官方 README** | `_src/taggui-README.md` 10,792 B/251 行/`8b2555e4f8644d95…` |
| `https://github.com/fpgaminer/joycaption`（1305★） | 每种打标模式的固定 prompt 原文与稳定性说明；`Straightforward` 模板的硬约束（不写画面没有的/不写分辨率/文字原样引号/提水印）；界面回显所用 prompt；bf16 约 17GB | **官方 README** | `_src/joycaption-README.md` 20,508 B/278 行/`87fc98909da0a1b8…` |
| dataset-deviser issue 区 | 0 个 open issue（2026-11-27 GitHub API 查） | 事实（无可读的用户反馈） | — |

### 9.4 结论边界

- 上述社区证据**没有一条**是「带完整配置 + 对照样图」的一手复现，除 `anima_lora` 的显存/成本测量外，其余只用于决定「先测什么、先怀疑什么」。
- 竞品结论只取 README 原文；`dataset-deviser` 的观感类说法（如「同一套衣服每张都出现→服装被学成角色」）在本仓库链条中仍是二手转述，见 `_competitor-gaps.md` §5。
- Reddit 与 Krea 官方 model card 的缺口见 §4、§8.2。

---

## 10. 训练噪声采样与训练底座（2026-11-27 新增）

### 10.1 MOSS 技术报告（一手，官方）

| 项 | 内容 | 可信度 |
|---|---|---|
| URL | `https://arxiv.org/html/2610.23374v1`（HTML 全文；`_src/arxiv-2610.23374-html.html` 433,863 B） | **官方论文** |
| 访问 | 2026-11-27，`urllib` + UA，HTTP 200 | |
| 关键口径 | Stage 3 的噪声采样分四段：1–140,000 步 = grid shift 3 + logit-normal(0,1)；140,001–207,000 = shift 1 + (−1.45,1.05)；207,001–274,000 = shift 1 + (−1.45,1.05) + cutoff 0.95；末期 274,001–310,000 = shift 1 + **截断 logit-normal（μ=−1.09, s=0.95）、cutoff 0.95、掺 1% 均匀**。`Z~N(μ,s²)`，`S=1−sigmoid(Z)`，`σ_j=j/T`，`p_j=(1−ρ)·[F(aj/T)−F(a(j−1)/T)]/F(a)+ρ/T`。**loss 不按 σ/SNR 加权**，文本 token 不参与预测 loss | 原文 Table 2 + 式 (4) |
| 用途 | 决定 `timestep_type` 的判据（σ 分布；换算约定 σ=t/T、t=σ·1000、σ=1 纯噪声），见 `moss_image.md` ③ | |

### 10.2 MOSS 官方仓库 README / USERGUIDE（一手，官方）

| 项 | 内容 | 可信度 |
|---|---|---|
| URL | `https://raw.githubusercontent.com/OpenMOSS/MOSS-Image/main/README.md`（17,403 B）、`.../USERGUIDE.md`（10,183 B） | **官方文档** |
| 访问 | 2026-11-27，HTTP 200 | |
| 底座口径 | `USERGUIDE.md:5`：「As a base model, MOSS-Image-Base has not undergone any post-training or distillation of inference parameters such as CFG… to make it easier for the community to train LoRAs.」；`:126-132`「If you post-train MOSS-Image-Base yourself, keep the text encoder frozen.」；`README.md:143`：SFT =「Fine-tuned from MOSS-Image-Base on curated data」；`README.md:47,67`：KohakuMossTrainer / DiffSynth-Studio 对 **Base 与 SFT 都**支持 LoRA | |

### 10.3 Krea 2 官方仓库（一手，官方）

| 项 | 内容 | 可信度 |
|---|---|---|
| URL | `https://raw.githubusercontent.com/krea-ai/krea-2/main/README.md`（5,478 B） | **官方仓库**（推理代码 + 说明，不是训练仓库） |
| 访问 | 2026-11-27，HTTP 200 | |
| 关键口径 | `:22/:24/:110`「**TRAIN on Raw and RUN on Turbo**」（Raw 非蒸馏、用于 fine-tune / post-train / LoRA）；`:46` 训练到 1k 分辨率；`:70-72` 推理 `--y1 0.5`（min 分辨率 mu）、`--y2 1.15`（max）、`--mu` 可固定（turbo 推荐 1.15）。**全文没有 LoRA 训练的 `timestep_type` 指导**，训练器外链 `fal.ai/krea-2-trainer`（未访问） | |

---

## 关键论断 → 来源索引

| 论断 | 来源 |
|---|---|
| 未知键会被静默忽略，只有 `job/config/name` 被校验 | `toolkit/config.py:38-51` + `config_modules.py` 全 `kwargs.get()` |
| `train.noise_scheduler: flowmatch` 是语义开关 | `BaseSDTrainProcess.py:1173-1210,1752-1755` |
| krea2/moss 上 `train_text_encoder: true` 是空操作 | `toolkit/lora_special.py:284,678-680`、`BaseSDTrainProcess.py:1931-1932` |
| `text_encoder_lr` 写了不读 | `BaseSDTrainProcess.py:2010-2021,2130-2140`（全仓 `grep` 0 命中） |
| SNR 真正生效在 SDTrainer | `extensions_built_in/sd_trainer/SDTrainer.py:1059-1070` |
| SDXL VAE 无 fp32 保护 | `toolkit/stable_diffusion_model.py:410-447`、`config_modules.py:690` |
| MOSS TE 必须冻结 | `OpenMOSS/MOSS-Image@main USERGUIDE.md` + `lora_special.py:284` |
| MOSS 布局 JSON 字段 | `USERGUIDE.md`「Layout Prompts」 |
| MOSS 推理 50 步 / CFG 3.0 / shift 5.0 / Euler / SLG 22（官方默认表） | HF `MOSS-Image-Base` README「Sampling」（§8.1）+ `USERGUIDE.md` |
| MOSS VAE 用 fp32（作者原文） | HF `MOSS-Image-Base` README「Quick Start」（§8.1） |
| MOSS 公开口径的最终依据是技术报告 | `arXiv:2610.23374`（§8.1） |
| Krea2 存在且体量（105,924 downloads / license other / gated auto） | HF `krea/Krea-2-Raw` api.json（§8.2） |
| Krea2 出图验收 yaml `guidance_scale: 5.5`（生效 4.5） | PR #15（`3e3797e2`）+ `krea2.py:455`；仍 UNVERIFIED |
| Krea2 训练用 Raw、Turbo 只验收 | `README.md:23` + SGLang cookbook |
| Krea2 DiT bf16 ≈24 GB | SGLang cookbook |
| SDXL 社区参数（rank 32、1e-4、120×张数） | HF 官方博客（作者实测） |
| SDXL VAE 需 fp32 | kohya `docs/sdxl_train_network.md`（`--no_half_vae`） |
| MOSS 预训练 timestep：末期 shift=1 网格 + 截断 logit-normal（μ=−1.09, s=0.95），shift 5.0 仅推理 | `arXiv:2610.23374` Table 2（§10.1，一手）+ kohya `docs/moss_image_train_network.md`（复刻口径） |
| MOSS 训 LoRA 用 Base（官方原话）、TE 冻结 | `USERGUIDE.md:5,126-132`（§10.2） |
| Krea2 官方只给「Raw 训 / Turbo 跑」与推理 mu（0.5/1.15），无训练 timestep 口径 | `krea-ai/krea-2 README`（§10.3） |
| `timestep_type` 判据：论文末期 σ 中位 0.741（σ=t/T 约定）；ai-toolkit `shift`(5.0) 0.833、`linear` 0.50 → `shift` 更接近，按 `moss_image.py:344` 赋值路径自算（该路径已核实） | 本仓库自算（论文式 (4) 采样 4M 次），§10.1 + `moss_image.md` ③ |
| MOSS 显存与 ConvRot INT8 训练可用性 | kohya README + `docs/moss_image_train_network.md` |
| Reddit 社区口径缺失 | 本文件 §4（三条路径均访问失败） |
| Krea2 训练用 Raw、采样器/头身比/霓虹染发/概念泄漏等失败模式 | 本文件 §9.1（二手汇编 + 单条用户帖；非一手复现） |
| Krea2「INT8 + layer offloading 同时开 → LoRA 无效」 | ai-toolkit issue #1057（§9.1，用户帖） |
| MOSS 16GB LoRA 方案与 caption 长度步时代价 | `anima_lora` `project/moss_lora/plan.md`（§9.2，一手实测） |
| MOSS 背景/主体发糊、文字渲染准确 | HF discussions #9/#12/#14（§9.2，用户帖） |
| MOSS chat template 是 attention sink | `anima_lora` `docs/moss-findings/attention_sink_chat_template.md`（§9.2，一手实测） |
| 竞品 caption 风格矩阵、规范标签靠 tagger、健康检查接丢弃列表 | dataset-deviser README（§9.3）+ `_competitor-gaps.md` |
| SD token 计数、用户词表补全 | taggui README（§9.3）+ `_competitor-gaps.md` G5 |
| 打标模式的固定 prompt 与 Straightforward 硬约束 | joycaption README（§9.3）+ `_competitor-gaps.md` G6/G7 |
