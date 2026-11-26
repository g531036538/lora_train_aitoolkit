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
  - License：pipeline Apache 2.0；技术报告 arXiv 2610.23374（**未访问**）

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
| `https://huggingface.co/krea/Krea-2-Raw` | Krea 2 Raw 官方 model card | **访问失败**：`Error: web fetch failed: TypeError: fetch failed` |
| `https://huggingface.co/krea/Krea-2-Raw/raw/main/README.md` | 同上（raw README） | **访问失败**：`TypeError: fetch failed` |
| `https://www.krea.ai/krea-2-open-source` | Krea 官方开源页 | **访问失败**：`TypeError: fetch failed` |
| `https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base` | MOSS Base 官方 model card | **访问失败**：`TypeError: fetch failed` |

> 影响：Krea 2 的参数量/权重细节、MOSS Base 的 model card 细节，本轮都没有一手来源。文档里已标 `UNVERIFIED`，并写明「确认需要能访问 HF 的网络」。

## 5. 二手转述与未访问来源（只作线索）

| 来源 | 类型 | 可信度 | 本轮处理 |
|---|---|---|---|
| `docs/models/_seed-from-lead.md`（主管手写，60 行） | 内部种子 | **二手转述**，未核实 | 已逐条对照源码/官方文档；核对结果见三份模型文档「未查实」表。该文件不是交付文档，需移出 `docs/models/` |
| `chengyansen-ai/krea2-lora-training`（MIT，约 6★） | 第三方训练仓库 | **二手转述**（种子转述，未直接访问） | 其 Krea2 参数（32/32、1e-4、2500 步、flowmatch、linear、桶 512/768/1024、caption_dropout 0.05、DOP）与源码机制不冲突，但**没有一条是它自己验证过的换算**；只作起点参考 |
| `EnragedAntelope/dataset-deviser` | 竞品仓库 | 未访问（仅种子提及） | 两条复盘（同一套衣服每张都出现→服装被学成角色；参考图道具被复制→道具被学成角色）记为**二手实战教训**，未在本仓库验证 |
| `jhc13/taggui`、`fpgaminer/joycaption` | 竞品仓库 | 未访问 | 竞品分析留待后续轮次 |
| Reddit / Civitai / Discord 社区经验 | 社区 | 传闻 | 本轮未做；`krea2.md`/`moss_image.md` 的「社区实战」条目均待补 |

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
| MOSS 推理 50 步 / CFG 3.0 / shift 5.0 / Euler | `USERGUIDE.md`「Inference Parameters」 |
| Krea2 训练用 Raw、Turbo 只验收 | `README.md:23` + SGLang cookbook |
| Krea2 DiT bf16 ≈24 GB | SGLang cookbook |
