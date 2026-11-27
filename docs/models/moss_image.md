# MOSS-Image（OpenMOSS-Team/MOSS-Image-Base）训练规格

> 证据口径：源码引用一律写 `ai-toolkit@73c84b0 <路径>:<行号>`（仓库 ostris/ai-toolkit，本地 `_src/ai-toolkit`，抓取日 2026-11-27，`VERSION = "0.13.43"` 见 `version.py:1`；无 .git，commit 由 GitHub API 查得）。官方引用写 `OpenMOSS/MOSS-Image@main <文件>`。`docs/ARCHITECTURE.md:<行号>` 指本项目接口契约。标 `UNVERIFIED` 的条目**不许**被代码当默认值用。
>
> 适用范围：`TargetModel.MOSS_IMAGE` → `build_config()` 产出的 `job: extension / process[type=sd_trainer]` YAML。
>
> **先读这条**：MOSS-Image 是「单流 42 层 DiT + Qwen3.5-9B 文本 + FLUX.2 VAE」，5B 量级，flow matching。官方 `USERGUIDE.md:5` 明确**Base 就是给社区训 LoRA 的**（未做 CFG 等推理后训练/蒸馏），`:126-132` 明确**训练时 TE 必须冻结**。注意调度器：**技术报告 Stage 3 末期是 shift=1 网格 + 截断 logit-normal（μ=−1.09, s=0.95, cutoff 0.95, 掺 1% 均匀），shift 5.0 只在推理时加**（`arXiv:2610.23374` Table 2，2026-11-27 读全文）；ai-toolkit 没有 logit-normal 分支，`shift` 又固定 5.0、且会被**应用两次**（见「训练调度器」行）→ 训练侧 `timestep_type` 只能取近似：**默认 `linear`（网格与预训练一致、密度不同），`shift` 降为可选**（判据与 σ↔t 换算约定见 ③）。文本长度上限 4096 token，是全项目最长的一档。

---

## ① 模型是什么

| 项 | 事实 | 出处 |
|---|---|---|
| 训练底模 | `OpenMOSS-Team/MOSS-Image-Base`（`SFT` 是 SFT 版，出图/验收用）。官方原话：「Base 没有做 CFG 等推理参数的后训练或蒸馏，目的是完整保留训练信息、让社区更容易训 LoRA」 | `OpenMOSS/MOSS-Image@main README.md`、`USERGUIDE.md`（2026-11-27 抓取）；`README.md:23`（本项目三模型表） |
| 规模 | **5B backbone**；42-layer diffusion transformer + Qwen3.5-9B TE + FLUX.2 VAE | `OpenMOSS/MOSS-Image@main README.md` |
| 推理默认（非训练） | 注册表 moss_image → `OpenMOSS-Team/MOSS-Image-SFT`，30 步 / guidance 3.0 / `convrot8` 量化 | `toolkit/models/registry.py:59` |
| 变体 | `moss_image:base`（Base + `uncond_skip_layers:[22]`）、`moss_image:edit`、`moss_image:base_edit` | `registry.py:60-62` |
| 加载器 | `MossImageModel(BaseModel)`，`arch = "moss_image"`，`latent_space_version="moss_image"` | `extensions_built_in/diffusion_models/moss_image/moss_image.py:159-169` |
| 主网络 | 单流 DiT `MossImageTransformer`：dim 3072、FFN 8192、24 头、**42 层**、text_dim 4096、freq_dim 256、**128 latent 通道**、rope_base 10000；image token 在前、text token 在后，42 个 block 共享一个 AdaLN；text 先过 2-block refiner | `src/transformer.py:32-42` |
| 文本编码器 | `Qwen/Qwen3.5-9B`；非 edit 用 `Qwen35TextOnlyEncoder`，edit 用 `Qwen35VLEncoder`；加载后 `eval()` + `requires_grad_(False)` | `moss_image.py:246-...`、`src/checkpoints.py:44-49` |
| TE 最大长度 | `model_kwargs.max_text_length`，默认 **4096** token | `moss_image.py:70,183` |
| VAE | FLUX.2 系 `AutoEncoder`（flux2_kl），f8 + **2×2 packing** → 桶对齐量 **16**；默认权重 `vae/diffusion_pytorch_model.safetensors`，可用 `vae_path` 覆盖。官方 README 原文：「The transformer and text encoder use bfloat16; **the VAE and initial noise use float32**. The bundled VAE uses the FLUX.2 small decoder.」 | `moss_image.py:305-320,207-209`、`src/checkpoints.py:44-49`、HF `OpenMOSS-Team/MOSS-Image-Base` README（2026-11-27） |
| 预测目标 | **flow matching**（`is_flow_matching=True`），预测 velocity | `moss_image.py:166` |
| 训练调度器 | `get_train_scheduler()` → `CustomFlowMatchEulerDiscreteScheduler(shift=5.0, num_train_timesteps=1000, use_dynamic_shifting=False)`；`load_model()` 内 `:344` 会**再赋值一次**同一对象，覆盖构造时传入的调度器。**`shift` 会被应用两次**：diffusers 的 `__init__` 在 `use_dynamic_shifting=False` 时先 shift 一次（`sigma_min` 0.001→0.00498，t≈4.98），`set_train_timesteps` 的 shift 分支（`custom_flowmatch_sampler.py:172`）再 shift 一次 → 实际训练 σ 中位 **0.9619**（只算一次是 0.833）。官方 `src/pipeline.py:27` 的 `training_shift: float = 1.0` 全仓库无调用点传过非默认值，即厂商训练网格是 **shift=1**（`:40` 用 `grid(total, training_shift)` 造网格、`:44-49` 把采样网格吸附上去） | `moss_image.py:72-75,202-205,344`、`src/pipeline.py:17,24-49` |
| 采样配方（官方） | **Euler + shift 5.0 + steps ≥50 + CFG 3.0（3.0–6.0）**；SLG 跳过层建议 **22**（22/24/26 均可）；关掉 SLG 更真实但细节会塌。官方 README 的 Sampling 表另注明：**默认 guidance 每步要跑两次 transformer forward**（条件 + 负条件分支），`uncond_skip_layers=[]` 可关 SLG 换单次 forward；`negative_prompt` 缺省为空 | `OpenMOSS/MOSS-Image@main USERGUIDE.md`「Inference Parameters」；HF `OpenMOSS-Team/MOSS-Image-Base` README「Sampling」 |
| 采样默认（源码） | `sample_shift` 默认 `DEFAULT_SHIFT=5.0`，采样时 `shift=self.sample_shift` | `moss_image.py:184,406`、`src/pipeline.py:17` |
| 无分类器引导跳过层 | `uncond_skip_layers` 默认 `(22,)`（推理期跳过第 22 层求 unconditional） | `src/pipeline.py:20`、`moss_image.py:183-195` |
| LoRA 目标 | `target_lora_modules = ["MossImageTransformer"]` | `moss_image.py:168` |
| 发布 / 许可 | HF 模型卡 `license: apache-2.0`、`language: en, zh`、`pipeline_tag: text-to-image`、未 gated；权重与打包推理代码 Apache 2.0 | HF `OpenMOSS-Team/MOSS-Image-Base` README（2026-11-27）、`hf-moss-image-base-api.json` |
| 运行依赖 | Python 3.10+、PyTorch 2.9+、`diffusers>=0.38`、`transformers>=5.3`、`accelerate>=1.10`、`safetensors>=0.6`、`Pillow>=10`；官方示例 `custom_pipeline="pipeline_moss_image"`、`trust_remote_code=True`、`torch_dtype={"default": bfloat16, "vae": float32}`，并重建 scheduler（`type(pipe.scheduler).from_config(pipe.scheduler.config)`） | HF `OpenMOSS-Team/MOSS-Image-Base` README「Quick Start」 |
| 技术报告 | 《MOSS-Image Technical Report》，MOSI. AI，2026，**`arXiv:2610.23374`**（比社区指南更硬的一手口径，遇分歧以论文为准） | HF 模型卡 Citation 段、`hf-moss-image-base-api.json` 的 `arxiv` 字段 |

**TEXT_LAYOUT 这一档的边界（已核实，不再是传闻）**：布局 JSON 是**官方正式格式**，`OpenMOSS/MOSS-Image@main USERGUIDE.md` 给了完整字段：

```json
{
  "summary": {"meta": ["poster"], "canvas": "single", "caption": "总体场景与风格描述"},
  "items": [
    {"label": "text", "caption": "顶部居中、深蓝大写标题", "text": "EXPLORE", "bbox": [-416, -384, -288, 384]},
    {"label": "object", "caption": "下方居中的蓝色扁平山形插画", "bbox": [-192, -288, 384, 288]}
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}
```

- `bbox_format: "yxyx"` = `[y_min, x_min, y_max, x_max]`。
- `bbox_coord_space: "abs_centralized"` = 以画布中心为原点的**绝对像素**，x 向右、y 向下；1024×1024 画布范围 −512..+512；换算方式是从左上原点坐标各减 `W/2`、`H/2`。
- 坐标必须与本次请求的宽高一致；换尺寸**不会**自动缩放框，必须重算。
- 官方另有「可编辑 draft」格式（左上原点 `abs`、`xyxy`、必须写 width/height），用仓库 `skills/moss-image-prompt-enhancer/scripts/compile_prompt.py` 编译成上面的 model 格式。

但仍要记住：**ai-toolkit 侧零解析**——扩展里没有任何 bbox/layout/JSON 代码（`grep -rn "bbox\|layout"` 只命中「权重文件 layout」注释），caption 原样喂给 TE。所以格式对不对，代码无法替你校验，产品端只能按官方 schema 生成、靠实测验收（⑤）。

---

## ② ai-toolkit 训练配置字段表

骨架同 SDXL/Krea2（`job: extension / config.name / process[type=sd_trainer]`）。

```yaml
job: extension
config:
  name: "<job 名>"
  process:
    - type: sd_trainer
      training_folder: "<输出根目录>"
      device: cuda:0
      trigger_word: "<触发词>"
      network: { type: lora, linear: 32, linear_alpha: 32 }
      save:    { dtype: float16, save_every: 250, max_step_saves_to_keep: 4, save_format: safetensors }
      datasets:
        - folder_path: "<数据集目录>"
          caption_ext: txt
          resolution: [512, 768, 1024]
          buckets: true
          bucket_tolerance: 64
          caption_dropout_rate: 0.05
          cache_latents_to_disk: true
      train:
        noise_scheduler: flowmatch      # ← 必写，否则 timestep_type 失效
        timestep_type: linear           # 默认：网格与预训练一致（shift=1）、密度不同；判据见 ③
        lr: 1e-4
        steps: 2500
      model:
        name_or_path: OpenMOSS-Team/MOSS-Image-Base
        arch: moss_image                # ← 必写，否则落到 sd1
        dtype: bf16
        vae_dtype: float32              # 官方加载口径：vae 用 fp32
        quantize: true
        qtype: convrot8
        model_kwargs: { max_text_length: 4096 }
      sample: { width: 1024, height: 1024, guidance_scale: 3.0, sample_steps: 50 }
```

### 顶层与 process

| 字段 | 必填 | 默认 | 取值/范围 | 出处 |
|---|---|---|---|---|
| `job` / `config` / `config.name` | 是 | — | 只校验这三个键 | `toolkit/config.py:38-51` |
| `process[].type` | 是 | — | `sd_trainer`（三模型共用） | `extensions_built_in/sd_trainer/__init__.py:8` |
| `process[].training_folder` | 是 | — | 路径 | 示例 `:6` |
| `process[].trigger_word` | 否 | `None` | 注入 caption 并继承给 dataset | `jobs/process/BaseSDTrainProcess.py:135,168-169`、`toolkit/dataloader_mixins.py:2367` |

### `network`

| 字段 | 必填 | 默认 | 取值/范围 | 出处 |
|---|---|---|---|---|
| `type` | 否 | `lora` | `lora`（`linear` 是别名） | `toolkit/config_modules.py:175-233` |
| `linear` / `linear_alpha` | 否 | `4` / `= alpha`(`1.0`) | rank / alpha | 同上 |
| `transformer_only` | 否 | `True` | MOSS 是 transformer，保持默认 | 同上 |

### `datasets[]`

| 字段 | 必填 | 默认 | 取值/范围 | 出处 |
|---|---|---|---|---|
| `folder_path` / `dataset_path` | 是（二选一） | — | 路径 | `config_modules.py:933,935` |
| `caption_ext` | 否 | `txt` | 不带点自动补点 | `:953-956` |
| `resolution` | 否 | `512` | int 或列表；列表 → 拆成 N 份 dataset | `:959,1116-1152` |
| `buckets` / `bucket_tolerance` | 否 | `True` / `64` | MOSS 对齐量 16，64 是整数倍 | `:961-962`、`moss_image.py:207-209` |
| `caption_dropout_rate` | 否 | `0.0` | 0–1 | `:968` |
| `shuffle_tokens` / `keep_tokens` | 否 | `False` / `0` | **MOSS 建议保持 false**：caption 是自然语言/可含 JSON，按逗号切分会破坏结构 | `:967,969` |
| `token_dropout_rate` | 否 | `0.0` | 0–1 | `:966` |
| `cache_latents_to_disk` | 否 | `False` | bool | `:1018` |
| `num_repeats` | 否 | `1` | int | `:1014` |

`resolution` 语义：面积上限，不是输出边长（`toolkit/buckets.py:17-48`）；小图不放大，只做对齐。官方推理推荐 1M / 1.5M / 2M 总像素（可到 0.25M–4M），宽高必须是 16 的倍数（`USERGUIDE.md`）。

### `train`

| 字段 | 必填 | 默认 | 取值/范围 | 出处 |
|---|---|---|---|---|
| **`noise_scheduler`** | **是（本模型）** | `ddpm` | **必须 `flowmatch`**：只有该值会调 `set_train_timesteps(...)`；其它值走 `set_timesteps()`，`timestep_type` 与 shift 全部不生效 | `config_modules.py:383`、`BaseSDTrainProcess.py:1173-1210` |
| `timestep_type` | 否 | `sigmoid` | `linear`/`sigmoid`/`shift`/`weighted`…；MOSS 默认取 **`linear`**（与预训练 shift=1 网格一致）。`shift` 分支用固定 `self.shift`（=5.0）做 `sigmas = shift*s/(1+(shift-1)*s)`，而 `__init__` 已先 shift 过一次 → **累计两次**（σ 中位 0.9619），不建议当默认；generic `get_sampler` 的 flux 兜底（`sampler.py:163-177`，shift 3.0 + dynamic=True）只在模型类没有 `get_train_scheduler` 时生效 | `config_modules.py:560`、`toolkit/samplers/custom_flowmatch_sampler.py:145-174` |
| `steps` / `lr` / `unet_lr` | 否 | `1000` / `1e-6` / `= lr` | — | `config_modules.py:386-388` |
| `optimizer` | 否 | `adamw` | 未知名抛 `ValueError` | `:393`、`toolkit/optimizer.py:4-119` |
| `lr_scheduler` | 否 | `constant` | `cosine/linear/constant/...` | `:395`、`toolkit/scheduler.py:6-57` |
| `batch_size` | 否 | `1` | int | `:399` |
| `gradient_accumulation_steps` | 否 | `1` | 与 `gradient_accumulation` 互斥（同给抛 `ValueError`） | `:461-468` |
| `dtype` | 否 | `fp32` | `fp32/fp16/bf16` | `:401`、`toolkit/train_tools.py:37-49` |
| `train_unet` | 否 | `True` | bool；即训 DiT | `:406` |
| **`train_text_encoder`** | 否 | `False` | **官方要求冻结 TE**（「keep the text encoder frozen. Do not fine-tune it.」）；且设 true 也训不动（Qwen3.5 类名不在 LoRA 目标列表里） | `USERGUIDE.md`「Post-Training」、`config_modules.py:407`、`toolkit/lora_special.py:284,678-680` |
| `content_or_style` | 否 | `balanced` | `content/style/balanced` | `:384-385`、`BaseSDTrainProcess.py:1231-1310` |
| `min_snr_gamma` / `snr_gamma` | 否 | `None` | 起步不设（见 ③） | `:411-412`、`SDTrainer.py:1059-1070` |
| `prompt_dropout_prob` | 否 | `0.0` | 0–1 | `:451` |
| `diff_output_preservation` | 否 | `False` | 需 `trigger_word`；与 `train_text_encoder: true` 冲突 | `:495`、`SDTrainer.py:95-102` |
| `gradient_checkpointing` | 否 | `True` | bool | `:419` |
| `max_grad_norm` | 否 | `1.0` | float | `:423` |
| `ema_config.use_ema` | 否 | `False` | `ema_decay` 默认 `0.999` | `:547-555,835-847` |

### `model`

| 字段 | 必填 | 默认 | 取值/范围 | 出处 |
|---|---|---|---|---|
| `name_or_path` | **是** | 无 → 抛 `ValueError('name_or_path must be specified')` | 训练用 `OpenMOSS-Team/MOSS-Image-Base` | `config_modules.py:675-676` |
| `arch` | **是（本模型）** | 省略 → 按 `is_*` 推，最终 `sd1` | 必须 `moss_image` | `:716,809-831` |
| `dtype` | 否 | `float16` | `fp16/bf16/fp32`；建议 `bf16`（官方加载口径 default=bf16） | `:655`、`USERGUIDE.md` |
| `vae_dtype` | 否 | `= dtype` | 官方 `torch_dtype={"default": bfloat16, "vae": float32}` → **建议显式 `float32`** | `:690`、`USERGUIDE.md` |
| `te_dtype` | 否 | `= dtype` | 同上；Qwen3.5-9B 用 bf16 | `:692` |
| `quantize` / `quantize_te` | 否 | `False` | bool；注册表推理两者都开 | `:695-696`、`registry.py:59` |
| `qtype` / `qtype_te` | 否 | `qfloat8` | MOSS 推理默认 `convrot8`（`CONVROT_QTYPES` 家族） | `:697-698`、`toolkit/util/ostris_quant.py:188-211` |
| `low_vram` | 否 | `False` | bool | `:699` |
| `model_kwargs` | 否 | `{}` | MOSS 认 `max_text_length`、`sample_shift`、`uncond_skip_layers`、`edit`、`kv_cache`；`vae_path` 走 `model.vae_path` | `moss_image.py:183-195,305-320` |

### 两条硬约束

**① `train.noise_scheduler: flowmatch` 是语义开关，不是采样器选择。**
`BaseSDTrainProcess.py:1752-1755` 先取模型类自带的 `get_train_scheduler`（MOSS 有，返回固定 shift=5.0 的 flowmatch 调度器），把它作为 `noise_scheduler=` 传给模型类；`moss_image.py:344`（`load_model()` 内）再赋一次同一对象。`stable_diffusion_model.py:659-660` 的 `self.noise_scheduler = pipe.scheduler` **只在传入为 None 时执行**，训练路径不满足；随后 timestep 准备按**字符串**分支（`:1173-1210`）：`== 'flowmatch'` 才调 `set_train_timesteps(timestep_type, latents, patch_size)`，否则 `set_timesteps(1000)`，`timestep_type` 与 shift 都不参与。

**② `train_text_encoder: true` 是静默空操作，而且官方也要求冻结。**
TE LoRA 的目标列表写死为 `["CLIPAttention","CLIPMLP"]`（`toolkit/lora_special.py:284,678-680`）；MOSS 的 TE 是 Qwen3.5（`Qwen35TextOnlyEncoder`），类名不匹配 → 0 个 TE LoRA，且只打印 skipped 数量不报错。`target_lora_modules` 只覆盖 DiT 侧（`BaseSDTrainProcess.py:1931-1932`、`moss_image.py:168`）。官方 USERGUIDE 的 LoRA 建议同样是「keep the text encoder frozen」——两边一致，这个字段在 MOSS 上没有任何正收益。

---

## ③ 推荐起点参数（第一轮训练）

| 参数 | 起点值 | 为什么 |
|---|---|---|
| `model.name_or_path` | `OpenMOSS-Team/MOSS-Image-Base` | 官方 `USERGUIDE.md:5` 原文：Base 未做 CFG 等推理后训练/蒸馏，「…to make it easier for the community to train LoRAs」；`:126-132` 的 Post-Training 整节只针对 Base 讲 LoRA（并要求冻结 TE）；`README.md:143` 写 SFT =「Fine-tuned from MOSS-Image-Base on curated data」。**ai-toolkit registry 的 `moss_image` 预置指向 SFT 是推理默认，不是训练底座**（`registry.py:59-60`）→ 训练配置必须写 Base，SFT 只用于出图/验收 |
| `model.arch` | `moss_image`（必写） | 不写落到 `sd1`（`config_modules.py:809-831`） |
| `train.noise_scheduler` | `flowmatch`（必写） | 见 ② 硬约束 ① |
| `train.timestep_type` | `linear`（近似，标 `UNVERIFIED`） | 官方配方在技术报告 **`arXiv:2610.23374` Table 2**：Stage 3 末期（274k–310k 步）是 **shift=1 网格 + 截断 logit-normal（μ=−1.09, s=0.95, cutoff 0.95）+ 1% 均匀**，只有最早 140k 步用 shift=3；kohya 文档复刻的是同一分布。**换算约定（不写清就无法复算）**：σ=噪声占比，σ=1 纯噪声、σ=0 干净图，`t=σ·T`、T=1000（论文 `σ_j=j/T`；ai-toolkit `timestep = sigma·NUM_TRAIN_TIMESTEPS`，`src/pipeline.py:141`）。**判据分两层（网格优先于密度）**：官方末期分布 = **σ 网格**（shift=1 的线性网格）+ **网格上的密度**（截断 logit-normal，按式 (4) 采样 4M 次得中位 0.741、IQR 0.601–0.842）。ai-toolkit 的 `timestep_type` 只在某个网格上均匀铺点，于是：`linear` = 线性网格 + 均匀密度（**网格对、密度不对**）；`shift` = shift-5 网格（还被应用两次）+ 均匀密度（**网格错、密度也错**）。网格错是结构性的——模型预训练时见过的 σ 集合就不是那批点；密度错只是采样权重偏。**按网格选，`linear` 胜出**，这与论文末期 shift=1 是同一句事实。按 `moss_image.py:74,202-205,344` 这条实际赋值路径自算的 σ 分位（P10/P25/P50/P75/P90，1000 点网格，可复算）：官方末期 0.456/0.601/0.741/0.842/0.899；`linear` 0.10/0.25/0.50/0.75/0.90；`sigmoid` 0.217/0.337/0.50/0.662/0.783；`shift` **单次** 0.359/0.626/0.834/0.938/0.978、**两次（实际生效）** 0.737/0.893/0.962/0.987/0.996。对官方末期的分位 L1 距离：`shift`（修正双重 shift 后）≈1.04、`linear` 1.04、`sigmoid` 1.04——**三者相同，说明 L1 根本分不出优劣**，这正是要改用「网格」判据的原因（先前的 0.39 是漏算第二次 shift 的结果）。仍不是官方配方：ai-toolkit 无 logit-normal 分支、`shift` 系数写死 5.0 且双次生效（kohya 用 3.0 近似，中位正好 0.75）→ 保留 UNVERIFIED，界面提示 A/B 自测 | `arXiv:2610.23374` Table 2 + 式 (4)（§10.1）；`toolkit/samplers/custom_flowmatch_sampler.py:111-235`；`moss_image.py:74,204`；kohya `docs/moss_image_train_network.md` |
| `network.linear` / `linear_alpha` | `32` / `32` | 契约默认（`docs/ARCHITECTURE.md:100-101`）。MOSS 无二手校准，沿用项目默认 |
| `train.steps` | `2500` | 契约默认（`:99`）。注意 `resolution` 列表会复制数据集，steps 要按公式换算 |
| `train.lr` | `1e-4` | 契约默认（`:102`）；源码默认 `1e-6`（`config_modules.py:387`） |
| `train.batch_size` / `gradient_accumulation_steps` | `1` / `1` | 42 层 3072 宽，24GB 先单批跑通 |
| 显存（社区实测） | bf16 权重 DiT ≈10.2GB + TE ≈15.9GB；~1MP/batch1/bf16/dim16：全量重算 **13.5GB**（1.29 s/step），不重算 55GB；ConvRot INT8 + compile **10.0GB**（0.84 s/step） | kohya 官方文档实测（RTX 4090：bf16 14.3GB/1.88s，INT8 10.0GB/1.00s，快 1.9×）→ 24GB 单卡可跑，但 `cache_text_embeddings` 会把 15.9GB 的 TE 输出压进来，谨慎开（`docs/moss_image_train_network.md`、sd-scripts README） |
| `train.train_text_encoder` | `false` | 官方要求冻结；设 true 也不起作用（硬约束 ②） |
| `model.dtype` / `te_dtype` | `bf16` / `bf16` | 官方加载口径 default=bf16（`USERGUIDE.md`）；9B TE + 3072 宽 DiT，fp16 易溢出 |
| `model.vae_dtype` | **`float32`** | 官方加载写 `{"vae": torch.float32}`，FLUX.2 VAE 低精度会坏；源码默认跟随 `dtype`，必须显式改（`config_modules.py:690`） |
| `model.quantize` / `qtype` | `true` / `convrot8` | 注册表 MOSS 推理就是 `convrot8`（`registry.py:59`）；不量化 24GB 大概率放不下——**训练侧需实测**，标 `UNVERIFIED` |
| `model.model_kwargs.max_text_length` | `4096` | 源码默认即 4096（`moss_image.py:70`）；不要为了省显存压低，长文字渲染是它的卖点 |
| `datasets[].resolution` | `[512, 768, 1024]` | 契约默认；官方推理推荐 1M/1.5M/2M 总像素、宽高 16 的倍数。注意列表会 ×3 复制数据集 |
| `datasets[].bucket_tolerance` | `64` | 对齐量 16 的整数倍（`dataloader_mixins.py:270,315`） |
| `datasets[].caption_dropout_rate` | `0.05` | 示例同值，防 caption 过拟合 |
| `save.save_every` / `max_step_saves_to_keep` | `250` / `4` | 示例同值 |
| `sample.sample_steps` / `guidance_scale` | `50` / `3.0` | 官方：steps **≥50**、CFG **3.0（3.0–6.0）**、shift 5.0、Euler。MOSS **不像 Krea2 那样减 1**（`moss_image.py:403` 直接传 `gen_config.guidance_scale`） |
| `sample.width` / `height` | `1024` / `1024` | 16 的倍数；官方示例用 1456×1456（2.1MP）也可，先与训练分辨率档对齐 |
| `model.model_kwargs.uncond_skip_layers` | `[22]`（仅推理侧） | 官方 SLG 推荐 22（22/24/26 均可），关掉更真实但细节会塌；训练侧未见使用（`src/pipeline.py:20`、`moss_image.py:407`） |

---

### 社区证据（2026-11-27 调研，详见 `_community-evidence.md`）

- **训练可行性有一手实测**：16GB 卡方案——LoRA 只落每个 block 的 6 个线性层（`q k v o ffn_up_gate ffn_down`，42×6=252 个），fp32 master + bf16 GEMM，TE 输出与 latent 全部预缓存（训练时不加载 TE/VAE），DiT 常驻 9.7 GiB，目标「一小时内训出可辨认结果、峰值 <15 GiB」（`sorryhyun/anima_lora` `project/moss_lora/plan.md`）。→ 24GB 跑得动，但必须缓存 + `gradient_checkpointing`。
- **caption 长度是步时杠杆（同一实测）**：单流结构 attention 是 `(N_img+N_txt)²`，每个文本 token 还要付完整 FFN/QKV；1776×1184 + 1.9k token prompt 时线性 +24%、attention +53% FLOPs。→ `max_text_length: 4096` 是上限不是目标。
- **背景/主体发糊**：两个用户独立报告「背景到处糊」「糊连主体身体也有」，同一套 prompt 在 Krea2 / Chroma / Flux 正常；官方未回复，机制未知（HF discussions #9、#12）。
- **风格 LoRA 训不像**：用户报告「style is not identical、形体崩」，未给配置（#12）。
- **文字能刻准**：1024² 无布局框的纯 T2I，footer 小字网格逐条引号写出，**每条都拼对**、栅格对齐（#14，带图与 recipe）。
- **i2i 上色 LoRA**：约 1 小时（5070 Ti），参考图经 Qwen3.5 vision tower 进 TE（#6）。注意这是 **i2i**，不是 T2I 角色 LoRA；作者自己的探测也表明图像条件只带回内容与粗布局，**不是像素对齐**。
- **Base 更适合当 LoRA 底座**（作者 5 条 prompt 的印象，明确说不是测量）；SFT 更完成、偏 gacha 光感。
- **attention sink 在 chat template 上**：10 个模板 token 吸走 0.24–0.34 的图像 query 质量，训好的 LoRA 几乎不动它 → 改 caption 格式/长度会先动这块，不是无关噪声。

## ④ caption 风格与禁忌

**官方口径（优先照做）**：官方给了 recaptioner prompt，输出固定 JSON：

```json
{"caption": {"long": "详细描述，含全部非水印可见文字", "medium": "一段话", "short": "一句话"},
 "watermark": [{"box_2d": [y_min, x_min, y_max, x_max], "label": "水印文字"}],
 "borders": ["none"]}
```

规则（`USERGUIDE.md`「Image Captioning」原文要点）：

- **语言用英文**（官方：英文训练数据最多；中英都支持，但第一批别拿中文赌）。
- `long` 必须把**所有非水印可见文字原样写进去**：不翻译、不改写、不罗马化、不归一化；并描述文字换行/顺序/相对位置/对齐/字号层级/字体风格/颜色/间距。
- 水印**只进 `watermark`**，坐标归一化 0–1000，不得写进任何 caption 字段。
- `borders` 记 `left/right/up/down`；无边框写 `["none"]`，不能和方向混写。
- 官方推荐打标 VLM：Gemma 4 / Qwen 3.5 27B / Qwen 3.5 35B-A3B，**开 thinking**。
- 若用标签序列：顺序 Quality→Score→Meta→Year→Character→Series→Artist→General（例 `best quality, highres, hatsune miku, vocaloid, @foo bar, 1girl, blue hair`）；负向词建议 `nsfw, worst quality, bad anatomy, watermark, jpeg artifact`。
- 换行：直接在 prompt 里写 `\n` 或真实换行。
- 布局：用 ① 的官方 JSON schema（`yxyx` + `abs_centralized` + 中心原点 + 坐标与宽高一致）；**别写猜的坐标**，坐标错比没有坐标更伤。官方给的是「可编辑草稿 → 编译」两步：draft 用左上原点 `abs` + `xyxy` 并显式带宽高，编译成 `abs_centralized` + `yxyx` 的紧凑 JSON（仓库 `skills/moss-image-prompt-enhancer/scripts/compile_prompt.py`）；交付给 Python API 时要 `json.dumps(layout, ensure_ascii=False)`（官方示例口径）。

**本项目落地建议**

- 打标词以「详细自然语言」为主，文字段落按官方要求原样保留；这是三模型里唯一鼓励「写长」的一档（上限 4096 token）。
- **一张图多 caption 候选**：官方训练数据是「原文 caption + 不同长度 recaption」混着，训练时每条随机取一个（kohya 用 `enable_multiline_caption` + 分隔行实现，见 `docs/moss_image_train_network.md`）。落到本产品：`long`/`medium`/`short` 三个字段别只存 `long`，按分隔符合成一个多候选 `.txt` 更接近官方分布。
- 触发词照常注入（`dataloader_mixins.py:2367`、`BaseSDTrainProcess.py:1617-1620`）。
- 写进 styler 的规则：`long` 存成 `.txt`；`watermark`/`borders` 先落地到 sidecar（本产品第一版不进训练 caption）；`TEXT_LAYOUT` 档才生成布局 JSON。

**禁忌（写了会坏事的写法）**

- **别开 `shuffle_tokens`**：按逗号重排会把自然语言句和 JSON 结构拆碎（`config_modules.py:967`）。
- **别把 JSON 和其它描述混着写给同一张图**：要么纯描述、要么官方 schema 的「描述 + 结构化字段」，混排会让 TE 分不清哪部分是指令（推断，`UNVERIFIED`）。
- **别把水印写进 caption**：官方明确要求剔除，写进去等于教模型复刻水印。
- **别改写图内文字**（翻译/简写/拼音化）：官方要求逐字保留，改写会污染文字渲染能力。
- **别写身份细节**（脸型/发型/肤色逐图写）：与触发词抢身份，换装即脸漂（`README.md:21`；社区经验）。
- **别把同一套衣服/道具逐图写全**：会被学成角色的一部分。这条来自竞品 README 的规则（trigger 学身份、caption 写变化），**不是实测结论**；判据用 ⑤ 的去触发词对照。
- **别依赖 4K token 写废话**：长上下文是容量不是建议长度；有效描述比堆词好。

---

## ⑤ 验收怎么测

**A. 配置层（无需 GPU）**

1. `validate_config(cfg, MOSS_IMAGE)` 返回空列表（见下方检查项表）。
2. `yaml.safe_dump(cfg)` 能被 `toolkit.config.get_config()` 读（`toolkit/config.py:38-51`）。
3. 回归：`train.noise_scheduler: ddpm` 必须报错（防 timestep 静默失效）。
4. 回归：`train_text_encoder: true` 必须报错（防空操作 + 违反官方冻结要求）。
5. 回归：`model.model_kwargs.max_text_length` 超出 4096 时警告（`moss_image.py:70` 是硬上限常量）。
6. 回归：`model.vae_dtype` 未显式给 `float32` 时警告（官方口径）。

**`validate_config` 检查项**

| # | 检查 | 判定 | 依据 |
|---|---|---|---|
| 1 | 未知键（`rank`/`alpha`/`text_encoder_lr`） | 报错 | `config_modules.py` 全 `kwargs.get()` |
| 2 | `model.name_or_path` 空 | 报错 | `:675-676` |
| 3 | `model.arch != moss_image` 且无 `is_*` 标志 | 报错（会落 `sd1`） | `:809-831` |
| 4 | `train.noise_scheduler != flowmatch` | 报错 | `BaseSDTrainProcess.py:1173-1210` |
| 5 | `timestep_type` 不在白名单 | 报错 | `custom_flowmatch_sampler.py:127-216` |
| 6 | `train.train_text_encoder: true` | 报错（Qwen3.5 + 官方要求冻结） | `lora_special.py:284,678-680`、`USERGUIDE.md` |
| 7 | `diff_output_preservation: true` 且无 `trigger_word` | 报错 | `SDTrainer.py:95-98` |
| 8 | `save_format` 非 `safetensors/diffusers` | 报错 | `config_modules.py:23-34` |
| 9 | `gradient_accumulation*` 双写 | 报错 | `:466-468` |
| 10 | `datasets[].resolution` 是列表 | 警告：数据集 ×N | `:1116-1152` |
| 11 | `model.qtype` 不在 `get_qtype` 已知家族 | 警告 | `toolkit/util/ostris_quant.py:188-211` |
| 12 | `model.vae_dtype != float32` | 警告（官方 VAE fp32 口径） | `USERGUIDE.md` |

**B. 训练层（首轮跑完看）**

1. 日志 `create LoRA for Text Encoder: 0 modules` 是预期（MOSS TE 无 LoRA 目标）。
2. 样图：**≥50 步 / CFG 3.0 / shift 5.0 / 1024×1024**（官方 `USERGUIDE.md` + `registry.py:59`）；采样侧 `shift=5.0` 由 `sample_shift` 承载（`moss_image.py:406`），与训练侧的 `timestep_type` 是两回事（见 ③）。
3. **三提示法定性**：分布内 / 泛化 / 去触发词对照（社区汇编口径，`UNVERIFIED`；见 `_community-evidence.md`）。
4. **文字验收单独测**：MOSS 的差异点是长文本渲染，样图里要专门放一条「含引号内文字、多行、中英混排」的 prompt，逐字核对字形/换行/位置；只看整体风格测不到卖点。
5. **布局验收（仅 `TEXT_LAYOUT` 档）**：按 ① 官方 schema 生成一张含 2–3 个 item 的布局 prompt，核对元素落位；坐标与请求宽高必须匹配。
6. **步数 sanity**：每 epoch 样本数 = 张数 × `num_repeats` × `len(resolutions)`。
7. **背景/主体发糊单独记一项**：同一 prompt 用**未加 LoRA 的 Base** 也出一张对照。若基线就糊 → 模型侧现象（社区两个独立报告，官方未解释），不能记成「LoRA 训坏」；若只在 LoRA 后出现 → 才是训练问题。
8. **timestep 对照**：固定数据与种子，`timestep_type: linear`（默认）与 `shift` 各跑 600–800 步，比文字渲染（MOSS 的强项）与背景/主体清晰度；若 `shift` 明显更好，说明「网格优先」判据仍有漏层，应回到 `src/pipeline.py:24-49` 与论文 Table 2 重查。
8. **caption 长度成本**：记录平均 caption token 数与 s/step。单流结构下步时随 `(N_img+N_txt)²` 涨，长 caption 导致的变慢先截断 caption 再怀疑配置（一手实测，见社区证据）。

---

## 未查实 / 需要什么才能确认

| # | 缺口 | 现状 | 确认需要 |
|---|---|---|---|
| 1 | 精确参数量与各组件权重体积 | HF model card 已于 2026-11-27 取得（`sources.md` §8.1）：确认 5B backbone、42 层、Qwen3.5-9B、FLUX.2 VAE、dtype 口径、Sampling 默认表、两种 forward 口径；仍缺 `config.json` 的精确参数量和 safetensors 分片大小 | 读 HF `config.json` / safetensors index（`urllib` 可达，无需登录） |
| 2 | 布局 JSON 在**训练** caption 里的正确用法 | 格式已是官方 schema（不再 UNVERIFIED），但 ai-toolkit 不解析；官方未说明训练时该用哪种 caption 形态 | 官方 issue/示例，或训练对照实验 |
| 3 | `timestep_type` 该用哪档 | 官方配方已知（`arXiv:2610.23374` Table 2：末期 shift=1 网格 + 截断 logit-normal μ=−1.09/s=0.95/cutoff 0.95 + 1% 均匀；σ 中位 0.741、IQR 0.601–0.842）。ai-toolkit 无 logit-normal，`shift` 系数写死 5.0 且会被应用两次（σ 中位 0.9619）；修正后按分位 L1，`shift`/`linear`/`sigmoid` 都是 ≈1.04（旧数字 0.39 漏算了第二次 shift）。改按「网格优先」判据 → 默认 `linear`（网格与预训练 shift=1 一致），**仍不是官方配方**；赋值路径已核实：`BaseSDTrainProcess.py:1752-1755` 取类方法 → `:1784` 传入 → `moss_image.py:344` 在 `load_model()` 内再赋一次；`stable_diffusion_model.py:659-660` 只在传入为 None 时用 `pipe.scheduler` | 同数据 600–800 步 `shift` vs `linear` 对照：文字渲染、结构、loss 曲线；以及「背景/主体发糊」是否随 timestep 分布变化 |
| 4 | 24GB 是否需要 `quantize`、`convrot8` 训练可用性 | 注册表只证明推理可用 | 24GB 实跑记峰值显存 + 出图对比 |
| 5 | MOSS 的 `lr=1e-4`/`rank=32`/2500 步 | 契约默认，无任何二手校准 | 对照训练 |
| 6 | `uncond_skip_layers` 在**训练**时是否该设 | 官方推荐 22 是推理口径；训练侧代码未见使用（`src/pipeline.py:20`、`moss_image.py:407`） | 读训练 loss 路径确认是否触发 uncond 分支 |
| 7 | `DatasetKind.TEXT_LAYOUT` 在本产品里的可行性 | 格式明确，但训练侧无解析、无示例 | 等 #2 确认后再定产品是否上这一档 |
| 8 | 社区 MOSS-Image 实战（T2I 角色 LoRA） | 已补：16GB 一手实测方案（`anima_lora`）、两个「背景/主体发糊」用户报告、一个文字渲染正面案例、一个 i2i 上色 LoRA；**仍缺**：公开的 T2I 角色 LoRA 完整配置与对照（步数/lr/rank/数据量），以及「发糊」到底来自模型还是采样参数 | 我们自己的 24GB 对照实跑；或官方对「发糊」的回应；Reddit 不可达（`sources.md` §4） |
| 9 | 官方 recaptioner 的 `long/medium/short` 哪个进训练 caption | 官方只说「可参考这套打标」 | 官方训练脚本/issue，或三档对照 |
