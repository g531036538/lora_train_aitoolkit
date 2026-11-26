# SDXL（stable-diffusion-xl-base-1.0）训练规格

> 证据口径：源码引用一律写 `ai-toolkit@73c84b0 <路径>:<行号>`（仓库 ostris/ai-toolkit，本地 `_src/ai-toolkit`，抓取日 2026-11-27，`VERSION = "0.13.43"` 见 `version.py:1`；无 .git，commit 由 GitHub API 查得）。`docs/ARCHITECTURE.md:<行号>` 指本项目接口契约。标 `UNVERIFIED` 的条目**不许**被代码当默认值用。
>
> 适用范围：`TargetModel.SDXL` → `build_config()` 产出的 `job: extension / process[type=sd_trainer]` YAML。

---

## ① 模型是什么

| 项 | 事实 | 出处 |
|---|---|---|
| 底模 | `stabilityai/stable-diffusion-xl-base-1.0`（diffusers 格式） | `ai-toolkit@73c84b0 toolkit/models/registry.py:50` |
| 加载器 | `StableDiffusionXLPipeline`；`arch=='sdxl'` 时 `is_xl=True` | `toolkit/stable_diffusion_model.py:410`、`toolkit/config_modules.py:791-792` |
| UNet | SDXL `UNet2DConditionModel`（`train_unet` 训练主体） | `toolkit/lora_special.py:281`（`UNET_TARGET_REPLACE_MODULE=["UNet2DConditionModel"]`） |
| 文本编码器 | **双 TE**：CLIP ViT-L + OpenCLIP ViT-bigG；加载后 `requires_grad_(False)` + `eval()`，默认不训 | `toolkit/stable_diffusion_model.py:410-447` |
| TE 开关 | `use_text_encoder_1` / `use_text_encoder_2`，默认都 `True`（指「参与前向」，不是「训练」） | `toolkit/config_modules.py:670-671` |
| VAE | `AutoencoderKL`，f8 下采样；`block_out_channels` 4 层 → 桶对齐量 `2**(4-1)*2 = 16` | `toolkit/stable_diffusion_model.py:364-372` |
| 预测目标 | `prediction_type` 默认 `epsilon`（`is_v_pred` 默认 `False`） | `toolkit/config_modules.py:654`、`jobs/process/BaseSDTrainProcess.py:1756-1769` |
| 原生分辨率 | 1024×1024；推理默认 25 步 / guidance 6.0 | `toolkit/models/registry.py:12`、`:50` |
| Refiner | 可选，默认不加载（`refiner_name_or_path=None`，`refiner_start_at=0.5`） | `toolkit/config_modules.py:657,659` |
| VAE 精度 | `vae_dtype` 默认**跟随 `model.dtype`**（即 `float16`），源码无 fp32 强制、无 NaN 保护 | `toolkit/config_modules.py:690`、`toolkit/stable_diffusion_model.py:188` |

要点：SDXL 是「UNet + 双 CLIP + f8 VAE、epsilon 目标、1024 原生」的老式扩散架构，不是 flow matching。`noise_scheduler` 走 `arch='sd'` 的 DDPM/DDIM 系调度器（`toolkit/sampler.py:120`，SD 配置见 `:31-50`：`scaled_linear` beta 0.00085→0.012、`timestep_spacing=leading`、`steps_offset=0`、1000 训练步）。

---

## ② ai-toolkit 训练配置字段表

生成的是如下骨架（字段名逐字对齐源码；整份 YAML 形状见 `ai-toolkit@73c84b0 config/examples/train_lora_flux_24gb.yaml:1-90`）：

```yaml
job: extension
config:
  name: "<job 名>"
  process:
    - type: sd_trainer
      training_folder: "<输出根目录>"
      device: cuda:0
      trigger_word: "<触发词>"          # 可选
      network: { type: lora, linear: 32, linear_alpha: 32 }
      save:    { dtype: float16, save_every: 250, max_step_saves_to_keep: 4, save_format: safetensors }
      datasets:
        - folder_path: "<数据集目录>"
          caption_ext: txt
          resolution: [512, 768, 1024]   # list → 拆成多个 dataset，见下方警告
          buckets: true
          bucket_tolerance: 64
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
      train:   { ... }                   # 见 train 表
      model:   { ... }                   # 见 model 表
      sample:  { ... }                   # 见 sample 表
```

### 顶层与 process

| 字段 | 必填 | 默认 | 取值/范围 | 出处 |
|---|---|---|---|---|
| `job` | 是 | — | 示例与注册表都用 `extension` | `toolkit/config.py:38-40`；示例 `:2` |
| `config` | 是 | — | 必须存在 | `toolkit/config.py:41-43` |
| `config.name` | 是 | — | 字符串；`[name]` 占位会被替换成它 | `toolkit/config.py:44-51` |
| `process[].type` | 是 | — | `sd_trainer`（SDXL/SD 系走它） | `config/examples/train_lora_flux_24gb.yaml:5`、`extensions_built_in/sd_trainer/` |
| `process[].training_folder` | 是 | — | 路径；训练产物落这里 | 示例 `:6` |
| `process[].device` | 否 | `cuda:0`（示例） | `cuda:N` | 示例 `:8` |
| `process[].trigger_word` | 否 | `None` | 字符串；会注入 caption，并**继承给未单独设置的每个 dataset** | `jobs/process/BaseSDTrainProcess.py:135,168-169` |
| `process[].performance_log_every` | 否 | 未设 | int | 示例 `:7`（注释） |

### `network`

| 字段 | 必填 | 默认 | 取值/范围 | 出处 |
|---|---|---|---|---|
| `type` | 否 | `lora` | `lora` / `lokr` 等；`linear` 是 `lora` 的别名 | `toolkit/config_modules.py:175-233` |
| `linear` | 否 | `4` | rank，int | `:175-233` |
| `linear_alpha` | 否 | `= alpha`（`alpha` 默认 `1.0`） | float | `:175-233` |
| `transformer_only` | 否 | `True` | bool；UNet 架构下只影响 transformer 类模型 | `:175-233` |
| `dropout` | 否 | `None` | float 或 `None` | `:175-233` |

注意：字段名是 **`linear` / `linear_alpha`**，不是 `rank` / `alpha`。本项目 `TrainingPlan.rank/alpha`（`docs/ARCHITECTURE.md:100-101`）必须映射成这两个键；写错不会报错，会被静默忽略（见下方「硬约束」）。

### `datasets[]`

| 字段 | 必填 | 默认 | 取值/范围 | 出处 |
|---|---|---|---|---|
| `folder_path` 或 `dataset_path` | 是（二选一） | — | 路径 | `toolkit/config_modules.py:933,935` |
| `caption_ext` | 否 | `txt` | 不带点会自动补点 | `:953-956` |
| `trigger_word` | 否 | `None` | 覆盖 process 级 | `:939` |
| `resolution` | 否 | `512` | **int 或 int 列表**；列表会被拆成 N 个 dataset | `:959`、`:1116-1152` |
| `buckets` | 否 | `True` | bool | `:961` |
| `bucket_tolerance` | 否 | `64` | int，桶宽高对齐量 | `:962`、`toolkit/dataloader_mixins.py:270,315` |
| `caption_dropout_rate` | 否 | `0.0` | 0–1 | `:968` |
| `shuffle_tokens` | 否 | `False` | bool，按逗号切分重排 | `:967` |
| `token_dropout_rate` | 否 | `0.0` | 0–1 | `:966` |
| `keep_tokens` | 否 | `0` | int，shuffle 时保留前 N 个 | `:969` |
| `num_repeats` | 否 | `1` | int | `:1014` |
| `cache_latents_to_disk` | 否 | `False` | bool | `:1018` |
| `random_scale` / `random_crop` / `square_crop` | 否 | `False` | bool | `:957,958,1070` |
| `flip_x` / `flip_y` | 否 | `False` | bool（角色/文字数据不要开） | `:970-971` |
| `num_workers` / `prefetch_factor` / `pin_memory` | 否 | `2` / `2` / `False` | int/int/bool | `:1056,1057,1063` |

**分辨率桶的实际语义（实测推断 + 源码）**：`get_bucket_for_image_size` 以 `max_pixels = resolution²` 为面积上限，按 `divisibility` 取候选（`toolkit/buckets.py:17-48`）；`total < max_pixels` 时 `scaler=1`，**小图不会被放大**，只做对齐。所以 `resolution` 不是「输出边长」而是「面积上限」。

**警告（会算错步数）**：`datasets[].resolution` 给列表时，`preprocess_dataset_raw_config` 会**复制数据集**——每个分辨率一份（`toolkit/config_modules.py:1116-1152`）。20 张图 × `[512,768,1024]` × `num_repeats=1` = 每个 epoch **60 个样本**。`steps` 是按样本数计的，所以：

```
steps ≈ 图片张数 × num_repeats × len(resolutions) × 目标 epochs
```

本项目 `TrainingPlan.steps=2500`（`docs/ARCHITECTURE.md:99`）是按「单次计算」的总步数，不是 epochs；`validate_config` 必须把这条换算关系提示给用户，别让 20 张图默认配 2500 步跑成 40+ epochs。

### `train`（SDXL 实际会读的字段）

| 字段 | 必填 | 默认 | 取值/范围 | 出处 |
|---|---|---|---|---|
| `steps` | 否 | `1000` | int > 0 | `toolkit/config_modules.py:386` |
| `lr` | 否 | `1e-6` | float | `:387` |
| `unet_lr` | 否 | `= lr` | float | `:388` |
| `text_encoder_lr` | — | **无效** | 写了也不读：两条优化器构造路径都写死 `'text_encoder_lr': self.train_config.lr` | `jobs/process/BaseSDTrainProcess.py:2010-2021, 2130-2140` |
| `optimizer` | 否 | `adamw` | `adam/adamw/lion/adagrad/adafactor/prodigy/prodigy8bit/dadaptation/dadaptationlion/adam8bit/adamw8bit/lion8bit/ademamix8bit/automagic1-4/adamconvrot`（未知名抛 `ValueError`） | `toolkit/config_modules.py:393`、`toolkit/optimizer.py:4-119` |
| `optimizer_params` | 否 | `{}` | dict | `:394` |
| `lr_scheduler` | 否 | `constant` | `cosine/cosine_with_restarts/step/linear/constant/constant_with_warmup` + diffusers 调度器名 | `:395`、`toolkit/scheduler.py:6-57` |
| `lr_scheduler_params` | 否 | `{}` | dict；会自动补 `total_iters=steps` | `:396`、`jobs/process/BaseSDTrainProcess.py:2216-2227` |
| `batch_size` | 否 | `1` | int | `:399` |
| `gradient_accumulation_steps` | 否 | `1` | int；**与 `gradient_accumulation` 互斥**，同时给抛 `ValueError` | `:461,465-468` |
| `dtype` | 否 | `fp32` | `fp32/float16/bf16`（`get_torch_dtype` 接受 fp32/fp16/bf16 及其别名） | `:401`、`toolkit/train_tools.py:37-49` |
| `noise_scheduler` | 否 | `ddpm` | `ddpm/ddim/pndm/lms/k_lms/euler/k_euler/euler_a/dpmsolver/dpmsolver++/dpmsingle/heun/dpm_2/dpm_2_a/lcm/custom_lcm/mean_flow/flowmatch` | `:383`、`toolkit/sampler.py:120-177` |
| `num_train_timesteps` | 否 | `1000` | int | `:446` |
| `train_unet` | 否 | `True` | bool | `:406` |
| `train_text_encoder` | 否 | `False` | bool；SDXL 下为两个 TE 各建 LoRA（`CLIPAttention`/`CLIPMLP`） | `:407`、`toolkit/lora_special.py:284,655-676` |
| `min_snr_gamma` | 否 | `None` | float，**须 >1e-6 才生效** | `:411`、`extensions_built_in/sd_trainer/SDTrainer.py:1059-1070` |
| `snr_gamma` | 否 | `None` | float；优先级 `learnable_snr_gos > snr_gamma > min_snr_gamma` | `:412`、同上 |

注：`BaseSDTrainProcess.py:904-915` 里也有个 `apply_snr`，但全仓库**没有任何调用点**（`grep "apply_snr("` 只命中定义）——真正生效的是 `SDTrainer.py:1059-1070`。写文档/写代码都以 `SDTrainer` 为准。
| `noise_offset` | 否 | `0.0` | float | `:416` |
| `content_or_style` | 否 | `balanced` | `balanced/content/style`，改 timestep 采样分布 | `:384-385`、`BaseSDTrainProcess.py:1231-1310` |
| `timestep_type` | 否 | `sigmoid` | `sigmoid/linear/...`（SD 系走 `set_timesteps`，只对 flowmatch 分支有额外语义） | `:560` |
| `max_grad_norm` | 否 | `1.0` | float | `:423` |
| `gradient_checkpointing` | 否 | `True` | bool | `:419` |
| `prompt_dropout_prob` | 否 | `0.0` | 0–1；在 encode 时随机把整条 prompt 换空 | `:451`、`toolkit/train_tools.py:413-417` |
| `min_denoising_steps` / `max_denoising_steps` | 否 | `0` / `999` | int | `:397-398` |
| `ema_config.use_ema` | 否 | `False` | bool；`ema_decay` 默认 `0.999` | `:547-555`、`:835-847` |
| `diff_output_preservation` | 否 | `False` | bool；**开启时必须有 trigger_word，且与 `train_text_encoder: true` 冲突（抛 ValueError）** | `:495`、`extensions_built_in/sd_trainer/SDTrainer.py:95-102` |
| `blank_prompt_preservation` | 否 | `False` | bool | `:501` |
| `skip_first_sample` | 否 | `False` | bool | `:417` |

### `model`

| 字段 | 必填 | 默认 | 取值/范围 | 出处 |
|---|---|---|---|---|
| `name_or_path` | **是** | `None` → 抛 `ValueError('name_or_path must be specified')` | HF repo id 或本地路径 | `toolkit/config_modules.py:675-676` |
| `arch` | 否 | 省略时按 `is_*` 推，最终 `sd1` | `sdxl/sd1/sd2/sd3/flux/...`；允许 `arch:tag` 形式（tag 被丢弃） | `:716,776-778,809-831` |
| `is_xl` | 否 | `False`；`arch=='sdxl'` 时自动置 `True` | bool | `:642,791-792` |
| `dtype` | 否 | `float16` | fp16/bf16/fp32 | `:655` |
| `vae_dtype` | 否 | `= dtype` | 同上；**SDXL 想稳就把 VAE 单独提成 `float32`** | `:690` |
| `te_dtype` | 否 | `= dtype` | 同上 | `:692` |
| `vae_path` | 否 | `None` | 覆盖 VAE 的路径 | `:656`、`stable_diffusion_model.py:408-409` |
| `use_text_encoder_1` / `use_text_encoder_2` | 否 | `True` / `True` | bool，仅 SDXL 系生效 | `:670-671` |
| `experimental_xl` | 否 | `False` | bool，注入 alt `text_projection` | `:673`、`stable_diffusion_model.py:449-461` |
| `refiner_name_or_path` | 否 | `None` | 路径 | `:657` |
| `refiner_start_at` | 否 | `0.5` | 0–1 | `:659` |
| `quantize` / `quantize_te` | 否 | `False` | bool；源码注释写明「只对 flux 目前生效」 | `:695-696` |
| `qtype` / `qtype_te` | 否 | `qfloat8` | 量化后端名（`convrot8`/`float8`/`nvfp4`/`orbit*`/`uintx*` 家族，见 `toolkit/util/ostris_quant.py:188-211`） | `:697-698` |
| `low_vram` | 否 | `False` | bool | `:699` |
| `only_if_contains` / `ignore_if_contains` | 否 | 未设 | 模块名列表，做分层训练 | `:704-705` |

### `sample`

| 字段 | 必填 | 默认 | 取值/范围 | 出处 |
|---|---|---|---|---|
| `sampler` | 否 | `ddpm` | **必须与 `train.noise_scheduler` 一致**（示例注释原文 "must match train.noise_scheduler"） | `toolkit/config_modules.py:82-125`、`config/examples/train_lora_flux_24gb.yaml:76` |
| `sample_every` | 否 | `100` | int | `:82-125` |
| `sample_start_step` | 否 | `0` | int | 同上 |
| `width` / `height` | 否 | `512` / `512` | int；SDXL 用 1024 | 同上 |
| `prompts` | 否 | 空 | 字符串列表；支持 `[trigger]` 占位替换 | 同上、`BaseSDTrainProcess.py:1079-1082`、示例 `:80` |
| `neg` | 否 | `False` | bool | 同上 |
| `guidance_scale` | 否 | `7` | float；SDXL 推理默认 6.0 | 同上、`registry.py:50` |
| `sample_steps` | 否 | `20` | int；SDXL 推理默认 25 | 同上、`registry.py:12` |
| `seed` | 否 | `0` | int | 同上 |
| `format` | 否 | `jpg` | 图片后缀 | 同上 |

`SampleItem` 允许每条 prompt 单独覆盖 `width/height/neg/seed/guidance_scale/sample_steps/sigmas/network_multiplier`（`toolkit/config_modules.py:44-80,110-118`）。

### `save` / `logging`

| 字段 | 必填 | 默认 | 取值/范围 | 出处 |
|---|---|---|---|---|
| `save.save_every` | 否 | `1000` | int | `toolkit/config_modules.py:23-34` |
| `save.dtype` | 否 | `float16` | 保存精度 | 同上 |
| `save.max_step_saves_to_keep` | 否 | `5` | int | 同上 |
| `save.save_format` | 否 | `safetensors` | **只接受 `safetensors` 或 `diffusers`**，否则抛 `ValueError` | 同上 |
| `logging.log_every` | 否 | `100` | int | `:35-42` |
| `logging.use_wandb` | 否 | `False` | bool | 同上 |

保存文件名：中间存 `{job.name}_{step:09d}.safetensors`，LoRA 存 `{lora_name}_{step:09d}.safetensors`，最终存无步号版本（`BaseSDTrainProcess.py:507-546`）。

### 硬约束：未知字段会被静默吞掉

`config_modules.py` 的配置类全部是 `self.x = kwargs.get('x', default)` 风格（如 `:381-631`），**没有未知键校验**；`toolkit/config.py:38-51` 只检查 `job` / `config` / `config.name` 三个键。所以 YAML 里写 `rank: 32`、`alpha: 32`、`text_encoder_lr: 1e-5` 这种「看起来对」的键不会报错，而是**直接失效**。`validate_config` 的第一职责就是查字段名白名单，而不是查值。

### `validate_config` 检查项（写死给代码用）

| # | 检查 | 判定 | 依据 |
|---|---|---|---|
| 1 | 未知键 | 报错，列出不认识的键路径 | 见上「静默吞掉」 |
| 2 | `model.name_or_path` | 空 → 报错 | `config_modules.py:675-676` |
| 3 | `network.linear` / `linear_alpha` | 缺失 → 报错（提示不要写 rank/alpha） | `:175-233` |
| 4 | `train.gradient_accumulation_steps` 与 `gradient_accumulation` 同时出现 | 报错 | `:466-468` |
| 5 | `save.save_format` | 不在 `{safetensors, diffusers}` → 报错 | `:23-34` |
| 6 | `optimizer` / `lr_scheduler` | 不在白名单 → 报错 | `optimizer.py:4-119`、`scheduler.py:6-57` |
| 7 | `diff_output_preservation: true` 且 `train_text_encoder: true` | 报错 | `SDTrainer.py:101-102` |
| 8 | `diff_output_preservation: true` 但无 `trigger_word`（process 或 dataset 级） | 报错 | `SDTrainer.py:95-98` |
| 9 | `datasets[].resolution` 是列表 | 警告：会复制数据集，steps 需 ×len | `config_modules.py:1116-1152` |
| 10 | `sample.sampler` ≠ `train.noise_scheduler` | 警告（示例注释要求一致） | 示例 `:76` |
| 11 | `model.arch` / `is_xl` 未声明 | 警告：会落到 `sd1` 分支，加载错 pipeline | `config_modules.py:809-831` |
| 12 | `min_snr_gamma` 在 (0, 1e-6] | 警告：等于没开 | `extensions_built_in/sd_trainer/SDTrainer.py:1059-1070` |
| 13 | `model.quantize: true` | 警告：源码注明仅 flux 生效 | `config_modules.py:695-696` |

---

## ③ 推荐起点参数（第一轮训练）

| 参数 | 起点值 | 为什么 |
|---|---|---|
| `model.name_or_path` | `stabilityai/stable-diffusion-xl-base-1.0` | 唯一被注册表与 README 同时列出的 SDXL 底座（`registry.py:50`、ai-toolkit README:39 行区） |
| `model.arch` | `sdxl`（或 `is_xl: true`） | 否则 arch 落到 `sd1`（`config_modules.py:809-831`），会按 SD1.5 加载 pipeline |
| `model.dtype` | `float16`（默认） | 默认值即可；24GB 下 bf16 更稳时可改 `train.dtype: bf16` 并把 `model.vae_dtype` 单独设 `float32` |
| `model.vae_dtype` | `float32` | 默认随 `model.dtype`（fp16），而 SDXL VAE **在 fp16 下不稳定**——kohya 官方文档直接要求 fp16 时必须开 `--no_half_vae`（`kohya-ss/sd-scripts docs/sdxl_train_network.md`）。**ai-toolkit 没有等价开关**（`config_modules.py:690`、`stable_diffusion_model.py:188` 只跟随 dtype），只能靠 `vae_dtype: float32` 兜住。依据等级：官方文档支持，本机未复现 NaN |
| `network.linear` / `linear_alpha` | `32` / `32` | 契约默认（`docs/ARCHITECTURE.md:100-101`）。社区实测同向：HF 官方博客把 **rank 32 当默认**，rank 64 皮肤「磨皮」不真实；kohya 官方 SDXL 示例 `--network_dim=32 --network_alpha=16`。两家都落在 32，alpha 取 rank 还是 rank/2 留作对照项（HF 博客、`kohya-ss/sd-scripts docs/sdxl_train_network.md`） |
| `train.steps` | `2500`（人脸小集按 **120×张数** 折算） | 契约默认（`ARCHITECTURE.md:99`），实际 epochs 按 ② 的公式核对。HF 官方博客作者实测：**6 张脸 → 720 步（120×张数）**；**27 张风格图用 2500 步明显过拟合**，1000 步（batch 1–2）最平衡 → 步数随张数和类型缩放，不是固定值（`huggingface/blog` @ `8000ba4` `sdxl_lora_advanced_script.md`，作者实测） |
| `train.lr` | `1e-4`（若开 TE 则降到 `5e-5`） | 契约默认（`:102`）；源码默认 `1e-6`（`config_modules.py:387`），1e-4 是本项目选择。社区口径：HF 博客 UNet `1e-4` / TE `5e-5`（普通微调）或 `3e-4`（pivotal tuning）；kohya 示例 UNet `1e-4` / TE1·TE2 各 `1e-5`。**两家都把 TE lr 设得比 UNet 小**，而 ai-toolkit 把 TE lr 写死等于 `lr` → 开 `train_text_encoder` 时全局 `lr` 取 `5e-5` 折中 |
| `train.train_text_encoder` | `false` | TE 加载即冻结；且 `text_encoder_lr` 写不进（见 ②），要单独调 TE lr 必须改代码。训 TE 只增显存与漂脸风险 |
| `train.noise_scheduler` | `ddpm`（默认） | SD 系调度器表推荐起点；和 SDXL 原生采样习惯一致（`sampler.py:31-50`） |
| `train.min_snr_gamma` | 首轮不设；对照实验用 `5.0` | 机制：阈值须 >1e-6 才生效（`SDTrainer.py:1059-1070`）。HF 官方博客作者推荐 **5.0**，但同文说**小数据集上差别不明显，大数据集才显著** → 首轮不设，第二组再开 5.0 做对照 |
| `train.content_or_style` | `balanced`（默认） | 角色/风格都吃得下；`style` 会向低噪声细节偏（`:1231-1310`） |
| `train.batch_size` / `gradient_accumulation_steps` | `1` / `1` | 24GB 起点；显存够再加 accum（只能用 `gradient_accumulation_steps`，不能同时给另一个键） |
| `train.optimizer` | `adamw`（默认） | 默认即合理；8bit 版省显存但需 bitsandbytes（`optimizer.py:4-119`） |
| `datasets[].resolution` | `[512, 768, 1024]` | 契约默认（`ARCHITECTURE.md:104`）与 flux 示例同构（示例 `:37`）；SDXL 原生 1024，512 档用来吃低清图。**代价是数据集×3**（见 ②） |
| `datasets[].buckets` / `bucket_tolerance` | `true` / `64` | 默认；64 对齐比 SDXL VAE 的 16 更粗，长宽更整齐（`dataloader_mixins.py:270,315`） |
| `datasets[].cache_latents_to_disk` | `true` | 示例注释「除非你知道自己在做什么，否则留着 true」（示例 `:36`） |
| `datasets[].caption_dropout_rate` | `0.05` | 示例同值（`:33`）；防 caption 过拟合 |
| `save.dtype` / `save_every` / `max_step_saves_to_keep` | `float16` / `250` / `4` | 示例同值（`:19-21`） |
| `sample.sampler` / `sample_steps` / `guidance_scale` | `ddpm` / `25` / `6.0` | SDXL 推理默认（`registry.py:12,50`），且 sampler 与训练一致 |
| `sample.sample_every` | `250` | 与 `save_every` 对齐，好定位过拟合拐点 |

### 数据集该长什么样（作者实测 + 官方文档）

| 结论 | 出处 | 可信度 |
|---|---|---|
| 质量 > 数量，人脸 LoRA **6–10 张**就够 | HF 官方博客（作者实测） | 作者实测 |
| 训脸时训练集里**不要出现别的脸** | 同上 | 作者实测 |
| 构图上**特写 + 全身都要有**（泛化姿势），**避免人物太远**（像素浪费） | 同上 | 作者实测 |
| 背景/服装/姿势/表情/光照要多样 | 同上 | 作者实测 |
| 同一套衣服或道具**别每张都出现** | dataset-deviser README 规则（非实测） | 竞品规则 |
| 风格 LoRA 27 张 + 人脸同参数会过拟合 → 风格类步数要更少 | HF 官方博客（作者实测） | 作者实测 |
| 24GB 单卡：batch 1 + gradient checkpointing 起步 | kohya 官方文档（同类量级） | 官方文档 |

> 只收「能核到作者本人实测」的条目。Reddit 本机不可达（见 `docs/sources.md` §4），论坛传闻一条没收。

---

## ④ caption 风格与禁忌

**风格（写进 styler 的规则）**
- 短标签、逗号分隔，**触发词在最前**（`README.md:20` 三模型对照表；`readme` 项目口径）。
- 触发词由 `[trigger]` 占位替换：caption 里已有 `[trigger]` 时会被换成 trigger_word；没有则注入（`toolkit/dataloader_mixins.py:2367`、`BaseSDTrainProcess.py:1617-1620`）。
- 逗号分隔还承担功能：`shuffle_tokens: true` 时按逗号切分重排（示例 `:34` 注释）。所以**不要用长句+从句**，否则 shuffle 语义失效。
- 长度上限：CLIP tokenizer 的 `model_max_length`（77 token）——`encode_prompts` 在未显式给 `max_length` 时用的是它（`toolkit/train_tools.py:402-422`），SDXL 两个 TE 都按此截断。超长部分直接丢。

**禁忌（写了会坏事的词/写法）**
- **把身份特征逐图写进 caption**：脸型、发型、肤色写多了会和触发词抢身份，换装时脸漂。身份交给触发词承载（`README.md:21` 同理；`UNVERIFIED`：无源码根据，属社区经验，标出来给主管判断）。
- **把不变的东西写全**：同一套衣服/同一个道具在每张图里都被描述 → 会被学成角色的一部分。这条来自竞品 README 的规则（trigger 学身份、caption 写变化；道具要先命名否则每张重画），**不是实测结论**，判据用 ⑤ 的去触发词对照。
- **写风格词**：数据集本身是风格 LoRA 时，caption 里再写 `anime style`、`oil painting` 会让触发词只覆盖「内容+风格」双重绑定；`DatasetKind.STYLE` 下 styler 应写内容不写风格（`docs/ARCHITECTURE.md:35`）。
- **中文/全角标点混入**：SDXL 双 CLIP 对中文基本无响应，caption 必须英文（架构推断，`UNVERIFIED`）。
- **空 caption 或纯触发词**：`prompt_dropout_prob`/`caption_dropout_rate` 已经负责「丢」，人工再写空等于放大丢概率。
- **token 超 77**：静默截断，后面的描述白写（见上）。

---

## ⑤ 验收怎么测

按 `docs/ARCHITECTURE.md` 的「配置被 ai-toolkit 读得进」+「权重真学对」两层验。

**A. 配置层（无需 GPU）**
1. `validate_config(cfg, SDXL)` 返回空列表（② 表 13 条）。
2. `yaml.safe_dump(cfg)` 后能被 `toolkit.config.get_config()` 读：`get_config(dict)` 会走 `preprocess_config`，校验 `job`/`config`/`config.name` 三个键（`toolkit/config.py:38-51`）。
3. 反向检查「未知键」：故意塞 `rank: 32` 进 `network`，`validate_config` 必须报错——这条是防止静默失效的回归测试。

**B. 训练层（首轮跑完看）**
1. `sample_every=250` 的样图里，`[trigger]` 提示词应稳定复现身份；25 步 / guidance 6.0 / 1024×1024 与推理默认一致（`registry.py:12,50`）。
2. **三提示法定性**（方法来自社区汇编口径，标 `UNVERIFIED`，见 `_community-evidence.md`；SDXL 同样适用）：① 分布内提示（触发词 + 训练集同款构图）② 泛化提示（触发词 + 没见过的场景/服装）③ 去触发词对照（同提示不带 trigger，应回到底座原有分布）。
3. **过拟合判据**：连续 save 的样图从「服装/背景被复刻」走向「姿势僵硬、只出训练集的构图」时，取更早的 checkpoint；`max_step_saves_to_keep=4` 就是为此（`config_modules.py:23-34`）。
4. **步数 sanity**：日志里每 epoch 的样本数应等于 `张数 × num_repeats × len(resolutions)`；对不上说明 resolution 拆数据集的行为被误读（`config_modules.py:1116-1152`）。
5. **过拟合拐点**：HF 博客作者实测「同参数 2500 步 → 服装/背景/神态全被复刻」，1000 步左右最平衡。样图出现「只会训练集构图、换背景即脸漂」时**先降步数，不要加 rank**。
6. **文字别当验收指标**：SDXL 底座本身文字渲染弱，数据集含文字（招牌/logo）时模板字模糊不算 LoRA 失败——这是底座限制（社区共识，`UNVERIFIED`）；文字档的验收对象是 MOSS。

---

## 未查实 / 需要什么才能确认

| # | 缺口 | 现状 | 确认需要 |
|---|---|---|---|
| 1 | `lr=1e-4`、`rank=32/alpha=32` 是否最优 | 已有两家社区实测支持 32 与 1e-4（HF 博客、kohya 示例）；alpha 取 32 还是 16 未验 | 一组对照：alpha 32/16 × lr 1e-4/5e-5，记录样图与脸漂 |
| 2 | SDXL VAE fp16 不稳定的具体触发条件 | 源码无保护；kohya 官方文档已确认「fp16 下不稳定、必须 `--no_half_vae`」，但触发阈值（哪些图会 NaN/灰噪）未查 | 官方 issue 或本机复现：fp16 VAE 编一批图，找出出错的样本特征 |
| 3 | `min_snr_gamma` 对 SDXL 的增益 | HF 博客推荐 5.0，但自述小数据集不明显 | 5.0 vs 不设的对照（同数据、同步数） |
| 4 | `content_or_style` 对角色 LoRA 的实际差异 | 只有 timestep 分布机制（`:1231-1310`） | 两档各跑一次同数据 |
| 5 | SDXL 12.9B/2.6B 等参数量数字 | 未在源码中核对 | 官方 model card（2026-11-27 HF 抓取失败，见 `sources.md`） |
| 6 | 社区经验的覆盖面 | 已补 HF 官方博客 + kohya 官方文档两类作者实测；**Reddit/Civitai 本机不可达**（fetch 失败、firecrawl 403，见 `sources.md` §4），论坛口径缺失 | 需要能访问 Reddit 的出口，或人工摘录 3–5 篇带图对照的帖子 |
