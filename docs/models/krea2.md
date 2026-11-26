# Krea 2（krea/Krea-2-Raw）训练规格

> 证据口径：源码引用一律写 `ai-toolkit@73c84b0 <路径>:<行号>`（仓库 ostris/ai-toolkit，本地 `_src/ai-toolkit`，抓取日 2026-11-27，`VERSION = "0.13.43"` 见 `version.py:1`；无 .git，commit 由 GitHub API 查得）。`docs/ARCHITECTURE.md:<行号>` 指本项目接口契约。标 `UNVERIFIED` 的条目**不许**被代码当默认值用。
>
> 适用范围：`TargetModel.KREA2` → `build_config()` 产出的 `job: extension / process[type=sd_trainer]` YAML。
>
> **先读这条**：Krea 2 不是 SDXL 的「换模型」，它有三个会让 SDXL 经验直接失效的机制——① flow matching（无 epsilon/v_prediction）② `train.noise_scheduler` 必须显式写 `flowmatch`，否则 `timestep_type` 完全失效 ③ 文本编码器是 Qwen3-VL，**`train_text_encoder: true` 是静默空操作**。详见 ② 的三条硬约束。

---

## ① 模型是什么

| 项 | 事实 | 出处 |
|---|---|---|
| 训练底模 | `krea/Krea-2-Raw`（`Turbo` 是蒸馏快版，只用于验收出图）。第三方官方 cookbook 佐证：两变体**同 backbone**，只差采样配方——Turbo 蒸馏 **8 步 / guidance 1.0（无 CFG）**，Raw 是**非蒸馏 base、~52 步 / guidance ≈4.5（diffusers 语义；ai-toolkit yaml 写 5.5）** | `README.md:23`（本项目三模型表）；SGLang 官方 cookbook（2026-11-27 抓取，见 `sources.md`）；HF `krea/Krea-2-Raw` api.json（`gated: auto`，105,924 downloads / 1,482 likes / `license: other` / 55 文件，2026-11-27）→ 权重细节仍 **`UNVERIFIED`**（model card 需登录接受许可，HTTP 401） |
| 推理默认（非训练） | 注册表里 krea2 指 `krea/Krea-2-Turbo`，8 步 / guidance 1.0 / 开量化 | `toolkit/models/registry.py:26` |
| 加载器 | `Krea2Model(BaseModel)`，`arch = "krea2"` | `extensions_built_in/diffusion_models/krea2/krea2.py:144-145` |
| 主网络 | 单流 MMDiT（`SingleStreamDiT`），宽 6144、28 层、48 头（KV 12 头）、patch 2、通道 16 | `krea2.py:66-80`（`KREA2_MMDIT_CONFIG`） |
| 规模 / 显存 | DiT **bf16 约 24 GB**；2×H200 上峰值 38 GB（tp=2 后 27 GB/卡）。与 6144×28 的参数规模自洽，也解释了为什么 24GB 单卡必须量化/offload | SGLang 官方 cookbook（第三方部署实测） |
| 文本编码器 | `Qwen/Qwen3-VL-4B-Instruct`，取 **12 层隐状态堆叠**（`SELECT_LAYERS`），加载后 `eval()` + `requires_grad_(False)` | `krea2.py:99,232-260`、`src/text_encoder.py:22` |
| TE 最大长度 | `model_kwargs.max_text_length`，默认 **512** token（Qwen 分词器，不是 CLIP 的 77） | `krea2.py:167-168` |
| VAE | `Qwen/Qwen-Image` 的 VAE，f8 下采样；`vae_scale_factor=8`，配合 `patch_size=2` → 桶对齐量 **16** | `krea2.py:100,163-164,204-206` |
| 预测目标 | **flow matching**（`is_flow_matching=True`），预测 velocity | `krea2.py:159` |
| 训练调度器 | `get_train_scheduler()` → `CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)` | `krea2.py:201-202` |
| 动态 shift | `use_dynamic_shifting=True`，mu 由图像 token 数线性插值：256-res→0.5、1280-res→1.15（`time_shift_type=exponential`，`base_shift 0.5`/`max_shift 1.15`，`num_train_timesteps 1000`） | `krea2.py:87-97` |
| LoRA 目标 | `target_lora_modules = ["SingleStreamDiT"]`，经 `network_kwargs['target_lin_modules']` 传入 | `krea2.py:158`、`jobs/process/BaseSDTrainProcess.py:1931-1932` |
| 编辑分支 | `is_edit` / `kv_cache` 由 `model_kwargs` 开（本项目不涉及） | `krea2.py:184,198` |

**TE 的编码口径（决定 ④ caption 怎么写）**：`encode_krea_prompt` 会把每条 prompt 包进固定指令模板：

```
<|im_start|>system
Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>
<|im_start|>user
<prompt><|im_end|>
<|im_start|>assistant
```

系统前缀 34 个 token 的隐状态在输出时被切掉（`src/text_encoder.py:29-42`）。也就是说：**模型被训练成「把 caption 当一段对画面颜色/形状/尺寸/材质/数量/文字/空间关系的描述」来读**，不是当 SDXL 那种标签串读。

---

## ② ai-toolkit 训练配置字段表

骨架与 SDXL 完全同形（`job / config.name / process[type=sd_trainer]`），差异只在 `train`/`model` 取值。整份 YAML 形状见 `config/examples/train_lora_flux_24gb.yaml:1-90`。

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
          shuffle_tokens: false
          cache_latents_to_disk: true
      train:
        noise_scheduler: flowmatch      # ← 必写，否则 timestep_type 失效
        timestep_type: linear
        lr: 1e-4
        steps: 2500
      model:
        name_or_path: krea/Krea-2-Raw
        arch: krea2                     # ← 必写，否则落到 sd1
        dtype: bf16
      sample: { width: 1024, height: 1024, guidance_scale: 5.5, sample_steps: 52 }   # yaml 写 5.5，生效 4.5
```

### 顶层与 process

与 SDXL 相同，逐字对齐源码：

| 字段 | 必填 | 默认 | 取值/范围 | 出处 |
|---|---|---|---|---|
| `job` | 是 | — | `extension` | `toolkit/config.py:38-40` |
| `config` / `config.name` | 是 | — | `config.name` 会被替换进 `[name]` 占位 | `toolkit/config.py:41-51` |
| `process[].type` | 是 | — | `sd_trainer`（所有 arch 共用这一个 trainer） | `extensions_built_in/sd_trainer/__init__.py:8` |
| `process[].training_folder` | 是 | — | 路径 | 示例 `:6` |
| `process[].trigger_word` | 否 | `None` | 字符串；注入 caption（`inject_trigger_into_prompt`）并继承给未单独设置的 dataset | `jobs/process/BaseSDTrainProcess.py:135,168-169`、`toolkit/dataloader_mixins.py:2367` |

### `network`

| 字段 | 必填 | 默认 | 取值/范围 | 出处 |
|---|---|---|---|---|
| `type` | 否 | `lora` | `lora`（`linear` 是别名） | `toolkit/config_modules.py:175-233` |
| `linear` | 否 | `4` | rank，int | 同上 |
| `linear_alpha` | 否 | `= alpha`（默认 `1.0`） | float | 同上 |
| `transformer_only` | 否 | `True` | Krea2 是 transformer，保持默认即只训 DiT | 同上 |
| `conv` / `conv_alpha` | 否 | `None` / 跟随 | 卷积层 LoRA rank | 同上 |

**注意**：字段名是 `linear`/`linear_alpha`，不是 `rank`/`alpha`；写错被静默忽略（`config_modules.py` 全是 `kwargs.get()` 风格）。

### `datasets[]`

| 字段 | 必填 | 默认 | 取值/范围 | 出处 |
|---|---|---|---|---|
| `folder_path` 或 `dataset_path` | 是（二选一） | — | 路径 | `config_modules.py:933,935` |
| `caption_ext` | 否 | `txt` | 不带点会自动补 | `:953-956` |
| `resolution` | 否 | `512` | int 或 int 列表；**列表会被拆成 N 份 dataset** | `:959,1116-1152` |
| `buckets` / `bucket_tolerance` | 否 | `True` / `64` | 桶宽高按 `bucket_tolerance` 对齐；Krea2 的 VAE+patch 对齐量是 16，64 是其整数倍，安全 | `:961-962`、`krea2.py:204-206`、`toolkit/dataloader_mixins.py:270,315` |
| `caption_dropout_rate` | 否 | `0.0` | 0–1 | `:968` |
| `shuffle_tokens` / `keep_tokens` | 否 | `False` / `0` | 按逗号切分重排；**Krea2 建议保持 false**（见 ④） | `:967,969` |
| `token_dropout_rate` | 否 | `0.0` | 0–1 | `:966` |
| `cache_latents_to_disk` | 否 | `False` | bool | `:1018` |
| `num_repeats` | 否 | `1` | int | `:1014` |

`resolution` 的语义同 SDXL：**面积上限，不是输出边长**（`toolkit/buckets.py:17-48`）。给列表会复制数据集、步数要 ×N（`config_modules.py:1116-1152`）。

### `train`（Krea 2 实际会读、且语义有差异的部分）

| 字段 | 必填 | 默认 | 取值/范围 | 出处 |
|---|---|---|---|---|
| **`noise_scheduler`** | **是（本模型）** | `ddpm` | **必须写 `flowmatch`**：只有这个值才会走 `set_train_timesteps(...)` 分支，`timestep_type`、动态 shift 才生效；写 `ddpm` 会落到 `set_timesteps(1000)`，训练仍能跑但 timestep 分布不受控 | `config_modules.py:383`、`BaseSDTrainProcess.py:1173-1210` |
| `timestep_type` | 否 | `sigmoid` | `linear` / `sigmoid` / `shift` / `flux_shift` / `lumina2_shift` / `weighted`；`linear` = 1000→1 均匀，`sigmoid` = 随机偏向中部，`shift` = 按 `use_dynamic_shifting` 复刻推理 shift | `config_modules.py:560`、`toolkit/samplers/custom_flowmatch_sampler.py:111-171` |
| `linear_timesteps` / `linear_timesteps2` | 否 | `False` | bool，任一为 true 等价于 `timestep_type: linear` | `BaseSDTrainProcess.py:1177-1183` |
| `num_train_timesteps` | 否 | `1000` | int | `config_modules.py:446` |
| `steps` | 否 | `1000` | int | `:386` |
| `lr` | 否 | `1e-6` | float | `:387` |
| `unet_lr` | 否 | `= lr` | float | `:388` |
| `optimizer` | 否 | `adamw` | `adamw/lion/adafactor/prodigy/adamw8bit/...`（未知名抛 `ValueError`） | `:393`、`toolkit/optimizer.py:4-119` |
| `lr_scheduler` | 否 | `constant` | `cosine/linear/constant/constant_with_warmup/...` | `:395`、`toolkit/scheduler.py:6-57` |
| `batch_size` | 否 | `1` | int | `:399` |
| `gradient_accumulation_steps` | 否 | `1` | int；与 `gradient_accumulation` 互斥（同时给抛 `ValueError`） | `:461-468` |
| `dtype` | 否 | `fp32` | `fp32/fp16/bf16` | `:401`、`toolkit/train_tools.py:37-49` |
| `train_unet` | 否 | `True` | bool；Krea2 下即「训 DiT」 | `:406` |
| **`train_text_encoder`** | 否 | `False` | 见下方硬约束 ②：**Krea2 上设 `true` 不会训任何东西** | `:407`、`toolkit/lora_special.py:678-680` |
| `content_or_style` | 否 | `balanced` | `content`/`style`/`balanced`，改 timestep 采样分布 | `:384-385`、`BaseSDTrainProcess.py:1231-1310` |
| `min_snr_gamma` / `snr_gamma` | 否 | `None` | 机制上 flow matching 也会生效（`SDTrainer.py:1059-1070`，`ignore_snr` 恒 False），但 SNR 口径按 sigma 算，对 flow 模型的推荐值无依据 → 起步不设 | 同上 |
| `prompt_dropout_prob` | 否 | `0.0` | 0–1，整条 prompt 置空 | `:451` |
| `diff_output_preservation` | 否 | `False` | 需 `trigger_word`；与 `train_text_encoder: true` 冲突（抛 `ValueError`） | `:495`、`extensions_built_in/sd_trainer/SDTrainer.py:95-102` |
| `gradient_checkpointing` | 否 | `True` | bool | `:419` |
| `max_grad_norm` | 否 | `1.0` | float | `:423` |
| `ema_config.use_ema` | 否 | `False` | `ema_decay` 默认 `0.999` | `:547-555,835-847` |

### `model`（Krea 2 必读）

| 字段 | 必填 | 默认 | 取值/范围 | 出处 |
|---|---|---|---|---|
| `name_or_path` | **是** | 无 → 抛 `ValueError('name_or_path must be specified')` | 训练用 `krea/Krea-2-Raw` | `config_modules.py:675-676` |
| `arch` | **是（本模型）** | 省略时按 `is_*` 推、最终 `sd1` | 必须 `krea2`；否则按 SD1.5 pipeline 加载 | `:716,809-831` |
| `dtype` | 否 | `float16` | `fp16/bf16/fp32`；24GB 建议 `bf16` | `:655` |
| `vae_dtype` | 否 | `= dtype` | 同上 | `:690` |
| `te_dtype` | 否 | `= dtype` | 同上；Qwen3-VL 用 bf16 更稳 | `:692` |
| `quantize` / `quantize_te` | 否 | `False` | bool；注册表推理默认开 | `:695-696`、`registry.py:26` |
| `qtype` / `qtype_te` | 否 | `qfloat8` | `qfloat8/convrot8/nvfp4/...`（`toolkit/util/ostris_quant.py:188-211`） | `:697-698` |
| `low_vram` | 否 | `False` | bool | `:699` |
| `model_kwargs` | 否 | `{}` | dict；Krea2 认 `text_encoder_path`、`vae_path`、`max_text_length`、`is_edit`、`kv_cache` 等 | `:714`、`krea2.py:167,184,198,234,263` |
| `latent_space_version` | 否 | 模型自定义 | Krea2 用默认 latent 空间 | `krea2.py:160` |

### 三条硬约束（SDXL 经验在这里失效）

**① `train.noise_scheduler` 决定 timestep 分支，不是「选采样器」。**
`BaseSDTrainProcess.py:1752-1755` 先看模型类有没有 `get_train_scheduler`：Krea2 有 → 调度器实例永远是 `CustomFlowMatchEulerDiscreteScheduler`。但后面的 timestep 准备按 **`train.noise_scheduler` 字符串**分支（`:1173-1210`）：只有 `== 'flowmatch'` 才调 `set_train_timesteps(timestep_type=..., latents=..., patch_size=...)`（`:1191-1205`），其它值走 `set_timesteps(num_train_timesteps)`，`timestep_type` 与动态 shift 全部不参与。所以 `noise_scheduler: flowmatch` 是**语义开关**，不是「选 ddpm 还是 flowmatch」的偏好。

**② `train_text_encoder: true` 在 Krea2 上是静默空操作。**
`lora_special.py:678-680` 训 TE 时用的固定列表是 `TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]`（`:284`）；Krea2 的 TE 是 Qwen3-VL，类名不匹配 → 建出 **0 个 TE LoRA**，源码只打印 skipped 数量、**不报错**。`get_model_class` 的 `target_lora_modules` 只覆盖 DiT 侧（`BaseSDTrainProcess.py:1931-1932`、`krea2.py:158`）。所以：想动 TE 只能改源码；`diff_output_preservation` 与 `train_text_encoder` 的冲突检查（`SDTrainer.py:101-102`）在 Krea2 上纯属自伤。

**③ `text_encoder_lr` 写了不读。**
两条优化器构造路径都写死 `'text_encoder_lr': self.train_config.lr`（`BaseSDTrainProcess.py:2010-2021,2130-2140`），全仓库 `grep train_config.text_encoder_lr` = 0 命中。

### `sample`（Krea2 特有的 guidance 偏移）

| 字段 | 必填 | 默认 | 取值/范围 | 出处 |
|---|---|---|---|---|
| `sampler` | 否 | `ddpm` | **Krea2 的 `generate_single_image` 不读它**（pipeline 自带 flow match 调度器），写了无效果 | `config_modules.py:82-125`、`krea2.py:411-467` |
| `width` / `height` | 否 | `512` | int；会被向下取整到 16 的倍数 | `:82-125`、`krea2.py:425-427` |
| `guidance_scale` | 否 | `7` | **实际生效值 = `max(0, 本值 - 1.0)`**：`krea2.py:455` 传 `guidance = gen_config.guidance_scale - 1.0` | `krea2.py:455` |
| `sample_steps` | 否 | `20` | int；Raw 建议按官方 52 步，Turbo 8 步 | `:82-125`、种子（`UNVERIFIED`） |
| `sample_every` / `sample_start_step` | 否 | `100` / `0` | int | 同上 |
| `prompts` | 否 | 空 | 列表，支持 `[trigger]` | 同上、`BaseSDTrainProcess.py:1079-1082` |
| `neg` / `seed` / `format` | 否 | `False` / `0` / `jpg` | — | 同上 |

### `validate_config` 检查项（写死给代码用）

| # | 检查 | 判定 | 依据 |
|---|---|---|---|
| 1 | 未知键（含 `rank`/`alpha`/`text_encoder_lr`） | 报错 | `config_modules.py` 全部 `kwargs.get()`，`config.py:38-51` 只校验 3 键 |
| 2 | `model.name_or_path` 为空 | 报错 | `config_modules.py:675-676` |
| 3 | `model.arch != krea2` 且未给 `is_xl` 等标志 | 报错（会落到 `sd1`） | `:809-831` |
| 4 | `train.noise_scheduler != flowmatch` | **报错**（Krea2 下 `timestep_type` 会静默失效） | `BaseSDTrainProcess.py:1173-1210` |
| 5 | `timestep_type` 不在 `{linear, sigmoid, shift, flux_shift, lumina2_shift, weighted}` | 报错 | `toolkit/samplers/custom_flowmatch_sampler.py:127-171` |
| 6 | `train.train_text_encoder: true` | 报错（TE 是 Qwen3-VL，LoRA 目标类名不匹配 → 空操作） | `toolkit/lora_special.py:284,678-680` |
| 7 | `train.diff_output_preservation: true` 且无 `trigger_word` | 报错 | `SDTrainer.py:95-98` |
| 8 | `save.save_format` 不在 `{safetensors, diffusers}` | 报错 | `config_modules.py:23-34` |
| 9 | `network.linear`/`linear_alpha` 缺失 | 报错（提示不要写 rank/alpha） | `:175-233` |
| 10 | `gradient_accumulation_steps` 与 `gradient_accumulation` 同时出现 | 报错 | `:466-468` |
| 11 | `datasets[].resolution` 是列表 | 警告：数据集 ×N，steps 要换算 | `:1116-1152` |
| 12 | `sample.guidance_scale <= 1.0` 却想用 CFG | 警告：实际生效值 = 本值 − 1 | `krea2.py:455` |
| 13 | `model.quantize: true` 且开了 `save` 后的 merge | 警告：量化模型不 merge（`model_is_prequantized` 分支） | `BaseSDTrainProcess.py:1985-2000` |

---

## ③ 推荐起点参数（第一轮训练）

| 参数 | 起点值 | 为什么 |
|---|---|---|
| `model.name_or_path` | `krea/Krea-2-Raw` | 训练用 Raw、Turbo 只验收（`README.md:23`、种子）。仓库名官方 model card 未访问成功 → `UNVERIFIED`，先按 README 口径 |
| `model.arch` | `krea2`（必写） | 不写会按 `sd1` 加载 pipeline（`config_modules.py:809-831`） |
| `train.noise_scheduler` | `flowmatch`（必写） | 唯一能进 `set_train_timesteps` 分支的值（见 ② 硬约束 ①） |
| `train.timestep_type` | `linear` | `set_train_timesteps` 的默认类型，1000→1 均匀；`sigmoid`（TrainConfig 默认）在 flow matching 下是随机分布，无依据。想更贴推理分布可试 `shift`（Krea2 `use_dynamic_shifting=True`，会按图像 token 数算 mu），但**本轮无实测**，标 `UNVERIFIED` |
| `network.linear` / `linear_alpha` | `32` / `32` | 契约默认（`docs/ARCHITECTURE.md:100-101`），与种子二手校准一致，两处独立指向同一值 |
| `train.steps` | `2500` | 契约默认（`:99`）；种子二手校准同值。实际 epochs 按「张数×num_repeats×len(resolutions)」核对 |
| `train.lr` | `1e-4` | 契约默认（`:102`）；种子同值（保守档 5e-5）。**源码默认是 `1e-6`**（`config_modules.py:387`），1e-4 是本项目选择不是源码结论 |
| `train.batch_size` / `gradient_accumulation_steps` | `1` / `1` | 24GB 起点；DiT 6144 宽，先单卡单批跑通 |
| `train.train_text_encoder` | `false` | 且设 `true` 也无效（硬约束 ②） |
| `model.dtype` / `te_dtype` | `bf16` / `bf16` | fp16 在 12B 级 DiT 上容易溢出；`get_torch_dtype` 接受 bf16（`train_tools.py:37-49`）。**这一条是工程判断，`UNVERIFIED`** |
| `model.quantize` / `qtype` | 24GB 先 `true` / `qfloat8` | 注册表推理就是 qfloat8（`registry.py:26`）；SGLang cookbook 实测 DiT bf16 约 24 GB，24GB 单卡光权重就占满（还要 TE、VAE、激活、LoRA 梯度）→ 量化/offload 不是可选项。训练侧显存仍未实测，标 `UNVERIFIED` |
| `model.vae_dtype` | `fp32`（可选） | 预防 Qwen-Image VAE 在低精度下出 NaN；源码无保护（`config_modules.py:690`），预防性设置 |
| `datasets[].resolution` | `[512, 768, 1024]` | 契约默认与种子一致；Krea2 原生 1024。注意数据集 ×3 |
| `datasets[].bucket_tolerance` | `64` | Krea2 对齐量是 16，64 是整数倍（`krea2.py:204-206`、`dataloader_mixins.py:270,315`） |
| `datasets[].caption_dropout_rate` | `0.05` | 示例同值（`train_lora_flux_24gb.yaml:33`），防 caption 过拟合 |
| `train.diff_output_preservation` | `false` 起步 | 种子建议开 `true` + `class person`；但需 trigger_word 且不能开 TE，先跑通再加。种子来源可信度：实测但非本仓库验证 |
| `save.save_every` / `max_step_saves_to_keep` | `250` / `4` | 示例同值；2500 步能留 10 个中间档 |
| `sample.sample_steps` / `sample.guidance_scale` | Raw：`52` / **`5.5`（生效 4.5）**；Turbo：`8` / `1.0` | 生效 CFG = 本值 − 1（`krea2.py:455`）→ yaml `5.5` 才是 4.5、`1.0` 才是 0（Turbo 无 CFG）。**取值口径已由 PR #15（`3e3797e2`）定为 yaml `5.5` / 生效 4.5**：SGLang cookbook 写的是 diffusers 语义「Raw ≈4.5（1.0=关）」，对齐到 ai-toolkit 的减 1 口径就是 yaml 5.5；种子旧口径「生效 3.5」已弃用。步数由 SGLang 佐证（Raw ~52、Turbo 8）。**仍未经实拍**，标 `UNVERIFIED` |
| `sample.width` / `height` | `1024` / `1024` | 原生分辨率 |

---

## ④ caption 风格与禁忌

**风格（写进 styler 的规则）**

- **描述性自然语言，1–3 句**，覆盖「颜色 / 形状 / 尺寸 / 材质 / 数量 / 图内文字 / 空间关系」（`src/text_encoder.py:29-38` 的固定 system 指令就是这些维度——模型是按这个 schema 被训练的）。
- **只写「这张图里变化的部分」**：Krea2 全家（编辑、多图）靠自然语言承载差异，不变的身份交给触发词（`README.md:21`）。
- 长度上限 512 token（`krea2.py:167-168`）——比 SDXL 的 77 宽得多，但仍别写论文；`max_text_length` 之外静默截断。
- 触发词照常注入（`dataloader_mixins.py:2367`），`[trigger]` 占位会被替换（`BaseSDTrainProcess.py:1617-1620`）。

**禁忌（写了会坏事的写法）**

- **别照搬 SDXL 的逗号标签串**（`1girl, solo, red dress, ...`）：Krea2 的 TE 是 Qwen3-VL，把词表喂进去得到的是「句子碎片」，与模板声明的描述任务不匹配（推断，`UNVERIFIED`，无对照实验）。
- **别写身份细节**（脸型、发型、肤色）：会和触发词抢身份，换装即脸漂（`README.md:21`；社区经验，`UNVERIFIED`）。
- **别开 `shuffle_tokens`**：它按逗号切分重排，而 Krea2 的 caption 是整句——重排会把句子切成碎片（`config_modules.py:967` 语义 + Krea2 caption 形态）。
- **别写风格套话**（`masterpiece, best quality, 8k`）：模板要求的是画面描述，套话不产生描述性信号。
- **别用中文写 caption**：Qwen3-VL 本身多语，但 Krea 2 的公开训练数据以英文描述为主；中文 caption 的效果未验证，标 `UNVERIFIED`，不建议在第一批数据上赌。
- **别把同一套衣服/道具逐图写全**：每张都描述 → 被学成角色的一部分（种子 `:54` 实测教训，来源 dataset-deviser 作者复盘）。

---

## ⑤ 验收怎么测

**A. 配置层（无需 GPU）**

1. `validate_config(cfg, KREA2)` 返回空列表（② 表 13 条）。
2. `yaml.safe_dump(cfg)` 后能被 `toolkit.config.get_config()` 读（`toolkit/config.py:38-51`）。
3. 回归：故意写 `train.noise_scheduler: ddpm` + `timestep_type: linear`，`validate_config` 必须报错（防「timestep 静默失效」）。
4. 回归：故意写 `train_text_encoder: true`，必须报错（防空操作）。

**B. 训练层（首轮跑完看）**

1. 日志里 `create LoRA for Text Encoder: 0 modules` 是**预期**（Krea2 TE 无 LoRA 目标）；若你确实需要 TE 微调，这条路当前不存在。
2. `sample` 样图：Raw 52 步 + `guidance_scale: 5.5`（生效 4.5）；Turbo 8 步 + 生效 CFG 0（`guidance_scale: 1.0`）——两档都要出图，用来区分「LoRA 学坏了」还是「采样参数不对」。取值按 PR #15（`3e3797e2`），未经实拍，`UNVERIFIED`。
3. **三提示法定性**（种子 `:32`，标 `UNVERIFIED`，方法可用于三模型）：① 分布内（触发词 + 训练集同款）② 泛化（触发词 + 没见过场景/服装）③ 去触发词对照（应回到底座分布）。
4. **过拟合判据**：连续 checkpoint 从「服装/背景被复刻」走向「姿势僵硬、只会训练集构图」时取更早的档（`max_step_saves_to_keep=4` 就为这个）。
5. **步数 sanity**：每 epoch 样本数 = 张数 × `num_repeats` × `len(resolutions)`；对不上说明 resolution 拆数据集被误读。

---

## 未查实 / 需要什么才能确认

| # | 缺口 | 现状 | 确认需要 |
|---|---|---|---|
| 1 | `krea/Krea-2-Raw` 是否为官方训练底座、参数量多少 | 变体性质与采样配方已由 SGLang cookbook 佐证（Raw = 非蒸馏 base，~52 步 / 生效 guidance 4.5 → yaml 5.5；Turbo = 8 步蒸馏）；HF `api.json` 确认存在、105,924 downloads、`gated: auto`、`license: other`、55 文件，但 **model card 需登录接受许可（HTTP 401）**，权重/参数量细节仍缺 | 接受许可后读 model card / `config.json`；或读 diffusers `Krea2Pipeline` 文档 |
| 2 | `lr=1e-4`、`rank=32`、2500 步是否为 Krea2 最优 | 契约默认 + 第三方仓库二手校准（约 6★），两条独立但都非实测 | 对照训练：rank 16/32/64 × lr 5e-5/1e-4，固定种子出样图 |
| 3 | `timestep_type: linear` vs `shift` 的训练差异 | 只有代码机制；`shift` 会走 `use_dynamic_shifting` 的 mu 插值 | 同数据两档对照，看收敛速度与细节/结构平衡 |
| 4 | 24GB 是否必须 `quantize: true` | SGLang cookbook 给 DiT bf16 ≈24 GB（第三方部署实测）→ 24GB 单卡几乎必然需要量化/offload；训练侧（LoRA 梯度 + 激活 + 9B TE）仍未实测 | 一次 24GB 实跑，记录峰值显存 |
| 5 | 量化 + `diff_output_preservation` 能否共存 | 量化模型 `can_merge_in=False`（`BaseSDTrainProcess.py:1985-2000`），DOP 依赖 base 输出对照 | 需读 DOP 的 loss 路径确认是否依赖 merge |
| 6 | Krea2 上 `content_or_style` 的实际差别 | 只有 timestep 分布机制（`BaseSDTrainProcess.py:1231-1310`） | 两档各跑一次同数据 |
| 7 | 社区 Krea2 实战（数据集规模/构成、失败模式） | 本轮补了社区来源，但**没有一条覆盖 Krea2**：HF 官方博客只讲 SDXL、kohya 只讲 SDXL/MOSS，Reddit 三条路径全部访问失败（见 `sources.md` §4） | 能访问 Reddit/Civitai 的出口，或人工摘录 Krea2 带图对照的帖子；否则只能等我们自己的实跑 |
| 8 | Krea2 出图验收的 guidance 口径（yaml 5.5 / 生效 4.5） | PR #15（`3e3797e2`）依据 `krea2.py:455` 的减 1 定值，逻辑自洽但**未经实拍**；SGLang cookbook 的 4.5 是 diffusers 语义，两者换算后一致 | 一次 Raw 实拍：yaml `5.5` 与 `4.5` 各出一张，比「是否过曝/结构崩」 |
