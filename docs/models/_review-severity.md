# 审核：`loraforge/configs/problems.py` 的 48 条判级

> 审核对象：`loraforge/configs/problems.py`（PR #15 合入 `3e3797e2`；本地抓取 `_src/problems.py` 727 行时代码 636 行）
> 文件指纹：`c31a976653a8edd100fe1314174b0b1029265ec3e08251c1b54591565d34b7b7`，27,068 bytes，636 行
> 审核日期：2026-11-27（Asia/Singapore）
> 方法：逐条读 `_REGISTRY` 的判级理由 + 对应校验实现，再对回 `_src/ai-toolkit`（`73c84b0`）源码与三份模型文档。**未改代码**。
> 判级标准沿用文件头：`error` = 会白跑几小时 / ai-toolkit 直接抛错 / 静默走错架构；`warn` = 效果打折、成本上升、可观测性下降，但配置能跑。

## 结论摘要

- 48 条里 **34 条判级我同意**，**14 条要改**：其中 8 条是 `error` 拦住了合法配置（假阳性最伤），2 条是 `error` 的理由站不住，3 条是条件写错导致误判，1 条是 `warn` 漏掉了真问题。
- 最急的两条：`network.rank-alpha-ratio` 会把 **ai-toolkit 自己的默认值（rank 4 / alpha 1.0）判成 error**；`_TIMESTEP_TYPES` 少了 `shift`/`flux_shift`/`lumina2_shift`，把源码明确支持的合法值判成「未知分支」——我们自己 MOSS 文档给的 `timestep_type: shift` 首当其冲。

## 一、要改级别的条目（按风险排序）

| # | code（problems.py 行） | 现 | 建议 | 理由与出处 |
|---|---|---|---|---|
| 1 | `network.rank-alpha-ratio`（`:535-544`） | error | **warn** | 阈值区间 `[0.5, 2.0]` 没有源码依据，是把「alpha=rank」当唯一惯例。`config_modules.py:187-191`：`rank` 缺省 4、`alpha` 缺省 **1.0** → `alpha/rank=0.25`，**工具自己的默认配置就被判 error**。kohya 官方 SDXL 示例是 `--network_dim=32 --network_alpha=16`（ratio 0.5，刚好压线）。ratio 偏小只是有效学习率（`lr × alpha/rank`）变小、训练偏保守，不是「不可用」 |
| 2 | `_TIMESTEP_TYPES` 白名单（`:67`） | —（判 error 的前提） | **补 `shift`/`flux_shift`/`lumina2_shift`**（`next_sample`/`one_step` 保留，它们确实有分支） | 白名单的真源头不止 `custom_flowmatch_sampler.py:127-216`：该文件 `:145` 明确处理 `flux_shift`/`lumina2_shift`/`shift` 三个名字，`BaseSDTrainProcess.py:1181-1261` 另外处理 `one_step`/`two_step`/`four_step`/`eight_step`/`next_sample`。当前白名单把三个**合法**分支判成「未知 = 等于没设」，而我们 MOSS 文档给的 `timestep_type: shift` 会直接命中这条 error |
| 3 | `train.timestep-type-sigmoid`（`:459-466`） | error | **warn** | 「sigmoid 会训坏」没有任何源码或官方文档支持。`custom_flowmatch_sampler.py:131-144` 只是分布不同（中心在 t≈500）；kohya 的 MOSS 文档同样只说「默认 sigmoid 与预训练分布差很多」，**仍能训**。Krea2 的 `linear` 是二手校准（6★ 仓库），不是硬约束 → 拦成 error 属假阳性 |
| 4 | `train.timestep-type-not-linear`（`:474-481`） | error | **warn** | 实现里的注释自己写着「白名单内但不是 linear：**能跑**，口径偏了」，登记表却给 error，自相矛盾。见 `docs/models/krea2.md` ③：`linear` 是推荐起点，`shift`/`weighted` 属未验证路线，不是会失败 |
| 5 | `train.text-encoder`（`:501-507`） | error | **warn** | `toolkit/lora_special.py:284,678-680`：krea2/moss 的 Qwen TE 建出 **0 个** LoRA，只把 skipped 数量打印出来，**不抛错、不中断训练**。MOSS 官方要求冻结（`USERGUIDE.md`）也只是「不该开」，不等于跑不起来。属「白搭」不属「白跑」——但文案要写明「开了也不会生效」 |
| 6 | `model.arch`（`:345-352`） | error | **error 保留，条件放宽** | 现在 `arch = model.get("arch") or ""`，只要没写 `arch` 就 error。但 `config_modules.py:809-831` 允许由 `is_xl` 推出 `sdxl`；官方 flux 示例就是只写 `is_flux: true`（`config/examples/train_lora_flux_24gb.yaml:63-64`），SDXL 同理。建议：`arch` 缺失但家族标志（`is_xl`）正确 → 不报；两者都缺 → error（会落 `sd1`） |
| 7 | `train.required-empty`（`:427-429`） | error | **warn** | 四个键全有默认值：`noise_scheduler` `ddpm`（`:383`）、`steps` 1000（`:386`）、`lr` 1e-6（`:387`）、`batch_size` 1（`:399`）。省略即走默认，配置能跑。要拦的是「显式写了空值/非法值」，不是「没写」 |
| 8 | `sample.size`（`:620-625`） | error | **只在键存在时校验** | `SampleConfig` 的 `width`/`height` 默认 512、`sample_steps` 默认 20（`:82-125`）；省略是合法写法，现在却报「必须为正数，实际 None」 |
| 9 | `datasets.resolution`（`:572-576`） | error | **error 保留，接受 int** | `config_modules.py:959`：`self.resolution: int = kwargs.get('resolution', 512)`——**单个 int 是合法值**；`:1122-1152` 只在 list 时才复制数据集。登记表把出处写成 `config_modules.py:899`（`:899` 不是这个字段），且「必须是非空列表」与源码不符：`resolution: 1024` 会被拦 |
| 10 | `datasets.folder-path`（`:555-558`） | error | **接受 `dataset_path` 别名** | `config_modules.py:935` 定义 `dataset_path`，`:1127` 把 `folder_path`/`dataset_path` 一起拆分。只认 `folder_path` 会拦掉合法写法（数据集段只给了 `dataset_path`） |
| 11 | `model.family-mismatch`（`:336-343`） | error | **warn** | 判定靠 `name_or_path` 子串（`:267-271`）：SDXL 只认 `stable-diffusion-xl`/`sdxl`，Krea 只认 `krea/` 前缀或 `krea-2`，MOSS 只认 `moss-image`。本地权重常见名 `sd_xl_base_1.0.safetensors`（含 `sd_xl` 不含 `sdxl`）、`moss_image_base.safetensors`（下划线）都会被判「拿错底座」→ 合法本地路径被一刀切。子串启发式不该有拦截权 |
| 12 | `model.arch-suffix`（`:353-361`） | error | **warn，且覆盖全 target** | `config_modules.py:776-778` 是通用逻辑，`krea2:raw`、`sdxl:foo` 的 tag 一样被丢弃；现在只查 `moss_image:base` 一种。丢 tag 不改变训练路径（变体由 `name_or_path` + `model_kwargs` 决定），属配置卫生问题 |
| 13 | `sample.sampler-mismatch`（`:610-617`） | warn | **krea2/moss 跳过，SDXL 保留** | `krea2.py:411-467`、`moss_image.py:371-411` 的 `generate_single_image` **不读** `gen_config.sampler`（登记表自己也引了 `krea2.py:175`）；且这两个模型的 `train.noise_scheduler` 必为 `flowmatch`，`sample.sampler` 无论填什么都不一致 → 每条 flow 系配置**恒响**一次，属噪音型假警报。SDXL 的 `sample.sampler` 会真的影响出图口径，保留 warn |
| 14 | MOSS 的 `timestep_type`（缺失校验） | 无 | **新增 error（白名单同 #2）** | timestep 分支只在 `t is KREA2`（`:457`）里跑，MOSS 完全不校验；而 MOSS 恰是最需要它的模型（预训练 logit-normal，见 `docs/models/moss_image.md` ③）。传非法值会在 `custom_flowmatch_sampler.py:230` 抛 `ValueError`，属真硬错 |

## 二、判级不改、但条件/文案要修

| # | code | 问题 | 建议 |
|---|---|---|---|
| 15 | `datasets.resolution-parallel` | 文案「训练量与耗时同倍增加」不准确：`steps` 固定时**训练时长不变**，变的是每张图的曝光次数（等效 epochs 除以 N） | 改为「同样步数下每张图被看到的次数变少；若要等价效果需把 steps ×N」 |
| 16 | `model.moss-uncond-skip-layers` | 对**每条** MOSS 配置都响（因为 `uncond_skip_layers` 缺省就不是 `[22]`），文案自己承认「不影响训练」 | 只在键**存在且不含 22** 时响；或降到 info 级 |
| 17 | `model.qtype-unknown` | `quantize: false` 时也判 qtype 家族，报无用警告 | 只在 `quantize`/`quantize_te` 为真时校验 |
| 18 | `train.lr-too-large` | `_LR_MAX` 阈值是工程判断（注释已标）；`> 1e-2` 严格大于，正好 1e-2 不响 | 判级可保留；阈值建议压到 `> 5e-3` 或写明「经验值，可关」 |
| 19 | `_SIZE_MULTIPLE`（`:81`）/ `sample.size-alignment` | 采样尺寸只要求 8 的倍数，但 krea2/moss 的 `get_bucket_divisibility()` 是 16（`krea2.py:204-206`、`moss_image.py:207-208`） | flow 系按 16 判；SDXL 保持 8 |

## 三、我同意、不用动的（避免误以为漏看）

`target.unknown`、`cfg.not-dict`、`job.missing`、`config.missing`、`config.name`、`config.process`、`config.process-item`、`process.type`、`model.missing`、`model.name-or-path`、`model.quantize-on-sdxl`、`model.vae-dtype`、`model.max-text-length`、`train.missing`、`train.noise-scheduler-not-flowmatch`、`train.noise-scheduler-flowmatch-on-sdxl`、`train.noise-scheduler-not-ddpm`、`train.timestep-type-unknown`（前提是修 #2 的白名单）、`train.diff-output-preservation-without-trigger`、`train.diff-output-preservation-with-te`、`train.steps`、`train.lr`、`network.rank`、`network.alpha`、`datasets.missing`、`datasets.item`、`datasets.resolution-value`、`datasets.resolution-alignment`、`datasets.caption-dropout-high`、`sample.missing`、`sample.prompts`。

其中判级正确但值得记一笔的：
- `train.noise-scheduler-flowmatch-on-sdxl` 我按 error 同意：`BaseSDTrainProcess.py:1756-1769` 对 SDXL 走 `arch='sd'` 的 epsilon 调度器，喂 flowmatch 语义的调度器属于静默走错路径。
- `model.vae-dtype` 保持 warn 合理：NaN 是「偶发」，且源码无保护，拦成 error 会让默认配置（`vae_dtype` 缺省跟随 fp16）全体被拦。

## 四、两个要主管/阿配拍板的策略问题

1. **alpha/rank 到底允不允许 <1**：kohya 官方示例是 32/16（0.5），ai-toolkit 默认 4/1.0（0.25）。如果产品只接受 `alpha == rank`，那 `_REGISTRY` 的 error 应改成「只允许 alpha==rank 或 alpha==rank/2」，且必须放过默认值；否则降到 warn。
2. **MOSS 的 timestep 政策（2026-11-27 已由技术报告解决）**：`arXiv:2610.23374` Table 2 给出 Stage 3 末期配方（shift=1 网格 + 截断 logit-normal μ=−1.09 / s=0.95 / cutoff 0.95 + 1% 均匀），ai-toolkit 没有 logit-normal 分支、`shift` 写死 5.0。按 σ 分位距离（σ=t/T 约定，按 `moss_image.py:344` 路径自算），`shift`（0.39）比 `linear`/`sigmoid`（各 1.04）更接近 → 默认取 `shift`，标 `UNVERIFIED`；校验层只放行白名单内，不替用户选。详见 `docs/models/moss_image.md` ③ 与 `docs/sources.md` §10。

## 五、对 `loraforge/configs/base.py` 默认值的影响（2026-11-27 追加）

| 默认值 | 现状（`d35b9a91`） | 建议 | 依据 |
|---|---|---|---|
| MOSS `train.timestep_type` | `shift` | **保留 `shift`**，但注释改成「近似官方末期 σ 分布，非官方配方」，并把 `moss_image.md` 的 UNVERIFIED 标注带进界面提示 | `arXiv:2610.23374` Table 2；σ 分位（σ=t/T）`shift` 0.39 vs `linear`/`sigmoid` 1.04；赋值路径 `moss_image.py:344` |
| MOSS `model.name_or_path` | Base（正确） | 保留；若界面按 registry 预置填了 SFT，需要拦一下——**`moss_image` 预置是推理默认，不是训练底座** | `USERGUIDE.md:5,126-132`；`registry.py:59-60` |
| Krea2 `train.timestep_type` | `linear` | 保留 `linear`，但注明「官方无训练口径、社区二手」；`shift`（1024² 时 mu≈0.91）作为对照写进注释 | `krea-ai/krea-2 README:70-72,110`；`krea2.py:87-97` |

- 这三条都是**默认值**，不是校验规则：`validate_config` 不因 `timestep_type` 取 `linear` 或 `shift` 报错（白名单内即可），只对白名单外的值报错。
- 需要阿配改代码的只有一处：MOSS 默认值的注释与界面提示文案。`shift` 的数值不用动。

## 六、待阿配实现：训练底座变体校验（MOSS Base / Krea2 Raw）

**现状：这两类误用没有任何校验能拦。** `loraforge/configs/problems.py` 的 `_resolve_arch_hint`（`:343-352`）只看家族子串——MOSS 判 `"moss-image" in name`，Krea2 判 `name.startswith("krea/") or "krea-2" in name`：

- `OpenMOSS-Team/MOSS-Image-SFT` → 小写含 `moss-image` → **通过**（但它是出图侧，官方建议用 Base 训 LoRA）
- `krea/Krea-2-Turbo` → 以 `krea/` 开头 → **通过**（但官方是 TRAIN on Raw、RUN on Turbo）
- 反过来 `moss_image_base.safetensors`（下划线）→ 不含 `moss-image` → **误杀**（review #11 已记）

`model.family-mismatch`（`:421-427`）只查家族，所以「用户照 ai-toolkit registry 预填 `moss_image`」会拿到 SFT，一路通过。

**建议：新增一条 `model.train-variant`，级别 `warn`（不要 `error`）。** 理由：kohya README 明确对 Base 与 SFT 都支持 LoRA（`:47,67`），训 SFT 不必然失败，只是偏离官方推荐；判 error 会变成假阳性。

| 项 | 建议 |
|---|---|
| code | `model.train-variant` |
| 触发 | `target == MOSS` 且小写 `name_or_path` 含 `sft`；或 `target == KREA2` 且含 `turbo` |
| 级别 | `warn` |
| message（MOSS） | `model.name_or_path 指向 MOSS-Image-SFT；官方建议用 MOSS-Image-Base 训练 LoRA（SFT 是出图侧）。见 USERGUIDE 第 5 节` |
| message（Krea2） | `model.name_or_path 指向 Krea-2-Turbo；官方口径是 TRAIN on Raw、RUN on Turbo` |
| 顺带修 | `_resolve_arch_hint` 先把 `name` 里的 `_` 和 `.` 归一成 `-` 再匹配，修掉 `moss_image_base.safetensors` 的误杀 |

依据：`OpenMOSS/MOSS-Image USERGUIDE.md:5`（Base 就是给社区训 LoRA 的）、`README.md:143`（SFT 由 Base 微调）；`krea-ai/krea-2 README:22,24,110`（TRAIN on Raw / RUN on Turbo）；`loraforge/configs/problems.py:343-352,421-427`；kohya `docs/moss_image_train_network.md` 对两者的支持表述。

这条与 §五 的「MOSS `name_or_path` 必须 Base」是同一件事：§五 是默认值，本条是校验层。

## 出处

- 源码：`_src/ai-toolkit`（`ai-toolkit@73c84b0`，VERSION 0.13.43，2026-11-27 抓取）；行号引用与 `docs/sources.md` §1 一致
- 模型文档：`docs/models/sdxl.md`、`docs/models/krea2.md`、`docs/models/moss_image.md`
- 官方文档：kohya `docs/sdxl_train_network.md`、`docs/moss_image_train_network.md`（2026-11-27，见 `docs/sources.md` §7）
- 待审代码：`loraforge/configs/problems.py`（`c31a9766…`，PR #15）
