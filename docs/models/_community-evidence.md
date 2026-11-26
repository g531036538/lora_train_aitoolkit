# 社区证据：Krea 2 与 MOSS-Image 实际怎么训

> 调研日期 2026-11-27。本文件只收「能指到出处」的经验，并把可信度分开标。
> 三档：**一手实测**（作者跑过、给了数字或日志）/ **二手汇编**（作者整理别人的经验，自己未复现）/ **用户帖**（单人单帖，无方法学）。
> Reddit 本条线不可达（`TypeError: fetch failed` + `firecrawl 403`），所以任何「r/StableDiffusion 实测」在本文件里都只能算二手转述。

---

## 0. 一句话结论

- **Krea 2**：社区共识集中在「Raw 训练 / Turbo 验收」和「`linear` + `flowmatch`」两条；失败模式有具体名字（头大身小、采样器漂脸、霓虹染发、概念泄漏），但**没有一个带完整配置的公开实测**能让我们照抄 rank/lr/steps。
- **MOSS-Image**：训练可行性已有**一手实测**（16GB 卡、LoRA 落 6 个线性层、缓存 TE/latent）；社区抱怨集中在「背景/主体糊」和「风格 LoRA 不像」；文字渲染有正面证据。
- 两个模型的社区证据都**不足以**替代我们自己的对照实验，只够用来决定「第一轮先测什么、失败先怀疑什么」。

---

## 1. Krea 2

### 1.1 训练/验收分工（多源一致）

| 说法 | 出处 | 可信度 |
|---|---|---|
| 训练用 `Krea-2-Raw`，出图/验收用 `Krea-2-Turbo` | SGLang cookbook（Raw 非蒸馏 ~52 步 guidance≈4.5；Turbo 蒸馏 8 步 guidance 1.0） | 第三方官方 |
| 「Train LoRAs on RAW and apply them on Turbo」（转述官方 FAQ 原话） | `chengyansen-ai/krea2-lora-training` README | 二手汇编（未见官方原文，Krea 官方站抓取仍失败） |
| 训练时想同时看 Turbo 预览，是一个真实需求 | ai-toolkit issue **#1070**「Krea 2 train on RAW preview on Turbo」（2026-10-01 开） | 用户帖（feature request） |

**对我们的意义**：`sample` 段只能二选一——训练侧监控用 Raw（52 步 / yaml 5.5），验收另行用 Turbo（8 步 / 1.0）。ai-toolkit 0.13.43 没有「训练一套、采样另一套」的开关（issue #1070 仍未实现），所以 `build_config` 无法同时给出两者，只能在 ⑤ 验收里单列一条 Turbo 复跑。

### 1.2 步数：只有一个区间，没有甜点

二手汇编引了 r/StableDiffusion 的几条实测（原帖不可达，无法核对）：

| 数据集 | 观察 | 出处 |
|---|---|---|
| 59 张 | 500 步不可识别 → 900 步约 70% → 1500 步约 90% | 二手汇编 §7 |
| 未给 | 750 步「踢进」，1700 步可用，3500 步继续提升 | 同上 |
| 未给（OneTrainer） | 3000 步已过拟合，甜点 500–700 | 同上 |

**结论**：Krea2 的 `steps` 没有社区公认值，只有「like 大约在 750–900 步开始出现」这一条被多源重复。这和契约默认 2500 不冲突（2500 是上限侧），但**验收时必须回看早期 ckpt**，不能只测最后一个。

### 1.3 失败模式（按「遇到先怀疑什么」排序）

| 现象 | 社区归因 | 处理 | 可信度 |
|---|---|---|---|
| 脸不像、换衣服就漂 | `timestep_type` 用了 FLUX 的 `sigmoid`/`weighted` | 改回 `linear` + `flowmatch` | 二手汇编（多源重复，与源码分支一致） |
| 脸不像 | 采样器 `res_2s` 系列对 likeness LoRA 漂脸 | 出图换 `Euler + simple` | 二手汇编 |
| 头大身小 | 训练集全身图头身比不对 | 全身图头部 ≤ 画面 20%、鞋完整；或先做人脸对齐 | 二手汇编（作者自述实战） |
| 过拟合 / 脸崩 | 步数过高、未开 EMA、未开 DOP | 2500→1500–2000；`ema_config.use_ema: true`（decay 0.99）；`diff_output_preservation: true, class: person` | 二手汇编 |
| 黑发被染成挑染、环境彩光附到头发 | Krea2 在高饱和粉/紫/霓虹光下的偏置 | 训练图与提示词避大范围粉紫霓虹光；显式写「黑发无挑染」 | 二手汇编（作者单次实战，`cysgirl` 数据集 034） |
| 概念泄漏：无触发词也出角色 | 过拟合 / 触发词不唯一 / caption 把身份写死 | 降 steps 或 lr 5e-5、DOP、caption 只写变化；用「无触发词对照」判断 | 二手汇编引 agihunt 实测（RTX5080 16GB、musubi 0.3.4、768px、1152 步，单次运行无 ablation） |
| 日志里数据集出现两次、bucket 还不一样 | `diff_output_preservation` 会额外加载一份数据集算差分损失 | 正常行为，不要当 bug | 二手汇编（与 `SDTrainer.py` 实现一致） |
| INT8 + layer offloading 100% 时 LoRA 完全无效 | 训练照跑，推理强度 1.0–2.0 都无感；关掉 offloading 就正常 | 先别同时开这两个 | 用户帖（ai-toolkit issue **#1057**，2026-09-21，作者称可复现） |
| 启动即 `Couldn't connect to hf-mirror.com` | `model_kwargs.vae_path` / `text_encoder_path` 留空 → 回退 HF repo id 联网 | 两项填本地绝对路径 | 二手汇编（报错文本具体） |
| Windows 上 webui 接管后 `run.py` 秒退 | 单条环境变量 > 32767 字符（accelerate 保存/恢复环境时崩） | 干净环境启动 | 二手汇编（给了 `ValueError: the environment variable is longer than 32767 characters` 原文） |

**注意**：`res_2s` 漂脸、霓虹染发、头大身小这三条都是**出图/数据侧**问题，不是训练配置能修的；写进 ⑤ 验收时要落到「看哪张图、怎么判」。

### 1.4 其他有信息量的点

- **LoKr 有争议**：二手汇编说 Reddit 指南用 factor 16（文件约 LoRA 的 5%、双人同图不串脸），ai-toolkit 社区倾向 4，折中 8；factor 与 rank 反比。我们没有 LoKr 产品需求，记录备查。
- **Krea2 edit 模式**：`arch: krea2` + `model_kwargs.edit: true`，配 `ostris/krea2_turbo_training_adapter`，参考图经 Qwen3-VL 编码后与 prompt 一起进 DiT，约 1750 步掌握编辑概念（二手汇编转述 Ostris 实测）。不在本轮范围。
- **musubi-tuner 路线**：12GB 可跑（预缓存 VAE latent + Qwen3-VL embedding、`--fp8_base --fp8_scaled`、`--blocks_to_swap N`、`--turbo_dit`）。说明「24GB 不够」不是模型问题而是 trainer 问题。
- **谁支持 Krea2**：ai-toolkit、musubi-tuner、DiffSynth-Studio、KONAKONA666/krea-2（二手汇编 §多后端）；**kohya sd-scripts README 全文无 Krea 字样**（2026-11-27 抓取，grep 0 命中）——kohya 不支持 Krea2。

---

## 2. MOSS-Image

### 2.1 训练可行性：有一手实测

`sorryhyun/anima_lora` 的 `project/moss_lora/plan.md`（commit `780028a`，2026-10-18）是一份把 MOSS LoRA 训到 16GB 卡上的工程记录，数字可直接引用：

| 事实 | 数值 / 做法 |
|---|---|
| LoRA 落点 | 每个 block 的 6 个线性层 `q k v o ffn_up_gate ffn_down`，42×6 = 252 个 |
| 精度 | fp32 master 权重、bf16 GEMM、存 bf16 |
| 缓存 | TE 输出（`last_hidden_state` + 真实长度，永不 padding）与 latent 都预缓存，训练时不加载 TE/VAE |
| 显存预算 | DiT 常驻 9.7 GiB；目标「16GB 卡上一小时内训出可辨认结果、峰值 <15 GiB」 |
| Base vs SFT | Base 更「生」、变化多，适合当 LoRA 底座；SFT 更完成、偏 gacha 光感。作者注明这是 5 条 prompt 的印象，不是测量 |
| 权重体积 | Base 27.2 GB（TE 16.8 + DiT 10.2 + VAE 0.25）；SFT 只多一份 DiT，TE/VAE 字节相同 |
| 2 MP 解码 | fp32 VAE decode 在 DiT 未卸载时 OOM，要先卸载 DiT 再解码 |

同一 repo 的 `docs/moss-findings/` 给了两条**对训练有解释力**的测量：

1. **chat template 就是 attention sink**：10 个模板 token（3 前缀 + 7 后缀）在 35/42 个 block 上吸走 ≥25% 头的 ≥30% 图像 query 质量；平均占 0.24–0.34 的图像 query 质量，首尾步最高。`<|im_start|>` 单 token 的质量约是普通图像 token 的 300 倍。「训好的 LoRA 几乎不动这个 sink」——意味着改 caption 长度/格式会先动到这块，不是无关噪声。
2. **caption 长度是直接的步时杠杆**：单流结构下 attention 是 `(N_img + N_txt)²`，每个文本 token 还要付完整 FFN/QKV；1776×1184 + 1.9k token prompt 时，线性 +24%、attention +53% FLOPs。→ 「长 caption 更好」和「步时」在这里是直接冲突的。

**i2i 案例**（用户帖 + 仓库，2026-10-19）：HF discussion #6（作者 sorryhyun）把 `qwen(image, caption)` + `vae(downscale 2)` 作为输入、目标图作为 label，在 5070 Ti 上约 1 小时训出漫画上色的 i2i LoRA。注意这是 **i2i**（参考图走 TE 的 vision tower），不是我们要的 T2I 角色 LoRA；作者自己的探测也证明：图像进 TE 能带回内容和粗布局，但**不是像素对齐**（64px luma PSNR 7.7–12.0，与纯 caption 基线相当），并且把 image-token 行丢掉后直接跑题（说明 DiT 真的在读那些行）。

### 2.2 失败模式

| 现象 | 描述 | 出处 | 可信度 |
|---|---|---|---|
| 背景/主体发糊 | 两个用户分别报告「背景到处都糊」「糊连主体身体也有」；同一套 prompt 在 Krea2 / Z-Image / Chroma / Flux 正常 | HF discussions #9（2026-10-22）、#12（2026-11-03） | 用户帖 |
| 是否「为强调文字而糊化图像」的机制 | 发帖人自己也只是猜；官方未回复 | #9 | 用户帖（假设） |
| 风格 LoRA 训不像、形体崩 | 「style is Not identical like in the images and deformed stuff」 | #12 | 用户帖（无配置） |
| 文字能刻准 | 1024² 无 layout box 的纯 T2I，footer 小字网格逐条引号写出，**每条都拼对**、栅格对齐、大字被两边裁切、人物不压小字 | #14 | 用户帖（带图与 recipe） |

**对我们的意义**：MOSS 的验收不能只测「像不像」，必须单列一张「背景糊/主体糊」的观察项——这是当前唯一被多人重复的现象，而且官方没有解释。

### 2.3 与官方文档对照后仍未知的

- Base 到底是哪一个训练阶段（anima_lora 指出官方技术报告正文说 Stage 3 step 310k，Table 1 又列了 Stage 4）→ 影响「原生分辨率上限」，我们在 moss_image.md 里标了 `UNVERIFIED`。
- 官方训练用「一图多 caption 候选、每条随机选一」（kohya 文档复刻），但社区没人报告过候选数量、长度组合的最优值。
- 长 caption（2k token 级）在训练里的收益/代价，没有公开对照。

---

## 3. 没取得 / 不再尝试

| 目标 | 结果 | 后续 |
|---|---|---|
| Reddit（r/StableDiffusion 等）| `web_fetch` → `TypeError: fetch failed`；`read_page` → firecrawl `403`；重试多次同样 | 不再试；sources.md 记「访问失败」 |
| Krea 官方 model card（`krea/Krea-2-Raw`）| gated `auto`，README HTTP **401** | 主管确认不再尝试，参数留在 `UNVERIFIED` |
| Civitai 训练笔记 | 未访问（本机通道不稳） | 下一轮可选 |
| dataset-deviser issue 区 | 仓库 0 个 open issue（2026-11-27） | 无需再读 |

---

## 4. 这些证据改变了文档里的什么

1. **krea2.md ⑤**：加「无触发词对照」判污染、加「早期 ckpt 先看」、加「采样器漂脸先换 Euler」——这三条以前只有笼统的「过拟合」。
2. **krea2.md ②/③**：`diff_output_preservation` 的「日志里数据集出现两次」是特性不是 bug，写清避免下游误判；INT8+offloading 组合写进「别同时开」。
3. **moss_image.md ③**：caption 长度与步时冲突写进起点表（`max_text_length` 4096 是上限不是目标）。
4. **moss_image.md ⑤**：加「背景/主体发糊」观察项与「文字渲染单测」。
