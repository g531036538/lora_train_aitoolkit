# 线索草稿（主管笔记，非交付文档）

> 这不是产品文档。立项阶段我手工整理的调研线索，**未经核实**，逐条结论一律以
> `docs/models/` 下三份正式文档为准。留在这里是为了记住「当初为什么这么查」，
> 以及还没做的竞品分析。

# 主管移交的调研种子（未经你复核，不要直接当结论用）

这些是我在立项前扒到的线索。**每一条你都要回到源头核一遍**，核过的写进正式文档并标出处，核不实或对不上的标 `UNVERIFIED` 或直接推翻。
尤其注意：数字类（步数、CFG、lr、rank）最容易被社区文章抄错或已过时，源码优先于博客，官方优先于二手。

---

## 1. ai-toolkit（ostris/ai-toolkit） — 待你核实

- 规模：约 12720 star / 1634 fork / 87 open issues，2026-11-26 仍有更新
- 支持列表里出现了：`krea/Krea-2-Raw`、`krea/Krea-2-Turbo`、`OpenMOSS-Team/MOSS-Image-SFT`、`MOSS-Image-Base`、`stabilityai/stable-diffusion-xl-base-1.0`
- 内置扩展目录：`extensions_built_in/diffusion_models/krea2/`、`extensions_built_in/diffusion_models/moss_image/`
- `toolkit/models/registry.py` 里我看到过（**要你逐行确认**）：
  - `krea2` → Krea-2-Turbo，采样 8 步 / guidance 1.0
  - `moss_image` → SFT，采样 30 步 / guidance 3.0，qtype `convrot8`
  - `moss_image:base` 额外带 `model_kwargs: {"uncond_skip_layers": [22]}`
  - 还有 `moss_image:edit` 变体
- **重要缺口**：`config/examples/` 里没有 sdxl / krea2 / moss_image 的示例 yaml（只有 flux、flux_kontext、flux_schnell、chroma、qwen_image、hidream、lumina、omnigen2、sd35_large、wan、flex）。所以这三款的字段只能从源码推。
- 数据集约定：同目录 `image2.jpg` + `image2.txt`；caption 里可写 `[trigger]` 占位，由 config 的 `trigger_word` 替换；只吃 jpg/jpeg/png
- 可能有用的能力：`only_if_contains` / `ignore_if_contains`（network_kwargs）做分层训练；`type: "lokr"` + `lokr_full_rank` + `lokr_factor`

## 2. Krea 2 — 待你核实

- 定位：Krea 自研首个基础模型（2026-05 发布），12.9B 单流 DiT + Qwen3-VL-4B 文本编码器 + Qwen-Image VAE
- 官方口径：**训练用 Krea-2-Raw，出图用 Krea-2-Turbo**
- 二手来源给的校准参数（来自 `chengyansen-ai/krea2-lora-training`，约 6 star，MIT，有 README.zh-CN.md 和 `references/krea2_train.yaml`）：
  - `timestep_type: linear` + `noise_scheduler: flowmatch` ——**禁止从 FLUX 搬 `sigmoid`**
  - rank/alpha 32/32；lr 1e-4（保守 5e-5）；首轮 2500 步（lr=5e-5 时 3000–4000）；batch 1 / acc 1（保守 acc 2）
  - `train_text_encoder: false`；量化 qfloat8（Transformer 和 TE 都）；分辨率桶 512+768+1024；caption dropout 0.05
  - `diff_output_preservation: true`（class `person`）；`content_or_style: balanced`
  - 数据集 15–40 张，40% 特写 / 30% 半身 / 30% 全身，头部不超过画面 20%
  - 验收：Turbo 8 步 / CFG 0 / mu 1.15；训练中监督 Raw 30 步 / CFG 4；最终 Raw 52 步 / CFG 3.5；三提示法（分布内 / 泛化 / 去触发词对照）
  - caption 模板：`[trigger], single person, [取景], [角度], [表情], [头发], [衣服], [光], [背景], [质感]`
  - **身份特征（脸型、发型、肤色）不写进打标词**，只写每张图变化的东西，否则换装时脸会漂
- Krea 2 提示词特性（官方博客口径）：歧义会让它多样化探索；风格参考最多 4 张、带 strength 滑块；moodboard 能出 taste profile / keywords / avoids

## 3. MOSS-Image — 待你核实

- OpenMOSS 开源，Apache-2.0，双 checkpoint：**Base 供训练 LoRA，SFT 供出图**
- 架构：42-block 单流 DiT，3072 宽，128 latent channels，FLUX.2 VAE（2x2 packed, BatchNorm）
- 接受三种提示形式：自然语言、标签序列、JSON layout
- 文字写法四段式：①对象 ②图内文字（每个词用引号，一行一个，标明 role 如 header/footer）③外观（纸墨字色光）④构图（谁在哪、留白多少）
- JSON layout 结构（**bbox 顺序是 y 先**）：
  `{"summary":{"meta":[...],"canvas":"single","caption":...},"items":[{"label":"text"|"object","caption":...,"text":...,"bbox":[y_min,x_min,y_max,x_max]}],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}`
  原点在画布中心，边长须 16 的倍数
- Base 采样：≥50 步、CFG 3.0（3.0–6.0）、shift 5.0、Euler、输出 1–2MP；负向提示建议 `nsfw, worst quality, bad anatomy, watermark, jpeg artifact`

## 4. 竞品与生态（用来取其精华，不是抄）

- `jhc13/taggui` 1368★ / 2026-11-20 / 80 open issues —— 标签管理 + 多种 captioner，最成熟，但重桌面 GUI
- `fpgaminer/joycaption` 1305★ / 2026-11-25 / 43 open issues —— 专为扩散训练做的 captioning VLM
- `roboflow/maestro` 2695★ —— 多模态微调框架，不是打标工具
- `EnragedAntelope/dataset-deviser` 9★（很新）—— 最接近的竞品：按目标模型切 caption 风格、可插拔 captioner、训练配置覆盖 ai-toolkit/kohya/musubi。**但不支持 MOSS-Image**
- 两条可复用的实战教训：同一套衣服出现在每张图 → LoRA 把服装学成角色；参考图里的道具被复制到每张 → LoRA 把道具学成角色

---

## 你要补的最大空白：SDXL

上面 SDXL 几乎是空的（我只知道它支持 SDXL 底模）。SDXL 的 LoRA 训练在 ai-toolkit 里怎么配（rank/alpha、lr、text encoder 训不训、分辨率桶、noise scheduler、min_snr、VAE 精度），要你从源码 + 社区实测补齐。这是三款里用户最可能先用的一款。
