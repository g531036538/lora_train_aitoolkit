# 三份可直接用的 ai-toolkit 配置 + 20 张图集说明

这三份 yaml 由 `loraforge.configs.build_config` 生成，**别手改、改生成器**；
`tests/test_configs.py::test_examples_are_built_by_the_builder` 会核对它们与生成器输出一致。

| 文件 | 训练底模 | 出图底模 | 关键差异 |
| --- | --- | --- | --- |
| [`sdxl.yaml`](sdxl.yaml) | `stabilityai/stable-diffusion-xl-base-1.0` | 同左 | `ddpm`、`vae_dtype: float32`、不设 `min_snr_gamma`，不开量化 |
| [`krea2.yaml`](krea2.yaml) | `krea/Krea-2-Raw` | `krea/Krea-2-Turbo` | `flowmatch` + `timestep_type: linear`，**不是 sigmoid**；训练期采样 52 步、yaml 写 guidance 5.5（生效 4.5 = 本值 − 1） |
| [`moss_image.yaml`](moss_image.yaml) | `OpenMOSS-Team/MOSS-Image-Base` | `OpenMOSS-Team/MOSS-Image-SFT` | `arch: moss_image` + `model_kwargs.uncond_skip_layers: [22]`；采样 50 步 / CFG 3.0 / shift 5 |

## 跑之前改这两行

```yaml
datasets:
  - folder_path: ./datasets/my_character   # 改成你的图集目录
process[0].training_folder: ./output        # 改成你的输出目录（示例里是 ./output）
```

然后（ai-toolkit 仓库根目录）：

```bash
python run.py /path/to/krea2.yaml
```

## 图集：20 张就够，但构成要对

图集**不进仓库**（隐私 + 体积），仓库里只有这份说明。目录长这样：

```
datasets/my_character/
├── 01.txt
├── 01.png
├── ...
├── 20.txt
└── 20.png
```

`caption_ext: txt` 表示每张图配一个**同名** `.txt`。开 `cache_latents_to_disk: true`
后第一次跑会生成缓存，改图后记得删缓存目录（`<training_folder>/_latent_cache`）。

### 20 张怎么分

| 数量 | 拍什么 | 为什么 |
| --- | --- | --- |
| 10 张 | 正面 / 侧面 / 45° 的清晰半身，背景朴素 | 让模型抓到身份本身 |
| 5 张 | 不同场景（室内、街道、户外） | 泛化用；全靠纯色背景会「只有背景会变」 |
| 3 张 | 不同光线（顺光、逆光、夜景） | 避免只认某一种光 |
| 2 张 | 全身 / 稍远构图 | 防止只学到脸 |

### 不要的

- 同一张图裁切出的多个版本（等于只用了几张）
- 多人合影、被别人挡住的脸
- 水印、贴纸、明显滤镜、AI 生成图
- 分辨率低于 1024 长边的图（SDXL 也要 bucket 到 512 起）

### 打标

打标用本仓库的 caption 侧（`loraforge/caption`）。三款模型的打标风格不同：

- **SDXL**：标签式，逗号分隔，例如 `mychar, 1girl, solo, upper body, plain background`
- **Krea 2**：自然语言句子，例如 `A photo of mychar, a young woman standing in a plain room.`
- **MOSS-Image**：偏精确的自然语言，把构图与光线写进去

触发词在 yaml 里是 `trigger_word: mychar`，caption 里出现 `[trigger]` 的位置会由
ai-toolkit 替换成它（`config/examples/train_lora_flux_24gb.yaml:70` 注释）。

## 参数出处

每个数值的出处写在 yaml 头部注释里，形如 `toolkit/config_modules.py:695-703`。
指的都是 [`ostris/ai-toolkit`](https://github.com/ostris/ai-toolkit) 主线源码。
找不到出处的值在注释里标 `UNVERIFIED`（例如 Krea 2 在 Raw 上训练期的采样步数/CFG）。

## 训完怎么验收

用 `loraforge.configs.acceptance`——采样参数（步数 / CFG / mu / shift / 负向提示）
和三提示法（分布内 / 泛化 / 去触发词对照）都在里面，界面直接读
`acceptance_dict(target)`：

```python
from loraforge.configs import acceptance_dict

acceptance_dict("krea2")
# {'verify_model': 'krea/Krea-2-Turbo', 'sampling': {...'mu': 0.90625}, 'prompts': [...], ...}
```
