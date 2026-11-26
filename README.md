# LoRAForge · ai-toolkit 多模型 LoRA 训练准备台

为一件事而生：把你手上的一批图，变成 **ai-toolkit 能直接开跑的打标词 + 训练配置**，并针对 **SDXL / Krea 2 / MOSS-Image** 三款模型各自的提示词习惯与训练特性，走不同的路子。

本项目**不安装、不驱动 ai-toolkit，也不训练任何模型**。它只产出两类文件：

1. 每张图旁一个同名 `.txt` 打标词
2. 一份 ai-toolkit 训练用 YAML 配置

你把这两样连同图集一起提交到云端，用 ai-toolkit 启动训练即可。

---

## 为什么不能一套打标走天下

三款模型的文本侧差别很大，同一套打标词喂进去，效果会差一截：

| | SDXL | Krea 2 | MOSS-Image |
|---|---|---|---|
| 打标风格 | 短标签为主，逗号分隔，触发词在前 | 描述性自然语言，只写「变化的」 | 详细自然语言；图内文字用引号，位置要求可用 JSON 布局 |
| 身份处理 | 触发词承载身份，其余描述 | 触发词承载身份，**身份特征不进打标词** | 同 Krea 2 原则 |
| 训练基底 | `stable-diffusion-xl-base-1.0` | `krea/Krea-2-Raw`（Turbo 只用于验收） | `OpenMOSS-Team/MOSS-Image-Base`（SFT 用于出图） |
| 关键参数 | 待研究校准 → [docs/models/sdxl.md](docs/models/sdxl.md) | `timestep_type: linear` + `noise_scheduler: flowmatch`（**禁止从 FLUX 照搬 sigmoid**） | 采样 30 步 / CFG 3.0 / shift 5；Base 需 `uncond_skip_layers: [22]` |

完整推导、证据与来源见 `docs/models/`。

---

## 工作流

```
图集(images/)
   │
   ├─ 1. 选目标模型 (SDXL / Krea 2 / MOSS)
   ├─ 2. 选数据集类型 (角色 / 风格 / 物件 / 文字排版)
   ├─ 3. 批量打标  ── 可插拔后端: ONNX 标签器 / 本地 VLM / OpenAI 兼容 API
   ├─ 4. 人工校对  ── 逐图过一遍，改完即存
   ├─ 5. 生成配置  ── 按模型 + 显存 + 数据量给出一份可跑的 YAML
   └─ 6. 导出打包  ── images/ + *.txt + train.yaml 一个 zip
```

---

## 安装与运行

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
python -m app
```

打开 <http://127.0.0.1:7861>。界面只绑本机，不暴露公网。

> 需要 GPU 的后端（本地 VLM）是可选项；纯 CPU 也能用 ONNX 标签器 + API 后端完成全部流程。

---

## 目录结构

```
lora_train_aitoolkit/
├── app/                      # 操作界面（本地 Web）
├── loraforge/                # 核心库
│   ├── caption/              # 打标：后端适配器 + 每模型模板 + 批处理
│   ├── configs/              # ai-toolkit YAML 生成与校验
│   └── export/               # 打包导出
├── configs/examples/         # 生成好的示例配置
├── docs/
│   ├── ARCHITECTURE.md       # 架构与接口契约（改代码前必读）
│   ├── models/               # 三模型打标与参数策略
│   └── sources.md            # 结论的证据来源
└── tests/
```

---

## 与 ai-toolkit 的关系

- 配置字段对齐 ai-toolkit 主线（2026-11，Krea 2 / MOSS-Image 已内置 `extensions_built_in/diffusion_models/`）。
- 本项目只生成配置，**不启动训练**；训练在你自己租的云 GPU 上跑。
- 配置里的模型路径用占位符，导出前在界面里填你自己的云端路径。

---

## 协作

开发流程见 [CONTRIBUTING.md](CONTRIBUTING.md)：**问题先在 Issues 提，主管给指导，成员开分支做，PR 由主管审**。
