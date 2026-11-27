# 第一个 LoRA：从图集到出图

> 对应 `main` 的 `d9da6418`（2026-11-27）。界面文案、按钮、报错、yaml 字段都按这一版的实际代码写；以后改版以代码为准。
>
> 每一步分两段：**你会看到什么**、**看到什么说明出错**。参数不在这份手册里发明——三款模型的完整字段表在 `docs/models/sdxl.md`、`docs/models/krea2.md`、`docs/models/moss_image.md`；训练完怎么判断往哪调，在 `docs/guide/iterating.md`。
>
> 标 **经验值** 的是社区做法、本项目未实测；标 **UNVERIFIED** 的是有出处但未实拍的数字。两者都不要当成定论。

---

## 0. 全程一页

先准备图集（§1），然后：

1. 启动界面（§2）
2. 选目标模型（§3）
3. 清点目录（§4）
4. 打标（§5；想用远程视觉模型看 §6）
5. 逐张校对（§7）
6. 选显存档位（§8）
7. 预览配置与校验（§9）
8. 渲染验收清单（§10）
9. 导出 zip（§11）
10. 拿到云上训练（§12）
11. 训练完回来判读（§13）

全程只有两件事会写文件：打标写 `.txt`，导出写 zip 和 `train.yaml`。界面不上传图片，除非你自己开了远程后端（§6）。

---

## 1. 图集怎么准备（动手之前）

这一步错了，后面每一步都在放大错误。下面每条都标了依据：前四条里的张数与多样性来自 HF 官方博客的作者实测，其余是经验值或竞品规则。

| 项目 | 建议 | 依据 |
|---|---|---|
| 张数 | 15–40 张；20 张高质量通常够 | 经验值。HF 官方博客作者实测：人脸 LoRA **6–10 张**就够，质量 > 数量（`docs/sources.md` §6） |
| 构图 | 约 40% 特写、30% 半身、30% 全身 | 经验值。HF 实测「特写 + 全身都要有（泛化姿势），避免人物太远」（`sdxl.md` §③） |
| 变化 | 背景、服装、姿势/角度、光照都要变 | HF 实测「背景/服装/姿势/表情/光照要多样」（同上） |
| 同一套衣服 | **不要每张都出现** | dataset-deviser README 规则（竞品规则，非实测） |
| 别人的脸 | 训练集里不要出现第二张人脸 | HF 实测「训脸时不要出现别的脸」（同上） |
| 道具 | 会被学成角色的一部分；要么先命名，要么别每张都有 | 竞品规则 + 去触发词对照判据（`sdxl.md` §④） |

四条硬性约定（都来自代码，不是建议）：

- **图片全部放一个目录的顶层**。扫描不递归，子目录会被列进「已跳过」（`loraforge/export/scan.py:47-82`）。
- **只认 `.jpg/.jpeg/.png`**，大小写不敏感。webp / heic / avif 先转成这三种。
- **一张图配一个同名 `.txt`**（`a.png` → `a.txt`）。同一目录里 `a.png` 和 `a.jpg` 会共用 `a.txt`，导出时报「同名冲突」。
- 文件名尽量只用 ASCII、不带空格——**经验值**：云端解压和 `folder_path` 会少一层转义麻烦。

还有两条**经验值**：原图长边别小于 1024（再小就是放大伤细节）；先把模糊、重复、带别人水印的图删掉，别指望训练时「稀释」。

最后先想好一个**触发词**。它要满足：全小写、没有语义、底座基本没见过（`mytoken`、`ohwx` 这类），不要用 `1girl`、`portrait` 这种本来就有含义的词。同一底座下训不同角色/风格，要换不同的词。

---

## 2. 第 1 步 · 启动界面

```bash
pip install -e .
python -m app
```

打开 <http://127.0.0.1:7861>。

**你会看到什么**

终端打印 `Uvicorn running on http://127.0.0.1:7861`；浏览器里是标题 **LoRAForge**，下面六个区块，从上到下：

1. 目标模型（三张卡片：训练底座 / 验收底座 / 打标风格 / 默认步数 / rank / alpha / 学习率 / 分辨率）
2. 打标后端（`wd_tagger`、`openai-http` 两行，带「可用 / 不可用」和说明）
3. 显存档位（8 / 12 / 16 / 24 / 48 GB 五行）
4. 图集与打标
5. 逐张校对
6. 训练参数与导出

**看到什么说明出错**

- `ModuleNotFoundError: No module named 'fastapi'`（或 `uvicorn`）→ 依赖没装，回上面第一条命令。
- `[Errno 10048]` / `error while attempting to bind on address` → 7861 端口被占。端口是写死的（`app/__main__.py:7-8`），先关掉旧的 `python -m app`。
- 浏览器「拒绝连接」→ 地址必须是 `http://127.0.0.1:7861`；界面只绑本机，不暴露公网（`README.md:52`）。

---

## 3. 第 2 步 · 选目标模型

选错模型，打标格式、yaml 字段、验收采样会一起错。一眼判断：

| 你要的效果 | 选它 | 理由 |
|---|---|---|
| 通用角色 / 画风 / 物件，生态最大、教程最多 | SDXL | 标签式语料与底座训练分布一致，24 GB 就能跑 |
| 写实质感、复杂指令、整句描述 | Krea 2 | 12.9B 单流 DiT + Qwen3-VL 文本编码器，吃自然语言 |
| 海报 / 包装 / 招牌，图里有字要精确 | MOSS-Image | 唯一把「图内文字 + 布局」当一等公民的底座 |

三款的训练底座和验收底座**不是同一个**，这是最容易被用错的地方（`loraforge/models.py:44-78`）：

| 模型 | 训练用 | 出图用 |
|---|---|---|
| SDXL | `stabilityai/stable-diffusion-xl-base-1.0` | 同一个 |
| Krea 2 | `krea/Krea-2-Raw` | `krea/Krea-2-Turbo` |
| MOSS-Image | `OpenMOSS-Team/MOSS-Image-Base` | `OpenMOSS-Team/MOSS-Image-SFT` |

### 3.1 SDXL

**什么时候选**：角色、画风、道具、构图，或者你只是想把整个流程先跑通。它是最省事的一档，24 GB 默认档就能训。

**要准备什么**：不需要额外权重；训练和出图用同一个 base。yaml 里 `vae_dtype: float32`（SDXL 的 VAE 半精度会坏，`docs/models/sdxl.md` ②）。

**打标词长什么样**：Danbooru 风格、逗号分隔、全小写、触发词在最前。wd_tagger 打完标后由 styler 自动整理成这样：

```text
mytoken, 1girl, solo, long hair, blue eyes, school uniform, outdoors, sunlight
```

**最容易用错**：

- 标签堆到 75 个以上。CLIP 文本编码器上限 77 token，超出部分**静默截断**（`toolkit/train_tools.py:402-422`），写在后半段的描述白写。校对器按「标签 ≤75、英文词 ≤60」做近似提示（`loraforge/caption/checks.py:45-46,178-201`），这是估算，不是精确 tokenizer 计数。
- 身份词和服装词混着写。SDXL 没有 Krea 那种身份过滤，写了就学；身份交给触发词，衣服写「这次这张图穿的」。
- 把一张图里所有细节都列进去。只写会变化的，固定描述挪到采样提示词。

### 3.2 Krea 2

**什么时候选**：要写实、要复杂指令、要整句描述而不是标签串。

**要准备什么**：

- 训练底座 `krea/Krea-2-Raw`，出图用 `krea/Krea-2-Turbo`。官方口径 **TRAIN on Raw, RUN on Turbo**（`models.py:59-66`）。
- `krea/Krea-2-Raw` 是 gated 权重：云端首次下载要先登录 Hugging Face 并接受许可，否则 401。
- 生成的 yaml 默认开 `qfloat8` 量化（`loraforge/configs/base.py:258-261`）；显存更小能不能跑，属工程建议、未实测（`plans.py` 档位备注）。
- 文本编码器**不训**：yaml 里 `train_text_encoder: false`，Krea 2 这条路径没有 TE LoRA 目标（`krea2.md` ⑤）。
- `timestep_type` 保持 `linear`，不要从 FLUX 搬 `sigmoid`（`models.py:64`、`krea2.md` ②）。

**打标词长什么样**：1–3 句英文描述，只写这张图里变化的部分；身份（脸型、发型、肤色、眼睛）不写——styler 会把这些从句整段剥掉（`loraforge/caption/stylers/krea2.py`）。

wd_tagger 是标签器，打完标 styler 只能把标签用逗号拼起来，得到的是「逗号句」：

```text
mytoken, a single young woman, school uniform, sunlight, outdoors, solo.
```

能用，但不是 Krea 要的 prose。想要整句，用远程视觉后端（§6）或手改：

```text
mytoken, a young woman in a navy school uniform stands in a sunlit classroom,
looking at the camera, soft light from the window.
```

**最容易用错**：照搬 SDXL 的标签串；写脸型/发型/肤色；每张都写同一套衣服（会被焊死在角色上）；改 `shuffle_tokens`（yaml 里是 `false`，按逗号重排会把句子拆碎）。

### 3.3 MOSS-Image

**什么时候选**：海报、包装、封面、招牌——图里有字，且字的位置/层级重要。

**要准备什么**：

- 训练底座是 **Base**，不是 SFT。官方口径 Base 没做推理后训练/蒸馏，就是给社区训 LoRA 的；SFT 用于出图（`docs/sources.md` §2）。
- 图内文字要**逐行给、加引号、标 role**。
- yaml 里 `quantize: true` + `qtype: convrot8`、`model_kwargs.uncond_skip_layers: [22]` 已经由工具填好（`configs/base.py:305-320`）。
- 需要精确位置时，要**手写布局 JSON**。界面里只有纯文本编辑框，没有可视化 bbox 编辑器——这不是坏了。JSON 的正式 schema 与坐标约定（中心原点、`yxyx`、绝对像素）见 `docs/models/moss_image.md` ① 的「TEXT_LAYOUT 边界」；styler 另外要求每条边长是 16 的倍数（`loraforge/caption/stylers/moss.py:70-100`）。

**打标词长什么样**：固定四段。

```text
Object: mytoken, a young woman in a school uniform
In-image text:
"CLASS OF 2026" (role: title)
Appearance: navy uniform, sunlight
Composition: outdoors, medium shot
```

没有文字时写 `In-image text: none`。缺的段 styler 会补 `not described`，不会报错。

**最容易用错**：

- 拿 SFT 当训练底座。能用但不符合官方口径；如果你手改 yaml 改成 SFT，配置校验会给一条 `model.train-variant` 黄条，不拦导出（`loraforge/configs/problems.py:554-567`）。
- 段名拼错或顺序乱。四段是 `Object` / `In-image text` / `Appearance` / `Composition`，工具不硬校验，写乱了只会悄悄教坏。
- 图内文字没加引号，或一行塞两段文字。MOSS 要求「一行一条、引号包住、带 role」。
- 以为 bbox 会被校验。ai-toolkit 侧对 JSON **零解析**，caption 原样喂给文本编码器（`moss_image.md:53`）；坐标写错不会报错。

---

## 4. 第 3 步 · 清点目录

在「图集与打标」里填**本机绝对路径**（例：`D:\datasets\my_character`），点 **清点目录**。这一步不写任何文件。

**你会看到什么**

- 「目录」后面是解析后的绝对路径；
- 「训练图 / 已有标注」两个数字，不等时是黄色，相等时绿色；
- 一张表：文件名 / 大小（字节）/ 已有 `.txt`；
- 如果有被跳过的文件，会多一块「已跳过（非 .jpg/.jpeg/.png）」并逐个列名。

**看到什么说明出错**

- `清点被拒绝（参数）` + `图集目录不能为空` 或 `图集目录不存在或不是目录：…` → 路径打错，或那个目录在本机确实不存在（界面与命令行必须是同一台机器）。
- 训练图 0 张，并提示 `目录里没有 .jpg/.jpeg/.png，打标无从跑起` → 后缀不在三种之内，或者图片在子目录里（不递归）。
- 「已跳过」里出现你以为是训练图的文件 → 看后缀；`.webp` 不认。
- 训练图数少于你放进去的图 → 只算了顶层，去子目录里找。

---

## 5. 第 4 步 · 打标

仍然在「图集与打标」，填完这些再点 **开始打标**：

- **图集目录**：和上一步同一个。
- **目标模型**：§3 选的那个。
- **数据集类型**：下拉里显示的是英文 key——`character` / `style` / `concept` / `text_layout`。含义：角色（身份交给触发词，只写变化）、风格（写画面内容，不写风格词）、物件/概念（写语境与关系）、文字排版（图内文字要精确）。
- **触发词**：§1 想好的那个。
- **打标后端**：默认 `wd_tagger`；不可用的项会带「（不可用）」且选不动。
- **覆盖已有 .txt**：**默认勾选**（`app/templates/index.html:239-240`）。勾着重跑，会把你手改过的标注整份覆盖，且没有二次确认。
- **重建后端**：换了模型权重才勾；否则会拿旧 session 打新图。

**你会看到什么**

- 一行「后端」；一行「写入 / 失败 / 跳过」的三个数字；
- 有失败时，列出每张图和后端给的原始原因，并出现按钮「只重跑这 N 张」；
- 每张成功打标的图，在同目录生成**同名 `.txt`**（UTF-8、LF）。图片本身不动。

**看到什么说明出错**

- 打标表单里的后端下拉不可选、后台上方「打标后端」表里那一行写 `model_weights_missing` → 本地 WD 模型没配：需要 `LORAFORGE_WD_MODEL_DIR` 指向一个含 `model.onnx` 和 `selected_tags.csv` 的目录（`loraforge/caption/backends/wd_tagger.py:91-107`）。没有模型就走 §6。
- `打标被拒绝（参数）`：`目录里没有可打标的图（只认 .jpg/.jpeg/.png）：…` → 回第 3 步。
- 写入 0、跳过 = 全部 → 你在「覆盖已有 .txt」没勾的情况下重跑了一遍（已存在的标注被跳过，不会报错）。
- 失败原因里出现 HTTP 4xx/5xx → 远程后端的问题，不是图的问题；先 `LORAFORGE_VLM_URL` / `MODEL` / `KEY` 三项对一遍。
- 你手改过的标注突然全没了 → 就是「覆盖已有 .txt」默认勾选干的。**以后重跑前先取消勾选，或先备份目录。**

> 一个容易误解的点：wd_tagger 输出的是 Danbooru 标签，但**保存的 `.txt` 不是它的原始输出**——中间过了 styler，会按目标模型改写成 SDXL 标签、Krea 逗号句、或 MOSS 四段（`loraforge/caption/stylers/`）。所以同一张图换目标模型重打，写出来的格式不一样。

---

## 6. 可选 · 用远程视觉模型打标（需要 API key）

默认是本地 `wd_tagger`，它只出标签。想让 Krea 2 / MOSS 得到整句描述，或本机没有 WD 模型，可以换 OpenAI 兼容的 Vision API。

启动界面**之前**在同一个终端里设两个必填变量、一个可选变量（`loraforge/caption/backends/openai_http.py:24-29`）：

```bash
# Windows PowerShell
$env:LORAFORGE_VLM_URL   = "https://your-endpoint/v1"
$env:LORAFORGE_VLM_MODEL = "your-vision-model"
$env:LORAFORGE_VLM_KEY   = "sk-..."     # 本地无鉴权端点可以不设
python -m app
```

**你会看到什么**

「打标后端」表里 `openai-http` 那行从「不可用」变成「可用」，说明列显示 `ok`；下拉框里能选中它。打标流程与本地完全一样，只是每张图会发一次网络请求。

**看到什么说明出错**

- 仍然不可用，说明列写 `未配置 LORAFORGE_VLM_URL` → 变量没设，或设在了另一个终端。改环境变量后要**重启** `python -m app`。
- 说明列写 `未配置 LORAFORGE_VLM_MODEL` → URL 有了，模型名没给。
- 说明列写 `缺少 httpx；无法调用 OpenAI 兼容端点` → 环境里没有 `httpx`，`pip install httpx`。
- 打标时每张都失败、原因带 401 → KEY 不对；404 → URL 少了 `/v1` 或模型名不对；超时 → 端点慢，调大 `LORAFORGE_VLM_TIMEOUT`。

**隐私提醒**：选了远程后端，图片会离开本机发给那个端点。别人的脸、未公开的稿子，先想清楚再开。

---

## 7. 第 5 步 · 逐张校对

在「逐张校对」里填同样的四项（目录 / 模型 / 类型 / 触发词），点 **载入校对**。

**你会看到什么**

- 顶部汇总：`红 N 黄 N 正常 N`，以及筛选按钮 全部 / 仅问题 / 仅红 / 仅黄；
- 每张图一张卡片：缩略图、文本框（就是 `.txt` 的内容）、右上角一个 `红 · code` 或 `黄 · code` 的徽章、徽章下面一行中文原因；
- 直接改文本框，点 **保存这张** → 状态变成「已写入 N 字节：<路径>」，并立即按新内容重新判级。

判级只有三档：**error 红**必须改，**warn 黄**看情况，**info 正常**。红黄来自 `loraforge/caption/checks.py`，界面只负责上色，不另造规则。

三条真实输出（照抄自代码，不是示意）：

```text
[红] 打标词为空；空打标会静默毁掉训练，必须重新生成
[红] 缺少触发词 "mytoken"；身份/风格会学不进去
[黄] 标签数 80（软上限 75）、英文词数 62（软上限 60） 超过 SDXL 近似上限，CLIP 可能在 77 token 处静默截断；…（后略）
     这是近似估算，不是精确 tokenizer 计数
```

**看到什么说明出错**

- `校对加载失败（参数）` + `目录里没有可校对的图（只认 .jpg/.jpeg/.png）：…` → 目录不对。
- 缩略图裂开 → 图片被移走或改名了；`.txt` 还在但图没了，重新清点目录。
- 红项是空标 → 这张图没生成 `.txt`，或生成后被人清空。重新打标，或直接在框里写。
- 红项是缺触发词 → 触发词和打标时用的不一致（复制过来少了/多了，或大小写不同）。先统一触发词；触发词在 caption 最前面，超长截断不会把它挤掉。
- 黄项「出现在 >80% 的图上」→ 服装/道具被每张都写，删掉或改写进变化词。
- 黄项「打标词过短」/「标签数只有 N 个」→ 描述太稀，补画面信息。
- 保存后徽章没变 → 点的是别的卡片；一次只保存当前这张。

> 两条**当前界面看不到**的检查：Krea 的「已移除 N 项身份描述」和 MOSS 的「第 N 行图内文字未加引号」。它们要拿打标后端的原文比对，而校对接口没把原文传下来（`app/main.py:262-284`）。所以这两项自己看：Krea 的 `.txt` 里不该出现脸型/发型/肤色；MOSS 的每行文字都该在引号里。

---

## 8. 第 6 步 · 选显存档位

在「显存档位」表里按自己的显卡挑一档：

| 档位 | 量化 · batch · 分辨率上限 | 说明 |
|---|---|---|
| 8 GB | qfloat8 · 1 · 512 | 工程建议，未实测：只跑小图；SDXL 自动加 `low_vram`，Krea2/MOSS 大概率跑不动 |
| 12 GB | qfloat8 · 1 · 768 | 工程建议，未实测：单图 768；SDXL 加 `low_vram` |
| 16 GB | qfloat8 · 1 · 1024 | 工程建议，未实测：单图 1024；SDXL 加 `low_vram` |
| **24 GB** | qfloat8 · 1 · 1024 | **默认档**：SDXL 不量化；Krea2/MOSS 按官方口径开 qfloat8 |
| 48 GB | 不量化 · 2 · 1024 | 不量化的对照档 |

切换档位只改批量、分辨率上限和量化，不改 rank / alpha（界面文案，`loraforge/configs/plans.py:42-95`）。三款的默认训练参数都是 **2500 步 / rank 32 / alpha 32 / lr 1e-4**。

**你会看到什么**

选完档位不会立刻有输出；下一次预览或导出时，yaml 里的 `datasets[0].resolution` 和 `train.batch_size` 会随档位变。量化不随档位变：SDXL 从不量化，Krea 2 / MOSS 的 builder 固定开量化（`configs/base.py`）。

**看到什么说明出错**

- 训练刚开始就 OOM → 档位高于实际显存，或分辨率上限选大了。降一档。
- Krea 2 / MOSS 在 8–12 GB 上失败 → 计划表里写明「大概率跑不动」，不是配置写错。
- SDXL 在 8–16 GB 上额外出现 `low_vram: true` → 正常，工具自动加的（`configs/base.py:214-216`）。

---

## 9. 第 7 步 · 预览配置与校验

在「训练参数与导出」里填：图集目录、导出 zip 路径、目标模型、触发词、显存档位；下面五个可以留空（留空 = 库默认）：训练步数、rank / alpha、学习率、批量。然后点 **预览配置与校验**。

**你会看到什么**

- 「能否导出」：`可以（没有 error）` 或 `不行（存在 error）`；
- 一行 `步数 / rank / alpha / lr / batch`，一行分辨率；
- 问题列表：每条 `[severity] code — message`；
- yaml 全文。

红项会让「生成 yaml 并导出」按钮变灰，改对之后自动恢复。

**看到什么说明出错**

- `参数有误（参数）` + `步数不是数字：'abc'` → 输入框里混了单位或全角字符。
- 「存在 error」→ 逐条读 message。这一版 yaml 大多由库生成，手工能填错的只有数字和路径；触发词为空在这一步**不报错**，要到导出时才拒（§11）。
- warn 不拦导出，但值得看一眼：它会在导出结果里再出现一次。

> **当前版本一个必须知道的限制**：这五个输入框（步数 / rank / alpha / lr / 批量）只影响「预览」和「验收清单」。**导出的 `train.yaml` 不读它们**——导出接口只接收目录、zip 路径、模型、触发词、档位、是否带图六项（`app/main.py:791-800`），yaml 永远是档位默认值。要改成非默认值，导出后直接编辑 `train.yaml`（zip 里一份，zip 同级还有一份，见 §11）。

---

## 10. 第 8 步 · 渲染验收清单

同一个表单，点 **渲染验收清单**。

**你会看到什么**

- 「验收清单 N 字节 / M 行（LF）；导出时作为 `验收.md` 进 zip」；
- 下面渲染出五个小节：`## 0. 出图用哪个模型`、`## 1. 采样参数`、`## 2. 三提示法`、`## 3. 判定口径`、`## 4. 参数出处`；
- 「查看 Markdown 原文」可以展开纯文本。

**看到什么说明出错**

- `验收清单渲染失败（参数）` → 和预览同一类参数错误，先修表单。
- 触发词空 → 拒绝。验收清单里的提示词需要 `[trigger]` 占位。

这份清单的作用是**训练完照着测**：第 0 节告诉你用哪个底座出图（Krea 2 用 Turbo、MOSS 用 SFT），第 1 节给采样参数，第 2 节是三提示法。三提示法的定义与判读在 `docs/guide/iterating.md` §1–§3。

---

## 11. 第 9 步 · 导出 zip

填好「导出 zip 路径」（必须以 `.zip` 结尾，建议绝对路径），确认「把训练图一起打进 zip」（默认勾选），点 **生成 yaml 并导出**。

**你会看到什么**

成功时：

```text
导出完成                                  D:\datasets\my_character.zip
图片 / 打标                               20 / 20
```

不干净时标题变成 **「已打包，但有问题需要处理」**，并可能多出：

- 「缺打标词的图」（红）——zip 里照样有这张图，但没有 `.txt`；
- 「没有对应图的 txt」（黄）——孤儿标注，不进包；
- 「同名冲突」（红）——`a.png` 和 `a.jpg` 共用 `a.txt`；
- 「校验提醒（不阻止导出）」——配置的 warn 项。

zip 里的结构（`loraforge/export/core.py:132-205`）：

```text
my_character/
  train.yaml
  frame_01.png
  frame_01.txt
  ...
  验收.md
```

同时会在 **zip 同级目录**写一份 `train.yaml`（`app/main.py:838-842`）。要改配置，改 zip 里那份或同级这份都行；如果重新导出，它会按表单值重新生成、覆盖你改过的内容。

**看到什么说明出错**

- `导出被拒绝（参数）` + `触发词不能为空：没有触发词，LoRA 学不进身份` → 回 §1 想一个。
- `导出被拒绝（参数）` + `导出路径要以 .zip 结尾：…` → 补后缀。
- `导出被拒绝（校验）` → 配置有 error，按 message 改；这是唯一会真正拦住导出的东西。
- 缺打标词的图 → **不会拦住导出**，但训练时会缺这一张的语义，必须在云上训之前补。这是最容易带着病往下一步走的地方。
- 同名冲突 → 改文件名（`a.png` 或 `a.jpg` 二选一）。
- `图集目录里没有可训练的图：…` → 目录里没有图片。

---

## 12. 第 10 步 · 拿到云上训练

不指名平台。任何能跑 CUDA 的机器都行，要点只有这些：

- 显存按 §8 的档位准备；24 GB 是默认档，Krea 2 / MOSS 别低于 24 GB（工程建议，未实测）。
- 有 CUDA 环境，装的是 ai-toolkit（本产品不装它，只给文件）。
- 把 zip 传上去解压，得到一个 `<图集目录名>/` 文件夹。

然后改 `train.yaml` 两处，**两处都必须改**：

```yaml
training_folder: /workspace/output          # 原来是 /path/to/output
datasets:
- folder_path: /workspace/my_character      # 原来是本机路径，例如 D:\datasets\my_character
```

- `training_folder`：LoRA 权重、样图、日志的输出目录。ai-toolkit 会在它下面建一个以 `config.name` 命名的子目录（`jobs/process/BaseTrainProcess.py:45`）。
- `folder_path`：解压出来的那个图集文件夹在云上的绝对路径。
- 首次运行会按 `model.name_or_path` 下载底座权重。Krea 2 的 Raw 是 gated，机器要先登录 Hugging Face 并接受许可，否则 401。

**你会看到什么**

训练启动后，日志里会周期性出现 step 数、loss、以及每 250 步存一次盘；`training_folder/<name>/` 下会出现 `config.yaml`、`samples/`、`.safetensors`。

**看到什么说明出错**

- `Dataset not found` / `folder_path` 相关的错 → 上一条没改，或改成了 zip 路径而不是解压后的目录。
- 401 / gated repo → 没登录 HF 或没接受 Krea 2 许可。
- OOM → 回 §8 降档，或把 yaml 里的 `resolution` 列表改小。
- 跑起来了但样图全是噪声 → 先看 `samples/` 的提示词里 `[trigger]` 有没有被替换成你的触发词；没替换说明 yaml 的 `trigger_word` 和 caption 不一致。

---

## 13. 第 11 步 · 训练完怎么判读

把 `training_folder/<name>/` 里的 `.safetensors`（LoRA 权重）和 `samples/` 拿回来，按 `验收.md` 第 0–2 节出图：

1. 用**验收底座**（Krea 2 是 Turbo，MOSS 是 SFT，SDXL 同一个），不是训练底座。
2. 采样参数照 `验收.md` 第 1 节，不要自己另设一套。
3. 三提示法：分布内（触发词 + 训练集同款构图）、泛化（触发词 + 没见过场景/服装）、去触发词（同提示不带触发词，应回到底座原有分布）。

然后按 `docs/guide/iterating.md` 判症状：过拟合、欠拟合、脸漂、概念泄漏、全景好特写崩。**一次只改一个变量**，改完记一行。

**看到什么说明出错**

- 训练底座直接出图 → 出图配方不对，先换验收底座再判断 LoRA 好坏。
- 去触发词对照也出了训练集的衣服/背景 → 概念泄漏，不是「学得好」。
- 触发词完全没反应 → 先确认 caption 里真的有触发词（§7 红项），再怀疑训练。

---

## 14. 常见错误速查

| 现象 | 最可能的原因 | 怎么做 |
|---|---|---|
| 页面打不开 | 没启动，或端口被占 | 看终端；关掉旧的 `python -m app` |
| 后端不可用，`model_weights_missing` | 没配 WD 模型目录 | 设 `LORAFORGE_WD_MODEL_DIR`，或改远程后端（§6） |
| 打标写入 0 张 | 目录没图 / 图在子目录 | 回清点目录（§4） |
| 手改的标注被覆盖 | 「覆盖已有 .txt」默认勾选 | 重跑前取消勾选，先备份 |
| 红：打标词为空 | 没生成或被人清空 | 重新打标或手写 |
| 红：缺触发词 | 触发词与打标时不一致 | 统一触发词（大小写也要一致） |
| 黄：标签数超 75 | 堆了太多通用标签 | 删通用标签，保留区分度高的 |
| 导出「已打包，但有问题」 | 缺标 / 同名冲突 / 孤儿 txt | 按列表逐条补或改名 |
| 云端 401（Krea 2） | gated 权重没接受许可 | 登录 HF 并接受许可 |
| 云端找不到数据集 | `folder_path` 还是本机路径 | 改成解压后的云上路径 |
| 训练 OOM | 档位或分辨率超显存 | 降档（§8） |
| 出图不像 | 先别怪训练 | 按 `iterating.md` §2 排除采样/触发词 |

---

## 15. 出处

代码引用（本仓库，`main` `d9da6418`）：

- 界面与按钮：`app/templates/index.html`（六个 section、表单字段、按钮、结果文案）。
- 启动与端口：`app/__main__.py:7-8`。
- 扫描规则：`loraforge/export/scan.py:47-82`。
- 后端清单与原因前缀：`loraforge/caption/pipeline.py:113-181`、`loraforge/caption/reasons.py`。
- 本地 tagger 的模型目录与依赖：`loraforge/caption/backends/wd_tagger.py:24-29,91-107`。
- 远程后端的环境变量：`loraforge/caption/backends/openai_http.py:24-29,61-70`。
- 打标判级与文案：`loraforge/caption/checks.py:45-46,126-201,205-330`。
- 三款 styler：`loraforge/caption/stylers/{sdxl,krea2,moss}.py`。
- 模型与训练/验收底座：`loraforge/models.py:44-78`。
- 显存档位与默认参数：`loraforge/configs/plans.py:42-95`。
- 配置生成：`loraforge/configs/base.py:90-330`。
- 导出与打包：`loraforge/export/core.py:132-205`、`app/main.py:791-876`。
- 校对接口未传原文：`app/main.py:262-284`。
- 验收清单五节：`loraforge/configs/acceptance.py:409-500`。

外部来源（URL + 访问日期见 `docs/sources.md`）：

- HF 官方博客（SDXL LoRA 进阶脚本，作者实测）——`docs/sources.md` §6，用于「6–10 张脸/质量优先/构图与多样性」。
- kohya-ss/sd-scripts 官方文档——`docs/sources.md` §7，用于 SDXL 的 VAE 精度与 MOSS 的复刻口径。
- MOSS-Image 官方 `USERGUIDE.md` / `README.md`——`docs/sources.md` §2，用于 Base/SFT 分工、布局 JSON schema、采样参数。
- ai-toolkit 源码（0.13.43，commit `73c84b0`）——`docs/sources.md` §1，用于 `training_folder/<name>/` 结构、77 token 截断、量化字段。
- dataset-deviser README——`docs/sources.md` §9.3，用于「同一套衣服别每张都有」这条**竞品规则**（非实测）。

**可信度说明**：§1 里只有「6–10 张 / 别出现别的脸 / 特写+全身 / 多样性」是 HF 作者实测；「15–40 张、20 张够、40/30/30」是经验值。档位备注全部是「工程建议，未实测」。Krea 2 的 24 GB 量化结论来自部署侧记录，训练侧未实测。§3.2 的 yaml 5.5 / 生效 4.5 仍标 `UNVERIFIED`（见 `krea2.md` ⑤）。
