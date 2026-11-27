# 团队工作流

这个仓库由一支 Bot 团队开发。本文件说明**怎么拿到代码、怎么提交、怎么开 Issue、怎么算交付**。
违规（越界改文件、无测试就报完成、跳过 Issue）一律打回。

## 0. 环境约束（先读，否则会浪费一小时）

本机 `git clone` / `curl` / `git push` **全部失败**：

```
schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS (0x8009030e)
```

`git -c http.sslBackend=openssl` 也不行（本机未编译 openssl 后端）。
这是 TLS 凭据链故障，不是网络不通。**Python 与 Node 的 HTTPS 是通的。**

所以：**不要试 git，直接用 `tools/gh.py`。** 它走 GitHub HTTP API，功能等价于 clone / commit / push / PR。

仓库是 **private**，所有请求都需要环境变量 `DSH_SECRET_GITHUB_TOKEN`（细粒度 PAT，权限见下）。

| 权限 | 级别 |
| --- | --- |
| Contents | Read and write |
| Issues | Read and write |
| Pull requests | Read and write |
| Metadata | Read（必选，自动带上） |

## 1. 开工前：同步代码

```bash
cd <你的工作目录>
python tools/gh.py fetch --dest .
```

`fetch` 会拉当前 main 的全量文件覆盖到本地。**开工前必做一次**，否则你测的是旧代码。

不要手改 `.git/`（本地没有 git 仓库）。

## 2. 开 Issue（每个任务一条）

先查有没有同题：

```bash
python tools/gh.py issues
```

没有就开一条，写清**你要解决什么、卡在哪、你打算怎么做**：

```bash
python tools/gh.py issue-create --title "caption: Krea2 styler 如何处理身份词" --body-file note.md --labels "area:caption"
```

标签用：`area:research` / `area:caption` / `area:config` / `area:ui` / `question` / `blocked`。

主管会在这条 Issue 下给开发指导。**拿到指导再动手**，尤其涉及接口变更时。

## 3. 开发：分支 + 本地测试

分支命名 `<type>/<area>-<slug>`，例如：

```
feat/caption-krea2-styler
feat/config-moss-image
fix/ui-export-zip-name
docs/research-sdxl-caption
```

```bash
python tools/gh.py branch feat/caption-krea2-styler --base main
```

写代码后**必须本地跑通测试**：

```bash
python -m pytest -q
```

没装的依赖按需装（`pip install -e ".[dev]"`）。测试失败不许提交。

## 4. 提交

**中文提交信息必须走 `--message-file`。** 本机控制台 locale 不是 UTF-8，中文经命令行参数传给 Python 会变成乱码并永久写进提交历史。

```bash
# 先把提交信息写进 UTF-8 文件
cat > .msg <<'EOF'
feat(caption): Krea2 styler 丢掉身份词，保留变化词

身份交给触发词；写进打标词会让换装时脸跟着漂。
EOF

python tools/gh.py commit --branch feat/caption-krea2-styler \
  --message-file .msg \
  --files loraforge/caption/stylers/krea2.py tests/test_caption_styler_krea2.py
```

纯英文信息用 `--message` 也可以。Issue 正文同理：中文一律 `--body-file`。

`--files` 只列你**自己写集内**的文件。跨写集 = 打回。

## 5. 开 PR

```bash
python tools/gh.py pr --head feat/caption-krea2-styler --title "feat(caption): Krea2 styler" --body-file pr.md
```

PR 描述必须包含：

1. **改了什么**（一两句）
2. **为什么这么改**（依据：源码位置 / 官方文档 / Issue 编号 / 实测数据）
3. **怎么验证的**（测试命令 + 真实输出，不是「应该没问题」）
4. **交付三数**：每个新增/修改文件的 `bytes` / `lines` / `sha256`

```bash
python -c "import hashlib,sys;[print(p, len(open(p,'rb').read()), sum(1 for _ in open(p,'rb')), hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]) for p in sys.argv[1:]]" 文件1 文件2
```

主管审 PR → 通过则合并；不通过会把理由写在 PR 里。

## 6. 写集边界（硬约束）

| 角色 | 可写 |
| --- | --- |
| 研究员 | `docs/models/**`、`docs/sources.md` |
| 打标工程师 | `loraforge/caption/**`、`tests/test_caption*` |
| 配置工程师 | `loraforge/configs/**`、`configs/examples/**`、`tests/test_configs*` |
| 界面工程师 | `app/**`、`loraforge/export/**`、`tests/test_app*`、`tests/test_export*` |
| 主管 | `README.md`、`CONTRIBUTING.md`、`docs/ARCHITECTURE.md`、`docs/WORKFLOW.md`、`.github/**`、`tools/**`、`pyproject.toml`、`loraforge/models.py`、`loraforge/caption/base.py`、`loraforge/testing/**`、`tests/conftest.py`、`tests/test_contracts.py`、合并 PR |

**共享契约**（`loraforge/models.py`、`loraforge/caption/base.py`、`docs/ARCHITECTURE.md`）只能由主管改。需要改接口 → 开 Issue 说明，主管定。

## 6.1 写测试的两个坑

- **不要用 `tmp_path`**。本机系统临时目录落在沙箱可写范围外，会报 `could not create numbered dir with prefix pytest-`。用 `tests/conftest.py` 里的 `scratch` fixture（仓库内 `.scratch/`，自动清理），或者 `images` fixture 拿三张假图片。
- **不要断言控制台打印**。本机 locale 不是 UTF-8，中文 print 出来是乱码，但数据本身可能是对的。要看内容就写文件再用 read 工具读。

- **检出后先查有没有反斜杠路径**。仓库曾有一批名字里含 `\` 的重复文件（`'loraforge\configs\base.py'` 这种），是早期提交路径没过正常分隔符留下的。在 Windows 上它们与真实文件**同路径、互相覆盖**，检出后你看到的是旧内容，测试会莫名变红。已在 `33373565` 从主分支删除；此后拉代码若发现某文件内容不对，先查这个。拉树时跳过路径里含 `\` 的条目。

## 7. 什么算「完成」

- 代码在分支上，测试本地跑通，输出贴进 PR
- 没有 `print` 调试残留、没有注释掉的死代码
- 新增行为有对应测试；配置/参数有出处（源码路径或文档链接）
- 不写「应该可以」「理论上没问题」——**没跑就是 NOT_RUN，并说明缺什么**

## 8. 本机环境事实（先看这条，能省很多弯路）

**可以从 PyPI 装包了（2026-11-27 起）。** 会话权限从 `workspace-write` 提到 `danger-full-access` 后：

- `python -m ensurepip --upgrade` 成功，**`pip 26.2.1` 可用**
- 已装 **`onnxruntime 1.32.0`**（Python 3.14 有 abi3 轮子）
- 装包仍要在会话内做，但**不许**把「装了什么」当成代码依赖的借口：核心功能必须在不装任何可选包时 import 正常、测试全绿

Python 是 **3.14.7**，已预装 138 个包。与本项目相关的：

| 有 | 没有（可装） |
| --- | --- |
| fastapi 0.136、uvicorn 0.48、jinja2 3.1、python-multipart 0.0.29 | **gradio**（不装，界面已定 FastAPI） |
| pillow、numpy、opencv-python、scipy、pyyaml | **transformers、accelerate**（VLM 用，量大，需要再议） |
| httpx、requests、pydantic、starlette | openai SDK |
| torch 2.12、torchvision 0.27、safetensors、einops、tqdm | pandas |
| pytest、huggingface-hub、**onnxruntime 1.32** | —— |

推论：

1. **界面用 FastAPI + Jinja2 + 原生 JS**，不是 Gradio。
2. **本地 tagger 现在能跑**：onnxruntime 已装，权重由用户放在 `LORAFORGE_WD_MODEL_DIR`（`model.onnx` + `selected_tags.csv`）。开发机权重在 `models/wd-swinv2-tagger-v3/`、`models/wd-eva02-large-tagger-v3/`（**不入库**，见 `.gitignore`）。
3. **远程 HTTP 后端仍然保留**：没有权重、没有网、或用户想用更强 VLM 时走它。
4. 可选依赖一律**延迟 import**：缺了就把 `availability()` 的 reason 写成 `missing_dependency:<name>` / `model_weights_missing`，**不许 import 就炸**。

## 8.1 跑测试

```bash
python -m pytest -q
```

不需要装任何东西，从仓库根跑即可。测试**不许联网、不许碰 GPU、不许下载模型**。

## 8.2 跑界面

```bash
python -m app      # http://127.0.0.1:7861
```

## 9. 机械闸门

GitHub Actions **暂时用不了**：当前 PAT 没有 `Workflows` 权限，往 `.github/workflows/` 下写文件会被 403 拒绝（`Resource not accessible by personal access token`）。CI 配置已写好放在本地，等权限到位再推。

所以合并前的机械闸门由主管在本地执行：

```bash
python -m pytest -q          # 全绿才合并
```

成员在 PR 里必须贴自己的 pytest **真实输出**；主管合并前会在 PR 分支上重跑一遍，对不上就打回重做。

## 9.1 交付说明怎么写（改了什么，必须自己说）

提交信息和 PR 正文的第一段，**先列清楚这一轮动了哪些文件、每个文件改了哪一节、为什么改**。

指纹表是**核对**用的，不是**汇报**用的——主管会拿它验 bytes/lines/sha256 是否与你的声明一致，但
「这个文件变了」不该由主管自己 diff 出来才发现。曾经发生过：交付说明写了三件事，实际改了四个文件，
第四个是文档内容大幅扩写，只能从指纹变化里看出来。这类交付一律退回补说明。

反过来也不许含糊：「按需调整了若干处」这种写法等于没写。

