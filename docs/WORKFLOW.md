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

```bash
python tools/gh.py commit --branch feat/caption-krea2-styler \
  --message "feat(caption): Krea2 styler 丢掉身份词，保留变化词" \
  --files loraforge/caption/stylers/krea2.py tests/test_caption_styler_krea2.py
```

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
| 主管 | `README.md`、`CONTRIBUTING.md`、`docs/ARCHITECTURE.md`、`docs/WORKFLOW.md`、`.github/**`、`tools/**`、`pyproject.toml`、`loraforge/models.py`、合并 PR |

**共享契约**（`loraforge/models.py`、`docs/ARCHITECTURE.md`）只能由主管改。需要改接口 → 开 Issue 说明，主管定。

## 7. 什么算「完成」

- 代码在分支上，测试本地跑通，输出贴进 PR
- 没有 `print` 调试残留、没有注释掉的死代码
- 新增行为有对应测试；配置/参数有出处（源码路径或文档链接）
- 不写「应该可以」「理论上没问题」——**没跑就是 NOT_RUN，并说明缺什么**

## 8. 本地开发

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest -q
```

界面（可选）：

```bash
pip install -e ".[ui,dev]"
python -m app.ui          # http://127.0.0.1:7861
```
