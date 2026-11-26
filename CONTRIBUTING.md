# 协作流程

本项目由一名主管 + 若干成员协作开发。规则只有一条目的：**同一个文件永远只有一个人写**。

## 1. 一切从 Issue 开始

- 要做什么、为什么、验收标准是什么，先开 Issue（用 `问题/需求` 或 `研究任务` 模板）。
- 主管在 Issue 下给出**开发指导**：范围、边界、验收口径。没有指导不进入编码。

## 2. 分支与提交

- 分支名：`feat/<area>/<短描述>`，例如 `feat/caption/wd-tagger-backend`
- `<area>` 只能是：`caption` / `configs` / `app` / `docs`
- 提交信息用中文，写清「做了什么 + 为什么」

## 3. 写集边界（硬规则）

| 角色 | 只能改 |
|---|---|
| 研究员 | `docs/models/**`、`docs/sources.md` |
| 打标工程师 | `loraforge/caption/**`、`tests/test_caption*` |
| 配置工程师 | `loraforge/configs/**`、`configs/examples/**`、`tests/test_configs*` |
| 界面工程师 | `app/**`、`loraforge/export/**`、`tests/test_app*` |
| 主管 | `README.md`、`CONTRIBUTING.md`、`docs/ARCHITECTURE.md`、`.github/**`、合并 PR |

**越界即打回。** 需要改别人的文件，先在 Issue 里说明，由主管派给对应负责人。

## 4. PR 要求

PR 描述必须写全四项，缺一项主管不审：

1. **范围**：改了哪些文件，为什么是这些
2. **验收**：怎么证明做完了（命令 + 输出，或截图）
3. **依赖**：是否需要别人先合入什么
4. **风险**：哪些地方没测、哪些是猜的

未验证的东西写「未验证」，不要写成「已通过」。

## 5. 接口变更

`docs/ARCHITECTURE.md` 里的接口契约是跨模块的唯一约定。要改契约，先开 Issue，主管确认后由主管改文件，其他人再跟进。

## 6. 真实数据不出仓

图片、训练好的模型、API key 一律不进仓库。`.gitignore` 已覆盖常见路径。
