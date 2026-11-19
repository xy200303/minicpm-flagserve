# 官方口径确认记录（FlagGems 基线 / FlagTree 版本）

本文件记录赛事方对两个环境问题的书面答复，及其对交付与复测的影响。所有时间为北京时间。

## 一、FlagGems 基线

- **2026-11-09 16:54**，赛道二负责人答复：**「以镜像为准」**。
- 含义：天数侧 FlagGems 使用官方镜像内的 `4df52d91`（镜像标记 `v5.3.5`），**不迁移**到公开 tag 对应的 `a7620cc`（两者相差 7 个提交）。
- 我们的执行：交付包的 FlagGems 树即以 `4df52d91` 重放，未合入那 7 个提交；见 `docs/experiments/runs/image-baseline-align/`。

## 二、FlagTree 版本

- **2026-11-09 17:02**，同一负责人：**「FlagTree 建议用新版」**。
- **2026-11-19**，明确天数侧最终统一复验使用 **`flagtree==0.7.1a2+iluvatar3.6`**，并提示**旧版本在部分场景可能引发精度问题**。

安装命令（赛事方给出）：

```bash
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
pip install flagtree==0.7.1a2+iluvatar3.6 $RES
```

- 我们的执行：天数侧交付与复测在 `0.7.1a2+iluvatar3.6` 下完成，记录见 `docs/experiments/runs/bi-flagtree-071a2/`（含升级前后的 `pip list`、内核级数值探针、4k/16k 逐轮数据与 MATH-500 结果）。

## 三、沐曦 C500

- **2026-11-19 14:54**，同一负责人：**「C500 复验的环境与镜像一致，无需额外更新 flagtree」**。
- 含义：C500 侧保持镜像环境，即 `triton 3.0.0+metax3.7.0.7`、未安装 flagtree；**不在源上升级到 `0.7.0.post2+metax3.6`**。
- 我们的执行：C500 的交付与复测（含 11-19 的 `--no-enable-prefix-caching` 官方口径复测）全部在镜像原环境上完成，未改动编译器。

## 四、对报告与 readme 的影响

| 位置 | 处理 |
|---|---|
| 报告环境表「软件包」行 | 天数改为 `FlagTree 0.7.1a2+iluvatar3.6`；C500 注明「镜像环境，无 flagtree」 |
| 报告 §8.3 复现注意事项 | 新增一条记录 FlagTree 升级后的复测结果与差异 |
| readme §0 / §5 | 同步版本与证据目录；补一句官方确认来源与日期 |

以上答复来自赛事方答疑群，日期与原文如上；如需截图原件，见交付包外的沟通记录。
