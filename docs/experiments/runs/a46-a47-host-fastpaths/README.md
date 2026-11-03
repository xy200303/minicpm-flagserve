# A46/A47：LibEntry 与连续 copy host 快速路径（2026-11-03）

## 动机与实验分组

A31 的 16k host profile 中，`LibEntry.run`、key 构造和动态 copy 包装有大量重复 Python 工作。为减少设备启动次数，将相关改动作为一个实验包，在同一设备窗口完成正确性测试和交错微基准；官方性能仍单独验证，不以微基准代替结论。

实验分支：`xy200303/FlagGems` 的 `exp/a46-a47-host-fastpaths`。

- A31 对照：`9c06143c`，远端 `/workspace/a31-gems`。
- A46：`1be4ae77`，预计算参数名、复用 key helper、减少 descriptor 判断、普通 dict 保序、缓存命中少一次哈希、grid 参数少一次字典重建。远端 `/workspace/a46-gems`。
- A47：`59a73960`，在 A46 上为连续、同形状、同 dtype 的 `copy_` 增加 1D Triton 内核，跳过通用动态形状包装；广播、类型转换、非连续和异常路径不变。`f20e89d3` 增加同形状非连续输入的回退测试。远端 `/workspace/a47-gems`。

回滚：恢复 `GEMS_ROOT=/workspace/a31-gems`；只撤销 copy 可 checkout `1be4ae77`，撤销全部实验可 checkout `9c06143c`。

## 验证计划

`run_micro.sh` 先对 A46/A47 运行 LibEntry key、decorator cascade、默认参数和完整 `test_copy.py`，再按 A31/A46/A47/A47/A46/A31 顺序运行真实 `copy_` 探针。探针记录实际导入路径、9 组原始时序和数值相等检查。

任务排在 A45 官方 A/B 完全退出后执行，避免 GPU 干扰。结果尚未产生；只有正确性通过且组合微基准有明确收益，才进入官方 4k/16k 交错 A/B。若组合达标，再分别保留 A46 与 A47 做消融。
