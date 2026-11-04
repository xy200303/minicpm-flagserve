# A55：A46/A47 host fastpath 与 A48 组合（2026-11-04）

A46 LibEntry 精简与 A47 连续 copy 专用内核在旧 A31 栈的官方 4k/16k 分别为 +0.0049%/+0.1175%，当时未达自定义门槛。当前按“可复现正收益即可合入”的新标准，在 A48 栈上重新验证组合。

FlagGems 分支 `exp/a55-a46-a47-a48-host-fusion`，最终提交 `51b097c2`，由当前集成 `3642366c` cherry-pick `1be4ae77`、`59a73960`、`f20e89d3`、`bd3ff824`。插件使用当前 A48 `4ea4a6e`。候选使用独立 `VLLM_CACHE_ROOT=/workspace/a55-compile-cache`。

## 正确性

相关测试为 **123 passed、1 skipped、3 deselected**，见 `gems-tests.log`。完整套件中另有一个 FlagTune/corex 不支持错误和后段 signal 场景退出；相同命令在当前 A48 基线也复现，见 `baseline-flagtune-test.log`、`baseline-full-tests.log`，不归因于 A55。

先执行一组官方 16k A/B 筛查；只有组合收益可复现才补第二组、4k 和 MATH-500。回滚为恢复 FlagGems `3642366c` 与当前 A48 缓存。
