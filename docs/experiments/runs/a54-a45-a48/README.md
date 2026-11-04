# A54：A45 metadata 缓存与 A48 门控融合组合（2026-11-04）

A45 在旧 A20+A31 基线上已有官方 4k +0.78%、TTFT 改善，但当时被自定义 3% 筛选线排除。当前交付已加入 A48，因此重新评估组合，而不直接沿用旧基线结论。

插件分支 `exp/a54-a45-a48-metadata-fusion`，提交 `0cb107d`，由当前集成插件 `58bae94`  cherry-pick A45 `16aa58d`。FlagGems 使用当前 A48 `f9085b6d`。A45 缓存 decode 行号索引，并将三处高级索引改为 `index_select`；A48 的 MLP 路径保持不变。

A45 两项切分/缓存测试与 A48 三项路由测试合计 **5 passed**，见 `plugin-tests.log`。候选使用独立 `VLLM_CACHE_ROOT=/workspace/a54-compile-cache`，基线使用 A48 缓存。官方 serve 参数、模型和 benchmark 不变。

## 官方 4k 当前进度

A1（当前 A48 栈）四轮汇总为 **5242.14 tok/s、3920.28ms TTFT**。B1/A2/B2 按串行交错执行，最终结果待完整对照后填写。回滚为恢复插件 `58bae94` 与当前 A48 缓存；合入前还需 MATH-500 Level 3 ≥0.95。
