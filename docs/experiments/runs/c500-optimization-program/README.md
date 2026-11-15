# C500 优化实验计划

本目录保存 2026-11-16 起在 C500 64GB / 128GB RAM 实例上的优化实验。每次实验必须单独建立子目录，并保存：

- `manifest.txt`：机器、软件版本、两个仓库 commit、环境变量和启动参数；
- `serve.log`：完整服务日志；
- `benchmark.log` / `benchmark.csv`：官方 benchmark 原始输出；
- `eval.log` / `math_500.json`：官方精度评测（如执行）；
- `summary.md`：逐轮数据、均值、TTFT、缓存命中和结论；
- `status`：`planned`、`running`、`passed`、`rejected` 或 `blocked`。

## 约束

1. 使用官网固定 `vllm serve` 参数，不改 benchmark，不使用量化或投机采样。
2. 优化必须落在提交的 vllm-plugin-FL / FlagGems 代码中，并保留可观察 fallback。
3. 每次实验先保存代码版本和环境，再启动服务；结束后立即把远程日志和结果复制到本地。
4. 只有同时满足正确性、MATH-500 >= 0.95、TTFT 不超过官方限制且端到端有收益的方案，才进入最终组合。
5. 失败方案也要保留证据，不删除远程或本地实验记录。

## 候选方案

- C500 attention：GQA KV 复用、paged decode layout、adaptive split-KV；先 profile，不能假设收益。
- Decode 融合：RMSNorm + QKV + RoPE + KV cache append；只在真实 decode 热点和形状满足时启用。
- MLP：Gate/Up + SwiGLU epilogue；已有 A48/A60 结果作为对照，避免重复实现。
- Sampling：现有 sort-free / top-p 路径的进一步融合；必须验证随机语义和 cudagraph replay。
- GEMM：C500 原生 tile layout / persistent decode shape；不得破坏大 M prefill fallback。
- KV 低比特：仅作为最后的研究性实验，精度不达标不得进入主路径。

方案进入开发前必须有 profile 或 microbenchmark 证据；单项 kernel 目标至少 10% 加速，端到端目标至少 3% 加速。未达到门槛的方案只保留为实验记录，不写成最终优化。
