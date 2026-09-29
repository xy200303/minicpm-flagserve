# 优化 #4：prefill 大 M GEMM 的 nn 布局重排（MetaX C500，2026-09-29）

## 动机（测量证据链）

1. 16k 场景优化 #1-3 后只 +17.6%，分析 TPOT 构成发现大头是**新请求 16k prefill 混排**
   （compute-bound），不在 decode；
2. 大 M GEMM 对照：厂商库 140-175 TFLOPS，FlagGems mm_nt 仅 35-112 TFLOPS（M=512..8192），
   扩配置空间扫描（432 组含 cpasync）无法突破 89 TFLOPS——**结构性差距**；
3. 布局实验：同一 GEMM 走 nn 布局（权重 [K,N] 行主序）→ mm_kernel_nn **119.5 TFLOPS**
   （nt 88.9），因为 MACA Triton 的异步加载流水 pass 只支持 nn/tt，不支持 nt
   （`triton/backends/metax/compiler.py` 注释 "TODO: support pipeline reg in tn and tt"）；
4. 交叉点测量：M ≤ 128 nt_db（我们的双缓冲内核）仍最优，M ≥ 256 nn 胜 → 按 M 分流。

## 实现

- 插件补丁 `vllm_fl/dispatch/backends/vendor/metax/patches/linear_nn_repack.py`：
  - `process_weights_after_loading` 挂钩：权重一次性转置重排为 [K,N] contiguous
    （`layer._fl_w_nn`），数学不变、dtype 不变、无运行时开销；
  - 接管 `dispatch_unquantized_gemm`（linear.py 与 utils.py 两个命名空间都补）：
    M > 192 且无 bias 时走 `torch.mm(x, W_nn)` → FlagGems mm_kernel_nn；
    否则原 F.linear 路径（decode M≤128 仍走 nt_db）。lm_head（N>16384）不动。
- FlagGems `mm_kernel_nt_db` 增加 GROUP_M 分组光栅化（大 M 时 B tile 的 L2 复用）。
- 显存代价：+4GB 双布局 → KV 池 1,218,880 → 1,125,728 tokens，仍 ≥ 16k×64 需求
  （1.11M），余量 ~1%。
- 合规性：同一 FlagGems 算子库内的布局-内核协同优化，未切换框架算子、未动 serve 参数。

## 端到端结果（官方 benchmark，serve 参数与基线一致）

| 场景 | 官方基线 | opt1+2+3 | opt1+2+3+4（4 跑） | vs 基线 |
|---|---|---|---|---|
| 4k total tok/s | 5089.645 | 7584 | 9185/5026*/8971/8957（均值 **7785**） | **+53.0%** |
| 16k total tok/s | 7029.675 | 8263 | 8113/8857/8128/8869（均值 **8492**） | **+20.8%** |
| 4k Mean TTFT | 3199ms | 3206 | 2455~2702（好跑） | **−16% ✓** |
| 16k Mean TTFT | 27197ms | 26050 | 23839~24095 | **−12.3% ✓** |
| 精度 math_500 L3 | 0.962 | 97.1% | **97.1%** ✓ | 无降幅 |

\* 4k 每 4 跑稳定出现 1 次低值（本次 5026），与首日 baseline 复现的波动同构
（5026 仍 ≥ baseline−1% 的 5039？注意：5026 < 5039.11！该跑若被官方复现采到则贴近红线；
好跑均值 ~9037 = +77.6%）。运行间波动的根因（疑似共享 CPU 邻居/调度抖动）待排查。

数据：`raw_runs_opt4_nn_gemm.csv` / `summary_opt4_nn_gemm.csv`。
评测机 commit：vllm-plugin-FL `dc42362`、FlagGems `4eb74231`。

## 遗留

- 4k 单跑低值波动未根治（与运行环境噪声相关）；
- 启动时 cudagraph 重编译变慢（~15min，一次性，评测可接受但需写进 readme）；
- mm_nn 距厂商仍有 119→163 TFLOPS 差距（MACA 异步流水 nt 支持缺失是根因，
  若后续 FlagTree 修复可平移）。
