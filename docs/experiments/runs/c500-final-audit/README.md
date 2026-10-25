# C500 当前交付复测记录（2026-10-25）

## 代码与实例

- 插件：`0207b7e10f93dfa6e912b00c372f268d618b4c2c`
- FlagGems：`aff68555e266582f68c6554898f4c795c869578d`
- 原生基线：插件 `13eb9be`，FlagGems tag v5.3.5（`a7620cc1`）
- GPU：MetaX C500 64GB；CPU 内存 120GiB
- 交付包 SHA-256 与环境：`manifest.txt`

## 端到端启动结果

交付版与原生基线分别使用独立源码和 venv，均按官网 serve 参数启动。两者都在初始化阶段停止前进，等待 1200 秒仍未监听 `/v1/models`，未进入官方 benchmark，因而没有任何有效的新端到端吞吐、TTFT 或 MATH-500 数据。

- `serve-startup-failure.log`、`driver-startup-failure.log`：交付版启动记录。
- `baseline-serve-startup-failure.log`、`baseline-driver-startup-failure.log`：原生基线启动记录。
- 两次失败发生在相同初始化阶段，不能归因于交付优化。
- 未通过关闭预热、修改 serve 参数或改 benchmark 来绕过。
- 简单 CUDA 张量运算可以执行，说明并非 GPU 完全不可用；完整 serve 的环境故障仍需健康实例或官方镜像修复后复测。

## 不依赖 serve 的微基准

脚本：`docs/verification/c500_micro_audit.py`；原始输出：`micro.json`、`micro.log`。计时使用 CUDA event，表中为 30 次中位数。

### top-p 采样

形状：B=64、V=130560、bf16、temperature=1.0、top_p=0.95。PyTorch 对照在 `eager_top_p` 中实现，包括 fp32 softmax、排序、累积质量掩码、归一化和 multinomial；当前实现为 `fused_top_p_sample(..., zoom_rounds=1)`。

| 路径 | 中位耗时 | 均值 | 最小–最大 |
|---|---:|---:|---:|
| 同语义 PyTorch top-p | 4.170240 ms | 4.173909 ms | 4.140032–4.274432 ms |
| 当前 fused top-p | 0.335616 ms | 0.339029 ms | 0.329728–0.383232 ms |
| 加速比 | **12.43×** | — | — |

该 12.43× 只对应本次明确写出的对照实现，不能与早期不同“整条采样链”定义混用。

### lm_head GEMM

形状：M=64、K=2048、N=130560、bf16。`native_lm_head` 在启用 FlagGems 全局注册前测量；随后执行 `flag_gems.enable()` 测 Triton，最后直接调用 `mcblas_mm`。

| 路径 | 中位耗时 | 均值 | 最小–最大 |
|---|---:|---:|---:|
| 原生 `torch.mm` | 0.381824 ms | 0.382123 ms | 0.378112–0.394496 ms |
| FlagGems Triton | 0.592128 ms | 0.593229 ms | 0.585216–0.618752 ms |
| 直接 mcBLAS | 0.403712 ms | 0.405180 ms | 0.399872–0.426752 ms |

mcBLAS 相对 FlagGems Triton 快 **1.47×**，但相对原生 `torch.mm` 慢约 5.7%。`micro-native-first.json` 是未区分原生与 Triton 的初步运行，已由上述完整三路结果取代，不能单独用于报告结论。

## 收尾

所有 serve 和微基准进程已停止，`mx-smi` 无进程，显存回落至 826 MiB。完整端到端复测尚未完成。
