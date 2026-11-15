# C500 最终代码组合实测记录（warmup off）

- 实验目录：`/workspace/xlang2026-final-c500-warmup-off-20261116-003450`
- 本地记录日期：2026-11-16
- GPU：MetaX C500 64 GiB；CPU RAM 128 GiB；无 swap
- 代码：`vllm-plugin-FL` `b5793cd4477a530eb420c58bf2a4e5d742a8846c`；`FlagGems` `ef5161373bc97c2ccabd1f6d51d92fb2a37ff814`
- 运行时：Driver 3.8.30，MACA 3.7.0.38，PyTorch 2.8.0+metax3.7.0.7，vLLM 0.24.0+empty，Triton 3.0.0+metax3.7.0.7
- 官方服务参数未修改；服务启动额外设置 `VLLM_FL_MM_WARMUP=0`。该变量关闭新增 GEMM autotune warmup，避免启动阶段超过 8 分钟仍未监听；没有改变请求参数、模型、量化或投机采样。

## 4k 官方场景

场景 `[4096,1024,64,256]`，4 轮，全部 `Successful requests=256`、`Failed requests=0`。

| 轮次 | Total token throughput (tok/s) | Mean TTFT (ms) | P99 TTFT (ms) |
|---:|---:|---:|---:|
| 1 | 11819.77 | 3007.92 | 15261.15 |
| 2 | 12460.63 | 1630.03 | 9621.01 |
| 3 | 12525.25 | 1624.91 | 9597.24 |
| 4 | 12411.15 | 1684.80 | 9754.43 |
| **均值** | **12304.20** | **1986.92** | **11058.46** |

## 16k 官方场景

场景 `[16384,1024,64,128]`，4 轮，全部 `Successful requests=128`、`Failed requests=0`。

| 轮次 | Total token throughput (tok/s) | Mean TTFT (ms) | P99 TTFT (ms) |
|---:|---:|---:|---:|
| 1 | 11082.18 | 18056.72 | 67223.12 |
| 2 | 11081.79 | 18061.35 | 67226.48 |
| 3 | 11079.33 | 18064.79 | 67253.56 |
| 4 | 11087.42 | 18044.09 | 67161.75 |
| **均值** | **11082.68** | **18056.74** | **67216.23** |

## 结论与限制

- 本次 4k、16k 官方 wrapper 均完整跑完四轮，退出码为 0，失败请求为 0；原始日志已保存在同目录。
- 当前结果是最终代码组合在 C500 上的真实实测，不使用历史结果替代。
- 与历史 C500 结果（4k 12527.62、16k 11143.77 tok/s；旧 commit）相比，当前 warmup-off 结果暂未显示吞吐提升，4k 和 16k 分别低约 1.8% 和 0.5%。历史结果不能直接视为同一代码版本对照。
- `VLLM_FL_MM_WARMUP=0` 是当前稳定启动所必需的运行开关；默认 warmup=1 的启动成本过高，需在精度验证、天数兼容性和最终交付策略确定后再决定是否修改默认值。
## MATH-500 Level 3 精度

- EvalScope 1.11.1，105 个 Level 3 样本，105 成功、0 错误、结果完整。
- Accuracy：**98.1%（0.981）**，达到官网要求的至少 0.95。
- 输出证据：`final-c500-warmup-off-20261116-003450-math_500.json`、`final-c500-warmup-off-20261116-003450-eval-level3.log`。

天数 BI-V150 复测尚未完成，因此本实验记录不能单独宣布项目最终交付完成。

## 证据文件

- `final-c500-warmup-off-20261116-003450-manifest.txt`
- `final-c500-warmup-off-20261116-003450-benchmark-partial.log`（4k）
- `final-c500-warmup-off-20261116-003450-benchmark-16k.log`（16k）
- `final-c500-warmup-off-20261116-003450-serve.log`
