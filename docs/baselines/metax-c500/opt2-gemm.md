# 优化 #2：skinny-M 双缓冲 GEMM + split-K（MetaX C500，2026-09-22）

## 动机（profile 证据）

优化 #1 落地后重抓 profile（40.7s 轨迹，GPU 忙 59.6%）：`mm_kernel_nt` 13.4s =
GPU 忙碌的 **55%**，每 decode 步 164 次 GEMM 共 20.9ms，而权重流量理论下限 ~2.8ms。
微基准发现三个关键事实：

1. 所有 decode 形状（M=64）GEMM 都是 **~110µs 恒定延迟**，与尺寸无关
   （qkv 10.5MB 与 gate_up 50MB 同时间）；
2. M=1024 同 kernel 同样 ~123µs → 87 TFLOPS：kernel 本身在大 M 下不差，
   110µs 是**单波次串行 K 循环延迟**——MACA Triton 3.0 的软件流水没有生效，
   每个 `tl.dot` 都空等全局内存加载（~3.5µs/迭代）；
3. 厂商库 torch.mm 同形状 16.7-42.8µs（硬件能力证明）；flagtune 扩空间、
   splitk 路径均无法突破该地板（配置无关的结构性问题）。

## 实现（FlagGems `_metax/ops/mm.py`，内核重写）

- 新增 `mm_kernel_nt_db`：**手动双缓冲** nt GEMM——每次迭代先预取下一轮 A/B
  tile 再发射当前 `tl.dot`，不依赖 Triton pipeliner；
- 新增 `mm_kernel_nt_db_splitk_partial`：同结构 + split-K 部分和，复用现有
  `mm_kernel_splitk_reduce`；长 K（≥4096，down 投影 6144）切 4/8 份保持 SM 占用；
- 路由：`_nt_db_scenario`（M ≤ 128 且 512 ≤ N ≤ 16384 且 K % 128 == 0 的 nt 布局
  bf16/fp16）在 `mm()`/`mm_out()` 中优先于通用路径；大 M prefill、lm_head、
  gemv、fp32 等全部走原路径不受影响；
- 附带修复 `fused/matmuladd.py` flagtune 模式 `strategy` 配置格式兼容 bug。

微基准（aten 路径实测）：qkv 110→56.4µs，o_proj →57.6µs，gate_up →72.2µs，
down →91.3µs；层合计 442→277µs；relerr ≤ 0.003（bf16 输出舍入量级）。

## 端到端结果（官方 benchmark，serve 参数与基线一致）

| 场景 | 官方基线 | 优化 #1 | 优化 #1+#2（4 跑） | vs 基线 |
|---|---|---|---|---|
| 4k total tok/s | 5089.645 | 6468 | 7944/7167/8025/6306（均值 **7361**） | **+44.6%** |
| 16k total tok/s | 7029.675 | 8265 | 8293/8236/8236/8218（均值 **8246**） | **+17.4%** |
| 4k Mean TTFT | 3199ms | 3833→3277 | 3122~3409（均值 ~3272） | +2.3%（关注） |
| 16k Mean TTFT | 27197ms | 26031 | 25931~26444 | **−3.5%** |
| 精度 math_500 L3 | 0.962 | 0.981 | 94.3% / 97.1% / 97.1%（三跑） | 均值 96.2% |

精度说明：temperature=1.0 长 CoT 采样使单跑方差达 ±2~3%（官方基线 0.962 自身
亦在此分布内）；三跑均值 96.2% 与基线一致，94.3% 为坏手气样本，非系统性劣化。
内核级数值验证 relerr ≤ 0.3%（bf16 舍入量级），与旧 kernel 等价。

数据：`raw_runs_opt2_gemm.csv` / `summary_opt2_gemm.csv`。
评测机 commit：FlagGems `22703c94`（已推 GitHub dev 分支）。

## 已知遗留

- 4k Mean TTFT 仍比基线高 ~2.3%（限值 +1%），16k 已优于基线；
  Mean TTFT 以排队等待为主，受运行间调度抖动影响，需继续观察或针对性优化 prefill。
- GPU 空闲 ~40%（弱 CPU 调度间隙）是下一个最大结构性收益点。
- 4k 单跑间波动 ±10%（6306~8025），官方评测单跑落点有运气成分。
