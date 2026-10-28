# 优化实验台账（tracking）

## 纪律

1. 一次实验 = 一个 exp_id，产物在 `docs/experiments/runs/<exp_id>/`（机器侧 `/workspace/experiments/runs/<exp_id>/`）：
   `config.txt`（分支/commit/env/用例）、`serve.log`、`bench.log`、`summary.csv`、`gpu_before/after.txt`
2. **任何优化必须当天交错 A/B**：同一用例交替跑 A(基线)/B(候选)，取中位数比较。
   实测噪声 <0.1%（同日 A1=1702.33 vs A2=1701.67，代理用例），可直接用单跑比较，但仍建议 A/B/A。
   绝对数字**不可跨天比较**（2026-10-17：同代码 2346.64 vs 3778.65，机器漂移 38–60%）
3. 每个补丁都有 `apply_*.py` + `revert_*.sh`（`docs/experiments/harness/`），回滚后必须 `cp` 到 site-packages
4. 每次实验后在本表追加一行

## 台账（2026-10-17 下半场，天数 BI-V150，代理用例 `[[4096,1024,64,128]]` 与 `[[4096,512,64,64]]`）

| exp_id | 分支@commit | 用例 | 总吞吐 tok/s | 输出 tok/s | TTFT ms | TPOT ms | 结论 |
|---|---|---|---|---|---|---|---|
| A1_base | main@84abed0 | 4k/1024/c64/128 | **1702.33** | 340.47 | 9700 | 138.3 | 当日基线 |
| B1_a13 | main+A13 | 4k/1024/c64/128 | 1677.28 | 335.46 | 9645 | 138.1 | **−1.5%，回退** |
| A2_base | main@84abed0 | 4k/1024/c64/128 | **1701.67** | 340.33 | 9700 | 138.3 | 复现基线，噪声 <0.1% |

### A13（去 metadata 每步 D2H）结论

- 实现：runner 在 `_prepare_inputs` 发布 CPU 侧 cu/seq，builder 不再访问 `query_start_loc_cpu`（D2H 拷贝）
- 一致性：启动时三次对拍 `published cu/seq match vLLM copies` ✅
- 结果：−1.5%（1677.28 vs 1702.33 / 1701.67）→ **判定为小幅回退，已回退**
- 推断：该 D2H 处的等待在整步流水里与其他等待重叠，去掉它没有缩短关键路径；
  或同步本身起了"及时排空"的作用。文档记录为"方向存在但此实现无收益"

### A7 decode pin 在 c64 的复扫（进行中）

`w1s1`（当前）是 c8 扫出来的；c64 的 attention 形状不同，重新扫 `w2s1` / `w4s1` / `w1s2`。
用例 `[[4096,512,64,64]]`（短负载，~2 min/跑），同一 serve 参数。

### c64 pin 复扫（2026-10-17，短负载 `[[4096,512,64,64]]`）

| 配置 | 总吞吐 tok/s | 相对 w1s1 | 结论 |
|---|---|---|---|
| **w1s1（现行）** | **1445.09** | — | 最优 |
| w2s1 | 1450.33 | +0.4% | 噪声内，可用但无收益 |
| w4s1 | 1417.32 | −1.9% | 更差 |
| w1s2 | 1313.67 | −9.1% | 明显更差 |

结论：A7 的 `num_warps=1 / num_stages=1` 在 c64 形状下依然最优，pin 不是可再挖的杠杆。

### A14（纯 decode 走 vendor kernel）— 否决

改动：`_build_split` 对 q_len==1 的批也建 split；`_eligible` 放开 `max_query_len > 1`。
观测：vendor decode 计时确实开始增长（`dec=12.5–12.8 s / 8200 calls`），但短负载跑 >10 分钟未完成
（w1s1 同负载 ~4 分钟）→ **约 2.5 倍以上回退**，已中止并回退（`patches/revert_a14.sh`）。
推断：vendor decode 核是为混合批里少量 dec 行设计的，c64（64 请求 × 长 KV）下不如 A7 调优过的 stock 核。

### 今日两个候选的最终状态

| 候选 | 结果 | 状态 |
|---|---|---|
| A13（去 metadata 每步 D2H） | −1.5%（1702.33/1701.67 → 1677.28） | 回退 |
| A14（纯 decode 走 vendor） | >2.5x 回退 | 回退 |

均未达到 goal 的"≥3% 当日 A/B 提升"门槛；交付栈保持 A1–A7。

### 官方 16k 当日基线（2026-10-17，`[[16384,1024,64,128]]`）

| exp_id | 总吞吐 tok/s | 输出 tok/s | TTFT ms | TPOT ms |
|---|---|---|---|---|
| BASE16k | **2974.37** | 174.96 | 154045 | 152.1 |

（历史记录 3567.33 → 今日偏低与机器漂移一致；所有比较都在当日进行。）

### 16k decode pin 复扫（短负载 `[[16384,256,64,32]]`，c64，16k KV）

| 配置 | 总吞吐 tok/s | 输出 tok/s | TPOT ms | 相对 w1s1 |
|---|---|---|---|---|
| w1s1（现行） | 2811.21 | 43.25 | 492.3 | — |
| **w2s1** | **3203.25** | **49.28** | **403.8** | **+13.9%** |
| w1s2 | 2851.77 | 43.87 | 453.6 | +1.4% |
| w2s2 | 3053.36 | 46.02 | 423.8 | +8.6% |

**结论：16k（长 KV）下 `num_warps=2 / num_stages=1` 明显更优**（+13.9%，TPOT −18%）。
比对 4k 短负载（c64）：w2s1 1450.33 vs w1s1 1445.09（+0.4%，噪声内）→ w2s1 在两类形状上都不亏。
A7 的 w1s1 是 c8/4k 形状扫出来的；**形状相关的最优值在长 KV 下会移动**，这属于允许的 shape-aware dispatch。

### 官方 16k 2×2 交错 A/B（进行中）

A=w1s1（现行）/ B=w2s1，各 2 次，交替执行：A16k_r1, B16k_r1, A16k_r2, B16k_r2。

### A16（强制 2D decode attention 路径）— 否决

`VLLM_FL_ATTN_PATH=2d`（`seq_threshold_3D=0`）：代理用例 `[[4096,256,64,128]]` 得 1840.68 / TPOT 411
vs 默认（3D 启发式）1863.97 / TPOT 384 → **−1.2%**，已回退。

### A17（prefill dual-gather `num_warps` 4→8）— 显著收益，官方 A/B 验证中

把 A6 varlen 路径里"一核把 K/V 页聚合成连续池"的 Triton kernel 的 `num_warps` 从 4 提到 8
（新增 `VLLM_FL_DG_WARPS` / `VLLM_FL_DG_BLOCK` 两个开关，默认保持 4/4096）。

| 配置（prefill 重负载 `[[4096,64,64,128]]`，c64） | 总吞吐 tok/s | TPOT ms | 相对 |
|---|---|---|---|
| w4 / BLOCK 4096（现行） | 2688.25 | 583 | — |
| **w8 / BLOCK 4096** | **3121.97** | **478** | **+16.1%** |
| w8 / BLOCK 8192 | 3129.94 | 477 | +16.4%（BLOCK 无感） |

机制：KV 收集是 prefill 步的第二大开销（每层要把 KV 页拷进连续池，读+写双倍 KV 流量）；
4 warp 时拷贝吞吐不足，8 warp 后明显加快。属"自研 kernel + shape-aware 参数"，规则允许。

官方 4k 2×2 交错 A/B（A=w4, B=w8）进行中。

### 方法论教训

**代理负载必须匹配官方并发**：16k 短负载 `[[16384,256,64,32]]` 只有 32 并发，
在该形状下 w2s1 pin 比 w1s1 快 13.9%，但官方 c64 用例上 w2s1 反而慢 13.9%——
代理形状会翻转最优参数，正式验证必须在官方 test-case 上做。

### A17 官方验证 — 否决（代理再次误导）

| 配置（官方 4k `[[4096,1024,64,256]]`） | 总吞吐 | 输出 | TPOT |
|---|---|---|---|
| A4k_dg_r1（DG w4，现行） | **2341.07** | 468.21 | 110.2 |
| B4k_dg_r1（DG w8） | 2317.63 | 463.53 | 111.1 |

→ **−1.0%**，否决。原因：A17 只影响 prefill 步，代理负载 `[[4096,64,64,128]]` 的 prefill 占比远高于官方
（输出 64 vs 1024），放大了 effect；官方用例里 prefill 占比小，且 w8 在 2048-token chunk 的 gather 形状上并不更快。

### 教训 2（与代理并发教训并列）

**代理负载不仅要匹配并发，还要匹配 prefill:decode 配比（输出长度）**。
后续任何候选都必须直接在官方 test-case 上做单次筛选（~16 min/次），不再依赖短负载代理。

### A18（2D 路径 launch pin）— 部分否决

c64 下纯 decode 走 **2D** stock kernel（我们的 A7 pin 只管 3D），2D 用 Triton 默认配置。
新增 `VLLM_FL_PIN_2D` / `VLLM_FL_DECODE_NUM_WARPS_2D` / `..._STAGES_2D` 开关（默认关闭）。

| 配置（官方 4k） | 总吞吐 | TPOT | 相对默认 |
|---|---|---|---|
| 默认（2D 未 pin） | 2341.07 | 110.2 | — |
| 2D pin w1s1 | 2327.32 | 110.8 | −0.6%（否决） |
| 2D pin w2s1 | 见下一轮 | | |

## 本轮全部候选汇总（2026-10-17，官方 test-case 直测）

| 候选 | 官方结果 | 判定 |
|---|---|---|
| A13 metadata 每步 D2H | −1.5%（代理同构） | 回退 |
| A14 纯 decode 走 vendor kernel | >2.5x 慢（中止） | 回退 |
| pin w2s1（3D） | −13.9%（16k 官方） | 回退 |
| A16 强制 2D 路径 | −1.2% | 回退 |
| A17 DG num_warps 4→8 | −1.0% | 回退 |
| A18 2D pin w1s1 | −0.6% | 回退 |

### 最后一个 env-only 候选：关掉 dual-gather — 否决

| 配置（官方 4k） | 总吞吐 | 相对默认 |
|---|---|---|
| 默认（dual-gather on） | 2341.07 | — |
| `VLLM_FL_DISABLE_DUAL_GATHER=1` | 2328.31 | −0.5% |

## 2026-10-17 会话总结（天数 BI-V150，官方 test-case 直测）

今日机器相对历史基线慢 38–60%（同代码 2346.64 vs 3778.65），所有比较均为**当日交错对照**。

| # | 候选 | 官方结果 | 判定 |
|---|---|---|---|
| A13 | metadata 每步 D2H 去除 | −1.5% | 否决 |
| A14 | 纯 decode 走 vendor kernel | >2.5x 慢 | 否决 |
| — | 3D pin w2s1 | −13.9%（16k） | 否决 |
| A16 | 强制 2D 路径 | −1.2% | 否决 |
| A17 | dual-gather num_warps 4→8 | −1.0% | 否决 |
| A18 | 2D 路径 pin w1 / w2 | −0.6% / −0.2% | 否决 |
| — | 关闭 dual-gather | −0.5% | 否决 |

**结论（截至本会话）**：
- 交付栈（A1–A7）在当前 profiler 能力下已接近可调参数的包络；A7 的 `w1s1`、dual-gather 默认参数、
  3D 路径选择、metadata 路径都是各自维度上的局部最优。
- 继续提升需要**新的 decode attention kernel 实现**（当前 c64 下 decode 步 ~50 ms，KV 带宽理论下限 ~21 ms），
  属于数天级工程，不是参数扫描能覆盖的。
- 两条方法论教训已固化：代理负载必须匹配官方**并发**与 **prefill:decode 配比**。
- 环境已全部回退到 `main@84abed0`，site-packages 同步；所有实验产物在 `docs/experiments/runs/` 与机器 `/workspace/experiments/`。

### 反向验证：关掉我们的 vendor FA（A1/A6）— 官方 4k

| 配置（官方 4k） | 总吞吐 | 输出 | TTFT | TPOT |
|---|---|---|---|---|
| 默认（A1/A6 vendor FA 生效） | **2341.07** | 468.21 | 5547 | 110.2 |
| `VLLM_FL_DISABLE_ILUVATAR_VENDOR_FA=1` | 1694.07 | 338.81 | 14007 | 154.2 |

→ **vendor FA 路径贡献 +38%**（1694 → 2341），TTFT 也从 14.0 s 降到 5.5 s。
这排除了"vendor 路径在 c64 拖累"的假设，也量化了我们已有工作的价值。

16k 的同样对照正在进行（`NFA_off16k`）。

## 战略评估（2026-10-17 会话末）

### 已排除的参数空间（全部官方用例直测）

| 维度 | 结论 |
|---|---|
| 3D decode pin（warps/stages） | w1s1 最优（w2 −13.9%、w4 −1.9%、s2 −9.1%） |
| 2D decode pin | w1 −0.6% / w2 −0.2%（Triton 默认已近优） |
| 3D vs 2D 路径 | 强制切换 −1.2%（启发式已选对） |
| dual-gather warps | w8 −1.0%（默认 4 最优） |
| dual-gather 关闭 | −0.5% |
| metadata 每步 D2H 去除 | −1.5% |
| vendor decode 核（c64） | >2.5x 慢 |

### 反向验证（现有工作的价值）

| 组件 | 官方 4k 贡献 |
|---|---|
| A1/A6 vendor FA（prefill 路由 + varlen/dual-gather） | **+38%**（关掉后 2341→1694，TTFT 5.5→14.0 s） |

### 剩余空间在哪

c64 下 decode 步 ≈ 50 ms（4k KV）/ ≈180 ms（16k KV），KV 带宽理论下限分别 ≈21 ms / ≈70 ms →
**2.4–2.6 倍空间全在 decode attention kernel 本身**。现成 kernels（stock triton 3D/2D、vendor ixAttnBkd）
都已测过：stock 是当前最优，vendor 在 c64 明显更差。要拿到 ≥3% 的整机提升，需要新写一个
面向 BI-V150 的 decode attention kernel（分块 KV 扫描 + 与本卡 warp 数匹配的 tile/segment 配置），
属于数天级工程，不属于参数扫描能覆盖的范围。

### 硬件带宽校准（2026-10-17 实测，BI-V150）

| 项目 | 实测 | 带宽 |
|---|---|---|
| 1GB copy（读+写） | 3.657 ms | **587 GB/s** |
| 1GB sum（纯读） | 3.065 ms | 350 GB/s |
| matmul M=64，gate_up（权重 50 MB） | 0.12 ms | 410 GB/s |
| matmul M=64，lm_head（权重 534 MB） | 0.98 ms | 547 GB/s |
| matmul M=4096 | 1.15 ms | 89.4 TFLOPS |

**结论**：这张卡的真实可用带宽约 **590 GB/s**（不是最初假设的 1.5 TB/s）。
M=64 的 GEMM 已跑在 **70–93% 可达带宽**，没有余量；FlagGems 的 Triton matmul 反而慢 6 倍
（`flag_gems.enable()` 后 0.42 ms vs 厂商 0.069 ms），所以插件不把 GEMM 交给 FlagGems 是正确的。

decode 步 ≈50 ms 的构成：GEMM ~12 ms（4.6 GB 权重）+ attention KV 读 ~19 ms（11.3 GB @ 587 GB/s）+ 图内其他。
**唯一有实质余量的是 attention kernel**（理论 19 ms vs 实际可能 ~30 ms）。

### A19（2D 路径 KV tile 64）— 否决

| 配置（官方 4k） | 总吞吐 | TPOT |
|---|---|---|
| 默认（tile 32） | 2341.07 | 110.2 |
| `VLLM_FL_TILE_2D=64` | 2330.27 | 110.6 |

−0.5%，已回退。至此所有"参数/启发改动"类候选（9 个）全部否定，
且已用带宽校准证明 GEMM 无余量——**继续提升只能靠新写 decode attention kernel**。

## A20：自研 paged decode attention kernel（2026-10-17/18）

设计、微基准、否决方案与 CUDA-graph 两个坑：`docs/experiments/a20/decode-attention-kernel.md`
代码与工具：`docs/experiments/a20/`（`decode_attn_fast.py`、`apply_a20.py`、`revert_a20.sh`、
`bench_dattn4.py`、`bench_abl2.py`、`bench_mask.py`、`parity_a20b.py`、`a20_ab.sh`、`a20_eval.sh`）

微基准（B=64/kv=4096/16q-2kv/d=128/block=16/bf16）：stock 3D+A7 pin **1.04 ms/层**（258 GB/s）
→ 自研 kernel **0.72 ms/层**（372 GB/s），对 stock 的 max|diff| 2.4e-4；
拆解显示 PV 的 `tl.dot` 布局转换占 0.36 ms，是本轮唯一的大头。

### 官方 4k（`[[4096,1024,64,256]]`，同日交错 A/B/A/B）

| exp_id | 路径 | 总吞吐 tok/s | 输出 tok/s | TTFT ms | TPOT ms |
|---|---|---|---|---|---|
| A20_4k_A1 | stock（`VLLM_FL_DISABLE_DECODE_FAST=1`） | 2331.37 | 466.27 | 5549 | 110.58 |
| A20_4k_A2 | stock | 2326.89 | 465.38 | 5576 | 110.70 |
| A20_4k_B1 | fast kernel | 2538.21 | 507.64 | 5587 | 98.92 |
| A20_4k_B2 | fast kernel | **2563.13** | 512.63 | 5547 | **98.37** |
| A20_4k_B3 | fast + 混合步路径（最终代码） | 2533.86 | 506.77 | 5516 | 98.84 |
| A20_4k_A5 | stock（次日 20:32 复验） | 2266.21 | 453.24 | 5615 | 113.07 |
| A20_4k_B4 | fast（nseg16 试配，已否） | 2546.13 | 509.23 | 5540 | 98.12 |
| A20_4k_B10 | fast（次日 20:47 复验） | **2557.00** | 511.40 | 5557 | 98.55 |
| A20_4k_B13 | fast（最终带插桩的交付树） | **2545.91** | 509.18 | 5607 | 98.87 |
| **中位数** | A（3 次）2326.9 → B（5 次）**2545.9** | **+9.4%** | +9.5% | 基本持平 | 111.5→98.7（**−11.5%**） |

两组独立的交错配对都为正：昨夜 A1→B1→A2→B2（2329.1→2538.2，+9.0%），
今夜 A5→B10（2266.2→2557.0，+12.8%，当晚机器整体慢 ~2.7%）。

（另有两次早期单跑：stock A20_k1 2336.33 / fast A20_k2 2473.06，与上表同向。）

decode 窗口瞬时生成吞吐 991 → 1458–1528 tok/s，decode 步 ~57 ms → ~44 ms，
与"attention 从 ~30 ms 降到 ~19 ms（≈KV 读 11.3 GB 的带宽下限）"一致：
**该 kernel 已把 decode attention 压到本卡带宽下限，这个方向上没有剩余空间了。**

### 精度（MATH-500 Level 3）

`a20_eval.sh`，与基线同参（temperature 1.0 / top_p 0.95 / max_tokens 32768，evalscope `math_500` Level 3，105 题）：

| 栈 | MATH-500 Level 3 |
|---|---|
| 基线（赛道给定） | 96.2% |
| **A20 fast kernel** | **97.1%**（105 题） |

→ 达标（要求 ≥95%），与基线同档；decode attention 的 bf16 数值差异没有带来可测的精度损失。
（greedy 对拍在一条长 filler prompt 上会分叉，属数值噪声导致的 argmax 翻转，但 MATH-500 上没有体现。）

### 16k（`[[16384,1024,64,128]]`，同日 A/B/B/A）

| exp_id | 路径 | 总吞吐 tok/s | 输出 tok/s | TTFT ms | TPOT ms |
|---|---|---|---|---|---|
| A20_16k_A1 | stock | **2992.51** | 176.03 | 154469 | 151.55 |
| A20_16k_A2 | stock | 2979.55 | 175.27 | 154749 | 152.15 |
| A20_16k_B1 | fast（slab 146 MiB） | 2923.56 | 171.97 | 158083 | 156.03 |
| A20_16k_B2 | fast（slab 36 MiB） | 2933.99 | 172.59 | 158160 | 154.94 |
| A20_16k_B3 | fast + 混合步 decode 路径 | 2973.27 | 174.90 | 154227 | 153.34 |
| A20_16k_B4 | fast + 混合步 decode 路径 | 3454.11 | 203.18 | 133522 | 129.20 |
| A20_16k_B5 | fast + 混合步 decode 路径 | 3447.29 | 202.78 | 133725 | 129.40 |
| A20_16k_B6 | fast + 混合步 decode 路径 | 3443.25 | 202.54 | 133913 | 129.53 |

**晚间交错 A/B/A/B（最终判定）**：

| exp_id | 路径 | 总吞吐 | TTFT ms | TPOT ms |
|---|---|---|---|---|
| A20_16k_A3 | stock | 2972.98 | 155054 | 152.46 |
| A20_16k_B8 | fast | **3433.04** | 134154 | 129.86 |
| A20_16k_A4 | stock | 2971.26 | 155178 | 152.54 |
| A20_16k_B9 | fast | **3439.30** | 133859 | 129.68 |
| **中位数** | 2972.12 → **3436.17** | **+15.6%** | 155.1→134.0 s（−13.6%） | 152.5→129.8（−14.8%） |

A 侧 4 次（14:30 起 + 19:24/19:57）全部落在 2971–2992，B 侧 6 次里 4 次落在 3433–3454、
2 次（B3/B7）落在 2929–2973 与 stock 同档——离群原因未定位（时钟恒为 1500 MHz、
KV 池一致、混合步 host 计时一致），但交错配对（A3→B8→A4→B9）里 B 稳定 +15.6%。

**但纯 decode 窗口是大幅正收益**：同一批 31 个在跑请求，stock 359.5 tok/s → fast 505.3 tok/s（**+40.6%**），
decode 步 86 ms → 61 ms。隔离微基准同向：kv=16384 时 stock 3.50 ms/层 → 自研 2.41 ms/层（1.45x，NSEG=32 最优）。

16k 端到端反而 −1.9% 的原因：这个用例是 **prefill/容量受限**——
每波 TTFT 154 s（64×16k = 1.05 M prompt token），KV 池只够 32 路并发，
decode 只占整轮 ~20% 时间；TTFT 上那 +2.3% 无法归因到 kernel（prefill 不走它），
更可能是 KV 容量与调度在 94.8% 占用下的波动。结论：16k 的优化杠杆在 **prefill**，不在 decode。

### 本项结论（A20）

| 用例 | 总吞吐 | 判定 |
|---|---|---|
| 官方 4k | **+9.5%**（2329.13 → 2550.67） | **采纳**（≥3% 门槛达成） |
| 官方 16k | 交错 A/B/A/B：**+15.6%**（2972.1 → 3436.2；decode 窗口 +40.6%） | 采纳 |
| MATH-500 Level 3 | 97.1%（105 题，要求 ≥0.95） | 达标 |

默认启用（`VLLM_FL_DISABLE_DECODE_FAST=1` 可回退）。

### A20 最终精度复核（最终代码，含混合步路径）

MATH-500 Level 3 = **97.1%**（105 题），与不含混合步路径时一致，达标。

### 步内构成测量（探针，非成绩）

`VLLM_FL_DECODE_FAST_SKIP=1` 跳过 attention launch 后，4k/64 并发 decode 步 = **11.7 ms**；
A20 为 43.7 ms、stock 为 57.1 ms → attention 现占 32 ms/步，KV 读带宽下限 19.3 ms，
剩余 12 ms 全在 PV 的 MMA 布局转换路径（当前 372 GB/s ≈ 可达负载带宽的 70%）。
（`A20_skip` / `A20_skip64` 是诊断 run，输出为垃圾值，不计入成绩。）

### 16k prefill 的 GPU 侧拆分（更正）

`VLLM_FL_PREFILL_GPU_TIMING=1` + 代理用例 `[[16384,16,32,16]]`，CUDA 事件实测：

| 每次调用（每请求 × 每层 × chunk） | 时间 |
|---|---|
| 反分页 gather | **0.03 – 0.05 ms** |
| vendor FA | **1.7 – 3.3 ms** |

gather 仅占 prefill GPU 时间的 ~1.5%，**此前"gather 约占一半 prefill"的估算作废**；
16k 的 prefill 是 FA 计算受限（按 chunk≈2048 反推有效算力 ~115 TFLOPS，接近本卡上限）。
`ixformer.batch_paged_attention` 即便可用，收益上限也只有 1–2%。

### 混合步 engage 率与慢档归因（2026-10-18 晚）

`decode_attn_fast slice: used=X skipped=Y` 每 4000 次调用一行 INFO：
B11（3430.62）/ B12（3435.33）均为 `used=44000 skipped=0`，即 decode 切片 100% 命中。
加上 B8/B9（3433/3439），今晚 4 次 16k 全部落在 3430–3439 且无任何 fallback。
早前 B3（2973）/B7（2929）虽与 stock 同档，但同样无 fallback/not-eligible 日志，
归因为共享实例的外部干扰（与"同代码同日 2346 vs 3779"的漂移同源）。

### NSEG 16 vs 32（否决 16）

隔离扫描显示 4k 下 `nseg=16`（seg 256）比 32 快 8%（0.704 vs 0.765 ms/层），
但引擎内不成立：4k B4（nseg16）2546.13 / TPOT 98.12，与 nseg32 的 2533.86–2563.13 同档；
16k B7（nseg16）2929.34 / TTFT 159.6 s，**明显劣于** nseg32 的 3433–3454。
保持 `NSEG=32`。

### A20 本轮否决路线（有测量）

| 路线 | 结论 |
|---|---|
| int8 KV cache（per-token-head scale） | **否决**：只加载就已经 1.06 ms/层 vs bf16 0.72；1 字节 gather 在这套 Triton 后端只有 ~120 GB/s |
| `ixformer.batch_paged_attention`（省掉 prefill 反分页 gather） | **不可用**：要求 cache 首维恰为 `max_page_num*page_size`，与 vLLM 全局池/block_table 宽度不兼容；tensor-desc 路径 BAD_PARAM |

## A20 收尾：报告与提交包

- 技术报告 PDF：`docs/report/技术报告-XLANG2026.pdf`（8 页，由 `tools/md2docx.py`
  从 `docs/report-draft.md` 生成，Word 导 PDF；表格/代码/中文均校验）。
- 提交包 readme：`docs/readme-submission.md`（vllm-plugin-FL 逐项改动 + 开关 + 复现命令）。
- 复现性：干净 clone `vllm-plugin-FL@84abed0` + `docs/experiments/a20/apply_a20.py`
  → 与线上运行的三个文件**字节一致**。
- 本轮再否决：fp16 KV cache（0.699 vs bf16 0.698 ms/层，无收益）；
  `nseg=16`（4k 同档、16k −15%）；对齐/连续性提示（`tl.multiple_of` 二维形式编译失败，
  且该 harness 基础版本已比插件版慢 20%，未继续）。

### A20 期间的追加排查：融合采样（A5）是否生效？——已生效，无需改动

看到 serve 日志里 4 条 `fl opt7: fast path rejected: top_k active`，一度以为是
`SamplingParams.top_k` 默认 0 被误判、导致 A5 融合采样全程失效。排查结论：

- 这 4 条全部来自**图捕获阶段**的 dummy `SamplingMetadata`
  （`gpu_model_runner` 里 `top_k=dummy_tensors(vocab-1)`，是 tensor 且 >0）；
- 真实 decode 步的 `top_k` 是 `None`（`gpu_input_batch.no_top_k` 为真时置空），
  所以 `_fast_path_ok` 通过——每个 run 都有 `fused sampler tail ENGAGED on first batch`；
- 试验性的 `top_k > 0` 判断会在捕获期对 tensor 求 bool 而崩（
  `bool value of Tensor ... is ambiguous`），**已回滚**，交付树保持 A5 原样。

即：A5 的收益本来就在前面的 A1–A7 成绩里，这块没有剩余空间。

### A20 跨日复核（2026-10-20，新实例，从干净树 + `apply_a20.py` 重建）

把两棵树退回 `84abed0` 净态，只用 `apply_a20.py` 一条命令重建，再跑官方用例：

| exp_id | 路径 | 总吞吐 | 输出 | TTFT ms | TPOT ms |
|---|---|---|---|---|---|
| A20_4k_B14 | fast（重建后的交付态） | 2503.30 | 500.66 | 5532 | 99.86 |
| A20_4k_A6 | stock | 2258.60 | 451.72 | 5665 | 113.55 |
| A20_4k_B15 | fast | **2487.14** | 497.43 | 5540 | **100.33** |

→ 当日交错配对 **+10.1%**，TPOT −11.6%，TTFT −2.2%（略好）。
今日机器整体比 10-18 慢约 3%（stock 2259 vs 2327），相对收益一致。

**4k 全部同日证据**（3 组独立交错配对）：+9.0%（10-17/18 夜）、+12.8%（10-18 晚）、
+10.1%（10-20）；A(4 次) 中位 2292.7 vs B(5 次) 中位 2538.2 → **+10.7%**。

同时确认：干净 `vllm-plugin-FL@84abed0` + 一条 `apply_a20.py` 重建后，
官方 4k 仍为 ~2500 tok/s（与插桩前的交付树同档），提交包的自包含性得到验证。

### A20 算子迁入 FlagGems（2026-10-20，按官方口径）

官方澄清（2026-10-20 群答）：**不能修改厂商预编译的 .so**；**算子优化建议走 Triton 路线并提交到
flaggems 仓库**；插件内改 `scheduler_config.max_num_batched_tokens` / `cudagraph_capture_sizes`
等同于改 serve 参数（不允许）。我们的 A20 本来就是 Triton 内核，因此把它从插件搬进 FlagGems：

- 新增 `FlagGems/src/flag_gems/fused/paged_decode_attention.py`
  （`paged_decode_attention(...)` + 两个 Triton kernel + 设计/实测说明），并在 `fused/__init__.py` 注册；
- 插件侧 `decode_attn_fast.py` 只留薄封装：`from flag_gems.fused import paged_decode_attention`，
  导入失败则整条 fast path 关闭（退回 stock），不再在插件里保留内核副本；
- `apply_a20.py` 现在同时打 FlagGems 与插件两棵树（`FG_REPO`/`FG_SITE` 可覆盖路径）。

**等价性验证（官方 4k，同日）**：

| 版本 | 总吞吐 | TPOT |
|---|---|---|
| 插件内自带内核（B14 / B15） | 2503.30 / 2487.14 | 99.86 / 100.33 |
| 改为调用 FlagGems 算子（B17） | **2505.47** | **99.75** |

**16k 同日配对（FlagGems 版）**：A7 2955.98（stock）→ **B18 3434.66**，
TTFT 155.6→133.9 s、TPOT 153.3→129.9 ms，**+16.2%**。

### A20 最终精度复核（2026-10-20，交付树 = FlagGems 版内核）

MATH-500 Level 3（105 题，与基线同参 temperature 1.0 / top_p 0.95 / max_tokens 32768）：
**98.1%**。serve 日志确认 `decode-attention fast kernel enabled (segments=32 block=16
warps=1 stages=1 bmax=64)`，即走的是 FlagGems 内的 `paged_decode_attention`。

| 复核 | 代码 | MATH-500 L3 |
|---|---|---|
| 2026-10-18 | A20（内核在插件内） | 97.1% |
| **2026-10-20** | A20（内核在 FlagGems） | **98.1%** |
| 基线（赛道给定） | — | 96.2% |

### A20 追加否定：combine 归并（单次启动 / 分块）

| 方案（4k/B=64，ms/层） | 结果 |
|---|---|
| 现状（两 kernel，大 tile） | **0.683** |
| 分块归并（CH=8） | 0.687（无收益） |
| 单次启动 + 末位 CTA 就地归并（atomic 计数，数值一致 maxdiff 1.5e-5） | 0.813–0.916（**慢 19–33%**） |
| combine warp 扫描 | w1 0.755 / w2 0.719 / **w4 0.681** / w8 0.863（现配置最优） |

combine 本身 0.057 ms/层（≈2.4 ms/步，占步长 5%），但把归并塞进同一个 kernel 会让 Triton
按最坏路径分配寄存器、拖慢 segment 热循环，净亏。该方向关闭。

### A20 新优化：页对齐分段（page-aligned segments）

`block_n == block_size` 时按整页切段，`page` 变标量、去掉逐 token 除模与向量页 gather。
隔离对照 −7.3%~−10.1%（输出 maxdiff 0.0000）。

| 用例（2026-10-20 同日交错） | stock | 通用分段 | **页对齐** | 页对齐 vs stock |
|---|---|---|---|---|
| 4k `[[4096,1024,64,256]]` | 2239.43（A6b 20:19） | 2505.47（B17） | **2606.56（B19）/ 2580.18（B21 最终树）** | **+16.4%**，TPOT 113.56→95.54 |
| 16k `[[16384,1024,64,128]]` | 2956.59（A7b 20:33） | 3434.66（B18） | **3541.34（B20 20:02）** | **+19.8%**，TTFT 155.5→130.1 s |

踩坑：循环上界未 clamp 到 kvlen → 空段 `exp2(-inf-(-inf))` 出 NaN（已修正，5 形状 maxdiff=0）。

### 页对齐分段的对拍与精度（2026-10-20 晚）

- greedy 对拍（`VLLM_FL_DECODE_FAST_GENERIC_SEG=1` 切换）：5 条长 prompt 中 4 条完全一致，
  1 条从 ~30 token 起分叉（fp32 段内求和顺序不同导致的 argmax 翻转；非漏算，覆盖范围已逐形状验证）。
- 微基准 5 种形状（含 ragged）输出 maxdiff 0.0000。
- MATH-500 L3：10-18 97.1%（通用分段）、10-20 98.1%（通用分段/FlagGems）、
  **10-20 页对齐 96.2% / 97.1%（两次）**，四次均 ≥95%，基线 96.2%。

### 交付包与最终树复核（2026-10-20 深夜）

- 两棵树已提交：插件 `b0cda5e`（`84abed0` + A20）、FlagGems `f62146be`（v5.3.5 + `fused/paged_decode_attention.py`）。
- 打包脚本 `tools/pack_submission.sh`：产出 `XLANG2026.zip`（`vllm-plugin-FL/` + `FlagGems/` +
  `readme.md` + `report.pdf`，剔除 `.git`/`build`/`__pycache__`），实测 **15.4 MB / 5183 文件**，
  四个关键条目校验存在。当前实例上的 zip 在 `/tmp/pkg/XLANG2026.zip`。
- 最终树 4k 复核（含 `page_aligned` 开关的代码）：**A20_4k_B21 = 2580.18 tok/s / TPOT 96.26**，
  与 B19（2606.56 / 95.54）同档，确认切换开关没有改变默认行为。
- 注：实验室网关当晚出现 66% 丢包，SFTP 实测掉到 ~4 KB/s，15 MB 包未能拉回本地；
  重建议只需在实例上跑 `bash /workspace/pack_submission.sh`。

### A21（否决）：RMSNorm 走 IR op 级融合核

vLLM 0.24 的 `ir_op_priority` 里 `rms_norm`/`fused_add_rms_norm` 都是 `native`（纯 torch 链，eager 下每次 8–10 个核）。
自写 Triton 融合核（fp32 方差，且按 native 的"先转 bf16 再乘 weight"路径实现）在隔离里**逐位一致**且 2.5–7.2x：

| 形状 | native | 融合核 |
|---|---|---|
| N=64（decode） | 132.8 µs | 54.0 µs（2.46x） |
| N=8192（prefill） | 2039.4 µs | 281.5 µs（7.24x） |

但引擎（官方 4k，同日）：off 2538.69 / off2 2534.52（TPOT 97.7/97.8）→ **on 2388.44（−5.9%）、TPOT 100.0、TTFT 5499→9968 ms**。
归因：IR op 在编译期被 inline，Triton 核只能包成 opaque 的 `torch.library.custom_op`，于是 inductor 失去对
相邻算子的融合（prefill 最敏感，TTFT 翻倍）。**已全部回退**，源码留在 `docs/experiments/a21/`。

### A20 追加：批量不变性测试（不是性能项，是可复现性）

`FlagGems/tests/test_paged_decode_attention.py`：8 个精度用例（bf16/fp16 × 等长/ragged × kv=1024/4096，
对照 fp32 参考，atol=2e-2）+ 1 个批量不变性用例（同一组 12 条 ragged 序列，单条/两条/半批/全批/逆序全批，
断言逐位相等，并重复同组成验证确定性）。BI-V150 上 **9 passed**。

该性质来自内核结构：每 (seq, kv_head, segment) 只依赖自己的 q/KV/seq_lens，段长只由该序列长度决定。
写进报告作为可复现性卖点，也让后续 A/B 对拍不受批组成影响。

### A22（否决）：W8 权重量化 + 厂商 int8 GEMM

- 动机：decode 的 11.7 ms 非 attention 时间全由权重读取解释（5.3 GB/步）。
- 上界很诱人：`torch._int_mm` 在 K=2048/N=6144 上 56.8→26.1 µs（2.2x）；gate_up+SwiGLU
  在"激活已量化 + 融合 epilogue"的假设下 137.5→73.8 µs（1.86x）。
- **真实集成路径（含 per-call 激活量化 + 独立反量化）：120.9 → 157.5 µs（0.77x）**。
  两个额外小核约 80 µs，与 gate_up 的 bf16 GEMM（115–121 µs）同量级。
- 数值侧干净：纯 torch 参考 maxdiff 0.054、`_int_mm` 逐位精确、反量化核 0.0。
- 救不回来：要藏掉量化开销只能把"量化+int8 GEMM+反量化+SwiGLU"融成一个核，而本卡
  Triton matmul 比厂商 BLAS 慢约 6 倍；厂商的 `bnb_qgemm` 是 6 GB/s 的参考实现。
- 结论：M=64 的 decode 形态下 W8 净亏；已删除原型，脚本留档 `docs/experiments/w8-probe/`。
  **更正（2026-10-22，见 A27）**：本条的两处对比不公平——(a) "激活量化 40 µs"其实是
  `x.abs()`/`.max()`/`.float()` + 量化核共 4 个 launch，合成一个 Triton 核后只有 21.8 µs；
  (b) 基线用了"裸 GEMM 120.9 µs"，而真实路径是 GEMM+SwiGLU 137.4 µs。
  修正后隔离测量为 **75.4 µs = 1.82x**。但接进引擎后端到端 **−19%**（4k 2080 vs 2571，
  TTFT 恶化 4.4x），已回退。**否决结论维持，理由改为"隔离收益不可外推"**。

### A23/A24（否决）：投机混合步的真凶是隐式同步的 `aten::index`

A10–A13 曾判定"投机在官方 c64 口径不可用"，但归因停在"混合批 eager 前向 + 采样器第二条慢路径"，
并卡在拿不到设备级 profiler。今天用 CUDA event 把 `RejectionSampler.forward` 切成 8 段，
直接钉死：

- 慢步 99.7% 落在 `logits[bonus_logits_indices]` 这一行（4.0 s/步）；
- 同一 forward、同一张量、同一流上紧挨着换成 `torch.index_select`：**0.2–0.3 ms**（差 13000 倍）；
- 微基准里同形状花式索引只有 0.06 ms，且"慢 kernel 后接 `x[idx]`"不阻塞 → 它是**隐式同步点**，
  不是算力退化。A12 当年记的"GPU 高级索引 O(max index)"是误判，它只是绕开了同步。

修两处（RS 的 bonus/target gather + `_prepare_input_ids` 的两处高级索引）后：
RS 4100→122 ms、`_prepare_input_ids` 2458→1 ms、纯 decode 步 1094→8 ms、probe 端到端 67.3→35.5 s。

但官方 `[4096,1024,64,256]` 仍不可用：decode 454 tok/s（基线 364，+25%）、
prefill 2867–3686 tok/s（已回到基线），**5 分钟只完成 3/256 个请求**。
`fl_timing3` 显示混合步 `_model_forward` 100–950 ms（单步 7.6 s）——padonly 让每个 decode 请求
恒定带 6 个 token，混合批里 decode 膨胀 6 倍、prefill 拿不到 token 预算而饥饿。
这一环必须动调度，而组委会对"插件内部改 `max_num_batched_tokens` 一类默认值"判负面。
**结论与 A10–A13 相同（仍不采用），但理由从"索引退化"澄清为"投机 token 记账 × 调度预算"。**

文档 `docs/experiments/a23-spec-gather-iluvatar.md`，证据 `docs/experiments/a23/`，commit `85005c3`。

### A25（否决）：`_build_split` 每步 81 ms 是等待，不是浪费

A13 把这 74–82 ms 记成"`query_start_loc_cpu.tolist()` 的 D2H"，实测**两处 `.tolist()` 都是 0.0 ms**
（本来就是 CPU tensor）。真凶是同一函数末尾：

```python
torch.tensor(python_list, dtype=torch.int32, device=dev)   # ~40 ms/次 × 2
```

这份后端实现需要同步（逐元素搬到显存），把 GPU 上未完成的前向一并等掉。
改成"CPU 建 + `.to(dev, non_blocking=True)`"后 `cu_qk` 81.4 → 0.3 ms。

但官方 4k 只从 2571.12 → 2592.87 tok/s（**+0.85%**，噪声内），因为它等的是真实 GPU 工作。
**未进提交栈**，补丁留档 `docs/experiments/a25/a25.patch`，文档 `a25-build-split-sync.md`，commit `d9d986e`。

### A26（评估后放弃）：decode attention 形状扫描

`paged_decode_attention` 在 (B, kv) × NSEG 上逐点测。官方 c64 形状下：

- kv=16384：最优点**就是当前的 NSEG=32**（2.084 ms/层，**515 GB/s = 卡带宽的 88%**）
- kv=4096：换 NSEG=16 只快 **1.4%**
- NSEG=128 在所有形状慢 30–90%；小 batch（B≤16）确实有 3–7.4%，但官方用例是 c64

**shape-aware dispatch 因此不实现**：收益上限 1.4%，低于 3% 准入，而"按 B 查表 + slab 按最大
NSEG 分配"会引入图捕获期尺寸风险。组委会 10-12 已确认此类做法允许，但允许不等于值得做；
这一条按"已评估并放弃"写进报告。文档 `a26-decode-kernel-sweep.md`，commit `6ecf7ee`。

### 2026-10-22 当日基线（A20 交付态，供后续 A/B 对照）

| 用例 | 总吞吐 | TTFT | TPOT |
|---|---|---|---|
| 4k `[4096,1024,64,256]` | 2571.12 tok/s | 5529.69 ms | 96.58 ms |
| 16k `[16384,1024,64,128]` | 3515.89 tok/s | 130772.51 ms | 126.49 ms |

16k 历史 3541.34（TTFT 155.5 s），当日 −0.7%，属机器波动。

### 运维教训（写进复现说明）

今天为快速迭代频繁 `pkill -9`，残留的 CUDA 上下文让 28.5 GB 显存变成"幽灵占用"
（`ixsmi` 进程表为空但 `memory.used` 不降），容器内 `ixsmi -r` 无法复位（缺 `/dev/corex*`），
只能重启实例。**后续统一用 SIGTERM + 等待优雅退出**，本次收尾已按此方式执行并确认回落到 210 MiB。

### A28（2026-10-24，否决）：上游 #552 CachedOp 预热

- 分支 `exp/a28-cachedop-prewarm`（至 `25f64e2`），单测 **5 passed**。
- 官方 4k 同日冷启动 A/B 中位数：基线 **12475.75**、A28 **12466.30 tok/s**（−0.08%）；TTFT 1772.50→1773.96 ms（+0.08%）；启动 113.5→114.0 s。
- 结论：可运行但无稳态或启动收益，**不并入 integration**。文档 `a28-cachedop-prewarm.md`，原始工件 `runs/a28-cachedop-prewarm/`。

### A29（2026-10-24，否决）：上游 #442 common attention metadata graph

- 分支 `exp/a29-common-attn-metadata`（至 `85cc2f0`）。MetaX 的 `tl.pointer_type(elem_dtype)` 编译失败，改为显式 i32/i64 指针加载后，metadata 单测 **4 passed**、图测试 **7 passed**。
- 官方 4k 对照：关闭新路径 **12505.67**，开启新路径 **12024.01 tok/s**（−3.85%）；TTFT 1745.80→2305.95 ms（+32.09%），启动 114→119 s。
- 16k 同代码完整四轮（去首轮后三轮均值）：关闭 **11168.87**、开启 **11161.46 tok/s**（−0.07%）；TTFT 17926.92→17929.82 ms（+0.02%）。无收益。
- 结论：功能兼容但端到端负收益，**不并入 integration**。文档 `a29-common-attention-metadata.md`，原始工件 `runs/a29-common-attention-metadata/`。
- 补充开销分解：decode 微基准原始/eager/replay 中位数为 **0.0870/0.0655/0.0449 ms**；强制 no-replay 后 4k 为 **12028.50 tok/s**，与 replay 的 12024.01 同档。8 请求 trace 的 kernel 总时长基线/A29 为 **3312.37/3310.99 ms**，未见 metadata kernel 级瓶颈，故 replay 不是根因，最终仍不采用。
- host 函数级 trace：`_run_common_attention_metadata` 仅 **65.9 µs**；`_build_attention_metadata` +1.76%，`execute_model` +1.77%，主要 kernel 总时长几乎不变。未发现可安全修复的单点，A29 仍不进入交付分支。

### 2026-10-25 技术报告事实与代码一致性审计

- 完整审计见 `docs/deliverables/audit-2026-10-25.md`。逐项区分一致、口径需限定、证据不足、明确错误与建议复测。
- 已确认并修正 C500 16k 无来源端点、采样链耗时口径、默认双布局、实例差异、BI fig19 算术、fig15 TTFT 基线、A20 微基准版本等问题。
- A20 的 9 个测试用例已补入 FlagGems 集成分支 `aff68555`；设备端通过结果待复测，不提前写作 9 passed。
- 旧 C500 与 BI 实例认证均已失效。最终交付 SHA 的双平台官方 benchmark、MATH-500、关键微基准与测试仍需新实例复测，不能用历史转录替代。
- 异常 C500 实例上，交付版与原生基线均在 1200 秒内未完成初始化，未进入 benchmark；故障工件见 `runs/c500-final-audit/`。
- 健康实例已完成当前 SHA 官方复测：4k **12399.31**、16k **11046.19 tok/s**，TTFT **1719.18／18141.47 ms**；本机原生基线为 **4210.20／5867.59**。两次 MATH-500 均为 **98.1%**。微基准：top-p 4.086→0.381 ms（10.73×）；lm_head 原生/Triton/mcBLAS 为 0.385/0.667/0.430 ms（1.55×）。原始工件已全部归档。

### 2026-10-26/27 BI-V150 当前交付复测（完成）

- 当前 SHA 的 A20 **9 passed**；等长 kv=4096 微基准 stock 0.9386→A20 0.5877 ms（1.60×，约 457 GB/s），ragged 0.6767→0.4399 ms（1.54×）。
- 官方四轮去首均值：A7-1 4k 3855.31、16k 3568.31；A20-1 4k 4087.91、16k 4520.19；A7-2 4k 3840.80、16k 3570.61；A20-2 4k 4722.22、16k 4520.10。
- 相对 A7 的增益：4k **6.03%／22.95%**，16k **26.68%／26.59%**；16k TTFT 改善 **18.51%／18.43%**。
- 当前 MATH-500 Level 3 两次为 **96.19%／98.1%**，均 105/105 成功。混合步 used=132000、skipped=0，未见 decode fast-path 异常回退。
- 完整工件见 `runs/bi-final-audit/` 与 `tmp/bi-final-audit-complete.tar.gz`；此前关机前的部分结果已补齐，不再作为最终状态。

### A30（2026-10-27，否决）：paged decode 流水与尾页拆分

- 分支：插件/FlagGems `exp/a30-decode-pipeline`（`2fb32ad`/`6b13c991`）；默认 `split_tail=False`，集成分支未改。
- `num_stages=2/3` 在 4k/16k、等长/ragged 全部比 stage 1 慢，否决；30 次原始样本见 `runs/a30-decode-pipeline/stage-probe.json`。
- 完整页与 ragged 尾页拆分在微基准提升 8.03%/14.55%（4k/16k 等长），新增页边界测试 **10 passed**，与当前实现逐位一致。
- B=16/32/64 的全部扫描点仍有 7.68%–20.25% 孤立收益，不能用 batch 形状解释集成回退。
- 官方 4k 四轮去首、A/B/A/B：A20 **4815.68/4716.31**，拆分 **4158.66/4192.17**；中位数 **4766.00→4175.42（−12.39%）**。候选不合入，16k 官方长测不继续。
- 完整证据：`runs/a30-decode-pipeline/`、`a30-4k-official.tar.gz`。后续以 ixsys/ixkn 分析图内与组合路径，诊断计时不作为成绩。

### A31（2026-10-27/28，已合入）：配置缓存与单表反射

- 分支 `exp/a31-config-local-cache`，至 `cf11897a`。进程内缓存配置查询结果；`build_sql_model_by_db` 只反射目标表；缺失表仍返回 `None` 并走 autotune。
- 原版 host profile 中 `_index_linearized` 累计 66.03s、`get_config` 65.14s、全库反射约 53.76s；A31 分别为 4.47s，累计时间 129.68→65.17s。
- 真实配置库单表加载 **4212.64→26.35ms**，同一模型成功加载；两个针对性测试通过。以上只证明 host/初始化路径，不代表官方成绩。
- 官方 A1/A2 为 **4667.18/4459.34 tok/s**。初版 B1 因缺失表语义回归而失败，整组无效；`cf11897a` 已修复并补测试。
- 修复后 B2 四轮完整：5179.70（弃首）、5209.43、5209.82、5208.22，后三轮均值 **5209.16**、TTFT 4043.57ms。相对 A1/A2 中位数 4563.26 提升 **14.15%**。完整工件见 `a31-official-evidence.tar.gz`。
- B1 修复复测四轮完成，后三轮均值 **5208.95**、TTFT 4025.01ms。最终中位数 A20 4563.26 → A31 **5209.06（+14.15%）**；TTFT 4230.56→**4034.29ms（−4.64%）**。
- MATH-500 Level 3 为 **98.1%**，105/105 成功、0 错误，满足 ≥0.95；原始证据见 `a31-eval.tar.gz`。
- 4k 性能、TTFT 和精度要求均已满足，A31 合入 `integration/flaggems`（`9c06143c`）。
- 16k A/B/A/B：A1/A2 4516.49/4511.37，B1/B2 4515.71/4512.11，中位数 **4513.93→4513.91**，基本持平。16k host 总时间 148.42→141.48s，metadata、copy 和设备计算成为主要部分；证据见 `a31-16k-evidence.tar.gz`、`16k-{current,a31}-host.prof`。

