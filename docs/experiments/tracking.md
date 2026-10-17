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
