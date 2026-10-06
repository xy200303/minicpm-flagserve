# opt7: Decode 采样尾巴融合 + lm_head nn 布局（MetaX C500）

日期：2026-10-06 ｜ 机器：C500 64GB（140.207.205.81，128GB 内存健康实例）

## 动机（profiler 实锤）

定稿栈（opt1-6 全生效）上抓 torch profiler trace（4k ctx、64 并发 decode）后发现
每个 decode step（约 19.8ms，含 profiler 开销）有一段图外 eager 尾巴：

| 内核 | 耗时/步 | 说明 |
|---|---|---|
| lm_head `linear_kernel`（vendor nt GEMM） | ~610-660 µs | [64,2048]×[2048,130560] |
| `cast` bf16→fp32 | ~38 µs | Sampler.forward 的 `logits.to(float32)` |
| `true_div`（÷temperature） | ~139 µs | `logits.div_(temp)`，fp32 读+写 |
| `gumbel_argmax` | ~218 µs | opt1 的 Gumbel-Max（读 fp32） |

合计 ≈ 1.0 ms/步 ≈ decode 步长的 5%。

## 改动 1：融合采样尾巴（FlagGems `fused/sampler_fused.py`）

三遍全词表遍历（cast + div + gumbel）融合为**一次 bf16 读取**：

- **temperature 折叠**：`argmax(x/T + g) == argmax(x + T·g)`（T>0 同除不改变 argmax），
  内核直接读 bf16 logits，内部转 fp32 计算。
- **混合 batch**：T < 1e-5 的行令 T:=0，噪声项归零 → 退化为该行纯 argmax，
  与 vLLM `torch.where(temp<eps, greedy, random)` 语义一致。
- **纯贪心**（SAMPLING=False）：无 RNG，tie 取最左，与 `torch.argmax` 一致。
- **并行度**：每行一个 program（grid=64）会把 C500 饿死（235µs）；
  改成两段式——pass1 按 vocab 切 split（[64,130560] 扫到 32 splits / BLOCK 4096 /
  num_warps=4 最优），pass2 每行 reduce 32 个 partial。

单测（test_sampler_tail.py）：贪心与 torch.argmax 100% 一致；采样分布 TV 距离与
torch.multinomial 同 N 校准持平；混合 batch 贪心行精确 argmax。

微基准 [64, 130560] bf16：

| 路径 | 采样 | 贪心 |
|---|---|---|
| eager（cast+div+gumbel_fp32 / cast+argmax） | 326 µs | 130 µs |
| 融合内核 | **104 µs（3.1x）** | **47 µs（2.8x）** |

## 改动 2：lm_head 走 vendor nn GEMM（linear_nn_repack.py 扩展）

`[64,2048]×[2048,130560]` 各候选实测（C500）：

| 候选 | 耗时 | 备注 |
|---|---|---|
| vendor F.linear（nt，现状） | 659 µs | |
| **vendor torch.mm（nn 重排权重）** | **378 µs** | M=1..128 全胜（M=1 持平） |
| Triton mm_kernel_nt_db 最优配置 | 736 µs | 巨大 N 下双缓冲内核输了 |
| Triton mm_kernel_nn_db 最优配置 | 871 µs | 同上 |

带宽下限 ~330µs（权重 534MB），vendor nn 已接近；Triton 自定义内核在该 N 下
没有竞争力（FlagGems 注释也写了 huge-N 走别处），不再死磕。

接入细节：

- lm_head 走 `UnquantizedEmbeddingMethod`（不是 UnquantizedLinearMethod），
  需要单独 hook `process_weights_after_loading`，按类名 `ParallelLMHead` 识别
  （避免误重排输入 embedding 白白浪费 534MB）。
- vLLM v1 的 lm_head 只吃 last-token hidden states，M ≤ 并发数，恒为 skinny-M
  → `_fl_always_nn=True` 全 M 路由 torch.mm。
- `tie_word_embeddings=false`，lm_head 独立权重；nt 副本保留作 fallback，
  净成本 +534MB 显存（KV 池足够）。
- M 扫描（bench_lm_head_msweep.py）：M=8..128 nn/nt 加速比 1.44-1.86x。

## 循环导入坑（重要）

metax patches 包是在 `vllm.v1.sample.sampler` 模块导入中途被平台插件发现机制
拉起的 → 补丁顶层 `from vllm.v1.sample.sampler import Sampler` 会抛
"partially initialized module"，且会导致 patches/__init__ 后续补丁全部加载失败。

解法（sampler_tail_fusion.py）：不直接 import Sampler；改为 hook
`TopKTopPSampler.__init__`（ops 子模块此时必定已完整导入，且 `Sampler.__init__`
无条件构造它）→ 引擎启动构造 Sampler 时一次性完成 forward 补丁。
非循环场景（单测）直接 `_apply()` 立即生效。

## 接入路径

- `flag_gems/fused/sampler_fused.py`：`fused_sample(logits, temperature=None, seed=0)`
- `vllm_fl/.../metax/patches/sampler_tail_fusion.py`：快速路径条件苛刻
  （无 logprobs/penalties/topk_topp/logitsprocs/generators/spec/thinking），
  不满足走原版 forward，功能语义完全保留。
- `linear_nn_repack.py`：新增 `UnquantizedEmbeddingMethod` hook + `_fl_always_nn`。

## opt7b：top_p 恒开的发现 + 全融合 top_p 采样

**关键发现**：MiniCPM5-2B 的 `generation_config.json` 设了 `top_p: 0.95`，vLLM 把它
当服务端默认采样参数 → **官方 benchmark 每个请求都跑 top_p 过滤**。之前认为
"top_k_top_p 内核在计分场景不会被调用"是错的。trace 实锤（1065 步）：
`_top_k_top_p_kernel`（opt1 内核）**1.61 ms/步**——grid=64 程序饥饿 + 每行
最多 23 趟词表扫描，是最大的单点。

### `fused/top_p_sample.py`：histogram-zoom top_p + 采样全融合

一趟流水线（全部 bf16 直读、中间 buffer 按 (batch,vocab) 缓存复用）：

1. `_row_stats_partial/reduce`：split 并行在线 max + Z（x/T 折叠进 max，
   T<eps 退化为 /1.0），reduce 顺手写初始阈值范围 [m−60, m]；
2. `_hist_zoom` ×2 + `_hist_walk` ×2：256-bin **质量直方图**
   （`atomic_add(exp(x−m))`）两轮缩放阈值，每轮精度 ×256，2 轮后 ~9e-4 logit；
   walk 用反序 load+cumsum 定位穿越 bin，tau 取 bin 下沿（保守，保证 mass≥p·Z）；
3. `_masked_sample_partial` + `_sample_reduce`：mask + Gumbel-Max 一趟完成。

修正过的坑：crossing bin 公式是 `b = sum(reach) − 1`（reach=反序 cumsum≥target）；
直方图 atomic 主导（134µs/趟 vs stats 22µs），接受不再优化。

单测（test_top_p_sample.py）：TV ours=0.3576 vs 参考 sort 实现 0.3596（持平）；
越界样本 0；混合 batch 贪心行精确；eager 链 1947µs vs 融合 766µs（**2.54x**）。
逐内核分解合计 ~496µs，其余为 launch/alloc → 加 `_BUF_CACHE` 后 782→766µs。

### spec_token_ids 空列表修复

vLLM 0.24 的 `SamplingMetadata.spec_token_ids` 是 `list[list[int]]`，无投机时
引擎传**全空列表**而非 None → `is not None` 判断会永久拒绝快速路径。
修复：仅当 `any(sm.spec_token_ids)` 为真才回退 stock（语义与
`_combine_outputs_with_spec_tokens` 的空 spec no-op 一致）。

## 实测结果（官方 benchmark，4 跑弃首取均值）

| 场景 | opt1-6 基线 | opt7b | Δ |
|---|---|---|---|
| 4k `[4096,1024,64,256]` | 8135.29 tok/s | **8273.58** | **+1.70%** |
| 16k `[16384,1024,64,128]` | 8683.08 tok/s | **8716.35** | **+0.38%** |

- 4k runs：8244.97（弃）/ 8278.64 / 8282.44 / 8259.67
- 16k runs：8718.65（弃）/ 8719.12 / 8712.96 / 8716.97
- 4k TTFT 2794.76ms，与基线持平；serve 日志全程零 fast-path 拒绝
  （仅 warmup 假元数据的 4 次 top_k 拒绝）。
- 16k 场景 prefill 占比高，采样尾巴收益被摊薄，符合预期。
- CSV 归档：`benchmark_results/summary_20261006_193814.csv`
- 精度复测：MATH-500 Level 3（evalscope，temp 1.0 / top_p 0.95，105 题）
  = **98.1%**，与定稿基线完全一致（要求 ≥0.95）。
  histogram-zoom top_p 的近似阈值对最终精度无可测影响。
