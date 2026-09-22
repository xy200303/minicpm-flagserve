# 优化 #1：免排序 top-k/top-p 采样内核（MetaX C500，2026-09-22）

## 动机（profile 证据）

torch profiler（64 并发、4096/512 负载、53.3s 轨迹，GPU 忙 70.9%）显示 decode 阶段
**~36% GPU 时间花在 eager top-p 采样**：`compute_global_hist`(8.9ms/步) + `sweep`(8×1.5ms/步)
+ `sort`(4ms/步) 等。原因链：

1. `MiniCPM5-2B/generation_config.json` 默认 `do_sample=true, temperature=1.0, top_p=0.95`
   → benchmark 所有请求都触发 top-p 过滤；
2. vLLM 0.24 的 Qrita Triton topk_topp 内核在 MetaX 编译失败（ttgir PassManager），
   插件补丁 `topk_topp_sampler.py` 退回 eager PyTorch；
3. eager 路径每步对 [64, 130560] logits 做**全词表排序**（补丁注释 TODO：
   "remove once FlagGems provides a MetaX-compatible topk_topp kernel"）。

## 实现（开发量落点）

- 新算子 `FlagGems/src/flag_gems/fused/top_k_top_p.py`：`apply_top_k_top_p(logits, k, p)`。
  **免排序**算法：① online logsumexp 单遍求行最大值与配分函数；② top-k 在 logit 区间
  二分求第 k 大阈值；③ top-p 在 k 保留集配分质量上二分求质量阈值；④ 单遍掩码写回。
  与参考实现（`apply_top_k_top_p_pytorch`）语义对齐（含 k 值并列保留、top-p 在
  top-k 保留集上重归一化）。
- 插件侧 `vllm_fl/dispatch/backends/vendor/metax/patches/topk_topp_sampler.py` 改为
  路由到 FlagGems 内核，PyTorch 路径保留为兜底（不改框架算子选择逻辑）。
- 正确性：fp32/bf16 × p∈{0.9,0.95,1.0} × k∈{None,50,vocab} 全组合，
  保留集 IoU ≥ 0.999、采样分布 TV ≤ 1e-4（p=1.0 边界差 ~1e-26 质量，无实际影响）。
- 内核级性能 [64×130560 p=0.95 fp32]：pytorch 9.06ms → **1.68ms（5.4×）**。

## 端到端结果（官方 benchmark，serve 参数与基线一致）

| 场景 | 官方基线 | 基线复现 | 优化后（4 跑） | vs 官方基线 |
|---|---|---|---|---|
| 4k total tok/s | 5089.645 | 4763（均值） | 6152 / 6213 / 6650 / 6857（均值 **6468**） | **+27.1%** |
| 16k total tok/s | 7029.675 | 7038.6（均值） | 8433 / 7815 / 8414 / 8396（均值 **8265**） | **+17.6%** |
| 4k Mean TTFT | 3199ms | 3234ms | 3833→3277ms（逐跑收敛） | 热身后 ~+1~2% |
| 16k Mean TTFT | 27197ms | 27111ms | 25991~26335ms | **−4%（更好）** |
| 精度 math_500 L3 | 0.962 | 0.981 | **0.981** | 无降幅 |

数据：`raw_runs_opt1_sampler.csv` / `summary_opt1_sampler.csv`。
评测机 commit：FlagGems `6f681ea8`、vllm-plugin-FL `4371ad7`（已推 GitHub dev 分支）。

## 已知遗留

- 4k 场景 Mean TTFT 前两跑偏高（冷启动 JIT），后两跑 3305/3277ms 接近基线；
  正式评测若为冷启动单次测量，TTFT 口径需关注（16k 已优于基线）。
- 二分迭代次数（2×16 遍）还有压缩空间（直方图预定位可降到 ~6 遍），内核本身
  1.68ms → 理论 <1ms，但对 e2e 的边际收益递减，优先级让位于其他热点。
