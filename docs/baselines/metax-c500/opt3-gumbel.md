# 优化 #3：Gumbel-Max 单遍融合采样（MetaX C500，2026-09-29，新实例恢复后）

## 动机（profile 证据）

优化 #1 的免排序阈值内核落地后，采样尾部仍是 eager PyTorch 四段式：
`softmax（物化 33MB）→ exponential_ 噪声（物化 33MB）→ div_（66MB 读写）→ argmax（33MB 读）`，
每 decode 步 ~0.39ms GPU + 4 次内核启动（弱 CPU 上 launch 开销显著），且全部在 cudagraph 外。

## 实现

- 新算子 `FlagGems/src/flag_gems/fused/gumbel_max_sample.py`：利用 Gumbel-Max 恒等式
  `argmax_i(x_i + g_i), g~Gumbel(0,1) ≡ sample~softmax(x)`，把采样尾部塌缩为
  **单内核、单遍 vocab 扫描**（Philox RNG 在 kernel 内生成，u clamp 防 log(0)）。
  顶层 mask（-inf）自然排除。参考：vLLM `v1/worker/gpu/sample/gumbel.py`、
  FlashInfer sampling、SonicSampler（arXiv 2607.20475）。
- 插件补丁 `topk_topp_sampler.py`（metax）扩展：接管 `TopKTopPSampler.forward_native`，
  在 `raw_logprobs` 模式、无 per-request generator、非 fp64-gumbel 时走融合内核，
  其余情形回退原生路径（可观测 fallback，不静默退化）。
- RNG 状态：标量 seed + host 侧步进计数器（该内核位于 cudagraph 外的采样尾部，
  无冻结风险；如需进图须改为指针传入，代码注释已注明）。

## 验证

- 统计一致性：与 torch.multinomial 同 N=2000 校准，TV 距离 0.1818 vs 0.1836
  （不可区分）；mask 集违规 0；同 step+seed 确定性、跨 step 变化、单 token 边界全过；
- 内核级 [64×130560 fp32]：eager 尾部 391µs → 融合 **218µs（1.8×）**，并省 3 次 launch；
- 顺带发现并关闭调研候选 #1：async scheduling 在 vLLM 0.24 默认已开启
  （serve 日志 "Asynchronous scheduling is enabled"），cudagraph 已是 FULL_AND_PIECEWISE。

## 端到端结果（官方 benchmark，serve 参数与基线一致）

| 场景 | 官方基线 | 优化 #1+#2 | 优化 #1+#2+#3（4 跑） | vs 基线 |
|---|---|---|---|---|
| 4k total tok/s | 5089.645 | 7361 | 7970/7169/7151/8046（均值 **7584**） | **+49.0%** |
| 16k total tok/s | 7029.675 | 8246 | 8297/8244/8253/8256（均值 **8263**） | **+17.6%** |
| 4k Mean TTFT | 3199ms | ~3272ms | 3129~3259（均值 **3206**） | **+0.2% ✓ 进限值** |
| 16k Mean TTFT | 27197ms | 26031 | 25898~26143（均值 26050） | **−4.2% ✓** |
| 精度 math_500 L3 | 0.962 | 96.2%（3 跑均值） | **97.1%** ✓ | 无降幅 |

注：本次在新实例上运行，先做了恢复验证（opt2 代码 4k 复测 5882/4947/6537/8115，
与旧实例同分布），环境一致性确认后才跑 opt3。

数据：`raw_runs_opt3_sampler_fused.csv` / `summary_opt3_sampler_fused.csv`。
评测机 commit：FlagGems `d4420081`、vllm-plugin-FL `d035e1b`（均已推 GitHub）。
