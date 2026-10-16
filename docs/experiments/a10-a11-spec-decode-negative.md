# A10/A11: n-gram 投机解码在 Iluvatar BI-V150 上的完整评估（阴性结论）

日期: 2026-10-16  卡: 天数 BI-V150 32GB(开发机)  栈: vLLM 0.24 + vllm-plugin-FL(flagos-2026-s2)
serve 参数与基线完全一致，仅额外加 `--speculative-config`（组委会 10-08 明确：改进投机采样策略本身允许，自训/外部 draft 模型不允许）。

## 1. 结论摘要

**ngram 投机的 draft 质量在这套 benchmark 负载上是完美的（接受率 100%、A/G 63.1%、理论 decode 加速 2.71x），
但在这套 vLLM+fork 栈上，无论哪种执行配置，净吞吐都远低于不投机的基线。故本赛道不采用投机解码。**

| 配置 | draft 接受率 | 平均接受长度 | 接受 token 占比 A/G | step 时间 | 净吞吐 (output tok/s) |
|---|---|---|---|---|---|
| **无投机（A1–A7 交付栈）** | — | 1 | — | **19.3 ms** | **364.11**（后3跑均值） |
| ngram（CPU proposer） | 100% | 6.00 | 63.1% | ~373 ms | 107.68 |
| ngram_gpu（stock） | ~99% | 1.6–2.2 | — | 78–131 ms | 30–100（抖动） |
| ngram_gpu + A11 均匀批补丁 | 100% | 6.00 | — | 5–55 s（病态） | 2–9 |

负载：4k 场景 `[4096,1024,8,32]` 的诊断切片（并发 8）与 `[4096,1024,8,16]`。基线对照为同参数无 spec（A7 栈）。

## 2. 为什么 draft 质量是好的

官方 benchmark 用随机 token 拼 prompt，模型输出很快退化成复读循环；ngram proposer 在循环里
"上一次出现位置之后的内容" 恰好就是下一步要生成的内容，因此 draft 几乎全中：

- 全程 82685/82685 命中，逐位置接受率 99.3–99.8%
- 生成 131k token 的窗口内 A/G = 63.1% → 理论上每步可白拿 2.71 倍 token

## 3. 为什么净吞吐仍然亏（三个原因，逐层定位）

1. **CPU proposer 本身贵**：单线程 Python 查找 ~43 ms/step，直接吃掉全部收益。
2. **stock ngram_gpu：裁剪同步 + 非均匀批**。
   `vllm.v1.spec_decode.ngram_proposer_gpu.update_scheduler_for_invalid_drafts` 每步
   `event.synchronize()` 阻塞流水线，并把每个请求的 draft 数裁到"上一步有效数"——
   各请求长度不一，导致 verify 批**非均匀**，`_is_uniform_decode()` 返回 False，
   `FULL_DECODE_ONLY` cudagraph 被绕过 → 整个 verify forward 走 eager（78–131 ms/step）。
3. **A11 均匀化补丁：图内 attention 路径病态**。A11 用占位 token 补齐 draft 槽位并禁用裁剪后，
   批次确实变均匀了（见下方 FL-DBG 实证），但 step 反而劣化到 **5–55 s/step**：
   cudagraph 已捕获的 decode 图内部执行的是 vLLM 原生 attention 实现；本提交的 vendor ixAttnBkd
   路由在**每步的 metadata builder 里做决策（图外）**，因此图内退化成该栈上 q_len=6 的病态路径
   （vendor FA 计时日志在 graphed 步期间停止递增即为佐证）。

### A11 生效性实证（FL-DBG 埋点输出）

```
[FL-DBG] step=17 cgmode=NONE ... maxq=300 qmin=6 ...      # 混合 prefill/decode 批 → 无图（正常）
[FL-DBG] step=18 cgmode=FULL bdesc=BatchDescriptor(num_tokens=48, num_reqs=8, uniform=True,...)
         maxq=6 qmin=6 qmax=6 tot=48                       # 纯 decode 批：均匀 + FULL 图 ✓
SpecDecoding metrics: Mean acceptance length: 6.00         # 接受长度满格 ✓
```

即：**批次均匀性、图派发、接受率三件事都做成了，但图内 attention 使这一步变成 5–55 秒。**

## 4. 环境关键发现（对复现与评审都重要）

- 运行时真正执行的 model runner 是**插件自带的 fork**：`vllm_fl/worker/model_runner.py::ModelRunnerFL`
  （文件头注明 "Adapted from vllm v0.19.0"），**不是** vllm 0.24 的 `vllm/v1/worker/gpu_model_runner.py`。
  任何针对 vllm 原版 runner 的 monkeypatch 都不会生效；必须打在插件模块的绑定上。
- `NgramProposerGPU` 等 spec-decode 类是从 `vllm.v1.spec_decode.*` 导入的共享类，打类方法补丁有效。
- `update_scheduler_for_invalid_drafts` 是被 `import` 进各 runner 模块命名空间的函数，
  必须逐个模块替换绑定才生效。

## 5. 可复现命令

```bash
# 对照（无投机，A7 栈）
VLLM_PLUGINS=fl vllm serve /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
# 投机（诊断用）
... --speculative-config '{"method": "ngram_gpu", "num_speculative_tokens": 5, "prompt_lookup_max": 5, "prompt_lookup_min": 2}'
# 负载
python3 vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm --port 9031 --test-cases '[[4096,1024,8,32]]'
```

实验产物（未进入提交代码）: `/workspace/opt_experiments/`（A11 补丁、FL-DBG 版 runner、阶段计时器）。

## 6. A9 结论修订

`docs/baselines/iluvatar-biv150/a8-a9-negative-results.md` 中 A9 的否决结论**结果正确、理由需要修订**：
不是"负载不适配投机"（恰恰相反，draft 质量完美、理论上限 2.71x），而是
"**这套 vLLM+fork 栈上投机步的执行开销无法压到收益以下**"（裁剪同步、非均匀批、图内 attention 病态）。
