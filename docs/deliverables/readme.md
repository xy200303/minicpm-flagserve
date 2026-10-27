# XLANG2026 ｜ FlagOS 赛道二提交说明

本包对应 `vllm-plugin-FL/` 与 `FlagGems/` 两个目录。基线为
`flagos-ai/vllm-plugin-FL` 分支 `flagos-2026-s2` 与 `FlagGems` tag `v5.3.5`。
**所有改动均在框架/算子内完成，`vllm serve` 启动参数与官方基线完全一致。**

## 0. 交付版本与打包方式

本包不是手工拼接的文件集合，而是从以下两个固定 Git 分支完整 clone 得到的源码树，**两个目录均保留 `.git` 和提交历史**，便于官方执行 `git log`、`git show`、`git diff` 核对开发过程：

| 目录 | 仓库分支 | 固定 commit | 内容 |
|---|---|---|---|
| `vllm-plugin-FL/` | `xy200303/vllm-plugin-FL` 的 `integration/vllm-plugin-fl` | `0207b7e1` | C500 opt1–opt11、BI-V150 A1–A7、A20 插件 hook |
| `FlagGems/` | `xy200303/FlagGems` 的 `integration/flaggems` | `aff68555` | C500 采样/GEMM 算子、BI-V150 A20 paged decode attention 与测试 |

这两条分支也是后续开发的**集成分支**：新实验从它们创建实验分支，验证通过后再回合；`xy200303/minicpm-flagserve` 的 `main` 只保存报告、实验台账与复现文档，不承担运行时代码。两个源码仓库均为对应上游项目的正式 fork，后续可直接向上游提交 PR。

复核或重新打包可使用（完整历史，不使用 `--depth 1`）：

```bash
git clone --single-branch --branch integration/vllm-plugin-fl \
  https://github.com/xy200303/vllm-plugin-FL.git vllm-plugin-FL
git clone --single-branch --branch integration/flaggems \
  https://github.com/xy200303/FlagGems.git FlagGems
# 打包时保留两个目录下的 .git；只剔除本机 build、__pycache__ 和虚拟环境
```

**当前交付复测（2026-10-25，C500 64GB / RAM 128GB）**：部署上表两个固定提交，显式设置 `PYTHONPATH=/workspace/vllm-plugin-FL:/workspace/FlagGems/src` 后按官方命令启动。交付版 293 秒启动，原生基线 161 秒启动；日志确认 `mcblas_mm`、采样融合与 lm_head nn 路由。两个版本均完成官方四轮 benchmark，交付版另完成两次 MATH-500 Level 3，均为 98.1%。

## 0.1 环境要求与启动耗时

| 项目 | 天数 BI-V150 | 沐曦 C500 |
|---|---|---|
| GPU / 显存 | BI-V150 / 32 GB（官方开发与评测） | C500（104 SM）/ 64 GB |
| CPU | Intel Xeon Gold 6330 @ 2.00GHz | Intel Core i7-8550U |
| **CPU 内存（最终实测实例）** | 503 GiB | 128 GiB；**复现建议 ≥128 GB**（见下） |
| PyTorch | 2.10.0+corex.4.5.0 | 2.8.0+metax3.7.0.7 |
| vLLM / Triton | 0.24.0 / 3.6.0 | 0.24.0 / 3.0.0+metax3.7.0.7 |
| 芯片软件栈 | IX-ML 4.4.0，驱动 4.5.0 | MACA 3.7.0.38，驱动 3.8.30 |
| evalscope | 1.11.1 | 1.11.1 |
| 启动命令中的 `--compilation-config` | `{"cudagraph_mode": "FULL_DECODE_ONLY"}` | **不要加**（与官方一致） |

除特别说明外，上述为最终复测实例的实际配置，不代表官网规定了相同 CPU 型号或内存容量。C500 CPU 型号来自同镜像开发实例记录，最终复测 manifest 未重复采集该字段；BI 的 Triton 版本取运行时 `triton.__version__`。两平台均使用插件 `0207b7e1` 和 FlagGems `aff68555`。

**C500 内存说明（实测，非官方约束）**：在模力方舟 **64 GB CPU 内存**规格实例上，
**纯镜像原版** `vllm serve` 也会在 inductor 编译阶段被内核 OOM killer 杀掉
（`memory.events` 的 `oom_kill` 逐次 +1，匿名内存峰值 63.4 GB，
被杀时 GPU 显存仅 858 MiB）。128 GB 规格实例全程稳定。
**这不是代码问题**，与负载、vllm 版本、我们的改动均无关。

**启动耗时**：C500 首次启动需要执行 GEMM autotune 预热。当前交付复测启动为
**293 秒**，原生基线为 **161 秒**；预热时间随实例和缓存状态变化。
**请耐心等待，不要误判为卡死**。如不需要预热可设 `VLLM_FL_MM_WARMUP=0`，
但 4k 吞吐会下降约 6%、TTFT 变差。

## 1. 性能结果

官方用例 `[input_len, output_len, concurrency, num_prompts]`，使用
`vllm bench serve`，无计时埋点。当前 C500 复测与历史 opt 系列均采用 4 轮去首、
后三轮均值；prefix cache 命中情况逐次检查，当前交付复测低于 1%。

| 平台 | 场景 | 官网基线 | 原生栈复测 | 当前交付 | 相对官网 |
|---|---|---:|---:|---:|---:|
| C500 | 4k `[4096,1024,64,256]` | 5089.645 | 4210.20 | **12399.31** | **2.44x** |
| C500 | 16k `[16384,1024,64,128]` | 7029.675 | 5867.59 | **11046.19** | **1.57x** |
| BI-V150 | 4k | 2028.01 | 1902.15 | **4087.91–4722.22** | **2.02x–2.33x** |
| BI-V150 | 16k | 915.15 | 861.98 | **4520.10–4520.19** | **4.94x** |

原生栈复测均使用未修改的官方栈、官方命令和四轮去首均值：C500 测于
2026-10-25，BI 测于 2026-10-10（32GB 官方实例）。BI 原生逐轮数据见
`docs/baselines/iluvatar-biv150/a1-iluvatar-vendor-fa.md`。跨日期、跨实例数据
不替代下文同日 A/B 对照。

`total tok/s` 同时计入输入与输出，且 4k／16k 请求数分别为 256／128，
不能直接据此比较单请求速度。BI 第一组两场景总 token 为 1.31M／2.23M，
因此 16k 总吞吐可接近 4k；同组 decode 输出吞吐由 817.58 降至 265.89 tok/s，
TTFT 由 4.33 s 增至 111.64 s，长上下文成本体现在这些分项指标中。

C500 的本机原生基线为 4k **4210.20**、16k **5867.59 tok/s**，当前交付相对其
分别为 **2.95x** 和 **1.88x**。BI-V150 的官方开发与评测环境均为 **32 GB**。
两组 A20/A7 对照分别为：4k **3855.31 → 4087.91（+6.0%）**、
**3840.80 → 4722.22（+22.9%）**；16k **3568.31 → 4520.19（+26.7%）**、
**3570.61 → 4520.10（+26.6%）**。两组均列出，不用单组高值代替整体结果。

**精度（MATH-500 Level 3，evalscope 官方原命令，105 题）**：
C500 当前交付两次复测均为 **98.1%**（105/105 请求成功）；BI-V150 为 **96.19%** 和 **98.1%**。
均 ≥ 0.95。

**TTFT**：C500 4k **1719.18 ms**（官网基线 3199.435 ms）、16k **18141.47 ms**
（官网基线 27197.135 ms）；BI-V150 4k 为 **4139.85–4327.86 ms**（官网基线
11573.53 ms），16k 为 **111.64–111.67 s**（官网基线 599262.03 ms，A7 对照为
136.86–137.00 s）。按官网基线判断，两平台各场景均满足 `baseline +1%` 上限。
同日 A/B 中 BI 4k 两组 TTFT 差异为 +2.1% 和 −1.9%，该对照用于分离 A20 收益，
不替代官网基线判断。

## 2. 天数 BI-V150 改动

### A1/A6：prefill attention 路由到 vendor ixAttnBkd

`vllm_fl/dispatch/backends/vendor/iluvatar/impl/vendor_flash_attn.py`

builder 一次性算出每步 decode/prefill 划分（避免每层 2 次 D2H）；
prefill 走 `ixformer.flash_attn_varlen_func`，KV 用 Triton `_dual_gather_kernel`
一次取出页内连续 K/V（比两次 `index_select` GPU 106→62 µs、host 68→37 µs）。
开关 `VLLM_FL_DISABLE_ILUVATAR_VENDOR_FA=1`。

### A7：decode unified-attention 发射配置 pin

`impl/decode_attn_pin.py`：仅对 3D 路径 pin `num_warps=1 / num_stages=1`，
输出与 stock **bit-exact**。开关 `VLLM_FL_DISABLE_DECODE_PIN=1`。

### A20：自研 paged decode attention 内核（本轮最大单项收益）

算子本体在 FlagGems：`FlagGems/src/flag_gems/fused/paged_decode_attention.py`
（Triton 实现，已在 `fused/__init__.py` 注册）。插件侧为薄封装与派发：
`impl/decode_attn_fast.py` + `vendor_flash_attn.py` / `iluvatar.py` 两处 hook。

- 一个 CTA = 1 warp = 一个 `(seq, kv_head, 段)`：`num_warps=1` 让 PV 的
  MMA 累加器→操作数布局转换留在 warp 内；
- `BLOCK_N=16`；base-2 在线 softmax（`scale` 预折 `log2(e)`）；
- 主循环无 mask，仅 ragged 尾部单独带 mask 的 block；
- **段数固定、段长由 device 端 `seq_lens` 现算** → CUDA-graph 安全；
  段缓存 slab 在 `apply()`（捕获前）一次性预分配；
- 段边界按整页对齐（`block_n == block_size` 时 page 变标量，
  去掉逐 token 除模与向量页 gather，再快 7–10%）；
- 混合步（prefill+decode 混批）的 decode 行也走本内核（`try_fast_decode`）。

开关：`VLLM_FL_DISABLE_DECODE_FAST=1` 整体回退；
`VLLM_FL_DECODE_FAST_SEGMENTS/_BLOCK/_WARPS/_STAGES` 调形状；
`VLLM_FL_DECODE_FAST_DEBUG=1` 打印不合格原因。

### 其它

- A3：builder 的 host 隐式同步修复；
- A4：GEMM pinned-config（libtuner 误选）；
- A5：采样链融合（跨平台移植；本 benchmark 的 top_k 路径不适用，自动退回）。

## 3. 沐曦 C500 改动（opt1–opt11）

代码位于 `vllm_fl/dispatch/backends/vendor/metax/` 与
`FlagGems/src/flag_gems/`（`fused/`、`runtime/backend/_metax/`）。

| 项 | 内容 |
|---|---|
| opt1/opt3 | **sort-free 采样链**：bf16 直读 + 温度折叠的 Gumbel-Max `argmax(x/T+g)`，<br>替换 eager PyTorch 的 fp32 全词表采样。`FlagGems/fused/sampler_fused.py` |
| opt7/opt11 | **top-p 直方图缩放**：默认以单轮 1024-bin 质量直方图求 top-p 阈值，<br>与 mask 采样一趟完成；`fused/top_p_sample.py`。<br>消除每步约 1.6 ms 的 `_top_k_top_p_kernel` 与 lm_head 的冗余 cast |
| opt2/opt4/opt6 | **双布局 GEMM**：默认小 M 回退 `F.linear`，大 M 走 nn/mcBLAS；<br>`nn_db` 是可选的非默认路径；`linear_nn_repack.py` |
| opt5 | **autotune 预热**：模型加载后按 (M,K,N) 触发一次，<br>避免运行时 autotune 抖动（4k +5.9%，见第 0 节启动耗时） |
| opt8 | **prefill GEMM 直调 vendor mcBLAS**：`patches/mcblas_mm.py` 用 ctypes<br>直绑 `libmcblas` 的 `mcblasGemmEx`（cuBLAS 兼容 ABI），<br>注册为 custom op 保证 compiled 区域安全 |
| opt10 | **消除 prefill 每层 DtoH 同步**：在 GPU 上对 `prefill_seq_lens`<br>执行 `cat` + `cumsum` 构建 attention metadata，消除每层两次隐式同步（4k +10.9%、16k +9.9%） |
| opt11 | lm_head nn 布局路由 + top-p zoom 单轮 1024-bin 定稿 |

## 4. 复现

### 4.1 安装

```bash
cd vllm-plugin-FL && pip install -e . --no-build-isolation
cd ../FlagGems && pip install -e . --no-build-isolation
```

### 4.2 环境变量

天数（Iluvatar）与 C500 均需保留 `VLLM_PLUGINS=fl`。若镜像已有其他插件版本，
同时设置 `PYTHONPATH=/workspace/vllm-plugin-FL:/workspace/FlagGems/src`，确保加载交付源码。
C500 的完整环境变量为：

```bash
export PATH=/opt/conda/bin:$PATH
export MACA_PATH=/opt/maca
export MACA_HOME=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/lib64
export VLLM_PLUGINS=fl
export PYTHONPATH=/workspace/vllm-plugin-FL:/workspace/FlagGems/src
```

### 4.3 启动（与官方一致，一字未改）

```bash
# 天数
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'

# 沐曦 C500（不要加 --compilation-config）
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072
```

### 4.4 性能评测

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

**注意**：本 benchmark 使用 `seed=0`，同一 serve 上连跑两次相同
`input_len` 的负载**可能命中 prefix cache**，须查看 serve 日志确认，不能认定必然命中。
每个数据点都应在新启动的 serve 上获取，或使用不同 `input_len`。

### 4.5 精度评测

```bash
evalscope eval --model minicpm \
  --api-url http://127.0.0.1:9031/v1/chat/completions --api-key EMPTY \
  --eval-type openai_api --datasets math_500 \
  --dataset-args '{"math_500": {"dataset_id": "/workspace/evalscope-datasets/math_500", "subset_list": ["Level 3"]}}' \
  --eval-batch-size 8 --timeout 3600 \
  --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
  --work-dir /workspace/evalscope-datasets/level3 --ignore-errors
```

结果在 `/workspace/evalscope-datasets/level3/<timestamp>/reports/minicpm/math_500.json`；当前 evalscope 版本的精度字段为 `metrics[0].score`。

## 5. 实验记录与事实来源

当前交付策略以本文第 2、3 节和实际代码为准。历史探索记录仅用于追溯，
不作为后续开发的直接输入；引用前须核对代码版本、硬件规格、测量口径和原始日志。
事实与代码审计见 `xy200303/minicpm-flagserve` 仓库的
`docs/deliverables/audit-2026-10-25.md`。

最终复测原始证据均保存在该文档仓库，可用以下路径直接核查：

| 结果 | 原始证据 |
|---|---|
| C500 当前交付性能 | `docs/experiments/runs/c500-final-audit/healthy-final-{raw,summary}.csv` 与同目录 serve／bench 日志 |
| C500 本机原生基线 | 同目录 `healthy-baseline-{raw,summary}.csv` 与 serve／bench 日志 |
| C500 两次精度与微基准 | 同目录 `healthy-eval1-math500.json`、`healthy-eval2-math500.json`、`micro-healthy.json` |
| BI-V150 两组 A7/A20 | `docs/experiments/runs/bi-final-audit/{1-a7,1-a20,2-a7,2-a20}/benchmark_results/` 及同级 serve／bench 日志 |
| BI-V150 精度与内核验证 | 同目录 `eval1-math500.json`、`eval2-math500.json`、`micro.json`、`pytest-a20.log` |
| 双平台环境与固定提交 | 两目录各自的 `manifest.txt`；C500 为 `healthy-final-manifest.txt` |

官网基线来源为[赛道二页面](https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn)“4.3 评审方案”的两张基线表（访问日期 2026-10-27）。原图归档于 `docs/deliverables/latex/official_baseline_biv150.png` 和 `official_baseline_c500.png`，报告附录给出原始图片链接。
