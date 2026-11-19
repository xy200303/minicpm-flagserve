# XLANG2026 ｜ FlagOS 赛道二提交说明

本包对应 `vllm-plugin-FL/` 与 `FlagGems/` 两个目录。基线为
`flagos-ai/vllm-plugin-FL` 分支 `flagos-2026-s2` 与 `FlagGems` tag `v5.3.5`。
**所有改动均在框架/算子内完成，`vllm serve` 启动参数与官方基线完全一致。**

## 0. 交付版本与打包方式

本包从以下两个固定 Git 分支按 commit 重建；开发仓库保留完整提交历史，官方可按下表的 commit 在对应 fork 中执行 `git log`、`git show`、`git diff` 核对开发过程。为避免提交包携带仓库元数据，官方 ZIP 不包含 `.git` 目录：

| 目录 | 仓库分支 | 固定 commit | 内容 |
|---|---|---|---|
| `vllm-plugin-FL/` | `xy200303/vllm-plugin-FL` 的 `integration/vllm-plugin-fl` | `e4b5854` | C500 opt1–opt11、BI-V150 A1–A7、A20 与 A48 插件路由 |
| `FlagGems/` | `xy200303/FlagGems` 的 `integration/flaggems` | `ef516137` | 基于官方镜像 `4df52d91`：C500 采样/GEMM 算子、BI-V150 A20/A31/A48/A60 算子与测试 |

这两条分支用于后续集成；`xy200303/minicpm-flagserve` 的 `main` 保存报告、实验台账和复现文档，不承载运行时代码。两个源码仓库均为对应上游项目的 fork。

复核或重新打包可使用（完整历史，不使用 `--depth 1`）：

```bash
git clone --single-branch --branch integration/vllm-plugin-fl \
  https://github.com/xy200303/vllm-plugin-FL.git vllm-plugin-FL
git clone --single-branch --branch integration/flaggems \
  https://github.com/xy200303/FlagGems.git FlagGems
# 按上表 commit 重建源码后打包；提交 ZIP 不携带 .git，只剔除本机 build、__pycache__ 和虚拟环境
```

> 为保证无 `.git` 的源码树也能构建，`FlagGems/pyproject.toml` 在 `[tool.setuptools_scm]` 下增加一行 `fallback_version = "5.3.5"`（插件原本已有该字段）。这行只在构建取版本时生效，不改变运行时行为，也不参与任何优化路径。
>
> 为提高跨平台解压兼容性，本包将上游 `vllm-plugin-FL/vllm_fl/configs/moe/E=256,N=512,device_name=nvidia.json` 软链接展开为普通文件，内容与包内目标 `E=256,N=512,device_name=nvidia,model=Qwen3.6-35B-A3B-H100.json` 一致。这是打包处理，不是内核改动。源仓库的 Git 记录仍保留原软链接；提交 ZIP 不包含 `.git`，解压后该路径显示为普通 JSON 文件。
>
> 本包在 `FlagGems/src/flag_gems/fused/DSA/` 下补了一个只含注释的 `__init__.py`。原因：上游 `setup.py` 用 `find_packages("src")` 发现包，该目录缺此文件时不会被打进 wheel，而 `fused/__init__.py` 又会 `import flag_gems.fused.DSA.bin_topk`，导致按官网命令 `pip install --no-build-isolation .` 装完后 `import flag_gems` 报 `ModuleNotFoundError`。上游 `4df52d91` 结构相同、`setup.py` 未改，属于上游打包缺陷；该文件不改变运行时行为，也不参与任何优化路径。
>
> 交付包在 commit 之上另有 4 处审查修复：`FlagGems/fused/top_p_sample.py` 的空 split NaN 防护（`_row_stats_partial` / `_row_stats_reduce`）；`LinearNNRepack` 的重排条件排除带 bias 的层；`_dual_gather_kernel` 仅在每页元素数为 4096 的整数倍时启用（否则走 `index_select`）；`flag_gems/utils/models/sql.py` 的进程内缓存键改用 `short_name`；另有两处规范整理：`_metax/ops/mm.py` 的后置 import 提到文件头（消除 E402），`decode_attn_pin.py` 与 `fused/DSA/__init__.py` 补上 Apache 许可证头。四处在 2026-11-19 的 C500 上装包验证：`import flag_gems, vllm_fl` 正常，`fused_top_p_sample` 在 vocab=151936（会触发空 split 的形状）下输出合法、不同 top-p 的选点分布符合预期。
>
> 打包元数据的差异：上游把不少文件标记为 `100755`，Windows 打包无法保留该位，因此本 ZIP 显式给 `*.sh`、`*.bash`、`*.py` 标记 `755`，其余文件 `644`；文本行尾除 `pyproject.toml`（CRLF）外均为 LF。逐文件内容与两个提交一致，可用 `git diff --ignore-all-space` 核对。

**C500 最终提交实测版本（64GB GPU / 128GB RAM）**：2026-11-16 在交付 commit `e4b5854` / `ef516137` 上完成 4k、16k 各四轮测试，实测二进制 `d69e766` 与 `e4b5854` 只差注释。服务命令与官网一致，未设额外环境变量；早期默认开启的启动期 GEMM 预热已改为可选、默认关闭。4k 后三轮均值 **12451.84 tok/s**，16k **11097.29 tok/s**，MATH-500 **96.19%**（101/105）。同日双布局提交 `58b750e` 的另一次评测为 103/105（98.1%），差异更可能来自 temperature=1.0 的采样随机性；2026-10-25 的早期交付栈两次也是 98.1%。逐轮数据、日志与 manifest 见 `docs/experiments/runs/c500-optimization-program/final-c500-warmup-off-20261116-003450-summary.md`。

下表中的 BI-V150 数值取自父提交（2026-11-11）。2026-11-17 按本说明的安装与评测命令在 BI-V150 上复测：4k/16k 后三轮均值 5191.26/4511.90 tok/s，TTFT 4050.56/111878.60 ms，MATH-500 97.1%（102/105 正确，105/105 请求成功）。4k 落在 11-11 三次测量（5131.78/5292.46/5292.99）的区间内，16k 比单次 4618.41 低 2.3%；同日可编辑安装下另测一次为 5162.05/4515.29 tok/s、98.1%，两次差 0.6% 以内。数据与日志见 `docs/experiments/runs/bi-final-official-replay-20261117/` 与 `docs/experiments/runs/bi-final-retest-20261117/`。2026-11-19 按赛事方口径把 FlagTree 升到 `0.7.1a2+iluvatar3.6` 后复测交付提交：4k/16k 后三轮均值 **5045.20 / 4401.06 tok/s**、平均 TTFT 4149.48 / 115735.63 ms，同机 A/B（当晚把 FlagTree 换回 0.6.2a3 只跑 4k）为 5048.83 tok/s，与 0.7.1a2 的 5045.20 相差 −0.07%，说明与 11-17 的 5191.26 之间约 2.8% 的差异来自实例、与编译器无关；两次 MATH-500 Level 3 为 95.24% 与 98.10%（100/105、103/105），证据见 `docs/experiments/runs/bi-flagtree-071a2/`。

## 0.1 环境要求与启动耗时

| 项目 | 天数 BI-V150 | 沐曦 C500 |
|---|---|---|
| GPU / 显存 | BI-V150 / 32 GB（官方开发与评测） | C500（104 SM）/ 64 GB |
| CPU | Intel Xeon Gold 6330 @ 2.00GHz | 型号未采集（最终实例仅记录内存） |
| **CPU 内存（最终实测实例）** | 503 GiB | 128 GiB；**复现建议 ≥128 GB**（见下） |
| PyTorch | 2.10.0+corex.4.5.0 | 2.8.0+metax3.7.0.7 |
| vLLM / Triton | 0.24.0 / 3.6.0 | 0.24.0 / 3.0.0+metax3.7.0.7 |
| FlagTree | 0.7.1a2+iluvatar3.6（官方统一复验版本） | 未安装（镜像环境） |
| 芯片软件栈 | IX-ML 4.4.0，驱动 4.5.0 | MACA 3.7.0.38，驱动 3.8.30 |
| evalscope | 1.11.1 | 1.11.1 |
| 启动命令中的 `--compilation-config` | `{"cudagraph_mode": "FULL_DECODE_ONLY"}` | **不要加**（与官方一致） |

除特别说明外，上述为最终复测实例的实际配置，不代表官网规定了相同 CPU 型号或内存容量。C500 CPU 型号来自同镜像开发实例记录，最终复测 manifest 未重复采集该字段；BI 的 Triton 版本取运行时 `triton.__version__`。当前交付为插件 `e4b5854` 和 FlagGems `ef516137`。BI 性能与精度在父提交 `58bae94`／`85b4ba82` 上测得；最终提交随后规范换行并调整 Metax 侧默认开关，BI 执行路径不变；C500 数据使用其专用实测提交，详见上文。天数 FlagTree 为 `0.6.2a3+iluvatar3.6`，FlagGems 重放基线为镜像提交 `4df52d91`。

**C500 内存说明（实测，非官方约束）**：在模力方舟 **64 GB CPU 内存**规格实例上，
**纯镜像原版** `vllm serve` 也会在 inductor 编译阶段被内核 OOM killer 杀掉
（`memory.events` 的 `oom_kill` 逐次 +1，匿名内存峰值 63.4 GB，
被杀时 GPU 显存仅 858 MiB）。128 GB 规格实例全程稳定。
日志支持的直接原因是 **CPU 内存达到容器限制**，不是 GPU 显存耗尽；原版也可复现，因此不能归因于我们的 nn 布局改动。但这不足以排除 vLLM／编译器版本对内存峰值的影响，尚未完成跨版本定位。

**启动耗时**：C500 冷启动实测约 **124 秒**（清空调优库与 Triton 编译缓存），原生基线为 **161 秒**。
早期默认开启的 GEMM autotune 预热会把启动拉长到 6–19 分钟，现已默认关闭；
需要在启动期完成分桶调优时设 `VLLM_FL_MM_WARMUP=1`。官方 benchmark 弃首轮，预热开关不影响后三轮成绩。

## 0.2 提交包结构与预期结果

解压后为 `XLANG2026/{FlagGems, vllm-plugin-FL, readme.md, report.pdf}`，与赛题要求的 `参赛队伍名/{FlagGems, readme.md, report.pdf, vllm-plugin-FL}` 一致。两个源码目录同时包含 C500 与 BI-V150 的改动，不需要按平台拆分。

复现时应看到的结果（官方用例、四轮去首）：

| 平台 | 4k total tok/s | 16k total tok/s | MATH-500 Level 3 |
|---|---:|---:|---:|
| 沐曦 C500（64 GB 显存 / 128 GB 内存） | ≈ 12451 | ≈ 11097 | ≈ 0.96 |
| 天数 BI-V150（32 GB 显存） | ≈ 5191 | ≈ 4512 | ≈ 0.97–0.98 |

同机型同镜像下 ±3% 属正常波动（BI 历史三次 4k 测量落在 5131–5293）。若结果明显低于该区间，先按 4.7 节检查 prefix cache 命中与回退日志，再判断是否复现失败。

完整流程的时间预算（单卡实例）：安装约 1 分钟、服务启动约 1 分钟、4k 四轮约 17 分钟、16k 四轮约 33 分钟、MATH-500 约 15 分钟，合计约 1 小时。

## 1. 性能结果

官方用例 `[input_len, output_len, concurrency, num_prompts]`，使用
`vllm bench serve`，无计时埋点。当前 C500 复测与历史 opt 系列均采用 4 轮去首、
后三轮均值；prefix cache 命中情况逐次检查，实测峰值低于 1%（C500 最终轮日志未回传，见 4.7 第 4 条）。

| 平台 | 场景 | 官网基线 | 原生栈复测 | 优化版本实测 | 相对官网 |
|---|---|---:|---:|---:|---:|
| C500 | 4k `[4096,1024,64,256]` | 5089.645 | 4210.20 | **12451.84** | **2.45x** |
| C500 | 16k `[16384,1024,64,128]` | 7029.675 | 5867.59 | **11097.29** | **1.58x** |
| BI-V150 | 4k | 2028.01 | 1902.15 | **5292.46** | **2.61x** |
| BI-V150 | 16k | 915.15 | 861.98 | **4618.41** | **5.05x** |

原生栈复测均使用未修改的官方栈、官方命令和四轮去首均值：C500 原生基线测于
2026-10-25，C500 最终提交性能实测采集于 2026-11-16；BI 原生基线测于 2026-10-10（32GB 官方实例）。
上述倍数因此是跨日期参考，不替代官网基线和同日 A/B。BI 原生逐轮数据见
`docs/baselines/iluvatar-biv150/a1-iluvatar-vendor-fa.md`。跨日期、跨实例数据
不替代下文同日 A/B 对照。

`total tok/s` 同时计入输入与输出，且 4k／16k 请求数分别为 256／128，
不能直接据此比较单请求速度。BI 第一组两场景总 token 为 1.31M／2.23M，
因此 16k 总吞吐可接近 4k；同组 decode 输出吞吐由 817.58 降至 265.89 tok/s，
TTFT 由 4.33 s 增至 111.64 s，长上下文成本体现在这些分项指标中。

C500 的本机原生基线为 4k **4210.20**、16k **5867.59 tok/s**，最终提交实测相对其
分别为 **2.96x** 和 **1.89x**。BI-V150 的官方开发与评测环境均为 **32 GB**；BI 表中数值为父提交测量；交付提交的复测值见第 0 节。
两组 A20/A7 对照分别为：4k **3855.31 → 4087.91（+6.0%）**、
**3840.80 → 4722.22（+22.9%）**；16k **3568.31 → 4520.19（+26.7%）**、
**3570.61 → 4520.10（+26.6%）**。两组均列出，不用单组高值代替整体结果。

A31 配置缓存与单表反射进一步将 4k 中位数从 **4563.26** 提升到
**5209.06 tok/s（+14.15%）**，TTFT 中位数从 4230.56 降至 4034.29ms（−4.64%）；
16k 中位数为 **4513.91 tok/s**，与 A20 基线基本持平。A31 的 MATH-500 为 **98.1%**。

A48 双累加器门控融合在两组同日交错 A/B 中继续取得小幅正收益：4k 中位数
**5201.98 → 5237.32 tok/s（+0.679%）**，TTFT **4048.75 → 3945.93ms**；
16k 中位数 **4524.55 → 4556.53 tok/s（+0.707%）**，TTFT
**111534.41 → 110251.38ms**。A48 的 MATH-500 为 **98.1%**，105/105 成功、0 错误。
原始数据见 `docs/experiments/runs/a48-dual-acc-silu-mul/`。

A60 在 A48 基础上将 gate/up 权重按 `BK×BN` 预打包，tile 内 K 维连续，缓存仅构建一次并保留原权重引用。插件侧按 M 派发：M≤128 与 M≥1024 走融合内核，128–1024 之间的混合批回退原路径。开发期在公开 tag 基线上的两组同日 A/B 为 4k **5238.28 → 5314.15 tok/s（+1.448%）**、16k **4551.39 → 4620.25 tok/s（+1.513%）**；该历史数据见 `docs/experiments/runs/a60-tile-packed/`。

官方天数镜像确认使用 FlagGems `4df52d91` 后，完整 A20+A31+A48+A60 栈已重放到该提交，测量提交为 `85b4ba82`；最终 `integration/flaggems` 为 `ef516137`，仅追加换行规范化。最终 4k 三次独立四轮测量为 **5131.78 / 5292.46 / 5292.99 tok/s**，中位数 **5292.46 tok/s**、TTFT **3908.25 ms**；16k 四轮汇总为 **4618.41 tok/s**、TTFT **118761.15 ms**。首测偏低的原因尚未完全定位，因此报告保留全部三次结果。完整证据见 `docs/experiments/runs/image-baseline-align/`。

**精度（MATH-500 Level 3，evalscope 官方原命令，105 题）**：
C500 最终提交为 **96.19%**（105/105 请求成功）；2026-10-25 的早期交付栈两次为 **98.1%**；BI-V150 父提交镜像对齐栈为 **98.1%**，105/105 成功、0 错误；交付提交复测（2026-11-17）为 **97.1%**（102/105 正确、105/105 请求成功），同日另一次为 **98.1%**。已验证结果均 ≥ 0.95。

**TTFT**：C500 最终提交 4k **1616.56 ms**（官网基线 3199.435 ms）、16k **17980.89 ms**
（官网基线 27197.135 ms）；BI-V150 父提交历史测量 4k 为 **3908.25 ms**，16k 为 **118.76 s**
（官网基线分别为 11573.53ms 和 599262.03ms）。C500 已按官网基线判断满足 `baseline +1%` 上限；BI 交付提交复测的平均 TTFT 为 4050.56/111878.60 ms，同样低于官网基线。同日 A/B 中 BI 4k 两组 TTFT 差异为 +2.1% 和 −1.9%，该对照用于分离 A20 收益，不替代官网基线判断。

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

### A20：自研 paged decode attention 内核（最大单项收益）

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

### A31：配置缓存与单表反射

- `src/flag_gems/utils/libentry.py`：为同一 shape 的配置查询和缺失结果增加进程内缓存，`set()` 同步更新；`FLAGGEMS_CONFIG_CACHE_LOCAL=0` 可回退到直接数据库读取。
- `src/flag_gems/utils/models/sql.py`：从全库 automap reflection 改为只加载目标表；配置表不存在时仍返回 `None` 并进入 autotune，保持原回退语义。
- 真实配置库单表加载 **4212.64→26.35ms**；host profile 的 `_index_linearized` 累计时间 **66.03→4.47s**。
- 官方 4k 中位数 **4563.26→5209.06 tok/s（+14.15%）**，16k 持平；MATH-500 **98.1%**。
- 16k 的原版配置获取约 48.47s，但 A31 后总 host 时间仅从 148.42s 降至 141.48s，attention metadata、copy 与设备计算成为主要部分，因此端到端没有提升。
- 该优化只改变配置和 host 元数据访问，不改变算子选择、模型计算、采样参数或官方启动命令。

### A48：双累加器 gate/up GEMM＋SiLU 门控融合

- `FlagGems/src/flag_gems/runtime/backend/_iluvatar/fused/mm_silu_mul.py` 使用两个独立 FP32 累加器分别计算 gate/up，共享输入 A tile，直接输出 `gate*sigmoid(gate)*up`，避免中间激活的额外读写。
- 插件 `impl/mm_silu_mul.py` 同时覆盖实际命中的 `LlamaMLP` 与仓库独立的 `MiniCPMMLP`，只在未量化、指定权重和 token 形状下启用，其余场景回退原路径；`VLLM_FL_DISABLE_MLP_FUSION=1` 可关闭。
- 官方形状测试 2 passed、插件路由测试 3 passed。4k/16k 两组中位数分别提升 **0.679%/0.707%**，TTFT 均改善；MATH-500 **98.1%**。

### A60：K 连续 tile 预打包权重布局

- 同一 `mm_silu_mul.py` 新增 `_tile_packed_silu_mul_kernel`，将 gate/up 分别打包为 `[K/BK, N/BN, BN, BK]` 的连续存储；逻辑点积保持 `[BK,BN]`，输出使用全局 N 偏移。
- `_tile_packed_weights` 以权重地址、设备、类型、形状和步长为键，缓存仅构建一次，并保留原权重引用；不满足 K/N tile 整除条件时回退 A48 内核。
- 官方形状与缓存测试 3 passed。开发期公开 tag 基线 A/B 为 **+1.448%/+1.513%**；镜像对齐后的最终栈 4k 中位数 **5292.46 tok/s**、16k **4618.41 tok/s**，MATH-500 **98.1%**。

### 其它

- A3：builder 的 host 隐式同步修复；
- A4：GEMM pinned-config（libtuner 误选）；
- A5：采样链融合（跨平台移植；本 benchmark 的 top_k 路径不适用，自动退回）。

## 3. 沐曦 C500 改动（opt1–opt11）

代码位于 `vllm_fl/dispatch/backends/vendor/metax/` 与
`FlagGems/src/flag_gems/`（`fused/`、`runtime/backend/_metax/`）。

改动分**算子层**与**执行路径层**。算子层交付两类自研内核：

- **采样链两个 Triton 内核**（`FlagGems/fused/sampler_fused.py`、`fused/top_p_sample.py`）：
  bf16 直读 + 温度折叠的 Gumbel-Max，以及以 1024-bin 质量直方图定位 top-p 阈值的单趟采样。
  B=64、词表 130560 时 **4.086 → 0.381 ms（10.7×）**，采样族占 GPU busy 由 **38.8% 降到 5.1%**。
- **GEMM 内核族 10 个**（`FlagGems/runtime/backend/_metax/ops/mm.py`）：通用 nn/nt 三布局、
  split-K 三件套、双缓冲 nt/nn 各两件。decode 形状下 qkv 110→56.4 µs、o_proj →57.6、
  gate_up →72.2、down →91.3 µs，**层合计 442→277 µs**。最终热路径经实测改由 mcBLAS 承担
  （mcBLAS 152–197 TFLOPS，Triton nn 为 119.5），内核族保留为回退与参考实现。

执行路径层的改动见下表。

| 项 | 内容 |
|---|---|
| opt1/opt3 | **sort-free 采样链**：bf16 直读 + 温度折叠的 Gumbel-Max `argmax(x/T+g)`，<br>替换 eager PyTorch 的 fp32 全词表采样。`FlagGems/fused/sampler_fused.py` |
| opt7/opt11 | **top-p 直方图缩放**：默认以单轮 1024-bin 质量直方图求 top-p 阈值，<br>与 mask 采样一趟完成；`fused/top_p_sample.py`。<br>消除每步约 1.6 ms 的 `_top_k_top_p_kernel` 与 lm_head 的冗余 cast |
| opt2/opt4/opt6 | **单 nn 布局 GEMM**：只保留 `[K,N]` 副本并释放 nt 存储（-4 GB），全 M 走 mcBLAS；<br>KV 池 1,111,744 → 1,203,904 tokens（+8.3%），官方 64 并发下吞吐中性；`VLLM_FL_NN_MIN_M=192` + `VLLM_FL_FREE_NT_WEIGHTS=0` 可回到双布局；`linear_nn_repack.py` |
| opt5 | **autotune 预热（可选，默认关闭）**：模型加载后按 (M,K,N) 触发一次，<br>避免运行时 autotune 抖动。早期默认开启时 4k 曾有 +5.9% 的单次记录，<br>但当前路由下分桶大多走不到，官方口径无收益，且冷启动被拉长到 6–19 分钟 |
| opt8 | **prefill GEMM 直调 vendor mcBLAS**：`patches/mcblas_mm.py` 用 ctypes<br>直绑 `libmcblas` 的 `mcblasGemmEx`（cuBLAS 兼容 ABI），<br>注册为 custom op 保证 compiled 区域安全 |
| opt10 | **消除 prefill 每层 DtoH 同步**：在 GPU 上对 `prefill_seq_lens`<br>执行 `cat` + `cumsum` 构建 attention metadata，消除每层两次隐式同步（4k +10.9%、16k +9.9%） |
| opt11 | lm_head nn 布局路由 + top-p zoom 单轮 1024-bin 定稿 |

## 4. 复现

### 4.1 安装

在官方镜像的原 Python 环境中操作，先备份原源码目录，再将包内两个目录完整替换到 `/workspace`，不要叠加解压到旧源码中。官网给出的安装选项为 `pip install --no-build-isolation`；以下补上当前目录 `.`。

两个项目的构建工具版本要求不同：插件要求 `setuptools>=77.0.3,<81`，FlagGems 要求 `setuptools>=64,<77`。禁用构建隔离时须按安装顺序分别准备，不能用同一个 setuptools 版本满足两份声明：

```bash
python -m pip install "setuptools>=77.0.3,<81" "setuptools-scm>=8,<10" "wheel==0.46.2"
cd /workspace/vllm-plugin-FL && pip install --no-build-isolation .
python -m pip install "setuptools>=64,<77" "setuptools-scm>=8,<10" "wheel==0.46.2"
cd /workspace/FlagGems && pip install --no-build-isolation .
python -m pip check
```

**实例无外网时**：跳过上面三行 `pip install "setuptools..."`，直接用镜像自带的 `setuptools 80.10.2`、`setuptools-scm 10.0.5`、`wheel 0.45.1` 跑两条 `pip install --no-build-isolation .`。2026-11-19 在一台无 PyPI 的 C500 实例上按此路径装完，`import flag_gems, vllm_fl` 正常，日志见 `docs/experiments/runs/c500-official-flag-20261119/install-*.log`。

使用镜像自带的厂商 PyTorch、Triton、vLLM，不另外安装通用 GPU 依赖或可选 extras。插件的 `fallback_version` 已由上游提供，FlagGems 增加该字段以支持不含 `.git` 的源码构建。构建工具检查不替代目标 GPU 上的服务启动和评测。

**编译与执行流程**：两个包都是纯 Python + Triton，没有需要单独编译的 C++ 扩展，`pip install` 完成构建即安装完毕；Triton 内核在首次遇到对应形状时 JIT 编译，因此首次启动或首轮 benchmark 偏慢属正常，官方脚本弃首轮也正是为此。`pip check` 在干净镜像上除镜像自带的 3 条版本偏差（compressed-tensors、fastapi、opencv-python-headless）外不应有输出，出现其他缺失依赖说明安装环境不完整。安装完成后用 `python -c import` 确认模块位置，再进入 4.3 启动。

### 4.2 环境变量

沿用官方镜像初始化的 PATH、MACA/COREX 和动态库环境，仅按官网设置：

```bash
export VLLM_PLUGINS=fl
```

不要覆盖镜像的 `LD_LIBRARY_PATH`，也不要用 `PYTHONPATH` 将旧源码放在已安装包之前。安装后在 `/workspace` 核对模块实际位置，确认使用当前 Python 环境中的安装产物：

```bash
cd /workspace
python -c 'import vllm_fl, flag_gems; print(vllm_fl.__file__); print(flag_gems.__file__)'
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

**官方复核时的前缀缓存开关**：赛事方在**性能测试**的启动命令上额外加 `--no-enable-prefix-caching`，**数学评测不加**。两平台分别对应：

```bash
# 性能测试用（天数）
vllm serve /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --no-enable-prefix-caching

# 性能测试用（沐曦 C500）
vllm serve /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --no-enable-prefix-caching
```

2026-11-19 已按官方口径（追加该开关）在 C500 上完整复测：4k/16k 后三轮均值 **12430.31 / 11106.09 tok/s**、平均 TTFT 1625.04 / 17996.83 ms，prefix cache 命中率 **0.0%**；与默认命令的 12451.84 / 11097.29 相差 **−0.17% / +0.08%**，说明该开关对本次提交无实质影响。逐轮数据与日志见 `docs/experiments/runs/c500-official-flag-20261119/`。

### 4.4 简单冒烟测试（可选）

启动后先用赛题页面给出的最小请求确认服务可用，再跑正式评测：

```python
import requests
response = requests.post(
    'http://127.0.0.1:9031/v1/chat/completions',
    headers={'Content-Type': 'application/json'},
    json={
        'model': 'minicpm',
        'messages': [{'role': 'user', 'content': 'introduce llm'}],
        'max_tokens': 256,
    },
)
print(response.json())
```

返回 `choices[0].message.content` 非空即说明模型加载、插件路由和采样链都正常。若这里就报错，先查 serve 日志里的 `patch failed` 与回退记录，不要继续跑 benchmark。

### 4.5 性能评测

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

**注意**：本 benchmark 使用 `seed=0`，同一 serve 上连跑两次相同
`input_len` 的负载**可能命中 prefix cache**，须查看 serve 日志确认，不能认定必然命中。
正式评测保持官方 `input_len` 和四轮脚本不变，并记录缓存命中率；需要独立复测时，在新启动的 serve 上重新执行完整官方脚本，不通过更改输入长度规避缓存。

### 4.6 精度评测

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

**数据集准备**：上述命令显式使用本地目录 `/workspace/evalscope-datasets/math_500`，不使用自动下载。

- 镜像已带该数据集时（推荐）：目录下应有 `eval.yaml`、`test.jsonl`、`README.md`，其中 `test.jsonl` 共 500 行。
- 需要自备时：把 MATH-500 的 `test.jsonl`（字段为 `problem`、`solution`、`answer`、`subject`、`level`、`unique_id`）与 `eval.yaml` 放进同一目录，并把 `--dataset-args` 的 `dataset_id` 指向该目录。
- 子集自检（Level 3 必须正好 105 题，与报告口径一致）：

```bash
python - <<'PY'
import json, collections
rows = [json.loads(l) for l in open('/workspace/evalscope-datasets/math_500/test.jsonl')]
print(len(rows), dict(collections.Counter(r['level'] for r in rows)))
PY
```

预期输出 `500 {1: 43, 2: 90, 3: 105, 4: 128, 5: 134}`；`level` 是整数，评测按 `subset_list: ["Level 3"]` 取其中 105 题。

### 4.7 结果校验

1. 吞吐：按四轮去首、后三轮均值与第 1 节比对，保留所有逐轮结果。按上述工作目录运行时，原始 CSV 位于 `/workspace/benchmark_results/raw_runs_<timestamp>.csv`，汇总为同目录 `summary_<timestamp>.csv`。波动与场景、实例有关，历史测量也有超过 1% 的情况。
2. 精度：`math_500.json` 的 `metrics[0].score` 应不低于 0.95；105/105 是请求成功数，不是答对题数。
3. 启动参数：`vllm serve` 命令行与 4.3 节一致，沿用镜像环境与默认优化开关。
4. 缓存：记录 prefix cache 命中率；本报告各次测量的峰值远低于 1%（BI 两次复测为 0.3%），这是实测状态而非官方额外门槛。C500 最终轮的 serve 日志未回传，无法就那一次单独核实，相邻两轮峰值为 0.7%。官方性能复核会用 `--no-enable-prefix-caching` 关闭前缀缓存，因此该指标只影响本队自测对照，不影响官方口径；异常偏高时检查是否复用同一服务的重复负载，必要时新起服务重测。
5. 生效核对：检查 `patch failed` 或回退警告。BI 的 `vendor flash prefill ENGAGED` 记录首次进入该分支，不能单独证明执行成功；还需核对后续异常。`decode_attn_fast slice` 仅每累计 4000 次混合步路径调用输出一次，缺少该行不能判定纯 decode 路径失效。C500 的 1,203,904 tokens 是本次实例的 KV 池测量值，不是固定验收值；池大小还受可用显存影响。安装路径、日志和实际负载的内核执行情况应结合判断。

### 4.8 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `import flag_gems` 报 `No module named 'flag_gems.fused.DSA'` | 用了未包含占位 `__init__.py` 的旧包，或把新包叠加解压到旧源码目录 | 用本包解压到干净目录后重新安装；本包已在 `FlagGems/src/flag_gems/fused/DSA/` 放入占位文件 |
| `pip install "setuptools>=77.0.3,<81"` 报网络错误 | 实例无外网，PyPI 不可达 | 跳过这步，用镜像自带的 setuptools 直接 `pip install --no-build-isolation .`，见 4.1 |
| `pip check` 报 setuptools 版本不符 | 插件要求 `>=77.0.3,<81`、FlagGems 要求 `>=64,<77`，两者没有交集 | 按 4.1 的顺序分两次安装；装完可执行 `pip install "setuptools>=77.0.3,<81"` 恢复，不影响已装好的两个包 |
| C500 启动阶段进程被杀 | 64 GB CPU 内存实例在 Inductor 编译期触顶，纯镜像原版同样会复现 | 换 ≥128 GB 内存的实例 |
| 吞吐偏高且 TTFT 异常低 | prefix cache 命中（benchmark 用 seed=0） | 新起 serve 重测，或换 `input_len` 后再测 |
| 首次启动或首轮很慢 | Triton 内核首次遇到形状时 JIT 编译 | 正常现象；官方脚本弃首轮，报告值取后三轮 |
| BI 上出现 `patch failed` 或回退记录 | 插件路由未命中，走了原始路径 | 核对启动时 `VLLM_PLUGINS=fl`、安装路径与 4.7 节的生效检查项 |

## 5. 实验记录与事实来源

当前交付策略以本文第 2、3 节和实际代码为准。历史探索记录仅用于追溯，
不作为后续开发的直接输入；引用前须核对代码版本、硬件规格、测量口径和原始日志。
`docs/deliverables/audit-2026-10-25.md` 只是当时的事实审计，最终提交以本说明和
`docs/experiments/runs/image-baseline-align/` 为准。

下表路径指向文档与实验记录仓库（https://github.com/xy200303/minicpm-flagserve）中的台账；官方提交 ZIP 按要求只包含源码、
本说明和 `report.pdf`，不会重复打包大体积原始日志。评审复现以本文命令、
报告中的关键结果和提交源码为准；BI-V150 的交付提交复测已于 2026-11-17 完成，证据见第 5 节表格。

最终复测原始证据均保存在该文档仓库，可用以下路径直接核查：

| 结果 | 原始证据 |
|---|---|
| C500 最终提交性能与精度 | `docs/experiments/runs/c500-final-e4b5854/` 下的 `bench-4k.log`、`bench-16k.log`、`math_500.json`、服务日志和 manifest |
| C500 本机原生基线 | `docs/experiments/runs/c500-final-audit/healthy-baseline-{raw,summary}.csv` 与 serve／bench 日志 |
| C500 两次精度与微基准 | `docs/experiments/runs/c500-final-audit/` 下的 `healthy-eval1-math500.json`、`healthy-eval2-math500.json`、`micro-healthy.json` |
| BI-V150 两组 A7/A20 | `docs/experiments/runs/bi-final-audit/{1-a7,1-a20,2-a7,2-a20}/benchmark_results/` 及同级 serve／bench 日志 |
| BI-V150 精度与内核验证 | 同目录 `eval1-math500.json`、`eval2-math500.json`、`micro.json`、`pytest-a20.log` |
| BI-V150 A31 4k/16k 与精度 | `docs/experiments/runs/a31-config-local-cache/` 下的 `a31-official-evidence.tar.gz`、`a31-16k-evidence.tar.gz`、`a31-eval.tar.gz`、`host.prof` |
| BI-V150 A48 性能与精度 | `docs/experiments/runs/a48-dual-acc-silu-mul/` 下的官方汇总、逐轮 CSV／日志与 `eval/` |
| BI-V150 A60 性能与精度 | `docs/experiments/runs/a60-tile-packed/` 下的 `official-{4k,16k}-summary.json`、逐轮原始工件、`gems-tests-v5.log` 与 `eval/` |
| BI-V150 镜像对齐最终复测 | `docs/experiments/runs/image-baseline-align/` 下的 `bench-evidence.tar.gz`、`eval-evidence.tar.gz`、`tests.tar.gz` 与 `manifest.txt` |
| 双平台环境与固定提交 | 两目录各自的 `manifest.txt`；C500 为 `healthy-final-manifest.txt` |
| BI-V150 交付提交复测（2026-11-17） | `docs/experiments/runs/bi-final-official-replay-20261117/`（按官方安装与评测命令的完整回放：安装日志、逐轮 CSV、bench/serve/eval 日志、MATH-500 JSON、manifest）与 `docs/experiments/runs/bi-final-retest-20261117/`（同日可编辑安装的对照） |
| BI 官方复验版本（FlagTree 0.7.1a2，2026-11-19） | `docs/experiments/runs/bi-flagtree-071a2/`：安装日志、内核探针、逐轮 CSV、bench/serve/eval 日志、两次 MATH-500 JSON 与 manifest |
| C500 官方口径复测（2026-11-19） | `docs/experiments/runs/c500-official-flag-20261119/` 下的 `raw_runs_20261119_112157.csv`、`summary_20261119_112157.csv`、`bench.log`、`serve.log` 与 `manifest.txt`（启动命令追加 `--no-enable-prefix-caching`） |

官网基线来源为[赛道二页面](https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn)“4.3 评审方案”的两张基线表（访问日期 2026-11-11）。原图归档于 `docs/deliverables/latex/official_baseline_biv150.png` 和 `official_baseline_c500.png`，报告附录给出原始图片链接。

天数最终 FlagGems 基线以镜像提交 `4df52d91` 为准，而不是公开 tag 另外包含的 7 个提交；FlagTree 按赛事方口径升到 `0.7.1a2+iluvatar3.6`（镜像原为 `0.6.2a3`），C500 维持镜像环境、未安装 flagtree。两版 FlagTree 的数据都在第 5 节索引里。