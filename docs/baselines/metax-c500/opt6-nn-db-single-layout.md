# 优化 #6：nn_db 内核与单 nn 布局探索（结论：保留为备选，定稿双布局）（MetaX C500，2026-09-30）

## 动机

opt4 的 nn 重排让 prefill 吃到了 MACA 流水，但代价是权重双份存储（+4GB）。
问题：能否把 decode 也迁到 nn 布局，全模型只存一份？

## nn_db：nn 布局的 skinny-M 双缓冲内核

把 nt_db 的手动 K 循环双缓冲移植到 nn 布局（B 为 [K,N] 行主序），
FlagGems `_metax/ops/mm.py` 新增 `mm_kernel_nn_db` 与 `mm_kernel_nn_db_splitk_partial`，
并把 `mm()`/`mm_out()` 的 dispatch 顺序调整为 **nt_db/nn_db 先于通用 split-K 场景**
（原顺序会让 skinny-M nn 形状落入慢速的 splitk 旧路径）。

关键教训（dispatch 取证）：内核正确性全过但首轮慢 8.5×——因为
`torch.mm` 的 nn 形状先被 `splitk_mm_scenario` 截胡，且钉死的 num_warps=4 配置错误。
扫描修复后（BN=32 喂满 104 个 SM + num_warps=8 + BK=128），M=64 逐形状对决：

| GEMM | nt_db | nn_db | 比值 |
|---|---|---|---|
| qkv (2048→2560) | 47.4µs | 44.9µs | 0.95 |
| o_proj (2048→2048) | 46.8µs | 43.9µs | 0.94 |
| gate_up (2048→12288) | 103.4µs | 90.2µs | 0.87 |
| down (6144→2048) | 80.5µs | 94.2µs | 1.17 |
| **层合计** | 278.1µs | **273.2µs** | **0.982** |

内核层面单布局可行：decode 用 nn_db 不输 nt_db，prefill 用 mm_nn，
权重单份（实测 KV 池 1,127,888 → 1,219,664 tokens，+8%）。

## 端到端 A/B（同一台健康机器，官方 benchmark，官方聚合口径=第2~4跑均值）

| 方案 | 4k total tok/s |
|---|---|
| B 双布局（nt_db + nn 分流） | **8135** |
| A 单 nn 布局（nn_db 通吃） | 8034 |

单布局 ~1.2% 落后（噪声边缘），双布局保持为我们测到的最大值，且与 opt1-4
的验证链一致。**定稿双布局**；单布局经 `VLLM_FL_NN_MIN_M=0 +
VLLM_FL_FREE_NT_WEIGHTS=1` 保留为显存紧张部署的备选（报告写作「未来方向」）。

## 最终成绩（双布局定稿，健康机器，官方 benchmark）

| 场景 | 官方基线 | 定稿 | 提升 |
|---|---|---|---|
| 4k total tok/s | 5089.645 | **8135** | **+59.9%** |
| 16k total tok/s | 7029.675 | **8683** | **+23.5%** |

（注：数字与昨日老 VM 的 8840/8858 存在宿主差异；同台对比均内部一致。
以官方评测机复测为准。）

## 附：平台问题调查（与代码无关，已证伪所有嫌疑）

评测期间模力方舟 64GB 内存规格实例（.53×3、.182×2）反复在启动/压测时
SIGKILL 服务。对照实验：镜像原版代码同样被杀；tiny 负载被杀；单个
`vllm bench` 子进程即可触发；而 60GB 内存加压、54GB 显存爬升、128 stream、
8 个 CUDA 子进程、safetensors+H2D 单项探针全部通过。

**2026-10-04 根因实锤（.182 纯镜像原版，启动即复现 4/4）**：cgroup v2
`memory.events` 里 `oom_kill` 计数随每次启动 +1（1→2→3→4），1s 级采样抓到
完整曲线——vLLM 启动阶段匿名内存以 ~2GB/s 爬升，**inductor compile_worker
子进程每个占 ~19-22GB RSS**，加上 APIServer/EngineCore，峰值冲到 63.4GB
撞穿 64GB cgroup 上限，内核 OOM killer 杀 EngineCore（GPU 显存当时仅
858MiB，与显存无关）。128GB 内存规格实例（.81）峰值远未触限，全程稳定。
结论：64GB 内存规格装不下这套栈原版启动的瞬时峰值，属平台规格问题，
与我们的代码、vllm 版本、负载模式均无关。（早期「57GB 空闲、cgroup 计数为 0」
的观测是死亡后采样/粗采样漏掉尖峰所致，以本次 1s 级 cgroup 取证为准。
证据：cgroup_watch_stock_182.log、stat_watch_stock_182.log。）

工具：tools/mem_pressure.py、tools/gpu_pressure.py、
tools/mini_bench.py（轻量压测客户端，绕开 12GB 的 vllm bench 子进程）。

**2026-10-05 追加（.81 新实例）**：stream 创建级死锁。`torch.cuda.Stream()` 挂死
（faulthandler 栈：triton autotuner `_bench` → `Stream.__new__`），基本 CUDA
算子正常；杀光持卡进程后仍挂死 → 驱动/虚拟化层 wedge，容器内不可恢复，
只能重启实例。触发点：verify_nt_db 首个 fp32 GEMM 的 autotune bench。
教训：这台栈上 LibTuner bench 会建 stream，实例不稳定时表现为「测试卡死」
而非崩溃，诊断路径：faulthandler.dump_traceback_later → 最小 stream 复现。
