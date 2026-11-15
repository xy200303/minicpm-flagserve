# C500 最终提交复测（插件 58b750e / FlagGems ef516137，2026-11-16）

本轮把 GEMM autotune 预热改为**默认关闭**后，从冷缓存重跑官方用例与精度评测。

## 环境与代码

| 项 | 值 |
|---|---|
| 实例 | MetaX C500 64 GiB，CPU 内存 128 GiB |
| 插件 | `58b750e`（`integration/vllm-plugin-fl`） |
| FlagGems | `ef516137`（`integration/flaggems`） |
| 预热 | 未设置 `VLLM_FL_MM_WARMUP`，取新默认（关闭） |
| 缓存 | 启动前清空 `~/.flaggems/config_cache/*.db` 与 `~/.triton` |
| 服务参数 | 与官网一致，未追加任何参数 |

## 启动

清冷缓存后 `Application startup complete` 用时 **124 秒**（`timing.txt`）。此前默认开启预热的版本冷启动为 367–1129 秒。

## 官方性能（四轮，官方口径取后三轮均值）

| 场景 | 逐轮 total tok/s | 后三轮均值 | 相对官网 | TTFT 均值 |
|---|---|---:|---:|---:|
| 4k `[4096,1024,64,256]` | 11818.93 / 12429.38 / 12448.58 / 12473.16 | **12450.37** | 2.45× | 1629.09 ms |
| 16k `[16384,1024,64,128]` | 11036.71 / 11096.24 / 11094.72 / 11094.47 | **11095.14** | 1.58× | 18031.98 ms |

全部请求成功，失败 0。4k 首轮比后三轮低 5.1%，是冷缓存下一次性调优/编译落在首个请求批次的成本；官方脚本弃首轮，不计入。

## 精度

MATH-500 Level 3：**98.1%（105/105）**，`math_500.json`。

## 单布局复测（A/B，未改默认）

同日同栈，用 `VLLM_FL_NN_MIN_M=0 VLLM_FL_FREE_NT_WEIGHTS=1` 释放 nt 权重副本，见 `ab-single-layout/`：

| 项 | 双布局（交付默认） | 单布局 |
|---|---:|---:|
| KV 池（tokens） | 1,111,744 | 1,204,064 |
| 16k 后三轮均值 | 11095.14 | 11130.01（+0.31%） |
| 4k 后三轮均值 | 12450.37 | 12485.28（+0.28%） |
| 冷启动 | 124 s | 110 s |

两场景同向高约 0.3%，与跨次波动同量级；单布局会释放 nt 存储并改变 `Parameter` 形状，因此保持双布局为默认。

## 文件

- `bench-4k.log` / `bench-16k.log`：官方 benchmark 原始输出
- `serve.log`：服务日志（含 KV 池、prefix cache 命中率）
- `eval-level3.log` / `math_500.json`：精度评测
- `manifest.txt`：实例、commit 与环境
- `timing.txt`：启动耗时
- `ab-single-layout/`：单布局 A/B 的 serve 与 benchmark 日志
