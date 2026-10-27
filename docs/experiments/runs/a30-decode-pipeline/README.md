# A30：paged decode 页循环流水与尾页拆分（2026-10-27）

## 代码与回退

- 基线：插件 `0207b7e1`、FlagGems `aff68555`。
- 实验：插件 `2fb32ad`、FlagGems `6b13c991`，均为 `exp/a30-decode-pipeline` 分支。
- 默认 `split_tail=False`，交付行为不变。候选仅在插件设置 `VLLM_FL_DECODE_FAST_SPLIT_TAIL=1` 且使用实验 FlagGems 时启用。
- 回退：恢复基线两个源码路径，或候选环境取消该开关。实验不改变 `vllm serve` 参数。

## 微基准口径

`docs/verification/bi_a30_pipeline_probe.py`；B=64、16 q heads / 2 KV heads、head_dim=128、page=16、NSEG=32、num_warps=1。每个配置 30 次 CUDA event 计时，kv=4096/16384，覆盖等长与 ragged；stock fp32 路径用于数值对照，JSON 保存全部样本。

基线现有测试 **9 passed**。候选新增页边界测试后为 **10 passed**，覆盖 1、15、16、17、255、256、257、4095 token，候选与当前实现逐位一致。

## stage 扫描：否决

| 场景 | stage 1 ms | stage 2 ms | stage 3 ms | 结论 |
|---|---:|---:|---:|---|
| 4k 等长 | 0.5891 | 0.6322 | 0.6552 | stage 2/3 更慢 |
| 4k ragged | 0.4418 | 0.4650 | 0.4773 | stage 2/3 更慢 |
| 16k 等长 | 2.1416 | 2.2830 | 2.3686 | stage 2/3 更慢 |
| 16k ragged | 1.9916 | 2.1210 | 2.1954 | stage 2/3 更慢 |

原始文件：`stage-probe.json`、`stage-probe.log`、`pytest-baseline.log`。stage 2/3 不进入官方长测。

## 完整页与 ragged 尾页拆分：通过微基准筛选

候选把 `_segment_kernel_paged` 热循环中的逐页分支移出：完整页只执行无 mask 循环，唯一不完整尾页单独处理。数学顺序、NSEG、scratch 和 combine 均不变。

| 场景 | 当前 A20 ms/层 | 候选两次中位数 ms/层 | 提升 |
|---|---:|---:|---:|
| 4k 等长 | 0.587712 | 0.544017 | **8.03%** |
| 4k ragged | 0.441248 | 0.396915 | **11.17%** |
| 16k 等长 | 2.139106 | 1.867355 | **14.55%** |
| 16k ragged | 1.989900 | 1.744195 | **14.09%** |

原始文件：`current-2.json`、`split-1.json`、`split-2.json`、对应 log 与 `pytest-split.log`。这些数字只用于筛选，不作为端到端成绩。

## 官方 A/B

按 A/B/A/B 顺序，每个数据点新启动 serve，官方脚本四轮去首均值。4k 为 `[[4096,1024,64,256]]`。运行脚本：`docs/experiments/harness/run_bi_case.sh`。

| 运行 | 路径 | 总吞吐 tok/s | TTFT ms |
|---|---|---:|---:|
| A1 | 当前 A20 | 4815.68 | 4098.57 |
| B1 | 尾页拆分 | 4158.66 | 4176.34 |
| A2 | 当前 A20 | 4716.31 | 4148.93 |
| B2 | 尾页拆分 | 4192.17 | 4161.20 |
| **中位数 A** | 当前 A20 | **4766.00** | 4123.75 |
| **中位数 B** | 尾页拆分 | **4175.42** | 4168.77 |

候选相对下降 **12.39%**，正式否决，不合入，也不继续做 16k 官方长测。完整四目录、serve/bench 日志、配置和原始 CSV 见 `a30-4k-official.tar.gz`，并已展开保存。

补充批量扫描：B=16/32/64、4k/16k、等长/ragged 的所有点上，候选孤立微基准仍有 **7.68%–20.25%** 收益（`batch-scan-{current,split}.json`）。因此官方回退不能由 batch 形状解释，需继续查 CUDA graph 与组合执行路径。
