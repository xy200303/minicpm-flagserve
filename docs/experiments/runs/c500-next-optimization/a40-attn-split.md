# A40：C500 16k attention 与 split-KV 调查（2026-11-01，否决）

## 测量与约束

16k 官方四轮基线此前已完成：11143.77 tok/s、TTFT 17961.76ms。尝试补抓 rank0 torch profiler 时，厂商运行时报告 `External init callback must run in same thread as registerClient`，随后 EngineDeadError；这是诊断路径故障，不改变官方基线，也不作为性能结论。日志保留为 `c500-torch-profile-16k-serve.log`。

改用独立 `flash_attn_with_kvcache` 微基准，B=64、16 Q heads、2 KV heads、head_dim=128、bf16、随机分页 KV：

| KV 长度 | 实测 | 1.6TB/s 理论下限 | 比值 |
|---|---:|---:|---:|
| 4096 | 211.9µs | 168µs | 79% |
| 16384 | 765.4µs | 671µs | 88% |
| 32768 | 1500.1µs | 1342µs | 89% |

原始输出：`c500-attn-decode-roofline.txt`。理论下限只用于判断是否值得继续扫实现，不等同于 vendor 实测峰值。

## split-KV 扫描

对 16k 扫描 `num_splits=0/1/2/4/8/16/32/64`，每点 30 次并保留全部样本：

- 自动 `num_splits=0`：**0.78464ms**，为全部配置中最快。
- 手动 16/32/64 分别为 0.8074/0.8027/0.8431ms，均更慢；1/2/4/8 更慢。
- 各配置与参考输出的 max diff 为 0.000244，属于 bf16 计算路径差异；没有采用任何候选。

自动 split-KV 已优于手动配置，A40 不修改集成代码、不执行官方长测。原始文件：`a40-attn-split-16k.json`，探针：`c500_attn_split_probe.py`。
