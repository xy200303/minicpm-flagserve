# A41：C500 16k 后续诊断与候选筛选（2026-11-01）

## 已完成

- 当前交付官方 16k：11143.77 tok/s、TTFT 17961.76ms。
- A40 独立 attention 微基准和 split-KV 全扫描完成，自动 `num_splits=0` 最快，候选否决。
- 组合消融已完成，当前 mcBLAS、采样等路径在最终栈中没有相互回退。

## 诊断阻塞

1. rank0 torch profiler：厂商运行时报告 `External init callback must run in same thread as registerClient`，随后 EngineDeadError，未生成 rank0 trace。证据：`c500-torch-profile-16k-serve.log`。
2. `execute_model` cProfile：服务完成启动，但首轮请求时 EngineCore 退出，四轮均为 FAILED。原始 `c500-current-16k-host-raw.csv` 保留为无效诊断记录，**不用于性能结论**。serve 日志：`c500-current-16k-host-serve.log`。

这些失败只发生在诊断包装路径，不影响未包装的官方 16k 基线。当前不再重复启动同类 profiler，也不根据缺失的 host 数据推断新候选。

## 当前判断

C500 4k/16k 的主要设备路径（attention、GEMM、采样）已完成实测；自动 split-KV、mcBLAS algo、oplist host 优化均未达到采纳门槛。继续优化需要新的实现机制或可用的厂商 trace，不能重复扫描相同参数或用微基准外推。BI-V150 的 A36 全页循环候选仍是下一阶段优先验证项。
