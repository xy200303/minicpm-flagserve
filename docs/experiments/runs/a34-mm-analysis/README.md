# A34：BI GEMM 组合路径调查（2026-10-28，否决）

## 范围与基线

基线为 A20+A31 当前交付栈，MiniCPM5-2B 的四个 GEMM 形状均使用已固定的 Iluvatar vendor 配置。A4 已完成主要 tile 配置扫描；本项只调查尚未覆盖的 `GROUP_M` tile 遍历顺序，不重复扫描 block 形状，也不把理论占用率直接外推为端到端收益。

## 实测

`prefill_gateup` 的 ixkn 理论占用率约 12.5%，受 189 个 vector register/thread 和 65536 bytes/block 动态共享内存限制；Memory Access Throughput 约 501.36 GB/s，L2 hit rate 79.90%。低占用率本身不能证明存在更快的内核。

固定其他参数，只改变 `GROUP_M`，每点 20 次 CUDA event 计时并取中位数：

- 首轮 2/4/8/16：`prefill_gateup` 在 16 时由 1.2963 降至 1.2601 ms，快 2.87%；其余形状均小于 1%。
- 扩展 8/16/32/64 复测：`prefill_gateup` 最佳为 64 的 1.2699 ms，相对 8 的 1.2919 ms 仅快 1.73%，首轮 2.87% 未稳定复现。
- 七个 decode 形状的最优变化均不超过 1%，且不同轮次最佳值漂移；prefill 其余形状也不超过 0.11%。

结论：`GROUP_M` 只改变 tile 顺序，没有形成稳定且足以影响官方总吞吐的候选。未修改集成分支，不执行官方长测，不合入。回滚即保持当前集成 SHA，无运行时补丁。

## RMSNorm 可行性旁查

在当前 BI 环境导入平台 IR kernels 后，`rms_norm` 和 `fused_add_rms_norm` 注册实现均为 `native/aiter/oink/vllm_c/xpu_kernels`，没有 FlagGems 实现；FL 默认优先级为 `native`。因此不存在可保留 Inductor 融合的现成 FlagGems 通道。新增 opaque custom op 会重复已否决的 A21，故停止该方向，不产生新补丁。

## 证据

- `vendor-all.json`：当前八个 GEMM 形状的基线。
- `decode-ixkn.log`、`prefill-ixkn.log`：资源与访存分析。
- `group-m.json`、`group-m-wide.json`：两轮原始计时。
- `machine-state.txt`：机器、GPU、源码 SHA 与软件版本。
- `rmsnorm-registry.txt`：实际 IR 实现注册表。
- 探针：`docs/verification/bi_a34_mm_probe.py`、`bi_a34_group_m_probe.py`。
