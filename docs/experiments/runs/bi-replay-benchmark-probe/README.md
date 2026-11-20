# FlagTree replay benchmarker 探针（2026-11-19）

## 背景

在 BI-V150 上启动服务时，日志会反复出现：

```
RuntimeWarning: active Triton backend does not provide a registered replay
benchmarker; falling back to event timing
```

FlagGems 的 autotune（`LibTuner`）希望用「graph replay」方式测每个候选 config：
把 kernel 调用录进 CUDA graph，再重放计时，比 CUDA event 计时少了 launch 开销。
是否可用由 FlagTree 侧的白名单决定，白名单按驱动模块名硬编码：

```python
_REPLAY_IMPLEMENTATIONS = {
    "triton.backends.nvidia.driver": "triton_cuda_graph_replay_v1",
    "triton.backends.amd.driver":    "triton_hip_graph_replay_v1",
    "triton.backends.metax.driver":  "triton_metax_graph_replay_v1",
    ...
}
```

天数的驱动是 `triton.backends.iluvatar.driver`，不在名单里，于是每次 autotune 都降级为
event 计时并打一条警告（一次 serve 生命周期内 125 条）。

## 探针一：CoreX 上 graph replay 是否可用

在 BI-V150 上用一个小 kernel 直接调用 `triton.testing.do_bench_cudagraph`：

| 方式 | 单次耗时 |
|---|---|
| event 计时 | 5.02 µs |
| graph replay | 2.58 µs |

差值就是 launch 开销，说明 **CoreX 支持 CUDA graph 计时**，技术上可以打开。

## 探针二：打开白名单后的端到端 4k

在 Triton 侧白名单补一行 `"triton.backends.iluvatar.driver"`，清空调优缓存后重启服务：

| 项目 | 默认（event） | 打开 replay |
|---|---|---|
| 警告条数 | 125 | **0** |
| KV 池 | 522,032 | 523,152 |
| 4k 第 1 轮 | 4048.35 | 4042.29 |
| 4k 第 2 轮 | 4796.58 | 5016.31 |
| 4k 第 3 轮 | 5101.64 | 4952.82 |
| 4k 第 4 轮 | 5237.38 | 5197.29 |
| 后三轮均值 | **5045.20** | **5055.47** |
| 平均 TTFT | 4149.48 ms | 4112.07 ms |

**差 +0.20%，在运行波动内**。也就是说：警告可以消掉，但端到端吞吐没有变化，
KV 池多出的 ~1.1k tokens 也不足以形成可见收益。

## 启动耗时的对照（重要）

同一台机器上三次启动的时间线（本地时间）：

| 场景 | serve 首行 | 打印 KV cache | 服务可响应 | 总耗时 |
|---|---|---|---|---|
| 默认（无补丁） | 19:50:31 | 19:51:59（+88s） | **19:54:09** | **≈218s** |
| 有补丁 + 提前开跑 | 22:59:33 | 23:00:47（+74s） | 未确认 | — |
| 有补丁 + 等就绪 | 23:18:27 前 | 23:18:27 | ≥23:22:59 | **≥270s** |

两点结论：

1. **提前开跑那次**在 `+109s` 就启动了基准，而**即使不打补丁**这台机器也需要约 218s 才能响应，
   所以 `+109s` 在任何配置下都会 `ConnectionRefused`。该次 0 tok/s 由时序造成，与补丁无关。
2. **补丁确实会拖慢启动**：无补丁约 218s，有补丁至少 270s（graph replay 为每个候选 config
   做一次 CUDA graph 捕获，显著增加 warmup 时间）。因此即便吞吐无差异，**也不建议启用**——
   它会让启动更慢、更依赖就绪判断，对按 4 轮计分的评测没有好处。

## 一次误判的记录

首次尝试时，第 1、2 轮返回 `0.0 tok/s`，一度被判断为「replay 基准与服务 CUDA graph
冲突」。复查 `bench-replay.log` 后确认真实原因是
`ConnectionRefusedError: Connect call failed ('127.0.0.1', 9031)`
——基准脚本在服务打印 `Application startup complete` 之前就启动了，请求全被拒。
按正确时序（等 startup complete 后 curl `/v1/models` 返回 200 再跑）重测，四轮全部
`SUCCESS`、`Successful Requests = 256`，即上表结果。

## 结论与处置

- 该警告是**预期降级**，不影响正确性，也不影响吞吐。
- 打开 replay 没有可测收益（+0.2%），却要改 FlagTree 安装包内的文件，属于环境改动，
  不在交付范围内，**因此不采用**，交付保持默认。
- 若评审看到该警告，不需要处理；也不要为了消警告去改白名单。

## 证据

- `replay-bench-4k.csv`：打开 replay 后的四轮原始数据
- `replay-bench-4k.log`：同一次运行的基准日志（含 `SUCCESS` 与 256/256 成功请求）
