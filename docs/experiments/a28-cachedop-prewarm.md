# A28：上游 #552 CachedOp 图捕获前预热（C500 实测）

## 目的与边界

参考上游 `vllm-plugin-FL#552`（commit `6182281`）的思路，在模型加载完成、Dynamo/CUDA Graph 捕获前解析并冻结 `CachedOp`，避免捕获期间反复进入 `OpManager`。本实验不是 merge 上游主分支，也不改变基线 serve 参数、benchmark 或模型。

- 实验分支：`exp/a28-cachedop-prewarm`，最终提交 `25f64e2`
- 基线：`integration/vllm-plugin-fl`（`0207b7e1`）
- 平台：沐曦 C500 64GB，CPU 内存 128GB
- 测试用例：官方 4k `[4096,1024,64,256]`
- 证据目录：`docs/experiments/runs/a28-cachedop-prewarm/`

## 实现与验证

实现包括：`prewarm_cached_ops()`、编译期只调用已选实现、政策 epoch 变化时解冻、IO dump 时不冻结，以及 `VLLM_FL_PREWARM_CACHED_OPS=0` 回退开关。单元测试覆盖冻结、编译期行为、政策切换、不可用算子和 IO dump。

- `tests/unit_tests/dispatch/test_cached_op_prewarm.py`：**5 passed**（见 `a28-pytest.log`）。
- 初次实现遗漏 `import torch`，单测立即发现；随后 `import os` 的开关也补齐。两次真实启动失败后均修复，最终服务可启动并正常返回请求。
- 启动健康检查及 `/v1/models`、实际 chat 请求均通过。

## 同日冷启动 A/B

每次均全新启动 serve，官方命令和 benchmark 不变。基线 commit 与实验 commit 只差 A28。

| run | 基线总吞吐 tok/s | A28 总吞吐 tok/s | 基线 TTFT ms | A28 TTFT ms | 基线启动 s | A28 启动 s |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 12468.78 | 12453.80 | 1770.31 | 1785.44 | 115 | 115 |
| 2 | 12482.72 | 12478.80 | 1774.68 | 1762.47 | 112 | 113 |
| **中位数** | **12475.75** | **12466.30** | **1772.50** | **1773.96** | **113.5** | **114.0** |

差异：吞吐 **−0.08%**，TTFT **+0.08%**，启动 **+0.44%**。均处于本机正常波动范围。

## 结论

#552 在当前 C500 栈上**可运行且测试通过，但没有可测得的稳态吞吐收益或启动收益**，不并入 `integration/vllm-plugin-fl`。保留实验分支和测试，后续若模型/图捕获结构改变，可重新评估。报告中的既有成绩不因此改变。

## 工件

- `base-1-*`、`base-2-*`：基线两次 serve/benchmark/driver 日志与 CSV
- `exp-3-*`、`exp-6-*`：A28 两次 serve/benchmark/driver 日志与 CSV
- `a28-pytest.log`：5 个单元测试
