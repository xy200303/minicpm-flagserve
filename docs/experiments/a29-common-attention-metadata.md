# A29：上游 #442 common attention metadata graph（C500 实测）

## 目的与边界

参考上游 `vllm-plugin-FL#442`（commit `71f6148`），把 slot mapping、computed-token count 和 block-table 尾部填充合并成可捕获/回放的公共 metadata 图。它与 opt10 的 prefill `cumsum` 同步消除不是同一层优化。实验不改变基线模型、benchmark 或官方 serve 参数；仅在实验分支用 `VLLM_FL_COMMON_ATTENTION_METADATA=1` 显式开启新路径。

- 实验分支：`exp/a29-common-attn-metadata`
- 基线：`integration/vllm-plugin-fl`（`0207b7e1`）
- 平台：沐曦 C500 64GB，CPU 内存 128GB
- 测试用例：官方 4k `[4096,1024,64,256]`
- 证据目录：`docs/experiments/runs/a29-common-attention-metadata/`

## MetaX 兼容修复

上游 NVIDIA 版 `_load_ptr` 使用 `tl.pointer_type(elem_dtype)`，C500 的 Triton 编译器报 `TypeError('element_ty is a constexpr.')`，无法编译。将其实改为 `_load_ptr_i32` / `_load_ptr_i64` 两个显式类型函数后，pointer-table kernel 可编译并回放。

- `tests/unit_tests/worker/test_common_attention_metadata.py`：**4 passed**。
- `tests/functional_tests/worker/test_common_attention_metadata_graph.py`：**7 passed**（opt-in 后）。
- 这证明的是功能兼容性，不代表端到端收益。

## 同日冷启动 A/B

| 配置 | 总吞吐 tok/s | 平均 TTFT ms | 启动 s |
|---|---:|---:|---:|
| `VLLM_FL_COMMON_ATTENTION_METADATA=0` | 12505.67 | 1745.80 | 114 |
| `VLLM_FL_COMMON_ATTENTION_METADATA=1` | 12024.01 | 2305.95 | 119 |
| 新路径相对关闭状态 | **−3.85%** | **+32.09%** | +5 s |

另与基线中位数 12475.75 tok/s 相比，开启新路径低约 3.62%。关闭新路径时，同一份 A29 代码回到基线档，说明主要负收益来自该图路径，而非其它合入内容。

## 结论

#442 在 C500 上**功能可行但端到端负收益**，不并入 `integration/vllm-plugin-fl`。保留实验分支和 MetaX 指针类型修复，供后续分析 graph replay 的开销；本次交付的既有性能数字不包含它。

## 工件

- `a29-0-*`：关闭新路径的 serve/benchmark/driver 日志与 CSV
- `a29-1-*`：开启新路径的 serve/benchmark/driver 日志与 CSV
- `a29-unit.log`：4 个 metadata 单测
- `a29-graph-fixed.log`：7 个图测试
- `a29-direct2.log`：最初的 MetaX 指针类型编译错误，供复核

## 开销分解补充

为排除“只是 graph replay 慢”的假设，在 C500 上用同一 `MultiGroupBlockTable` 做 200 次同步测量（decode 形状，64 请求）：

| 路径 | 中位单步耗时 |
|---|---:|
| 原始 `BlockTable.compute_slot_mapping` + computed-token 更新 | 0.0870 ms |
| 公共 metadata eager | 0.0655 ms |
| 公共 metadata graph replay | 0.0449 ms |

强制 `VLLM_FL_COMMON_ATTENTION_METADATA_NO_REPLAY=1` 后，官方 4k 为 **12028.50 tok/s**，与 replay 的 12024.01 基本同档，说明 replay 不是负收益根因。

进一步对相同 8 请求的 rank0 trace 汇总，基线与 A29 的 kernel 总时长分别为 **3312.37 ms 和 3310.99 ms**，差约 −0.04%。`flash_fwd_splitkv`、mcBLAS GEMM 等主要 kernel 的时间也几乎一致。结合 eager/no-replay 对照，当前不能把负收益归因于某个 metadata kernel；更像 host 侧执行路径或图交互的差异，但尚未定位到可安全修复的单点。保留这些原始分析，避免把“上游有优化”误判为“本卡必然受益”。

## 最终取舍

A29 不进入交付分支。若未来继续研究，应先用生产形状下的 host 侧分段计时定位 `unified_attention_with_output` 附近的额外开销，而不是直接采用上游实现。完整汇总见 `a29-kernel-summary.txt`、`a29-profile-decode.log`、`profiler_out_0.txt`；大体积 trace 文件保留在实验实例上，未复制进文档仓库。
