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

## host 函数级补充

对相同 8 请求的 `python_function` trace 进一步核对：`_run_common_attention_metadata` 在 A29 中总计约 **65.9 µs**，`_build_attention_metadata` 由 226.77 ms 增至 230.76 ms（+1.76%），`execute_model` 由 4.929 s 增至 5.017 s（+1.77%）。主要 `flash_fwd_splitkv` 与 GEMM kernel 总时长则几乎不变。`synchronize_input_prep` 的等待时间有所增加，但它不是 metadata kernel 本身，且现有数据不足以支持一个无风险的局部修复。

因此，A29 的最终判断是：**功能已验证，端到端负收益已确认，但没有证据支持把 replay 或某个公共 metadata kernel 当成唯一根因，也不支持为了上游形式而强行合入。** 实验分支和指针类型修复保留作后续定位用。

## 16k 完整四轮补充验证

在 commit `85cc2f0` 上对同一份代码仅切换 `VLLM_FL_COMMON_ATTENTION_METADATA`，使用官方 16k `[16384,1024,64,128]` 用例。每组冷启动后连续运行 4 轮，全部 128/128 请求成功，按官方脚本丢弃首轮、对后三轮取均值：

| 配置 | 总吞吐 tok/s | 平均 TTFT ms | 启动 s |
|---|---:|---:|---:|
| 关闭新路径 | 11168.87 | 17926.92 | 112 |
| 开启新路径 | 11161.46 | 17929.82 | 113 |
| 开启相对关闭 | **−0.07%** | **+0.02%** | +1 s |

16k 差异属于噪声，没有改变 4k 回退 3.85%、TTFT 回退 32.09% 的否决结论。因而不再重复 MATH-500：该实验没有进入候选准入，交付代码和报告既有性能数字均不变。分支最终提交为 `85cc2f0`，不是早期中间提交。

- `a29-16k-{on,off}-*.{log,csv}`：完整 driver、serve、benchmark、四轮原始数据及官方汇总。
- `run_a28_16k.sh`：本次启动与测试驱动；官方 benchmark 仍为 4 轮，参数未变，外层超时只用于等待完整运行。
- `a29-16k-timeout240-*`：首次误设 240 秒外层超时而中断的无效试验，单独保留，不计入结果。
- 四轮原始 CSV SHA-256：开启 `d61b9fc885dbba0c8c5ec202e204d620cdd294dfeb6a883e5275358ee74b8da4`，关闭 `f073736f0aeb19068a8d31fdfb28f1c29ba32b39759b9caa1c5c1cf76c502481`。
- 收尾 `mx-smi` 无进程，显存回落至 826 MiB；CPU 内存 120 GiB，未见内存异常。
