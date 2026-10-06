# opt8: Prefill GEMM 路由到 vendor mcBLAS（ctypes 直绑）

日期：2026-10-06 ｜ 机器：C500 64GB（140.207.205.81，128GB 内存健康实例）

## 动机（16k prefill trace 实锤）

对 16k prefill 阶段抓 torch profiler trace（16 条 ~15k token prompt，chunk=2048，
121 步）：GPU busy 81%，热点极度集中：

| 内核 | 总耗时 | 占比 | 说明 |
|---|---|---|---|
| `mm_kernel_nn`（FlagGems Triton） | 7.64s | 49% | 层内 GEMM，n=19824，avg 385µs |
| vendor flash_fwd_splitkv | 6.55s | 42% | attention（vendor，已近优） |
| 其余（SiLU/RMSNorm/cache） | <1.4s | <9% | |

GEMM 四个形状（M=2048，42 层）：qkv K2048·N2560、o K2048·N2048、
gate_up K2048·N12288、down K6144·N2048。

## 关键发现 1：Triton 配置已到顶，vendor 快 1.35-1.67x

把 `mm_kernel_nn` 的 tune 空间从 8 组扩到 126 组（BLOCK_M/N/K × stages × warps
全扫）后，最优配置仍是原空间里的 (256,256,32,s2,w8)，serve 早已在用。
Triton 最佳 vs vendor aten::mm（M=2048）：

| 形状 | vendor | Triton 最佳 | 差距 |
|---|---|---|---|
| qkv | 131µs (163TF) | 178µs | 1.36x |
| o_proj | 113µs (152TF) | 177µs | 1.56x |
| gate_up | 522µs (197TF) | 707µs | 1.35x |
| down | 294µs (175TF) | 492µs | 1.67x |

vendor mcBLAS 在大 M 上逼近芯片峰值（~197TF），Triton 无法靠调配置追平。

## 关键发现 2：FlagGems 注册后 vendor GEMM 从 Python 不可达

FlagGems 用 `torch.library.Library("aten","IMPL")` 把 aten::mm/linear/addmm/bmm
的 CUDA 实现全替换成了自家 Triton 内核（进程级、无 TLS 旁路）。实测
torch.mm / aten.mm / matmul / F.linear / addmm / bmm 全部落到 Triton。
要在大 M 用上 vendor GEMM，只能绕过 dispatcher 直调 mcBLAS。

## 改动：mcblas_mm.py（ctypes → mcblasGemmEx）

- 直绑 `/opt/maca/lib/libmcblas.so` 的 `mcblasCreate/SetStream/GemmEx`
  （C 符号，cuBLAS 兼容 ABI）。行主序 C=A@B 用列主序恒等式 C^T=B^TA^T 下单。
- 注册为自定义算子 `vllm_fl_metax::mcblas_mm`（带 fake impl），dynamo 把它
  当不透明外部调用，compiled 区域内安全。
- 正确性：与 aten vendor mm **逐位一致**（同库同算法）；与 Triton 路径
  rel_err ~2-4e-3（bf16 正常累加顺序差）。
- 单测（test_mcblas_route.py）：gems 开启下 M=2048 全走路由成功、时延与
  vendor 一致；M=64 仍走原 nt/Triton decode 路径不动。

## 接入（linear_nn_repack.py 一处改动）

`_metax_unquantized_gemm` 的 nn 分支：bf16 且 contiguous 时由
`torch.mm`（被劫持到 Triton）换成 `mcblas_mm`（vendor）。
decode 细瘦 M 路径（nt_db/nn_db）不变；`VLLM_FL_DISABLE_MCBLAS_MM=1` 可 A/B。

预期收益：每层每步省 (178-131)+(177-113)+(707-522)+(492-294) ≈ 494µs，
×42 层 ≈ 20.7ms / 129ms prefill 步 ≈ prefill 时间 -16%。

## 实测结果（官方 benchmark，4 跑弃首取均值）

| 场景 | opt7 | opt8 | Δ | 相对官网基线 |
|---|---|---|---|---|
| 4k `[4096,1024,64,256]` | 8273.58 | **10946.86** | **+32.3%** | **+115.1%** |
| 16k `[16384,1024,64,128]` | 8716.35 | **10001.65** | **+14.7%** | **+42.3%** |

- 4k runs：10780.66（弃）/ 10922.51 / 10953.97 / 10964.10
- 16k runs：10010.42（弃）/ 10008.58 / 9996.60 / 9999.77
- TTFT 同步改善：4k 2795→2174ms，16k 23896→20814ms（prefill 变快的直接体现）
- CSV 归档：`benchmark_results/summary_20261006_212430.csv`
- 4k 提升比 16k 更大：4k 场景 prefill token 占输入 4:1，GEMM 加速的
  权重更高；16k 里 attention（vendor，42%）摊薄了 GEMM 收益。
- 精度：MATH-500 Level 3 两次复测 **97.1% / 98.1%**（105 题，temp 1.0
  top_p 0.95 采样有 ±1.4%/σ 的自然波动；第二次与基线完全一致），
  远高于 0.95 及格线。GEMM 换 vendor 后与 Triton 路径 rel_err ~2e-3
  （fp32 累加的顺序差），对精度无可测影响。
