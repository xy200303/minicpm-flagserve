# opt11: lm_head 路由修复（vpe 绑定）+ top-p 直方图单轮化

日期：2026-10-07 ｜ 机器：C500 64GB（140.207.205.81）

## 背景

opt10 后重抓 decode trace（64 并发 × 4k ctx × 1024 步）：GPU 空闲已从
14.6% 降到 **8.1%**（gap>20µs 仅 4.0%），但发现两个残留问题：

1. `linear_kernel`（FlagGems Triton）每步 603µs × 1065 步——是 lm_head，
   **opt7 的 nn 重排 + opt8 的 mcBLAS 路由对它从未生效**；
2. `_hist_zoom`（opt7b top-p 直方图）每步 2×311µs，atomic 主导。

## 修复 1：lm_head 从未走优化路径（import-time 绑定陷阱）

`vocab_parallel_embedding.py` 在模块 import 时就
`from ...layers.utils import dispatch_unquantized_gemm`——本地名字绑死了
**原始**工厂函数；我们 patch `layer_utils`/`linear_mod` 的模块属性对它无效。
普通 linear 层走 `linear.py`（调用时才查模块属性）所以正常，
唯独 lm_head（ParallelLMHead→UnquantizedEmbeddingMethod.apply）一直裸奔在
FlagGems Triton linear_kernel 上。

修复：补丁里同时替换 `vocab_parallel_embedding.dispatch_unquantized_gemm`。
效果：lm_head `[64,2048]×[2048,130560]` 从 603µs（Triton）→ **379µs**
（vendor mcBLAS nn），decode 每步省 224µs。

教训：**patch 模块属性前必须确认目标模块是"调用时查找"还是"导入时绑定"**，
from-import 的模块名要逐个 patch。

## 修复 2：top-p 直方图 2 轮 → 1 轮（1024 bins）

原设计 2×256 轮缩放（阈值精度 ~1e-3 logit）。变体扫描
（test_zoom_variants.py，[64,130560] bf16，top_p=0.95，2 万次采样）：

| 变体 | 耗时 | TV 距离 | 越界样本 |
|---|---|---|---|
| 2×256（原） | 813µs | 0.2938 | 0 |
| **1×1024** | **310µs** | 0.2884 | 0 |
| 1×2048 | 344µs | 0.2867 | 0 |
| 1×4096 | 408µs | 0.3054 | 0 |

单轮 1024 bins 统计上与双轮无差异（TV 差 < 采样噪声），整条采样链
282µs vs eager 1948µs（**6.9x**）。tau 恒取穿越 bin 下沿，保证
kept mass ≥ p·Z，永不欠覆盖。

（测试插曲：变体测试第一版 TV=1.0 全挂，查了半天是**测试脚本自己**的
bug——sort 后的概率没 scatter 回原始词表顺序；采样器一直是对的。）

## 实测（官方 benchmark，4 跑弃首取均值）

| 场景 | opt10 | opt11 | Δ | 相对官网基线 |
|---|---|---|---|---|
| 4k `[4096,1024,64,256]` | 12145.31 | **12529.39** | **+3.2%** | **+146.2%** |
| 16k `[16384,1024,64,128]` | 10991.50 | **11088.13** | **+0.9%** | **+57.7%** |

- 4k runs：12444.34（弃）/ 12535.37 / 12540.24 / 12512.56
- 16k runs：11092.98（弃）/ 11090.54 / 11087.32 / 11086.54
- TTFT：4k 1618ms、16k 17992ms，均优于 opt10
- CSV：`benchmark_results/summary_20261007_162801.csv`（注：raw 文件名
  raw_runs_20261007_162801.csv）
- 精度：MATH-500 Level 3 = **97.1%**（与历史各版 97.1-98.1% 的采样波动
  区间一致，远高于 0.95 及格线）。
