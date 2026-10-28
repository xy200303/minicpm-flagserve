# A36：decode 全页循环与尾页分离（2026-10-28）

## 候选与实现

基线为 A20+A31。A20 页对齐内核此前在每次循环内执行 `if t0 + BS <= kvlen`，即使绝大多数迭代都是全页，也保留尾页分支。A36 先计算 `main_end`，全页主循环只执行无 mask 的 K/V 加载和 online softmax，唯一尾页在循环外处理。页号标量化、NSEG、数值顺序、分段归并和 CUDA graph 接口均不变；与 A30 的尾页拆分、A33 的双页展开是不同机制。

实验分支：`xy200303/FlagGems` 的 `exp/a36-fullpage-loop`，提交 `d259d497`。一键回滚：在集成工作树执行 `git apply -R official.patch`，或不使用候选目录而恢复 `GEMS_ROOT=/workspace/a31-gems`。

## 正确性与微基准

`pytest -q tests/test_paged_decode_attention.py`：**9 passed**。每点 30 次 CUDA event 计时并取中位数，B=64：

| 场景 | A20 | A36 | 延迟下降 | 加速比 |
|---|---:|---:|---:|---:|
| 4k 等长 | 0.58784 ms | 0.54413 ms | 7.44% | 1.0803× |
| 4k ragged | 0.44156 ms | 0.39699 ms | 10.10% | 1.1123× |
| 16k 等长 | 2.13931 ms | 1.86885 ms | 12.64% | 1.1447× |
| 16k ragged | 1.99413 ms | 1.74549 ms | 12.48% | 1.1425× |

微基准确认候选实际从 `/workspace/a36-gems-full/src` 加载。首次仅上传单文件导致 import 回落到 site-packages 的无效测量已覆盖，不用于结论。

## 官方验证状态

官方 4k A/B/A/B 使用 `run_bi_case_v5.sh`，保持官网 serve 参数；A 为 `/workspace/a31-gems`，B 为 `/workspace/a36-official-gems`（A31 加本补丁）。运行 ID：`a36-4k-a1`、`a36-4k-b1`、`a36-4k-a2`、`a36-4k-b2`。

A1 已完整归档：**5208.52 tok/s，TTFT 4060.54 ms**。B1 运行至实例关机前仍未生成汇总；00:00 后连接认证失效，无法取回完整结果，A2/B2 与 MATH-500 未执行。B1 的启动配置、源码路径和日志已保存，但不据此计算提升。当前不合入；下次可用实例需重新完成至少两组当日交错 A/B 和精度验证。

## 文件

- `current.json`、`candidate.json`：全部原始计时时序与正确性结果。
- `pytest.log`：算子测试。
- `a36.patch`：相对 A31 的实验分支补丁；`official.patch`：远端官方验证树的实际差异。
- `official-source-state.txt`：远端基线 SHA 与候选文件状态。
- 官方 CSV、serve/bench 日志、机器状态将在各轮结束后归档。
- `docs/verification/summarize_official_ab.py`：四个完整运行目录的中位数汇总；校验官方案例一致，并检查吞吐 ≥3% 与 TTFT ≤对照中位数 +1%。残缺 B1 已验证会被拒绝。
