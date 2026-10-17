# 优化实验台账（tracking）

## 纪律

1. 一次实验 = 一个 exp_id，产物在 `docs/experiments/runs/<exp_id>/`（机器侧 `/workspace/experiments/runs/<exp_id>/`）：
   `config.txt`（分支/commit/env/用例）、`serve.log`、`bench.log`、`summary.csv`、`gpu_before/after.txt`
2. **任何优化必须当天交错 A/B**：同一用例交替跑 A(基线)/B(候选)，取中位数比较。
   实测噪声 <0.1%（同日 A1=1702.33 vs A2=1701.67，代理用例），可直接用单跑比较，但仍建议 A/B/A。
   绝对数字**不可跨天比较**（2026-10-17：同代码 2346.64 vs 3778.65，机器漂移 38–60%）
3. 每个补丁都有 `apply_*.py` + `revert_*.sh`（`docs/experiments/harness/`），回滚后必须 `cp` 到 site-packages
4. 每次实验后在本表追加一行

## 台账（2026-10-17 下半场，天数 BI-V150，代理用例 `[[4096,1024,64,128]]` 与 `[[4096,512,64,64]]`）

| exp_id | 分支@commit | 用例 | 总吞吐 tok/s | 输出 tok/s | TTFT ms | TPOT ms | 结论 |
|---|---|---|---|---|---|---|---|
| A1_base | main@84abed0 | 4k/1024/c64/128 | **1702.33** | 340.47 | 9700 | 138.3 | 当日基线 |
| B1_a13 | main+A13 | 4k/1024/c64/128 | 1677.28 | 335.46 | 9645 | 138.1 | **−1.5%，回退** |
| A2_base | main@84abed0 | 4k/1024/c64/128 | **1701.67** | 340.33 | 9700 | 138.3 | 复现基线，噪声 <0.1% |

### A13（去 metadata 每步 D2H）结论

- 实现：runner 在 `_prepare_inputs` 发布 CPU 侧 cu/seq，builder 不再访问 `query_start_loc_cpu`（D2H 拷贝）
- 一致性：启动时三次对拍 `published cu/seq match vLLM copies` ✅
- 结果：−1.5%（1677.28 vs 1702.33 / 1701.67）→ **判定为小幅回退，已回退**
- 推断：该 D2H 处的等待在整步流水里与其他等待重叠，去掉它没有缩短关键路径；
  或同步本身起了"及时排空"的作用。文档记录为"方向存在但此实现无收益"

### A7 decode pin 在 c64 的复扫（进行中）

`w1s1`（当前）是 c8 扫出来的；c64 的 attention 形状不同，重新扫 `w2s1` / `w4s1` / `w1s2`。
用例 `[[4096,512,64,64]]`（短负载，~2 min/跑），同一 serve 参数。
