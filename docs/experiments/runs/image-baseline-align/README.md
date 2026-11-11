# 官方镜像基线对齐与最终复测（2026-11-11）

本次不是新的优化实验，而是把已有的 A20+A31+A48+A60 交付栈从公开 `v5.3.5` 提交 `a7620cc1` 重放到天数官方镜像实际使用的 `4df52d91` 上，解决基线版本不一致，并重新完成性能与精度验证。

## 环境与代码

| 项目 | 最终值 |
|---|---|
| 实例 | `8f8a906a307e`，官方重置后的 BI-V150 32GB 实例 |
| FlagGems 镜像基线 | `4df52d9168ac55c1c1061cefb4638c570292d896`，镜像内 `v5.3.5` 指向该提交 |
| FlagGems 交付分支 | `integration/flaggems`，`ef5161373bc97c2ccabd1f6d51d92fb2a37ff814`；性能/精度测量于父提交 `85b4ba82`，最终提交只规范化 CRLF |
| vllm-plugin-FL | `integration/vllm-plugin-fl`，`b5793cd4477a530eb420c58bf2a4e5d742a8846c`；性能/精度测量于父提交 `58bae94`，最终提交只规范化 CRLF |
| FlagTree | `0.6.2a3+iluvatar3.6` |
| Torch / vLLM | `2.10.0+corex.4.5.0` / `0.24.0+empty` |
| CPU / 内存 | Xeon Gold 6330，503GiB |
| GPU | Iluvatar BI-V150，32768MiB |

镜像提交比公开 tag 少 7 个提交。重放时唯一冲突是镜像尚无 `_iluvatar/fused/matmul_bias_activation.py`，按镜像基线解决，只保留本工作的 `mm_silu_mul` 注册。19 个线性提交落在 `4df52d91` 上，完整历史保留在 `integration/flaggems`。旧公开 tag 集成历史归档为 tag `archive/flaggems-public-tag`（`e9390c54`）。

## 正确性

在镜像实例上直接从源码导入：

```bash
export PYTHONPATH=/workspace/vllm-plugin-image-align:/workspace/flag-image-align/src
export VLLM_PLUGINS=fl
cd /workspace/flag-image-align
python -m pytest -q tests/test_iluvatar_mm_silu_mul.py tests/test_paged_decode_attention.py
cd /workspace/vllm-plugin-image-align
python -m pytest -q tests/unit_tests/models/test_mm_silu_mul.py
```

结果：FlagGems **12 passed**，插件路由 **3 passed**。完整日志见 `tests.tar.gz`。

## 官方性能复测

启动参数与官网完全一致：

```bash
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072
```

benchmark 脚本执行官方命令，每个场景四轮、自动去首轮。4k 共执行三次独立四轮测量，避免把设备冷态误判为性能：

| 4k 独立测量 | 三轮均值（tok/s） | 平均 TTFT（ms） | 说明 |
|---|---:|---:|---|
| 第 1 次 | 5131.78 | 3956.52 | 服务启动后的首个场景，首轮及后续仍受冷态影响 |
| 第 2 次 | 5292.46 | 3908.25 | 稳定态 |
| 第 3 次 | 5292.99 | 3888.19 | 稳定态 |
| 中位数 | **5292.46** | **3908.25** | 报告采用该值 |

16k 执行一次完整四轮测量，三轮均值 **4618.41 tok/s**、平均 TTFT **118761.15 ms**。16k 结果与旧实例的 4620.25／118779.35ms 基本一致，说明基线回放没有改变长上下文性能。

最终报告值采用 4k 三次测量中位数和 16k 四轮汇总：

| 场景 | 官网基线 | 本提交 | 相对官网 |
|---|---:|---:|---:|
| 4k | 2028.01 | **5292.46** | **2.61×** |
| 16k | 915.15 | **4618.41** | **5.05×** |

4k 最低一次为 5131.78，仍为官网基线的 2.53 倍；报告同时保留该值，不隐藏冷态波动。原始逐轮 CSV 与 summary 见 `bench-evidence.tar.gz` 中的 `benchmark_results/`。

## 精度

官方 evalscope 命令，MATH-500 Level 3：**0.981（98.1%）**，105/105 请求成功、0 错误，满足 ≥0.95。报告位于 `eval-evidence.tar.gz` 内的 `20261111_132906/reports/minicpm/math_500.json`。

## 证据与回滚

- 性能证据：`bench-evidence.tar.gz`，SHA256 `61c783a2fd810df7a7f3d6b229569b5ea9c5737d35e35dfa2d6ac59ff029d6c4`
- 精度证据：`eval-evidence.tar.gz`，SHA256 `3379990ad4b977f0f04d48dadb2d7c251059f2997c12d4b7045e1cfaad821704`
- 测试与最终包审计：`tests.tar.gz`，SHA256 `2ee44352312cf4c7e340898ab35a89d4e11cb5536e3e469b0ab020e034a47988`；包含最终包测试日志、Git 干净性、包 SHA256、真实请求响应和 smoke 服务日志
- 回滚代码：将 `integration/flaggems` 恢复到旧标签 `archive/flaggems-public-tag`（`e9390c54`）即可；本次没有修改插件。
- 回滚性能环境：恢复对应 `VLLM_CACHE_ROOT` 或重新编译；本次使用 `/workspace/image-align-cache`。
