# A59：事件同步与异步调度分析（2026-11-06）

基于已下载到本地的 `a58-dual-acc-warps16/ixsys-summary.json` 和当前 `model_runner.py` 代码，分析设备 API 中最大的同步项。

## 观测

- `cudaEventSynchronize`：2554 次，累计 **26.079s**，平均约 **10.21ms**。当前诊断覆盖约 1276 个 decode step，基本对应每步两次同步。
- `cudaStreamSynchronize`：524 次，累计约 **0.263s**。
- `cuLaunchKernel`：197584 次，累计约 **1.081s**。
- `cudaGetDevice`、`cuPointerGetAttribute` 分别累计约 **0.313s/0.279s**，不值得单独优化。

## 代码归因

`AsyncGPUModelRunnerOutput.get_output()` 中的 `async_copy_ready_event.synchronize()` 用于等待异步 D2H 输出完成；`synchronize_input_prep()` 中的 `prepare_inputs_event.synchronize()` 用于防止下一步复用尚未完成传输的 CPU 输入 buffer。两者都是异步调度正确性所需的同步，不是无意义的 host 自旋。

删除或合并这些事件会造成输出未就绪就读取、输入 tensor 被覆盖等数据竞争，因此不形成安全优化候选。下一步若要降低等待，只能缩短事件之前的 GPU/拷贝工作，而不是删除同步本身。本项不修改代码、不执行 benchmark。

原始 ixsys 大文件仍保留在服务器 `/workspace/a58-current-ixsys.tar.gz`，SHA256 为 `3dd52c433c437bc0edd06e1d7b00e6aac3465f171d88c2cf11c6c79c95a7f193`。连接超时导致完整原始包尚未下载，本地已有内核汇总 JSON；连接恢复后需补齐并校验。
