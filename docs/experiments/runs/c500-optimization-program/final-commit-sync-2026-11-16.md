# Final commit sync: 2026-11-16

The C500 instance was synchronized from the team's GitHub forks without editing source files manually:

```text
vllm-plugin-FL b5793cd4477a530eb420c58bf2a4e5d742a8846c
FlagGems      ef5161373bc97c2ccabd1f6d51d92fb2a37ff814
```

Both worktrees were detached at the exact final delivery commits and reported clean status after checkout. The instance environment remains PyTorch 2.8.0+metax3.7.0.7, vLLM 0.24.0+empty, Triton 3.0.0+metax3.7.0.7, C500 64GB and 128GiB RAM.

The final-commit C500 service was started with the official serve parameters and with `VLLM_FL_MM_WARMUP=0` to disable the newly added GEMM autotune warmup. The official 4k and 16k benchmark cases were then run separately with the unchanged benchmark program. Both cases completed four rounds with zero failed requests; detailed evidence is in `final-c500-warmup-off-20261116-003450-summary.md` and the accompanying raw logs. MATH-500 Level 3 evaluation is running separately and its result is not yet available. This synchronization is recorded separately from the performance evidence so old C500 measurements are not accidentally presented as final-commit measurements.
