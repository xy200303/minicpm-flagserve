# 报告重建记录（2026-11-16）

- 源文件：`tmp/latex/main.tex`
- 生成命令：`latexmk -g -xelatex -shell-escape -interaction=nonstopmode -halt-on-error -outdir=build main.tex`
- 输出：`tmp/latex/main.pdf`
- 编译结果：通过；最终 `build/main.log` 无 `Underfull`/`Overfull`/`Error`。
- 本轮同步的 C500 最终提交数据：4k `12304.20 tok/s`，16k `11082.68 tok/s`，平均 `11693.44 tok/s`；平均 TTFT 分别为 `1986.92 ms`、`18056.74 ms`；MATH-500 Level 3 `98.1%`（105/105，0 错误）。
- BI-V150 数据在报告中明确标注为父提交历史测量；最终提交复测仍未验证，不以历史值替代。
- 图 `fig21_headline_speedup.pdf`、`fig22_throughput_both.pdf`、`fig23_ttft_ratio.pdf` 已按上述 C500 数据重生成，并在图注中注明 BI 历史测量性质。
- 交付 ZIP 尚未重建，等待 BI-V150 最终提交复测及最终审计。
