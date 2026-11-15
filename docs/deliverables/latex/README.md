# 报告源码与编译方法

`main.tex` 使用 XeLaTeX 编译，`minted` 宏包需要 `-shell-escape`：

```bash
latexmk -xelatex -shell-escape -outdir=build main.tex
```

依赖与路径：

- 代码清单（`\PluginCode`、`\GemsCode`）从 `../pkg_build/XLANG2026/vllm-plugin-FL` 与
  `../pkg_build/XLANG2026/FlagGems` 读取。把提交包解压出的两个源码目录放到该位置即可，
  版本为插件 `b5793cd4`、FlagGems `ef516137`。
- 图片从 `../submission/` 读取，仓库内已包含报告中使用的全部 PDF 图。
- 图 21–26、28 由 `make_figs_v3.py` 生成，数据来自 `../experiments/runs/` 下的
  benchmark 日志与 CSV；`fig8_accuracy` 由 `make_figs_nature.py` 生成。

报告正文中的官网基线数字取自赛题页面“4.3 评审方案”的两张基线截图，原图保存在
`../submission/official_baseline_biv150.png` 与 `official_baseline_c500.png`（本目录亦保留副本）。
