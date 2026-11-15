"""Report figures added in the 2026-11-13 polish pass (fig21–fig29).

Every number here is read from the raw benchmark CSVs / summary JSONs under
main-checkout/docs/experiments/runs/, or is a value already stated in the
report text. Nothing is estimated or interpolated.

Style follows make_figs_nature.py (Times New Roman + SimHei, same palette).
Outputs: submission/<name>.pdf|.svg and a copy of the PDF in latex/.
"""
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor" / "nature-figure" / "scripts"))

import matplotlib as mpl  # noqa: E402

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator, NullLocator  # noqa: E402

import nature_api as na  # noqa: E402

na.apply_publication_style(font_size=7.5, axes_linewidth=0.6)
mpl.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.family": ["Times New Roman", "SimHei"],
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.handlelength": 1.0,
    "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.2,
    "legend.labelspacing": 0.3,
    "axes.unicode_minus": False,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
})

RUNS = HERE / "main-checkout" / "docs" / "experiments" / "runs"
OUT = HERE / "submission"
LATEX = HERE / "latex"
CM = 1 / 2.54

P = na.PALETTE
HERO = P["blue_main"]
HERO_SOFT = P["blue_secondary"]
ACCENT = P["red_strong"]
VIOLET = P["violet"]
TEAL = P["teal"]
BASE = P["neutral_light"]
BASE_EDGE = "#9A9A9A"
INK = P["neutral_black"]
MID = P["neutral_mid"]
GRID = "#E6E6E6"
DASH = (0, (4, 3))


def save(fig, stem):
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.03)
    shutil.copyfile(OUT / f"{stem}.pdf", LATEX / f"{stem}.pdf")
    plt.close(fig)
    print("saved", stem)


def panel_title(ax, text):
    ax.set_title(text, loc="left", fontsize=8, pad=6)


def raw_rounds(path):
    """Per-round rows of an official raw_runs CSV, grouped by prefill length."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.setdefault(int(float(row["Prefill"])), []).append(
                {k: float(v) for k, v in row.items()
                 if k not in ("Run Status",)})
    return out


def log_rounds(path):
    """Per-round metrics parsed from the result blocks of a benchmark log."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    alias = {
        "Total token throughput (tok/s)": "Total tok/s",
        "Output token throughput (tok/s)": "Output tok/s",
    }
    rows = []
    for block in re.findall(r"Serving Benchmark Result =+\n(.*?)\n=+\n", text, re.S):
        row = {}
        for line in block.splitlines():
            m = re.match(r"([A-Za-z][^:]*?):\s+([0-9.]+)", line)
            if m:
                row[alias.get(m.group(1).strip(), m.group(1).strip())] = float(m.group(2))
        if row:
            rows.append(row)
    return rows


def tail_mean(rows, key):
    """Official aggregation: drop the first round, average the rest."""
    return float(np.mean([r[key] for r in rows[1:]]))


# --------------------------------------------------------------------------- #
# Source data
# --------------------------------------------------------------------------- #
OFFICIAL = {"C500": {"4k": 5089.645, "16k": 7029.675},
            "BI": {"4k": 2028.01, "16k": 915.15}}
NATIVE = {"C500": {"4k": 4210.20, "16k": 5867.59},
          "BI": {"4k": 1902.15, "16k": 861.98}}
OFFICIAL_TTFT = {"C500": {"4k": 3199.435, "16k": 27197.135},
                 "BI": {"4k": 11573.53, "16k": 599262.03}}

C500_4K = log_rounds(HERE.parent / "docs" / "experiments" / "runs"
                     / "c500-optimization-program"
                     / "final-c500-warmup-off-20261116-003450-benchmark-partial.log")
C500_16K = log_rounds(HERE.parent / "docs" / "experiments" / "runs"
                      / "c500-optimization-program"
                      / "final-c500-warmup-off-20261116-003450-benchmark-16k.log")
_bi1 = raw_rounds(RUNS / "image-baseline-align/bench/benchmark_results/"
                  "raw_runs_20261111_123240.csv")
BI_4K_RUNS = [
    _bi1[4096],
    raw_rounds(RUNS / "image-baseline-align/bench/benchmark_results/"
               "raw_runs_20261111_135130.csv")[4096],
    raw_rounds(RUNS / "image-baseline-align/bench/benchmark_results/"
               "raw_runs_20261111_141134.csv")[4096],
]
BI_16K = _bi1[16384]

FINAL = {
    "C500": {"4k": tail_mean(C500_4K, "Total tok/s"),
             "16k": tail_mean(C500_16K, "Total tok/s")},
    "BI": {"4k": float(np.median([tail_mean(r, "Total tok/s") for r in BI_4K_RUNS])),
           "16k": tail_mean(BI_16K, "Total tok/s")},
}
BI_4K_MEANS = [tail_mean(r, "Total tok/s") for r in BI_4K_RUNS]
FINAL_TTFT = {
    "C500": {"4k": tail_mean(C500_4K, "Mean TTFT (ms)"),
             "16k": tail_mean(C500_16K, "Mean TTFT (ms)")},
    "BI": {"4k": tail_mean(BI_4K_RUNS[1], "Mean TTFT (ms)"),
           "16k": tail_mean(BI_16K, "Mean TTFT (ms)")},
}


def report_numbers():
    """Print every derived number used in the LaTeX tables."""
    print("== throughput (tok/s), ratio to official, ratio to native")
    for plat in ("C500", "BI"):
        for sc in ("4k", "16k"):
            f, o, n = FINAL[plat][sc], OFFICIAL[plat][sc], NATIVE[plat][sc]
            print(f"{plat} {sc}: final={f:.2f} x_official={f / o:.3f} x_native={f / n:.3f}")
        fa = np.mean(list(FINAL[plat].values()))
        oa = np.mean(list(OFFICIAL[plat].values()))
        na_ = np.mean(list(NATIVE[plat].values()))
        print(f"{plat} avg: final={fa:.2f} official={oa:.2f} native={na_:.2f} "
              f"x_official={fa / oa:.3f} x_native={fa / na_:.3f}")
    print("BI 4k independent means:", [f"{v:.2f}" for v in BI_4K_MEANS])
    print("== TTFT (ms): official, limit(+1%), measured, delta")
    for plat in ("C500", "BI"):
        for sc in ("4k", "16k"):
            o, m = OFFICIAL_TTFT[plat][sc], FINAL_TTFT[plat][sc]
            print(f"{plat} {sc}: official={o:.2f} limit={o * 1.01:.2f} "
                  f"measured={m:.2f} delta={(m / o - 1) * 100:+.1f}%")
    print("== latency profile (rounds 2-4 mean)")
    keys = ["Output tok/s", "Mean TTFT (ms)", "Median TTFT (ms)", "P99 TTFT (ms)",
            "Mean TPOT (ms)", "P99 TPOT (ms)", "Median ITL (ms)", "P99 ITL (ms)"]
    for name, rows in (("C500 4k", C500_4K), ("C500 16k", C500_16K),
                       ("BI 4k (run 2)", BI_4K_RUNS[1]), ("BI 16k", BI_16K)):
        print(name, " | ".join(f"{k}={tail_mean(rows, k):.2f}" for k in keys))
    print("== round spread (rounds 2-4, (max-min)/mean %)")
    for name, rows in (("C500 4k", C500_4K), ("C500 16k", C500_16K),
                       ("BI 4k r1", BI_4K_RUNS[0]), ("BI 4k r2", BI_4K_RUNS[1]),
                       ("BI 4k r3", BI_4K_RUNS[2]), ("BI 16k", BI_16K)):
        v = np.array([r["Total tok/s"] for r in rows[1:]])
        print(f"{name}: rounds={[r['Total tok/s'] for r in rows]} "
              f"spread={(v.max() - v.min()) / v.mean() * 100:.2f}%")


# --------------------------------------------------------------------------- #
# 21. Headline: speedup over the official baseline
# --------------------------------------------------------------------------- #
def fig_headline():
    groups = ["4k 场景", "16k 场景", "两场景平均\n（官方指标）"]
    ratios = {}
    for plat in ("C500", "BI"):
        f, o = FINAL[plat], OFFICIAL[plat]
        ratios[plat] = [f["4k"] / o["4k"], f["16k"] / o["16k"],
                        np.mean(list(f.values())) / np.mean(list(o.values()))]
    fig, ax = plt.subplots(figsize=(15.4 * CM, 5.6 * CM))
    x = np.arange(3)
    w = 0.32
    for off, plat, color, label in ((-w / 2, "C500", HERO, "沐曦 C500"),
                                    (w / 2, "BI", VIOLET, "天数 BI-V150")):
        bars = ax.bar(x + off, ratios[plat], w * 0.92, color=color, label=label,
                      edgecolor="white", linewidth=0.4, zorder=3)
        for bar, v in zip(bars, ratios[plat]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.08, f"{v:.2f}×",
                    ha="center", va="bottom", fontsize=8, color=color,
                    fontweight="bold")
    ax.axhline(1.0, color=ACCENT, linewidth=0.7, linestyle=DASH, zorder=2)
    ax.text(-0.83, 1.06, "官网基线 = 1×", color=ACCENT, fontsize=6.8,
            ha="left", va="bottom")
    ax.axvspan(1.5, 2.5, color="#F3F5F9", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=7.5)
    ax.set_xlim(-0.85, 2.5)
    ax.set_ylim(0, 5.8)
    ax.set_ylabel("总吞吐相对官网基线（×）")
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.grid(axis="y", color=GRID, linewidth=0.45, zorder=0)
    ax.legend(loc="upper left", frameon=False, ncol=2)
    save(fig, "fig21_headline_speedup")


# --------------------------------------------------------------------------- #
# 22. Three-way throughput, both platforms in one figure
# --------------------------------------------------------------------------- #
def fig_throughput_both():
    fig, axes = plt.subplots(1, 2, figsize=(15.4 * CM, 6.2 * CM))
    labels = ["4k 场景", "16k 场景"]
    x = np.arange(2)
    w = 0.25
    for ax, plat, color, title, ymax, dy in (
            (axes[0], "C500", HERO, "a  沐曦 C500（实测于 2026-11-16）", 17800, 230),
            (axes[1], "BI", VIOLET, "b  天数 BI-V150（实测于 2026-11-11）", 7500, 95)):
        series = [
            ("官网基线", [OFFICIAL[plat][s] for s in ("4k", "16k")], BASE, BASE_EDGE, MID),
            ("原生栈复测", [NATIVE[plat][s] for s in ("4k", "16k")], "#9DB4D3"
             if plat == "C500" else "#CFA8C8", "white", MID),
            ("优化版本实测", [FINAL[plat][s] for s in ("4k", "16k")], color, "white", color),
        ]
        for i, (label, vals, fc, ec, tc) in enumerate(series):
            bars = ax.bar(x + (i - 1) * w, vals, w * 0.92, color=fc, edgecolor=ec,
                          linewidth=0.4, label=label, zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + dy, f"{v:,.0f}",
                        ha="center", va="bottom", fontsize=6.6, color=tc)
        for xi, sc in zip(x, ("4k", "16k")):
            ratio = FINAL[plat][sc] / OFFICIAL[plat][sc]
            ax.text(xi + w, FINAL[plat][sc] + dy * 4.3, f"{ratio:.2f}×",
                    ha="center", va="bottom", fontsize=8.5, color=color,
                    fontweight="bold")
        if plat == "BI":
            lo, hi = min(BI_4K_MEANS), max(BI_4K_MEANS)
            ax.errorbar(x[0] + w, FINAL["BI"]["4k"],
                        yerr=[[FINAL["BI"]["4k"] - lo], [hi - FINAL["BI"]["4k"]]],
                        fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.5,
                        capthick=0.8, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("总吞吐（tok/s）")
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis="y", color=GRID, linewidth=0.45, zorder=0)
        ax.legend(loc="upper left", frameon=False, ncol=3, columnspacing=0.8,
                  fontsize=6.5, bbox_to_anchor=(0.0, 1.0))
        panel_title(ax, title)
    fig.subplots_adjust(wspace=0.30)
    save(fig, "fig22_throughput_both")


# --------------------------------------------------------------------------- #
# 23. TTFT relative to the official baseline (half-width panel)
# --------------------------------------------------------------------------- #
def fig_ttft_ratio():
    rows = [("C500 4k", "C500", "4k", HERO), ("C500 16k", "C500", "16k", HERO),
            ("BI-V150 4k", "BI", "4k", VIOLET), ("BI-V150 16k", "BI", "16k", VIOLET)]
    fig, ax = plt.subplots(figsize=(7.4 * CM, 6.0 * CM))
    y = np.arange(len(rows))[::-1]
    for yi, (label, plat, sc, color) in zip(y, rows):
        o, m = OFFICIAL_TTFT[plat][sc], FINAL_TTFT[plat][sc]
        pct = m / o * 100
        ax.barh(yi, pct, height=0.42, color=color, zorder=3)
        ax.text(pct + 1.6, yi, f"{pct:.1f}%（{m / 1000:.2f} s）", va="center",
                ha="left", fontsize=6.8, color=INK)
    ax.axvline(101, color=ACCENT, linewidth=0.7, linestyle=DASH, zorder=4)
    ax.text(99.5, 3.62, "上限 101%", color=ACCENT, fontsize=6.5, ha="right",
            va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 108)
    ax.set_ylim(-0.6, 3.9)
    ax.set_xlabel("平均 TTFT 占官网基线的比例（%，越低越好）", fontsize=7.5)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    save(fig, "fig23_ttft_ratio")


# --------------------------------------------------------------------------- #
# 24. BI-V150 same-day interleaved A/B pairs, stage by stage
# --------------------------------------------------------------------------- #
def fig_bi_stage_ab():
    # (label, baseline tok/s, candidate tok/s, adopted)
    stages_4k = [
        ("A7 发射配置", 2738.65, 3778.65, True),
        ("A20（第 1 组）", 3855.31, 4087.91, True),
        ("A20（第 2 组）", 3840.80, 4722.22, True),
        ("A31 配置缓存", 4563.26, 5209.06, True),
        ("A36 尾页分离", 5139.415, 4499.19, False),
        ("A48 门控融合", 5201.975, 5237.315, True),
        ("A60 权重预打包", 5238.28, 5314.145, True),
    ]
    stages_16k = [
        ("A7 发射配置", 2558.37, 3567.33, True),
        ("A20（第 1 组）", 3568.31, 4520.19, True),
        ("A20（第 2 组）", 3570.61, 4520.10, True),
        ("A31 配置缓存", 4513.93, 4513.91, True),
        ("A48 门控融合", 4524.545, 4556.53, True),
        ("A60 权重预打包", 4551.39, 4620.245, True),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15.4 * CM, 7.6 * CM))
    for ax, stages, sc, xlim in ((axes[0], stages_4k, "4k", (2500, 6750)),
                                 (axes[1], stages_16k, "16k", (2350, 5900))):
        y = np.arange(len(stages))[::-1]
        for yi, (label, a, b, ok) in zip(y, stages):
            color = HERO if ok else ACCENT
            ax.plot([a, b], [yi, yi], color=color, linewidth=1.3, zorder=2,
                    solid_capstyle="round")
            ax.scatter([a], [yi], s=26, facecolor="white", edgecolor=MID,
                       linewidths=0.9, zorder=3)
            ax.scatter([b], [yi], s=30, color=color, zorder=4,
                       marker="o" if ok else "X")
            delta = (b / a - 1) * 100
            txt = "±0.0%" if abs(delta) < 0.05 else f"{delta:+.1f}%"
            ax.text(xlim[1], yi, txt, va="center", ha="right",
                    fontsize=7, color=color, fontweight="bold")
        ax.axvline(FINAL["BI"][sc], color=VIOLET, linewidth=0.7, linestyle=DASH,
                   zorder=1)
        ax.text(FINAL["BI"][sc], len(stages) - 0.42,
                f"镜像对齐交付 {FINAL['BI'][sc]:,.0f}", color=VIOLET, fontsize=6.5,
                ha="right", va="bottom")
        ax.set_yticks(y)
        ax.set_yticklabels([s[0] for s in stages], fontsize=7)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.6, len(stages) - 0.1)
        ax.set_xlabel(f"{sc} 总吞吐（tok/s）")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis="x", color=GRID, linewidth=0.45, zorder=0)
        panel_title(ax, f"{'a' if sc == '4k' else 'b'}  {sc} 场景")
    h = [plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
                    markeredgecolor=MID, markersize=4.5, label="当日对照（基线）"),
         plt.Line2D([], [], marker="o", linestyle="", color=HERO, markersize=4.5,
                    label="候选（已采用）"),
         plt.Line2D([], [], marker="X", linestyle="", color=ACCENT, markersize=5,
                    label="候选（未采用）")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.045), columnspacing=1.6)
    fig.subplots_adjust(wspace=0.52, bottom=0.2)
    save(fig, "fig24_bi_stage_ab")


# --------------------------------------------------------------------------- #
# 25. C500 leave-one-out ablation (4k)
# --------------------------------------------------------------------------- #
def fig_c500_ablation():
    variants = [
        ("Full（11-01 组合）", 12527.62, 1623.73),
        ("w/o lm_head nn", 12458.35, 1635.81),
        ("w/o 采样融合", 11785.47, 1644.35),
        ("w/o prefill mcBLAS", 9371.69, 2072.08),
        ("w/o mcBLAS 与采样融合", 8946.83, 2093.73),
    ]
    full_tp, full_ttft = variants[0][1], variants[0][2]
    fig, axes = plt.subplots(1, 2, figsize=(15.4 * CM, 5.8 * CM),
                             gridspec_kw={"width_ratios": [1.45, 1.0]},
                             sharey=True)
    y = np.arange(len(variants))[::-1]
    ax = axes[0]
    for yi, (label, tp, _) in zip(y, variants):
        full = label.startswith("Full")
        ax.barh(yi, tp, height=0.5, color=HERO if full else "#9DB4D3", zorder=3)
        txt = f"{tp:,.0f}" if full else f"{tp:,.0f}（{(tp / full_tp - 1) * 100:+.1f}%）"
        ax.text(tp + 150, yi, txt, va="center", ha="left", fontsize=7,
                color=HERO if full else INK, fontweight="bold" if full else "normal")
    ax.axvline(OFFICIAL["C500"]["4k"], color=ACCENT, linewidth=0.7, linestyle=DASH,
               zorder=4)
    ax.text(OFFICIAL["C500"]["4k"] + 120, -0.52, "官网基线", color=ACCENT,
            fontsize=6.5, ha="left", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([v[0] for v in variants], fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 16200)
    ax.set_ylim(-0.75, len(variants) - 0.4)
    ax.set_xlabel("4k 总吞吐（tok/s）")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    panel_title(ax, "a  吞吐")
    ax = axes[1]
    for yi, (label, _, ttft) in zip(y, variants):
        full = label.startswith("Full")
        ax.barh(yi, ttft, height=0.5, color=HERO if full else "#9DB4D3", zorder=3)
        txt = f"{ttft:,.0f}" if full else f"{ttft:,.0f}（{(ttft / full_ttft - 1) * 100:+.1f}%）"
        ax.text(ttft + 40, yi, txt, va="center", ha="left", fontsize=7, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 3200)
    ax.set_xlabel("4k 平均 TTFT（ms，越低越好）")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    panel_title(ax, "b  首 token 延迟")
    fig.subplots_adjust(wspace=0.06)
    save(fig, "fig25_c500_ablation")


# --------------------------------------------------------------------------- #
# 26. Round-level stability of the delivered measurements
# --------------------------------------------------------------------------- #
def fig_round_stability():
    fig, axes = plt.subplots(1, 2, figsize=(15.4 * CM, 6.0 * CM), sharey=True)
    rounds = np.arange(1, 5)

    def draw(ax, rows, ref, color, marker, label, ls="-"):
        v = np.array([r["Total tok/s"] for r in rows]) / ref * 100
        ax.plot(rounds, v, color=color, linewidth=0.9, linestyle=ls, zorder=2)
        ax.scatter(rounds[:1], v[:1], s=24, facecolor="white", edgecolor=color,
                   marker=marker, linewidths=0.9, zorder=3)
        ax.scatter(rounds[1:], v[1:], s=24, color=color, marker=marker, zorder=3)
        ax.plot([], [], color=color, linewidth=0.9, linestyle=ls, marker=marker,
                markersize=4, label=label)

    ax = axes[0]
    draw(ax, C500_4K, FINAL["C500"]["4k"], HERO, "o", "4k")
    draw(ax, C500_16K, FINAL["C500"]["16k"], TEAL, "s", "16k")
    panel_title(ax, "a  沐曦 C500")
    ax = axes[1]
    for i, (rows, ls) in enumerate(zip(BI_4K_RUNS, ("-", "--", ":"))):
        draw(ax, rows, FINAL["BI"]["4k"], VIOLET, "o", f"4k 第 {i + 1} 次独立测量", ls)
    draw(ax, BI_16K, FINAL["BI"]["16k"], TEAL, "s", "16k")
    panel_title(ax, "b  天数 BI-V150")
    for ax in axes:
        ax.axhline(100, color=MID, linewidth=0.6, linestyle=DASH, zorder=1)
        ax.axvspan(0.6, 1.5, color="#F3F5F9", zorder=0)
        ax.text(1.05, 103.4, "首轮（剔除）", fontsize=6.5, color=MID,
                ha="center", va="bottom")
        ax.set_xticks(rounds)
        ax.set_xticklabels([f"第 {i} 轮" for i in rounds])
        ax.set_xlim(0.6, 4.4)
        ax.set_ylim(80, 106)
        ax.grid(axis="y", color=GRID, linewidth=0.45, zorder=0)
        ax.legend(loc="lower right", frameon=False, fontsize=6.5,
                  handlelength=2.2)
    axes[0].set_ylabel("单轮总吞吐 / 报告值（%）")
    fig.subplots_adjust(wspace=0.06)
    save(fig, "fig26_round_stability")


# --------------------------------------------------------------------------- #
# 27. BI-A31: host-side configuration overhead
# --------------------------------------------------------------------------- #
def fig_a31_host():
    fig, axes = plt.subplots(1, 2, figsize=(15.4 * CM, 5.4 * CM),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    items = [("真实配置库\n单表加载", 4212.64 / 1000, 26.35 / 1000),
             ("_index_linearized\n累计时间", 66.03, 4.47)]
    x = np.arange(len(items))
    w = 0.3
    before = [i[1] for i in items]
    after = [i[2] for i in items]
    ax.bar(x - w / 2, before, w * 0.92, color=BASE, edgecolor=BASE_EDGE,
           linewidth=0.4, label="BI-A20（原实现）", zorder=3)
    ax.bar(x + w / 2, after, w * 0.92, color=HERO, label="BI-A31", zorder=3)
    for xi, b, a in zip(x, before, after):
        ax.text(xi - w / 2, b * 1.18, f"{b:.2f} s" if b >= 1 else f"{b * 1000:.0f} ms",
                ha="center", va="bottom", fontsize=6.8, color=MID)
        ax.text(xi + w / 2, a * 1.18, f"{a:.2f} s" if a >= 1 else f"{a * 1000:.1f} ms",
                ha="center", va="bottom", fontsize=6.8, color=HERO)
        ax.text(xi, max(b, a) * 3.6, f"÷{b / a:.0f}" if b / a >= 20 else f"÷{b / a:.1f}",
                ha="center", va="bottom", fontsize=8.5, color=ACCENT,
                fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(0.008, 40000)
    ax.set_xticks(x)
    ax.set_xticklabels([i[0] for i in items], fontsize=7)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("耗时（s，对数轴）")
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(axis="y", color=GRID, linewidth=0.45, zorder=0)
    ax.legend(loc="upper left", frameon=False, ncol=2, columnspacing=1.0)
    panel_title(ax, "a  配置读取与模型反射")

    ax = axes[1]
    labels = ["4k", "16k"]
    base = [4563.26, 4513.93]
    cand = [5209.06, 4513.91]
    x = np.arange(2)
    ax.bar(x - w / 2, base, w * 0.92, color=BASE, edgecolor=BASE_EDGE,
           linewidth=0.4, zorder=3)
    ax.bar(x + w / 2, cand, w * 0.92, color=HERO, zorder=3)
    for xi, b, c in zip(x, base, cand):
        ax.text(xi - w / 2, b + 60, f"{b:,.0f}", ha="center", va="bottom",
                fontsize=6.8, color=MID)
        ax.text(xi + w / 2, c + 60, f"{c:,.0f}", ha="center", va="bottom",
                fontsize=6.8, color=HERO)
        d = (c / b - 1) * 100
        ax.text(xi, max(b, c) + 420, "±0.0%" if abs(d) < 0.005 else f"{d:+.2f}%",
                ha="center", va="bottom", fontsize=8.5, color=ACCENT,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} 场景" for s in labels])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 6600)
    ax.set_ylabel("总吞吐（tok/s，同日中位数）")
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="y", color=GRID, linewidth=0.45, zorder=0)
    panel_title(ax, "b  官方场景端到端效果")
    fig.subplots_adjust(wspace=0.36)
    save(fig, "fig27_a31_host")


# --------------------------------------------------------------------------- #
# 28. Micro-benchmark gain versus end-to-end gain for late candidates
# --------------------------------------------------------------------------- #
def fig_micro_vs_e2e():
    # (label, micro gain %, end-to-end 4k gain %, adopted)
    items = [
        ("BI-A60 权重预打包", 10.0, 1.448, True),
        ("BI-A58 warps=16", 3.64, -0.0336, False),
        ("BI-A36 尾页分离（等长）", 8.38, -12.457, False),
        ("BI-A36 尾页分离（ragged）", 11.64, -12.457, False),
    ]
    fig, ax = plt.subplots(figsize=(15.4 * CM, 5.6 * CM))
    for label, micro, e2e, ok in items:
        color = HERO if ok else ACCENT
        ax.plot([0, 1], [micro, e2e], color=color, linewidth=1.1, zorder=2,
                marker="o", markersize=4.5, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=0.6)
    left = [("BI-A36（ragged）", 11.64, 12.6, ACCENT), ("BI-A60", 10.0, 10.3, HERO),
            ("BI-A36（等长）", 8.38, 7.9, ACCENT), ("BI-A58", 3.64, 3.64, ACCENT)]
    for name, v, ytxt, color in left:
        ax.text(-0.035, ytxt, f"{name}  {v:+.1f}%", ha="right", va="center",
                fontsize=7, color=color)
    right = [("BI-A60  采用", 1.448, 2.4, HERO), ("BI-A58  不合入", -0.0336, -1.3, ACCENT),
             ("BI-A36  不合入", -12.457, -12.457, ACCENT)]
    for name, v, ytxt, color in right:
        ax.text(1.035, ytxt, f"{v:+.2f}%  {name}", ha="left", va="center",
                fontsize=7, color=color)
    ax.axhline(0, color=MID, linewidth=0.6, linestyle=DASH, zorder=1)
    ax.set_xlim(-0.62, 1.62)
    ax.set_ylim(-15, 15)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["孤立内核微基准", "官方 4k 端到端（同日交错 A/B）"], fontsize=8)
    ax.set_ylabel("相对对照的变化（%）")
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.grid(axis="y", color=GRID, linewidth=0.45, zorder=0)
    for side in ("bottom",):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0)
    save(fig, "fig28_micro_vs_e2e")


if __name__ == "__main__":
    report_numbers()
    fig_headline()
    fig_throughput_both()
    fig_ttft_ratio()
    fig_bi_stage_ab()
    fig_c500_ablation()
    fig_round_stability()
    fig_a31_host()
    fig_micro_vs_e2e()
