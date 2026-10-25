"""Report figures built through the nature-figure workflow.

Flow (SKILL.md → backend fragment → api reference):
  1. style      : nature_api.apply_publication_style()
  2. primitives : make_grouped_bar / make_heatmap / make_trend / is_dark
  3. export     : finalize_figure() — tight_layout, panel-alignment gate,
                  multi-format write (SVG + PDF)
  4. QA         : audit_pdf_text.py and audit_figure_collisions.py on the PDFs

Panels the API reference has no helper for (horizontal bars in the three
"constraint" panels and the A7 gain panel) are drawn with plain Matplotlib but
keep the same palette, tick and spine behaviour.

Type: the skill mandate is Arial; this report's typographic standard is Times
New Roman for Latin and SimHei for Chinese, so only the family is overridden
(the rest of the rcParams stay as published).
"""
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor" / "nature-figure" / "scripts"))

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FixedLocator, MaxNLocator, NullLocator  # noqa: E402

import nature_api as na  # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

na.apply_publication_style(font_size=7.5, axes_linewidth=0.6)
mpl.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.family": ["Times New Roman", "SimHei"],
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.handlelength": 0.8,
    "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.2,
    "legend.labelspacing": 0.3,
})

BUILD = HERE / "figs_build"
OUT = HERE / "submission"
CM = 1 / 2.54

P = na.PALETTE
HERO = P["blue_main"]
HERO_SOFT = P["blue_secondary"]
ACCENT = P["red_strong"]
BASE = P["neutral_light"]
BASE_EDGE = "#9A9A9A"
INK = P["neutral_black"]
MID = P["neutral_mid"]
FAINT = "#B8B8B8"

CMAP_SEQ = LinearSegmentedColormap.from_list(
    "nat_blue", ["#F6F8FC", "#D8E1EE", "#A9BDD7", "#7395BE", "#3D6BA3", "#0F4D92"]
)


def save(fig, name, pad=1.2):
    stem = name.replace(".png", "")
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(BUILD / f"{stem}.alignment.json"),
        overlay_svg=str(BUILD / f"{stem}.alignment.svg"),
        strict=True,
    )
    na.finalize_figure(fig, BUILD / stem, formats=["svg", "pdf"], dpi=400, pad=pad)
    pdf_path = BUILD / f"{stem}.pdf"
    svg_path = BUILD / f"{stem}.svg"
    shutil.copyfile(pdf_path, OUT / f"{stem}.pdf")
    shutil.copyfile(svg_path, OUT / f"{stem}.svg")
    print("saved", name)


def label_above(ax, x, y, text, color, size=7, dy=None):
    ax.text(x, y, text, ha="center", va="bottom", fontsize=size, color=color)


# --------------------------------------------------------------------------- #
# 1. C500 throughput
# --------------------------------------------------------------------------- #
def fig_c500_throughput():
    labels = ["4k 场景", "16k 场景"]
    official = np.array([5089.645, 7029.675])
    local = np.array([4210.20, 5867.59])
    current = np.array([12399.31, 11046.19])
    fig, ax = plt.subplots(figsize=(15.4 * CM, 6.4 * CM))
    cont = na.make_grouped_bar(
        ax, labels, [official, local, current],
        ["官网基线", "本机原生基线", "当前交付"],
        ylabel="总吞吐（tokens/s）", colors=[BASE, HERO_SOFT, HERO], bar_width=0.65,
        edgecolor=BASE_EDGE, linewidth=0.4)
    ax.legend(loc="upper left").set_zorder(5)
    for series, vals, color in zip(cont, [official, local, current],
                                   [MID, HERO_SOFT, HERO]):
        for i, bar in enumerate(series):
            cx = bar.get_x() + bar.get_width() / 2
            label_above(ax, cx, vals[i] + 230, f"{vals[i]:,.0f}", color, 6.8)
    ax.set_ylim(0, 14300)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    save(fig, "fig1_c500_throughput")


# --------------------------------------------------------------------------- #
# 2. Sorted-free sampling chain
# --------------------------------------------------------------------------- #
def fig_sampler():
    labels = ["PyTorch top-p\n参考实现", "fused_top_p_sample\n当前交付"]
    vals = [4.08576, 0.3808]
    fig, ax = plt.subplots(figsize=(15.4 * CM, 6.4 * CM))
    cont = na.make_grouped_bar(
        ax, labels, [np.array(vals)], ["采样链"], ylabel="单次调用耗时（ms，对数轴）",
        colors=[BASE], bar_width=0.16, edgecolor=BASE_EDGE, linewidth=0.4)
    ax.get_legend().remove()
    cont[0][1].set_facecolor(HERO)
    cont[0][1].set_edgecolor(HERO)
    for bar, v in zip(cont[0], vals):
        label_above(ax, bar.get_x() + bar.get_width() / 2, v * 1.18,
                    f"{v:.3f} ms", HERO if v < 1 else MID, 8)
    ax.annotate("", xy=(0.52, 0.37), xytext=(0.52, 3.80),
                arrowprops=dict(arrowstyle="<|-|>", color=ACCENT, linewidth=0.7,
                                shrinkA=0, shrinkB=0, mutation_scale=6))
    ax.text(0.55, 1.2, "约 10.7×", color=ACCENT, fontsize=8.5, fontweight="bold",
            va="center", ha="left")
    ax.set_yscale("log")
    ax.set_ylim(0.2, 8.0)
    ax.set_xlim(-0.55, 1.55)
    ax.yaxis.set_major_locator(FixedLocator([0.3, 0.5, 1, 2, 4, 8]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    save(fig, "fig2_sampler")


# --------------------------------------------------------------------------- #
# 3. C500 optimisation ladder
# --------------------------------------------------------------------------- #
def fig_ladder():
    labels = ["官方基线", "opt1–6", "opt7", "opt8", "opt10", "opt11"]
    vals = np.array([5089.645, 8135.29, 8273.58, 10946.86, 12145.31, 12529.39])
    fig, ax = plt.subplots(figsize=(15.4 * CM, 6.6 * CM))
    cont = na.make_grouped_bar(
        ax, labels, [vals], ["累计吞吐"], ylabel="4k 场景总吞吐（tokens/s）",
        colors=[HERO], bar_width=0.46, edgecolor=P["neutral_light"],
        linewidth=0.5)
    ax.get_legend().remove()
    bars = cont[0]
    bars[0].set_facecolor(BASE)
    bars[0].set_edgecolor(BASE_EDGE)
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, vals[i] + 210,
                f"{vals[i]:,.0f}", ha="center", va="bottom", fontsize=7,
                color=MID if i == 0 else INK)
        if i:
            gain = (vals[i] / vals[i - 1] - 1) * 100
            if gain >= 1.0:
                ax.text(bar.get_x() + bar.get_width() / 2, vals[i] + 900,
                        f"+{gain:.0f}%", ha="center", va="bottom",
                        fontsize=6.5, color=ACCENT)
    ax.set_ylim(0, 15300)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.text(-0.45, 14300, f"整体 ×{vals[-1] / vals[0]:.2f}", ha="left",
            va="center", fontsize=8.5, color=ACCENT, fontweight="bold")
    save(fig, "fig3_c500_ladder")


# --------------------------------------------------------------------------- #
# 4. Decode attention bandwidth
# --------------------------------------------------------------------------- #
def fig_decode_bw():
    labels = ["stock 3D", "A20 页对齐", "实测 copy 参考"]
    vals = np.array([258, 427, 587])
    fig, ax = plt.subplots(figsize=(15.4 * CM, 6.4 * CM))
    cont = na.make_grouped_bar(
        ax, labels, [vals], ["有效带宽"], ylabel="decode attention 有效带宽（GB/s）",
        colors=[BASE], bar_width=0.28, edgecolor=BASE_EDGE, linewidth=0.5)
    ax.get_legend().remove()
    bars = cont[0]
    bars[1].set_facecolor(HERO)
    bars[1].set_edgecolor(HERO)
    bars[2].set_facecolor("white")
    bars[2].set_edgecolor(MID)
    bars[2].set_hatch("///")
    for bar, v, c in zip(bars, vals, [MID, HERO, MID]):
        label_above(ax, bar.get_x() + bar.get_width() / 2, v + 9, f"{v}", c, 8)
    ax.axhline(587, color=ACCENT, linewidth=0.7, linestyle=(0, (4, 3)), alpha=0.9)
    ax.text(0.0, 620, "实测 copy 参考 587 GB/s", ha="left", va="bottom",
            fontsize=7, color=ACCENT)
    ax.set_ylim(0, 680)
    ax.set_yticks([0, 150, 300, 450])
    save(fig, "fig4_decode_bw")


# --------------------------------------------------------------------------- #
# 5. Iluvatar same-day A/B
# --------------------------------------------------------------------------- #
def fig_ab_same_day():
    labels = ["4k 场景", "16k 场景"]
    base = np.array([2239.43, 2956.59])
    ours = np.array([2606.56, 3541.34])
    gain = (ours / base - 1) * 100
    fig, ax = plt.subplots(figsize=(15.4 * CM, 6.6 * CM))
    for i in range(2):
        x0, x1 = i - 0.15, i + 0.15
        ax.plot([x0, x1], [base[i], ours[i]], color=FAINT, linewidth=1.1,
                zorder=1)
        ax.scatter([x0], [base[i]], s=32, color=BASE_EDGE, zorder=3,
                   edgecolors="white", linewidths=0.7)
        ax.scatter([x1], [ours[i]], s=36, color=HERO, zorder=3,
                   edgecolors="white", linewidths=0.7)
        ax.text(x0, base[i] - 145, f"{base[i]:,.0f}", ha="center", va="top",
                fontsize=7.5, color=MID)
        ax.text(x1, ours[i] + 95, f"{ours[i]:,.0f}", ha="center", va="bottom",
                fontsize=8, color=HERO)
        ax.text(i + 0.30, ours[i] - 40, f"+{gain[i]:.1f}%", ha="left",
                va="center", fontsize=8.5, color=ACCENT, fontweight="bold")
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels)
    ax.set_ylabel("总吞吐（tokens/s）")
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(1850, 3900)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.plot([], [], marker="o", color=BASE_EDGE, linewidth=0, markersize=4,
            label="A7 栈（关闭 A20）")
    ax.plot([], [], marker="o", color=HERO, linewidth=0, markersize=4,
            label="A20 自研内核")
    ax.legend(loc="upper left", frameon=False)
    save(fig, "fig5_iluvatar_ab")


# --------------------------------------------------------------------------- #
# 6. Stock decode kernel breakdown (horizontal bars)
# --------------------------------------------------------------------------- #
def fig_decode_breakdown():
    labels = ["loads + QK", "PV 的 tl.dot\n（MMA 布局转换）", "softmax"]
    vals = [0.54, 0.36, 0.02]
    colors = [HERO_SOFT, ACCENT, BASE]
    fig, ax = plt.subplots(figsize=(15.4 * CM, 6.0 * CM))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, height=0.32, color=colors,
            edgecolor=["white", "white", BASE_EDGE], linewidth=0.6)
    for yi, v, c in zip(y, vals, colors):
        inside = v > 0.3
        ax.text(v - 0.012 if inside else v + 0.010, yi, f"{v:.2f} ms",
                va="center", ha="right" if inside else "left", fontsize=7.5,
                color="white" if inside and na.is_dark(c) else INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("单步耗时（ms；stock 3D decode kernel，B=64 / kv=4096）")
    ax.set_xlim(0, 0.62)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.text(0.618, 2.52, "合计 0.92 ms/层", ha="right", va="center",
            fontsize=7, color=MID)
    save(fig, "fig6_decode_breakdown")


# --------------------------------------------------------------------------- #
# 7. A7 launch configuration (horizontal bars)
# --------------------------------------------------------------------------- #
def fig_a7_config():
    labels = ["4k 场景", "16k 场景"]
    gains = [38.0, 39.4]
    fig, ax = plt.subplots(figsize=(15.4 * CM, 5.2 * CM))
    y = np.arange(2)[::-1]
    ax.barh(y, gains, height=0.20, color=HERO)
    for yi, v in zip(y, gains):
        ax.text(v + 0.8, yi, f"+{v:.1f}%", va="center", ha="left", fontsize=9,
                color=HERO, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("相对 stock 的端到端吞吐提升（%）")
    ax.set_xlim(0, 48)
    ax.set_ylim(-0.5, 1.5)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    save(fig, "fig7_a7_config")


# --------------------------------------------------------------------------- #
# 8/9. Constraint panels (half column, horizontal bars)
# --------------------------------------------------------------------------- #
def fig_accuracy():
    labels = ["C500 当前复测 1", "C500 当前复测 2", "BI 页对齐 1", "BI 页对齐 2"]
    vals = [98.1, 98.1, 96.2, 97.1]
    colors = [HERO, HERO] + [P["violet"]] * 2
    fig, ax = plt.subplots(figsize=(7.4 * CM, 6.0 * CM))
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, height=0.40, color=colors,
            edgecolor="none", linewidth=0.4)
    for yi, v in zip(y, vals):
        ax.text(v + 0.11, yi, f"{v:.2f}", va="center", ha="left", fontsize=7)
    ax.axvline(95.0, color=ACCENT, linewidth=0.7, linestyle=(0, (4, 3)), alpha=0.9)
    ax.text(95.25, 3.62, "赛题门槛 95%", color=ACCENT, fontsize=6.5, ha="left",
            va="center", bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(93, 99.6)
    ax.set_ylim(-0.6, 3.9)
    ax.set_xlabel("MATH-500 Level 3 准确率（%）", fontsize=7.5)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    save(fig, "fig8_accuracy")


def fig_ttft():
    labels = ["C500 官方基线", "C500 本提交", "BI-V150 A20 前", "BI-V150 A20 后"]
    secs = [27.2, 17.9, 155.5, 130.1]
    colors = [BASE, HERO, BASE, HERO]
    fig, ax = plt.subplots(figsize=(7.4 * CM, 6.0 * CM))
    y = np.arange(len(secs))[::-1]
    ax.barh(y, secs, height=0.40, color=colors,
            edgecolor=[BASE_EDGE, "none", BASE_EDGE, "none"], linewidth=0.4)
    for yi, v, c in zip(y, secs, colors):
        ax.text(v + 5.0, yi, f"{v:.1f} s", va="center", ha="left", fontsize=7,
                color=HERO if c == HERO else MID)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 196)
    ax.set_xlabel("16k 场景 TTFT（s，越低越好）", fontsize=7.5)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    save(fig, "fig9_ttft")


# --------------------------------------------------------------------------- #
# 10/11/12. NSEG heatmaps
# --------------------------------------------------------------------------- #
B = [8, 16, 32, 64]
NSEG = [16, 32, 64, 128]
MS_4K = np.array([
    [0.168, 0.173, 0.190, 0.325],
    [0.248, 0.242, 0.258, 0.438],
    [0.377, 0.368, 0.403, 0.639],
    [0.620, 0.629, 0.703, 1.168],
])
MS_16K = np.array([
    [0.363, 0.392, 0.390, 0.509],
    [0.679, 0.651, 0.632, 0.800],
    [1.194, 1.121, 1.124, 1.347],
    [2.131, 2.084, 2.121, 2.566],
])
GB_16K = np.array([
    [370, 343, 344, 263],
    [395, 412, 425, 336],
    [450, 479, 478, 399],
    [504, 515, 506, 418],
])


def heat(data, name, unit, best_is_min=True, figsize=(7.3 * CM, 6.0 * CM),
         cellfs=7.4):
    fig, ax = plt.subplots(figsize=figsize)
    im = na.make_heatmap(ax, data, x_labels=NSEG, y_labels=B, cmap=CMAP_SEQ,
                         cbar_label=unit, annotate=True,
                         fmt="{:.3f}" if unit == "ms" else "{:.0f}",
                         fontsize=cellfs)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    ax.tick_params(axis="both", length=0)
    ax.set_xlabel("NSEG（每序列的分段数）", fontsize=7.5)
    ax.set_ylabel("并发批大小 B", fontsize=7.5)
    idx = np.unravel_index(np.argmin(data) if best_is_min else np.argmax(data),
                           data.shape)
    ax.add_patch(plt.Rectangle((idx[1] - 0.5, idx[0] - 0.5), 1, 1, fill=False,
                               edgecolor=ACCENT, linewidth=1.2))
    cb = im.colorbar
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1.8, width=0.5, labelsize=6.5)
    cb.ax.yaxis.label.set_size(7)
    save(fig, name)


# --------------------------------------------------------------------------- #
# 13. Decode bandwidth evolution
# --------------------------------------------------------------------------- #
def fig_decode_evolution():
    labels = ["stock 2D", "stock 3D", "stock 3D\n+A7 pin", "A20\n自研内核",
              "硬件 copy\n上限"]
    vals = np.array([117, 153, 258, 427, 587])
    fig, ax = plt.subplots(figsize=(14.5 * CM, 6.8 * CM))
    cont = na.make_grouped_bar(
        ax, labels, [vals], ["有效带宽"], ylabel="decode attention 有效带宽（GB/s）",
        colors=[BASE], bar_width=0.40, edgecolor=BASE_EDGE, linewidth=0.5)
    ax.get_legend().remove()
    bars = cont[0]
    for bar, color in zip(bars, [None, None, HERO_SOFT, HERO, "white"]):
        if color:
            bar.set_facecolor(color)
            bar.set_edgecolor("none" if color != "white" else MID)
    bars[-1].set_hatch("///")
    for bar, v, c in zip(bars, vals, [MID, MID, HERO, HERO, MID]):
        label_above(ax, bar.get_x() + bar.get_width() / 2, v + 9, f"{v}", c, 7.5)
    ax.axhline(587, color=ACCENT, linewidth=0.7, linestyle=(0, (4, 3)), alpha=0.9)
    ax.set_ylim(0, 680)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    save(fig, "fig12_decode_evolution")


# --------------------------------------------------------------------------- #
# 14. Benchmark run-to-run variance (2 panels)
# --------------------------------------------------------------------------- #
def fig_run_variance():
    k4 = [5078.77, 4440.19, 4745.85, 5103.93]
    k16 = [7024.44, 7049.04, 7038.42, 7042.59]
    fig, axes = plt.subplots(1, 2, figsize=(15.0 * CM, 5.8 * CM))
    x = np.arange(1, 5)
    for ax, vals, title in zip(axes, [k4, k16], ["4k 场景", "16k 场景"]):
        mean = float(np.mean(vals))
        spread = (max(vals) - min(vals)) / mean * 100
        na.make_trend(ax, x, [np.array(vals)], ["吞吐"], colors=[HERO],
                      ylabel="total tokens/s", xlabel="run 序号",
                      show_shadow=False, lw=1.0, markersize=4)
        ax.get_legend().remove()
        ax.axhline(mean, color=ACCENT, linewidth=0.7, linestyle=(0, (4, 3)),
                   alpha=0.9)
        ax.text(0.97, 0.04, f"均值 {mean:.0f}", transform=ax.transAxes,
                color=ACCENT, va="bottom", ha="right", fontsize=7)
        ax.set_title(f"{title}（4 次连续 run，极差 {spread:.1f}%）", fontsize=8,
                     pad=5)
        ax.set_xticks(x)
        ax.set_xlim(0.6, 4.4)
        pad = (max(vals) - min(vals)) * 0.30
        ax.set_ylim(min(vals) - pad, max(vals) + pad)
        ax.yaxis.set_major_locator(MaxNLocator(4))
    fig.subplots_adjust(wspace=0.30)
    save(fig, "fig13_run_variance", pad=1.0)


if __name__ == "__main__":
    BUILD.mkdir(exist_ok=True)
    fig_c500_throughput()
    fig_sampler()
    fig_ladder()
    fig_decode_bw()
    fig_ab_same_day()
    fig_decode_breakdown()
    fig_a7_config()
    fig_accuracy()
    fig_ttft()
    heat(MS_16K, "fig10_nseg_heatmap_16k", "ms")
    heat(GB_16K, "fig11_nseg_bw_16k", "GB/s", best_is_min=False)
    heat(MS_4K, "fig10b_nseg_heatmap_4k", "ms",
         figsize=(15.4 * CM, 7.0 * CM), cellfs=9)
    fig_decode_evolution()
    fig_run_variance()


# --------------------------------------------------------------------------- #
# 15. Per-operator hot-path composition (C500, from the two kernel profiles)
#     source: docs/baselines/metax-c500/profile_baseline.txt / profile_post_opt1.txt
# --------------------------------------------------------------------------- #
def fig_op_hotpath():
    fams = ["采样（排序/直方图/argmax）", "GEMM / Linear", "decode attention",
            "归一化与逐元素", "host 搬运（Memcpy）"]
    share = np.array([
        [38.8, 5.1],      # 采样
        [36.7, 56.9],     # GEMM
        [19.1, 29.8],     # attention
        [4.6, 6.9],       # 归一化
        [0.8, 1.2],       # host
    ])
    secs = np.array([
        [14.61, 1.24],
        [13.82, 13.81],
        [7.19, 7.22],
        [1.72, 1.67],
        [0.29, 0.29],
    ])
    delta = secs[:, 1] - secs[:, 0]
    fig, ax = plt.subplots(figsize=(15.4 * CM, 6.0 * CM))
    im = ax.imshow(share, cmap=CMAP_SEQ, aspect="auto", vmin=0, vmax=60)
    ax.set_xticks(range(2))
    ax.set_xticklabels(["基线", "采样链融合后"], fontsize=8)
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels(fams, fontsize=7.5)
    ax.tick_params(axis="both", length=0)
    for i in range(len(fams)):
        for j in range(2):
            ax.text(j, i, f"{share[i, j]:.1f}%", ha="center", va="center",
                    fontsize=8, color="white" if share[i, j] > 34 else INK)
    for i in range(len(fams)):
        ax.text(1.72, i, f"{delta[i]:+.2f} s", ha="left", va="center",
                fontsize=7.5, color=ACCENT if delta[i] < -1 else MID)
    ax.text(1.72, -0.62, "Δ 内核时间", ha="left", va="center", fontsize=7,
            color=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.34, fraction=0.030)
    cb.set_label("占内核总时间（%）", fontsize=7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1.8, width=0.5, labelsize=6.5)
    save(fig, "fig14_op_hotpath")


# --------------------------------------------------------------------------- #
# 16. Constraint compliance margins (both platforms)
# --------------------------------------------------------------------------- #
def fig_compliance():
    rows = ["4k 吞吐", "16k 吞吐", "MATH-500 精度", "16k TTFT 改善"]
    cols = ["沐曦 C500", "天数 BI-V150"]
    margin = np.array([
        [(12399.31 / 5089.645 - 1) * 100, (2606.56 / 2028.01 - 1) * 100],
        [(11046.19 / 7029.675 - 1) * 100, (3541.34 / 915.15 - 1) * 100],
        [(98.10 / 95.0 - 1) * 100, (96.20 / 95.0 - 1) * 100],
        [(1 - 18.14147 / 27.197135) * 100, (1 - 130.1 / 599.262) * 100],
    ])
    fig, ax = plt.subplots(figsize=(7.4 * CM, 6.0 * CM))
    im = ax.imshow(np.log10(margin), cmap=CMAP_SEQ, aspect="auto",
                   vmin=0, vmax=2.6)
    ax.set_xticks(range(2))
    ax.set_xticklabels(cols, fontsize=7.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7.5)
    ax.tick_params(axis="both", length=0)
    xpos = np.tile(np.arange(2), (4, 1))
    for i in range(len(rows)):
        for j in range(2):
            v = margin[i, j]
            ax.text(xpos[i, j], i, f"+{v:.1f}%" if v < 100 else f"+{v:.0f}%",
                    ha="center", va="center", fontsize=7.5,
                    color="white" if np.log10(v) > 1.55 else INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.04, fraction=0.05)
    cb.set_label("log₁₀（达标裕度 %）", fontsize=6.8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1.8, width=0.5, labelsize=6.5)
    save(fig, "fig15_compliance")


# --------------------------------------------------------------------------- #
# 16. Incremental gains, with separate measurement scopes for the two platforms
# --------------------------------------------------------------------------- #
def fig_incremental_gains():
    fig, axes = plt.subplots(1, 2, figsize=(15.4 * CM, 7.0 * CM),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    labels = ["官网基线", "opt1–6", "opt7", "opt8", "opt10", "opt11"]
    vals = np.array([5089.645, 8135.29, 8273.58, 10946.86, 12145.31, 12529.39])
    x = np.arange(len(vals))
    colors = [BASE, HERO_SOFT, HERO_SOFT, HERO, HERO, HERO]
    bars = ax.bar(x, vals, width=0.62, color=colors,
                  edgecolor=[BASE_EDGE] + ["white"] * 5, linewidth=0.45)
    for i, (bar, val) in enumerate(zip(bars, vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 220,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=6.8,
                color=MID if i == 0 else INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylabel("4k 总吞吐（tokens/s）")
    ax.set_ylim(0, 14500)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_title("a  C500：历史版本记录（跨实例）", loc="left", fontsize=8, pad=6)

    ax = axes[1]
    labels = ["A6", "A7", "A20"]
    gains4 = np.array([5.83, 38.0, 16.4])
    gains16 = np.array([1.01, 39.4, 19.8])
    x = np.arange(len(labels))
    width = 0.28
    b1 = ax.bar(x - width / 2, gains4, width, color=HERO, label="4k")
    b2 = ax.bar(x + width / 2, gains16, width, color=HERO_SOFT, label="16k")
    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"+{v:.1f}%",
                    ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("相对当日对照的吞吐提升（%）")
    ax.set_ylim(0, 47)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.set_title("b  BI-V150：各实验当日对照（非累计）", loc="left", fontsize=8, pad=6)
    fig.subplots_adjust(wspace=0.30)
    save(fig, "fig16_incremental_gains", pad=1.0)


# --------------------------------------------------------------------------- #
# 17. Local kernel speedup versus end-to-end throughput gain
# --------------------------------------------------------------------------- #
def fig_speedup_benefit():
    records = [
        ("C500 opt1\n采样", 5.4, (6573.22 / 5089.645 - 1) * 100,
         (8208.44 / 7029.675 - 1) * 100, HERO),
        ("C500 opt8\nGEMM", (1.36 + 1.56 + 1.35 + 1.67) / 4, 32.3, 14.7, HERO),
        ("BI-V150 A7\ndecode 配置", 1.76, 38.0, 39.4, P["violet"]),
        ("BI-V150 A20\ndecode 内核", 1.04 / 0.629, 16.4, 19.8, P["violet"]),
    ]
    fig, ax = plt.subplots(figsize=(15.4 * CM, 6.8 * CM))
    for name, speedup, gain4, gain16, color in records:
        ax.scatter(speedup, gain4, s=38, marker="o", color=color,
                   edgecolors="white", linewidths=0.6, zorder=3)
        ax.scatter(speedup, gain16, s=38, marker="s", color=color,
                   edgecolors="white", linewidths=0.6, zorder=3)
        ax.annotate(name, (speedup, max(gain4, gain16)),
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    va="bottom", fontsize=6.8, color=color)
    ax.set_xscale("log")
    ax.set_xlim(1.15, 6.7)
    ax.set_ylim(0, 49)
    ax.set_xticks([1.25, 1.5, 2.0, 3.0, 4.0, 5.4])
    ax.set_xticklabels(["1.25", "1.5", "2.0", "3.0", "4.0", "5.4"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.set_xlabel("局部内核加速比（×，对数轴）")
    ax.set_ylabel("官方场景端到端吞吐提升（%）")
    ax.plot([], [], marker="o", linestyle="", color=MID, label="4k 场景")
    ax.plot([], [], marker="s", linestyle="", color=MID, label="16k 场景")
    ax.plot([], [], marker="o", linestyle="", color=HERO, label="C500")
    ax.plot([], [], marker="o", linestyle="", color=P["violet"], label="BI-V150")
    ax.legend(loc="upper right", frameon=False, ncol=2, columnspacing=1.0)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.45, zorder=0)
    save(fig, "fig17_speedup_benefit")


# --------------------------------------------------------------------------- #
# 18. Memory-bandwidth utilization and GEMM arithmetic throughput
# --------------------------------------------------------------------------- #
def fig_efficiency():
    fig, axes = plt.subplots(1, 2, figsize=(15.4 * CM, 6.7 * CM),
                             gridspec_kw={"width_ratios": [0.9, 1.35]})
    ax = axes[0]
    labels = ["BI stock", "BI A20", "C500 vendor"]
    absolute = np.array([258, 427, 1466])
    reference = np.array([587, 587, 1610])
    utilization = absolute / reference * 100
    bars = ax.bar(np.arange(3), utilization, width=0.48,
                  color=[BASE, HERO, P["teal"]],
                  edgecolor=[BASE_EDGE, "white", "white"], linewidth=0.5)
    for bar, pct, bw in zip(bars, utilization, absolute):
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 2.0,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=7.5)
        ax.text(bar.get_x() + bar.get_width() / 2, pct / 2,
                f"{bw}\nGB/s", ha="center", va="center", fontsize=7,
                color="white" if bar.get_facecolor()[0] < 0.8 else INK)
    ax.axhline(100, color=ACCENT, linewidth=0.7, linestyle=(0, (4, 3)))
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("相对实测带宽参考（%）")
    ax.set_ylim(0, 113)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_title("a  decode attention 带宽利用", loc="left", fontsize=8, pad=6)
    ax.text(0.0, -0.22, "参考：BI copy 587 GB/s；C500 sum 1610 GB/s",
            transform=ax.transAxes, fontsize=6.4, color=MID)

    ax = axes[1]
    labels = ["qkv", "o_proj", "gate_up", "down"]
    vendor = np.array([163, 152, 197, 175], dtype=float)
    speedup = np.array([1.36, 1.56, 1.35, 1.67])
    triton = vendor / speedup
    x = np.arange(4)
    width = 0.32
    b1 = ax.bar(x - width / 2, triton, width, color=BASE,
                edgecolor=BASE_EDGE, linewidth=0.4, label="Triton 最优配置")
    b2 = ax.bar(x + width / 2, vendor, width, color=HERO, label="vendor mcBLAS")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                    f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("GEMM 计算吞吐（TFLOPS）")
    ax.set_ylim(0, 230)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("b  C500 prefill GEMM（M=2048）", loc="left", fontsize=8, pad=6)
    fig.subplots_adjust(wspace=0.32, bottom=0.22)
    save(fig, "fig18_efficiency", pad=1.0)


# --------------------------------------------------------------------------- #
# 19. BI-V150 prefill kernel and host-stall profiles
# --------------------------------------------------------------------------- #
def fig_biv150_bottleneck():
    fig, axes = plt.subplots(1, 2, figsize=(15.4 * CM, 6.1 * CM),
                             gridspec_kw={"width_ratios": [0.85, 1.35]})
    ax = axes[0]
    ax.barh([0], [91.8], height=0.32, color=HERO, label="unified attention")
    ax.barh([0], [8.2], left=[91.8], height=0.32, color=BASE,
            edgecolor=BASE_EDGE, linewidth=0.4, label="其他内核")
    ax.text(45.9, 0, "91.8%", ha="center", va="center", color="white", fontsize=9)
    ax.text(95.9, 0.23, "其他 8.2%", ha="center", va="bottom", color=INK, fontsize=6.5)
    ax.text(0, 0.36, "36.8 ms／层／2048-token chunk", ha="left", va="bottom",
            fontsize=7, color=MID)
    ax.set_yticks([0])
    ax.set_yticklabels(["初始 16k prefill"])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.65, 0.72)
    ax.set_xlabel("占 GPU 内核时间（%）")
    ax.legend(loc="lower left", frameon=False, ncol=1)
    ax.set_title("a  attention 占 GPU 忙时的 91.8%", loc="left", fontsize=8, pad=6)

    ax = axes[1]
    gpu = np.array([677.0 * 0.413, 353.0 * 0.763])
    wall = np.array([677.0, 353.0])
    wait = wall - gpu
    y = np.array([1, 0])
    ax.barh(y, gpu, height=0.34, color=HERO_SOFT, label="GPU 工作")
    ax.barh(y, wait, left=gpu, height=0.34, color=BASE,
            edgecolor=BASE_EDGE, linewidth=0.4, label="host 等待")
    for yi, g, w, total in zip(y, gpu, wait, wall):
        ax.text(g / 2, yi, f"{g:.0f} ms", ha="center", va="center",
                color="white", fontsize=7.5)
        ax.text(g + w / 2, yi, f"{w:.0f} ms", ha="center", va="center",
                color=INK, fontsize=7.5)
        ax.text(total + 12, yi, f"wall {total:.0f} ms", ha="left", va="center",
                color=MID, fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(["A1.5 后", "A3 同步修复后"])
    ax.set_xlim(0, 770)
    ax.set_ylim(-0.65, 1.65)
    ax.set_xlabel("16k 混合步时间分解（ms）")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.legend(loc="upper right", frameon=False, ncol=2)
    ax.set_title("b  GPU busy：41.3% → 76.3%", loc="left", fontsize=8, pad=6)
    fig.subplots_adjust(wspace=0.38)
    save(fig, "fig19_biv150_bottleneck", pad=1.0)


fig_incremental_gains()
fig_speedup_benefit()
fig_efficiency()
fig_biv150_bottleneck()
