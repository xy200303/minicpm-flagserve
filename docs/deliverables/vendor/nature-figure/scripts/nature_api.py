"""Helper layer transcribed from the nature-figure API reference.

Source: nature-figure ``references/api.md`` (Apache-2.0, © the nature-skills
authors). Functions are kept behaviour-identical to the published reference;
the only additions are optional ``edgecolor`` / ``linewidth`` / ``annot_size``
kwargs on ``make_grouped_bar`` so small journal-width panels can use hairline
bar outlines instead of the reference's large-panel 1.5 pt default.

Everything else in the report's plotting script (style, layout, export, QA) is
driven from this module, so the figures follow the skill's documented flow
rather than a bespoke one.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from audit_panel_alignment import require_matplotlib_panel_alignment

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral_light": "#CFCECE",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
    "gold": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "magenta": "#EA84DD",
}

DEFAULT_COLORS = [
    PALETTE["blue_main"],
    PALETTE["green_3"],
    PALETTE["red_strong"],
    PALETTE["teal"],
    PALETTE["violet"],
    PALETTE["neutral_light"],
]


def apply_publication_style(font_size=16, axes_linewidth=2.5, use_tex=False):
    """Apply Nature-style rcParams. Call once before creating any figures."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = axes_linewidth
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["pdf.fonttype"] = 42
    if use_tex:
        plt.rcParams["text.usetex"] = True


def is_dark(hex_color, threshold=128):
    """Return True if hex color is dark (use white text on it)."""
    c = hex_color.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) < threshold


def add_panel_label(ax, label, x=0, y=1, x_offset_pt=-4, y_offset_pt=3,
                    fontsize=8, color="black", fontweight="bold", va="bottom"):
    """Place a panel label with a fixed physical offset from an axes anchor."""
    from matplotlib.transforms import ScaledTranslation

    offset = ScaledTranslation(x_offset_pt / 72, y_offset_pt / 72,
                               ax.figure.dpi_scale_trans)
    ax.text(x, y, label, transform=ax.transAxes + offset, fontsize=fontsize,
            fontweight=fontweight, color=color, ha="left", va=va)


def make_grouped_bar(ax, categories, series, labels, ylabel="Value",
                     colors=None, annotate=False, bar_width=0.8,
                     series_spread=None, error_kw=None,
                     edgecolor="black", linewidth=1.5, annot_size=10,
                     label_fmt="{:.2f}"):
    """Grouped bar chart (see the API reference for the full contract)."""
    if colors is None:
        colors = DEFAULT_COLORS
    if error_kw is None:
        error_kw = {"elinewidth": 2, "capthick": 2, "capsize": 10}
    n_groups = len(series)
    n_cats = len(categories)
    w = bar_width / n_groups
    x = np.arange(n_cats)
    flat = np.concatenate([np.asarray(v, dtype=float) for v in series])
    label_pad = 0.02 * max(float(np.ptp(flat)), float(np.max(np.abs(flat))), 1.0)
    containers = []
    for i, (vals, label, color) in enumerate(zip(series, labels, colors)):
        spread = None if series_spread is None else np.asarray(series_spread[i],
                                                               dtype=float)
        offset = (i - (n_groups - 1) / 2) * w
        bars = ax.bar(x + offset, vals, width=w, label=label, color=color,
                      edgecolor=edgecolor, linewidth=linewidth,
                      yerr=spread, error_kw=error_kw)
        containers.append(bars)
        if annotate:
            for j, (bar, val) in enumerate(zip(bars, vals)):
                upper = bar.get_height() + (0 if spread is None else spread[j])
                ax.text(bar.get_x() + bar.get_width() / 2, upper + label_pad,
                        label_fmt.format(val), ha="center", va="bottom",
                        fontsize=annot_size)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.legend()
    return containers


def make_trend(ax, x, y_series, labels, colors=None, ylabel=None, xlabel=None,
               show_shadow=True, shadow_alpha=0.15, lw=2.5, marker="o",
               markersize=8):
    """Multi-line trend plot (see the API reference for the full contract)."""
    if colors is None:
        colors = DEFAULT_COLORS
    for y, label, color in zip(y_series, labels, colors):
        y = np.asarray(y)
        if y.ndim == 2:
            mean, std = y.mean(0), y.std(0)
        else:
            mean, std = y, None
        ax.plot(x, mean, color=color, lw=lw, marker=marker,
                markersize=markersize, label=label)
        if show_shadow and std is not None:
            ax.fill_between(x, mean - std, mean + std, color=color,
                            alpha=shadow_alpha)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.legend()


def make_heatmap(ax, matrix, x_labels=None, y_labels=None, cmap="magma",
                 cbar_label=None, annotate=False, fmt="{:.2f}", fontsize=12,
                 cell_colors=None):
    """2D heatmap with optional colorbar and cell annotations."""
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    if cbar_label:
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
    if x_labels:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=30, ha="right",
                           rotation_mode="anchor")
    if y_labels:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
    if annotate:
        norm = mpl.colors.Normalize(vmin=matrix.min(), vmax=matrix.max())
        cm_obj = plt.get_cmap(cmap)
        for (i, j), val in np.ndenumerate(matrix):
            r, g, b, _ = cm_obj(norm(val))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            color = "white" if lum < 0.5 else "black"
            if cell_colors is not None:
                color = cell_colors
            ax.text(j, i, fmt.format(val), ha="center", va="center",
                    fontsize=fontsize, color=color)
    ax.set_frame_on(False)
    return im


def finalize_figure(fig, out_path, formats=None, dpi=300, pad=2,
                    bbox_inches=None, close=True, alignment_options=None):
    """tight_layout + panel-alignment gate + multi-format export."""
    base = Path(out_path)
    if formats is None:
        formats = [base.suffix.lstrip(".") or "png"]
        base = base.with_suffix("")
    os.makedirs(base.parent, exist_ok=True)
    fig.tight_layout(pad=pad)
    options = {} if alignment_options is None else dict(alignment_options)
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(base) + ".alignment.json",
        overlay_svg=str(base) + ".alignment.svg",
        strict=True,
        **options,
    )
    saved = []
    for fmt in formats:
        p = str(base) + f".{fmt}"
        kw = {}
        if bbox_inches is not None:
            kw["bbox_inches"] = bbox_inches
        fig.savefig(p, dpi=dpi, **kw)
        saved.append(p)
    if close:
        plt.close(fig)
    return saved
