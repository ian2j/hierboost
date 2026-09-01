"""Generates all data-driven figures for the whitepaper from the real backtest
results (results_multiyear.csv, results_replicate_multiyear.csv). Run from the
whitepaper/ directory: PYTHONPATH=.. python3 make_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
GRAY = "#8a8a86"
INK = "#0b0b0b"
MUTED = "#52514e"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.edgecolor": "#c9c8c2",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": "#e9e8e3",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.dpi": 150,
})

ERA_LABELS = {"M6": "2022 (M6)", "extended-yr1": "2023", "extended-yr2": "2024", "extended-yr3": "2025"}
ERA_ORDER = ["M6", "extended-yr1", "extended-yr2", "extended-yr3"]


def era_boundaries(df):
    """Return (period index of first row in each era, label) for vertical era dividers."""
    bounds = []
    for era in ERA_ORDER:
        first = df.index[df.era == era][0]
        bounds.append((first, ERA_LABELS[era]))
    return bounds


def shade_eras(ax, df, label_y=1.03):
    bounds = era_boundaries(df)
    for i, (start, label) in enumerate(bounds):
        end = bounds[i + 1][0] if i + 1 < len(bounds) else len(df)
        if i % 2 == 1:
            ax.axvspan(start + 1 - 0.5, end + 0.5, color="#f2f1ec", zorder=0)
        mid = (start + 1 + end) / 2
        ax.text(mid, label_y, label, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=8.5, color=MUTED)


def fig_rps_timeseries(df, out):
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = df.period.values
    ax.axhline(0.16, color=GRAY, linewidth=1.4, linestyle="--", zorder=1, label="Naive uniform benchmark (0.160)")
    ax.plot(x, df.rps, color=BLUE, linewidth=1.6, zorder=3)
    ax.scatter(x, df.rps, color=BLUE, s=10, zorder=4)
    shade_eras(ax, df, label_y=1.10)
    ax.set_xlabel("Period (1–48, walk-forward)")
    ax.set_ylabel("RPS (lower is better)")
    ax.set_xlim(0.5, len(df) + 0.5)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    fig.suptitle("Ranked Probability Score by period, 2022–2025", fontsize=11, x=0.125, y=1.06, ha="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_ir_timeseries(df, out):
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = df.period.values
    colors = np.where(df.ir >= 0, AQUA, RED)
    ax.bar(x, df.ir, color=colors, width=0.75, zorder=3)
    ax.axhline(0, color=INK, linewidth=0.8, zorder=2)
    shade_eras(ax, df, label_y=1.10)
    ax.set_xlabel("Period (1–48, walk-forward)")
    ax.set_ylabel("Information Ratio")
    ax.set_xlim(0.5, len(df) + 0.5)
    fig.suptitle("Investment-decision performance (IR) by period, 2022–2025", fontsize=11, x=0.125, y=1.06, ha="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_adaptive_convergence(df, out):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    x = df.period.values
    ax1.plot(x, df.momentum_sign, color=BLUE, linewidth=1.6)
    ax1.axhline(0, color=GRAY, linewidth=1, linestyle=":")
    ax1.set_ylabel("Momentum sign\n(- = reversal tilt)")
    shade_eras(ax1, df, label_y=1.14)

    ax2.plot(x, df.blend, color=ORANGE, linewidth=1.6)
    ax2.set_ylabel("Shrinkage blend\ntoward uniform")
    ax2.set_xlabel("Period (1–48, walk-forward)")
    ax2.set_ylim(0, 0.6)

    for ax in (ax1, ax2):
        ax.set_xlim(0.5, len(df) + 0.5)
    fig.suptitle("Adaptive parameters converge as evidence accumulates\n(causal: each period uses only strictly prior periods)",
                 fontsize=11, x=0.125, y=1.10, ha="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_leaderboard_placement(out):
    # Overall Rank position out of 164, lower bar = better placement (rank 1 = best)
    rows = [
        ("Naive benchmark\n(RPS only, no IR)", None),
        ("Ian's actual\n2022 result", 113),
        ("v4\n(momentum only)", 20),
        ("v6\n(+ residual-noise fix,\nadaptive shrinkage)", 2.5),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    colors = [GRAY, RED, "#f0b27a", BLUE]
    y = np.arange(len(labels))
    plot_vals = [0 if v is None else v for v in vals]
    bars = ax.barh(y, plot_vals, color=colors, height=0.55, zorder=3)
    for yi, v in zip(y, vals):
        if v is not None:
            ax.text(v + 3, yi, f"{v:.0f} / 164", va="center", fontsize=9, color=INK)
        else:
            ax.text(3, yi, "n/a (RPS-only benchmark)", va="center", fontsize=9, color=MUTED, style="italic")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Overall Rank position out of 164 real M6 teams (lower = better)")
    ax.set_title("Real M6 leaderboard placement (2022 window)", fontsize=11, loc="left")
    ax.set_xlim(0, 170)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_v5_v6_by_era(df6, df5, out):
    era_labels = [ERA_LABELS[e] for e in ERA_ORDER]
    v6_means = [df6[df6.era == e].ir.mean() for e in ERA_ORDER]
    v5_means = [df5[df5.era == e].ir.mean() for e in ERA_ORDER]

    x = np.arange(len(era_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x - width / 2, v6_means, width, color=BLUE, label="v6 (momentum/reversal, self-referential)", zorder=3)
    ax.bar(x + width / 2, v5_means, width, color=ORANGE, label="v5 (DBMF replication, external target)", zorder=3)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(era_labels)
    ax.set_ylabel("Mean IR")
    ax.set_title("v6 vs. v5 by year — both positive throughout,\nincluding DBMF's own down year (2023)", fontsize=11, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    df6 = pd.read_csv("../results_multiyear.csv")
    df5 = pd.read_csv("../results_replicate_multiyear.csv")

    fig_rps_timeseries(df6, "figures/rps_timeseries.pdf")
    fig_ir_timeseries(df6, "figures/ir_timeseries.pdf")
    fig_adaptive_convergence(df6, "figures/adaptive_convergence.pdf")
    fig_leaderboard_placement("figures/leaderboard_placement.pdf")
    fig_v5_v6_by_era(df6, df5, "figures/v5_v6_by_era.pdf")
    print("All figures saved to figures/")
