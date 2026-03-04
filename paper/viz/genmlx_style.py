"""
GenMLX Paper Visualization Standards

Adapted from GenJAX Research Visualization Standards (GRVS).
Provides consistent styling for all GenMLX paper figures to match
the GenJAX paper aesthetic: clean white backgrounds, bold labels,
colorblind-friendly palette, minimal clutter.

Usage:
    from genmlx_style import setup, COLORS, SIZES, save_fig
    setup()
    fig, ax = plt.subplots(figsize=SIZES["benchmark"])
    ...
    save_fig(fig, "output.pdf")
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.legend_handler
from matplotlib.ticker import MaxNLocator
import numpy as np

# =============================================================================
# FIGURE SIZES
# =============================================================================

SIZES = {
    # Benchmark figures (wide, short — matches GenJAX perfbench)
    "benchmark": (10, 3.5),
    # Standard single panel
    "single": (6.5, 4.875),
    # Two-panel horizontal
    "two_panel": (12, 5),
    # Three-panel horizontal (scaling plots)
    "three_panel": (18, 5),
    # Posterior density plots
    "posteriors": (10, 4),
    # Heatmap / matrix
    "heatmap": (8, 5),
    # Small inline
    "small": (4.33, 3.25),
}

# =============================================================================
# COLOR PALETTE — colorblind-friendly, consistent with GenJAX
# =============================================================================

COLORS = {
    # Systems (matching GenJAX perfbench palette)
    "genmlx": "#8B5CF6",           # Purple — our system
    "genmlx_vec": "#A78BFA",       # Lighter purple (vectorized variant)
    "genjl": "darkblue",           # Gen.jl
    "genjax": "deepskyblue",       # GenJAX
    "handcoded": "gold",           # Handcoded baseline
    "numpyro": "coral",            # NumPyro
    "stan": "#029E73",             # Stan (green)

    # Inference algorithms
    "compiled_mh": "#8B5CF6",      # Purple
    "vec_traj_mh": "#A78BFA",      # Light purple
    "hmc": "#DE8F05",              # Orange
    "nuts": "#029E73",             # Green
    "advi": "#CC3311",             # Red
    "vec_is": "#0173B2",           # Blue
    "is": "#B19CD9",              # Light purple
    "smc": "#56B4E9",              # Sky blue
    "gibbs": "#E69F00",            # Yellow-orange

    # Data elements
    "data": "#CC3311",             # Red (observations)
    "truth": "#333333",           # Dark gray (ground truth)
    "analytic": "#333333",         # Dark gray (analytic posterior)

    # Neutral
    "baseline_line": "gray",
    "grid": "gray",
}

# IS variant colors (for particle-count comparisons)
IS_COLORS = {
    50: "#B19CD9",
    100: "#9B7ED8",
    500: "#8B5CF6",
    1000: "#0173B2",
    5000: "#029E73",
    10000: "#DE8F05",
}

# =============================================================================
# TYPOGRAPHY — matching GenJAX GRVS
# =============================================================================

FONTS = {
    "base": 18,
    "axis_label": 18,
    "tick": 16,
    "legend": 14,
    "title": 20,
    "annotation": 12,
    "bar_label": 10,
}

# =============================================================================
# VISUAL SPECS
# =============================================================================

LINE = {
    "main": {"linewidth": 3, "alpha": 0.9},
    "secondary": {"linewidth": 2.5, "alpha": 0.9},
    "samples": {"linewidth": 1, "alpha": 0.1},
    "truth": {"linewidth": 3, "alpha": 0.8, "linestyle": "--"},
    "grid": {"linewidth": 1.5, "alpha": 0.3},
    "baseline": {"linewidth": 1.5, "alpha": 0.6, "linestyle": ":"},
}

MARKER = {
    "data": {"s": 120, "zorder": 10, "edgecolor": "white", "linewidth": 2, "alpha": 0.9},
    "star": {"s": 400, "marker": "*", "edgecolor": "black", "linewidth": 3, "zorder": 100},
    "small_star": {"s": 200, "marker": "*", "edgecolor": "darkgoldenrod", "linewidth": 1.5, "zorder": 10},
}

BAR = {
    "alpha": 0.9,
    "edgecolor": "black",
    "linewidth": 0.5,
}

LEGEND = {
    "framealpha": 0.9,
    "fancybox": True,
    "shadow": True,
    "fontsize": 12,
    "edgecolor": "black",
}

SAVE = {
    "dpi": 300,
    "bbox_inches": "tight",
    "pad_inches": 0.05,
}

# =============================================================================
# SETUP
# =============================================================================

def setup():
    """Apply GenMLX paper visualization standards (matches GenJAX GRVS)."""
    import seaborn as sns
    sns.set_style("white")

    plt.rcParams.update({
        "font.size": FONTS["base"],
        "axes.titlesize": FONTS["title"],
        "axes.labelsize": FONTS["axis_label"],
        "axes.labelweight": "bold",
        "xtick.labelsize": FONTS["tick"],
        "ytick.labelsize": FONTS["tick"],
        "legend.fontsize": FONTS["legend"],
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def clean_axes(ax, grid=False):
    """Remove top/right spines, optionally add subtle grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, alpha=0.3, which="major")
    else:
        ax.grid(False)


def set_ticks(ax, x=3, y=3):
    """Set minimal tick count (GRVS standard: 3 per axis)."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=x, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=y, prune="both"))


def log_y_ticks(ax, vmin, vmax):
    """Set log-scale y-axis with LaTeX-formatted tick labels."""
    ax.set_yscale("log")
    ticks = []
    labels = []
    exp = int(np.floor(np.log10(vmin)))
    while 10**exp <= vmax * 10:
        ticks.append(10**exp)
        labels.append(f"$10^{{{exp}}}$")
        exp += 2
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(vmin, vmax)


def bottom_legend(fig, handles, labels, ncol=None):
    """Add single-row legend below the plot (GenJAX perfbench style)."""
    if ncol is None:
        ncol = len(labels)
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, -0.08),
        frameon=True,
        **LEGEND,
    )
    plt.subplots_adjust(bottom=0.20)


def smaller_is_better(ax):
    """Add 'Smaller is better' annotation (top-left, italic)."""
    ax.text(
        0.02, 0.98, "Smaller is better",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=FONTS["annotation"], style="italic", alpha=0.7,
    )


def add_speedup_labels(ax, x_positions, times, baseline_time):
    """Add 'X.Xx' speedup labels above bars."""
    for x, t in zip(x_positions, times):
        if t > 0 and baseline_time > 0:
            factor = t / baseline_time
            text = f"{factor:.1f}x" if factor < 10 else f"{int(factor)}x"
            ax.annotate(
                text, xy=(x, t), xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=FONTS["bar_label"], fontweight="bold",
            )


def add_baseline_line(ax, value, label=None):
    """Add horizontal dotted baseline line."""
    ax.axhline(y=value, color="gray", **{k: v for k, v in LINE["baseline"].items()})
    if label:
        ax.annotate(
            label, xy=(0.01, value),
            xycoords=("axes fraction", "data"),
            xytext=(0, 2), textcoords="offset points",
            ha="left", va="bottom",
            fontsize=FONTS["bar_label"], color="gray", alpha=0.8,
        )


def save_fig(fig, path, also_png=True):
    """Save figure as PDF (and optionally PNG) with publication settings."""
    fig.tight_layout()
    fig.savefig(path, **SAVE)
    if also_png and path.endswith(".pdf"):
        fig.savefig(path.replace(".pdf", ".png"), **SAVE)
    plt.close(fig)
    print(f"Saved: {path}")


def make_legend_handle(color, label):
    """Create a rectangle legend handle."""
    return mpatches.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black",
                              linewidth=0.5, alpha=0.9, label=label)
