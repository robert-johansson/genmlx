"""
Generate Figures 10-14 for the GenMLX paper: Analytical Inference.

Covers auto-conjugacy ESS improvement, variance reduction across L2/L3/L3.5,
MVN dimension scaling, combinator conjugacy, and analytical MCMC.

Usage:
    cd paper/viz
    python fig_analytical.py
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from genmlx_style import (
    setup, COLORS, SIZES, FONTS, BAR, LINE,
    clean_axes, save_fig, make_legend_handle, smaller_is_better,
)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Paths
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
FIGS = os.path.join(ROOT, "paper", "figs")
os.makedirs(FIGS, exist_ok=True)


def load_json(relpath):
    with open(os.path.join(ROOT, "results", relpath)) as f:
        return json.load(f)


# =============================================================================
# Figure 10: Auto-Conjugacy ESS Improvement
# =============================================================================

def fig10_conjugacy_ess():
    """Bar chart of ESS improvement ratio per conjugate family."""
    data = load_json("conjugacy/data.json")
    improvements = data["summary"]["ess_improvements"]
    n_particles = data["config"]["n_particles"]

    families = [e["label"] for e in improvements]
    ratios = [e["ratio"] for e in improvements]

    # Gather absolute ESS values from sub_experiments
    sub = data["sub_experiments"]
    sub_keys = ["exp5a", "exp5b", "exp5c", "exp5d", "exp5e"]
    l2_ess = [sub[k]["l2"]["ess_mean"] for k in sub_keys]
    l3_ess = [sub[k]["l3"]["ess_mean"] for k in sub_keys]

    max_ratio = max(ratios)

    fig, ax = plt.subplots(figsize=SIZES["benchmark"])
    x = np.arange(len(families))

    colors = [COLORS["genmlx"] if r < max_ratio else "#6D28D9" for r in ratios]
    edge_colors = ["black" if r < max_ratio else "gold" for r in ratios]
    edge_widths = [BAR["linewidth"] if r < max_ratio else 2.5 for r in ratios]

    bars = ax.bar(x, ratios, color=colors, alpha=BAR["alpha"],
                  edgecolor=edge_colors, linewidth=edge_widths)

    # Annotate absolute ESS (L2 -> L3) below each bar
    for i, (bar, l2, l3) in enumerate(zip(bars, l2_ess, l3_ess)):
        l2_str = f"{l2:.0f}" if l2 >= 10 else f"{l2:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, -max_ratio * 0.04,
                f"{l2_str}\u2192{l3:.0f}",
                ha="center", va="top", fontsize=FONTS["bar_label"] - 1,
                color="gray")

    # Annotate ratio on top of bars
    for bar, r in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_ratio * 0.015,
                f"{r:.1f}x", ha="center", va="bottom",
                fontsize=FONTS["annotation"], fontweight="bold")

    # Mark exact conjugacy bars (ESS = N particles)
    for i, (l3, r) in enumerate(zip(l3_ess, ratios)):
        if abs(l3 - n_particles) < 0.01:
            ax.text(x[i], ratios[i] + max_ratio * 0.08, "exact",
                    ha="center", va="bottom",
                    fontsize=FONTS["bar_label"], fontstyle="italic",
                    color="#6D28D9")

    ax.set_ylabel("ESS Improvement Ratio (L3/L2)")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("-", "\n", 1) for f in families],
                       fontsize=FONTS["tick"] - 2)
    ax.set_title("Auto-Conjugacy ESS Improvement (N=50 particles)",
                 fontweight="bold")
    ax.set_ylim(-max_ratio * 0.1, max_ratio * 1.3)
    clean_axes(ax, grid=True)

    save_fig(fig, os.path.join(FIGS, "fig10_conjugacy_ess.pdf"))


# =============================================================================
# Figure 11: Variance Reduction L2 vs L3/L3.5
# =============================================================================

def fig11_variance_reduction():
    """Grouped bar chart of log-ML variance: L2 (gray) vs L3/L3.5 (purple)."""
    conj = load_json("conjugacy/data.json")
    rb = load_json("rao-blackwell/data.json")
    mvn = load_json("l3.5-mvn-conjugacy/data.json")

    # Collect groups: (label, l2_var, l3_var)
    groups = []

    # From conjugacy sub_experiments
    sub = conj["sub_experiments"]
    for key in ["exp5a", "exp5b", "exp5c", "exp5d", "exp5e"]:
        entry = sub[key]
        groups.append((entry["label"], entry["l2"]["log_ml_var"],
                        entry["l3"]["log_ml_var"]))

    # From rao-blackwell
    groups.append(("Rao-Blackwell",
                   rb["conditions"]["L2-no-analytical"]["log-ml-var"],
                   rb["conditions"]["L3-with-analytical"]["log-ml-var"]))

    # From l3.5-mvn
    groups.append(("L3.5-MVN",
                   mvn["conditions"]["L2-standard-IS"]["log-ml-var"],
                   mvn["conditions"]["L3.5-analytical"]["log-ml-var"]))

    labels = [g[0] for g in groups]
    l2_vars = [g[1] for g in groups]
    l3_vars = [g[2] for g in groups]

    # For log scale, replace 0 with a small floor value
    FLOOR = 1e-8
    l3_plot = [v if v > 0 else FLOOR for v in l3_vars]
    l2_plot = [v if v > 0 else FLOOR for v in l2_vars]

    fig, ax = plt.subplots(figsize=SIZES["benchmark"])
    x = np.arange(len(labels))
    width = 0.35

    bars_l2 = ax.bar(x - width / 2, l2_plot, width, color="gray",
                     alpha=BAR["alpha"], edgecolor=BAR["edgecolor"],
                     linewidth=BAR["linewidth"], label="L2 (standard IS)")
    bars_l3 = ax.bar(x + width / 2, l3_plot, width, color=COLORS["genmlx"],
                     alpha=BAR["alpha"], edgecolor=BAR["edgecolor"],
                     linewidth=BAR["linewidth"], label="L3/L3.5 (analytical)")

    ax.set_yscale("log")

    # Annotate "exact (0)" for zero-variance entries
    for i, v in enumerate(l3_vars):
        if v == 0:
            ax.annotate("exact (0)",
                        xy=(x[i] + width / 2, FLOOR),
                        xytext=(0, -14), textcoords="offset points",
                        ha="center", va="top",
                        fontsize=FONTS["bar_label"], fontstyle="italic",
                        color=COLORS["genmlx"])

    ax.set_ylabel("Log-ML Variance (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("-", "\n", 1) for l in labels],
                       fontsize=FONTS["tick"] - 4)
    ax.set_title("Variance Reduction: L2 Standard IS vs L3/L3.5 Analytical",
                 fontweight="bold")
    clean_axes(ax, grid=True)
    smaller_is_better(ax)

    handles = [make_legend_handle("gray", "L2 (standard IS)"),
               make_legend_handle(COLORS["genmlx"], "L3/L3.5 (analytical)")]
    ax.legend(handles=handles, loc="upper left", fontsize=FONTS["legend"])

    save_fig(fig, os.path.join(FIGS, "fig11_variance_reduction.pdf"))


# =============================================================================
# Figure 12: MVN Conjugacy Dimension Scaling
# =============================================================================

def fig12_mvn_scaling():
    """2-panel: (A) timing vs dimension, (B) ESS + posterior error summary."""
    data = load_json("l3.5-mvn-conjugacy/data.json")
    scaling = data["dimension-scaling"]
    conds = data["conditions"]

    dims = [e["d"] for e in scaling]
    analytical_ms = [e["analytical-ms"] for e in scaling]
    standard_ms = [e["standard-ms"] for e in scaling]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=SIZES["two_panel"])

    # Panel A: Timing vs dimension
    ax_a.plot(dims, standard_ms, color="gray", marker="o",
              label="L2 standard IS", **LINE["main"])
    ax_a.plot(dims, analytical_ms, color=COLORS["genmlx"], marker="s",
              label="L3.5 analytical", **LINE["main"])
    ax_a.set_xlabel("Dimension (d)")
    ax_a.set_ylabel("Time (ms)")
    ax_a.set_title("(A) Inference Time vs Dimension", fontweight="bold")
    ax_a.legend(fontsize=FONTS["legend"])
    clean_axes(ax_a, grid=True)

    # Panel B: Summary metrics (ESS improvement + posterior error)
    ess_impr = data["ess-improvement"]
    l2_ess = conds["L2-standard-IS"]["ess-mean"]
    l35_ess = conds["L3.5-analytical"]["ess-mean"]
    l2_err = conds["L2-standard-IS"]["posterior-mean-error"]
    l35_err = conds["L3.5-analytical"]["posterior-mean-error"]
    n_particles = data["config"]["n-particles"]

    metrics = ["ESS", "Posterior Error"]
    l2_vals = [l2_ess, l2_err]
    l35_vals = [l35_ess, l35_err]

    # Use two sub-axes via twin for different scales
    # Simpler: two grouped bar clusters
    x = np.arange(2)
    width = 0.3

    # ESS bars (left cluster)
    ax_b.bar(0 - width / 2, l2_ess, width, color="gray",
             alpha=BAR["alpha"], edgecolor=BAR["edgecolor"],
             linewidth=BAR["linewidth"])
    ax_b.bar(0 + width / 2, l35_ess, width, color=COLORS["genmlx"],
             alpha=BAR["alpha"], edgecolor=BAR["edgecolor"],
             linewidth=BAR["linewidth"])

    ax_b.text(0 - width / 2, l2_ess + 20, f"{l2_ess:.1f}",
              ha="center", va="bottom", fontsize=FONTS["bar_label"],
              fontweight="bold")
    ax_b.text(0 + width / 2, l35_ess + 20, f"{l35_ess:.0f}",
              ha="center", va="bottom", fontsize=FONTS["bar_label"],
              fontweight="bold")

    ax_b.set_ylabel("ESS (N=1000 particles)")
    ax_b.set_title("(B) ESS and Posterior Error (d=5)", fontweight="bold")

    # Posterior error on secondary y-axis
    ax_b2 = ax_b.twinx()
    ax_b2.bar(1 - width / 2, l2_err, width, color="gray",
              alpha=0.5, edgecolor=BAR["edgecolor"],
              linewidth=BAR["linewidth"], hatch="//")
    ax_b2.bar(1 + width / 2, l35_err, width, color=COLORS["genmlx"],
              alpha=0.5, edgecolor=BAR["edgecolor"],
              linewidth=BAR["linewidth"], hatch="//")

    ax_b2.text(1 - width / 2, l2_err * 1.1, f"{l2_err:.2f}",
               ha="center", va="bottom", fontsize=FONTS["bar_label"],
               fontweight="bold")
    ax_b2.text(1 + width / 2, l35_err * 2.5, f"{l35_err:.1e}",
               ha="center", va="bottom", fontsize=FONTS["bar_label"],
               fontweight="bold", color=COLORS["genmlx"])

    ax_b2.set_ylabel("Posterior Mean Error")
    ax_b2.spines["top"].set_visible(False)

    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(["ESS", "Posterior Error"], fontsize=FONTS["tick"])
    clean_axes(ax_b, grid=False)

    # Annotation: ESS improvement ratio
    ax_b.annotate(f"{ess_impr:.0f}x ESS improvement",
                  xy=(0.5, 0.95), xycoords="axes fraction",
                  ha="center", va="top",
                  fontsize=FONTS["annotation"] + 2, fontweight="bold",
                  color=COLORS["genmlx"],
                  bbox=dict(boxstyle="round,pad=0.3", fc="white",
                            ec=COLORS["genmlx"], alpha=0.8))

    handles = [make_legend_handle("gray", "L2 standard IS"),
               make_legend_handle(COLORS["genmlx"], "L3.5 analytical")]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04), fontsize=FONTS["legend"],
               frameon=True, edgecolor="black")

    save_fig(fig, os.path.join(FIGS, "fig12_mvn_scaling.pdf"))


# =============================================================================
# Figure 13: Combinator Conjugacy Scaling
# =============================================================================

def fig13_combinator_conjugacy():
    """Grouped bar chart: L2 vs L3.5 log-ML variance by Map combinator size K."""
    data = load_json("l3.5-combinator-conjugacy/data.json")
    results = data["results-by-size"]

    ks = [r["K"] for r in results]
    l2_vars = [r["L2-standard"]["log-ml-var"] for r in results]
    l35_vars = [r["L3.5-analytical"]["log-ml-var"] for r in results]
    ess_imprs = [r["ess-improvement"] for r in results]

    # Floor for log scale (L3.5 variance is 0)
    FLOOR = 1e-4
    l35_plot = [v if v > 0 else FLOOR for v in l35_vars]

    fig, ax = plt.subplots(figsize=SIZES["single"])
    x = np.arange(len(ks))
    width = 0.3

    bars_l2 = ax.bar(x - width / 2, l2_vars, width, color="gray",
                     alpha=BAR["alpha"], edgecolor=BAR["edgecolor"],
                     linewidth=BAR["linewidth"], label="L2 standard IS")
    bars_l35 = ax.bar(x + width / 2, l35_plot, width, color=COLORS["genmlx"],
                      alpha=BAR["alpha"], edgecolor=BAR["edgecolor"],
                      linewidth=BAR["linewidth"], label="L3.5 analytical")

    ax.set_yscale("log")

    # Annotate "exact (0)" for L3.5
    for i, v in enumerate(l35_vars):
        if v == 0:
            ax.annotate("exact (0)",
                        xy=(x[i] + width / 2, FLOOR),
                        xytext=(0, -14), textcoords="offset points",
                        ha="center", va="top",
                        fontsize=FONTS["bar_label"], fontstyle="italic",
                        color=COLORS["genmlx"])

    # Annotate ESS improvement + L2 variance combined above L2 bars
    for i, (ei, v) in enumerate(zip(ess_imprs, l2_vars)):
        bar_top = bars_l2[i].get_height()
        var_str = f"{v:.0f}" if v >= 1 else f"{v:.1f}"
        ax.annotate(f"var={var_str}\n{ei:.0f}x ESS",
                    xy=(x[i] - width / 2, bar_top),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=FONTS["bar_label"], fontweight="bold",
                    color="dimgray")

    ax.set_ylabel("Log-ML Variance (log scale)")
    ax.set_xlabel("Map Combinator Size (K)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in ks], fontsize=FONTS["tick"])
    ax.set_title("Combinator Conjugacy: L2 Variance Explodes, L3.5 Stays Exact",
                 fontweight="bold")
    clean_axes(ax, grid=True)
    smaller_is_better(ax)

    handles = [make_legend_handle("gray", "L2 standard IS"),
               make_legend_handle(COLORS["genmlx"], "L3.5 analytical")]
    ax.legend(handles=handles, loc="upper right", fontsize=FONTS["legend"])

    save_fig(fig, os.path.join(FIGS, "fig13_combinator_conjugacy.pdf"))


# =============================================================================
# Figure 14: Analytical MCMC Dimension Reduction
# =============================================================================

def fig14_analytical_mcmc():
    """2-panel: (A) ESS per parameter, (B) posterior std per parameter."""
    data = load_json("l3.5-regenerate-analytical/data.json")
    conds = data["conditions"]
    full = conds["full-mh-5d"]
    anal = conds["analytical-mh-2d"]

    params = list(full["ess"].keys())  # scale, offset, mu1, mu2, mu3

    full_ess = [full["ess"][p] for p in params]
    anal_ess = [anal["ess"][p] for p in params]

    full_std = [full["posterior-stds"][p] for p in params]
    anal_std = [anal["posterior-stds"][p] for p in params]

    # Identify which params are eliminated (mu1, mu2, mu3)
    eliminated = {"mu1", "mu2", "mu3"}

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=SIZES["two_panel"])
    x = np.arange(len(params))
    width = 0.3

    # Panel A: ESS per parameter
    bars_full_a = ax_a.bar(x - width / 2, full_ess, width, color="gray",
                           alpha=BAR["alpha"], edgecolor=BAR["edgecolor"],
                           linewidth=BAR["linewidth"])
    bars_anal_a = ax_a.bar(x + width / 2, anal_ess, width,
                           color=COLORS["genmlx"],
                           alpha=BAR["alpha"], edgecolor=BAR["edgecolor"],
                           linewidth=BAR["linewidth"])

    # Highlight eliminated params
    for i, p in enumerate(params):
        if p in eliminated:
            ax_a.axvspan(i - 0.45, i + 0.45, color=COLORS["genmlx"],
                         alpha=0.05, zorder=0)

    # Annotate ESS improvement for eliminated params
    for i, p in enumerate(params):
        if p in eliminated:
            impr = anal_ess[i] / full_ess[i]
            ax_a.text(x[i] + width / 2, anal_ess[i] + 5,
                      f"{impr:.1f}x",
                      ha="center", va="bottom",
                      fontsize=FONTS["bar_label"], fontweight="bold",
                      color=COLORS["genmlx"])

    ax_a.set_ylabel("Effective Sample Size")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(params, fontsize=FONTS["tick"])
    ax_a.set_title("(A) ESS per Parameter (500 MH samples)",
                   fontweight="bold")
    clean_axes(ax_a, grid=True)

    # Panel B: Posterior std per parameter
    bars_full_b = ax_b.bar(x - width / 2, full_std, width, color="gray",
                           alpha=BAR["alpha"], edgecolor=BAR["edgecolor"],
                           linewidth=BAR["linewidth"])
    bars_anal_b = ax_b.bar(x + width / 2, anal_std, width,
                           color=COLORS["genmlx"],
                           alpha=BAR["alpha"], edgecolor=BAR["edgecolor"],
                           linewidth=BAR["linewidth"])

    # Highlight eliminated params
    for i, p in enumerate(params):
        if p in eliminated:
            ax_b.axvspan(i - 0.45, i + 0.45, color=COLORS["genmlx"],
                         alpha=0.05, zorder=0)

    # Annotate std reduction for eliminated params
    for i, p in enumerate(params):
        if p in eliminated:
            reduction = full_std[i] / anal_std[i]
            ax_b.text(x[i], max(full_std[i], anal_std[i]) + 0.05,
                      f"{reduction:.1f}x tighter",
                      ha="center", va="bottom",
                      fontsize=FONTS["bar_label"], fontweight="bold",
                      color=COLORS["genmlx"])

    ax_b.set_ylabel("Posterior Std")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(params, fontsize=FONTS["tick"])
    ax_b.set_title("(B) Posterior Std per Parameter",
                   fontweight="bold")
    clean_axes(ax_b, grid=True)

    # Shared legend
    handles = [make_legend_handle("gray", "Full MH (5D)"),
               make_legend_handle(COLORS["genmlx"], "Analytical MH (2D)")]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04), fontsize=FONTS["legend"],
               frameon=True, edgecolor="black")

    # Annotation: dimension reduction
    fig.text(0.5, 0.97, "Dimension Reduction: 5D \u2192 2D (3 conjugate params marginalized)",
             ha="center", va="top", fontsize=FONTS["annotation"] + 1,
             fontstyle="italic", color="gray")

    save_fig(fig, os.path.join(FIGS, "fig14_analytical_mcmc.pdf"))


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    setup()
    print("Generating analytical inference figures (10-14)...\n")

    fig10_conjugacy_ess()
    fig11_variance_reduction()
    fig12_mvn_scaling()
    fig13_combinator_conjugacy()
    fig14_analytical_mcmc()

    print("\nAll analytical figures generated in paper/figs/")
