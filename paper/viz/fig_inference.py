#!/usr/bin/env python3
"""
Generate Figures 5-9 for the GenMLX paper: inference algorithm comparisons.

Figures produced:
    fig05_linreg_pareto.pdf    — LinReg algorithm Pareto (time vs accuracy)
    fig06_hmm.pdf              — HMM IS vs SMC (2-panel: error bars + scatter)
    fig07_gmm.pdf              — GMM weight collapse (bar chart)
    fig08_funnel.pdf           — Neal's Funnel (horizontal bar chart)
    fig09_changepoint.pdf      — Changepoint IS vs SMC (ESS ratio bar chart)

Usage:
    /Users/robert/code/genmlx/.venv-genjax/bin/python paper/viz/fig_inference.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from genmlx_style import (
    setup, COLORS, SIZES, FONTS, LINE, BAR,
    clean_axes, save_fig, make_legend_handle,
)

RESULTS = Path(__file__).parent.parent.parent / "results"
FIGS = Path(__file__).parent.parent / "figs"


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ============================================================================
# Fig 5: LinReg Algorithm Pareto (time vs slope error)
# ============================================================================

def fig05_linreg_pareto():
    data = load_json(RESULTS / "linreg" / "data.json")
    results = data["results"]
    gt_slope = data["ground-truth"]["slope-mean"]

    # Map algorithm names to color keys
    color_map = {
        "Compiled MH": COLORS["compiled_mh"],
        "HMC": COLORS["hmc"],
        "NUTS": COLORS["nuts"],
        "VIS-1K": COLORS["vec_is"],
        "VIS-10K": COLORS["vec_is"],
    }

    fig, ax = plt.subplots(figsize=SIZES["single"])

    for r in results:
        alg = r["algorithm"]
        time_ms = r["mean-ms"]
        error = r["slope-error"]
        color = color_map.get(alg, "gray")

        ax.scatter(time_ms, error, s=160, color=color, edgecolor="black",
                   linewidth=1.2, zorder=10, alpha=0.9)

        # Label positioning: offset based on algorithm to avoid overlaps
        offsets = {
            "Compiled MH": (10, -12),
            "HMC": (10, 5),
            "NUTS": (-15, 10),
            "VIS-1K": (10, 5),
            "VIS-10K": (10, -12),
        }
        dx, dy = offsets.get(alg, (10, 5))
        ax.annotate(alg, xy=(time_ms, error), xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=FONTS["annotation"], fontweight="bold",
                    ha="left", va="center")

    ax.set_xscale("log")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Slope estimation error (absolute)")

    # Add ground-truth line at zero error
    ax.axhline(y=0, color=COLORS["truth"], linestyle="--", linewidth=1.5,
               alpha=0.5, label="Perfect accuracy")

    clean_axes(ax, grid=True)

    # Legend
    handles = [
        make_legend_handle(COLORS["compiled_mh"], "Compiled MH"),
        make_legend_handle(COLORS["hmc"], "HMC"),
        make_legend_handle(COLORS["nuts"], "NUTS"),
        make_legend_handle(COLORS["vec_is"], "VIS"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=FONTS["legend"],
              framealpha=0.9, edgecolor="black")

    save_fig(fig, str(FIGS / "fig05_linreg_pareto.pdf"))


# ============================================================================
# Fig 6: HMM IS vs SMC (2-panel)
# ============================================================================

def fig06_hmm():
    data = load_json(RESULTS / "hmm" / "data.json")
    exact_log_ml = data["exact_log_ml"]
    algorithms = data["algorithms"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SIZES["two_panel"])

    # --- Panel A: Bar chart of absolute log-ML error ---
    names = [a["algorithm"] for a in algorithms]
    errors = [a["error_mean"] for a in algorithms]
    error_stds = [a["error_std"] for a in algorithms]

    # Color mapping
    def hmm_color(alg):
        if "SMC" in alg:
            return COLORS["smc"]
        if "Seq" in alg:
            return COLORS["is"]
        return COLORS["vec_is"]

    colors = [hmm_color(n) for n in names]

    x = np.arange(len(names))
    bars = ax1.bar(x, errors, yerr=error_stds, capsize=4,
                   color=colors, edgecolor=BAR["edgecolor"],
                   linewidth=BAR["linewidth"], alpha=BAR["alpha"])

    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace("_", "\n") for n in names],
                        fontsize=FONTS["tick"] - 2)
    ax1.set_ylabel("Absolute log-ML error (nats)")
    ax1.set_title("(A) Log-ML accuracy", fontsize=FONTS["title"], fontweight="bold")

    # Annotate bars with values
    for i, (e, s) in enumerate(zip(errors, error_stds)):
        ax1.text(i, e + s + 1.5, f"{e:.1f}", ha="center", va="bottom",
                 fontsize=FONTS["bar_label"], fontweight="bold")

    clean_axes(ax1, grid=True)

    # --- Panel B: Scatter of time vs error ---
    # Per-algorithm offsets to avoid label overlap
    label_offsets = {
        "VIS_1000": (10, 5),
        "Seq_IS_100": (10, 5),
        "Batched_SMC_100": (10, 5),
        "Batched_SMC_200": (10, -14),
    }
    for a in algorithms:
        name = a["algorithm"]
        color = hmm_color(name)
        ax2.scatter(a["time_ms_mean"], a["error_mean"], s=160, color=color,
                    edgecolor="black", linewidth=1.2, zorder=10, alpha=0.9)
        dx, dy = label_offsets.get(name, (10, 5))
        ax2.annotate(name.replace("_", " "), xy=(a["time_ms_mean"], a["error_mean"]),
                     xytext=(dx, dy), textcoords="offset points",
                     fontsize=FONTS["annotation"], fontweight="bold")

    ax2.set_xscale("log")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Absolute log-ML error (nats)")
    ax2.set_title("(B) Time vs accuracy", fontsize=FONTS["title"], fontweight="bold")
    clean_axes(ax2, grid=True)

    # Shared legend
    handles = [
        make_legend_handle(COLORS["vec_is"], "Vectorized IS"),
        make_legend_handle(COLORS["is"], "Sequential IS"),
        make_legend_handle(COLORS["smc"], "Batched SMC"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.05), frameon=True,
               fontsize=FONTS["legend"], edgecolor="black", framealpha=0.9)

    save_fig(fig, str(FIGS / "fig06_hmm.pdf"))


# ============================================================================
# Fig 7: GMM Weight Collapse (bar chart)
# ============================================================================

def fig07_gmm():
    data = load_json(RESULTS / "gmm" / "data.json")
    results = data["results"]

    fig, ax = plt.subplots(figsize=SIZES["single"])

    # Build labels distinguishing VIS by particle count
    labels = []
    for r in results:
        alg = r["algorithm"]
        particles = r["particles"]
        if alg == "seq-IS":
            labels.append(f"seq-IS\n({particles}p)")
        else:
            labels.append(f"VIS\n({particles // 1000}K)" if particles >= 1000
                          else f"VIS\n({particles}p)")

    log_mls = [r["log-ml"] for r in results]

    # Color: seq-IS gets is color, VIS gets gradient of vec_is
    vis_shades = ["#4DA3D9", "#0173B2"]  # lighter to darker blue
    colors = [COLORS["is"]]
    vis_idx = 0
    for r in results[1:]:
        colors.append(vis_shades[vis_idx] if vis_idx < len(vis_shades) else COLORS["vec_is"])
        vis_idx += 1

    x = np.arange(len(results))
    bars = ax.bar(x, log_mls, color=colors, edgecolor=BAR["edgecolor"],
                  linewidth=BAR["linewidth"], alpha=BAR["alpha"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONTS["tick"] - 2)
    ax.set_ylabel("Log marginal likelihood")

    # Annotate ESS and time above/below bars
    for i, r in enumerate(results):
        ess_val = r["ess"]
        time_ms = r["mean-ms"]

        # Time annotation above bar
        ax.text(i, log_mls[i] + 0.3, f"{time_ms:.0f} ms",
                ha="center", va="bottom", fontsize=FONTS["bar_label"],
                fontweight="bold")

        # ESS annotation below time
        if ess_val is not None:
            ess_str = f"ESS={ess_val:.1f}"
        else:
            ess_str = "ESS=N/A"
        ax.text(i, log_mls[i] + 1.5, ess_str,
                ha="center", va="bottom", fontsize=FONTS["bar_label"],
                color="gray")

    clean_axes(ax, grid=True)

    # Legend
    handles = [
        make_legend_handle(COLORS["is"], "Sequential IS"),
        make_legend_handle(COLORS["vec_is"], "Vectorized IS"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=FONTS["legend"],
              framealpha=0.9, edgecolor="black")

    save_fig(fig, str(FIGS / "fig07_gmm.pdf"))


# ============================================================================
# Fig 8: Neal's Funnel (horizontal bar chart)
# ============================================================================

def fig08_funnel():
    data = load_json(RESULTS / "funnel" / "data.json")
    results = data["results"]
    gt_v_mean = data["ground-truth"]["v-mean"]

    fig, ax = plt.subplots(figsize=SIZES["single"])

    # Color mapping
    color_map = {
        "HMC-100": COLORS["hmc"],
        "NUTS-100": COLORS["nuts"],
        "compiled-MH-100": COLORS["compiled_mh"],
    }

    names = [r["algorithm"] for r in results]
    times = [r["elapsed-ms"] for r in results]
    colors = [color_map.get(n, "gray") for n in names]

    y = np.arange(len(results))
    bars = ax.barh(y, times, color=colors, edgecolor=BAR["edgecolor"],
                   linewidth=BAR["linewidth"], alpha=BAR["alpha"], height=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=FONTS["tick"])
    ax.set_xscale("log")
    ax.set_xlabel("Time (ms)")

    # Annotate v-mean and v-std beside each bar
    for i, r in enumerate(results):
        v_mean = r["v-mean"]
        v_std = r["v-std"]
        time_ms = r["elapsed-ms"]
        label = f"v={v_mean:.2f} (std={v_std:.2f})"
        ax.text(time_ms * 1.3, i, label,
                ha="left", va="center", fontsize=FONTS["annotation"],
                fontweight="bold")

    # Ground truth line annotation
    ax.text(0.98, 0.02, f"Ground truth: v-mean = {gt_v_mean}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FONTS["annotation"], style="italic", color=COLORS["truth"],
            alpha=0.8)

    clean_axes(ax, grid=True)

    # Extend x-axis to make room for annotations
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax * 8)

    save_fig(fig, str(FIGS / "fig08_funnel.pdf"))


# ============================================================================
# Fig 9: Changepoint IS vs SMC (ESS ratio bar chart)
# ============================================================================

def fig09_changepoint():
    data = load_json(RESULTS / "changepoint" / "data.json")
    results = data["results"]

    fig, ax = plt.subplots(figsize=SIZES["single"])

    # Color mapping
    def cp_color(alg):
        if "SMC" in alg:
            return COLORS["smc"]
        return COLORS["is"]

    names = [r["algorithm"] for r in results]
    ess_ratios = [r["ess-ratio"] for r in results]
    times = [r["elapsed-ms"] for r in results]
    colors = [cp_color(n) for n in names]

    x = np.arange(len(results))
    bars = ax.bar(x, ess_ratios, color=colors, edgecolor=BAR["edgecolor"],
                  linewidth=BAR["linewidth"], alpha=BAR["alpha"])

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=FONTS["tick"] - 2)
    ax.set_ylabel("ESS ratio (ESS / N)")
    ax.set_ylim(0, 1.15)

    # Annotate with ESS ratio values and time
    for i, r in enumerate(results):
        ratio = r["ess-ratio"]
        time_ms = r["elapsed-ms"]
        # ESS ratio above bar
        ax.text(i, ratio + 0.02, f"{ratio:.3f}",
                ha="center", va="bottom", fontsize=FONTS["bar_label"] + 1,
                fontweight="bold")
        # Time below the ratio label
        if time_ms >= 1000:
            time_str = f"{time_ms / 1000:.0f}s"
        else:
            time_str = f"{time_ms:.0f}ms"
        ax.text(i, ratio + 0.08, time_str,
                ha="center", va="bottom", fontsize=FONTS["bar_label"],
                color="gray")

    # Ideal line at 1.0
    ax.axhline(y=1.0, color=COLORS["truth"], linestyle="--", linewidth=1.5,
               alpha=0.4, label="Ideal ESS ratio")

    clean_axes(ax, grid=True)

    handles = [
        make_legend_handle(COLORS["is"], "Sequential IS"),
        make_legend_handle(COLORS["smc"], "Batched SMC"),
    ]
    ax.legend(handles=handles, loc="center right", fontsize=FONTS["legend"],
              framealpha=0.9, edgecolor="black")

    save_fig(fig, str(FIGS / "fig09_changepoint.pdf"))


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    setup()
    FIGS.mkdir(parents=True, exist_ok=True)

    print("Generating Figures 5-9...")
    fig05_linreg_pareto()
    fig06_hmm()
    fig07_gmm()
    fig08_funnel()
    fig09_changepoint()
    print("Done. All figures saved to paper/figs/")
