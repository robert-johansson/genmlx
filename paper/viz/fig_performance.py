"""
Figures 1-4 for the GenMLX paper: compilation ladder, vectorization,
particle scaling, and compiled inference speedups.
"""

import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(__file__))
from genmlx_style import (
    setup, COLORS, SIZES, BAR, LINE, FONTS,
    clean_axes, save_fig, make_legend_handle, bottom_legend,
)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results")
FIGS = os.path.join(os.path.dirname(__file__), "..", "figs")


def load(name):
    with open(os.path.join(RESULTS, name, "data.json")) as f:
        return json.load(f)


# ── helpers ──────────────────────────────────────────────────────────────

def annotate_speedup(ax, x, y, text, above=True):
    """Place a bold speedup label above (or below) a bar."""
    offset = 6 if above else -14
    ax.annotate(
        text, xy=(x, y), xytext=(0, offset),
        textcoords="offset points",
        ha="center", va="bottom" if above else "top",
        fontsize=FONTS["bar_label"], fontweight="bold",
    )


GRAY = "#999999"


# ═════════════════════════════════════════════════════════════════════════
# Figure 1: Compilation Ladder
# ═════════════════════════════════════════════════════════════════════════

def fig01_compilation_ladder():
    data = load("compilation-ladder")
    results = {r["level"]: r for r in data["results"]}
    speedups = data["speedups"]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=SIZES["two_panel"])

    # ── Panel A: single-call latencies ───────────────────────────────
    single_levels = ["L0-dynamic", "L0-static", "L1", "L3", "L4-fit"]
    labels_a = ["L0\ndynamic", "L0\nstatic", "L1\ncompiled", "L3\nconjugacy", "L4\nfit"]
    colors_a = [GRAY, GRAY, COLORS["genmlx"], COLORS["genmlx"], COLORS["genmlx"]]

    means_a = [results[k]["mean-ms"] for k in single_levels]
    stds_a = [results[k]["std-ms"] for k in single_levels]
    x_a = np.arange(len(single_levels))

    bars_a = ax_a.bar(x_a, means_a, yerr=stds_a, capsize=3,
                      color=colors_a, **BAR)
    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(labels_a, fontsize=FONTS["tick"] - 2)
    ax_a.set_ylabel("Latency (ms)")
    ax_a.set_title("A. Single-Call Latency", fontweight="bold", loc="left")
    clean_axes(ax_a)

    # ── Panel B: iterative inference latencies (log scale) ───────────
    iter_levels = ["L2-MH-handler", "L2-MH-compiled", "L2-HMC",
                   "L4-learn", "L4-handler-loop"]
    labels_b = ["MH\nhandler", "MH\ncompiled", "HMC\n100 samp",
                "Adam\ncompiled", "Adam\nhandler"]
    colors_b = [GRAY, COLORS["genmlx"], GRAY,
                COLORS["genmlx"], GRAY]

    means_b = [results[k]["mean-ms"] for k in iter_levels]
    stds_b = [results[k]["std-ms"] for k in iter_levels]
    x_b = np.arange(len(iter_levels))

    bars_b = ax_b.bar(x_b, means_b, color=colors_b, **BAR)
    ax_b.set_yscale("log")
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(labels_b, fontsize=FONTS["tick"] - 2)
    ax_b.set_ylabel("Latency (ms, log scale)")
    ax_b.set_title("B. Iterative Inference (200 steps)", fontweight="bold", loc="left")
    clean_axes(ax_b)

    # Annotate MH speedup
    mh_sp = speedups["mh-handler-to-compiled"]
    annotate_speedup(ax_b, 1, means_b[1], f"{mh_sp:.1f}x")

    # Annotate Adam speedup (compiled vs handler, using compiled-optimizer data)
    adam_sp = means_b[4] / means_b[3]  # handler / compiled (but note L4-learn > L4-handler-loop)
    # Actually L4-learn is compiled Adam (1003ms) and L4-handler-loop is handler (234ms).
    # The compiled optimizer result from the other file shows 7.2x. For the compilation
    # ladder data, L4-learn (compiled Adam 200 iter) is slower because it runs a different
    # optimizer config. Let's just annotate absolute values since the speedup comparison
    # is in Fig 4.

    h_compiled = make_legend_handle(COLORS["genmlx"], "Compiled / Optimized")
    h_handler = make_legend_handle(GRAY, "Handler / Baseline")
    bottom_legend(fig, [h_compiled, h_handler],
                  ["Compiled / Optimized", "Handler / Baseline"])

    save_fig(fig, os.path.join(FIGS, "fig01_compilation_ladder.pdf"))


# ═════════════════════════════════════════════════════════════════════════
# Figure 2: Vectorization + FFI Scaling
# ═════════════════════════════════════════════════════════════════════════

def fig02_vectorization():
    vec_data = load("vectorization")
    ffi_data = load("ffi-scaling")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=SIZES["two_panel"])

    # ── Panel A: vectorization speedup bars ──────────────────────────
    benchmarks = vec_data["results"]
    n = len(benchmarks)
    x = np.arange(n)
    width = 0.35

    seq_means = [b["sequential"]["mean-ms"] for b in benchmarks]
    bat_means = [b["batched"]["mean-ms"] for b in benchmarks]
    speedups = [b["speedup"] for b in benchmarks]
    bench_labels = [b["benchmark"] for b in benchmarks]

    ax_a.bar(x - width / 2, seq_means, width, color=GRAY, label="Sequential", **BAR)
    ax_a.bar(x + width / 2, bat_means, width, color=COLORS["genmlx"], label="Batched", **BAR)
    ax_a.set_yscale("log")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(bench_labels, fontsize=FONTS["tick"] - 3, rotation=15, ha="right")
    ax_a.set_ylabel("Latency (ms, log scale)")
    ax_a.set_title("A. Shape-Based Batching", fontweight="bold", loc="left")
    clean_axes(ax_a)

    for i, sp in enumerate(speedups):
        if sp >= 1000:
            label = f"{sp/1000:.1f}kx"
        else:
            label = f"{sp:.0f}x"
        annotate_speedup(ax_a, x[i] + width / 2, bat_means[i], label)

    # ── Panel B: FFI scaling line plot ───────────────────────────────
    ffi_results = ffi_data["results"]
    dims = [r["D"] for r in ffi_results]
    sim_sp = [r["simulate-speedup"] for r in ffi_results]
    is_sp = [r["is-speedup"] for r in ffi_results]

    ax_b.plot(dims, sim_sp, color=COLORS["genmlx"], marker="o",
              label="Simulate speedup", **LINE["main"])
    ax_b.plot(dims, is_sp, color=COLORS["hmc"], marker="s",
              label="IS speedup", **LINE["secondary"])

    ax_b.set_xlabel("Model Dimension (D)")
    ax_b.set_ylabel("Speedup (vectorized / per-site)")
    ax_b.set_title("B. FFI Overhead Amortization", fontweight="bold", loc="left")
    ax_b.legend(fontsize=FONTS["legend"], frameon=True)
    clean_axes(ax_b, grid=True)

    # Annotate endpoints
    annotate_speedup(ax_b, dims[-1], sim_sp[-1], f"{sim_sp[-1]:.0f}x")
    annotate_speedup(ax_b, dims[-1], is_sp[-1], f"{is_sp[-1]:.0f}x")

    fig.tight_layout()
    save_fig(fig, os.path.join(FIGS, "fig02_vectorization.pdf"))


# ═════════════════════════════════════════════════════════════════════════
# Figure 3: Particle Scaling
# ═════════════════════════════════════════════════════════════════════════

def fig03_particle_scaling():
    data = load("particle-scaling")
    scaling = data["scaling"]
    ratios = data["ratios"]

    particles = [s["particles"] for s in scaling]
    times = [s["mean-ms"] for s in scaling]
    ess_vals = [s["ess"] for s in scaling]

    fig, ax1 = plt.subplots(figsize=SIZES["single"])

    color_time = COLORS["genmlx"]
    color_ess = COLORS["hmc"]

    # Left Y: time
    ln1 = ax1.plot(particles, times, color=color_time, marker="o",
                   label="Time (ms)", **LINE["main"])
    ax1.set_xscale("log")
    ax1.set_xlabel("Particles")
    ax1.set_ylabel("Latency (ms)", color=color_time)
    ax1.tick_params(axis="y", labelcolor=color_time)
    clean_axes(ax1)

    # Right Y: ESS
    ax2 = ax1.twinx()
    ln2 = ax2.plot(particles, ess_vals, color=color_ess, marker="s",
                   linestyle="--", label="ESS", **LINE["secondary"])
    ax2.set_ylabel("Effective Sample Size", color=color_ess)
    ax2.tick_params(axis="y", labelcolor=color_ess)
    ax2.spines["top"].set_visible(False)

    # Combined legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper left", fontsize=FONTS["legend"], frameon=True)

    # Annotate the key ratio
    ratio_10k = ratios["10k-vs-100"]
    ax1.annotate(
        f"100x particles\n{ratio_10k:.2f}x time",
        xy=(10000, times[-1]),
        xytext=(-80, 30), textcoords="offset points",
        fontsize=FONTS["annotation"] + 1, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.9),
    )

    ax1.set_title("Particle Scaling (Vectorized IS)", fontweight="bold", loc="left")

    save_fig(fig, os.path.join(FIGS, "fig03_particle_scaling.pdf"))


# ═════════════════════════════════════════════════════════════════════════
# Figure 4: Compiled MH & Adam Speedup
# ═════════════════════════════════════════════════════════════════════════

def fig04_compiled_speedup():
    ci_data = load("compiled-inference")
    co_data = load("compiled-optimizer")

    mh = ci_data["comparisons"]["mh-chain"]
    score = ci_data["comparisons"]["score-fn"]

    # Three comparison pairs
    labels = ["MH Chain\n(200 steps)", "Score Function", "Adam Optimizer\n(200 iter)"]
    handler_means = [
        mh["handler"]["mean-ms"],
        score["uncompiled"]["mean-ms"],
        co_data["handler_loop"]["mean_ms"],
    ]
    compiled_means = [
        mh["compiled"]["mean-ms"],
        score["compiled"]["mean-ms"],
        co_data["compiled_adam"]["mean_ms"],
    ]
    speedup_vals = [
        mh["speedup"],
        score["speedup"],
        co_data["speedup"],
    ]

    fig, ax = plt.subplots(figsize=SIZES["benchmark"])

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, handler_means, width, color=GRAY, **BAR)
    ax.bar(x + width / 2, compiled_means, width, color=COLORS["genmlx"], **BAR)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONTS["tick"] - 1)
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_title("Compiled Path Speedups", fontweight="bold", loc="left")
    clean_axes(ax)

    # Annotate speedups above compiled bars
    for i, sp in enumerate(speedup_vals):
        annotate_speedup(ax, x[i] + width / 2, compiled_means[i], f"{sp:.1f}x")

    h_handler = make_legend_handle(GRAY, "Handler")
    h_compiled = make_legend_handle(COLORS["genmlx"], "Compiled")
    bottom_legend(fig, [h_handler, h_compiled], ["Handler", "Compiled"])

    save_fig(fig, os.path.join(FIGS, "fig04_compiled_speedup.pdf"))


# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    setup()
    os.makedirs(FIGS, exist_ok=True)

    fig01_compilation_ladder()
    fig02_vectorization()
    fig03_particle_scaling()
    fig04_compiled_speedup()

    print("\nAll figures generated.")
