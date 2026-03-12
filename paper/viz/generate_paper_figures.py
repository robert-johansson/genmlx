"""
Generate all figures for the TOPML paper from benchmark data.

Usage:
    cd paper/viz
    python generate_paper_figures.py

Reads data from results/paper/exp{01-11}_*/ and outputs to paper/figs/.
"""

import json
import os
import sys
import numpy as np

# Add viz directory to path for style import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from genmlx_style import (
    setup, COLORS, SIZES, FONTS, LINE, BAR, LEGEND, MARKER,
    clean_axes, save_fig, add_baseline_line, log_y_ticks,
    make_legend_handle, smaller_is_better, IS_COLORS,
)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Paths
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RESULTS = os.path.join(ROOT, "results", "paper")
FIGS = os.path.join(ROOT, "paper", "figs")

os.makedirs(FIGS, exist_ok=True)

def load_json(path):
    with open(os.path.join(RESULTS, path)) as f:
        return json.load(f)


# =============================================================================
# Figure 1: Compilation Ladder (THE HERO FIGURE)
# =============================================================================

def fig_compilation_ladder():
    """Bar chart showing performance at each compilation level."""
    data = load_json("exp01_ladder/ladder_results.json")
    results = data["results"]

    # Extract data for the bar chart — select key levels
    levels = []
    times = []
    colors = []

    level_config = [
        ("L0", "L0-dynamic", COLORS["genmlx"], "Handler\n(dynamic)"),
        ("L1", "L1", "#A78BFA", "Compiled\ngenerate"),
        ("L2-VIS", "L2-VIS", COLORS["vec_is"], "Vectorized IS\n(1000 ptcls)"),
        ("L2-MH\n(handler)", "L2-MH-handler", "#CC3311", "Handler\nMH loop"),
        ("L2-MH\n(compiled)", "L2-MH-compiled", COLORS["compiled_mh"], "Compiled\nMH chain"),
        ("L3", "L3", COLORS["nuts"], "Auto-\nconjugacy"),
        ("L4-fit", "L4-fit", COLORS["hmc"], "fit API"),
    ]

    labels = []
    for label, level_name, color, _ in level_config:
        r = next((r for r in results if r["level"] == level_name), None)
        if r:
            levels.append(label)
            times.append(r["mean_ms"])
            colors.append(color)
            labels.append(label)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(levels))
    bars = ax.bar(x, times, color=colors, **BAR)

    ax.set_yscale("log")
    ax.set_ylabel("Time (ms)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=12)
    ax.set_title("GenMLX Compilation Ladder: Same Model, Progressive Optimization", fontsize=16, fontweight="bold")
    clean_axes(ax, grid=True)

    # Annotate times on bars
    for bar, t in zip(bars, times):
        if t < 1:
            label = f"{t:.2f}ms"
        elif t < 100:
            label = f"{t:.1f}ms"
        else:
            label = f"{t:.0f}ms"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
                label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Add MH speedup annotation
    mh_handler = next(r["mean_ms"] for r in results if r["level"] == "L2-MH-handler")
    mh_compiled = next(r["mean_ms"] for r in results if r["level"] == "L2-MH-compiled")
    speedup = mh_handler / mh_compiled
    ax.annotate(f"{speedup:.0f}x faster",
                xy=(4, mh_compiled), xytext=(4.6, mh_compiled * 5),
                fontsize=12, fontweight="bold", color=COLORS["compiled_mh"],
                arrowprops=dict(arrowstyle="->", color=COLORS["compiled_mh"], lw=2))

    save_fig(fig, os.path.join(FIGS, "fig_ladder.pdf"))


# =============================================================================
# Figure 2: Vectorization Speedups (Bar chart)
# =============================================================================

def fig_vectorization_speedup():
    """Bar chart: sequential vs batched for 3 methods."""
    data = load_json("exp02_vectorization/method_speedup.json")
    results = data["results"]

    methods = [r["method"].replace("_", "\n") for r in results]
    seq_times = [r["sequential"]["mean"] for r in results]
    bat_times = [r["batched"]["mean"] for r in results]
    speedups = [r["speedup"] for r in results]

    fig, ax = plt.subplots(figsize=SIZES["benchmark"])
    x = np.arange(len(methods))
    w = 0.35

    bars_seq = ax.bar(x - w/2, seq_times, w, label="Sequential", color="#CC3311", **BAR)
    bars_bat = ax.bar(x + w/2, bat_times, w, label="Batched (MLX)", color=COLORS["genmlx"], **BAR)

    ax.set_yscale("log")
    ax.set_ylabel("Time (ms)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=14)
    ax.set_title("Shape-Based Batching: Sequential vs Vectorized (N=1000)", fontweight="bold")
    clean_axes(ax, grid=True)

    # Speedup annotations
    for i, (s, bt) in enumerate(zip(speedups, bat_times)):
        ax.text(i + w/2, bt * 0.3, f"{s:,.0f}x", ha="center", va="top",
                fontsize=13, fontweight="bold", color="white")

    ax.legend(fontsize=14, loc="upper right", framealpha=0.9)
    save_fig(fig, os.path.join(FIGS, "fig_vectorization.pdf"))


# =============================================================================
# Figure 3: Cross-System IS Comparison (Grouped bars)
# =============================================================================

def fig_cross_system_is():
    """Grouped bar chart: GenMLX vs Gen.jl vs JAX for IS."""
    comparison = load_json("exp07_system/cross_system_comparison.json")
    is_data = comparison["comparisons"]["importance_sampling_1000"]

    models = ["LinReg", "GMM", "HMM"]
    genmlx = [is_data["linreg"]["genmlx_vis_ms"], is_data["gmm"]["genmlx_vis_ms"], is_data["hmm"]["genmlx_vis_ms"]]
    genjl = [is_data["linreg"]["genjl_ms"], is_data["gmm"]["genjl_ms"], is_data["hmm"]["genjl_ms"]]
    jax = [is_data["linreg"]["jax_ms"], is_data["gmm"]["jax_ms"], None]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    w = 0.25

    bars1 = ax.bar(x - w, genmlx, w, label="GenMLX (VIS, Metal)", color=COLORS["genmlx"], **BAR)
    bars2 = ax.bar(x, genjl, w, label="Gen.jl (CPU)", color=COLORS["genjl"], **BAR)
    jax_vals = [v if v is not None else 0 for v in jax]
    bars3 = ax.bar(x + w, jax_vals, w, label="JAX (CPU, JIT)", color=COLORS["genjax"], **BAR)

    # Hide the missing HMM JAX bar
    if jax[2] is None:
        bars3[2].set_alpha(0)

    ax.set_yscale("log")
    ax.set_ylabel("Time (ms)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=16)
    ax.set_title("Importance Sampling (1000 particles): Cross-System Comparison", fontweight="bold")
    clean_axes(ax, grid=True)

    # Annotate GenMLX times
    for bar, t in zip(bars1, genmlx):
        ax.text(bar.get_x() + bar.get_width()/2, t * 1.4,
                f"{t:.1f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=COLORS["genmlx"])

    ax.legend(fontsize=13, loc="upper left", framealpha=0.9)

    # Add gold stars where GenMLX wins
    for i, (g, j) in enumerate(zip(genmlx, genjl)):
        if g < j:
            ax.scatter(i - w, g * 0.5, **MARKER["small_star"], color="gold")

    save_fig(fig, os.path.join(FIGS, "fig_cross_system_is.pdf"))


# =============================================================================
# Figure 4: Particle Scaling (Sublinear)
# =============================================================================

def fig_particle_scaling():
    """Line plot: time vs particle count showing sublinear GPU scaling."""
    comparison = load_json("exp07_system/cross_system_comparison.json")
    ps = comparison["comparisons"]["particle_scaling"]

    particles = [1000, 10000]
    genmlx_times = [ps["linreg_genmlx_1k_ms"], ps["linreg_genmlx_10k_ms"]]
    jax_times = [ps["jax_1k_ms"], ps["jax_10k_ms"]]

    # Add HMM batched SMC data from exp04b
    hmm_particles = [100, 250, 1000]
    hmm_times = [296, 318, 319]  # From benchmark output

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SIZES["two_panel"])

    # Panel 1: IS particle scaling
    ax1.plot(particles, genmlx_times, "o-", color=COLORS["genmlx"], markersize=10,
             label=f"GenMLX ({ps['scaling_factor']:.2f}x)", **LINE["main"])
    ax1.plot(particles, jax_times, "s-", color=COLORS["genjax"], markersize=10,
             label=f"JAX ({ps['jax_scaling_factor']:.2f}x)", **LINE["secondary"])

    # Linear reference
    linear = [genmlx_times[0], genmlx_times[0] * 10]
    ax1.plot(particles, linear, color="gray", **LINE["baseline"], label="Linear scaling")

    ax1.set_xlabel("Particles", fontweight="bold")
    ax1.set_ylabel("Time (ms)", fontweight="bold")
    ax1.set_title("IS: Particle Scaling (LinReg)", fontweight="bold")
    ax1.set_xscale("log")
    ax1.legend(fontsize=12, loc="upper left")
    clean_axes(ax1, grid=True)

    ax1.annotate(f"10x particles\n{ps['scaling_factor']:.0f}% more time",
                 xy=(10000, genmlx_times[1]), xytext=(3000, genmlx_times[1] * 3),
                 fontsize=11, fontweight="bold", color=COLORS["genmlx"],
                 arrowprops=dict(arrowstyle="->", color=COLORS["genmlx"], lw=2))

    # Panel 2: HMM batched SMC flat scaling
    ax2.plot(hmm_particles, hmm_times, "o-", color=COLORS["smc"], markersize=10,
             label="Batched SMC (HMM)", **LINE["main"])
    linear_smc = [hmm_times[0], hmm_times[0] * (1000/100)]
    ax2.plot([100, 1000], linear_smc, color="gray", **LINE["baseline"], label="Linear scaling")

    ax2.set_xlabel("Particles", fontweight="bold")
    ax2.set_ylabel("Time (ms)", fontweight="bold")
    ax2.set_title("SMC: Flat Particle Scaling (HMM)", fontweight="bold")
    ax2.legend(fontsize=12, loc="upper left")
    clean_axes(ax2, grid=True)

    ax2.annotate("10x particles\nsame time!",
                 xy=(1000, 319), xytext=(400, 1500),
                 fontsize=11, fontweight="bold", color=COLORS["smc"],
                 arrowprops=dict(arrowstyle="->", color=COLORS["smc"], lw=2))

    save_fig(fig, os.path.join(FIGS, "fig_particle_scaling.pdf"))


# =============================================================================
# Figure 5: HMM Algorithm Comparison
# =============================================================================

def fig_hmm_comparison():
    """Bar chart: sequential vs batched SMC, IS."""
    # Data from exp04b output
    algorithms = ["SMC\n(seq, 100)", "SMC\n(seq, 250)", "SMC\n(bat, 100)", "SMC\n(bat, 250)", "SMC\n(bat, 1000)", "IS\n(seq, 1000)", "VIS\n(1000)"]
    times = [22419, 49337, 296, 318, 319, 63413, 84]
    errors = [0.45, 0.41, 0.45, 0.41, 0.12, 19.74, 22.00]
    colors_list = ["#CC3311", "#CC3311", COLORS["smc"], COLORS["smc"], COLORS["smc"], "#CC3311", COLORS["vec_is"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SIZES["two_panel"])

    # Panel 1: Time comparison
    x = np.arange(len(algorithms))
    bars = ax1.bar(x, times, color=colors_list, **BAR)
    ax1.set_yscale("log")
    ax1.set_ylabel("Time (ms)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, fontsize=9)
    ax1.set_title("HMM: Time Comparison", fontweight="bold")
    clean_axes(ax1, grid=True)

    # Speedup annotations for batched
    for i, (t, alg) in enumerate(zip(times, algorithms)):
        if t < 500:
            ax1.text(i, t * 1.5, f"{t}ms", ha="center", fontsize=9, fontweight="bold")

    # Panel 2: Accuracy (log-ML error)
    bars2 = ax2.bar(x, errors, color=colors_list, **BAR)
    ax2.set_ylabel("|log-ML error| (nats)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, fontsize=9)
    ax2.set_title("HMM: Inference Accuracy", fontweight="bold")
    ax2.axhline(y=1.0, color="gray", **LINE["baseline"])
    ax2.text(0.02, 1.2, "1 nat error threshold", fontsize=9, color="gray", alpha=0.8)
    clean_axes(ax2, grid=True)

    # Legend
    handles = [
        make_legend_handle("#CC3311", "Sequential"),
        make_legend_handle(COLORS["smc"], "Batched SMC"),
        make_legend_handle(COLORS["vec_is"], "Vectorized IS"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=12,
               bbox_to_anchor=(0.5, -0.05), framealpha=0.9)

    save_fig(fig, os.path.join(FIGS, "fig_hmm_comparison.pdf"))


# =============================================================================
# Figure 6: L3 Conjugacy (Observation Scaling)
# =============================================================================

def fig_conjugacy():
    """Two-panel: L3 vs L2 observation scaling + Rao-Blackwellization."""
    # Data from L3 evaluation output
    n_obs = [5, 10, 20, 50]
    l3_logml = [-7.708, -12.649, -22.184, -50.211]
    l2_logml = [-7.932, -12.923, -22.501, -50.543]
    l2_std = [0.404, 0.414, 0.412, 0.417]
    l2_ess = [11.4, 7.8, 5.2, 3.1]
    ess_ratio = [e/200 * 100 for e in l2_ess]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SIZES["two_panel"])

    # Panel 1: log-ML accuracy
    ax1.plot(n_obs, l3_logml, "o-", color=COLORS["nuts"], markersize=10,
             label="L3 (exact)", **LINE["main"])
    ax1.errorbar(n_obs, l2_logml, yerr=l2_std, fmt="s-", color=COLORS["is"],
                 markersize=8, capsize=5, label="L2 (200 IS)", **LINE["secondary"])
    ax1.set_xlabel("Number of observations", fontweight="bold")
    ax1.set_ylabel("log P(y)", fontweight="bold")
    ax1.set_title("L3 vs L2: Marginal Log-Likelihood", fontweight="bold")
    ax1.legend(fontsize=13, loc="lower left")
    clean_axes(ax1, grid=True)

    # Panel 2: ESS collapse
    ax2.bar(range(len(n_obs)), ess_ratio, color=COLORS["is"], **BAR)
    ax2.axhline(y=100, color=COLORS["nuts"], **LINE["truth"], label="L3 (100% effective)")
    ax2.set_xticks(range(len(n_obs)))
    ax2.set_xticklabels([str(n) for n in n_obs], fontsize=14)
    ax2.set_xlabel("Number of observations", fontweight="bold")
    ax2.set_ylabel("ESS / N (%)", fontweight="bold")
    ax2.set_title("L2 Effective Sample Size Collapse", fontweight="bold")
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=13, loc="upper right")
    clean_axes(ax2, grid=True)

    # Annotate
    for i, (n, e) in enumerate(zip(n_obs, ess_ratio)):
        ax2.text(i, e + 2, f"{e:.1f}%", ha="center", fontsize=11, fontweight="bold")

    save_fig(fig, os.path.join(FIGS, "fig_conjugacy.pdf"))


# =============================================================================
# Figure 7: L3 Rao-Blackwellization
# =============================================================================

def fig_rao_blackwell():
    """Bar chart: L3 vs L2 for mixed model."""
    methods = ["L3\n(3/5 dims eliminated)", "L2\n(all 5 dims sampled)"]
    logml_means = [-28.932, -66.488]
    logml_stds = [0.440, 14.728]
    ess = [7.7, 1.1]
    colors_list = [COLORS["nuts"], COLORS["is"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SIZES["two_panel"])

    # Panel 1: log-ML with error bars
    x = np.arange(len(methods))
    bars = ax1.bar(x, [-m for m in logml_means], yerr=logml_stds, capsize=8,
                   color=colors_list, **BAR)
    ax1.set_ylabel("-log P(y) (lower is better)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=13)
    ax1.set_title("Rao-Blackwellization: log-ML Accuracy", fontweight="bold")
    clean_axes(ax1)

    ax1.annotate(f"33.5x lower\nvariance",
                 xy=(0, 28.932), xytext=(0.5, 50),
                 fontsize=12, fontweight="bold", color=COLORS["nuts"],
                 arrowprops=dict(arrowstyle="->", color=COLORS["nuts"], lw=2))

    # Panel 2: ESS
    bars2 = ax2.bar(x, ess, color=colors_list, **BAR)
    ax2.set_ylabel("ESS (out of 200)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=13)
    ax2.set_title("Rao-Blackwellization: Sample Efficiency", fontweight="bold")
    clean_axes(ax2)

    for i, e in enumerate(ess):
        ax2.text(i, e + 0.3, f"ESS={e}", ha="center", fontsize=13, fontweight="bold")

    ax2.annotate(f"7.2x higher\nESS",
                 xy=(0, 7.7), xytext=(0.5, 5),
                 fontsize=12, fontweight="bold", color=COLORS["nuts"],
                 arrowprops=dict(arrowstyle="->", color=COLORS["nuts"], lw=2))

    save_fig(fig, os.path.join(FIGS, "fig_rao_blackwell.pdf"))


# =============================================================================
# Figure 8: L4 Compiled Adam Speedup
# =============================================================================

def fig_compiled_adam():
    """Bar chart: handler vs compiled Adam optimization."""
    data = load_json("exp10_optimization/optimization_results.json")
    adam = data["exp10a_compiled_adam"]

    methods = ["Handler Adam\n(manual loop)", "Compiled Adam\n(co/learn)"]
    times = [adam["handler"]["mean_ms"], adam["compiled"]["mean_ms"]]
    stds = [adam["handler"]["std_ms"], adam["compiled"]["std_ms"]]
    colors_list = ["#CC3311", COLORS["genmlx"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, times, yerr=stds, capsize=8, color=colors_list, **BAR)

    ax.set_ylabel("Time (ms, 200 iterations)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=14)
    ax.set_title("L4: Compiled Adam vs Handler Loop", fontweight="bold", fontsize=16)
    clean_axes(ax)

    # Speedup annotation
    speedup = adam["speedup"]
    ax.annotate(f"{speedup:.0f}x faster",
                xy=(1, times[1] + stds[1]), xytext=(1.3, times[0] * 0.4),
                fontsize=16, fontweight="bold", color=COLORS["genmlx"],
                arrowprops=dict(arrowstyle="->", color=COLORS["genmlx"], lw=3))

    # Time labels
    for i, (t, s) in enumerate(zip(times, stds)):
        label = f"{t:.0f}ms" if t > 100 else f"{t:.1f}ms"
        ax.text(i, t + s + times[0] * 0.02, label, ha="center", fontsize=12, fontweight="bold")

    save_fig(fig, os.path.join(FIGS, "fig_compiled_adam.pdf"))


# =============================================================================
# Figure 9: Method Selection Decision Tree
# =============================================================================

def fig_method_selection():
    """Table-style visualization of method selection accuracy."""
    data = load_json("exp10_optimization/optimization_results.json")
    ms_results = data["exp10b_method_selection"]["results"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    # Build table data
    headers = ["Model", "Eliminated", "Residual", "Selected Method", "Correct"]
    rows = []
    for r in ms_results:
        rows.append([
            r["label"],
            str(r["n_eliminated"]),
            str(r["n_residual"]),
            r["actual"],
            "Yes" if r["pass"] else "No"
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="center", colColours=[COLORS["genmlx"]]*5)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Color header text white
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color="white", fontweight="bold")
        if col == 4 and row > 0:
            cell.set_facecolor("#d4edda" if rows[row-1][4] == "Yes" else "#f8d7da")

    ax.set_title("L4 Method Selection: 6/6 Correct", fontsize=16, fontweight="bold", pad=20)
    save_fig(fig, os.path.join(FIGS, "fig_method_selection.pdf"))


# =============================================================================
# Figure 10: Compiled Inference Speedups
# =============================================================================

def fig_compiled_inference():
    """Bar chart: compilation speedups for MH, score-fn, vectorized MH."""
    data = load_json("exp06_compiled/compiled_speedup.json")

    benchmarks = ["MH\n(500 steps)", "Score\nfunction", "Vectorized MH\n(10 chains)"]
    uncompiled = [
        data["bench1_gfi_mh_vs_compiled_mh"]["gfi_mh"]["mean"],
        data["bench2_score_fn_compilation"]["uncompiled"]["mean"],
        data["bench4_serial_vs_vectorized_mh"]["serial_extrapolated"]["mean"],
    ]
    compiled = [
        data["bench1_gfi_mh_vs_compiled_mh"]["compiled_mh"]["mean"],
        data["bench2_score_fn_compilation"]["compiled"]["mean"],
        data["bench4_serial_vs_vectorized_mh"]["vectorized"]["mean"],
    ]
    speedups = [u/c for u, c in zip(uncompiled, compiled)]

    fig, ax = plt.subplots(figsize=SIZES["benchmark"])
    x = np.arange(len(benchmarks))
    w = 0.35

    bars1 = ax.bar(x - w/2, uncompiled, w, label="Uncompiled / Serial", color="#CC3311", **BAR)
    bars2 = ax.bar(x + w/2, compiled, w, label="Compiled / Vectorized", color=COLORS["genmlx"], **BAR)

    ax.set_yscale("log")
    ax.set_ylabel("Time (ms)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=13)
    ax.set_title("Compiled Inference Speedups", fontweight="bold")
    clean_axes(ax, grid=True)
    ax.legend(fontsize=13, loc="upper right")

    # Speedup labels
    for i, (s, c) in enumerate(zip(speedups, compiled)):
        ax.text(i + w/2, c * 0.3, f"{s:.1f}x", ha="center", va="top",
                fontsize=13, fontweight="bold", color="white")

    save_fig(fig, os.path.join(FIGS, "fig_compiled_inference.pdf"))


# =============================================================================
# Figure 11: Verification Suite
# =============================================================================

def fig_verification():
    """Horizontal bar chart: test pass rates by suite."""
    data = load_json("exp11_verification/verification_results.json")
    suites = data["suites"]

    names = [s["name"] for s in suites]
    totals = [s["total"] for s in suites]
    passes = [s["pass"] for s in suites]
    rates = [p/t * 100 for p, t in zip(passes, totals)]

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(names))

    bars = ax.barh(y, rates, color=[COLORS["nuts"] if r == 100 else COLORS["hmc"] for r in rates],
                   **BAR)

    ax.set_xlim(95, 101)
    ax.set_xlabel("Pass Rate (%)", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_title("Verification Suite: 740/744 (99.5%)", fontweight="bold")
    clean_axes(ax)
    ax.axvline(x=100, color="gray", **LINE["baseline"])

    for i, (p, t, r) in enumerate(zip(passes, totals, rates)):
        ax.text(r - 0.3, i, f"{p}/{t}", ha="right", va="center",
                fontsize=11, fontweight="bold", color="white")

    save_fig(fig, os.path.join(FIGS, "fig_verification.pdf"))


# =============================================================================
# Figure 12: Cross-System Full Comparison
# =============================================================================

def fig_cross_system_full():
    """Comprehensive cross-system comparison with multiple benchmarks."""
    benchmarks = ["LinReg\nIS (1K)", "GMM\nIS (1K)", "HMM\nIS (1K)", "LinReg\nMH (5K)", "HMM\nSMC (100)"]

    genmlx = [1.036, 2.095, 8.165, 777.9, 296]
    genjl = [1.713, 5.311, 46.722, 40.305, 603.5]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(benchmarks))
    w = 0.35

    bars1 = ax.bar(x - w/2, genmlx, w, label="GenMLX (Metal GPU)", color=COLORS["genmlx"], **BAR)
    bars2 = ax.bar(x + w/2, genjl, w, label="Gen.jl (CPU)", color=COLORS["genjl"], **BAR)

    ax.set_yscale("log")
    ax.set_ylabel("Time (ms)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=12)
    ax.set_title("GenMLX vs Gen.jl: Full Benchmark Suite", fontweight="bold", fontsize=16)
    clean_axes(ax, grid=True)
    ax.legend(fontsize=14, loc="upper left")

    # Star winners
    for i, (g, j) in enumerate(zip(genmlx, genjl)):
        winner_x = i - w/2 if g < j else i + w/2
        winner_t = min(g, j)
        ax.scatter(winner_x, winner_t * 0.4, **MARKER["small_star"], color="gold")

        ratio = max(g, j) / min(g, j)
        winner = "GenMLX" if g < j else "Gen.jl"
        ax.text(i, max(g, j) * 1.5, f"{ratio:.1f}x", ha="center",
                fontsize=9, fontweight="bold",
                color=COLORS["genmlx"] if g < j else COLORS["genjl"])

    save_fig(fig, os.path.join(FIGS, "fig_cross_system_full.pdf"))


# =============================================================================
# Figure 13: GMM Comparison
# =============================================================================

def fig_gmm_comparison():
    """Two-panel: accuracy + time for GMM."""
    algorithms = ["Exact", "Seq IS\n(1000)", "VIS\n(1000)", "Gibbs\n(500)"]
    errors = [0, 4.49, 5.96, 0]  # |log-ML error|
    times = [4, 13139, 22, 107172]
    colors_list = [COLORS["truth"], "#CC3311", COLORS["vec_is"], COLORS["gibbs"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SIZES["two_panel"])

    # Panel 1: Time
    x = np.arange(len(algorithms))
    bars = ax1.bar(x, times, color=colors_list, **BAR)
    ax1.set_yscale("log")
    ax1.set_ylabel("Time (ms)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, fontsize=13)
    ax1.set_title("GMM: Inference Time", fontweight="bold")
    clean_axes(ax1, grid=True)

    # Speedup
    ax1.annotate(f"605x faster", xy=(2, 22), xytext=(2.5, 2000),
                 fontsize=12, fontweight="bold", color=COLORS["vec_is"],
                 arrowprops=dict(arrowstyle="->", color=COLORS["vec_is"], lw=2))

    # Panel 2: Accuracy
    accuracy = [1.0, 0, 0, 1.0]  # Gibbs and exact achieve perfect assignment
    accuracy_labels = ["Exact\n(analytic)", "IS\n(ESS=1.2)", "VIS\n(ESS=1.9)", "Gibbs\n(acc=1.0)"]
    ess_vals = [None, 1.2, 1.9, None]
    gibbs_acc = [None, None, None, 1.0]

    bars2 = ax2.bar(x, [1.0 if a else 0 for a in [True, False, False, True]],
                    color=colors_list, **BAR)
    ax2.set_ylabel("Assignment Accuracy", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(accuracy_labels, fontsize=10)
    ax2.set_title("GMM: Inference Quality", fontweight="bold")
    ax2.set_ylim(0, 1.2)
    clean_axes(ax2)

    save_fig(fig, os.path.join(FIGS, "fig_gmm_comparison.pdf"))


# =============================================================================
# Figure 14: Funnel Posterior Recovery
# =============================================================================

def fig_funnel():
    """Bar chart: sampler comparison on Neal's funnel."""
    algorithms = ["NUTS\n(3000)", "HMC\n(2000)", "MALA\n(2000)", "Compiled MH\n(5000)"]
    v_means = [0.849, 0.643, -1.503, -1.750]
    v_stds = [2.327, 2.114, 2.493, 4.534]
    rhats = [1.595, 3.308, 17.291, 4.604]
    times = [202849, 45290, 3443, 2731]
    colors_list = [COLORS["nuts"], COLORS["hmc"], "#CC3311", COLORS["compiled_mh"]]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=SIZES["three_panel"])
    x = np.arange(len(algorithms))

    # Panel 1: v posterior mean (truth = 0)
    ax1.bar(x, v_means, yerr=v_stds, capsize=5, color=colors_list, **BAR)
    ax1.axhline(y=0, color=COLORS["truth"], **LINE["truth"], label="Ground truth (v=0)")
    ax1.set_ylabel("v posterior mean", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, fontsize=10)
    ax1.set_title("Posterior Recovery", fontweight="bold")
    ax1.legend(fontsize=10)
    clean_axes(ax1)

    # Panel 2: R-hat (lower is better, target 1.0)
    bars2 = ax2.bar(x, rhats, color=colors_list, **BAR)
    ax2.axhline(y=1.0, color=COLORS["truth"], **LINE["truth"], label="R-hat = 1.0 (converged)")
    ax2.set_ylabel("R-hat", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, fontsize=10)
    ax2.set_title("Convergence (R-hat)", fontweight="bold")
    ax2.legend(fontsize=10, loc="upper left")
    clean_axes(ax2)
    smaller_is_better(ax2)

    for i, r in enumerate(rhats):
        ax2.text(i, r + 0.3, f"{r:.1f}", ha="center", fontsize=10, fontweight="bold")

    # Panel 3: Time
    bars3 = ax3.bar(x, [t/1000 for t in times], color=colors_list, **BAR)
    ax3.set_ylabel("Time (seconds)", fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(algorithms, fontsize=10)
    ax3.set_title("Wall-Clock Time", fontweight="bold")
    clean_axes(ax3)
    smaller_is_better(ax3)

    for i, t in enumerate(times):
        ax3.text(i, t/1000 + 5, f"{t/1000:.0f}s", ha="center", fontsize=10, fontweight="bold")

    save_fig(fig, os.path.join(FIGS, "fig_funnel.pdf"))


# =============================================================================
# Figure 15: Changepoint Detection
# =============================================================================

def fig_changepoint():
    """Bar chart: SMC vs IS for changepoint model."""
    algorithms = ["SMC\n(N=100)", "SMC\n(N=250)", "SMC\n(N=500)", "IS\n(N=1000)"]
    errors = [51.56, 7.93, 5.33, 263.92]
    times = [55993, 131209, 172645, 95148]
    colors_list = [COLORS["smc"], COLORS["smc"], COLORS["smc"], "#CC3311"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SIZES["two_panel"])
    x = np.arange(len(algorithms))

    # Panel 1: Error
    bars = ax1.bar(x, errors, color=colors_list, **BAR)
    ax1.set_ylabel("|log-ML error| (nats)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, fontsize=13)
    ax1.set_title("Changepoint: Inference Accuracy", fontweight="bold")
    clean_axes(ax1)
    smaller_is_better(ax1)

    for i, e in enumerate(errors):
        ax1.text(i, e + 5, f"{e:.1f}", ha="center", fontsize=11, fontweight="bold")

    # Panel 2: Time
    bars2 = ax2.bar(x, [t/1000 for t in times], color=colors_list, **BAR)
    ax2.set_ylabel("Time (seconds)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, fontsize=13)
    ax2.set_title("Changepoint: Wall-Clock Time", fontweight="bold")
    clean_axes(ax2)

    save_fig(fig, os.path.join(FIGS, "fig_changepoint.pdf"))


# =============================================================================
# Figure 16: LinReg Algorithm Comparison
# =============================================================================

def fig_linreg_algorithms():
    """Bar chart: time and accuracy for all linreg algorithms."""
    algorithms = ["Compiled\nMH", "Multi-chain\nMH", "HMC", "NUTS", "ADVI", "VIS\n(12K)"]
    times = [4841, 2045, 34208, 216017, 13238, 8]
    slope_errors = [0.0001, 0.0031, 0.0002, 0.0023, 0.062, 0.011]
    colors_list = [COLORS["compiled_mh"], COLORS["genmlx_vec"], COLORS["hmc"],
                   COLORS["nuts"], COLORS["advi"], COLORS["vec_is"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SIZES["two_panel"])
    x = np.arange(len(algorithms))

    # Panel 1: Time
    bars = ax1.bar(x, times, color=colors_list, **BAR)
    ax1.set_yscale("log")
    ax1.set_ylabel("Time (ms)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, fontsize=11)
    ax1.set_title("LinReg: Inference Time", fontweight="bold")
    clean_axes(ax1, grid=True)
    smaller_is_better(ax1)

    # VIS callout
    ax1.annotate("8ms!", xy=(5, 8), xytext=(4.3, 100),
                 fontsize=14, fontweight="bold", color=COLORS["vec_is"],
                 arrowprops=dict(arrowstyle="->", color=COLORS["vec_is"], lw=2))

    # Panel 2: Accuracy
    bars2 = ax2.bar(x, slope_errors, color=colors_list, **BAR)
    ax2.set_yscale("log")
    ax2.set_ylabel("|slope error|", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, fontsize=11)
    ax2.set_title("LinReg: Parameter Accuracy", fontweight="bold")
    clean_axes(ax2, grid=True)
    smaller_is_better(ax2)

    save_fig(fig, os.path.join(FIGS, "fig_linreg_algorithms.pdf"))


# =============================================================================
# Figure 17: Equivalent Particle Count (L3 vs L2)
# =============================================================================

def fig_equivalent_particles():
    """Shows how many L2 particles are needed to match L3 accuracy."""
    # Data from L3 evaluation benchmark C
    particles_10 = [50, 200, 500]
    stds_10 = [0.4704, 0.2999, 0.1709]
    within_10 = [2, 4, 6]  # out of 10

    particles_50 = [50, 200, 500]
    stds_50 = [1.8457, 0.5298, 0.2981]
    within_50 = [2, 1, 2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SIZES["two_panel"])

    # Panel 1: n=10 observations
    ax1.plot(particles_10, stds_10, "o-", color=COLORS["is"], markersize=10,
             label="L2 IS", **LINE["main"])
    ax1.axhline(y=0, color=COLORS["nuts"], **LINE["truth"], label="L3 (exact, std=0)")
    ax1.axhline(y=0.1, color="gray", **LINE["baseline"])
    ax1.text(50, 0.12, "Target: std < 0.1", fontsize=10, color="gray")
    ax1.set_xlabel("Number of IS particles", fontweight="bold")
    ax1.set_ylabel("log-ML standard deviation", fontweight="bold")
    ax1.set_title("N=10 observations", fontweight="bold")
    ax1.legend(fontsize=12)
    clean_axes(ax1, grid=True)

    # Annotate within-threshold counts
    for p, s, w in zip(particles_10, stds_10, within_10):
        ax1.text(p, s + 0.02, f"{w}/10 within 0.1", fontsize=9, ha="center")

    # Panel 2: n=50 observations
    ax2.plot(particles_50, stds_50, "o-", color=COLORS["is"], markersize=10,
             label="L2 IS", **LINE["main"])
    ax2.axhline(y=0, color=COLORS["nuts"], **LINE["truth"], label="L3 (exact, std=0)")
    ax2.axhline(y=0.1, color="gray", **LINE["baseline"])
    ax2.text(50, 0.12, "Target: std < 0.1", fontsize=10, color="gray")
    ax2.set_xlabel("Number of IS particles", fontweight="bold")
    ax2.set_ylabel("log-ML standard deviation", fontweight="bold")
    ax2.set_title("N=50 observations", fontweight="bold")
    ax2.legend(fontsize=12)
    clean_axes(ax2, grid=True)

    for p, s, w in zip(particles_50, stds_50, within_50):
        ax2.text(p, s + 0.05, f"{w}/10 within 0.1", fontsize=9, ha="center")

    save_fig(fig, os.path.join(FIGS, "fig_equivalent_particles.pdf"))


# =============================================================================
# Figure 18: Hero Summary (Key Speedups)
# =============================================================================

def fig_hero_speedups():
    """Horizontal bar chart: all key speedup numbers."""
    metrics = [
        "Vectorized IS\n(seq → batched)",
        "Vectorized IS\n(HMM)",
        "Vectorized IS\n(GMM)",
        "Compiled Adam\n(handler → fused)",
        "Batched SMC\n(seq → batched)",
        "Score-fn\n(compiled)",
        "log-ML variance\n(L3 conjugacy)",
        "Compiled MH\n(handler → compiled)",
        "Vectorized MH\n(serial → 10 chains)",
        "ESS improvement\n(L3 vs L2)",
    ]
    speedups = [7813, 759, 605, 78.9, 75.7, 40.6, 33.5, 15.8, 14.1, 7.2]

    colors_list = [
        COLORS["vec_is"], COLORS["vec_is"], COLORS["vec_is"],
        COLORS["genmlx"], COLORS["smc"], COLORS["genmlx"],
        COLORS["nuts"], COLORS["compiled_mh"], COLORS["genmlx_vec"],
        COLORS["nuts"],
    ]

    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(metrics))
    bars = ax.barh(y, speedups, color=colors_list, **BAR)

    ax.set_xscale("log")
    ax.set_xlabel("Speedup Factor (x)", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=12)
    ax.set_title("GenMLX Key Performance Gains", fontweight="bold", fontsize=18)
    clean_axes(ax, grid=True)

    for i, s in enumerate(speedups):
        label = f"{s:,.0f}x" if s >= 10 else f"{s:.1f}x"
        ax.text(s * 1.2, i, label, va="center", fontsize=11, fontweight="bold")

    save_fig(fig, os.path.join(FIGS, "fig_hero_speedups.pdf"))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    setup()
    print("Generating all paper figures...\n")

    figures = [
        ("fig_ladder", fig_compilation_ladder),
        ("fig_vectorization", fig_vectorization_speedup),
        ("fig_cross_system_is", fig_cross_system_is),
        ("fig_particle_scaling", fig_particle_scaling),
        ("fig_hmm_comparison", fig_hmm_comparison),
        ("fig_conjugacy", fig_conjugacy),
        ("fig_rao_blackwell", fig_rao_blackwell),
        ("fig_compiled_adam", fig_compiled_adam),
        ("fig_method_selection", fig_method_selection),
        ("fig_compiled_inference", fig_compiled_inference),
        ("fig_verification", fig_verification),
        ("fig_cross_system_full", fig_cross_system_full),
        ("fig_gmm_comparison", fig_gmm_comparison),
        ("fig_funnel", fig_funnel),
        ("fig_changepoint", fig_changepoint),
        ("fig_linreg_algorithms", fig_linreg_algorithms),
        ("fig_equivalent_particles", fig_equivalent_particles),
        ("fig_hero_speedups", fig_hero_speedups),
    ]

    generated = 0
    failed = 0
    for name, func in figures:
        try:
            func()
            generated += 1
        except Exception as e:
            print(f"FAILED: {name} — {e}")
            failed += 1

    print(f"\nDone: {generated} figures generated, {failed} failed")
    print(f"Output: {FIGS}/")
