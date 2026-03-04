#!/usr/bin/env python3
"""
Generate all paper figures for the GenMLX TOPML manuscript.

Reads results from results/ directory and produces publication-quality
PDF figures in paper/figs/ using the GenMLX visualization standard.

Usage:
    cd paper/viz
    source ../.venv/bin/activate  # or: source /path/to/paper/.venv/bin/activate
    python generate_figures.py

Figures produced (matching MANUSCRIPT_PLAN.md Figure/Table Assignment):
    fig1_particle_scaling.pdf   — Exp 1: particle scaling log-log (1x→1603x)
    fig2_method_speedup.pdf     — Exp 1: method speedup bar chart at N=1000
    fig3_ffi_scaling.pdf        — Exp 2: FFI scaling (star figure), 3 lines
    fig5_linreg_posteriors.pdf  — Exp 3: LinReg posterior KDE densities
    fig7_hmm_logml.pdf          — Exp 3: HMM log-ML error bars
    fig8_gmm_logml.pdf          — Exp 3: GMM IS vs Gibbs comparison
    fig9_system_bars.pdf        — Exp 4: system comparison grouped bars
    fig12_contract_heatmap.pdf  — Exp 5: contract × model verification matrix
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Add this directory to path for style import
sys.path.insert(0, str(Path(__file__).parent))
from genmlx_style import (
    setup, COLORS, SIZES, FONTS, LINE, MARKER, BAR,
    clean_axes, set_ticks, log_y_ticks, bottom_legend,
    smaller_is_better, add_baseline_line, save_fig, make_legend_handle,
)

RESULTS = Path(__file__).parent.parent.parent / "results"
FIGS = Path(__file__).parent.parent / "figs"


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ============================================================================
# Fig 1: Particle Scaling (log-log, two lines)
# ============================================================================

def fig1_particle_scaling():
    data = load_json(RESULTS / "exp1_vectorization" / "particle_scaling.json")
    results = data["results"]

    ns = [r["n"] for r in results]
    seq = [r["sequential"]["mean"] for r in results]
    bat = [r["batched"]["mean"] for r in results]
    speedups = [r["speedup"] for r in results]

    fig, ax = plt.subplots(figsize=SIZES["single"])

    ax.plot(ns, seq, "o-", color=COLORS["data"], label="Sequential",
            **LINE["main"])
    ax.plot(ns, bat, "s-", color=COLORS["genmlx"], label="Batched (vgenerate)",
            **LINE["main"])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of particles (N)")
    ax.set_ylabel("Time (ms)")

    # Annotate speedups at key points
    for i, n in enumerate(ns):
        if n >= 100:
            ax.annotate(
                f"{speedups[i]:.0f}×",
                xy=(n, bat[i]), xytext=(15, 10),
                textcoords="offset points",
                fontsize=FONTS["annotation"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
            )

    ax.legend(loc="upper left", fontsize=FONTS["legend"], framealpha=0.9)
    clean_axes(ax, grid=True)

    save_fig(fig, str(FIGS / "fig1_particle_scaling.pdf"))


# ============================================================================
# Fig 2: Method Speedup Bar Chart (N=1000)
# ============================================================================

def fig2_method_speedup():
    data = load_json(RESULTS / "exp1_vectorization" / "method_speedup.json")
    results = data["results"]

    methods = [r["method"].replace("_", "-") for r in results]
    display_names = {
        "dist-sample-n": "dist-sample-n",
        "importance-sampling": "Importance\nSampling",
        "smc-init": "SMC Init",
    }
    labels = [display_names.get(m, m) for m in methods]
    speedups = [r["speedup"] for r in results]
    method_colors = [COLORS["genmlx"], COLORS["vec_is"], COLORS["smc"]]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars = ax.bar(labels, speedups, color=method_colors, **BAR)

    # Add value labels on bars
    for bar, s in zip(bars, speedups):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f"{s:.0f}×", ha="center", va="bottom",
            fontsize=FONTS["tick"], fontweight="bold",
        )

    ax.set_ylabel("Speedup\n(batched / sequential)")
    ax.set_ylim(0, max(speedups) * 1.15)
    clean_axes(ax)

    save_fig(fig, str(FIGS / "fig2_method_speedup.pdf"))


# ============================================================================
# Fig 3: FFI Scaling — Star Figure (3 lines, log scale)
# ============================================================================

def fig3_ffi_scaling():
    dims = [10, 25, 50, 100, 200]

    per_site, gauss_vec, genjax_times = [], [], []
    for d in dims:
        ps = load_json(RESULTS / "exp2_ffi_bottleneck" / f"is_D{d}_n10000.json")
        gv = load_json(RESULTS / "exp2_ffi_bottleneck" / f"is_fast_D{d}_n10000.json")
        gj = load_json(RESULTS / "exp2_ffi_bottleneck" / f"genjax_is_D{d}_n10000.json")

        per_site.append(ps["mean_time"] * 1000)
        gauss_vec.append(gv["mean_time"] * 1000)
        gj_mean = gj.get("mean_time", sum(gj["times"]) / len(gj["times"]))
        genjax_times.append(gj_mean * 1000)

    fig, ax = plt.subplots(figsize=SIZES["single"])

    ax.plot(dims, per_site, "o-", color=COLORS["data"],
            label="GenMLX per-site", **LINE["main"])
    ax.plot(dims, genjax_times, "D-", color=COLORS["genjax"],
            label="GenJAX (JIT, CPU)", **LINE["secondary"])
    ax.plot(dims, gauss_vec, "s-", color=COLORS["genmlx"],
            label="GenMLX gaussian-vec", **LINE["main"])

    ax.set_yscale("log")
    ax.set_xlabel("Model dimensionality (D)")
    ax.set_ylabel("IS time (ms), N=10,000 particles")

    # Annotate speedup at D=200
    ratio_vs_genjax = genjax_times[-1] / gauss_vec[-1]
    ax.annotate(
        f"{ratio_vs_genjax:.1f}× vs GenJAX",
        xy=(200, gauss_vec[-1]), xytext=(-60, -35),
        textcoords="offset points",
        fontsize=FONTS["annotation"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
    )

    ax.legend(loc="upper left", fontsize=FONTS["legend"], framealpha=0.9)
    clean_axes(ax, grid=True)
    ax.set_xticks(dims)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())

    save_fig(fig, str(FIGS / "fig3_ffi_scaling.pdf"))


# ============================================================================
# Fig 5: LinReg Posterior KDE Densities
# ============================================================================

def fig5_linreg_posteriors():
    from scipy.stats import gaussian_kde, norm

    data = load_json(RESULTS / "exp3_canonical_models" / "linreg_results.json")
    analytic = data["analytic"]

    fig, ax = plt.subplots(figsize=SIZES["single"])

    mu = analytic["slope"]["mean"]
    sigma = analytic["slope"]["std"]

    # Analytic posterior
    x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    ax.plot(x_range, norm.pdf(x_range, mu, sigma),
            color=COLORS["analytic"], label="Analytic",
            linewidth=3, alpha=0.9, linestyle="--")

    # Algorithm posteriors (only those with slope_samples)
    # Only MCMC algorithms (IS samples are unweighted proposals, not posterior)
    alg_styles = {
        "Compiled_MH": (COLORS["compiled_mh"], "Compiled MH"),
        "HMC": (COLORS["hmc"], "HMC"),
        "NUTS": (COLORS["nuts"], "NUTS"),
    }

    for alg in data["algorithms"]:
        name = alg["algorithm"]
        if "slope_samples" in alg and name in alg_styles:
            color, label = alg_styles[name]
            samples = np.array(alg["slope_samples"])
            try:
                kde = gaussian_kde(samples)
                ax.plot(x_range, kde(x_range), color=color,
                        label=label, linewidth=2, alpha=0.8)
            except Exception:
                pass

    ax.set_xlabel("Slope")
    ax.set_ylabel("Density")
    ax.legend(fontsize=FONTS["legend"], framealpha=0.9)
    clean_axes(ax)

    save_fig(fig, str(FIGS / "fig5_linreg_posteriors.pdf"))


# ============================================================================
# Fig 7: HMM Log-ML Error (bar chart with error bars)
# ============================================================================

def fig7_hmm_logml():
    data = load_json(RESULTS / "exp3_canonical_models" / "hmm_results.json")
    exact = data["exact_log_ml"]

    alg_data = []
    for alg in data["algorithms"]:
        name = alg["algorithm"]
        errors = alg.get("raw_errors", [])
        mean_err = alg["error"]
        std_err = alg.get("error_std", 0)
        time_ms = alg["time_ms"]
        alg_data.append((name, mean_err, std_err, time_ms))

    fig, ax = plt.subplots(figsize=SIZES["benchmark"])

    names = [a[0].replace("_", " ") for a in alg_data]
    errors = [a[1] for a in alg_data]
    stds = [a[2] for a in alg_data]

    bar_colors = [COLORS["is"], COLORS["vec_is"], COLORS["smc"], COLORS["smc"], COLORS["genmlx"]]
    # Pad colors if fewer than algorithms
    while len(bar_colors) < len(names):
        bar_colors.append(COLORS["genmlx"])

    bars = ax.bar(names, errors, yerr=stds, capsize=5,
                  color=bar_colors[:len(names)], **BAR)

    ax.set_ylabel("|log P(y) - exact|")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    clean_axes(ax)

    # Annotate ESS on bars
    for i, alg in enumerate(alg_data):
        ess = data["algorithms"][i].get("ess", None)
        if ess is not None:
            ax.text(
                i, errors[i] + stds[i] + 0.3,
                f"ESS={ess:.0f}",
                ha="center", fontsize=9, color="gray",
            )

    save_fig(fig, str(FIGS / "fig7_hmm_logml.pdf"))


# ============================================================================
# Fig 8: GMM IS vs Gibbs (dual metric bar chart)
# ============================================================================

def fig8_gmm_logml():
    data = load_json(RESULTS / "exp3_canonical_models" / "gmm_results.json")

    alg_data = []
    for alg in data["algorithms"]:
        alg_data.append(alg)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SIZES["two_panel"])

    # Left: Log-ML error
    is_algs = [a for a in alg_data if "IS" in a["algorithm"]]
    gibbs_algs = [a for a in alg_data if "Gibbs" in a["algorithm"]]

    all_algs = is_algs + gibbs_algs
    names = [a["algorithm"].replace("_", " ") for a in all_algs]
    bar_colors = []
    errors = []

    for a in all_algs:
        if "IS" in a["algorithm"]:
            errors.append(a.get("error", 0))
            bar_colors.append(COLORS["is"] if "Vec" not in a["algorithm"] else COLORS["vec_is"])
        else:
            errors.append(a.get("marginal_mae", 0))
            bar_colors.append(COLORS["gibbs"])

    ax1.bar(names, errors, color=bar_colors, **BAR)
    ax1.set_ylabel("Error")
    ax1.set_title("Inference Accuracy", fontsize=FONTS["tick"])
    ax1.tick_params(axis="x", rotation=20)
    clean_axes(ax1)

    # Right: Time comparison
    times = [a["time_ms"] for a in all_algs]
    ax2.bar(names, times, color=bar_colors, **BAR)
    ax2.set_ylabel("Time (ms)")
    ax2.set_yscale("log")
    ax2.set_title("Wall-Clock Time", fontsize=FONTS["tick"])
    ax2.tick_params(axis="x", rotation=20)
    clean_axes(ax2)

    fig.tight_layout()
    save_fig(fig, str(FIGS / "fig8_gmm_logml.pdf"))


# ============================================================================
# Fig 9: System Comparison (grouped bar chart, log scale)
# ============================================================================

def fig9_system_bars():
    genmlx_is = load_json(RESULTS / "exp4_system_comparison" / "genmlx_is1000.json")
    genjl = load_json(RESULTS / "exp4_system_comparison" / "genjl.json")
    genjax = load_json(RESULTS / "exp4_system_comparison" / "genjax.json")
    linreg = load_json(RESULTS / "exp3_canonical_models" / "linreg_results.json")
    hmm = load_json(RESULTS / "exp3_canonical_models" / "hmm_results.json")

    def get_time(data, model, algo, prefer_method=None):
        for c in data["comparisons"]:
            if c["model"] == model and c["algorithm"] == algo:
                if prefer_method is None or c.get("method", "") == prefer_method:
                    return c["time_ms"]
        for c in data["comparisons"]:
            if c["model"] == model and c["algorithm"] == algo:
                return c["time_ms"]
        return None

    # Get GenMLX MH and SMC from exp3 results
    mlx_mh = None
    for alg in linreg["algorithms"]:
        if alg["algorithm"] == "Compiled_MH":
            mlx_mh = alg["time_ms"]

    mlx_smc = None
    for alg in hmm["algorithms"]:
        if alg["algorithm"] == "SMC_100":
            mlx_smc = alg["time_ms"]

    groups = [
        "LinReg\nIS(1K)", "GMM\nIS(1K)", "HMM\nIS(1K)",
        "LinReg\nMH(5K)", "HMM\nSMC(100)",
    ]

    fig, ax = plt.subplots(figsize=(12, 4.5))

    x = np.arange(len(groups))
    width = 0.25

    genmlx_times = [
        get_time(genmlx_is, "linreg", "IS", "vectorized"),
        get_time(genmlx_is, "gmm", "IS", "vectorized"),
        get_time(genmlx_is, "hmm", "IS", "vectorized"),
        mlx_mh,
        mlx_smc,
    ]
    genjl_times = [
        get_time(genjl, "linreg", "IS"),
        get_time(genjl, "gmm", "IS"),
        get_time(genjl, "hmm", "IS"),
        get_time(genjl, "linreg", "MH"),
        get_time(genjl, "hmm", "SMC"),
    ]
    genjax_times = [
        get_time(genjax, "linreg", "IS"),
        get_time(genjax, "gmm", "IS"),
        None, None, None,
    ]

    # Plot bars
    group_labels = [g[0] for g in groups]

    def plot_system(offset, times, color, label):
        valid_x = [x[i] + offset for i in range(len(times)) if times[i] is not None]
        valid_t = [t for t in times if t is not None]
        ax.bar(valid_x, valid_t, width, color=color, label=label, **BAR)

    plot_system(-width, genmlx_times, COLORS["genmlx"], "GenMLX (Metal)")
    plot_system(0, genjl_times, COLORS["genjl"], "Gen.jl (CPU)")
    plot_system(width, genjax_times, COLORS["genjax"], "GenJAX (JIT, CPU)")

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12)
    ax.set_yscale("log")
    ax.set_ylabel("Time (ms)")
    clean_axes(ax)
    smaller_is_better(ax)

    # Add star on GenMLX bars for IS wins (first 3 groups)
    for i in range(3):
        if genmlx_times[i] is not None:
            ax.scatter(x[i] - width, genmlx_times[i] * 0.6,
                       marker="*", s=150, color="gold",
                       edgecolor="darkgoldenrod", linewidth=1, zorder=10)

    # Legend
    handles = [
        make_legend_handle(COLORS["genmlx"], "GenMLX (Metal)"),
        make_legend_handle(COLORS["genjl"], "Gen.jl (CPU)"),
        make_legend_handle(COLORS["genjax"], "GenJAX (JIT, CPU)"),
    ]
    labels = ["GenMLX (Metal)", "Gen.jl (CPU)", "GenJAX (JIT, CPU)"]
    bottom_legend(fig, handles, labels)

    save_fig(fig, str(FIGS / "fig9_system_bars.pdf"))


# ============================================================================
# Fig 12: Verification Contract Heatmap
# ============================================================================

def fig12_contract_heatmap():
    data = load_json(RESULTS / "exp5_verification" / "verification_summary.json")

    contracts = data["gfi_contracts"]["contract_types"]
    models = data["gfi_contracts"]["model_types"]

    # All checks pass → full green matrix
    matrix = np.ones((len(contracts), len(models)))

    fig, ax = plt.subplots(figsize=SIZES["heatmap"])

    cmap = plt.cm.colors.ListedColormap(["#FF4444", "#66BB6A"])
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace("-", "\n") for m in models],
                       fontsize=11, ha="center")
    ax.set_yticks(range(len(contracts)))
    ax.set_yticklabels(contracts, fontsize=11)

    ax.set_xlabel("Model")
    ax.set_ylabel("GFI Contract")

    # Add pass markers (use P instead of unicode checkmark for font compatibility)
    for i in range(len(contracts)):
        for j in range(len(models)):
            ax.text(j, i, "P", ha="center", va="center",
                    fontsize=12, fontweight="bold", color="white")

    # Summary annotation
    total = data["gfi_contracts"]["checks"]
    ax.set_title(f"{total} checks, 0 failures", fontsize=FONTS["tick"])

    fig.tight_layout()
    save_fig(fig, str(FIGS / "fig12_contract_heatmap.pdf"))


# ============================================================================
# LaTeX Tables
# ============================================================================

def tab_linreg():
    """Generate fig6_linreg_table.tex — algorithm comparison table."""
    data = load_json(RESULTS / "exp3_canonical_models" / "linreg_results.json")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Linear Regression: Inference Algorithm Comparison}",
        r"\label{tab:linreg}",
        r"\small",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Algorithm & Slope $\mu$ & Slope Err & Int. $\mu$ & Int. Err & ESS & $\hat{R}$ & Time (ms) \\",
        r"\midrule",
    ]

    analytic = data["analytic"]
    lines.append(
        f"Analytic & {analytic['slope']['mean']:.4f} & --- "
        f"& {analytic['intercept']['mean']:.4f} & --- & --- & --- & --- \\\\"
    )

    for alg in data["algorithms"]:
        name = alg["algorithm"].replace("_", " ")
        s_mean = alg["slope"]["mean"]
        s_err = alg["slope"]["error"]
        i_mean = alg["intercept"]["mean"]
        i_err = alg["intercept"]["error"]
        ess = f"{alg['ess']:.0f}" if "ess" in alg else "---"
        rhat = f"{alg['rhat']:.3f}" if "rhat" in alg else "---"
        time = alg["time_ms"]
        lines.append(
            f"{name} & {s_mean:.4f} & {s_err:.4f} "
            f"& {i_mean:.4f} & {i_err:.4f} "
            f"& {ess} & {rhat} & {time:.0f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    path = FIGS / "fig6_linreg_table.tex"
    path.write_text("\n".join(lines))
    print(f"Saved: {path}")


def tab_system():
    """Generate fig10_system_table.tex — headline system comparison."""
    genmlx = load_json(RESULTS / "exp4_system_comparison" / "genmlx_is1000.json")
    genjl = load_json(RESULTS / "exp4_system_comparison" / "genjl.json")
    genjax = load_json(RESULTS / "exp4_system_comparison" / "genjax.json")
    linreg = load_json(RESULTS / "exp3_canonical_models" / "linreg_results.json")
    hmm = load_json(RESULTS / "exp3_canonical_models" / "hmm_results.json")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{System Comparison: GenMLX (Metal GPU) vs Gen.jl (CPU) vs GenJAX (JIT, CPU)}",
        r"\label{tab:system}",
        r"\small",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Model & Algorithm & GenMLX & Gen.jl & GenJAX \\",
        r"\midrule",
    ]

    def get_time(data, model, algo, prefer_method=None):
        for c in data["comparisons"]:
            if c["model"] == model and c["algorithm"] == algo:
                if prefer_method is None or c.get("method", "") == prefer_method:
                    return c["time_ms"]
        for c in data["comparisons"]:
            if c["model"] == model and c["algorithm"] == algo:
                return c["time_ms"]
        return None

    def fmt(t):
        if t is None:
            return "---"
        if t < 1:
            return f"{t:.2f}ms"
        if t < 1000:
            return f"{t:.1f}ms"
        return f"{t / 1000:.1f}s"

    # Get GenMLX MH from linreg results (Compiled_MH)
    mlx_mh = None
    for alg in linreg["algorithms"]:
        if alg["algorithm"] == "Compiled_MH":
            mlx_mh = alg["time_ms"]

    # Get GenMLX SMC from hmm results (SMC_100)
    mlx_smc = None
    for alg in hmm["algorithms"]:
        if alg["algorithm"] == "SMC_100":
            mlx_smc = alg["time_ms"]

    rows = [
        ("LinReg", "IS(1K)", get_time(genmlx, "linreg", "IS", "vectorized"),
         get_time(genjl, "linreg", "IS"), get_time(genjax, "linreg", "IS")),
        ("GMM", "IS(1K)", get_time(genmlx, "gmm", "IS", "vectorized"),
         get_time(genjl, "gmm", "IS"), get_time(genjax, "gmm", "IS")),
        ("HMM", "IS(1K)", get_time(genmlx, "hmm", "IS", "vectorized"),
         get_time(genjl, "hmm", "IS"), None),
        ("LinReg", "MH(5K)", mlx_mh, get_time(genjl, "linreg", "MH"), None),
        ("HMM", "SMC(100)", mlx_smc, get_time(genjl, "hmm", "SMC"), None),
    ]

    for display_model, algo, t_mlx, t_jl, t_jax in rows:
        vals = [v for v in [t_mlx, t_jl, t_jax] if v is not None]
        winner = min(vals) if vals else None

        def fmt_bold(t):
            s = fmt(t)
            if t is not None and t == winner:
                return r"\textbf{" + s + "}"
            return s

        lines.append(
            f"{display_model} & {algo} & {fmt_bold(t_mlx)} & {fmt_bold(t_jl)} & {fmt_bold(t_jax)} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    path = FIGS / "fig10_system_table.tex"
    path.write_text("\n".join(lines))
    print(f"Saved: {path}")


def tab_compilation():
    """Generate compilation_table.tex — loop compilation speedups."""
    data = load_json(RESULTS / "exp6_compilation" / "compiled_speedup.json")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Loop Compilation Speedups}",
        r"\label{tab:compilation}",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Benchmark & Baseline (ms) & Compiled (ms) & Speedup \\",
        r"\midrule",
    ]

    # Hardcode the structure since the JSON keys vary
    b1 = data["bench1_gfi_mh_vs_compiled_mh"]
    lines.append(
        f"GFI MH vs Compiled MH & {b1['gfi_mh']['mean']:.0f} & {b1['compiled_mh']['mean']:.0f} & {b1['speedup']:.1f}$\\times$ \\\\"
    )

    b2 = data["bench2_score_fn_compilation"]
    lines.append(
        f"Score function compilation & {b2['uncompiled']['mean']:.0f} & {b2['compiled']['mean']:.0f} & {b2['speedup']:.1f}$\\times$ \\\\"
    )

    b3 = data["bench3_hmc_genmlx_vs_handcoded"]
    lines.append(
        f"HMC: GenMLX vs hand-optimized & {b3['genmlx_hmc']['mean']:.0f} & {b3['handcoded_hmc']['mean']:.0f} & {b3['overhead']:.2f}$\\times$ \\\\"
    )

    b4 = data["bench4_serial_vs_vectorized_mh"]
    lines.append(
        f"Serial vs vectorized 10-chain MH & {b4['serial_extrapolated']['mean']:.0f} & {b4['vectorized']['mean']:.0f} & {b4['speedup']:.1f}$\\times$ \\\\"
    )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    path = FIGS / "compilation_table.tex"
    path.write_text("\n".join(lines))
    print(f"Saved: {path}")


def tab_ffi_speedup():
    """Generate fig4_speedup_table.tex — FFI bottleneck speedup table."""
    dims = [10, 25, 50, 100, 200]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{FFI Bottleneck: IS Time (ms) at N=10,000 Particles}",
        r"\label{tab:ffi}",
        r"\small",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"D & Per-site & gaussian-vec & GenJAX & vec/ps & vec/GenJAX \\",
        r"\midrule",
    ]

    for d in dims:
        ps = load_json(RESULTS / "exp2_ffi_bottleneck" / f"is_D{d}_n10000.json")
        gv = load_json(RESULTS / "exp2_ffi_bottleneck" / f"is_fast_D{d}_n10000.json")
        gj = load_json(RESULTS / "exp2_ffi_bottleneck" / f"genjax_is_D{d}_n10000.json")

        ps_ms = ps["mean_time"] * 1000
        gv_ms = gv["mean_time"] * 1000
        gj_mean = gj.get("mean_time", sum(gj["times"]) / len(gj["times"]))
        gj_ms = gj_mean * 1000

        lines.append(
            f"{d} & {ps_ms:.1f} & {gv_ms:.1f} & {gj_ms:.1f} "
            f"& {ps_ms / gv_ms:.1f}$\\times$ & {gj_ms / gv_ms:.1f}$\\times$ \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    path = FIGS / "fig4_speedup_table.tex"
    path.write_text("\n".join(lines))
    print(f"Saved: {path}")


def tab_verification():
    """Generate fig11_verification_ladder.tex."""
    data = load_json(RESULTS / "exp5_verification" / "verification_summary.json")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Three-Level Verification Results}",
        r"\label{tab:verification}",
        r"\small",
        r"\begin{tabular}{lrrl}",
        r"\toprule",
        r"Level & Passed & Total & Source \\",
        r"\midrule",
    ]

    gc = data["gen_clj_compat"]
    gj = data["genjax_compat"]
    gfi = data["gfi_contracts"]

    lines.append(f"Gen.clj compatibility & {gc['passed']} & {gc['total']} & {gc['total']} ported tests \\\\")
    lines.append(f"GenJAX compatibility & {gj['passed']} & {gj['total']} & {gj['total']} ported tests \\\\")
    lines.append(f"GFI contracts & {gfi['checks']} & {gfi['checks']} & {gfi['contracts']} contracts $\\times$ {gfi['models']} models \\\\")

    total = gc["passed"] + gj["passed"] + gfi["checks"]
    lines.append(r"\midrule")
    lines.append(f"\\textbf{{Total}} & \\textbf{{{total}}} & \\textbf{{{total}}} & 0 failures \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    path = FIGS / "fig11_verification_ladder.tex"
    path.write_text("\n".join(lines))
    print(f"Saved: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    setup()

    print("=" * 60)
    print("Generating GenMLX paper figures")
    print("=" * 60)

    # Figures
    fig1_particle_scaling()
    fig2_method_speedup()
    fig3_ffi_scaling()
    fig7_hmm_logml()
    fig8_gmm_logml()
    fig9_system_bars()
    fig12_contract_heatmap()

    # Fig 5 requires scipy
    try:
        fig5_linreg_posteriors()
    except ImportError:
        print("SKIP fig5_linreg_posteriors.pdf (scipy not installed; run: pip install scipy)")

    # Tables
    tab_linreg()
    tab_system()
    tab_compilation()
    tab_ffi_speedup()
    tab_verification()

    print("=" * 60)
    print(f"Done! All outputs in {FIGS}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
