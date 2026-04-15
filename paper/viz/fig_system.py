"""
Figures 15-20 for the GenMLX paper: system-level evaluation figures.

Generates:
  fig15_method_spectrum.pdf   — GenMLX method spectrum (bar chart)
  fig16_method_selection.pdf  — Automatic method selection (2-panel)
  fig17_test_dashboard.pdf    — Test suite dashboard (horizontal stacked bars)
  fig18_property_coverage.pdf — Property test coverage (treemap)
  fig19_hero.pdf              — Hero performance landscape (2x2 panels)
  fig20_recommendation.pdf    — Inference recommendation map (table figure)
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))
from genmlx_style import (
    setup, COLORS, SIZES, FONTS, BAR, LINE, LEGEND,
    save_fig, clean_axes, log_y_ticks, smaller_is_better,
    make_legend_handle, bottom_legend,
)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RESULTS = os.path.join(BASE, "results")
FIGS = os.path.join(BASE, "paper", "figs")


def load(path):
    with open(os.path.join(RESULTS, path)) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────
# Figure 15: GenMLX Method Spectrum
# ─────────────────────────────────────────────────────────────────────

def fig15_method_spectrum():
    data = load("cross-system/data.json")
    genjl = load("cross-system/genjl.json")
    genjax = load("cross-system/genjax.json")

    # Organize GenMLX results by config
    genmlx_results = {r["config"]: r for r in data["results"]}

    # Define groups for the chart — tasks we can compare across systems
    # Group 1: IS linreg (1000 particles)
    # Group 2: IS gmm (1000 particles)
    # Group 3: MH linreg
    # Also show GenMLX-only methods

    # Build comparison groups
    groups = []

    # IS LinReg 1K
    genjl_is_linreg = next(
        (c for c in genjl["comparisons"]
         if c["model"] == "linreg" and c["algorithm"] == "IS"), None)
    genjax_is_linreg = next(
        (c for c in genjax["comparisons"]
         if c["model"] == "linreg" and c["algorithm"] == "IS"), None)
    groups.append({
        "label": "IS LinReg\n(1K particles)",
        "genmlx": genmlx_results.get("VIS-linreg", {}).get("mean_ms"),
        "genjl": genjl_is_linreg["time_ms"] if genjl_is_linreg else None,
        "genjax": genjax_is_linreg["time_ms"] if genjax_is_linreg else None,
    })

    # IS GMM 1K
    genjl_is_gmm = next(
        (c for c in genjl["comparisons"]
         if c["model"] == "gmm" and c["algorithm"] == "IS"), None)
    genjax_is_gmm = next(
        (c for c in genjax["comparisons"]
         if c["model"] == "gmm" and c["algorithm"] == "IS"), None)
    groups.append({
        "label": "IS GMM\n(1K particles)",
        "genmlx": genmlx_results.get("VIS-gmm", {}).get("mean_ms"),
        "genjl": genjl_is_gmm["time_ms"] if genjl_is_gmm else None,
        "genjax": genjax_is_gmm["time_ms"] if genjax_is_gmm else None,
    })

    # MH LinReg
    genjl_mh = next(
        (c for c in genjl["comparisons"]
         if c["model"] == "linreg" and c["algorithm"] == "MH"), None)
    groups.append({
        "label": "MH LinReg\n(compiled)",
        "genmlx": genmlx_results.get("fused-MH-linreg", {}).get("mean_ms"),
        "genjl": genjl_mh["time_ms"] if genjl_mh else None,
        "genjax": None,
    })

    # GenMLX-only methods
    groups.append({
        "label": "VIS LinReg\n(10K particles)",
        "genmlx": genmlx_results.get("VIS-linreg", {}).get("mean_ms")
        if "VIS-linreg" in genmlx_results else None,
        "genjl": None,
        "genjax": None,
    })
    # Use the 10K-particle VIS-linreg entry
    vis_10k = next(
        (r for r in data["results"]
         if r["config"] == "VIS-linreg" and r.get("particles") == 10000), None)
    if vis_10k:
        groups[-1]["genmlx"] = vis_10k["mean_ms"]

    groups.append({
        "label": "L3 Exact\nPosterior",
        "genmlx": genmlx_results.get("L3-exact", {}).get("mean_ms"),
        "genjl": None,
        "genjax": None,
    })

    groups.append({
        "label": "HMC\n(100 samples)",
        "genmlx": genmlx_results.get("HMC-linreg", {}).get("mean_ms"),
        "genjl": None,
        "genjax": None,
    })

    groups.append({
        "label": "NUTS\n(100 samples)",
        "genmlx": genmlx_results.get("NUTS-linreg", {}).get("mean_ms"),
        "genjl": None,
        "genjax": None,
    })

    fig, ax = plt.subplots(figsize=SIZES["benchmark"])

    n_groups = len(groups)
    x = np.arange(n_groups)
    width = 0.25

    # Plot bars for each system
    genmlx_vals = []
    genjl_vals = []
    genjax_vals = []
    genmlx_x = []
    genjl_x = []
    genjax_x = []

    for i, g in enumerate(groups):
        if g["genmlx"] is not None:
            genmlx_vals.append(g["genmlx"])
            genmlx_x.append(x[i])
        if g["genjl"] is not None:
            genjl_vals.append(g["genjl"])
            genjl_x.append(x[i])
        if g["genjax"] is not None:
            genjax_vals.append(g["genjax"])
            genjax_x.append(x[i])

    # Offset bars for groups with multiple systems
    for i, g in enumerate(groups):
        n_systems = sum(1 for v in [g["genmlx"], g["genjl"], g["genjax"]]
                        if v is not None)
        if n_systems == 1:
            if g["genmlx"] is not None:
                ax.bar(x[i], g["genmlx"], width, color=COLORS["genmlx"], **BAR)
        elif n_systems == 2:
            off = width / 2
            if g["genmlx"] is not None:
                ax.bar(x[i] - off, g["genmlx"], width,
                       color=COLORS["genmlx"], **BAR)
            if g["genjl"] is not None:
                ax.bar(x[i] + off, g["genjl"], width,
                       color=COLORS["genjl"], **BAR)
        else:  # 3 systems
            if g["genmlx"] is not None:
                ax.bar(x[i] - width, g["genmlx"], width,
                       color=COLORS["genmlx"], **BAR)
            if g["genjl"] is not None:
                ax.bar(x[i], g["genjl"], width,
                       color=COLORS["genjl"], **BAR)
            if g["genjax"] is not None:
                ax.bar(x[i] + width, g["genjax"], width,
                       color=COLORS["genjax"], **BAR)

    ax.set_yscale("log")
    ax.set_ylabel("Time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([g["label"] for g in groups], fontsize=FONTS["bar_label"])
    ax.set_title("GenMLX Method Spectrum", fontsize=FONTS["title"], fontweight="bold")
    smaller_is_better(ax)
    clean_axes(ax, grid=True)

    handles = [
        make_legend_handle(COLORS["genmlx"], "GenMLX"),
        make_legend_handle(COLORS["genjl"], "Gen.jl"),
        make_legend_handle(COLORS["genjax"], "GenJAX"),
    ]
    bottom_legend(fig, handles, ["GenMLX", "Gen.jl", "GenJAX"])

    save_fig(fig, os.path.join(FIGS, "fig15_method_spectrum.pdf"))


# ─────────────────────────────────────────────────────────────────────
# Figure 16: Automatic Method Selection
# ─────────────────────────────────────────────────────────────────────

def fig16_method_selection():
    data = load("method-selection/data.json")
    ms_results = data["method_selection"]["results"]
    fit_results = data["fit_api"]["results"]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=SIZES["two_panel"],
                                      gridspec_kw={"width_ratios": [1.3, 1]})

    # Panel A: Method selection matrix
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(-0.5, len(ms_results) - 0.5)
    ax_a.invert_yaxis()
    ax_a.set_title("A. Method Selection Accuracy", fontsize=FONTS["title"],
                    fontweight="bold", loc="left")

    col_x = [0.0, 3.8, 5.8, 7.8, 9.2]
    headers = ["Model", "Expected", "Actual", "Elim.", "Status"]
    for cx, h in zip(col_x, headers):
        ax_a.text(cx, -0.9, h, fontsize=FONTS["bar_label"], fontweight="bold",
                  ha="left", va="bottom")

    for i, r in enumerate(ms_results):
        # Shorten label
        label = r["label"].replace("(fully conjugate)", "(conj.)")
        label = label.replace("(dynamic addresses)", "(dyn.)")
        label = label.replace("(partial conjugate)", "(part.)")
        label = label.replace("(sub-model call)", "(splice)")
        label = label.replace("(no trace sites)", "(empty)")
        label = label.replace("(15+ latents)", "(large)")

        color = "#e8f5e9" if r["pass"] else "#ffebee"
        ax_a.axhspan(i - 0.4, i + 0.4, color=color, alpha=0.5)

        ax_a.text(col_x[0], i, label, fontsize=FONTS["bar_label"],
                  va="center", ha="left")
        ax_a.text(col_x[1], i, r["expected"], fontsize=FONTS["bar_label"],
                  va="center", ha="left", family="monospace")
        ax_a.text(col_x[2], i, r["actual"], fontsize=FONTS["bar_label"],
                  va="center", ha="left", family="monospace",
                  fontweight="bold")
        elim_str = f"{r['n_eliminated']}/{r['n_eliminated'] + r['n_residual']}"
        ax_a.text(col_x[3], i, elim_str, fontsize=FONTS["bar_label"],
                  va="center", ha="left")
        icon = "PASS" if r["pass"] else "FAIL"
        icon_color = "#2e7d32" if r["pass"] else "#c62828"
        ax_a.text(col_x[4], i, icon, fontsize=FONTS["bar_label"],
                  va="center", ha="left", color=icon_color, fontweight="bold")

    ax_a.text(5.0, len(ms_results) + 0.3,
              f"{data['method_selection']['correct']}/{data['method_selection']['total']} correct",
              fontsize=FONTS["legend"], fontweight="bold", ha="center",
              va="top", color="#2e7d32")
    ax_a.axis("off")

    # Panel B: Fit API timing bars
    models = [r["model"] for r in fit_results]
    times = [r["elapsed_ms"] for r in fit_results]
    methods = [r["method"] for r in fit_results]

    method_colors = {
        "exact": COLORS["genmlx"],
        "handler-is": COLORS["vec_is"],
        "hmc": COLORS["hmc"],
    }

    bars = ax_b.barh(models, times, color=[method_colors.get(m, "gray")
                                            for m in methods],
                     **BAR)

    for bar, method, t in zip(bars, methods, times):
        ax_b.text(bar.get_width() + max(times) * 0.02,
                  bar.get_y() + bar.get_height() / 2,
                  f"{method}\n({t:.0f} ms)", fontsize=FONTS["bar_label"],
                  va="center", ha="left")

    ax_b.set_xscale("log")
    ax_b.set_xlabel("Time (ms)")
    ax_b.set_title("B. fit API End-to-End", fontsize=FONTS["title"],
                    fontweight="bold", loc="left")
    clean_axes(ax_b, grid=True)

    fig.tight_layout(w_pad=3)
    save_fig(fig, os.path.join(FIGS, "fig16_method_selection.pdf"))


# ─────────────────────────────────────────────────────────────────────
# Figure 17: Test Suite Dashboard
# ─────────────────────────────────────────────────────────────────────

def fig17_test_dashboard():
    verification = load("verification-suite/data.json")
    gfi_laws = load("gfi-law-verification/data.json")
    property_tests = load("property-test-stats/data.json")
    dist_audit = load("distribution-audit/data.json")
    mutation = load("mutation-boundary/data.json")

    categories = [
        ("Verification Suite", verification["totals"]),
        ("GFI Law Verification", gfi_laws["totals"]),
        ("Property Tests", property_tests["totals"]),
        ("Distribution Audit", dist_audit["totals"]),
        ("Mutation Boundary", mutation["totals"]),
    ]

    fig, ax = plt.subplots(figsize=SIZES["benchmark"])

    labels = [c[0] for c in categories]
    passes = [c[1]["pass"] for c in categories]
    fails = [c[1]["fail"] for c in categories]
    errors = [c[1].get("error", 0) for c in categories]

    y = np.arange(len(labels))
    h = 0.6

    green = COLORS["genmlx"]
    red = "#e53935"
    orange = "#fb8c00"

    # Stacked horizontal bars
    bars_pass = ax.barh(y, passes, h, color=green, **BAR, label="Pass")
    bars_fail = ax.barh(y, fails, h, left=passes, color=red, **BAR,
                        label="Fail")
    lefts_err = [p + f for p, f in zip(passes, fails)]
    bars_err = ax.barh(y, errors, h, left=lefts_err, color=orange, **BAR,
                       label="Error")

    # Annotate totals
    for i, (p, f, e) in enumerate(zip(passes, fails, errors)):
        total = p + f + e
        ax.text(total + max(p + f + e for p, f, e in
                            zip(passes, fails, errors)) * 0.02,
                i, f"{p}/{total}",
                fontsize=FONTS["bar_label"], va="center", fontweight="bold")

    grand_pass = sum(passes)
    grand_total = sum(p + f + e for p, f, e in zip(passes, fails, errors))
    ax.set_title(
        f"Test Suite Dashboard ({grand_pass:,}/{grand_total:,} passing)",
        fontsize=FONTS["title"], fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FONTS["tick"])
    ax.set_xlabel("Assertions")
    ax.invert_yaxis()
    clean_axes(ax, grid=True)

    handles = [
        make_legend_handle(green, "Pass"),
        make_legend_handle(red, "Fail"),
        make_legend_handle(orange, "Error"),
    ]
    ax.legend(handles=handles, labels=["Pass", "Fail", "Error"],
              loc="lower right", **LEGEND)

    save_fig(fig, os.path.join(FIGS, "fig17_test_dashboard.pdf"))


# ─────────────────────────────────────────────────────────────────────
# Figure 18: Property Test Coverage (treemap)
# ─────────────────────────────────────────────────────────────────────

def fig18_property_coverage():
    import squarify

    data = load("property-test-stats/data.json")
    suites = data["suites"]

    # Clean names
    def clean_name(name):
        return (name.replace("_property_test", "")
                .replace("_", " ")
                .title())

    names = [clean_name(s["name"]) for s in suites]
    sizes = [s["total-assertions"] for s in suites]
    durations = [s["duration-ms"] for s in suites]

    # Color by duration (log scale)
    log_dur = np.log10(np.array(durations) + 1)
    norm_dur = (log_dur - log_dur.min()) / (log_dur.max() - log_dur.min() + 1e-9)

    # Purple gradient from light to dark
    base_rgb = np.array([139, 92, 246]) / 255.0  # COLORS["genmlx"]
    colors = []
    for nd in norm_dur:
        factor = 0.3 + 0.7 * nd  # range 0.3..1.0 intensity
        c = base_rgb * factor + np.array([1, 1, 1]) * (1 - factor)
        colors.append(c)

    fig, ax = plt.subplots(figsize=SIZES["heatmap"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    normed = squarify.normalize_sizes(sizes, 100, 100)
    rects = squarify.squarify(normed, 0, 0, 100, 100)

    for r, name, sz, dur, color in zip(rects, names, sizes, durations, colors):
        rect = mpatches.FancyBboxPatch(
            (r["x"] + 0.3, r["y"] + 0.3),
            r["dx"] - 0.6, r["dy"] - 0.6,
            boxstyle="round,pad=0.3",
            facecolor=color, edgecolor="white", linewidth=2)
        ax.add_patch(rect)

        cx = r["x"] + r["dx"] / 2
        cy = r["y"] + r["dy"] / 2

        # Only label if rectangle is large enough
        if r["dx"] > 8 and r["dy"] > 8:
            # Truncate long names
            display_name = name if len(name) < 16 else name[:14] + "."
            ax.text(cx, cy + 1.5, display_name, ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])
            ax.text(cx, cy - 2.5, f"{sz}", ha="center", va="center",
                    fontsize=7, color="white",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])
        elif r["dx"] > 5 and r["dy"] > 5:
            ax.text(cx, cy, f"{sz}", ha="center", va="center",
                    fontsize=6, color="white", fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    ax.set_title(
        f"Property Test Coverage ({data['totals']['total-assertions']} assertions, "
        f"{len(suites)} suites)",
        fontsize=FONTS["title"], fontweight="bold")
    ax.axis("off")

    # Color bar legend
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.Purples,
        norm=plt.Normalize(vmin=min(durations), vmax=max(durations)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Duration (ms)", fontsize=FONTS["bar_label"])

    save_fig(fig, os.path.join(FIGS, "fig18_property_coverage.pdf"))


# ─────────────────────────────────────────────────────────────────────
# Figure 19: Hero Performance Landscape
# ─────────────────────────────────────────────────────────────────────

def fig19_hero():
    comp = load("compilation-ladder/data.json")
    vec = load("vectorization/data.json")
    conj = load("conjugacy/data.json")
    scale = load("particle-scaling/data.json")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panel_labels = ["A", "B", "C", "D"]

    # Panel A: Compile (MH handler vs compiled)
    ax = axes[0, 0]
    mh_handler = next(r for r in comp["results"]
                      if r["level"] == "L2-MH-handler")
    mh_compiled = next(r for r in comp["results"]
                       if r["level"] == "L2-MH-compiled")
    speedup = comp["speedups"]["mh-handler-to-compiled"]

    bars_a = ax.bar(
        ["Handler\nMH", "Compiled\nMH"],
        [mh_handler["mean-ms"], mh_compiled["mean-ms"]],
        color=[COLORS["genmlx_vec"], COLORS["genmlx"]],
        width=0.5, **BAR)
    ax.set_ylabel("Time (ms)")
    ax.set_title("A. Compile", fontsize=FONTS["title"], fontweight="bold",
                 loc="left")
    clean_axes(ax, grid=True)

    # Annotate speedup
    ax.annotate(f"{speedup:.1f}x faster",
                xy=(1, mh_compiled["mean-ms"]),
                xytext=(1.3, mh_handler["mean-ms"] * 0.5),
                fontsize=FONTS["legend"], fontweight="bold",
                color=COLORS["genmlx"],
                arrowprops=dict(arrowstyle="->", color=COLORS["genmlx"],
                                lw=2))

    # Panel B: Vectorize (2182x IS speedup)
    ax = axes[0, 1]
    vis_data = next(r for r in vec["results"]
                    if r["benchmark"] == "vectorized-IS")
    seq_ms = vis_data["sequential"]["mean-ms"]
    batch_ms = vis_data["batched"]["mean-ms"]
    vis_speedup = vis_data["speedup"]

    bars_b = ax.bar(
        ["Sequential\nIS", "Vectorized\nIS"],
        [seq_ms, batch_ms],
        color=[COLORS["genmlx_vec"], COLORS["genmlx"]],
        width=0.5, **BAR)
    ax.set_yscale("log")
    ax.set_ylabel("Time (ms)")
    ax.set_title("B. Vectorize", fontsize=FONTS["title"], fontweight="bold",
                 loc="left")
    clean_axes(ax, grid=True)

    ax.annotate(f"{vis_speedup:.0f}x",
                xy=(1, batch_ms),
                xytext=(1.3, batch_ms * 30),
                fontsize=FONTS["base"], fontweight="bold",
                color=COLORS["genmlx"],
                arrowprops=dict(arrowstyle="->", color=COLORS["genmlx"],
                                lw=2))

    # Panel C: Eliminate (L2 ESS vs L3 ESS for LinReg)
    ax = axes[1, 0]
    linreg = conj["sub_experiments"]["exp5e"]
    l2_ess = linreg["l2"]["ess_mean"]
    l3_ess = linreg["l3"]["ess_mean"]
    ess_ratio = linreg["ess_improvement"]

    bars_c = ax.bar(
        ["L2\n(IS)", "L3\n(Analytical)"],
        [l2_ess, l3_ess],
        color=[COLORS["genmlx_vec"], COLORS["genmlx"]],
        width=0.5, **BAR)
    ax.set_ylabel("ESS")
    ax.set_title("C. Eliminate", fontsize=FONTS["title"], fontweight="bold",
                 loc="left")
    clean_axes(ax, grid=True)

    ax.annotate(f"{ess_ratio:.0f}x ESS",
                xy=(1, l3_ess),
                xytext=(1.3, l3_ess * 0.6),
                fontsize=FONTS["legend"], fontweight="bold",
                color=COLORS["genmlx"],
                arrowprops=dict(arrowstyle="->", color=COLORS["genmlx"],
                                lw=2))

    # Panel D: Scale (flat time curve)
    ax = axes[1, 1]
    particles = [s["particles"] for s in scale["scaling"]]
    times = [s["mean-ms"] for s in scale["scaling"]]
    stds = [s["std-ms"] for s in scale["scaling"]]

    ax.plot(particles, times, color=COLORS["genmlx"], marker="o",
            markersize=8, **LINE["main"])
    ax.fill_between(particles,
                    [t - s for t, s in zip(times, stds)],
                    [t + s for t, s in zip(times, stds)],
                    alpha=0.2, color=COLORS["genmlx"])
    ax.set_xlabel("Particles")
    ax.set_ylabel("Time (ms)")
    ax.set_xscale("log")
    ax.set_title("D. Scale", fontsize=FONTS["title"], fontweight="bold",
                 loc="left")
    clean_axes(ax, grid=True)

    ratio_10k = scale["ratios"]["10k-vs-100"]
    ax.annotate(f"10K particles\n{ratio_10k:.2f}x time",
                xy=(10000, times[-1]),
                xytext=(3000, max(times) * 1.15),
                fontsize=FONTS["bar_label"], fontweight="bold",
                color=COLORS["genmlx"],
                arrowprops=dict(arrowstyle="->", color=COLORS["genmlx"],
                                lw=1.5))

    fig.suptitle("GenMLX Performance: Four Techniques",
                 fontsize=FONTS["title"] + 2, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, os.path.join(FIGS, "fig19_hero.pdf"))


# ─────────────────────────────────────────────────────────────────────
# Figure 20: Inference Recommendation Map
# ─────────────────────────────────────────────────────────────────────

def fig20_recommendation():
    data = load("method-selection/data.json")
    ms_results = data["method_selection"]["results"]

    # Build recommendation table
    rows = [
        {
            "characteristic": "Fully conjugate",
            "example": "Normal-Normal LinReg",
            "method": "exact",
            "evidence": "L3 auto-conjugacy",
            "latents": "All eliminated",
        },
        {
            "characteristic": "Partially conjugate",
            "example": "Mixed model",
            "method": "hmc",
            "evidence": "Rao-Blackwellization",
            "latents": "1 residual dim",
        },
        {
            "characteristic": "Dynamic addresses",
            "example": "Dynamic LinReg",
            "method": "handler-is",
            "evidence": "Shape methods N/A",
            "latents": "3 residual dims",
        },
        {
            "characteristic": "Trivial (no latents)",
            "example": "Empty model",
            "method": "exact",
            "evidence": "Trivially solved",
            "latents": "None",
        },
        {
            "characteristic": "High-dimensional",
            "example": "15+ latent model",
            "method": "vi",
            "evidence": ">10 dims, VI preferred",
            "latents": "15 residual dims",
        },
        {
            "characteristic": "Sub-model structure",
            "example": "Splice model",
            "method": "smc",
            "evidence": "Splice sites present",
            "latents": "1 residual dim",
        },
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(-1, len(rows) + 0.5)
    ax.invert_yaxis()

    col_x = [0.0, 3.2, 5.5, 7.0, 9.0]
    col_headers = ["Characteristic", "Example", "Method", "Evidence", "Latents"]

    # Header row
    for cx, h in zip(col_x, col_headers):
        ax.text(cx, -0.5, h, fontsize=FONTS["legend"], fontweight="bold",
                ha="left", va="center")
    ax.axhline(y=0.0, color="black", linewidth=1.5)

    method_colors = {
        "exact": COLORS["genmlx"],
        "hmc": COLORS["hmc"],
        "handler-is": COLORS["vec_is"],
        "vi": COLORS["advi"],
        "smc": COLORS["smc"],
    }

    for i, row in enumerate(rows):
        y_pos = i + 0.5
        # Alternate row shading
        if i % 2 == 0:
            ax.axhspan(y_pos - 0.45, y_pos + 0.45, color="#f5f5f5", zorder=0)

        ax.text(col_x[0], y_pos, row["characteristic"],
                fontsize=FONTS["bar_label"], va="center", ha="left")
        ax.text(col_x[1], y_pos, row["example"],
                fontsize=FONTS["bar_label"], va="center", ha="left",
                style="italic")

        # Method as colored badge
        mc = method_colors.get(row["method"], "gray")
        badge = mpatches.FancyBboxPatch(
            (col_x[2] - 0.1, y_pos - 0.3), 1.4, 0.6,
            boxstyle="round,pad=0.1",
            facecolor=mc, edgecolor="none", alpha=0.8)
        ax.add_patch(badge)
        ax.text(col_x[2] + 0.6, y_pos, row["method"],
                fontsize=FONTS["bar_label"], va="center", ha="center",
                color="white", fontweight="bold")

        ax.text(col_x[3], y_pos, row["evidence"],
                fontsize=FONTS["bar_label"], va="center", ha="left")
        ax.text(col_x[4], y_pos, row["latents"],
                fontsize=FONTS["bar_label"], va="center", ha="left")

    ax.set_title("Inference Recommendation Map (6/6 auto-selected correctly)",
                 fontsize=FONTS["title"], fontweight="bold")
    ax.axis("off")

    # Method legend
    handles = [make_legend_handle(c, m) for m, c in method_colors.items()]
    ax.legend(handles=handles,
              labels=list(method_colors.keys()),
              loc="lower center",
              ncol=len(method_colors),
              bbox_to_anchor=(0.5, -0.12),
              **LEGEND)

    save_fig(fig, os.path.join(FIGS, "fig20_recommendation.pdf"))


# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup()
    os.makedirs(FIGS, exist_ok=True)

    fig15_method_spectrum()
    fig16_method_selection()
    fig17_test_dashboard()
    fig18_property_coverage()
    fig19_hero()
    fig20_recommendation()

    print("\nAll figures 15-20 generated successfully.")
