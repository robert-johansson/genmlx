"""
Figures 3, 4, 5 for the paper "Three Learning Phenomena, One Function":
per-model simulated learning curves.

  Figure 3 (habituation): Gershman (2024) habituation as optimal filtering.
    4 conditions × 10 reps, normalized response vs repetition.
    Source CSV: paper/figs/data/habituation_fig3.csv
    Driver: examples/habituation.cljs (Demo 1, paper Fig 3 reproduction).

  Figure 4 (rate-estimation): Gershman (2025) online Bayesian rate estimation.
    Learning curve of lambda_B and lambda_CS over ~2000 steps.
    Source CSV: paper/figs/data/rate_estimation_learning_curve.csv
    Driver: examples/rate_estimation.cljs (Demo 1, cell 2 reproduction).

  Figure 5 (CRP-operant): Lloyd-Leslie (2013) CRP-based decision making.
    Serial reversal: mean errors per period across 8 reversals.
    Source CSV: paper/figs/data/crp_serial_reversal.csv
    Driver: examples/crp_operant.cljs (Demo 5, paper Fig 7 reproduction).

Run from repo root:
    .venv/bin/python paper/viz/fig_three_phenomena.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from genmlx_style import (  # noqa: E402
    BAR,
    COLORS,
    FONTS,
    LINE,
    SIZES,
    clean_axes,
    save_fig,
    set_ticks,
    setup,
)

import matplotlib.pyplot as plt  # noqa: E402 — must follow setup-aware imports

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "DeHouwer_paper", "figs", "data")
FIGS = os.path.join(HERE, "..", "DeHouwer_paper", "figs")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Habituation (Gershman 2024)
# ─────────────────────────────────────────────────────────────────────────────

def fig_habituation():
    df = pd.read_csv(os.path.join(DATA, "habituation_fig3.csv"))

    # Grouped color scheme: intensity selects hue (blue = low, orange = high),
    # frequency selects saturation (lighter = low freq, darker = high freq).
    color_for_condition = {
        "Intensity: Low,  Frequency: Low":   COLORS["smc"],     # sky blue
        "Intensity: Low,  Frequency: High":  COLORS["vec_is"],  # darker blue
        "Intensity: High, Frequency: Low":   COLORS["gibbs"],   # yellow-orange
        "Intensity: High, Frequency: High":  COLORS["hmc"],     # orange
    }
    label_for_condition = {
        "Intensity: Low,  Frequency: Low":   "Low intensity, Low freq",
        "Intensity: Low,  Frequency: High":  "Low intensity, High freq",
        "Intensity: High, Frequency: Low":   "High intensity, Low freq",
        "Intensity: High, Frequency: High":  "High intensity, High freq",
    }

    fig, ax = plt.subplots(figsize=SIZES["single"])

    for condition, group in df.groupby("condition", sort=False):
        ax.plot(
            group["rep"], group["norm_resp"],
            color=color_for_condition.get(condition, "gray"),
            label=label_for_condition.get(condition, condition),
            marker="o", markersize=6,
            **LINE["main"],
        )

    ax.axhline(100, color="gray", **LINE["baseline"])
    ax.text(
        df["rep"].max(), 100, "baseline",
        ha="right", va="bottom",
        fontsize=FONTS["annotation"], color="gray", alpha=0.7,
    )

    ax.set_xlabel("Repetition")
    ax.set_ylabel("Normalized response (%)")
    ax.legend(loc="best", fontsize=FONTS["legend"], framealpha=0.9)
    clean_axes(ax, grid=False)
    set_ticks(ax)
    save_fig(fig, os.path.join(FIGS, "fig_habituation.pdf"))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Rate estimation (Gershman 2025)
# ─────────────────────────────────────────────────────────────────────────────

def fig_rate_estimation():
    df = pd.read_csv(os.path.join(DATA, "rate_estimation_learning_curve.csv"))

    fig, ax = plt.subplots(figsize=SIZES["single"])

    ax.plot(
        df["step"], df["lambda_B"],
        color=COLORS["truth"],
        label=r"$\hat{\lambda}_B$  (background)",
        marker="o", markersize=4,
        **LINE["main"],
    )
    ax.plot(
        df["step"], df["lambda_CS"],
        color=COLORS["hmc"],
        label=r"$\hat{\lambda}_{CS}$  (CS-driven)",
        marker="o", markersize=4,
        **LINE["main"],
    )

    # True rates as horizontal dashed lines
    ax.axhline(0.5, color=COLORS["truth"], **LINE["truth"])
    ax.axhline(1.5, color=COLORS["hmc"], **LINE["truth"])

    x_text = df["step"].max() * 0.97
    ax.text(
        x_text, 0.5, "true 0.5",
        ha="right", va="bottom",
        fontsize=FONTS["annotation"], color=COLORS["truth"], alpha=0.85,
    )
    ax.text(
        x_text, 1.5, "true 1.5",
        ha="right", va="bottom",
        fontsize=FONTS["annotation"], color=COLORS["hmc"], alpha=0.85,
    )

    ax.set_xlabel("Time step")
    ax.set_ylabel(r"Estimated rate $\hat{\lambda}$")
    ax.legend(loc="center right", fontsize=FONTS["legend"], framealpha=0.9)
    clean_axes(ax, grid=False)
    set_ticks(ax)
    save_fig(fig, os.path.join(FIGS, "fig_rate_estimation.pdf"))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — CRP-operant serial reversal (Lloyd-Leslie 2013)
# ─────────────────────────────────────────────────────────────────────────────

def fig_crp_operant():
    df = pd.read_csv(os.path.join(DATA, "crp_serial_reversal.csv"))

    fig, ax = plt.subplots(figsize=SIZES["single"])

    bars = ax.bar(
        df["period"], df["mean_errors"],
        color=COLORS["genmlx"], **BAR,
    )

    for x, v, arm in zip(df["period"], df["mean_errors"], df["correct_arm"]):
        ax.annotate(
            f"{v:.1f}",
            xy=(x, v), xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=FONTS["bar_label"], fontweight="bold",
        )
        ax.annotate(
            arm,
            xy=(x, 0), xytext=(0, 2),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=FONTS["annotation"], color="white", fontweight="bold",
        )

    ax.set_xlabel("Training period (24 trials, alternating)")
    ax.set_ylabel("Mean errors per period")
    ax.set_xticks(df["period"])
    ax.set_ylim(0, max(df["mean_errors"]) * 1.2)
    clean_axes(ax, grid=False)
    save_fig(fig, os.path.join(FIGS, "fig_crp_operant.pdf"))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup()
    fig_habituation()
    fig_rate_estimation()
    fig_crp_operant()
