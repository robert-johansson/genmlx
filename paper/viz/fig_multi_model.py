"""
Figure 7 for the paper "Three Learning Phenomena, One Function":
multi-model embodied agent loop.

  Top panel:    Bayesian model posterior P(model | rewards) over time
                — three lines, one per kernel (habituation, rate-estimation,
                CRP-operant), with regime boundaries marked.

  Bottom panel: cumulative log-likelihood log P(rewards | model) per kernel,
                showing the per-cycle scoring that drives the posterior above.

Source CSV: paper/figs/data/multi_model_agent.csv
Driver:     examples/multi_model_agent.cljs

Run from repo root:
    .venv/bin/python paper/viz/fig_multi_model.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from genmlx_style import (  # noqa: E402
    COLORS,
    FONTS,
    LINE,
    clean_axes,
    save_fig,
    set_ticks,
    setup,
)

import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "DeHouwer_paper", "figs", "data")
FIGS = os.path.join(HERE, "..", "DeHouwer_paper", "figs")


def fig_multi_model_agent():
    df = pd.read_csv(os.path.join(DATA, "multi_model_agent.csv"))

    color_map = {
        "hab":  COLORS["smc"],     # sky blue
        "rate": COLORS["hmc"],     # orange
        "crp":  COLORS["genmlx"],  # purple (system color)
    }
    label_map = {
        "hab":  "Habituation",
        "rate": "Rate-estimation",
        "crp":  "CRP-operant",
    }

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 6.5),
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.10},
        sharex=True,
    )

    # ── Top panel: model posterior over time ─────────────────────────────
    for m in ("hab", "rate", "crp"):
        ax_top.plot(
            df["cycle"], df[f"p_{m}"],
            color=color_map[m],
            label=label_map[m],
            **LINE["main"],
        )

    ax_top.set_ylabel(r"$P(\mathrm{model}\,|\,\mathrm{rewards})$")
    ax_top.set_ylim(-0.04, 1.06)
    ax_top.legend(
        loc="center left", bbox_to_anchor=(1.005, 0.5),
        fontsize=FONTS["legend"], framealpha=0.9,
    )

    # ── Bottom panel: cumulative log-likelihood ──────────────────────────
    for m in ("hab", "rate", "crp"):
        ax_bot.plot(
            df["cycle"], df[f"{m}_logL"],
            color=color_map[m],
            label=label_map[m],
            **LINE["main"],
        )

    ax_bot.set_xlabel("Cycle")
    ax_bot.set_ylabel(r"$\log P(\mathrm{rewards}_{1:t}\,|\,\mathrm{model})$")

    # ── Regime boundaries on both panels ─────────────────────────────────
    for ax in (ax_top, ax_bot):
        for x in (50, 100):
            ax.axvline(x, color="gray", **LINE["baseline"])
        clean_axes(ax, grid=False)
        set_ticks(ax, x=4, y=3)

    # ── Regime labels above the top panel ────────────────────────────────
    regime_y = 1.10  # axes fraction
    for cx, lab in [(25, "Regime 1\nstable"),
                    (75, "Regime 2\nreversal"),
                    (125, "Regime 3\nstim-driven")]:
        ax_top.text(
            cx, regime_y, lab,
            transform=ax_top.get_xaxis_transform(),
            ha="center", va="bottom",
            fontsize=FONTS["annotation"], alpha=0.85, fontweight="bold",
        )

    save_fig(fig, os.path.join(FIGS, "fig_multi_model_agent.pdf"))


if __name__ == "__main__":
    setup()
    fig_multi_model_agent()
