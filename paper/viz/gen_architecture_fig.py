#!/usr/bin/env python3
"""Generate the GenMLX architecture stack figure (Figure 1).

Layers 0-8, matching CLAUDE.md architecture spec.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from genmlx_style import setup, save_fig

setup()

# 9 layers (0-8), bottom to top
layers = [
    ("Layer 0:  MLX + Runtime",    "mlx.cljs, random.cljs",                "mutable"),
    ("Layer 1:  Core Data",        "ChoiceMap, Trace, Selection",           "pure"),
    ("Layer 2:  GFI & Execution",  "protocols, handler, edit, diff",        "pure"),
    ("Layer 3:  DSL + Schema",     "gen macro, DynamicGF, schema",          "pure"),
    ("Layer 4:  Distributions",    "27 types via defdist",                  "pure"),
    ("Layer 5:  Combinators",      "Map, Unfold, Switch, Scan, ...",        "pure"),
    ("Layer 6:  Inference",        "IS, MCMC, SMC, VI, ADEV",              "pure"),
    ("Layer 7:  Vectorized",       "batched execution, VIS",                "pure"),
    ("Layer 8:  Verification",     "contracts, verify",                     "pure"),
]

# Colors: gradient from red (mutable) through yellow to green (pure top)
layer_colors = [
    "#E74C3C",  # 0: red (mutable)
    "#F7DC6F",  # 1: pale yellow
    "#F0E68C",  # 2: khaki
    "#C5E17A",  # 3: yellow-green
    "#A8D86D",  # 4: light green
    "#8BC960",  # 5: green
    "#6DBF53",  # 6: medium green
    "#50B546",  # 7: deeper green
    "#3AAB39",  # 8: top green
]

fig, ax = plt.subplots(figsize=(10, 8))

n = len(layers)
bar_height = 0.8
gap = 0.12
total_height = n * (bar_height + gap)

# Use data coordinates scaled for wide bars
x_max = 12.0  # wide coordinate system

for i, (name, desc, purity) in enumerate(layers):
    y = i * (bar_height + gap)
    # Slight indent for visual stacking — wider bars at bottom
    indent = 0.15 * (n - 1 - i)
    width = x_max - 2 * indent

    rect = mpatches.FancyBboxPatch(
        (indent, y), width, bar_height,
        boxstyle="round,pad=0.08",
        facecolor=layer_colors[i],
        edgecolor="#333333",
        linewidth=1.5,
    )
    ax.add_patch(rect)

    # Layer name (left-aligned)
    ax.text(indent + 0.35, y + bar_height / 2, name,
            fontsize=14, fontweight="bold", va="center", ha="left",
            fontfamily="serif", color="#1a1a1a")

    # Description (center)
    ax.text(x_max * 0.55, y + bar_height / 2, desc,
            fontsize=12, va="center", ha="left",
            fontfamily="serif", color="#333333")

    # Purity tag (right-aligned, with fixed margin from right edge)
    color = "#CC3311" if purity == "mutable" else "#2E7D32"
    ax.text(x_max - indent - 0.35, y + bar_height / 2, purity,
            fontsize=12, fontweight="bold", va="center", ha="right",
            fontfamily="serif", color=color,
            bbox=dict(facecolor=layer_colors[i], edgecolor='none', pad=1))

ax.set_xlim(-0.3, x_max + 0.3)
ax.set_ylim(-0.3, total_height + 0.5)
ax.axis("off")

ax.set_title("GenMLX Architecture Stack", fontsize=20, fontweight="bold",
             fontfamily="serif", pad=15)

fig.tight_layout()

outdir = os.path.join(os.path.dirname(__file__), "..", "TOPML_system", "figs")
save_fig(fig, os.path.join(outdir, "fig0_architecture.pdf"))
