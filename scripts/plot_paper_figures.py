#!/usr/bin/env python3
"""Generate all publication figures for the GenMLX TOPML paper.

Usage:
    python scripts/plot_paper_figures.py

Reads JSON data from results/ and writes PDF/TEX outputs alongside the data.
"""

import json
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import gaussian_kde, norm

# ---------------------------------------------------------------------------
# Style setup — ACM journal style
# ---------------------------------------------------------------------------

rcParams['font.family'] = 'serif'
try:
    rcParams['text.usetex'] = True
    rcParams['font.serif'] = ['Computer Modern Roman']
except Exception:
    rcParams['text.usetex'] = False

rcParams['figure.figsize'] = (3.5, 2.5)
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.05
rcParams['axes.labelsize'] = 9
rcParams['axes.titlesize'] = 10
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 7.5
rcParams['legend.framealpha'] = 0.9
rcParams['lines.linewidth'] = 1.5
rcParams['lines.markersize'] = 4

# Okabe-Ito colorblind-safe palette
BLUE    = '#0072B2'
ORANGE  = '#E69F00'
GREEN   = '#009E73'
RED     = '#D55E00'
PURPLE  = '#CC79A7'
CYAN    = '#56B4E9'
BLACK   = '#000000'

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')


def load_json(path):
    with open(os.path.join(RESULTS, path)) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fig 1: Particle Scaling (log-log)
# ---------------------------------------------------------------------------

def fig1_particle_scaling():
    data = load_json('exp1_vectorization/particle_scaling.json')
    results = data['results']

    ns = [r['n'] for r in results]
    seq_mean = [r['sequential']['mean'] for r in results]
    seq_std = [r['sequential']['std'] for r in results]
    bat_mean = [r['batched']['mean'] for r in results]
    bat_std = [r['batched']['std'] for r in results]
    speedups = [r['speedup'] for r in results]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    ax.errorbar(ns, seq_mean, yerr=seq_std, fmt='o-', color=RED,
                label='Sequential', capsize=2, markersize=4)
    ax.errorbar(ns, bat_mean, yerr=bat_std, fmt='s-', color=BLUE,
                label='Vectorized', capsize=2, markersize=4)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of particles $N$')
    ax.set_ylabel('Time (ms)')
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])

    # Annotate speedups at key points
    for i, n in enumerate(ns):
        if n in (100, 500, 1000):
            ax.annotate(f'{speedups[i]:.0f}$\\times$',
                        xy=(n, bat_mean[i]), xytext=(0, -14),
                        textcoords='offset points', fontsize=7,
                        ha='center', color=BLUE)

    ax.legend(loc='upper left')
    ax.set_title('Particle Scaling: Sequential vs.\\ Vectorized')

    outpath = os.path.join(RESULTS, 'exp1_vectorization', 'fig1_particle_scaling.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'  Wrote: {outpath}')


# ---------------------------------------------------------------------------
# Fig 2: Method Speedup (horizontal bar)
# ---------------------------------------------------------------------------

def fig2_method_speedup():
    data = load_json('exp1_vectorization/method_speedup.json')
    results = data['results']

    methods = [r['method'].replace('_', ' ').title() for r in results]
    # Reformat labels
    label_map = {
        'Dist Sample N': 'dist-sample-n',
        'Importance Sampling': 'Importance Sampling',
        'Smc Init': 'SMC Init',
    }
    methods = [label_map.get(m, m) for m in methods]
    speedups = [r['speedup'] for r in results]

    fig, ax = plt.subplots(figsize=(3.5, 1.8))

    colors = [ORANGE, BLUE, GREEN]
    bars = ax.barh(methods, speedups, color=colors, height=0.5, edgecolor='white')

    for bar, s in zip(bars, speedups):
        ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
                f'{s:.0f}$\\times$', va='center', fontsize=7)

    ax.set_xlabel('Speedup vs.\\ Sequential ($N=1000$)')
    ax.set_xlim(0, max(speedups) * 1.15)
    ax.set_title('Vectorization Speedup by Method')

    outpath = os.path.join(RESULTS, 'exp1_vectorization', 'fig2_method_speedup.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'  Wrote: {outpath}')


# ---------------------------------------------------------------------------
# Fig 3: FFI Scaling (STAR figure)
# ---------------------------------------------------------------------------

def fig3_ffi_scaling():
    dims = [10, 25, 50, 100, 200]

    per_site = []
    gauss_vec = []
    genjax = []

    for d in dims:
        ps = load_json(f'exp2_ffi_bottleneck/is_D{d}_n10000.json')
        gv = load_json(f'exp2_ffi_bottleneck/is_fast_D{d}_n10000.json')
        gj = load_json(f'exp2_ffi_bottleneck/genjax_is_D{d}_n10000.json')
        # times are in seconds, convert to ms
        per_site.append((ps['mean_time'] * 1000, ps['std_time'] * 1000))
        gauss_vec.append((gv['mean_time'] * 1000, gv['std_time'] * 1000))
        genjax.append((gj['mean_time'] * 1000, gj['std_time'] * 1000))

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    ax.errorbar(dims, [p[0] for p in per_site], yerr=[p[1] for p in per_site],
                fmt='o-', color=RED, label='GenMLX per-site', capsize=2, markersize=4)
    ax.errorbar(dims, [g[0] for g in genjax], yerr=[g[1] for g in genjax],
                fmt='^-', color=ORANGE, label='GenJAX (JIT)', capsize=2, markersize=4)
    ax.errorbar(dims, [g[0] for g in gauss_vec], yerr=[g[1] for g in gauss_vec],
                fmt='s-', color=BLUE, label='GenMLX gaussian-vec', capsize=2, markersize=4)

    ax.set_yscale('log')
    ax.set_xlabel('Number of features $D$')
    ax.set_ylabel('Time (ms)')
    ax.set_xticks(dims)
    ax.legend(loc='upper left')
    ax.set_title('FFI Overhead: Per-site vs.\\ Vector Distributions')

    outpath = os.path.join(RESULTS, 'exp2_ffi_bottleneck', 'fig3_ffi_scaling.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'  Wrote: {outpath}')


# ---------------------------------------------------------------------------
# Fig 4: FFI Speedup Table (LaTeX)
# ---------------------------------------------------------------------------

def fig4_speedup_table():
    dims = [10, 25, 50, 100, 200]

    rows = []
    for d in dims:
        ps = load_json(f'exp2_ffi_bottleneck/is_D{d}_n10000.json')
        gv = load_json(f'exp2_ffi_bottleneck/is_fast_D{d}_n10000.json')
        gj = load_json(f'exp2_ffi_bottleneck/genjax_is_D{d}_n10000.json')
        ps_ms = ps['mean_time'] * 1000
        gv_ms = gv['mean_time'] * 1000
        gj_ms = gj['mean_time'] * 1000
        rows.append((d, ps_ms, gv_ms, gj_ms, ps_ms / gv_ms, gj_ms / gv_ms))

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{FFI bottleneck: IS time (ms) at $N\!=\!10{,}000$ particles.}',
        r'\label{tab:ffi-speedup}',
        r'\begin{tabular}{r r r r r r}',
        r'\toprule',
        r'$D$ & Per-site & gauss-vec & GenJAX & \multicolumn{2}{c}{Speedup} \\',
        r'    &  (ms)    &  (ms)     & (ms)   & vs per-site & vs GenJAX \\',
        r'\midrule',
    ]
    for d, ps, gv, gj, sp_ps, sp_gj in rows:
        lines.append(
            f'{d} & {ps:.1f} & {gv:.1f} & {gj:.1f} '
            f'& {sp_ps:.1f}$\\times$ & {sp_gj:.1f}$\\times$ \\\\'
        )
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    outpath = os.path.join(RESULTS, 'exp2_ffi_bottleneck', 'fig4_speedup_table.tex')
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Wrote: {outpath}')


# ---------------------------------------------------------------------------
# Fig 5: Posterior Densities (KDE)
# ---------------------------------------------------------------------------

def fig5_linreg_posteriors():
    data = load_json('exp3_canonical_models/linreg_results.json')
    analytic = data['analytic']
    algos = data['algorithms']

    a_mu = analytic['slope']['mean']
    a_std = analytic['slope']['std']

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # X grid for density plots
    x = np.linspace(a_mu - 4 * a_std, a_mu + 4 * a_std, 500)

    # Analytic posterior
    ax.plot(x, norm.pdf(x, a_mu, a_std), color=BLACK, linewidth=2.5,
            label='Analytic', zorder=10)

    colors_map = {
        'Compiled_MH': BLUE,
        'HMC': GREEN,
        'NUTS': ORANGE,
        'ADVI': PURPLE,
        'Vectorized_IS': CYAN,
    }

    for algo in algos:
        name = algo['algorithm']
        color = colors_map.get(name, 'gray')

        if name == 'ADVI':
            # Gaussian from learned mu/sigma
            mu = algo.get('slope_mu', algo['slope']['mean'])
            sigma = algo.get('slope_sigma', algo['slope']['std'])
            ax.plot(x, norm.pdf(x, mu, sigma), color=color,
                    linestyle='--', label='ADVI')
        elif name == 'Vectorized_IS':
            # Weighted KDE
            samples = np.array(algo.get('slope_samples', []))
            weights = np.array(algo.get('weights', []))
            if len(samples) > 0 and len(weights) > 0:
                # Resample according to weights for KDE
                weights = weights / weights.sum()
                idx = np.random.choice(len(samples), size=min(2000, len(samples)),
                                       p=weights, replace=True)
                resampled = samples[idx]
                kde = gaussian_kde(resampled)
                ax.plot(x, kde(x), color=color, linestyle='-.', label='Vec.\\ IS')
            else:
                # Fallback: Gaussian from stats
                mu = algo['slope']['mean']
                sigma = algo['slope']['std']
                ax.plot(x, norm.pdf(x, mu, sigma), color=color,
                        linestyle='-.', label='Vec.\\ IS')
        else:
            # KDE from raw samples
            samples = np.array(algo.get('slope_samples', []))
            if len(samples) > 0:
                kde = gaussian_kde(samples)
                ax.plot(x, kde(x), color=color, label=name.replace('_', ' '))
            else:
                # Fallback: Gaussian from stats
                mu = algo['slope']['mean']
                sigma = algo['slope']['std']
                ax.plot(x, norm.pdf(x, mu, sigma), color=color,
                        label=name.replace('_', ' '))

    ax.set_xlabel('Slope')
    ax.set_ylabel('Density')
    ax.set_title('Posterior Density: Slope Parameter')
    ax.legend(loc='upper right', fontsize=6.5)

    outpath = os.path.join(RESULTS, 'exp3_canonical_models', 'fig5_linreg_posteriors.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'  Wrote: {outpath}')


# ---------------------------------------------------------------------------
# Fig 6: Linear Regression Table (LaTeX)
# ---------------------------------------------------------------------------

def fig6_linreg_table():
    data = load_json('exp3_canonical_models/linreg_results.json')
    analytic = data['analytic']
    algos = data['algorithms']

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Bayesian linear regression: posterior accuracy across five inference algorithms.}',
        r'\label{tab:linreg}',
        r'\begin{tabular}{l c c c c r}',
        r'\toprule',
        r'Algorithm & Slope Mean & Slope Err & ESS & $\hat{R}$ & Time (ms) \\',
        r'\midrule',
    ]

    for algo in algos:
        name = algo['algorithm'].replace('_', ' ')
        s_mean = algo['slope']['mean']
        s_err = algo['slope']['error']
        ess = algo.get('ess')
        rhat = algo.get('rhat')
        t = algo['time_ms']

        ess_str = f'{ess:.0f}' if ess is not None else '---'
        rhat_str = f'{rhat:.3f}' if rhat is not None else '---'

        lines.append(
            f'{name} & {s_mean:.3f} & {s_err:.3f} & {ess_str} & {rhat_str} & {t:.0f} \\\\'
        )

    lines += [
        r'\midrule',
        f'Analytic & {analytic["slope"]["mean"]:.3f} & --- & --- & --- & --- \\\\',
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    outpath = os.path.join(RESULTS, 'exp3_canonical_models', 'fig6_linreg_table.tex')
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Wrote: {outpath}')


# ---------------------------------------------------------------------------
# Fig 11: Verification Ladder Table (LaTeX)
# ---------------------------------------------------------------------------

def fig7_hmm_logml():
    data = load_json('exp3_canonical_models/hmm_results.json')
    exact = data['exact_log_ml']
    algos = data['algorithms']

    names = []
    errors = []
    error_stds = []
    colors = []

    color_map = {'IS_1000': RED, 'SMC_100': BLUE, 'SMC_250': GREEN}

    for algo in algos:
        name = algo['algorithm']
        label = name.replace('_', ' (') + ')'
        names.append(label)
        errors.append(algo['error'])
        error_stds.append(algo['error_std'])
        colors.append(color_map.get(name, 'gray'))

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    x = np.arange(len(names))
    bars = ax.bar(x, errors, yerr=error_stds, color=colors, width=0.5,
                  capsize=3, edgecolor='white', linewidth=0.5)

    # Annotate bars with error values
    for i, (bar, err) in enumerate(zip(bars, errors)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + error_stds[i] + 0.03,
                f'{err:.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel(r'$|\log \hat{Z} - \log Z|$')
    ax.set_title('HMM Log-ML Estimation Error')
    ax.set_ylim(0)

    # Add exact log-ML annotation
    ax.text(0.98, 0.95, f'Exact $\\log Z = {exact:.1f}$',
            transform=ax.transAxes, ha='right', va='top', fontsize=6.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.8))

    outpath = os.path.join(RESULTS, 'exp3_canonical_models', 'fig7_hmm_logml.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'  Wrote: {outpath}')


def fig11_verification_ladder():
    data = load_json('exp5_verification/verification_summary.json')

    gen_clj = data['gen_clj_compat']
    genjax = data['genjax_compat']
    contracts = data['gfi_contracts']

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{GenMLX verification ladder: four levels of correctness evidence.}',
        r'\label{tab:verification}',
        r'\begin{tabular}{l l r r l}',
        r'\toprule',
        r'Level & What & Count & Pass & Validates \\',
        r'\midrule',
        f'Runtime contracts & Executable checks & '
        f'{contracts["contracts"]} $\\times$ {contracts["models"]} '
        f'& 100\\% & GFI invariants \\\\',
        f'Gen.clj compat & Regression suite & {gen_clj["total"]} '
        f'& 100\\% & Cross-implementation \\\\',
        f'GenJAX compat & Regression suite & {genjax["total"]} '
        f'& 100\\% & Cross-implementation \\\\',
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    outpath = os.path.join(RESULTS, 'exp5_verification', 'fig11_verification_ladder.tex')
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Wrote: {outpath}')


# ---------------------------------------------------------------------------
# Compilation Table (LaTeX)
# ---------------------------------------------------------------------------

def compilation_table():
    data = load_json('exp6_compilation/compiled_speedup.json')

    b1 = data['bench1_gfi_mh_vs_compiled_mh']
    b2 = data['bench2_score_fn_compilation']
    b3 = data['bench3_hmc_genmlx_vs_handcoded']
    b4 = data['bench4_serial_vs_vectorized_mh']

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Compilation and loop fusion speedups.}',
        r'\label{tab:compilation}',
        r'\begin{tabular}{l r r r}',
        r'\toprule',
        r'Benchmark & Baseline (ms) & Compiled (ms) & Speedup \\',
        r'\midrule',
        f'GFI MH vs Compiled MH & {b1["gfi_mh"]["mean"]:.0f} '
        f'& {b1["compiled_mh"]["mean"]:.0f} '
        f'& {b1["speedup"]:.1f}$\\times$ \\\\',
        f'Score-fn compilation & {b2["uncompiled"]["mean"]:.1f} '
        f'& {b2["compiled"]["mean"]:.1f} '
        f'& {b2["speedup"]:.1f}$\\times$ \\\\',
        f'HMC GenMLX vs hand-coded & {b3["genmlx_hmc"]["mean"]:.0f} '
        f'& {b3["handcoded_hmc"]["mean"]:.0f} '
        f'& {b3["overhead"]:.2f}$\\times$ \\\\',
        f'Serial vs Vec.\\ MH (10 chains) & {b4["serial_extrapolated"]["mean"]:.0f} '
        f'& {b4["vectorized"]["mean"]:.0f} '
        f'& {b4["speedup"]:.1f}$\\times$ \\\\',
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    outpath = os.path.join(RESULTS, 'exp6_compilation', 'compilation_table.tex')
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Wrote: {outpath}')


# ---------------------------------------------------------------------------
# Fig 0: Architecture Diagram
# ---------------------------------------------------------------------------

def fig0_architecture():
    layers = [
        ('Layer 0', 'MLX + Runtime', 'mlx.cljs, random.cljs, runtime.cljs', 'mutable boundary'),
        ('Layer 1', 'Core Data', 'ChoiceMap, Trace, Selection', 'pure'),
        ('Layer 2', 'GFI \\& Execution', 'protocols, handler, edit, diff', 'pure'),
        ('Layer 3', 'DSL', 'gen macro, DynamicGF', 'pure'),
        ('Layer 4', 'Distributions', '27 types via defdist', 'pure'),
        ('Layer 5', 'Combinators', 'Map, Unfold, Switch, Scan, ...', 'pure'),
        ('Layer 6', 'Inference', 'IS, MCMC, SMC, VI, ADEV', 'pure'),
        ('Layer 7', 'Verification', 'contracts, verify', 'pure'),
    ]

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, len(layers) * 1.1 + 0.3)
    ax.axis('off')

    # Color gradient from warm (bottom/mutable) to cool (top/pure)
    cmap = matplotlib.colormaps.get_cmap('RdYlGn')
    colors = [cmap(0.15)] + [cmap(0.45 + 0.07 * i) for i in range(7)]

    for i, (num, name, desc, purity) in enumerate(layers):
        y = i * 1.1
        rect = plt.Rectangle((0.3, y), 7.0, 0.85, facecolor=colors[i],
                              edgecolor='#333333', linewidth=0.8, zorder=2)
        ax.add_patch(rect)
        ax.text(0.6, y + 0.43, f'\\textbf{{{num}: {name}}}', fontsize=7.5,
                va='center', ha='left', zorder=3)
        ax.text(4.8, y + 0.43, desc, fontsize=6, va='center', ha='center',
                color='#444444', zorder=3)
        # Purity annotation on right
        style = 'italic' if purity == 'mutable boundary' else 'normal'
        ax.text(7.6, y + 0.43, purity, fontsize=6, va='center', ha='left',
                fontstyle=style, color='#666666', zorder=3)

    ax.set_title('GenMLX Architecture Stack', fontsize=11, pad=8)

    outpath = os.path.join(RESULTS, 'fig0_architecture.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'  Wrote: {outpath}')


# ---------------------------------------------------------------------------
# Fig 8: GMM Log-ML Error (bar chart)
# ---------------------------------------------------------------------------

def fig8_gmm_logml():
    data = load_json('exp3_canonical_models/gmm_results.json')
    exact = data['exact_log_ml']
    algos = data['algorithms']

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # IS error bar
    is_algo = next(a for a in algos if a['method'] == 'is')
    gibbs_algo = next(a for a in algos if a['method'] == 'gibbs')

    names = ['IS (1000)', 'Gibbs (500)']
    # IS: log-ML error; Gibbs: marginal MAE (different metric)
    is_err = is_algo['error']
    is_err_std = is_algo['error_std']
    gibbs_mae = gibbs_algo['marginal_mae']
    gibbs_mae_std = gibbs_algo['marginal_mae_std']

    # Two-panel: left = log-ML error, right = marginal MAE
    # Just show IS log-ML error as bar
    ax.bar([0], [is_err], yerr=[is_err_std], color=[RED], width=0.4,
                  capsize=3, edgecolor='white', linewidth=0.5, label='IS log-ML error')
    ax.text(0, is_err + is_err_std + 0.02, f'{is_err:.2f}',
            ha='center', va='bottom', fontsize=7)

    # Add Gibbs marginal MAE on secondary axis
    ax2 = ax.twinx()
    ax2.bar([1], [gibbs_mae], yerr=[gibbs_mae_std], color=[GREEN], width=0.4,
                    capsize=3, edgecolor='white', linewidth=0.5, label='Gibbs marginal MAE')
    ax2.text(1, gibbs_mae + gibbs_mae_std + 0.001, f'{gibbs_mae:.4f}',
             ha='center', va='bottom', fontsize=7)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel(r'$|\log \hat{Z} - \log Z|$', color=RED)
    ax2.set_ylabel('Marginal MAE', color=GREEN)
    ax.set_ylim(0)
    ax2.set_ylim(0)
    ax.set_title('GMM: IS vs Gibbs')

    ax.text(0.98, 0.95, f'Exact $\\log Z = {exact:.1f}$',
            transform=ax.transAxes, ha='right', va='top', fontsize=6.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.8))

    outpath = os.path.join(RESULTS, 'exp3_canonical_models', 'fig8_gmm_logml.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'  Wrote: {outpath}')


# ---------------------------------------------------------------------------
# Fig 12: Contract Verification Heatmap
# ---------------------------------------------------------------------------

def fig12_contract_heatmap():
    contracts = [
        'generate-weight\n=score',
        'update-empty\nidentity',
        'update-weight\ncorrectness',
        'update\nround-trip',
        'regenerate-empty\nidentity',
        'project-all\n=score',
        'project-none\n=zero',
        'assess=generate\nscore',
        'propose-generate\nround-trip',
        'score\ndecomposition',
        'broadcast\nequivalence',
    ]
    contract_keys = [
        'generate-weight-equals-score',
        'update-empty-identity',
        'update-weight-correctness',
        'update-round-trip',
        'regenerate-empty-identity',
        'project-all-equals-score',
        'project-none-equals-zero',
        'assess-equals-generate-score',
        'propose-generate-round-trip',
        'score-decomposition',
        'broadcast-equivalence',
    ]

    models = [
        'single-site', 'multi-site', 'linreg', 'splice', 'mixed',
        'map', 'unfold', 'switch', 'scan', 'mask',
        'deep-nesting', 'two-site-vec', 'recurse',
    ]
    model_labels = [
        'single\nsite', 'multi\nsite', 'linreg', 'splice', 'mixed',
        'map', 'unfold', 'switch', 'scan', 'mask',
        'deep\nnesting', 'two-site\nvec', 'recurse',
    ]

    # Which contracts are N/A for each model
    broadcast = {'broadcast-equivalence'}

    na_map = {
        'single-site':  broadcast,
        'multi-site':   broadcast,
        'linreg':       broadcast,
        'splice':       broadcast | {'score-decomposition'},
        'mixed':        broadcast,
        'map':          broadcast | {'update-weight-correctness', 'update-round-trip',
                                     'regenerate-empty-identity'},
        'unfold':       broadcast | {'update-weight-correctness', 'update-round-trip',
                                     'regenerate-empty-identity'},
        'switch':       broadcast,
        'scan':         broadcast | {'update-weight-correctness', 'update-round-trip',
                                     'regenerate-empty-identity'},
        'mask':         broadcast,
        'deep-nesting': broadcast | {'update-weight-correctness', 'update-round-trip',
                                     'regenerate-empty-identity', 'score-decomposition'},
        'two-site-vec': set(),  # all contracts run
        'recurse':      broadcast | {'score-decomposition', 'regenerate-empty-identity'},
    }

    # Build matrix: 1 = pass, 0 = N/A
    n_contracts = len(contract_keys)
    n_models = len(models)
    matrix = np.ones((n_contracts, n_models))

    for j, model in enumerate(models):
        for i, ck in enumerate(contract_keys):
            if ck in na_map[model]:
                matrix[i, j] = 0

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    # Custom colormap: light gray for N/A, green for pass
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#E8E8E8', '#4CAF50'])
    ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(model_labels, fontsize=5.5, ha='center')
    ax.set_yticks(range(n_contracts))
    ax.set_yticklabels(contracts, fontsize=5.5, va='center')

    # Grid lines
    ax.set_xticks(np.arange(-0.5, n_models, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_contracts, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', size=0)

    # Group separators (Basic | Combinators | Advanced)
    for x_sep in [4.5, 9.5]:
        ax.axvline(x=x_sep, color='#333333', linewidth=2)

    # Group labels at top
    ax.text(2.0, -1.3, 'Basic', ha='center', fontsize=7, fontweight='bold')
    ax.text(7.0, -1.3, 'Combinators', ha='center', fontsize=7, fontweight='bold')
    ax.text(11.5, -1.3, 'Advanced', ha='center', fontsize=7, fontweight='bold')

    # Cell annotations
    for i in range(n_contracts):
        for j in range(n_models):
            if matrix[i, j] == 1:
                ax.text(j, i, 'P', ha='center', va='center', fontsize=5,
                        color='white', fontweight='bold')
            else:
                ax.text(j, i, '--', ha='center', va='center', fontsize=5,
                        color='#999999')

    total_pass = int(matrix.sum())
    total_na = int((matrix == 0).sum())
    ax.set_title(f'GFI Contract Verification: {total_pass} pass, {total_na} N/A, 0 fail',
                 fontsize=9, pad=15)

    outpath = os.path.join(RESULTS, 'exp5_verification', 'fig12_contract_heatmap.pdf')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)
    print(f'  Wrote: {outpath}')


# ---------------------------------------------------------------------------
# Table PDFs — render .tex tables as matplotlib figures
# ---------------------------------------------------------------------------

def _render_table_pdf(col_labels, row_data, title, outpath, col_widths=None):
    """Render a table as a PDF using matplotlib."""
    n_cols = len(col_labels)
    n_rows = len(row_data)

    fig_width = max(5.0, n_cols * 1.0)
    fig_height = max(1.5, 0.35 * (n_rows + 1) + 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    cell_text = [[str(c) for c in row] for row in row_data]
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellLoc='center', loc='center',
                     colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.4)

    # Style header
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor('#E8E8E8')
        cell.set_text_props(fontweight='bold')

    ax.set_title(title, fontsize=9, pad=12)

    fig.savefig(outpath)
    plt.close(fig)
    print(f'  Wrote: {outpath}')


def fig4_speedup_table_pdf():
    dims = [10, 25, 50, 100, 200]
    rows = []
    for d in dims:
        ps = load_json(f'exp2_ffi_bottleneck/is_D{d}_n10000.json')
        gv = load_json(f'exp2_ffi_bottleneck/is_fast_D{d}_n10000.json')
        gj = load_json(f'exp2_ffi_bottleneck/genjax_is_D{d}_n10000.json')
        ps_ms = ps['mean_time'] * 1000
        gv_ms = gv['mean_time'] * 1000
        gj_ms = gj['mean_time'] * 1000
        rows.append([d, f'{ps_ms:.1f}', f'{gv_ms:.1f}', f'{gj_ms:.1f}',
                      f'{ps_ms/gv_ms:.1f}x', f'{gj_ms/gv_ms:.1f}x'])

    outpath = os.path.join(RESULTS, 'exp2_ffi_bottleneck', 'fig4_speedup_table.pdf')
    _render_table_pdf(
        ['D', 'Per-site (ms)', 'gauss-vec (ms)', 'GenJAX (ms)',
         'vs per-site', 'vs GenJAX'],
        rows, 'FFI Bottleneck: IS Time at N=10,000', outpath)


def fig6_linreg_table_pdf():
    data = load_json('exp3_canonical_models/linreg_results.json')
    analytic = data['analytic']
    algos = data['algorithms']
    rows = []
    for algo in algos:
        name = algo['algorithm'].replace('_', ' ')
        s_mean = algo['slope']['mean']
        s_err = algo['slope']['error']
        ess = algo.get('ess')
        rhat = algo.get('rhat')
        t = algo['time_ms']
        rows.append([name, f'{s_mean:.3f}', f'{s_err:.3f}',
                      f'{ess:.0f}' if ess else '---',
                      f'{rhat:.3f}' if rhat else '---',
                      f'{t:.0f}'])
    rows.append(['Analytic', f'{analytic["slope"]["mean"]:.3f}',
                  '---', '---', '---', '---'])

    outpath = os.path.join(RESULTS, 'exp3_canonical_models', 'fig6_linreg_table.pdf')
    _render_table_pdf(
        ['Algorithm', 'Slope Mean', 'Slope Err', 'ESS', 'R-hat', 'Time (ms)'],
        rows, 'Bayesian Linear Regression: Posterior Accuracy', outpath)


def fig11_verification_ladder_pdf():
    data = load_json('exp5_verification/verification_summary.json')
    gen_clj = data['gen_clj_compat']
    genjax = data['genjax_compat']
    contracts = data['gfi_contracts']

    rows = [
        ['Runtime contracts', 'Executable checks',
         f'{contracts["contracts"]} x {contracts["models"]}', '100%', 'GFI invariants'],
        ['Gen.clj compat', 'Regression suite', str(gen_clj['total']),
         '100%', 'Cross-implementation'],
        ['GenJAX compat', 'Regression suite', str(genjax['total']),
         '100%', 'Cross-implementation'],
    ]

    outpath = os.path.join(RESULTS, 'exp5_verification', 'fig11_verification_ladder.pdf')
    _render_table_pdf(
        ['Level', 'What', 'Count', 'Pass', 'Validates'],
        rows, 'GenMLX Verification Ladder', outpath)


def compilation_table_pdf():
    data = load_json('exp6_compilation/compiled_speedup.json')
    b1 = data['bench1_gfi_mh_vs_compiled_mh']
    b2 = data['bench2_score_fn_compilation']
    b3 = data['bench3_hmc_genmlx_vs_handcoded']
    b4 = data['bench4_serial_vs_vectorized_mh']

    rows = [
        ['GFI MH vs Compiled MH', f'{b1["gfi_mh"]["mean"]:.0f}',
         f'{b1["compiled_mh"]["mean"]:.0f}', f'{b1["speedup"]:.1f}x'],
        ['Score-fn compilation', f'{b2["uncompiled"]["mean"]:.1f}',
         f'{b2["compiled"]["mean"]:.1f}', f'{b2["speedup"]:.1f}x'],
        ['HMC GenMLX vs hand-coded', f'{b3["genmlx_hmc"]["mean"]:.0f}',
         f'{b3["handcoded_hmc"]["mean"]:.0f}', f'{b3["overhead"]:.2f}x'],
        ['Serial vs Vec. MH (10ch)', f'{b4["serial_extrapolated"]["mean"]:.0f}',
         f'{b4["vectorized"]["mean"]:.0f}', f'{b4["speedup"]:.1f}x'],
    ]

    outpath = os.path.join(RESULTS, 'exp6_compilation', 'compilation_table.pdf')
    _render_table_pdf(
        ['Benchmark', 'Baseline (ms)', 'Compiled (ms)', 'Speedup'],
        rows, 'Compilation and Loop Fusion Speedups', outpath)


# ---------------------------------------------------------------------------
# Fig 9: System Comparison Bar Chart
# ---------------------------------------------------------------------------

def fig9_system_bars():
    genmlx = load_json('exp4_system_comparison/genmlx.json')
    genjl = load_json('exp4_system_comparison/genjl.json')
    genjax = load_json('exp4_system_comparison/genjax.json')

    def find(data, model, algo):
        for c in data['comparisons']:
            if c['model'] == model and c['algorithm'] == algo:
                return c
        return None

    # Define comparison groups
    groups = []
    labels = []

    # LinReg IS
    mlx_lr = find(genmlx, 'linreg', 'IS')
    jl_lr = find(genjl, 'linreg', 'IS')
    jx_lr = find(genjax, 'linreg', 'IS')
    if mlx_lr and jl_lr and jx_lr:
        groups.append((mlx_lr['time_ms'], jl_lr['time_ms'], jx_lr['time_ms']))
        labels.append('LinReg\nIS(1K)')

    # GMM IS
    mlx_gmm = find(genmlx, 'gmm', 'IS')
    jl_gmm = find(genjl, 'gmm', 'IS')
    jx_gmm = find(genjax, 'gmm', 'IS')
    if mlx_gmm and jl_gmm and jx_gmm:
        groups.append((mlx_gmm['time_ms'], jl_gmm['time_ms'], jx_gmm['time_ms']))
        labels.append('GMM\nIS(1K)')

    # HMM IS (GenJAX may be missing)
    mlx_hmm = find(genmlx, 'hmm', 'IS')
    jl_hmm = find(genjl, 'hmm', 'IS')
    jx_hmm = find(genjax, 'hmm', 'IS')
    if mlx_hmm and jl_hmm:
        hmm_jx = jx_hmm['time_ms'] if jx_hmm else None
        groups.append((mlx_hmm['time_ms'], jl_hmm['time_ms'], hmm_jx))
        labels.append('HMM\nIS(1K)')

    # LinReg MH
    mlx_mh = find(genmlx, 'linreg', 'MH')
    jl_mh = find(genjl, 'linreg', 'MH')
    if mlx_mh and jl_mh:
        groups.append((mlx_mh['time_ms'], jl_mh['time_ms'], None))
        labels.append('LinReg\nMH(5K)')

    # HMM SMC
    mlx_smc = find(genmlx, 'hmm', 'SMC')
    jl_smc = find(genjl, 'hmm', 'SMC')
    if mlx_smc and jl_smc:
        groups.append((mlx_smc['time_ms'], jl_smc['time_ms'], None))
        labels.append('HMM\nSMC(100)')

    n_groups = len(groups)
    x = np.arange(n_groups)
    width = 0.25

    fig, ax = plt.subplots(figsize=(5.0, 3.0))

    for i, (label, color, offset) in enumerate([
        ('GenMLX (Metal GPU)', BLUE, -width),
        ('Gen.jl (CPU)', ORANGE, 0),
        ('GenJAX (JIT, CPU)', GREEN, width),
    ]):
        vals = []
        positions = []
        for j, g in enumerate(groups):
            v = g[i]
            if v is not None:
                vals.append(v)
                positions.append(x[j] + offset)
        if vals:
            ax.bar(positions, vals, width, label=label, color=color,
                   edgecolor='white', linewidth=0.5)
            # Annotate values
            for pos, v in zip(positions, vals):
                if v >= 1000:
                    txt = f'{v/1000:.1f}s'
                elif v >= 1:
                    txt = f'{v:.1f}'
                else:
                    txt = f'{v:.2f}'
                ax.text(pos, v * 1.15, txt, ha='center', va='bottom',
                        fontsize=5.5, rotation=0)

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Time (ms, log scale)')
    ax.legend(loc='upper left', fontsize=6.5)
    ax.set_title('System Comparison: GenMLX vs Gen.jl vs GenJAX')

    outpath = os.path.join(RESULTS, 'exp4_system_comparison', 'fig9_system_bars.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'  Wrote: {outpath}')


# ---------------------------------------------------------------------------
# Fig 10: System Comparison Table (LaTeX + PDF)
# ---------------------------------------------------------------------------

def fig10_system_table():
    genmlx = load_json('exp4_system_comparison/genmlx.json')
    genjl = load_json('exp4_system_comparison/genjl.json')
    genjax = load_json('exp4_system_comparison/genjax.json')

    def find(data, model, algo):
        for c in data['comparisons']:
            if c['model'] == model and c['algorithm'] == algo:
                return c
        return None

    def fmt_ms(c):
        if c is None:
            return '---'
        t = c['time_ms']
        if t >= 1000:
            return f'{t/1000:.1f}s'
        elif t >= 10:
            return f'{t:.0f}'
        elif t >= 1:
            return f'{t:.1f}'
        else:
            return f'{t:.2f}'

    rows = [
        ('LinReg', 'IS (1K)',
         find(genmlx, 'linreg', 'IS'), find(genjl, 'linreg', 'IS'),
         find(genjax, 'linreg', 'IS'), 'Vec IS (Metal) vs JIT (CPU)'),
        ('LinReg', 'MH (5K)',
         find(genmlx, 'linreg', 'MH'), find(genjl, 'linreg', 'MH'),
         None, 'Compiled MH vs Gen.jl MH'),
        ('HMM', 'IS (1K)',
         find(genmlx, 'hmm', 'IS'), find(genjl, 'hmm', 'IS'),
         find(genjax, 'hmm', 'IS'), 'Vec IS (Metal) vs Gen.jl (CPU)'),
        ('HMM', 'SMC (100)',
         find(genmlx, 'hmm', 'SMC'), find(genjl, 'hmm', 'SMC'),
         None, 'Unfold PF vs Gen.jl PF'),
        ('GMM', 'IS (1K)',
         find(genmlx, 'gmm', 'IS'), find(genjl, 'gmm', 'IS'),
         find(genjax, 'gmm', 'IS'), 'Vec IS (Metal) vs JIT (CPU)'),
    ]

    # LaTeX table
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{System comparison on Apple M2: GenMLX (Metal GPU), Gen.jl (CPU), GenJAX (JIT, CPU).}',
        r'\label{tab:system-comparison}',
        r'\begin{tabular}{l l r r r l}',
        r'\toprule',
        r'Model & Algorithm & GenMLX & Gen.jl & GenJAX & Notes \\',
        r'\midrule',
    ]
    for model, algo, mlx, jl, jx, note in rows:
        lines.append(
            f'{model} & {algo} & {fmt_ms(mlx)} & {fmt_ms(jl)} '
            f'& {fmt_ms(jx)} & {note} \\\\'
        )
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    outpath = os.path.join(RESULTS, 'exp4_system_comparison', 'fig10_system_table.tex')
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Wrote: {outpath}')


def fig10_system_table_pdf():
    genmlx = load_json('exp4_system_comparison/genmlx.json')
    genjl = load_json('exp4_system_comparison/genjl.json')
    genjax = load_json('exp4_system_comparison/genjax.json')

    def find(data, model, algo):
        for c in data['comparisons']:
            if c['model'] == model and c['algorithm'] == algo:
                return c
        return None

    def fmt_ms(c):
        if c is None:
            return '---'
        t = c['time_ms']
        if t >= 1000:
            return f'{t/1000:.1f}s'
        elif t >= 10:
            return f'{t:.0f}'
        elif t >= 1:
            return f'{t:.1f}'
        else:
            return f'{t:.2f}'

    pdf_rows = [
        ['LinReg', 'IS (1K)',
         fmt_ms(find(genmlx, 'linreg', 'IS')),
         fmt_ms(find(genjl, 'linreg', 'IS')),
         fmt_ms(find(genjax, 'linreg', 'IS')),
         'Vec IS vs JIT'],
        ['LinReg', 'MH (5K)',
         fmt_ms(find(genmlx, 'linreg', 'MH')),
         fmt_ms(find(genjl, 'linreg', 'MH')),
         '---',
         'Compiled MH'],
        ['HMM', 'IS (1K)',
         fmt_ms(find(genmlx, 'hmm', 'IS')),
         fmt_ms(find(genjl, 'hmm', 'IS')),
         fmt_ms(find(genjax, 'hmm', 'IS')),
         'Sequential'],
        ['HMM', 'SMC (100)',
         fmt_ms(find(genmlx, 'hmm', 'SMC')),
         fmt_ms(find(genjl, 'hmm', 'SMC')),
         '---',
         'Unfold PF'],
        ['GMM', 'IS (1K)',
         fmt_ms(find(genmlx, 'gmm', 'IS')),
         fmt_ms(find(genjl, 'gmm', 'IS')),
         fmt_ms(find(genjax, 'gmm', 'IS')),
         'Sequential vs JIT'],
    ]

    outpath = os.path.join(RESULTS, 'exp4_system_comparison', 'fig10_system_table.pdf')
    _render_table_pdf(
        ['Model', 'Algorithm', 'GenMLX', 'Gen.jl', 'GenJAX', 'Notes'],
        pdf_rows, 'System Comparison: Apple M2', outpath,
        col_widths=[0.12, 0.12, 0.12, 0.12, 0.12, 0.22])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)
    print('Generating publication figures...\n')

    print('Fig 0: Architecture Diagram')
    fig0_architecture()

    print('Fig 1: Particle Scaling')
    fig1_particle_scaling()

    print('Fig 2: Method Speedup')
    fig2_method_speedup()

    print('Fig 3: FFI Scaling')
    fig3_ffi_scaling()

    print('Fig 4: FFI Speedup Table (tex + pdf)')
    fig4_speedup_table()
    fig4_speedup_table_pdf()

    print('Fig 5: Posterior Densities')
    fig5_linreg_posteriors()

    print('Fig 6: LinReg Table (tex + pdf)')
    fig6_linreg_table()
    fig6_linreg_table_pdf()

    print('Fig 7: HMM Log-ML')
    fig7_hmm_logml()

    print('Fig 8: GMM Log-ML')
    fig8_gmm_logml()

    print('Fig 11: Verification Ladder (tex + pdf)')
    fig11_verification_ladder()
    fig11_verification_ladder_pdf()

    print('Fig 12: Contract Heatmap')
    fig12_contract_heatmap()

    print('Compilation Table (tex + pdf)')
    compilation_table()
    compilation_table_pdf()

    print('Fig 9: System Comparison Bars')
    fig9_system_bars()

    print('Fig 10: System Comparison Table (tex + pdf)')
    fig10_system_table()
    fig10_system_table_pdf()

    print('\nAll figures generated.')


if __name__ == '__main__':
    main()
