#!/usr/bin/env python3
"""
Experiment 4: Extract and combine GenMLX timings.

Reads:
  - results/exp4_system_comparison/genmlx_is1000.json  (IS N=1000 for all 3 models)
  - results/exp3_canonical_models/linreg_results.json  (MH timing)
  - results/exp3_canonical_models/hmm_results.json     (SMC timing)

Writes:
  results/exp4_system_comparison/genmlx.json

Usage: .venv/bin/python3 scripts/exp4_extract_genmlx.py
"""

import json
import os

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')


def load_json(path):
    with open(os.path.join(RESULTS, path)) as f:
        return json.load(f)


def main():
    # IS(1000) timings from dedicated benchmark
    is_data = load_json('exp4_system_comparison/genmlx_is1000.json')

    # Existing exp3 data for MH and SMC
    linreg = load_json('exp3_canonical_models/linreg_results.json')
    hmm = load_json('exp3_canonical_models/hmm_results.json')

    # For each (model, algorithm, n_particles) group, prefer vectorized over sequential
    raw = is_data['comparisons']
    best = {}
    for entry in raw:
        key = (entry['model'], entry['algorithm'], entry.get('n_particles'))
        prev = best.get(key)
        if prev is None or entry.get('method') == 'vectorized':
            best[key] = entry
    comparisons = list(best.values())

    # Add LinReg MH from exp3
    for algo in linreg['algorithms']:
        if algo['algorithm'] == 'Compiled_MH':
            comparisons.append({
                'model': 'linreg',
                'algorithm': 'MH',
                'n_samples': algo['samples'],
                'time_ms': algo['time_ms'],
                'note': 'Compiled MH from exp3',
            })
            break

    # Add HMM SMC from exp3
    for algo in hmm['algorithms']:
        if algo['algorithm'] == 'SMC_100':
            comparisons.append({
                'model': 'hmm',
                'algorithm': 'SMC',
                'n_particles': 100,
                'n_steps': 50,
                'time_ms': algo['time_ms'],
                'time_ms_std': algo['time_ms_std'],
                'note': 'SMC(100) from exp3',
            })
            break

    output = {
        'system': 'genmlx',
        'version': '0.1.0',
        'hardware': 'Apple M2',
        'backend': 'Metal GPU',
        'timing_protocol': '5 warmup, 20 runs, performance.now',
        'comparisons': comparisons,
    }

    outpath = os.path.join(RESULTS, 'exp4_system_comparison', 'genmlx.json')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'Wrote: {outpath}')


if __name__ == '__main__':
    main()
