#!/usr/bin/env python3
"""
Experiment 4: GenJAX benchmarks for system comparison.

Models: LinReg, HMM, GMM — same data as GenMLX exp3.
Protocol: 5 warmup, 20 timed runs, time.perf_counter.
Output: results/exp4_system_comparison/genjax.json

Run: PYENV_VERSION=genjax-05 pyenv exec python scripts/exp4_genjax_benchmarks.py
"""

import os
import time
import json
import numpy as np
import jax
import jax.numpy as jnp
from genjax import gen, normal, categorical, ChoiceMap

# ---------------------------------------------------------------------------
# Data (hardcoded from exp3 JSONs — byte-identical inputs)
# ---------------------------------------------------------------------------

LINREG_XS = jnp.array([
    -2.5, -2.236842105263158, -1.973684210526316, -1.7105263157894737,
    -1.4473684210526316, -1.1842105263157896, -0.9210526315789473,
    -0.6578947368421053, -0.3947368421052633, -0.13157894736842124,
    0.1315789473684208, 0.3947368421052633, 0.6578947368421053,
    0.9210526315789473, 1.1842105263157894, 1.4473684210526319,
    1.7105263157894735, 1.973684210526316, 2.2368421052631575, 2.5
])

LINREG_YS = jnp.array([
    -4.645086392760277, -2.232466120468943, -4.743588723634419,
    -1.3603573410134566, -3.226216689536446, -3.294092987713061,
    -1.7810833391390348, -1.0100273317412327, 0.3002335711529378,
    -0.06383435977132734, 0.1810731197658333, 1.4528234914729472,
    2.0677273100928257, 2.639107370062878, 2.240536426243029,
    3.3727551091854515, 4.502427813253904, 4.419489033912358,
    7.114258427368966, 5.38109839707613
])

HMM_T = 50
HMM_TRANS = jnp.array([[0.9, 0.1], [0.1, 0.9]])
HMM_INIT = jnp.array([0.5, 0.5])
HMM_MEANS = jnp.array([-2.0, 2.0])
HMM_SIGMA = 1.0

HMM_YS = jnp.array([
    3.069579601287842, 1.29532390832901, 2.305454909801483,
    -3.5862162113189697, -3.4873690605163574, -2.0841751396656036,
    2.0168902575969696, 0.928423285484314, 3.504786491394043,
    1.6985026597976685, 4.161322116851807, -1.769323617219925,
    -1.535953313112259, -0.8051328659057617, -3.148682117462158,
    -1.7772709429264069, -2.746698319911957, -1.2596661448478699,
    -1.9912031944841146, -4.088978290557861, -2.291228175163269,
    -1.472139060497284, -1.9415821731090546, -2.348399966955185,
    -2.660956621170044, -2.9283525347709656, -2.5985397696495056,
    -1.1935933232307434, -3.5146095752716064, -1.9819734804332256,
    -2.570974826812744, -2.3379217088222504, -2.9130019545555115,
    -1.7724827527999878, 0.14748835563659668, -1.9908120324835181,
    2.650322377681732, 2.6598562598228455, 1.8102920204401016,
    2.1211230158805847, 2.97222638130188, 5.103949785232544,
    1.4137526154518127, 3.213983416557312, 2.1396096646785736,
    1.7250688076019287, 2.607529580593109, 1.4608569145202637,
    2.41530442237854, 2.0015695926267654
])

GMM_K = 3
GMM_N = 8
GMM_MEANS_V = jnp.array([-4.0, 0.0, 4.0])
GMM_SIGMA = 1.0
GMM_WEIGHTS = jnp.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
GMM_LOG_WEIGHTS = jnp.log(GMM_WEIGHTS)

GMM_YS = jnp.array([
    0.8350437879562378, 4.188530474901199, -2.888857841491699,
    -4.104988642036915, 5.638246417045593, 0.10598962754011154,
    -2.8335559368133545, -4.310329109430313
])


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def bench(f, warmup=5, runs=20):
    """Run f with warmup, return list of times in ms."""
    for _ in range(warmup):
        result = f()
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, tuple) and hasattr(result[0], 'block_until_ready'):
            result[0].block_until_ready()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = f()
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, tuple) and hasattr(result[0], 'block_until_ready'):
            result[0].block_until_ready()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return times


# ---------------------------------------------------------------------------
# Model A: Linear Regression
# ---------------------------------------------------------------------------

@gen
def linreg_model(xs):
    slope = normal(0.0, 2.0) @ "slope"
    intercept = normal(0.0, 2.0) @ "intercept"
    for j in range(len(xs)):
        normal(slope * xs[j] + intercept, 1.0) @ f"y{j}"
    return slope


linreg_obs = ChoiceMap.d({f"y{i}": float(LINREG_YS[i]) for i in range(len(LINREG_YS))})

# ---------------------------------------------------------------------------
# Model C: Gaussian Mixture Model
# ---------------------------------------------------------------------------

@gen
def gmm_model(ys):
    for i in range(len(ys)):
        z = categorical(GMM_LOG_WEIGHTS) @ f"z{i}"
        mu = jnp.take(GMM_MEANS_V, z)
        normal(mu, GMM_SIGMA) @ f"y{i}"


gmm_obs = ChoiceMap.d({f"y{i}": float(GMM_YS[i]) for i in range(GMM_N)})

# ---------------------------------------------------------------------------
# Model B: HMM (flat loop)
# ---------------------------------------------------------------------------

@gen
def hmm_model(T):
    z_prev = categorical(jnp.log(HMM_INIT)) @ "z0"
    normal(jnp.take(HMM_MEANS, z_prev), HMM_SIGMA) @ "y0"
    for t in range(1, T):
        trans_probs = jnp.log(HMM_TRANS[z_prev])
        z = categorical(trans_probs) @ f"z{t}"
        normal(jnp.take(HMM_MEANS, z), HMM_SIGMA) @ f"y{t}"
        z_prev = z


hmm_obs = ChoiceMap.d({f"y{t}": float(HMM_YS[t]) for t in range(HMM_T)})


# ---------------------------------------------------------------------------
# Vectorized IS via jax.vmap + jax.jit
# ---------------------------------------------------------------------------

def make_is_fn(model, args, obs, n_particles):
    """Create a JIT-compiled vectorized IS function."""
    def one_is(key):
        tr, w = model.generate(key, obs, args)
        return w

    keys_fn = lambda key: jax.random.split(key, n_particles)

    @jax.jit
    def run_is(key):
        keys = keys_fn(key)
        weights = jax.vmap(one_is)(keys)
        log_ml = jax.scipy.special.logsumexp(weights) - jnp.log(n_particles)
        return log_ml

    return run_is


# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

print(f"\n=== GenJAX System Comparison Benchmarks ===")
print(f"  JAX {jax.__version__} ({jax.devices()[0].platform})")
print(f"  Protocol: 5 warmup, 20 timed runs")
print()

comparisons = []

# --- LinReg IS (N=1000) ---
print("-- LinReg IS (N=1000) --")
linreg_is = make_is_fn(linreg_model, (LINREG_XS,), linreg_obs, 1000)
key = jax.random.key(42)

times = bench(lambda: linreg_is(key))
comparisons.append({
    "model": "linreg",
    "algorithm": "IS",
    "n_particles": 1000,
    "time_ms": float(np.mean(times)),
    "time_ms_std": float(np.std(times)),
    "time_ms_min": float(np.min(times)),
    "times_ms": [float(t) for t in times],
})
print(f"  Mean: {np.mean(times):.3f} ms")
print(f"  Std:  {np.std(times):.3f} ms")
print(f"  Min:  {np.min(times):.3f} ms")

# --- GMM IS (N=1000) ---
print("\n-- GMM IS (N=1000) --")
gmm_is = make_is_fn(gmm_model, (GMM_YS,), gmm_obs, 1000)

times = bench(lambda: gmm_is(key))
comparisons.append({
    "model": "gmm",
    "algorithm": "IS",
    "n_particles": 1000,
    "time_ms": float(np.mean(times)),
    "time_ms_std": float(np.std(times)),
    "time_ms_min": float(np.min(times)),
    "times_ms": [float(t) for t in times],
})
print(f"  Mean: {np.mean(times):.3f} ms")
print(f"  Std:  {np.std(times):.3f} ms")
print(f"  Min:  {np.min(times):.3f} ms")

# --- HMM IS (N=1000) ---
print("\n-- HMM IS (N=1000) --")
try:
    hmm_is = make_is_fn(hmm_model, (HMM_T,), hmm_obs, 1000)
    times = bench(lambda: hmm_is(key))
    comparisons.append({
        "model": "hmm",
        "algorithm": "IS",
        "n_particles": 1000,
        "time_ms": float(np.mean(times)),
        "time_ms_std": float(np.std(times)),
        "time_ms_min": float(np.min(times)),
        "times_ms": [float(t) for t in times],
    })
    print(f"  Mean: {np.mean(times):.3f} ms")
    print(f"  Std:  {np.std(times):.3f} ms")
    print(f"  Min:  {np.min(times):.3f} ms")
except Exception as e:
    print(f"  HMM IS SKIPPED: {e}")

# ---------------------------------------------------------------------------
# Write JSON output
# ---------------------------------------------------------------------------

output = {
    "system": "genjax",
    "version": "0.10.3",
    "jax_version": jax.__version__,
    "hardware": "Apple M2",
    "backend": f"JAX CPU ({jax.devices()[0].platform})",
    "timing_protocol": "5 warmup, 20 runs, time.perf_counter",
    "comparisons": comparisons,
}

outpath = os.path.join(os.path.dirname(__file__), "..", "results",
                       "exp4_system_comparison", "genjax.json")
os.makedirs(os.path.dirname(outpath), exist_ok=True)
with open(outpath, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nWrote: {outpath}")
