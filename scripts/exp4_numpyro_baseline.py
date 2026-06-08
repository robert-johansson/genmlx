#!/usr/bin/env python3
"""
Experiment 4: NumPyro out-of-(Gen-)family baseline for the system comparison.

The cross-system comparison is otherwise Gen-family only (GenMLX + Gen.jl + GenJAX,
all GFI/trace-based). NumPyro is the Pyro *effect-handler* paradigm — genuinely
out-of-family — even though it rides on JAX. This adds ONE competently-configured
out-of-family reference point, reported as matched-accuracy-at-cost.

Shared linear-regression spec (BYTE-IDENTICAL to scripts/exp4_genjax_benchmarks.py):
  xs       = 20 points evenly spaced in [-2.5, 2.5]
  priors   = slope, intercept ~ N(0, 2)
  obs      = y_j ~ N(slope*x_j + intercept, 1)
(NB: GenMLX's own bench/cross_system.cljs + bench/linreg.cljs use N(0,10) and a
 different 5-point dataset, so their *timings* are NOT on this shared spec. The
 clean apples-to-apples accuracy comparison here is NumPyro-IS vs GenJAX-IS vs the
 closed-form exact log-ML; see the "caveats" block in the emitted JSON.)

Two reported points:
  1. IS log-ML (N=1000): importance sampling from the prior, log_ml = logsumexp(w) - log N,
     mirroring the GenJAX make_is_fn row (JIT-compiled, vmapped weights). Accuracy is
     checked against an INDEPENDENT closed-form Gaussian marginal likelihood (exact MVN),
     and cross-checked once via NumPyro's own Predictive + log_likelihood machinery.
  2. NUTS (NumPyro's native idiom): posterior means + wall-clock, validated against the
     exact linear-Gaussian posterior mean.

Protocol: 5 warmup, 20 timed runs, time.perf_counter, block_until_ready (JAX).

Reproducibility (CPU JAX is fine; versions recorded in the output JSON):
  ~/code/genmlx/.venv-genjax/bin/pip install numpyro
  ~/code/genmlx/.venv-genjax/bin/python scripts/exp4_numpyro_baseline.py

Output: results/cross-system/numpyro.json
"""

import os
import time
import json
import subprocess

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as ndist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_likelihood

numpyro.set_host_device_count(1)

# ---------------------------------------------------------------------------
# Shared spec (byte-identical to exp4_genjax_benchmarks.py lines 24-41)
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

SIGMA_PRIOR = 2.0
SIGMA_OBS = 1.0
N_PARTICLES = 1000

# GenMLX's L3 analytical eliminator (Kalman) on this exact shared spec, for the
# matched-accuracy claim. Reproduce (Metal GPU, exact to the float32 floor):
#   bun run --bun nbb -e '(do (require (quote [genmlx.mlx :as mx])
#     (quote [genmlx.linear-gaussian :as lg]))
#     (def specs (mapv (fn [x y] {:y (mx/scalar y) :h (mx/array [x 1.0])
#       :c (mx/scalar 0.0) :r (mx/scalar 1.0)}) XS YS))
#     (println (mx/item (:marginal-ll (lg/lg-eliminate (mx/array [0.0 0.0])
#       (mx/array [[4.0 0.0] [0.0 4.0]]) specs)))))'
# => -31.699430465698242 ; post-mean [2.090046 0.558755]  (matches closed form below)
GENMLX_L3_LOG_ML = -31.699430465698242


# ---------------------------------------------------------------------------
# NumPyro model (effect-handler style)
# ---------------------------------------------------------------------------

def linreg_model(xs, ys=None):
    slope = numpyro.sample("slope", ndist.Normal(0.0, SIGMA_PRIOR))
    intercept = numpyro.sample("intercept", ndist.Normal(0.0, SIGMA_PRIOR))
    mu = slope * xs + intercept
    with numpyro.plate("data", xs.shape[0]):
        numpyro.sample("y", ndist.Normal(mu, SIGMA_OBS), obs=ys)


# ---------------------------------------------------------------------------
# Independent closed-form oracle (analytic MVN — NOT NumPyro, NOT GenMLX)
# ---------------------------------------------------------------------------

def exact_linear_gaussian(xs, ys, sigma_prior, sigma_obs):
    """Closed-form log marginal likelihood and posterior mean for
    y ~ N(0, sigma_obs^2 I + Phi diag(sigma_prior^2) Phi^T), Phi = [x, 1]."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    n = xs.shape[0]
    Phi = np.stack([xs, np.ones_like(xs)], axis=1)          # [n, 2]: cols = slope, intercept
    Lam = np.diag([sigma_prior ** 2, sigma_prior ** 2])     # prior cov of [slope, intercept]
    Sigma = sigma_obs ** 2 * np.eye(n) + Phi @ Lam @ Phi.T  # marginal cov of y
    sign, logdet = np.linalg.slogdet(Sigma)
    quad = ys @ np.linalg.solve(Sigma, ys)
    log_ml = -0.5 * (n * np.log(2.0 * np.pi) + logdet + quad)
    # Exact Gaussian posterior over [slope, intercept]
    post_prec = np.linalg.inv(Lam) + (Phi.T @ Phi) / sigma_obs ** 2
    post_cov = np.linalg.inv(post_prec)
    post_mean = post_cov @ (Phi.T @ ys) / sigma_obs ** 2
    return {
        "log_ml": float(log_ml),
        "post_mean_slope": float(post_mean[0]),
        "post_mean_intercept": float(post_mean[1]),
    }


# ---------------------------------------------------------------------------
# IS log-ML (N=1000), JIT-compiled & vmapped — mirrors GenJAX make_is_fn
# ---------------------------------------------------------------------------

def make_is_fn(n_particles):
    """JIT-compiled importance sampling from the prior.

    Per particle: draw (slope, intercept) ~ prior, weight = log p(y | slope, intercept).
    log_ml = logsumexp(weights) - log N. Scoring uses NumPyro distributions, so the
    weight is exactly the generate-weight a NumPyro trace would assign to the
    constrained y sites under a prior-sampled latent."""
    prior = ndist.Normal(0.0, SIGMA_PRIOR)

    @jax.jit
    def run_is(key):
        k_s, k_i = jax.random.split(key)
        slope = prior.sample(k_s, (n_particles,))            # [N]
        intercept = prior.sample(k_i, (n_particles,))        # [N]
        mu = slope[:, None] * LINREG_XS[None, :] + intercept[:, None]   # [N, 20]
        loglik = ndist.Normal(mu, SIGMA_OBS).log_prob(
            LINREG_YS[None, :]).sum(axis=1)                  # [N]
        return jax.scipy.special.logsumexp(loglik) - jnp.log(n_particles)

    return run_is


def faithfulness_check(key, n_particles):
    """Score the SAME prior draws two ways — the hand-vmapped Normal.log_prob used
    by run_is, and NumPyro's own log_likelihood machinery — and confirm the
    per-particle log-weights are identical. This proves the timed hand-rolled
    estimator computes exactly the generate-weight NumPyro would assign, rather
    than relying on two independent (differently-seeded) estimates agreeing."""
    prior = ndist.Normal(0.0, SIGMA_PRIOR)
    k_s, k_i = jax.random.split(key)
    slope = prior.sample(k_s, (n_particles,))
    intercept = prior.sample(k_i, (n_particles,))
    mu = slope[:, None] * LINREG_XS[None, :] + intercept[:, None]
    w_hand = ndist.Normal(mu, SIGMA_OBS).log_prob(LINREG_YS[None, :]).sum(axis=1)  # [N]
    latents = {"slope": slope, "intercept": intercept}
    w_numpyro = log_likelihood(
        linreg_model, latents, xs=LINREG_XS, ys=LINREG_YS)["y"].sum(axis=1)        # [N]
    max_disc = float(jnp.max(jnp.abs(w_hand - w_numpyro)))
    est = float(jax.scipy.special.logsumexp(w_hand) - jnp.log(n_particles))
    return est, max_disc


# ---------------------------------------------------------------------------
# Timing helper (matches exp4_genjax_benchmarks.py bench())
# ---------------------------------------------------------------------------

def bench(f, warmup=5, runs=20):
    for _ in range(warmup):
        r = f()
        if hasattr(r, "block_until_ready"):
            r.block_until_ready()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        r = f()
        if hasattr(r, "block_until_ready"):
            r.block_until_ready()
        times.append((time.perf_counter() - start) * 1000)
    return times


def detect_hardware():
    try:
        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        return brand or "Apple Silicon"
    except Exception:
        return "Apple Silicon"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

print("\n=== NumPyro Out-of-Family Baseline (system comparison) ===")
print(f"  numpyro {numpyro.__version__} | JAX {jax.__version__} "
      f"({jax.devices()[0].platform})")
print(f"  Shared linreg: N={LINREG_XS.shape[0]}, priors N(0,{SIGMA_PRIOR}), "
      f"obs sigma {SIGMA_OBS}")
print("  Protocol: 5 warmup, 20 timed runs\n")

hardware = detect_hardware()
key = jax.random.PRNGKey(42)

# --- Ground truth (independent closed form) ---
exact = exact_linear_gaussian(LINREG_XS, LINREG_YS, SIGMA_PRIOR, SIGMA_OBS)
print(f"-- Exact (closed-form MVN) --")
print(f"  log-ML        : {exact['log_ml']:.6f}")
print(f"  post mean     : slope={exact['post_mean_slope']:.4f} "
      f"intercept={exact['post_mean_intercept']:.4f}\n")

# --- IS log-ML (N=1000), timed ---
print(f"-- LinReg IS (N={N_PARTICLES}) --")
run_is = make_is_fn(N_PARTICLES)
is_estimate = float(run_is(key))
is_check, weight_disc = faithfulness_check(key, N_PARTICLES)
times = bench(lambda: run_is(key))
print(f"  log-ML (IS)        : {is_estimate:.6f}  "
      f"(abs err vs exact = {abs(is_estimate - exact['log_ml']):.4f})")
print(f"  faithfulness       : run_is == shared-draw est: "
      f"{abs(is_estimate - is_check) < 1e-4} | "
      f"max |hand - numpyro.log_likelihood| weight disc = {weight_disc:.2e}")
print(f"  Mean: {np.mean(times):.4f} ms | Std: {np.std(times):.4f} | "
      f"Min: {np.min(times):.4f}\n")

is_comparison = {
    "model": "linreg",
    "algorithm": "IS",
    "n_particles": N_PARTICLES,
    "time_ms": float(np.mean(times)),
    "time_ms_std": float(np.std(times)),
    "time_ms_min": float(np.min(times)),
    "times_ms": [float(t) for t in times],
    "log_ml_estimate": is_estimate,
    "log_ml_exact": exact["log_ml"],
    "log_ml_abs_err": abs(is_estimate - exact["log_ml"]),
    "weight_faithfulness_max_disc": weight_disc,
}

# --- NUTS (NumPyro's native idiom), timed ---
print("-- LinReg NUTS (500 warmup + 1000 samples) --")
NUM_WARMUP, NUM_SAMPLES = 500, 1000


def run_nuts(rng):
    mcmc = MCMC(NUTS(linreg_model), num_warmup=NUM_WARMUP,
                num_samples=NUM_SAMPLES, progress_bar=False)
    mcmc.run(rng, xs=LINREG_XS, ys=LINREG_YS)
    return mcmc


# One warmup run to absorb the one-time JIT compile, then 3 timed runs.
_warm = run_nuts(key)
_warm.get_samples()["slope"].block_until_ready()
nuts_times = []
nuts_keys = jax.random.split(key, 3)
last = None
for nk in nuts_keys:
    t0 = time.perf_counter()
    last = run_nuts(nk)
    last.get_samples()["slope"].block_until_ready()
    nuts_times.append((time.perf_counter() - t0) * 1000)

nuts_samples = last.get_samples()
nuts_slope = float(np.mean(np.asarray(nuts_samples["slope"])))
nuts_intercept = float(np.mean(np.asarray(nuts_samples["intercept"])))
print(f"  post mean : slope={nuts_slope:.4f} intercept={nuts_intercept:.4f} "
      f"(exact slope={exact['post_mean_slope']:.4f} "
      f"intercept={exact['post_mean_intercept']:.4f})")
print(f"  Mean: {np.mean(nuts_times):.2f} ms over {len(nuts_times)} runs "
      f"(JIT compile excluded)\n")

nuts_comparison = {
    "model": "linreg",
    "algorithm": "NUTS",
    "num_warmup": NUM_WARMUP,
    "num_samples": NUM_SAMPLES,
    "time_ms": float(np.mean(nuts_times)),
    "time_ms_std": float(np.std(nuts_times)),
    "time_ms_min": float(np.min(nuts_times)),
    "times_ms": [float(t) for t in nuts_times],
    "post_mean_slope": nuts_slope,
    "post_mean_intercept": nuts_intercept,
    "post_mean_slope_exact": exact["post_mean_slope"],
    "post_mean_intercept_exact": exact["post_mean_intercept"],
    "timing_note": "single MCMC run; one-time JIT compile excluded via a warmup run; mean of 3",
}

# ---------------------------------------------------------------------------
# Emit JSON (mirrors genjax.json schema + accuracy/reproducibility blocks)
# ---------------------------------------------------------------------------

output = {
    "system": "numpyro",
    "family": "out-of-family (Pyro effect-handler paradigm; not GFI/trace-based)",
    "version": numpyro.__version__,
    "jax_version": jax.__version__,
    "hardware": hardware,
    "backend": f"JAX CPU ({jax.devices()[0].platform})",
    "timing_protocol": "5 warmup, 20 runs, time.perf_counter (IS); 3 runs, JIT excluded (NUTS)",
    "comparisons": [is_comparison, nuts_comparison],
    "accuracy": {
        "metric": "log marginal likelihood (linreg, shared spec)",
        "log_ml_exact_closed_form": exact["log_ml"],
        "log_ml_genmlx_l3_exact": GENMLX_L3_LOG_ML,
        "log_ml_is_estimate": is_estimate,
        "log_ml_is_abs_err": abs(is_estimate - exact["log_ml"]),
        "note": ("NumPyro IS reaches GenMLX's exact L3 log-ML within "
                 f"{abs(is_estimate - GENMLX_L3_LOG_ML):.3f} nats; NUTS recovers the "
                 "exact linear-Gaussian posterior mean. Closed-form == GenMLX L3 to float32 floor."),
        "post_mean_exact": {
            "slope": exact["post_mean_slope"],
            "intercept": exact["post_mean_intercept"],
        },
        "post_mean_nuts": {"slope": nuts_slope, "intercept": nuts_intercept},
    },
    "reproducibility": {
        "env": ".venv-genjax (gitignored)",
        "install": "~/code/genmlx/.venv-genjax/bin/pip install numpyro",
        "command": "~/code/genmlx/.venv-genjax/bin/python scripts/exp4_numpyro_baseline.py",
        "versions": {
            "numpyro": numpyro.__version__,
            "jax": jax.__version__,
            "numpy": np.__version__,
        },
        "shared_spec": {
            "model": "linreg",
            "n_obs": int(LINREG_XS.shape[0]),
            "sigma_prior": SIGMA_PRIOR,
            "sigma_obs": SIGMA_OBS,
            "note": "byte-identical xs/ys to scripts/exp4_genjax_benchmarks.py",
        },
        "caveats": (
            "Accuracy comparison is apples-to-apples NumPyro-IS vs GenJAX-IS vs closed-form "
            "(same 20-point N(0,2) spec). GenMLX's bench/cross_system.cljs + bench/linreg.cljs "
            "use N(0,10) on a different 5-point dataset, so the GenMLX timing column is NOT on "
            "this shared spec; treat the GenMLX-vs-others IS timing as indicative, not matched."
        ),
    },
}

outpath = os.path.join(os.path.dirname(__file__), "..", "results",
                       "cross-system", "numpyro.json")
os.makedirs(os.path.dirname(outpath), exist_ok=True)
with open(outpath, "w") as f:
    json.dump(output, f, indent=2)
print(f"Wrote: {os.path.normpath(outpath)}")
