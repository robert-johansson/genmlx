"""
GenJAX benchmark — comprehensive comparison with GenMLX.

Models:
  A: Gaussian conjugate (4 sites: mu, y0, y1, y2)
  B: Linear regression  (11 sites: slope, intercept, y0..y8)
  C: Many parameters    (52 sites: z0..z49, obs_mean, obs_var)

Sections:
  1. GFI primitives: simulate, generate (10 calls)
  2. Vectorized importance sampling (N=100, N=1000)
  3. MCMC single chain (200 steps): MH, MALA, HMC
  4. Vectorized MCMC (10 chains, 200 steps): MH, MALA, HMC
  5. Scaling test (52-site model)

Protocol: 3 warmup, median of 7 runs, time.perf_counter()
Sync: jax.block_until_ready() on all outputs

Run: pixi run python test/benchmark/genjax_benchmark.py
"""

import os
import sys
import time
import json
import numpy as np
import jax
import jax.numpy as jnp
from genjax import gen, normal, sel, Const, const, seed, init, chain, mh, mala
from genjax.inference import hmc


# ---------------------------------------------------------------------------
# Timing — 3 warmup, median of 7 runs
# ---------------------------------------------------------------------------

def bench(f, warmup=3, runs=7):
    """Run f with warmup, return median time in ms."""
    for _ in range(warmup):
        result = f()
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = f()
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    times.sort()
    return times[runs // 2]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Model A: Gaussian Conjugate (4 sites)
# mu ~ N(0, 10), y_i ~ N(mu, 1) for i=0,1,2
@gen
def model_a():
    mu = normal(0.0, 10.0) @ "mu"
    normal(mu, 1.0) @ "y0"
    normal(mu, 1.0) @ "y1"
    normal(mu, 1.0) @ "y2"
    return mu

obs_a = {"y0": 3.0, "y1": 3.1, "y2": 2.9}

# Model B: Linear Regression (11 sites)
# slope ~ N(0,10), intercept ~ N(0,10), y_j ~ N(slope*x_j + intercept, 1)
@gen
def model_b(xs):
    slope = normal(0.0, 10.0) @ "slope"
    intercept = normal(0.0, 10.0) @ "intercept"
    for j in range(len(xs)):
        normal(slope * xs[j] + intercept, 1.0) @ f"y{j}"
    return slope

xs_b = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
obs_b = {f"y{i}": v for i, v in enumerate(
    [3.1, 5.2, 6.9, 9.1, 10.8, 12.9, 15.1, 17.0, 19.2])}
full_b = {"slope": 2.0, "intercept": 1.0, **obs_b}

# Model C: Many Parameters (52 sites)
# z_i ~ N(0, 1) for i=0..49, obs_mean ~ N(mean(z), 0.1), obs_var ~ N(var(z), 0.1)
@gen
def model_c():
    zs = []
    for i in range(50):
        z = normal(0.0, 1.0) @ f"z{i}"
        zs.append(z)
    z_arr = jnp.array(zs)
    mean_z = jnp.mean(z_arr)
    var_z = jnp.var(z_arr)
    normal(mean_z, 0.1) @ "obs_mean"
    normal(var_z, 0.1) @ "obs_var"
    return mean_z

obs_c = {"obs_mean": 0.0, "obs_var": 1.0}


# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

print("\n=== GenJAX Comprehensive Benchmark ===")
print(f"  Runtime: JAX {jax.__version__} ({jax.devices()[0].platform})")
print(f"  Protocol: 3 warmup, median of 7 runs, time.perf_counter()")
print()

results = {}

# ---------------------------------------------------------------------------
# 1. GFI Primitives: simulate, generate (10 calls each)
# ---------------------------------------------------------------------------

print("=" * 60)
print("SECTION 1: GFI Primitives (10 calls each)")
print("=" * 60)

# -- Simulate --
print("\n-- Simulate (10 calls) --")

def sim_a():
    for _ in range(10):
        model_a.simulate()

def sim_b():
    for _ in range(10):
        model_b.simulate(xs_b)

def sim_c():
    for _ in range(10):
        model_c.simulate()

r = bench(sim_a)
print(f"  Model A (4-site):   {r:.3f}ms")
results["simulate_a"] = r

r = bench(sim_b)
print(f"  Model B (11-site):  {r:.3f}ms")
results["simulate_b"] = r

r = bench(sim_c)
print(f"  Model C (52-site):  {r:.3f}ms")
results["simulate_c"] = r

# -- Generate --
print("\n-- Generate (10 calls) --")

def gen_a():
    for _ in range(10):
        model_a.generate(obs_a)

def gen_b():
    for _ in range(10):
        model_b.generate(full_b, xs_b)

def gen_c():
    for _ in range(10):
        model_c.generate(obs_c)

r = bench(gen_a)
print(f"  Model A (4-site):   {r:.3f}ms")
results["generate_a"] = r

r = bench(gen_b)
print(f"  Model B (11-site):  {r:.3f}ms")
results["generate_b"] = r

r = bench(gen_c)
print(f"  Model C (52-site):  {r:.3f}ms")
results["generate_c"] = r


# ---------------------------------------------------------------------------
# 2. Vectorized Importance Sampling
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SECTION 2: Vectorized Importance Sampling")
print("=" * 60)

for N in [100, 1000]:
    print(f"\n-- IS N={N} --")

    def run_is_a(n=N):
        particles = seed(lambda: init(model_a, (), const(n), obs_a))(jax.random.key(0))
        lml = particles.log_marginal_likelihood()
        jax.block_until_ready(lml)
        return lml

    def run_is_b(n=N):
        particles = seed(lambda: init(model_b, (xs_b,), const(n), obs_b))(jax.random.key(0))
        lml = particles.log_marginal_likelihood()
        jax.block_until_ready(lml)
        return lml

    r = bench(run_is_a)
    print(f"  Model A (4-site):   {r:.3f}ms")
    results[f"is_{N}_a"] = r

    r = bench(run_is_b)
    print(f"  Model B (11-site):  {r:.3f}ms")
    results[f"is_{N}_b"] = r


# ---------------------------------------------------------------------------
# 3. MCMC Single Chain (200 steps)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SECTION 3: MCMC Single Chain (200 steps)")
print("=" * 60)

# -- MH --
print("\n-- MH 200 steps --")

def run_mh_chain(model, obs, selection, n_steps=200, *args):
    init_trace, _ = model.generate(obs, *args)
    def kernel(trace):
        return mh(trace, selection)
    runner = seed(chain(kernel))
    result = runner(jax.random.key(0), init_trace, n_steps=Const(n_steps))
    return result

r = bench(lambda: run_mh_chain(model_a, obs_a, sel("mu")))
print(f"  Model A (4-site):   {r:.2f}ms")
results["mh200_a"] = r

r = bench(lambda: run_mh_chain(model_b, full_b, sel("slope") | sel("intercept"), 200, xs_b))
print(f"  Model B (11-site):  {r:.2f}ms")
results["mh200_b"] = r

# -- MH pre-compiled (reuse runner) --
print("\n-- MH 200 steps (pre-compiled runner) --")

init_trace_a, _ = model_a.generate(obs_a)
def mh_kernel_a(trace):
    return mh(trace, sel("mu"))
mh_runner_a = seed(chain(mh_kernel_a))
# Warmup JIT
_ = mh_runner_a(jax.random.key(0), init_trace_a, n_steps=Const(200))
_ = mh_runner_a(jax.random.key(0), init_trace_a, n_steps=Const(200))

r = bench(lambda: mh_runner_a(jax.random.key(0), init_trace_a, n_steps=Const(200)))
print(f"  Model A (4-site):   {r:.2f}ms")
results["mh200_a_precompiled"] = r

init_trace_b, _ = model_b.generate(full_b, xs_b)
def mh_kernel_b(trace):
    return mh(trace, sel("slope") | sel("intercept"))
mh_runner_b = seed(chain(mh_kernel_b))
_ = mh_runner_b(jax.random.key(0), init_trace_b, n_steps=Const(200))
_ = mh_runner_b(jax.random.key(0), init_trace_b, n_steps=Const(200))

r = bench(lambda: mh_runner_b(jax.random.key(0), init_trace_b, n_steps=Const(200)))
print(f"  Model B (11-site):  {r:.2f}ms")
results["mh200_b_precompiled"] = r

# -- MALA --
print("\n-- MALA 200 steps --")

def run_mala_chain(model, obs, selection, step_size=0.1, n_steps=200, *args):
    init_trace, _ = model.generate(obs, *args)
    def kernel(trace):
        return mala(trace, selection, step_size)
    runner = seed(chain(kernel))
    result = runner(jax.random.key(0), init_trace, n_steps=Const(n_steps))
    return result

r = bench(lambda: run_mala_chain(model_a, obs_a, sel("mu"), 0.1))
print(f"  Model A (4-site):   {r:.2f}ms")
results["mala200_a"] = r

r = bench(lambda: run_mala_chain(model_b, full_b, sel("slope") | sel("intercept"), 0.1, 200, xs_b))
print(f"  Model B (11-site):  {r:.2f}ms")
results["mala200_b"] = r

# -- MALA pre-compiled --
print("\n-- MALA 200 steps (pre-compiled runner) --")

def mala_kernel_a(trace):
    return mala(trace, sel("mu"), 0.1)
mala_runner_a = seed(chain(mala_kernel_a))
_ = mala_runner_a(jax.random.key(0), init_trace_a, n_steps=Const(200))
_ = mala_runner_a(jax.random.key(0), init_trace_a, n_steps=Const(200))

r = bench(lambda: mala_runner_a(jax.random.key(0), init_trace_a, n_steps=Const(200)))
print(f"  Model A (4-site):   {r:.2f}ms")
results["mala200_a_precompiled"] = r

def mala_kernel_b(trace):
    return mala(trace, sel("slope") | sel("intercept"), 0.1)
mala_runner_b = seed(chain(mala_kernel_b))
_ = mala_runner_b(jax.random.key(0), init_trace_b, n_steps=Const(200))
_ = mala_runner_b(jax.random.key(0), init_trace_b, n_steps=Const(200))

r = bench(lambda: mala_runner_b(jax.random.key(0), init_trace_b, n_steps=Const(200)))
print(f"  Model B (11-site):  {r:.2f}ms")
results["mala200_b_precompiled"] = r

# -- HMC --
print("\n-- HMC 200 steps, L=10 --")

def run_hmc_chain(model, obs, selection, step_size=0.01, n_leapfrog=10, n_steps=200, *args):
    init_trace, _ = model.generate(obs, *args)
    def kernel(trace):
        return hmc(trace, selection, step_size=step_size, n_steps=n_leapfrog)
    runner = seed(chain(kernel))
    result = runner(jax.random.key(0), init_trace, n_steps=Const(n_steps))
    return result

r = bench(lambda: run_hmc_chain(model_a, obs_a, sel("mu"), 0.01, 10, 200))
print(f"  Model A (4-site):   {r:.2f}ms")
results["hmc200_a"] = r

r = bench(lambda: run_hmc_chain(model_b, full_b, sel("slope") | sel("intercept"), 0.01, 10, 200, xs_b))
print(f"  Model B (11-site):  {r:.2f}ms")
results["hmc200_b"] = r

# -- HMC pre-compiled --
print("\n-- HMC 200 steps, L=10 (pre-compiled runner) --")

def hmc_kernel_a(trace):
    return hmc(trace, sel("mu"), step_size=0.01, n_steps=10)
hmc_runner_a = seed(chain(hmc_kernel_a))
_ = hmc_runner_a(jax.random.key(0), init_trace_a, n_steps=Const(200))
_ = hmc_runner_a(jax.random.key(0), init_trace_a, n_steps=Const(200))

r = bench(lambda: hmc_runner_a(jax.random.key(0), init_trace_a, n_steps=Const(200)))
print(f"  Model A (4-site):   {r:.2f}ms")
results["hmc200_a_precompiled"] = r

def hmc_kernel_b(trace):
    return hmc(trace, sel("slope") | sel("intercept"), step_size=0.01, n_steps=10)
hmc_runner_b = seed(chain(hmc_kernel_b))
_ = hmc_runner_b(jax.random.key(0), init_trace_b, n_steps=Const(200))
_ = hmc_runner_b(jax.random.key(0), init_trace_b, n_steps=Const(200))

r = bench(lambda: hmc_runner_b(jax.random.key(0), init_trace_b, n_steps=Const(200)))
print(f"  Model B (11-site):  {r:.2f}ms")
results["hmc200_b_precompiled"] = r


# ---------------------------------------------------------------------------
# 4. Vectorized MCMC (10 chains, 200 steps)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SECTION 4: Vectorized MCMC (10 chains, 200 steps)")
print("=" * 60)

N_CHAINS = 10

# -- Multi-chain MH --
print(f"\n-- MH 200 steps, {N_CHAINS} chains --")

def run_mh_multi_a():
    init_trace, _ = model_a.generate(obs_a)
    runner = seed(chain(mh_kernel_a))
    result = runner(jax.random.key(0), init_trace,
                    n_steps=Const(200), n_chains=Const(N_CHAINS))
    return result

def run_mh_multi_b():
    init_trace, _ = model_b.generate(full_b, xs_b)
    runner = seed(chain(mh_kernel_b))
    result = runner(jax.random.key(0), init_trace,
                    n_steps=Const(200), n_chains=Const(N_CHAINS))
    return result

r = bench(run_mh_multi_a)
print(f"  Model A (4-site):   {r:.2f}ms")
results["mh200_10chains_a"] = r

r = bench(run_mh_multi_b)
print(f"  Model B (11-site):  {r:.2f}ms")
results["mh200_10chains_b"] = r

# -- Multi-chain MALA --
print(f"\n-- MALA 200 steps, {N_CHAINS} chains --")

def run_mala_multi_a():
    init_trace, _ = model_a.generate(obs_a)
    runner = seed(chain(mala_kernel_a))
    result = runner(jax.random.key(0), init_trace,
                    n_steps=Const(200), n_chains=Const(N_CHAINS))
    return result

def run_mala_multi_b():
    init_trace, _ = model_b.generate(full_b, xs_b)
    runner = seed(chain(mala_kernel_b))
    result = runner(jax.random.key(0), init_trace,
                    n_steps=Const(200), n_chains=Const(N_CHAINS))
    return result

r = bench(run_mala_multi_a)
print(f"  Model A (4-site):   {r:.2f}ms")
results["mala200_10chains_a"] = r

r = bench(run_mala_multi_b)
print(f"  Model B (11-site):  {r:.2f}ms")
results["mala200_10chains_b"] = r

# -- Multi-chain HMC --
print(f"\n-- HMC 200 steps, L=10, {N_CHAINS} chains --")

def run_hmc_multi_a():
    init_trace, _ = model_a.generate(obs_a)
    runner = seed(chain(hmc_kernel_a))
    result = runner(jax.random.key(0), init_trace,
                    n_steps=Const(200), n_chains=Const(N_CHAINS))
    return result

def run_hmc_multi_b():
    init_trace, _ = model_b.generate(full_b, xs_b)
    runner = seed(chain(hmc_kernel_b))
    result = runner(jax.random.key(0), init_trace,
                    n_steps=Const(200), n_chains=Const(N_CHAINS))
    return result

r = bench(run_hmc_multi_a)
print(f"  Model A (4-site):   {r:.2f}ms")
results["hmc200_10chains_a"] = r

r = bench(run_hmc_multi_b)
print(f"  Model B (11-site):  {r:.2f}ms")
results["hmc200_10chains_b"] = r


# ---------------------------------------------------------------------------
# 5. Scaling Test (52-site model)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SECTION 5: Scaling Test (52-site model)")
print("=" * 60)

# HMC on 5 selected parameters from model C
selection_c = sel("z0") | sel("z1") | sel("z2") | sel("z3") | sel("z4")

print("\n-- HMC 50 steps, L=5 on 5 params --")

def run_hmc_c():
    init_trace, _ = model_c.generate(obs_c)
    def kernel(trace):
        return hmc(trace, selection_c, step_size=0.01, n_steps=5)
    runner = seed(chain(kernel))
    result = runner(jax.random.key(0), init_trace, n_steps=Const(50))
    return result

r = bench(run_hmc_c)
print(f"  Model C (52-site):  {r:.2f}ms")
results["hmc50_c_scaling"] = r


# ---------------------------------------------------------------------------
# 6. Correctness Checks
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SECTION 6: Correctness Checks")
print("=" * 60)

correctness = {}

# Model A: Gaussian conjugate posterior
# Prior: mu ~ N(0, 10), Likelihood: y_i ~ N(mu, 1), obs = [3.0, 3.1, 2.9]
# Posterior: mu ~ N(2.990, 0.577^2)
print("\n-- Model A: Gaussian Conjugate Posterior --")

try:
    # Run longer chain for correctness
    def mh_kernel_a_correct(trace):
        return mh(trace, sel("mu"))
    runner_a = seed(chain(mh_kernel_a_correct))
    init_a, _ = model_a.generate(obs_a)
    result_a = runner_a(jax.random.key(42), init_a,
                        n_steps=Const(2000), burn_in=Const(500))
    # Extract mu samples
    mu_samples = jax.device_get(result_a.traces.get_choices()["mu"])
    mu_mean = float(np.mean(mu_samples))
    err_a = abs(mu_mean - 2.990)
    pass_a = err_a < 0.3
    print(f"  E[mu] = {mu_mean:.3f} (expected ~2.990, error={err_a:.3f})")
    print(f"  RESULT: {'PASS' if pass_a else 'FAIL'}")
    correctness["model_a_mu_mean"] = mu_mean
    correctness["model_a_pass"] = pass_a
except Exception as e:
    print(f"  ERROR: {e}")
    correctness["model_a_pass"] = False

# Model B: Linear regression posterior
# Check slope ~2.0, intercept ~1.0
print("\n-- Model B: Linear Regression Posterior --")

try:
    def mala_kernel_b_correct(trace):
        return mala(trace, sel("slope") | sel("intercept"), 0.05)
    runner_b = seed(chain(mala_kernel_b_correct))
    init_b, _ = model_b.generate(full_b, xs_b)
    result_b = runner_b(jax.random.key(42), init_b,
                        n_steps=Const(2000), burn_in=Const(500))
    slope_samples = jax.device_get(result_b.traces.get_choices()["slope"])
    intercept_samples = jax.device_get(result_b.traces.get_choices()["intercept"])
    slope_mean = float(np.mean(slope_samples))
    intercept_mean = float(np.mean(intercept_samples))
    err_slope = abs(slope_mean - 2.0)
    err_intercept = abs(intercept_mean - 1.0)
    pass_b = err_slope < 0.5 and err_intercept < 0.5
    print(f"  E[slope] = {slope_mean:.3f} (expected ~2.0, error={err_slope:.3f})")
    print(f"  E[intercept] = {intercept_mean:.3f} (expected ~1.0, error={err_intercept:.3f})")
    print(f"  RESULT: {'PASS' if pass_b else 'FAIL'}")
    correctness["model_b_slope_mean"] = slope_mean
    correctness["model_b_intercept_mean"] = intercept_mean
    correctness["model_b_pass"] = pass_b
except Exception as e:
    print(f"  ERROR: {e}")
    correctness["model_b_pass"] = False

# Model C: z0 should be near 0
print("\n-- Model C: Many Parameters Posterior --")

try:
    def mh_kernel_c_correct(trace):
        return mh(trace, selection_c)
    runner_c = seed(chain(mh_kernel_c_correct))
    init_c, _ = model_c.generate(obs_c)
    result_c = runner_c(jax.random.key(42), init_c,
                        n_steps=Const(2000), burn_in=Const(500))
    z0_samples = jax.device_get(result_c.traces.get_choices()["z0"])
    z0_mean = float(np.mean(z0_samples))
    err_z0 = abs(z0_mean)
    pass_c = err_z0 < 0.5
    print(f"  E[z0] = {z0_mean:.3f} (expected ~0.0, error={err_z0:.3f})")
    print(f"  RESULT: {'PASS' if pass_c else 'FAIL'}")
    correctness["model_c_z0_mean"] = z0_mean
    correctness["model_c_pass"] = pass_c
except Exception as e:
    print(f"  ERROR: {e}")
    correctness["model_c_pass"] = False


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

output = {
    "framework": "GenJAX",
    "jax_version": jax.__version__,
    "device": str(jax.devices()[0].platform),
    "protocol": "3 warmup, median of 7 runs",
    "timings": results,
    "correctness": correctness,
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "genjax_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")

print("\n=== Benchmark complete ===")
