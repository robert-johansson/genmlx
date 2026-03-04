#!/usr/bin/env julia
"""
Experiment 4: Gen.jl benchmarks for system comparison.

Models: LinReg, HMM, GMM — same data as GenMLX exp3.
Protocol: 5 warmup, 20 timed runs, @elapsed.
Output: results/exp4_system_comparison/genjl.json

Run: julia scripts/exp4_genjl_benchmarks.jl
"""

using Gen
using JSON
using Statistics

# ---------------------------------------------------------------------------
# Data (hardcoded from exp3 JSONs — byte-identical inputs)
# ---------------------------------------------------------------------------

# LinReg: 20 centered x-values, 20 y-values
const LINREG_XS = Float64[
    -2.5, -2.236842105263158, -1.973684210526316, -1.7105263157894737,
    -1.4473684210526316, -1.1842105263157896, -0.9210526315789473,
    -0.6578947368421053, -0.3947368421052633, -0.13157894736842124,
    0.1315789473684208, 0.3947368421052633, 0.6578947368421053,
    0.9210526315789473, 1.1842105263157894, 1.4473684210526319,
    1.7105263157894735, 1.973684210526316, 2.2368421052631575, 2.5
]

const LINREG_YS = Float64[
    -4.645086392760277, -2.232466120468943, -4.743588723634419,
    -1.3603573410134566, -3.226216689536446, -3.294092987713061,
    -1.7810833391390348, -1.0100273317412327, 0.3002335711529378,
    -0.06383435977132734, 0.1810731197658333, 1.4528234914729472,
    2.0677273100928257, 2.639107370062878, 2.240536426243029,
    3.3727551091854515, 4.502427813253904, 4.419489033912358,
    7.114258427368966, 5.38109839707613
]

# HMM: T=50, 2-state, emission means [-2, 2], sigma=1
const HMM_T = 50
const HMM_TRANS = [0.9 0.1; 0.1 0.9]
const HMM_INIT = [0.5, 0.5]
const HMM_MEANS = [-2.0, 2.0]
const HMM_SIGMA = 1.0

const HMM_YS = Float64[
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
]

# GMM: K=3, N=8, means [-4, 0, 4], sigma=1
const GMM_K = 3
const GMM_N = 8
const GMM_MEANS = [-4.0, 0.0, 4.0]
const GMM_SIGMA = 1.0
const GMM_WEIGHTS = [1.0/3.0, 1.0/3.0, 1.0/3.0]

const GMM_YS = Float64[
    0.8350437879562378, 4.188530474901199, -2.888857841491699,
    -4.104988642036915, 5.638246417045593, 0.10598962754011154,
    -2.8335559368133545, -4.310329109430313
]

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

function bench(f; warmup=5, runs=20)
    # Warmup
    for _ in 1:warmup
        f()
    end
    # Timed runs
    times = Float64[]
    for _ in 1:runs
        t = @elapsed f()
        push!(times, t * 1000)  # convert to ms
    end
    return times
end

# ---------------------------------------------------------------------------
# Model A: Linear Regression
# ---------------------------------------------------------------------------

@gen function linreg_model(xs::Vector{Float64})
    slope = {:slope} ~ normal(0.0, 2.0)
    intercept = {:intercept} ~ normal(0.0, 2.0)
    for j in 1:length(xs)
        {(:y, j)} ~ normal(slope * xs[j] + intercept, 1.0)
    end
    return slope
end

# Observations choicemap
function linreg_obs()
    cm = choicemap()
    for j in 1:length(LINREG_YS)
        cm[(:y, j)] = LINREG_YS[j]
    end
    return cm
end

# ---------------------------------------------------------------------------
# Model B: Hidden Markov Model (flat loop)
# ---------------------------------------------------------------------------

@gen function hmm_model(T::Int)
    # Draw initial state
    z_prev = {:z1} ~ categorical(HMM_INIT)
    {:y1} ~ normal(HMM_MEANS[z_prev], HMM_SIGMA)

    for t in 2:T
        trans_probs = HMM_TRANS[z_prev, :]
        z = {(:z, t)} ~ categorical(trans_probs)
        {(:y, t)} ~ normal(HMM_MEANS[z], HMM_SIGMA)
        z_prev = z
    end
end

function hmm_obs()
    cm = choicemap()
    cm[:y1] = HMM_YS[1]
    for t in 2:HMM_T
        cm[(:y, t)] = HMM_YS[t]
    end
    return cm
end

# ---------------------------------------------------------------------------
# Model C: Gaussian Mixture Model (flat loop)
# ---------------------------------------------------------------------------

@gen function gmm_model(ys::Vector{Float64})
    log_weights = log.(GMM_WEIGHTS)
    for i in 1:length(ys)
        z = {(:z, i)} ~ categorical(GMM_WEIGHTS)
        {(:y, i)} ~ normal(GMM_MEANS[z], GMM_SIGMA)
    end
end

function gmm_obs()
    cm = choicemap()
    for i in 1:GMM_N
        cm[(:y, i)] = GMM_YS[i]
    end
    return cm
end

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

println("\n=== Gen.jl System Comparison Benchmarks ===")
println("  Gen.jl v$(pkgversion(Gen))")
println("  Julia v$(VERSION)")
println("  CPU: Apple M2")
println("  Protocol: 5 warmup, 20 timed runs")
println()

comparisons = []

# --- LinReg IS (N=1000) ---
println("-- LinReg IS (N=1000) --")
obs_lr = linreg_obs()

times_lr_is = bench() do
    importance_resampling(linreg_model, (LINREG_XS,), obs_lr, 1000)
end

push!(comparisons, Dict(
    "model" => "linreg",
    "algorithm" => "IS",
    "n_particles" => 1000,
    "time_ms" => mean(times_lr_is),
    "time_ms_std" => std(times_lr_is),
    "time_ms_min" => minimum(times_lr_is),
    "times_ms" => times_lr_is,
))
println("  Mean: $(round(mean(times_lr_is), digits=2)) ms")
println("  Std:  $(round(std(times_lr_is), digits=2)) ms")
println("  Min:  $(round(minimum(times_lr_is), digits=2)) ms")

# --- LinReg MH (5000 samples) ---
println("\n-- LinReg MH (5000 samples) --")

function linreg_mh()
    trace, _ = generate(linreg_model, (LINREG_XS,), obs_lr)
    sel = select(:slope, :intercept)
    for _ in 1:5000
        trace, _ = mh(trace, sel)
    end
    return trace
end

times_lr_mh = bench(linreg_mh)

push!(comparisons, Dict(
    "model" => "linreg",
    "algorithm" => "MH",
    "n_samples" => 5000,
    "time_ms" => mean(times_lr_mh),
    "time_ms_std" => std(times_lr_mh),
    "time_ms_min" => minimum(times_lr_mh),
    "times_ms" => times_lr_mh,
))
println("  Mean: $(round(mean(times_lr_mh), digits=2)) ms")
println("  Std:  $(round(std(times_lr_mh), digits=2)) ms")
println("  Min:  $(round(minimum(times_lr_mh), digits=2)) ms")

# --- HMM IS (N=1000) ---
println("\n-- HMM IS (N=1000) --")
obs_hmm = hmm_obs()

times_hmm_is = bench() do
    importance_resampling(hmm_model, (HMM_T,), obs_hmm, 1000)
end

push!(comparisons, Dict(
    "model" => "hmm",
    "algorithm" => "IS",
    "n_particles" => 1000,
    "time_ms" => mean(times_hmm_is),
    "time_ms_std" => std(times_hmm_is),
    "time_ms_min" => minimum(times_hmm_is),
    "times_ms" => times_hmm_is,
))
println("  Mean: $(round(mean(times_hmm_is), digits=2)) ms")
println("  Std:  $(round(std(times_hmm_is), digits=2)) ms")
println("  Min:  $(round(minimum(times_hmm_is), digits=2)) ms")

# --- HMM SMC (N=100, T=50) ---
println("\n-- HMM SMC (N=100, T=50) --")

function hmm_smc()
    # Build per-step observation sequences
    obs_first = choicemap()
    obs_first[:z1] = nothing  # no constraint on z
    obs_first[:y1] = HMM_YS[1]

    # Initialize with first step
    state = initialize_particle_filter(hmm_model, (1,),
        choicemap((:y1, HMM_YS[1])), 100)

    # Extend for each subsequent step
    for t in 2:HMM_T
        obs_t = choicemap()
        obs_t[(:y, t)] = HMM_YS[t]

        maybe_resample!(state, ess_threshold=50.0)
        particle_filter_step!(state, (t,), (UnknownChange(),),
            obs_t)
    end
    return log_ml_estimate(state)
end

# Quick test first
try
    hmm_smc()
    times_hmm_smc = bench(hmm_smc)

    push!(comparisons, Dict(
        "model" => "hmm",
        "algorithm" => "SMC",
        "n_particles" => 100,
        "n_steps" => HMM_T,
        "time_ms" => mean(times_hmm_smc),
        "time_ms_std" => std(times_hmm_smc),
        "time_ms_min" => minimum(times_hmm_smc),
        "times_ms" => times_hmm_smc,
    ))
    println("  Mean: $(round(mean(times_hmm_smc), digits=2)) ms")
    println("  Std:  $(round(std(times_hmm_smc), digits=2)) ms")
    println("  Min:  $(round(minimum(times_hmm_smc), digits=2)) ms")
catch e
    println("  SMC SKIPPED: $e")
end

# --- GMM IS (N=1000) ---
println("\n-- GMM IS (N=1000) --")
obs_gmm = gmm_obs()

times_gmm_is = bench() do
    importance_resampling(gmm_model, (GMM_YS,), obs_gmm, 1000)
end

push!(comparisons, Dict(
    "model" => "gmm",
    "algorithm" => "IS",
    "n_particles" => 1000,
    "time_ms" => mean(times_gmm_is),
    "time_ms_std" => std(times_gmm_is),
    "time_ms_min" => minimum(times_gmm_is),
    "times_ms" => times_gmm_is,
))
println("  Mean: $(round(mean(times_gmm_is), digits=2)) ms")
println("  Std:  $(round(std(times_gmm_is), digits=2)) ms")
println("  Min:  $(round(minimum(times_gmm_is), digits=2)) ms")

# ---------------------------------------------------------------------------
# Write JSON output
# ---------------------------------------------------------------------------

output = Dict(
    "system" => "genjl",
    "version" => string(pkgversion(Gen)),
    "julia_version" => string(VERSION),
    "hardware" => "Apple M2",
    "backend" => "CPU",
    "timing_protocol" => "5 warmup, 20 runs, @elapsed",
    "comparisons" => comparisons,
)

outpath = joinpath(@__DIR__, "..", "results", "exp4_system_comparison", "genjl.json")
mkpath(dirname(outpath))
open(outpath, "w") do f
    JSON.print(f, output, 2)
end
println("\nWrote: $outpath")
