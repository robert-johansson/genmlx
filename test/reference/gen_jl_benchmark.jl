#!/usr/bin/env julia
#
# gen_jl_benchmark.jl — Benchmark Gen.jl operations for speed comparison with GenMLX.
#
# Run:     julia test/reference/gen_jl_benchmark.jl
# Output:  test/reference/gen_jl_benchmark.json
# Requires: Julia 1.9+ with Gen.jl installed
#
# Protocol: 3 warmup runs, median of 7 measured runs, wall-clock ms via @elapsed.
# No BenchmarkTools — we want wall-clock ms comparable to js/Date.now.

using Gen
using JSON
using Dates

# =============================================================================
# Timing infrastructure
# =============================================================================

function bench(f; warmup=3, runs=7)
    for _ in 1:warmup
        f()
    end
    times_ms = Float64[]
    for _ in 1:runs
        t = @elapsed f()
        push!(times_ms, t * 1000.0)  # seconds -> ms
    end
    sort!(times_ms)
    median_ms = times_ms[div(runs, 2) + 1]
    return (times_ms=times_ms, median_ms=median_ms)
end

# =============================================================================
# Model definitions (matching gen_jl_reference.jl)
# =============================================================================

# Model 1: Single Gaussian (1 site)
@gen function model1_single_gaussian()
    x = {:x} ~ normal(0, 1)
    return x
end

# Model 2: Linear Regression (5 observation sites + 2 latents = 7 sites)
@gen function model2_linear_regression(xs::Vector{Float64})
    slope = {:slope} ~ normal(0, 10)
    intercept = {:intercept} ~ normal(0, 10)
    for (j, x) in enumerate(xs)
        {Symbol("y$(j-1)")} ~ normal(slope * x + intercept, 1)
    end
    return slope
end

# Model 3: Mixed discrete/continuous (2 sites)
@gen function model3_mixed()
    coin = {:coin} ~ bernoulli(0.5)
    if coin
        x = {:x} ~ normal(10, 1)
    else
        x = {:x} ~ normal(0, 1)
    end
    return x
end

# Model 4: Map combinator kernel + Map (3 elements)
@gen function model4_kernel(x::Float64)
    y = {:y} ~ normal(x, 1)
    return y
end
model4_map = Map(model4_kernel)

# Model 5: Unfold combinator (3 steps)
@gen function model5_kernel(t::Int, state::Float64)
    x = {:x} ~ normal(state, 1)
    return x
end
model5_unfold = Unfold(model5_kernel)

# Model 6: Many addresses (11 sites)
@gen function model6_many_addresses()
    mu = {:mu} ~ normal(0, 10)
    for i in 1:10
        {Symbol("y$(i-1)")} ~ normal(mu, 1)
    end
    return mu
end

# =============================================================================
# Constraint builders
# =============================================================================

function constraints_single_gaussian()
    choicemap((:x, 0.5))
end

function constraints_linear_regression()
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    c = choicemap(
        (:slope, 2.0),
        (:intercept, 1.0),
        (:y0, 3.1), (:y1, 5.2), (:y2, 6.9), (:y3, 9.1), (:y4, 10.8)
    )
    return (xs, c)
end

function observations_linear_regression()
    # Observations only (not latents) for inference
    choicemap((:y0, 3.1), (:y1, 5.2), (:y2, 6.9), (:y3, 9.1), (:y4, 10.8))
end

function constraints_mixed()
    choicemap((:coin, true), (:x, 10.5))
end

function constraints_map()
    choicemap(
        (1 => :y, 1.5),
        (2 => :y, 2.5),
        (3 => :y, 3.5)
    )
end

function constraints_unfold()
    choicemap(
        (1 => :x, 0.5),
        (2 => :x, 1.0),
        (3 => :x, 1.5)
    )
end

function constraints_many_addresses()
    c = choicemap((:mu, 2.0))
    for i in 0:9
        c[Symbol("y$i")] = Float64(i) * 0.5
    end
    return c
end

function observations_many_addresses()
    c = choicemap()
    for i in 0:9
        c[Symbol("y$i")] = Float64(i) * 0.5
    end
    return c
end

# =============================================================================
# Benchmark runners
# =============================================================================

results = Dict[]

function add_result!(model_name, operation, iterations, times_ms, median_ms)
    push!(results, Dict(
        "model" => model_name,
        "operation" => operation,
        "iterations" => iterations,
        "times_ms" => times_ms,
        "median_ms" => median_ms
    ))
    println("  $(model_name) / $(operation) [$(iterations)x]: $(round(median_ms, digits=2)) ms")
end

# ---------------------------------------------------------------------------
# simulate benchmarks (all 6 models)
# ---------------------------------------------------------------------------

println("\n=== Simulate benchmarks ===")

let
    r = bench(; warmup=3, runs=7) do
        for _ in 1:100
            simulate(model1_single_gaussian, ())
        end
    end
    add_result!("single_gaussian", "simulate", 100, r.times_ms, r.median_ms)
end

let
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    r = bench(; warmup=3, runs=7) do
        for _ in 1:100
            simulate(model2_linear_regression, (xs,))
        end
    end
    add_result!("linear_regression", "simulate", 100, r.times_ms, r.median_ms)
end

let
    r = bench(; warmup=3, runs=7) do
        for _ in 1:100
            simulate(model3_mixed, ())
        end
    end
    add_result!("mixed", "simulate", 100, r.times_ms, r.median_ms)
end

let
    xs = [1.0, 2.0, 3.0]
    r = bench(; warmup=3, runs=7) do
        for _ in 1:100
            simulate(model4_map, (xs,))
        end
    end
    add_result!("map_combinator", "simulate", 100, r.times_ms, r.median_ms)
end

let
    r = bench(; warmup=3, runs=7) do
        for _ in 1:100
            simulate(model5_unfold, (3, 0.0))
        end
    end
    add_result!("unfold_combinator", "simulate", 100, r.times_ms, r.median_ms)
end

let
    r = bench(; warmup=3, runs=7) do
        for _ in 1:100
            simulate(model6_many_addresses, ())
        end
    end
    add_result!("many_addresses", "simulate", 100, r.times_ms, r.median_ms)
end

# ---------------------------------------------------------------------------
# generate benchmarks (all 6 models, fully constrained)
# ---------------------------------------------------------------------------

println("\n=== Generate benchmarks ===")

let
    c = constraints_single_gaussian()
    r = bench(; warmup=3, runs=7) do
        for _ in 1:100
            generate(model1_single_gaussian, (), c)
        end
    end
    add_result!("single_gaussian", "generate", 100, r.times_ms, r.median_ms)
end

let
    (xs, c) = constraints_linear_regression()
    r = bench(; warmup=3, runs=7) do
        for _ in 1:100
            generate(model2_linear_regression, (xs,), c)
        end
    end
    add_result!("linear_regression", "generate", 100, r.times_ms, r.median_ms)
end

let
    c = constraints_mixed()
    r = bench(; warmup=3, runs=7) do
        for _ in 1:100
            generate(model3_mixed, (), c)
        end
    end
    add_result!("mixed", "generate", 100, r.times_ms, r.median_ms)
end

let
    xs = [1.0, 2.0, 3.0]
    c = constraints_map()
    r = bench(; warmup=3, runs=7) do
        for _ in 1:100
            generate(model4_map, (xs,), c)
        end
    end
    add_result!("map_combinator", "generate", 100, r.times_ms, r.median_ms)
end

let
    c = constraints_unfold()
    r = bench(; warmup=3, runs=7) do
        for _ in 1:100
            generate(model5_unfold, (3, 0.0), c)
        end
    end
    add_result!("unfold_combinator", "generate", 100, r.times_ms, r.median_ms)
end

let
    c = constraints_many_addresses()
    r = bench(; warmup=3, runs=7) do
        for _ in 1:100
            generate(model6_many_addresses, (), c)
        end
    end
    add_result!("many_addresses", "generate", 100, r.times_ms, r.median_ms)
end

# ---------------------------------------------------------------------------
# update benchmarks (models 1, 2, 6)
# ---------------------------------------------------------------------------

println("\n=== Update benchmarks ===")

let
    c = constraints_single_gaussian()
    trace, _ = generate(model1_single_gaussian, (), c)
    new_c = choicemap((:x, -0.5))
    r = bench(; warmup=3, runs=7) do
        for _ in 1:100
            update(trace, (), (UnknownChange(),), new_c)
        end
    end
    add_result!("single_gaussian", "update", 100, r.times_ms, r.median_ms)
end

let
    (xs, c) = constraints_linear_regression()
    trace, _ = generate(model2_linear_regression, (xs,), c)
    new_c = choicemap((:slope, 3.0))
    r = bench(; warmup=3, runs=7) do
        for _ in 1:100
            update(trace, (xs,), (UnknownChange(),), new_c)
        end
    end
    add_result!("linear_regression", "update", 100, r.times_ms, r.median_ms)
end

let
    c = constraints_many_addresses()
    trace, _ = generate(model6_many_addresses, (), c)
    new_c = choicemap((:mu, 3.0))
    r = bench(; warmup=3, runs=7) do
        for _ in 1:100
            update(trace, (), (UnknownChange(),), new_c)
        end
    end
    add_result!("many_addresses", "update", 100, r.times_ms, r.median_ms)
end

# ---------------------------------------------------------------------------
# importance_sampling benchmarks (all 6 models, 100 particles, 10 calls)
# ---------------------------------------------------------------------------

println("\n=== Importance Sampling benchmarks ===")

let
    obs = constraints_single_gaussian()
    r = bench(; warmup=3, runs=7) do
        for _ in 1:10
            traces = [generate(model1_single_gaussian, (), obs) for _ in 1:100]
        end
    end
    add_result!("single_gaussian", "importance_sampling", 10, r.times_ms, r.median_ms)
end

let
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    obs = observations_linear_regression()
    r = bench(; warmup=3, runs=7) do
        for _ in 1:10
            traces = [generate(model2_linear_regression, (xs,), obs) for _ in 1:100]
        end
    end
    add_result!("linear_regression", "importance_sampling", 10, r.times_ms, r.median_ms)
end

let
    obs = choicemap((:x, 10.5))
    r = bench(; warmup=3, runs=7) do
        for _ in 1:10
            traces = [generate(model3_mixed, (), obs) for _ in 1:100]
        end
    end
    add_result!("mixed", "importance_sampling", 10, r.times_ms, r.median_ms)
end

let
    xs = [1.0, 2.0, 3.0]
    obs = constraints_map()
    r = bench(; warmup=3, runs=7) do
        for _ in 1:10
            traces = [generate(model4_map, (xs,), obs) for _ in 1:100]
        end
    end
    add_result!("map_combinator", "importance_sampling", 10, r.times_ms, r.median_ms)
end

let
    obs = constraints_unfold()
    r = bench(; warmup=3, runs=7) do
        for _ in 1:10
            traces = [generate(model5_unfold, (3, 0.0), obs) for _ in 1:100]
        end
    end
    add_result!("unfold_combinator", "importance_sampling", 10, r.times_ms, r.median_ms)
end

let
    obs = observations_many_addresses()
    r = bench(; warmup=3, runs=7) do
        for _ in 1:10
            traces = [generate(model6_many_addresses, (), obs) for _ in 1:100]
        end
    end
    add_result!("many_addresses", "importance_sampling", 10, r.times_ms, r.median_ms)
end

# ---------------------------------------------------------------------------
# MH benchmarks (models 1, 2, 6 — 200 steps)
# ---------------------------------------------------------------------------

println("\n=== MH benchmarks ===")

let
    obs = constraints_single_gaussian()
    trace, _ = generate(model1_single_gaussian, (), obs)
    sel = select(:x)
    r = bench(; warmup=3, runs=7) do
        t = trace
        for _ in 1:200
            t, _ = mh(t, sel)
        end
    end
    add_result!("single_gaussian", "mh", 200, r.times_ms, r.median_ms)
end

let
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    obs = observations_linear_regression()
    trace, _ = generate(model2_linear_regression, (xs,), obs)
    sel = select(:slope, :intercept)
    r = bench(; warmup=3, runs=7) do
        t = trace
        for _ in 1:200
            t, _ = mh(t, sel)
        end
    end
    add_result!("linear_regression", "mh", 200, r.times_ms, r.median_ms)
end

let
    obs = observations_many_addresses()
    trace, _ = generate(model6_many_addresses, (), obs)
    sel = select(:mu)
    r = bench(; warmup=3, runs=7) do
        t = trace
        for _ in 1:200
            t, _ = mh(t, sel)
        end
    end
    add_result!("many_addresses", "mh", 200, r.times_ms, r.median_ms)
end

# ---------------------------------------------------------------------------
# HMC benchmarks (models 2, 6 — 50 steps, L=10)
# ---------------------------------------------------------------------------

println("\n=== HMC benchmarks ===")

# Gen.jl's built-in HMC kernel
let
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    obs = observations_linear_regression()
    trace, _ = generate(model2_linear_regression, (xs,), obs)
    sel = select(:slope, :intercept)
    r = bench(; warmup=3, runs=7) do
        t = trace
        for _ in 1:50
            t, _ = hmc(t, sel; L=10, eps=0.01)
        end
    end
    add_result!("linear_regression", "hmc", 50, r.times_ms, r.median_ms)
end

let
    obs = observations_many_addresses()
    trace, _ = generate(model6_many_addresses, (), obs)
    sel = select(:mu)
    r = bench(; warmup=3, runs=7) do
        t = trace
        for _ in 1:50
            t, _ = hmc(t, sel; L=10, eps=0.01)
        end
    end
    add_result!("many_addresses", "hmc", 50, r.times_ms, r.median_ms)
end


# =============================================================================
# Write JSON output
# =============================================================================

output = Dict(
    "metadata" => Dict(
        "generator" => "gen_jl_benchmark.jl",
        "gen_version" => string(pkgversion(Gen)),
        "julia_version" => string(VERSION),
        "date" => string(Dates.now()),
        "protocol" => Dict(
            "warmup" => 3,
            "runs" => 7,
            "timing" => "@elapsed (wall-clock seconds converted to ms)"
        )
    ),
    "benchmarks" => results
)

outpath = joinpath(@__DIR__, "gen_jl_benchmark.json")
open(outpath, "w") do f
    JSON.print(f, output, 2)
end

println("\n=== Summary ===")
println("Wrote $(outpath)")
println("Total benchmarks: $(length(results))")
for r in results
    println("  $(r["model"]) / $(r["operation"]): $(round(r["median_ms"], digits=2)) ms")
end
