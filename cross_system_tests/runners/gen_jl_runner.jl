# Gen.jl cross-system verification runner
# Reads JSON from stdin, writes results to stdout.
using Gen
using JSON
using SpecialFunctions: loggamma
using LinearAlgebra: logdet, inv

# --- Distribution log-prob dispatch ---

function eval_logprob(spec)
    dist_name = spec["dist"]
    value = spec["value"]
    params = spec["params"]

    try
        lp = if dist_name == "normal"
            logpdf(normal, value, params["mu"], params["sigma"])
        elseif dist_name == "uniform"
            logpdf(uniform, value, params["lo"], params["hi"])
        elseif dist_name == "bernoulli"
            logpdf(bernoulli, value == 1 ? true : false, params["p"])
        elseif dist_name == "beta"
            logpdf(beta, value, params["alpha"], params["beta"])
        elseif dist_name == "gamma"
            # Gen.jl uses (shape, SCALE), spec uses rate. scale = 1/rate
            logpdf(gamma, value, params["shape"], 1.0 / params["rate"])
        elseif dist_name == "exponential"
            logpdf(exponential, value, params["rate"])
        elseif dist_name == "laplace"
            logpdf(laplace, value, params["loc"], params["scale"])
        elseif dist_name == "cauchy"
            logpdf(cauchy, value, params["loc"], params["scale"])
        elseif dist_name == "poisson"
            logpdf(poisson, value, params["rate"])
        elseif dist_name == "binomial"
            logpdf(binom, value, Int(params["n"]), params["p"])
        elseif dist_name == "geometric"
            logpdf(geometric, value, params["p"])
        elseif dist_name == "lognormal"
            mu = params["mu"]; sigma = params["sigma"]
            -log(value) - 0.5*log(2*pi) - log(sigma) - 0.5*((log(value) - mu)/sigma)^2
        elseif dist_name == "inv_gamma"
            logpdf(inv_gamma, value, params["shape"], params["scale"])
        elseif dist_name == "student_t"
            df = params["df"]; loc = params["loc"]; scale = params["scale"]
            z = (value - loc) / scale
            loggamma((df + 1) / 2) - loggamma(df / 2) - 0.5 * log(df * pi) - log(scale) - ((df + 1) / 2) * log(1 + z^2 / df)
        elseif dist_name == "categorical"
            # Spec uses logits (0-indexed). Gen.jl categorical takes probs (1-indexed).
            logits = Float64.(params["logits"])
            probs = exp.(logits) ./ sum(exp.(logits))  # softmax
            value_1indexed = Int(value) + 1
            logpdf(categorical, value_1indexed, probs)
        elseif dist_name == "dirichlet"
            alpha = Float64.(params["alpha"])
            val = Float64.(value)
            logpdf(dirichlet, val, alpha)
        elseif dist_name == "mvn"
            mu = Float64.(params["mu"])
            # JSON nested array → Julia matrix (each row is a row of cov)
            cov_rows = params["cov"]
            cov = Float64.(hcat([Float64.(row) for row in cov_rows]...)')
            val = Float64.(value)
            logpdf(mvnormal, val, mu, cov)
        else
            return Dict("id" => spec["id"], "error" => "unsupported dist: $dist_name")
        end
        Dict("id" => spec["id"], "logprob" => lp)
    catch e
        Dict("id" => spec["id"], "error" => string(e))
    end
end

# --- GFI models ---

@gen function single_normal_model()
    x = @trace(normal(0.0, 1.0), :x)
    return x
end

@gen function two_normals_model()
    mu = @trace(normal(0.0, 10.0), :mu)
    x = @trace(normal(mu, 1.0), :x)
    return x
end

@gen function beta_bernoulli_model()
    p = @trace(beta(2.0, 2.0), :p)
    x = @trace(bernoulli(p), :x)
    return x
end

@gen function linear_regression_model(xs::Vector{Float64})
    slope = @trace(normal(0.0, 10.0), :slope)
    intercept = @trace(normal(0.0, 10.0), :intercept)
    for (j, x) in enumerate(xs)
        @trace(normal(slope * x + intercept, 1.0), Symbol("y$(j-1)"))
    end
    return slope
end

# Phase 1: single_gaussian (alias — same as single_normal but named differently in specs)
@gen function single_gaussian_model()
    x = @trace(normal(0.0, 1.0), :x)
    return x
end

# Phase 1: mixed model (bernoulli + branching)
@gen function mixed_model()
    coin = @trace(bernoulli(0.5), :coin)
    if coin
        x = @trace(normal(10.0, 1.0), :x)
    else
        x = @trace(normal(0.0, 1.0), :x)
    end
    return x
end

# Phase 1: many_addresses model (mu + 10 obs)
@gen function many_addresses_model()
    mu = @trace(normal(0.0, 10.0), :mu)
    for i in 0:9
        @trace(normal(mu, 1.0), Symbol("y$i"))
    end
    return mu
end

# Phase 1: linear_regression_5 (5-observation variant)
@gen function linear_regression_5_model(xs::Vector{Float64})
    slope = @trace(normal(0.0, 10.0), :slope)
    intercept = @trace(normal(0.0, 10.0), :intercept)
    for (j, x) in enumerate(xs)
        @trace(normal(slope * x + intercept, 1.0), Symbol("y$(j-1)"))
    end
    return slope
end

MODEL_LOOKUP = Dict(
    "single_normal" => single_normal_model,
    "single_gaussian" => single_gaussian_model,
    "two_normals" => two_normals_model,
    "beta_bernoulli" => beta_bernoulli_model,
    "linear_regression" => linear_regression_model,
    "linear_regression_5" => linear_regression_5_model,
    "mixed" => mixed_model,
    "many_addresses" => many_addresses_model,
)

function make_choicemap(choices_dict)
    cm = Gen.choicemap()
    for (k, v) in choices_dict
        addr = Symbol(k)
        if v isa Bool || (v isa Number && (v == 0 || v == 1) && v isa Integer)
            cm[addr] = v == 1 ? true : v == 0 ? false : v
        else
            cm[addr] = Float64(v)
        end
    end
    return cm
end

function get_model_args(model_name, spec)
    raw_args = get(spec, "args", [])
    if model_name in ("linear_regression", "linear_regression_5")
        if isempty(raw_args)
            # Default args when spec omits them
            model_name == "linear_regression" ? ([1.0, 2.0, 3.0],) : ([1.0, 2.0, 3.0, 4.0, 5.0],)
        else
            (Float64.(raw_args[1]),)
        end
    else
        ()
    end
end

function fix_bool_choices!(cm, model_name, choices_dict)
    # Bernoulli returns Bool in Gen.jl — convert 0/1 integers only if key exists
    if model_name == "beta_bernoulli" && haskey(choices_dict, "x")
        cm[:x] = choices_dict["x"] == 1 ? true : false
    elseif model_name == "mixed" && haskey(choices_dict, "coin")
        cm[:coin] = choices_dict["coin"] == 1 ? true : false
    end
end

function eval_assess(spec)
    model_name = spec["model"]
    model = MODEL_LOOKUP[model_name]
    args = get_model_args(model_name, spec)
    cm = make_choicemap(spec["choices"])

    try
        fix_bool_choices!(cm, model_name, spec["choices"])
        (weight, retval) = assess(model, args, cm)
        Dict("id" => spec["id"], "weight" => weight)
    catch e
        Dict("id" => spec["id"], "error" => string(e))
    end
end

function eval_generate(spec)
    model_name = spec["model"]
    model = MODEL_LOOKUP[model_name]
    args = get_model_args(model_name, spec)
    cm = make_choicemap(spec["constraints"])

    try
        fix_bool_choices!(cm, model_name, spec["constraints"])
        (trace, weight) = generate(model, args, cm)
        score = get_score(trace)
        Dict("id" => spec["id"], "weight" => weight, "score" => score)
    catch e
        Dict("id" => spec["id"], "error" => string(e))
    end
end

# --- Phase 1: Score Decomposition ---

function compute_component_logprob(comp)
    dist_name = comp["dist"]
    value = comp["value"]
    params = comp["params"]

    if dist_name == "normal"
        logpdf(normal, Float64(value), Float64(params["mu"]), Float64(params["sigma"]))
    elseif dist_name == "bernoulli"
        logpdf(bernoulli, value == 1 ? true : false, Float64(params["p"]))
    elseif dist_name == "beta"
        logpdf(beta, Float64(value), Float64(params["alpha"]), Float64(params["beta"]))
    else
        error("unsupported component dist: $dist_name")
    end
end

function eval_score_decomposition(spec)
    model_name = spec["model"]
    model = MODEL_LOOKUP[model_name]
    args = get_model_args(model_name, spec)
    cm = make_choicemap(spec["choices"])

    try
        fix_bool_choices!(cm, model_name, spec["choices"])
        (trace, _) = generate(model, args, cm)
        total_score = get_score(trace)

        components = Dict{String, Float64}()
        for (addr, comp) in spec["expected_components"]
            lp = compute_component_logprob(comp)
            components[addr] = lp
        end

        sum_components = sum(values(components))

        Dict("id" => spec["id"],
             "total_score" => total_score,
             "components" => components,
             "sum_components" => sum_components)
    catch e
        Dict("id" => spec["id"], "error" => string(e))
    end
end

# --- Phase 2: Update ---

function eval_update(spec)
    model_name = spec["model"]
    model = MODEL_LOOKUP[model_name]
    args = get_model_args(model_name, spec)

    # Step 1: create initial trace via fully-constrained generate
    initial_cm = make_choicemap(spec["initial_choices"])
    fix_bool_choices!(initial_cm, model_name, spec["initial_choices"])

    try
        (trace, _) = generate(model, args, initial_cm)
        old_score = get_score(trace)

        # Step 2: update with new constraints
        update_cm = make_choicemap(spec["update_constraints"])
        fix_bool_choices!(update_cm, model_name, spec["update_constraints"])

        # For models with args: pass same args + UnknownChange argdiffs
        # For models with no args: pass empty args + empty argdiffs
        argdiffs = if model_name in ("linear_regression", "linear_regression_5")
            (UnknownChange(),)
        else
            ()
        end

        (new_trace, weight, _, _) = update(trace, args, argdiffs, update_cm)
        new_score = get_score(new_trace)

        Dict("id" => spec["id"],
             "old_score" => old_score,
             "new_score" => new_score,
             "weight" => weight)
    catch e
        Dict("id" => spec["id"], "error" => string(e))
    end
end

# --- Main: read stdin, write stdout ---

function main()
    input = JSON.parse(read(stdin, String))
    test_type = input["test_type"]
    results = []

    if test_type == "logprob"
        for spec in input["tests"]
            push!(results, eval_logprob(spec))
        end
    elseif test_type == "assess"
        for spec in input["assess_tests"]
            push!(results, eval_assess(spec))
        end
    elseif test_type == "generate"
        for spec in input["generate_tests"]
            push!(results, eval_generate(spec))
        end
    elseif test_type == "score_decomposition"
        for spec in input["score_decomposition_tests"]
            push!(results, eval_score_decomposition(spec))
        end
    elseif test_type == "update"
        for spec in input["tests"]
            push!(results, eval_update(spec))
        end
    else
        println(stderr, "Unknown test type: $test_type")
        exit(1)
    end

    output = Dict("system" => "gen_jl", "test_type" => test_type, "results" => results)
    JSON.print(stdout, output, 2)
    println()
end

main()
