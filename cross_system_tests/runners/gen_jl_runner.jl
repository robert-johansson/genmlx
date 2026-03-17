# Gen.jl cross-system verification runner
# Reads JSON from stdin, writes results to stdout.
using Gen
using JSON
using SpecialFunctions: loggamma, digamma
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

# --- Phase 3: Project + Regenerate ---

function make_selection(sel_spec)
    sel_type = sel_spec["type"]
    if sel_type == "addrs"
        select([Symbol(a) for a in sel_spec["addrs"]]...)
    elseif sel_type == "all"
        selectall()
    else  # "none"
        select()
    end
end

function eval_project(spec)
    model_name = spec["model"]
    model = MODEL_LOOKUP[model_name]
    args = get_model_args(model_name, spec)
    cm = make_choicemap(spec["choices"])

    try
        fix_bool_choices!(cm, model_name, spec["choices"])
        (trace, _) = generate(model, args, cm)
        sel = make_selection(spec["selection"])
        weight = project(trace, sel)
        score = get_score(trace)
        Dict("id" => spec["id"], "weight" => weight, "score" => score)
    catch e
        Dict("id" => spec["id"], "error" => string(e))
    end
end

function eval_regenerate(spec)
    model_name = spec["model"]
    model = MODEL_LOOKUP[model_name]
    args = get_model_args(model_name, spec)
    cm = make_choicemap(spec["choices"])

    try
        fix_bool_choices!(cm, model_name, spec["choices"])
        (trace, _) = generate(model, args, cm)
        old_score = get_score(trace)
        sel = make_selection(spec["selection"])
        (new_trace, weight) = regenerate(trace, sel)
        new_score = get_score(new_trace)
        Dict("id" => spec["id"], "old_score" => old_score, "new_score" => new_score, "weight" => weight)
    catch e
        Dict("id" => spec["id"], "error" => string(e))
    end
end

# --- Phase 4: Combinator models ---

@gen function map_kernel(x::Float64)
    y = {:y} ~ normal(x, 1.0)
    return y
end
map_model = Map(map_kernel)

@gen function unfold_kernel(t::Int, state::Float64)
    x = {:x} ~ normal(state, 1.0)
    return x
end
unfold_model = Unfold(unfold_kernel)

@gen function switch_branch_a()
    x = {:x} ~ normal(0.0, 1.0)
    return x
end
@gen function switch_branch_b()
    x = {:x} ~ normal(10.0, 1.0)
    return x
end
switch_model = Switch(switch_branch_a, switch_branch_b)

function make_combinator_choicemap(comb_type, choices_dict)
    cm = Gen.choicemap()
    if comb_type == "switch"
        # Switch choices are flat
        for (addr_str, val) in choices_dict
            cm[Symbol(addr_str)] = Float64(val)
        end
    else
        # Map/Unfold: hierarchical with 0-based → 1-based conversion
        for (idx_str, sub_dict) in choices_dict
            idx = parse(Int, idx_str) + 1  # 0-based → 1-based
            for (addr_str, val) in sub_dict
                cm[(idx => Symbol(addr_str))] = Float64(val)
            end
        end
    end
    return cm
end

function get_combinator_model(comb_type)
    if comb_type == "map"
        map_model
    elseif comb_type == "unfold"
        unfold_model
    elseif comb_type == "switch"
        switch_model
    else
        error("Unknown combinator type: $comb_type")
    end
end

function get_combinator_args(comb_type, raw_args)
    if comb_type == "map"
        (Float64.(raw_args[1]),)
    elseif comb_type == "unfold"
        (Int(raw_args[1]), Float64(raw_args[2]))
    elseif comb_type == "switch"
        # 0-based → 1-based for Gen.jl Switch
        (Int(raw_args[1]) + 1,)
    else
        error("Unknown combinator type: $comb_type")
    end
end

function eval_combinator(spec)
    comb_type = spec["combinator_type"]
    operation = spec["operation"]
    raw_args = spec["args"]
    model = get_combinator_model(comb_type)
    args = get_combinator_args(comb_type, raw_args)

    try
        if operation == "assess"
            cm = make_combinator_choicemap(comb_type, spec["choices"])
            (weight, _) = assess(model, args, cm)
            Dict("id" => spec["id"], "weight" => weight)

        elseif operation == "score_decomposition"
            cm = make_combinator_choicemap(comb_type, spec["choices"])
            (trace, _) = generate(model, args, cm)
            total_score = get_score(trace)
            components = spec["expected_components"]
            comp_scores = Dict{String, Float64}()
            for comp in components
                lp = compute_component_logprob(comp)
                comp_scores["$(comp["index"])_$(comp["addr"])"] = lp
            end
            sum_comp = sum(values(comp_scores))
            Dict("id" => spec["id"],
                 "total_score" => total_score,
                 "components" => comp_scores,
                 "sum_components" => sum_comp)

        elseif operation == "generate"
            cm = make_combinator_choicemap(comb_type, spec["constraints"])
            (trace, weight) = generate(model, args, cm)
            score = get_score(trace)
            Dict("id" => spec["id"], "weight" => weight, "score" => score)

        elseif operation == "update"
            init_cm = make_combinator_choicemap(comb_type, spec["initial_choices"])
            (trace, _) = generate(model, args, init_cm)
            old_score = get_score(trace)
            upd_cm = make_combinator_choicemap(comb_type, spec["update_choices"])
            # Combinator update: no argdiffs needed for same args
            (new_trace, weight, _, _) = update(trace, args, map(_ -> NoChange(), args), upd_cm)
            new_score = get_score(new_trace)
            Dict("id" => spec["id"],
                 "old_score" => old_score,
                 "new_score" => new_score,
                 "weight" => weight)
        else
            Dict("id" => spec["id"], "error" => "unsupported operation: $operation")
        end
    catch e
        Dict("id" => spec["id"], "error" => string(e))
    end
end

# --- Gradient dispatch ---

function eval_gradient(spec)
    dist_name = spec["dist"]
    value = Float64(spec["value"])
    params = spec["params"]
    grad_wrt = spec["grad_wrt"]

    try
        grad_val = if dist_name == "normal"
            mu = Float64(params["mu"]); sigma = Float64(params["sigma"])
            (gv, gmu, gsigma) = logpdf_grad(normal, value, mu, sigma)
            if grad_wrt == "value"
                gv
            elseif grad_wrt == "mu"
                gmu
            elseif grad_wrt == "sigma"
                gsigma
            else
                error("unsupported grad_wrt for normal: $grad_wrt")
            end

        elseif dist_name == "beta"
            alpha = Float64(params["alpha"]); b = Float64(params["beta"])
            if grad_wrt == "value"
                # d/dx log Beta(x;a,b) = (a-1)/x - (b-1)/(1-x)
                (alpha - 1.0) / value - (b - 1.0) / (1.0 - value)
            elseif grad_wrt == "alpha"
                # d/dalpha = log(x) - digamma(alpha) + digamma(alpha+beta)
                log(value) - digamma(alpha) + digamma(alpha + b)
            else
                error("unsupported grad_wrt for beta: $grad_wrt")
            end

        elseif dist_name == "gamma"
            shape = Float64(params["shape"]); rate = Float64(params["rate"])
            # Gen.jl gamma uses scale = 1/rate
            if grad_wrt == "value"
                # d/dx log Gamma(x;k,rate) = (k-1)/x - rate
                (shape - 1.0) / value - rate
            elseif grad_wrt == "shape"
                # d/dk = log(x) + log(rate) - digamma(k)
                log(value) + log(rate) - digamma(shape)
            else
                error("unsupported grad_wrt for gamma: $grad_wrt")
            end

        elseif dist_name == "exponential"
            rate = Float64(params["rate"])
            if grad_wrt == "value"
                -rate
            else
                error("unsupported grad_wrt for exponential: $grad_wrt")
            end

        elseif dist_name == "laplace"
            loc = Float64(params["loc"]); scale = Float64(params["scale"])
            if grad_wrt == "value"
                -sign(value - loc) / scale
            else
                error("unsupported grad_wrt for laplace: $grad_wrt")
            end

        elseif dist_name == "cauchy"
            loc = Float64(params["loc"]); scale = Float64(params["scale"])
            if grad_wrt == "value"
                -2.0 * (value - loc) / (scale^2 + (value - loc)^2)
            else
                error("unsupported grad_wrt for cauchy: $grad_wrt")
            end

        elseif dist_name == "lognormal"
            mu = Float64(params["mu"]); sigma = Float64(params["sigma"])
            if grad_wrt == "value"
                -(1.0 + (log(value) - mu) / sigma^2) / value
            else
                error("unsupported grad_wrt for lognormal: $grad_wrt")
            end

        elseif dist_name == "student_t"
            df = Float64(params["df"]); loc = Float64(params["loc"]); scale = Float64(params["scale"])
            if grad_wrt == "value"
                -(df + 1.0) * (value - loc) / (df * scale^2 + (value - loc)^2)
            else
                error("unsupported grad_wrt for student_t: $grad_wrt")
            end

        else
            return Dict("id" => spec["id"], "error" => "unsupported dist: $dist_name")
        end

        Dict("id" => spec["id"], "gradient" => grad_val)
    catch e
        Dict("id" => spec["id"], "error" => string(e))
    end
end

# --- Utility ---

function logsumexp(xs)
    m = maximum(xs)
    m + log(sum(exp.(xs .- m)))
end

# --- Phase 8: Inference Quality models ---

@gen function nn_inference_model(observations)
    mu = {:mu} ~ normal(0, 1)
    for (i, obs) in enumerate(observations)
        {Symbol("x$(i-1)")} ~ normal(mu, 1)
    end
    return mu
end

@gen function bb_inference_model(observations)
    p = {:p} ~ beta(1, 1)
    for (i, obs) in enumerate(observations)
        {Symbol("x$(i-1)")} ~ bernoulli(p)
    end
    return p
end

@gen function linreg_inference_model(xs)
    slope = {:slope} ~ normal(0, 10)
    for (i, x) in enumerate(xs)
        {Symbol("y$(i-1)")} ~ normal(slope * x, 1)
    end
    return slope
end

@gen function gp_inference_model(observations)
    lambda = {:lambda} ~ gamma(2, 1.0)  # shape=2, scale=1.0
    for (i, obs) in enumerate(observations)
        {Symbol("x$(i-1)")} ~ poisson(lambda)
    end
    return lambda
end

@gen function dc_inference_model(observations)
    p = {:p} ~ dirichlet([1.0, 1.0, 1.0])
    for (i, obs) in enumerate(observations)
        {Symbol("x$(i-1)")} ~ categorical(p)  # Gen.jl categorical: probs, 1-indexed
    end
    return p
end

INFERENCE_MODEL_LOOKUP = Dict(
    "normal_normal" => nn_inference_model,
    "beta_bernoulli_iid" => bb_inference_model,
    "normal_linreg" => linreg_inference_model,
    "gamma_poisson" => gp_inference_model,
    "dirichlet_categorical" => dc_inference_model,
)

function make_inference_observations(model_name, data)
    cm = Gen.choicemap()
    if model_name == "normal_normal"
        for (i, obs) in enumerate(data["observations"])
            cm[Symbol("x$(i-1)")] = Float64(obs)
        end
    elseif model_name == "beta_bernoulli_iid"
        for (i, obs) in enumerate(data["observations"])
            cm[Symbol("x$(i-1)")] = obs == 1 ? true : false
        end
    elseif model_name == "normal_linreg"
        for (i, y) in enumerate(data["ys"])
            cm[Symbol("y$(i-1)")] = Float64(y)
        end
    elseif model_name == "gamma_poisson"
        for (i, obs) in enumerate(data["observations"])
            cm[Symbol("x$(i-1)")] = Int(obs)
        end
    elseif model_name == "dirichlet_categorical"
        for (i, obs) in enumerate(data["observations"])
            # Gen.jl categorical is 1-indexed
            cm[Symbol("x$(i-1)")] = Int(obs) + 1
        end
    end
    return cm
end

function get_inference_args(model_name, data)
    if model_name == "normal_normal"
        (Float64.(data["observations"]),)
    elseif model_name == "beta_bernoulli_iid"
        (Float64.(data["observations"]),)
    elseif model_name == "normal_linreg"
        (Float64.(data["xs"]),)
    elseif model_name == "gamma_poisson"
        (Float64.(data["observations"]),)
    elseif model_name == "dirichlet_categorical"
        (Float64.(data["observations"]),)
    end
end

function extract_trace_value(trace, target_addr, target_component)
    val = trace[target_addr]
    if target_component !== nothing
        # Vector-valued (e.g. Dirichlet): extract component (0-indexed in spec)
        return val[target_component + 1]
    else
        return Float64(val)
    end
end

function eval_inference(spec)
    model_name = spec["model"]
    model = INFERENCE_MODEL_LOOKUP[model_name]
    algorithm = spec["algorithm"]
    algo_params = spec["algorithm_params"]
    data = spec["data"]
    target_addr = Symbol(spec["target_addr"])
    target_component = get(spec, "target_component", nothing)
    comparison = get(spec, "comparison", nothing)
    args = get_inference_args(model_name, data)
    obs_cm = make_inference_observations(model_name, data)

    try
        if algorithm == "importance_sampling"
            n_particles = get(algo_params, "n_particles", 1000)
            (traces, log_weights, _) = importance_sampling(model, args, obs_cm, n_particles)
            # Compute log-ML estimate
            log_ml = logsumexp(log_weights) - log(n_particles)
            # Compute weighted mean
            max_w = maximum(log_weights)
            weights = exp.(log_weights .- max_w)
            weights ./= sum(weights)
            vals = [extract_trace_value(traces[i], target_addr, target_component) for i in 1:length(traces)]
            posterior_mean = sum(vals .* weights)
            result = Dict("id" => spec["id"], "posterior_mean" => posterior_mean)
            if comparison in ("log_ml", "log_ml_analytical")
                result["log_ml"] = log_ml
            end
            return result
        elseif algorithm == "mh"
            n_steps = get(algo_params, "n_steps", 2000)
            burn = get(algo_params, "burn", 500)
            (trace, _) = generate(model, args, obs_cm)
            sel = select(target_addr)
            samples = Float64[]
            accepted_count = 0
            for step in 1:n_steps
                (trace, accepted) = mh(trace, sel)
                if step > burn
                    push!(samples, extract_trace_value(trace, target_addr, target_component))
                    if accepted; accepted_count += 1; end
                end
            end
            posterior_mean = sum(samples) / length(samples)
            accept_rate = accepted_count / length(samples)
            return Dict("id" => spec["id"], "posterior_mean" => posterior_mean, "acceptance_rate" => accept_rate)

        elseif algorithm == "hmc"
            n_steps = get(algo_params, "n_steps", 500)
            burn = get(algo_params, "burn", 200)
            L = Int(get(algo_params, "leapfrog_steps", 10))
            eps = get(algo_params, "step_size", 0.01)
            (trace, _) = generate(model, args, obs_cm)
            sel = select(target_addr)
            samples = Float64[]
            accepted_count = 0
            for step in 1:(burn + n_steps)
                (trace, accepted) = hmc(trace, sel; L=L, eps=eps)
                if step > burn
                    push!(samples, trace[target_addr])
                    if accepted; accepted_count += 1; end
                end
            end
            posterior_mean = sum(samples) / length(samples)
            accept_rate = accepted_count / n_steps
            return Dict("id" => spec["id"], "posterior_mean" => posterior_mean, "acceptance_rate" => accept_rate)

        elseif algorithm == "mala"
            n_steps = get(algo_params, "n_steps", 500)
            burn = get(algo_params, "burn", 200)
            tau = get(algo_params, "step_size", 0.01)
            (trace, _) = generate(model, args, obs_cm)
            sel = select(target_addr)
            samples = Float64[]
            accepted_count = 0
            for step in 1:(burn + n_steps)
                (trace, accepted) = mala(trace, sel, tau)
                if step > burn
                    push!(samples, trace[target_addr])
                    if accepted; accepted_count += 1; end
                end
            end
            posterior_mean = sum(samples) / length(samples)
            accept_rate = accepted_count / n_steps
            return Dict("id" => spec["id"], "posterior_mean" => posterior_mean, "acceptance_rate" => accept_rate)

        else
            return Dict("id" => spec["id"], "error" => "unsupported algorithm: $algorithm")
        end
    catch e
        Dict("id" => spec["id"], "error" => string(e))
    end
end

# --- Phase 12: SMC — Linear-Gaussian SSM via particle filter ---

@gen function ssm_model(T::Int)
    x_prev = {:x1} ~ normal(0.0, 1.0)
    {:y1} ~ normal(x_prev, 0.5)
    for t in 2:T
        x = {(:x, t)} ~ normal(x_prev, 1.0)
        {(:y, t)} ~ normal(x, 0.5)
        x_prev = x
    end
end

function run_smc_ssm(observations, n_particles)
    T = length(observations)
    # Initialize at t=1
    init_obs = choicemap()
    init_obs[:y1] = observations[1]
    state = initialize_particle_filter(ssm_model, (1,), init_obs, n_particles)

    # Step through t=2..T
    for t in 2:T
        obs_t = choicemap()
        obs_t[(:y, t)] = observations[t]
        maybe_resample!(state, ess_threshold=n_particles/2.0)
        particle_filter_step!(state, (t,), (UnknownChange(),), obs_t)
    end

    log_ml = log_ml_estimate(state)
    # Compute ESS from final weights
    log_weights = state.log_weights
    max_w = maximum(log_weights)
    weights = exp.(log_weights .- max_w)
    weights ./= sum(weights)
    ess = 1.0 / sum(weights .^ 2)

    return (log_ml, ess)
end

function run_smc_single(model, args, obs_cm, n_particles)
    (traces, log_weights, _) = importance_sampling(model, args, obs_cm, n_particles)
    log_ml = logsumexp(log_weights) - log(n_particles)
    return log_ml
end

function eval_inference_smc(spec)
    algorithm = spec["algorithm"]
    algo_params = spec["algorithm_params"]
    data = spec["data"]
    comparison = get(spec, "comparison", "log_ml")

    try
        if algorithm == "smc"
            observations = Float64.(data["observations"])
            n_particles = Int(get(algo_params, "n_particles", 500))
            (log_ml, ess) = run_smc_ssm(observations, n_particles)

            if comparison == "ess"
                return Dict("id" => spec["id"], "ess" => ess, "log_ml" => log_ml)
            else
                return Dict("id" => spec["id"], "log_ml" => log_ml)
            end
        elseif algorithm == "smc_single"
            model_name = spec["model"]
            model = INFERENCE_MODEL_LOOKUP[model_name]
            args = get_inference_args(model_name, data)
            obs_cm = make_inference_observations(model_name, data)
            n_particles = Int(get(algo_params, "n_particles", 500))
            log_ml = run_smc_single(model, args, obs_cm, n_particles)
            return Dict("id" => spec["id"], "log_ml" => log_ml)
        else
            return Dict("id" => spec["id"], "error" => "unsupported smc algorithm: $algorithm")
        end
    catch e
        Dict("id" => spec["id"], "error" => string(e))
    end
end

# --- Phase 13: VI — GenMLX-only (analytical comparison), Gen.jl returns skip ---

function eval_inference_vi(spec)
    # Gen.jl VI is not implemented for cross-system comparison.
    # Return skip so orchestrator uses analytical_only comparison.
    return Dict("id" => spec["id"], "skip" => true)
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
    elseif test_type == "project"
        for spec in input["project_tests"]
            push!(results, eval_project(spec))
        end
    elseif test_type == "regenerate"
        for spec in input["regenerate_tests"]
            push!(results, eval_regenerate(spec))
        end
    elseif test_type == "combinator"
        for spec in input["combinator_tests"]
            push!(results, eval_combinator(spec))
        end
    elseif test_type == "gradient"
        for spec in input["gradient_tests"]
            push!(results, eval_gradient(spec))
        end
    elseif test_type == "inference_quality"
        for spec in input["tests"]
            algo = spec["algorithm"]
            if algo in ("smc", "smc_single")
                push!(results, eval_inference_smc(spec))
            elseif algo == "vi"
                push!(results, eval_inference_vi(spec))
            else
                push!(results, eval_inference(spec))
            end
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
