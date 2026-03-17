# Gen.jl cross-system verification runner
# Reads JSON from stdin, writes results to stdout.
using Gen
using JSON
using SpecialFunctions: loggamma

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

MODEL_LOOKUP = Dict(
    "single_normal" => single_normal_model,
    "two_normals" => two_normals_model,
    "beta_bernoulli" => beta_bernoulli_model,
    "linear_regression" => linear_regression_model,
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

function eval_assess(spec)
    model_name = spec["model"]
    model = MODEL_LOOKUP[model_name]
    raw_args = get(spec, "args", [])
    args = if model_name == "linear_regression"
        (Float64.(raw_args[1]),)
    else
        ()
    end
    cm = make_choicemap(spec["choices"])

    try
        if model_name == "beta_bernoulli"
            cm[:x] = spec["choices"]["x"] == 1 ? true : false
        end
        (weight, retval) = assess(model, args, cm)
        Dict("id" => spec["id"], "weight" => weight)
    catch e
        Dict("id" => spec["id"], "error" => string(e))
    end
end

function eval_generate(spec)
    model_name = spec["model"]
    model = MODEL_LOOKUP[model_name]
    raw_args = get(spec, "args", [])
    args = if model_name == "linear_regression"
        (Float64.(raw_args[1]),)
    else
        ()
    end
    cm = make_choicemap(spec["constraints"])

    try
        (trace, weight) = generate(model, args, cm)
        score = get_score(trace)
        Dict("id" => spec["id"], "weight" => weight, "score" => score)
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
    else
        println(stderr, "Unknown test type: $test_type")
        exit(1)
    end

    output = Dict("system" => "gen_jl", "test_type" => test_type, "results" => results)
    JSON.print(stdout, output, 2)
    println()
end

main()
