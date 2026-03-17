"""GenJAX cross-system verification runner.
Reads JSON from stdin, writes results to stdout."""
import sys
import json
import math
import jax
import jax.numpy as jnp
import genjax


# --- Distribution log-prob dispatch ---

def eval_logprob(spec):
    dist_name = spec["dist"]
    value = spec["value"]
    params = spec["params"]

    try:
        if dist_name == "normal":
            lp = eval_logprob_via_assess(genjax.normal, value,
                                         params["mu"], params["sigma"])
        elif dist_name == "uniform":
            lp = eval_logprob_via_assess(genjax.uniform, value,
                                         params["lo"], params["hi"])
        elif dist_name == "bernoulli":
            lp = eval_logprob_via_assess(genjax.bernoulli, value,
                                         probs=params["p"])
        elif dist_name == "beta":
            lp = eval_logprob_via_assess(genjax.beta, value,
                                         params["alpha"], params["beta"])
        elif dist_name == "gamma":
            lp = eval_logprob_via_assess(genjax.gamma, value,
                                         params["shape"], params["rate"])
        elif dist_name == "exponential":
            lp = eval_logprob_via_assess(genjax.exponential, value,
                                         params["rate"])
        elif dist_name == "laplace":
            lp = eval_logprob_via_assess(genjax.laplace, value,
                                         params["loc"], params["scale"])
        elif dist_name == "cauchy":
            lp = eval_logprob_via_assess(genjax.cauchy, value,
                                         params["loc"], params["scale"])
        elif dist_name == "poisson":
            lp = eval_logprob_via_assess(genjax.poisson, value,
                                         params["rate"])
        elif dist_name == "binomial":
            jax_val = jnp.array(int(value), dtype=jnp.int32)
            n = jnp.array(int(params["n"]), dtype=jnp.int32)
            p = jnp.array(params["p"], dtype=jnp.float32)
            density, _ = genjax.binomial.assess(jax_val, n, p)
            lp = float(density)
        elif dist_name == "geometric":
            if not hasattr(genjax, 'geometric'):
                return {"id": spec["id"], "error": "genjax has no geometric distribution"}
            lp = eval_logprob_via_assess(genjax.geometric, value,
                                         params["p"])
        elif dist_name == "lognormal":
            lp = eval_logprob_via_assess(genjax.log_normal, value,
                                         params["mu"], params["sigma"])
        elif dist_name == "inv_gamma":
            lp = eval_logprob_via_assess(genjax.inverse_gamma, value,
                                         params["shape"], params["scale"])
        elif dist_name == "student_t":
            lp = eval_logprob_via_assess(genjax.student_t, value,
                                         params["df"], params["loc"],
                                         params["scale"])
        else:
            return {"id": spec["id"], "error": f"unsupported dist: {dist_name}"}

        return {"id": spec["id"], "logprob": float(lp)}
    except Exception as e:
        return {"id": spec["id"], "error": str(e)}


def eval_logprob_via_assess(dist, value, *args, **kwargs):
    """Use GenJAX's assess to get log-prob of a distribution at a point."""
    jax_value = jnp.array(value, dtype=jnp.float32)
    density, _ = dist.assess(jax_value, *args, **kwargs)
    return float(density)


# --- GFI models ---

@genjax.gen
def single_normal_model():
    x = genjax.normal(0.0, 1.0) @ "x"
    return x

@genjax.gen
def two_normals_model():
    mu = genjax.normal(0.0, 10.0) @ "mu"
    x = genjax.normal(mu, 1.0) @ "x"
    return x

@genjax.gen
def beta_bernoulli_model():
    p = genjax.beta(2.0, 2.0) @ "p"
    x = genjax.bernoulli(probs=p) @ "x"
    return x

@genjax.gen
def linear_regression_model(xs):
    slope = genjax.normal(0.0, 10.0) @ "slope"
    intercept = genjax.normal(0.0, 10.0) @ "intercept"
    y0 = genjax.normal(slope * xs[0] + intercept, 1.0) @ "y0"
    y1 = genjax.normal(slope * xs[1] + intercept, 1.0) @ "y1"
    y2 = genjax.normal(slope * xs[2] + intercept, 1.0) @ "y2"
    return slope

MODEL_LOOKUP = {
    "single_normal": single_normal_model,
    "two_normals": two_normals_model,
    "beta_bernoulli": beta_bernoulli_model,
    "linear_regression": linear_regression_model,
}


def make_choices_dict(choices, model_name):
    """Convert spec choices to GenJAX constraint dict."""
    result = {}
    for k, v in choices.items():
        if isinstance(v, bool):
            result[k] = jnp.array(v, dtype=jnp.bool_)
        elif isinstance(v, int) and model_name == "beta_bernoulli" and k == "x":
            result[k] = jnp.array(v, dtype=jnp.bool_)
        elif isinstance(v, int):
            result[k] = jnp.array(float(v), dtype=jnp.float32)
        else:
            result[k] = jnp.array(v, dtype=jnp.float32)
    return result


def eval_assess(spec):
    model_name = spec["model"]
    model = MODEL_LOOKUP[model_name]
    raw_args = spec.get("args", [])

    if model_name == "linear_regression":
        args = (jnp.array(raw_args[0], dtype=jnp.float32),)
    else:
        args = ()

    choices = make_choices_dict(spec["choices"], model_name)

    try:
        density, retval = model.assess(choices, *args)
        return {"id": spec["id"], "weight": float(density)}
    except Exception as e:
        return {"id": spec["id"], "error": str(e)}


def eval_generate(spec):
    model_name = spec["model"]
    model = MODEL_LOOKUP[model_name]
    raw_args = spec.get("args", [])

    if model_name == "linear_regression":
        args = (jnp.array(raw_args[0], dtype=jnp.float32),)
    else:
        args = ()

    constraints = make_choices_dict(spec["constraints"], model_name)

    try:
        tr, weight = model.generate(constraints, *args)
        score = float(tr.get_score())
        # GenJAX score is NEGATIVE log-prob — negate to match Gen.jl/GenMLX
        return {"id": spec["id"], "weight": float(weight), "score": -score}
    except Exception as e:
        return {"id": spec["id"], "error": str(e)}


# --- JSON sanitization ---

def sanitize(obj):
    """Replace -inf/inf/nan with JSON-safe strings."""
    if isinstance(obj, float):
        if math.isinf(obj):
            return "-Inf" if obj < 0 else "Inf"
        if math.isnan(obj):
            return "NaN"
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


# --- Main: read stdin, write stdout ---

def main():
    input_data = json.loads(sys.stdin.read())
    test_type = input_data["test_type"]
    results = []

    if test_type == "logprob":
        for spec in input_data["tests"]:
            results.append(eval_logprob(spec))
    elif test_type == "assess":
        for spec in input_data["assess_tests"]:
            results.append(eval_assess(spec))
    elif test_type == "generate":
        for spec in input_data["generate_tests"]:
            results.append(eval_generate(spec))
    else:
        print(f"Unknown test type: {test_type}", file=sys.stderr)
        sys.exit(1)

    output = sanitize({"system": "genjax", "test_type": test_type, "results": results})
    json.dump(output, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
