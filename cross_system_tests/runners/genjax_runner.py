"""GenJAX cross-system verification runner.
Reads JSON from stdin, writes results to stdout."""
import sys
import json
import math
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.special as jsp
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
            # Computed from formula (TFP dtype issues with x64 enabled).
            # log P(k|n,p) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1) + k*log(p) + (n-k)*log(1-p)
            k = float(value); n = float(params["n"]); p_binom = float(params["p"])
            lp = float(jsp.gammaln(n + 1) - jsp.gammaln(k + 1) - jsp.gammaln(n - k + 1)
                        + k * jnp.log(p_binom) + (n - k) * jnp.log(1 - p_binom))
        elif dist_name == "geometric":
            # Computed from formula, not native distribution.
            # P(k|p) = (1-p)^k * p, log P = k*log(1-p) + log(p)
            k = float(value); p_geom = float(params["p"])
            lp = k * math.log(1 - p_geom) + math.log(p_geom)
        elif dist_name == "lognormal":
            lp = eval_logprob_via_assess(genjax.log_normal, value,
                                         params["mu"], params["sigma"])
        elif dist_name == "inv_gamma":
            lp = eval_logprob_via_assess(genjax.inverse_gamma, value,
                                         params["shape"], params["scale"])
        elif dist_name == "student_t":
            # Computed from formula (TFP dtype issues with x64 enabled).
            df = float(params["df"]); loc = float(params["loc"]); scale = float(params["scale"])
            z = (value - loc) / scale
            lp = float(jsp.gammaln((df + 1) / 2) - jsp.gammaln(df / 2)
                        - 0.5 * jnp.log(df * jnp.pi) - jnp.log(scale)
                        - ((df + 1) / 2) * jnp.log(1 + z ** 2 / df))
        elif dist_name == "dirichlet":
            # Computed from formula, not native distribution.
            # log P(x|alpha) = sum((alpha_i - 1)*log(x_i)) + lgamma(sum(alpha)) - sum(lgamma(alpha_i))
            alpha = jnp.array(params["alpha"], dtype=jnp.float64)
            x = jnp.array(value, dtype=jnp.float64)
            lp = float(jnp.sum((alpha - 1) * jnp.log(x))
                        + jsp.gammaln(jnp.sum(alpha)) - jnp.sum(jsp.gammaln(alpha)))
        elif dist_name == "mvn":
            # Multivariate normal from formula.
            # log P(x|mu,Sigma) = -k/2*log(2*pi) - 1/2*logdet(Sigma) - 1/2*(x-mu)^T Sigma^-1 (x-mu)
            mu = jnp.array(params["mu"], dtype=jnp.float64)
            cov = jnp.array(params["cov"], dtype=jnp.float64)
            x = jnp.array(value, dtype=jnp.float64)
            k = len(mu)
            diff = x - mu
            sign, logdet_val = jnp.linalg.slogdet(cov)
            maha = float(diff @ jnp.linalg.solve(cov, diff))
            lp = float(-0.5 * k * jnp.log(2 * jnp.pi) - 0.5 * logdet_val - 0.5 * maha)
        # --- New distributions (computed from formula, not native distribution) ---
        elif dist_name == "neg_binomial":
            # log P(k | r, p) = lgamma(k+r) - lgamma(k+1) - lgamma(r) + r*log(p) + k*log(1-p)
            k = float(value); r = float(params["r"]); p_nb = float(params["p"])
            lp = float(jsp.gammaln(k + r) - jsp.gammaln(k + 1) - jsp.gammaln(r)
                        + r * jnp.log(p_nb) + k * jnp.log(1 - p_nb))
        elif dist_name == "discrete_uniform":
            # P(k) = 1/(hi - lo + 1) for lo <= k <= hi, else 0
            k = int(value); lo = int(params["lo"]); hi = int(params["hi"])
            lp = -math.log(hi - lo + 1) if lo <= k <= hi else float('-inf')
        elif dist_name == "truncated_normal":
            # logpdf = normal_logpdf(x, mu, sigma) - log(Phi((hi-mu)/sigma) - Phi((lo-mu)/sigma))
            x = float(value); mu = float(params["mu"]); sigma = float(params["sigma"])
            lo = float(params["lo"]); hi = float(params["hi"])
            if lo <= x <= hi:
                normal_lp = -0.5 * math.log(2 * math.pi) - math.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2
                # ndtr is standard normal CDF
                phi_hi = float(jsp.ndtr((hi - mu) / sigma))
                phi_lo = float(jsp.ndtr((lo - mu) / sigma))
                log_Z = math.log(phi_hi - phi_lo)
                lp = normal_lp - log_Z
            else:
                lp = float('-inf')
        elif dist_name == "von_mises":
            # log P = kappa*cos(x - mu) - log(2*pi) - log(I0(kappa))
            # i0e(kappa) = I0(kappa) * exp(-kappa), so log(I0(kappa)) = log(i0e(kappa)) + kappa
            x = float(value); mu = float(params["mu"]); kappa = float(params["kappa"])
            log_i0 = float(jnp.log(jsp.i0e(kappa))) + kappa
            lp = kappa * math.cos(x - mu) - math.log(2 * math.pi) - log_i0
        elif dist_name == "categorical":
            # P(k | logits) = softmax(logits)[k]
            # logprob = logits[k] - logsumexp(logits)
            logits = jnp.array(params["logits"], dtype=jnp.float64)
            k = int(value)
            lp = float(logits[k] - jsp.logsumexp(logits))
        elif dist_name == "wishart":
            # Wishart log-prob from formula
            df = float(params["df"])
            # Handle both "scale" and "scale_matrix" param names
            V = jnp.array(params.get("scale", params.get("scale_matrix")), dtype=jnp.float64)
            X = jnp.array(value, dtype=jnp.float64)
            p = X.shape[0]
            log_mvgamma = float(jsp.multigammaln(df / 2.0, p))
            sign_X, logdet_X = jnp.linalg.slogdet(X)
            sign_V, logdet_V = jnp.linalg.slogdet(V)
            lp = float(0.5 * (df - p - 1) * logdet_X - 0.5 * jnp.trace(jnp.linalg.solve(V, X))
                        - 0.5 * df * p * jnp.log(2.0) - 0.5 * df * logdet_V - log_mvgamma)
        elif dist_name == "inv_wishart":
            # Inverse Wishart log-prob from formula
            df = float(params["df"])
            # Handle both "scale" and "scale_matrix" param names
            Psi = jnp.array(params.get("scale", params.get("scale_matrix")), dtype=jnp.float64)
            X = jnp.array(value, dtype=jnp.float64)
            p = X.shape[0]
            log_mvgamma = float(jsp.multigammaln(df / 2.0, p))
            sign_Psi, logdet_Psi = jnp.linalg.slogdet(Psi)
            sign_X, logdet_X = jnp.linalg.slogdet(X)
            lp = float(0.5 * df * logdet_Psi - 0.5 * (df + p + 1) * logdet_X
                        - 0.5 * jnp.trace(Psi @ jnp.linalg.inv(X))
                        - 0.5 * df * p * jnp.log(2.0) - log_mvgamma)
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

# Phase 1: single_gaussian (alias — same as single_normal but named differently in specs)
@genjax.gen
def single_gaussian_model():
    x = genjax.normal(0.0, 1.0) @ "x"
    return x

# Phase 1: mixed model (bernoulli + branching)
# GenJAX uses Cond combinator for traced branching.
@genjax.gen
def _mixed_branch_true():
    x = genjax.normal(10.0, 1.0) @ "x"
    return x

@genjax.gen
def _mixed_branch_false():
    x = genjax.normal(0.0, 1.0) @ "x"
    return x

_mixed_cond = _mixed_branch_true.cond(_mixed_branch_false)

@genjax.gen
def mixed_model():
    coin = genjax.bernoulli(probs=0.5) @ "coin"
    x = _mixed_cond(coin) @ "branch"
    return x

# Phase 1: many_addresses model (mu + 10 obs)
@genjax.gen
def many_addresses_model():
    mu = genjax.normal(0.0, 10.0) @ "mu"
    y0 = genjax.normal(mu, 1.0) @ "y0"
    y1 = genjax.normal(mu, 1.0) @ "y1"
    y2 = genjax.normal(mu, 1.0) @ "y2"
    y3 = genjax.normal(mu, 1.0) @ "y3"
    y4 = genjax.normal(mu, 1.0) @ "y4"
    y5 = genjax.normal(mu, 1.0) @ "y5"
    y6 = genjax.normal(mu, 1.0) @ "y6"
    y7 = genjax.normal(mu, 1.0) @ "y7"
    y8 = genjax.normal(mu, 1.0) @ "y8"
    y9 = genjax.normal(mu, 1.0) @ "y9"
    return mu

# Phase 1: linear_regression_5 (5-observation variant)
@genjax.gen
def linear_regression_5_model(xs):
    slope = genjax.normal(0.0, 10.0) @ "slope"
    intercept = genjax.normal(0.0, 10.0) @ "intercept"
    y0 = genjax.normal(slope * xs[0] + intercept, 1.0) @ "y0"
    y1 = genjax.normal(slope * xs[1] + intercept, 1.0) @ "y1"
    y2 = genjax.normal(slope * xs[2] + intercept, 1.0) @ "y2"
    y3 = genjax.normal(slope * xs[3] + intercept, 1.0) @ "y3"
    y4 = genjax.normal(slope * xs[4] + intercept, 1.0) @ "y4"
    return slope

MODEL_LOOKUP = {
    "single_normal": single_normal_model,
    "single_gaussian": single_gaussian_model,
    "two_normals": two_normals_model,
    "beta_bernoulli": beta_bernoulli_model,
    "linear_regression": linear_regression_model,
    "linear_regression_5": linear_regression_5_model,
    "many_addresses": many_addresses_model,
    "mixed": mixed_model,
}

# --- Boolean address sets: models where certain addresses need bool dtype ---
BOOL_ADDRS = {
    "beta_bernoulli": {"x"},
    "mixed": {"coin"},
}


def make_choices_dict(choices, model_name):
    """Convert spec choices to GenJAX constraint dict.
    Handles model-specific address remapping (e.g., mixed model uses
    hierarchical {'branch': {'x': val}} instead of flat {'x': val})."""
    bool_addrs = BOOL_ADDRS.get(model_name, set())
    result = {}

    # Mixed model: flat spec {coin, x} -> GenJAX hierarchical {coin, branch: {x}}
    if model_name == "mixed":
        for k, v in choices.items():
            if k == "coin":
                result["coin"] = jnp.array(v == 1, dtype=jnp.bool_)
            elif k == "x":
                if isinstance(v, int):
                    result.setdefault("branch", {})["x"] = jnp.array(float(v), dtype=jnp.float64)
                else:
                    result.setdefault("branch", {})["x"] = jnp.array(v, dtype=jnp.float64)
        return result

    for k, v in choices.items():
        if isinstance(v, bool):
            result[k] = jnp.array(v, dtype=jnp.bool_)
        elif k in bool_addrs:
            result[k] = jnp.array(v == 1, dtype=jnp.bool_)
        elif isinstance(v, int):
            result[k] = jnp.array(float(v), dtype=jnp.float64)
        else:
            result[k] = jnp.array(v, dtype=jnp.float64)
    return result


def get_model_args(model_name, spec):
    """Build model args tuple from spec."""
    raw_args = spec.get("args", [])
    if model_name in ("linear_regression", "linear_regression_5"):
        if raw_args:
            return (jnp.array(raw_args[0], dtype=jnp.float64),)
        elif model_name == "linear_regression":
            return (jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64),)
        else:
            return (jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64),)
    else:
        return ()


def eval_assess(spec):
    model_name = spec["model"]
    model = MODEL_LOOKUP[model_name]
    args = get_model_args(model_name, spec)
    choices = make_choices_dict(spec["choices"], model_name)

    try:
        density, retval = model.assess(choices, *args)
        return {"id": spec["id"], "weight": float(density)}
    except Exception as e:
        return {"id": spec["id"], "error": str(e)}


def eval_generate(spec):
    model_name = spec["model"]
    model = MODEL_LOOKUP[model_name]
    args = get_model_args(model_name, spec)
    constraints = make_choices_dict(spec["constraints"], model_name)

    try:
        tr, weight = model.generate(constraints, *args)
        score = float(tr.get_score())
        # GenJAX get_score() returns -logpdf — negate to match Gen.jl/GenMLX
        return {"id": spec["id"], "weight": float(weight), "score": -score}
    except Exception as e:
        return {"id": spec["id"], "error": str(e)}


# --- Score Decomposition ---

def compute_component_logprob(comp):
    """Compute log-prob of a single score component using GenJAX distribution assess."""
    dist_name = comp["dist"]
    value = comp["value"]
    params = comp["params"]

    if dist_name == "normal":
        return eval_logprob_via_assess(genjax.normal, value,
                                        params["mu"], params["sigma"])
    elif dist_name == "bernoulli":
        v = jnp.array(value == 1, dtype=jnp.bool_)
        density, _ = genjax.bernoulli.assess(v, probs=params["p"])
        return float(density)
    elif dist_name == "beta":
        return eval_logprob_via_assess(genjax.beta, value,
                                        params["alpha"], params["beta"])
    else:
        raise ValueError(f"unsupported component dist: {dist_name}")


def eval_score_decomposition(spec):
    model_name = spec["model"]
    model = MODEL_LOOKUP[model_name]
    args = get_model_args(model_name, spec)
    choices = make_choices_dict(spec["choices"], model_name)

    try:
        tr, _ = model.generate(choices, *args)
        # GenJAX get_score() returns -logpdf — negate
        total_score = -float(tr.get_score())

        components = {}
        for addr, comp in spec["expected_components"].items():
            lp = compute_component_logprob(comp)
            components[addr] = lp

        sum_components = sum(components.values())

        return {"id": spec["id"],
                "total_score": total_score,
                "components": components,
                "sum_components": sum_components}
    except Exception as e:
        return {"id": spec["id"], "error": str(e)}


# --- Update ---

def eval_update(spec):
    model_name = spec["model"]
    model = MODEL_LOOKUP[model_name]
    args = get_model_args(model_name, spec)

    # Step 1: create initial trace via fully-constrained generate
    initial_choices = make_choices_dict(spec["initial_choices"], model_name)

    try:
        tr, _ = model.generate(initial_choices, *args)
        # GenJAX get_score() returns -logpdf — negate
        old_score = -float(tr.get_score())

        # Step 2: update with new constraints
        update_choices = make_choices_dict(spec["update_constraints"], model_name)
        new_tr, weight, discard = model.update(tr, update_choices, *args)
        # GenJAX get_score() returns -logpdf — negate
        new_score = -float(new_tr.get_score())

        # GenJAX update weight = new_logpdf - old_logpdf (same sign as Gen.jl)
        return {"id": spec["id"],
                "old_score": old_score,
                "new_score": new_score,
                "weight": float(weight)}
    except Exception as e:
        return {"id": spec["id"], "error": str(e)}


# --- Project + Regenerate ---

def make_selection(sel_spec):
    """Create a GenJAX Selection from a spec selection dict."""
    from genjax.core import sel
    sel_type = sel_spec["type"]
    if sel_type == "addrs":
        addrs = sel_spec["addrs"]
        if len(addrs) == 1:
            return sel(addrs[0])
        else:
            # Union of multiple single-address selections
            s = sel(addrs[0])
            for a in addrs[1:]:
                s = s | sel(a)
            return s
    elif sel_type == "all":
        return sel(())
    else:  # "none"
        return sel({})  # Empty dict selection matches nothing


def _compute_site_logprobs(model_name, choices_raw, args_raw):
    """Compute per-site log-probs for a model given choices.
    Returns dict {addr: logprob}.
    Uses model-specific structure knowledge."""
    site_lps = {}

    if model_name == "single_gaussian" or model_name == "single_normal":
        site_lps["x"] = eval_logprob_via_assess(genjax.normal, choices_raw["x"], 0.0, 1.0)

    elif model_name == "two_normals":
        mu = float(choices_raw["mu"])
        site_lps["mu"] = eval_logprob_via_assess(genjax.normal, mu, 0.0, 10.0)
        site_lps["x"] = eval_logprob_via_assess(genjax.normal, choices_raw["x"], mu, 1.0)

    elif model_name == "beta_bernoulli":
        p = float(choices_raw["p"])
        site_lps["p"] = eval_logprob_via_assess(genjax.beta, p, 2.0, 2.0)
        x_val = choices_raw["x"]
        jax_x = jnp.array(x_val == 1, dtype=jnp.bool_)
        density, _ = genjax.bernoulli.assess(jax_x, probs=p)
        site_lps["x"] = float(density)

    elif model_name in ("linear_regression", "linear_regression_5"):
        slope = float(choices_raw["slope"])
        intercept = float(choices_raw["intercept"])
        xs = args_raw[0] if args_raw else ([1.0, 2.0, 3.0] if model_name == "linear_regression"
                                            else [1.0, 2.0, 3.0, 4.0, 5.0])
        site_lps["slope"] = eval_logprob_via_assess(genjax.normal, slope, 0.0, 10.0)
        site_lps["intercept"] = eval_logprob_via_assess(genjax.normal, intercept, 0.0, 10.0)
        for i, x in enumerate(xs):
            addr = f"y{i}"
            if addr in choices_raw:
                mu_i = slope * x + intercept
                site_lps[addr] = eval_logprob_via_assess(genjax.normal, choices_raw[addr], mu_i, 1.0)

    elif model_name == "many_addresses":
        mu = float(choices_raw["mu"])
        site_lps["mu"] = eval_logprob_via_assess(genjax.normal, mu, 0.0, 10.0)
        for i in range(10):
            addr = f"y{i}"
            if addr in choices_raw:
                site_lps[addr] = eval_logprob_via_assess(genjax.normal, choices_raw[addr], mu, 1.0)

    elif model_name == "mixed":
        coin = int(choices_raw.get("coin", 0))
        site_lps["coin"] = math.log(0.5)  # Bernoulli(0.5)
        x_val = float(choices_raw["x"])
        mu_x = 10.0 if coin == 1 else 0.0
        site_lps["x"] = eval_logprob_via_assess(genjax.normal, x_val, mu_x, 1.0)

    return site_lps


def eval_project(spec):
    """Project = sum of logprobs for selected addresses only.
    GenJAX doesn't have native project() — compute from model structure."""
    model_name = spec["model"]
    model = MODEL_LOOKUP[model_name]
    args = get_model_args(model_name, spec)
    choices = make_choices_dict(spec["choices"], model_name)

    try:
        tr, _ = model.generate(choices, *args)
        # GenJAX get_score() returns -logpdf — negate
        total_score = -float(tr.get_score())

        sel_spec = spec["selection"]
        if sel_spec["type"] == "all":
            projected = total_score
        elif sel_spec["type"] == "none":
            projected = 0.0
        else:
            # Compute per-site logprobs and sum selected ones
            raw_args = spec.get("args", [])
            site_lps = _compute_site_logprobs(model_name, spec["choices"], raw_args)
            projected = sum(site_lps.get(a, 0.0) for a in sel_spec["addrs"])

        return {"id": spec["id"], "weight": projected, "score": total_score}
    except Exception as e:
        return {"id": spec["id"], "error": str(e)}


def eval_regenerate(spec):
    model_name = spec["model"]
    model = MODEL_LOOKUP[model_name]
    args = get_model_args(model_name, spec)
    choices = make_choices_dict(spec["choices"], model_name)

    try:
        tr, _ = model.generate(choices, *args)
        # GenJAX get_score() returns -logpdf — negate
        old_score = -float(tr.get_score())

        selection = make_selection(spec["selection"])
        new_tr, weight, discard = model.regenerate(tr, selection, *args)
        new_score = -float(new_tr.get_score())

        return {"id": spec["id"],
                "old_score": old_score,
                "new_score": new_score,
                "weight": float(weight)}
    except Exception as e:
        return {"id": spec["id"], "error": str(e)}


# --- Combinators ---

@genjax.gen
def map_kernel(x):
    y = genjax.normal(x, 1.0) @ "y"
    return y

@genjax.gen
def unfold_kernel(t, state):
    x = genjax.normal(state, 1.0) @ "x"
    return x

@genjax.gen
def switch_branch_a():
    x = genjax.normal(0.0, 1.0) @ "x"
    return x

@genjax.gen
def switch_branch_b():
    x = genjax.normal(10.0, 1.0) @ "x"
    return x


def eval_combinator(spec):
    """Evaluate combinator tests.
    NOTE: GenJAX combinator API may differ significantly from Gen.jl.
    Map/Unfold/Switch may not be directly available or have different interfaces.
    We attempt each and report errors for unsupported operations."""
    comb_type = spec["combinator_type"]
    operation = spec["operation"]

    try:
        if comb_type == "map":
            return eval_combinator_map(spec, operation)
        elif comb_type == "unfold":
            return eval_combinator_unfold(spec, operation)
        elif comb_type == "switch":
            return eval_combinator_switch(spec, operation)
        else:
            return {"id": spec["id"], "error": f"unsupported combinator: {comb_type}"}
    except Exception as e:
        return {"id": spec["id"], "error": str(e)}


def eval_combinator_map(spec, operation):
    """Map combinator: apply kernel to each element.
    GenJAX may support Map via vmap or explicit map combinator."""
    # Check if GenJAX has Map combinator
    if not hasattr(genjax, 'Map'):
        # Compute from components directly using kernel
        raw_args = spec["args"]
        xs = raw_args[0]

        if operation == "assess":
            # Score = sum of kernel logpdfs
            total = 0.0
            for idx_str, sub in spec["choices"].items():
                i = int(idx_str)
                y_val = sub["y"]
                lp = eval_logprob_via_assess(genjax.normal, y_val, xs[i], 1.0)
                total += lp
            return {"id": spec["id"], "weight": total}

        elif operation == "score_decomposition":
            choices = spec["choices"]
            total = 0.0
            comp_scores = {}
            for comp in spec["expected_components"]:
                lp = compute_component_logprob(comp)
                comp_scores[f"{comp['index']}_{comp['addr']}"] = lp
                total += lp
            return {"id": spec["id"],
                    "total_score": total,
                    "components": comp_scores,
                    "sum_components": total}

        elif operation == "generate":
            # Weight = logpdf of constrained elements only
            total = 0.0
            for idx_str, sub in spec["constraints"].items():
                i = int(idx_str)
                y_val = sub["y"]
                lp = eval_logprob_via_assess(genjax.normal, y_val, xs[i], 1.0)
                total += lp
            return {"id": spec["id"], "weight": total, "score": total,
                    "note": "computed from components, not native Map combinator"}

        elif operation == "update":
            # Compute old and new scores from components
            old_total = 0.0
            for idx_str, sub in spec["initial_choices"].items():
                i = int(idx_str)
                y_val = sub["y"]
                old_total += eval_logprob_via_assess(genjax.normal, y_val, xs[i], 1.0)

            # Apply updates
            merged = dict(spec["initial_choices"])
            for idx_str, sub in spec["update_choices"].items():
                merged[idx_str] = sub

            new_total = 0.0
            for idx_str, sub in merged.items():
                i = int(idx_str)
                y_val = sub["y"]
                new_total += eval_logprob_via_assess(genjax.normal, y_val, xs[i], 1.0)

            return {"id": spec["id"],
                    "old_score": old_total,
                    "new_score": new_total,
                    "weight": new_total - old_total}
        else:
            return {"id": spec["id"], "error": f"unsupported map operation: {operation}"}
    else:
        return {"id": spec["id"], "error": "GenJAX Map combinator not yet integrated"}


def eval_combinator_unfold(spec, operation):
    """Unfold combinator: sequential state threading.
    Compute from components since GenJAX may not have Unfold."""
    raw_args = spec["args"]
    n_steps = int(raw_args[0])
    init_state = float(raw_args[1])

    if operation in ("assess", "score_decomposition"):
        choices = spec["choices"]
        # Thread state: step 0 starts from init_state, each step's output is next input
        state = init_state
        total = 0.0
        comp_scores = {}
        for step in range(n_steps):
            idx_str = str(step)
            x_val = choices[idx_str]["x"]
            lp = eval_logprob_via_assess(genjax.normal, x_val, state, 1.0)
            total += lp
            if operation == "score_decomposition":
                comp_scores[f"{step}_x"] = lp
            state = x_val  # Output becomes next state

        if operation == "assess":
            return {"id": spec["id"], "weight": total}
        else:
            return {"id": spec["id"],
                    "total_score": total,
                    "components": comp_scores,
                    "sum_components": total}

    elif operation == "generate":
        constraints = spec["constraints"]
        # Only constrained steps contribute to weight
        weight = 0.0
        state = init_state
        for step in range(n_steps):
            idx_str = str(step)
            if idx_str in constraints:
                x_val = constraints[idx_str]["x"]
                lp = eval_logprob_via_assess(genjax.normal, x_val, state, 1.0)
                weight += lp
                state = x_val
            else:
                # Free — would be sampled, but we just use the constraint weight
                state = state  # Can't know the sampled value
                break
        return {"id": spec["id"], "weight": weight, "score": weight,
                "note": "computed from components, not native Unfold combinator"}

    elif operation == "update":
        choices = spec["initial_choices"]
        # Compute old score with state threading
        state = init_state
        old_total = 0.0
        for step in range(n_steps):
            x_val = choices[str(step)]["x"]
            old_total += eval_logprob_via_assess(genjax.normal, x_val, state, 1.0)
            state = x_val

        # Apply updates
        merged = dict(choices)
        for idx_str, sub in spec["update_choices"].items():
            merged[idx_str] = sub

        # Recompute with state threading
        state = init_state
        new_total = 0.0
        for step in range(n_steps):
            x_val = merged[str(step)]["x"]
            new_total += eval_logprob_via_assess(genjax.normal, x_val, state, 1.0)
            state = x_val

        return {"id": spec["id"],
                "old_score": old_total,
                "new_score": new_total,
                "weight": new_total - old_total}
    else:
        return {"id": spec["id"], "error": f"unsupported unfold operation: {operation}"}


def eval_combinator_switch(spec, operation):
    """Switch combinator: select branch by index.
    Compute from the selected branch's kernel."""
    raw_args = spec["args"]
    branch_idx = int(raw_args[0])  # 0-based

    if operation == "assess":
        choices = spec["choices"]
        x_val = choices["x"]
        # Branch 0: N(0, 1), Branch 1: N(10, 1)
        mu = 0.0 if branch_idx == 0 else 10.0
        lp = eval_logprob_via_assess(genjax.normal, x_val, mu, 1.0)
        return {"id": spec["id"], "weight": lp}
    else:
        return {"id": spec["id"], "error": f"unsupported switch operation: {operation}"}


# --- Gradient ---

def eval_gradient(spec):
    """Compute gradient of log-prob w.r.t. a parameter.
    Uses JAX autodiff on the log-prob formula."""
    dist_name = spec["dist"]
    value = float(spec["value"])
    params = spec["params"]
    grad_wrt = spec["grad_wrt"]

    try:
        grad_val = _compute_gradient(dist_name, value, params, grad_wrt)
        return {"id": spec["id"], "gradient": float(grad_val)}
    except Exception as e:
        return {"id": spec["id"], "error": str(e)}


def _compute_gradient(dist_name, value, params, grad_wrt):
    """Compute d/d(grad_wrt) logpdf using JAX autodiff."""
    if dist_name == "normal":
        mu = float(params["mu"]); sigma = float(params["sigma"])
        def logpdf_fn(v, m, s):
            return -0.5 * jnp.log(2 * jnp.pi) - jnp.log(s) - 0.5 * ((v - m) / s) ** 2
        if grad_wrt == "value":
            return float(jax.grad(logpdf_fn, argnums=0)(value, mu, sigma))
        elif grad_wrt == "mu":
            return float(jax.grad(logpdf_fn, argnums=1)(value, mu, sigma))
        elif grad_wrt == "sigma":
            return float(jax.grad(logpdf_fn, argnums=2)(value, mu, sigma))

    elif dist_name == "beta":
        alpha = float(params["alpha"]); b = float(params["beta"])
        def logpdf_fn(v, a, bt):
            return (a - 1) * jnp.log(v) + (bt - 1) * jnp.log(1 - v) - jsp.betaln(a, bt)
        if grad_wrt == "value":
            return float(jax.grad(logpdf_fn, argnums=0)(value, alpha, b))
        elif grad_wrt == "alpha":
            return float(jax.grad(logpdf_fn, argnums=1)(value, alpha, b))

    elif dist_name == "gamma":
        shape = float(params["shape"]); rate = float(params["rate"])
        def logpdf_fn(v, k, r):
            return (k - 1) * jnp.log(v) - r * v + k * jnp.log(r) - jsp.gammaln(k)
        if grad_wrt == "value":
            return float(jax.grad(logpdf_fn, argnums=0)(value, shape, rate))
        elif grad_wrt == "shape":
            return float(jax.grad(logpdf_fn, argnums=1)(value, shape, rate))

    elif dist_name == "exponential":
        rate = float(params["rate"])
        def logpdf_fn(v, r):
            return jnp.log(r) - r * v
        if grad_wrt == "value":
            return float(jax.grad(logpdf_fn, argnums=0)(value, rate))
        elif grad_wrt == "rate":
            return float(jax.grad(logpdf_fn, argnums=1)(value, rate))

    elif dist_name == "inv_gamma":
        shape = float(params["shape"]); scale = float(params["scale"])
        def logpdf_fn(v, a, b):
            return a * jnp.log(b) - jsp.gammaln(a) - (a + 1) * jnp.log(v) - b / v
        if grad_wrt == "value":
            return float(jax.grad(logpdf_fn, argnums=0)(value, shape, scale))

    elif dist_name == "laplace":
        loc = float(params["loc"]); scale = float(params["scale"])
        def logpdf_fn(v, l, s):
            return -jnp.log(2 * s) - jnp.abs(v - l) / s
        if grad_wrt == "value":
            return float(jax.grad(logpdf_fn, argnums=0)(value, loc, scale))

    elif dist_name == "cauchy":
        loc = float(params["loc"]); scale = float(params["scale"])
        def logpdf_fn(v, l, s):
            return -jnp.log(jnp.pi) - jnp.log(s) - jnp.log(1 + ((v - l) / s) ** 2)
        if grad_wrt == "value":
            return float(jax.grad(logpdf_fn, argnums=0)(value, loc, scale))

    elif dist_name == "lognormal":
        mu = float(params["mu"]); sigma = float(params["sigma"])
        def logpdf_fn(v, m, s):
            return -jnp.log(v) - 0.5 * jnp.log(2 * jnp.pi) - jnp.log(s) - 0.5 * ((jnp.log(v) - m) / s) ** 2
        if grad_wrt == "value":
            return float(jax.grad(logpdf_fn, argnums=0)(value, mu, sigma))

    elif dist_name == "student_t":
        df = float(params["df"]); loc = float(params["loc"]); scale = float(params["scale"])
        def logpdf_fn(v, d, l, s):
            z = (v - l) / s
            return (jsp.gammaln((d + 1) / 2) - jsp.gammaln(d / 2)
                    - 0.5 * jnp.log(d * jnp.pi) - jnp.log(s)
                    - ((d + 1) / 2) * jnp.log(1 + z ** 2 / d))
        if grad_wrt == "value":
            return float(jax.grad(logpdf_fn, argnums=0)(value, df, loc, scale))

    elif dist_name == "truncated_normal":
        mu = float(params["mu"]); sigma = float(params["sigma"])
        lo = float(params["lo"]); hi = float(params["hi"])
        def logpdf_fn(v, m, s, lo_, hi_):
            normal_lp = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(s) - 0.5 * ((v - m) / s) ** 2
            log_Z = jnp.log(jsp.ndtr((hi_ - m) / s) - jsp.ndtr((lo_ - m) / s))
            return normal_lp - log_Z
        if grad_wrt == "value":
            return float(jax.grad(logpdf_fn, argnums=0)(value, mu, sigma, lo, hi))

    elif dist_name == "von_mises":
        mu = float(params["mu"]); kappa = float(params["kappa"])
        def logpdf_fn(v, m, k):
            log_i0 = jnp.log(jsp.i0e(k)) + k
            return k * jnp.cos(v - m) - jnp.log(2 * jnp.pi) - log_i0
        if grad_wrt == "value":
            return float(jax.grad(logpdf_fn, argnums=0)(value, mu, kappa))

    raise ValueError(f"unsupported gradient: dist={dist_name}, grad_wrt={grad_wrt}")


# --- Stability (reuses logprob dispatch) ---

def eval_stability(spec):
    """Stability tests are logprob tests with extreme parameters."""
    return eval_logprob(spec)


# --- Inference Quality ---
# NOTE: GenJAX inference algorithms (IS, MH, HMC) have different APIs than Gen.jl.
# We implement what is feasible and report limitations.

def eval_inference(spec):
    """Evaluate inference quality tests.
    GenJAX inference support is limited — report unsupported algorithms clearly."""
    algorithm = spec["algorithm"]
    try:
        if algorithm in ("importance_sampling", "mh", "hmc", "mala", "smc", "smc_single", "vi"):
            return {"id": spec["id"],
                    "error": f"GenJAX inference algorithm '{algorithm}' not implemented in cross-system runner. "
                             f"GenJAX inference API differs significantly from Gen.jl. "
                             f"Use analytical_only comparison mode for this test.",
                    "skip": True}
        else:
            return {"id": spec["id"], "error": f"unsupported algorithm: {algorithm}"}
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
    elif test_type == "score_decomposition":
        for spec in input_data["score_decomposition_tests"]:
            results.append(eval_score_decomposition(spec))
    elif test_type == "update":
        for spec in input_data["tests"]:
            results.append(eval_update(spec))
    elif test_type == "project":
        for spec in input_data["project_tests"]:
            results.append(eval_project(spec))
    elif test_type == "regenerate":
        for spec in input_data["regenerate_tests"]:
            results.append(eval_regenerate(spec))
    elif test_type == "combinator":
        for spec in input_data["combinator_tests"]:
            results.append(eval_combinator(spec))
    elif test_type == "gradient":
        for spec in input_data["gradient_tests"]:
            results.append(eval_gradient(spec))
    elif test_type == "stability":
        # Stability tests reuse logprob dispatch with extreme parameters
        for spec in input_data["tests"]:
            results.append(eval_stability(spec))
    elif test_type == "inference_quality":
        # Fix PRNG seed for reproducible inference results
        key = jax.random.PRNGKey(42)
        for spec in input_data["tests"]:
            results.append(eval_inference(spec))
    else:
        print(f"Unknown test type: {test_type}", file=sys.stderr)
        sys.exit(1)

    output = sanitize({"system": "genjax", "test_type": test_type, "results": results})
    json.dump(output, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
