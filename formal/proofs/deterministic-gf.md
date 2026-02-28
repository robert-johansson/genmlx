# Deterministic Generative Function Wrappers — TODO 10.16

> Proposition: CustomGradientGF and NeuralNetGF satisfy GFI contracts C1-C5
> trivially, with score = 0 and empty trace type. Custom gradients and
> neural network forward passes compose correctly with probabilistic models.

---

## 1. Deterministic GF Type

### 1.1 Definition

A **deterministic generative function** is a generative function with
empty trace type:

```
G_det = G_{} η
```

Its denotation is the pair:

```
⟦G_{} η⟧ = (δ_∗, f)
```

where:
- **δ_∗** is the Dirac measure on the trivial (empty) trace space {∗}
- **f : {∗} → ⟦η⟧** is the deterministic return function

Since the trace type is empty, there are no random choices, and the
score (density of δ_∗ w.r.t. counting measure on {∗}) is 1 (or 0 in
log-space).

### 1.2 GFI Operations on Deterministic GFs

All GFI operations simplify dramatically for G_{} η:

```
simulate(G_det, args)           = ({}, f(args), 0)       -- score = 0
generate(G_det, args, obs)      = ({}, f(args), 0, 0)    -- weight = 0
update(G_det, tr, constraints)  = ({}, f(args), 0, 0, {}) -- weight = 0
regenerate(G_det, tr, sel)      = ({}, f(args), 0, 0)    -- weight = 0
project(G_det, tr, sel)         = 0                       -- nothing to project
assess(G_det, args, choices)    = (f(args), 0)           -- weight = 0
propose(G_det, args)            = ({}, f(args), 0)       -- weight = 0
```

All weights are 0 (log-space), all choice maps are empty, all discards
are empty. The only non-trivial output is the return value f(args).

---

## 2. CustomGradientGF Satisfies C1-C5

### 2.1 Record Structure

CustomGradientGF (ref: `custom_gradient.cljs:13-55`) wraps a deterministic
computation with an optional custom gradient function:

```
CustomGradientGF = {
  forward-fn  : Args → η,            -- deterministic forward computation
  gradient-fn : (Args × η × η) → [η],  -- custom backward pass (optional)
  arg-grads   : [𝔹]                   -- which arguments accept gradients
}
```

### 2.2 Contract Verification

**C1 (simulate correctness):** `simulate` (ref: `custom_gradient.cljs:15-20`)
calls `(apply forward-fn args)` and returns a Trace with:
- `choices = cm/EMPTY` (empty choice map)
- `score = (mx/scalar 0.0)`
- `retval = (apply forward-fn args)`

The denotation is (δ_∗, forward-fn), matching G_{} η. ✓

**C2 (generate correctness):** `generate` (ref: `custom_gradient.cljs:23-26`)
returns `{:trace (simulate ...) :weight (mx/scalar 0.0)}`.

Weight = 0 because there are no constrained addresses (empty trace type).
The constraint argument is ignored since dom(γ) = ∅. ✓

**C3 (assess correctness):** `assess` (ref: `custom_gradient.cljs:29-31`)
returns `{:retval (apply forward-fn args) :weight (mx/scalar 0.0)}`.

The weight (log-density) is 0 because the only trace is ∗, and
log δ_∗(∗) = 0. ✓

**C4 (update correctness):** `update` (ref: `custom_gradient.cljs:40-43`)
re-executes forward-fn and returns:
- New trace with same empty choices
- Weight = 0 (no density change: 0 - 0 = 0)
- Discard = cm/EMPTY

Since dom(constraints) ∩ dom(γ) = ∅ ∩ ∅ = ∅, no addresses change,
and the weight is correctly 0. ✓

**C5 (regenerate correctness):** `regenerate` (ref: `custom_gradient.cljs:46-48`)
re-executes and returns weight = 0.

Since selected(sel) ∩ dom(γ) = ∅, no addresses are resampled,
and the weight is correctly 0. ✓

### 2.3 Additional Operations

**propose:** (ref: `custom_gradient.cljs:34-37`) Returns
`{:choices cm/EMPTY :weight 0.0 :retval (apply forward-fn args)}`. ✓

**project:** (ref: `custom_gradient.cljs:51-52`) Returns `0.0`.
Since there are no addresses to select, the projected log-probability
is 0. ✓

---

## 3. NeuralNetGF Satisfies C1-C5

### 3.1 Record Structure

NeuralNetGF (ref: `nn.cljs:37-74`) wraps an MLX neural network module:

```
NeuralNetGF = {
  module : nn.Module    -- MLX neural network with .forward method
}
```

The forward computation is `(.forward module input)`, which is
deterministic (no sampling).

### 3.2 Contract Verification

The verification is structurally identical to CustomGradientGF:

**C1 (simulate):** `simulate` (ref: `nn.cljs:39-43`) calls
`(.forward module (first args))`, returns Trace with score = 0
and choices = cm/EMPTY. ✓

**C2 (generate):** (ref: `nn.cljs:46-47`) Returns trace + weight = 0. ✓

**C3 (assess):** (ref: `nn.cljs:50-51`) Returns retval + weight = 0. ✓

**C4 (update):** (ref: `nn.cljs:59-62`) Re-executes, weight = 0,
discard = cm/EMPTY. ✓

**C5 (regenerate):** (ref: `nn.cljs:65-67`) Re-executes, weight = 0. ✓

**project:** (ref: `nn.cljs:70-71`) Returns 0. ✓

### 3.3 Differentiability

NeuralNetGF declares `has-argument-grads = [true]` (ref: `nn.cljs:74`),
indicating that gradients can flow through the first argument.

The forward pass `(.forward module input)` is differentiable via MLX's
automatic differentiation system. The nn.Module's parameters are
differentiable by construction (MLX tracks gradients for all nn.Module
parameters).

---

## 4. Gradient Interaction with Probabilistic Models

### 4.1 Deterministic GF in Probabilistic Context

When a deterministic GF (either CustomGradientGF or NeuralNetGF) is
spliced into a probabilistic model via `dyn/splice`, the combined model
has trace type γ_parent ⊕ {} = γ_parent. The deterministic GF contributes:

- **Score contribution:** 0 (no trace sites)
- **Gradient contribution:** Through the return value

### 4.2 Gradient Flow

Consider a model:

```
(gen [x]
  (let [h (dyn/splice :nn neural-net-gf [x])     -- deterministic
        y (dyn/trace :y (dist/gaussian h 1.0))]   -- probabilistic
    y))
```

The score of this model is:

```
score = 0 + log p(y | h)      -- neural net contributes 0
      = log N(y; h, 1)
```

The gradient ∇_x score flows through:
1. ∇_h log N(y; h, 1) — gradient of Gaussian log-prob w.r.t. mean
2. ∇_x h = ∇_x nn(x) — gradient of neural net forward pass

This chain of gradients is computed correctly by MLX autograd because:
- The neural net forward pass is tracked in the computation graph
- The Gaussian log-prob is differentiable
- The score accumulation is a sum of differentiable terms

### 4.3 IHasArgumentGrads Protocol

Both CustomGradientGF and NeuralNetGF implement `IHasArgumentGrads`:

```
CustomGradientGF: arg-grads = user-specified vector of booleans
NeuralNetGF:      arg-grads = [true]
```

This protocol (ref: `custom_gradient.cljs:54-55`, `nn.cljs:74`) informs
the gradient system which arguments can receive gradients. When
`arg-grads[i] = true`, the gradient computation includes ∇_{args[i]}
in its backward pass.

### 4.4 Custom Gradient Override

For CustomGradientGF with a `gradient-fn`, the backward pass uses the
custom gradient instead of MLX autograd:

```
∂L/∂args = gradient-fn(args, retval, ∂L/∂retval)
```

This enables:
1. **Efficient gradients:** Custom implementations that avoid materializing
   large intermediate Jacobians
2. **Approximate gradients:** Straight-through estimators, surrogate
   gradients for non-differentiable operations
3. **External code:** Gradients for functions implemented outside MLX

The custom gradient must satisfy the **gradient contract**: for
correctness of upstream gradient computations, `gradient-fn` should
return the true gradient ∂retval/∂args (or a consistent approximation).
If the custom gradient is incorrect, optimization may fail to converge,
but GFI contracts C1-C5 are unaffected (they concern only the forward
pass and score computation).

---

## 5. Composition with Combinators

### 5.1 Proposition (Combinator Compatibility)

**Statement.** Since CustomGradientGF and NeuralNetGF satisfy C1-C5,
all combinator compositionality results from `combinators.md` apply.

**Proof.** The combinator proofs in `combinators.md` require only that
each component generative function satisfies C1-C5. The proofs do not
assume non-empty trace types, non-zero scores, or stochastic behavior.
Therefore, deterministic GFs compose correctly with:

- **Map(det-gf):** Applies the deterministic computation to each element.
  Score = 0 for each application. Total score = 0. ✓
- **Switch(det-gf, prob-gf):** When the deterministic branch is selected,
  score contribution is 0. ✓
- **Unfold(det-gf):** Sequential deterministic computation. Total score = 0. ✓
- **Scan(det-gf):** Accumulating deterministic computation. Total score = 0. ✓

### 5.2 Mixed Deterministic-Probabilistic Composition

The typical use case is combining a deterministic GF with probabilistic
GFs. For example, in an amortized inference model:

```
(gen [obs]
  (let [params (dyn/splice :encoder encoder-gf [obs])  -- NeuralNetGF
        z      (dyn/trace :z (dist/gaussian
                  (:mu params) (:sigma params)))]       -- probabilistic
    z))
```

The trace type is γ = {:z → ℝ} (only the probabilistic address).
The encoder contributes score = 0 but its output parameterizes the
distribution at :z. Gradients flow from the ELBO objective through
:z's log-prob, through the encoder parameters, enabling end-to-end
training via ADEV or VI.

---

## 6. Implementation Correspondence

| Formal Concept | Implementation | Location |
|----------------|---------------|----------|
| G_{} η (deterministic GF) | CustomGradientGF / NeuralNetGF | `custom_gradient.cljs:13` / `nn.cljs:37` |
| forward-fn | `:forward` in config map | `custom_gradient.cljs:57-63` |
| gradient-fn | `:gradient` in config map | `custom_gradient.cljs:57-63` |
| nn.Module forward | `(.forward module input)` | `nn.cljs:40` |
| score = 0 | `(mx/scalar 0.0)` | `custom_gradient.cljs:20` / `nn.cljs:43` |
| choices = EMPTY | `cm/EMPTY` | `custom_gradient.cljs:18` / `nn.cljs:42` |
| weight = 0 | `(mx/scalar 0.0)` | `custom_gradient.cljs:26` / `nn.cljs:47` |
| IHasArgumentGrads | `has-argument-grads` protocol method | `custom_gradient.cljs:54-55` / `nn.cljs:74` |
| custom-gradient-gf factory | `(custom-gradient-gf config)` | `custom_gradient.cljs:57-63` |
| nn->gen-fn factory | `(nn->gen-fn module)` | `nn.cljs:80-84` |
| Training utilities | `value-and-grad`, `optimizer`, `step!` | `nn.cljs:90-119` |
