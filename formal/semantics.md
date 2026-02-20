# Denotational Semantics of λ_MLX — TODO 10.5

> QBS interpretation of λ_MLX types and terms, extending λ_GEN (Figure 11,
> POPL 2026) with handler transition semantics as a state monad, splice
> semantics, and the formal meaning of each GFI mode.

---

## 1. Type Denotations

Following the paper's use of quasi-Borel spaces (QBS), each type τ maps to
a space ⟦τ⟧.

### 1.1 Ground and Standard Types (same as λ_GEN)

```
⟦𝔹⟧             = 𝔹                          (two-element set)
⟦ℝ⟧             = ℝ                          (real line)
⟦ℝ_{>0}⟧        = ℝ_{>0}                     (positive reals)
⟦T[n]⟧          = ⟦T⟧ⁿ                       (n-fold product)
⟦1⟧             = {∗}                         (singleton)
⟦η₁ × η₂⟧      = ⟦η₁⟧ × ⟦η₂⟧               (product)
⟦{k₁:η₁,…}⟧    = ⟦η₁⟧ × ⋯ × ⟦ηₙ⟧          (labeled product)
⟦τ₁ → τ₂⟧      = ⟦τ₁⟧ → ⟦τ₂⟧               (QBS morphism space)
⟦τ₁ × τ₂⟧      = ⟦τ₁⟧ × ⟦τ₂⟧               (product)
```

### 1.2 Probabilistic Types (same as λ_GEN)

```
⟦D η⟧           = 𝒫_≪(⟦η⟧)                  (measures abs. continuous w.r.t. ν_η)
⟦P η⟧           = 𝒫(⟦η⟧)                    (probability measures)
⟦G_γ η⟧         = 𝒫_≪(⟦γ⟧) × (⟦γ⟧ → ⟦η⟧)  (trace measure × return function)
```

A generative function of type G_γ η is a pair (μ, f) where:
- **μ** is a measure on ⟦γ⟧, absolutely continuous w.r.t. the stock
  measure ν_γ, representing the distribution over traces
- **f : ⟦γ⟧ → ⟦η⟧** maps traces to return values
- The density (score) at trace u is w(u) = dμ/dν_γ(u)

### 1.3 Handler Type (NEW)

```
⟦H(σ, τ)⟧       = ⟦σ⟧ → ⟦τ⟧ × ⟦σ⟧          (state-passing function)
```

The handler type H(σ, τ) denotes a state monad transformer. A value of
type H(σ, τ) is a function that takes a state ⟦σ⟧, produces a result
⟦τ⟧ and an updated state ⟦σ⟧. This directly models the pure transitions
in `handler.cljs`.

### 1.4 Auxiliary Types (NEW)

```
⟦EditReq(γ)⟧    = ⟦γ⟧ + ⟦Sel(γ)⟧ + (⟦G_{γ'} η'⟧ × Args × ⟦G_{γ''} η''⟧ × Args)
⟦Sel(γ)⟧        = 𝒫(dom(γ))                  (subsets of trace addresses)
⟦Δ⟧             = {NoChange} + {UnknownChange} + (⟦η⟧ × ⟦η⟧) + 𝒫(ℕ) + …
```

---

## 2. Stock Measures

Stock measures are defined by induction on ground type η:

```
ν_ℝ             = λ_Leb                       (Lebesgue measure on ℝ)
ν_{ℝ_{>0}}      = λ_Leb |_{ℝ_{>0}}           (Lebesgue restricted to ℝ_{>0})
ν_𝔹             = #                            (counting measure on {true, false})
ν_1             = δ_∗                          (Dirac at the singleton)
ν_{T[n]}        = ν_T^{⊗n}                    (n-fold product of stock measure)
ν_{η₁ × η₂}    = ν_{η₁} ⊗ ν_{η₂}            (product measure)
ν_{{k₁:η₁,…}}  = ν_{η₁} ⊗ ⋯ ⊗ ν_{ηₙ}       (product of component measures)
```

These are the reference measures against which densities (scores) are computed.
Continuous types use Lebesgue measure; discrete types use counting measure.

---

## 3. Term Denotations (inherited from λ_GEN)

Selected term denotations from Figure 11:

```
⟦()⟧(ρ)                = ∗
⟦c⟧(ρ)                 = c                       (constant)
⟦x⟧(ρ)                 = ρ(x)                    (variable lookup)
⟦λx.t⟧(ρ)              = λv. ⟦t⟧(ρ[x ↦ v])      (abstraction)
⟦t₁ t₂⟧(ρ)             = ⟦t₁⟧(ρ)(⟦t₂⟧(ρ))       (application)
⟦(t₁, t₂)⟧(ρ)          = (⟦t₁⟧(ρ), ⟦t₂⟧(ρ))     (pair)
⟦π_i t⟧(ρ)             = (⟦t⟧(ρ))_i              (projection)
⟦t[k]⟧(ρ)              = (⟦t⟧(ρ))_k              (field access)
⟦let x = t₁ in t₂⟧(ρ)  = ⟦t₂⟧(ρ[x ↦ ⟦t₁⟧(ρ)])  (let binding)

⟦return_P t⟧(ρ)         = δ_{⟦t⟧(ρ)}              (Dirac distribution)
⟦sample t⟧(ρ)           = ⟦t⟧(ρ)                  (sample = identity on measures)
⟦return_G t⟧(ρ)         = (δ_{∗}, λ(). ⟦t⟧(ρ))    (empty trace, constant return)

⟦do_P{x ← t; m}⟧(ρ)(A) = ∫ ⟦t⟧(ρ, du)  ⟦do_P{m}⟧(ρ[x ↦ u])(A)
```

For generative functions (subscript ₁ = measure, ₂ = return function):

```
⟦do_G{x ← t; m}⟧₁(ρ)(A) = ∫ ⟦t⟧₁(ρ, du)  ⟦do_G{m}⟧₁(ρ[x ↦ ⟦t⟧₂(ρ)(u)])(A')
  where A = {(u, u') | u ∈ π_{grade(t)}(A), u' ∈ A'}

⟦do_G{x ← t; m}⟧₂(ρ) = λtr. ⟦do_G{m}⟧₂(ρ[x ↦ ⟦t⟧₂(ρ)(π_{grade(t)}(tr))])
                                           (π_{grade(do_G{m})}(tr))
```

---

## 4. Handler Transition Semantics (NEW)

Each handler mode defines a semantics for the `trace(a, d)` effect
operation as a state transition in H(σ_m, η). These directly correspond
to the pure transition functions in `handler.cljs`.

All transitions are written in multiplicative notation for scores (matching
the paper). The implementation uses log-space: multiply ↔ `mx/add`,
density ↔ `dc/dist-log-prob`.

### 4.1 Simulate Transition

```
⟦trace(a, d)⟧_simulate : H(σ_sim, η)

⟦trace(a, d)⟧_simulate(σ) =
  let (k₁, k₂)  = split(σ.key)
      v          = sample(d, k₂)
      w          = density_d(v)                    -- dμ_d/dν_η(v)
  in (v, { key     : k₁,
            choices : σ.choices[a ↦ v],
            score   : σ.score · w })
```

**Implementation**: `simulate-transition` at `handler.cljs:72-81`.

### 4.2 Generate Transition

```
⟦trace(a, d)⟧_generate : H(σ_gen, η)

⟦trace(a, d)⟧_generate(σ) =
  if a ∈ dom(σ.constraints) then
    let v  = σ.constraints(a)
        w  = density_d(v)
    in (v, { key         : σ.key,                  -- key NOT consumed
              choices     : σ.choices[a ↦ v],
              score       : σ.score · w,
              weight      : σ.weight · w,           -- weight tracks constrained
              constraints : σ.constraints })
  else
    let (v, σ') = ⟦trace(a, d)⟧_simulate(σ ↾ σ_sim)
    in (v, σ' ⊎ { weight      : σ.weight,          -- weight unchanged
                    constraints : σ.constraints })
```

When the address is constrained, both score and weight accumulate the
density. When unconstrained, we delegate to simulate and weight is
unchanged — the unconstrained site contributes to the score (for
correctness of the joint density) but not to the importance weight.

**Implementation**: `generate-transition` at `handler.cljs:83-94`.

### 4.3 Update Transition

```
⟦trace(a, d)⟧_update : H(σ_upd, η)

⟦trace(a, d)⟧_update(σ) =
  if a ∈ dom(σ.constraints) then
    -- Case 1: New constraint provided
    let v_new  = σ.constraints(a)
        w_new  = density_d(v_new)
        v_old  = σ.old-choices(a)                  -- may be absent
        w_old  = if a ∈ dom(σ.old-choices)
                 then density_d(v_old) else 1
    in (v_new, { key         : σ.key,
                  choices     : σ.choices[a ↦ v_new],
                  score       : σ.score · w_new,
                  weight      : σ.weight · (w_new / w_old),
                  constraints : σ.constraints,
                  old-choices : σ.old-choices,
                  discard     : if v_old ≠ ⊥
                                then σ.discard[a ↦ v_old]
                                else σ.discard })

  else if a ∈ dom(σ.old-choices) then
    -- Case 2: Keep old value
    let v  = σ.old-choices(a)
        w  = density_d(v)
    in (v, { …, choices : σ.choices[a ↦ v],
                 score   : σ.score · w })
             -- weight unchanged (same value, same density)

  else
    -- Case 3: New address, sample fresh
    ⟦trace(a, d)⟧_simulate(σ ↾ σ_sim) ⊎ {update fields}
```

**Implementation**: `update-transition` at `handler.cljs:96-124`.

### 4.4 Regenerate Transition

```
⟦trace(a, d)⟧_regenerate : H(σ_reg, η)

⟦trace(a, d)⟧_regenerate(σ) =
  if a ∈ selected(σ.selection) then
    -- Resample: draw new value, compute weight adjustment
    let (k₁, k₂)  = split(σ.key)
        v_new      = sample(d, k₂)
        w_new      = density_d(v_new)
        v_old      = σ.old-choices(a)
        w_old      = if a ∈ dom(σ.old-choices)
                     then density_d(v_old) else 1
    in (v_new, { key         : k₁,
                  choices     : σ.choices[a ↦ v_new],
                  score       : σ.score · w_new,
                  weight      : σ.weight · (w_new / w_old),
                  old-choices : σ.old-choices,
                  selection   : σ.selection })
  else
    -- Not selected: keep old value
    let v  = σ.old-choices(a)
        w  = density_d(v)
    in (v, { …, choices : σ.choices[a ↦ v],
                 score   : σ.score · w })
             -- weight unchanged
```

Note: The regenerate weight at the DynamicGF level is computed as
`new_score - old_score - proposal_ratio` (see `dynamic.cljs:89-107`),
where proposal_ratio is the accumulated weight from the transitions.
This ensures the MH acceptance ratio is correct.

**Implementation**: `regenerate-transition` at `handler.cljs:126-153`.

### 4.5 Project Transition

```
⟦trace(a, d)⟧_project : H(σ_proj, η)

⟦trace(a, d)⟧_project(σ) =
  let v  = σ.old-choices(a)                        -- replay old value
      w  = density_d(v)
  in (v, { key     : σ.key,
            choices : σ.choices[a ↦ v],
            score   : σ.score · w,
            weight  : if a ∈ selected(σ.selection)
                      then σ.weight · w             -- accumulate for selected
                      else σ.weight })               -- skip unselected
```

**Implementation**: `project-transition` at `handler.cljs:155-171`.

---

## 5. Splice Semantics (NEW)

When a generative function body contains `splice(k, g, args)`, the
handler delegates to the sub-GF's own GFI operation, scoping the
relevant state fields under address k:

```
⟦splice(k, g, args)⟧_mode(σ) =
  let sub-constraints = σ.constraints ↾ k          -- scope constraints to k
      sub-old-choices = σ.old-choices ↾ k           -- scope old choices to k
      sub-selection   = σ.selection ↾ k             -- scope selection to k

      sub-result = mode{g}(args,
                           constraints=sub-constraints,
                           old-choices=sub-old-choices,
                           selection=sub-selection,
                           key=σ.key)

  in (sub-result.retval,
      merge-sub-result(σ, k, sub-result))
```

where `merge-sub-result` nests the sub-result's choices, score, weight,
and discard under address k in the parent state.

**Implementation**: `trace-gf!` and `execute-sub` at `handler.cljs:420-448`
and `dynamic.cljs:148-194`.

---

## 6. Batched Transition Semantics (NEW)

For vectorized execution with N particles, each transition produces
[N]-shaped values. The key insight is that the transition functions are
**structurally identical** to their scalar counterparts — only the
sampling and log-prob operations change shape:

```
⟦trace(a, d)⟧_simulate^N(σ) =
  let (k₁, k₂) = split(σ.key)
      v         = sample_n(d, k₂, N)              -- [N]-shaped
      w         = density_d(v)                     -- [N]-shaped (broadcasts)
  in (v, { key     : k₁,
            choices : σ.choices[a ↦ v],
            score   : σ.score + w })               -- [N] + [N] = [N]
                                                    -- or scalar + [N] = [N]
```

The structural identity between scalar and batched transitions is what
makes the broadcasting correctness theorem (see `proofs/broadcasting.md`)
possible. The handler transitions never inspect array shapes — they work
with whatever shapes the sampling and scoring operations produce.

**Implementation**: `batched-simulate-transition` at `handler.cljs:177-187`,
and analogously for generate, update, regenerate.
