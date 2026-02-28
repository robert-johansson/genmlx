# Program Transformations for λ_MLX — TODO 10.6

> Source-to-source program transformations for all GFI operations, extending
> λ_GEN (Figure 12, POPL 2026) which formalizes only simulate{−} and
> assess{−}. λ_MLX adds generate{−}, update{−}, regenerate{−}, edit{−},
> project{−}, and propose{−}.

---

## 1. Inherited Transformations (from λ_GEN)

### 1.1 simulate{−} — Forward Sampling

**On types:**
```
simulate{D η}   = P (η × ℝ)
simulate{G_γ η} = P (γ × η × ℝ)
```

**On terms:**
```
simulate{d : D η} =
  do_P{ v ← sample d;
        w := density_d(v);
        return_P (v, w) }

simulate{return_G t} =
  return_P ({}, simulate{t}, 1)

simulate{trace(k, t : D η)} =
  do_P{ (v, w) ← simulate{t};
        return_P ({k : v}, v, w) }

simulate{trace(k, t : G_γ η)} =
  do_P{ (u, x, w) ← simulate{t};
        return_P ({k : u}, x, w) }

simulate{do_G{x ← t; m}} =
  do_P{ (u, x, w)    ← simulate{t};
        (u', y, w')   ← simulate{do_G{m}[x ↦ x]};
        return_P (u ⊕ u', y, w · w') }
```

### 1.2 assess{−} — Density Evaluation

**On types:**
```
assess{D η}   = η → ℝ
assess{G_γ η} = γ → η × ℝ
```

**On terms:**
```
assess{d : D η} =
  λv. density_d(v)

assess{return_G t} =
  λu. (assess{t}, 1)

assess{trace(k, t : D η)} =
  λu. let w = assess{t}(u[k]) in (u[k], w)

assess{do_G{x ← t; m}} =
  λu. let (x, w)  = assess{t}(π_{grade(t)}(u)) in
      let (y, w') = assess{do_G{m}}(π_{grade(do_G{m})}(u)) in
      (y, w · w')
```

Both transformations act homomorphically on product and function types and
leave ground types unchanged.

---

## 2. generate{−} — Constrained Sampling (NEW)

Generate produces a trace where some addresses are constrained (observed)
and the rest are sampled freely. The weight accumulates the density at
constrained sites.

### 2.1 Type Transformation

```
generate{D η}   = η_obs → P (η × ℝ × ℝ)        -- value × score × weight
generate{G_γ η} = γ_obs → P (γ × η × ℝ × ℝ)    -- trace × retval × score × weight
```

where γ_obs ⊆ γ is a partial constraint map (subset of addresses).

### 2.2 Term Transformation

```
generate{d : D η}(obs) =
  if obs ≠ ∅ then
    let v = obs
        w = density_d(v)
    in return_P (v, w, w)                          -- score = weight = density
  else
    do_P{ v ← sample d;
          w := density_d(v);
          return_P (v, w, 0) }                     -- weight = 0 (unconstrained)

generate{return_G t}(obs) =
  return_P ({}, generate{t}, 1, 0)

generate{trace(k, t : D η)}(obs) =
  if k ∈ dom(obs) then
    let v = obs(k)
        w = density_d(v)
    in return_P ({k : v}, v, w, w)                 -- constrained: weight = density
  else
    do_P{ (v, w, _) ← simulate{t};
          return_P ({k : v}, v, w, 0) }            -- unconstrained: weight = 0

generate{trace(k, t : G_γ η)}(obs) =
  let sub-obs = obs ↾ k                           -- scope constraints
  in do_P{ (u, x, w, w_c) ← generate{t}(sub-obs);
           return_P ({k : u}, x, w, w_c) }

generate{do_G{x ← t; m}}(obs) =
  do_P{ (u, x, w, w_c)     ← generate{t}(obs ↾ grade(t));
        (u', y, w', w_c')   ← generate{do_G{m}[x ↦ x]}(obs ↾ grade(do_G{m}));
        return_P (u ⊕ u', y, w · w', w_c + w_c') }
```

Note: In the implementation, score and weight are both tracked in log-space.
The final weight from `generate` is the sum of log-densities at all
constrained addresses.

**Implementation**: `generate-transition` (`handler.cljs:83-94`) and
`DynamicGF/generate` (`dynamic.cljs:53-66`).

---

## 3. update{−} — Trace Modification (NEW)

Update takes an existing trace and new constraints, producing a modified
trace with a weight reflecting the density change and a discard map
containing displaced old values.

### 3.1 Type Transformation

```
update{D η}   = (η × ℝ) → η_new → P (η × ℝ × ℝ × η_disc)
update{G_γ η} = (γ × η × ℝ) → γ_new → P (γ × η × ℝ × ℝ × γ_disc)
```

where the output weight = new_score - old_score (in log-space).

### 3.2 Term Transformation

```
update{trace(k, t : D η)}(tr_old, constraints) =
  if k ∈ dom(constraints) then
    -- Case 1: New constraint replaces old value
    let v_new = constraints(k)
        w_new = density_d(v_new)
        v_old = tr_old[k]
        w_old = density_d(v_old)
    in return_P ({k : v_new}, v_new, w_new,
                  w_new / w_old,                   -- weight = ratio of densities
                  {k : v_old})                     -- discard = old value

  else if k ∈ dom(tr_old) then
    -- Case 2: Keep old value
    let v = tr_old[k]
        w = density_d(v)
    in return_P ({k : v}, v, w,
                  1,                                -- weight = 1 (no change)
                  {})                               -- discard = empty

  else
    -- Case 3: New address, sample fresh
    do_P{ (v, w, _) ← simulate{t};
          return_P ({k : v}, v, w, 1, {}) }

update{do_G{x ← t; m}}(tr_old, constraints) =
  let tr_t = π_{grade(t)}(tr_old)
      c_t  = constraints ↾ grade(t)
      tr_m = π_{grade(do_G{m})}(tr_old)
      c_m  = constraints ↾ grade(do_G{m})
  in do_P{ (u, x, w, δw, disc)       ← update{t}(tr_t, c_t);
           (u', y, w', δw', disc')    ← update{do_G{m}[x ↦ x]}(tr_m, c_m);
           return_P (u ⊕ u', y, w · w', δw · δw', disc ⊕ disc') }
```

Note: In the implementation, the weight returned by `DynamicGF/update` is
computed as `new_score - old_score` (see `dynamic.cljs:85`), which equals
the sum of per-site log density ratios.

**Implementation**: `update-transition` (`handler.cljs:96-124`) and
`DynamicGF/update` (`dynamic.cljs:69-86`).

---

## 4. regenerate{−} — Selected Resample (NEW)

Regenerate takes an existing trace and a selection, resampling the selected
addresses while keeping unselected addresses unchanged.

### 4.1 Type Transformation

```
regenerate{D η}   = (η × ℝ) → Sel → P (η × ℝ × ℝ)
regenerate{G_γ η} = (γ × η × ℝ) → Sel(γ) → P (γ × η × ℝ)
```

### 4.2 Term Transformation

```
regenerate{trace(k, t : D η)}(tr_old, sel) =
  if k ∈ selected(sel) then
    -- Resample: draw new value
    do_P{ v_new ← sample d;
          let w_new = density_d(v_new)
              v_old = tr_old[k]
              w_old = density_d(v_old)
          in return_P ({k : v_new}, v_new, w_new,
                        w_new / w_old) }           -- proposal ratio
  else
    -- Keep old value
    let v = tr_old[k]
        w = density_d(v)
    in return_P ({k : v}, v, w, 1)

regenerate{do_G{x ← t; m}}(tr_old, sel) =
  do_P{ (u, x, w, r)    ← regenerate{t}(π_{grade(t)}(tr_old),
                                          sel ↾ grade(t));
        (u', y, w', r')  ← regenerate{do_G{m}[x ↦ x]}(
                              π_{grade(do_G{m})}(tr_old),
                              sel ↾ grade(do_G{m}));
        return_P (u ⊕ u', y, w · w', r · r') }
```

The final regenerate weight (at DynamicGF level) is:
```
weight = new_score - old_score - proposal_ratio
```
where proposal_ratio = Σ_{a ∈ selected} (new_lp(a) - old_lp(a)).

This ensures the MH acceptance ratio `min(1, exp(weight))` is correct:
accepting the proposal with this weight maintains detailed balance.

**Implementation**: `regenerate-transition` (`handler.cljs:126-153`) and
`DynamicGF/regenerate` (`dynamic.cljs:89-108`).

---

## 5. project{−} — Selected Log-Weight (NEW)

Project computes the total log-probability of selected choices in an
existing trace, without modifying the trace.

### 5.1 Type Transformation

```
project{D η}   = (η × ℝ) → Sel → ℝ
project{G_γ η} = (γ × η × ℝ) → Sel(γ) → ℝ
```

### 5.2 Term Transformation

```
project{trace(k, t : D η)}(tr, sel) =
  let v = tr[k]
      w = density_d(v)
  in if k ∈ selected(sel) then w else 1

project{do_G{x ← t; m}}(tr, sel) =
  let w₁ = project{t}(π_{grade(t)}(tr), sel ↾ grade(t))
      w₂ = project{do_G{m}}(π_{grade(do_G{m})}(tr), sel ↾ grade(do_G{m}))
  in w₁ · w₂
```

In log-space: `project = Σ_{a ∈ selected(sel)} log_prob_d(a)(tr[a])`.

**Implementation**: `project-transition` (`handler.cljs:155-171`) and
`DynamicGF/project` (`dynamic.cljs:133-144`).

---

## 6. propose{−} — Forward Sample + Score (NEW)

Propose is structurally identical to simulate but returns the score as
the importance weight.

### 6.1 Type Transformation

```
propose{D η}   = P (η × ℝ)
propose{G_γ η} = P (γ × η × ℝ)
```

### 6.2 Term Transformation

```
propose{t} = simulate{t}
```

with the weight field set to the score (rather than 0). Propose is
operationally identical to simulate — the difference is in how the
caller uses the result:
- `simulate` returns `(trace, retval, score)` — score used for future weight computations
- `propose` returns `(choices, retval, weight)` — weight = score, used immediately

**Implementation**: `DynamicGF/propose` (`dynamic.cljs:122-131`).

---

## 7. edit{−} — Parametric Edit (NEW)

Edit generalizes update and regenerate into a single parametric operation.
Each edit request type dispatches differently.

### 7.1 Type Transformation

```
edit{G_γ η} = (γ × η × ℝ) → EditReq(γ) → P (γ × η × ℝ × EditReq(γ))
```

The return includes a backward EditReq that reverses the forward edit.

### 7.2 Term Transformation

```
edit{g}(tr, ConstraintEdit(c)) =
  let (tr', weight, disc) = update{g}(tr, c)
  in (tr', weight, ConstraintEdit(disc))           -- backward = discarded values

edit{g}(tr, SelectionEdit(s)) =
  let (tr', weight) = regenerate{g}(tr, s)
  in (tr', weight, SelectionEdit(s))               -- backward = same selection

edit{g}(tr, ProposalEdit(f, f_args, b, b_args)) =
  let (choices_f, w_f)         = propose{f}(f_args ∪ {tr.choices})
      (tr', w_upd, disc)      = update{g}(tr, choices_f)
      (_, w_b)                 = assess{b}(b_args ∪ {tr'.choices}, disc)
      weight                   = w_upd + w_b − w_f
  in (tr', weight, ProposalEdit(b, b_args, f, f_args))
```

The ProposalEdit backward request **swaps** forward and backward GFs,
enabling the MH acceptance ratio to maintain detailed balance.

**Implementation**: `edit-dispatch` (`edit.cljs:64-108`).

---

## 8. Homomorphic Extension

All transformations extend homomorphically to product types, function
types, and record types:

```
op{τ₁ × τ₂}         = op{τ₁} × op{τ₂}
op{τ₁ → τ₂}         = op{τ₁} → op{τ₂}
op{{k₁:τ₁,…,kₙ:τₙ}} = {k₁ : op{τ₁}, …, kₙ : op{τₙ}}
op{η}                 = η                 (ground types unchanged)
```

This ensures that all GFI operations compose naturally through the
type structure.

---

## 9. Summary: All Transformations at a Glance

| Transformation | Input Types | Output Type | Source |
|---------------|-------------|-------------|--------|
| simulate{G_γ η} | args | P (γ × η × ℝ) | λ_GEN Fig.12 |
| assess{G_γ η} | args × γ | η × ℝ | λ_GEN Fig.12 |
| generate{G_γ η} | args × γ_obs | P (γ × η × ℝ) | NEW |
| update{G_γ η} | (γ × η × ℝ) × γ_new | P (γ × η × ℝ × γ_disc) | NEW |
| regenerate{G_γ η} | (γ × η × ℝ) × Sel(γ) | P (γ × η × ℝ) | NEW |
| project{G_γ η} | (γ × η × ℝ) × Sel(γ) | ℝ | NEW |
| propose{G_γ η} | args | P (γ × η × ℝ) | NEW |
| edit{G_γ η} | (γ × η × ℝ) × EditReq(γ) | P (γ × η × ℝ × EditReq(γ)) | NEW |

The six new transformations (generate through edit) constitute the core
novel contribution of λ_MLX beyond λ_GEN. Together with simulate and
assess, they formalize the complete Generative Function Interface.

### 9.1 Extended Transformations

The following transformations operate on the gradient and inference
level, building on the GFI operations above:

| Transformation | Input Types | Output Type | Source |
|---------------|-------------|-------------|--------|
| adev{G_γ η} | args × Key | σ_adev | `proofs/adev.md` |
| adev-surrogate | cost × reinforce-lp × baseline | ℝ (surrogate loss) | `proofs/adev.md` §4 |
| hmc-step{G_γ η} | q × Key × ε × L × M | q' (trace) | `proofs/hmc-nuts.md` §3 |
| elbo{G_γ η} | q-params × Key × K | ℝ (bound) | `proofs/vi.md` §1 |
| smcp3-step | traces × weights × obs | traces × weights | `proofs/smcp3.md` §3 |

These are not GFI operations per se (they do not appear in the GFI
protocol), but they build on GFI operations and have formal correctness
proofs. See the referenced proof files for details.
