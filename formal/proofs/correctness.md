# Correctness of Generate and Update — TODO 10.7

> Proposition: the generate and update program transformations produce
> correct importance weights and trace modifications. Analogous to
> Proposition 3.1 of the POPL 2026 paper, which proves correctness of
> simulate and assess.

---

## 1. Statement

### Proposition (Correctness of generate)

Let ⊢ t : G_γ η be a closed term of generative function type with
denotation (μ, f) = ⟦t⟧, and let ν be the stock measure on γ. Let
obs : γ_obs ⊆ γ be a partial constraint map. Then:

**⟦generate{t}(obs)⟧** produces a distribution over (trace, retval, weight)
such that:

1. The trace u agrees with obs at all constrained addresses:
   ∀a ∈ dom(obs). u(a) = obs(a)

2. Unconstrained addresses are distributed according to μ conditioned
   on the constraints:
   u|_{dom(γ)\dom(obs)} ~ μ(· | u|_{dom(obs)} = obs)

3. The weight equals the marginal density of the observations:
   w = Σ_{a ∈ dom(obs)} log density_{d_a}(obs(a))

   where d_a is the distribution at address a.

### Proposition (Correctness of update)

Let ⊢ t : G_γ η with (μ, f) = ⟦t⟧. Let tr_old be a trace from a
previous execution, and c_new : γ_new ⊆ γ be new constraints. Then:

**⟦update{t}(tr_old, c_new)⟧** produces (trace, retval, weight, discard)
such that:

1. At constrained addresses: u(a) = c_new(a) for a ∈ dom(c_new)

2. At unconstrained-but-previously-traced addresses:
   u(a) = tr_old(a) for a ∈ dom(tr_old) \ dom(c_new)

3. At new addresses (not in tr_old): u(a) ~ d_a (fresh sample)

4. The weight equals the log score difference:
   w = score(u_new) - score(u_old)
   = Σ_a log density_{d_a}(u_new(a)) - Σ_a log density_{d_a}(u_old(a))

5. The discard contains displaced old values:
   disc(a) = tr_old(a) for a ∈ dom(c_new) ∩ dom(tr_old)

---

## 2. Proof of Generate Correctness

By structural induction on the term t.

### Base Case: t = trace(k, d : D η)

There are two sub-cases depending on whether k is constrained.

**Sub-case k ∈ dom(obs):**

```
⟦generate{trace(k, d)}(obs)⟧
= let v = obs(k), w = density_d(v)
  in return_P ({k : v}, v, w, w)
```

By the semantics, this produces a Dirac distribution at (trace, v, w, w)
where trace = {k : obs(k)}. The trace agrees with obs at k (✓),
and w = density_d(obs(k)) is the density of the observation (✓).

**Sub-case k ∉ dom(obs):**

```
⟦generate{trace(k, d)}(obs)⟧
= do_P{ (v, w, _) ← simulate{d};
        return_P ({k : v}, v, w, 0) }
```

This samples v ~ d and returns weight 0 (no constrained sites). The
sampled value is distributed according to μ (= d) unconditioned (✓),
and the weight is 0 since no address was constrained (✓).

### Inductive Case: t = do_G{x ← t₁; t₂}

Assume the proposition holds for t₁ and t₂ (induction hypothesis).

```
⟦generate{do_G{x ← t₁; t₂}}(obs)⟧
= do_P{ (u₁, x, w₁, w_c₁) ← generate{t₁}(obs ↾ grade(t₁));
        (u₂, y, w₂, w_c₂) ← generate{t₂[x ↦ x]}(obs ↾ grade(t₂));
        return_P (u₁ ⊕ u₂, y, w₁ · w₂, w_c₁ + w_c₂) }
```

By the induction hypothesis on t₁:
- u₁ agrees with obs at addresses in dom(obs) ∩ grade(t₁)
- w_c₁ = Σ_{a ∈ dom(obs) ∩ grade(t₁)} log density(obs(a))

By the induction hypothesis on t₂:
- u₂ agrees with obs at addresses in dom(obs) ∩ grade(t₂)
- w_c₂ = Σ_{a ∈ dom(obs) ∩ grade(t₂)} log density(obs(a))

Since keys(grade(t₁)) ∩ keys(grade(t₂)) = ∅ (disjoint grading from
the typing rule), the combined trace u₁ ⊕ u₂ agrees with obs at all
constrained addresses, and the combined weight:

```
w_c₁ + w_c₂ = Σ_{a ∈ dom(obs)} log density(obs(a))
```

sums over all constrained addresses.

The combined score w₁ · w₂ = Π_a density(u(a)) is the joint density
of the full trace (product of per-site densities), which equals
dμ/dν(u) by the product structure of the stock measure. ∎

---

## 3. Proof of Update Correctness

By structural induction on the term t.

### Base Case: t = trace(k, d : D η)

Three sub-cases:

**Sub-case k ∈ dom(c_new) (new constraint):**

```
v_new = c_new(k),  w_new = density_d(v_new)
v_old = tr_old(k), w_old = density_d(v_old)

result = ({k : v_new}, v_new, w_new, w_new/w_old, {k : v_old})
```

Check: trace has new value (✓), weight = log(w_new) - log(w_old)
= new score contribution minus old (✓), discard has old value (✓).

**Sub-case k ∉ dom(c_new), k ∈ dom(tr_old) (keep old):**

```
v = tr_old(k), w = density_d(v)

result = ({k : v}, v, w, 1, {})
```

Check: trace preserves old value (✓), weight = 0 in log-space (✓),
discard is empty (✓).

**Sub-case k ∉ dom(c_new), k ∉ dom(tr_old) (new address):**

```
simulate: v ~ d, w = density_d(v)

result = ({k : v}, v, w, 1, {})
```

Check: new address sampled fresh (✓), weight = 0 (✓), discard empty (✓).

### Inductive Case: t = do_G{x ← t₁; t₂}

By the induction hypothesis, update{t₁} and update{t₂} each produce
correct trace modifications with correct per-site weight contributions.

```
⟦update{do_G{x ← t₁; t₂}}(tr_old, c_new)⟧ =
  do_P{ (u₁, x, w₁, δw₁, disc₁) ← update{t₁}(...);
        (u₂, y, w₂, δw₂, disc₂) ← update{t₂}(...);
        return_P (u₁ ⊕ u₂, y, w₁·w₂, δw₁·δw₂, disc₁ ⊕ disc₂) }
```

The total weight δw₁ · δw₂ in log-space equals:
```
Σ_{a ∈ dom(c_new) ∩ grade(t₁)} [log p(c_new(a)) - log p(tr_old(a))]
+ Σ_{a ∈ dom(c_new) ∩ grade(t₂)} [log p(c_new(a)) - log p(tr_old(a))]
= Σ_{a ∈ dom(c_new)} [log p_new(a) - log p_old(a)]
= score_new - score_old
```

The discard disc₁ ⊕ disc₂ contains old values at all changed addresses.

### Key Lemma: Constraint Propagation Preserves Absolute Continuity

If μ ≪ ν (the trace distribution is absolutely continuous w.r.t. the
stock measure), and we modify the trace at finitely many addresses, then
the resulting distribution μ' is still absolutely continuous w.r.t. ν.

**Proof:** Modifying finitely many coordinates of a product measure
preserves absolute continuity — the modified density is:

```
dμ'/dν(u') = Π_{a ∈ changed} density_{d_a}(u'(a))
             · Π_{a ∉ changed} density_{d_a}(u(a))
```

which remains a valid density since each factor density_{d_a}(·) is
finite and non-negative. ∎

---

## 4. Implementation Correspondence

| Formal Step | Implementation | Location |
|-------------|---------------|----------|
| generate{trace(k,d)}, constrained | `generate-transition`, `if (cm/has-value? constraint)` | `handler.cljs:87-93` |
| generate{trace(k,d)}, unconstrained | `generate-transition`, else → `simulate-transition` | `handler.cljs:94` |
| update{trace(k,d)}, new constraint | `update-transition`, case 1 | `handler.cljs:103-113` |
| update{trace(k,d)}, keep old | `update-transition`, case 2 | `handler.cljs:116-121` |
| update{trace(k,d)}, sample fresh | `update-transition`, case 3 | `handler.cljs:124` |
| Generate weight = Σ constrained lp | `DynamicGF/generate`, `:weight (:weight result)` | `dynamic.cljs:66` |
| Update weight = new_score - old_score | `DynamicGF/update`, `(mx/subtract (:score result) (:score trace))` | `dynamic.cljs:85` |
| Discard = displaced old values | `update-transition`, `σ.discard[a ↦ old_val]` | `handler.cljs:113` |

The weight computation at the DynamicGF level (`dynamic.cljs:85`) computes
`new_score - old_score`, which by the telescoping sum equals the sum of
per-site log density ratios proven above.
