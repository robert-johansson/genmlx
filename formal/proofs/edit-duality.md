# Edit/Backward Duality — TODO 10.12

> Theorem: the backward EditRequest correctly inverts the forward edit,
> making the Metropolis-Hastings acceptance ratio valid. The forward-backward
> swap in ProposalEdit ensures detailed balance.

---

## 1. Background

The edit interface (`edit.cljs`) generalizes trace modification into three
parametric cases, each returning a **backward request** that reverses the
forward operation:

```
edit(gf, trace, request) → { trace', weight, discard, backward-request }
```

The backward request is critical for:
- MH kernels: the acceptance ratio depends on the reversibility
- SMCP3: sequential proposals use forward-backward pairs
- Involutive MCMC: the backward move must exactly reverse the forward move

---

## 2. Duality Statements

### 2.1 ConstraintEdit Duality

**Claim:** For a ConstraintEdit with constraints c:

```
forward:  edit(g, tr, ConstraintEdit(c))
          → { trace' = update(tr, c).trace,
              weight = update(tr, c).weight,
              backward = ConstraintEdit(discard) }

backward: edit(g, tr', ConstraintEdit(discard))
          → { trace'' = tr   (original trace recovered),
              weight' = -weight,
              backward' = ConstraintEdit(c) }
```

**Proof:**
The forward edit applies constraints c, producing trace' and discard.
At each address a ∈ dom(c):
- trace'(a) = c(a) (new value)
- discard(a) = tr(a) (old value)

The backward edit applies ConstraintEdit(discard) to trace':
At each address a ∈ dom(discard):
- trace''(a) = discard(a) = tr(a) (original value restored)
- discard'(a) = trace'(a) = c(a)

So trace'' = tr and backward' = ConstraintEdit(c). ✓

The backward weight:
```
weight' = score(trace'') - score(trace')
        = score(tr) - score(tr')
        = -(score(tr') - score(tr))
        = -weight
```
✓

**Implementation:** `edit-dispatch`, ConstraintEdit case (`edit.cljs:68-72`).

### 2.2 SelectionEdit Duality

**Claim:** For a SelectionEdit with selection s:

```
forward:  edit(g, tr, SelectionEdit(s))
          → { trace' = regenerate(tr, s).trace,
              weight,
              backward = SelectionEdit(s) }
```

The backward request is the **same selection** — regenerate is its own
inverse in the following sense:

```
backward: edit(g, tr', SelectionEdit(s))
          → { trace'' (fresh resample of selected addresses),
              weight' }
```

**Note:** Unlike ConstraintEdit, applying the backward request does NOT
recover the original trace — it produces a fresh resample. This is
correct for MH because the regenerate weight already accounts for the
proposal distribution (see `dynamic.cljs:89-107`):

```
weight = new_score - old_score - proposal_ratio
```

The proposal_ratio term makes the acceptance ratio:
```
α = min(1, exp(weight)) = min(1, p(tr')/p(tr) · q(tr|tr')/q(tr'|tr))
```
satisfy detailed balance, regardless of whether the backward move
recovers the original trace.

**Implementation:** `edit-dispatch`, SelectionEdit case (`edit.cljs:74-81`).

### 2.3 ProposalEdit Duality (Core Result)

**Claim (Detailed Balance):** For a ProposalEdit with forward GF f and
backward GF b:

```
forward:  edit(g, tr, ProposalEdit(f, f_args, b, b_args))
          → { trace', weight, backward = ProposalEdit(b, b_args, f, f_args) }
```

The backward request **swaps** the forward and backward GFs. This ensures:

```
weight_forward(tr → tr') + weight_backward(tr' → tr) = 0
```

in expectation, which is the detailed balance condition for MH.

---

## 3. Proof of ProposalEdit Detailed Balance

### Forward Direction

```
edit(g, tr, ProposalEdit(f, f_args, b, b_args)):

1. (choices_f, w_f) = propose(f, f_args ∪ {tr.choices})
   -- Sample proposed choices from f, score = w_f

2. (tr', w_upd, disc) = update(g, tr, choices_f)
   -- Apply proposed choices to model

3. (_, w_b) = assess(b, b_args ∪ {tr'.choices}, disc)
   -- Score the discard under backward proposal

4. weight = w_upd + w_b - w_f
```

### Backward Direction

```
edit(g, tr', ProposalEdit(b, b_args, f, f_args)):

1. (choices_b, w_b') = propose(b, b_args ∪ {tr'.choices})
   -- Sample proposed choices from b

2. (tr'', w_upd', disc') = update(g, tr', choices_b)
   -- Apply backward-proposed choices to model

3. (_, w_f') = assess(f, f_args ∪ {tr''.choices}, disc')
   -- Score the discard under forward proposal

4. weight' = w_upd' + w_f' - w_b'
```

### Detailed Balance Argument

The MH acceptance ratio for the forward move tr → tr' is:

```
α = min(1, exp(weight_forward))
  = min(1, exp(w_upd + w_b - w_f))
```

Expanding each term:

```
w_upd = score(tr') - score(tr)        -- update weight (from correctness.md)
w_f   = score_f(choices_f)            -- forward proposal log-probability
w_b   = score_b(disc)                 -- backward scoring of discard
```

So:
```
weight = [score(tr') - score(tr)] + score_b(disc) - score_f(choices_f)
       = log[p(tr')/p(tr)] + log[q_b(disc)/q_f(choices_f)]
       = log[p(tr') · q_b(disc)] - log[p(tr) · q_f(choices_f)]
```

This is exactly the log of the MH ratio:
```
p(tr') · q_b(disc | tr')
─────────────────────────
p(tr)  · q_f(choices_f | tr)
```

For the backward move tr' → tr:
```
weight' = log[p(tr) · q_f(disc')] - log[p(tr') · q_b(choices_b)]
```

The ratio for the backward move is the reciprocal of the forward ratio,
confirming detailed balance:

```
exp(weight) · exp(weight') = 1
```

when the proposals satisfy the **consistency condition** defined below. ∎

### Proposal Consistency Condition

The detailed balance argument above requires that the forward and
backward proposals are **consistent**: they must describe the same
bijection between old and new trace spaces, viewed from opposite
directions.

**Definition (Consistent proposal pair).** A forward GF f and backward
GF b form a consistent proposal pair if:

**(P1) Support coverage.** The support of f (given the old trace)
must cover all addresses that the model update will modify:

```
dom(choices_f) ⊇ dom(constraints applied to model)
```

That is, f must propose values for every address that will change in
the model trace. If f proposes fewer addresses than the model expects,
the update will sample the missing addresses from the prior, introducing
randomness not accounted for in the weight calculation.

**(P2) Backward support matching.** The backward GF b must be able to
score any discard that the forward move produces:

```
dom(assess(b, …, disc)) = dom(disc)
```

That is, b's support must include the old values at all addresses
displaced by the forward move. If b cannot score some discarded values
(they fall outside b's support), the backward weight w_b is -∞ and the
MH acceptance probability is 0 — the move is always rejected. This is
correct but wasteful.

**(P3) Bijection condition (for involutive MCMC).** For the stronger
property that the forward-backward pair defines an involution on the
extended state space (trace, auxiliary), f and b must define inverse
bijections:

```
If f(old_trace) → (choices_f, aux_f)
and update(model, old_trace, choices_f) → (new_trace, disc)
then b(new_trace) deterministically produces disc
and update(model, new_trace, disc) → (old_trace, choices_f)
```

This is the condition under which
`exp(weight_forward) · exp(weight_backward) = 1` holds **exactly**
(not just in expectation).

**When consistency holds automatically:**

1. **ConstraintEdit:** f and b are both identity-like (they just swap
   values). Consistency is automatic — the discard is always the set
   of displaced old values, and applying the discard as constraints
   recovers the old trace.

2. **SelectionEdit:** f = b = same selection. Consistency holds in the
   sense that the regenerate weight already includes the proposal
   correction term (proposal_ratio). No bijection is needed.

3. **ProposalEdit with deterministic bijection:** When f and b implement
   a deterministic invertible transformation (e.g., a reversible jump),
   P1-P3 all hold.

**When consistency can fail:**

1. **Mismatched supports:** If f proposes at addresses {a, b, c} but
   the model only has addresses {a, b}, the extra proposal at c is
   ignored, and the backward assessment of b over {a, b} will miss c.
   The weight calculation is incorrect.

2. **Stochastic proposals with wrong backward:** If f samples auxiliary
   randomness (e.g., a random perturbation) and b does not correctly
   account for the inverse perturbation, the MH ratio is biased.

3. **Support gaps:** If b assigns zero density to some discard values
   that f can produce, the chain cannot traverse those states in the
   backward direction, breaking ergodicity (though each individual
   step is still valid — it just always rejects).

**Practical guidance:** Most GenMLX users will use ConstraintEdit or
SelectionEdit, which are automatically consistent. ProposalEdit is
for advanced users implementing custom MCMC kernels (involutive MCMC,
SMCP3), who must verify P1-P3 for their specific proposal pair.

---

## 4. Connection to Involutive MCMC

The ProposalEdit mechanism generalizes the involutive MCMC framework
(Neklyudov et al. 2020, Cusumano-Towner et al. 2019). In involutive MCMC:

- An **involution** T maps (trace, auxiliary) → (trace', auxiliary')
- Detailed balance holds when T is its own inverse: T(T(x)) = x

The ProposalEdit achieves this by the forward-backward swap:
```
forward:  ProposalEdit(f, f_args, b, b_args)
backward: ProposalEdit(b, b_args, f, f_args)
```

Applying the backward request to the result of the forward request
produces a new ProposalEdit that is again ProposalEdit(f, f_args, b, b_args)
— the original request. This two-step cycle is the discrete analog of
an involution.

For the special case f = b (self-inverse proposals), the ProposalEdit
reduces to a standard involutive MCMC kernel.

---

## 5. Implementation Correspondence

| Formal Step | Implementation | Location |
|-------------|---------------|----------|
| Forward propose | `(p/propose forward-gf fwd-args)` | `edit.cljs:86-89` |
| Forward update | `(p/update gf trace fwd-choices)` | `edit.cljs:91-93` |
| Backward assess | `(p/assess backward-gf bwd-args disc)` | `edit.cljs:95-98` |
| Weight computation | `(mx/add update-weight (mx/subtract bwd-score fwd-score))` | `edit.cljs:100` |
| Backward swap | `(->ProposalEdit backward-gf backward-args forward-gf forward-args)` | `edit.cljs:104-105` |
| ConstraintEdit backward | `(->ConstraintEdit discard)` | `edit.cljs:72` |
| SelectionEdit backward | `(->SelectionEdit (:selection edit-request))` | `edit.cljs:81` |
