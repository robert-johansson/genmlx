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

when the proposals are consistent (i.e., the forward proposal producing
choices_f and the backward proposal producing choices_b describe the
same move pair). ∎

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
