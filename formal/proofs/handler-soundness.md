# Handler Soundness — TODO 10.10

> Theorem: each handler mode correctly implements its GFI operation.
> The volatile!/vreset! wrapper is semantically invisible — the state
> threading semantics is identical to the pure state monad H(σ, τ).

---

## 1. Statement

### Theorem (Handler Soundness)

For each mode m ∈ {simulate, generate, update, regenerate, project}, and
for any generative function body with k trace effect operations:

```
⟦run-handler(m-transition, σ₀, body)⟧ = m{body}(σ₀)
```

That is, the result of executing the body under the m-handler with initial
state σ₀ equals the denotation of the m-transformation applied to the body.

Specifically:
- The returned choices equal m{body}'s trace
- The returned score equals m{body}'s score (joint log-density)
- The returned weight (where applicable) equals m{body}'s weight
- The returned discard (where applicable) equals m{body}'s discard

---

## 2. Architecture Recap

The handler system has two layers:

**Layer 1: Pure state transitions** — functions of type
`(σ, addr, dist) → (value, σ')`, defined in `handler.cljs:72-171`.

**Layer 2: Volatile wrapper** — `run-handler` creates a `volatile!`
holding the current state, and handler functions (`simulate-handler`,
`generate-handler`, etc.) read/write this volatile via `@*state*` and
`vreset!`.

```clojure
;; Pure transition (Layer 1)
(defn- simulate-transition [state addr dist]
  (let [[k1 k2] (rng/split (:key state))
        value (dc/dist-sample dist k2)
        lp    (dc/dist-log-prob dist value)]
    [value (-> state ...)]))

;; Volatile wrapper (Layer 2)
(defn simulate-handler [addr dist]
  (let [[value state'] (simulate-transition @*state* addr dist)]
    (vreset! *state* state')
    value))

;; Orchestrator
(defn run-handler [handler-fn init-state body-fn]
  (binding [*handler* handler-fn
            *state*   (volatile! init-state)]
    (let [retval (body-fn)]
      (assoc @*state* :retval retval))))
```

---

## 3. Proof

By induction on the number of trace effect operations n in the body.

### Base Case: n = 0

The body contains no `trace` calls. It is a pure computation that produces
a return value retval without any handler interaction.

```
run-handler(m-transition, σ₀, body)
= binding { *handler* = m-handler, *state* = volatile!(σ₀) }
    let retval = body()
    in assoc(@*state*, :retval, retval)
= { ...σ₀, retval: body() }
```

Since no handler was invoked, the state σ₀ is unchanged. This corresponds
to the m-transformation of `return_G(body())`:

```
m{return_G t}(σ₀) = (t, σ₀)    for all modes m
```

The choices remain empty (EMPTY), score remains 0, weight remains 0
(where applicable). ✓

### Inductive Case: n → n+1

Assume the theorem holds for bodies with n trace operations (induction
hypothesis). Consider a body with n+1 trace operations. We can decompose
it as:

```
body = do_G{ x ← trace(a₁, d₁);        -- first trace operation
             rest }                       -- remaining n operations
```

**Step 1: First trace operation.**

When body() executes and reaches the first `trace(a₁, d₁)` call:

1. `dyn/trace` calls `h/trace-choice!`
2. `trace-choice!` dispatches to `*handler*` = m-handler
3. m-handler executes:
   ```
   let [value, σ₁] = m-transition(@*state*, a₁, d₁)
   vreset!(*state*, σ₁)
   return value
   ```
4. The state transitions from σ₀ to σ₁

By the definition of m-transition (see `semantics.md` §4), σ₁ contains:
- choices updated with a₁ ↦ value
- score updated with density_d₁(value)
- weight updated according to mode m
- Other fields updated as specified

This matches ⟦trace(a₁, d₁)⟧_m(σ₀) from the denotational semantics.

**Step 2: Remaining operations.**

After the first trace operation, the state is σ₁ and the body continues
with `rest` (which has n trace operations). By the induction hypothesis:

```
run-handler(m-transition, σ₁, rest) = m{rest}(σ₁)
```

But `run-handler` was already running with the volatile holding σ₁
(from Step 1's vreset!), so the remaining execution continues in the
same binding scope. The volatile acts as a threading mechanism: each
subsequent trace operation reads σᵢ and writes σᵢ₊₁.

**Step 3: Composition.**

The full execution is:
```
let (x, σ₁) = m-transition(σ₀, a₁, d₁)
let (retval, σ_final) = m{rest[x ↦ x]}(σ₁)
result = (retval, σ_final)
```

This is exactly the denotation of `do_G{x ← trace(a₁, d₁); rest}` under
mode m, which sequences the monadic operations:

```
⟦do_G{x ← trace(a₁, d₁); rest}⟧_m(σ₀)
= let (x, σ₁)        = ⟦trace(a₁, d₁)⟧_m(σ₀)
  let (retval, σ_n+1) = ⟦rest⟧_m(σ₁)
  in (retval, σ_n+1)
```

Therefore `run-handler(m-transition, σ₀, body) = m{body}(σ₀)`. ∎

---

## 4. Volatile! Invisibility

The key property enabling the proof is that the volatile!/vreset! boundary
is **semantically invisible**: it implements sequential state threading
without observable side effects.

### Property: State Threading Equivalence

The volatile-based execution:
```
state_vol = volatile!(σ₀)
x₁ = handler(a₁, d₁)       -- reads state_vol, writes σ₁
x₂ = handler(a₂, d₂)       -- reads σ₁, writes σ₂
...
xₙ = handler(aₙ, dₙ)       -- reads σₙ₋₁, writes σₙ
result = @state_vol          -- reads σₙ
```

is equivalent to the pure state monad:
```
let (x₁, σ₁) = transition(σ₀, a₁, d₁)
let (x₂, σ₂) = transition(σ₁, a₂, d₂)
...
let (xₙ, σₙ) = transition(σₙ₋₁, aₙ, dₙ)
result = σₙ
```

This equivalence holds because:

1. **Isolation:** The volatile is created inside `binding` and never
   escapes the scope (`handler.cljs:457-461`)

2. **Single-threaded access:** Within a single `run-handler` execution,
   the volatile is accessed sequentially — each handler call completes
   before the next begins (ClojureScript is single-threaded)

3. **No aliasing:** The state map values are immutable persistent data
   structures — `vreset!` replaces the entire state atomically

4. **No observation of mutation:** The body function observes handler
   return values (the `value` from transitions), never the volatile
   itself. State changes are only observable through future handler calls.

### Corollary: Handler Purity

Despite using mutation (volatile!), the handler system is **observationally
pure**: for any body function, the result of `run-handler` depends only on:
- The transition function (determines which mode)
- The initial state σ₀ (determines starting conditions)
- The body function (determines which trace operations occur)
- The PRNG key (determines random samples)

No global state is read or written. The dynamic vars `*handler*` and
`*state*` are lexically scoped to the `binding` block.

---

## 5. Splice Soundness

For bodies containing `splice(k, g, args)`, the proof extends by treating
the splice as a macro-step.

### 5.1 Scalar Splice

```
trace-gf!(k, g, args) =
  let sub-result = execute-sub(g, args, scoped-state)
  vswap!(*state*, merge-sub-result, k, sub-result)
  return sub-result.retval
```

By the induction hypothesis applied to the sub-GF's own execution:
`execute-sub` correctly implements the sub-GF's GFI operation. The
`merge-sub-result` function correctly nests the sub-result under address k.

**State scoping correctness.** The scalar splice (`handler.cljs:432-444`)
extracts sub-state fields scoped to address k:

```
sub-constraints = σ.constraints ↾ k
sub-old-choices = σ.old-choices ↾ k
sub-selection   = σ.selection ↾ k
```

The sub-GF executes with these scoped fields via the parent's `:executor`
function, which calls the appropriate GFI operation on g. The sub-result
is then merged back:

```
σ' = σ with {
  choices[k] ← sub.choices,
  score += sub.score,
  weight += sub.weight,
  discard[k] ← sub.discard
}
```

This is correct because:
1. The sub-GF's addresses are relative — nesting under k in the parent
   ensures no address collisions with the parent's other trace sites
2. Score and weight are additive — the sub-GF's contribution adds to
   the parent's running totals
3. The discard is nested under k, preserving the hierarchical structure

### 5.2 Batched Splice

The batched variant (`batched-splice-transition` at `handler.cljs:363-418`)
differs from scalar splice in a critical way: it creates a **nested
`run-handler` scope** with a fresh `volatile!`.

**Theorem (Batched Splice Soundness).** The batched splice transition
correctly implements the sub-GF's GFI operation in batched mode, with
the nested `run-handler` scope isolated from the parent scope.

**Proof.** The batched splice performs five steps:

**Step 1: Key splitting** (`handler.cljs:368`).
```
(k₁, k₂) = split(σ_parent.key)
```
k₂ is passed to the sub-GF; k₁ becomes the parent's new key. This
ensures the sub-GF's randomness is independent of the parent's
subsequent randomness (by the PRNG splitting contract).

**Step 2: Mode detection** (`handler.cljs:377-410`).
The batched splice determines which handler mode to use by examining
which state fields are present:
- If `sub-selection` is present → regenerate mode
- Else if `sub-old-choices` is non-empty → update mode
- Else if `sub-constraints` is non-empty → generate mode
- Else → simulate mode

This mirrors the scalar splice's delegation logic, adapted for the
batched case where we must choose the batched handler variant.

**Step 3: Nested run-handler** (`handler.cljs:412-413`).
```
sub-result = run-handler(batched-m-handler, init-state, #(apply body-fn args))
```

This call creates a **fresh volatile** (`handler.cljs:458`):
```
(binding [*handler* batched-m-handler
          *state*   (volatile! init-state)    ;; ← FRESH volatile
          mx/*batched-exec?* true]
  ...)
```

**Isolation property:** The fresh volatile is completely independent
of the parent's volatile. During the sub-GF's execution:
- The parent's `*state*` volatile is NOT accessible (shadowed by
  the `binding` block's new binding)
- The parent's `*handler*` is NOT accessible (similarly shadowed)
- The sub-GF's handler reads and writes ONLY the fresh volatile

This isolation means the sub-GF's execution cannot observe or corrupt
the parent's state. The sub-GF sees a clean initial state with:
- `choices: EMPTY` (no prior choices)
- `score: 0` (score starts fresh)
- `weight: 0` (weight starts fresh)
- `key: k₂` (independent PRNG stream)
- `batch-size: N` (propagated from parent)
- Mode-specific fields (constraints, old-choices, selection)

**Step 4: Sub-GF execution.**
By the handler soundness theorem (§3) applied to the sub-GF's body
with the batched handler:
```
⟦run-handler(batched-m-transition, init-state, body)⟧ = m{body}^N(init-state)
```

The sub-GF's body executes under the batched handler, producing
[N]-shaped choices, scores, and weights. By the broadcasting
correctness theorem (`broadcasting.md` §4), these represent N
independent executions in SoA format.

**Step 5: Result merging** (`handler.cljs:415-418`).
```
σ_parent' = merge-sub-result(σ_parent{key: k₁}, k, sub-result)
```

The merge function (`handler.cljs:333-361`) combines:
- `choices[k] ← sub.choices` (nest sub-GF's [N]-shaped choices under k)
- `score += sub.score` (add [N]-shaped sub-score to parent's [N]-shaped score)
- `weight += sub.weight` (same for weight)
- `discard[k] ← sub.discard` (nest discard under k)

The parent volatile is updated exactly once (via `vreset!` at
`handler.cljs:427-428`) with the fully-computed merged state. No
intermediate parent states are visible to any other code.

**Volatile nesting diagram:**

```
Parent run-handler scope:
  *state* → volatile!(σ_parent)    ← parent handler reads/writes this
  │
  ├── trace(a₁, d₁)  → updates σ_parent
  ├── trace(a₂, d₂)  → updates σ_parent
  │
  ├── splice(k, g, args):
  │   │
  │   └── Child run-handler scope:
  │       *state* → volatile!(σ_child)   ← child handler reads/writes this
  │       │                                 (parent volatile is SHADOWED)
  │       ├── trace(b₁, e₁) → updates σ_child
  │       ├── trace(b₂, e₂) → updates σ_child
  │       └── returns sub-result
  │
  │   σ_parent ← merge(σ_parent, k, sub-result)  ← single atomic update
  │
  ├── trace(a₃, d₃)  → updates σ_parent (with merged sub-result)
  └── returns final result
```

The nesting can be arbitrarily deep (sub-GFs can themselves contain
splices), and isolation is maintained at each level by the fresh
`binding` + `volatile!` in each `run-handler` call. ∎

---

## 6. Implementation Correspondence

| Formal Concept | Implementation | Location |
|----------------|---------------|----------|
| run-handler(m, σ₀, body) | `run-handler` | `handler.cljs:454-461` |
| H(σ, τ) state monad | `volatile!` + `vreset!` | `handler.cljs:264-326` |
| Monadic bind (do_H) | Sequential handler calls in body | implicit |
| Monadic return (return_H) | Body return value | `handler.cljs:461` |
| State isolation | `binding` block scope | `handler.cljs:457-459` |
| Sub-GF delegation | `trace-gf!` / `execute-sub` | `handler.cljs:420-448` |
| Mode dispatch | `*handler*` dynamic var | `handler.cljs:62` |

Each handler function (`simulate-handler`, etc.) at `handler.cljs:264-325`
is a thin wrapper that reads the volatile, calls the pure transition, and
writes the result back. The pure transitions at `handler.cljs:72-171`
contain all the semantics.
