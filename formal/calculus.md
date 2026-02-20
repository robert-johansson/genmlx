# λ_MLX Calculus — TODO 10.4

> Full type grammar, term grammar, and typing rules for λ_MLX, extending
> λ_GEN (Figure 10, POPL 2026) with handler state types, edit requests,
> diff types, and additional term constructors.

---

## 1. Type Grammar

### 1.1 Base Types (same as λ_GEN)

```
Base types       B  ::=  𝔹 | ℝ | ℝ_{>0}
Batched types    T  ::=  B | T[n]
Ground types     η  ::=  1 | T | η₁ × η₂ | {k₁ : η₁, …, kₙ : ηₙ}
Trace types      γ  ::=  {k₁ : η₁, …, kₙ : ηₙ}
```

Ground types include the unit type 1 (empty record {}), batched types T,
products η₁ × η₂, and string-keyed record types. Trace types γ are record
types that grade generative functions, equipped with a monoid structure ⊕
for concatenation (with {} as identity).

### 1.2 Full Type System

```
Types  τ  ::=  η                          Ground types
            |  τ₁ → τ₂                    Function types
            |  τ₁ × τ₂                    Product types
            |  D η                         Density-carrying distributions
            |  P η                         Stochastic computations
            |  G_γ η                       Generative functions (graded by γ)
            |  H(σ, τ)                     Handler computations (NEW)
            |  EditReq(γ)                  Edit request types (NEW)
            |  Δ                           Diff types (NEW)
            |  Sel(γ)                      Selection types (NEW)
```

### 1.3 Handler State Types (NEW)

The handler computation type H(σ, τ) is unique to λ_MLX. It formalizes
GenMLX's handler state machine as a state-passing monad. Each pure handler
transition has type:

```
transition : Addr × D η → H(σ, η)
```

which unfolds to:

```
transition : Addr × D η → σ → η × σ
```

Handler state schemas, with exact correspondence to `handler.cljs`:

```
σ_sim  = { key    : Key,         -- PRNG key (split at each trace site)
            choices : γ,           -- accumulated choice map
            score   : ℝ }         -- accumulated log-density

σ_gen  = { key         : Key,
            choices     : γ,
            score       : ℝ,
            weight      : ℝ,      -- importance weight (constrained sites)
            constraints : γ_obs }  -- observed values

σ_upd  = { key         : Key,
            choices     : γ,
            score       : ℝ,
            weight      : ℝ,      -- score difference at changed sites
            constraints : γ_new,   -- new constraint values
            old-choices : γ,       -- previous trace choices
            discard     : γ_disc } -- displaced old values

σ_reg  = { key         : Key,
            choices     : γ,
            score       : ℝ,
            weight      : ℝ,      -- proposal ratio (new_lp - old_lp)
            old-choices : γ,       -- previous trace choices
            selection   : Sel(γ) } -- addresses to resample

σ_proj = { key         : Key,
            choices     : γ,
            score       : ℝ,
            weight      : ℝ,      -- accumulated selected log-prob
            old-choices : γ,       -- previous trace choices
            selection   : Sel(γ),  -- addresses to project
            constraints : γ }      -- (unused, present for splice compat)
```

**Batched variants** add two fields:

```
σ_batch(σ_m) = σ_m ∪ { batch-size : ℕ, batched? : 𝔹 }
```

All other keys and semantics are identical — MLX broadcasting handles
the shape difference between scalar and [N]-shaped values.

### 1.4 Edit Request Types (NEW)

```
EditReq(γ)  ::=  ConstraintEdit(γ_new)                       -- like update
              |  SelectionEdit(Sel(γ))                        -- like regenerate
              |  ProposalEdit(G_{γ'} η', Args,                -- SMCP3-style
                              G_{γ''} η'', Args)
```

### 1.5 Diff Types (NEW)

```
Δ  ::=  NoChange                         -- value unchanged
     |  UnknownChange                    -- conservatively assume changed
     |  ValueChange(η, η)               -- old and new values
     |  VectorDiff(𝒫(ℕ))               -- set of changed indices
     |  MapDiff(𝒫(K), 𝒫(K), 𝒫(K))     -- changed, added, removed keys
```

### 1.6 Selection Types

```
Sel(γ)  ::=  AllSelection                -- select all addresses
          |  NoneSelection               -- select no addresses
          |  KeySelection({k₁, …, kₙ})  -- select specific addresses
          |  ComplementSelection(Sel(γ)) -- select everything not in sel
          |  HierarchicalSelection({k₁ : Sel(γ₁), …}) -- nested
```

---

## 2. Term Grammar

### 2.1 λ_GEN Terms (inherited)

```
Terms  t  ::=  ()                              unit
            |  c                               constants (c ∈ T)
            |  p                               primitives
            |  x                               variables
            |  (t₁, t₂)                        pairs
            |  π_i t                            projections (i ∈ {1, 2})
            |  t[k]                             record field access
            |  {k₁ : t₁, …, kₙ : tₙ}          record literals
            |  λx.t                             abstraction
            |  t₁ t₂                            application
            |  let x = t₁ in t₂                let binding
            |  select(t₁, t₂, t₃)              conditional selection
            |  trace(k, t)                      traced random choice
            |  return_G t                       embed deterministic into G
            |  return_P t                       embed deterministic into P
            |  do_G{x ← t; m}                  sequence in G monad
            |  do_P{x ← t; m}                  sequence in P monad
            |  sample t                          sample from distribution

Monadic  m  ::=  t | x ← t; m

Primitives p  ::=  cos | sin | exp | log | add | mul | subtract
                 | dot | sum | stack | reshape | where
                 | fold | scan | reduce
                 | gaussian | uniform | bernoulli | …  (27 distributions)
```

### 2.2 λ_MLX Additional Terms (NEW)

```
Additional terms:
  t  ::=  …                                           all λ_GEN terms
       |  splice(k, g, args)                           call sub-GF at address k
       |  param(k, default)                            read trainable parameter
       |  fix(f)                                       fixed point (recursion)

Edit/update terms:
       |  update(tr, constraints)                      modify trace
       |  regenerate(tr, sel)                          resample selected
       |  edit(tr, req)                                parametric edit
       |  project(tr, sel)                             selected log-weight
       |  propose(args)                                forward sample + score

Handler terms (internal — not exposed in DSL):
       |  run-handler(transition, σ₀, body)            execute under handler
       |  return_H t                                   embed into H monad
       |  do_H{x ← t; m}                              sequence in H monad
```

---

## 3. Typing Rules

### 3.1 Inherited from λ_GEN (Figure 10)

```
  Γ ⊢ t : η
  ─────────────────
  Γ ⊢ return_P t : P η

  Γ ⊢ t : D η
  ─────────────────
  Γ ⊢ sample t : P η

  Γ ⊢ t : P η   Γ, x : η ⊢ do_P{m} : P η'
  ─────────────────────────────────────────────
  Γ ⊢ do_P{x ← t; m} : P η'

  Γ ⊢ t : η
  ─────────────────────
  Γ ⊢ return_G t : G_{} η

  Γ ⊢ t : G_γ η   Γ, x : η ⊢ do_G{m} : G_{γ'} η'   keys(γ) ∩ keys(γ') = ∅
  ─────────────────────────────────────────────────────────────────────────────
  Γ ⊢ do_G{x ← t; m} : G_{γ ⊕ γ'} η'

  k ∈ Str   Γ ⊢ t : D η
  ─────────────────────────
  Γ ⊢ trace(k, t) : G_{k↦η} η

  k ∈ Str   Γ ⊢ t : G_γ η
  ──────────────────────────
  Γ ⊢ trace(k, t) : G_{k↦γ} η
```

### 3.2 Handler Computation Rules (NEW)

```
  Γ ⊢ t : τ
  ──────────────────────
  Γ ⊢ return_H t : H(σ, τ)

  Γ ⊢ t : H(σ, τ₁)   Γ, x : τ₁ ⊢ do_H{m} : H(σ, τ₂)
  ────────────────────────────────────────────────────────
  Γ ⊢ do_H{x ← t; m} : H(σ, τ₂)

  Γ ⊢ transition : Addr × D η → H(σ, η)
  Γ ⊢ σ₀ : σ     Γ ⊢ body : η'    (body may contain trace effects)
  ──────────────────────────────────────────────────────────────────
  Γ ⊢ run-handler(transition, σ₀, body) : σ ⊗ {retval : η'}
```

The `run-handler` rule captures the operational semantics: it executes
`body` intercepting every `trace(k, d)` call via `transition`, threading
state σ through all intercepts.

### 3.3 Splice Rule (NEW)

```
  k ∈ Str   Γ ⊢ g : G_{γ'} η'   Γ ⊢ args : τ_args
  ─────────────────────────────────────────────────────
  Γ ⊢ splice(k, g, args) : G_{k↦γ'} η'
```

Splice calls a sub-generative function, nesting its trace under address k.
The sub-GF's trace type γ' is nested under k in the parent's grading.

### 3.4 Update Rule (NEW)

```
  Γ ⊢ g : G_γ η   Γ ⊢ tr : γ × η × ℝ   Γ ⊢ c : γ_new ⊆ γ
  ──────────────────────────────────────────────────────────────
  Γ ⊢ update(tr, c) : P (γ × η × ℝ × γ_disc)
```

where γ_new ⊆ γ means dom(γ_new) ⊆ dom(γ), and γ_disc records the
displaced old values at addresses in dom(γ_new) ∩ dom(γ).

### 3.5 Regenerate Rule (NEW)

```
  Γ ⊢ g : G_γ η   Γ ⊢ tr : γ × η × ℝ   Γ ⊢ s : Sel(γ)
  ──────────────────────────────────────────────────────────
  Γ ⊢ regenerate(tr, s) : P (γ × η × ℝ)
```

### 3.6 Edit Rule (NEW)

```
  Γ ⊢ g : G_γ η   Γ ⊢ tr : γ × η × ℝ   Γ ⊢ req : EditReq(γ)
  ──────────────────────────────────────────────────────────────────
  Γ ⊢ edit(tr, req) : P (γ × η × ℝ × EditReq(γ))
```

The return includes a backward EditReq that reverses the forward edit.

### 3.7 Project Rule (NEW)

```
  Γ ⊢ g : G_γ η   Γ ⊢ tr : γ × η × ℝ   Γ ⊢ s : Sel(γ)
  ──────────────────────────────────────────────────────────
  Γ ⊢ project(tr, s) : ℝ
```

### 3.8 Propose Rule (NEW)

```
  Γ ⊢ g : G_γ η   Γ ⊢ args : τ_args
  ─────────────────────────────────────
  Γ ⊢ propose(args) : P (γ × η × ℝ)
```

Propose is structurally identical to simulate but returns the score as
the weight (importance weight = joint log-probability of all choices).

### 3.9 Fixed Point Rule (NEW)

```
  Γ ⊢ f : (G_γ η → G_γ η)
  ──────────────────────────
  Γ ⊢ fix(f) : G_γ η
```

Corresponds to the `RecurseCombinator` where `maker : self → GF`:

```
  Γ ⊢ maker : G_γ η → G_γ η
  ────────────────────────────
  Γ ⊢ Recurse(maker) : G_γ η
```

### 3.10 Trainable Parameter Rule (NEW)

```
  k ∈ Str   Γ ⊢ default : T
  ─────────────────────────────
  Γ ⊢ param(k, default) : T
```

The `param` effect reads from the active parameter store if one is bound,
otherwise returns the default value. Unlike `trace`, param does not
contribute to the trace type γ or the score.

---

## 4. Implementation Correspondence

| Formal Type | Implementation | File |
|-------------|---------------|------|
| G_γ η | `DynamicGF` record | `dynamic.cljs` |
| D η | `Distribution` record | `dist/core.cljs` |
| γ | `ChoiceMap` (Node/Value) | `choicemap.cljs` |
| γ × η × ℝ | `Trace` record | `trace.cljs` |
| H(σ, τ) | Pure transition fn + `volatile!` | `handler.cljs` |
| σ_sim | `{:key :choices :score}` map | `handler.cljs:72-81` |
| σ_gen | `{:key :choices :score :weight :constraints}` map | `handler.cljs:83-94` |
| σ_upd | `{:key :choices :score :weight :constraints :old-choices :discard}` map | `handler.cljs:96-124` |
| σ_reg | `{:key :choices :score :weight :old-choices :selection}` map | `handler.cljs:126-153` |
| σ_proj | `{:key :choices :score :weight :old-choices :selection :constraints}` map | `handler.cljs:155-171` |
| Sel(γ) | Selection algebra | `selection.cljs` |
| EditReq(γ) | `ConstraintEdit`/`SelectionEdit`/`ProposalEdit` records | `edit.cljs` |
| Δ | Diff maps `{:diff-type ...}` | `diff.cljs` |
| P η | Implicit (handler execution produces values) | — |
| trace(k, d) | `(dyn/trace :k (dist/gaussian ...))` | `dynamic.cljs:216-219` |
| splice(k, g, args) | `(dyn/splice :k model args)` | `dynamic.cljs:221-225` |
| param(k, v) | `(dyn/param :k v)` | `dynamic.cljs:333-339` |
| fix(maker) | `(combinators/recurse maker)` | `combinators.cljs:445-522` |
| run-handler | `(h/run-handler handler-fn init-state body-fn)` | `handler.cljs:454-461` |

| Formal Combinator | Implementation | Type Signature |
|-------------------|---------------|----------------|
| Map(g) | `MapCombinator` | `[η₁] → G_{[i:γ]} [η]` |
| Unfold(g) | `UnfoldCombinator` | `(ℕ × η₁ × …) → G_{[t:γ]} [η]` |
| Switch(g₁,…,gₙ) | `SwitchCombinator` | `ℤₙ → G_{γᵢ} η` |
| Scan(g) | `ScanCombinator` | `(C × [η₁]) → G_{[t:γ]} (C × [O])` |
| Mask(g) | `MaskCombinator` | `𝔹 → G_{γ\|{}} η` |
| Mix(ws, g₁,…,gₙ) | `MixCombinator` | `G_{{:idx}⊕γᵢ} η` |
| Contramap(f, g) | `ContramapGF` | `G_γ η` (args transformed) |
| Dimap(f, h, g) | `ContramapGF ∘ MapRetvalGF` | `G_γ η'` |
| Recurse(maker) | `RecurseCombinator` | `G_γ η` (fixed point) |
