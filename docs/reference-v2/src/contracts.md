# Contracts

The GFI algebraic-theory module provides a data-driven registry of 85 measure-theoretic invariants ("laws") that every GFI-compliant generative function must satisfy. Each law is a named theorem paired with a check function that tests the invariant over random traces.

Use `verify` to run the full law suite against any model. This is the primary tool for validating that new generative functions, combinators, or compiled paths preserve GFI semantics.

```clojure
(require '[genmlx.gfi :as gfi])
```

Source: `src/genmlx/gfi.cljs`

---

## Overview

The `laws` var is a **vector** of law maps. Each law is `{:name keyword, :from string, :theorem string, :tags set, :check fn}`. Each `:check` function takes `{:keys [model args]}` and returns a boolean (it may internally simulate traces as needed). The verifier runs each law over multiple random trials to catch violations that only appear under certain sampled values.

---

## verify

### `verify`

```clojure
(gfi/verify model args & {:keys [law-names tags n-trials]})
```

Run GFI laws over multiple random trials and return a structured report.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | IGenerativeFunction | The generative function to verify |
| `args` | vector | Arguments to the model |
| `law-names` | collection (optional) | Subset of law keywords to run (default: all) |
| `tags` | collection (optional) | Run laws matching ANY of these tag keywords |
| `n-trials` | integer (optional) | Number of independent trials per law (default: 10) |

**Returns:** Map with keys:

| Key | Type | Description |
|-----|------|-------------|
| `:results` | vector | One map per law: `{:name :from :theorem :passes :fails :pass?}` |
| `:total-pass` | integer | Sum of all passes across all laws |
| `:total-fail` | integer | Sum of all failures across all laws |
| `:all-pass?` | boolean | `true` if every trial of every law passed |
| `:n-laws` | integer | Number of laws selected |
| `:n-trials` | integer | Trials per law |

**Example:**
```clojure
(def model
  (gen [x]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y (dist/gaussian mu 1)))))

;; Run all laws, 10 trials each
(def report (gfi/verify model [0.0]))
(println (:all-pass? report))  ;; => true

;; Run a subset of laws
(def report (gfi/verify model [0.0]
              :law-names [:generate-full-weight-equals-score
                          :update-identity]
              :n-trials 100))

;; Run laws by tag
(def report (gfi/verify model [0.0] :tags [:update :regenerate]))
```

A companion `print-report` pretty-prints the result:

```clojure
(gfi/print-report report)
```

---

## check-law

### `check-law`

```clojure
(gfi/check-law law-name model args)
```

Run a single law (one trial) against a model. Returns `{:name :pass? :theorem}`, plus an `:error` string if the check threw. Throws `ex-info` if `law-name` is not a known law.

```clojure
(gfi/check-law :generate-full-weight-equals-score model [0.0])
;; => {:name :generate-full-weight-equals-score :pass? true :theorem "..."}
```

---

## Law List

The laws are grouped by the GFI operation they verify. A representative selection (use `(map :name gfi/laws)` for the full list):

### `:simulate-produces-trace`

**Theorem:** `simulate(P, x)` returns trace `t = (P, x, tau)` where `tau` is in `supp(p(.; x))` and score is finite.

---

### `:simulate-score-is-log-density`

**Theorem:** `trace.score = log p(tau; x) = assess(P, x, tau).weight`.

The trace score is the joint log-density and matches the weight returned by `assess` on the same choices.

---

### `:generate-empty-is-simulate`

**Theorem:** `generate(P, x, {}).weight = 0` (equivalent to simulate).

With empty constraints, `generate` is equivalent to `simulate` and the importance weight is zero.

---

### `:generate-full-weight-equals-score`

**Theorem:** when fully constrained, `generate(model, args, trace.choices).weight = trace.score`.

When `generate` is given a complete set of constraints (all choices pinned), the returned weight equals the trace score. The importance weight under full observation is the joint log-density.

---

### `:assess-equals-generate-score`

**Theorem:** `assess(model, args, choices).weight = generate(model, args, choices).score`.

The weight returned by `assess` matches the score of a trace generated with the same choices. Both compute the joint log-density.

---

### `:update-identity`

**Theorem:** updating a trace with its own choices is a no-op; the weight is zero.

The new and old scores are equal, so the update weight is zero.

---

### `:update-density-ratio`

**Theorem:** `update(model, trace, constraint).weight = new_score - old_score`.

The update weight equals the difference between the new and old trace scores -- the fundamental correctness property of incremental recomputation.

---

### `:update-round-trip`

**Theorem:** `update(trace, c)` then re-applying the discard recovers the original trace values.

Applying an update and then applying the discard as a constraint recovers the original trace, verifying that update and discard are inverses.

---

### `:regenerate-empty-identity`

**Theorem:** `regenerate(model, trace, sel/none).weight = 0`, choices unchanged.

Regenerating with an empty selection is a no-op: the weight is zero and all choices are preserved.

---

### `:regenerate-weight-formula`

**Theorem:** the regenerate weight is the retained-only density ratio `W = Σ_retained [lp(v; new) − lp(v; old)]`.

Selected, fresh, and removed sites cancel to zero; only retained sites contribute to the weight.

---

### `:project-all-equals-score`

**Theorem:** `project(model, trace, sel/all) = trace.score`.

Projecting onto all addresses recovers the full trace score, verifying that the score decomposes correctly over addresses.

---

### `:project-none-equals-zero`

**Theorem:** `project(model, trace, sel/none) = 0`.

Projecting onto no addresses yields zero -- no choices selected means no log-density contribution.

---

### `:score-decomposition`

**Theorem:** the sum of per-address projections equals `trace.score`.

The total trace score equals the sum of individual per-address projections, verifying additive decomposition of the log-density.

---

### `:vsimulate-shape-correctness`

**Theorem:** `vsimulate(model, args, N)` produces a batch of `N` traces with finite scores of shape `[N]`.

Vectorized simulation produces a batch of `N` traces, each with a finite score, verifying that shape-based batching works for the model.

---

## laws

### `laws`

```clojure
gfi/laws
```

The law registry -- a **vector** of `{:name keyword, :from string, :theorem string, :tags set, :check fn}` maps. Can be used directly for custom verification workflows.

**Example:**
```clojure
;; Run a single law's check manually
(let [law  (first (filter #(= :generate-full-weight-equals-score (:name %))
                          gfi/laws))
      check (:check law)]
  (check {:model model :args [0.0]}))
;; => true
```
