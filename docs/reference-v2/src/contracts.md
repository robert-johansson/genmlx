# Contracts

The contracts module provides a data-driven registry of 11 measure-theoretic invariants that every GFI-compliant generative function must satisfy. Each contract is a named theorem paired with a check function that tests the invariant over random traces.

Use `verify-gfi-contracts` to run the full contract suite against any model. This is the primary tool for validating that new generative functions, combinators, or compiled paths preserve GFI semantics.

```clojure
(require '[genmlx.contracts :as ct])
```

Source: `src/genmlx/contracts.cljs`

---

## Overview

The `contracts` var holds a map from contract keyword to `{:theorem string, :check fn}`. Each `:check` function takes `{:keys [model args trace]}` and returns a boolean. The verifier runs each contract over multiple random traces to catch violations that only appear under certain sampled values.

---

## verify-gfi-contracts

### `verify-gfi-contracts`

```clojure
(ct/verify-gfi-contracts model args & {:keys [n-trials contract-keys]})
```

Run GFI contracts over multiple random trials and return a structured report.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | IGenerativeFunction | The generative function to verify |
| `args` | vector | Arguments to the model |
| `n-trials` | integer (optional) | Number of random traces per contract (default: 50) |
| `contract-keys` | collection (optional) | Subset of contract keys to run (default: all 11) |

**Returns:** Map with keys:

| Key | Type | Description |
|-----|------|-------------|
| `:results` | map | `{contract-key -> {:pass int :fail int :theorem string}}` |
| `:total-pass` | integer | Sum of all passes across all contracts |
| `:total-fail` | integer | Sum of all failures across all contracts |
| `:all-pass?` | boolean | `true` if every trial of every contract passed |

**Example:**
```clojure
(def model
  (gen [x]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y (dist/gaussian mu 1)))))

;; Run all 11 contracts, 50 trials each
(def report (ct/verify-gfi-contracts model [0.0]))
(println (:all-pass? report))  ;; => true

;; Run a subset of contracts
(def report (ct/verify-gfi-contracts model [0.0]
              :contract-keys [:generate-weight-equals-score
                              :update-empty-identity]
              :n-trials 100))
```

---

## Contract List

The 11 contracts, grouped by the GFI operation they verify:

### 1. `:generate-weight-equals-score`

**Theorem:** `generate(model, args, trace.choices).weight = trace.score` when fully constrained.

When `generate` is given a complete set of constraints (all choices pinned), the returned weight equals the trace score. This verifies that the importance weight under full observation is the joint log-density.

---

### 2. `:update-empty-identity`

**Theorem:** `update(model, trace, trace.choices).weight = 0` (no-op update).

Updating a trace with its own choices is a no-op. The weight is zero because the new and old scores are equal.

---

### 3. `:update-weight-correctness`

**Theorem:** `update(model, trace, constraint).weight = new_score - old_score`.

The update weight equals the difference between the new and old trace scores. This is the fundamental correctness property of incremental recomputation.

---

### 4. `:update-round-trip`

**Theorem:** `update(trace, c)` then `update(trace', discard)` recovers original values.

Applying an update and then applying the discard as a constraint recovers the original trace values. This verifies that update and discard are inverses.

---

### 5. `:regenerate-empty-identity`

**Theorem:** `regenerate(model, trace, sel/none).weight = 0`, choices unchanged.

Regenerating with an empty selection is a no-op. The weight is zero and all choices are preserved.

---

### 6. `:project-all-equals-score`

**Theorem:** `project(model, trace, sel/all) = trace.score`.

Projecting onto all addresses recovers the full trace score. This verifies that the score decomposes correctly over addresses.

---

### 7. `:project-none-equals-zero`

**Theorem:** `project(model, trace, sel/none) = 0`.

Projecting onto no addresses yields zero. No choices selected means no log-density contribution.

---

### 8. `:assess-equals-generate-score`

**Theorem:** `assess(model, args, choices).weight = generate(model, args, choices).score`.

The weight returned by `assess` matches the score of a trace generated with the same choices. Both compute the joint log-density.

---

### 9. `:propose-generate-round-trip`

**Theorem:** `propose(model, args)` produces choices; `generate` with those choices has finite weight.

A model can propose choices and those choices can be fed back to `generate` to produce a valid trace with finite weight.

---

### 10. `:score-decomposition`

**Theorem:** Sum of `project(trace, {addr_i})` over all leaf addresses equals `trace.score`.

The total trace score equals the sum of individual per-address projections. This verifies additive decomposition of the log-density.

---

### 11. `:broadcast-equivalence`

**Theorem:** `vsimulate(model, args, N)` produces finite scores with shape `[N]`.

Vectorized simulation produces a batch of `N` traces, each with a finite score. This verifies that shape-based batching works correctly for the model.

---

## contracts

### `contracts`

```clojure
ct/contracts
```

The contract registry -- a map from contract keyword to `{:theorem string, :check fn}`. Can be used directly for custom verification workflows.

**Example:**
```clojure
;; Run a single contract manually
(let [{:keys [check]} (:generate-weight-equals-score ct/contracts)
      trace (p/simulate model args)]
  (check {:model model :args args :trace trace}))
;; => true
```
