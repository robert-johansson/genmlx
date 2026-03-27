# Validate

Static and dynamic validation for generative functions. Run a model through a validation handler that checks for structural problems: execution errors, duplicate addresses, non-finite scores, empty models, and materialization calls that break vectorized execution.

Unlike `contracts` (which verifies measure-theoretic GFI invariants), `validate` focuses on catching programming mistakes in model definitions.

```clojure
(require '[genmlx.verify :as v])
```

Source: `src/genmlx/verify.cljs`

---

## validate-gen-fn

### `validate-gen-fn`

```clojure
(v/validate-gen-fn gf args)
(v/validate-gen-fn gf args opts)
```

Validate a generative function for structural correctness. Performs both static analysis (source form inspection) and dynamic analysis (execution with a validation handler).

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | DynamicGF | The generative function to validate |
| `args` | vector | Arguments to pass to the model |
| `opts` | map (optional) | Options (see below) |

**Options:**

| Key | Type | Description |
|-----|------|-------------|
| `:key` | MLX PRNG key | Random key for sampling (default: fresh key) |
| `:n-trials` | integer | Number of independent validation runs (default: 1) |

Multiple trials can catch conditional problems that only manifest in certain execution paths (e.g., duplicate addresses inside branches).

**Returns:** Map with keys:

| Key | Type | Description |
|-----|------|-------------|
| `:valid?` | boolean | `true` if no errors were found (warnings do not affect this) |
| `:violations` | vector | All violations found across static analysis and all trials |
| `:trace` | Trace or nil | Trace from the last trial (nil if execution failed) |

**Example:**
```clojure
(def model
  (gen [x]
    (let [mu (trace :mu (dist/gaussian 0 1))]
      (trace :y (dist/gaussian mu 1)))))

(def result (v/validate-gen-fn model [0.0]))
(:valid? result)      ;; => true
(:violations result)  ;; => []

;; Run multiple trials to catch conditional issues
(def result (v/validate-gen-fn model [0.0] {:n-trials 10}))
```

---

## Violation Types

Each violation is a map with `:type`, `:severity`, and `:message`. Severity is either `:error` (model is broken) or `:warning` (model works but may have issues).

### `:execution-error`

**Severity:** `:error`

The model body threw an exception during execution. The message includes the error details. The trace will be `nil`.

```clojure
{:type :execution-error
 :severity :error
 :message "Model execution failed: Cannot read properties of undefined"}
```

---

### `:duplicate-address`

**Severity:** `:error`

The same address was traced more than once in a single execution. This violates the GFI requirement that trace addresses are unique. Includes the duplicate address in `:addr`.

```clojure
{:type :duplicate-address
 :severity :error
 :message "Address :x traced more than once"
 :addr :x}
```

---

### `:non-finite-score`

**Severity:** `:error`

The total trace score is not a finite number (NaN or infinity). This typically indicates a distribution parameterization problem (e.g., zero standard deviation in a Gaussian).

```clojure
{:type :non-finite-score
 :severity :error
 :message "Model score is -Infinity"}
```

---

### `:empty-model`

**Severity:** `:warning`

The model body contains no `trace` calls. The choicemap is empty. This is not an error -- deterministic generative functions are valid -- but it is usually unintentional.

```clojure
{:type :empty-model
 :severity :warning
 :message "Model body contains no trace calls"}
```

---

### `:materialization-in-body`

**Severity:** `:warning`

The model source form contains `eval!` or `item` calls. These force eager evaluation of MLX computation graphs, which breaks vectorized execution (`vsimulate`/`vgenerate`). Detected statically from the source form -- no execution needed.

```clojure
{:type :materialization-in-body
 :severity :warning
 :message "Found eval! in model body — breaks vectorized execution"}
```
