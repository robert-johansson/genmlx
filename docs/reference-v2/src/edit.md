# Edit Interface

The parametric edit interface generalizes `update` and `regenerate` into a single operation with typed edit requests. Each edit type carries a backward request that enables automatic computation of acceptance weights for reversible kernels -- the foundation of SMCP3.

Every trace mutation (constrain new observations, resample latent variables, apply a custom proposal) becomes an instance of `edit`. The backward request reverses the edit, which is what makes reversible-jump and SMCP3 kernels work.

```clojure
(require '[genmlx.edit :as edit])
```

Source: `src/genmlx/edit.cljs`

---

## Edit Types

Three record types represent the three kinds of trace edits:

### `ConstraintEdit`

```clojure
(edit/->ConstraintEdit constraints)
```

Equivalent to `update`: replace choices at specific addresses with new values. The backward request is a `ConstraintEdit` containing the discarded (overwritten) values.

| Field | Type | Description |
|-------|------|-------------|
| `constraints` | ChoiceMap | New values to constrain |

---

### `SelectionEdit`

```clojure
(edit/->SelectionEdit selection)
```

Equivalent to `regenerate`: resample the selected addresses from their prior. The backward request is a `SelectionEdit` with the same selection (regeneration is its own inverse in terms of the proposal).

| Field | Type | Description |
|-------|------|-------------|
| `selection` | Selection | Addresses to resample |

---

### `ProposalEdit`

```clojure
(edit/->ProposalEdit forward-gf forward-args backward-gf backward-args)
```

Apply a custom proposal for SMCP3-style reversible kernels. The forward generative function proposes new choices; the backward generative function scores the reverse move. The combined weight accounts for the proposal ratio.

| Field | Type | Description |
|-------|------|-------------|
| `forward-gf` | IGenerativeFunction | Proposal that generates new choices |
| `forward-args` | vector or nil | Arguments to the forward proposal (default: `[trace.choices]`) |
| `backward-gf` | IGenerativeFunction | Scores the reverse move |
| `backward-args` | vector or nil | Arguments to the backward proposal (default: `[new-trace.choices]`) |

---

## Constructors

### `constraint-edit`

```clojure
(edit/constraint-edit constraints)
```

Create a `ConstraintEdit` -- equivalent to calling `update` on a trace.

| Parameter | Type | Description |
|-----------|------|-------------|
| `constraints` | ChoiceMap | New values to constrain |

**Returns:** `ConstraintEdit`

**Example:**
```clojure
(def req (edit/constraint-edit (cm/choicemap :x (mx/scalar 1.0))))
(edit/edit-dispatch model trace req)
```

---

### `selection-edit`

```clojure
(edit/selection-edit selection)
```

Create a `SelectionEdit` -- equivalent to calling `regenerate` on a trace.

| Parameter | Type | Description |
|-----------|------|-------------|
| `selection` | Selection | Addresses to resample |

**Returns:** `SelectionEdit`

**Example:**
```clojure
(def req (edit/selection-edit (sel/select :z)))
(edit/edit-dispatch model trace req)
```

---

### `proposal-edit`

```clojure
(edit/proposal-edit forward-gf backward-gf)
(edit/proposal-edit forward-gf forward-args backward-gf backward-args)
```

Create a `ProposalEdit` for SMCP3-style reversible kernels. In the two-argument form, both `forward-args` and `backward-args` default to `nil` (the dispatch function will use `[trace.choices]` and `[new-trace.choices]` respectively).

| Parameter | Type | Description |
|-----------|------|-------------|
| `forward-gf` | IGenerativeFunction | Proposal that generates new choices |
| `forward-args` | vector or nil | Arguments to the forward proposal |
| `backward-gf` | IGenerativeFunction | Scores the reverse move |
| `backward-args` | vector or nil | Arguments to the backward proposal |

**Returns:** `ProposalEdit`

---

## Dispatch

### `edit-dispatch`

```clojure
(edit/edit-dispatch gf trace edit-request)
```

Generic edit implementation that dispatches on the `EditRequest` type. This is the single entry point for all trace edits.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gf` | IGenerativeFunction | The generative function that produced the trace |
| `trace` | Trace | The trace to edit |
| `edit-request` | ConstraintEdit, SelectionEdit, or ProposalEdit | The edit to apply |

**Returns:** Map with keys:

| Key | Type | Description |
|-----|------|-------------|
| `:trace` | Trace | The updated trace |
| `:weight` | MLX scalar | Log importance weight of the edit |
| `:discard` | ChoiceMap | Values that were overwritten |
| `:backward-request` | EditRequest | The reverse edit (for acceptance ratio computation) |

**Dispatch behavior:**

- **ConstraintEdit** -- delegates to `p/update`. Backward request is a `ConstraintEdit` containing the discarded values.
- **SelectionEdit** -- delegates to `p/regenerate`. Backward request is a `SelectionEdit` with the same selection.
- **ProposalEdit** -- runs `p/propose` on the forward GF, applies results via `p/update`, scores the reverse move with `p/assess` on the backward GF, and computes the combined weight as `update-weight + backward-score - forward-score`. Backward request swaps forward and backward GFs.

**Example:**
```clojure
;; ConstraintEdit -- update observed data
(let [req (edit/constraint-edit (cm/choicemap :y0 (mx/scalar 2.5)))
      {:keys [trace weight backward-request]} (edit/edit-dispatch model trace req)]
  ;; backward-request is a ConstraintEdit with the old :y0 value
  (println "Weight:" (mx/item weight)))

;; SelectionEdit -- resample latent variables
(let [req (edit/selection-edit (sel/select :slope :intercept))
      {:keys [trace weight]} (edit/edit-dispatch model trace req)]
  (println "Weight:" (mx/item weight)))
```

---

## IEdit Protocol

```clojure
(defprotocol IEdit
  (edit [gf trace edit-request]))
```

Protocol for generative functions that support the edit interface directly. `edit-dispatch` provides a default implementation that works with any GFI-compliant generative function, so implementing `IEdit` is optional -- it is available for generative functions that want to provide optimized edit paths.

| Method | Description |
|--------|-------------|
| `edit` | Apply an edit request to a trace, returning `{:trace :weight :discard :backward-request}` |
