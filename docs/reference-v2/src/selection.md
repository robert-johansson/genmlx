# Selections

Selections are a composable address selection algebra. They specify which addresses to target in GFI operations like `regenerate` (resample selected choices), `project` (compute log-probability of selected choices), and MCMC proposals (select which variables to perturb).

Source: `src/genmlx/selection.cljs`

## ISelection Protocol

All selections implement the `ISelection` protocol:

```clojure
(defprotocol ISelection
  (selected?        [s addr] "Is this address selected?")
  (get-subselection [s addr] "Get the selection for addresses under this one."))
```

### `selected?`

```clojure
(selected? selection addr)
```

Test whether a specific address is selected.

| Parameter | Type | Description |
|-----------|------|-------------|
| `selection` | ISelection | The selection to query |
| `addr` | Keyword/Any | The address to test |

**Returns:** Boolean.

### `get-subselection`

```clojure
(get-subselection selection addr)
```

Get the selection that applies to sub-addresses beneath `addr`. This is how hierarchical selections work: when a combinator encounters a splice address, it calls `get-subselection` to obtain the selection for the sub-model's addresses.

| Parameter | Type | Description |
|-----------|------|-------------|
| `selection` | ISelection | The selection to query |
| `addr` | Keyword/Any | The parent address |

**Returns:** ISelection for the sub-addresses.

## Built-in Selections

### `all`

```clojure
sel/all
```

Selects every address at every depth. Both `selected?` and `get-subselection` return `true` and `all` respectively for any address.

```clojure
(sel/selected? sel/all :anything)        ;; => true
(sel/get-subselection sel/all :anything) ;; => sel/all
```

### `none`

```clojure
sel/none
```

Selects no addresses at any depth. Both `selected?` and `get-subselection` return `false` and `none` respectively for any address.

```clojure
(sel/selected? sel/none :anything)        ;; => false
(sel/get-subselection sel/none :anything) ;; => sel/none
```

## Creating Selections

### `select`

```clojure
(select & addrs)
```

Create a flat selection of specific addresses. When a selected address contains sub-addresses (e.g., a splice site), `get-subselection` returns `all` -- meaning the entire sub-model is selected.

| Parameter | Type | Description |
|-----------|------|-------------|
| `addrs` | Keywords/Any (variadic) | Addresses to select |

**Returns:** ISelection that selects exactly the given addresses.

```clojure
(sel/select :slope :intercept)
;; Selects :slope and :intercept but not :y, :x, etc.

(sel/selected? (sel/select :slope :intercept) :slope) ;; => true
(sel/selected? (sel/select :slope :intercept) :y)     ;; => false
```

### `from-set`

```clojure
(from-set s)
```

Create a flat selection from a set of addresses. Equivalent to calling `select` with the elements of the set.

| Parameter | Type | Description |
|-----------|------|-------------|
| `s` | Set | Set of addresses to select |

**Returns:** ISelection that selects exactly the addresses in the set.

```clojure
(sel/from-set #{:slope :intercept})
;; Same as (sel/select :slope :intercept)
```

### `hierarchical`

```clojure
(hierarchical & kvs)
```

Create a hierarchical selection that maps parent addresses to sub-selections. This is used when you need to select specific addresses within sub-models reached via `splice`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `kvs` | Keyword/ISelection pairs (variadic) | Alternating address, sub-selection pairs |

**Returns:** ISelection with per-address sub-selections.

For addresses present in the map, `selected?` returns `true` and `get-subselection` returns the associated sub-selection. For addresses not in the map, `selected?` returns `false` and `get-subselection` returns `none`.

```clojure
;; Select :x and :y under :sub1, everything under :sub2
(sel/hierarchical :sub1 (sel/select :x :y)
                  :sub2 sel/all)

;; Useful for models with splice calls:
(def model
  (gen [x]
    (let [params (splice :prior prior-model [])]
      (trace :y (dist/gaussian (:mu params) (:sigma params))))))

;; Select only :mu within the :prior sub-model
(p/regenerate model trace
  (sel/hierarchical :prior (sel/select :mu)))
```

## Composition

### `complement-sel`

```clojure
(complement-sel s)
```

Create the complement of a selection: selects everything **not** selected by `s`. The complement of the complement is the original selection.

| Parameter | Type | Description |
|-----------|------|-------------|
| `s` | ISelection | The selection to complement |

**Returns:** ISelection that inverts the selection logic.

Sub-selections are also complemented: `get-subselection` on the complement returns the complement of the inner selection's sub-selection.

```clojure
;; Select everything except :slope
(sel/complement-sel (sel/select :slope))

;; Use with regenerate: resample everything except observed data
(p/regenerate model trace
  (sel/complement-sel (sel/select :y0 :y1 :y2)))
```

## Usage with GFI Operations

Selections are primarily used with three GFI operations:

### regenerate

Resample the selected addresses from their prior, keeping all other choices fixed. Returns `{:trace :weight}` where the weight is the proposal ratio for MH acceptance.

```clojure
(require '[genmlx.protocols :as p]
         '[genmlx.selection :as sel])

;; Resample slope from its prior
(p/regenerate model trace (sel/select :slope))

;; Resample all latent variables
(p/regenerate model trace sel/all)
```

### project

Compute the log-probability of the selected choices in a trace. Returns an MLX scalar.

```clojure
;; Log-probability of just the slope choice
(p/project model trace (sel/select :slope))

;; Log-probability of all choices
(p/project model trace sel/all)
```

### MCMC proposals

Selections drive which variables MCMC kernels propose changes to:

```clojure
(require '[genmlx.inference.mcmc :as mcmc])

;; MH targeting only :slope
(mcmc/mh {:samples 1000 :selection (sel/select :slope)}
         model args obs)

;; Block Gibbs: cycle through variable groups
(mcmc/gibbs {:samples 1000
             :blocks [(sel/select :slope)
                      (sel/select :intercept)]}
            model args obs)
```
