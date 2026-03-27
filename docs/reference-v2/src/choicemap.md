# Choicemaps

Choicemaps are hierarchical, persistent data structures that map addresses to values. They store the random choices made during model execution and serve as the primary data exchange format for the Generative Function Interface (GFI).

A choicemap is built from two record types: **Value** (a leaf holding a single choice) and **Node** (an internal node mapping addresses to sub-choicemaps). All operations are purely functional -- every modification returns a new choicemap via structural sharing.

```clojure
(require '[genmlx.choicemap :as cm])
```

Source: `src/genmlx/choicemap.cljs`

---

## Creating Choicemaps

### `choicemap`

```clojure
(cm/choicemap & kvs)
```

Smart constructor that builds a choicemap from keyword-value pairs. Scalar values are wrapped in `Value` leaves. Plain maps are recursively converted into nested `Node` structures. Existing `IChoiceMap` values are preserved as-is.

| Parameter | Type | Description |
|-----------|------|-------------|
| `kvs` | keyword-value pairs | Alternating keys and values |

**Returns:** `Node` -- a choicemap with the given entries.

**Example:**
```clojure
;; Flat choicemap with two entries
(cm/choicemap :x 1.0 :y 2.0)

;; Nested choicemap -- plain maps become sub-nodes
(cm/choicemap :params {:slope 0.5 :intercept 1.2})

;; Single entry
(cm/choicemap :z (mx/scalar 3.14))

;; Use as constraints for generate
(def obs (cm/choicemap :y0 (mx/scalar 1.5) :y1 (mx/scalar 2.3)))
(p/generate model [xs] obs)
```

---

### `from-map`

```clojure
(cm/from-map m)
```

Convert a plain nested Clojure map into a choicemap. Nested maps become `Node`s; all other values become `Value` leaves.

| Parameter | Type | Description |
|-----------|------|-------------|
| `m` | map | A nested Clojure map |

**Returns:** `Node` if `m` is a map, `Value` if `m` is a non-map value.

**Example:**
```clojure
(cm/from-map {:slope 0.5 :intercept 1.2})
;; => Node with :slope -> Value(0.5), :intercept -> Value(1.2)

(cm/from-map {:params {:a 1 :b 2} :sigma 0.1})
;; => Nested: :params -> Node{:a -> Value(1), :b -> Value(2)}, :sigma -> Value(0.1)
```

---

### `from-flat-map`

```clojure
(cm/from-flat-map m)
```

Build a choicemap from a flat `{keyword -> value}` map. Each key becomes a top-level `Value` entry. Unlike `from-map`, this does not recurse into nested maps -- all values are treated as leaves.

| Parameter | Type | Description |
|-----------|------|-------------|
| `m` | map | A flat keyword-to-value map |

**Returns:** `Node` -- a choicemap with one `Value` leaf per entry, or `EMPTY` if the map is empty.

**Example:**
```clojure
(cm/from-flat-map {:x 1.0 :y 2.0 :z 3.0})
;; => Node with :x -> Value(1.0), :y -> Value(2.0), :z -> Value(3.0)

(cm/from-flat-map {})
;; => EMPTY
```

---

### `EMPTY`

```clojure
cm/EMPTY
```

The empty choicemap -- a `Node` with no children. Has no addresses and no values.

**Example:**
```clojure
(cm/addresses cm/EMPTY)
;; => []

(cm/has-value? cm/EMPTY)
;; => false
```

---

## Accessing Values

### `has-value?`

```clojure
(cm/has-value? cm)
```

Test whether a choicemap node holds a leaf value. Returns `true` for `Value` nodes, `false` for `Node`s, `nil`, or non-choicemap values.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cm` | IChoiceMap or nil | A choicemap to test |

**Returns:** `boolean`

**Example:**
```clojure
(cm/has-value? (cm/->Value 1.0))
;; => true

(cm/has-value? (cm/choicemap :x 1.0))
;; => false (it is a Node, not a Value)

(cm/has-value? (cm/get-submap (cm/choicemap :x 1.0) :x))
;; => true (the submap at :x is a Value)
```

---

### `get-value`

```clojure
(cm/get-value cm)
```

Extract the leaf value from a `Value` node. Returns `nil` if `cm` is not a `Value`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cm` | IChoiceMap or nil | A choicemap leaf to unwrap |

**Returns:** The stored value, or `nil`.

**Example:**
```clojure
(def choices (cm/choicemap :slope (mx/scalar 2.0)))

;; First get the submap, then extract the value
(cm/get-value (cm/get-submap choices :slope))
;; => #mlx 2.0
```

---

### `get-submap`

```clojure
(cm/get-submap cm addr)
```

Get the sub-choicemap at a single address level. Returns `EMPTY` if the address does not exist.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cm` | IChoiceMap or nil | A choicemap to look up in |
| `addr` | keyword | The address to look up |

**Returns:** `IChoiceMap` -- the sub-choicemap at `addr`, or `EMPTY`.

**Example:**
```clojure
(def choices (cm/choicemap :params {:slope 0.5 :intercept 1.2}))

(cm/get-submap choices :params)
;; => Node with :slope -> Value(0.5), :intercept -> Value(1.2)

(cm/get-submap choices :missing)
;; => EMPTY
```

---

### `get-choice`

```clojure
(cm/get-choice cm path)
```

Get the value at a path of addresses (a vector of keywords). Traverses nested `Node`s by calling `get-submap` at each level, then extracts the leaf value.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cm` | IChoiceMap | A choicemap to traverse |
| `path` | vector | Vector of keyword addresses |

**Returns:** The leaf value at the given path.

**Example:**
```clojure
(def choices (cm/choicemap :params {:slope (mx/scalar 0.5)}))

(cm/get-choice choices [:params :slope])
;; => #mlx 0.5

;; Equivalent to:
(-> choices (cm/get-submap :params) (cm/get-submap :slope) cm/get-value)
```

---

## Manipulation

### `set-value`

```clojure
(cm/set-value cm addr value)
```

Set a `Value` at a single keyword address in a `Node`. The value is wrapped in a `Value` leaf. Returns a new choicemap (persistent, immutable).

This is the fast path used by handler transitions where `cm` is always a `Node` and the value is always a raw (unwrapped) value.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cm` | Node | The choicemap to update |
| `addr` | keyword | The address to set |
| `value` | any | The raw value (will be wrapped in `Value`) |

**Returns:** `Node` -- a new choicemap with the value set.

**Example:**
```clojure
(def choices (cm/choicemap :x 1.0))
(cm/set-value choices :y 2.0)
;; => Node with :x -> Value(1.0), :y -> Value(2.0)
```

---

### `set-submap`

```clojure
(cm/set-submap cm addr sub-cm)
```

Set a sub-choicemap at a single address in a `Node`. The value must already be an `IChoiceMap` (it is not wrapped). Returns a new choicemap.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cm` | Node | The choicemap to update |
| `addr` | keyword | The address to set |
| `sub-cm` | IChoiceMap | The sub-choicemap to insert |

**Returns:** `Node` -- a new choicemap with the sub-choicemap set.

**Example:**
```clojure
(def root (cm/choicemap :x 1.0))
(def params (cm/choicemap :slope 0.5 :intercept 1.2))
(cm/set-submap root :params params)
;; => Node with :x -> Value(1.0), :params -> Node{:slope, :intercept}
```

---

### `set-choice`

```clojure
(cm/set-choice cm path value)
```

Set a value at a path of addresses, creating intermediate `Node`s as needed. Returns a new choicemap. If the value satisfies `IChoiceMap`, it is inserted directly; otherwise it is wrapped in a `Value`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cm` | IChoiceMap | The choicemap to update |
| `path` | vector | Vector of keyword addresses |
| `value` | any | The value to set (raw or IChoiceMap) |

**Returns:** `Node` -- a new choicemap with the value set at the given path.

**Example:**
```clojure
(cm/set-choice cm/EMPTY [:params :slope] 0.5)
;; => Node with :params -> Node{:slope -> Value(0.5)}

;; Create deeply nested structure from nothing
(cm/set-choice cm/EMPTY [:a :b :c] 42)
;; => Node with :a -> Node{:b -> Node{:c -> Value(42)}}
```

---

### `merge-cm`

```clojure
(cm/merge-cm a b)
```

Merge two choicemaps. Values in `b` override values in `a` at the same address. Nested nodes are merged recursively.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | IChoiceMap or nil | Base choicemap |
| `b` | IChoiceMap or nil | Overriding choicemap |

**Returns:** `IChoiceMap` -- the merged choicemap.

**Example:**
```clojure
(def a (cm/choicemap :x 1.0 :y 2.0))
(def b (cm/choicemap :y 99.0 :z 3.0))
(cm/merge-cm a b)
;; => Node with :x -> 1.0, :y -> 99.0, :z -> 3.0

;; Nested merge
(def a (cm/choicemap :params {:slope 0.5 :intercept 1.0}))
(def b (cm/choicemap :params {:slope 0.8}))
(cm/merge-cm a b)
;; => :params -> {:slope 0.8, :intercept 1.0}
```

---

## Traversal

### `addresses`

```clojure
(cm/addresses cm)
```

Return all leaf address paths as a vector of vectors. Each inner vector is the full path from root to a `Value` leaf.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cm` | IChoiceMap or nil | The choicemap to enumerate |

**Returns:** `vector` of `vector` -- all leaf address paths.

**Example:**
```clojure
(cm/addresses (cm/choicemap :x 1 :params {:a 2 :b 3}))
;; => [[:x] [:params :a] [:params :b]]

(cm/addresses cm/EMPTY)
;; => []
```

---

### `to-map`

```clojure
(cm/to-map cm)
```

Convert a choicemap to a plain nested Clojure map. `Node`s become maps, `Value` leaves become their unwrapped values.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cm` | IChoiceMap or nil | The choicemap to convert |

**Returns:** A nested Clojure map.

**Example:**
```clojure
(cm/to-map (cm/choicemap :x 1.0 :params {:slope 0.5}))
;; => {:x 1.0, :params {:slope 0.5}}
```

---

### `to-flat-map` (via `to-map`)

Choicemaps do not have a dedicated `to-flat-map` function. For flat choicemaps (no nesting), `to-map` returns a flat map directly. For nested choicemaps, use `addresses` and `get-choice` together:

```clojure
;; Flatten any choicemap to {path -> value}
(into {}
  (map (fn [path] [path (cm/get-choice cm path)])
       (cm/addresses cm)))
```

---

## Batched Operations

These functions support vectorized inference by converting between scalar choicemaps and choicemaps with `[N]`-shaped leaf arrays.

### `stack-choicemaps`

```clojure
(cm/stack-choicemaps cms mlx-stack-fn)
```

Stack `N` choicemaps into one choicemap where each leaf is an `[N]`-shaped array. All `N` choicemaps must have the same address structure.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cms` | vector | Vector of choicemaps with identical address structure |
| `mlx-stack-fn` | function | Stacking function, typically `mx/stack` |

**Returns:** `IChoiceMap` -- a single choicemap with `[N]`-shaped leaves.

**Example:**
```clojure
(require '[genmlx.mlx :as mx])

(def cm1 (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0)))
(def cm2 (cm/choicemap :x (mx/scalar 3.0) :y (mx/scalar 4.0)))

(cm/stack-choicemaps [cm1 cm2] mx/stack)
;; => Node with :x -> Value([1.0, 3.0]), :y -> Value([2.0, 4.0])
```

---

### `unstack-choicemap`

```clojure
(cm/unstack-choicemap cm n mlx-index-fn scalar-leaf-fn)
```

Split a choicemap with `[N]`-shaped leaves into `N` individual scalar choicemaps. The inverse of `stack-choicemaps`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cm` | IChoiceMap | A choicemap with `[N]`-shaped leaves |
| `n` | integer | Number of particles/elements to split into |
| `mlx-index-fn` | function | Indexing function `(fn [array i] ...)` |
| `scalar-leaf-fn` | function | Predicate `(fn [value] ...)` returning true if value is scalar |

**Returns:** `vector` of `IChoiceMap` -- `N` choicemaps with scalar leaves.

**Example:**
```clojure
(def stacked (cm/choicemap :x (mx/array [1.0 2.0 3.0])))

(cm/unstack-choicemap stacked 3
  (fn [arr i] (mx/index arr i))
  (fn [v] (= 0 (count (mx/shape v)))))
;; => [Node{:x -> 1.0}, Node{:x -> 2.0}, Node{:x -> 3.0}]
```

---

## Protocols

### IChoiceMap

The protocol implemented by both `Value` and `Node`:

```clojure
(defprotocol IChoiceMap
  (-has-value? [cm])
  (-get-value  [cm])
  (-get-submap [cm addr])
  (-submaps    [cm]))
```

| Method | Description |
|--------|-------------|
| `-has-value?` | Returns `true` if this node holds a leaf value |
| `-get-value` | Returns the leaf value; throws on `Node` |
| `-get-submap` | Returns the sub-choicemap at the given address |
| `-submaps` | Returns a sequence of `[address sub-choicemap]` pairs |

In practice, use the public API functions (`has-value?`, `get-value`, `get-submap`) rather than calling protocol methods directly. The public functions handle `nil` and non-choicemap inputs gracefully.

### Value

```clojure
(cm/->Value v)
```

Leaf record wrapping a single random choice value. `has-value?` returns `true`. Typically you do not construct `Value` records directly -- use `choicemap`, `from-map`, or `set-value` instead.

### Node

```clojure
(cm/->Node m)
```

Internal record holding a persistent map of `{address -> IChoiceMap}`. `has-value?` returns `false`. Calling `get-value` on a `Node` throws an error listing the available sub-addresses. As with `Value`, prefer the smart constructors over direct `Node` construction.
