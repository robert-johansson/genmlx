# Serialization

Save and load traces and choicemaps as JSON. Two serialization modes are available:

- **Choices-only** (recommended): saves only the choicemap, reconstructs the full trace via `generate` on load. Compact and reliable.
- **Full-trace**: saves choices, args, retval, and score. Best-effort for retval (closures and protocol instances will not survive serialization).

Gen-fns are never serialized. The user provides the gen-fn during deserialization. This follows the conventions of GenSerialization.jl.

```clojure
(require '[genmlx.serialize :as ser])
```

Source: `src/genmlx/serialize.cljs`

---

## Traces

### `save-trace`

```clojure
(ser/save-trace trace & {:keys [gen-fn-id]})
```

Serialize a full trace to a JSON string. Includes choices, args, score, and retval (best-effort for retval -- closures and protocol instances will not round-trip).

| Parameter | Type | Description |
|-----------|------|-------------|
| `trace` | Trace | The trace to serialize |
| `gen-fn-id` | string (optional) | Human-readable identifier for the gen-fn |

**Returns:** JSON string (pretty-printed).

---

### `load-trace`

```clojure
(ser/load-trace gen-fn json-str)
```

Deserialize a full trace from a JSON string. Reconstructs the trace by running `generate` with the saved choices and deserialized args.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gen-fn` | IGenerativeFunction | The generative function that produced the trace |
| `json-str` | string | JSON string from `save-trace` |

**Returns:** `Trace` -- a fully reconstructed trace.

---

### `save-trace-to-file`

```clojure
(ser/save-trace-to-file trace path & {:keys [gen-fn-id]})
```

Save a full trace to a JSON file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `trace` | Trace | The trace to serialize |
| `path` | string | File path to write |
| `gen-fn-id` | string (optional) | Human-readable identifier for the gen-fn |

**Returns:** `nil` (writes to disk).

---

### `load-trace-from-file`

```clojure
(ser/load-trace-from-file gen-fn path)
```

Load a full trace from a JSON file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gen-fn` | IGenerativeFunction | The generative function that produced the trace |
| `path` | string | File path to read |

**Returns:** `Trace` -- a fully reconstructed trace.

---

## Choicemaps

### `save-choices`

```clojure
(ser/save-choices trace & {:keys [gen-fn-id]})
```

Serialize a trace's choices to a JSON string. This is the recommended serialization mode -- it produces a compact representation that can be reliably reconstructed.

| Parameter | Type | Description |
|-----------|------|-------------|
| `trace` | Trace | The trace whose choices to serialize |
| `gen-fn-id` | string (optional) | Human-readable identifier for the gen-fn |

**Returns:** JSON string (pretty-printed).

---

### `load-choices`

```clojure
(ser/load-choices json-str)
```

Deserialize a JSON string to a `ChoiceMap`. Does not require a gen-fn -- returns the raw choicemap.

| Parameter | Type | Description |
|-----------|------|-------------|
| `json-str` | string | JSON string from `save-choices` |

**Returns:** `ChoiceMap`

---

### `save-choices-to-file`

```clojure
(ser/save-choices-to-file trace path & {:keys [gen-fn-id]})
```

Save a trace's choices to a JSON file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `trace` | Trace | The trace whose choices to serialize |
| `path` | string | File path to write |
| `gen-fn-id` | string (optional) | Human-readable identifier for the gen-fn |

**Returns:** `nil` (writes to disk).

---

### `load-choices-from-file`

```clojure
(ser/load-choices-from-file path)
```

Load choices from a JSON file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | File path to read |

**Returns:** `ChoiceMap`

---

## Reconstruction

### `reconstruct-trace`

```clojure
(ser/reconstruct-trace gen-fn args json-str)
```

Reconstruct a full trace from a gen-fn, args, and a serialized choices JSON string. Runs `generate` with the deserialized choices to produce a valid trace with correct score, retval, and metadata.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gen-fn` | IGenerativeFunction | The generative function |
| `args` | vector | Arguments to the generative function |
| `json-str` | string | JSON string from `save-choices` |

**Returns:** `Trace`

**Example:**
```clojure
;; Save choices
(def trace (p/simulate model [xs]))
(def json (ser/save-choices trace :gen-fn-id "linear-regression"))

;; Later, reconstruct
(def restored (ser/reconstruct-trace model [xs] json))
```

---

### `reconstruct-trace-from-file`

```clojure
(ser/reconstruct-trace-from-file gen-fn args path)
```

Reconstruct a trace from a gen-fn, args, and a choices JSON file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gen-fn` | IGenerativeFunction | The generative function |
| `args` | vector | Arguments to the generative function |
| `path` | string | File path to read |

**Returns:** `Trace`

---

## Serialization Format

Both modes use JSON with the following structure:

**Choices-only** (`genmlx-choices-v1`):
```json
{
  "version": 1,
  "format": "genmlx-choices-v1",
  "gen_fn_id": "optional-id",
  "choices": { ... }
}
```

**Full-trace** (`genmlx-trace-v1`):
```json
{
  "version": 1,
  "format": "genmlx-trace-v1",
  "gen_fn_id": "optional-id",
  "choices": { ... },
  "args": [ ... ],
  "score": 0.0,
  "retval": ...
}
```

MLX arrays are encoded as:
- Scalars: `{"type": "scalar", "value": 1.0, "dtype": "float32"}`
- Arrays: `{"type": "array", "value": [1.0, 2.0], "shape": [2], "dtype": "float32"}`
- Clojure values: `{"type": "clj", "value": "<pr-str representation>"}`
