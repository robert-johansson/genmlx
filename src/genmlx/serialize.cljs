(ns genmlx.serialize
  "Trace serialization for GenMLX.
   Save/load traces as JSON. Two modes:
   - choices-only (recommended): saves choices, reconstructs trace via generate
   - full-trace (convenience): saves choices + args + retval, best-effort

   Gen-fns are NOT serialized. User provides gen-fn during deserialization.
   Follows GenSerialization.jl conventions."
  (:require [genmlx.mlx :as mx]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.protocols :as p]
            [cljs.reader :as reader]
            [clojure.string :as str]))

(def ^:private fs (js/require "fs"))

;; ---------------------------------------------------------------------------
;; Dtype string <-> MLX dtype mapping
;; ---------------------------------------------------------------------------

(def ^:private dtype-code->str
  "Forward map: numeric NAPI DType enum → string name.
   Covers the full enum — uint32 is the categorical/token index dtype,
   so every LLM token trace hits it (genmlx-000i)."
  {0 "float32"
   1 "int32"
   2 "float16"
   3 "bfloat16"
   4 "uint32"
   5 "uint8"})

(def ^:private str->dtype-map
  "Reverse map: string name → MLX dtype constant."
  {"float32"  mx/float32
   "float64"  mx/float64
   "int32"    mx/int32
   "int64"    mx/int32
   "bool"     mx/bool-dt
   "float16"  mx/float16
   "bfloat16" mx/bfloat16
   "uint32"   mx/uint32
   "uint8"    mx/uint8})

(defn- dtype->str [dtype]
  (or (get dtype-code->str dtype)
      (throw (ex-info (str "Unknown dtype code: " dtype) {:dtype dtype}))))

(defn- str->dtype [s]
  (or (get str->dtype-map s)
      (throw (ex-info (str "Unknown dtype: " s) {:dtype s}))))

;; ---------------------------------------------------------------------------
;; MLX value <-> serializable data
;; ---------------------------------------------------------------------------

(defn- mlx-value->data
  "Convert an MLX array to a serializable map."
  [arr]
  (mx/eval! arr)
  (let [sh (mx/shape arr)
        dt (dtype->str (mx/dtype arr))]
    (if (empty? sh)
      {:type "scalar" :value (mx/item arr) :dtype dt}
      {:type "array" :value (mx/->clj arr) :shape sh :dtype dt})))

(defn- reject-nulls!
  "Refuse a JSON `null` standing where a number must be. `Float32Array.from`
   maps nil to 0, so without this a legacy row written before genmlx-94tu
   silently restores -Inf as 0.0 — and for an SMC log-weight that turns the
   impossible particle into the best one. Unrecoverable, so fail loudly."
  [v where]
  (cond
    (nil? v)
    (throw (ex-info (str "genmlx.serialize: JSON null where a number must be, in " where
                         ". This row predates the non-finite token codec (genmlx-94tu); "
                         "the original value (NaN or ±Inf) is not recoverable.")
                    {:genmlx/error :non-finite-lost :where where}))
    (sequential? v)
    (do (when (some nil? v)
          (throw (ex-info (str "genmlx.serialize: JSON null(s) where numbers must be, in " where
                               ". This row predates the non-finite token codec (genmlx-94tu); "
                               "the original values (NaN or ±Inf) are not recoverable.")
                          {:genmlx/error :non-finite-lost :where where
                           :n-null (count (filter nil? v))})))
        v)
    :else v))

(defn- data->mlx-value
  "Convert a serializable map back to an MLX array."
  [data]
  (let [dt (str->dtype (:dtype data))
        v  (reject-nulls! (:value data) (str "an MLX " (:type data) " leaf"))]
    (if (= "scalar" (:type data))
      (mx/scalar v dt)
      (mx/array v dt))))

;; ---------------------------------------------------------------------------
;; Address codec
;; ---------------------------------------------------------------------------
;; JSON object keys must be strings, but choicemap addresses are keywords,
;; integers (Map/Unfold/Scan element indices), or strings. A type prefix
;; keeps them distinct through the round trip — (name 0) used to throw, so
;; no combinator trace could be saved, and the load side keywordized
;; everything anyway (genmlx-000i). Unprefixed keys (legacy v1 files)
;; decode as keywords, the only address type the old codec produced.

(defn- addr->str
  "Encode a choicemap address as a prefixed string."
  [addr]
  (cond
    (keyword? addr) (str "k:" (subs (str addr) 1))  ;; keeps namespace
    (integer? addr) (str "i:" addr)
    (string? addr)  (str "s:" addr)
    :else (throw (ex-info (str "Unserializable address type: " (pr-str addr))
                          {:addr addr}))))

(defn- str->addr
  "Decode a prefixed address string (or the keyword js->clj made of it).
   Unprefixed strings decode as keywords for legacy-file compatibility."
  [s]
  (let [s (if (keyword? s) (subs (str s) 1) s)]
    (cond
      (str/starts-with? s "k:") (keyword (subs s 2))
      (str/starts-with? s "i:") (js/parseInt (subs s 2) 10)
      (str/starts-with? s "s:") (subs s 2)
      :else (keyword s))))

;; ---------------------------------------------------------------------------
;; ChoiceMap <-> serializable data
;; ---------------------------------------------------------------------------

(defn- choicemap->data
  "Recursively convert a ChoiceMap to a serializable map."
  [cm-node]
  (cond
    (nil? cm-node) {}
    (cm/has-value? cm-node)
    (let [v (cm/get-value cm-node)]
      (if (mx/array? v)
        (mlx-value->data v)
        ;; Non-MLX value (e.g., product distribution values)
        {:type "clj" :value (pr-str v)}))

    (instance? cm/Node cm-node)
    (-> (update-keys (:m cm-node) addr->str)
        (update-vals choicemap->data))

    :else {}))

(defn- mlx-data?
  "Is data a serialized MLX leaf (a map tagged :type \"scalar\" or \"array\")?"
  [data]
  (and (map? data) (#{"scalar" "array"} (:type data))))

(defn- data->choicemap
  "Recursively convert serializable data back to a ChoiceMap."
  [data]
  (cond
    ;; Leaf: MLX scalar or array
    (mlx-data? data)
    (cm/->Value (data->mlx-value data))

    ;; Leaf: CLJ value (pr-str'd)
    (and (map? data) (= "clj" (:type data)))
    (cm/->Value (reader/read-string (:value data)))

    ;; Node: map of encoded-address -> sub
    (map? data)
    (cm/->Node (-> (update-keys data str->addr)
                   (update-vals data->choicemap)))

    :else cm/EMPTY))

;; ---------------------------------------------------------------------------
;; Value serialization (for args, retval)
;; ---------------------------------------------------------------------------

(defn- serialize-value
  "Serialize a value that may contain MLX arrays. Map keys use the same
   prefixed address codec as choicemaps, so integer/string keys survive."
  [v]
  (cond
    (mx/array? v)   (mlx-value->data v)
    (sequential? v) (mapv serialize-value v)
    (map? v)        (-> (update-keys v addr->str) (update-vals serialize-value))
    (keyword? v)    {:type "keyword" :value (subs (str v) 1)}
    :else           v))

(defn- deserialize-value
  "Deserialize a value that may contain MLX arrays."
  [v]
  (cond
    (mlx-data? v)
    (data->mlx-value v)

    (and (map? v) (= "keyword" (:type v)))
    (keyword (:value v))

    (sequential? v) (mapv deserialize-value v)
    (map? v)        (-> (update-keys v str->addr) (update-vals deserialize-value))
    :else           v))

;; ---------------------------------------------------------------------------
;; Public composition surface (genmlx.memory composes these codecs)
;; ---------------------------------------------------------------------------
;; genmlx.memory is the durable PERSISTENCE face of the Bun membrane; it owns
;; the backend (bun:sqlite) but reuses these value<->data codecs rather than
;; re-deriving the dtype/address/MLX-array machinery. They are the same
;; functions the choices/trace round trip uses, exposed under stable names.

(defn choices->data
  "ChoiceMap -> JSON-serializable data (the recursive choicemap codec, MLX
   leaves included). Inverse: `data->choices`."
  [cm-node]
  (choicemap->data cm-node))

(defn data->choices
  "JSON-serializable data -> ChoiceMap. Inverse of `choices->data`."
  [data]
  (data->choicemap data))

(defn value->data
  "Serialize an arbitrary value that may contain MLX arrays plus nested
   maps/seqs (a score, weight, args vector, or parameter store). Map keys use
   the prefixed address codec, so keyword/int/string keys survive. Inverse:
   `data->value`."
  [v]
  (serialize-value v))

(defn data->value
  "Deserialize a value produced by `value->data` back to MLX arrays + CLJS
   data. Inverse of `value->data`."
  [v]
  (deserialize-value v))

;; ---------------------------------------------------------------------------
;; Non-finite tokens — the JSON round trip's one lossy spot
;; ---------------------------------------------------------------------------
;; `JSON.stringify` maps NaN and ±Infinity to `null`, and `Float32Array.from`
;; then maps `null` to 0. So a non-finite value written and read back returns
;; **0.0**, with no error at either end. For an SMC log-weight that is
;; maximally perverse: -Inf means "impossible particle" and 0.0 means
;; "log-weight 0", so after logsumexp normalization the dead particle restores
;; as the single BEST one. Measured 2026-08-01 (genmlx-94tu), before this fix:
;;   save-particles! payload : "weight":{"value":[null,-2,null]}
;;   restore-particles       : [0 -2 0]      <- input was [##-Inf -2 ##Inf]
;;
;; A -Inf score is ROUTINE, not exotic — any out-of-support observation
;; produces one, and the genmlx-4x5w support-guard work makes them strictly
;; more common. So the codec must round-trip them, not reject them.
;;
;; `genmlx.memory` fixed this on its own hashing path (genmlx-7qbr) and left
;; the payload codec untouched. These live HERE, in the lower layer, so there
;; is one implementation and memory delegates to it rather than keeping a
;; second copy that can drift (genmlx-pif1).

(def non-finite-tokens
  "Token <-> value table for the JSON round trip. Public so `genmlx.memory`
   composes this rather than defining its own."
  {"#nan" js/NaN "#+inf" js/Infinity "#-inf" (- js/Infinity)})

(defn- non-finite-token [x]
  (cond (js/Number.isNaN x) "#nan"
        (pos? x)            "#+inf"
        :else               "#-inf"))

(defn tag-non-finite
  "Replace non-finite numbers with their tokens, and escape any genuine string
   that could be read back as one (a leading `#` is doubled). Map KEYS are left
   alone — they are already the prefixed-address strings the codec produced."
  [x]
  (cond
    (number? x) (if (js/isFinite x) x (non-finite-token x))
    (string? x) (if (str/starts-with? x "#") (str "#" x) x)
    (map? x)    (update-vals x tag-non-finite)
    (vector? x) (mapv tag-non-finite x)
    (seq? x)    (mapv tag-non-finite x)
    :else       x))

(defn untag-non-finite
  "Inverse of `tag-non-finite`."
  [x]
  (cond
    (string? x) (cond
                  (contains? non-finite-tokens x) (get non-finite-tokens x)
                  (str/starts-with? x "##")       (subs x 1)
                  :else                           x)
    (map? x)    (update-vals x untag-non-finite)
    (vector? x) (mapv untag-non-finite x)
    (seq? x)    (mapv untag-non-finite x)
    :else       x))

(defn- stringify
  "`JSON.stringify` with non-finite numbers tokenized first. Every save path
   in this namespace goes through here — a bare `js/JSON.stringify` on trace
   data is the bug above."
  [data]
  (js/JSON.stringify (clj->js (tag-non-finite data)) nil 2))

;; ---------------------------------------------------------------------------
;; Public API: choices-only (recommended)
;; ---------------------------------------------------------------------------

(def ^:private serialization-version 1)

(defn save-choices
  "Serialize a trace's choices to a JSON string.
   Options:
     :gen-fn-id - optional string identifier for the gen-fn"
  [trace & {:keys [gen-fn-id]}]
  (let [data (cond-> {:version serialization-version
                      :format "genmlx-choices-v1"
                      :choices (choicemap->data (:choices trace))}
               gen-fn-id (assoc :gen-fn-id gen-fn-id))]
    (stringify data)))

(defn- parse-versioned
  "Parse a JSON string and assert it is serialization version 1."
  [json-str]
  ;; untag BEFORE anything else, so every downstream decoder sees real
  ;; NaN/±Inf rather than tokens. Rows written before genmlx-94tu contain a
  ;; bare JSON `null` where a number belongs; those are unrecoverable, and
  ;; `data->mlx-value`/`data->value` refuse them loudly rather than handing
  ;; back a silent 0.0.
  (let [data (untag-non-finite
              (js->clj (js/JSON.parse json-str) :keywordize-keys true))]
    (when (not= serialization-version (:version data))
      (throw (ex-info "Unsupported serialization version"
                      {:expected serialization-version :got (:version data)})))
    data))

(defn load-choices
  "Deserialize a JSON string to a ChoiceMap."
  [json-str]
  (data->choicemap (:choices (parse-versioned json-str))))

(defn reconstruct-trace
  "Reconstruct a full trace from a gen-fn, args, and serialized choices JSON.
   Runs generate with the deserialized choices to produce a valid trace."
  [gen-fn args json-str]
  (let [choices (load-choices json-str)
        {:keys [trace]} (p/generate gen-fn args choices)]
    trace))

;; ---------------------------------------------------------------------------
;; Public API: full trace
;; ---------------------------------------------------------------------------

(defn save-trace
  "Serialize a full trace to a JSON string.
   Includes choices, args, and retval (best-effort for retval).
   The optional :omega field (encapsulated randomness, genmlx-qbaa §4.5) is
   intentionally NOT persisted: it is internal randomness, and load-trace
   reconstructs via p/generate, which redraws a fresh omega for an
   EncapsulatedGF (the saved :score is reproduced in expectation, not exactly).
   Options:
     :gen-fn-id - optional string identifier for the gen-fn"
  [trace & {:keys [gen-fn-id]}]
  (let [score (:score trace)
        data {:version serialization-version
              :format "genmlx-trace-v1"
              :choices (choicemap->data (:choices trace))
              :args (mapv serialize-value (:args trace))
              ;; 0-d scores stay plain numbers (legacy format); [N]-shaped
              ;; batched scores serialize as array data — mx/realize on
              ;; them used to throw (item needs size 1; genmlx-000i).
              :score (if (and (mx/array? score) (seq (mx/shape score)))
                       (mlx-value->data score)
                       (mx/realize score))
              ;; Declare the score's encoding (genmlx-lbae): a marginal
              ;; (analytically-eliminated) trace's saved score is NOT a joint
              ;; density. load-trace re-generates from choices, so the loaded
              ;; trace is freshly joint-scored regardless of this field.
              :score-type (name (tr/score-type trace))}
        ;; retval is best-effort — closures, protocol instances won't survive
        data (try
               (assoc data :retval (serialize-value (:retval trace)))
               (catch :default _
                 data))
        data (cond-> data
               gen-fn-id (assoc :gen-fn-id gen-fn-id))]
    (stringify data)))

(defn load-trace
  "Deserialize a full trace JSON string. Requires the gen-fn.
   Reconstructs the trace via generate with the saved choices and args."
  [gen-fn json-str]
  (let [data (parse-versioned json-str)
        choices (data->choicemap (:choices data))
        args (mapv deserialize-value (:args data))
        {:keys [trace]} (p/generate gen-fn args choices)]
    trace))

;; ---------------------------------------------------------------------------
;; File I/O convenience
;; ---------------------------------------------------------------------------

(defn save-choices-to-file!
  "Save a trace's choices to a JSON file."
  [trace path & {:keys [gen-fn-id]}]
  (let [json (save-choices trace :gen-fn-id gen-fn-id)]
    (.writeFileSync fs path json "utf8")))

(defn load-choices-from-file
  "Load choices from a JSON file."
  [path]
  (let [json (.readFileSync fs path "utf8")]
    (load-choices json)))

(defn reconstruct-trace-from-file
  "Reconstruct a trace from a gen-fn, args, and a choices JSON file."
  [gen-fn args path]
  (let [json (.readFileSync fs path "utf8")]
    (reconstruct-trace gen-fn args json)))

(defn save-trace-to-file!
  "Save a full trace to a JSON file."
  [trace path & {:keys [gen-fn-id]}]
  (let [json (save-trace trace :gen-fn-id gen-fn-id)]
    (.writeFileSync fs path json "utf8")))

(defn load-trace-from-file
  "Load a full trace from a JSON file. Requires the gen-fn."
  [gen-fn path]
  (let [json (.readFileSync fs path "utf8")]
    (load-trace gen-fn json)))
