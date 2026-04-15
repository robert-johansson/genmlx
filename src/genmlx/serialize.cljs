(ns genmlx.serialize
  "Trace serialization for GenMLX.
   Save/load traces as JSON. Two modes:
   - choices-only (recommended): saves choices, reconstructs trace via generate
   - full-trace (convenience): saves choices + args + retval, best-effort

   Gen-fns are NOT serialized. User provides gen-fn during deserialization.
   Follows GenSerialization.jl conventions."
  (:require [genmlx.mlx :as mx]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [cljs.reader :as reader]))

(def ^:private fs (js/require "fs"))

;; ---------------------------------------------------------------------------
;; Dtype string <-> MLX dtype mapping
;; ---------------------------------------------------------------------------

(def ^:private dtype-code->str
  "Forward map: numeric NAPI DType enum → string name."
  {0 "float32"
   1 "int32"})

(def ^:private str->dtype-map
  "Reverse map: string name → MLX dtype constant."
  {"float32" mx/float32
   "float64" mx/float64
   "int32"   mx/int32
   "int64"   mx/int32
   "bool"    mx/bool-dt})

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

(defn- data->mlx-value
  "Convert a serializable map back to an MLX array."
  [data]
  (let [dt (str->dtype (:dtype data))]
    (if (= "scalar" (:type data))
      (mx/scalar (:value data) dt)
      (mx/array (:value data) dt))))

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
    (into {}
      (map (fn [[k sub]]
             [(name k) (choicemap->data sub)])
           (cm/-submaps cm-node)))

    :else {}))

(defn- data->choicemap
  "Recursively convert serializable data back to a ChoiceMap."
  [data]
  (cond
    ;; Leaf: MLX scalar or array
    (and (map? data) (contains? data :type)
         (or (= "scalar" (:type data)) (= "array" (:type data))))
    (cm/->Value (data->mlx-value data))

    ;; Leaf: CLJ value (pr-str'd)
    (and (map? data) (= "clj" (:type data)))
    (cm/->Value (reader/read-string (:value data)))

    ;; Node: map of string -> sub
    (map? data)
    (cm/->Node
      (into {}
        (map (fn [[k v]]
               [(keyword k) (data->choicemap v)])
             data)))

    :else cm/EMPTY))

;; ---------------------------------------------------------------------------
;; Value serialization (for args, retval)
;; ---------------------------------------------------------------------------

(defn- serialize-value
  "Serialize a value that may contain MLX arrays."
  [v]
  (cond
    (mx/array? v)   (mlx-value->data v)
    (vector? v)     (mapv serialize-value v)
    (sequential? v) (mapv serialize-value v)
    (map? v)        (into {} (map (fn [[k val]] [(name k) (serialize-value val)]) v))
    (keyword? v)    {:type "keyword" :value (name v)}
    :else           v))

(defn- deserialize-value
  "Deserialize a value that may contain MLX arrays."
  [v]
  (cond
    (and (map? v) (contains? v :type)
         (or (= "scalar" (:type v)) (= "array" (:type v))))
    (data->mlx-value v)

    (and (map? v) (= "keyword" (:type v)))
    (keyword (:value v))

    (vector? v)     (mapv deserialize-value v)
    (sequential? v) (mapv deserialize-value v)
    (map? v)        (into {} (map (fn [[k val]] [(keyword k) (deserialize-value val)]) v))
    :else           v))

;; ---------------------------------------------------------------------------
;; Public API: choices-only (recommended)
;; ---------------------------------------------------------------------------

(defn save-choices
  "Serialize a trace's choices to a JSON string.
   Options:
     :gen-fn-id - optional string identifier for the gen-fn"
  [trace & {:keys [gen-fn-id]}]
  (let [data {:version 1
              :format "genmlx-choices-v1"
              :choices (choicemap->data (:choices trace))}
        data (if gen-fn-id (assoc data :gen-fn-id gen-fn-id) data)]
    (js/JSON.stringify (clj->js data) nil 2)))

(defn load-choices
  "Deserialize a JSON string to a ChoiceMap."
  [json-str]
  (let [data (js->clj (js/JSON.parse json-str) :keywordize-keys true)]
    (when (not= 1 (:version data))
      (throw (ex-info "Unsupported serialization version"
                      {:expected 1 :got (:version data)})))
    (data->choicemap (:choices data))))

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
   Options:
     :gen-fn-id - optional string identifier for the gen-fn"
  [trace & {:keys [gen-fn-id]}]
  (let [data {:version 1
              :format "genmlx-trace-v1"
              :choices (choicemap->data (:choices trace))
              :args (mapv serialize-value (:args trace))
              :score (mx/realize (:score trace))}
        ;; retval is best-effort — closures, protocol instances won't survive
        data (try
               (assoc data :retval (serialize-value (:retval trace)))
               (catch :default _
                 data))
        data (if gen-fn-id (assoc data :gen-fn-id gen-fn-id) data)]
    (js/JSON.stringify (clj->js data) nil 2)))

(defn load-trace
  "Deserialize a full trace JSON string. Requires the gen-fn.
   Reconstructs the trace via generate with the saved choices and args."
  [gen-fn json-str]
  (let [data (js->clj (js/JSON.parse json-str) :keywordize-keys true)]
    (when (not= 1 (:version data))
      (throw (ex-info "Unsupported serialization version"
                      {:expected 1 :got (:version data)})))
    (let [choices (data->choicemap (:choices data))
          args (mapv deserialize-value (:args data))
          {:keys [trace]} (p/generate gen-fn args choices)]
      trace)))

;; ---------------------------------------------------------------------------
;; File I/O convenience
;; ---------------------------------------------------------------------------

(defn save-choices-to-file
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

(defn save-trace-to-file
  "Save a full trace to a JSON file."
  [trace path & {:keys [gen-fn-id]}]
  (let [json (save-trace trace :gen-fn-id gen-fn-id)]
    (.writeFileSync fs path json "utf8")))

(defn load-trace-from-file
  "Load a full trace from a JSON file. Requires the gen-fn."
  [gen-fn path]
  (let [json (.readFileSync fs path "utf8")]
    (load-trace gen-fn json)))
