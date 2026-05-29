(ns genmlx.llm.vision
  "VLM-as-generative-function: per-cell classification with structural composition.

   Two layers, one per concern:

   - **Async I/O** wraps `@mlx-node/lm`'s `loadSession` / `send`. Used to load
     the VLM and classify individual cells. Returns promises.
   - **Sync GFI** wraps the per-cell categorical structure as a `gen` function.
     Each cell is a trace site at a keyword address; observations from the VLM
     become constraints when invoking via `p/generate`.

   The split follows the GenMLX async boundary convention: I/O is async,
   GFI math is sync. Run perception once via `classify-grid`, then feed the
   result into the gen-fn via constraints.

   See `examples/vlm_grid_gf.cljs` for the end-to-end demo.

   Operational note: `classify-cell` calls `session.reset()` on every call. Without
   that, conversation history accumulates across independent classifications and
   per-call latency grows monotonically with N (verified empirically — 0.6s rises
   to 116s by the 18th call). With reset, latency is flat. See
   `dev/docs/INVESTIGATION_VLM_PER_CELL_PROBE.md`."
  (:require ["@mlx-node/lm" :as mlx-lm]
            [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [promesa.core :as pr]
            [clojure.string :as str]))

;; ---------------------------------------------------------------------------
;; Async I/O layer
;; ---------------------------------------------------------------------------

(defn load-vlm
  "Load a VLM session via `@mlx-node/lm`'s `loadSession`. Returns a promise.

   Supports any model whose `model_type` is recognized by mlx-node and whose
   safetensors include vision_tower / visual / embed_vision weights. Verified
   today: Qwen3.6-35B-A3B-4bit (`Qwen3_5MoeForConditionalGeneration` architecture
   loaded via `qwen3_5_moe`)."
  [model-path]
  (.loadSession mlx-lm model-path))

(defn- option->line
  "Format one option for the classification prompt. Accepts a bare string or a
   `{:label :description}` map; the description (when present) gives the model
   the visual hint it needs for accurate classification."
  [opt]
  (if (map? opt)
    (str "- " (:label opt) " (" (:description opt) ")")
    (str "- " opt)))

(defn- option->label
  "Extract the canonical label string from an option (string or map)."
  [opt]
  (if (map? opt) (:label opt) opt))

(defn classify-cell
  "Classify a single cell image into one of `options`. Returns a promise of the
   selected option label (lowercased, alphabetic-only canonical form).

   `options` is a vector of either bare strings or `{:label :description}` maps.
   Including descriptions improves accuracy substantially — without them the
   model has only the label name to disambiguate, which loses visual hints
   (e.g. \"empty (a blank white cell with thin gray borders)\" beats just
   \"empty\").

   Always calls `session.reset()` before the send so history doesn't accumulate
   across independent calls. Disables the model's reasoning trace and the
   default repetition penalty so short single-word answers come through cleanly."
  [session image-bytes options]
  (pr/let [_ (.reset session)
           opts-text (str/join "\n" (map option->line options))
           prompt (str "What is shown in this image? Answer with EXACTLY ONE of:\n"
                       opts-text
                       "\n\nOutput ONLY the single word, nothing else.")
           config #js {:maxNewTokens 8
                       :reasoningEffort "none"
                       :repetitionPenalty 1.0}
           result (.send session prompt
                          #js {:images #js [image-bytes]
                               :config config})]
    (-> (or (.-text result) "")
        str/trim
        (str/split #"\s+")
        first
        (or "")
        str/lower-case
        (str/replace #"[^a-z]" ""))))

(defn classify-grid
  "Classify all cells of an N×M grid. Returns a promise of
   `{:rows N :cols M :labels <2D vector of strings>}`.

   `crop-fn` takes `[r c]` and returns a Uint8Array of cell image bytes (or a
   promise of one). `options` is the vector of allowed labels.

   Cells are processed strictly sequentially. A single ChatSession does not
   support concurrent send()s — reset() will throw if one is in flight — and
   the model's Metal kernels serialize anyway, so concurrency wouldn't help.

   `progress-fn` (optional) is called as `(progress-fn r c label dt-ms)` after
   each cell completes."
  ([session rows cols crop-fn options]
   (classify-grid session rows cols crop-fn options nil))
  ([session rows cols crop-fn options progress-fn]
   (let [coords (vec (for [r (range rows) c (range cols)] [r c]))]
     (pr/let
      [results (reduce
                (fn [acc-promise [r c]]
                  (pr/let [acc acc-promise
                           t0 (.now js/performance)
                           bytes (crop-fn r c)
                           label (classify-cell session bytes options)
                           dt (- (.now js/performance) t0)]
                    (when progress-fn (progress-fn r c label dt))
                    (conj acc [r c label])))
                (pr/resolved [])
                coords)]
       (let [lookup (into {} (map (fn [[r c lbl]] [[r c] lbl]) results))]
         {:rows rows :cols cols
          :labels (vec (for [r (range rows)]
                         (vec (for [c (range cols)]
                                (get lookup [r c] "?")))))})))))

;; ---------------------------------------------------------------------------
;; Sync GFI layer
;; ---------------------------------------------------------------------------

(defn cell-addr
  "Trace address for cell (r, c). Keyword like `:r0-c1`."
  [r c]
  (keyword (str "r" r "-c" c)))

(defn label->index
  "Look up a label in `cell-types` (case-insensitive). Accepts string or map
   options. Returns -1 if not found."
  [cell-types label]
  (let [target (str/lower-case (or label ""))]
    (or (first (keep-indexed
                 (fn [i v]
                   (when (= (str/lower-case (option->label v)) target) i))
                 cell-types))
        -1)))

(defn labels->constraints
  "Build a choicemap-friendly map from a 2D label grid plus `cell-types`,
   suitable for passing to `p/generate`. Cells whose label is unrecognized
   (label->index returns -1) are simply omitted."
  [labels cell-types]
  (into {}
        (for [r (range (count labels))
              c (range (count (first labels)))
              :let [lbl (get-in labels [r c])
                    idx (label->index cell-types lbl)]
              :when (>= idx 0)]
          [(cell-addr r c) (mx/scalar idx mx/int32)])))

(defn make-grid-gf
  "Build a gen-fn over an N×M grid. Each cell at (r, c) is a categorical trace
   site at address `(cell-addr r c)` with `(count cell-types)` outcomes.

   Default prior is uniform (zero logits). Pass `prior-logits` — a 2D vector
   whose `[r][c]` is a length-K vector of logits — to use per-cell priors.

   The gen-fn returns the 2D grid of sampled indices (or constrained values
   when invoked via `p/generate`). Recover labels with
   `(map #(map (fn [i] (nth cell-types i)) %) grid)`."
  ([rows cols cell-types]
   (make-grid-gf rows cols cell-types nil))
  ([rows cols cell-types prior-logits]
   (let [n-types (count cell-types)
         uniform (mx/array (vec (repeat n-types 0.0)))]
     (dyn/auto-key
       (gen []
         (vec
          (for [r (range rows)]
            (vec
             (for [c (range cols)]
               (let [logits (if prior-logits
                              (mx/array (get-in prior-logits [r c]))
                              uniform)]
                 (trace (cell-addr r c) (dist/categorical logits))))))))))))

(defn idx-grid->labels
  "Convert a 2D vector of integer indices into a 2D vector of label strings,
   given `cell-types` (strings or `{:label :description}` maps). Accepts MLX
   scalars or plain JS numbers."
  [idx-grid cell-types]
  (vec
   (for [row idx-grid]
     (vec
      (for [v row]
        (let [i (if (number? v) v (try (mx/item v) (catch :default _ -1)))]
          (if (and (number? i) (<= 0 i) (< i (count cell-types)))
            (option->label (nth cell-types i))
            "?")))))))
