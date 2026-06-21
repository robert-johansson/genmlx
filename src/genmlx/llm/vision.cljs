(ns genmlx.llm.vision
  "VLM-as-generative-function: per-cell classification with structural composition.

   Two layers, one per concern:

   - **Async I/O** wraps the native `@genmlx/core` VL forward (`load` +
     `chatSessionStart` with image bytes per message). Used to load the VLM and
     classify individual cells. Returns promises.
   - **Sync GFI** wraps the per-cell categorical structure as a `gen` function.
     Each cell is a trace site at a keyword address; observations from the VLM
     become constraints when invoking via `p/generate`.

   The split follows the GenMLX async boundary convention: I/O is async,
   GFI math is sync. Run perception once via `classify-grid`, then feed the
   result into the gen-fn via constraints.

   See `examples/vlm_grid_gf.cljs` for the end-to-end demo.

   Operational note: each `classify-cell` runs a fresh turn-1 `chatSessionStart`,
   so conversation history does not accumulate across independent classifications.
   (The old @mlx-node/lm session path needed an explicit `session.reset()` per call
   to stop per-call latency growing monotonically with N — 0.6s rising to 116s by
   the 18th call; see
   `../genmlx-lab/dev/docs/INVESTIGATION_VLM_PER_CELL_PROBE.md`.)"
  (:require [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [promesa.core :as pr]
            [clojure.string :as str]
            ["fs" :as fs]))

;; Single @genmlx/core addon (js/require) — bean genmlx-qt34. Same reason as
;; genmlx.llm.backend: GenMLX rides on ONE MLX-linking .node addon; the former
;; @mlx-node/lm dependency pulled in a second core (upstream @mlx-node/core) that
;; SIGTRAPs alongside @genmlx/core. The VL-capable native classes (Qwen35Model /
;; Qwen35MoeModel / Gemma4Model) accept image bytes per chatSessionStart message.
(defonce ^:private mlx-core (js/require "@genmlx/core"))

;; ---------------------------------------------------------------------------
;; Async I/O layer
;; ---------------------------------------------------------------------------

(defn load-vlm
  "Load a vision-language model via the native @genmlx/core forward, dispatching on
   config.json model_type. Returns a promise of the native model instance
   (Qwen35Model / Qwen35MoeModel / Gemma4Model), whose `chatSessionStart` accepts
   image bytes per message. The checkpoint's safetensors must include the vision
   tower; verified target: Qwen3.6-35B-A3B-4bit (model_type qwen3_5_moe).

   NOTE (bean genmlx-qt34): migrated off @mlx-node/lm's generic loadSession to the
   single @genmlx/core addon. The VLM *generation* path (classify-cell) has no
   automated test — the verified VL checkpoint is a large MoE — so this I/O layer
   is exercised via the live demo (examples/vlm_grid_gf.cljs), not the test suite."
  [model-path]
  (let [mt (-> (.readFileSync fs (str model-path "/config.json") "utf8")
               js/JSON.parse
               .-model_type)]
    (if-let [cls (case mt
                   "qwen3_5"     (.-Qwen35Model mlx-core)
                   "qwen3_5_moe" (.-Qwen35MoeModel mlx-core)
                   "gemma4"      (.-Gemma4Model mlx-core)
                   nil)]
      (.load cls model-path)
      (throw (ex-info (str "genmlx.llm.vision/load-vlm: no native VL-capable "
                           "@genmlx/core model class for model_type " (pr-str mt))
                      {:genmlx/error :unsupported-vlm-type :model-type mt
                       :model-path model-path})))))

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

   `vlm` is a native model from `load-vlm`. Each call runs a fresh turn-1
   `chatSessionStart` (the image bytes ride in the user message), so history
   doesn't accumulate across independent calls — no explicit reset needed.
   Disables the model's reasoning trace and the default repetition penalty so
   short single-word answers come through cleanly."
  [vlm image-bytes options]
  (pr/let [opts-text (str/join "\n" (map option->line options))
           prompt (str "What is shown in this image? Answer with EXACTLY ONE of:\n"
                       opts-text
                       "\n\nOutput ONLY the single word, nothing else.")
           messages (clj->js [{:role "user" :content prompt :images [image-bytes]}])
           config #js {:maxNewTokens 8
                       :reasoningEffort "none"
                       :repetitionPenalty 1.0}
           result (.chatSessionStart vlm messages config)]
    ;; Keep the WHOLE answer (lowercased, alphanumerics + single spaces):
    ;; taking only the first word made multi-word labels ("fire truck")
    ;; unmatchable in label->index (genmlx-xwxh). label->index normalizes
    ;; cell-type labels the same way.
    (-> (or (.-text result) "")
        str/lower-case
        (str/replace #"[^a-z0-9 ]" "")
        (str/replace #"\s+" " ")
        str/trim)))

(defn classify-grid
  "Classify all cells of an N×M grid. Returns a promise of
   `{:rows N :cols M :labels <2D vector of strings>}`.

   `vlm` is a native model from `load-vlm`. `crop-fn` takes `[r c]` and returns a
   Uint8Array of cell image bytes (or a promise of one). `options` is the vector
   of allowed labels.

   Cells are processed strictly sequentially — the model's Metal kernels serialize
   anyway, so concurrency wouldn't help.

   `progress-fn` (optional) is called as `(progress-fn r c label dt-ms)` after
   each cell completes."
  ([vlm rows cols crop-fn options]
   (classify-grid vlm rows cols crop-fn options nil))
  ([vlm rows cols crop-fn options progress-fn]
   (let [coords (vec (for [r (range rows) c (range cols)] [r c]))]
     (pr/let
      [results (reduce
                (fn [acc-promise [r c]]
                  (pr/let [acc acc-promise
                           t0 (.now js/performance)
                           bytes (crop-fn r c)
                           label (classify-cell vlm bytes options)
                           dt (- (.now js/performance) t0)]
                    (when progress-fn (progress-fn r c label dt))
                    (conj acc [r c label])))
                (pr/resolved [])
                coords)]
       {:rows rows :cols cols
        :labels (mapv vec (partition cols (map peek results)))}))))

;; ---------------------------------------------------------------------------
;; Sync GFI layer
;; ---------------------------------------------------------------------------

(defn cell-addr
  "Trace address for cell (r, c). Keyword like `:r0-c1`."
  [r c]
  (keyword (str "r" r "-c" c)))

(defn- normalize-label
  "Shared normalization for VLM answers and cell-type labels: lowercase,
   strip everything but alphanumerics and spaces, collapse whitespace. Both
   sides of the match go through this, so \"Fire Truck!\" == \"fire truck\"."
  [s]
  (-> (or s "")
      str/lower-case
      (str/replace #"[^a-z0-9 ]" "")
      (str/replace #"\s+" " ")
      str/trim))

(defn- word-contained?
  "Whole-word containment in either direction between two normalized strings:
   \"a tree\" contains the label \"tree\"; the terse answer \"fire\" is
   contained in the label \"fire truck\"."
  [a b]
  (let [pa (str " " a " ") pb (str " " b " ")]
    (or (str/includes? pa pb) (str/includes? pb pa))))

(defn label->index
  "Look up a label in `cell-types` (case/punctuation/whitespace insensitive —
   see normalize-label). Accepts string or map options. Exact normalized
   match first; otherwise whole-word containment in either direction
   (answer \"a fire truck\" matches label \"fire truck\"; terse answer
   \"fire\" matches \"fire truck\"), first cell-type winning ties.
   Returns nil if not found."
  [cell-types label]
  (let [target (normalize-label label)
        norm   (mapv #(normalize-label (option->label %)) cell-types)]
    (when (seq target)
      (or (first (keep-indexed (fn [i v] (when (= v target) i)) norm))
          (first (keep-indexed
                   (fn [i v]
                     (when (and (seq v) (word-contained? target v)) i))
                   norm))))))

(defn labels->constraints
  "Build a PLAIN {addr -> mx int32 scalar} map from a 2D label grid plus
   `cell-types`. NOT itself a choicemap: wrap with cm/from-map (as the
   examples do) before passing to p/generate. Cells whose label is
   unrecognized (label->index returns nil) are simply omitted."
  [labels cell-types]
  (into {}
        (for [r (range (count labels))
              c (range (count (first labels)))
              :let [lbl (get-in labels [r c])
                    idx (label->index cell-types lbl)]
              :when idx]
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
