(ns genmlx.llm.forward
  "f6ov: model-family dispatch for the GenMLX-owned LLM forward. Reads
   config.json's `model_type` and routes to the vanilla Qwen3 forward
   (genmlx.llm.qwen3-forward) or the Qwen3.5 hybrid GatedDeltaNet forward
   (genmlx.llm.qwen35-forward). Both families expose the same 6-fn interface
   (load-model / forward / next-token-logits / prefill / step / init-cache), so
   CljsForwardModel in backend.cljs drives whichever family the checkpoint
   declares with no per-family branching of its own.

   This keeps the proven vanilla-Qwen3 path (qwen3-forward) untouched: the only
   backend change is requiring this façade instead of qwen3-forward directly."
  (:require [genmlx.llm.qwen3-forward :as q3]
            [genmlx.llm.qwen35-forward :as q35]
            ["fs" :as fs]))

(def supported-model-types
  "config.json model_type strings the GenMLX-owned forward actually implements.
   Anything else must use the upstream forward (load-model {:cljs-forward? false})."
  #{"qwen3" "qwen3_5"})

(defn- detect-model-type [dir]
  (-> (.readFileSync fs (str dir "/config.json") "utf8")
      (js/JSON.parse)
      (.-model_type)))

(defn loadable-weights?
  "True if the owned single-file loader can read this checkpoint's weights — i.e.
   a single `model.safetensors` exists. The owned loader (q3/q35 load-model ->
   mx/load-safetensors) does NOT yet read HuggingFace sharded / index.json
   layouts (model.safetensors-0000N-of-... + model.safetensors.index.json),
   tracked by genmlx-o94r. Until then such checkpoints must use the upstream
   loader even when their model_type is owned-supported."
  [dir]
  (.existsSync fs (str dir "/model.safetensors")))

(defn supported?
  "True if the owned forward implements this checkpoint's config.json model_type
   AND its loader can read the weights (a single model.safetensors — see
   loadable-weights?). backend/load-model's smart default uses this: the owned
   forward only when BOTH hold, the upstream forward otherwise (so a supported
   family with a sharded/index.json checkpoint safely falls back rather than
   erroring, and auto-upgrades once the owned loader learns the layout, o94r)."
  [dir]
  (and (contains? supported-model-types (detect-model-type dir))
       (loadable-weights? dir)))

(defn load-model
  "Load a checkpoint, dispatching on config.json model_type. Returns the family's
   {:config :weights ..} tagged with :impl so the other fns route correctly.
   Throws on a model_type the owned forward does not implement, instead of
   silently mis-routing it to the vanilla-Qwen3 forward."
  [dir]
  (let [mt (detect-model-type dir)]
    (case mt
      "qwen3_5" (assoc (q35/load-model dir) :impl :qwen3_5)
      "qwen3"   (assoc (q3/load-model dir)  :impl :qwen3)
      (throw (ex-info (str "genmlx.llm.forward: the GenMLX-owned forward does not "
                           "implement model_type " (pr-str mt) "; load with "
                           "{:cljs-forward? false} to use the upstream forward.")
                      {:model-type mt :dir dir :supported supported-model-types})))))

(defn- q35? [m] (= :qwen3_5 (:impl m)))

(defn forward            [m ids]            (if (q35? m) (q35/forward m ids)            (q3/forward m ids)))
(defn next-token-logits  [m ids]            (if (q35? m) (q35/next-token-logits m ids)  (q3/next-token-logits m ids)))
(defn init-cache         [m]                (if (q35? m) (q35/init-cache m)             (q3/init-cache m)))
(defn prefill            [m ids]            (if (q35? m) (q35/prefill m ids)            (q3/prefill m ids)))
(defn step               [m cache offset id] (if (q35? m) (q35/step m cache offset id)  (q3/step m cache offset id)))
