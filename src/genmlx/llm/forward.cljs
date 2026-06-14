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

(defn- detect-model-type [dir]
  (-> (.readFileSync fs (str dir "/config.json") "utf8")
      (js/JSON.parse)
      (.-model_type)))

(defn load-model
  "Load a checkpoint, dispatching on config.json model_type. Returns the family's
   {:config :weights ..} tagged with :impl so the other fns route correctly."
  [dir]
  (if (= "qwen3_5" (detect-model-type dir))
    (assoc (q35/load-model dir) :impl :qwen3_5)
    (assoc (q3/load-model dir)  :impl :qwen3)))

(defn- q35? [m] (= :qwen3_5 (:impl m)))

(defn forward            [m ids]            (if (q35? m) (q35/forward m ids)            (q3/forward m ids)))
(defn next-token-logits  [m ids]            (if (q35? m) (q35/next-token-logits m ids)  (q3/next-token-logits m ids)))
(defn init-cache         [m]                (if (q35? m) (q35/init-cache m)             (q3/init-cache m)))
(defn prefill            [m ids]            (if (q35? m) (q35/prefill m ids)            (q3/prefill m ids)))
(defn step               [m cache offset id] (if (q35? m) (q35/step m cache offset id)  (q3/step m cache offset id)))
