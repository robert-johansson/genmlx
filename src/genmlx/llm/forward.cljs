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
  (:require [genmlx.mlx :as mx]
            [genmlx.llm.qwen3-forward :as q3]
            [genmlx.llm.qwen35-forward :as q35]
            [genmlx.llm.qwen35-vision-forward :as vis]
            ["fs" :as fs]))

(def supported-model-types
  "config.json model_type strings the GenMLX-owned forward actually implements.
   Anything else must use the upstream forward (load-model {:cljs-forward? false}).
   qwen3_5_moe (Ornith, Qwen3.6-A3B — genmlx-g6vk) rides the qwen3_5 hybrid
   forward with a per-layer sparse-MoE MLP branch."
  #{"qwen3" "qwen3_5" "qwen3_5_moe"})

(defn- detect-model-type [dir]
  (-> (.readFileSync fs (str dir "/config.json") "utf8")
      (js/JSON.parse)
      (.-model_type)))

(defn loadable-weights?
  "True if the owned loader can read this checkpoint's weights: either a
   single `model.safetensors`, or the HuggingFace sharded layout
   (model-0000N-of-… + model.safetensors.index.json), which q3/load-weights
   resolves via the index's weight_map and merges shard-by-shard
   (genmlx-sbif; the single-file-only era was genmlx-o94r)."
  [dir]
  (or (.existsSync fs (str dir "/model.safetensors"))
      (.existsSync fs (str dir "/model.safetensors.index.json"))))

(defn supported?
  "True if the owned forward implements this checkpoint's config.json model_type
   AND its loader can read the weights (single-file or sharded — see
   loadable-weights?) AND any declared quantization is one the owned loader can
   dequantize at load (q3/dequantizable? — affine 2/4/8-bit, globally and for
   every per-tensor override; exotic or odd-bit schemes fall back to the
   upstream forward, which drives the native quantized kernels).
   backend/load-model's smart default uses this: the owned forward only when
   ALL hold, the upstream forward otherwise."
  [dir]
  (and (contains? supported-model-types (detect-model-type dir))
       (loadable-weights? dir)
       (let [qz (q3/load-quantization dir)]
         (or (nil? qz) (q3/dequantizable? qz)))))

(defn load-model
  "Load a checkpoint, dispatching on config.json model_type. Returns the family's
   {:config :weights ..} tagged with :impl so the other fns route correctly,
   plus :dir and — for qwen3.5-family VLM checkpoints — :vcfg (the parsed
   vision_config; nil for text-only checkpoints), so the backend can route an
   image-bearing forward-prefill to the owned VLM prefill (genmlx-jq6l; the
   tower weights are in :weights already — the loader reads every tensor).
   Throws on a model_type the owned forward does not implement, instead of
   silently mis-routing it to the vanilla-Qwen3 forward."
  [dir]
  (let [mt (detect-model-type dir)]
    (case mt
      "qwen3_5"     (assoc (q35/load-model dir) :impl :qwen3_5 :dir dir
                           :vcfg (vis/load-vision-config dir))
      "qwen3_5_moe" (assoc (q35/load-model dir) :impl :qwen3_5 :dir dir
                           :vcfg (vis/load-vision-config dir))
      "qwen3"       (assoc (q3/load-model dir)  :impl :qwen3 :dir dir)
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

;; --- [K]-particle batch axis (genmlx-9uyg) ---------------------------------

(defn step-batched
  "Advance K lockstep lanes one token each: `tok` is a [K]-shaped int array,
   `cache` a [K …]-tiled per-layer cache. Returns [logits [K vocab] cache']."
  [m cache offset tok]
  (if (q35? m) (q35/step-batched m cache offset tok) (q3/step-batched m cache offset tok)))

(defn prefill-batched
  "K equal-length prompts through ONE [B T] forward over a fresh cache.
   Returns [last-logits [B vocab] cache]. The particle path itself prefills
   at B=1 and tiles with broadcast-cache; this entry exists for genuinely
   different prompts (and the batch-independence gate)."
  [m prompts]
  (if (q35? m) (q35/prefill-batched m prompts) (q3/prefill-batched m prompts)))

(defn broadcast-cache
  "Tile a B=1 per-layer cache to B=k for lockstep [K]-lane decode. Family-
   agnostic: every entry is a map of arrays with leading batch dim 1
   ({:k :v} full-attn / {:conv :rec} GDN), and mx/broadcast-to over immutable
   MxArray values shares — never copies or aliases-mutably — the prefix, so
   the B=1 original stays live and valid (the same persistent-value property
   the owned branch ledger rides, genmlx-7f93)."
  [cache k]
  (mapv (fn [ce]
          (when ce
            (reduce-kv
             (fn [m' kk arr]
               (let [sh (mx/shape arr)]
                 (when-not (= 1 (first sh))
                   (throw (ex-info (str "broadcast-cache: entry " kk
                                        " has leading dim " (first sh)
                                        " — expected a B=1 cache")
                                   {:key kk :shape (vec sh)})))
                 (assoc m' kk (mx/broadcast-to arr (into [k] (rest sh))))))
             {} ce)))
        cache))
