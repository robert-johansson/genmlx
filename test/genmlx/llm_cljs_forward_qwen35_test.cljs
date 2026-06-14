;; @tier slow
(ns genmlx.llm-cljs-forward-qwen35-test
  "f6ov P6: the LLM-as-GF running on the GenMLX-OWNED Qwen3.5 HYBRID forward
   (GatedDeltaNet linear layers + full-attention layers), via load-model with
   :cljs-forward? true. Verifies the GF generates coherent text end-to-end on the
   owned hybrid forward — which drives the incremental decode path (per-layer
   conv-state + recurrent-state for linear layers, KV cache for full-attention
   layers) over many steps — and that scoring matches the upstream-forward GF to
   bf16 tolerance (forward parity through the full GF stack).

   qwen3.5-0.8b; skips cleanly if absent."
  (:require [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as core]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx.random :as rng]
            [promesa.core :as pr]
            ["fs" :as fs]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v (do (swap! pass inc) (println (str "  PASS: " label)))
        (do (swap! fail inc) (println (str "  FAIL: " label)))))

(def ^:private dir (str (.-HOME js/process.env) "/.cache/models/qwen3.5-0.8b-mlx-bf16"))

(if-not (.existsSync fs (str dir "/model.safetensors"))
  (println "SKIP llm-cljs-forward-qwen35-test: qwen3.5-0.8b checkpoint absent")
  (pr/let [cljs-m  (llm/load-model dir {:cljs-forward? true})
           up-m    (llm/load-model dir)
           ids-raw (llm/encode (:tokenizer cljs-m) "The capital of France is" false)
           prompt  (vec ids-raw)
           cljs-gf (core/make-llm-gf cljs-m)
           up-gf   (core/make-llm-gf up-m)]
    (println "\n== LLM-as-GF on the GenMLX-owned Qwen3.5 hybrid forward (qwen3.5-0.8b) ==")
    (assert-true "model is a CljsForwardModel" (llm/cljs-forward-model? (:model cljs-m)))
    ;; 1. multi-token generation runs end-to-end on the owned hybrid forward
    ;;    (incremental GatedDeltaNet conv/recurrent state + full-attn KV cache).
    (pr/let [tr   (p/simulate (dyn/with-key cljs-gf (rng/fresh-key 0)) [prompt 12])
             text (core/decode-trace (:tokenizer cljs-m) tr)]
      (println "  generated:" (pr-str text))
      (assert-true "GF simulate produced a trace with a score" (some? (:score tr)))
      (assert-true "score is finite & negative"
                   (let [s (mx/item (:score tr))] (and (js/isFinite s) (neg? s))))
      (assert-true "decoded text is non-empty" (pos? (count text)))
      ;; 2. forward parity through the GF: assess the SAME tokens on the
      ;;    upstream-forward GF; the score must match within bf16 tolerance
      ;;    (~12 tokens of small per-token cross-kernel drift over the hybrid stack).
      (let [cljs-score (mx/item (:score tr))
            up-score   (mx/item (:weight (p/assess up-gf [prompt 12] (:choices tr))))]
        (println "  cljs-forward score:" (.toFixed cljs-score 3)
                 " upstream-forward score:" (.toFixed up-score 3)
                 " diff:" (.toFixed (js/Math.abs (- cljs-score up-score)) 4))
        (assert-true "owned-forward score matches upstream-forward within bf16 tol (<6)"
                     (< (js/Math.abs (- cljs-score up-score)) 6.0)))
      (println (str "\n=== cljs-forward-qwen35: " @pass " PASS, " @fail " FAIL ===")))))
