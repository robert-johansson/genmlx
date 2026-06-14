;; @tier slow
(ns genmlx.llm-cljs-forward-test
  "f6ov P4: the LLM-as-GF running on the GenMLX-OWNED forward (load-model with
   :cljs-forward? true), decoupled from upstream's model structs. Verifies the
   GF generates coherent text on the owned forward and that scoring matches the
   upstream-forward GF to bf16 tolerance (forward parity through the full GF).
   Vanilla Qwen3 (qwen3-0.6b); skips cleanly if absent."
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

(def ^:private dir (str (.-HOME js/process.env) "/.cache/models/qwen3-0.6b-mlx-bf16"))

(if-not (.existsSync fs (str dir "/model.safetensors"))
  (println "SKIP llm-cljs-forward-test: qwen3-0.6b checkpoint absent")
  (pr/let [cljs-m  (llm/load-model dir {:cljs-forward? true})
           up-m    (llm/load-model dir)
           ids-raw (llm/encode (:tokenizer cljs-m) "The capital of France is" false)
           prompt  (vec ids-raw)
           cljs-gf (core/make-llm-gf cljs-m)
           up-gf   (core/make-llm-gf up-m)]
    (println "\n== LLM-as-GF on the GenMLX-owned forward (qwen3-0.6b) ==")
    (assert-true "model is a CljsForwardModel" (llm/cljs-forward-model? (:model cljs-m)))
    ;; 1. generation runs end-to-end on the owned forward
    (pr/let [tr   (p/simulate (dyn/with-key cljs-gf (rng/fresh-key 0)) [prompt 12])
             text (core/decode-trace (:tokenizer cljs-m) tr)]
      (println "  generated:" (pr-str text))
      (assert-true "GF simulate produced a trace with a score" (some? (:score tr)))
      (assert-true "score is finite & negative"
                   (let [s (mx/item (:score tr))] (and (js/isFinite s) (neg? s))))
      (assert-true "decoded text is non-empty" (pos? (count text)))
      ;; 2. forward parity through the GF: assess the SAME tokens on the
      ;;    upstream-forward GF; the score must match within bf16 tolerance
      ;;    (~12 tokens of small per-token cross-kernel drift).
      (let [cljs-score (mx/item (:score tr))
            up-score   (mx/item (:weight (p/assess up-gf [prompt 12] (:choices tr))))]
        (println "  cljs-forward score:" (.toFixed cljs-score 3)
                 " upstream-forward score:" (.toFixed up-score 3)
                 " diff:" (.toFixed (js/Math.abs (- cljs-score up-score)) 4))
        (assert-true "owned-forward score matches upstream-forward within bf16 tol (<4)"
                     (< (js/Math.abs (- cljs-score up-score)) 4.0)))
      (println (str "\n=== cljs-forward: " @pass " PASS, " @fail " FAIL ===")))))
