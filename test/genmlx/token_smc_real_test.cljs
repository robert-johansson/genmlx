;; @tier slow
(ns genmlx.token-smc-real-test
  "genmlx-5qk7: token-SMC real-model smokes.

   V5 — dense 0.8B (gated on ~/.cache/models/qwen3.5-0.8b-mlx-bf16):
        grammar-twisted generation (\\d{3}-\\d{4} style): all N outputs match
        the grammar, finite log-ML, sane ESS trajectory; runtime + decoder
        kind printed. Uses the native decoder when the model exposes the
        branch surface, the correct-but-O(T) replay decoder otherwise (the
        documented R3 asymmetry).
   V6 — 80B MoE (gated on GENMLX_MOE_MODEL): N=4, short completion, R1/R2
        hold on the NATIVE branch surface, no abort.

   Run: bunx --bun nbb@1.4.208 test/genmlx/token_smc_real_test.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.grammar :as gram]
            [genmlx.llm.smc :as tsmc]
            [promesa.core :as pr]
            [clojure.string :as str]
            ["fs" :as fs]
            ["os" :as os]
            ["path" :as path]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

;; A GENUINELY dense bf16 checkpoint. NOTE: qwen3.5-0.8b-mlx-bf16 is 4-bit
;; quantized despite its name (see the x76x root-cause) and the CLJS forward
;; path cannot read packed embeddings (genmlx-vmks), so V5 rides the 0.6B.
(def dense-dir
  (let [cands [(path/join (os/homedir) ".cache" "models" "qwen3-0.6b-mlx-bf16")
               (path/join (os/homedir) ".cache" "models" "qwen3-0.6b")]]
    (or (first (filter #(.existsSync fs (path/join % "tokenizer.json")) cands))
        (first cands))))
(def moe-dir (some-> js/process .-env .-GENMLX_MOE_MODEL))

(defn- summary []
  (println (str "\n== token-smc-real: " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

(defn- v5-dense []
  (if-not (.existsSync fs (path/join dense-dir "tokenizer.json"))
    (do (println "  SKIP V5 — no dense model at" dense-dir) (pr/resolved nil))
    (pr/let [mm (llm/load-model dense-dir)
             {:keys [model tokenizer]} mm
             enc (llm/encode tokenizer "My phone number is ")]
      (pr/let [constraint (gram/compile-constraint tokenizer "[0-9]{3}-[0-9]{4}")
            decoder (tsmc/decoder-for mm)
            native? (llm/supports-branching? model)
            prompt (vec enc)
            t0 (js/Date.now)
            max-live (atom 0)
            r0 (tsmc/token-smc {:particles 4 :max-tokens 10
                               :eos-id (llm/eos-token-id tokenizer)
                               :proposal :grammar-masked :constraint constraint
                               :decoder decoder :key (rng/fresh-key 42)
                               :callback (fn [_] (swap! max-live max (tsmc/live-handles decoder)))}
                              mm prompt)
            r (tsmc/decode-particles! mm r0)
            secs (/ (- (js/Date.now) t0) 1000.0)
            texts (mapv :text (:particles r))]
        (println (str "    V5 decoder=" (if native? "native" "replay")
                      " " (.toFixed secs 1) "s; texts: " (pr-str texts)))
        (assert-true "V5: all N outputs match the grammar (digit prefix)"
                     (every? #(re-matches #"[0-9]{3}-[0-9]{4}" (or % "")) texts))
        (assert-true "V5: finite log-ML"
                     (js/isFinite (mx/realize (:log-ml-estimate r))))
        (assert-true "V5: sane ESS trajectory (every entry in (0, N])"
                     (every? #(and (pos? %) (<= % 4.0001)) (:ess-trajectory r)))
        (assert-true (str "V5: R1 bounded (" @max-live " <= 5)") (<= @max-live 5))
        (assert-true "V5: R2 no leak after return" (zero? (tsmc/live-handles decoder)))))))

(defn- v6-moe []
  (if-not (and moe-dir (.existsSync fs moe-dir))
    (do (println "  SKIP V6 — GENMLX_MOE_MODEL not set / missing") (pr/resolved nil))
    (pr/let [mm (llm/load-model moe-dir)
             {:keys [model tokenizer]} mm
             enc (llm/encode tokenizer "# Returns the ")]
      (if-not (llm/supports-branching? model)
        (do (println "  SKIP V6 — model lacks the native branch surface") nil)
        (pr/let [decoder (tsmc/native-decoder model)
              prompt (vec enc)
              max-live (atom 0)
              r0 (tsmc/token-smc {:particles 4 :max-tokens 8
                                 :eos-id (llm/eos-token-id tokenizer)
                                 :decoder decoder :key (rng/fresh-key 7)
                                 :callback (fn [_] (swap! max-live max (tsmc/live-handles decoder)))}
                                mm prompt)
              r (tsmc/decode-particles! mm r0)]
          (println (str "    V6 texts: " (pr-str (mapv :text (:particles r)))))
          (assert-true "V6: filter completed on the 80B (no abort), 4 particles"
                       (= 4 (count (:particles r))))
          (assert-true (str "V6: R1 bounded on the native surface (" @max-live " <= 5)")
                       (<= @max-live 5))
          (assert-true "V6: R2 no leak" (zero? (tsmc/live-handles decoder))))))))

(-> (pr/do (v5-dense) (v6-moe))
    (pr/then (fn [_] (summary)))
    (pr/catch (fn [e]
                (swap! fail inc)
                (println "  FAIL (uncaught)" (.-message e))
                (summary))))
