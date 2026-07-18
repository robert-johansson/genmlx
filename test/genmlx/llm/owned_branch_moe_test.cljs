;; @tier exclude — loads Ornith-1.0-35B-4bit (~20 GB resident). Run manually:
;;   bunx --bun nbb@1.4.208 test/genmlx/llm_owned_branch_moe_test.cljs
;; GENMLX_OWNED_MOE_MODEL overrides the checkpoint dir. Keep a MemAvailable
;; watchdog beside it on this box (genmlx-h3p5).
(ns genmlx.llm.owned-branch-moe-test
  "genmlx-7f93 GATE: token-SMC on the OWNED Ornith-35B MoE — the two headline
   capabilities (token-SMC + owned forward) meeting in one place. The owned
   qwen3_5_moe forward now exposes the branch surface (persistent-value cache
   forks), so the filter's particles are O(1) branch handles instead of the
   O(T)/step replay fallback.

   Gates (logit/weight-level only — the quantized MoE expert path jitters
   in situ, genmlx-ba06/cnhi, so sampled TEXT is never asserted):
     M1 owned load (smart default) + supports-branching? true
     M2 forced-token weight parity: per-site SMC weight increments (grammar
        mask log-normalizers) along ONE token path, branch decoder vs the
        replay-decoder fallback (uncached owned forward) — median within the
        repeated-assess jitter band, max within its tail
     M3 token-SMC (5qk7 V5 shape) runs on the owned 35B: N=4 digit-grammar
        particles, all outputs grammar-valid, finite log-ML, R1 <= N+1,
        R2 no leak"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.grammar :as gram]
            [genmlx.llm.smc :as tsmc]
            [promesa.core :as pr]
            ["fs" :as fs]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

(def model-dir
  (or (some-> js/process .-env .-GENMLX_OWNED_MOE_MODEL)
      (let [base (str (.-HOME js/process.env)
                      "/.cache/huggingface/hub/models--mlx-community--Ornith-1.0-35B-4bit/snapshots")]
        (when (.existsSync fs base)
          (str base "/" (first (js->clj (.readdirSync fs base))))))))

(defn- mat [a] (mx/materialize! a) a)
(defn- lse [logits] (mx/realize (mx/logsumexp logits)))
(defn- median [xs]
  (let [s (vec (sort xs)) n (count s)]
    (cond (zero? n) 0.0
          (odd? n) (nth s (quot n 2))
          :else (/ (+ (nth s (dec (quot n 2))) (nth s (quot n 2))) 2.0))))

(defn- gmask [c dfa logits] (gram/apply-mask c dfa logits))
(defn- gadvance [c dfa tok-id]
  (if (= tok-id (:eos-id c))
    dfa
    (gram/dfa-advance-string (:dfa c) dfa (nth (:token-index c) tok-id ""))))

(defn- summary []
  (println (str "\n== llm-owned-branch-moe: " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

(if-not (and model-dir (.existsSync fs model-dir))
  (do (println "SKIP llm-owned-branch-moe — Ornith-1.0-35B-4bit not cached") (summary))
  (->
   (pr/let [mm (llm/load-model model-dir)   ; smart default => OWNED qwen3_5_moe
            {:keys [model tokenizer]} mm
            constraint (gram/compile-constraint tokenizer "[0-9]{3}-[0-9]{4}")
            enc (llm/encode tokenizer "My phone number is ")]
     (mx/force-gc!)                          ; trim the dequant transient
     (let [prompt (vec enc)]
       (println "== owned Ornith-35B branch surface (" (count prompt) "-token prompt)")

       ;; ---- M1 ----
       (assert-true "M1: smart default loads the OWNED forward" (llm/cljs-forward-model? model))
       (assert-true "M1: supports-branching? => true on the owned MoE"
                    (llm/supports-branching? model))

       ;; ---- M2: forced-token weight parity, branch vs replay ----
       ;; One greedy-masked path from the BRANCH decoder; the replay decoder
       ;; is then FORCED through the same tokens (never compare two sampled
       ;; paths on a jittering model).
       (let [steps 7
             bwalk (let [d (tsmc/native-decoder model)
                         {:keys [root logits]} (tsmc/dec-prefill! d prompt)
                         h (tsmc/dec-fork! d root)]
                     (loop [i 0, lg logits, dfa (:start (:dfa constraint)), out []]
                       (if (>= i steps)
                         (do (tsmc/dec-dispose! d h) (tsmc/dec-dispose! d root) out)
                         (let [masked (gmask constraint dfa lg)
                               inc-w (- (lse masked) (lse lg))
                               tok (mx/item (mx/argmax masked))
                               lg' (mat (tsmc/dec-step! d h tok))]
                           (recur (inc i) lg' (gadvance constraint dfa tok)
                                  (conj out {:tok tok :inc-w inc-w}))))))
             toks (mapv :tok bwalk)
             rwalk (let [d (tsmc/replay-decoder model)
                         {:keys [root logits]} (tsmc/dec-prefill! d prompt)
                         h (tsmc/dec-fork! d root)]
                     (loop [i 0, lg logits, dfa (:start (:dfa constraint)), out []]
                       (if (>= i steps)
                         (do (tsmc/dec-dispose! d h) (tsmc/dec-dispose! d root) out)
                         (let [masked (gmask constraint dfa lg)
                               inc-w (- (lse masked) (lse lg))
                               tok (nth toks i)   ; FORCED to the branch path
                               lg' (mat (tsmc/dec-step! d h tok))]
                           (recur (inc i) lg' (gadvance constraint dfa tok)
                                  (conj out {:inc-w inc-w}))))))
             ds (mapv #(js/Math.abs (- (:inc-w %1) (:inc-w %2))) bwalk rwalk)
             dW (js/Math.abs (- (reduce + (map :inc-w bwalk))
                                (reduce + (map :inc-w rwalk))))
             md (median ds) mx- (reduce max 0 ds)]
         (println (str "    [info] per-site |Δinc-w|: median " (.toFixed md 4)
                       " max " (.toFixed mx- 4) " | total |ΔW| " (.toFixed dW 4)
                       " over " steps " sites"))
         ;; Bands: repeated-assess on identical choices spreads 0.125-0.625
         ;; nats on this MoE (cnhi — kernel-level gather_mm nondeterminism);
         ;; each increment is an lse difference on the SAME quantized expert
         ;; path both sides, so the median sits inside that band and the max
         ;; inside its tail.
         (assert-true (str "M2: median per-site weight increment within jitter band ("
                           (.toFixed md 4) " < 0.4)")
                      (< md 0.4))
         (assert-true (str "M2: max per-site weight increment within jitter tail ("
                           (.toFixed mx- 4) " < 1.5)")
                      (< mx- 1.5)))

       ;; ---- M3: token-SMC on the owned 35B ----
       (let [decoder (tsmc/native-decoder model)
             max-live (atom 0)
             t0 (js/Date.now)
             r0 (tsmc/token-smc
                 {:particles 4 :max-tokens 10
                  :eos-id (llm/eos-token-id tokenizer)
                  :proposal :grammar-masked :constraint constraint
                  :decoder decoder :key (rng/fresh-key 42)
                  :callback (fn [_] (swap! max-live max (tsmc/live-handles decoder)))}
                 mm prompt)]
         (pr/let [r (tsmc/decode-particles! mm r0)]
           (let [secs (/ (- (js/Date.now) t0) 1000.0)
                 texts (mapv :text (:particles r))]
             (println (str "    [info] " (.toFixed secs 1) "s; texts: " (pr-str texts)))
             (assert-true "M3: filter completed with 4 particles"
                          (= 4 (count (:particles r))))
             (assert-true "M3: all outputs grammar-valid on the owned 35B"
                          (every? #(re-matches #"[0-9]{3}-[0-9]{4}" (or % "")) texts))
             (assert-true "M3: finite log-ML"
                          (js/isFinite (mx/realize (:log-ml-estimate r))))
             (assert-true (str "M3: R1 bounded (" @max-live " <= 5)") (<= @max-live 5))
             (assert-true "M3: R2 no leak after return" (zero? (tsmc/live-handles decoder)))
             (summary))))))
   (pr/catch (fn [e]
               (swap! fail inc)
               (println "  FAIL (uncaught)" (or (.-message e) e))
               (summary)))))
