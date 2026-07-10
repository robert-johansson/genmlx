;; @tier slow
(ns genmlx.llm-lanes-smc-test
  "genmlx-k7nj: batched-lane token-SMC — K particles ride the lane axis of
   ONE [K]-batched owned branch (run-filter-lanes), one lockstep forward per
   round, resample = one lane gather. Gates on the dense 0.6b (deterministic
   forward on this box):

     L1 :model structural: K particles, tokens bounded, log-ML exactly 0
        (no twist, p == q), ESS pinned at K, no resample
     L2 weight replay-equivalence (THE sweep gate): with resampling disabled,
        every particle's reported log-w equals the scalar-replay recomputation
        of the grammar-mask log-normalizer sum over its token sequence —
        a broken lane/cache ancestry diverges by tens of nats
     L3 forced-resample coherence: ess-threshold 1.01 resamples on every
        weight-diverging round; run completes, every decoded text matches the
        grammar end-to-end (post-resample lanes stay coherent), ledger empty
     L4 guard matrix: :twist / :decoder / fn-proposal refused with typed
        errors; T=0 returns prompt-only particles and log-ML 0
     L5 engine parity: lanes vs per-branch log-ML on the same tight grammar
        within a stochastic band; wall-clock printed for both
     L6 (gated on GENMLX_MOE_MODEL): the 35B MoE wall-clock gate — K=8
        lanes sweep vs per-branch on the big model, where the forward
        dominates and the lane axis pays off; asserts a strict speedup

   Run: bunx --bun nbb@1.4.208 test/genmlx/llm_lanes_smc_test.cljs"
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

(def dense-dir
  (let [cands [(path/join (os/homedir) ".cache" "models" "qwen3-0.6b-mlx-bf16")
               (path/join (os/homedir) ".cache" "models" "qwen3-0.6b")]]
    (or (first (filter #(.existsSync fs (path/join % "tokenizer.json")) cands))
        (first cands))))

(defn- summary []
  (println (str "\n== llm-lanes-smc: " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

(defn- branch-count [model] (count (:branches @(:branches model))))

(defn- mat [a] (mx/materialize! a) a)

(defn- replay-weight
  "Scalar-replay recomputation of a particle's grammar-mask weight:
   Σ_t [lse(masked_t) − lse(raw_t)] along its token sequence, on the owned
   SCALAR path — the independent oracle for the lane engine's weights."
  [model constraint eos-id prompt toks]
  (llm/init-cache! model)
  (try
    (loop [logits (mat (llm/forward-prefill model (vec prompt)))
           dfa (:start (:dfa constraint))
           ts (seq toks)
           w 0.0]
      (if-not ts
        w
        (let [masked (gram/apply-mask constraint dfa logits)
              lse-m (mx/realize (mx/logsumexp masked))
              lse-r (mx/realize (mx/logsumexp logits))
              tok (long (first ts))
              dfa' (if (= tok eos-id)
                     dfa
                     (gram/dfa-advance-string (:dfa constraint) dfa
                                              (nth (:token-index constraint) tok "")))
              w' (+ w (- lse-m lse-r))]
          (if (next ts)
            (recur (mat (llm/forward-step model tok)) dfa' (next ts) w')
            w'))))
    (finally (llm/reset-cache! model))))

(defn- thrown-kind [f]
  (try (f) nil (catch :default e (:genmlx/error (ex-data e)))))

(defn- run-gates []
  (pr/let [mm (llm/load-model dense-dir)
           {:keys [model tokenizer]} mm
           enc (llm/encode tokenizer "My phone number is ")
           constraint (gram/compile-constraint tokenizer "[0-9]{3}-[0-9]{4}")]
    (let [prompt (vec enc)
          eos (llm/eos-token-id tokenizer)]
      (if-not (llm/supports-branching? model)
        (println "  SKIP — model lacks the owned branch surface")
        (do
          ;; ---- L1: :model structural ----
          (println "\n-- L1 :model structural --")
          (let [r (tsmc/token-smc {:particles 4 :max-tokens 6 :eos-id eos
                                   :lanes? true :proposal :model
                                   :key (rng/fresh-key 11)}
                                  mm prompt)]
            (assert-true "L1: 4 particles" (= 4 (count (:particles r))))
            (assert-true "L1: tokens bounded by max-tokens"
                         (every? #(<= (count (:tokens %)) 6) (:particles r)))
            (assert-true "L1: log-ML exactly 0 (p == q, no twist)"
                         (< (js/Math.abs (mx/realize (:log-ml-estimate r))) 1e-4))
            (assert-true "L1: ESS pinned at K (uniform weights, no resample)"
                         (every? #(> % 3.999) (:ess-trajectory r)))
            (assert-true "L1: ledger empty after return" (zero? (branch-count model))))

          ;; ---- L2: weight replay-equivalence (resampling disabled) ----
          (println "\n-- L2 weight replay-equivalence --")
          (let [r (tsmc/token-smc {:particles 4 :max-tokens 10 :eos-id eos
                                   :lanes? true :proposal :grammar-masked
                                   :constraint constraint :ess-threshold 0.0
                                   :key (rng/fresh-key 42)}
                                  mm prompt)
                diffs (mapv (fn [pt]
                              (js/Math.abs
                               (- (mx/realize (:log-w pt))
                                  (replay-weight model constraint eos prompt
                                                 (:tokens pt)))))
                            (:particles r))]
            (println (str "    replay |Δw| per particle: "
                          (pr-str (mapv #(.toFixed % 4) diffs))))
            (assert-true "L2: every particle's weight matches scalar replay (< 1.0 nat; broken ancestry = tens of nats)"
                         (every? #(< % 1.0) diffs))
            (assert-true "L2: finite log-ML"
                         (js/isFinite (mx/realize (:log-ml-estimate r))))
            (assert-true "L2: ledger empty after return" (zero? (branch-count model))))

          ;; ---- L3: forced-resample coherence ----
          (println "\n-- L3 forced-resample coherence --")
          (pr/let [r0 (pr/resolved
                       (tsmc/token-smc {:particles 4 :max-tokens 10 :eos-id eos
                                        :lanes? true :proposal :grammar-masked
                                        :constraint constraint :ess-threshold 1.01
                                        :key (rng/fresh-key 7)}
                                       mm prompt))
                   r (tsmc/decode-particles! mm r0)]
            (let [texts (mapv :text (:particles r))]
              (println (str "    texts: " (pr-str texts)))
              (assert-true "L3: every output grammar-valid through resamples"
                           (every? #(re-matches #"[0-9]{3}-[0-9]{4}" (or % "")) texts))
              (assert-true "L3: finite log-ML"
                           (js/isFinite (mx/realize (:log-ml-estimate r))))
              (assert-true "L3: ledger empty after return" (zero? (branch-count model)))

              ;; ---- L4: guard matrix + T=0 ----
              (println "\n-- L4 guards --")
              (assert-true "L4: :twist refused"
                           (= :lanes-twist-unsupported
                              (thrown-kind #(tsmc/token-smc {:particles 2 :max-tokens 2
                                                             :lanes? true
                                                             :twist (fn [_ _] 0.0)}
                                                            mm prompt))))
              (assert-true "L4: :decoder refused"
                           (= :lanes-decoder-unsupported
                              (thrown-kind #(tsmc/token-smc {:particles 2 :max-tokens 2
                                                             :lanes? true
                                                             :decoder (tsmc/decoder-for mm)}
                                                            mm prompt))))
              (assert-true "L4: fn proposal refused"
                           (= :lanes-proposal-unsupported
                              (thrown-kind #(tsmc/token-smc {:particles 2 :max-tokens 2
                                                             :lanes? true
                                                             :proposal (fn [_ lg] lg)}
                                                            mm prompt))))
              (let [r0 (tsmc/token-smc {:particles 3 :max-tokens 0 :eos-id eos
                                        :lanes? true :proposal :model}
                                       mm prompt)]
                (assert-true "L4: T=0 → prompt-only particles, log-ML 0"
                             (and (= 3 (count (:particles r0)))
                                  (every? #(empty? (:tokens %)) (:particles r0))
                                  (zero? (mx/realize (:log-ml-estimate r0)))
                                  (zero? (branch-count model)))))

              ;; ---- L5: engine parity (lanes vs per-branch) ----
              (println "\n-- L5 engine parity --")
              (let [t0 (js/Date.now)
                    rl (tsmc/token-smc {:particles 8 :max-tokens 10 :eos-id eos
                                        :lanes? true :proposal :grammar-masked
                                        :constraint constraint
                                        :key (rng/fresh-key 123)}
                                       mm prompt)
                    t1 (js/Date.now)
                    rb (tsmc/token-smc {:particles 8 :max-tokens 10 :eos-id eos
                                        :proposal :grammar-masked
                                        :constraint constraint
                                        :key (rng/fresh-key 123)}
                                       mm prompt)
                    t2 (js/Date.now)
                    ml-l (mx/realize (:log-ml-estimate rl))
                    ml-b (mx/realize (:log-ml-estimate rb))]
                (println (str "    lanes " (- t1 t0) "ms  per-branch " (- t2 t1)
                              "ms  log-ML lanes=" (.toFixed ml-l 3)
                              " branch=" (.toFixed ml-b 3)))
                (assert-true "L5: engines agree on log-ML (|Δ| < 5, same estimand)"
                             (< (js/Math.abs (- ml-l ml-b)) 5.0))
                (assert-true "L5: ledger empty after both" (zero? (branch-count model)))))))))))

(def moe-dir (some-> js/process .-env .-GENMLX_MOE_MODEL))

(defn- run-moe-bench []
  (if-not (and moe-dir (.existsSync fs moe-dir))
    (do (println "\n  SKIP L6 — GENMLX_MOE_MODEL not set / missing") (pr/resolved nil))
    (pr/let [mm (llm/load-model moe-dir)
             {:keys [model tokenizer]} mm
             enc (llm/encode tokenizer "My phone number is ")
             constraint (gram/compile-constraint tokenizer "[0-9]{3}-[0-9]{4}")]
      (println "\n-- L6 35B MoE lanes wall-clock gate --")
      (if-not (llm/supports-branching? model)
        (println "  SKIP L6 — model lacks the owned branch surface")
        (let [prompt (vec enc)
              eos (llm/eos-token-id tokenizer)
              ;; warm both engines first: the FIRST B=K batched run pays a
              ;; one-time NVRTC JIT for the batch-K MoE kernel variants
              ;; (~8s measured on the 35B) that would otherwise swamp the
              ;; timing gate
              _ (tsmc/token-smc {:particles 8 :max-tokens 3 :eos-id eos
                                 :lanes? true :proposal :model
                                 :key (rng/fresh-key 1)} mm prompt)
              _ (tsmc/token-smc {:particles 8 :max-tokens 3 :eos-id eos
                                 :proposal :model
                                 :key (rng/fresh-key 1)} mm prompt)
              t0 (js/Date.now)
              rl (tsmc/token-smc {:particles 8 :max-tokens 10 :eos-id eos
                                  :lanes? true :proposal :grammar-masked
                                  :constraint constraint
                                  :key (rng/fresh-key 123)}
                                 mm prompt)
              t1 (js/Date.now)
              rb (tsmc/token-smc {:particles 8 :max-tokens 10 :eos-id eos
                                  :proposal :grammar-masked
                                  :constraint constraint
                                  :key (rng/fresh-key 123)}
                                 mm prompt)
              t2 (js/Date.now)
              ms-l (- t1 t0) ms-b (- t2 t1)
              ml-l (mx/realize (:log-ml-estimate rl))
              ml-b (mx/realize (:log-ml-estimate rb))]
          (println (str "    lanes " ms-l "ms  per-branch " ms-b "ms  ("
                        (.toFixed (/ ms-b (max 1 ms-l)) 2) "x)  log-ML lanes="
                        (.toFixed ml-l 3) " branch=" (.toFixed ml-b 3)))
          (assert-true "L6: lanes strictly faster than per-branch at K=8 on the 35B"
                       (< ms-l ms-b))
          (assert-true "L6: engines agree on log-ML (|Δ| < 5)"
                       (< (js/Math.abs (- ml-l ml-b)) 5.0))
          (assert-true "L6: ledger empty after both" (zero? (branch-count model))))))))

(-> (run-gates)
    (pr/then (fn [_] (run-moe-bench)))
    (pr/then (fn [_] (summary)))
    (pr/catch (fn [e]
                (swap! fail inc)
                (println "  FAIL (uncaught)" (.-message e) (pr-str (ex-data e)))
                (summary))))
