;; @tier fast
(ns genmlx.token-smc-test
  "genmlx-5qk7: token-SMC over branchable caches — model-free validation
   (V1-V4) on the tiny coupled-grammar categorical model (the
   llm_token_mcmc_test pattern; enumerable posterior, no GPU model). The
   real-model smokes (V5 dense, V6 80B MoE) live in token_smc_real_test.cljs
   (env-gated). The V7 baseline-cost comparison lives in
   bench/token_smc_bench.cljs.

   The synthetic decoder is fully deterministic per seed, so the statistical
   bounds here are frozen observations with generous margins, not flaky
   draws.

   Run: bunx --bun nbb@1.4.208 test/genmlx/token_smc_test.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.llm.grammar :as grammar]
            [genmlx.llm.smc :as tsmc])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

(defn assert-close [label expected actual tol]
  (assert-true (str label " (" actual " ~ " expected ")")
               (and (js/isFinite actual)
                    (<= (js/Math.abs (- expected actual)) tol))))

;; ===========================================================================
;; The coupled-grammar categorical "token" model (a=0 b=1 c=2)
;; ===========================================================================

(def L [0.4 0.1 -0.2])
(def logits (mx/array L))
(def A 0) (def B 1) (def C 2)

(def coupled-constraint
  {:dfa (grammar/compile-regex "(ab|ac|ba|bc)")
   :token-index ["a" "b" "c"] :eos-id 999 :masks nil})

(def td-logits (fn [_prompt _toks] logits))

;; GLOBAL twisted target: π(t0,t1) ∝ e^{L[t0]} e^{L[t1]} over the 4 valid
;; pairs (token-smc's semantics: p_LM · grammar-indicator, globally
;; renormalized — NOT the per-step locally-renormalized product).
(def valid-pairs [[A B] [A C] [B A] [B C]])
(def ^:private eL (mapv #(js/Math.exp %) L))
(def S (reduce + eL))
(def Z-total (/ (reduce + (map (fn [[t0 t1]] (* (nth eL t0) (nth eL t1))) valid-pairs))
                (* S S)))
(def exact-log-ml (js/Math.log Z-total))
(def exact-post
  (into {} (map (fn [[t0 t1 :as pr]]
                  [pr (/ (/ (* (nth eL t0) (nth eL t1)) (* S S)) Z-total)])
                valid-pairs)))

(defn- tv [p q]
  (* 0.5 (reduce + (map (fn [k] (js/Math.abs (- (get p k 0.0) (get q k 0.0))))
                        (into (set (keys p)) (keys q))))))

(defn- weighted-posterior
  "Normalized weight mass per token pair from a token-smc result."
  [result]
  (let [ws (mapv #(mx/realize (:log-w %)) (:particles result))
        m (apply max ws)
        es (mapv #(js/Math.exp (- % m)) ws)
        z (reduce + es)]
    (reduce (fn [acc [pt e]]
              (update acc (vec (:tokens pt)) (fnil + 0.0) (/ e z)))
            {} (map vector (:particles result) es))))

(defn- run-smc [seed n & [extra]]
  (tsmc/token-smc (merge {:particles n :max-tokens 2 :eos-id 999
                          :proposal :grammar-masked
                          :constraint coupled-constraint
                          :decoder (tsmc/table-decoder td-logits)
                          :key (rng/fresh-key seed)}
                         extra)
                  nil [7 7]))

;; ===========================================================================
(println "\n-- V1 exactness: log-ML + posterior TV vs the enumerable target --")

(let [seeds [11 23 47 61 83]
      results (mapv #(run-smc % 64) seeds)
      mls (mapv #(mx/realize (:log-ml-estimate %)) results)
      mean-ml (/ (reduce + mls) (count mls))
      tvs (mapv #(tv exact-post (weighted-posterior %)) results)]
  (println "    exact log-ML =" exact-log-ml "| estimates =" (pr-str mls))
  (println "    TVs =" (pr-str (mapv #(js/Number (.toFixed % 3)) tvs)))
  (assert-close "V1: mean log-ML over 5 seeds ~ exact log-Z" exact-log-ml mean-ml 0.05)
  (assert-true "V1: every seed's log-ML within 0.3 of exact"
               (every? #(< (js/Math.abs (- % exact-log-ml)) 0.3) mls))
  (assert-true "V1: posterior TV at N=64 below 0.2 on every seed, mean < 0.12"
               (and (every? #(< % 0.2) tvs)
                    (< (/ (reduce + tvs) (count tvs)) 0.12)))
  (assert-true "V1: only valid sequences appear"
               (every? (set (map vec valid-pairs))
                       (mapcat #(map (comp vec :tokens) (:particles %)) results))))

;; ===========================================================================
(println "\n-- V2 agreement: smc ~ one-shot IS ~ token-MCMC (uncoupled grammar) --")
;; Uncoupled masks (t1's valid set independent of t0) make the local (cmodel)
;; and global (token-smc) targets coincide, so all three methods estimate the
;; same conditional p(t0 | t1 = c).

(def uncoupled-constraint
  {:dfa (grammar/compile-regex "[ab][bc]")
   :token-index ["a" "b" "c"] :eos-id 999 :masks nil})

(def base-model (gen [] (trace :t0 (dist/categorical logits))
                        (trace :t1 (dist/categorical logits)) nil))
(def cmodel (grammar/constrain base-model uncoupled-constraint))

(let [;; observe t1 = c through the twist: -Inf unless the step-1 token is c
      obs-twist (fn [{:keys [step]} toks]
                  (if (and (= step 1) (not= C (peek toks)))
                    js/Number.NEGATIVE_INFINITY
                    0.0))
      result (tsmc/token-smc {:particles 256 :max-tokens 2 :eos-id 999
                              :proposal :grammar-masked
                              :constraint uncoupled-constraint
                              :twist obs-twist
                              :decoder (tsmc/table-decoder td-logits)
                              :key (rng/fresh-key 5)}
                             nil [7 7])
      post (weighted-posterior result)
      smc-pa (reduce-kv (fn [acc pr w] (if (= A (first pr)) (+ acc w) acc)) 0.0 post)
      ;; exact conditional (uncoupled: Z1 independent of t0)
      exact-pa (/ (nth eL A) (+ (nth eL A) (nth eL B)))
      ;; long token-MCMC chain on cmodel with the same observation
      traces (mcmc/mh {:samples 3000 :burn 300 :selection (sel/select :t0)
                       :key (rng/fresh-key 123)}
                      cmodel [] (cm/from-map {:t1 (mx/array C)}))
      t0s (map (fn [t] (mx/item (cm/get-value (cm/get-submap (:choices t) :t0)))) traces)
      mcmc-pa (/ (count (filter #(= A %) t0s)) (double (count t0s)))]
  (println "    p(t0=a | t1=c): exact" (.toFixed exact-pa 3)
           "| smc" (.toFixed smc-pa 3) "| mcmc" (.toFixed mcmc-pa 3))
  (assert-close "V2: token-smc conditional ~ exact" exact-pa smc-pa 0.08)
  (assert-close "V2: token-MCMC chain ~ exact (same target, uncoupled)" exact-pa mcmc-pa 0.05)
  (assert-close "V2: smc ~ mcmc" mcmc-pa smc-pa 0.1))

;; ===========================================================================
(println "\n-- V3 GFI conformance: exported traces satisfy assess == score --")

(let [ccmodel (grammar/constrain base-model coupled-constraint)
      result (run-smc 31 8)]
  (assert-true "V3: every exported particle trace has assess == score (exact)"
               (every? (fn [pt]
                         (let [tr (tsmc/particle->trace ccmodel 2 pt (rng/fresh-key 9))
                               {:keys [weight]} (p/assess (dyn/with-key ccmodel (rng/fresh-key 9))
                                                          [] (:choices tr))]
                           (< (js/Math.abs (- (mx/realize (:score tr)) (mx/realize weight))) 1e-4)))
                       (:particles result))))

;; ===========================================================================
(println "\n-- V4 resource properties (R1 bounded, R2 no leak) --")

(let [dec (tsmc/table-decoder td-logits)
      n 16
      max-live (atom 0)
      _ (tsmc/token-smc {:particles n :max-tokens 2 :eos-id 999
                         :proposal :grammar-masked :constraint coupled-constraint
                         :decoder dec :key (rng/fresh-key 3)
                         :callback (fn [_] (swap! max-live max (tsmc/live-handles dec)))}
                        nil [7 7])]
  (println "    max live handles =" @max-live "| after return =" (tsmc/live-handles dec))
  (assert-true (str "R1: live handles bounded by N+1 during the run (" @max-live " <= " (inc n) ")")
               (<= @max-live (inc n)))
  (assert-true "R2: zero live handles after token-smc returns"
               (zero? (tsmc/live-handles dec))))

(let [dec (tsmc/table-decoder td-logits)
      out (tsmc/with-token-smc* {:particles 4 :max-tokens 2 :eos-id 999
                                 :proposal :grammar-masked :constraint coupled-constraint
                                 :decoder dec :key (rng/fresh-key 4)}
                                nil [7 7]
                                (fn [res]
                                  {:live-inside (tsmc/live-handles dec)
                                   :n (count (:particles res))}))]
  (assert-true "R2: with-token-smc* exposes live branches inside the scope"
               (pos? (:live-inside out)))
  (assert-true "R2: with-token-smc* tears everything down after the scope"
               (zero? (tsmc/live-handles dec))))

;; twist throwing: propagate + dispose (R2 on the error path)
(let [dec (tsmc/table-decoder td-logits)
      threw (try (tsmc/token-smc {:particles 4 :max-tokens 2 :eos-id 999
                                  :proposal :grammar-masked :constraint coupled-constraint
                                  :twist (fn [_ _] (throw (ex-info "boom" {})))
                                  :decoder dec :key (rng/fresh-key 6)}
                                 nil [7 7])
                 false
                 (catch :default _ true))]
  (assert-true "edge: a throwing twist propagates" threw)
  (assert-true "edge: all handles disposed after the throw" (zero? (tsmc/live-handles dec))))

;; ===========================================================================
(println "\n-- edge cases: T=0, EOS, mask deadlock --")

(let [r (tsmc/token-smc {:particles 4 :max-tokens 0 :eos-id 999
                         :decoder (tsmc/table-decoder td-logits)
                         :key (rng/fresh-key 8)}
                        nil [7 7])]
  (assert-true "T=0: prompt-only particles, log-ml 0"
               (and (every? (comp empty? :tokens) (:particles r))
                    (zero? (mx/realize (:log-ml-estimate r)))
                    (every? :finished? (:particles r)))))

;; EOS: token id 2 (c) as EOS — particles stop as soon as c is sampled
(let [r (tsmc/token-smc {:particles 32 :max-tokens 4 :eos-id C
                         :decoder (tsmc/table-decoder td-logits)
                         :key (rng/fresh-key 12)}
                        nil [7 7])]
  (assert-true "EOS: a particle sampling the eos id stops there (weight frozen)"
               (every? (fn [pt]
                         (let [toks (:tokens pt)
                               eos-at (.indexOf (to-array toks) C)]
                           (or (neg? eos-at) (= eos-at (dec (count toks))))))
                       (:particles r))))

;; full mask deadlock: after 'ab' NOTHING is allowed but T=3 keeps decoding →
;; every particle deadlocks → -Inf all → loud :degenerate-particles (ng9t)
(let [deadlock-constraint {:dfa (grammar/compile-regex "ab")
                           :token-index ["a" "b" "c"] :eos-id 999 :masks nil}
      kind (try (tsmc/token-smc {:particles 4 :max-tokens 3 :eos-id 999
                                 :proposal :grammar-masked
                                 :constraint deadlock-constraint
                                 :decoder (tsmc/table-decoder td-logits)
                                 :key (rng/fresh-key 14)}
                                nil [7 7])
                :no-throw
                (catch :default e (or (:genmlx/error (ex-data e)) :other)))]
  (assert-true "edge: all-particle mask deadlock throws :degenerate-particles"
               (= :degenerate-particles kind)))

;; ===========================================================================
(println "\n-- rejuvenation (v1: end-of-filter token-MCMC, weight-preserving) --")

(let [ccmodel (grammar/constrain base-model coupled-constraint)
      r (tsmc/token-smc {:particles 16 :max-tokens 2 :eos-id 999
                         :proposal :grammar-masked :constraint coupled-constraint
                         :decoder (tsmc/table-decoder td-logits)
                         :rejuvenation {:steps 3 :selection (sel/select :t0)
                                        :gf ccmodel :key (rng/fresh-key 21)}
                         :key (rng/fresh-key 19)}
                        nil [7 7])]
  (assert-true "rejuvenation: outputs remain valid sequences"
               (every? (set (map vec valid-pairs)) (map (comp vec :tokens) (:particles r))))
  (assert-true "rejuvenation: log-ML unchanged semantics (finite)"
               (js/isFinite (mx/realize (:log-ml-estimate r)))))

;; ===========================================================================
(println (str "\n== token-smc: " @pass " passed, " @fail " failed =="))
(when (pos? @fail) (set! (.-exitCode js/process) 1))
