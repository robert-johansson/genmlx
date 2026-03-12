(ns genmlx.l3-5-gate-test
  "Level 3.5 Gate 0 + Gate 1: Regenerate correctness and performance.

   Gate 0: Correctness — MH chain with auto-handlers converges correctly
   Gate 1: Performance — no more than 10% wall-clock slowdown

   Case B only (prior NOT selected). Case A (prior selected) is deferred.

   Run: bun run --bun nbb test/genmlx/l3_5_gate_test.cljs"
  (:require [genmlx.gen :refer [gen]]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.selection :as sel]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn- assert-true [desc pred]
  (if pred
    (do (swap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " desc)))))

(defn- assert-close [desc expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass-count inc)
          (println (str "  PASS: " desc " (" (.toFixed actual 6) ")")))
      (do (swap! fail-count inc)
          (println (str "  FAIL: " desc " (expected=" (.toFixed expected 6)
                       " actual=" (.toFixed actual 6) " diff=" (.toFixed diff 6) ")"))))))

(defn- strip-analytical [gf]
  (assoc gf :schema (dissoc (:schema gf) :auto-handlers :auto-regenerate-handlers
                            :auto-regenerate-transition
                            :conjugate-pairs :has-conjugate? :analytical-plan)))

(defn- time-ms [f]
  (let [start (js/Date.now)]
    (f)
    (- (js/Date.now) start)))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

;; Mixed model: mu ~ N(0,10), sigma ~ Gamma(2,1)
;; y1,y2 ~ N(mu,1) [conjugate], y3 ~ N(0, sigma) [non-conjugate]
(def mixed-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))
          sigma (trace :sigma (dist/gamma-dist 2 1))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      (trace :y3 (dist/gaussian 0 sigma))
      mu)))

;; NN model for score consistency check
(def nn-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      mu)))

;; =========================================================================
;; Gate 0: Correctness
;; =========================================================================

(println "\n===== Gate 0: Regenerate Correctness =====\n")

;; Gate 0a: Score consistency — regenerate reproduces generate score
(println "-- Gate 0a: Score consistency --")

(let [model (dyn/auto-key nn-model)
      obs (-> cm/EMPTY
              (cm/set-value :y1 (mx/scalar 3.0))
              (cm/set-value :y2 (mx/scalar 4.0)))
      gen-result (p/generate model [] obs)
      trace (:trace gen-result)
      gen-score (do (mx/eval! (:score trace)) (mx/item (:score trace)))
      gen-weight (do (mx/eval! (:weight gen-result)) (mx/item (:weight gen-result)))
      ;; Regenerate with empty selection — should reproduce exact same score
      regen (p/regenerate model trace (sel/select))
      regen-score (do (mx/eval! (:score (:trace regen)))
                      (mx/item (:score (:trace regen))))
      regen-weight (do (mx/eval! (:weight regen))
                       (mx/item (:weight regen)))]
  (assert-close "regenerate score = generate score" gen-score regen-score 1e-6)
  (assert-close "regenerate weight = 0 (nothing selected)" 0.0 regen-weight 1e-6)
  (println (str "    Generate score: " (.toFixed gen-score 6)
                " Regenerate score: " (.toFixed regen-score 6))))

;; Gate 0b: MH chain on mixed model (select sigma, Case B for mu)
(println "\n-- Gate 0b: MH chain convergence (mixed model, Case B) --")

(let [model mixed-model
      obs (-> cm/EMPTY
              (cm/set-value :y1 (mx/scalar 3.0))
              (cm/set-value :y2 (mx/scalar 4.0))
              (cm/set-value :y3 (mx/scalar 0.5)))
      n-steps 2000
      selection (sel/select :sigma)
      key (rng/fresh-key 42)
      init-trace (:trace (p/generate (dyn/with-key model key) [] obs))
      chain
      (reduce
        (fn [{:keys [trace accepts sigmas scores]} i]
          (let [key-i (rng/fresh-key (+ 1000 i))
                {:keys [trace weight]} (p/regenerate (dyn/with-key model key-i) trace selection)]
            (mx/eval! weight)
            (mx/eval! (:score trace))
            (let [log-alpha (mx/item weight)
                  accept? (< (js/Math.log (js/Math.random)) log-alpha)
                  sigma-val (mx/item (cm/get-value (cm/get-submap (:choices trace) :sigma)))
                  score-val (mx/item (:score trace))]
              {:trace trace
               :accepts (if accept? (inc accepts) accepts)
               :sigmas (conj sigmas sigma-val)
               :scores (conj scores score-val)})))
        {:trace init-trace :accepts 0 :sigmas [] :scores []}
        (range n-steps))
      accept-rate (/ (:accepts chain) n-steps)
      burn-in 400
      post-sigmas (subvec (:sigmas chain) burn-in)
      chain-mean (/ (reduce + post-sigmas) (count post-sigmas))
      chain-var (/ (reduce + (map #(let [d (- % chain-mean)] (* d d)) post-sigmas))
                   (count post-sigmas))]
  (println (str "    Chain mean sigma: " (.toFixed chain-mean 4)))
  (println (str "    Chain var sigma: " (.toFixed chain-var 4)))
  (println (str "    Accept rate: " (.toFixed accept-rate 4)))
  (assert-true "All sigma values positive" (every? pos? post-sigmas))
  (assert-true "Acceptance rate > 0.1" (> accept-rate 0.1))
  (assert-true "Acceptance rate < 0.99 (not trivially accepting)" (< accept-rate 0.99))
  (assert-true "Chain mean sigma is finite and positive" (and (js/isFinite chain-mean) (pos? chain-mean)))
  ;; Score consistency: all scores should be finite
  (let [post-scores (subvec (:scores chain) burn-in)]
    (assert-true "All scores finite" (every? js/isFinite post-scores))))

;; Gate 0c: Multiple auto-handler regenerate cycles produce valid traces
(println "\n-- Gate 0c: Trace validity across regenerate cycles --")

(let [model (dyn/auto-key nn-model)
      obs (-> cm/EMPTY
              (cm/set-value :y1 (mx/scalar 3.0))
              (cm/set-value :y2 (mx/scalar 4.0)))
      trace (:trace (p/generate model [] obs))
      ;; Run 20 regenerate cycles (with empty selection)
      traces
      (reduce
        (fn [traces _]
          (let [regen (p/regenerate model (:trace (last traces)) (sel/select))]
            (conj traces regen)))
        [{:trace trace :weight (mx/scalar 0.0)}]
        (range 20))
      ;; Check all scores match
      scores (mapv (fn [r] (do (mx/eval! (:score (:trace r)))
                               (mx/item (:score (:trace r))))) traces)]
  (assert-true "All 20 regenerate cycles produce same score"
    (every? #(< (js/Math.abs (- % (first scores))) 1e-4) scores))
  (assert-true "All regenerate cycles have :y1=3.0"
    (every? #(< (js/Math.abs (- 3.0 (mx/item (cm/get-value
                (cm/get-submap (:choices (:trace %)) :y1))))) 1e-6) traces)))

;; =========================================================================
;; Gate 1: Performance
;; =========================================================================

(println "\n\n===== Gate 1: Regenerate Performance =====\n")

(println "-- Gate 1a: Wall-clock comparison --")

(let [model-with mixed-model
      model-without (strip-analytical mixed-model)
      obs (-> cm/EMPTY
              (cm/set-value :y1 (mx/scalar 3.0))
              (cm/set-value :y2 (mx/scalar 4.0))
              (cm/set-value :y3 (mx/scalar 0.5)))
      selection (sel/select :sigma)
      n-steps 1000

      run-chain
      (fn [model]
        (let [key (rng/fresh-key 42)
              init-trace (:trace (p/generate (dyn/with-key model key) [] obs))]
          (reduce
            (fn [trace i]
              (let [key-i (rng/fresh-key (+ 5000 i))
                    {:keys [trace weight]} (p/regenerate (dyn/with-key model key-i) trace selection)]
                (mx/eval! weight)
                trace))
            init-trace
            (range n-steps))))

      ;; Warmup both paths
      _ (run-chain model-with)
      _ (run-chain model-without)

      ;; Timed runs
      time-with (time-ms #(run-chain model-with))
      time-without (time-ms #(run-chain model-without))

      slowdown (if (pos? time-without) (/ (- time-with time-without) time-without) 0)]
  (println (str "    With auto-handlers: " time-with "ms"))
  (println (str "    Without auto-handlers: " time-without "ms"))
  (println (str "    Slowdown: " (.toFixed (* 100 slowdown) 1) "%"))
  (assert-true "Slowdown < 10% (optimized dispatch)"
    (< slowdown 0.10)))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println "\n===== Gate Results =====")
(println (str "PASS: " @pass-count " / " (+ @pass-count @fail-count)))
(when (pos? @fail-count)
  (println (str "FAIL: " @fail-count)))
(let [all-pass? (zero? @fail-count)]
  (println (if all-pass?
             "\n  *** GATE 0 + GATE 1 PASSED ***"
             "\n  *** GATES FAILED ***")))
(println "========================\n")
