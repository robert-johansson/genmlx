(ns genmlx.l3-5-score-fn-test
  "Level 3.5 WP-4: Score function integration tests.

   Verifies that compiled MCMC score functions automatically exclude
   analytically eliminated addresses, reducing parameter dimensionality.

   Run: bun run --bun nbb test/genmlx/l3_5_score_fn_test.cljs"
  (:require [genmlx.gen :refer [gen]]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.util :as u]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.mlx.random :as rng]))

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

(defn- assert-equal [desc expected actual]
  (if (= expected actual)
    (do (swap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " desc " (expected=" (pr-str expected)
                     " actual=" (pr-str actual) ")")))))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

;; Pure conjugate: mu ~ N(0,10), y1,y2 ~ N(mu,1)
(def nn-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      mu)))

;; Mixed: mu ~ N(0,10) [conjugate], sigma ~ Gamma(2,1) [non-conjugate]
(def mixed-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))
          sigma (trace :sigma (dist/gamma-dist 2 1))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      (trace :y3 (dist/gaussian 0 sigma))
      mu)))

;; Non-conjugate: no analytical elimination possible
;; Use uniform priors (not in any conjugate family with gaussian obs)
(def non-conjugate-model
  (gen []
    (let [x (trace :x (dist/uniform -5 5))
          y (trace :y (dist/uniform -5 5))]
      (trace :obs (dist/gaussian (mx/add x y) 1))
      (mx/add x y))))

;; =========================================================================
;; Test 1: get-eliminated-addresses
;; =========================================================================

(println "\n===== WP-4: Score Function Integration =====\n")

(println "-- Test 1: get-eliminated-addresses --")

(let [nn-elim (u/get-eliminated-addresses nn-model)
      mixed-elim (u/get-eliminated-addresses mixed-model)
      nc-elim (u/get-eliminated-addresses non-conjugate-model)]
  (assert-true "NN model: :mu eliminated" (contains? nn-elim :mu))
  (assert-equal "NN model: only :mu eliminated" #{:mu} nn-elim)
  (assert-true "Mixed model: :mu eliminated" (contains? mixed-elim :mu))
  (assert-true "Mixed model: :sigma NOT eliminated" (not (contains? mixed-elim :sigma)))
  (assert-true "Non-conjugate model: nothing eliminated"
    (or (nil? nc-elim) (empty? nc-elim))))

;; =========================================================================
;; Test 2: filter-addresses
;; =========================================================================

(println "\n-- Test 2: filter-addresses --")

(assert-equal "Filters eliminated addresses"
  [:sigma] (u/filter-addresses [:mu :sigma] #{:mu}))
(assert-equal "No filtering when eliminated is nil"
  [:mu :sigma] (u/filter-addresses [:mu :sigma] nil))
(assert-equal "No filtering when eliminated is empty"
  [:mu :sigma] (u/filter-addresses [:mu :sigma] #{}))
(assert-equal "All filtered → empty"
  [] (u/filter-addresses [:mu] #{:mu}))

;; =========================================================================
;; Test 3: make-conjugate-aware-score-fn
;; =========================================================================

(println "\n-- Test 3: make-conjugate-aware-score-fn --")

(let [obs (-> cm/EMPTY
              (cm/set-value :y1 (mx/scalar 3.0))
              (cm/set-value :y2 (mx/scalar 4.0))
              (cm/set-value :y3 (mx/scalar 0.5)))
      result (u/make-conjugate-aware-score-fn mixed-model [] obs [:mu :sigma])]
  (assert-true "reduced? is true for conjugate model" (:reduced? result))
  (assert-equal "addresses reduced to [:sigma]" [:sigma] (:addresses result))
  (assert-equal "eliminated is #{:mu}" #{:mu} (:eliminated result))
  ;; Score fn works with reduced params
  (let [w ((:score-fn result) (mx/array [1.5]))]
    (mx/eval! w)
    (assert-true "score-fn returns finite value" (js/isFinite (mx/item w)))))

;; Non-conjugate model: no reduction
(let [obs (-> cm/EMPTY (cm/set-value :obs (mx/scalar 1.0)))
      result (u/make-conjugate-aware-score-fn non-conjugate-model [] obs [:x :y])]
  (assert-true "reduced? is false for non-conjugate model" (not (:reduced? result)))
  (assert-equal "addresses unchanged" [:x :y] (:addresses result)))

;; =========================================================================
;; Test 4: Score consistency — reduced score fn matches full
;; =========================================================================

(println "\n-- Test 4: Score consistency --")

(let [obs (-> cm/EMPTY
              (cm/set-value :y1 (mx/scalar 3.0))
              (cm/set-value :y2 (mx/scalar 4.0))
              (cm/set-value :y3 (mx/scalar 0.5)))
      model (dyn/auto-key mixed-model)
      ;; Full score fn (mu and sigma)
      full-fn (u/make-score-fn model [] obs [:mu :sigma])
      ;; Reduced score fn (only sigma)
      reduced-fn (u/make-score-fn model [] obs [:sigma])
      ;; Test at sigma=1.5 — mu=0 for full (should be marginalized anyway)
      full-w (full-fn (mx/array [0.0 1.5]))
      reduced-w (reduced-fn (mx/array [1.5]))]
  (mx/eval! full-w)
  (mx/eval! reduced-w)
  (assert-close "full and reduced scores match" (mx/item full-w) (mx/item reduced-w) 1e-4))

;; =========================================================================
;; Test 5: prepare-mcmc-score auto-filters
;; =========================================================================

(println "\n-- Test 5: prepare-mcmc-score auto-filters --")

(let [obs (-> cm/EMPTY
              (cm/set-value :y1 (mx/scalar 3.0))
              (cm/set-value :y2 (mx/scalar 4.0))
              (cm/set-value :y3 (mx/scalar 0.5)))
      model (dyn/auto-key mixed-model)
      {:keys [trace]} (p/generate model [] obs)
      result (u/prepare-mcmc-score model [] obs [:mu :sigma] trace)]
  (assert-equal "n-params reduced to 1" 1 (:n-params result))
  (assert-equal "init-params shape is [1]" [1] (mx/shape (:init-params result)))
  ;; Score fn works
  (let [w ((:score-fn result) (:init-params result))]
    (mx/eval! w)
    (assert-true "score-fn from prepare returns finite" (js/isFinite (mx/item w)))))

;; Non-conjugate: no reduction
(let [obs (-> cm/EMPTY (cm/set-value :obs (mx/scalar 1.0)))
      model (dyn/auto-key non-conjugate-model)
      {:keys [trace]} (p/generate model [] obs)
      result (u/prepare-mcmc-score model [] obs [:x :y] trace)]
  (assert-equal "non-conjugate: n-params unchanged at 2" 2 (:n-params result)))

;; =========================================================================
;; Test 6: NN model — fully eliminated
;; =========================================================================

(println "\n-- Test 6: NN model fully eliminated --")

(let [obs (-> cm/EMPTY
              (cm/set-value :y1 (mx/scalar 3.0))
              (cm/set-value :y2 (mx/scalar 4.0)))
      result (u/make-conjugate-aware-score-fn nn-model [] obs [:mu])]
  (assert-true "NN model: all latents eliminated" (:reduced? result))
  (assert-equal "NN model: addresses empty after elimination" [] (:addresses result)))

;; =========================================================================
;; Test 7: Compiled MH chain in reduced dimension
;; =========================================================================

(println "\n-- Test 7: Compiled MH in reduced dimension --")

(let [obs (-> cm/EMPTY
              (cm/set-value :y1 (mx/scalar 3.0))
              (cm/set-value :y2 (mx/scalar 4.0))
              (cm/set-value :y3 (mx/scalar 0.5)))
      ;; Use prepare-mcmc-score to verify dimension reduction, then run
      ;; a manual MH loop with per-step eval to avoid Metal exhaustion
      model (dyn/auto-key mixed-model)
      {:keys [trace]} (p/generate model [] obs)
      {:keys [score-fn init-params n-params]}
      (u/prepare-mcmc-score model [] obs [:mu :sigma] trace)
      _ (assert-equal "prepare-mcmc-score gives n-params=1" 1 n-params)
      ;; Manual MH loop with reduced dimension
      n-steps 300
      burn 100
      std 0.3
      chain
      (reduce
        (fn [{:keys [params accepts sigmas]} i]
          (let [noise (mx/multiply (mx/scalar std)
                        (rng/normal (rng/fresh-key (+ 2000 i)) (mx/shape params)))
                proposal (mx/add params noise)
                s-cur (score-fn params)
                s-prop (score-fn proposal)]
            (mx/eval! s-cur)
            (mx/eval! s-prop)
            (let [log-alpha (- (mx/item s-prop) (mx/item s-cur))
                  accept? (< (js/Math.log (js/Math.random)) log-alpha)
                  new-params (if accept? proposal params)]
              (mx/eval! new-params)
              {:params new-params
               :accepts (if accept? (inc accepts) accepts)
               :sigmas (conj sigmas (mx/item new-params))})))
        {:params init-params :accepts 0 :sigmas []}
        (range n-steps))
      post-sigmas (subvec (:sigmas chain) burn)]
  (assert-true "Got chain samples" (= n-steps (count (:sigmas chain))))
  ;; All sigma values should be positive (Gamma prior constrains)
  ;; Note: MH proposals may go negative but score will reject them
  (let [mean-sigma (/ (reduce + post-sigmas) (count post-sigmas))]
    (assert-true "Mean sigma is finite" (js/isFinite mean-sigma))
    (assert-true "Mean sigma > 0" (> mean-sigma 0))
    (println (str "    Mean sigma: " (.toFixed mean-sigma 4)
                  " Accept rate: " (.toFixed (/ (:accepts chain) n-steps) 4)))))

;; =========================================================================
;; Test 8: Fallback — non-conjugate model unchanged
;; =========================================================================

(println "\n-- Test 8: Fallback for non-conjugate model --")

(let [obs (-> cm/EMPTY (cm/set-value :obs (mx/scalar 1.0)))
      samples (mcmc/compiled-mh
                {:samples 50 :burn 20 :addresses [:x :y]
                 :proposal-std 0.5 :compile? false :device :cpu}
                non-conjugate-model [] obs)]
  (assert-equal "Non-conjugate samples have dimension 2"
    2 (count (first samples)))
  (assert-true "Got 50 samples" (= 50 (count samples))))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println "\n===== Score Function Test Results =====")
(println (str "PASS: " @pass-count " / " (+ @pass-count @fail-count)))
(when (pos? @fail-count)
  (println (str "FAIL: " @fail-count)))
(let [all-pass? (zero? @fail-count)]
  (println (if all-pass?
             "\n  *** ALL WP-4 TESTS PASSED ***"
             "\n  *** WP-4 TESTS FAILED ***")))
(println "======================================\n")
