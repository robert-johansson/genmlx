;; @tier medium
(ns genmlx.bugfix-test
  "Tests for three bug fixes:
   1. Update weight correctness for dependent variables
   2. Vectorized switch produces distinct samples
   3. Conditional SMC uses the reference trace"
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.gen :refer [gen]]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.combinators :as comb]
            [genmlx.inference.smc :as smc]))

(deftest update-weight-dependent-variables
  (testing "update weight: dependent variables"
    (let [model (dyn/auto-key (gen [obs-val]
                  (let [mu (trace :mu (dist/gaussian 0 10))]
                    (trace :obs (dist/gaussian mu 1))
                    mu)))
          obs-val 5.0
          init-constraints (cm/choicemap :mu (mx/scalar 0.0) :obs (mx/scalar obs-val))
          {:keys [trace]} (p/generate model [obs-val] init-constraints)
          old-score (:score trace)
          new-constraints (cm/choicemap :mu (mx/scalar 5.0))
          {:keys [trace weight]} (p/update model trace new-constraints)
          new-score (:score trace)
          expected-weight (- (mx/realize new-score) (mx/realize old-score))
          actual-weight (mx/realize weight)]
      (is (h/close? expected-weight actual-weight 0.001) "update weight = new_score - old_score")
      (is (> actual-weight 0) "update weight > 0 (mu moved toward obs)"))))

(deftest update-weight-map-combinator
  (testing "update weight: MapCombinator"
    (let [kernel (gen [x]
                  (let [mu (trace :mu (dist/gaussian 0 10))]
                    (trace :obs (dist/gaussian mu 1))
                    mu))
          model (comb/map-combinator (dyn/auto-key kernel))
          args [[0.0 0.0]]
          constraints (cm/choicemap
                        0 (cm/choicemap :mu (mx/scalar 1.0) :obs (mx/scalar 5.0))
                        1 (cm/choicemap :mu (mx/scalar 2.0) :obs (mx/scalar 5.0)))
          {:keys [trace]} (p/generate model args constraints)
          old-score (:score trace)
          new-constraints (cm/choicemap
                            0 (cm/choicemap :mu (mx/scalar 5.0))
                            1 (cm/choicemap :mu (mx/scalar 5.0)))
          {:keys [trace weight]} (p/update model trace new-constraints)
          new-score (:score trace)
          expected-weight (- (mx/realize new-score) (mx/realize old-score))
          actual-weight (mx/realize weight)]
      (is (h/close? expected-weight actual-weight 0.001) "map update weight = new_score - old_score")
      (is (> actual-weight 0) "map update weight > 0 (mu moved toward obs)"))))

(deftest update-weight-scan-combinator
  (testing "update weight: ScanCombinator"
    (let [kernel (gen [carry input]
                  (let [x (trace :x (dist/gaussian carry 1))]
                    (trace :obs (dist/gaussian x 0.5))
                    [x x]))
          model (comb/scan-combinator (dyn/auto-key kernel))
          args [(mx/scalar 0.0) [1 2]]
          constraints (cm/choicemap
                        0 (cm/choicemap :x (mx/scalar 1.0) :obs (mx/scalar 3.0))
                        1 (cm/choicemap :x (mx/scalar 2.0) :obs (mx/scalar 3.0)))
          {:keys [trace]} (p/generate model args constraints)
          old-score (:score trace)
          new-constraints (cm/choicemap
                            0 (cm/choicemap :x (mx/scalar 3.0)))
          {:keys [trace weight]} (p/update model trace new-constraints)
          new-score (:score trace)
          expected-weight (- (mx/realize new-score) (mx/realize old-score))
          actual-weight (mx/realize weight)]
      (is (h/close? expected-weight actual-weight 0.001) "scan update weight = new_score - old_score"))))

(deftest vectorized-switch-distinct-samples
  (testing "vectorized switch: distinct samples"
    (let [branches [(dist/gaussian 0 1) (dist/gaussian 10 1)]
          n 20
          index (mx/zeros [n] mx/int32)
          result (comb/vectorized-switch branches index [])
          values (cm/get-value (:choices result))
          vals-list (mx/->clj values)]
      (is (= (mx/shape values) [n]) "vectorized switch: values are [N]-shaped")
      (let [unique-count (count (set vals-list))]
        (is (> unique-count 1) "vectorized switch: values are distinct (not identical)"))
      (let [mean-val (mx/realize (mx/mean values))]
        (is (< (js/Math.abs mean-val) 2.0) "vectorized switch: mean near 0 for branch 0")))))

(deftest vectorized-switch-mixed-branches
  (testing "vectorized switch: mixed branch selection"
    (let [branches [(dist/gaussian 0 0.1) (dist/gaussian 100 0.1)]
          n 10
          index (mx/array [0 0 0 0 0 1 1 1 1 1] mx/int32)
          result (comb/vectorized-switch branches index [])
          values (cm/get-value (:choices result))
          vals-list (mx/->clj values)
          branch0-vals (take 5 vals-list)
          branch1-vals (drop 5 vals-list)]
      (is (< (/ (reduce + branch0-vals) 5) 5.0) "branch 0 values near 0")
      (is (> (/ (reduce + branch1-vals) 5) 90.0) "branch 1 values near 100")
      (is (= (mx/shape (:score result)) [n]) "vectorized switch: scores are [N]-shaped"))))

(deftest csmc-reference-trace
  (testing "conditional SMC: reference trace used"
    (let [model (dyn/auto-key (gen []
                  (let [x (trace :x (dist/gaussian 0 10))]
                    (trace :obs (dist/gaussian x 1))
                    x)))
          ref-x 42.0
          ref-constraints (cm/choicemap :x (mx/scalar ref-x) :obs (mx/scalar 42.5))
          {:keys [trace]} (p/generate model [] ref-constraints)
          reference-trace trace
          obs (cm/choicemap :obs (mx/scalar 42.5))
          result (smc/csmc {:particles 10 :key (rng/fresh-key)}
                           model [] [obs] reference-trace)
          ref-particle-trace (first (:traces result))
          ref-x-val (mx/realize (cm/get-choice (:choices ref-particle-trace) [:x]))]
      (is (h/close? ref-x ref-x-val 0.001) "cSMC: reference particle x = 42.0")
      (let [other-xs (mapv (fn [t]
                             (mx/realize (cm/get-choice (:choices t) [:x])))
                           (rest (:traces result)))
            different-count (count (filter #(> (js/Math.abs (- % ref-x)) 5) other-xs))]
        (is (> different-count 0) "cSMC: other particles differ from reference")))))

;; ---------------------------------------------------------------------------
;; 4. Mix regenerate: selected inner sites resample on a same-index resample
;;    (genmlx-uizc — blind retention froze them with weight 0)
;; ---------------------------------------------------------------------------

(deftest mix-regenerate-selected-inner-sites
  (testing "Mix + sel/all resamples inner sites on same-idx outcomes; idx-only selection still retains"
    (let [c0 (dyn/auto-key (gen [] (trace :x (dist/gaussian -50 0.1))))
          c1 (dyn/auto-key (gen [] (trace :x (dist/gaussian 50 0.1))))
          mix (comb/mix-combinator [c0 c1] (fn [_] (mx/array [0.0 0.0])))
          tr0 (p/simulate (dyn/with-key mix (rng/fresh-key 1)) [])
          old-x (mx/realize (cm/get-choice (:choices tr0) [:x]))
          old-idx (mx/realize (cm/get-choice (:choices tr0) [:component-idx]))
          regen (fn [sel seed]
                  (let [r (p/regenerate (dyn/with-key mix (rng/fresh-key seed)) tr0 sel)]
                    {:x (mx/realize (cm/get-choice (:choices (:trace r)) [:x]))
                     :idx (mx/realize (cm/get-choice (:choices (:trace r)) [:component-idx]))
                     :w (mx/realize (:weight r))}))
          all-runs (mapv #(regen sel/all (+ 100 %)) (range 12))
          same-idx-all (filterv #(= (:idx %) old-idx) all-runs)
          idx-only-runs (mapv #(regen (sel/select :component-idx) (+ 300 %)) (range 12))
          same-idx-only (filterv #(= (:idx %) old-idx) idx-only-runs)]
      ;; symmetric 2-component mix: same-idx outcomes occur w.p. 1/2 per run
      (is (pos? (count same-idx-all)) "sel/all: at least one same-idx outcome sampled")
      (is (every? #(not= (:x %) old-x) same-idx-all)
          "sel/all + same idx: selected :x RESAMPLED, never frozen")
      (is (every? #(< (js/Math.abs (:w %)) 1e-6) all-runs)
          "sel/all: full-prior resample weight is exactly 0")
      (is (pos? (count same-idx-only)) "idx-only: at least one same-idx outcome sampled")
      (is (every? #(= (:x %) old-x) same-idx-only)
          "(select :component-idx) + same idx: unselected :x RETAINED (genmlx-zek9)"))))

;; ---------------------------------------------------------------------------
;; 5. Poisson sampler moments at large rate (genmlx-2nec — the Knuth product
;;    loop underflowed exp(-rate) for rate >= ~708, pinning samples near 700)
;; ---------------------------------------------------------------------------

(deftest poisson-large-rate-moments
  (testing "poisson(2000) samples have the right mean/variance (was pinned ~700)"
    (let [n 150
          xs (mapv (fn [i] (mx/realize (dc/dist-sample (dist/poisson 2000)
                                                       (rng/fresh-key (+ 7000 i)))))
                   (range n))
          m (/ (reduce + xs) n)
          v (/ (reduce + (map #(let [d (- % m)] (* d d)) xs)) n)]
      ;; mean 2000, sd ~44.7: sample-mean SE ~3.7 (a ~7-sigma band);
      ;; var estimate SE ~ var*sqrt(2/n) ~ 231 (a ~5x band)
      (is (h/close? 2000 m 25) (str "mean ~2000 (got " m ")"))
      (is (h/close? 2000 v 1200) (str "variance ~2000 (got " v ")"))
      (is (> (apply min xs) 1700) "no sample anywhere near the old ~700 pin")))
  (testing "poisson small/branch-boundary rates still correct"
    (doseq [[rate n tol] [[3 300 0.5] [9.9 300 1.2] [10.1 300 1.2]]]
      (let [xs (mapv (fn [i] (mx/realize (dc/dist-sample (dist/poisson rate)
                                                         (rng/fresh-key (+ 9000 i)))))
                     (range n))
            m (/ (reduce + xs) n)]
        (is (h/close? rate m tol) (str "poisson(" rate ") mean (got " m ")"))))))

(cljs.test/run-tests)
