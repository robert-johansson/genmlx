(ns genmlx.l3-5-score-fn-test
  "Level 3.5 WP-4: Score function integration tests."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.gen :refer [gen]]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.util :as u]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.mlx.random :as rng]))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

(def nn-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      mu)))

(def mixed-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))
          sigma (trace :sigma (dist/gamma-dist 2 1))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      (trace :y3 (dist/gaussian 0 sigma))
      mu)))

(def non-conjugate-model
  (gen []
    (let [x (trace :x (dist/uniform -5 5))
          y (trace :y (dist/uniform -5 5))]
      (trace :obs (dist/gaussian (mx/add x y) 1))
      (mx/add x y))))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest get-eliminated-addresses-test
  (testing "get-eliminated-addresses"
    (let [nn-elim (u/get-eliminated-addresses nn-model)
          mixed-elim (u/get-eliminated-addresses mixed-model)
          nc-elim (u/get-eliminated-addresses non-conjugate-model)]
      (is (contains? nn-elim :mu) "NN model: :mu eliminated")
      (is (= #{:mu} nn-elim) "NN model: only :mu eliminated")
      (is (contains? mixed-elim :mu) "Mixed model: :mu eliminated")
      (is (not (contains? mixed-elim :sigma)) "Mixed model: :sigma NOT eliminated")
      (is (or (nil? nc-elim) (empty? nc-elim)) "Non-conjugate model: nothing eliminated"))))

(deftest filter-addresses-test
  (testing "filter-addresses"
    (is (= [:sigma] (u/filter-addresses [:mu :sigma] #{:mu})) "Filters eliminated addresses")
    (is (= [:mu :sigma] (u/filter-addresses [:mu :sigma] nil)) "No filtering when eliminated is nil")
    (is (= [:mu :sigma] (u/filter-addresses [:mu :sigma] #{})) "No filtering when eliminated is empty")
    (is (= [] (u/filter-addresses [:mu] #{:mu})) "All filtered -> empty")))

(deftest make-conjugate-aware-score-fn-test
  (testing "make-conjugate-aware-score-fn: conjugate model"
    (let [obs (-> cm/EMPTY
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0))
                  (cm/set-value :y3 (mx/scalar 0.5)))
          result (u/make-conjugate-aware-score-fn mixed-model [] obs [:mu :sigma])]
      (is (:reduced? result) "reduced? is true for conjugate model")
      (is (= [:sigma] (:addresses result)) "addresses reduced to [:sigma]")
      (is (= #{:mu} (:eliminated result)) "eliminated is #{:mu}")
      (let [w ((:score-fn result) (mx/array [1.5]))]
        (mx/eval! w)
        (is (js/isFinite (mx/item w)) "score-fn returns finite value"))))

  (testing "make-conjugate-aware-score-fn: non-conjugate model"
    (let [obs (-> cm/EMPTY (cm/set-value :obs (mx/scalar 1.0)))
          result (u/make-conjugate-aware-score-fn non-conjugate-model [] obs [:x :y])]
      (is (not (:reduced? result)) "reduced? is false for non-conjugate model")
      (is (= [:x :y] (:addresses result)) "addresses unchanged"))))

(deftest score-consistency-test
  (testing "Score consistency"
    (let [obs (-> cm/EMPTY
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0))
                  (cm/set-value :y3 (mx/scalar 0.5)))
          model (dyn/auto-key mixed-model)
          full-fn (u/make-score-fn model [] obs [:mu :sigma])
          reduced-fn (u/make-score-fn model [] obs [:sigma])
          full-w (full-fn (mx/array [0.0 1.5]))
          reduced-w (reduced-fn (mx/array [1.5]))]
      (mx/eval! full-w)
      (mx/eval! reduced-w)
      (is (> (js/Math.abs (- (mx/item full-w) (mx/item reduced-w))) 0.1)
          "full and reduced scores differ (joint vs marginal)"))))

(deftest prepare-mcmc-score-auto-filters-test
  (testing "prepare-mcmc-score auto-filters: conjugate model"
    (let [obs (-> cm/EMPTY
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0))
                  (cm/set-value :y3 (mx/scalar 0.5)))
          model (dyn/auto-key mixed-model)
          {:keys [trace]} (p/generate model [] obs)
          result (u/prepare-mcmc-score model [] obs [:mu :sigma] trace)]
      (is (= 1 (:n-params result)) "n-params reduced to 1")
      (is (= [1] (mx/shape (:init-params result))) "init-params shape is [1]")
      (let [w ((:score-fn result) (:init-params result))]
        (mx/eval! w)
        (is (js/isFinite (mx/item w)) "score-fn from prepare returns finite"))))

  (testing "prepare-mcmc-score auto-filters: non-conjugate model"
    (let [obs (-> cm/EMPTY (cm/set-value :obs (mx/scalar 1.0)))
          model (dyn/auto-key non-conjugate-model)
          {:keys [trace]} (p/generate model [] obs)
          result (u/prepare-mcmc-score model [] obs [:x :y] trace)]
      (is (= 2 (:n-params result)) "non-conjugate: n-params unchanged at 2"))))

(deftest nn-fully-eliminated-test
  (testing "NN model fully eliminated"
    (let [obs (-> cm/EMPTY
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0)))
          result (u/make-conjugate-aware-score-fn nn-model [] obs [:mu])]
      (is (:reduced? result) "NN model: all latents eliminated")
      (is (= [] (:addresses result)) "NN model: addresses empty after elimination"))))

(deftest compiled-mh-reduced-dimension-test
  (testing "Compiled MH in reduced dimension"
    (let [obs (-> cm/EMPTY
                  (cm/set-value :y1 (mx/scalar 3.0))
                  (cm/set-value :y2 (mx/scalar 4.0))
                  (cm/set-value :y3 (mx/scalar 0.5)))
          model (dyn/auto-key mixed-model)
          {:keys [trace]} (p/generate model [] obs)
          {:keys [score-fn init-params n-params]}
          (u/prepare-mcmc-score model [] obs [:mu :sigma] trace)
          _ (is (= 1 n-params) "prepare-mcmc-score gives n-params=1")
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
      (is (= n-steps (count (:sigmas chain))) "Got chain samples")
      (let [mean-sigma (/ (reduce + post-sigmas) (count post-sigmas))]
        (is (js/isFinite mean-sigma) "Mean sigma is finite")
        (is (> mean-sigma 0) "Mean sigma > 0")))))

(deftest fallback-non-conjugate-test
  (testing "Fallback for non-conjugate model"
    (let [obs (-> cm/EMPTY (cm/set-value :obs (mx/scalar 1.0)))
          samples (mcmc/compiled-mh
                    {:samples 50 :burn 20 :addresses [:x :y]
                     :proposal-std 0.5 :compile? false :device :cpu}
                    non-conjugate-model [] obs)]
      (is (= 2 (count (first samples))) "Non-conjugate samples have dimension 2")
      (is (= 50 (count samples)) "Got 50 samples"))))

(cljs.test/run-tests)
