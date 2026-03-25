(ns genmlx.vectorized-test
  "Tests for vectorized inference: dist-sample-n, vsimulate, vgenerate,
   resample, vectorized IS/SMC, splice batching, and fallback paths."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.vectorized :as vec]
            [genmlx.inference.importance :as is]
            [genmlx.inference.smc :as smc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; -------------------------------------------------------------------------
;; dist-sample-n shape correctness
;; -------------------------------------------------------------------------

(deftest dist-sample-n-shape-test
  (testing "dist-sample-n produces correct shapes"
    (let [n 50
          key (rng/fresh-key)]
      (let [samples (dc/dist-sample-n (dist/gaussian 0 1) key n)]
        (is (= [n] (h/realize-shape samples)) "gaussian shape"))
      (let [samples (dc/dist-sample-n (dist/uniform 0 1) key n)]
        (is (= [n] (h/realize-shape samples)) "uniform shape"))
      (let [samples (dc/dist-sample-n (dist/bernoulli 0.5) key n)]
        (is (= [n] (h/realize-shape samples)) "bernoulli shape"))
      (let [samples (dc/dist-sample-n (dist/exponential 2.0) key n)]
        (is (= [n] (h/realize-shape samples)) "exponential shape"))
      (let [samples (dc/dist-sample-n (dist/laplace 0 1) key n)]
        (is (= [n] (h/realize-shape samples)) "laplace shape"))
      (let [samples (dc/dist-sample-n (dist/log-normal 0 1) key n)]
        (is (= [n] (h/realize-shape samples)) "log-normal shape"))
      (let [samples (dc/dist-sample-n (dist/delta (mx/scalar 42.0)) key n)]
        (is (= [n] (h/realize-shape samples)) "delta shape")
        (is (h/close? 42.0 (mx/realize (mx/index samples 0)) 0.001) "delta all same")))))

;; -------------------------------------------------------------------------
;; log-prob broadcasting
;; -------------------------------------------------------------------------

(deftest log-prob-broadcasting-test
  (testing "log-prob broadcasting with [N]-shaped values"
    (let [n 50
          key (rng/fresh-key)]
      (let [d (dist/gaussian 0 1)
            samples (dc/dist-sample-n d key n)
            lp (dc/dist-log-prob d samples)]
        (is (= [n] (h/realize-shape lp)) "gaussian log-prob shape")
        (let [max-lp (mx/realize (mx/amax lp))]
          (is (< max-lp 0.01) "gaussian log-probs negative")))
      (let [d (dist/uniform 0 1)
            samples (dc/dist-sample-n d key n)
            lp (dc/dist-log-prob d samples)]
        (is (= [n] (h/realize-shape lp)) "uniform log-prob shape")
        (is (h/close? 0.0 (mx/realize (mx/mean lp)) 0.001) "uniform log-prob value")))))

;; -------------------------------------------------------------------------
;; vsimulate
;; -------------------------------------------------------------------------

(deftest vsimulate-test
  (testing "vsimulate shape correctness"
    (let [model (gen []
                  (trace :x (dist/gaussian 0 1))
                  (trace :y (dist/uniform -1 1))
                  nil)
          n 100
          key (rng/fresh-key)
          vtrace (dyn/vsimulate model [] n key)]
      (is (instance? vec/VectorizedTrace vtrace) "vsimulate returns VectorizedTrace")
      (is (= n (:n-particles vtrace)) "vsimulate n-particles")
      (let [x-val (cm/get-value (cm/get-submap (:choices vtrace) :x))
            y-val (cm/get-value (cm/get-submap (:choices vtrace) :y))]
        (mx/eval! x-val y-val)
        (is (= [n] (mx/shape x-val)) "vsimulate :x shape")
        (is (= [n] (mx/shape y-val)) "vsimulate :y shape"))
      (let [score (:score vtrace)]
        (is (= [n] (h/realize-shape score)) "vsimulate score shape")))))

;; -------------------------------------------------------------------------
;; vgenerate
;; -------------------------------------------------------------------------

(deftest vgenerate-test
  (testing "vgenerate with constraints"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 1))]
                    (trace :y (dist/gaussian x 0.1))
                    nil))
          n 100
          key (rng/fresh-key)
          obs (cm/choicemap :y (mx/scalar 2.0))
          vtrace (dyn/vgenerate model [] obs n key)]
      (is (instance? vec/VectorizedTrace vtrace) "vgenerate returns VectorizedTrace")
      (let [x-val (cm/get-value (cm/get-submap (:choices vtrace) :x))]
        (is (= [n] (h/realize-shape x-val)) "vgenerate :x shape"))
      (let [y-val (cm/get-value (cm/get-submap (:choices vtrace) :y))]
        (is (h/close? 2.0 (h/realize y-val) 0.001) "vgenerate :y is constrained"))
      (let [w (:weight vtrace)]
        (is (= [n] (h/realize-shape w)) "vgenerate weight shape"))
      (let [log-ml (vec/vtrace-log-ml-estimate vtrace)]
        (is (js/isFinite (h/realize log-ml)) "vgenerate log-ml is finite"))
      (let [ess (vec/vtrace-ess vtrace)]
        (is (> ess 0) "vgenerate ESS > 0")
        (is (<= ess n) "vgenerate ESS <= N")))))

;; -------------------------------------------------------------------------
;; Statistical equivalence
;; -------------------------------------------------------------------------

(deftest statistical-equivalence-test
  (testing "Sequential vs batched mean score equivalence"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 5.0 2.0))]
                    x))
          n 500
          seq-results (mapv (fn [_] (p/generate (dyn/auto-key model) [] cm/EMPTY)) (range n))
          seq-scores (mapv (fn [r] (h/realize (:score (:trace r)))) seq-results)
          seq-mean-score (/ (reduce + seq-scores) n)
          key (rng/fresh-key)
          vtrace (dyn/vsimulate model [] n key)
          batch-scores (:score vtrace)
          _ (mx/eval! batch-scores)
          batch-mean-score (mx/realize (mx/mean batch-scores))]
      (is (h/close? seq-mean-score batch-mean-score 0.5)
          "mean score similar (within 0.5)"))))

;; -------------------------------------------------------------------------
;; resample-vtrace
;; -------------------------------------------------------------------------

(deftest resample-vtrace-test
  (testing "resample-vtrace"
    (let [model (gen []
                  (trace :x (dist/gaussian 0 1))
                  nil)
          n 50
          key (rng/fresh-key)
          [k1 k2] (rng/split key)
          vtrace (dyn/vsimulate model [] n k1)
          resampled (vec/resample-vtrace vtrace k2)]
      (is (instance? vec/VectorizedTrace resampled) "resampled is VectorizedTrace")
      (let [x-val (cm/get-value (cm/get-submap (:choices resampled) :x))]
        (is (= [n] (h/realize-shape x-val)) "resampled :x shape"))
      (let [w (:weight resampled)]
        (is (h/close? 0.0 (h/realize (mx/mean w)) 0.001)
            "resampled weights are zero")))))

;; -------------------------------------------------------------------------
;; vectorized importance sampling
;; -------------------------------------------------------------------------

(deftest vectorized-importance-sampling-test
  (testing "Vectorized importance sampling"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 10))]
                    (trace :obs (dist/gaussian x 1))
                    x))
          obs (cm/choicemap :obs (mx/scalar 5.0))
          {:keys [vtrace log-ml-estimate]}
          (is/vectorized-importance-sampling {:samples 200} model [] obs)]
      (is (instance? vec/VectorizedTrace vtrace) "vis returns VectorizedTrace")
      (is (js/isFinite (h/realize log-ml-estimate)) "vis log-ml is finite")
      (let [ess (vec/vtrace-ess vtrace)]
        (is (> ess 1) "vis ESS > 1")))))

;; -------------------------------------------------------------------------
;; vsmc-init
;; -------------------------------------------------------------------------

(deftest vsmc-init-test
  (testing "vsmc-init"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 10))]
                    (trace :obs (dist/gaussian x 1))
                    x))
          obs (cm/choicemap :obs (mx/scalar 3.0))
          {:keys [vtrace log-ml-estimate]}
          (smc/vsmc-init model [] obs 100 nil)]
      (is (instance? vec/VectorizedTrace vtrace) "vsmc-init returns VectorizedTrace")
      (is (js/isFinite (h/realize log-ml-estimate)) "vsmc-init log-ml is finite"))))

;; -------------------------------------------------------------------------
;; Vectorized splice: vsimulate
;; -------------------------------------------------------------------------

(deftest splice-vsimulate-test
  (testing "Vectorized splice vsimulate"
    (let [sub-model (gen []
                      (trace :z (dist/gaussian 0 1))
                      (trace :w (dist/uniform -1 1))
                      nil)
          model (gen []
                  (let [x (trace :x (dist/gaussian 0 10))]
                    (splice :sub sub-model)
                    x))
          n 50
          key (rng/fresh-key)
          vtrace (dyn/vsimulate model [] n key)]
      (is (instance? vec/VectorizedTrace vtrace) "vsimulate+splice returns VectorizedTrace")
      (let [x-val (cm/get-value (cm/get-submap (:choices vtrace) :x))]
        (is (= [n] (h/realize-shape x-val)) "splice vsimulate :x shape"))
      (let [sub-cm (cm/get-submap (:choices vtrace) :sub)
            z-val (cm/get-value (cm/get-submap sub-cm :z))
            w-val (cm/get-value (cm/get-submap sub-cm :w))]
        (mx/eval! z-val w-val)
        (is (= [n] (mx/shape z-val)) "splice vsimulate :sub :z shape")
        (is (= [n] (mx/shape w-val)) "splice vsimulate :sub :w shape"))
      (let [score (:score vtrace)]
        (is (= [n] (h/realize-shape score)) "splice vsimulate score shape")))))

;; -------------------------------------------------------------------------
;; Vectorized splice: vgenerate
;; -------------------------------------------------------------------------

(deftest splice-vgenerate-test
  (testing "Vectorized splice vgenerate"
    (let [sub-model (gen [mu]
                      (trace :z (dist/gaussian mu 1))
                      nil)
          model (gen []
                  (let [x (trace :x (dist/gaussian 0 10))]
                    (splice :sub sub-model x)
                    x))
          n 50
          key (rng/fresh-key)
          obs (cm/choicemap :sub (cm/choicemap :z (mx/scalar 2.0)))
          vtrace (dyn/vgenerate model [] obs n key)]
      (is (instance? vec/VectorizedTrace vtrace) "vgenerate+splice returns VectorizedTrace")
      (let [x-val (cm/get-value (cm/get-submap (:choices vtrace) :x))]
        (is (= [n] (h/realize-shape x-val)) "splice vgenerate :x shape"))
      (let [sub-cm (cm/get-submap (:choices vtrace) :sub)
            z-val (cm/get-value (cm/get-submap sub-cm :z))]
        (is (h/close? 2.0 (h/realize z-val) 0.001) "splice vgenerate :sub :z constrained"))
      (let [w (:weight vtrace)]
        (is (= [n] (h/realize-shape w)) "splice vgenerate weight shape")))))

;; -------------------------------------------------------------------------
;; Vectorized splice: vupdate
;; -------------------------------------------------------------------------

(deftest splice-vupdate-test
  (testing "Vectorized splice vupdate"
    (let [sub-model (gen []
                      (trace :z (dist/gaussian 0 1))
                      nil)
          model (gen []
                  (let [x (trace :x (dist/gaussian 0 10))]
                    (splice :sub sub-model)
                    x))
          n 50
          key (rng/fresh-key)
          [k1 k2] (rng/split key)
          vtrace (dyn/vsimulate model [] n k1)
          new-obs (cm/choicemap :sub (cm/choicemap :z (mx/scalar 3.0)))
          {:keys [vtrace weight]} (dyn/vupdate model vtrace new-obs k2)]
      (is (instance? vec/VectorizedTrace vtrace) "vupdate+splice returns VectorizedTrace")
      (let [sub-cm (cm/get-submap (:choices vtrace) :sub)
            z-val (cm/get-value (cm/get-submap sub-cm :z))]
        (is (h/close? 3.0 (h/realize z-val) 0.001) "splice vupdate :sub :z updated"))
      (is (= [n] (h/realize-shape weight)) "splice vupdate weight shape"))))

;; -------------------------------------------------------------------------
;; Vectorized splice: vregenerate
;; -------------------------------------------------------------------------

(deftest splice-vregenerate-test
  (testing "Vectorized splice vregenerate"
    (let [sub-model (gen []
                      (trace :z (dist/gaussian 0 1))
                      nil)
          model (gen []
                  (let [x (trace :x (dist/gaussian 0 10))]
                    (splice :sub sub-model)
                    x))
          n 50
          key (rng/fresh-key)
          [k1 k2] (rng/split key)
          vtrace (dyn/vsimulate model [] n k1)
          sel (sel/hierarchical :sub (sel/select :z))
          {:keys [vtrace weight]} (dyn/vregenerate model vtrace sel k2)]
      (is (instance? vec/VectorizedTrace vtrace) "vregenerate+splice returns VectorizedTrace")
      (let [sub-cm (cm/get-submap (:choices vtrace) :sub)
            z-val (cm/get-value (cm/get-submap sub-cm :z))]
        (is (= [n] (h/realize-shape z-val)) "splice vregenerate :sub :z shape"))
      (is (= [n] (h/realize-shape weight)) "splice vregenerate weight shape"))))

;; -------------------------------------------------------------------------
;; Non-DynamicGF fallback
;; -------------------------------------------------------------------------

(deftest non-dynamic-gf-fallback-test
  (testing "Non-DynamicGF combinator fallback in batched mode"
    (let [model (gen []
                  (splice :d (dist/gaussian 0 1))
                  nil)
          vtrace (dyn/vsimulate model [] 10 nil)
          d-sub (cm/get-submap (:choices vtrace) :d)]
      (is (cm/has-value? d-sub) "non-DynamicGF splice works in batched mode")
      (let [v (cm/get-value d-sub)]
        (is (= [10] (h/realize-shape v)) "non-DynamicGF splice shape")))))

;; -------------------------------------------------------------------------
;; Nested 3-level splice
;; -------------------------------------------------------------------------

(deftest nested-3-level-splice-test
  (testing "Nested 3-level splice"
    (let [inner (gen []
                  (trace :a (dist/gaussian 0 1)))
          middle (gen []
                   (trace :b (dist/uniform -1 1))
                   (splice :inner inner))
          outer (gen []
                  (trace :c (dist/exponential 1.0))
                  (splice :mid middle)
                  nil)
          n 50
          key (rng/fresh-key)
          vtrace (dyn/vsimulate outer [] n key)]
      (is (instance? vec/VectorizedTrace vtrace) "nested splice returns VectorizedTrace")
      (let [c-val (cm/get-value (cm/get-submap (:choices vtrace) :c))]
        (is (= [n] (h/realize-shape c-val)) "nested splice :c shape"))
      (let [mid-cm (cm/get-submap (:choices vtrace) :mid)
            b-val (cm/get-value (cm/get-submap mid-cm :b))]
        (is (= [n] (h/realize-shape b-val)) "nested splice :mid :b shape"))
      (let [mid-cm (cm/get-submap (:choices vtrace) :mid)
            inner-cm (cm/get-submap mid-cm :inner)
            a-val (cm/get-value (cm/get-submap inner-cm :a))]
        (is (= [n] (h/realize-shape a-val)) "nested splice :mid :inner :a shape"))
      (let [score (:score vtrace)]
        (is (= [n] (h/realize-shape score)) "nested splice score shape")))))

;; -------------------------------------------------------------------------
;; Sequential fallback (beta)
;; -------------------------------------------------------------------------

(deftest sequential-fallback-beta-test
  (testing "Sequential fallback for beta distribution"
    (let [d (dist/beta-dist 2 5)
          key (rng/fresh-key)
          n 20
          samples (dc/dist-sample-n d key n)]
      (is (= [n] (h/realize-shape samples)) "beta fallback shape")
      (let [min-val (mx/realize (mx/amin samples))
            max-val (mx/realize (mx/amax samples))]
        (is (and (> min-val 0) (< max-val 1)) "beta samples in (0,1)")))))

(cljs.test/run-tests)
