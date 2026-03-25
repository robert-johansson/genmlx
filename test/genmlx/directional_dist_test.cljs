(ns genmlx.directional-dist-test
  "Tests for directional statistics distributions: von-mises, wrapped-cauchy, wrapped-normal."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.gradients :as grad]
            [genmlx.vectorized :as vec])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Circular mean helper
;; ---------------------------------------------------------------------------

(defn- circular-mean
  "Compute circular mean of a vector of angles."
  [angles]
  (let [n (count angles)
        s (reduce + (map js/Math.sin angles))
        c (reduce + (map js/Math.cos angles))]
    (js/Math.atan2 (/ s n) (/ c n))))

(defn- circular-variance
  "Compute circular variance (1 - R-bar) of a vector of angles."
  [angles]
  (let [n (count angles)
        s (reduce + (map js/Math.sin angles))
        c (reduce + (map js/Math.cos angles))
        r-bar (/ (js/Math.sqrt (+ (* s s) (* c c))) n)]
    (- 1.0 r-bar)))

;; ---------------------------------------------------------------------------
;; Von Mises tests
;; ---------------------------------------------------------------------------

(deftest von-mises-mean-direction
  (testing "von-mises mean direction recovery"
    (let [d (dist/von-mises 0 5)
          key (rng/fresh-key)
          samples (mapv (fn [k] (mx/item (dc/dist-sample d k)))
                        (rng/split-n key 2000))
          cm (circular-mean samples)]
      (is (h/close? 0.0 cm 0.15) "mean direction recovery (mu=0, kappa=5)")
      (is (every? #(and (>= % (- js/Math.PI)) (< % js/Math.PI)) samples)
          "all samples in [-pi, pi)"))))

(deftest von-mises-shifted-mean
  (testing "von-mises shifted mean"
    (mx/clear-cache!)
    (let [d (dist/von-mises 2.0 10)
          key (rng/fresh-key)
          samples (mapv (fn [k] (mx/item (dc/dist-sample d k)))
                        (rng/split-n key 2000))
          cm (circular-mean samples)]
      (is (h/close? 2.0 cm 0.15) "mean direction recovery (mu=2, kappa=10)"))))

(deftest von-mises-concentration
  (testing "von-mises concentration"
    (mx/clear-cache!)
    (let [d-lo (dist/von-mises 0 1)
          d-hi (dist/von-mises 0 20)
          key (rng/fresh-key)
          [k1 k2] (rng/split key)
          s-lo (mapv (fn [k] (mx/item (dc/dist-sample d-lo k))) (rng/split-n k1 1000))
          s-hi (mapv (fn [k] (mx/item (dc/dist-sample d-hi k))) (rng/split-n k2 1000))
          cv-lo (circular-variance s-lo)
          cv-hi (circular-variance s-hi)]
      (is (> cv-lo cv-hi) "higher kappa -> lower circular variance"))))

(deftest von-mises-log-prob-finite
  (testing "von-mises log-prob finite"
    (mx/clear-cache!)
    (let [d (dist/von-mises 1.0 3.0)
          lp (mx/item (dist/log-prob d (mx/scalar 1.0)))]
      (is (js/isFinite lp) "log-prob is finite"))))

(deftest von-mises-batch-sample
  (testing "von-mises dist-sample-n"
    (let [d (dist/von-mises 0 5)
          key (rng/fresh-key)
          batch (dc/dist-sample-n d key 50)]
      (is (= [50] (mx/shape batch)) "batch shape is [50]"))))

;; ---------------------------------------------------------------------------
;; Wrapped Cauchy tests
;; ---------------------------------------------------------------------------

(deftest wrapped-cauchy-mean-direction
  (testing "wrapped-cauchy mean direction"
    (mx/clear-cache!)
    (let [d (dist/wrapped-cauchy 0 0.5)
          key (rng/fresh-key)
          samples (mapv (fn [k] (mx/item (dc/dist-sample d k)))
                        (rng/split-n key 2000))
          cm (circular-mean samples)]
      (is (h/close? 0.0 cm 0.15) "mean direction recovery (mu=0, rho=0.5)")
      (is (every? #(and (>= % (- js/Math.PI)) (< % js/Math.PI)) samples)
          "all samples in [-pi, pi)"))))

(deftest wrapped-cauchy-shifted-mean
  (testing "wrapped-cauchy shifted mean"
    (mx/clear-cache!)
    (let [d (dist/wrapped-cauchy 1.5 0.8)
          key (rng/fresh-key)
          samples (mapv (fn [k] (mx/item (dc/dist-sample d k)))
                        (rng/split-n key 2000))
          cm (circular-mean samples)]
      (is (h/close? 1.5 cm 0.15) "mean direction recovery (mu=1.5, rho=0.8)"))))

(deftest wrapped-cauchy-concentration
  (testing "wrapped-cauchy concentration"
    (mx/clear-cache!)
    (let [d-lo (dist/wrapped-cauchy 0 0.2)
          d-hi (dist/wrapped-cauchy 0 0.9)
          key (rng/fresh-key)
          [k1 k2] (rng/split key)
          s-lo (mapv (fn [k] (mx/item (dc/dist-sample d-lo k))) (rng/split-n k1 1000))
          s-hi (mapv (fn [k] (mx/item (dc/dist-sample d-hi k))) (rng/split-n k2 1000))
          cv-lo (circular-variance s-lo)
          cv-hi (circular-variance s-hi)]
      (is (> cv-lo cv-hi) "higher rho -> lower circular variance"))))

(deftest wrapped-cauchy-log-prob-finite
  (testing "wrapped-cauchy log-prob finite"
    (mx/clear-cache!)
    (let [d (dist/wrapped-cauchy 0 0.5)
          lp (mx/item (dist/log-prob d (mx/scalar 0.5)))]
      (is (js/isFinite lp) "log-prob is finite"))))

(deftest wrapped-cauchy-batch-sample
  (testing "wrapped-cauchy dist-sample-n"
    (let [d (dist/wrapped-cauchy 0 0.5)
          key (rng/fresh-key)
          batch (dc/dist-sample-n d key 50)]
      (is (= [50] (mx/shape batch)) "batch shape is [50]"))))

;; ---------------------------------------------------------------------------
;; Wrapped Normal tests
;; ---------------------------------------------------------------------------

(deftest wrapped-normal-mean-direction
  (testing "wrapped-normal mean direction"
    (mx/clear-cache!)
    (let [d (dist/wrapped-normal 0 1.0)
          key (rng/fresh-key)
          samples (mapv (fn [k] (mx/item (dc/dist-sample d k)))
                        (rng/split-n key 2000))
          cm (circular-mean samples)]
      (is (h/close? 0.0 cm 0.15) "mean direction recovery (mu=0, sigma=1)")
      (is (every? #(and (>= % (- js/Math.PI)) (< % js/Math.PI)) samples)
          "all samples in [-pi, pi)"))))

(deftest wrapped-normal-shifted-mean
  (testing "wrapped-normal shifted mean"
    (mx/clear-cache!)
    (let [d (dist/wrapped-normal 2.0 0.5)
          key (rng/fresh-key)
          samples (mapv (fn [k] (mx/item (dc/dist-sample d k)))
                        (rng/split-n key 2000))
          cm (circular-mean samples)]
      (is (h/close? 2.0 cm 0.15) "mean direction recovery (mu=2, sigma=0.5)"))))

(deftest wrapped-normal-spread
  (testing "wrapped-normal spread"
    (mx/clear-cache!)
    (let [d-lo (dist/wrapped-normal 0 0.3)
          d-hi (dist/wrapped-normal 0 2.0)
          key (rng/fresh-key)
          [k1 k2] (rng/split key)
          s-lo (mapv (fn [k] (mx/item (dc/dist-sample d-lo k))) (rng/split-n k1 1000))
          s-hi (mapv (fn [k] (mx/item (dc/dist-sample d-hi k))) (rng/split-n k2 1000))
          cv-lo (circular-variance s-lo)
          cv-hi (circular-variance s-hi)]
      (is (< cv-lo cv-hi) "smaller sigma -> lower circular variance"))))

(deftest wrapped-normal-log-prob-finite
  (testing "wrapped-normal log-prob finite"
    (mx/clear-cache!)
    (let [d (dist/wrapped-normal 0 1.0)
          lp (mx/item (dist/log-prob d (mx/scalar 0.5)))]
      (is (js/isFinite lp) "log-prob is finite"))))

(deftest wrapped-normal-batch-sample
  (testing "wrapped-normal dist-sample-n"
    (mx/clear-cache!)
    (let [d (dist/wrapped-normal 0 1.0)
          key (rng/fresh-key)
          batch (dc/dist-sample-n d key 50)]
      (is (= [50] (mx/shape batch)) "batch shape is [50]"))))

;; ---------------------------------------------------------------------------
;; Parameter validation tests
;; ---------------------------------------------------------------------------

(deftest von-mises-parameter-validation
  (testing "von-mises rejects invalid kappa"
    (is (thrown? js/Error (dist/von-mises 0 0)) "von-mises rejects kappa=0")
    (is (thrown? js/Error (dist/von-mises 0 -1)) "von-mises rejects kappa<0")))

(deftest wrapped-cauchy-parameter-validation
  (testing "wrapped-cauchy rejects invalid rho"
    (is (thrown? js/Error (dist/wrapped-cauchy 0 0)) "wrapped-cauchy rejects rho=0")
    (is (thrown? js/Error (dist/wrapped-cauchy 0 1)) "wrapped-cauchy rejects rho=1")
    (is (thrown? js/Error (dist/wrapped-cauchy 0 -0.5)) "wrapped-cauchy rejects rho<0")))

(deftest wrapped-normal-parameter-validation
  (testing "wrapped-normal rejects invalid sigma"
    (is (thrown? js/Error (dist/wrapped-normal 0 0)) "wrapped-normal rejects sigma=0")
    (is (thrown? js/Error (dist/wrapped-normal 0 -1)) "wrapped-normal rejects sigma<0")))

;; ---------------------------------------------------------------------------
;; Edge case tests
;; ---------------------------------------------------------------------------

(deftest von-mises-small-kappa
  (testing "von-mises very small kappa (0.01)"
    (let [d (dist/von-mises 0 0.01)
          key (rng/fresh-key)
          s (mx/item (dc/dist-sample d key))]
      (is (js/isFinite s) "sample is finite with small kappa"))))

(deftest von-mises-large-kappa
  (testing "von-mises very large kappa (100)"
    (mx/clear-cache!)
    (let [d (dist/von-mises 1.0 100)
          key (rng/fresh-key)
          samples (mapv (fn [k] (mx/item (dc/dist-sample d k)))
                        (rng/split-n key 200))
          cm (circular-mean samples)]
      (is (h/close? 1.0 cm 0.1) "concentrated near mu=1 with large kappa"))))

(deftest wrapped-cauchy-extreme-rho
  (testing "wrapped-cauchy rho near 0 and near 1"
    (mx/clear-cache!)
    (let [d-low (dist/wrapped-cauchy 0 0.01)
          d-high (dist/wrapped-cauchy 0 0.99)
          key (rng/fresh-key)
          [k1 k2] (rng/split key)
          s1 (mx/item (dc/dist-sample d-low k1))
          s2 (mx/item (dc/dist-sample d-high k2))]
      (is (js/isFinite s1) "sample finite with rho near 0")
      (is (js/isFinite s2) "sample finite with rho near 1"))))

;; ---------------------------------------------------------------------------
;; Von Mises log-prob numerical accuracy
;; ---------------------------------------------------------------------------

(deftest von-mises-log-prob-accuracy
  (testing "von-mises log-prob numerical accuracy"
    (let [d (dist/von-mises 0 2)
          lp (mx/item (dist/log-prob d (mx/scalar 1.0)))
          ref-i0 (+ 1.0 1.0 0.25 (/ 1.0 36.0) (/ 1.0 576.0) (/ 1.0 14400.0) (/ 1.0 518400.0))
          ref-lp (- (* 2.0 (js/Math.cos 1.0)) (js/Math.log (* 2.0 js/Math.PI)) (js/Math.log ref-i0))]
      (is (h/close? ref-lp lp 0.01) "von-mises log-prob at x=1, mu=0, kappa=2"))))

(deftest von-mises-log-prob-mode
  (testing "von-mises log-prob at mode vs off mode"
    (let [d (dist/von-mises 0 5)
          lp-mode (mx/item (dist/log-prob d (mx/scalar 0.0)))
          lp-off  (mx/item (dist/log-prob d (mx/scalar 1.0)))]
      (is (> lp-mode lp-off) "log-prob at mode > log-prob off mode"))))

(deftest von-mises-log-prob-symmetry
  (testing "von-mises log-prob symmetry"
    (let [d (dist/von-mises 1.0 3.0)
          lp-plus  (mx/item (dist/log-prob d (mx/scalar 1.5)))
          lp-minus (mx/item (dist/log-prob d (mx/scalar 0.5)))]
      (is (h/close? lp-plus lp-minus 1e-5) "log-prob symmetric around mu"))))

;; ---------------------------------------------------------------------------
;; Von Mises GFI integration (generate, update, assess)
;; ---------------------------------------------------------------------------

(deftest von-mises-simulate
  (testing "von-mises simulate"
    (mx/clear-cache!)
    (let [model (dyn/auto-key (gen []
                  (let [angle (trace :angle (dist/von-mises 0 5))]
                    angle)))
          tr (p/simulate model [])
          angle-val (cm/get-value (cm/get-submap (:choices tr) :angle))]
      (mx/eval! angle-val)
      (is (js/isFinite (mx/item angle-val)) "simulate: angle is finite")
      (is (js/isFinite (mx/item (:score tr))) "simulate: score is finite"))))

(deftest von-mises-generate
  (testing "von-mises generate with constraint"
    (mx/clear-cache!)
    (let [model (dyn/auto-key (gen []
                  (let [angle (trace :angle (dist/von-mises 0 5))]
                    angle)))
          constraints (cm/choicemap :angle (mx/scalar 1.0))
          {:keys [trace weight]} (p/generate model [] constraints)]
      (let [v (mx/item (cm/get-value (cm/get-submap (:choices trace) :angle)))]
        (is (h/close? 1.0 v 1e-5) "generate: constrained angle = 1.0"))
      (mx/eval! weight)
      (is (js/isFinite (mx/item weight)) "generate: weight is finite"))))

(deftest von-mises-assess
  (testing "von-mises assess"
    (mx/clear-cache!)
    (let [model (dyn/auto-key (gen []
                  (let [angle (trace :angle (dist/von-mises 0 5))]
                    angle)))
          choices (cm/choicemap :angle (mx/scalar 0.5))
          {:keys [weight]} (p/assess model [] choices)]
      (mx/eval! weight)
      (is (js/isFinite (mx/item weight)) "assess: weight is finite"))))

(deftest von-mises-update
  (testing "von-mises update"
    (mx/clear-cache!)
    (let [model (dyn/auto-key (gen []
                  (let [angle (trace :angle (dist/von-mises 0 5))]
                    angle)))
          constraints (cm/choicemap :angle (mx/scalar 1.0))
          {:keys [trace]} (p/generate model [] constraints)
          new-constraints (cm/choicemap :angle (mx/scalar 2.0))
          {:keys [trace weight discard]} (p/update model trace new-constraints)]
      (let [v (mx/item (cm/get-value (cm/get-submap (:choices trace) :angle)))]
        (is (h/close? 2.0 v 1e-5) "update: angle changed to 2.0"))
      (mx/eval! weight)
      (is (js/isFinite (mx/item weight)) "update: weight is finite")
      (is (some? discard) "update: discard returned"))))

;; ---------------------------------------------------------------------------
;; Von Mises gradient test
;; ---------------------------------------------------------------------------

(deftest von-mises-gradient-at-mode
  (testing "von-mises gradient at mode"
    (mx/clear-cache!)
    (let [model (dyn/auto-key (gen []
                  (trace :angle (dist/von-mises 0 5))))
          {:keys [trace]} (p/generate model [] (cm/choicemap :angle (mx/scalar 0.0)))
          grads (grad/choice-gradients model trace [:angle])
          g (mx/item (:angle grads))]
      (is (h/close? 0.0 g 0.1) "gradient at mode ~ 0"))))

(deftest von-mises-gradient-off-mode
  (testing "von-mises gradient off mode"
    (mx/clear-cache!)
    (let [model (dyn/auto-key (gen []
                  (trace :angle (dist/von-mises 0 5))))
          {:keys [trace]} (p/generate model [] (cm/choicemap :angle (mx/scalar 1.0)))
          grads (grad/choice-gradients model trace [:angle])
          g (mx/item (:angle grads))]
      (is (js/isFinite g) "gradient off-mode is finite")
      (is (> (js/Math.abs g) 0.1) "gradient off-mode is non-zero"))))

;; ---------------------------------------------------------------------------
;; Von Mises vectorized inference (vsimulate, vgenerate)
;; ---------------------------------------------------------------------------

(deftest von-mises-vsimulate
  (testing "von-mises vsimulate"
    (mx/clear-cache!)
    (let [model (gen []
                  (trace :angle (dist/von-mises 0 5))
                  nil)
          n 50
          key (rng/fresh-key)
          vtrace (dyn/vsimulate model [] n key)]
      (is (instance? vec/VectorizedTrace vtrace) "vsimulate: returns VectorizedTrace")
      (let [v (cm/get-value (cm/get-submap (:choices vtrace) :angle))]
        (mx/eval! v)
        (is (= [n] (mx/shape v)) "vsimulate: angle shape is [N]"))
      (let [score (:score vtrace)]
        (mx/eval! score)
        (is (= [n] (mx/shape score)) "vsimulate: score shape is [N]")))))

(deftest von-mises-vgenerate
  (testing "von-mises vgenerate"
    (mx/clear-cache!)
    (let [model (gen []
                  (let [angle (trace :angle (dist/von-mises 0 5))]
                    (trace :obs (dist/gaussian (mx/cos angle) 0.1))
                    nil))
          n 50
          key (rng/fresh-key)
          obs (cm/choicemap :obs (mx/scalar 0.5))
          vtrace (dyn/vgenerate model [] obs n key)]
      (is (instance? vec/VectorizedTrace vtrace) "vgenerate: returns VectorizedTrace")
      (let [v (cm/get-value (cm/get-submap (:choices vtrace) :angle))]
        (mx/eval! v)
        (is (= [n] (mx/shape v)) "vgenerate: angle shape is [N]"))
      (let [w (:weight vtrace)]
        (mx/eval! w)
        (is (= [n] (mx/shape w)) "vgenerate: weight shape is [N]")))))

(cljs.test/run-tests)
