(ns genmlx.vectorized-test
  (:require [genmlx.mlx :as mx]
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

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (do (println "  FAIL:" msg "- expected truthy")
        (println "    actual:" actual))))

(defn assert-equal [msg expected actual]
  (if (= expected actual)
    (println "  PASS:" msg)
    (do (println "  FAIL:" msg)
        (println "    expected:" expected)
        (println "    actual:  " actual))))

(defn assert-close [msg expected actual tolerance]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tolerance)
      (println "  PASS:" msg)
      (do (println "  FAIL:" msg)
          (println "    expected:" expected "+/-" tolerance)
          (println "    actual:  " actual)))))

(println "\n=== Vectorized Inference Tests ===\n")

;; ---------------------------------------------------------------------------
;; Step 1: dist-sample-n shape correctness
;; ---------------------------------------------------------------------------

(println "-- dist-sample-n shape correctness --")

(let [n 50
      key (rng/fresh-key)]

  ;; Gaussian
  (let [d (dist/gaussian 0 1)
        samples (dc/dist-sample-n d key n)]
    (mx/eval! samples)
    (assert-equal "gaussian shape" [n] (mx/shape samples)))

  ;; Uniform
  (let [d (dist/uniform 0 1)
        samples (dc/dist-sample-n d key n)]
    (mx/eval! samples)
    (assert-equal "uniform shape" [n] (mx/shape samples)))

  ;; Bernoulli
  (let [d (dist/bernoulli 0.5)
        samples (dc/dist-sample-n d key n)]
    (mx/eval! samples)
    (assert-equal "bernoulli shape" [n] (mx/shape samples)))

  ;; Exponential
  (let [d (dist/exponential 2.0)
        samples (dc/dist-sample-n d key n)]
    (mx/eval! samples)
    (assert-equal "exponential shape" [n] (mx/shape samples)))

  ;; Laplace
  (let [d (dist/laplace 0 1)
        samples (dc/dist-sample-n d key n)]
    (mx/eval! samples)
    (assert-equal "laplace shape" [n] (mx/shape samples)))

  ;; Log-Normal
  (let [d (dist/log-normal 0 1)
        samples (dc/dist-sample-n d key n)]
    (mx/eval! samples)
    (assert-equal "log-normal shape" [n] (mx/shape samples)))

  ;; Delta
  (let [d (dist/delta (mx/scalar 42.0))
        samples (dc/dist-sample-n d key n)]
    (mx/eval! samples)
    (assert-equal "delta shape" [n] (mx/shape samples))
    (assert-close "delta all same" 42.0 (mx/realize (mx/index samples 0)) 0.001)))

;; ---------------------------------------------------------------------------
;; Step 2: log-prob broadcasting with [N]-shaped values
;; ---------------------------------------------------------------------------

(println "\n-- log-prob broadcasting --")

(let [n 50
      key (rng/fresh-key)]

  ;; Gaussian log-prob with [N] values should return [N]
  (let [d (dist/gaussian 0 1)
        samples (dc/dist-sample-n d key n)
        lp (dc/dist-log-prob d samples)]
    (mx/eval! lp)
    (assert-equal "gaussian log-prob shape" [n] (mx/shape lp))
    ;; All log-probs should be negative (for standard normal)
    (let [max-lp (mx/realize (mx/amax lp))]
      (assert-true "gaussian log-probs negative" (< max-lp 0.01))))

  ;; Uniform log-prob
  (let [d (dist/uniform 0 1)
        samples (dc/dist-sample-n d key n)
        lp (dc/dist-log-prob d samples)]
    (mx/eval! lp)
    (assert-equal "uniform log-prob shape" [n] (mx/shape lp))
    ;; log(1/(1-0)) = 0 for all points in [0,1]
    (assert-close "uniform log-prob value" 0.0 (mx/realize (mx/mean lp)) 0.001)))

;; ---------------------------------------------------------------------------
;; Step 3: vsimulate shape correctness
;; ---------------------------------------------------------------------------

(println "\n-- vsimulate --")

(let [model (gen []
              (dyn/trace :x (dist/gaussian 0 1))
              (dyn/trace :y (dist/uniform -1 1))
              nil)
      n 100
      key (rng/fresh-key)
      vtrace (dyn/vsimulate model [] n key)]
  (assert-true "vsimulate returns VectorizedTrace"
               (instance? vec/VectorizedTrace vtrace))
  (assert-equal "vsimulate n-particles" n (:n-particles vtrace))

  ;; Check choice shapes
  (let [x-val (cm/get-value (cm/get-submap (:choices vtrace) :x))
        y-val (cm/get-value (cm/get-submap (:choices vtrace) :y))]
    (mx/eval! x-val y-val)
    (assert-equal "vsimulate :x shape" [n] (mx/shape x-val))
    (assert-equal "vsimulate :y shape" [n] (mx/shape y-val)))

  ;; Score should be [N]-shaped
  (let [score (:score vtrace)]
    (mx/eval! score)
    (assert-equal "vsimulate score shape" [n] (mx/shape score))))

;; ---------------------------------------------------------------------------
;; Step 4: vgenerate with constraints
;; ---------------------------------------------------------------------------

(println "\n-- vgenerate --")

(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))]
                (dyn/trace :y (dist/gaussian x 0.1))
                nil))
      n 100
      key (rng/fresh-key)
      obs (cm/choicemap :y (mx/scalar 2.0))
      vtrace (dyn/vgenerate model [] obs n key)]
  (assert-true "vgenerate returns VectorizedTrace"
               (instance? vec/VectorizedTrace vtrace))

  ;; :x should be [N]-shaped (unconstrained)
  (let [x-val (cm/get-value (cm/get-submap (:choices vtrace) :x))]
    (mx/eval! x-val)
    (assert-equal "vgenerate :x shape" [n] (mx/shape x-val)))

  ;; :y should be scalar (constrained)
  (let [y-val (cm/get-value (cm/get-submap (:choices vtrace) :y))]
    (mx/eval! y-val)
    (assert-close "vgenerate :y is constrained" 2.0 (mx/realize y-val) 0.001))

  ;; Weight should be [N]-shaped
  (let [w (:weight vtrace)]
    (mx/eval! w)
    (assert-equal "vgenerate weight shape" [n] (mx/shape w)))

  ;; Log ML estimate
  (let [log-ml (vec/vtrace-log-ml-estimate vtrace)]
    (mx/eval! log-ml)
    (assert-true "vgenerate log-ml is finite" (js/isFinite (mx/realize log-ml))))

  ;; ESS
  (let [ess (vec/vtrace-ess vtrace)]
    (assert-true "vgenerate ESS > 0" (> ess 0))
    (assert-true "vgenerate ESS <= N" (<= ess n))))

;; ---------------------------------------------------------------------------
;; Step 5: Statistical equivalence — sequential vs batched
;; ---------------------------------------------------------------------------

(println "\n-- statistical equivalence --")

(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 5.0 2.0))]
                x))
      n 500

      ;; Sequential: N separate generate calls
      seq-results (mapv (fn [_] (p/generate model [] cm/EMPTY)) (range n))
      seq-scores  (mapv (fn [r] (mx/realize (:score (:trace r)))) seq-results)
      seq-mean-score (/ (reduce + seq-scores) n)

      ;; Batched: single vsimulate
      key (rng/fresh-key)
      vtrace (dyn/vsimulate model [] n key)
      batch-scores (:score vtrace)
      _ (mx/eval! batch-scores)
      batch-mean-score (mx/realize (mx/mean batch-scores))]

  ;; Mean scores should be similar (both are log-probs of N(5,2))
  (assert-close "mean score similar (within 0.5)" seq-mean-score batch-mean-score 0.5))

;; ---------------------------------------------------------------------------
;; Step 6: resample-vtrace
;; ---------------------------------------------------------------------------

(println "\n-- resample-vtrace --")

(let [model (gen []
              (dyn/trace :x (dist/gaussian 0 1))
              nil)
      n 50
      key (rng/fresh-key)
      [k1 k2] (rng/split key)
      vtrace (dyn/vsimulate model [] n k1)
      resampled (vec/resample-vtrace vtrace k2)]
  (assert-true "resampled is VectorizedTrace"
               (instance? vec/VectorizedTrace resampled))
  (let [x-val (cm/get-value (cm/get-submap (:choices resampled) :x))]
    (mx/eval! x-val)
    (assert-equal "resampled :x shape" [n] (mx/shape x-val)))
  ;; After resampling, weights should be uniform (zeros)
  (let [w (:weight resampled)]
    (mx/eval! w)
    (assert-close "resampled weights are zero" 0.0 (mx/realize (mx/mean w)) 0.001)))

;; ---------------------------------------------------------------------------
;; Step 7: vectorized-importance-sampling
;; ---------------------------------------------------------------------------

(println "\n-- vectorized importance sampling --")

(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 10))]
                (dyn/trace :obs (dist/gaussian x 1))
                x))
      obs (cm/choicemap :obs (mx/scalar 5.0))
      {:keys [vtrace log-ml-estimate]}
      (is/vectorized-importance-sampling {:samples 200} model [] obs)]
  (assert-true "vis returns VectorizedTrace"
               (instance? vec/VectorizedTrace vtrace))
  (mx/eval! log-ml-estimate)
  (assert-true "vis log-ml is finite" (js/isFinite (mx/realize log-ml-estimate)))
  ;; ESS should be reasonable
  (let [ess (vec/vtrace-ess vtrace)]
    (assert-true "vis ESS > 1" (> ess 1))))

;; ---------------------------------------------------------------------------
;; Step 8: vsmc-init
;; ---------------------------------------------------------------------------

(println "\n-- vsmc-init --")

(let [model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 10))]
                (dyn/trace :obs (dist/gaussian x 1))
                x))
      obs (cm/choicemap :obs (mx/scalar 3.0))
      {:keys [vtrace log-ml-estimate]}
      (smc/vsmc-init model [] obs 100 nil)]
  (assert-true "vsmc-init returns VectorizedTrace"
               (instance? vec/VectorizedTrace vtrace))
  (mx/eval! log-ml-estimate)
  (assert-true "vsmc-init log-ml is finite" (js/isFinite (mx/realize log-ml-estimate))))

;; ---------------------------------------------------------------------------
;; Step 9: vectorized splice (batched sub-GF calls)
;; ---------------------------------------------------------------------------

(println "\n-- vectorized splice: vsimulate --")

(let [sub-model (gen []
                  (dyn/trace :z (dist/gaussian 0 1))
                  (dyn/trace :w (dist/uniform -1 1))
                  nil)
      model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 10))]
                (dyn/splice :sub sub-model)
                x))
      n 50
      key (rng/fresh-key)
      vtrace (dyn/vsimulate model [] n key)]
  (assert-true "vsimulate+splice returns VectorizedTrace"
               (instance? vec/VectorizedTrace vtrace))
  ;; Top-level :x should be [N]-shaped
  (let [x-val (cm/get-value (cm/get-submap (:choices vtrace) :x))]
    (mx/eval! x-val)
    (assert-equal "splice vsimulate :x shape" [n] (mx/shape x-val)))
  ;; Nested :sub :z and :sub :w should be [N]-shaped
  (let [sub-cm (cm/get-submap (:choices vtrace) :sub)
        z-val (cm/get-value (cm/get-submap sub-cm :z))
        w-val (cm/get-value (cm/get-submap sub-cm :w))]
    (mx/eval! z-val w-val)
    (assert-equal "splice vsimulate :sub :z shape" [n] (mx/shape z-val))
    (assert-equal "splice vsimulate :sub :w shape" [n] (mx/shape w-val)))
  ;; Score should be [N]-shaped
  (let [score (:score vtrace)]
    (mx/eval! score)
    (assert-equal "splice vsimulate score shape" [n] (mx/shape score))))

(println "\n-- vectorized splice: vgenerate --")

(let [sub-model (gen [mu]
                  (dyn/trace :z (dist/gaussian mu 1))
                  nil)
      model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 10))]
                (dyn/splice :sub sub-model x)
                x))
      n 50
      key (rng/fresh-key)
      ;; Constrain the sub-model's :z site via hierarchical choicemap
      obs (cm/choicemap :sub (cm/choicemap :z (mx/scalar 2.0)))
      vtrace (dyn/vgenerate model [] obs n key)]
  (assert-true "vgenerate+splice returns VectorizedTrace"
               (instance? vec/VectorizedTrace vtrace))
  ;; :x should be [N]-shaped (unconstrained)
  (let [x-val (cm/get-value (cm/get-submap (:choices vtrace) :x))]
    (mx/eval! x-val)
    (assert-equal "splice vgenerate :x shape" [n] (mx/shape x-val)))
  ;; :sub :z should be constrained scalar
  (let [sub-cm (cm/get-submap (:choices vtrace) :sub)
        z-val (cm/get-value (cm/get-submap sub-cm :z))]
    (mx/eval! z-val)
    (assert-close "splice vgenerate :sub :z constrained" 2.0 (mx/realize z-val) 0.001))
  ;; Weight should be [N]-shaped (log-prob depends on [N]-shaped x)
  (let [w (:weight vtrace)]
    (mx/eval! w)
    (assert-equal "splice vgenerate weight shape" [n] (mx/shape w))))

(println "\n-- vectorized splice: vupdate --")

(let [sub-model (gen []
                  (dyn/trace :z (dist/gaussian 0 1))
                  nil)
      model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 10))]
                (dyn/splice :sub sub-model)
                x))
      n 50
      key (rng/fresh-key)
      [k1 k2] (rng/split key)
      vtrace (dyn/vsimulate model [] n k1)
      ;; Update: constrain :sub :z to a new value
      new-obs (cm/choicemap :sub (cm/choicemap :z (mx/scalar 3.0)))
      {:keys [vtrace weight]} (dyn/vupdate model vtrace new-obs k2)]
  (assert-true "vupdate+splice returns VectorizedTrace"
               (instance? vec/VectorizedTrace vtrace))
  ;; :sub :z should now be 3.0
  (let [sub-cm (cm/get-submap (:choices vtrace) :sub)
        z-val (cm/get-value (cm/get-submap sub-cm :z))]
    (mx/eval! z-val)
    (assert-close "splice vupdate :sub :z updated" 3.0 (mx/realize z-val) 0.001))
  ;; Weight should be [N]-shaped
  (mx/eval! weight)
  (assert-equal "splice vupdate weight shape" [n] (mx/shape weight)))

(println "\n-- vectorized splice: vregenerate --")

(let [sub-model (gen []
                  (dyn/trace :z (dist/gaussian 0 1))
                  nil)
      model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 10))]
                (dyn/splice :sub sub-model)
                x))
      n 50
      key (rng/fresh-key)
      [k1 k2] (rng/split key)
      vtrace (dyn/vsimulate model [] n k1)
      ;; Regenerate: resample :sub :z
      sel (sel/hierarchical :sub (sel/select :z))
      {:keys [vtrace weight]} (dyn/vregenerate model vtrace sel k2)]
  (assert-true "vregenerate+splice returns VectorizedTrace"
               (instance? vec/VectorizedTrace vtrace))
  ;; :sub :z should be [N]-shaped (resampled)
  (let [sub-cm (cm/get-submap (:choices vtrace) :sub)
        z-val (cm/get-value (cm/get-submap sub-cm :z))]
    (mx/eval! z-val)
    (assert-equal "splice vregenerate :sub :z shape" [n] (mx/shape z-val)))
  ;; Weight should be [N]-shaped
  (mx/eval! weight)
  (assert-equal "splice vregenerate weight shape" [n] (mx/shape weight)))

(println "\n-- vectorized splice: non-DynamicGF guard --")

(let [;; A distribution used as a GF has no :body-fn
      model (gen []
              (dyn/splice :d (dist/gaussian 0 1))
              nil)
      caught? (try
                (dyn/vsimulate model [] 10 nil)
                false
                (catch :default e
                  (boolean (re-find #"non-DynamicGF" (.-message e)))))]
  (assert-true "non-DynamicGF splice in batched mode throws" caught?))

(println "\n-- vectorized splice: nested (3 levels) --")

(let [inner (gen []
              (dyn/trace :a (dist/gaussian 0 1)))
      middle (gen []
               (dyn/trace :b (dist/uniform -1 1))
               (dyn/splice :inner inner))
      outer (gen []
              (dyn/trace :c (dist/exponential 1.0))
              (dyn/splice :mid middle)
              nil)
      n 50
      key (rng/fresh-key)
      vtrace (dyn/vsimulate outer [] n key)]
  (assert-true "nested splice returns VectorizedTrace"
               (instance? vec/VectorizedTrace vtrace))
  ;; :c at top level
  (let [c-val (cm/get-value (cm/get-submap (:choices vtrace) :c))]
    (mx/eval! c-val)
    (assert-equal "nested splice :c shape" [n] (mx/shape c-val)))
  ;; :mid :b
  (let [mid-cm (cm/get-submap (:choices vtrace) :mid)
        b-val (cm/get-value (cm/get-submap mid-cm :b))]
    (mx/eval! b-val)
    (assert-equal "nested splice :mid :b shape" [n] (mx/shape b-val)))
  ;; :mid :inner :a
  (let [mid-cm (cm/get-submap (:choices vtrace) :mid)
        inner-cm (cm/get-submap mid-cm :inner)
        a-val (cm/get-value (cm/get-submap inner-cm :a))]
    (mx/eval! a-val)
    (assert-equal "nested splice :mid :inner :a shape" [n] (mx/shape a-val)))
  ;; Score should be [N]-shaped
  (let [score (:score vtrace)]
    (mx/eval! score)
    (assert-equal "nested splice score shape" [n] (mx/shape score))))

;; ---------------------------------------------------------------------------
;; Step 10: sequential fallback for beta (non-batchable)
;; ---------------------------------------------------------------------------

(println "\n-- sequential fallback (beta) --")

(let [d (dist/beta-dist 2 5)
      key (rng/fresh-key)
      n 20
      samples (dc/dist-sample-n d key n)]
  (mx/eval! samples)
  (assert-equal "beta fallback shape" [n] (mx/shape samples))
  ;; All samples should be in (0, 1)
  (let [min-val (mx/realize (mx/amin samples))
        max-val (mx/realize (mx/amax samples))]
    (assert-true "beta samples in (0,1)" (and (> min-val 0) (< max-val 1)))))

(println "\nAll vectorized tests complete.")
