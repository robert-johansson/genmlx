(ns genmlx.iid-conjugacy-test
  "M2 Step 4: Conjugacy + auto-analytical for iid-gaussian.
   Verifies normal-iid-normal conjugate pair detection, update math,
   handler integration, and end-to-end p/generate with auto-analytical."
  (:require [genmlx.inference.auto-analytical :as aa]
            [genmlx.conjugacy :as conj]
            [genmlx.handler :as h]
            [genmlx.runtime :as rt]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def ^:private pass-count (volatile! 0))
(def ^:private fail-count (volatile! 0))

(defn- assert-true [desc pred]
  (if pred
    (do (vswap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (vswap! fail-count inc)
        (println (str "  FAIL: " desc)))))

(defn- assert-equal [desc expected actual]
  (if (= expected actual)
    (do (vswap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (vswap! fail-count inc)
        (println (str "  FAIL: " desc " — expected " expected ", got " actual)))))

(defn- assert-close [desc expected actual tol]
  (let [e (if (number? expected) expected (mx/item expected))
        a (if (number? actual) actual (mx/item actual))
        diff (js/Math.abs (- e a))]
    (if (<= diff tol)
      (do (vswap! pass-count inc)
          (println (str "  PASS: " desc " (diff=" (.toExponential diff 2) ")")))
      (do (vswap! fail-count inc)
          (println (str "  FAIL: " desc " — expected " e ", got " a " (diff=" diff ")"))))))

;; ---------------------------------------------------------------------------
;; 1. Conjugacy table entry
;; ---------------------------------------------------------------------------

(println "\n=== 1. Conjugacy table: :gaussian + :iid-gaussian ===")

(let [entry (get conj/conjugacy-table [:gaussian :iid-gaussian])]
  (assert-true "entry exists" (some? entry))
  (assert-equal "family" :normal-iid-normal (:family entry))
  (assert-equal "natural-param-idx" 0 (:natural-param-idx entry))
  (assert-equal "prior-mean-key" :mu (:prior-mean-key entry))
  (assert-equal "prior-std-key" :sigma (:prior-std-key entry))
  (assert-equal "obs-mean-key" :mu (:obs-mean-key entry))
  (assert-equal "obs-noise-key" :sigma (:obs-noise-key entry)))

;; ---------------------------------------------------------------------------
;; 2. Conjugate pair detection on iid model
;; ---------------------------------------------------------------------------

(println "\n=== 2. Conjugate pair detection ===")

(def iid-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :ys (dist/iid-gaussian mu (mx/scalar 1.0) 5))
      mu)))

(let [pairs (conj/detect-conjugate-pairs (:schema iid-model))]
  (assert-equal "1 conjugate pair" 1 (count pairs))
  (let [pair (first pairs)]
    (assert-equal "prior-addr" :mu (:prior-addr pair))
    (assert-equal "obs-addr" :ys (:obs-addr pair))
    (assert-equal "family" :normal-iid-normal (:family pair))
    (assert-equal "dep-type direct" :direct (get-in pair [:dependency-type :type]))))

;; Augmented schema
(let [aug (conj/augment-schema-with-conjugacy (:schema iid-model))]
  (assert-true "has-conjugate?" (:has-conjugate? aug))
  (assert-equal "conjugate-pairs count" 1 (count (:conjugate-pairs aug))))

;; ---------------------------------------------------------------------------
;; 3. nn-iid-update-step math correctness
;; ---------------------------------------------------------------------------

(println "\n=== 3. nn-iid-update-step math ===")

;; Prior: N(0, 100). Obs: ys=[1,2,3,4,5], sigma=1.
;; Posterior precision = 1/100 + 5/1 = 5.01
;; Posterior var = 1/5.01 ≈ 0.1996
;; Posterior mean = 0.1996 * (0/100 + 15/1) = 0.1996 * 15 ≈ 2.994
(let [prior {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
      obs (mx/array [1.0 2.0 3.0 4.0 5.0])
      obs-var (mx/scalar 1.0)
      result (aa/nn-iid-update-step prior obs obs-var)]
  (mx/eval!)
  (assert-close "posterior mean ≈ 2.994" 2.994 (:mean result) 0.01)
  (assert-close "posterior var ≈ 0.1996" 0.1996 (:var result) 0.01)
  (assert-true "ll is finite" (js/isFinite (mx/item (:ll result))))
  (assert-true "ll is negative" (neg? (mx/item (:ll result)))))

;; Single observation: should match nn-update-step
(let [prior {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
      obs-single (mx/array [3.0])
      obs-var (mx/scalar 1.0)
      iid-result (aa/nn-iid-update-step prior obs-single obs-var)
      scalar-result (aa/nn-update-step prior (mx/scalar 3.0) obs-var)]
  (mx/eval!)
  (assert-close "T=1: mean matches nn-update" (mx/item (:mean scalar-result)) (mx/item (:mean iid-result)) 1e-6)
  (assert-close "T=1: var matches nn-update" (mx/item (:var scalar-result)) (mx/item (:var iid-result)) 1e-6)
  (assert-close "T=1: ll matches nn-update" (mx/item (:ll scalar-result)) (mx/item (:ll iid-result)) 1e-6))

;; Large T: posterior should be tight around sample mean
(let [prior {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
      obs (mx/array (vec (repeat 100 5.0)))
      obs-var (mx/scalar 1.0)
      result (aa/nn-iid-update-step prior obs obs-var)]
  (mx/eval!)
  ;; Posterior precision = 1/100 + 100/1 = 100.01
  ;; Posterior var ≈ 0.01, Posterior mean ≈ 4.995
  (assert-close "T=100: posterior mean ≈ 5.0" 5.0 (:mean result) 0.01)
  (assert-close "T=100: posterior var ≈ 0.01" 0.01 (:var result) 0.001))

;; ---------------------------------------------------------------------------
;; 4. Handler integration: build-auto-handlers with iid pair
;; ---------------------------------------------------------------------------

(println "\n=== 4. Handler integration ===")

(let [pairs [{:prior-addr :mu :obs-addr :ys :family :normal-iid-normal}]
      handlers (aa/build-auto-handlers pairs)]
  (assert-true "has :mu handler" (contains? handlers :mu))
  (assert-true "has :ys handler" (contains? handlers :ys))
  (assert-equal "2 handlers total" 2 (count handlers)))

;; ---------------------------------------------------------------------------
;; 5. run-handler with auto-analytical transition (iid-gaussian obs)
;; ---------------------------------------------------------------------------

(println "\n=== 5. run-handler + iid-gaussian ===")

(let [model (dyn/auto-key
              (gen []
                (let [mu (trace :mu (dist/gaussian 0 10))]
                  (trace :ys (dist/iid-gaussian mu (mx/scalar 1.0) 5))
                  mu)))
      pairs (conj/detect-conjugate-pairs (:schema model))
      handlers (aa/build-auto-handlers pairs)
      transition (aa/make-address-dispatch h/generate-transition handlers)
      obs-data (mx/array [1.0 2.0 3.0 4.0 5.0])
      constraints (-> cm/EMPTY (cm/set-value :ys obs-data))
      init {:choices cm/EMPTY :score (mx/scalar 0.0) :weight (mx/scalar 0.0)
            :key (rng/fresh-key 42) :constraints constraints :auto-posteriors {}}
      result (rt/run-handler transition init
               (fn [rt] (apply (:body-fn model) rt [])))]
  (mx/eval!)
  ;; Weight should be the marginal LL
  (assert-true "weight is finite" (js/isFinite (mx/item (:weight result))))
  (assert-true "weight is negative" (neg? (mx/item (:weight result))))
  (assert-true "score is finite" (js/isFinite (mx/item (:score result))))
  ;; Choices should contain :mu (posterior mean) and :ys (observed)
  (assert-true "choices has :mu" (some? (cm/get-submap (:choices result) :mu)))
  (assert-true "choices has :ys" (some? (cm/get-submap (:choices result) :ys)))
  ;; :mu should be posterior mean
  (let [mu-val (mx/item (cm/get-value (cm/get-submap (:choices result) :mu)))
        ;; Reference: posterior mean
        ref (aa/nn-iid-update-step {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
                                    obs-data (mx/scalar 1.0))]
    (mx/eval!)
    (assert-close "mu = posterior mean" (mx/item (:mean ref)) mu-val 1e-6)
    (assert-close "weight = marginal LL" (mx/item (:ll ref)) (mx/item (:weight result)) 1e-6)
    (assert-close "score = marginal LL" (mx/item (:ll ref)) (mx/item (:score result)) 1e-6)))

;; ---------------------------------------------------------------------------
;; 6. End-to-end: p/generate with auto-analytical elimination
;; ---------------------------------------------------------------------------

(println "\n=== 6. p/generate end-to-end ===")

(def iid-model-e2e
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :ys (dist/iid-gaussian mu (mx/scalar 1.0) 5))
      mu)))

(let [gf (dyn/auto-key iid-model-e2e)
      obs (cm/choicemap :ys (mx/array [1.0 2.0 3.0 4.0 5.0]))
      result (p/generate gf [] obs)]
  (mx/eval!)
  (assert-true "e2e: weight is finite" (js/isFinite (mx/item (:weight result))))
  (assert-true "e2e: weight is negative" (neg? (mx/item (:weight result))))
  ;; The weight from auto-analytical is the marginal LL
  ;; Reference calculation
  (let [ref (aa/nn-iid-update-step {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
                                    (mx/array [1.0 2.0 3.0 4.0 5.0])
                                    (mx/scalar 1.0))]
    (mx/eval!)
    (assert-close "e2e: weight = marginal LL" (mx/item (:ll ref)) (mx/item (:weight result)) 1e-4)))

;; ---------------------------------------------------------------------------
;; 7. Multi-obs: iid-gaussian + scalar gaussian on same prior
;; ---------------------------------------------------------------------------

(println "\n=== 7. Mixed iid + scalar obs ===")

(def mixed-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :ys (dist/iid-gaussian mu (mx/scalar 1.0) 3))
      (trace :y-extra (dist/gaussian mu 1))
      mu)))

(let [pairs (conj/detect-conjugate-pairs (:schema mixed-model))]
  ;; Should detect 2 pairs: mu->ys (normal-iid-normal) and mu->y-extra (normal-normal)
  (assert-equal "mixed: 2 pairs detected" 2 (count pairs))
  (let [families (set (map :family pairs))]
    (assert-true "mixed: has normal-iid-normal" (contains? families :normal-iid-normal))
    (assert-true "mixed: has normal-normal" (contains? families :normal-normal))))

;; ---------------------------------------------------------------------------
;; 8. Regenerate handlers for iid-gaussian
;; ---------------------------------------------------------------------------

(println "\n=== 8. Regenerate handlers ===")

(let [pairs [{:prior-addr :mu :obs-addr :ys :family :normal-iid-normal}]
      handlers (aa/build-regenerate-handlers pairs)]
  (assert-true "regen: has :mu handler" (contains? handlers :mu))
  (assert-true "regen: has :ys handler" (contains? handlers :ys))
  (assert-equal "regen: 2 handlers total" 2 (count handlers)))

;; ---------------------------------------------------------------------------
;; 9. Variance reduction: auto-analytical should reduce IS weight variance
;; ---------------------------------------------------------------------------

(println "\n=== 9. Variance reduction ===")

(def var-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 5))]
      (trace :ys (dist/iid-gaussian mu (mx/scalar 1.0) 10))
      mu)))

(let [gf (dyn/auto-key var-model)
      obs (cm/choicemap :ys (mx/array [2.0 2.1 1.9 2.0 2.1 1.9 2.0 2.1 1.9 2.0]))
      ;; Run 50 generate calls and collect weights
      weights (vec (for [_ (range 50)]
                     (mx/item (:weight (p/generate gf [] obs)))))
      mean-w (/ (reduce + weights) (count weights))
      var-w (/ (reduce + (map #(* (- % mean-w) (- % mean-w)) weights)) (count weights))]
  ;; With conjugate elimination, ALL weights should be identical
  ;; (prior is analytically marginalized — no sampling variance)
  (assert-close "variance reduction: weight variance ≈ 0" 0.0 var-w 1e-6)
  (assert-true "variance reduction: all weights equal"
    (every? #(< (js/Math.abs (- % (first weights))) 1e-6) weights)))

;; ---------------------------------------------------------------------------
;; 10. Score accounting: score = weight for fully constrained model
;; ---------------------------------------------------------------------------

(println "\n=== 10. Score = weight accounting ===")

(let [gf (dyn/auto-key iid-model-e2e)
      obs (cm/choicemap :ys (mx/array [1.0 2.0 3.0 4.0 5.0]))
      results (for [_ (range 10)]
                (let [r (p/generate gf [] obs)
                      tr (:trace r)]
                  {:weight (mx/item (:weight r))
                   :score (mx/item (:score tr))}))]
  ;; When only obs are constrained and prior is marginalized,
  ;; score should equal weight
  (doseq [r results]
    (assert-close "score ≈ weight" (:weight r) (:score r) 1e-6)))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println "\n========================================")
(println (str "M2 Step 4 (IID Conjugacy): " @pass-count "/" (+ @pass-count @fail-count)
              " passed" (when (pos? @fail-count) (str ", " @fail-count " FAILED"))))
(println "========================================")

(when (pos? @fail-count)
  (js/process.exit 1))
