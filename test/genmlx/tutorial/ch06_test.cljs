(ns genmlx.tutorial.ch06-test
  "Test file for Tutorial Chapter 6: Composition — Splice and Combinators."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.selection :as sel]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.combinators :as comb]
            [genmlx.vmap :as vmap])
  (:require-macros [genmlx.gen :refer [gen]]))

(def pass (atom 0))
(def fail (atom 0))

(defn assert-true [msg pred]
  (if pred
    (do (swap! pass inc) (println "  PASS:" msg))
    (do (swap! fail inc) (println "  FAIL:" msg))))

(defn assert-close [msg expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass inc) (println "  PASS:" msg))
      (do (swap! fail inc) (println "  FAIL:" msg (str "expected=" expected " actual=" actual))))))

;; ============================================================
;; Listing 6.1: splice — calling sub-models
;; ============================================================
(println "\n== Listing 6.1: splice ==")

(def prior-model
  (gen []
    (let [slope (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 5))]
      {:slope slope :intercept intercept})))

(def obs-model
  (gen [params xs]
    (let [slope (:slope params)
          intercept (:intercept params)]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                      intercept) 1))))
    nil))

(def composed-model
  (gen [xs]
    (let [params (splice :prior prior-model [])
          _ (splice :obs obs-model [params xs])]
      params)))

(let [model (dyn/auto-key composed-model)
      trace (p/simulate model [[1.0 2.0 3.0]])]
  (assert-true "composed model simulates" (some? trace))
  (assert-true "has :prior submap" (not= cm/EMPTY (cm/get-submap (:choices trace) :prior)))
  (let [prior-sub (cm/get-submap (:choices trace) :prior)]
    (assert-true "prior has :slope" (cm/has-value? (cm/get-submap prior-sub :slope)))
    (assert-true "prior has :intercept" (cm/has-value? (cm/get-submap prior-sub :intercept))))
  (assert-true "score is finite" (js/Number.isFinite (mx/item (:score trace)))))

;; ============================================================
;; Listing 6.2: Map combinator
;; ============================================================
(println "\n== Listing 6.2: Map combinator ==")

(def noisy-obs
  (gen [x]
    (trace :y (dist/gaussian (mx/scalar x) 1))))

(let [mapped (comb/map-combinator noisy-obs)
      model (dyn/auto-key mapped)
      trace (p/simulate model [[[1.0] [2.0] [3.0] [4.0]]])]
  (assert-true "map simulates" (some? trace))
  ;; Choices indexed by integers
  (assert-true "has index 0" (not= cm/EMPTY (cm/get-submap (:choices trace) 0)))
  (assert-true "has index 3" (not= cm/EMPTY (cm/get-submap (:choices trace) 3)))
  ;; Each sub-choicemap has :y
  (let [sub0 (cm/get-submap (:choices trace) 0)]
    (assert-true "index 0 has :y" (cm/has-value? (cm/get-submap sub0 :y))))
  (assert-true "score is finite" (js/Number.isFinite (mx/item (:score trace)))))

;; ============================================================
;; Listing 6.3: Unfold combinator (HMM-like)
;; ============================================================
(println "\n== Listing 6.3: Unfold combinator ==")

(def step-kernel
  (gen [t state]
    (let [next-state (trace :z (dist/gaussian state 1))]
      (trace :obs (dist/gaussian next-state 0.5))
      next-state)))

(let [unfolded (comb/unfold-combinator step-kernel)
      model (dyn/auto-key unfolded)
      ;; Unfold takes [T init-state] as args (flat, not nested)
      trace (p/simulate model [3 0.0])]
  (assert-true "unfold simulates" (some? trace))
  ;; Choices indexed by timestep (0-based)
  (assert-true "has timestep 0" (not= cm/EMPTY (cm/get-submap (:choices trace) 0)))
  (assert-true "has timestep 2" (not= cm/EMPTY (cm/get-submap (:choices trace) 2)))
  ;; Each step has :z and :obs
  (let [step0 (cm/get-submap (:choices trace) 0)]
    (assert-true "step 0 has :z" (cm/has-value? (cm/get-submap step0 :z)))
    (assert-true "step 0 has :obs" (cm/has-value? (cm/get-submap step0 :obs))))
  (assert-true "score is finite" (js/Number.isFinite (mx/item (:score trace)))))

;; ============================================================
;; Listing 6.4: Switch combinator
;; ============================================================
(println "\n== Listing 6.4: Switch combinator ==")

(def branch-a (dyn/auto-key (gen [] (trace :x (dist/gaussian 0 1)))))
(def branch-b (dyn/auto-key (gen [] (trace :x (dist/gaussian 10 1)))))

(let [switched (comb/switch-combinator branch-a branch-b)
      model (dyn/auto-key switched)
      ;; Switch takes [index] as arg
      trace-a (p/simulate model [0])
      trace-b (p/simulate model [1])]
  (assert-true "switch(0) simulates" (some? trace-a))
  (assert-true "switch(1) simulates" (some? trace-b))
  ;; Branch A samples near 0, Branch B near 10
  (let [val-a (mx/item (cm/get-choice (:choices trace-a) [:x]))
        val-b (mx/item (cm/get-choice (:choices trace-b) [:x]))]
    (assert-true "branch A near 0" (< (js/Math.abs val-a) 5))
    (assert-true "branch B near 10" (< (js/Math.abs (- val-b 10)) 5))))

;; ============================================================
;; Listing 6.5: Scan combinator
;; ============================================================
(println "\n== Listing 6.5: Scan combinator ==")

;; Scan kernel: (fn [carry input] -> [output carry'])
;; Must return a 2-vector [output new-carry]
(def scan-step
  (gen [carry input]
    (let [x (trace :x (dist/gaussian (mx/add carry input) 1))]
      [x x])))

(let [scanned (comb/scan-combinator scan-step)
      model (dyn/auto-key scanned)
      inputs [(mx/scalar 0.0) (mx/scalar 1.0) (mx/scalar 2.0)]
      trace (p/simulate model [(mx/scalar 0.0) inputs])]
  (assert-true "scan simulates" (some? trace))
  (assert-true "has timestep 0" (not= cm/EMPTY (cm/get-submap (:choices trace) 0)))
  (assert-true "has timestep 2" (not= cm/EMPTY (cm/get-submap (:choices trace) 2)))
  (assert-true "score is finite" (js/Number.isFinite (mx/item (:score trace)))))

;; ============================================================
;; Listing 6.6: Mix combinator
;; ============================================================
(println "\n== Listing 6.6: Mix combinator ==")

(def component-a (gen [] (trace :x (dist/gaussian -5 1))))
(def component-b (gen [] (trace :x (dist/gaussian 5 1))))

(let [mixed (comb/mix-combinator [component-a component-b]
                                  (mx/array [(js/Math.log 0.5) (js/Math.log 0.5)]))
      model (dyn/auto-key mixed)
      ;; Sample 20 times and check we get values from both components
      samples (mapv (fn [_]
                      (let [t (p/simulate model [])]
                        (mx/item (cm/get-choice (:choices t) [:x]))))
                    (range 20))
      has-negative (some neg? samples)
      has-positive (some pos? samples)]
  (assert-true "mix simulates" (= 20 (count samples)))
  ;; With 50/50 weights, we should see both negative and positive values
  (assert-true "samples from both components" (and has-negative has-positive)))

;; ============================================================
;; Listing 6.7: Recurse combinator
;; ============================================================
(println "\n== Listing 6.7: Recurse combinator ==")

(let [tree (comb/recurse
             (fn [self]
               (dyn/auto-key
                (gen [depth]
                  (let [x (trace :x (dist/gaussian 0 1))]
                    (mx/eval! x)
                    (if (> depth 0)
                      (do (splice :child self [(dec depth)])
                          x)
                      x))))))
      model (dyn/auto-key tree)
      trace (p/simulate model [2])]
  (assert-true "recurse simulates" (some? trace))
  (assert-true "has :x" (cm/has-value? (cm/get-submap (:choices trace) :x)))
  (assert-true "score is finite" (js/Number.isFinite (mx/item (:score trace)))))

;; Note: contramap-gf, map-retval, and dimap are covered in Chapter 10 (extensions)
;; as they interact with the compilation system in ways that need careful handling.

;; ============================================================
;; Listing 6.9: Composing combinators (Map of models)
;; ============================================================
(println "\n== Listing 6.9: composing combinators ==")

(let [series-model (dyn/auto-key (gen [x] (trace :y (dist/gaussian (mx/scalar x) 1))))
      mapped (comb/map-combinator series-model)
      model (dyn/auto-key mapped)
      trace (p/simulate model [[[1.0] [2.0] [3.0]]])]
  (assert-true "composed combinator simulates" (some? trace))
  (assert-true "has 3 indices" (and (not= cm/EMPTY (cm/get-submap (:choices trace) 0))
                                     (not= cm/EMPTY (cm/get-submap (:choices trace) 1))
                                     (not= cm/EMPTY (cm/get-submap (:choices trace) 2)))))

;; ============================================================
;; Summary
;; ============================================================
(println (str "\n== Chapter 6 tests: " @pass " PASS, " @fail " FAIL =="))
(when (pos? @fail) (js/process.exit 1))
