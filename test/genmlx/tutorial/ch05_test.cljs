(ns genmlx.tutorial.ch05-test
  "Test file for Tutorial Chapter 5: Updating and Regenerating.
   Exercises update, regenerate, assess, project, and building MH by hand."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.selection :as sel]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.util :as u])
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
      (do (swap! fail inc) (println "  FAIL:" msg (str "expected=" expected " actual=" actual " diff=" diff))))))

;; Reuse linear model
(def linear-model
  (gen [xs]
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                      intercept) 1)))
      slope)))

(def xs [1.0 2.0 3.0])

;; ============================================================
;; Listing 5.1: p/update — change choices
;; ============================================================
(println "\n== Listing 5.1: p/update ==")

(let [model (dyn/auto-key linear-model)
      trace (p/simulate model [xs])
      old-slope (mx/item (cm/get-choice (:choices trace) [:slope]))
      ;; Update: set slope to 2.0
      new-constraints (cm/choicemap :slope (mx/scalar 2.0))
      {:keys [trace weight discard]} (p/update model trace new-constraints)]
  (assert-close "new slope is 2.0" 2.0
                (mx/item (cm/get-choice (:choices trace) [:slope])) 0.001)
  (assert-true "weight is finite" (js/Number.isFinite (mx/item weight)))
  (assert-true "discard is a choicemap" (some? discard))
  ;; Discard contains the old slope value
  (assert-close "discard has old slope" old-slope
                (mx/item (cm/get-choice discard [:slope])) 0.001)
  ;; Other addresses unchanged
  (assert-true "intercept still present"
               (cm/has-value? (cm/get-submap (:choices trace) :intercept))))

;; ============================================================
;; Listing 5.2: Weight semantics
;; ============================================================
(println "\n== Listing 5.2: weight = log(new_score / old_score) ==")

(let [model (dyn/auto-key linear-model)
      trace (p/simulate model [xs])
      old-score (mx/item (:score trace))
      new-constraints (cm/choicemap :slope (mx/scalar 2.0))
      {:keys [trace weight]} (p/update model trace new-constraints)
      new-score (mx/item (:score trace))
      w (mx/item weight)
      expected-w (- new-score old-score)]
  (assert-close "weight = new_score - old_score" expected-w w 0.01))

;; ============================================================
;; Listing 5.3: The discard
;; ============================================================
(println "\n== Listing 5.3: the discard ==")

(let [model (dyn/auto-key linear-model)
      trace (p/simulate model [xs])
      ;; Update both slope and intercept
      new-constraints (cm/choicemap :slope (mx/scalar 2.0) :intercept (mx/scalar 0.5))
      {:keys [discard]} (p/update model trace new-constraints)]
  (assert-true "discard has :slope" (cm/has-value? (cm/get-submap discard :slope)))
  (assert-true "discard has :intercept" (cm/has-value? (cm/get-submap discard :intercept)))
  ;; Discard does NOT have :y0 (not changed)
  (assert-true "discard does not have :y0" (= cm/EMPTY (cm/get-submap discard :y0))))

;; ============================================================
;; Listing 5.4: p/regenerate — resample selected addresses
;; ============================================================
(println "\n== Listing 5.4: p/regenerate ==")

(let [model (dyn/auto-key linear-model)
      trace (p/simulate model [xs])
      old-intercept (mx/item (cm/get-choice (:choices trace) [:intercept]))
      ;; Regenerate only :slope
      {:keys [trace weight]} (p/regenerate model trace (sel/select :slope))]
  (assert-true "regenerate returns a trace" (some? trace))
  (assert-true "weight is finite" (js/Number.isFinite (mx/item weight)))
  ;; Intercept should be kept (not selected)
  (assert-close "intercept unchanged" old-intercept
                (mx/item (cm/get-choice (:choices trace) [:intercept])) 0.001)
  ;; Slope was resampled (might be different)
  (assert-true "slope exists" (cm/has-value? (cm/get-submap (:choices trace) :slope))))

;; ============================================================
;; Listing 5.5: Selections target specific addresses
;; ============================================================
(println "\n== Listing 5.5: selections ==")

(let [model (dyn/auto-key linear-model)
      trace (p/simulate model [xs])
      old-slope (mx/item (cm/get-choice (:choices trace) [:slope]))
      old-intercept (mx/item (cm/get-choice (:choices trace) [:intercept]))
      ;; Regenerate both slope and intercept
      {:keys [trace]} (p/regenerate model trace (sel/select :slope :intercept))]
  ;; Both were resampled — they exist but values may differ
  (assert-true ":slope exists after regen" (cm/has-value? (cm/get-submap (:choices trace) :slope)))
  (assert-true ":intercept exists after regen" (cm/has-value? (cm/get-submap (:choices trace) :intercept)))
  ;; y values should still exist (kept from old trace)
  (assert-true ":y0 exists" (cm/has-value? (cm/get-submap (:choices trace) :y0))))

;; ============================================================
;; Listing 5.6: p/assess — score fully-specified choices
;; ============================================================
(println "\n== Listing 5.6: p/assess ==")

(let [model (dyn/auto-key linear-model)
      trace (p/simulate model [xs])
      ;; Assess the trace's own choices
      {:keys [retval weight]} (p/assess model [xs] (:choices trace))
      score (mx/item (:score trace))
      assess-weight (mx/item weight)]
  (assert-true "assess weight is finite" (js/Number.isFinite assess-weight))
  (assert-close "assess weight matches trace score" score assess-weight 0.01))

;; ============================================================
;; Listing 5.7: p/project — score a selection
;; ============================================================
(println "\n== Listing 5.7: p/project ==")

(let [model (dyn/auto-key linear-model)
      trace (p/simulate model [xs])
      ;; Project all addresses
      all-proj (mx/item (p/project model trace sel/all))
      score (mx/item (:score trace))]
  (assert-close "project(all) = score" score all-proj 0.01)
  ;; Project none
  (let [none-proj (mx/item (p/project model trace sel/none))]
    (assert-close "project(none) = 0" 0.0 none-proj 0.01))
  ;; Project a subset — log-prob of just :slope
  (let [slope-proj (mx/item (p/project model trace (sel/select :slope)))]
    (assert-true "project(:slope) is finite" (js/Number.isFinite slope-proj))
    (assert-true "project(:slope) is not zero" (not= 0.0 slope-proj))))

;; ============================================================
;; Listing 5.8: MH by hand — regenerate + accept/reject
;; ============================================================
(println "\n== Listing 5.8: MH by hand ==")

(let [model (dyn/auto-key linear-model)
      obs (cm/choicemap :y0 (mx/scalar 2.5) :y1 (mx/scalar 4.5) :y2 (mx/scalar 6.5))
      ;; Initialize with generate
      init-result (p/generate model [xs] obs)
      init-trace (:trace init-result)]
  (assert-true "initial trace exists" (some? init-trace))
  ;; One MH step: regenerate :slope, accept/reject
  (let [{:keys [trace weight]} (p/regenerate model init-trace (sel/select :slope))
        log-alpha (mx/item weight)
        log-u (js/Math.log (js/Math.random))
        accept? (< log-u log-alpha)
        next-trace (if accept? trace init-trace)]
    (assert-true "MH step produces a trace" (some? next-trace))
    (assert-true "log-alpha is finite" (js/Number.isFinite log-alpha))))

;; ============================================================
;; Listing 5.9: MH chain — 200 steps
;; ============================================================
(println "\n== Listing 5.9: MH chain ==")

(let [model (dyn/auto-key linear-model)
      obs (cm/choicemap :y0 (mx/scalar 2.5) :y1 (mx/scalar 4.5) :y2 (mx/scalar 6.5))
      init-trace (:trace (p/generate model [xs] obs))
      selection (sel/select :slope :intercept)
      n-steps 200
      burn 50
      ;; Run chain
      result (loop [i 0, trace init-trace, samples [], accepted 0]
               (if (>= i n-steps)
                 {:samples samples :acceptance-rate (/ accepted n-steps)}
                 (let [{t :trace w :weight} (p/regenerate model trace selection)
                       log-alpha (mx/item w)
                       accept? (< (js/Math.log (js/Math.random)) log-alpha)
                       next-trace (if accept? t trace)
                       _ (when (zero? (mod i 50)) (mx/sweep-dead-arrays!))]
                   (recur (inc i)
                          next-trace
                          (if (>= i burn)
                            (conj samples (mx/item (cm/get-choice (:choices next-trace) [:slope])))
                            samples)
                          (if accept? (inc accepted) accepted)))))]
  (assert-true "collected samples after burn-in" (pos? (count (:samples result))))
  (assert-true "acceptance rate > 0" (> (:acceptance-rate result) 0))
  (assert-true "acceptance rate < 1" (< (:acceptance-rate result) 1))
  ;; Posterior mean should be near 2.0
  (let [samples (:samples result)
        mean (/ (reduce + samples) (count samples))]
    (assert-true "posterior mean near 2 (within 3)"
                 (< (js/Math.abs (- mean 2.0)) 3.0))))

;; ============================================================
;; Summary
;; ============================================================
(println (str "\n== Chapter 5 tests: " @pass " PASS, " @fail " FAIL =="))
(when (pos? @fail) (js/process.exit 1))
