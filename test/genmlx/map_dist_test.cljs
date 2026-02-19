(ns genmlx.map-dist-test
  "Tests for map->dist: creating distributions from plain maps."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-close [msg expected actual tolerance]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tolerance)
      (println "  PASS:" msg)
      (do (println "  FAIL:" msg)
          (println "    expected:" expected "+/-" tolerance)
          (println "    actual:  " actual)))))

(println "\n=== map->dist Tests ===\n")

;; ---------------------------------------------------------------------------
;; 1. Basic usage — custom uniform-like distribution
;; ---------------------------------------------------------------------------

(println "-- basic map->dist usage --")
(let [;; Create a uniform(0,1) equivalent via map->dist
      my-uniform (dc/map->dist
                   {:type :test-uniform
                    :sample (fn [key]
                              (rng/uniform (rng/ensure-key key) []))
                    :log-prob (fn [value]
                                ;; log(1) = 0 for values in [0,1]
                                (mx/scalar 0.0))})
      ;; Sample
      v (dc/dist-sample my-uniform nil)]
  (mx/eval! v)
  (let [val (mx/realize v)]
    (assert-true "sample in [0,1]" (and (>= val 0) (<= val 1))))
  ;; Log-prob
  (let [lp (dc/dist-log-prob my-uniform (mx/scalar 0.5))]
    (mx/eval! lp)
    (assert-close "log-prob is 0" 0.0 (mx/realize lp) 1e-6)))

;; ---------------------------------------------------------------------------
;; 2. GFI integration — use inside gen body
;; ---------------------------------------------------------------------------

(println "\n-- GFI integration --")
(let [my-dist (dc/map->dist
                {:type :test-gaussian-bridge
                 :sample (fn [key]
                           (let [key (rng/ensure-key key)
                                 z (rng/normal key [])]
                             ;; N(5, 0.1): 5 + 0.1*z
                             (mx/add (mx/scalar 5.0)
                                     (mx/multiply (mx/scalar 0.1) z))))
                 :log-prob (fn [value]
                             (let [z (mx/divide (mx/subtract value (mx/scalar 5.0))
                                                (mx/scalar 0.1))]
                               (mx/subtract
                                 (mx/negative (mx/multiply (mx/scalar 0.5)
                                                           (mx/multiply z z)))
                                 (mx/scalar (+ (* 0.5 (js/Math.log (* 2 js/Math.PI)))
                                               (js/Math.log 0.1))))))})
      model (gen []
              (let [x (dyn/trace :x my-dist)]
                (mx/eval! x) (mx/item x)))
      ;; simulate
      trace (p/simulate model [])
      retval (:retval trace)]
  (assert-true "simulate: retval near 5" (< (js/Math.abs (- retval 5)) 1.0))
  (assert-true "simulate: score finite" (js/isFinite (mx/realize (:score trace))))

  ;; generate with constraint
  (let [{:keys [trace weight]} (p/generate model [] (cm/choicemap :x (mx/scalar 5.0)))]
    (mx/eval! weight)
    (assert-true "generate: weight finite" (js/isFinite (mx/realize weight)))
    (assert-close "generate: constrained value" 5.0
      (mx/realize (cm/get-value (cm/get-submap (:choices trace) :x))) 1e-6))

  ;; assess
  (let [{:keys [weight]} (p/assess model [] (cm/choicemap :x (mx/scalar 5.0)))]
    (mx/eval! weight)
    (assert-true "assess: weight finite" (js/isFinite (mx/realize weight)))))

;; ---------------------------------------------------------------------------
;; 3. Optional reparam
;; ---------------------------------------------------------------------------

(println "\n-- optional reparam --")
(let [my-dist (dc/map->dist
                {:type :test-reparam
                 :sample (fn [key]
                           (rng/normal (rng/ensure-key key) []))
                 :log-prob (fn [value]
                             (mx/negative (mx/multiply (mx/scalar 0.5)
                                                       (mx/multiply value value))))
                 :reparam (fn [key]
                            ;; Reparameterized: same as sample for standard normal
                            (rng/normal (rng/ensure-key key) []))})
      v (dc/dist-reparam my-dist nil)]
  (mx/eval! v)
  (assert-true "reparam: returns finite value" (js/isFinite (mx/realize v))))

;; ---------------------------------------------------------------------------
;; 4. Optional sample-n
;; ---------------------------------------------------------------------------

(println "\n-- optional sample-n --")
(let [my-dist (dc/map->dist
                {:type :test-batch
                 :sample (fn [key]
                           (rng/normal (rng/ensure-key key) []))
                 :log-prob (fn [value]
                             (mx/scalar 0.0))
                 :sample-n (fn [key n]
                             (rng/normal (rng/ensure-key key) [n]))})
      samples (dc/dist-sample-n my-dist (rng/fresh-key) 50)]
  (mx/eval! samples)
  (assert-true "sample-n: shape [50]" (= [50] (mx/shape samples))))

;; ---------------------------------------------------------------------------
;; 5. Auto-generated type keyword
;; ---------------------------------------------------------------------------

(println "\n-- auto-generated type --")
(let [d1 (dc/map->dist {:sample (fn [k] (mx/scalar 1.0))
                         :log-prob (fn [v] (mx/scalar 0.0))})
      d2 (dc/map->dist {:sample (fn [k] (mx/scalar 2.0))
                         :log-prob (fn [v] (mx/scalar 0.0))})]
  (assert-true "different auto types" (not= (:type d1) (:type d2)))
  (let [v1 (dc/dist-sample d1 nil)
        v2 (dc/dist-sample d2 nil)]
    (mx/eval! v1 v2)
    (assert-close "d1 samples 1.0" 1.0 (mx/realize v1) 1e-6)
    (assert-close "d2 samples 2.0" 2.0 (mx/realize v2) 1e-6)))

(println "\nAll map->dist tests complete.")
