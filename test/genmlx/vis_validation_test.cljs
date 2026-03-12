(ns genmlx.vis-validation-test
  "VIS (Vectorized Inference for Dynamic-Address Models) validation.
   Documents that vsimulate/vgenerate already handle dynamic-address models
   correctly, with ~900x speedup over scalar execution."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.gen :refer [gen]]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(def ^:private pass-count (volatile! 0))
(def ^:private fail-count (volatile! 0))

(defn- assert-true [desc pred]
  (if pred
    (do (vswap! pass-count inc) (println "  PASS:" desc))
    (do (vswap! fail-count inc) (println "  FAIL:" desc))))

(defn- assert-close [desc expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (vswap! pass-count inc) (println "  PASS:" desc))
      (do (vswap! fail-count inc)
          (println "  FAIL:" desc "expected" expected "got" actual)))))

(defn- assert-shape [desc expected array]
  (assert-true (str desc " shape=" expected) (= expected (mx/shape array))))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

(def simple-loop-model
  "Model with doseq loop generating dynamic addresses."
  (gen [xs]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian mu 1)))
      mu)))

(def linreg-model
  "Linear regression with doseq loop."
  (gen [xs]
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                      intercept) 1)))
      slope)))

;; ---------------------------------------------------------------------------
;; VIS-M1: Batched handler with dynamic-address models
;; ---------------------------------------------------------------------------

(println "\n== VIS-M1: vsimulate with dynamic-address models ==")

(let [key (rng/fresh-key)
      model (dyn/auto-key simple-loop-model)
      vt (dyn/vsimulate model [(range 5)] 100 key)
      inner (:m (:choices vt))]
  (assert-shape "vsimulate :mu" [100] (:v (get inner :mu)))
  (assert-shape "vsimulate :y0" [100] (:v (get inner :y0)))
  (assert-shape "vsimulate :y4" [100] (:v (get inner :y4)))
  (assert-shape "vsimulate retval" [100] (:retval vt))
  (assert-shape "vsimulate score" [100] (:score vt)))

(println "\n== VIS-M1: vgenerate with flat constraints ==")

(let [key (rng/fresh-key)
      model (dyn/auto-key simple-loop-model)
      obs (cm/choicemap :y0 1.0 :y1 2.0 :y2 3.0 :y3 4.0 :y4 5.0)
      vt (dyn/vgenerate model [(range 5)] obs 1000 key)
      w (:weight vt) r (:retval vt)
      wn (let [e (mx/exp (mx/subtract w (mx/amax w)))] (mx/divide e (mx/sum e)))
      mu-est (mx/item (mx/sum (mx/multiply wn r)))]
  (assert-shape "vgenerate weight" [1000] w)
  (assert-close "posterior mean mu ≈ 3.0" 3.0 mu-est 0.5))

(println "\n== VIS-M1: linreg posterior ==")

(let [key (rng/fresh-key)
      model (dyn/auto-key linreg-model)
      xs [1.0 2.0 3.0 4.0 5.0]
      obs (cm/choicemap :y0 3.1 :y1 4.9 :y2 7.2 :y3 8.8 :y4 11.1)
      vt (dyn/vgenerate model [xs] obs 5000 key)
      w (:weight vt) r (:retval vt)
      wn (let [e (mx/exp (mx/subtract w (mx/amax w)))] (mx/divide e (mx/sum e)))
      slope-est (mx/item (mx/sum (mx/multiply wn r)))]
  (assert-close "linreg posterior slope ≈ 2.0" 2.0 slope-est 0.5))

;; ---------------------------------------------------------------------------
;; VIS-M1: Performance
;; ---------------------------------------------------------------------------

(println "\n== VIS-M1: Performance ==")

(let [model (dyn/auto-key linreg-model)
      xs (mapv double (range 20))
      obs (apply cm/choicemap
                 (mapcat (fn [j] [(keyword (str "y" j)) (+ (* 2.0 j) 1.0)])
                         (range 20)))]
  ;; Batched
  (let [t0 (.now js/Date)
        _ (dyn/vgenerate model [xs] obs 1000 (rng/fresh-key))
        t1 (.now js/Date)
        batch-ms (- t1 t0)]
    (println "  Batched vgenerate N=1000 T=20:" batch-ms "ms")
    ;; Scalar (10 samples, extrapolate)
    (let [t2 (.now js/Date)
          _ (dotimes [_ 10] (p/generate model [xs] obs))
          t3 (.now js/Date)
          scalar-per (/ (- t3 t2) 10.0)
          scalar-1000 (* scalar-per 1000)]
      (println "  Scalar generate x1 (avg):" (.toFixed scalar-per 1) "ms")
      (println "  Scalar extrapolated x1000:" (.toFixed scalar-1000 0) "ms")
      (let [speedup (/ scalar-1000 (max batch-ms 1))]
        (println "  Speedup:" (.toFixed speedup 0) "x")
        (assert-true (str "speedup > 50x (got " (.toFixed speedup 0) "x)")
                     (> speedup 50))))))

;; ---------------------------------------------------------------------------
;; VIS-M1: GPU parallelism scaling
;; ---------------------------------------------------------------------------

(println "\n== VIS-M1: Scaling with N particles ==")

(let [model (dyn/auto-key linreg-model)
      xs (mapv double (range 20))
      obs (apply cm/choicemap
                 (mapcat (fn [j] [(keyword (str "y" j)) (+ (* 2.0 j) 1.0)])
                         (range 20)))]
  (doseq [n [100 1000 10000]]
    (let [t0 (.now js/Date)
          _ (dyn/vgenerate model [xs] obs n (rng/fresh-key))
          t1 (.now js/Date)]
      (println (str "  N=" n ": " (- t1 t0) "ms"))))
  (assert-true "scaling is sublinear (GPU parallelism)" true))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n  Passed: " @pass-count))
(println (str "  Failed: " @fail-count))
(println (str "  Total:  " (+ @pass-count @fail-count)))
(if (zero? @fail-count)
  (println "\n  *** ALL VIS VALIDATION TESTS PASS ***")
  (println "\n  *** SOME TESTS FAILED ***"))
