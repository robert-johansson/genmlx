(ns genmlx.iid-test
  "Tests for iid and iid-gaussian distributions."
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
    (do (vswap! pass-count inc)
        (println "  PASS:" desc))
    (do (vswap! fail-count inc)
        (println "  FAIL:" desc))))

(defn- assert-close [desc expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (vswap! pass-count inc)
          (println "  PASS:" desc))
      (do (vswap! fail-count inc)
          (println "  FAIL:" desc "expected" expected "got" actual "diff" diff)))))

(defn- assert-shape [desc expected-shape array]
  (assert-true (str desc " shape=" expected-shape)
               (= expected-shape (mx/shape array))))

;; ---------------------------------------------------------------------------
;; iid distribution
;; ---------------------------------------------------------------------------

(println "\n-- iid: sample shape --")
(let [base (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0))
      d (dist/iid base 5)
      s (dc/dist-sample d (rng/fresh-key))]
  (assert-shape "iid sample" [5] s))

(println "\n-- iid: sample-n shape --")
(let [base (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0))
      d (dist/iid base 5)
      sn (dc/dist-sample-n d (rng/fresh-key) 10)]
  (assert-shape "iid sample-n" [10 5] sn))

(println "\n-- iid: log-prob matches sum of element log-probs --")
(let [base (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0))
      d (dist/iid base 3)
      vals (mx/array [1.0 2.0 3.0])
      iid-lp (mx/item (dc/dist-log-prob d vals))
      manual (+ (mx/item (dc/dist-log-prob base (mx/scalar 1.0)))
                (mx/item (dc/dist-log-prob base (mx/scalar 2.0)))
                (mx/item (dc/dist-log-prob base (mx/scalar 3.0))))]
  (assert-close "iid log-prob = sum of element log-probs" manual iid-lp 1e-5))

(println "\n-- iid: reparam shape --")
(let [base (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0))
      d (dist/iid base 4)
      s (dc/dist-reparam d (rng/fresh-key))]
  (assert-shape "iid reparam" [4] s))

;; ---------------------------------------------------------------------------
;; iid-gaussian distribution
;; ---------------------------------------------------------------------------

(println "\n-- iid-gaussian: sample shape --")
(let [d (dist/iid-gaussian (mx/scalar 0.0) (mx/scalar 1.0) 5)
      s (dc/dist-sample d (rng/fresh-key))]
  (assert-shape "iid-gaussian sample" [5] s))

(println "\n-- iid-gaussian: sample-n shape --")
(let [d (dist/iid-gaussian (mx/scalar 0.0) (mx/scalar 1.0) 5)
      sn (dc/dist-sample-n d (rng/fresh-key) 10)]
  (assert-shape "iid-gaussian sample-n" [10 5] sn))

(println "\n-- iid-gaussian: log-prob matches iid --")
(let [d1 (dist/iid (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)) 3)
      d2 (dist/iid-gaussian (mx/scalar 0.0) (mx/scalar 1.0) 3)
      vals (mx/array [1.0 2.0 3.0])
      lp1 (mx/item (dc/dist-log-prob d1 vals))
      lp2 (mx/item (dc/dist-log-prob d2 vals))]
  (assert-close "iid-gaussian matches iid log-prob" lp1 lp2 1e-5))

(println "\n-- iid-gaussian: [T]-shaped mu --")
(let [means (mx/array [1.0 2.0 3.0])
      d (dist/iid-gaussian means (mx/scalar 1.0) 3)]
  (assert-shape "[T] mu sample" [3] (dc/dist-sample d (rng/fresh-key)))
  ;; Log-prob at the means should be the maximum (all z=0)
  (let [lp-at-means (mx/item (dc/dist-log-prob d means))
        lp-away (mx/item (dc/dist-log-prob d (mx/array [10.0 20.0 30.0])))]
    (assert-true "[T] mu lp at means > lp away" (> lp-at-means lp-away))))

(println "\n-- iid-gaussian: [N,T] log-prob broadcasting --")
(let [means (mx/array [1.0 2.0 3.0])
      d (dist/iid-gaussian means (mx/scalar 1.0) 3)
      vals (mx/array [[1.0 2.0 3.0] [0.5 1.5 2.5]])]
  (let [lp (dc/dist-log-prob d vals)]
    (assert-shape "[N,T] broadcasting" [2] lp)
    ;; First row (at means) should have higher log-prob
    (let [lps (mx/->clj lp)]
      (assert-true "[N,T] lp at means > lp away" (> (first lps) (second lps))))))

(println "\n-- iid-gaussian: reparam shape --")
(let [d (dist/iid-gaussian (mx/scalar 0.0) (mx/scalar 1.0) 4)
      s (dc/dist-reparam d (rng/fresh-key))]
  (assert-shape "iid-gaussian reparam" [4] s))

;; ---------------------------------------------------------------------------
;; Model integration: scalar
;; ---------------------------------------------------------------------------

(println "\n-- iid in model: scalar simulate --")
(def iid-model
  (gen [t]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :ys (dist/iid-gaussian mu (mx/scalar 1.0) t))
      mu)))

(let [tr (p/simulate (dyn/auto-key iid-model) [5])]
  (assert-shape "scalar simulate :ys" [5]
                (cm/get-value (cm/get-submap (:choices tr) :ys))))

(println "\n-- iid in model: scalar generate with stacked obs --")
(let [obs (cm/choicemap :ys (mx/array [1.0 2.0 3.0 4.0 5.0]))
      result (p/generate (dyn/auto-key iid-model) [5] obs)]
  (assert-true "scalar generate weight is finite"
               (js/isFinite (mx/item (:weight result)))))

;; ---------------------------------------------------------------------------
;; Model integration: vectorized
;; ---------------------------------------------------------------------------

(println "\n-- iid in model: vsimulate --")
(let [vt (dyn/vsimulate (dyn/auto-key iid-model) [5] 100 (rng/fresh-key))
      inner (:m (:choices vt))]
  (assert-shape "vsimulate :mu" [100] (:v (get inner :mu)))
  (assert-shape "vsimulate :ys" [100 5] (:v (get inner :ys)))
  (assert-shape "vsimulate score" [100] (:score vt)))

(println "\n-- iid in model: vgenerate with posterior check --")
(let [obs (cm/choicemap :ys (mx/array [1.0 2.0 3.0 4.0 5.0]))
      vt (dyn/vgenerate (dyn/auto-key iid-model) [5] obs 5000 (rng/fresh-key))
      w (:weight vt) r (:retval vt)
      wn (let [e (mx/exp (mx/subtract w (mx/amax w)))] (mx/divide e (mx/sum e)))
      mu-est (mx/item (mx/sum (mx/multiply wn r)))]
  (assert-shape "vgenerate weight" [5000] w)
  (assert-close "posterior mean mu ≈ 3.0" 3.0 mu-est 0.5))

;; ---------------------------------------------------------------------------
;; Performance
;; ---------------------------------------------------------------------------

(println "\n-- iid performance --")
(let [model (dyn/auto-key iid-model)
      obs (cm/choicemap :ys (mx/array (mapv #(+ (* 2.0 %) 1.0) (range 50))))
      key (rng/fresh-key)
      t0 (.now js/Date)
      _ (dyn/vgenerate model [50] obs 10000 key)
      t1 (.now js/Date)]
  (println "  vgenerate N=10000 T=50:" (- t1 t0) "ms")
  (assert-true "vgenerate < 100ms" (< (- t1 t0) 100)))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n  Passed: " @pass-count))
(println (str "  Failed: " @fail-count))
(println (str "  Total:  " (+ @pass-count @fail-count)))
(if (zero? @fail-count)
  (println "\n  *** ALL IID TESTS PASS ***")
  (println "\n  *** SOME TESTS FAILED ***"))
