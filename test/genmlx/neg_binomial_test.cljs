(ns genmlx.neg-binomial-test
  "Tests for the negative binomial distribution."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
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

(println "\n=== Negative Binomial Tests ===\n")

;; NB(r=5, p=0.5)
;; Mean = r*(1-p)/p = 5*0.5/0.5 = 5.0
;; Var  = r*(1-p)/p^2 = 5*0.5/0.25 = 10.0
(let [d (dist/neg-binomial 5 0.5)]

  ;; -- Log-prob spot checks --
  ;; NB(k; r, p) = C(k+r-1, k) * p^r * (1-p)^k
  ;; For NB(5, 0.5):
  ;;   k=0: C(4,0)*0.5^5*0.5^0 = 1*0.03125 = 0.03125 → log = -3.4657
  ;;   k=3: C(7,3)*0.5^5*0.5^3 = 35*0.5^8 = 35/256 → log = -1.989
  ;;   k=5: C(9,5)*0.5^5*0.5^5 = 126*0.5^10 = 126/1024 → log = -2.096
  (println "-- log-prob spot checks --")
  (let [lp0 (dist/log-prob d (mx/scalar 0))]
    (mx/eval! lp0)
    (assert-close "log-prob at k=0" -3.4657 (mx/item lp0) 0.01))
  (let [lp3 (dist/log-prob d (mx/scalar 3))]
    (mx/eval! lp3)
    (assert-close "log-prob at k=3" -1.989 (mx/item lp3) 0.02))
  (let [lp5 (dist/log-prob d (mx/scalar 5))]
    (mx/eval! lp5)
    (assert-close "log-prob at k=5" -2.096 (mx/item lp5) 0.02))

  ;; -- Samples are non-negative integers --
  (println "\n-- sample validity --")
  (let [samples (repeatedly 20 #(let [v (dist/sample d)]
                                   (mx/eval! v)
                                   (mx/item v)))]
    (assert-true "all samples non-negative"
                 (every? #(>= % 0) samples))
    (assert-true "all samples are integers"
                 (every? #(== % (js/Math.floor %)) samples)))

  ;; -- Sample mean --
  (println "\n-- sample mean --")
  (let [n 2000
        samples (repeatedly n #(let [v (dist/sample d)]
                                  (mx/eval! v)
                                  (mx/item v)))
        mean (/ (reduce + samples) n)]
    (assert-close "sample mean ≈ 5.0" 5.0 mean 1.5))

  ;; -- Sample variance --
  (println "\n-- sample variance --")
  (let [n 2000
        samples (vec (repeatedly n #(let [v (dist/sample d)]
                                       (mx/eval! v)
                                       (mx/item v))))
        mean (/ (reduce + samples) n)
        variance (/ (reduce + (map #(* (- % mean) (- % mean)) samples)) n)]
    (assert-close "sample variance ≈ 10.0" 10.0 variance 4.0)))

;; -- Generate weight = log-prob --
(println "\n-- generate weight --")
(let [model (gen []
              (dyn/trace :k (dist/neg-binomial 5 0.5)))
      obs (cm/choicemap :k (mx/scalar 3))
      {:keys [trace weight]} (p/generate model [] obs)]
  (mx/eval! weight)
  (let [w (mx/item weight)
        lp (dist/log-prob (dist/neg-binomial 5 0.5) (mx/scalar 3))]
    (mx/eval! lp)
    (assert-close "generate weight = log-prob" (mx/item lp) w 0.01)))

(println "\nAll negative binomial tests complete.")
