(ns genmlx.contracts-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.contracts :as contracts])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(println "\n=== Contract Registry Tests ===")

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

;; Standard model (with eval!/item — works for scalar GFI ops)
(def two-site
  (gen []
    (let [x (dyn/trace :x (dist/gaussian 0 1))
          y (dyn/trace :y (dist/gaussian 0 1))]
      (mx/eval! x y)
      (mx/item x))))

;; Vectorization-compatible model (no eval!/item in body)
(def two-site-vec
  (gen []
    (let [x (dyn/trace :x (dist/gaussian 0 1))
          y (dyn/trace :y (dist/gaussian 0 1))]
      x)))

(def model two-site)
(def args [])
(def trace (p/simulate model args))

;; For broadcast-equivalence, use the vec-compatible model
(def vec-model two-site-vec)
(def vec-trace (p/simulate vec-model args))

;; ---------------------------------------------------------------------------
;; Run each contract
;; ---------------------------------------------------------------------------

(println "\n-- contract checks on 2-site gaussian --")

(let [scalar-ctx {:model model :args args :trace trace}
      vec-ctx {:model vec-model :args args :trace vec-trace}]
  (doseq [[k {:keys [theorem check]}] (sort-by (comp str key) contracts/contracts)]
    (let [ctx (if (= k :broadcast-equivalence) vec-ctx scalar-ctx)
          result (check ctx)]
      (assert-true (str (name k) " — " theorem) result))))

;; ---------------------------------------------------------------------------
;; Registry structure checks
;; ---------------------------------------------------------------------------

(println "\n-- registry structure --")

(assert-true "contracts is a map"
             (map? contracts/contracts))

(assert-true "11 contracts defined"
             (= 11 (count contracts/contracts)))

(assert-true "every contract has :theorem and :check"
             (every? (fn [[_ v]]
                       (and (string? (:theorem v))
                            (fn? (:check v))))
                     contracts/contracts))

(println "\nAll contract registry tests complete.")
