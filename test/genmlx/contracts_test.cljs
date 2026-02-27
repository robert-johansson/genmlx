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

;; ---------------------------------------------------------------------------
;; verify-gfi-contracts tests
;; ---------------------------------------------------------------------------

(println "\n-- verify-gfi-contracts --")

;; Run on scalar model (all contracts except broadcast-equivalence)
(let [scalar-keys (disj (set (keys contracts/contracts)) :broadcast-equivalence)
      report (contracts/verify-gfi-contracts model args :n-trials 5 :contract-keys scalar-keys)]
  (assert-true "all-pass? true for scalar contracts"
               (:all-pass? report))
  (assert-true "total-pass = 50 (10 contracts × 5 trials)"
               (= 50 (:total-pass report)))
  (assert-true "total-fail = 0"
               (= 0 (:total-fail report)))
  (assert-true "each result has :pass :fail :theorem"
               (every? (fn [[_ v]]
                         (and (integer? (:pass v))
                              (integer? (:fail v))
                              (string? (:theorem v))))
                       (:results report))))

;; Run broadcast-equivalence on vec-compatible model
(let [report (contracts/verify-gfi-contracts vec-model args
               :n-trials 3
               :contract-keys #{:broadcast-equivalence})]
  (assert-true "broadcast-equivalence passes on vec model"
               (:all-pass? report))
  (assert-true "broadcast total-pass = 3"
               (= 3 (:total-pass report))))

(println "\nAll contract registry tests complete.")
