(ns genmlx.product-dist-test
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
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

(println "\n=== Product Distribution Tests ===")

;; ---------------------------------------------------------------------------
;; Vector form: basic sampling
;; ---------------------------------------------------------------------------

(println "\n-- Vector form: sample shape --")
(let [d (dist/product [(dist/gaussian 0 1) (dist/bernoulli 0.5) (dist/uniform 0 1)])
      v (dc/dist-sample d (rng/fresh-key))]
  (assert-true "vector form returns vector" (vector? v))
  (assert-true "vector form has 3 elements" (= 3 (count v)))
  (assert-true "each element is MLX array" (every? mx/array? v)))

;; ---------------------------------------------------------------------------
;; Map form: basic sampling
;; ---------------------------------------------------------------------------

(println "\n-- Map form: sample shape --")
(let [d (dist/product {:x (dist/gaussian 0 1) :y (dist/uniform 0 1)})
      v (dc/dist-sample d (rng/fresh-key))]
  (assert-true "map form returns map" (map? v))
  (assert-true "map form has correct keys" (= #{:x :y} (set (keys v))))
  (assert-true "each value is MLX array" (every? mx/array? (vals v))))

;; ---------------------------------------------------------------------------
;; Log-prob = sum of component log-probs
;; ---------------------------------------------------------------------------

(println "\n-- Log-prob = sum of component log-probs --")
(let [d1 (dist/gaussian 0 1)
      d2 (dist/uniform 0 1)
      prod (dist/product [d1 d2])
      v1 (mx/scalar 0.5)
      v2 (mx/scalar 0.3)
      lp1 (dc/dist-log-prob d1 v1)
      lp2 (dc/dist-log-prob d2 v2)
      lp-sum (mx/add lp1 lp2)
      lp-prod (dc/dist-log-prob prod [v1 v2])]
  (mx/eval! lp-sum lp-prod)
  (assert-close "log-prob = sum of components"
                (mx/item lp-sum) (mx/item lp-prod) 1e-6))

;; Map form log-prob
(println "\n-- Map form log-prob --")
(let [d1 (dist/gaussian 2 1)
      d2 (dist/gaussian -1 3)
      prod (dist/product {:a d1 :b d2})
      va (mx/scalar 1.0)
      vb (mx/scalar 0.0)
      lp1 (dc/dist-log-prob d1 va)
      lp2 (dc/dist-log-prob d2 vb)
      lp-sum (mx/add lp1 lp2)
      lp-prod (dc/dist-log-prob prod {:a va :b vb})]
  (mx/eval! lp-sum lp-prod)
  (assert-close "map log-prob = sum of components"
                (mx/item lp-sum) (mx/item lp-prod) 1e-6))

;; ---------------------------------------------------------------------------
;; Reparam works for all-continuous
;; ---------------------------------------------------------------------------

(println "\n-- Reparam for all-continuous --")
(let [prod (dist/product [(dist/gaussian 0 1) (dist/gaussian 5 2)])
      v (dc/dist-reparam prod (rng/fresh-key))]
  (assert-true "reparam returns vector" (vector? v))
  (assert-true "reparam vector has 2 elements" (= 2 (count v))))

;; Reparam throws for non-reparameterizable component
(println "\n-- Reparam throws for discrete component --")
(let [prod (dist/product [(dist/gaussian 0 1) (dist/bernoulli 0.5)])]
  (assert-true "reparam throws for mixed"
               (try (dc/dist-reparam prod (rng/fresh-key))
                    false
                    (catch :default _ true))))

;; ---------------------------------------------------------------------------
;; Support: Cartesian product for discrete
;; ---------------------------------------------------------------------------

(println "\n-- Support: Cartesian product --")
(let [prod (dist/product [(dist/bernoulli 0.5) (dist/bernoulli 0.3)])
      sup (dc/dist-support prod)]
  (assert-true "support has 4 elements (2x2)" (= 4 (count sup)))
  (assert-true "support elements are vectors" (every? vector? sup)))

;; Map form support
(println "\n-- Map form support --")
(let [prod (dist/product {:a (dist/bernoulli 0.5) :b (dist/bernoulli 0.7)})
      sup (dc/dist-support prod)]
  (assert-true "map support has 4 elements" (= 4 (count sup)))
  (assert-true "map support elements are maps" (every? map? sup)))

;; Support throws for continuous
(println "\n-- Support throws for continuous --")
(let [prod (dist/product [(dist/gaussian 0 1)])]
  (assert-true "support throws for continuous"
               (try (dc/dist-support prod)
                    false
                    (catch :default _ true))))

;; ---------------------------------------------------------------------------
;; GFI integration: trace a product inside a gen body
;; ---------------------------------------------------------------------------

(println "\n-- GFI integration --")
(let [model (dyn/auto-key
              (gen []
                (let [v (trace :joint (dist/product [(dist/gaussian 0 1)
                                                     (dist/gaussian 5 2)]))]
                  v)))
      tr (p/simulate model [])]
  (assert-true "trace retval is vector" (vector? (:retval tr)))
  (assert-true "trace choices has :joint" (cm/has-value? (cm/get-submap (:choices tr) :joint))))

;; GFI generate with constraints
(println "\n-- GFI generate with constraints --")
(let [prod (dist/product [(dist/gaussian 0 1) (dist/gaussian 5 2)])
      constrained-val [(mx/scalar 1.0) (mx/scalar 3.0)]
      {:keys [trace weight]} (p/generate prod [] (cm/->Value constrained-val))]
  (assert-true "generate weight is finite" (js/isFinite (mx/item weight)))
  (assert-true "generate retval matches constraint"
               (let [rv (:retval trace)]
                 (and (= 2 (count rv))
                      (< (js/Math.abs (- 1.0 (mx/item (nth rv 0)))) 1e-6)
                      (< (js/Math.abs (- 3.0 (mx/item (nth rv 1)))) 1e-6)))))

;; ---------------------------------------------------------------------------
;; sample-n: correct batch shapes
;; ---------------------------------------------------------------------------

(println "\n-- sample-n: batch shapes --")
(let [prod (dist/product [(dist/gaussian 0 1) (dist/uniform 0 1)])
      samples (dc/dist-sample-n prod (rng/fresh-key) 10)]
  (assert-true "sample-n returns vector" (vector? samples))
  (assert-true "sample-n vector has 2 components" (= 2 (count samples)))
  (assert-true "component 0 shape is [10]" (= [10] (mx/shape (nth samples 0))))
  (assert-true "component 1 shape is [10]" (= [10] (mx/shape (nth samples 1)))))

;; Map form sample-n
(println "\n-- Map form sample-n --")
(let [prod (dist/product {:x (dist/gaussian 0 1) :y (dist/uniform 0 1)})
      samples (dc/dist-sample-n prod (rng/fresh-key) 5)]
  (assert-true "map sample-n returns map" (map? samples))
  (assert-true "map sample-n has correct keys" (= #{:x :y} (set (keys samples))))
  (assert-true ":x shape is [5]" (= [5] (mx/shape (:x samples))))
  (assert-true ":y shape is [5]" (= [5] (mx/shape (:y samples)))))

;; ---------------------------------------------------------------------------
;; Statistical validation: sample means
;; ---------------------------------------------------------------------------

(println "\n-- Statistical validation --")
(let [prod (dist/product [(dist/gaussian 3.0 0.5) (dist/gaussian -2.0 1.0)])
      samples (dc/dist-sample-n prod (rng/fresh-key) 5000)
      mean0 (mx/item (mx/mean (nth samples 0)))
      mean1 (mx/item (mx/mean (nth samples 1)))]
  (assert-close "component 0 mean ≈ 3.0" 3.0 mean0 0.2)
  (assert-close "component 1 mean ≈ -2.0" -2.0 mean1 0.2))

(println "\n=== Product Distribution Tests Complete ===")
