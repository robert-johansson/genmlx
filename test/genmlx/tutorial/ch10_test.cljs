(ns genmlx.tutorial.ch10-test
  "Test file for Tutorial Chapter 10: Extensions and Verification."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.gfi :as gfi]
            [genmlx.verify :as verify])
  (:require-macros [genmlx.gen :refer [gen]]
                   [genmlx.dist.macros :refer [defdist]]))

(def pass (atom 0))
(def fail (atom 0))

(defn assert-true [msg pred]
  (if pred
    (do (swap! pass inc) (println "  PASS:" msg))
    (do (swap! fail inc) (println "  FAIL:" msg))))

;; ============================================================
;; Listing 10.1: defdist — custom distribution
;; ============================================================
(println "\n== Listing 10.1: defdist ==")

(defdist my-uniform-int
  "Uniform integer distribution on [lo, hi]."
  [lo hi]
  (sample [key]
    (let [range-size (- hi lo -1)
          u (rng/uniform key [] mx/float32)
          idx (mx/astype (mx/floor (mx/multiply u (mx/scalar range-size))) mx/int32)]
      (mx/add idx (mx/scalar lo mx/int32))))
  (log-prob [value]
    (let [range-size (- hi lo -1)]
      (mx/negative (mx/log (mx/scalar range-size))))))

(let [d (my-uniform-int 1 6)
      key (rng/fresh-key)
      v (dc/dist-sample d key)
      lp (dc/dist-log-prob d v)]
  (assert-true "custom dist samples" (mx/array? v))
  (assert-true "sample in [1,6]" (let [x (mx/item v)] (and (>= x 1) (<= x 6))))
  (assert-true "log-prob is MLX array" (mx/array? lp)))

;; ============================================================
;; Listing 10.2: map->dist — quick custom distribution
;; ============================================================
(println "\n== Listing 10.2: map->dist ==")

(let [d (dc/map->dist {:type :my-spike
                        :sample (fn [key] (mx/scalar 42.0))
                        :log-prob (fn [value] (mx/scalar 0.0))})
      v (dc/dist-sample d (rng/fresh-key))]
  (assert-true "map->dist creates a dist" (instance? dc/Distribution d))
  (assert-true "samples 42" (= 42.0 (mx/item v))))

;; ============================================================
;; Listing 10.3: Custom dist in a model
;; ============================================================
(println "\n== Listing 10.3: custom dist in model ==")

(let [model (dyn/auto-key (gen []
              (trace :roll (my-uniform-int 1 6))))
      trace (p/simulate model [])]
  (assert-true "model with custom dist simulates" (some? trace))
  (assert-true "roll is in [1,6]"
               (let [v (mx/item (cm/get-choice (:choices trace) [:roll]))]
                 (and (>= v 1) (<= v 6)))))

;; ============================================================
;; Listing 10.4: GFI contracts
;; ============================================================
(println "\n== Listing 10.4: GFI laws ==")

(let [model (dyn/auto-key (gen []
              (let [mu (trace :mu (dist/gaussian 0 1))]
                (trace :x (dist/gaussian mu 1))
                mu)))
      result (gfi/verify model [] :n-trials 1 :tags [:core])]
  (assert-true "verify returns a result" (map? result))
  (assert-true "has :total-pass" (contains? result :total-pass))
  (assert-true "has :total-fail" (contains? result :total-fail))
  (assert-true "has :all-pass?" (contains? result :all-pass?))
  (let [tp (:total-pass result)
        tf (:total-fail result)]
    (assert-true "some laws pass" (pos? tp))
    (println (str "  laws: " tp " pass, " tf " fail"))))

;; ============================================================
;; Listing 10.5: Static validation
;; ============================================================
(println "\n== Listing 10.5: static validation ==")

(let [model (dyn/auto-key (gen []
              (trace :x (dist/gaussian 0 1))))
      result (verify/validate-gen-fn model [] {:n-trials 3})]
  (assert-true "validation returns a result" (map? result))
  (assert-true "valid model passes" (:valid? result)))

;; ============================================================
;; Summary
;; ============================================================
(println (str "\n== Chapter 10 tests: " @pass " PASS, " @fail " FAIL =="))
(when (pos? @fail) (js/process.exit 1))
