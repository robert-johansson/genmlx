(ns genmlx.project-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.dist.core :as dc]
            [genmlx.combinators :as comb])
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

(println "\n=== Project Tests ===\n")

;; --- Basic: project with sel/all should equal trace score ---
(println "-- project all = score --")
(let [model (gen []
              (dyn/trace :x (dist/gaussian 0 1))
              (dyn/trace :y (dist/gaussian 0 1))
              nil)
      trace (p/simulate model [])
      weight (p/project model trace sel/all)]
  (mx/eval! weight (:score trace))
  (assert-close "project all = trace score"
                (mx/item (:score trace)) (mx/item weight) 0.001))

;; --- Project with sel/none should return 0 ---
(println "\n-- project none = 0 --")
(let [model (gen []
              (dyn/trace :x (dist/gaussian 0 1))
              (dyn/trace :y (dist/gaussian 0 1))
              nil)
      trace (p/simulate model [])
      weight (p/project model trace sel/none)]
  (mx/eval! weight)
  (assert-close "project none = 0" 0.0 (mx/item weight) 0.001))

;; --- Project with subset selection ---
(println "\n-- project subset --")
(let [model (gen []
              (dyn/trace :x (dist/gaussian 0 1))
              (dyn/trace :y (dist/gaussian 0 1))
              nil)
      constraints (cm/choicemap :x (mx/scalar 2.0) :y (mx/scalar 3.0))
      {:keys [trace]} (p/generate model [] constraints)
      ;; Project just :x
      weight-x (p/project model trace (sel/select :x))
      ;; Manually compute expected: log-prob of x=2.0 under N(0,1)
      expected-lp-x (dc/dist-log-prob (dist/gaussian 0 1) (mx/scalar 2.0))]
  (mx/eval! weight-x expected-lp-x)
  (assert-close "project :x = log-prob of x"
                (mx/item expected-lp-x) (mx/item weight-x) 0.001))

;; --- Project subset: :y only ---
(println "\n-- project :y only --")
(let [model (gen []
              (dyn/trace :x (dist/gaussian 0 1))
              (dyn/trace :y (dist/gaussian 0 1))
              nil)
      constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar -1.0))
      {:keys [trace]} (p/generate model [] constraints)
      weight-y (p/project model trace (sel/select :y))
      expected-lp-y (dc/dist-log-prob (dist/gaussian 0 1) (mx/scalar -1.0))]
  (mx/eval! weight-y expected-lp-y)
  (assert-close "project :y = log-prob of y"
                (mx/item expected-lp-y) (mx/item weight-y) 0.001))

;; --- Complement selection ---
(println "\n-- project complement --")
(let [model (gen []
              (dyn/trace :x (dist/gaussian 0 1))
              (dyn/trace :y (dist/gaussian 0 1))
              nil)
      constraints (cm/choicemap :x (mx/scalar 1.5) :y (mx/scalar -0.5))
      {:keys [trace]} (p/generate model [] constraints)
      weight-not-x (p/project model trace (sel/complement-sel (sel/select :x)))
      expected-lp-y (dc/dist-log-prob (dist/gaussian 0 1) (mx/scalar -0.5))]
  (mx/eval! weight-not-x expected-lp-y)
  (assert-close "project complement(:x) = log-prob of y"
                (mx/item expected-lp-y) (mx/item weight-not-x) 0.001))

;; --- Project on distribution directly ---
(println "\n-- project distribution --")
(let [d (dist/gaussian 5 2)
      trace (p/simulate d [])
      weight (p/project d trace sel/all)]
  (mx/eval! weight (:score trace))
  (assert-close "dist project = score" (mx/item (:score trace)) (mx/item weight) 0.001))

;; --- Splice: model with sub-GF ---
(println "\n-- project with splice --")
(let [sub-model (gen [mu]
                  (dyn/trace :z (dist/gaussian mu 1))
                  nil)
      model (gen []
              (let [x (dyn/trace :x (dist/gaussian 0 1))]
                (dyn/splice :sub sub-model (mx/item x))
                nil))
      trace (p/simulate model [])
      weight-all (p/project model trace sel/all)]
  (mx/eval! weight-all (:score trace))
  (assert-close "project all with splice = trace score"
                (mx/item (:score trace)) (mx/item weight-all) 0.001))

;; --- Map combinator ---
(println "\n-- project map combinator --")
(let [kernel (gen [x]
               (dyn/trace :y (dist/gaussian x 1))
               nil)
      mapped (comb/map-combinator kernel)
      constraints (cm/choicemap
                    0 (cm/choicemap :y (mx/scalar 1.0))
                    1 (cm/choicemap :y (mx/scalar 2.0))
                    2 (cm/choicemap :y (mx/scalar 3.0)))
      {:keys [trace]} (p/generate mapped [[0.0 0.0 0.0]] constraints)
      weight-all (p/project mapped trace sel/all)]
  (mx/eval! weight-all (:score trace))
  (assert-close "map project all = score"
                (mx/item (:score trace)) (mx/item weight-all) 0.001))

;; --- Map combinator: select single element ---
(println "\n-- project map combinator subset --")
(let [kernel (gen [x]
               (dyn/trace :y (dist/gaussian x 1))
               nil)
      mapped (comb/map-combinator kernel)
      constraints (cm/choicemap
                    0 (cm/choicemap :y (mx/scalar 1.0))
                    1 (cm/choicemap :y (mx/scalar 2.0)))
      {:keys [trace]} (p/generate mapped [[0.0 0.0]] constraints)
      ;; Select only element 0
      weight-0 (p/project mapped trace (sel/hierarchical 0 sel/all))
      expected-lp (dc/dist-log-prob (dist/gaussian 0 1) (mx/scalar 1.0))]
  (mx/eval! weight-0 expected-lp)
  (assert-close "map project element 0 = log-prob of y=1"
                (mx/item expected-lp) (mx/item weight-0) 0.001))

;; --- Switch combinator ---
(println "\n-- project switch combinator --")
(let [branch-a (gen []
                 (dyn/trace :v (dist/gaussian 0 1))
                 nil)
      branch-b (gen []
                 (dyn/trace :v (dist/gaussian 10 1))
                 nil)
      sw (comb/switch-combinator branch-a branch-b)
      constraints (cm/choicemap :v (mx/scalar 0.5))
      {:keys [trace]} (p/generate sw [0] constraints)
      weight (p/project sw trace sel/all)]
  (mx/eval! weight (:score trace))
  (assert-close "switch project = score"
                (mx/item (:score trace)) (mx/item weight) 0.001))

;; --- Unfold combinator ---
(println "\n-- project unfold combinator --")
(let [kernel (gen [t state]
               (let [v (dyn/trace :v (dist/gaussian state 1))]
                 (mx/eval! v)
                 (mx/item v)))
      uf (comb/unfold-combinator kernel)
      ;; 3 timesteps, init-state=0
      constraints (cm/choicemap
                    0 (cm/choicemap :v (mx/scalar 1.0))
                    1 (cm/choicemap :v (mx/scalar 2.0))
                    2 (cm/choicemap :v (mx/scalar 3.0)))
      {:keys [trace]} (p/generate uf [3 0.0] constraints)
      weight-all (p/project uf trace sel/all)]
  (mx/eval! weight-all (:score trace))
  (assert-close "unfold project all = score"
                (mx/item (:score trace)) (mx/item weight-all) 0.001))

;; --- Unfold combinator: subset ---
(println "\n-- project unfold subset --")
(let [kernel (gen [t state]
               (let [v (dyn/trace :v (dist/gaussian state 1))]
                 (mx/eval! v)
                 (mx/item v)))
      uf (comb/unfold-combinator kernel)
      constraints (cm/choicemap
                    0 (cm/choicemap :v (mx/scalar 1.0))
                    1 (cm/choicemap :v (mx/scalar 2.0)))
      {:keys [trace]} (p/generate uf [2 0.0] constraints)
      ;; Select only timestep 0
      weight-0 (p/project uf trace (sel/hierarchical 0 sel/all))
      ;; v=1.0, state=0 → N(0,1) log-prob
      expected-lp (dc/dist-log-prob (dist/gaussian 0 1) (mx/scalar 1.0))]
  (mx/eval! weight-0 expected-lp)
  (assert-close "unfold project timestep 0"
                (mx/item expected-lp) (mx/item weight-0) 0.001))

;; --- Mask combinator ---
(println "\n-- project mask combinator --")
(let [inner (gen []
              (dyn/trace :v (dist/gaussian 0 1))
              nil)
      masked (comb/mask-combinator inner)
      ;; Active=true
      constraints-active (cm/choicemap :v (mx/scalar 1.5))
      {:keys [trace]} (p/generate masked [true] constraints-active)
      weight-active (p/project masked trace sel/all)]
  (mx/eval! weight-active (:score trace))
  (assert-close "mask active project = score"
                (mx/item (:score trace)) (mx/item weight-active) 0.001))

(let [inner (gen []
              (dyn/trace :v (dist/gaussian 0 1))
              nil)
      masked (comb/mask-combinator inner)
      ;; Active=false → score is 0, project should be 0
      trace (p/simulate masked [false])
      weight (p/project masked trace sel/all)]
  (mx/eval! weight)
  (assert-close "mask inactive project = 0" 0.0 (mx/item weight) 0.001))

;; --- Manual verification: known values ---
(println "\n-- manual verification --")
(let [model (gen []
              (dyn/trace :x (dist/gaussian 0 1))
              (dyn/trace :y (dist/gaussian 0 1))
              nil)
      constraints (cm/choicemap :x (mx/scalar 0.0) :y (mx/scalar 0.0))
      {:keys [trace]} (p/generate model [] constraints)
      ;; log-prob of 0 under N(0,1) = -0.5*ln(2*pi) ~ -0.9189
      weight-x (p/project model trace (sel/select :x))
      weight-y (p/project model trace (sel/select :y))
      weight-all (p/project model trace sel/all)]
  (mx/eval! weight-x weight-y weight-all)
  (assert-close "project x=0 under N(0,1)" -0.9189 (mx/item weight-x) 0.01)
  (assert-close "project y=0 under N(0,1)" -0.9189 (mx/item weight-y) 0.01)
  (assert-close "project all = x + y" (+ (mx/item weight-x) (mx/item weight-y))
                (mx/item weight-all) 0.001))

(println "\nAll project tests complete.")
