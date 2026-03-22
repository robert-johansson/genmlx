(ns genmlx.dist-property-test
  "Property-based distribution invariant tests using test.check.
   Verifies that all distributions satisfy core contracts:
   log-prob of own samples is finite, sample-n shape is correct,
   reparam samples carry gradients, and log-prob is deterministic.

   Uses pre-built distribution instances via gen/elements to avoid
   test.check shrinking issues with mx/scalar in nbb/SCI (alpha)."
  (:require [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [cljs.test :as t]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.inference.adev :as adev])
  (:require-macros [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Pre-built distribution instances (avoids shrinking mx/scalar in nbb)
;; ---------------------------------------------------------------------------

(defn- s [v] (mx/scalar (double v)))

;; Each: {:dist Distribution, :label string, :reparam? bool, :continuous? bool}
(def all-dists
  [{:dist (dist/gaussian (s 0) (s 1))       :label "gaussian(0,1)"      :reparam? true  :continuous? true}
   {:dist (dist/gaussian (s 3) (s 0.5))     :label "gaussian(3,0.5)"    :reparam? true  :continuous? true}
   {:dist (dist/gaussian (s -2) (s 5))      :label "gaussian(-2,5)"     :reparam? true  :continuous? true}
   {:dist (dist/uniform (s 0) (s 1))        :label "uniform(0,1)"       :reparam? true  :continuous? true}
   {:dist (dist/uniform (s -5) (s 5))       :label "uniform(-5,5)"      :reparam? true  :continuous? true}
   {:dist (dist/exponential (s 1))           :label "exponential(1)"     :reparam? true  :continuous? true}
   {:dist (dist/exponential (s 0.5))         :label "exponential(0.5)"   :reparam? true  :continuous? true}
   {:dist (dist/laplace (s 0) (s 1))        :label "laplace(0,1)"       :reparam? true  :continuous? true}
   {:dist (dist/laplace (s 2) (s 3))        :label "laplace(2,3)"       :reparam? true  :continuous? true}
   {:dist (dist/log-normal (s 0) (s 1))     :label "log-normal(0,1)"    :reparam? true  :continuous? true}
   {:dist (dist/log-normal (s 1) (s 0.5))   :label "log-normal(1,0.5)"  :reparam? true  :continuous? true}
   {:dist (dist/cauchy (s 0) (s 1))         :label "cauchy(0,1)"        :reparam? true  :continuous? true}
   {:dist (dist/bernoulli (s 0.5))           :label "bernoulli(0.5)"     :reparam? false :continuous? false}
   {:dist (dist/bernoulli (s 0.1))           :label "bernoulli(0.1)"     :reparam? false :continuous? false}
   {:dist (dist/bernoulli (s 0.9))           :label "bernoulli(0.9)"     :reparam? false :continuous? false}
   {:dist (dist/beta-dist (s 2) (s 2))      :label "beta(2,2)"          :reparam? false :continuous? true}
   {:dist (dist/beta-dist (s 0.5) (s 0.5))  :label "beta(0.5,0.5)"     :reparam? false :continuous? true}
   {:dist (dist/gamma-dist (s 2) (s 1))     :label "gamma(2,1)"         :reparam? false :continuous? true}
   {:dist (dist/gamma-dist (s 0.5) (s 2))   :label "gamma(0.5,2)"      :reparam? false :continuous? true}
   {:dist (dist/poisson (s 3))               :label "poisson(3)"         :reparam? false :continuous? false}
   {:dist (dist/poisson (s 0.5))             :label "poisson(0.5)"       :reparam? false :continuous? false}
   {:dist (dist/student-t (s 3) (s 0) (s 1)) :label "student-t(3,0,1)" :reparam? false :continuous? true}
   {:dist (dist/student-t (s 10) (s 0) (s 1)) :label "student-t(10,0,1)" :reparam? false :continuous? true}
   {:dist (dist/delta (s 3.14))              :label "delta(3.14)"        :reparam? false :continuous? false}
   {:dist (dist/delta (s -1))                :label "delta(-1)"          :reparam? false :continuous? false}
   {:dist (dist/geometric (s 0.3))           :label "geometric(0.3)"     :reparam? false :continuous? false}
   {:dist (dist/geometric (s 0.8))           :label "geometric(0.8)"     :reparam? false :continuous? false}])

(def gen-dist (gen/elements all-dists))
(def gen-continuous-dist (gen/elements (filterv :continuous? all-dists)))
(def gen-reparam-dist (gen/elements (filterv :reparam? all-dists)))

;; ---------------------------------------------------------------------------
;; Property 1: log-prob of own sample is finite
;; ---------------------------------------------------------------------------

(defspec log-prob-sample-is-finite 200
  (prop/for-all [d gen-dist]
    (let [key (rng/fresh-key)
          v (dc/dist-sample (:dist d) key)
          lp (dc/dist-log-prob (:dist d) v)]
      (mx/eval! lp)
      (js/isFinite (mx/item lp)))))

;; ---------------------------------------------------------------------------
;; Property 2: log-prob is deterministic (same input -> same output)
;; ---------------------------------------------------------------------------

(defspec log-prob-is-deterministic 100
  (prop/for-all [d gen-dist]
    (let [key (rng/fresh-key)
          v (dc/dist-sample (:dist d) key)
          lp1 (dc/dist-log-prob (:dist d) v)
          lp2 (dc/dist-log-prob (:dist d) v)]
      (mx/eval! lp1 lp2)
      (== (mx/item lp1) (mx/item lp2)))))

;; ---------------------------------------------------------------------------
;; Property 3: sample-n produces correct shape [N]
;; ---------------------------------------------------------------------------

(defspec sample-n-shape-is-n-for-scalar-distributions 100
  (prop/for-all [d gen-dist
                 n (gen/choose 1 50)]
    (let [key (rng/fresh-key)
          samples (dc/dist-sample-n (:dist d) key n)]
      (mx/eval! samples)
      (= n (first (mx/shape samples))))))

;; ---------------------------------------------------------------------------
;; Property 4: sample-n elements have finite log-prob
;; ---------------------------------------------------------------------------

(defspec sample-n-elements-have-finite-log-prob 100
  (prop/for-all [d gen-dist]
    (let [key (rng/fresh-key)
          n 10
          samples (dc/dist-sample-n (:dist d) key n)
          lps (dc/dist-log-prob (:dist d) samples)]
      (mx/eval! lps)
      (let [shape (mx/shape lps)]
        (every? js/isFinite
                (for [i (range (first shape))]
                  (mx/item (mx/index lps i))))))))

;; ---------------------------------------------------------------------------
;; Property 5: reparam sample has finite log-prob
;; ---------------------------------------------------------------------------

(defspec reparam-sample-has-finite-log-prob 100
  (prop/for-all [d gen-reparam-dist]
    (let [key (rng/fresh-key)
          v (dc/dist-reparam (:dist d) key)
          lp (dc/dist-log-prob (:dist d) v)]
      (mx/eval! lp)
      (js/isFinite (mx/item lp)))))

;; ---------------------------------------------------------------------------
;; Property 6: has-reparam? matches expected
;; ---------------------------------------------------------------------------

(defspec has-reparam-is-consistent 100
  (prop/for-all [d gen-dist]
    (= (:reparam? d) (adev/has-reparam? (:dist d)))))

;; ---------------------------------------------------------------------------
;; Property 7: sample is in support for bounded distributions
;; ---------------------------------------------------------------------------

(def uniform-dists
  [{:lo 0.0 :hi 1.0} {:lo -5.0 :hi 5.0} {:lo -100.0 :hi -1.0} {:lo 0.1 :hi 0.2}])

(defspec uniform-sample-is-in-lo-hi 50
  (prop/for-all [spec (gen/elements uniform-dists)]
    (let [lo (:lo spec) hi (:hi spec)
          d (dist/uniform (s lo) (s hi))
          key (rng/fresh-key)
          v (dc/dist-sample d key)]
      (mx/eval! v)
      (let [x (mx/item v)]
        (and (>= x lo) (<= x hi))))))

(defspec bernoulli-sample-is-0-or-1 50
  (prop/for-all [d (gen/elements [(dist/bernoulli (s 0.1))
                                   (dist/bernoulli (s 0.5))
                                   (dist/bernoulli (s 0.9))])]
    (let [key (rng/fresh-key)
          v (dc/dist-sample d key)]
      (mx/eval! v)
      (let [x (mx/item v)]
        (or (== x 0.0) (== x 1.0))))))

(defspec exponential-sample-is-non-negative 50
  (prop/for-all [d (gen/elements [(dist/exponential (s 0.1))
                                   (dist/exponential (s 1))
                                   (dist/exponential (s 5))])]
    (let [key (rng/fresh-key)
          v (dc/dist-sample d key)]
      (mx/eval! v)
      (>= (mx/item v) 0.0))))

(defspec beta-sample-is-in-0-1 50
  (prop/for-all [d (gen/elements [(dist/beta-dist (s 2) (s 2))
                                   (dist/beta-dist (s 0.5) (s 0.5))
                                   (dist/beta-dist (s 1) (s 3))])]
    (let [key (rng/fresh-key)
          v (dc/dist-sample d key)]
      (mx/eval! v)
      (let [x (mx/item v)]
        (and (> x 0.0) (< x 1.0))))))

(defspec gamma-sample-is-positive 50
  (prop/for-all [d (gen/elements [(dist/gamma-dist (s 2) (s 1))
                                   (dist/gamma-dist (s 0.5) (s 2))
                                   (dist/gamma-dist (s 5) (s 0.5))])]
    (let [key (rng/fresh-key)
          v (dc/dist-sample d key)]
      (mx/eval! v)
      (> (mx/item v) 0.0))))

(defspec log-normal-sample-is-positive 50
  (prop/for-all [d (gen/elements [(dist/log-normal (s 0) (s 1))
                                   (dist/log-normal (s 1) (s 0.5))
                                   (dist/log-normal (s -1) (s 2))])]
    (let [key (rng/fresh-key)
          v (dc/dist-sample d key)]
      (mx/eval! v)
      (> (mx/item v) 0.0))))

;; ---------------------------------------------------------------------------
;; Property 8: delta distribution always returns its value
;; ---------------------------------------------------------------------------

(defspec delta-sample-equals-its-value 30
  (prop/for-all [d (gen/elements [{:v 3.14} {:v -1.0} {:v 0.0} {:v 100.0}])]
    (let [dist (dist/delta (s (:v d)))
          key (rng/fresh-key)
          v (dc/dist-sample dist key)]
      (mx/eval! v)
      (< (js/Math.abs (- (:v d) (mx/item v))) 1e-6))))

(defspec delta-log-prob-is-0-at-value--inf-elsewhere 30
  (prop/for-all [d (gen/elements [{:v 3.14} {:v -1.0} {:v 0.0}])]
    (let [dist (dist/delta (s (:v d)))
          lp-at (dc/dist-log-prob dist (s (:v d)))
          lp-away (dc/dist-log-prob dist (s (+ (:v d) 1.0)))]
      (mx/eval! lp-at lp-away)
      (and (< (js/Math.abs (mx/item lp-at)) 1e-6)
           (= ##-Inf (mx/item lp-away))))))

;; ---------------------------------------------------------------------------
;; Property 9: geometric sample is non-negative integer
;; ---------------------------------------------------------------------------

(defspec geometric-sample-is-non-negative-integer 50
  (prop/for-all [d (gen/elements [(dist/geometric (s 0.1))
                                   (dist/geometric (s 0.5))
                                   (dist/geometric (s 0.9))])]
    (let [key (rng/fresh-key)
          v (dc/dist-sample d key)]
      (mx/eval! v)
      (let [x (mx/item v)]
        (and (>= x 0) (== x (js/Math.floor x)))))))

;; ---------------------------------------------------------------------------
;; Property 10: GFI bridge -- dist as generative function
;; ---------------------------------------------------------------------------

(defspec dist-simulate-score-equals-log-prob-sample 100
  (prop/for-all [d gen-dist]
    (let [trace (dc/dist-simulate (:dist d))
          v (:retval trace)
          lp (dc/dist-log-prob (:dist d) v)]
      (mx/eval! (:score trace) lp)
      (< (js/Math.abs (- (mx/item (:score trace)) (mx/item lp))) 0.01))))

(defspec dist-generate-with-constraint-weight-equals-log-prob 100
  (prop/for-all [d gen-continuous-dist]
    (let [key (rng/fresh-key)
          v (dc/dist-sample (:dist d) key)
          constraint (cm/->Value v)
          {:keys [weight]} (dc/dist-generate (:dist d) constraint)
          lp (dc/dist-log-prob (:dist d) v)]
      (mx/eval! weight lp)
      (< (js/Math.abs (- (mx/item weight) (mx/item lp))) 0.01))))

;; ---------------------------------------------------------------------------
;; Multi-dimensional distribution invariants
;; ---------------------------------------------------------------------------

;; Pre-built MVN instances: varying dimensionality and covariance structure
(def mvn-pool
  [{:dim 2 :label "MVN(2d, I)"
    :dist (dist/multivariate-normal (mx/zeros [2]) (mx/eye 2))}
   {:dim 3 :label "MVN(3d, I)"
    :dist (dist/multivariate-normal (mx/zeros [3]) (mx/eye 3))}
   {:dim 5 :label "MVN(5d, I)"
    :dist (dist/multivariate-normal (mx/zeros [5]) (mx/eye 5))}
   {:dim 2 :label "MVN(2d, diag)"
    :dist (dist/multivariate-normal (mx/array [1.0 -1.0])
                                    (mx/array [[2.0 0.0] [0.0 0.5]]))}
   {:dim 3 :label "MVN(3d, shifted)"
    :dist (dist/multivariate-normal (mx/array [3.0 -2.0 1.0])
                                    (mx/eye 3))}])

;; Pre-built Dirichlet instances: varying dimension and concentration
(def dirichlet-pool
  [{:k 2 :label "Dir([1,1])"
    :dist (dist/dirichlet (mx/array [1.0 1.0]))}
   {:k 3 :label "Dir([2,3,1])"
    :dist (dist/dirichlet (mx/array [2.0 3.0 1.0]))}
   {:k 5 :label "Dir([0.5 x5])"
    :dist (dist/dirichlet (mx/array [0.5 0.5 0.5 0.5 0.5]))}
   {:k 4 :label "Dir([10,10,10,10])"
    :dist (dist/dirichlet (mx/array [10.0 10.0 10.0 10.0]))}])

(defspec mvn-log-prob-sample-is-finite 50
  (prop/for-all [spec (gen/elements mvn-pool)]
    (let [key (rng/fresh-key)
          v (dc/dist-sample (:dist spec) key)
          lp (dc/dist-log-prob (:dist spec) v)]
      (mx/eval! lp)
      (js/isFinite (mx/item lp)))))

(defspec mvn-sample-shape-equals-d 50
  (prop/for-all [spec (gen/elements mvn-pool)]
    (let [key (rng/fresh-key)
          v (dc/dist-sample (:dist spec) key)]
      (mx/eval! v)
      (= [(:dim spec)] (mx/shape v)))))

(defspec dirichlet-samples-sum-to-1 50
  (prop/for-all [spec (gen/elements dirichlet-pool)]
    (let [key (rng/fresh-key)
          v (dc/dist-sample (:dist spec) key)]
      (mx/eval! v)
      (let [total (mx/item (mx/sum v))
            vals (mx/->clj v)]
        (and (< (js/Math.abs (- total 1.0)) 1e-4)
             (every? #(>= % 0.0) vals))))))

(t/run-tests)
