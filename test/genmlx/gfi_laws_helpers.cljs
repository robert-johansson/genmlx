(ns genmlx.gfi-laws-helpers
  "Shared model families, pools, generators, and helper functions for the
   GFI law test suite. Split files (gfi_laws_test_p1..p9) require this."
  (:require [clojure.test.check.generators :as gen]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.gfi :as gfi]
            [genmlx.test-helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn ev
  "Realize MLX scalar. Alias for h/realize."
  [x] (h/realize x))

(defn close?
  "Like h/close? but with finiteness guard and wider default tolerance
   appropriate for property-based tests across diverse model families."
  ([a b] (close? a b 0.05))
  ([a b tol] (and (h/finite? a) (h/finite? b) (h/close? a b tol))))

;; ---------------------------------------------------------------------------
;; Model families
;; ---------------------------------------------------------------------------

;; Family 1: Single-site continuous
(def single-gaussian
  {:model (dyn/auto-key (gen [] (trace :x (dist/gaussian 0 1))))
   :args []
   :label "single-gaussian"})

(def single-uniform
  {:model (dyn/auto-key (gen [] (trace :x (dist/uniform 0 1))))
   :args []
   :label "single-uniform"})

(def single-exponential
  {:model (dyn/auto-key (gen [] (trace :x (dist/exponential 1))))
   :args []
   :label "single-exponential"})

(def single-beta
  {:model (dyn/auto-key (gen [] (trace :x (dist/beta-dist 2 5))))
   :args []
   :label "single-beta"})

;; Family 2: Multi-site independent
(def two-independent
  {:model (dyn/auto-key (gen []
                             (let [x (trace :x (dist/gaussian 0 1))
                                   y (trace :y (dist/uniform 0 1))]
                               x)))
   :args []
   :label "two-independent"})

(def three-independent
  {:model (dyn/auto-key (gen []
                             (let [a (trace :a (dist/gaussian 0 2))
                                   b (trace :b (dist/exponential 1))
                                   c (trace :c (dist/uniform -1 1))]
                               a)))
   :args []
   :label "three-independent"})

;; Family 3: Chain models (dependent sites)
(def gaussian-chain
  {:model (dyn/auto-key (gen []
                             (let [x (trace :x (dist/gaussian 0 1))]
                               (trace :y (dist/gaussian x 0.5)))))
   :args []
   :label "gaussian-chain"})

(def three-chain
  {:model (dyn/auto-key (gen []
                             (let [a (trace :a (dist/gaussian 0 1))
                                   b (trace :b (dist/gaussian a 1))]
                               (trace :c (dist/gaussian (mx/add a b) 0.5)))))
   :args []
   :label "three-chain"})

;; Family 4: Mixed discrete/continuous
(def mixed-disc-cont
  {:model (dyn/auto-key (gen []
                             (let [b (trace :b (dist/bernoulli 0.5))
                                   x (trace :x (dist/gaussian 0 1))]
                               x)))
   :args []
   :label "mixed-disc-cont"})

;; Family 5: Branching models (stochastic control flow)
(def branching-model
  {:model (dyn/auto-key (gen []
                             (let [coin (trace :coin (dist/bernoulli 0.3))]
                               (mx/eval! coin)
                               (if (pos? (mx/item coin))
                                 (trace :heads (dist/gaussian 10 1))
                                 (trace :tails (dist/uniform 0 1))))))
   :args []
   :label "branching"})

;; Family 6: Models with arguments
(def linear-regression
  {:model (dyn/auto-key (gen [x-val]
                             (let [slope (trace :slope (dist/gaussian 0 5))
                                   intercept (trace :intercept (dist/gaussian 0 5))]
                               (trace :y (dist/gaussian (mx/add (mx/multiply slope x-val)
                                                                intercept) 1)))))
   :args [(mx/scalar 2.0)]
   :label "linear-regression"})

;; Family 6b: More models with arguments (gradient coverage)
(def single-arg-model
  {:model (dyn/auto-key (gen [mu]
                             (trace :x (dist/gaussian mu 1))))
   :args [(mx/scalar 3.0)]
   :label "single-arg"})

(def two-arg-model
  {:model (dyn/auto-key (gen [mu sigma]
                             (trace :x (dist/gaussian mu sigma))))
   :args [(mx/scalar 0.0) (mx/scalar 2.0)]
   :label "two-arg"})

;; Family 7: Splice models (compositional)
(def splice-inner
  (gen [] (trace :a (dist/gaussian 0 1))))

(def splice-dependent
  {:model (dyn/auto-key
           (gen []
                (let [a (splice :inner splice-inner)]
                  (trace :b (dist/gaussian a 1)))))
   :args []
   :label "splice-dependent"})

(def splice-independent
  {:model (dyn/auto-key
           (gen []
                (let [_a (splice :inner splice-inner)]
                  (trace :b (dist/gaussian 0 1)))))
   :args []
   :label "splice-independent"})

;; Family 7b: Nested splice (2-level nesting for compositionality)
(def splice-inner-inner
  (gen [] (trace :a (dist/gaussian 0 1))))

(def splice-mid
  (gen []
       (let [a (splice :inner splice-inner-inner)]
         (trace :b (dist/gaussian a 1)))))

(def splice-nested
  {:model (dyn/auto-key
           (gen []
                (let [b (splice :mid splice-mid)]
                  (trace :c (dist/gaussian b 1)))))
   :args []
   :label "splice-nested"})

(def splice-pool [splice-dependent splice-independent splice-nested])

;; Family 8: Larger models (5+ trace sites, diverse distributions)
(def five-site
  {:model (dyn/auto-key (gen []
                             (let [a (trace :a (dist/gaussian 0 1))
                                   b (trace :b (dist/exponential 1))
                                   c (trace :c (dist/uniform -1 1))
                                   d (trace :d (dist/gaussian a 0.5))
                                   e (trace :e (dist/laplace (mx/add b c) 1))]
                               e)))
   :args []
   :label "five-site"})

;; Family 9: Argument-dependent branching
(def arg-branching
  {:model (dyn/auto-key (gen [threshold]
                             (let [x (trace :x (dist/gaussian 0 1))]
                               (mx/eval! x)
                               (if (pos? (mx/item (mx/subtract x threshold)))
                                 (trace :high (dist/exponential 1))
                                 (trace :low (dist/uniform -1 0))))))
   :args [(mx/scalar 0.0)]
   :label "arg-branching"})

;; ---------------------------------------------------------------------------
;; Model pools and generators
;; ---------------------------------------------------------------------------

(def model-pool
  [single-gaussian single-uniform single-exponential single-beta
   two-independent three-independent
   gaussian-chain three-chain
   mixed-disc-cont
   branching-model arg-branching
   linear-regression single-arg-model two-arg-model
   splice-dependent splice-independent splice-nested
   five-site])

(def non-branching-pool
  [single-gaussian single-uniform single-exponential single-beta
   two-independent three-independent
   gaussian-chain three-chain
   mixed-disc-cont
   linear-regression single-arg-model two-arg-model
   splice-dependent splice-independent splice-nested
   five-site])

(def multi-site-pool
  (filter #(let [t (p/simulate (:model %) (:args %))
                 addrs (cm/addresses (:choices t))]
             (and (> (count addrs) 1)
                  (every? (fn [path] (= 1 (count path))) addrs)))
          non-branching-pool))

(def gen-model
  (gen/elements model-pool))

(def gen-nonbranching
  (gen/elements non-branching-pool))

(def gen-multisite
  (gen/elements multi-site-pool))

(def gen-splice
  (gen/elements splice-pool))

(def vectorized-pool
  (filterv #(not (contains? (set (map :label splice-pool)) (:label %)))
           non-branching-pool))

(def gen-vectorizable
  (gen/elements vectorized-pool))

;; ---------------------------------------------------------------------------
;; Differentiable model pools (gradient tests)
;; ---------------------------------------------------------------------------

(def differentiable-pool
  [single-gaussian single-uniform single-exponential single-beta
   two-independent three-independent
   gaussian-chain three-chain
   linear-regression single-arg-model two-arg-model])

(def gen-differentiable
  (gen/elements differentiable-pool))

(def models-with-args
  (filterv #(seq (:args %)) differentiable-pool))

(def gen-with-args
  (gen/elements models-with-args))

;; ---------------------------------------------------------------------------
;; Inference helpers
;; ---------------------------------------------------------------------------

(defn logsumexp
  "Numerically stable log-sum-exp: log(sum(exp(xs)))."
  [xs]
  (let [max-x (apply max xs)]
    (+ max-x (js/Math.log (reduce + (map #(js/Math.exp (- % max-x)) xs))))))

;; ---------------------------------------------------------------------------
;; Moment-matching helpers (Dist laws)
;; ---------------------------------------------------------------------------

(def N-moment-samples 5000)

(defn collect-samples
  "Simulate N traces and collect values at given addresses."
  [model args addrs n]
  (let [results (repeatedly n #(let [tr (p/simulate model args)]
                                 (into {} (map (fn [a]
                                                 [a (ev (cm/get-value
                                                         (cm/get-submap (:choices tr) a)))])
                                               addrs))))]
    (into {} (map (fn [a] [a (mapv #(get % a) results)]) addrs))))

(defn sample-mean [xs] (/ (reduce + xs) (count xs)))
(defn sample-var [xs]
  (let [m (sample-mean xs)]
    (/ (reduce + (map #(* (- % m) (- % m)) xs)) (count xs))))
(defn sample-cov [xs ys]
  (let [mx (sample-mean xs) my (sample-mean ys)]
    (/ (reduce + (map #(* (- %1 mx) (- %2 my)) xs ys)) (count xs))))

;; ---------------------------------------------------------------------------
;; Compiled model pools (compiled path equivalence tests)
;; ---------------------------------------------------------------------------

(def compiled-pool
  (filterv #(some? (get-in % [:model :schema :compiled-simulate]))
           non-branching-pool))

(def gen-compiled
  (gen/elements compiled-pool))

(def compiled-multisite-pool
  (filterv #(let [t (p/simulate (:model %) (:args %))
                  addrs (cm/addresses (:choices t))]
              (and (> (count addrs) 1)
                   (every? (fn [path] (= 1 (count path))) addrs)))
           compiled-pool))

(def gen-compiled-multisite
  (gen/elements compiled-multisite-pool))
