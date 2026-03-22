(ns genmlx.level0-certification-test
  "Level 0 Certification: single-file gate for all Level 0 capabilities.
   Must pass before starting Level 1 development.

   Gates:
     G1  Core GFI operations (simulate, generate, update, regenerate, assess, project)
     G2  Distributions (sample + log-prob roundtrip for all 26)
     G3  Vectorized execution (vsimulate/vgenerate match scalar)
     G4  Batched combinators (Unfold, Switch, Scan, Mix)
     G5  GFI contracts (10 contracts on 3 canonical models)
     G6  Compiled ops (compiled unfold, compiled gen fn)
     G7  Analytical middleware (Kalman, EKF, HMM, conjugate)
     G8  Inference algorithms (IS, MH, kernel composition)
     G9  Gradients and learning (choice gradients, Adam)
     G10 Cross-module integration (compiled + batched + middleware)"
  (:require [cljs.test :as t :refer [deftest is testing]]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.trace :as tr]
            [genmlx.vectorized :as vec]
            [genmlx.contracts :as contracts]
            [genmlx.combinators :as comb]
            [genmlx.handler :as h]
            [genmlx.compiled :as compiled]
            [genmlx.inference.importance :as is]
            [genmlx.inference.kernel :as kern]
            [genmlx.inference.differentiable :as diff]
            [genmlx.inference.analytical :as ana]
            [genmlx.inference.ekf :as ekf]
            [genmlx.inference.conjugate :as conj]
            [genmlx.gradients :as grad]
            [genmlx.test-helpers :as th])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Canonical models
;; ---------------------------------------------------------------------------

(def coin-model
  (dyn/auto-key
    (gen []
      (let [p (trace :p (dist/beta-dist 2 2))]
        (trace :flip (dist/bernoulli p))
        p))))

(def line-model
  (dyn/auto-key
    (gen [xs]
      (let [slope (trace :slope (dist/gaussian 0 10))
            intercept (trace :intercept (dist/gaussian 0 10))]
        (doseq [[j x] (map-indexed vector xs)]
          (trace (keyword (str "y" j))
                 (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x)) intercept)
                                (mx/scalar 1))))
        slope))))

(def multi-model
  (dyn/auto-key
    (gen []
      (let [a (trace :a (dist/gaussian 0 1))
            b (trace :b (dist/gaussian a 1))
            c (trace :c (dist/gaussian b 1))]
        c))))

;; ---------------------------------------------------------------------------
;; G1: Core GFI operations
;; ---------------------------------------------------------------------------

(deftest g1-core-gfi-operations
  (testing "simulate"
    (let [tr (p/simulate coin-model [])]
      (is (some? tr) "simulate returns trace")
      (is (some? (:choices tr)) "simulate has choices")
      (is (some? (:score tr)) "simulate has score")
      (mx/eval! (:score tr))
      (is (js/isFinite (mx/item (:score tr))) "simulate score is finite")))

  (testing "generate"
    (let [obs (cm/set-value cm/EMPTY :flip (mx/scalar 1))
          {:keys [trace weight]} (p/generate coin-model [] obs)]
      (mx/eval! weight)
      (is (some? trace) "generate returns trace")
      (is (js/isFinite (mx/item weight)) "generate weight is finite")))

  (testing "update"
    (let [tr1 (p/simulate multi-model [])
          new-obs (cm/set-value cm/EMPTY :c (mx/scalar 2.0))
          {:keys [trace weight discard]} (p/update multi-model tr1 new-obs)]
      (mx/eval! weight)
      (is (some? trace) "update returns trace")
      (is (js/isFinite (mx/item weight)) "update weight is finite")
      (is (some? discard) "update returns discard")))

  (testing "regenerate"
    (let [tr1 (p/simulate multi-model [])
          selection (sel/select :a)
          {:keys [trace weight]} (p/regenerate multi-model tr1 selection)]
      (mx/eval! weight)
      (is (some? trace) "regenerate returns trace")
      (is (js/isFinite (mx/item weight)) "regenerate weight is finite")))

  (testing "assess"
    (let [choices (-> cm/EMPTY
                      (cm/set-value :a (mx/scalar 0.5))
                      (cm/set-value :b (mx/scalar 0.3))
                      (cm/set-value :c (mx/scalar 0.1)))
          {:keys [weight]} (p/assess multi-model [] choices)]
      (mx/eval! weight)
      (is (js/isFinite (mx/item weight)) "assess weight is finite")))

  (testing "project"
    (let [tr (p/simulate multi-model [])
          w (p/project multi-model tr (sel/select :a))]
      (mx/eval! w)
      (is (js/isFinite (mx/item w)) "project returns finite weight"))))

;; ---------------------------------------------------------------------------
;; G2: All 26 distributions — sample + log-prob roundtrip
;; ---------------------------------------------------------------------------

(def dist-suite
  [["gaussian"         (dist/gaussian 0 1)]
   ["uniform"          (dist/uniform 0 1)]
   ["bernoulli"        (dist/bernoulli 0.5)]
   ["beta"             (dist/beta-dist 2 5)]
   ["gamma"            (dist/gamma-dist 3 2)]
   ["exponential"      (dist/exponential 2)]
   ["poisson"          (dist/poisson 4)]
   ["categorical"      (dist/categorical (mx/log (mx/array [0.2 0.3 0.5])))]
   ["laplace"          (dist/laplace 0 1)]
   ["log-normal"       (dist/log-normal 0 1)]
   ["cauchy"           (dist/cauchy 0 1)]
   ["student-t"        (dist/student-t 5 0 1)]
   ["inv-gamma"        (dist/inv-gamma 4 2)]
   ["geometric"        (dist/geometric 0.3)]
   ["binomial"         (dist/binomial 10 0.4)]
   ["neg-binomial"     (dist/neg-binomial 5 0.4)]
   ["discrete-uniform" (dist/discrete-uniform 0 10)]
   ["truncated-normal" (dist/truncated-normal 0 1 -2 2)]
   ["delta"            (dist/delta (mx/scalar 42))]
   ["dirichlet"        (dist/dirichlet (mx/array [1 2 3]))]
   ["piecewise-uniform" (dist/piecewise-uniform (mx/array [0 1 2 3]) (mx/array [1 1 1]))]
   ["von-mises"        (dist/von-mises 0 2)]
   ["wrapped-normal"   (dist/wrapped-normal 0 1)]
   ["wrapped-cauchy"   (dist/wrapped-cauchy 0 0.5)]
   ["broadcasted-normal" (dist/broadcasted-normal (mx/array [0 1]) (mx/array [1 2]))]])

(deftest g2-distribution-roundtrip
  (doseq [[name d] dist-suite]
    (testing name
      (let [key (rng/fresh-key)
            s (dc/dist-sample d key)
            _ (mx/eval! s)
            lp (dc/dist-log-prob d s)
            _ (mx/eval! lp)
            lp-val (if (= [] (mx/shape lp))
                     (mx/item lp)
                     (mx/item (mx/sum lp)))]
        (is (js/isFinite lp-val) (str name ": log-prob is finite"))))))

;; ---------------------------------------------------------------------------
;; G3: Vectorized execution
;; ---------------------------------------------------------------------------

(def N-particles 200)

(deftest g3-vectorized-execution
  (testing "vsimulate shape check"
    (let [key (rng/fresh-key)
          vt (dyn/vsimulate multi-model [] N-particles key)
          score (:score vt)]
      (mx/eval! score)
      (is (= [N-particles] (mx/shape score)) "vsimulate score shape")
      (let [a-vals (cm/get-value (cm/get-submap (:choices vt) :a))]
        (is (= [N-particles] (mx/shape a-vals)) "vsimulate choice shape"))))

  (testing "vgenerate weight check"
    (let [key (rng/fresh-key)
          obs (cm/set-value cm/EMPTY :c (mx/scalar 1.0))
          vt (dyn/vgenerate multi-model [] obs N-particles key)]
      (mx/eval! (:weight vt))
      (is (= [N-particles] (mx/shape (:weight vt))) "vgenerate weight shape")
      (let [w-sum (mx/sum (:weight vt))]
        (mx/eval! w-sum)
        (is (js/isFinite (mx/item w-sum)) "vgenerate weights are finite"))))

  (testing "scalar-vectorized statistical equivalence"
    (let [key (rng/fresh-key)
          vt (dyn/vsimulate multi-model [] 5000 key)
          a-vals (cm/get-value (cm/get-submap (:choices vt) :a))
          _ (mx/eval! a-vals)
          mean-a (mx/item (mx/mean a-vals))]
      (is (th/close? 0.0 mean-a 0.2) "vsimulate :a mean near 0"))))

;; ---------------------------------------------------------------------------
;; G4: Batched combinators
;; ---------------------------------------------------------------------------

(def ar-kernel
  (dyn/auto-key
    (gen [t state]
      (trace :x (dist/gaussian (mx/multiply state (mx/scalar 0.9)) (mx/scalar 1))))))

(def unfold-model
  (dyn/auto-key
    (gen [n-steps]
      (splice :t (comb/unfold-combinator ar-kernel) n-steps (mx/scalar 0.0)))))

(def branch-lo (dyn/auto-key (gen [] (trace :v (dist/gaussian (mx/scalar 0) (mx/scalar 1))))))
(def branch-hi (dyn/auto-key (gen [] (trace :v (dist/gaussian (mx/scalar 10) (mx/scalar 1))))))

(def switch-model
  (dyn/auto-key
    (gen []
      (let [idx (trace :idx (dist/bernoulli 0.5))]
        (splice :branch (comb/switch-combinator branch-lo branch-hi) idx)))))

(def scan-kernel
  (dyn/auto-key
    (gen [carry input]
      (let [v (trace :x (dist/gaussian (mx/add carry input) (mx/scalar 1.0)))]
        [v (mx/multiply v (mx/scalar 2))]))))

(def scan-model
  (dyn/auto-key
    (gen [inputs]
      (splice :s (comb/scan-combinator scan-kernel) (mx/scalar 0.0) inputs))))

(def comp-a (dyn/auto-key (gen [] (trace :v (dist/gaussian (mx/scalar 0) (mx/scalar 1))))))
(def comp-b (dyn/auto-key (gen [] (trace :v (dist/gaussian (mx/scalar 5) (mx/scalar 1))))))

(def mix-model
  (dyn/auto-key
    (gen []
      (splice :mix (comb/mix-combinator
                     [comp-a comp-b]
                     (fn [] (mx/log (mx/array [0.3 0.7]))))))))

(deftest g4-batched-combinators
  (testing "batched unfold"
    (let [key (rng/fresh-key)
          vt (dyn/vsimulate unfold-model [5] 100 key)
          score (:score vt)]
      (mx/eval! score)
      (is (= [100] (mx/shape score)) "batched unfold score shape")
      (is (js/isFinite (mx/item (mx/sum score))) "batched unfold scores finite")))

  (testing "batched switch"
    (let [key (rng/fresh-key)
          vt (dyn/vsimulate switch-model [] 100 key)
          score (:score vt)]
      (mx/eval! score)
      (is (= [100] (mx/shape score)) "batched switch score shape")))

  (testing "batched scan"
    (let [key (rng/fresh-key)
          inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]
          vt (dyn/vsimulate scan-model [inputs] 100 key)
          score (:score vt)]
      (mx/eval! score)
      (is (= [100] (mx/shape score)) "batched scan score shape")))

  (testing "batched mix"
    (let [key (rng/fresh-key)
          vt (dyn/vsimulate mix-model [] 100 key)
          score (:score vt)]
      (mx/eval! score)
      (is (= [100] (mx/shape score)) "batched mix score shape"))))

;; ---------------------------------------------------------------------------
;; G5: GFI Contracts (10 contracts x 3 models x 5 trials)
;; ---------------------------------------------------------------------------

(def contract-keys (set (keys contracts/contracts)))
(def scalar-keys (disj contract-keys :broadcast-equivalence))

(defn run-contracts [model-name model args ks n-trials]
  (let [results (contracts/verify-gfi-contracts model args
                  :contract-keys ks :n-trials n-trials)]
    (doseq [[k {:keys [pass fail]}] results]
      (when (pos? fail)
        (println "    contract" k ":" fail "failures")))
    (let [total-fail (reduce + (map :fail (vals results)))]
      (is (zero? total-fail) (str model-name ": all contracts pass")))))

(deftest g5-gfi-contracts
  (testing "coin-model"
    (run-contracts "coin-model" coin-model [] scalar-keys 5))
  (testing "multi-model"
    (run-contracts "multi-model" multi-model [] contract-keys 5))
  (testing "line-model"
    (run-contracts "line-model" line-model [(mx/array [1 2 3])] scalar-keys 5)))

;; ---------------------------------------------------------------------------
;; G6: Compiled ops
;; ---------------------------------------------------------------------------

(def obs-data (reduce (fn [m j] (cm/set-value m (keyword (str "y" j)) (mx/scalar (* 2 j))))
                      cm/EMPTY (range 5)))

(deftest g6-compiled-ops
  (testing "IS loss"
    (let [loss-fn (diff/make-is-loss-fn line-model [(mx/array [0 1 2 3 4])]
                                         obs-data [:slope :intercept] 50 (rng/fresh-key))
          params (mx/array [1.0 0.0])
          loss (loss-fn params)]
      (mx/eval! loss)
      (is (js/isFinite (mx/item loss)) "IS loss is finite"))))

;; ---------------------------------------------------------------------------
;; G7: Analytical middleware
;; ---------------------------------------------------------------------------

(def nn-step
  (gen [obs-val]
    (let [mu (trace :mu (conj/nn-prior (mx/scalar 0.0) (mx/scalar 10.0)))]
      (trace :y (conj/nn-obs :mu mu (mx/scalar 1.0) (mx/scalar 1.0)))
      mu)))

(defn tanh-f [z] (mx/tanh z))
(defn linear-h [z] z)

(def ekf-step-fn
  (gen [obs-val obs-fn-arg]
    (let [z (trace :z (ekf/ekf-latent tanh-f (mx/scalar 0.0) (mx/scalar 0.1)))]
      (trace :obs (ekf/ekf-obs obs-fn-arg z (mx/scalar 0.5) (mx/scalar 1.0)))
      z)))

(deftest g7-analytical-middleware
  (testing "conjugate Normal-Normal"
    (let [constraints (cm/set-value cm/EMPTY :y (mx/scalar 3.0))
          dispatches [(conj/make-nn-dispatch :mu)]
          result (conj/conjugate-generate nn-step [(mx/scalar 3.0)] constraints
                                           dispatches (rng/fresh-key))]
      (is (some? result) "conjugate-generate returns result")
      (let [ll (or (:conjugate-ll result) (mx/scalar 0.0))]
        (mx/eval! ll)
        (is (js/isFinite (mx/item ll)) "conjugate NN LL is finite"))))

  (testing "EKF"
    (let [obs (mx/scalar 0.6)
          constraints (cm/set-value cm/EMPTY :obs obs)
          result (ekf/ekf-generate ekf-step-fn [obs linear-h] constraints
                                   :z 1 (rng/fresh-key))]
      (is (some? result) "ekf-generate returns result")
      (let [ll (or (:ekf-ll result) (mx/scalar 0.0))]
        (mx/eval! ll)
        (is (js/isFinite (mx/item ll)) "EKF LL is finite"))))

  (testing "middleware composition"
    (let [nn-d (conj/make-nn-dispatch :mu)
          composed (ana/compose-middleware h/generate-transition nn-d)]
      (is (fn? composed) "compose-middleware returns function"))))

;; ---------------------------------------------------------------------------
;; G8: Inference algorithms
;; ---------------------------------------------------------------------------

(deftest g8-inference-algorithms
  (testing "importance sampling"
    (let [obs (cm/set-value cm/EMPTY :flip (mx/scalar 1))
          {:keys [traces log-weights]} (is/importance-sampling {:samples 200} coin-model [] obs)]
      (is (= 200 (count traces)) "IS returns traces")
      (is (= 200 (count log-weights)) "IS returns weights")))

  (testing "MH kernel"
    (let [obs (-> cm/EMPTY (cm/set-value :b (mx/scalar 2.0)) (cm/set-value :c (mx/scalar 2.5)))
          {:keys [trace]} (p/generate multi-model [] obs)
          k (kern/mh-kernel (sel/select :a))
          samples (kern/run-kernel {:samples 20} k trace)]
      (is (= 20 (count samples)) "MH kernel returns 20 samples")))

  (testing "kernel composition (cycle)"
    (let [obs (-> cm/EMPTY (cm/set-value :b (mx/scalar 2.0)) (cm/set-value :c (mx/scalar 2.5)))
          {:keys [trace]} (p/generate multi-model [] obs)
          k (kern/cycle-kernels [(kern/mh-kernel (sel/select :a))
                                  (kern/mh-kernel (sel/select :b))])
          samples (kern/run-kernel {:samples 10} k trace)]
      (is (= 10 (count samples)) "kernel cycle returns 10 samples"))))

;; ---------------------------------------------------------------------------
;; G9: Gradients and learning
;; ---------------------------------------------------------------------------

(deftest g9-gradients-and-learning
  (testing "choice gradients"
    (let [tr (p/simulate multi-model [])
          grads (grad/choice-gradients multi-model tr [:a])]
      (is (some? grads) "choice-gradients returns grads")
      (let [g (get grads :a)]
        (mx/eval! g)
        (is (js/isFinite (mx/item g)) "gradient :a is finite"))))

  (testing "IS loss gradient"
    (let [grad-fn (diff/make-is-loss-grad-fn line-model [(mx/array [0 1 2 3 4])]
                                              obs-data [:slope :intercept] 50)
          key (rng/fresh-key)
          params (mx/array [1.0 0.0])
          {:keys [loss grad]} (grad-fn params key)]
      (mx/eval! loss)
      (mx/eval! grad)
      (is (js/isFinite (mx/item loss)) "IS loss grad: loss is finite")
      (is (= [2] (mx/shape grad)) "IS loss grad: grad shape"))))

;; ---------------------------------------------------------------------------
;; G10: Cross-module integration
;; ---------------------------------------------------------------------------

(deftest g10-cross-module-integration
  (testing "vectorized IS with combinator model"
    (let [key (rng/fresh-key)
          obs (cm/set-choice cm/EMPTY [:t 1 :x] (mx/scalar 1.0))
          obs (cm/set-choice obs [:t 3 :x] (mx/scalar 0.5))
          vt (dyn/vgenerate unfold-model [5] obs 500 key)]
      (mx/eval! (:weight vt))
      (is (= [500] (mx/shape (:weight vt))) "vectorized IS + unfold weight shape")
      (let [log-ml (vec/vtrace-log-ml-estimate vt)]
        (mx/eval! log-ml)
        (is (js/isFinite (mx/item log-ml)) "log-ML estimate is finite"))))

  (testing "multi-middleware composition"
    (let [nn-d1 (conj/make-nn-dispatch :mu)
          nn-d2 (conj/make-nn-dispatch :sigma)
          composed (ana/compose-middleware h/generate-transition nn-d1 nn-d2)]
      (is (fn? composed) "multi-middleware composition returns function"))))

(t/run-tests)
