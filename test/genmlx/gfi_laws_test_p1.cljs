(ns genmlx.gfi-laws-test-p1
  "GFI law tests part 1: SIMULATE laws"
  (:require [cljs.test :as t]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.diff :as diff]
            [genmlx.dynamic :as dyn]
            [genmlx.edit :as edit]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.gfi :as gfi]
            [genmlx.gradients :as grad]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.util :as u]
            [genmlx.learning :as learn]
            [genmlx.test-helpers :as h]
            [genmlx.combinators :as comb]
            [genmlx.inference.smc :as smc]
            [genmlx.verify :as verify]
            [genmlx.gfi-laws-helpers :as glh
             :refer [ev close? gen-model gen-nonbranching gen-multisite gen-splice
                     gen-vectorizable gen-differentiable gen-with-args gen-compiled
                     gen-compiled-multisite model-pool non-branching-pool multi-site-pool
                     vectorized-pool differentiable-pool models-with-args compiled-pool
                     compiled-multisite-pool splice-pool
                     single-gaussian single-uniform single-exponential single-beta
                     two-independent three-independent gaussian-chain three-chain
                     mixed-disc-cont branching-model linear-regression single-arg-model
                     two-arg-model splice-inner splice-dependent splice-independent
                     splice-inner-inner splice-mid splice-nested five-site arg-branching
                     logsumexp N-moment-samples collect-samples sample-mean sample-var
                     sample-cov]])
  (:require-macros [genmlx.gen :refer [gen]]
                   [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; SIMULATE laws [T] Def 2.1.16, §2.3.1
;; ---------------------------------------------------------------------------

(defspec law:simulate-produces-trace 100
  ;; [T] Def 2.1.16, §2.3.1 SIMULATE
  ;; simulate(P, x) returns trace with finite score and correct gen-fn
  (prop/for-all [m gen-model]
                (let [t (p/simulate (:model m) (:args m))]
                  (and (some? (:choices t))
                       (= (:gen-fn t) (:model m))
                       (h/finite? (ev (:score t)))))))

(defspec law:simulate-score-is-log-density 100
  ;; [T] §2.3.1 LOGPDF, [D] get_score
  ;; trace.score = assess(P, x, tau).weight
  (prop/for-all [m gen-model]
                (let [t (p/simulate (:model m) (:args m))
                      s (ev (:score t))
                      {:keys [weight]} (p/assess (:model m) (:args m) (:choices t))
                      w (ev weight)]
                  (close? s w 0.01))))

(defspec law:halts-with-probability-one 100
  ;; [T] Def 2.1.16 — simulate(P, x) terminates with probability 1
  (prop/for-all [m gen-model]
                (let [t (p/simulate (:model m) (:args m))]
                  (some? t))))

;; ---------------------------------------------------------------------------
;; GENERATE laws [T] §2.3.1
;; ---------------------------------------------------------------------------

(defspec law:generate-empty-is-simulate 100
  ;; [D] generate with empty constraints = simulate
  ;; weight should be 0
  (prop/for-all [m gen-model]
                (let [{:keys [weight]} (p/generate (:model m) (:args m) cm/EMPTY)]
                  (close? 0.0 (ev weight) 0.01))))

(defspec law:generate-full-weight-equals-score 100
  ;; [T] §2.3.1 GENERATE
  ;; Fully constrained: weight = trace.score = log p-bar(sigma)
  (prop/for-all [m gen-nonbranching]
                (let [t (p/simulate (:model m) (:args m))
                      {:keys [trace weight]} (p/generate (:model m) (:args m)
                                                         (:choices t))]
                  (close? (ev (:score trace)) (ev weight) 0.01))))

(defspec law:return-value-independence 100
  ;; [T] §2.3.1 — f(x,τ) depends only on x and τ
  (prop/for-all [m gen-nonbranching]
                (let [{:keys [model args]} m
                      t (p/simulate model args)
                      r1 (:trace (p/generate model args (:choices t)))
                      r2 (:trace (p/generate model args (:choices t)))]
                  (close? (ev (:retval r1)) (ev (:retval r2)) 1e-10))))

;; ---------------------------------------------------------------------------
;; ASSESS laws [D] assess
;; ---------------------------------------------------------------------------

(defspec law:assess-equals-generate-score 100
  ;; [D] assess(P, x, tau).weight = generate(P, x, tau).trace.score
  (prop/for-all [m gen-nonbranching]
                (let [t (p/simulate (:model m) (:args m))
                      {:keys [weight]} (p/assess (:model m) (:args m) (:choices t))
                      {:keys [trace]} (p/generate (:model m) (:args m) (:choices t))]
                  (close? (ev weight) (ev (:score trace)) 0.01))))

;; ---------------------------------------------------------------------------
;; PROPOSE laws [D] propose
;; ---------------------------------------------------------------------------

(defspec law:propose-weight-equals-generate 100
  ;; [D] propose(P,x).weight = generate(P,x, propose.choices).weight
  (prop/for-all [m gen-nonbranching]
                (let [{:keys [choices weight]} (p/propose (:model m) (:args m))
                      pw (ev weight)
                      {:keys [weight]} (p/generate (:model m) (:args m) choices)
                      gw (ev weight)]
                  (close? pw gw 0.01))))

(defspec law:propose-is-simulate-plus-score 100
  ;; [T] Def 2.1.16: propose weight = assess weight for same choices
  (prop/for-all [m gen-nonbranching]
                (let [{:keys [choices weight]} (p/propose (:model m) (:args m))
                      pw (ev weight)
                      {:keys [weight]} (p/assess (:model m) (:args m) choices)
                      aw (ev weight)]
                  (close? pw aw 1e-5))))

(t/run-tests)
