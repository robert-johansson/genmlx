(ns genmlx.gfi-laws-test-p2
  "GFI law tests part 2: UPDATE + REGENERATE + PROJECT laws"
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

(defspec law:update-identity 100
  ;; [T] h_update(tau, tau) = (tau, {})
  ;; update with same choices: weight = 0
  (prop/for-all [m gen-nonbranching]
                (let [t (p/simulate (:model m) (:args m))
                      {:keys [weight]} (p/update (:model m) t (:choices t))]
                  (close? 0.0 (ev weight) 0.01))))

(defspec law:update-density-ratio 100
  ;; [T] §2.3.1 UPDATE: weight = log(p(tau';x')/p(tau;x))
  (prop/for-all [m gen-nonbranching]
                (let [t1 (p/simulate (:model m) (:args m))
                      t2 (p/simulate (:model m) (:args m))
                      old-score (ev (:score t1))
                      {:keys [trace weight]} (p/update (:model m) t1 (:choices t2))
                      new-score (ev (:score trace))
                      w (ev weight)]
                  (close? w (- new-score old-score) 0.1))))

(defspec law:update-round-trip 100
  ;; [T] Proposition 2.3.1
  ;; update(update(t,sigma).trace, discard) recovers original
  (prop/for-all [m gen-nonbranching]
                (let [t (p/simulate (:model m) (:args m))
                      orig-score (ev (:score t))
                      ;; Update with a different trace's choices
                      t2 (p/simulate (:model m) (:args m))
                      {:keys [trace discard]} (p/update (:model m) t (:choices t2))
                      ;; Round-trip back
                      {:keys [trace]} (p/update (:model m) trace discard)
                      recovered-score (ev (:score trace))]
                  (close? orig-score recovered-score 0.05))))

(defspec law:update-discard-completeness 100
  ;; [T] §2.3.1 UPDATE — discard holds original values at overwritten addresses
  (prop/for-all [m gen-nonbranching]
                (let [{:keys [model args]} m
                      t1 (p/simulate model args)
                      t2 (p/simulate model args)
                      {:keys [discard]} (p/update model t1 (:choices t2))]
                  (every? (fn [path]
                            (close? (ev (cm/get-choice (:choices t1) path))
                                    (ev (cm/get-choice discard path))
                                    1e-6))
                          (cm/addresses discard)))))

;; ---------------------------------------------------------------------------
;; UPDATE-WITH-DIFFS laws [T] §2.3.1 (optimization extension)
;; ---------------------------------------------------------------------------

(defspec law:update-with-diffs-equivalence 100
  ;; update-with-diffs with :unknown argdiffs = update
  (prop/for-all [m gen-nonbranching]
                (let [t1 (p/simulate (:model m) (:args m))
                      t2 (p/simulate (:model m) (:args m))
                      sigma (:choices t2)
                      upd (p/update (:model m) t1 sigma)
                      uwd (p/update-with-diffs (:model m) t1 sigma :unknown)]
                  (and (close? (ev (:weight upd)) (ev (:weight uwd)) 1e-6)
                       (close? (ev (:score (:trace upd)))
                               (ev (:score (:trace uwd))) 1e-6)))))

(t/deftest law:update-with-diffs-nochange-identity
  ;; NoChange + empty constraints = identity (weight = 0)
  (t/testing "update-with-diffs with no-change diffs and empty constraints"
    (let [model (:model gaussian-chain)
          t (p/simulate model [])
          result (p/update-with-diffs model t cm/EMPTY diff/no-change)
          w (ev (:weight result))]
      (t/is (close? w 0.0 1e-6)
            (str "Expected weight=0, got " w)))))

;; ---------------------------------------------------------------------------
;; REGENERATE laws [D] regenerate
;; ---------------------------------------------------------------------------

(defspec law:regenerate-empty-identity 100
  ;; regenerate(t, none).weight = 0
  (prop/for-all [m gen-nonbranching]
                (let [t (p/simulate (:model m) (:args m))
                      {:keys [weight]} (p/regenerate (:model m) t sel/none)]
                  (close? 0.0 (ev weight) 0.01))))

(defspec law:regenerate-preserves-unselected 100
  ;; regenerate(t, S) preserves values at addresses NOT in S
  (prop/for-all [m gen-multisite]
                (let [t (p/simulate (:model m) (:args m))
                      addrs (cm/addresses (:choices t))
                      selected-addr (first (first addrs))
                      unselected (map first (rest addrs))
                      orig-vals (into {} (map (fn [a]
                                                [a (ev (cm/get-value
                                                        (cm/get-submap (:choices t) a)))])
                                              unselected))
                      {:keys [trace]} (p/regenerate (:model m) t
                                                    (sel/select selected-addr))]
                  (every? (fn [a]
                            (let [v (ev (cm/get-value (cm/get-submap (:choices trace) a)))]
                              (close? (get orig-vals a) v 1e-6)))
                          unselected))))

;; ---------------------------------------------------------------------------
;; PROJECT laws [D] project
;; ---------------------------------------------------------------------------

(defspec law:project-all-equals-score 100
  ;; project(t, all) = trace.score
  (prop/for-all [m gen-nonbranching]
                (let [t (p/simulate (:model m) (:args m))
                      proj (ev (p/project (:model m) t sel/all))
                      s (ev (:score t))]
                  (close? proj s 0.01))))

(defspec law:project-none-equals-zero 100
  ;; project(t, none) = 0
  (prop/for-all [m gen-nonbranching]
                (let [t (p/simulate (:model m) (:args m))
                      proj (ev (p/project (:model m) t sel/none))]
                  (close? 0.0 proj 0.01))))

(t/run-tests)
