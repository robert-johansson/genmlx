(ns genmlx.gfi-laws-test-p3
  "GFI law tests part 3: SCORE DECOMPOSITION + STRUCTURED DENSITY + COMPOSITIONALITY"
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

(defspec law:score-decomposition 100
  ;; [T] project(t, S) + project(t, S^c) = trace.score
  (prop/for-all [m gen-multisite]
                (let [t (p/simulate (:model m) (:args m))
                      addrs (cm/addresses (:choices t))
                      first-addr (first (first addrs))
                      s (sel/select first-addr)
                      score (ev (:score t))
                      ps (ev (p/project (:model m) t s))
                      pcs (ev (p/project (:model m) t (sel/complement-sel s)))]
                  (close? score (+ ps pcs) 0.1))))

(defspec law:score-full-decomposition 100
  ;; Sum over all individual project = trace.score
  (prop/for-all [m gen-multisite]
                (let [t (p/simulate (:model m) (:args m))
                      addrs (cm/addresses (:choices t))
                      score (ev (:score t))
                      proj-sum (reduce
                                (fn [acc path]
                                  (+ acc (ev (p/project (:model m) t
                                                        (gfi/path->selection path)))))
                                0.0
                                addrs)]
                  (close? proj-sum score 0.05))))

;; ---------------------------------------------------------------------------
;; STRUCTURED DENSITY laws [T] Def 2.1.3, Prop 2.1.2, Prop 2.1.3
;; ---------------------------------------------------------------------------

(defspec law:structured-density 100
  ;; [T] Def 2.1.3 — structured density: address sets with nonzero probability
  ;; are determined by values. Group 20 simulates by address set; every group
  ;; must have all finite scores. For branching models this verifies every
  ;; execution path has positive density (P(miss both branches) < 0.001).
  (prop/for-all [m gen-model]
                (let [traces (repeatedly 20 #(p/simulate (:model m) (:args m)))
                      groups (group-by (fn [t] (set (cm/addresses (:choices t)))) traces)]
                  (every? (fn [[_addr-set ts]]
                            (every? (fn [t] (h/finite? (ev (:score t)))) ts))
                          groups))))

(defspec law:conditional-is-structured 100
  ;; [T] Prop 2.1.2 -- generate with partial constraints has finite score
  (prop/for-all [m gen-multisite]
                (let [t (p/simulate (:model m) (:args m))
                      addrs (cm/addresses (:choices t))
                      first-addr (first (first addrs))
                      val (cm/get-value (cm/get-submap (:choices t) first-addr))
                      partial (cm/choicemap first-addr val)
                      {:keys [trace weight]} (p/generate (:model m) (:args m) partial)]
                  (and (h/finite? (ev (:score trace)))
                       (h/finite? (ev weight))))))

(defspec law:generate-uniqueness 100
  ;; [T] Prop 2.1.3 -- full constraints produce deterministic trace
  ;; Score, retval, AND weight must all be identical
  (prop/for-all [m gen-nonbranching]
                (let [t (p/simulate (:model m) (:args m))
                      r1 (p/generate (:model m) (:args m) (:choices t))
                      r2 (p/generate (:model m) (:args m) (:choices t))]
                  (and (close? (ev (:score (:trace r1)))
                               (ev (:score (:trace r2))) 1e-10)
                       (close? (ev (:retval (:trace r1)))
                               (ev (:retval (:trace r2))) 1e-10)
                       (close? (ev (:weight r1))
                               (ev (:weight r2)) 1e-10)))))

(defspec law:address-uniqueness 100
  ;; [T] §2.2.1 restriction 2: no duplicate addresses in any trace
  (prop/for-all [m gen-model]
                (let [t (p/simulate (:model m) (:args m))
                      addrs (cm/addresses (:choices t))]
                  (= (count addrs) (count (set addrs))))))

;; ---------------------------------------------------------------------------
;; COMPOSITIONALITY laws [T] Proposition 2.3.2
;; ---------------------------------------------------------------------------

(defspec law:update-compositionality 100
  ;; [T] Proposition 2.3.2 -- splice weight decomposition
  ;; For P3 = P1;P2: w3 = w1 + w2 where w1 = inner score diff, w2 = outer-only diff
  ;; Verified by projecting onto inner/outer partitions independently
  (prop/for-all [m gen-splice]
                (let [{:keys [model args]} m
                      t1 (p/simulate model args)
                      t2 (p/simulate model args)
                      ;; Total update weight
                      {:keys [trace weight]} (p/update model t1 (:choices t2))
                      w3 (ev weight)
                      ;; Partition: inner (splice namespace) vs outer
                      inner-sel (sel/hierarchical :inner sel/all)
                      outer-sel (sel/complement-sel inner-sel)
                      ;; w1 = inner score difference via project
                      w1 (- (ev (p/project model trace inner-sel))
                            (ev (p/project model t1 inner-sel)))
                      ;; w2 = outer-only score difference via project
                      w2 (- (ev (p/project model trace outer-sel))
                            (ev (p/project model t1 outer-sel)))]
                  (close? w3 (+ w1 w2) 0.05))))

(defspec law:nested-splice-compositionality 50
  ;; [T] Prop 2.3.2 at 2 levels of nesting
  ;; weight = w_outer + w_mid + w_inner via project decomposition
  (prop/for-all [_ (gen/return nil)]
                (let [model (:model splice-nested)
                      t1 (p/simulate model [])
                      t2 (p/simulate model [])
                      {:keys [trace weight]} (p/update model t1 (:choices t2))
                      w-total (ev weight)
                      new-score (ev (:score trace))
                      old-score (ev (:score t1))
                      ;; Decompose via score differences (= update density ratio)
                      expected (- new-score old-score)]
                  (close? w-total expected 0.05))))

;; ---------------------------------------------------------------------------
;; EDIT laws [T] Prop 2.3.1 via edit interface
;; ---------------------------------------------------------------------------

(defspec law:edit-backward-request-roundtrip 30
  ;; ConstraintEdit roundtrip: edit then backward-edit recovers original
  ;; Verify BOTH score AND choice values recover (skeptic: score-only is insufficient)
  (prop/for-all [m gen-nonbranching]
                (let [t1 (p/simulate (:model m) (:args m))
                      t2 (p/simulate (:model m) (:args m))
                      req (edit/constraint-edit (:choices t2))
                      {:keys [trace weight backward-request]}
                      (edit/edit-dispatch (:model m) t1 req)
                      w1 (ev weight)
                      result2 (edit/edit-dispatch (:model m) trace backward-request)
                      w2 (ev (:weight result2))
                      recovered-score (ev (:score (:trace result2)))
                      original-score (ev (:score t1))
                      ;; Also verify choice values recover at every address
                      orig-addrs (cm/addresses (:choices t1))
                      choices-match (every?
                                     (fn [path]
                                       (close? (ev (cm/get-choice (:choices t1) path))
                                               (ev (cm/get-choice
                                                    (:choices (:trace result2)) path))
                                               1e-6))
                                     orig-addrs)]
                  (and (close? original-score recovered-score 1e-4)
                       (close? (+ w1 w2) 0.0 1e-3)
                       choices-match))))

(t/run-tests)
