(ns genmlx.gfi-laws-test-p4
  "GFI law tests part 4: AGREEMENT + GRADIENT + IS laws"
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
;; CROSS-OPERATION consistency laws
;; ---------------------------------------------------------------------------

(defspec law:generate-assess-agreement 100
  ;; generate(P,x,sigma).trace.score = assess(P,x,sigma).weight
  ;; (for fully constrained sigma)
  (prop/for-all [m gen-nonbranching]
                (let [t (p/simulate (:model m) (:args m))
                      {:keys [trace]} (p/generate (:model m) (:args m) (:choices t))
                      gs (ev (:score trace))
                      {:keys [weight]} (p/assess (:model m) (:args m) (:choices t))
                      aw (ev weight)]
                  (close? gs aw 0.01))))

(defspec law:propose-assess-agreement 100
  ;; Cross-operation consistency: propose weight = assess weight
  ;; Complements law:propose-weight-equals-generate (propose vs generate)
  (prop/for-all [m gen-nonbranching]
                (let [{:keys [choices weight]} (p/propose (:model m) (:args m))
                      pw (ev weight)
                      {:keys [weight]} (p/assess (:model m) (:args m) choices)
                      aw (ev weight)]
                  (close? pw aw 0.01))))

;; ---------------------------------------------------------------------------
;; GRADIENT laws [T] Eq 2.12, §2.3.1
;; ---------------------------------------------------------------------------

(defspec law:gradient-choice-correctness 50
  ;; [T] Eq 2.12 — choice gradients match finite differences
  (prop/for-all [m gen-differentiable]
                (let [{:keys [model args]} m
                      t (p/simulate model args)
                      addrs (->> (:choices t) cm/addresses (mapv first))
                      grads (grad/choice-gradients model t addrs)
                      h 1e-3]
                  (every?
                   (fn [addr]
                     (let [v (ev (cm/get-value (cm/get-submap (:choices t) addr)))
                           choices-plus (cm/set-choice (:choices t) [addr]
                                                       (mx/scalar (+ v h)))
                           choices-minus (cm/set-choice (:choices t) [addr]
                                                        (mx/scalar (- v h)))
                           score-plus (-> (p/generate model args choices-plus)
                                          :trace :score ev)
                           score-minus (-> (p/generate model args choices-minus)
                                           :trace :score ev)
                           fd-grad (/ (- score-plus score-minus) (* 2 h))
                           analytical (ev (get grads addr))]
                       (close? analytical fd-grad 0.05)))
                   addrs))))

(defspec law:gradient-argument-correctness 50
  ;; [T] §2.3.1 — AD argument gradients match finite differences
  ;; Uses gen-with-args so every trial exercises a model with arguments
  (prop/for-all [m gen-with-args]
                (let [{:keys [model args]} m
                      t (p/simulate model args)
                      choices (:choices t)
                      args-v (vec args)
                      h 1e-3]
                  (every?
                   (fn [i]
                     (let [x-val (nth args-v i)
                            ;; AD gradient
                           score-fn (fn [x-arr]
                                      (:weight (p/generate model
                                                           (assoc args-v i x-arr)
                                                           choices)))
                           analytical (ev ((mx/grad score-fn) x-val))
                            ;; FD gradient
                           x-num (ev x-val)
                           sp (ev (:weight (p/generate model
                                                       (assoc args-v i (mx/scalar (+ x-num h)))
                                                       choices)))
                           sm (ev (:weight (p/generate model
                                                       (assoc args-v i (mx/scalar (- x-num h)))
                                                       choices)))
                           fd (/ (- sp sm) (* 2 h))]
                       (close? analytical fd 0.05)))
                   (range (count args-v))))))

;; ---------------------------------------------------------------------------
;; INFERENCE laws (importance sampling) [T] Alg 2, Eq 3.2-3.5
;; ---------------------------------------------------------------------------

(defspec law:is-weight-formula 100
  ;; [T] Alg 2, Eq 3.2 — IS weight = project(trace, obs_selection)
  (prop/for-all [m gen-multisite]
                (let [{:keys [model args]} m
                      t (p/simulate model args)
                      obs-addr (first (first (cm/addresses (:choices t))))
                      obs-val (cm/get-value (cm/get-submap (:choices t) obs-addr))
                      obs (cm/choicemap obs-addr obs-val)
                      {:keys [trace weight]} (p/generate model args obs)
                      w (ev weight)
                      expected (ev (p/project model trace (sel/select obs-addr)))]
                  (close? w expected 0.01))))

(defspec law:proposal-support-coverage 100
  ;; [T] Eq 3.3 — no -Inf weights means proposal covers model support
  (prop/for-all [m gen-multisite]
                (let [{:keys [model args]} m
                      t (p/simulate model args)
                      obs-addr (first (first (cm/addresses (:choices t))))
                      obs-val (cm/get-value (cm/get-submap (:choices t) obs-addr))
                      obs (cm/choicemap obs-addr obs-val)
                      weights (repeatedly 20
                                          #(ev (:weight (p/generate model args obs))))]
                  (every? h/finite? weights))))

(defspec law:log-ml-estimate-well-defined 50
  ;; [T] Eq 3.5 — IS log-ML estimate is finite (well-defined) for any model.
  ;; Convergence to analytical value tested in log-ml-convergence-analytical.
  (prop/for-all [m gen-multisite]
                (let [{:keys [model args]} m
                      t (p/simulate model args)
                      obs-addr (first (first (cm/addresses (:choices t))))
                      obs-val (cm/get-value (cm/get-submap (:choices t) obs-addr))
                      obs (cm/choicemap obs-addr obs-val)
                      weights (repeatedly 100
                                          #(ev (:weight (p/generate model args obs))))
                      max-w (apply max weights)
                      log-ml (+ max-w
                                (js/Math.log
                                 (/ (reduce + (map #(js/Math.exp (- % max-w)) weights))
                                    (count weights))))]
                  (h/finite? log-ml))))

(t/deftest log-ml-convergence-analytical
  (t/testing "IS log-ML converges to analytical value for Normal-Normal conjugate"
    (let [conj-model (dyn/auto-key
                      (gen []
                           (let [mu (trace :mu (dist/gaussian 0 2))]
                             (trace :y (dist/gaussian mu 1)))))
          obs (cm/choicemap :y (mx/scalar 1.5))
          ;; Analytical: log N(1.5; 0, sqrt(5))
          ;; = -0.5*log(2*pi) - 0.5*log(5) - 0.5*(2.25/5)
          analytical (- (- (* 0.5 (js/Math.log (* 2 js/Math.PI))))
                        (* 0.5 (js/Math.log 5))
                        (* 0.5 (/ 2.25 5)))
          n-samples 5000
          weights (repeatedly n-samples
                              #(ev (:weight (p/generate conj-model [] obs))))
          log-ml (- (logsumexp weights) (js/Math.log n-samples))]
      (t/is (close? log-ml analytical 0.1)
            (str "log-ML " log-ml " not close to analytical " analytical)))))

(t/run-tests)
