(ns genmlx.gfi-laws-test-p7
  "GFI law tests part 7: SELECTION MH + COMPILED PATH EQUIVALENCE laws"
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
;; SELECTION MH + KERNEL COMPOSITION laws [T] §3.4, §4.1, Prop 3.4.1
;; ---------------------------------------------------------------------------

;; --- Law #30: Optimal Proposal Weight [T] §4.1.3 ---
;; For conjugate models, generate weight = log marginal likelihood.
;; Normal-Normal conjugate: mu ~ N(0,2), y ~ N(mu,1), observe y=3.
;; Analytical: log N(3; 0, sqrt(5)) = -2.62366

(t/deftest law:optimal-proposal-weight
  (t/testing "conjugate generate weight = log marginal likelihood"
    (let [model (dyn/auto-key
                  (gen [] (let [mu (trace :mu (dist/gaussian 0 2))]
                            (trace :y (dist/gaussian mu 1)))))
          obs (cm/choicemap :y (mx/scalar 3.0))
          analytical (- (* -0.5 (js/Math.log (* 2 js/Math.PI)))
                        (* 0.5 (js/Math.log 5.0))
                        (* 0.5 (/ 9.0 5.0)))
          n-trials 10
          results (for [_ (range n-trials)]
                    (let [{:keys [weight]} (p/generate model [] obs)
                          w (ev weight)]
                      (close? w analytical 1e-3)))]
      (t/is (every? true? results)
            (str "Optimal proposal weight failed in "
                 (count (remove true? results)) "/" n-trials " trials")))))

;; --- Law #31: Selection MH Correctness [T] Alg 15 ---
;; regenerate weight = (new_score - old_score) - (new_proj - old_proj)
;; Verified for first leaf address across multi-site models.

(defspec law:selection-mh-correctness 100
  ;; [T] Alg 15 — regenerate weight decomposes as score change at unselected sites
  (prop/for-all [m gen-multisite]
                (let [{:keys [model args]} m
                      t (p/simulate model args)
                      addrs (cm/addresses (:choices t))
                      sel-addr (first (first addrs))
                      sel (sel/select sel-addr)
                      {:keys [trace weight]} (p/regenerate model t sel)
                      w (ev weight)
                      new-score (ev (:score trace))
                      old-score (ev (:score t))
                      old-proj (ev (p/project model t sel))
                      new-proj (ev (p/project model trace sel))
                      expected (- (- new-score old-score)
                                  (- new-proj old-proj))]
                  (and (h/finite? w)
                       (close? w expected 0.01)))))

;; --- Law #32: Simulate Address Set Consistency [T] §4.1.4 ---
;; For static models, simulate always produces the same address set.

(defspec law:simulate-address-set-consistency 100
  ;; [T] §4.1.4, Alg 16 — prerequisite: consistent address structure
  (prop/for-all [m gen-nonbranching]
                (let [{:keys [model args]} m
                      t1 (p/simulate model args)
                      t2 (p/simulate model args)]
                  (and (h/finite? (ev (:score t1)))
                       (h/finite? (ev (:score t2)))
                       (= (set (cm/addresses (:choices t1)))
                          (set (cm/addresses (:choices t2))))))))

;; --- Law #16: Mixture Kernel Stationarity [T] Prop 3.4.1 ---
;; Mix of stationary kernels is stationary. Random-mix and deterministic-cycle
;; of MH kernels on different addresses converge to the same posterior.

(t/deftest law:mixture-kernel-stationarity
  (t/testing "mix-kernels and cycle-kernels both converge to analytical posterior"
    ;; x ~ N(0,1), y ~ N(0,1), obs ~ N(x+y, 0.5), observe obs=2
    ;; s = x+y: prior N(0, sqrt(2)), likelihood N(s, 0.5)
    ;; Posterior mean = 8/4.5 = 1.778
    (let [model (dyn/auto-key
                  (gen []
                    (let [x (trace :x (dist/gaussian 0 1))
                          y (trace :y (dist/gaussian 0 1))]
                      (trace :obs (dist/gaussian (mx/add x y) 0.5)))))
          obs (cm/choicemap :obs (mx/scalar 2.0))
          init (:trace (p/generate model [] obs))
          analytical-mean (/ 8.0 4.5)
          run-chain
          (fn [select-fn n-steps burn]
            (loop [t init i 0 acc [] k (rng/fresh-key 13)]
              (if (>= i n-steps)
                acc
                (let [[k1 k2] (rng/split k)
                      sel (select-fn i k1)
                      {:keys [trace weight]} (p/regenerate model t sel)
                      w (ev weight)
                      accept? (u/accept-mh? w k1)
                      next-t (if accept? trace t)
                      x-val (ev (cm/get-value (cm/get-submap (:choices next-t) :x)))
                      y-val (ev (cm/get-value (cm/get-submap (:choices next-t) :y)))
                      _ (when (zero? (mod i 200)) (mx/force-gc!))]
                  (recur next-t (inc i)
                         (if (>= i burn)
                           (conj acc (+ x-val y-val))
                           acc)
                         k2)))))
          ;; 8000 steps, 2000 burn-in, 6000 post-burn samples.
          ;; Posterior var of x+y: Var[x]+Var[y]+2Cov = 0.4+0.4+0 = 0.8 (approx, ignoring correlation)
          ;; SE = sqrt(0.8/N_eff). Conservative N_eff ~ 1000. SE ~ 0.028, 3.5*SE ~ 0.10.
          mix-samples (run-chain
                       (fn [_ k] (sel/select (if (pos? (mx/item (rng/bernoulli k 0.5 []))) :x :y)))
                       8000 2000)
          cycle-samples (run-chain
                         (fn [i _k] (sel/select (if (even? i) :x :y)))
                         8000 2000)
          mean-mix (/ (reduce + mix-samples) (count mix-samples))
          mean-cycle (/ (reduce + cycle-samples) (count cycle-samples))]
      (t/is (close? mean-mix analytical-mean 0.15)
            (str "Mix mean=" mean-mix " expected=" analytical-mean))
      (t/is (close? mean-cycle analytical-mean 0.15)
            (str "Cycle mean=" mean-cycle " expected=" analytical-mean)))))

;; ---------------------------------------------------------------------------
;; HMC ACCEPTANCE laws [T] Alg 6
;; ---------------------------------------------------------------------------

;; --- Law #52: HMC Acceptance Correctness [T] Alg 6, §3.4.3 ---
;; HMC acceptance probability: alpha = min{1, exp(H(q,p) - H(q',p'))}
;; Leapfrog approximately preserves the Hamiltonian, yielding high acceptance.
;; Verified on Normal-Normal conjugate: x ~ N(0,1), y ~ N(x, 0.5), y=2.
;; Posterior: x|y ~ N(1.6, sqrt(0.2)).

(t/deftest law:hmc-acceptance-correctness
  ;; Normal-Normal conjugate: x ~ N(0,1), y ~ N(x, 0.5), observe y=2.
  ;; Posterior: x|y ~ N(1.6, sqrt(0.2)). Run HMC once, check both moments.
  (let [hmc-model (dyn/auto-key
                   (gen []
                        (let [x (trace :x (dist/gaussian 0 1))]
                          (trace :y (dist/gaussian x 0.5)))))
        obs (cm/choicemap :y (mx/scalar 2.0))
        samples (mcmc/hmc {:samples 500 :burn 200 :step-size 0.05
                           :leapfrog-steps 10 :addresses [:x]
                           :compile? false :device :cpu}
                          hmc-model [] obs)
        x-vals (mapv first samples)
        mean-x (/ (reduce + x-vals) (count x-vals))
        var-x (/ (reduce + (map #(let [d (- % mean-x)] (* d d)) x-vals))
                 (dec (count x-vals)))]
    (t/testing "HMC converges to correct posterior mean"
      (t/is (close? mean-x 1.6 0.3)
            (str "HMC posterior mean=" mean-x " expected=1.6")))
    (t/testing "HMC posterior variance is approximately correct"
      (t/is (close? var-x 0.2 0.15)
            (str "HMC posterior var=" var-x " expected=0.2")))))

;; ---------------------------------------------------------------------------
;; PROPOSAL TRAINING laws [T] Eq 3.8-3.9
;; ---------------------------------------------------------------------------

;; --- Law #53: Proposal Training Objective [T] Eq 3.8-3.9 ---
;; max_theta E_{(sigma+rho)~p}[log q(sigma; rho, theta)]
;;   = min_theta E_rho[D_KL(p(.|rho) || q(.; rho, theta))]
;;
;; Training a parametric model via gradient descent on negative log-likelihood
;; decreases the loss and converges to the MLE. This is the variational
;; principle that underlies proposal training (wake phase).

(t/deftest law:proposal-training-objective
  ;; Model: x ~ N(mu_param, 1), observe x=3. MLE: mu -> 3.
  ;; Train once, verify three properties of the variational principle.
  (let [train-model (dyn/auto-key
                     (gen []
                          (let [mu (param :mu (mx/scalar 0.0))
                                x (trace :x (dist/gaussian mu 1))]
                            x)))
        obs (cm/choicemap :x (mx/scalar 3.0))
        loss-grad-fn (learn/make-param-loss-fn train-model [] obs [:mu])
        init-params (mx/array [0.0])
        {:keys [loss-history params]}
        (learn/train {:iterations 100 :optimizer :adam :lr 0.05
                      :key (rng/fresh-key 42)}
                     loss-grad-fn init-params)
        first-10 (take 10 loss-history)
        last-10 (take-last 10 loss-history)
        mean-first (/ (reduce + first-10) (count first-10))
        mean-last (/ (reduce + last-10) (count last-10))
        final-mu (mx/realize (mx/index params 0))
        final-loss (last loss-history)
        theoretical-min (* 0.5 (js/Math.log (* 2 js/Math.PI)))]
    (t/testing "gradient descent decreases negative log-likelihood"
      (t/is (< mean-last mean-first)
            (str "Loss should decrease: first-10-mean=" mean-first
                 " last-10-mean=" mean-last)))
    (t/testing "optimized parameter converges to MLE"
      (t/is (close? final-mu 3.0 0.5)
            (str "MLE mu=" final-mu " expected=3.0")))
    (t/testing "loss at convergence approaches theoretical minimum"
      ;; Theoretical minimum NLL for N(mu, 1) at MLE: 0.5*log(2*pi) = 0.919
      (t/is (close? final-loss theoretical-min 0.1)
            (str "Final loss=" final-loss " theoretical-min=" theoretical-min)))))

;; ---------------------------------------------------------------------------
;; COMPILED PATH EQUIVALENCE laws [T] Ch 5
;; Laws #47-50: Compiled execution paths must produce the same probability
;; density p(tau; x) as the handler (interpreter) path.
;; ---------------------------------------------------------------------------

;; --- Law #47: Compiled Simulate Equivalence ---
;; compiled simulate score = handler assess weight for the same choices.

(defspec law:compiled-simulate-equivalence 50
  ;; [T] Ch 5 -- compiled simulate preserves p(tau; x)
  ;; Simulate via compiled path, then verify the score via handler assess
  ;; on the same choices. Avoids needing matched PRNG keys.
  (prop/for-all [m gen-compiled]
                (let [{:keys [model args]} m
                      compiled-trace (p/simulate model args)
                      compiled-score (ev (:score compiled-trace))
                      handler-model (dyn/auto-key (gfi/strip-compiled model))
                      {:keys [weight]} (p/assess handler-model args
                                                  (:choices compiled-trace))
                      handler-score (ev weight)]
                  (close? compiled-score handler-score 1e-4))))

;; --- Law #48: Compiled Generate Equivalence ---
;; Fully constrained generate via compiled and handler paths produces
;; identical scores, weights, and return values.

(defspec law:compiled-generate-equivalence 50
  ;; [T] Ch 5 -- compiled generate preserves scores and weights
  (prop/for-all [m gen-compiled]
                (let [{:keys [model args]} m
                      source-trace (p/simulate model args)
                      constraints (:choices source-trace)
                      ;; Compiled generate
                      {:keys [trace weight]} (p/generate model args constraints)
                      compiled-score (ev (:score trace))
                      compiled-weight (ev weight)
                      compiled-retval (ev (:retval trace))
                      ;; Handler generate
                      handler-model (dyn/auto-key (gfi/strip-compiled model))
                      {:keys [trace weight]} (p/generate handler-model args
                                                         constraints)
                      handler-score (ev (:score trace))
                      handler-weight (ev weight)
                      handler-retval (ev (:retval trace))]
                  (and (close? compiled-score handler-score 1e-4)
                       (close? compiled-weight handler-weight 1e-4)
                       (close? compiled-retval handler-retval 1e-4)))))

;; --- Law #49: Compiled Update Equivalence ---
;; Update with new constraints via compiled and handler paths produces
;; identical weights (density ratios) and new scores.

(defspec law:compiled-update-equivalence 50
  ;; [T] Ch 5 -- compiled update preserves density ratios
  (prop/for-all [m gen-compiled]
                (let [{:keys [model args]} m
                      t1 (p/simulate model args)
                      t2 (p/simulate model args)
                      ;; Compiled update
                      {:keys [trace weight]} (p/update model t1 (:choices t2))
                      compiled-weight (ev weight)
                      compiled-new-score (ev (:score trace))
                      ;; Handler update: reconstruct starting trace via generate
                      handler-model (dyn/auto-key (gfi/strip-compiled model))
                      handler-t1 (:trace (p/generate handler-model args
                                                     (:choices t1)))
                      {:keys [trace weight]} (p/update handler-model handler-t1
                                                       (:choices t2))
                      handler-weight (ev weight)
                      handler-new-score (ev (:score trace))]
                  (and (close? compiled-weight handler-weight 1e-4)
                       (close? compiled-new-score handler-new-score 1e-4)))))

;; --- Law #50: Compiled Regenerate Equivalence ---
;; Regenerated trace score equals handler-assessed density for the same
;; choices. Weight is finite. Unselected addresses are preserved.

(defspec law:compiled-regenerate-equivalence 50
  ;; [T] Ch 5 -- compiled regenerate preserves weight semantics
  (prop/for-all [m gen-compiled-multisite]
                (let [{:keys [model args]} m
                      t (p/simulate model args)
                      addrs (cm/addresses (:choices t))
                      sel-addr (first (first addrs))
                      sel (sel/select sel-addr)
                      ;; Compiled regenerate
                      {:keys [trace weight]} (p/regenerate model t sel)
                      compiled-score (ev (:score trace))
                      compiled-weight (ev weight)
                      ;; Verify score via handler assess on regenerated choices
                      handler-model (dyn/auto-key (gfi/strip-compiled model))
                      {:keys [weight]} (p/assess handler-model args
                                                  (:choices trace))
                      handler-score (ev weight)
                      ;; Verify unselected addresses preserved
                      unselected (map first (rest addrs))]
                  (and (close? compiled-score handler-score 1e-4)
                       (h/finite? compiled-weight)
                       (every? (fn [a]
                                 (close?
                                  (ev (cm/get-value (cm/get-submap (:choices t) a)))
                                  (ev (cm/get-value (cm/get-submap (:choices trace) a)))
                                  1e-6))
                               unselected)))))

(t/run-tests)
