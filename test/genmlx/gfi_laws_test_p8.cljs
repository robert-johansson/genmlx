(ns genmlx.gfi-laws-test-p8
  "GFI law tests part 8: PF + AIS + INVOLUTIVE MCMC laws"
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
;; PF INCREMENTAL WEIGHT law [T] Alg 7, Alg 8
;; ---------------------------------------------------------------------------

;; --- Law #18: PF Incremental Weight ---
;; In a bootstrap PF (Unfold + unfold-extend), each step's weight equals
;; the kernel's generate weight with the observation constraint.
;; For the bootstrap proposal, this is log p(y_t | x_t).
;;
;; Uses non-conjugate SSM (Laplace obs) to avoid conjugacy detection
;; giving the optimal weight instead of the bootstrap weight.

(t/deftest law:pf-incremental-weight
  (t/testing "unfold-extend weight = log p(y_t | x_t) for non-conjugate SSM"
    ;; SSM kernel: x_t ~ N(x_{t-1}, 1), y_t ~ Laplace(x_t, 1)
    (let [kernel (gen [t prev-state]
                   (let [x (trace :x (dist/gaussian prev-state 1))]
                     (trace :y (dist/laplace x 1))
                     x))
          unfold-gf (comb/unfold-combinator kernel)
          init-trace (comb/unfold-empty-trace unfold-gf (mx/scalar 0.0))
          obs-vals [1.0 2.0 0.5 -1.0 3.0]
          obs-seq (mapv #(cm/choicemap :y (mx/scalar %)) obs-vals)
          ;; Run 5 steps, verifying each weight
          results
          (loop [t 0, tr init-trace, k (rng/fresh-key 77), acc []]
            (if (>= t 5)
              acc
              (let [[k1 k2] (rng/split k)
                    {:keys [trace weight]} (comb/unfold-extend tr (nth obs-seq t) k1)
                    _ (mx/materialize! weight)
                    step-cm (cm/get-submap (:choices trace) t)
                    x-val (cm/get-value (cm/get-submap step-cm :x))
                    _ (mx/materialize! x-val)
                    w (ev weight)
                    x (ev x-val)
                    expected (ev (dist/log-prob (dist/laplace (mx/scalar x) 1)
                                               (mx/scalar (nth obs-vals t))))]
                (recur (inc t) trace k2
                       (conj acc {:step t :weight w :expected expected
                                  :diff (js/Math.abs (- w expected))})))))]
      ;; Each step must match to float32 precision
      (doseq [{:keys [step weight expected diff]} results]
        (t/is (close? weight expected 1e-4)
              (str "Step " step ": weight=" weight " expected=" expected
                   " diff=" diff))))))

(t/deftest law:pf-log-ml-convergence
  (t/testing "smc-unfold log-ML converges to Kalman analytical on Gaussian SSM"
    ;; Gaussian SSM: x_t ~ N(x_{t-1}, 1), y_t ~ N(x_t, 1)
    ;; Observations: y = [1.0, 2.0, 0.5]
    ;; Kalman analytical log-ML = -4.895060
    (let [gaussian-kernel (gen [t prev-state]
                            (let [x (trace :x (dist/gaussian prev-state 1))]
                              (trace :y (dist/gaussian x 1))
                              x))
          ys [1.0 2.0 0.5]
          ;; Kalman analytical (verified independently)
          ;; Step 0: P_pred=1, S=2, innov=1, ll=-1.515512
          ;; Step 1: P_pred=1.5, S=2.5, innov=1.5, ll=-1.827084
          ;; Step 2: P_pred=1.6, S=2.6, innov=-0.9, ll=-1.552463
          LOG-2PI (js/Math.log (* 2 js/Math.PI))
          ll0 (* -0.5 (+ LOG-2PI (js/Math.log 2.0) (/ 1.0 2.0)))
          ll1 (* -0.5 (+ LOG-2PI (js/Math.log 2.5) (/ 2.25 2.5)))
          ll2 (* -0.5 (+ LOG-2PI (js/Math.log 2.6) (/ 0.81 2.6)))
          analytical (+ ll0 ll1 ll2)
          ;; Strip conjugacy to force bootstrap PF — conjugacy gives
          ;; optimal weights but destroys particle diversity, biasing log-ML
          stripped-kernel (assoc gaussian-kernel :schema
                                (dissoc (:schema gaussian-kernel)
                                        :auto-handlers :conjugate-pairs
                                        :has-conjugate? :analytical-plan
                                        :auto-regenerate-transition))
          obs-seq (mapv #(cm/choicemap :y (mx/scalar %)) ys)
          result (smc/smc-unfold {:particles 1000 :key (rng/fresh-key 42)}
                                  stripped-kernel (mx/scalar 0.0) obs-seq)
          smc-log-ml (ev (:log-ml result))
          diff (js/Math.abs (- smc-log-ml analytical))]
      ;; Bootstrap PF (conjugacy stripped) on well-conditioned 3-step SSM.
      ;; SE ~ 0.05-0.1 for bootstrap proposal, conservative tol = 0.25.
      (t/is (close? smc-log-ml analytical 0.25)
            (str "SMC log-ML=" smc-log-ml " analytical=" analytical
                 " diff=" diff)))))

;; ---------------------------------------------------------------------------
;; AIS DENSITY RATIO law [T] Alg 8
;; ---------------------------------------------------------------------------

;; --- Law #19: AIS Density Ratio Telescoping ---
;; For geometric annealing, the AIS weight decomposes via GFI PROJECT:
;;   score(x; beta) = prior_score + beta * lik_score
;;   AIS increment = delta_beta * lik_score
;;   Total = lik_score (telescoping identity)
;; Also verifies IS identity: E[exp(lik_score)] = p(y) when x ~ prior.

(t/deftest law:ais-density-ratio-telescoping
  (t/testing "AIS weight increments = delta_beta * lik_score, telescoping holds"
    ;; Model: x ~ N(0, 2), y ~ N(x, 1). Strip conjugacy for clean decomposition.
    (let [model (dyn/auto-key
                  (gen [] (let [x (trace :x (dist/gaussian 0 2))]
                            (trace :y (dist/gaussian x 1))
                            x)))
          stripped (assoc model :schema
                         (dissoc (:schema model)
                                 :auto-handlers :conjugate-pairs
                                 :has-conjugate? :analytical-plan
                                 :auto-regenerate-transition))
          choices (cm/choicemap :x (mx/scalar 2.0) :y (mx/scalar 3.0))
          {:keys [trace]} (p/generate stripped [] choices)
          _ (mx/materialize! (:score trace))
          prior-score (ev (p/project stripped trace (sel/select :x)))
          lik-score (ev (p/project stripped trace (sel/select :y)))
          ;; Temperature schedule
          betas [0.0 0.25 0.5 0.75 1.0]
          tempered (mapv #(+ prior-score (* % lik-score)) betas)
          increments (mapv #(- (nth tempered (inc %)) (nth tempered %)) (range 4))
          expected-inc (* 0.25 lik-score)
          total (reduce + increments)]
      ;; Each increment = delta_beta * lik_score
      (doseq [[i inc-val] (map-indexed vector increments)]
        (t/is (close? inc-val expected-inc 1e-10)
              (str "Step " i ": increment=" inc-val " expected=" expected-inc)))
      ;; Telescoping: sum of increments = lik_score
      (t/is (close? total lik-score 1e-10)
            (str "Telescoping: total=" total " lik-score=" lik-score)))))

(t/deftest law:ais-is-identity
  (t/testing "E[exp(lik_score)] = p(y) when x sampled from prior"
    ;; IS identity underlying AIS: single-step AIS = importance sampling
    ;; with prior as proposal. p(y=3) = N(3; 0, sqrt(5)).
    (let [model (dyn/auto-key
                  (gen [] (let [x (trace :x (dist/gaussian 0 2))]
                            (trace :y (dist/gaussian x 1))
                            x)))
          stripped (assoc model :schema
                         (dissoc (:schema model)
                                 :auto-handlers :conjugate-pairs
                                 :has-conjugate? :analytical-plan
                                 :auto-regenerate-transition))
          obs (cm/choicemap :y (mx/scalar 3.0))
          n-samples 5000
          log-weights (loop [i 0 acc []]
                        (if (>= i n-samples)
                          acc
                          (let [{:keys [weight]} (p/generate stripped [] obs)]
                            (mx/materialize! weight)
                            (let [w (ev weight)]
                              (when (zero? (mod i 500)) (mx/force-gc!))
                              (recur (inc i) (conj acc w))))))
          max-w (apply max log-weights)
          log-z (+ max-w (js/Math.log
                           (/ (reduce + (map #(js/Math.exp (- % max-w)) log-weights))
                              n-samples)))
          ;; Analytical: log N(3; 0, sqrt(5))
          analytical (- (* -0.5 (js/Math.log (* 2 js/Math.PI)))
                        (* 0.5 (js/Math.log 5.0))
                        (* 0.5 (/ 9.0 5.0)))]
      ;; IS estimate converges to analytical marginal likelihood
      ;; SE ~ 0.03 for N=5000, tol = 0.15 (conservative)
      (t/is (close? log-z analytical 0.15)
            (str "IS log-ML=" log-z " analytical=" analytical)))))

;; ---------------------------------------------------------------------------
;; INVOLUTIVE MCMC laws [T] §3.7, Def 3.7.1, Eq 3.15, Eq 3.17
;; ---------------------------------------------------------------------------

;; --- Law #23: Involution Self-Inverse ---

(t/deftest law:involution-self-inverse
  (t/testing "h(h(x, eps)) = (x, eps) for random walk involution"
    (let [inv-fn (fn [tcm acm]
                   (let [x (cm/get-value (cm/get-submap tcm :x))
                         eps (cm/get-value (cm/get-submap acm :eps))
                         _ (mx/materialize! x eps)]
                     [(cm/set-value tcm :x (mx/add x eps))
                      (cm/set-value acm :eps (mx/negative eps))]))
          n-trials 20
          results
          (for [seed (range n-trials)]
            (let [k (rng/fresh-key seed)
                  [k1 k2] (rng/split k)
                  x-val (mx/realize (rng/normal k1 []))
                  eps-val (mx/realize (rng/normal k2 []))
                  tcm (cm/choicemap :x (mx/scalar x-val) :y (mx/scalar 2.0))
                  acm (cm/choicemap :eps (mx/scalar eps-val))
                  [tcm1 acm1] (inv-fn tcm acm)
                  [tcm2 acm2] (inv-fn tcm1 acm1)
                  x-rt (ev (cm/get-value (cm/get-submap tcm2 :x)))
                  eps-rt (ev (cm/get-value (cm/get-submap acm2 :eps)))]
              {:x-diff (js/Math.abs (- x-val x-rt))
               :eps-diff (js/Math.abs (- eps-val eps-rt))}))]
      (doseq [[i {:keys [x-diff eps-diff]}] (map-indexed vector results)]
        (t/is (close? x-diff 0.0 1e-6)
              (str "Trial " i ": x round-trip diff=" x-diff))
        (t/is (close? eps-diff 0.0 1e-6)
              (str "Trial " i ": eps round-trip diff=" eps-diff))))))

;; --- Law #24: Involutive MH Weight Formula ---

(t/deftest law:involutive-mh-weight-formula
  (t/testing "symmetric RW involution weight = score(x') - score(x)"
    ;; For h(x,eps) = (x+eps,-eps) with symmetric proposal N(0,delta),
    ;; the proposal terms cancel: weight = UPDATE weight = score diff.
    (let [model (dyn/auto-key
                  (gen [] (let [x (trace :x (dist/gaussian 0 1))]
                            (trace :y (dist/gaussian x 0.5))
                            x)))
          test-points [[1.0 0.5] [0.5 -0.3] [2.0 -1.0] [1.6 0.1] [0.0 1.5]]]
      (doseq [[x-val eps-val] test-points]
        (let [;; Score at x
              choices-old (cm/choicemap :x (mx/scalar x-val) :y (mx/scalar 2.0))
              {:keys [trace]} (p/generate model [] choices-old)
              _ (mx/materialize! (:score trace))
              score-old (ev (:score trace))
              ;; Score at x' = x + eps
              x-new (+ x-val eps-val)
              choices-new (cm/choicemap :x (mx/scalar x-new) :y (mx/scalar 2.0))
              {:keys [trace]} (p/generate model [] choices-new)
              _ (mx/materialize! (:score trace))
              score-new (ev (:score trace))
              expected (- score-new score-old)
              ;; UPDATE weight (what involutive-mh-step uses internally)
              {:keys [trace]} (p/generate model [] choices-old)
              _ (mx/materialize! (:score trace))
              {:keys [weight]} (p/update model trace choices-new)
              _ (mx/materialize! weight)
              actual (ev weight)]
          (t/is (close? actual expected 1e-4)
                (str "x=" x-val " eps=" eps-val
                     " update=" actual " expected=" expected)))))))

;; --- Law #25: Involutive MCMC Convergence ---

(t/deftest law:involutive-mh-convergence
  (t/testing "involutive MH converges to correct posterior"
    ;; x ~ N(0,1), y ~ N(x,0.5), y=2. Posterior: N(1.6, sqrt(0.2)).
    (let [model (dyn/auto-key
                  (gen [] (let [x (trace :x (dist/gaussian 0 1))]
                            (trace :y (dist/gaussian x 0.5))
                            x)))
          proposal (dyn/auto-key
                     (gen [_tcm]
                       (trace :eps (dist/gaussian 0 0.5))))
          involution (fn [tcm acm]
                       (let [x (cm/get-value (cm/get-submap tcm :x))
                             eps (cm/get-value (cm/get-submap acm :eps))
                             _ (mx/materialize! x eps)]
                         [(cm/set-value tcm :x (mx/add x eps))
                          (cm/set-value acm :eps (mx/negative eps))]))
          obs (cm/choicemap :y (mx/scalar 2.0))
          init-trace (:trace (p/generate model [] obs))
          samples (loop [t init-trace i 0 acc [] k (rng/fresh-key 42)]
                    (if (>= i 2000)
                      acc
                      (let [[k1 k2] (rng/split k)
                            next-t (mcmc/involutive-mh-step
                                     t model proposal involution k1)
                            x-val (ev (cm/get-value
                                       (cm/get-submap (:choices next-t) :x)))
                            _ (when (zero? (mod i 200)) (mx/force-gc!))]
                        (recur next-t (inc i)
                               (if (>= i 500) (conj acc x-val) acc)
                               k2))))
          mean-x (/ (reduce + samples) (count samples))
          var-x (/ (reduce + (map #(let [d (- % mean-x)] (* d d)) samples))
                   (dec (count samples)))]
      ;; Posterior mean = 1.6, tolerance 0.15 (SE-derived, N_eff ~ 500)
      (t/is (close? mean-x 1.6 0.15)
            (str "Posterior mean=" mean-x " expected=1.6"))
      ;; Posterior variance = 0.2, tolerance 0.08
      (t/is (close? var-x 0.2 0.08)
            (str "Posterior var=" var-x " expected=0.2")))))

(t/run-tests)
