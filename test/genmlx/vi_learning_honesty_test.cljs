;; @tier medium
(ns genmlx.vi-learning-honesty-test
  "Regression tests for the genmlx-7sqe audit cluster: REINFORCE score term
   identically zero, fit :vi mislabeling MAP as VI, ADVI key dishonesty,
   soft-resample ensemble collapse, and the false 'gradients through
   p/generate are always zero' membrane claim. Oracles are closed-form
   identities and hand-computed densities, never the path under test."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.fit :as fit]
            [genmlx.inference.vi :as vi]
            [genmlx.inference.differentiable-resample :as dr])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; (1) REINFORCE estimator: analytic gradient oracle
;; ---------------------------------------------------------------------------
;; For q(z; mu) = N(mu, 1), f(z) = z, samples held fixed (non-reparam):
;; the REINFORCE gradient of E_q[f] at mu is E[(f(z) - b)·(z - mu)]
;; ≈ Cov(f, z) = Var(z) = 1. Pre-fix the score term was identically zero
;; through estimate-objective, so the gradient was 0.

(deftest reinforce-gradient-matches-covariance-identity
  (testing "REINFORCE gradient ≈ Var(z) = 1 for f(z) = z under N(mu,1)"
    (let [k 4096
          samples (rng/normal (rng/fresh-key 42) [k])
          psf identity
          surrogate (fn [params]
                      ((vi/reinforce-estimator
                        psf
                        (fn [z] (dist/log-prob
                                 (dist/gaussian (mx/index params 0) 1) z)))
                       samples))
          g ((mx/grad surrogate) (mx/array [0.0]))
          g-val (h/realize (mx/index g 0))]
      (is (h/close? 1.0 g-val 0.15)
          (str "REINFORCE grad " (.toFixed g-val 4) " ≈ 1.0 (MC tol 0.15)")))))

;; ---------------------------------------------------------------------------
;; (1b) End-to-end: non-reparameterized sampler must still learn
;; ---------------------------------------------------------------------------
;; Target p(z) = N(3, 1); guide q(z; mu) = N(mu, 1) sampled WITH
;; stop-gradient (no pathwise flow). :pwake maximizes E_q[log p], so only
;; the score-function term can move mu. Pre-fix the gradient was exactly
;; zero and mu stayed at its 0.0 init.

(deftest reinforce-trains-nonreparam-guide
  (testing "programmable-vi :pwake + :reinforce moves mu toward the target
            with a stop-gradient sampler"
    (let [log-p (fn [z-arr] (dist/log-prob (dist/gaussian 3 1)
                                           (mx/index z-arr 0)))
          log-q (fn [z-arr params]
                  (dist/log-prob (dist/gaussian (mx/index params 0) 1)
                                 (mx/index z-arr 0)))
          sample-fn (fn [params key n]
                      (mx/stop-gradient
                       (mx/add (mx/index params 0)
                               (rng/normal (rng/ensure-key key) [n 1]))))
          {:keys [params]} (vi/programmable-vi
                            {:iterations 400 :learning-rate 0.05
                             :n-samples 32 :objective :pwake
                             :estimator :reinforce
                             :key (rng/fresh-key 7)}
                            log-p log-q sample-fn (mx/array [0.0]))
          mu-val (h/realize (mx/index params 0))]
      (is (> mu-val 2.0)
          (str "mu " (.toFixed mu-val 3) " moved toward 3 (pre-fix: stuck at 0)")))))

;; ---------------------------------------------------------------------------
;; (1c) :reinforce on sample-coupled objectives throws honestly
;; ---------------------------------------------------------------------------

(deftest reinforce-on-iwelbo-throws
  (testing ":reinforce + :iwelbo is rejected (its score-function form IS :vimco)"
    (let [log-p (fn [z-arr] (dist/log-prob (dist/gaussian 0 1)
                                           (mx/index z-arr 0)))
          log-q (fn [z-arr params]
                  (dist/log-prob (dist/gaussian (mx/index params 0) 1)
                                 (mx/index z-arr 0)))
          sample-fn (fn [params key n]
                      (mx/add (mx/index params 0)
                              (rng/normal (rng/ensure-key key) [n 1])))]
      (is (thrown? js/Error
                   (vi/programmable-vi
                    {:iterations 2 :objective :iwelbo :estimator :reinforce
                     :key (rng/fresh-key 8)}
                    log-p log-q sample-fn (mx/array [0.0])))
          "iwelbo + reinforce throws instead of silently dropping the score term"))))

;; ---------------------------------------------------------------------------
;; (3) ADVI: :key makes the whole run reproducible; sample-fn is honest
;; ---------------------------------------------------------------------------

(defn- advi-target
  "log N(v; 2, 1) over a [1]-shaped parameter vector."
  [v]
  (dist/log-prob (dist/gaussian 2 1) (mx/index v 0)))

(deftest advi-key-reproducibility
  (testing "two vi runs with the same :key produce identical mu and history"
    (let [run #(vi/vi {:iterations 40 :learning-rate 0.05 :elbo-samples 5
                       :key (rng/fresh-key 99)}
                      advi-target (mx/array [0.0]))
          r1 (run)
          r2 (run)]
      (is (= (h/realize (mx/index (:mu r1) 0))
             (h/realize (mx/index (:mu r2) 0)))
          "same :key → bit-identical mu (pre-fix: gradient sampling unkeyed)")
      (is (= (:elbo-history r1) (:elbo-history r2))
          "same :key → identical elbo history"))))

(deftest advi-sample-fn-honest-entropy
  (testing "sample-fn draws fresh entropy per unkeyed call; keyed calls
            are deterministic"
    (let [{:keys [sample-fn]} (vi/vi {:iterations 20 :learning-rate 0.05
                                      :elbo-samples 5 :key (rng/fresh-key 5)}
                                     advi-target (mx/array [0.0]))]
      (is (not= (sample-fn 5) (sample-fn 5))
          "unkeyed calls differ (pre-fix: identical — one fixed key)")
      (is (= (sample-fn 5 (rng/fresh-key 123))
             (sample-fn 5 (rng/fresh-key 123)))
          "keyed calls are deterministic"))))

;; ---------------------------------------------------------------------------
;; (2) fit :vi reports :log-joint, not :log-ml
;; ---------------------------------------------------------------------------
;; Model z ~ N(0,1), x ~ N(z,1), x = 2 observed. MAP z* = 1;
;; joint at z* = logN(1;0,1) + logN(2;1,1) = -log(2π) - 1 ≈ -2.8379.
;; True log-ml = logN(2; 0, sqrt 2) ≈ -2.2655 — the joint-at-optimum is
;; NOT a marginal likelihood, which is exactly why :log-ml must be nil.

(deftest fit-vi-relabels-map-result
  (testing "fit :vi returns :log-joint (joint at MAP) and a nil :log-ml"
    (let [model (gen []
                  (let [z (trace :z (dist/gaussian 0 1))]
                    (trace :x (dist/gaussian z 1))
                    z))
          data (cm/choicemap :x (mx/scalar 2.0))
          result (fit/fit model [] data {:method :vi
                                         :residual-addrs [:z]
                                         :iterations 800 :lr 0.05
                                         :key (rng/fresh-key 3)})
          expected-joint (+ (h/gaussian-lp 1.0 0.0 1.0)
                            (h/gaussian-lp 2.0 1.0 1.0))]
      (is (nil? (:log-ml result)) ":log-ml is nil — MAP yields no evidence")
      (is (h/close? expected-joint (:log-joint result) 0.1)
          (str ":log-joint " (:log-joint result)
               " ≈ joint at MAP " (.toFixed expected-joint 4))))))

;; ---------------------------------------------------------------------------
;; (5) soft-resample: gathers survivors, importance-reweights, differentiable
;; ---------------------------------------------------------------------------

(deftest soft-resample-gathers-not-averages
  (testing "output particles are members of the input set, not its mean"
    ;; Pre-fix every output row was the SAME convex combination (the
    ;; weighted ensemble mean) — total particle collapse.
    (let [particles (mx/array [[0.0] [1.0] [2.0] [3.0] [4.0]])
          log-w (mx/array [0.0 0.0 0.0 0.0 5.0])
          {:keys [particles indices]} (dr/soft-resample particles log-w 0.5
                                                        (rng/fresh-key 21))
          vals (mapv first (h/realize-vec particles))
          idxs (h/realize-vec indices)]
      (is (every? #(some (fn [m] (h/close? m % 1e-6)) [0.0 1.0 2.0 3.0 4.0])
                  vals)
          "every output is one of the input particles")
      (is (= vals (mapv double idxs))
          "gathered values match the returned ancestor indices"))))

(deftest soft-resample-importance-weights-exact
  (testing "returned log-weights equal log w_a - log q(a) computed by hand"
    (let [alpha 0.5
          n 5
          lw [0.5 -0.3 1.2 0.0 -1.0]
          log-w (mx/array lw)
          {:keys [indices log-weights]} (dr/soft-resample
                                         (mx/array [[0.0] [1.0] [2.0] [3.0] [4.0]])
                                         log-w alpha (rng/fresh-key 22))
          idxs (h/realize-vec indices)
          got (h/realize-vec log-weights)
          ;; Hand-computed: normalized w, mixture q = α·w + (1-α)/N
          z (reduce + (map js/Math.exp lw))
          w (mapv #(/ (js/Math.exp %) z) lw)
          q (mapv #(+ (* alpha %) (/ (- 1.0 alpha) n)) w)]
      (doseq [[k a] (map-indexed vector idxs)]
        (is (h/close? (- (js/Math.log (nth w a)) (js/Math.log (nth q a)))
                      (nth got k) 1e-5)
            (str "survivor " k " (ancestor " a ") weight = log w_a - log q_a"))))))

(deftest soft-resample-gradient-flows
  (testing "gradients flow through the returned log-weights"
    (let [particles (mx/array [[0.0] [1.0] [2.0]])
          loss (fn [lw]
                 (mx/sum (:log-weights
                          (dr/soft-resample particles lw 0.5
                                            (rng/fresh-key 23)))))
          g ((mx/grad loss) (mx/array [0.1 0.2 0.3]))
          g-vals (h/realize-vec g)]
      (is (some #(> (js/Math.abs %) 1e-6) g-vals)
          "log-weight gradient is nonzero (the method's entire point)"))))

;; ---------------------------------------------------------------------------
;; (6) Gradients DO flow through p/generate
;; ---------------------------------------------------------------------------
;; amortized.cljs claimed the handler's volatile! makes gradients through
;; p/generate 'always zero'. The volatile! threads host state only; the
;; weight stays one connected lazy graph. Oracle: d/dz [log N(z;0,1) +
;; log N(1; z, 0.5)] at z = 0.5 is -z + (1-z)/0.25 = 1.5.

(deftest gradient-flows-through-p-generate
  (testing "grad of the generate weight wrt a constraint input is analytic"
    (let [model (dyn/auto-key
                 (gen []
                   (let [z (trace :z (dist/gaussian 0 1))]
                     (trace :x (dist/gaussian z 0.5))
                     z)))
          loss (fn [v]
                 (let [cmap (-> cm/EMPTY
                                (cm/set-choice [:z] (mx/index v 0))
                                (cm/set-choice [:x] (mx/scalar 1.0)))]
                   (mx/negative (:weight (p/generate model [] cmap)))))
          g ((mx/grad loss) (mx/array [0.5]))
          g-val (h/realize (mx/index g 0))]
      (is (h/close? -1.5 g-val 1e-4)
          (str "grad " (.toFixed g-val 5) " = -1.5 analytic — NOT zero")))))

(cljs.test/run-tests)
