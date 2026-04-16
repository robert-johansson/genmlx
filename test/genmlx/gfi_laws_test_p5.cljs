(ns genmlx.gfi-laws-test-p5
  "GFI law tests part 5: MH + REGENERATE WEIGHT + INTERNAL PROPOSAL laws"
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
;; MH + REGENERATE WEIGHT laws [T] Eq 4.1, Alg 5, §3.4.2
;; ---------------------------------------------------------------------------

(defspec law:regenerate-weight-formula 100
  ;; [T] Eq 4.1 — regenerate weight equals score change at unselected sites
  ;; w = (new_score - old_score) - (project(new, S) - project(old, S))
  ;; Verified for BOTH first and last leaf addresses.
  (prop/for-all [m gen-multisite]
                (let [{:keys [model args]} m
                      t (p/simulate model args)
                      addrs (cm/addresses (:choices t))
                      check-addr (fn [addr]
                                   (let [sel (sel/select addr)
                                         old-score (ev (:score t))
                                         old-proj-s (ev (p/project model t sel))
                                         {:keys [trace weight]}
                                         (p/regenerate model t sel)
                                         new-score (ev (:score trace))
                                         new-proj-s (ev (p/project model trace sel))
                                         w (ev weight)
                                         expected (- (- new-score old-score)
                                                     (- new-proj-s old-proj-s))]
                                     (close? w expected 0.01)))
                      first-addr (first (first addrs))
                      last-addr (first (last addrs))]
                  (and (check-addr first-addr)
                       (check-addr last-addr)))))

(defspec law:mh-acceptance-correctness 10
  ;; [T] Alg 5, §3.4.2 — MH with regenerate weight converges to correct posterior.
  ;; Prior: x ~ N(0,1), Likelihood: y ~ N(x, 0.5), observe y=2.
  ;; Posterior: x | y=2 ~ N(1.6, sqrt(0.2)).
  ;; 10 trials x 2000 steps (500 burn-in, 1500 post-burn samples).
  ;; SE = sqrt(0.2/1500) * 3.5 = 0.040. Tolerance 0.15 (conservative for ESS < N).
  (prop/for-all [_ (gen/return nil)]
                (let [mh-model (dyn/auto-key
                                (gen []
                                     (let [x (trace :x (dist/gaussian 0 1))]
                                       (trace :y (dist/gaussian x 0.5)))))
                      obs (cm/choicemap :y (mx/scalar 2.0))
                      init-trace (:trace (p/generate mh-model [] obs))
                      sel (sel/select :x)
                      samples (loop [t init-trace i 0 acc [] k (rng/fresh-key 7)]
                                (if (>= i 2000)
                                  acc
                                  (let [[k1 k2] (rng/split k)
                                        {:keys [trace weight]}
                                        (p/regenerate mh-model t sel)
                                        w (ev weight)
                                        accept? (u/accept-mh? w k1)
                                        next-t (if accept? trace t)
                                        x-val (ev (cm/get-value
                                                   (cm/get-submap (:choices next-t) :x)))
                                        _ (when (zero? (mod i 200)) (mx/force-gc!))]
                                    (recur next-t (inc i)
                                           (if (>= i 500) (conj acc x-val) acc)
                                           k2))))
                      mean (/ (reduce + samples) (count samples))]
                  (close? mean 1.6 0.15))))

(defspec law:mh-proposal-reversibility 100
  ;; [T] §3.4.2 — regenerate-based MH is reversible: forward and reverse
  ;; moves both produce valid traces with finite weights
  (prop/for-all [m gen-multisite]
                (let [{:keys [model args]} m
                      t (p/simulate model args)
                      sel-addr (first (first (cm/addresses (:choices t))))
                      sel (sel/select sel-addr)
                      ;; Forward move
                      {fwd-trace :trace fwd-weight :weight}
                      (p/regenerate model t sel)
                      w-fwd (ev fwd-weight)
                      ;; Reverse move
                      {rev-trace :trace rev-weight :weight}
                      (p/regenerate model fwd-trace sel)
                      w-rev (ev rev-weight)]
                  (and (js/Number.isFinite w-fwd)
                       (js/Number.isFinite w-rev)
                       ;; Reverse trace preserves address set
                       (= (set (cm/addresses (:choices t)))
                          (set (cm/addresses (:choices rev-trace))))))))

(t/deftest regenerate-weight-chain-model-exact
  (t/testing "Regenerate weight on gaussian chain matches child log-prob change"
    ;; x ~ N(0,1), y ~ N(x, 0.5). Regenerate S={:x}.
    ;; weight should equal lp(y; x', 0.5) - lp(y; x, 0.5)
    ;; because y's distribution changed but y's value didn't.
    (let [n-trials 20
          results
          (for [_ (range n-trials)]
            (let [t (p/simulate (:model gaussian-chain) [])
                  old-x (-> (:choices t) (cm/get-submap :x) cm/get-value)
                  old-y (-> (:choices t) (cm/get-submap :y) cm/get-value)
                  {:keys [trace weight]} (p/regenerate (:model gaussian-chain)
                                                       t (sel/select :x))
                  new-x (-> (:choices trace) (cm/get-submap :x) cm/get-value)
                  new-y (-> (:choices trace) (cm/get-submap :y) cm/get-value)
                  w (ev weight)
                  ;; y must be unchanged (unselected)
                  _ (assert (close? (ev old-y) (ev new-y) 1e-8)
                            "y should be preserved")
                  ;; Independently compute the expected weight:
                  ;; change in child's log-prob due to parent change
                  old-child-lp (ev (dist/log-prob
                                    (dist/gaussian old-x 0.5) old-y))
                  new-child-lp (ev (dist/log-prob
                                    (dist/gaussian new-x 0.5) old-y))
                  expected (- new-child-lp old-child-lp)]
              (close? w expected 0.01)))]
      (t/is (every? true? results)
            (str "Regenerate weight mismatch in "
                 (count (remove true? results)) "/" n-trials " trials")))))

;; ---------------------------------------------------------------------------
;; INTERNAL PROPOSAL laws [T] Ch. 4 — Cusumano-Towner 2020
;; ---------------------------------------------------------------------------

;; --- Law #27: Internal Proposal Support [T] Def 4.1.1 ---
;; q(v; x, sigma) > 0 iff p(v|sigma; x) > 0
;;
;; For forward-sampling: q samples unconstrained addresses from their conditional
;; priors. Same support as the model by construction. Distinguished from
;; proposal-support-coverage (#12) by testing at SUPPORT BOUNDARIES specifically.

(t/deftest law:internal-proposal-support
  ;; [T] Def 4.1.1 — support equivalence at boundaries
  ;; Deterministic test: fixed models with known boundary values.
  (t/testing "in-support boundary values produce finite weights"
    (let [;; Uniform(0,1) at x=0 and x=1 (boundary of support)
          uni-model (dyn/auto-key
                      (gen [] (let [x (trace :x (dist/uniform 0 1))]
                                (trace :y (dist/gaussian x 1)))))
          w0 (ev (:weight (p/generate uni-model [] (cm/choicemap :x (mx/scalar 0.0)))))
          w1 (ev (:weight (p/generate uni-model [] (cm/choicemap :x (mx/scalar 1.0)))))]
      (t/is (h/finite? w0) "Uniform boundary x=0 should be finite")
      (t/is (close? w0 0.0 1e-6) "Uniform(0;0,1) log-prob = 0")
      (t/is (h/finite? w1) "Uniform boundary x=1 should be finite")
      (t/is (close? w1 0.0 1e-6) "Uniform(1;0,1) log-prob = 0"))
    (let [;; Beta(2,5) at x=0.01 and x=0.99 (near boundary)
          beta-model (dyn/auto-key
                       (gen [] (let [p (trace :p (dist/beta-dist 2 5))]
                                 (trace :x (dist/bernoulli p)))))
          w-lo (ev (:weight (p/generate beta-model [] (cm/choicemap :p (mx/scalar 0.01)))))
          w-hi (ev (:weight (p/generate beta-model [] (cm/choicemap :p (mx/scalar 0.99)))))]
      (t/is (h/finite? w-lo) "Beta near-boundary p=0.01 should be finite")
      (t/is (h/finite? w-hi) "Beta near-boundary p=0.99 should be finite"))
    (let [;; Exponential(1) at x=0 (boundary of support [0,inf))
          exp-model (dyn/auto-key
                      (gen [] (let [x (trace :x (dist/exponential 1))]
                                (trace :y (dist/gaussian x 1)))))
          w0 (ev (:weight (p/generate exp-model [] (cm/choicemap :x (mx/scalar 0.0)))))]
      (t/is (h/finite? w0) "Exponential boundary x=0 should be finite")
      (t/is (close? w0 0.0 1e-6) "Exp(0;1) log-prob = 0")))
  (t/testing "out-of-support values produce non-finite log-prob (reverse direction)"
    ;; Reverse direction of support equivalence: q(v;x,sigma) = 0 when
    ;; p(v|sigma;x) = 0. Verified at the distribution primitive level,
    ;; which defines the proposal's density function.
    (let [cases [[(dist/exponential 1) (mx/scalar -1.0) "Exp(1) at x=-1"]
                 [(dist/uniform 0 1) (mx/scalar 2.0) "Uniform(0,1) at x=2"]
                 [(dist/uniform 0 1) (mx/scalar -0.5) "Uniform(0,1) at x=-0.5"]]]
      (doseq [[d v label] cases]
        (let [lp (ev (dist/log-prob d v))]
          (t/is (not (h/finite? lp))
                (str label " should have non-finite log-prob, got " lp)))))))

;; --- Law #28: Extended Generate Weight [T] §4.1.1 ---
;; log w = log p(tau; x) / q(v; x, sigma)
;;       = sum_{a in sigma} log p(sigma[a] | parents(a))
;;
;; For forward-sampling, the weight equals the sum of log-probs at constrained
;; addresses only (the marginal likelihood of the constraints).
;; Distinguished from is-weight-formula (#11): tests with MULTIPLE constrained
;; addresses simultaneously.

(t/deftest law:generate-weight-is-marginal-likelihood
  (t/testing "weight = sum of log-probs at constrained addresses, multi-constraint"
    ;; Non-conjugate model: a ~ Uniform(0,1), b ~ Laplace(a, 1), c ~ Exp(1)
    ;; No conjugate pairs, so uses compiled/handler path (not analytical).
    (let [model (dyn/auto-key
                  (gen [] (let [a (trace :a (dist/uniform 0 1))
                                b (trace :b (dist/laplace a 1))]
                            (trace :c (dist/exponential 1)))))
          ;; Test 1: single constraint
          ;; Constrain a=0.5 only: weight = log Uniform(0.5; 0,1) = 0
          c1 (cm/choicemap :a (mx/scalar 0.5))
          w1 (ev (:weight (p/generate model [] c1)))
          ;; Test 2: two constraints
          ;; Constrain a=0.5 and c=0.5: weight = 0 + log Exp(0.5; 1)
          ;; Analytical: log Exp(0.5; 1) = log(1) - 1*0.5 = -0.5
          c2 (cm/choicemap :a (mx/scalar 0.5) :c (mx/scalar 0.5))
          {:keys [trace weight]} (p/generate model [] c2)
          w2 (ev weight)
          ;; Test 3: weight = project(trace, constrained selection)
          proj (ev (p/project model trace (sel/from-set #{:a :c})))]
      (t/is (close? w1 0.0 0.01) "single constraint: Uniform(0.5) = 0")
      (t/is (close? w2 -0.5 0.01) "multi constraint: 0 + (-0.5) = -0.5")
      (t/is (close? w2 proj 0.01) "weight = project for constrained selection"))))

;; --- Law #29: Forward Sampling Factorization [T] §4.1.3 ---
;; q(v; x, sigma) = product_i p({a_i->v[a_i]} | context)^[a_i not in sigma]
;;
;; For each unconstrained address, the proposal samples from the conditional
;; prior. Verified via: weight = score - sum(log-prob at unconstrained addrs).

(t/deftest law:forward-sampling-factorization
  (t/testing "weight = score - log q(unconstrained), Laplace chain model"
    ;; a ~ Uniform(0,1), b ~ Laplace(a, 1), c ~ Laplace(a+b, 0.5)
    ;; No conjugate pairs.
    (let [model (dyn/auto-key
                  (gen [] (let [a (trace :a (dist/uniform 0 1))
                                b (trace :b (dist/laplace a 1))]
                            (trace :c (dist/laplace (mx/add a b) 0.5)))))
          ;; Case 1: constrain c=3.0 only. a, b unconstrained.
          ;; weight = log Laplace(3; a+b, 0.5).
          ;; Factorization: weight = score - [lp(a) + lp(b|a)].
          ;;
          ;; Fixed analytical example (verified by hand):
          ;; a=0.5, b=1.0 => score = 0 + (-0.693-0.5) + (-3.0) = -4.193
          ;; log q = 0 + (-0.693-0.5) = -1.193
          ;; weight = -4.193 - (-1.193) = -3.0
          ;; Direct: log Laplace(3; 1.5, 0.5) = -log(1) - |3-1.5|/0.5 = -3.0
          results-c
          (for [seed (range 20)]
            (let [m (dyn/with-key model (rng/fresh-key seed))
                  {:keys [trace weight]} (p/generate m []
                                           (cm/choicemap :c (mx/scalar 3.0)))
                  w (ev weight)
                  s (ev (:score trace))
                  a (ev (cm/get-value (cm/get-submap (:choices trace) :a)))
                  b (ev (cm/get-value (cm/get-submap (:choices trace) :b)))
                  lp-a (ev (dist/log-prob (dist/uniform 0 1) (mx/scalar a)))
                  lp-b (ev (dist/log-prob (dist/laplace (mx/scalar a) 1)
                                           (mx/scalar b)))
                  log-q (+ lp-a lp-b)]
              (close? w (- s log-q) 0.01)))
          ;; Case 2: constrain a=0.7 and c=3.0. b unconstrained.
          ;; Factorization: weight = score - lp(b|a=0.7).
          results-ac
          (for [seed (range 20)]
            (let [m (dyn/with-key model (rng/fresh-key seed))
                  constraints (cm/choicemap :a (mx/scalar 0.7) :c (mx/scalar 3.0))
                  {:keys [trace weight]} (p/generate m [] constraints)
                  w (ev weight)
                  s (ev (:score trace))
                  b (ev (cm/get-value (cm/get-submap (:choices trace) :b)))
                  log-q (ev (dist/log-prob (dist/laplace (mx/scalar 0.7) 1)
                                            (mx/scalar b)))]
              (close? w (- s log-q) 0.01)))]
      (t/is (every? true? results-c)
            (str "Factorization failed (constrain c only) in "
                 (count (remove true? results-c)) "/20 trials"))
      (t/is (every? true? results-ac)
            (str "Factorization failed (constrain a+c) in "
                 (count (remove true? results-ac)) "/20 trials")))))

(t/run-tests)
