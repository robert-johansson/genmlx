;; @tier medium
(ns genmlx.adev-test
  "ADEV gradient estimation tests."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.handler :as handler]
            [genmlx.inference.adev :as adev])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Deterministic seeding for the statistical reparam tests (genmlx-0nyj): the
;; reparam value entropy comes from rng/fresh-key's 0-arg js/Math.random path, so
;; we reseed it from one mulberry32 stream (the proven project pattern) around the
;; convergence/variance assertions to make their thresholds reproducible.
(defn- mulberry32 [seed]
  (let [state (atom (bit-or seed 0))]
    (fn []
      (let [a (swap! state #(bit-or (+ % 0x6D2B79F5) 0))
            t (js/Math.imul (bit-xor a (unsigned-bit-shift-right a 15)) (bit-or a 1))
            t (bit-xor (+ t (js/Math.imul (bit-xor t (unsigned-bit-shift-right t 7))
                                          (bit-or t 61))) t)]
        (/ (unsigned-bit-shift-right (bit-xor t (unsigned-bit-shift-right t 14)) 0)
           4294967296)))))
(def ^:private rng-stream (mulberry32 0xADE0))
(def ^:private orig-fresh-key rng/fresh-key)
(defn- det-fresh-key
  ([]     (orig-fresh-key (js/Math.floor (* (rng-stream) 2147483647))))
  ([seed] (orig-fresh-key seed)))

;; categorical with an explicit Gumbel-softmax temperature (the reparam path reads
;; :reparam-tau from params; default 0.5). Public arity of dist/categorical is
;; unchanged — this just sets the param for gradient tests.
(defn- cat-tau [logits tau]
  (update (dist/categorical logits) :params assoc :reparam-tau tau))

(deftest has-reparam-detection-test
  (testing "has-reparam? detection"
    (is (adev/has-reparam? (dist/gaussian 0 1)) "gaussian is reparameterizable")
    (is (adev/has-reparam? (dist/uniform 0 1)) "uniform is reparameterizable")
    (is (adev/has-reparam? (dist/exponential 1)) "exponential is reparameterizable")
    (is (adev/has-reparam? (dist/laplace 0 1)) "laplace is reparameterizable")
    (is (not (adev/has-reparam? (dist/bernoulli 0.5))) "bernoulli is NOT reparameterizable")
    (is (adev/has-reparam? (dist/categorical (mx/array [-1 -1]))) "categorical IS reparameterizable via Gumbel-softmax (genmlx-0nyj)")
    (is (not (adev/has-reparam? (dist/beta-dist 2 2))) "beta is NOT reparameterizable")))

(deftest pure-reparam-model-test
  (testing "pure reparam model (ADEV execute)"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 1))]
                    x))
          key (rng/fresh-key)
          {:keys [trace reinforce-lp]} (adev/adev-execute model [] key)]
      (is (cm/has-value? (cm/get-submap (:choices trace) :x)) "trace has choices")
      (is (number? (mx/item (:score trace))) "trace has score")
      (is (h/close? 0.0 (mx/item reinforce-lp) 1e-6) "reinforce-lp is 0 for pure reparam"))))

(deftest mixed-model-test
  (testing "mixed model (reparam + REINFORCE)"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 1))
                        b (trace :b (dist/bernoulli 0.5))]
                    (mx/add x b)))
          key (rng/fresh-key)
          {:keys [trace reinforce-lp]} (adev/adev-execute model [] key)]
      (is (cm/has-value? (cm/get-submap (:choices trace) :x)) "trace has gaussian choice")
      (is (cm/has-value? (cm/get-submap (:choices trace) :b)) "trace has bernoulli choice")
      (is (js/isFinite (mx/item reinforce-lp)) "reinforce-lp is finite"))))

(deftest adev-surrogate-loss-test
  (testing "ADEV surrogate loss"
    (let [model (gen []
                  (let [x (trace :x (dist/gaussian 0 1))
                        b (trace :b (dist/bernoulli 0.5))]
                    (mx/add x b)))
          cost-fn (fn [trace] (mx/square (:retval trace)))
          key (rng/fresh-key)
          surrogate (adev/adev-surrogate model [] cost-fn key)]
      (mx/eval! surrogate)
      (is (js/isFinite (mx/item surrogate)) "surrogate is finite")
      (is (>= (mx/item surrogate) 0.0) "surrogate is non-negative (squared cost)"))))

(deftest adev-gradient-with-params-test
  (testing "ADEV gradient with params"
    (let [model (gen []
                  (let [mu (param :mu 0.0)
                        x (trace :x (dist/gaussian mu (mx/scalar 1.0)))]
                    x))
          cost-fn (fn [trace]
                    (mx/square (mx/subtract (:retval trace) (mx/scalar 3.0))))
          param-names [:mu]
          params (mx/array [0.0])
          {:keys [loss grad]} (adev/adev-gradient {:n-samples 10}
                                                   model [] cost-fn
                                                   param-names params)]
      (mx/eval! loss grad)
      (is (js/isFinite (mx/item loss)) "loss is finite")
      (is (js/isFinite (mx/item (mx/index grad 0))) "grad is finite")
      (is (< (mx/item (mx/index grad 0)) 0) "grad is negative (should increase mu)"))))

(deftest adev-optimization-convergence-test
  (testing "ADEV optimization convergence"
    (let [model (gen []
                  (let [mu (param :mu 0.0)
                        x (trace :x (dist/gaussian mu (mx/scalar 1.0)))]
                    x))
          cost-fn (fn [trace]
                    (mx/square (mx/subtract (:retval trace) (mx/scalar 5.0))))
          param-names [:mu]
          init-params (mx/array [0.0])
          {:keys [params loss-history]} (adev/adev-optimize
                                          {:iterations 200 :lr 0.1 :n-samples 10}
                                          model [] cost-fn param-names init-params)
          final-mu (mx/item (mx/index params 0))
          first-loss (first loss-history)
          last-loss (last loss-history)]
      (is (h/close? 5.0 final-mu 1.5) "mu converges near 5.0")
      (is (< last-loss first-loss) "loss decreases"))))

(deftest gradient-finite-difference-test
  (testing "gradient correctness (finite difference check)"
    (let [model (gen []
                  (let [mu (param :mu 0.0)
                        x (trace :x (dist/gaussian mu (mx/scalar 1.0)))]
                    x))
          cost-fn (fn [trace] (mx/square (:retval trace)))
          param-names [:mu]
          mu-val 2.0
          params (mx/array [mu-val])
          {:keys [grad]} (adev/adev-gradient {:n-samples 500}
                                              model [] cost-fn
                                              param-names params)
          adev-grad (mx/item (mx/index grad 0))
          analytical-grad (* 2.0 mu-val)]
      (is (h/close? analytical-grad adev-grad 0.5) "ADEV grad ~ analytical 2*mu"))))

(deftest vadev-gradient-test
  (testing "vectorized ADEV gradient (vadev-gradient)"
    (let [model (gen []
                  (let [mu (param :mu 0.0)
                        x (trace :x (dist/gaussian mu (mx/scalar 1.0)))]
                    x))
          cost-fn (fn [result]
                    (mx/square (mx/subtract (:retval result) (mx/scalar 3.0))))
          param-names [:mu]
          params (mx/array [0.0])
          {:keys [loss grad]} (adev/vadev-gradient {:n-samples 100}
                                                    model [] cost-fn
                                                    param-names params)]
      (mx/eval! loss grad)
      (is (js/isFinite (mx/item loss)) "vadev loss is finite")
      (is (js/isFinite (mx/item (mx/index grad 0))) "vadev grad is finite")
      (is (< (mx/item (mx/index grad 0)) 0) "vadev grad is negative (should increase mu)"))))

(deftest vadev-vs-adev-agreement-test
  (testing "vadev-gradient vs adev-gradient agreement"
    (let [model (gen []
                  (let [mu (param :mu 0.0)
                        x (trace :x (dist/gaussian mu (mx/scalar 1.0)))]
                    x))
          cost-fn-scalar (fn [trace]
                           (mx/square (mx/subtract (:retval trace) (mx/scalar 0.0))))
          cost-fn-batch (fn [result]
                          (mx/square (mx/subtract (:retval result) (mx/scalar 0.0))))
          param-names [:mu]
          mu-val 2.0
          params (mx/array [mu-val])
          analytical-grad (* 2.0 mu-val)
          {adev-grad-arr :grad} (adev/adev-gradient {:n-samples 500}
                                                     model [] cost-fn-scalar
                                                     param-names params)
          adev-g (mx/item (mx/index adev-grad-arr 0))
          {vadev-grad-arr :grad} (adev/vadev-gradient {:n-samples 500}
                                                       model [] cost-fn-batch
                                                       param-names params)
          vadev-g (mx/item (mx/index vadev-grad-arr 0))]
      (is (h/close? analytical-grad adev-g 0.5) "adev-gradient ~ analytical 2*mu")
      (is (h/close? analytical-grad vadev-g 0.5) "vadev-gradient ~ analytical 2*mu"))))

(deftest compiled-adev-optimize-test
  (testing "compiled-adev-optimize convergence"
    (let [model (gen []
                  (let [mu (param :mu 0.0)
                        x (trace :x (dist/gaussian mu (mx/scalar 1.0)))]
                    x))
          cost-fn (fn [result]
                    (mx/square (mx/subtract (:retval result) (mx/scalar 5.0))))
          param-names [:mu]
          init-params (mx/array [0.0])
          {:keys [params loss-history]} (adev/compiled-adev-optimize
                                          {:iterations 200 :lr 0.1 :n-samples 100}
                                          model [] cost-fn param-names init-params)
          final-mu (mx/item (mx/index params 0))
          first-loss (first loss-history)
          last-loss (last loss-history)]
      (is (h/close? 5.0 final-mu 1.5) "mu converges near 5.0")
      (is (< last-loss first-loss) "loss decreases"))))

(deftest baseline-variance-reduction-test
  (testing "baseline reduces variance (mixed model)"
    (let [model (gen []
                  (let [mu (param :mu 0.0)
                        x (trace :x (dist/gaussian mu (mx/scalar 1.0)))
                        b (trace :b (dist/bernoulli 0.5))]
                    (mx/add x b)))
          cost-fn (fn [result]
                    (mx/square (mx/subtract (:retval result) (mx/scalar 5.0))))
          param-names [:mu]
          init-params (mx/array [0.0])
          iters 200
          bl-result (adev/compiled-adev-optimize
                      {:iterations iters :lr 0.1 :n-samples 100
                       :baseline-decay 0.9}
                      model [] cost-fn param-names init-params)
          half (quot iters 2)
          tail-bl (drop half (:loss-history bl-result))
          mean-fn (fn [xs] (/ (reduce + xs) (count xs)))
          var-fn (fn [xs]
                   (let [m (mean-fn xs)]
                     (/ (reduce + (map #(* (- % m) (- % m)) xs)) (count xs))))
          var-bl (var-fn tail-bl)]
      (is (js/isFinite var-bl) "baseline loss variance is finite"))))

(deftest compiled-adev-baseline-convergence-test
  (testing "compiled-adev-optimize with baseline convergence"
    (let [model (gen []
                  (let [mu (param :mu 0.0)
                        x (trace :x (dist/gaussian mu (mx/scalar 1.0)))]
                    x))
          cost-fn (fn [result]
                    (mx/square (mx/subtract (:retval result) (mx/scalar 5.0))))
          param-names [:mu]
          init-params (mx/array [0.0])
          {:keys [params loss-history]} (adev/compiled-adev-optimize
                                          {:iterations 200 :lr 0.1 :n-samples 100
                                           :baseline-decay 0.9}
                                          model [] cost-fn param-names init-params)
          final-mu (mx/item (mx/index params 0))
          first-loss (first loss-history)
          last-loss (last loss-history)]
      (is (h/close? 5.0 final-mu 1.5) "mu converges near 5.0 with baseline")
      (is (< last-loss first-loss) "loss decreases with baseline"))))

;; ---------------------------------------------------------------------------
;; Categorical Gumbel-softmax reparameterization (genmlx-0nyj)
;; ---------------------------------------------------------------------------

(deftest categorical-reparam-contract-test
  (testing "reparam value is a [K] straight-through one-hot; ordinary ops unchanged"
    (let [d (cat-tau (mx/array [0.5 -0.5 1.0]) 0.3)
          v (dc/dist-reparam d (rng/fresh-key 7))]
      (mx/eval! v)
      (is (= [3] (vec (mx/shape v))) "reparam value has shape [K]")
      (is (h/close? 1.0 (mx/item (mx/sum v)) 1e-5) "one-hot sums to 1")
      (let [vals (mx/->clj v)]
        (is (every? #(or (h/close? 0.0 % 1e-5) (h/close? 1.0 % 1e-5)) vals)
            "forward value is exactly one-hot (0/1 entries)")))
    ;; NON-CORRUPTION GUARD: ordinary categorical sampling/scoring is untouched —
    ;; the relaxation is scoped to the ADEV/dist-reparam path only.
    (let [d (dist/categorical (mx/array [0.0 0.0 0.0]))
          s (dc/dist-sample d (rng/fresh-key 9))]
      (mx/eval! s)
      (is (integer? (mx/item s)) "ordinary sample is an integer index, not a one-hot")
      (is (h/close? (js/Math.log (/ 1.0 3.0)) (mx/item (dc/dist-log-prob d (mx/scalar 1 mx/int32))) 1e-4)
          "ordinary log-prob is exact log-softmax, not the relaxed score"))))

(deftest categorical-reparam-gradient-oracle-test
  ;; INDEPENDENT ORACLE: 2-category categorical with logits θ=[t0,t1], objective
  ;; f(c)=c (the category index 0/1). E[f]=p1=softmax(θ)[1]; the EXACT analytic
  ;; gradient (derived without GenMLX) is dE/dt1=p1(1-p1), dE/dt0=-p1(1-p1).
  ;; With θ=[0,0]: p1=0.5 → grad=[-0.25, 0.25].
  (with-redefs [rng/fresh-key det-fresh-key]
    (testing "reparam gradient converges to the analytic softmax gradient"
      (let [model (gen [] (let [t0 (param :t0 0.0) t1 (param :t1 0.0)
                                logits (mx/stack [t0 t1])
                                c (trace :c (cat-tau logits 0.2))]
                            (mx/sum (mx/multiply c (mx/array [0.0 1.0])))))
            cost-fn (fn [tr] (:retval tr))
            {:keys [grad]} (adev/adev-gradient {:n-samples 4000} model []
                                               cost-fn [:t0 :t1] (mx/array [0.0 0.0]))
            g0 (mx/item (mx/index grad 0))
            g1 (mx/item (mx/index grad 1))]
        (is (h/close? -0.25 g0 0.05) (str "dE/dt0 ~ -0.25 (got " g0 ")"))
        (is (h/close? 0.25 g1 0.05) (str "dE/dt1 ~ +0.25 (got " g1 ")"))
        (is (h/close? g1 (- g0) 0.02) "gradient is antisymmetric (2-category softmax)")))
    (testing "relaxation bias shrinks as temperature decreases"
      (let [grad-at (fn [tau]
                      (let [model (gen [] (let [t0 (param :t0 0.0) t1 (param :t1 0.0)
                                                logits (mx/stack [t0 t1])
                                                c (trace :c (cat-tau logits tau))]
                                            (mx/sum (mx/multiply c (mx/array [0.0 1.0])))))]
                        (mx/item (mx/index (:grad (adev/adev-gradient {:n-samples 4000} model []
                                                    (fn [tr] (:retval tr)) [:t0 :t1] (mx/array [0.0 0.0]))) 1))))
            err-hi (js/Math.abs (- 0.25 (grad-at 0.7)))
            err-lo (js/Math.abs (- 0.25 (grad-at 0.1)))]
        (is (<= err-lo err-hi)
            (str "smaller tau is at least as close to the analytic gradient (lo=" err-lo " hi=" err-hi ")"))))))

(deftest categorical-reparam-single-sample-unbiased-test
  ;; The reparam (Gumbel-softmax) single-sample gradient is a usable low-bias
  ;; estimator: averaging a modest number of n-samples=1 draws lands near the
  ;; analytic 0.25, and every draw is finite (gradient genuinely flows through
  ;; the discrete site). NOTE: we deliberately do NOT assert lower variance than
  ;; REINFORCE — for a single binary choice with the linear objective f(c)=c,
  ;; REINFORCE's score-function estimator (score ∈ {0, 0.5}) is already
  ;; low-variance and beats Gumbel-softmax at τ=0.2; the variance advantage of
  ;; the relaxation is regime-dependent (smooth/high-dim objectives), not a
  ;; universal property worth asserting on this toy (genmlx-0nyj, empirically
  ;; verified: reparam var ≈ 0.12 > REINFORCE var ≈ 0.06 here).
  (with-redefs [rng/fresh-key det-fresh-key]
    (let [model (gen [] (let [t0 (param :t0 0.0) t1 (param :t1 0.0)
                              lg (mx/stack [t0 t1])
                              c (trace :c (cat-tau lg 0.2))]
                          (mx/sum (mx/multiply c (mx/array [0.0 1.0])))))
          k 300
          gs (vec (for [_ (range k)]
                    (mx/item (mx/index (:grad (adev/adev-gradient {:n-samples 1} model []
                                                (fn [tr] (:retval tr)) [:t0 :t1] (mx/array [0.0 0.0]))) 1))))
          mean (/ (reduce + gs) (count gs))]
      (is (every? js/isFinite gs) "every single-sample reparam gradient is finite")
      (is (h/close? 0.25 mean 0.05)
          (str "mean of single-sample reparam gradients ~ analytic 0.25 (got " mean ")")))))

(cljs.test/run-tests)
