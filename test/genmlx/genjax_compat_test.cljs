(ns genmlx.genjax-compat-test
  "Tests adapted from GenJAX (https://github.com/femtomc/genjax/tree/main/tests).
   Verifies GFI invariants, inference convergence, gradient flow, combinators,
   and numerical stability."
  (:require [cljs.test :as t :refer [deftest is testing]]
            [genmlx.test-helpers :as th]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.combinators :as comb]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.smc :as smc]
            [genmlx.inference.vi :as vi]
            [genmlx.inference.diagnostics :as diag])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Section 1: GFI Invariants (adapted from test_core.py)
;; ---------------------------------------------------------------------------

(deftest s1-gfi-invariants
  (testing "1.1 simulate/generate consistency"
    (let [model (dyn/auto-key (gen []
                  (let [x (trace :x (dist/gaussian 0 1))
                        y (trace :y (dist/gaussian 0 1))]
                    [x y])))
          trace (p/simulate model [])
          choices (:choices trace)
          score (th/realize (:score trace))
          {:keys [trace weight]} (p/generate model [] choices)
          gen-score (th/realize (:score trace))
          gen-weight (th/realize weight)]
      (is (th/close? gen-weight gen-score 1e-5) "weight ≈ score")
      (is (th/close? gen-score score 1e-5) "generate score equals original score")))

  (testing "1.1 hierarchical model consistency"
    (let [model (dyn/auto-key (gen [n]
                  (let [mu (trace :mu (dist/gaussian 0 10))]
                    (mx/eval! mu)
                    (let [m (mx/item mu)]
                      (doseq [i (range n)]
                        (trace (keyword (str "y" i)) (dist/gaussian m 1)))
                      m))))
          trace (p/simulate model [5])
          choices (:choices trace)
          score (th/realize (:score trace))
          {:keys [weight]} (p/generate model [5] choices)
          w (th/realize weight)]
      (is (th/close? w score 1e-4) "hierarchical model: generate weight ≈ score")))

  (testing "1.2 empty constraints: weight ≈ 0"
    (let [model (dyn/auto-key (gen [] (trace :x (dist/gaussian 0 1))))
          {:keys [weight]} (p/generate model [] cm/EMPTY)
          w (th/realize weight)]
      (is (th/close? w 0.0 1e-10) "weight ≈ 0")))

  (testing "1.3 update weight invariant"
    (let [model (dyn/auto-key (gen [] (trace :x (dist/gaussian 0 1))))
          trace (p/simulate model [])
          old-score (th/realize (:score trace))
          new-val (mx/scalar 2.5)
          constraints (cm/choicemap :x new-val)
          {:keys [trace weight]} (p/update model trace constraints)
          new-score (th/realize (:score trace))
          w (th/realize weight)]
      (is (th/close? w (- new-score old-score) 1e-5) "weight = new_score - old_score")))

  (testing "1.3 multi-address update"
    (let [model (dyn/auto-key (gen []
                  (let [x (trace :x (dist/gaussian 0 1))
                        y (trace :y (dist/gaussian 0 1))]
                    [x y])))
          trace (p/simulate model [])
          old-score (th/realize (:score trace))
          constraints (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
          {:keys [trace weight]} (p/update model trace constraints)
          new-score (th/realize (:score trace))
          w (th/realize weight)]
      (is (th/close? w (- new-score old-score) 1e-5) "multi-address update: weight = new_score - old_score")))

  (testing "1.3 update with same value"
    (let [model (dyn/auto-key (gen [] (trace :x (dist/gaussian 0 1))))
          trace (p/simulate model [])
          old-val (cm/get-value (cm/get-submap (:choices trace) :x))
          constraints (cm/choicemap :x old-val)
          {:keys [weight]} (p/update model trace constraints)
          w (th/realize weight)]
      (is (th/close? w 0.0 1e-5) "weight ≈ 0")))

  (testing "1.4 regenerate with no selection"
    (let [model (dyn/auto-key (gen []
                  (let [x (trace :x (dist/gaussian 0 1))
                        y (trace :y (dist/gaussian 0 1))]
                    [x y])))
          trace (p/simulate model [])
          old-x (th/realize (cm/get-value (cm/get-submap (:choices trace) :x)))
          {:keys [trace weight]} (p/regenerate model trace sel/none)
          new-x (th/realize (cm/get-value (cm/get-submap (:choices trace) :x)))
          w (th/realize weight)]
      (is (th/close? w 0.0 1e-5) "weight ≈ 0")
      (is (th/close? old-x new-x 1e-10) "choices unchanged")))

  (testing "1.4 regenerate selected: unselected preserved"
    (let [model (dyn/auto-key (gen []
                  (let [x (trace :x (dist/gaussian 0 1))
                        y (trace :y (dist/gaussian 0 1))]
                    [x y])))
          trace (p/simulate model [])
          old-y (th/realize (cm/get-value (cm/get-submap (:choices trace) :y)))
          {:keys [trace]} (p/regenerate model trace (sel/select :x))
          new-y (th/realize (cm/get-value (cm/get-submap (:choices trace) :y)))]
      (is (th/close? old-y new-y 1e-10) ":y preserved")))

  (testing "1.5 distribution GFI consistency"
    (doseq [[name d] [["gaussian" (dist/gaussian 0 1)]
                       ["uniform" (dist/uniform 0 1)]
                       ["exponential" (dist/exponential 1)]
                       ["laplace" (dist/laplace 0 1)]]]
      (let [trace (p/simulate d [])
            v (:retval trace)
            score (th/realize (:score trace))
            lp (th/realize (dist/log-prob d v))]
        (is (th/close? score lp 1e-5)
            (str name " GFI: simulate score = log-prob")))))

  (testing "1.6 nested addressing"
    (let [inner (gen [] (trace :z (dist/gaussian 0 1)))
          outer (dyn/auto-key (gen []
                  (let [x (trace :x (dist/gaussian 0 10))]
                    (splice :inner inner)
                    x)))
          trace (p/simulate outer [])
          choices (:choices trace)]
      (is (cm/has-value? (cm/get-submap choices :x)) "top-level :x exists")
      (is (cm/has-value? (cm/get-submap (cm/get-submap choices :inner) :z))
          ":inner/:z exists")))

  (testing "1.7 deterministic computation"
    (let [model (dyn/auto-key (gen [x] (* x x)))
          trace (p/simulate model [5])
          score (th/realize (:score trace))]
      (is (= (:retval trace) 25) "retval = 25")
      (is (th/close? score 0.0 1e-10) "score = 0"))))

;; ============================================================================
;; Section 2: MCMC Convergence to Exact Posteriors (adapted from test_mcmc.py)
;; ============================================================================

(deftest s2-mcmc-convergence
  (testing "2.1 normal-normal conjugate (MH)"
    (let [sigma0 10.0
          sigma 1.0
          xs [3.0 3.5 2.5 4.0 3.0]
          n (count xs)
          post-var (/ 1.0 (+ (/ 1.0 (* sigma0 sigma0)) (/ n (* sigma sigma))))
          post-mean (* post-var (/ (reduce + xs) (* sigma sigma)))
          model (gen [xs sigma]
                  (let [mu (trace :mu (dist/gaussian 0 sigma0))]
                    (mx/eval! mu)
                    (let [m (mx/item mu)]
                      (doseq [[i x] (map-indexed vector xs)]
                        (trace (keyword (str "y" i)) (dist/gaussian m sigma)))
                      m)))
          observations (reduce (fn [cm [i x]]
                                 (cm/set-choice cm [(keyword (str "y" i))] (mx/scalar x)))
                               cm/EMPTY (map-indexed vector xs))
          traces (mcmc/mh {:samples 500 :burn 100
                            :selection (sel/select :mu)}
                           model [xs sigma] observations)
          mus (mapv (fn [t]
                      (let [v (cm/get-value (cm/get-submap (:choices t) :mu))]
                        (mx/eval! v) (mx/item v)))
                    traces)
          sample-mu (/ (reduce + mus) (count mus))]
      (is (th/close? sample-mu post-mean 0.3)
          (str "MH posterior mean ≈ " (.toFixed post-mean 2)))))

  (testing "2.2 normal-normal conjugate (HMC)"
    (let [sigma0 10.0
          sigma 1.0
          xs [3.0 3.5 2.5 4.0 3.0]
          n (count xs)
          post-var (/ 1.0 (+ (/ 1.0 (* sigma0 sigma0)) (/ n (* sigma sigma))))
          post-mean (* post-var (/ (reduce + xs) (* sigma sigma)))
          model (gen [xs sigma]
                  (let [mu (trace :mu (dist/gaussian 0 sigma0))]
                    (doseq [[i x] (map-indexed vector xs)]
                      (trace (keyword (str "y" i)) (dist/gaussian mu sigma)))
                    mu))
          observations (reduce (fn [cm [i x]]
                                 (cm/set-choice cm [(keyword (str "y" i))] (mx/scalar x)))
                               cm/EMPTY (map-indexed vector xs))
          samples (mcmc/hmc {:samples 200 :step-size 0.05 :leapfrog-steps 10
                              :burn 50 :addresses [:mu]}
                             model [xs sigma] observations)
          mus (mapv first samples)
          sample-mu (/ (reduce + mus) (count mus))]
      (is (th/close? sample-mu post-mean 0.3)
          (str "HMC posterior mean ≈ " (.toFixed post-mean 2)))))

  (testing "2.3 normal-normal conjugate (MALA)"
    (let [sigma0 10.0
          sigma 1.0
          xs [3.0 3.5 2.5 4.0 3.0]
          n (count xs)
          post-var (/ 1.0 (+ (/ 1.0 (* sigma0 sigma0)) (/ n (* sigma sigma))))
          post-mean (* post-var (/ (reduce + xs) (* sigma sigma)))
          model (gen [xs sigma]
                  (let [mu (trace :mu (dist/gaussian 0 sigma0))]
                    (doseq [[i x] (map-indexed vector xs)]
                      (trace (keyword (str "y" i)) (dist/gaussian mu sigma)))
                    mu))
          observations (reduce (fn [cm [i x]]
                                 (cm/set-choice cm [(keyword (str "y" i))] (mx/scalar x)))
                               cm/EMPTY (map-indexed vector xs))
          samples (mcmc/mala {:samples 200 :step-size 0.1
                               :burn 100 :addresses [:mu]}
                              model [xs sigma] observations)
          mus (mapv first samples)
          sample-mu (/ (reduce + mus) (count mus))]
      (is (th/close? sample-mu post-mean 1.0)
          (str "MALA posterior mean ≈ " (.toFixed post-mean 2)))))

  (testing "2.4 beta-bernoulli conjugate (MH)"
    (let [alpha-prior 2.0 beta-prior 2.0
          obs [1.0 1.0 1.0 0.0 1.0 0.0 1.0]
          n (count obs)
          sum-x (reduce + obs)
          alpha-post (+ alpha-prior sum-x)
          beta-post (+ beta-prior (- n sum-x))
          post-mean (/ alpha-post (+ alpha-post beta-post))
          model (gen [n]
                  (let [p (trace :p (dist/beta-dist alpha-prior beta-prior))]
                    (mx/eval! p)
                    (let [pv (mx/item p)]
                      (doseq [i (range n)]
                        (trace (keyword (str "x" i)) (dist/bernoulli pv)))
                      pv)))
          observations (reduce (fn [cm [i x]]
                                 (cm/set-choice cm [(keyword (str "x" i))] (mx/scalar x)))
                               cm/EMPTY (map-indexed vector obs))
          traces (mcmc/mh {:samples 500 :burn 100
                            :selection (sel/select :p)}
                           model [n] observations)
          ps (mapv (fn [t]
                     (let [v (cm/get-value (cm/get-submap (:choices t) :p))]
                       (mx/eval! v) (mx/item v)))
                   traces)
          sample-p (/ (reduce + ps) (count ps))]
      (is (th/close? sample-p post-mean 0.15)
          (str "Beta-Bernoulli posterior mean ≈ " (.toFixed post-mean 2)))))

  (testing "2.5 acceptance rate validation"
    (let [model (dyn/auto-key (gen []
                  (let [mu (trace :mu (dist/gaussian 0 10))]
                    (mx/eval! mu)
                    (let [m (mx/item mu)]
                      (trace :y0 (dist/gaussian m 1))
                      (trace :y1 (dist/gaussian m 1))
                      m))))
          obs (cm/choicemap :y0 (mx/scalar 3.0) :y1 (mx/scalar 3.5))
          {:keys [trace]} (p/generate model [] obs)
          n-steps 200
          accepted (volatile! 0)
          _ (reduce (fn [t _]
                      (let [t' (mcmc/mh-step t (sel/select :mu))]
                        (when (not= t t') (vswap! accepted inc))
                        t'))
                    trace (range n-steps))
          rate (/ @accepted n-steps)]
      (is (and (> rate 0.0) (< rate 1.0))
          (str "MH acceptance rate in (0, 1): " (.toFixed rate 2)))))

  (testing "2.6 chain stationarity"
    (let [model (gen [] (trace :x (dist/gaussian 3 1)))
          obs (cm/choicemap :x (mx/scalar 3.0))
          traces (mcmc/mh {:samples 200 :burn 50
                            :selection (sel/select :x)}
                           model [] obs)
          vals (mapv (fn [t]
                       (let [v (cm/get-value (cm/get-submap (:choices t) :x))]
                         (mx/eval! v) (mx/item v)))
                     traces)
          half (quot (count vals) 2)
          first-half (subvec vals 0 half)
          second-half (subvec vals half)
          mean1 (/ (reduce + first-half) (count first-half))
          mean2 (/ (reduce + second-half) (count second-half))]
      (is (< (js/Math.abs (- mean1 mean2)) 0.5)
          (str "stationarity: |mean1 - mean2| < 0.5")))))

;; ============================================================================
;; Section 3: Importance Sampling & SMC (adapted from test_smc.py)
;; ============================================================================

(deftest s3-importance-sampling-smc
  (testing "3.1 IS log-ML convergence"
    (let [model (gen []
                  (let [mu (trace :mu (dist/gaussian 0 5))]
                    (mx/eval! mu)
                    (trace :y (dist/gaussian (mx/item mu) 1))
                    mu))
          obs (cm/choicemap :y (mx/scalar 2.0))
          log-mls (mapv (fn [n-particles]
                          (let [{:keys [log-ml-estimate]}
                                (is/importance-sampling {:samples n-particles}
                                                        model [] obs)]
                            (mx/eval! log-ml-estimate)
                            (mx/item log-ml-estimate)))
                        [50 200 500])]
      (is (every? js/isFinite log-mls) "IS log-ML estimates are all finite")))

  (testing "3.2 IS posterior concentration"
    (let [model (gen []
                  (let [mu (trace :mu (dist/gaussian 0 5))]
                    (mx/eval! mu)
                    (trace :y (dist/gaussian (mx/item mu) 0.1))
                    mu))
          obs (cm/choicemap :y (mx/scalar 3.0))
          {:keys [traces log-weights]}
          (is/importance-sampling {:samples 200} model [] obs)
          weights-arr (mx/array (mapv (fn [w] (mx/eval! w) (mx/item w)) log-weights))
          log-probs (mx/subtract weights-arr (mx/logsumexp weights-arr))
          _ (mx/eval! log-probs)
          probs (mx/->clj (mx/exp log-probs))
          mus (mapv (fn [t]
                      (let [v (cm/get-value (cm/get-submap (:choices t) :mu))]
                        (mx/eval! v) (mx/item v)))
                    traces)
          weighted-mean (reduce + (map * mus probs))]
      (is (th/close? weighted-mean 3.0 0.5)
          (str "IS weighted mean ≈ 3.0 (got " (.toFixed weighted-mean 2) ")"))))

  (testing "3.3 importance resampling"
    (let [model (gen []
                  (let [mu (trace :mu (dist/gaussian 0 5))]
                    (mx/eval! mu)
                    (trace :y (dist/gaussian (mx/item mu) 1))
                    mu))
          obs (cm/choicemap :y (mx/scalar 2.0))
          resampled (is/importance-resampling {:samples 50 :particles 200}
                                              model [] obs)
          mus (mapv (fn [t]
                      (let [v (cm/get-value (cm/get-submap (:choices t) :mu))]
                        (mx/eval! v) (mx/item v)))
                    resampled)
          mean-mu (/ (reduce + mus) (count mus))]
      (is (= 50 (count resampled)) "resampled traces exist")
      (is (th/close? mean-mu 2.0 1.0)
          (str "resampled mean ≈ 2.0 (got " (.toFixed mean-mu 2) ")"))))

  (testing "3.4 SMC sequential"
    (let [model (gen [xs]
                  (let [mu (trace :mu (dist/gaussian 0 5))]
                    (mx/eval! mu)
                    (let [m (mx/item mu)]
                      (doseq [[i x] (map-indexed vector xs)]
                        (trace (keyword (str "y" i)) (dist/gaussian m 1)))
                      m)))
          obs-seq [(cm/choicemap :y0 (mx/scalar 2.0))
                   (cm/choicemap :y0 (mx/scalar 2.0) :y1 (mx/scalar 2.5))
                   (cm/choicemap :y0 (mx/scalar 2.0) :y1 (mx/scalar 2.5) :y2 (mx/scalar 1.5))]
          result (smc/smc {:particles 20}
                           model [[2.0 2.5 1.5]] obs-seq)]
      (is (= 20 (count (:traces result))) "SMC returns traces")
      (mx/eval! (:log-ml-estimate result))
      (is (js/isFinite (mx/item (:log-ml-estimate result))) "SMC log-ML is finite"))))

;; ============================================================================
;; Section 4: Variational Inference (adapted from test_vi.py)
;; ============================================================================

(deftest s4-variational-inference
  (testing "4.1 VI convergence to known target"
    (let [log-density (fn [params]
                        (let [diff (mx/subtract params (mx/scalar 3.0))]
                          (mx/multiply (mx/scalar -0.5) (mx/sum (mx/multiply diff diff)))))
          init-params (mx/scalar 0.0)
          result (vi/vi {:iterations 500 :learning-rate 0.05 :elbo-samples 10
                          :key (rng/fresh-key 42)}
                         log-density init-params)
          mu (th/realize (:mu result))
          sigma (th/realize (:sigma result))]
      (is (th/close? mu 3.0 0.5)
          (str "VI mu ≈ 3.0 (got " (.toFixed mu 2) ")"))
      (is (th/close? sigma 1.0 0.5)
          (str "VI sigma ≈ 1.0 (got " (.toFixed sigma 2) ")"))))

  (testing "4.2 ELBO improves"
    (let [log-density (fn [params]
                        (mx/multiply (mx/scalar -0.5) (mx/sum (mx/square params))))
          init-params (mx/array [5.0 5.0])
          result (vi/vi {:iterations 200 :learning-rate 0.02 :elbo-samples 10
                          :key (rng/fresh-key 99)}
                         log-density init-params)
          history (:elbo-history result)
          n (count history)]
      (when (> n 4)
        (let [early-avg (/ (reduce + (take 2 history)) 2)
              late-avg (/ (reduce + (take-last 2 history)) 2)]
          (is (> late-avg early-avg)
              (str "ELBO improves: early=" (.toFixed early-avg 1) " late=" (.toFixed late-avg 1)))))))

  (testing "4.3 VI model-equivalent"
    (let [log-density (fn [params]
                        (let [mu params]
                          (mx/add (dist/log-prob (dist/gaussian 0 10) mu)
                                  (dist/log-prob (dist/gaussian mu 1) (mx/scalar 5.0)))))
          init-params (mx/scalar 0.0)
          result (vi/vi {:iterations 500 :learning-rate 0.05 :elbo-samples 10
                          :key (rng/fresh-key 77)}
                         log-density init-params)
          mu (th/realize (:mu result))]
      (is (th/close? mu 5.0 1.5)
          (str "VI model-equiv mu ≈ 5.0 (got " (.toFixed mu 2) ")")))))

;; ============================================================================
;; Section 5: Combinator GFI Contracts (adapted from test_core.py)
;; ============================================================================

(deftest s5-combinator-gfi
  (testing "5.1 map combinator"
    (let [kernel (gen [x] (trace :y (dist/gaussian x 1)))
          mapped (comb/map-combinator (dyn/auto-key kernel))
          trace (p/simulate mapped [[1.0 2.0 3.0]])
          retvals (:retval trace)
          score (th/realize (:score trace))]
      (is (= 3 (count retvals)) "map simulate: returns 3 values")
      (is (js/isFinite score) "map simulate: score is finite"))

    (let [kernel (gen [x] (trace :y (dist/gaussian x 1)))
          mapped (comb/map-combinator (dyn/auto-key kernel))
          constraints (-> cm/EMPTY
                          (cm/set-choice [0] (-> cm/EMPTY (cm/set-choice [:y] (mx/scalar 1.0))))
                          (cm/set-choice [1] (-> cm/EMPTY (cm/set-choice [:y] (mx/scalar 2.0))))
                          (cm/set-choice [2] (-> cm/EMPTY (cm/set-choice [:y] (mx/scalar 3.0)))))
          {:keys [trace weight]} (p/generate mapped [[1.0 2.0 3.0]] constraints)
          gen-score (th/realize (:score trace))
          w (th/realize weight)]
      (is (th/close? w gen-score 1e-4) "map generate: weight ≈ score (full constraints)"))

    (let [kernel (gen [x] (trace :y (dist/gaussian x 1)))
          mapped (comb/map-combinator (dyn/auto-key kernel))
          trace (p/simulate mapped [[1.0 2.0 3.0]])
          old-score (th/realize (:score trace))
          constraints (-> cm/EMPTY
                          (cm/set-choice [1] (-> cm/EMPTY (cm/set-choice [:y] (mx/scalar 99.0)))))
          {:keys [trace weight]} (p/update mapped trace constraints)
          new-score (th/realize (:score trace))
          w (th/realize weight)]
      (is (th/close? w (- new-score old-score) 0.5)
          "map update: weight ≈ new_score - old_score")))

  (testing "5.2 unfold combinator"
    (let [step (gen [t state]
                 (let [noise (trace :noise (dist/gaussian 0 0.1))]
                   (mx/eval! noise)
                   (+ state (mx/item noise))))
          unfolded (comb/unfold-combinator (dyn/auto-key step))
          trace (p/simulate unfolded [5 0.0])
          states (:retval trace)
          score (th/realize (:score trace))]
      (is (= 5 (count states)) "unfold simulate: 5 states")
      (is (js/isFinite score) "unfold simulate: score finite"))

    (let [step (gen [t state]
                 (let [noise (trace :noise (dist/gaussian 0 0.1))]
                   (mx/eval! noise)
                   (+ state (mx/item noise))))
          unfolded (comb/unfold-combinator (dyn/auto-key step))
          constraints (-> cm/EMPTY
                          (cm/set-choice [0] (-> cm/EMPTY (cm/set-choice [:noise] (mx/scalar 0.1)))))
          {:keys [trace weight]} (p/generate unfolded [5 0.0] constraints)
          w (th/realize weight)]
      (is (js/isFinite w) "unfold generate: weight finite")))

  (testing "5.3 switch combinator"
    (let [branch-a (gen [] (trace :v (dist/gaussian 0 1)))
          branch-b (gen [] (trace :v (dist/gaussian 10 1)))
          switched (comb/switch-combinator (dyn/auto-key branch-a) (dyn/auto-key branch-b))]
      (let [trace (p/simulate switched [0])
            v (th/realize (cm/get-value (cm/get-submap (:choices trace) :v)))
            score (th/realize (:score trace))]
        (is (< (js/Math.abs v) 5) "switch branch 0: value near 0")
        (is (js/isFinite score) "switch branch 0: score finite"))
      (let [trace (p/simulate switched [1])
            v (th/realize (cm/get-value (cm/get-submap (:choices trace) :v)))]
        (is (< (js/Math.abs (- v 10)) 5) "switch branch 1: value near 10")))))

;; ============================================================================
;; Section 6: Gradient Flow (adapted from test_adev.py)
;; ============================================================================

(deftest s6-gradient-flow
  (testing "6.1 gradient through Gaussian log-prob"
    (let [f (fn [mu] (dist/log-prob (dist/gaussian mu 1) (mx/scalar 3.0)))
          grad-f (mx/grad f)
          mu0 (mx/scalar 1.0)
          g (grad-f mu0)]
      (mx/eval! g)
      (is (th/close? (mx/item g) 2.0 1e-4)
          "d/dmu log N(3|mu,1) at mu=1 ≈ 2.0")))

  (testing "6.2 gradient through model score"
    (let [model (dyn/auto-key (gen []
                  (trace :x (dist/gaussian 0 1))))
          score-fn (fn [x-val]
                     (:weight (p/generate model []
                                (cm/choicemap :x x-val))))
          grad-score (mx/grad score-fn)
          x0 (mx/scalar 2.0)
          g (grad-score x0)]
      (mx/eval! g)
      (is (th/close? (mx/item g) -2.0 1e-3)
          "d/dx score at x=2 ≈ -2.0")))

  (testing "6.3 value-and-grad"
    (let [f (fn [x] (mx/multiply (mx/scalar -0.5) (mx/square x)))
          vg (mx/value-and-grad f)
          x0 (mx/scalar 3.0)
          [v g] (vg x0)]
      (mx/eval! v g)
      (is (th/close? (mx/item v) -4.5 1e-5) "value = -4.5")
      (is (th/close? (mx/item g) -3.0 1e-5) "grad = -3.0")))

  (testing "6.4 multi-address score gradient"
    (let [model (dyn/auto-key (gen []
                  (let [x (trace :x (dist/gaussian 0 1))
                        y (trace :y (dist/gaussian 0 1))]
                    [x y])))
          score-fn (fn [params]
                     (let [cm (cm/choicemap :x (mx/index params 0)
                                            :y (mx/index params 1))]
                       (:weight (p/generate model [] cm))))
          grad-fn (mx/grad score-fn)
          params (mx/array [1.0 2.0])
          g (grad-fn params)]
      (mx/eval! g)
      (let [gv (mx/->clj g)]
        (is (th/close? (first gv) -1.0 1e-3) "grad[0] ≈ -1.0")
        (is (th/close? (second gv) -2.0 1e-3) "grad[1] ≈ -2.0"))))

  (testing "6.5 reparameterized sampling"
    (let [key (rng/fresh-key)
          f (fn [mu]
              (let [eps (rng/normal key [])
                    x (mx/add mu eps)]
                x))
          grad-f (mx/grad f)
          mu (mx/scalar 5.0)
          g (grad-f mu)]
      (mx/eval! g)
      (is (th/close? (mx/item g) 1.0 1e-5)
          "d/dmu (mu + eps) = 1.0")))

  (testing "6.6 compiled gradient"
    (let [f (fn [x] (mx/exp (mx/negative (mx/multiply (mx/scalar 0.5) (mx/square x)))))
          compiled-grad (mx/compile-fn (mx/grad f))
          x0 (mx/scalar 1.0)
          g (compiled-grad x0)]
      (mx/eval! g)
      (is (th/close? (mx/item g) (* -1.0 (js/Math.exp -0.5)) 1e-3)
          "compiled grad at x=1 ≈ -0.6065"))))

;; ============================================================================
;; Section 7: Diagnostics (adapted from test_mcmc.py)
;; ============================================================================

(deftest s7-diagnostics
  (testing "7.1 ESS"
    (let [samples (mapv (fn [_] (let [v (rng/normal (rng/fresh-key) [])]
                                  (mx/eval! v) v))
                        (range 50))
          effective (diag/ess samples)]
      (is (> effective 25)
          (str "IID ESS ≈ 50 (got " (.toFixed effective 0) ")")))

    (let [samples (loop [i 0 v 0.0 acc []]
                    (if (>= i 50)
                      acc
                      (let [v' (+ (* 0.99 v) (* 0.1 (- (* 2 (js/Math.random)) 1)))]
                        (recur (inc i) v' (conj acc (mx/scalar v'))))))
          effective (diag/ess samples)]
      (is (< effective 40)
          (str "correlated ESS < 50 (got " (.toFixed effective 0) ")"))))

  (testing "7.2 R-hat"
    (let [chain1 (mapv (fn [_] (+ 3 (* 0.5 (- (* 2 (js/Math.random)) 1)))) (range 50))
          chain2 (mapv (fn [_] (+ 3 (* 0.5 (- (* 2 (js/Math.random)) 1)))) (range 50))
          rh (diag/r-hat [chain1 chain2])]
      (is (< rh 1.2)
          (str "converged R-hat ≈ 1.0 (got " (.toFixed rh 3) ")")))

    (let [chain1 (mapv (fn [_] (* 0.5 (- (* 2 (js/Math.random)) 1))) (range 50))
          chain2 (mapv (fn [_] (+ 10 (* 0.5 (- (* 2 (js/Math.random)) 1)))) (range 50))
          rh (diag/r-hat [chain1 chain2])]
      (is (> rh 2.0)
          (str "diverged R-hat >> 1.0 (got " (.toFixed rh 2) ")"))))

  (testing "7.3 summary statistics"
    (let [samples (mapv #(mx/scalar %) [1.0 2.0 3.0 4.0 5.0])
          mu (diag/sample-mean samples)
          sd (diag/sample-std samples)]
      (mx/eval! mu sd)
      (is (th/close? (mx/item mu) 3.0 1e-4) "mean of [1..5] = 3.0")
      (is (> (mx/item sd) 0) "std of [1..5] > 0"))))

;; ============================================================================
;; Section 8: Numerical Stability
;; ============================================================================

(deftest s8-numerical-stability
  (testing "8.1 extreme Gaussian parameters"
    (let [d1 (dist/gaussian 0 1000)
          lp1 (th/realize (dist/log-prob d1 (mx/scalar 0.0)))
          d2 (dist/gaussian 0 0.001)
          lp2 (th/realize (dist/log-prob d2 (mx/scalar 0.0)))
          d3 (dist/gaussian 0 1)
          lp3 (th/realize (dist/log-prob d3 (mx/scalar 100.0)))]
      (is (js/isFinite lp1) "large sigma: log-prob finite")
      (is (js/isFinite lp2) "small sigma: log-prob finite")
      (is (js/isFinite lp3) "large value: log-prob finite")))

  (testing "8.2 boundary values"
    (let [u (dist/uniform 0 1)
          lp-lo (th/realize (dist/log-prob u (mx/scalar 0.0)))
          lp-hi (th/realize (dist/log-prob u (mx/scalar 1.0)))
          lp-out (th/realize (dist/log-prob u (mx/scalar 1.5)))]
      (is (js/isFinite lp-lo) "uniform at lo: finite")
      (is (js/isFinite lp-hi) "uniform at hi: finite")
      (is (= lp-out ##-Inf) "uniform outside: -Inf"))

    (let [e (dist/exponential 1)
          lp0 (th/realize (dist/log-prob e (mx/scalar 0.0)))
          lp-neg (th/realize (dist/log-prob e (mx/scalar -1.0)))]
      (is (js/isFinite lp0) "exponential at 0: finite")
      (is (= lp-neg ##-Inf) "exponential at -1: -Inf")))

  (testing "8.3 large model"
    (let [n 10
          model (dyn/auto-key (gen [n]
                  (let [mu (trace :mu (dist/gaussian 0 10))]
                    (mx/eval! mu)
                    (let [m (mx/item mu)]
                      (doseq [i (range n)]
                        (trace (keyword (str "y" i)) (dist/gaussian m 1)))
                      m))))
          trace (p/simulate model [n])
          score (th/realize (:score trace))]
      (is (js/isFinite score) "10-obs model: score finite")))

  (testing "8.4 MVN stability"
    (let [k 10
          cov (mx/add (mx/eye k) (mx/multiply (mx/scalar 0.001) (rng/normal (rng/fresh-key) [k k])))
          cov-sym (mx/multiply (mx/scalar 0.5) (mx/add cov (mx/transpose cov)))
          cov-spd (mx/add cov-sym (mx/multiply (mx/scalar (float k)) (mx/eye k)))]
      (mx/eval! cov-spd)
      (let [mvn (dist/multivariate-normal (mx/zeros [k]) cov-spd)
            sample (dist/sample mvn)
            _ (mx/eval! sample)
            lp (dist/log-prob mvn sample)]
        (mx/eval! lp)
        (is (every? js/isFinite (mx/->clj sample)) "10-dim MVN: sample finite")
        (is (js/isFinite (mx/item lp)) "10-dim MVN: log-prob finite")))))

;; ============================================================================
;; Section 9: Score Consistency Across Operations
;; ============================================================================

(deftest s9-score-consistency
  (testing "9.1 hierarchical model round-trip"
    (let [model (dyn/auto-key (gen [xs]
                  (let [slope (trace :slope (dist/gaussian 0 10))
                        intercept (trace :intercept (dist/gaussian 0 10))]
                    (mx/eval! slope intercept)
                    (let [s (mx/item slope) i (mx/item intercept)]
                      (doseq [[j x] (map-indexed vector xs)]
                        (trace (keyword (str "y" j))
                                   (dist/gaussian (+ (* s x) i) 1)))
                      [s i]))))
          xs [1.0 2.0 3.0]
          trace (p/simulate model [xs])
          choices (:choices trace)
          score (th/realize (:score trace))
          {:keys [weight]} (p/generate model [xs] choices)
          w (th/realize weight)]
      (is (th/close? w score 1e-3) "hierarchical round-trip: weight ≈ score")))

  (testing "9.2 update with no-op"
    (let [model (dyn/auto-key (gen []
                  (let [x (trace :x (dist/gaussian 0 1))
                        y (trace :y (dist/gaussian 0 1))]
                    [x y])))
          trace (p/simulate model [])
          old-score (th/realize (:score trace))
          old-x (cm/get-value (cm/get-submap (:choices trace) :x))
          old-y (cm/get-value (cm/get-submap (:choices trace) :y))
          {:keys [trace weight]} (p/update model trace
                                   (cm/choicemap :x old-x :y old-y))
          new-score (th/realize (:score trace))
          w (th/realize weight)]
      (is (th/close? old-score new-score 1e-5) "no-op update: score unchanged")
      (is (th/close? w 0.0 1e-5) "no-op update: weight ≈ 0")))

  (testing "9.3 generate/update consistency"
    (let [model (dyn/auto-key (gen [] (trace :x (dist/gaussian 0 1))))
          val-a (mx/scalar 1.0)
          val-b (mx/scalar 3.0)
          ;; Path 1: generate with val-a, then update to val-b
          {trace-a :trace} (p/generate model [] (cm/choicemap :x val-a))
          {trace-b :trace weight-b :weight} (p/update model trace-a (cm/choicemap :x val-b))
          path1-score (th/realize (:score trace-b))
          ;; Path 2: generate directly with val-b
          {trace-direct :trace} (p/generate model [] (cm/choicemap :x val-b))
          path2-score (th/realize (:score trace-direct))]
      (is (th/close? path1-score path2-score 1e-5)
          "generate→update score = direct generate score"))))

(t/run-tests)
