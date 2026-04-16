(ns genmlx.gfi-laws-test-p6
  "GFI law tests part 6: DENOTATIONAL SEMANTICS laws"
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
;; DENOTATIONAL SEMANTICS laws [T] Chapter 2.2.2, Figure 2-1
;; Cusumano-Towner 2020 PhD thesis, §2.2.2
;;
;; The thesis defines three semantic functions for a toy modeling language:
;;   Addrs⟦E⟧          — set of addresses used by expression E
;;   Val⟦E⟧(σ)(τ)      — value of E given environment σ and choices τ
;;   Dist⟦E⟧(σ)(τ)     — probability distribution on choices for E
;;
;; The top-level denotation maps a gen definition to a generative function:
;;   ⟦@gen function(X₁,...,Xₙ) E end⟧ = (Rⁿ, R, λx,τ.Dist⟦E⟧, λx,τ.Val⟦E⟧)
;; ---------------------------------------------------------------------------

;; --- Law #38: Addrs⟦E⟧ correctness ---
;; GenMLX's schema/extract-schema walks the gen body source form at construction
;; time and extracts :trace-sites. For static models (no branches, no dynamic
;; address construction), the schema addresses must exactly equal the runtime
;; trace addresses produced by simulate.

(t/deftest law:addrs-correctness-two-site
  (t/testing "Addrs⟦E⟧: schema addresses = runtime addresses (2-site chain)"
    ;; Model: x ~ N(0,1), y ~ N(x, 0.5)
    ;; Addrs⟦E⟧ = {:x, :y}
    (let [model (:model gaussian-chain)
          schema (:schema model)
          schema-addrs (set (map :addr (:trace-sites schema)))
          trace (p/simulate model [])
          trace-addrs (set (map first (cm/addresses (:choices trace))))]
      (t/is (:static? schema)
            "gaussian-chain should be classified as static")
      (t/is (= schema-addrs trace-addrs)
            (str "Addrs mismatch: schema=" schema-addrs " trace=" trace-addrs)))))

(t/deftest law:addrs-correctness-three-site
  (t/testing "Addrs⟦E⟧: schema addresses = runtime addresses (3-site chain)"
    ;; Model: a ~ N(0,1), b ~ N(a,1), c ~ N(a+b, 0.5)
    ;; Addrs⟦E⟧ = {:a, :b, :c}
    (let [model (:model three-chain)
          schema (:schema model)
          schema-addrs (set (map :addr (:trace-sites schema)))
          trace (p/simulate model [])
          trace-addrs (set (map first (cm/addresses (:choices trace))))]
      (t/is (:static? schema)
            "three-chain should be classified as static")
      (t/is (= schema-addrs trace-addrs)
            (str "Addrs mismatch: schema=" schema-addrs " trace=" trace-addrs)))))

(t/deftest law:addrs-dependency-graph
  (t/testing "Addrs⟦E⟧: dependency structure matches semantic analysis"
    ;; For three-chain: :a has no deps, :b depends on :a, :c depends on {:a, :b}
    ;; This is the dependency structure that Addrs⟦E⟧ implicitly encodes:
    ;; each trace site's distribution depends on previously-traced values.
    (let [schema (:schema (:model three-chain))
          dep-map (into {} (map (fn [s] [(:addr s) (:deps s)])
                                (:trace-sites schema)))]
      (t/is (= #{} (get dep-map :a))
            ":a should have no dependencies")
      (t/is (= #{:a} (get dep-map :b))
            ":b should depend only on :a")
      (t/is (= #{:a :b} (get dep-map :c))
            ":c should depend on both :a and :b"))))

;; --- Law #39: Val⟦E⟧ determinism ---
;; Val⟦E⟧(σ)(τ) is a mathematical function: given the same environment (args)
;; and choices (τ), the return value is deterministic. This is the denotational
;; semantics perspective on law #8 (return-value-independence): the value
;; function f in the GFI tuple (X, Y, p, f) is well-defined as a mathematical
;; function from (X × T) → Y, not a stochastic procedure.

(t/deftest law:val-determinism-chain
  (t/testing "Val⟦E⟧: f(args, τ) is deterministic (chain model)"
    ;; Model B: (gen [] (let [x (trace :x ...)] (trace :y ...)))
    ;; f([], τ) = τ[:y] (return value is the last trace expression)
    (let [model (:model gaussian-chain)
          tau (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0))
          r1 (p/generate model [] tau)
          r2 (p/generate model [] tau)
          rv1 (ev (:retval (:trace r1)))
          rv2 (ev (:retval (:trace r2)))]
      ;; Tolerance: 1e-10 (algebraic identity — no randomness, rounding only)
      (t/is (close? rv1 rv2 1e-10)
            (str "Val not deterministic: " rv1 " vs " rv2))
      ;; Verify retval = τ[:y]
      (t/is (close? rv1 1.0 1e-6)
            (str "Expected retval=1.0 (τ[:y]), got " rv1)))))

(t/deftest law:val-determinism-retval-not-last-trace
  (t/testing "Val⟦E⟧: retval = body return value, not necessarily last trace"
    ;; Model A: (gen [] (let [x (trace :x ...) y (trace :y ...)] x))
    ;; f([], τ) = τ[:x] (body returns x, not y)
    (let [model (:model two-independent)
          tau (cm/choicemap :x (mx/scalar 3.14) :y (mx/scalar -2.0))
          r1 (p/generate model [] tau)
          r2 (p/generate model [] tau)
          rv1 (ev (:retval (:trace r1)))
          rv2 (ev (:retval (:trace r2)))]
      ;; Tolerance: 1e-10 (algebraic identity)
      (t/is (close? rv1 rv2 1e-10)
            (str "Val not deterministic: " rv1 " vs " rv2))
      ;; Verify retval = τ[:x], not τ[:y]
      (t/is (close? rv1 3.14 1e-4)
            (str "Expected retval=3.14 (τ[:x]), got " rv1)))))

;; --- Law #40: Dist⟦E⟧ analytical match ---
;; For static models, Dist⟦E⟧ defines the joint density on choices.
;; We derive analytical moments of the joint distribution and verify
;; that simulated traces match within statistically justified tolerances.
;;
;; Tolerance policy for moment-matching (N=5000 iid samples):
;;   SE(mean) = sqrt(Var / N)
;;   SE(variance) = sqrt(2 * Var^2 / N)  [Gaussian kurtosis = 3]
;;   SE(covariance) = sqrt((Var_x * Var_y + Cov_xy^2) / N)  [Isserlis]
;;   Tolerance = 3.5 * SE  (p(false positive) < 0.001 for Gaussian)

(t/deftest law:dist-model-A-independent-gaussians
  (t/testing "Dist⟦E⟧ Model A: x ~ N(0,1), y ~ N(0,2) independent"
    ;; Analytical derivation:
    ;;   p(τ) = N(τ[:x]; 0, 1) * N(τ[:y]; 0, 2)
    ;;   E[x] = 0, Var[x] = 1
    ;;   E[y] = 0, Var[y] = sigma^2 = 4
    ;;   Cov(x,y) = 0  (independent)
    (let [model (dyn/auto-key (gen [] (let [x (trace :x (dist/gaussian 0 1))
                                            y (trace :y (dist/gaussian 0 2))]
                                        x)))
          samples (collect-samples model [] [:x :y] N-moment-samples)
          xs (get samples :x)
          ys (get samples :y)
          ;; Tolerances (N=5000, z=3.5):
          ;;   SE(mean_x) = sqrt(1/5000) = 0.01414, tol = 0.050
          ;;   SE(mean_y) = sqrt(4/5000) = 0.02828, tol = 0.099
          ;;   SE(var_x) = sqrt(2/5000) = 0.0200, tol = 0.070
          ;;   SE(var_y) = sqrt(2*16/5000) = 0.0800, tol = 0.280
          ;;   SE(cov_xy) = sqrt(1*4/5000) = 0.0283, tol = 0.099  [independent: Isserlis simplifies]
          ]
      (t/is (close? (sample-mean xs) 0.0 0.050)
            (str "E[x]: expected 0, got " (sample-mean xs)))
      (t/is (close? (sample-mean ys) 0.0 0.099)
            (str "E[y]: expected 0, got " (sample-mean ys)))
      (t/is (close? (sample-var xs) 1.0 0.070)
            (str "Var[x]: expected 1, got " (sample-var xs)))
      (t/is (close? (sample-var ys) 4.0 0.280)
            (str "Var[y]: expected 4, got " (sample-var ys)))
      (t/is (close? (sample-cov xs ys) 0.0 0.099)
            (str "Cov(x,y): expected 0, got " (sample-cov xs ys))))))

(t/deftest law:dist-model-B-gaussian-chain
  (t/testing "Dist⟦E⟧ Model B: x ~ N(0,1), y|x ~ N(x, 0.5)"
    ;; Analytical derivation:
    ;;   y = x + eps, eps ~ N(0, 0.5) independent of x
    ;;   E[x] = 0, Var[x] = 1
    ;;   E[y] = E[x + eps] = 0
    ;;   Var[y] = Var[x] + Var[eps] = 1 + 0.25 = 1.25
    ;;   Cov(x,y) = Cov(x, x+eps) = Var(x) = 1
    ;;   Corr(x,y) = 1/sqrt(1 * 1.25) = 2/sqrt(5) = 0.89443
    (let [model (:model gaussian-chain)
          samples (collect-samples model [] [:x :y] N-moment-samples)
          xs (get samples :x) ys (get samples :y)
          ;; Tolerances (N=5000, z=3.5):
          ;;   SE(mean_y) = sqrt(1.25/5000) = 0.01581, tol = 0.055
          ;;   SE(var_y) = sqrt(2*1.5625/5000) = 0.0250, tol = 0.088
          ;;   SE(cov) = sqrt((1*1.25 + 1^2)/5000) = sqrt(2.25/5000) = 0.02121, tol = 0.074
          ;;   For corr: delta method, tol ~ 0.06
          obs-cov (sample-cov xs ys)
          obs-corr (/ obs-cov (js/Math.sqrt (* (sample-var xs) (sample-var ys))))]
      (t/is (close? (sample-mean xs) 0.0 0.050)
            (str "E[x]: expected 0, got " (sample-mean xs)))
      (t/is (close? (sample-mean ys) 0.0 0.055)
            (str "E[y]: expected 0, got " (sample-mean ys)))
      (t/is (close? (sample-var xs) 1.0 0.070)
            (str "Var[x]: expected 1, got " (sample-var xs)))
      (t/is (close? (sample-var ys) 1.25 0.088)
            (str "Var[y]: expected 1.25, got " (sample-var ys)))
      (t/is (close? obs-cov 1.0 0.074)
            (str "Cov(x,y): expected 1, got " obs-cov))
      (t/is (close? obs-corr (/ 2.0 (js/Math.sqrt 5.0)) 0.06)
            (str "Corr(x,y): expected 2/sqrt(5)=" (/ 2.0 (js/Math.sqrt 5.0))
                 ", got " obs-corr)))))

(t/deftest law:dist-model-C-three-site-chain
  (t/testing "Dist⟦E⟧ Model C: a ~ N(0,1), b|a ~ N(a,1), c|a,b ~ N(a+b, 0.5)"
    ;; Analytical derivation:
    ;;   b = a + eps1, eps1 ~ N(0,1)
    ;;   c = (a + b) + eps2, eps2 ~ N(0, 0.5)
    ;;
    ;;   E[a] = 0, Var[a] = 1
    ;;   E[b] = 0, Var[b] = Var[a] + Var[eps1] = 2
    ;;   Cov(a,b) = Cov(a, a+eps1) = Var(a) = 1
    ;;   Var[a+b] = Var[a] + Var[b] + 2*Cov(a,b) = 1 + 2 + 2 = 5
    ;;   E[c] = 0, Var[c] = Var[a+b] + Var[eps2] = 5 + 0.25 = 5.25
    ;;   Cov(a,c) = Cov(a, a+b+eps2) = Var(a) + Cov(a,b) = 1 + 1 = 2
    ;;   Cov(b,c) = Cov(b, a+b+eps2) = Cov(b,a) + Var(b) = 1 + 2 = 3
    (let [model (:model three-chain)
          samples (collect-samples model [] [:a :b :c] N-moment-samples)
          as (get samples :a) bs (get samples :b) cs (get samples :c)
          ;; Tolerances (N=5000, z=3.5):
          ;;   SE(mean_b) = sqrt(2/5000) = 0.0200, tol = 0.070
          ;;   SE(mean_c) = sqrt(5.25/5000) = 0.0324, tol = 0.113
          ;;   SE(var_b) = sqrt(2*4/5000) = 0.0400, tol = 0.140
          ;;   SE(var_c) = sqrt(2*27.5625/5000) = 0.1050, tol = 0.367
          ;;   SE(cov_ab) = sqrt((1*2+1)/5000) = 0.0245, tol = 0.086
          ;;   SE(cov_ac) = sqrt((1*5.25+4)/5000) = 0.0430, tol = 0.150
          ;;   SE(cov_bc) = sqrt((2*5.25+9)/5000) = 0.0625, tol = 0.219
          ]
      (t/is (close? (sample-mean as) 0.0 0.050)
            (str "E[a]: expected 0, got " (sample-mean as)))
      (t/is (close? (sample-mean bs) 0.0 0.070)
            (str "E[b]: expected 0, got " (sample-mean bs)))
      (t/is (close? (sample-mean cs) 0.0 0.113)
            (str "E[c]: expected 0, got " (sample-mean cs)))
      (t/is (close? (sample-var as) 1.0 0.070)
            (str "Var[a]: expected 1, got " (sample-var as)))
      (t/is (close? (sample-var bs) 2.0 0.140)
            (str "Var[b]: expected 2, got " (sample-var bs)))
      (t/is (close? (sample-var cs) 5.25 0.367)
            (str "Var[c]: expected 5.25, got " (sample-var cs)))
      (t/is (close? (sample-cov as bs) 1.0 0.086)
            (str "Cov(a,b): expected 1, got " (sample-cov as bs)))
      (t/is (close? (sample-cov as cs) 2.0 0.150)
            (str "Cov(a,c): expected 2, got " (sample-cov as cs)))
      (t/is (close? (sample-cov bs cs) 3.0 0.219)
            (str "Cov(b,c): expected 3, got " (sample-cov bs cs))))))

;; --- Law #41: Gen function denotation ---
;; ⟦gen body⟧ = P = (X, Y, p, f) — all four components verified simultaneously.
;;
;; For a concrete model, generate with full constraints and verify:
;;   - X: model accepts declared arguments
;;   - Y: retval type and value match f(args, τ)
;;   - p: score = log p(τ; x) matches analytical joint density
;;   - f: retval = f(x, τ) is deterministic function of (args, choices)

(t/deftest law:gen-denotation-model-B
  (t/testing "⟦gen⟧ = (X, Y, p, f): all four components for gaussian chain"
    ;; Model B: (gen [] (let [x (trace :x (dist/gaussian 0 1))]
    ;;                     (trace :y (dist/gaussian x 0.5))))
    ;; X = []
    ;; Y = R
    ;; p(τ) = N(τ[:x]; 0, 1) * N(τ[:y]; τ[:x], 0.5)
    ;; f([], τ) = τ[:y]
    ;;
    ;; At τ = {:x 0.5, :y 1.0}:
    ;;
    ;; log N(0.5; 0, 1) = -0.5*log(2π) - log(1) - 0.5*(0.5/1)²
    ;;                   = -0.91893853 - 0 - 0.125
    ;;                   = -1.04393853
    ;;
    ;; log N(1.0; 0.5, 0.5) = -0.5*log(2π) - log(0.5) - 0.5*((1.0-0.5)/0.5)²
    ;;                       = -0.91893853 + 0.69314718 - 0.5
    ;;                       = -0.72579135
    ;;
    ;; score = -1.04393853 + -0.72579135 = -1.76972988
    ;; retval = τ[:y] = 1.0
    ;; Tolerance: 1e-4 (float32 accumulation in two log-prob computations + addition)
    (let [model (:model gaussian-chain)
          tau (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0))
          result (p/generate model [] tau)
          trace (:trace result)
          ;; Analytical score
          log-2pi-half (* 0.5 (js/Math.log (* 2.0 js/Math.PI)))
          lp-x (- (- log-2pi-half) (* 0.5 0.25)) ;; N(0.5; 0, 1)
          lp-y (- (- (- log-2pi-half) (js/Math.log 0.5)) ;; N(1.0; 0.5, 0.5)
                  (* 0.5 1.0)) ;; z=(1-0.5)/0.5=1, z²=1
          analytical-score (+ lp-x lp-y)]
      ;; Component X: model accepts [] arguments (no exception)
      (t/is (some? trace)
            "X: model should accept [] arguments")
      ;; Component Y: retval = f([], τ) = τ[:y] = 1.0
      (t/is (close? (ev (:retval trace)) 1.0 1e-6)
            (str "Y/f: expected retval=1.0, got " (ev (:retval trace))))
      ;; Component p: score = log p(τ; [])
      (t/is (close? (ev (:score trace)) analytical-score 1e-4)
            (str "p: expected score=" analytical-score
                 ", got " (ev (:score trace))))
      ;; Cross-check: assess agrees with generate
      (let [assess-w (ev (:weight (p/assess model [] tau)))]
        (t/is (close? assess-w (ev (:score trace)) 1e-6)
              (str "p cross-check: assess=" assess-w
                   " vs generate.score=" (ev (:score trace)))))
      ;; Weight = score (fully constrained, proposal = prior)
      (t/is (close? (ev (:weight result)) (ev (:score trace)) 1e-6)
            "weight should equal score for fully constrained generate"))))

(t/deftest law:gen-denotation-model-C
  (t/testing "⟦gen⟧ = (X, Y, p, f): all four components for three-site chain"
    ;; Model C: a ~ N(0,1), b|a ~ N(a,1), c|a,b ~ N(a+b, 0.5)
    ;; f([], τ) = τ[:c]  (last trace expression)
    ;;
    ;; At τ = {:a 1.0, :b 0.5, :c 2.0}:
    ;;
    ;; log N(1.0; 0, 1) = -0.5*log(2π) - 0.5*1.0 = -1.41893853
    ;; log N(0.5; 1.0, 1) = -0.5*log(2π) - 0.5*(0.5)² = -0.91894 - 0.125 = -1.04393853
    ;; log N(2.0; 1.5, 0.5) = -0.5*log(2π) - log(0.5) - 0.5*((2-1.5)/0.5)²
    ;;                       = -0.91894 + 0.69315 - 0.5 = -0.72579135
    ;; Total = -1.41894 + -1.04394 + -0.72579 = -3.18867
    ;; Tolerance: 1e-4 (float32 accumulation across 3 log-prob terms)
    (let [model (:model three-chain)
          tau (cm/choicemap :a (mx/scalar 1.0) :b (mx/scalar 0.5) :c (mx/scalar 2.0))
          result (p/generate model [] tau)
          trace (:trace result)
          log-2pi-half (* 0.5 (js/Math.log (* 2.0 js/Math.PI)))
          lp-a (- (- log-2pi-half) (* 0.5 1.0)) ;; N(1.0; 0, 1)
          lp-b (- (- log-2pi-half) (* 0.5 0.25)) ;; N(0.5; 1.0, 1): z=-0.5
          lp-c (- (- (- log-2pi-half) (js/Math.log 0.5)) ;; N(2.0; 1.5, 0.5): z=1.0
                  (* 0.5 1.0))
          analytical-score (+ lp-a lp-b lp-c)]
      ;; X: accepts [] arguments
      (t/is (some? trace) "X: model should accept [] arguments")
      ;; Y/f: retval = τ[:c] = 2.0
      (t/is (close? (ev (:retval trace)) 2.0 1e-6)
            (str "Y/f: expected retval=2.0, got " (ev (:retval trace))))
      ;; p: score matches analytical joint density
      (t/is (close? (ev (:score trace)) analytical-score 1e-4)
            (str "p: expected score=" analytical-score
                 ", got " (ev (:score trace))))
      ;; Cross-check via assess
      (let [assess-w (ev (:weight (p/assess model [] tau)))]
        (t/is (close? assess-w (ev (:score trace)) 1e-6)
              (str "p cross-check: assess=" assess-w
                   " vs generate.score=" (ev (:score trace))))))))

(t/deftest law:gen-denotation-with-args
  (t/testing "⟦gen⟧ = (X, Y, p, f): model with arguments"
    ;; linear-regression: (gen [x-val] ...)
    ;; X = [R] (one scalar argument)
    ;; At x-val=2.0, τ = {:slope 1.0, :intercept 0.5, :y 3.0}:
    ;;   slope ~ N(0, 5):     log N(1.0; 0, 5) = -0.5*log(2π) - log(5) - 0.5*(1/5)²
    ;;                       = -0.91894 - 1.60944 - 0.02 = -2.54838
    ;;   intercept ~ N(0, 5): log N(0.5; 0, 5) = -0.91894 - 1.60944 - 0.5*(0.1)²
    ;;                       = -0.91894 - 1.60944 - 0.005 = -2.53338
    ;;   y ~ N(slope*x + intercept, 1) = N(2.5, 1):
    ;;     log N(3.0; 2.5, 1) = -0.91894 - 0.5*(0.5)² = -0.91894 - 0.125 = -1.04394
    ;; Total = -2.54838 + -2.53338 + -1.04394 = -6.12569
    ;; Tolerance: 1e-4 (float32 across 3 terms)
    (let [model (:model linear-regression)
          x-val (mx/scalar 2.0)
          tau (cm/choicemap :slope (mx/scalar 1.0)
                            :intercept (mx/scalar 0.5)
                            :y (mx/scalar 3.0))
          result (p/generate model [x-val] tau)
          trace (:trace result)
          log-2pi-half (* 0.5 (js/Math.log (* 2.0 js/Math.PI)))
          lp-slope (- (- (- log-2pi-half) (js/Math.log 5.0))
                      (* 0.5 (/ (* 1.0 1.0) (* 5.0 5.0))))
          lp-intercept (- (- (- log-2pi-half) (js/Math.log 5.0))
                          (* 0.5 (/ (* 0.5 0.5) (* 5.0 5.0))))
          lp-y (- (- log-2pi-half)
                  (* 0.5 (* 0.5 0.5)))
          analytical-score (+ lp-slope lp-intercept lp-y)]
      ;; X: model accepts [x-val]
      (t/is (some? trace)
            "X: model should accept [x-val] arguments")
      ;; p: score matches analytical
      (t/is (close? (ev (:score trace)) analytical-score 1e-4)
            (str "p: expected score=" analytical-score
                 ", got " (ev (:score trace)))))))

(t/run-tests)
