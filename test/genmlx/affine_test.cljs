(ns genmlx.affine-test
  "Tests for affine expression analysis.
   Gate 3: Correct coefficient extraction for 10+ expression patterns.
   Gate 4: Auto-Kalman correctness (deferred until WP-2 auto-wiring lands)."
  (:require [genmlx.affine :as aff]
            [genmlx.schema :as schema]
            [genmlx.conjugacy :as conj]))

;; =========================================================================
;; Test helpers
;; =========================================================================

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn assert-true [desc pred]
  (if pred
    (do (swap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " desc)))))

(defn assert-equal [desc expected actual]
  (if (= expected actual)
    (do (swap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " desc " — expected " (pr-str expected) " got " (pr-str actual))))))

;; =========================================================================
;; Tests: analyze-affine — base cases
;; =========================================================================

(println "\n== analyze-affine: base cases ==")

(let [r (aff/analyze-affine 'mu 'mu {})]
  (assert-true "identity: affine" (:affine? r))
  (assert-true "identity: has-target" (:has-target? r))
  (assert-equal "identity: coefficient" 1 (:coefficient r))
  (assert-equal "identity: offset" 0 (:offset r)))

(let [r (aff/analyze-affine 42 'mu {})]
  (assert-true "number literal: affine" (:affine? r))
  (assert-true "number literal: no target" (not (:has-target? r)))
  (assert-equal "number literal: coefficient" 0 (:coefficient r))
  (assert-equal "number literal: offset" 42 (:offset r)))

(let [r (aff/analyze-affine 'sigma 'mu {})]
  (assert-true "other symbol: affine (constant)" (:affine? r))
  (assert-true "other symbol: no target" (not (:has-target? r)))
  (assert-equal "other symbol: offset is symbol" 'sigma (:offset r)))

(let [r (aff/analyze-affine 'sigma 'mu {'sigma #{:mu}})]
  (assert-true "dependent symbol via env: nonlinear (conservative)"
    (not (:affine? r))))

;; =========================================================================
;; Tests: analyze-affine — arithmetic operations
;; =========================================================================

(println "\n== analyze-affine: arithmetic ==")

;; Addition
(let [r (aff/analyze-affine '(mx/add mu 3) 'mu {})]
  (assert-true "add constant: affine" (:affine? r))
  (assert-equal "add constant: coefficient" 1 (:coefficient r))
  (assert-equal "add constant: offset" 3 (:offset r)))

(let [r (aff/analyze-affine '(mx/add 5 mu) 'mu {})]
  (assert-true "add constant (reversed): affine" (:affine? r))
  (assert-equal "add constant (reversed): coefficient" 1 (:coefficient r))
  (assert-equal "add constant (reversed): offset" 5 (:offset r)))

;; Multiplication
(let [r (aff/analyze-affine '(mx/multiply 2 mu) 'mu {})]
  (assert-true "multiply constant: affine" (:affine? r))
  (assert-equal "multiply constant: coefficient" 2 (:coefficient r))
  (assert-equal "multiply constant: offset" 0 (:offset r)))

(let [r (aff/analyze-affine '(mx/multiply mu 3) 'mu {})]
  (assert-true "multiply constant (reversed): affine" (:affine? r))
  (assert-equal "multiply constant (reversed): coefficient" 3 (:coefficient r))
  (assert-equal "multiply constant (reversed): offset" 0 (:offset r)))

(let [r (aff/analyze-affine '(mx/multiply mu mu) 'mu {})]
  (assert-true "multiply self: nonlinear (quadratic)" (not (:affine? r))))

;; Variable coefficient
(let [r (aff/analyze-affine '(mx/multiply slope mu) 'mu {})]
  (assert-true "multiply by symbol: affine" (:affine? r))
  (assert-equal "multiply by symbol: coefficient" 'slope (:coefficient r))
  (assert-equal "multiply by symbol: offset" 0 (:offset r)))

;; Subtraction
(let [r (aff/analyze-affine '(mx/subtract mu 5) 'mu {})]
  (assert-true "subtract constant: affine" (:affine? r))
  (assert-equal "subtract constant: coefficient" 1 (:coefficient r)))

;; Negation
(let [r (aff/analyze-affine '(mx/negate mu) 'mu {})]
  (assert-true "negate: affine" (:affine? r))
  (assert-equal "negate: coefficient" -1 (:coefficient r))
  (assert-equal "negate: offset" 0 (:offset r)))

;; Division by constant
(let [r (aff/analyze-affine '(mx/divide mu 2) 'mu {})]
  (assert-true "divide by constant: affine" (:affine? r))
  (assert-true "divide by constant: has target" (:has-target? r)))

;; Division BY target — nonlinear
(let [r (aff/analyze-affine '(mx/divide 1 mu) 'mu {})]
  (assert-true "divide by target: nonlinear" (not (:affine? r))))

;; =========================================================================
;; Tests: analyze-affine — composite expressions
;; =========================================================================

(println "\n== analyze-affine: composite ==")

;; Full affine: slope * mu + intercept
(let [r (aff/analyze-affine '(mx/add (mx/multiply slope mu) intercept) 'mu {})]
  (assert-true "full affine: affine" (:affine? r))
  (assert-equal "full affine: coefficient" 'slope (:coefficient r))
  (assert-equal "full affine: offset" 'intercept (:offset r)))

;; Nested: 2 * (mu + 3) => coefficient=2, offset=(mx/multiply 2 3)
(let [r (aff/analyze-affine '(mx/multiply 2 (mx/add mu 3)) 'mu {})]
  (assert-true "nested 2*(mu+3): affine" (:affine? r))
  (assert-equal "nested 2*(mu+3): coefficient" 2 (:coefficient r))
  ;; offset is (mx/multiply 2 3) as a source form
  (assert-true "nested 2*(mu+3): offset is multiply form"
    (and (seq? (:offset r))
         (= 'mx/multiply (first (:offset r))))))

;; Nonlinear: exp(mu)
(let [r (aff/analyze-affine '(mx/exp mu) 'mu {})]
  (assert-true "exp: nonlinear" (not (:affine? r))))

;; Nonlinear: sin(mu)
(let [r (aff/analyze-affine '(mx/sin mu) 'mu {})]
  (assert-true "sin: nonlinear" (not (:affine? r))))

;; Nonlinear: log(mu)
(let [r (aff/analyze-affine '(mx/log mu) 'mu {})]
  (assert-true "log: nonlinear" (not (:affine? r))))

;; Constants only: 2 + 3
(let [r (aff/analyze-affine '(mx/add 2 3) 'mu {})]
  (assert-true "constants: affine (no target)" (:affine? r))
  (assert-true "constants: no target" (not (:has-target? r))))

;; mx/scalar passthrough
(let [r (aff/analyze-affine '(mx/scalar mu) 'mu {})]
  (assert-true "scalar passthrough: affine" (:affine? r))
  (assert-equal "scalar passthrough: coefficient" 1 (:coefficient r)))

;; =========================================================================
;; Tests: classify-affine-dependency with real schemas
;; =========================================================================

(println "\n== classify-affine-dependency ==")

;; Direct: y ~ N(mu, 1)
(let [s (schema/extract-schema '([x]
          (let [mu (trace :mu (dist/gaussian 0 10))]
            (trace :y (dist/gaussian mu 1)))))
      y-site (second (:trace-sites s))]
  (assert-equal "direct dependency"
    {:type :direct}
    (aff/classify-affine-dependency :mu y-site 0)))

;; Affine: y ~ N(0.9 * z, 0.5)
(let [s (schema/extract-schema '([x]
          (let [z (trace :z (dist/gaussian 0 1))]
            (trace :y (dist/gaussian (mx/multiply 0.9 z) 0.5)))))
      y-site (second (:trace-sites s))]
  (let [r (aff/classify-affine-dependency :z y-site 0)]
    (assert-equal "affine multiply: type" :affine (:type r))
    (assert-equal "affine multiply: coefficient" 0.9 (:coefficient r))
    (assert-equal "affine multiply: offset" 0 (:offset r))))

;; Affine: y ~ N(2.5 * mu + 3, 1)
(let [s (schema/extract-schema '([x]
          (let [mu (trace :mu (dist/gaussian 0 10))]
            (trace :y (dist/gaussian (mx/add (mx/multiply 2.5 mu) 3) 1)))))
      y-site (second (:trace-sites s))]
  (let [r (aff/classify-affine-dependency :mu y-site 0)]
    (assert-equal "full affine: type" :affine (:type r))
    (assert-equal "full affine: coefficient" 2.5 (:coefficient r))
    (assert-equal "full affine: offset" 3 (:offset r))))

;; Nonlinear: y ~ N(exp(mu), 1)
(let [s (schema/extract-schema '([x]
          (let [mu (trace :mu (dist/gaussian 0 10))]
            (trace :y (dist/gaussian (mx/exp mu) 1)))))
      y-site (second (:trace-sites s))]
  (assert-equal "nonlinear dependency"
    {:type :nonlinear}
    (aff/classify-affine-dependency :mu y-site 0)))

;; Temporal chain: z1 ~ N(0.9 * z0, 0.5)
(let [s (schema/extract-schema '([x]
          (let [z0 (trace :z0 (dist/gaussian 0 1))
                z1 (trace :z1 (dist/gaussian (mx/multiply 0.9 z0) 0.5))]
            z1)))
      z1-site (second (:trace-sites s))]
  (let [r (aff/classify-affine-dependency :z0 z1-site 0)]
    (assert-equal "temporal chain: type" :affine (:type r))
    (assert-equal "temporal chain: coefficient" 0.9 (:coefficient r))))

;; No dependency: y ~ N(0, 1) — prior doesn't appear in args
(let [s (schema/extract-schema '([x]
          (let [mu (trace :mu (dist/gaussian 0 10))]
            (trace :y (dist/gaussian 0 1)))))
      y-site (second (:trace-sites s))]
  (assert-equal "no dependency"
    {:type :nonlinear}
    (aff/classify-affine-dependency :mu y-site 0)))

;; Bernoulli with beta prior — natural param is position 0
(let [s (schema/extract-schema '([x]
          (let [p (trace :p (dist/beta-dist 1 1))]
            (trace :y (dist/bernoulli p)))))
      y-site (second (:trace-sites s))]
  (assert-equal "beta-bernoulli direct"
    {:type :direct}
    (aff/classify-affine-dependency :p y-site 0)))

;; =========================================================================
;; Tests: multi-dependency (target has multiple deps)
;; =========================================================================

(println "\n== multi-dependency edge cases ==")

;; Two trace deps: y ~ N(mu + sigma, 1) — affine in mu, but sigma is also a trace
;; When analyzing for :mu, sigma should be treated as constant
(let [s (schema/extract-schema '([x]
          (let [mu (trace :mu (dist/gaussian 0 10))
                sigma (trace :sigma (dist/gaussian 1 1))]
            (trace :y (dist/gaussian (mx/add mu sigma) 1)))))
      y-site (nth (:trace-sites s) 2)]
  ;; sigma is in env as depending on :sigma trace addr
  ;; When analyzing for :mu, sigma doesn't resolve to target 'mu
  ;; But it IS in the env with deps. Need to check behavior.
  (let [r (aff/classify-affine-dependency :mu y-site 0)]
    (assert-equal "two deps, affine in target: type" :affine (:type r))
    (assert-equal "two deps, affine in target: coefficient" 1 (:coefficient r))))

;; =========================================================================
;; Tests: detect-kalman-chains
;; =========================================================================

(println "\n== detect-kalman-chains ==")

;; 3-step temporal model: z0 -> z1 -> z2 with obs y0, y1, y2
(let [s (schema/extract-schema '([obs]
          (let [z0 (trace :z0 (dist/gaussian 0 1))
                z1 (trace :z1 (dist/gaussian (mx/multiply 0.9 z0) 0.5))
                z2 (trace :z2 (dist/gaussian (mx/multiply 0.9 z1) 0.5))]
            (trace :y0 (dist/gaussian z0 0.3))
            (trace :y1 (dist/gaussian z1 0.3))
            (trace :y2 (dist/gaussian z2 0.3))
            z2)))
      pairs (conj/detect-conjugate-pairs s)
      chains (aff/detect-kalman-chains pairs)]
  (assert-equal "3-step: 1 chain" 1 (count chains))
  (let [c (first chains)]
    (assert-equal "3-step: 3 latent nodes" [:z0 :z1 :z2] (:latent-addrs c))
    (assert-equal "3-step: 3 obs nodes" [:y0 :y1 :y2] (:obs-addrs c))
    (assert-equal "3-step: 3 steps" 3 (count (:steps c)))
    (assert-equal "3-step: step 0 latent" :z0 (:latent (nth (:steps c) 0)))
    (assert-equal "3-step: step 0 obs" [:y0] (:observations (nth (:steps c) 0)))
    (assert-equal "3-step: step 0 transition coeff" 0.9
      (get-in (nth (:steps c) 0) [:transition :coefficient]))
    (assert-equal "3-step: step 0 noise" 0.5
      (:noise-std (nth (:steps c) 0)))
    (assert-equal "3-step: last step no transition" nil
      (:transition (nth (:steps c) 2)))))

;; Single NN pair (not a chain — only 1 latent)
(let [s (schema/extract-schema '([x]
          (let [mu (trace :mu (dist/gaussian 0 10))]
            (trace :y (dist/gaussian mu 1)))))
      pairs (conj/detect-conjugate-pairs s)
      chains (aff/detect-kalman-chains pairs)]
  (assert-equal "single pair: no chain (need >= 2 latents)" 0 (count chains)))

;; No pairs at all
(let [chains (aff/detect-kalman-chains [])]
  (assert-true "no pairs: nil or empty" (or (nil? chains) (empty? chains))))

;; Chain with direct (identity) transitions
(let [s (schema/extract-schema '([obs]
          (let [z0 (trace :z0 (dist/gaussian 0 1))
                z1 (trace :z1 (dist/gaussian z0 0.5))
                z2 (trace :z2 (dist/gaussian z1 0.5))]
            (trace :y0 (dist/gaussian z0 0.3))
            (trace :y1 (dist/gaussian z1 0.3))
            (trace :y2 (dist/gaussian z2 0.3))
            z2)))
      pairs (conj/detect-conjugate-pairs s)
      chains (aff/detect-kalman-chains pairs)]
  (assert-equal "direct chain: 1 chain" 1 (count chains))
  (assert-equal "direct chain: transition is direct"
    :direct (get-in (first chains) [:steps 0 :transition :type])))

;; Chain with no separate observations (latent-only: z0 -> z1 -> z2)
;; z2 is terminal (not a prior), so chain is [z0, z1] with z2 as "obs" of z1
(let [s (schema/extract-schema '([x]
          (let [z0 (trace :z0 (dist/gaussian 0 1))
                z1 (trace :z1 (dist/gaussian (mx/multiply 0.8 z0) 0.3))
                z2 (trace :z2 (dist/gaussian (mx/multiply 0.8 z1) 0.3))]
            z2)))
      pairs (conj/detect-conjugate-pairs s)
      chains (aff/detect-kalman-chains pairs)]
  (assert-equal "latent-only chain: 1 chain" 1 (count chains))
  ;; z2 is terminal — treated as obs of z1 in the chain
  (assert-equal "latent-only chain: latents [z0 z1]" [:z0 :z1] (:latent-addrs (first chains)))
  (assert-equal "latent-only chain: z2 is obs" [:z2] (:obs-addrs (first chains))))

;; Mixed: chain + independent conjugate pair
(let [s (schema/extract-schema '([x]
          (let [z0 (trace :z0 (dist/gaussian 0 1))
                z1 (trace :z1 (dist/gaussian (mx/multiply 0.9 z0) 0.5))
                mu (trace :mu (dist/gaussian 0 10))]
            (trace :y0 (dist/gaussian z0 0.3))
            (trace :y1 (dist/gaussian z1 0.3))
            (trace :obs (dist/gaussian mu 1))
            z1)))
      pairs (conj/detect-conjugate-pairs s)
      chains (aff/detect-kalman-chains pairs)]
  (assert-equal "mixed: 1 chain (mu-obs is just conjugate, not a chain)" 1 (count chains))
  (assert-equal "mixed: chain is z0->z1" [:z0 :z1] (:latent-addrs (first chains))))

;; =========================================================================
;; Gate 4: Auto-Kalman handler correctness
;; =========================================================================

(println "\n-- Gate 4: Auto-Kalman handler tests --")

(require '[genmlx.mlx :as mx]
         '[genmlx.choicemap :as cm]
         '[genmlx.dist :as dist]
         '[genmlx.inference.auto-analytical :as auto])

(defn assert-close [desc expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass-count inc)
          (println (str "  PASS: " desc)))
      (do (swap! fail-count inc)
          (println (str "  FAIL: " desc " — expected " expected " got " actual " (diff " diff ")"))))))

;; --- Test 1: 3-step chain, batch execution order (z0,z1,z2,y0,y1,y2) ---
(println "\n  3-step Kalman chain (batch order)")
(let [chain {:steps [{:latent :z0
                      :observations [:y0]
                      :obs-dep-types [{:type :direct}]
                      :transition {:type :direct}
                      :noise-std (mx/scalar 0.5)
                      :next-latent :z1}
                     {:latent :z1
                      :observations [:y1]
                      :obs-dep-types [{:type :direct}]
                      :transition {:type :direct}
                      :noise-std (mx/scalar 0.5)
                      :next-latent :z2}
                     {:latent :z2
                      :observations [:y2]
                      :obs-dep-types [{:type :direct}]
                      :transition nil
                      :noise-std nil
                      :next-latent nil}]
             :latent-addrs [:z0 :z1 :z2]
             :obs-addrs [:y0 :y1 :y2]}
      handlers (auto/make-auto-kalman-handlers chain)
      constraints (-> (cm/choicemap)
                      (cm/set-value :y0 (mx/scalar 1.0))
                      (cm/set-value :y1 (mx/scalar 0.5))
                      (cm/set-value :y2 (mx/scalar -0.3)))
      state {:choices (cm/choicemap)
             :constraints constraints
             :score (mx/scalar 0.0)
             :weight (mx/scalar 0.0)
             :auto-kalman-beliefs {}
             :auto-kalman-noise-vars {}}
      z0-d (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0))
      z1-d (dist/gaussian (mx/scalar 0.0) (mx/scalar 0.5))
      z2-d (dist/gaussian (mx/scalar 0.0) (mx/scalar 0.5))
      y0-d (dist/gaussian (mx/scalar 0.0) (mx/scalar 0.3))
      y1-d (dist/gaussian (mx/scalar 0.0) (mx/scalar 0.3))
      y2-d (dist/gaussian (mx/scalar 0.0) (mx/scalar 0.3))
      ;; Batch order
      [_ s1] ((get handlers :z0) state :z0 z0-d)
      [_ s2] ((get handlers :z1) s1 :z1 z1-d)
      [_ s3] ((get handlers :z2) s2 :z2 z2-d)
      [_ s4] ((get handlers :y0) s3 :y0 y0-d)
      [_ s5] ((get handlers :y1) s4 :y1 y1-d)
      [_ s6] ((get handlers :y2) s5 :y2 y2-d)
      score (mx/item (:score s6))
      weight (mx/item (:weight s6))
      z2-mean (mx/item (:mean (get-in s6 [:auto-kalman-beliefs :z2])))
      z2-var (mx/item (:var (get-in s6 [:auto-kalman-beliefs :z2])))]
  (assert-true "handlers built for all 6 addresses" (= 6 (count handlers)))
  (assert-close "batch: score matches manual Kalman" -3.5510 score 0.001)
  (assert-close "batch: weight = score" score weight 1e-8)
  (assert-close "batch: z2 posterior mean" -0.1053 z2-mean 0.001)
  (assert-close "batch: z2 posterior var" 0.0703 z2-var 0.001)

  ;; Interleaved order (z0,y0,z1,y1,z2,y2)
  (println "\n  3-step Kalman chain (interleaved order)")
  (let [[_ t1] ((get handlers :z0) state :z0 z0-d)
        [_ t2] ((get handlers :y0) t1 :y0 y0-d)
        [_ t3] ((get handlers :z1) t2 :z1 z1-d)
        [_ t4] ((get handlers :y1) t3 :y1 y1-d)
        [_ t5] ((get handlers :z2) t4 :z2 z2-d)
        [_ t6] ((get handlers :y2) t5 :y2 y2-d)
        iscore (mx/item (:score t6))]
    (assert-close "interleaved: score matches batch" score iscore 1e-6)
    (assert-close "interleaved: z2 posterior mean"
                  z2-mean (mx/item (:mean (get-in t6 [:auto-kalman-beliefs :z2]))) 1e-6)
    (assert-close "interleaved: z2 posterior var"
                  z2-var (mx/item (:var (get-in t6 [:auto-kalman-beliefs :z2]))) 1e-6)))

;; --- Test 2: 2-step chain with affine transition ---
(println "\n  2-step chain with affine transition (coeff=0.9)")
(let [chain {:steps [{:latent :x0
                      :observations [:obs0]
                      :obs-dep-types [{:type :direct}]
                      :transition {:type :affine :coefficient 0.9 :offset 0}
                      :noise-std (mx/scalar 0.3)
                      :next-latent :x1}
                     {:latent :x1
                      :observations [:obs1]
                      :obs-dep-types [{:type :direct}]
                      :transition nil
                      :noise-std nil
                      :next-latent nil}]
             :latent-addrs [:x0 :x1]
             :obs-addrs [:obs0 :obs1]}
      handlers (auto/make-auto-kalman-handlers chain)
      constraints (-> (cm/choicemap)
                      (cm/set-value :obs0 (mx/scalar 2.0))
                      (cm/set-value :obs1 (mx/scalar 1.5)))
      state {:choices (cm/choicemap)
             :constraints constraints
             :score (mx/scalar 0.0)
             :weight (mx/scalar 0.0)
             :auto-kalman-beliefs {}
             :auto-kalman-noise-vars {}}
      x0-d (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0))
      x1-d (dist/gaussian (mx/scalar 0.0) (mx/scalar 0.3))
      o0-d (dist/gaussian (mx/scalar 0.0) (mx/scalar 0.5))
      o1-d (dist/gaussian (mx/scalar 0.0) (mx/scalar 0.5))
      ;; Batch order
      [_ s1] ((get handlers :x0) state :x0 x0-d)
      [_ s2] ((get handlers :x1) s1 :x1 x1-d)
      [_ s3] ((get handlers :obs0) s2 :obs0 o0-d)
      [_ s4] ((get handlers :obs1) s3 :obs1 o1-d)
      score (mx/item (:score s4))]
  (assert-true "affine chain: handlers for 4 addresses" (= 4 (count handlers)))
  ;; Manual Kalman: x0 prior N(0,1), obs0=2 with obs_var=0.25
  ;; x0 update: S=1.25, K=0.8, mean=1.6, var=0.2
  ;; x1 predict: mean=0.9*1.6=1.44, var=0.81*0.2+0.09=0.252
  ;; x1 update: S=0.252+0.25=0.502, innov=1.5-1.44=0.06
  ;; Total LL should be finite and negative
  (assert-true "affine chain: score is finite" (js/isFinite score))
  (assert-true "affine chain: score is negative" (< score 0))
  ;; Check x0 posterior after obs0
  (let [x0-belief (get-in s3 [:auto-kalman-beliefs :x0])
        x0-mean (mx/item (:mean x0-belief))
        x0-var (mx/item (:var x0-belief))]
    (assert-close "affine chain: x0 posterior mean" 1.6 x0-mean 0.01)
    (assert-close "affine chain: x0 posterior var" 0.2 x0-var 0.01))
  ;; Verify batch vs interleaved
  (let [[_ t1] ((get handlers :x0) state :x0 x0-d)
        [_ t2] ((get handlers :obs0) t1 :obs0 o0-d)
        [_ t3] ((get handlers :x1) t2 :x1 x1-d)
        [_ t4] ((get handlers :obs1) t3 :obs1 o1-d)]
    (assert-close "affine chain: interleaved matches batch"
                  score (mx/item (:score t4)) 1e-6)))

;; --- Test 3: Unconstrained obs falls through ---
(println "\n  Unconstrained observation fallthrough")
(let [chain {:steps [{:latent :z0
                      :observations [:y0]
                      :obs-dep-types [{:type :direct}]
                      :transition nil
                      :noise-std nil
                      :next-latent nil}]
             :latent-addrs [:z0]
             :obs-addrs [:y0]}
      handlers (auto/make-auto-kalman-handlers chain)
      ;; No constraints — obs handler should return nil
      state {:choices (cm/choicemap)
             :constraints (cm/choicemap)
             :score (mx/scalar 0.0)
             :weight (mx/scalar 0.0)
             :auto-kalman-beliefs {}
             :auto-kalman-noise-vars {}}
      z0-d (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0))
      y0-d (dist/gaussian (mx/scalar 0.0) (mx/scalar 0.5))
      [_ s1] ((get handlers :z0) state :z0 z0-d)
      result ((get handlers :y0) s1 :y0 y0-d)]
  (assert-true "unconstrained obs returns nil (fallthrough)" (nil? result)))

;; --- Test 4: Single-step chain (just prior + obs, no chain edge) ---
;; detect-kalman-chains requires >= 2 latent nodes, so this tests the handler
;; factory directly for a 1-step "chain" (degenerate case)
(println "\n  Single-step degenerate chain")
(let [chain {:steps [{:latent :z
                      :observations [:y]
                      :obs-dep-types [{:type :direct}]
                      :transition nil
                      :noise-std nil
                      :next-latent nil}]
             :latent-addrs [:z]
             :obs-addrs [:y]}
      handlers (auto/make-auto-kalman-handlers chain)
      constraints (-> (cm/choicemap) (cm/set-value :y (mx/scalar 3.0)))
      state {:choices (cm/choicemap)
             :constraints constraints
             :score (mx/scalar 0.0)
             :weight (mx/scalar 0.0)
             :auto-kalman-beliefs {}
             :auto-kalman-noise-vars {}}
      z-d (dist/gaussian (mx/scalar 0.0) (mx/scalar 2.0))
      y-d (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0))
      [_ s1] ((get handlers :z) state :z z-d)
      [_ s2] ((get handlers :y) s1 :y y-d)
      ;; z prior: N(0,4), y|z ~ N(z,1)
      ;; Marginal: y ~ N(0, 5)
      ;; ll = -0.5*(log(2pi) + log(5) + 9/5)
      expected-ll (* -0.5 (+ 1.8378770664093453 (js/Math.log 5.0) (/ 9.0 5.0)))]
  (assert-close "degenerate chain: marginal LL" expected-ll (mx/item (:score s2)) 0.001)
  (assert-close "degenerate chain: posterior mean" 2.4 (mx/item (:mean (get-in s2 [:auto-kalman-beliefs :z]))) 0.01)
  (assert-close "degenerate chain: posterior var" 0.8 (mx/item (:var (get-in s2 [:auto-kalman-beliefs :z]))) 0.01))

;; --- Test 5: build-auto-kalman-handlers merges multiple chains ---
(println "\n  build-auto-kalman-handlers")
(let [chain1 {:steps [{:latent :a0 :observations [:b0] :obs-dep-types [{:type :direct}]
                       :transition {:type :direct} :noise-std nil :next-latent :a1}
                      {:latent :a1 :observations [:b1] :obs-dep-types [{:type :direct}]
                       :transition nil :noise-std nil :next-latent nil}]
              :latent-addrs [:a0 :a1] :obs-addrs [:b0 :b1]}
      chain2 {:steps [{:latent :c0 :observations [:d0] :obs-dep-types [{:type :direct}]
                       :transition {:type :direct} :noise-std nil :next-latent :c1}
                      {:latent :c1 :observations [:d1] :obs-dep-types [{:type :direct}]
                       :transition nil :noise-std nil :next-latent nil}]
              :latent-addrs [:c0 :c1] :obs-addrs [:d0 :d1]}
      handlers (auto/build-auto-kalman-handlers [chain1 chain2])]
  (assert-equal "multi-chain: 8 handlers" 8 (count handlers))
  (assert-true "multi-chain: has :a0" (contains? handlers :a0))
  (assert-true "multi-chain: has :c1" (contains? handlers :c1))
  (assert-true "multi-chain: has :d0" (contains? handlers :d0)))

;; =========================================================================
;; Summary
;; =========================================================================

(println (str "\n== RESULTS: " @pass-count " passed, " @fail-count " failed =="))
