(ns genmlx.affine-test
  "Tests for affine expression analysis.
   Correct coefficient extraction for 10+ expression patterns.
   Auto-Kalman correctness."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.choicemap :as cm]
            [genmlx.dist :as dist]
            [genmlx.affine :as aff]
            [genmlx.schema :as schema]
            [genmlx.conjugacy :as conj]
            [genmlx.inference.auto-analytical :as auto]))

;; =========================================================================
;; Tests: analyze-affine -- base cases
;; =========================================================================

(deftest analyze-affine-base-cases
  (testing "identity"
    (let [r (aff/analyze-affine 'mu 'mu {})]
      (is (:affine? r) "identity: affine")
      (is (:has-target? r) "identity: has-target")
      (is (= 1 (:coefficient r)) "identity: coefficient")
      (is (= 0 (:offset r)) "identity: offset")))

  (testing "number literal"
    (let [r (aff/analyze-affine 42 'mu {})]
      (is (:affine? r) "number literal: affine")
      (is (not (:has-target? r)) "number literal: no target")
      (is (= 0 (:coefficient r)) "number literal: coefficient")
      (is (= 42 (:offset r)) "number literal: offset")))

  (testing "other symbol"
    (let [r (aff/analyze-affine 'sigma 'mu {})]
      (is (:affine? r) "other symbol: affine (constant)")
      (is (not (:has-target? r)) "other symbol: no target")
      (is (= 'sigma (:offset r)) "other symbol: offset is symbol")))

  (testing "dependent symbol via env"
    (let [r (aff/analyze-affine 'sigma 'mu {'sigma #{:mu}})]
      (is (not (:affine? r)) "dependent symbol via env: nonlinear (conservative)"))))

;; =========================================================================
;; Tests: analyze-affine -- arithmetic operations
;; =========================================================================

(deftest analyze-affine-arithmetic
  (testing "add constant"
    (let [r (aff/analyze-affine '(mx/add mu 3) 'mu {})]
      (is (:affine? r) "add constant: affine")
      (is (= 1 (:coefficient r)) "add constant: coefficient")
      (is (= 3 (:offset r)) "add constant: offset")))

  (testing "add constant reversed"
    (let [r (aff/analyze-affine '(mx/add 5 mu) 'mu {})]
      (is (:affine? r) "add constant (reversed): affine")
      (is (= 1 (:coefficient r)) "add constant (reversed): coefficient")
      (is (= 5 (:offset r)) "add constant (reversed): offset")))

  (testing "multiply constant"
    (let [r (aff/analyze-affine '(mx/multiply 2 mu) 'mu {})]
      (is (:affine? r) "multiply constant: affine")
      (is (= 2 (:coefficient r)) "multiply constant: coefficient")
      (is (= 0 (:offset r)) "multiply constant: offset")))

  (testing "multiply constant reversed"
    (let [r (aff/analyze-affine '(mx/multiply mu 3) 'mu {})]
      (is (:affine? r) "multiply constant (reversed): affine")
      (is (= 3 (:coefficient r)) "multiply constant (reversed): coefficient")
      (is (= 0 (:offset r)) "multiply constant (reversed): offset")))

  (testing "multiply self"
    (let [r (aff/analyze-affine '(mx/multiply mu mu) 'mu {})]
      (is (not (:affine? r)) "multiply self: nonlinear (quadratic)")))

  (testing "multiply by symbol"
    (let [r (aff/analyze-affine '(mx/multiply slope mu) 'mu {})]
      (is (:affine? r) "multiply by symbol: affine")
      (is (= 'slope (:coefficient r)) "multiply by symbol: coefficient")
      (is (= 0 (:offset r)) "multiply by symbol: offset")))

  (testing "subtract constant"
    (let [r (aff/analyze-affine '(mx/subtract mu 5) 'mu {})]
      (is (:affine? r) "subtract constant: affine")
      (is (= 1 (:coefficient r)) "subtract constant: coefficient")))

  (testing "negate"
    (let [r (aff/analyze-affine '(mx/negate mu) 'mu {})]
      (is (:affine? r) "negate: affine")
      (is (= -1 (:coefficient r)) "negate: coefficient")
      (is (= 0 (:offset r)) "negate: offset")))

  (testing "divide by constant"
    (let [r (aff/analyze-affine '(mx/divide mu 2) 'mu {})]
      (is (:affine? r) "divide by constant: affine")
      (is (:has-target? r) "divide by constant: has target")))

  (testing "divide by target"
    (let [r (aff/analyze-affine '(mx/divide 1 mu) 'mu {})]
      (is (not (:affine? r)) "divide by target: nonlinear"))))

;; =========================================================================
;; Tests: analyze-affine -- composite expressions
;; =========================================================================

(deftest analyze-affine-composite
  (testing "full affine: slope * mu + intercept"
    (let [r (aff/analyze-affine '(mx/add (mx/multiply slope mu) intercept) 'mu {})]
      (is (:affine? r) "full affine: affine")
      (is (= 'slope (:coefficient r)) "full affine: coefficient")
      (is (= 'intercept (:offset r)) "full affine: offset")))

  (testing "nested 2*(mu+3)"
    (let [r (aff/analyze-affine '(mx/multiply 2 (mx/add mu 3)) 'mu {})]
      (is (:affine? r) "nested 2*(mu+3): affine")
      (is (= 2 (:coefficient r)) "nested 2*(mu+3): coefficient")
      ;; offset is (mx/multiply 2 3) as a source form
      (is (and (seq? (:offset r))
               (= 'mx/multiply (first (:offset r)))) "nested 2*(mu+3): offset is multiply form")))

  (testing "exp: nonlinear"
    (let [r (aff/analyze-affine '(mx/exp mu) 'mu {})]
      (is (not (:affine? r)) "exp: nonlinear")))

  (testing "sin: nonlinear"
    (let [r (aff/analyze-affine '(mx/sin mu) 'mu {})]
      (is (not (:affine? r)) "sin: nonlinear")))

  (testing "log: nonlinear"
    (let [r (aff/analyze-affine '(mx/log mu) 'mu {})]
      (is (not (:affine? r)) "log: nonlinear")))

  (testing "constants only: 2 + 3"
    (let [r (aff/analyze-affine '(mx/add 2 3) 'mu {})]
      (is (:affine? r) "constants: affine (no target)")
      (is (not (:has-target? r)) "constants: no target")))

  (testing "mx/scalar passthrough"
    (let [r (aff/analyze-affine '(mx/scalar mu) 'mu {})]
      (is (:affine? r) "scalar passthrough: affine")
      (is (= 1 (:coefficient r)) "scalar passthrough: coefficient"))))

;; =========================================================================
;; Tests: classify-affine-dependency with real schemas
;; =========================================================================

(deftest classify-affine-dependency
  (testing "direct dependency: y ~ N(mu, 1)"
    (let [s (schema/extract-schema '([x]
              (let [mu (trace :mu (dist/gaussian 0 10))]
                (trace :y (dist/gaussian mu 1)))))
          y-site (second (:trace-sites s))]
      (is (= {:type :direct} (aff/classify-affine-dependency :mu y-site 0))
          "direct dependency")))

  (testing "affine multiply: y ~ N(0.9 * z, 0.5)"
    (let [s (schema/extract-schema '([x]
              (let [z (trace :z (dist/gaussian 0 1))]
                (trace :y (dist/gaussian (mx/multiply 0.9 z) 0.5)))))
          y-site (second (:trace-sites s))]
      (let [r (aff/classify-affine-dependency :z y-site 0)]
        (is (= :affine (:type r)) "affine multiply: type")
        (is (= 0.9 (:coefficient r)) "affine multiply: coefficient")
        (is (= 0 (:offset r)) "affine multiply: offset"))))

  (testing "full affine: y ~ N(2.5 * mu + 3, 1)"
    (let [s (schema/extract-schema '([x]
              (let [mu (trace :mu (dist/gaussian 0 10))]
                (trace :y (dist/gaussian (mx/add (mx/multiply 2.5 mu) 3) 1)))))
          y-site (second (:trace-sites s))]
      (let [r (aff/classify-affine-dependency :mu y-site 0)]
        (is (= :affine (:type r)) "full affine: type")
        (is (= 2.5 (:coefficient r)) "full affine: coefficient")
        (is (= 3 (:offset r)) "full affine: offset"))))

  (testing "nonlinear dependency: y ~ N(exp(mu), 1)"
    (let [s (schema/extract-schema '([x]
              (let [mu (trace :mu (dist/gaussian 0 10))]
                (trace :y (dist/gaussian (mx/exp mu) 1)))))
          y-site (second (:trace-sites s))]
      (is (= {:type :nonlinear} (aff/classify-affine-dependency :mu y-site 0))
          "nonlinear dependency")))

  (testing "temporal chain: z1 ~ N(0.9 * z0, 0.5)"
    (let [s (schema/extract-schema '([x]
              (let [z0 (trace :z0 (dist/gaussian 0 1))
                    z1 (trace :z1 (dist/gaussian (mx/multiply 0.9 z0) 0.5))]
                z1)))
          z1-site (second (:trace-sites s))]
      (let [r (aff/classify-affine-dependency :z0 z1-site 0)]
        (is (= :affine (:type r)) "temporal chain: type")
        (is (= 0.9 (:coefficient r)) "temporal chain: coefficient"))))

  (testing "no dependency: y ~ N(0, 1)"
    (let [s (schema/extract-schema '([x]
              (let [mu (trace :mu (dist/gaussian 0 10))]
                (trace :y (dist/gaussian 0 1)))))
          y-site (second (:trace-sites s))]
      (is (= {:type :nonlinear} (aff/classify-affine-dependency :mu y-site 0))
          "no dependency")))

  (testing "beta-bernoulli direct"
    (let [s (schema/extract-schema '([x]
              (let [p (trace :p (dist/beta-dist 1 1))]
                (trace :y (dist/bernoulli p)))))
          y-site (second (:trace-sites s))]
      (is (= {:type :direct} (aff/classify-affine-dependency :p y-site 0))
          "beta-bernoulli direct"))))

;; =========================================================================
;; Tests: multi-dependency (target has multiple deps)
;; =========================================================================

(deftest multi-dependency-edge-cases
  (testing "two deps, affine in target"
    (let [s (schema/extract-schema '([x]
              (let [mu (trace :mu (dist/gaussian 0 10))
                    sigma (trace :sigma (dist/gaussian 1 1))]
                (trace :y (dist/gaussian (mx/add mu sigma) 1)))))
          y-site (nth (:trace-sites s) 2)]
      (let [r (aff/classify-affine-dependency :mu y-site 0)]
        (is (= :affine (:type r)) "two deps, affine in target: type")
        (is (= 1 (:coefficient r)) "two deps, affine in target: coefficient")))))

;; =========================================================================
;; Tests: detect-kalman-chains
;; =========================================================================

(deftest detect-kalman-chains-3step
  (testing "3-step temporal model"
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
      (is (= 1 (count chains)) "3-step: 1 chain")
      (let [c (first chains)]
        (is (= [:z0 :z1 :z2] (:latent-addrs c)) "3-step: 3 latent nodes")
        (is (= [:y0 :y1 :y2] (:obs-addrs c)) "3-step: 3 obs nodes")
        (is (= 3 (count (:steps c))) "3-step: 3 steps")
        (is (= :z0 (:latent (nth (:steps c) 0))) "3-step: step 0 latent")
        (is (= [:y0] (:observations (nth (:steps c) 0))) "3-step: step 0 obs")
        (is (= 0.9 (get-in (nth (:steps c) 0) [:transition :coefficient])) "3-step: step 0 transition coeff")
        (is (= 0.5 (:noise-std (nth (:steps c) 0))) "3-step: step 0 noise")
        (is (= nil (:transition (nth (:steps c) 2))) "3-step: last step no transition")))))

(deftest detect-kalman-chains-single-pair
  (testing "Single NN pair (not a chain)"
    (let [s (schema/extract-schema '([x]
              (let [mu (trace :mu (dist/gaussian 0 10))]
                (trace :y (dist/gaussian mu 1)))))
          pairs (conj/detect-conjugate-pairs s)
          chains (aff/detect-kalman-chains pairs)]
      (is (= 0 (count chains)) "single pair: no chain (need >= 2 latents)"))))

(deftest detect-kalman-chains-empty
  (testing "No pairs"
    (let [chains (aff/detect-kalman-chains [])]
      (is (or (nil? chains) (empty? chains)) "no pairs: nil or empty"))))

(deftest detect-kalman-chains-direct
  (testing "Chain with direct (identity) transitions"
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
      (is (= 1 (count chains)) "direct chain: 1 chain")
      (is (= :direct (get-in (first chains) [:steps 0 :transition :type])) "direct chain: transition is direct"))))

(deftest detect-kalman-chains-latent-only
  (testing "Chain with no separate observations"
    (let [s (schema/extract-schema '([x]
              (let [z0 (trace :z0 (dist/gaussian 0 1))
                    z1 (trace :z1 (dist/gaussian (mx/multiply 0.8 z0) 0.3))
                    z2 (trace :z2 (dist/gaussian (mx/multiply 0.8 z1) 0.3))]
                z2)))
          pairs (conj/detect-conjugate-pairs s)
          chains (aff/detect-kalman-chains pairs)]
      (is (= 1 (count chains)) "latent-only chain: 1 chain")
      (is (= [:z0 :z1] (:latent-addrs (first chains))) "latent-only chain: latents [z0 z1]")
      (is (= [:z2] (:obs-addrs (first chains))) "latent-only chain: z2 is obs"))))

(deftest detect-kalman-chains-mixed
  (testing "Mixed: chain + independent conjugate pair"
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
      (is (= 1 (count chains)) "mixed: 1 chain (mu-obs is just conjugate, not a chain)")
      (is (= [:z0 :z1] (:latent-addrs (first chains))) "mixed: chain is z0->z1"))))

;; =========================================================================
;; Gate 4: Auto-Kalman handler correctness
;; =========================================================================

(deftest auto-kalman-3step-batch
  (testing "3-step Kalman chain (batch order)"
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
      (is (= 6 (count handlers)) "handlers built for all 6 addresses")
      (is (h/close? -3.5510 score 0.001) "batch: score matches manual Kalman")
      (is (h/close? score weight 1e-8) "batch: weight = score")
      (is (h/close? -0.1053 z2-mean 0.001) "batch: z2 posterior mean")
      (is (h/close? 0.0703 z2-var 0.001) "batch: z2 posterior var"))))

(deftest auto-kalman-3step-interleaved
  (testing "3-step Kalman chain (interleaved order)"
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
          ;; Batch order for reference score
          [_ s1b] ((get handlers :z0) state :z0 z0-d)
          [_ s2b] ((get handlers :z1) s1b :z1 z1-d)
          [_ s3b] ((get handlers :z2) s2b :z2 z2-d)
          [_ s4b] ((get handlers :y0) s3b :y0 y0-d)
          [_ s5b] ((get handlers :y1) s4b :y1 y1-d)
          [_ s6b] ((get handlers :y2) s5b :y2 y2-d)
          batch-score (mx/item (:score s6b))
          batch-z2-mean (mx/item (:mean (get-in s6b [:auto-kalman-beliefs :z2])))
          batch-z2-var (mx/item (:var (get-in s6b [:auto-kalman-beliefs :z2])))
          ;; Interleaved order
          [_ t1] ((get handlers :z0) state :z0 z0-d)
          [_ t2] ((get handlers :y0) t1 :y0 y0-d)
          [_ t3] ((get handlers :z1) t2 :z1 z1-d)
          [_ t4] ((get handlers :y1) t3 :y1 y1-d)
          [_ t5] ((get handlers :z2) t4 :z2 z2-d)
          [_ t6] ((get handlers :y2) t5 :y2 y2-d)
          iscore (mx/item (:score t6))]
      (is (h/close? batch-score iscore 1e-6) "interleaved: score matches batch")
      (is (h/close? batch-z2-mean (mx/item (:mean (get-in t6 [:auto-kalman-beliefs :z2]))) 1e-6)
          "interleaved: z2 posterior mean")
      (is (h/close? batch-z2-var (mx/item (:var (get-in t6 [:auto-kalman-beliefs :z2]))) 1e-6)
          "interleaved: z2 posterior var"))))

(deftest auto-kalman-2step-affine
  (testing "2-step chain with affine transition (coeff=0.9)"
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
      (is (= 4 (count handlers)) "affine chain: handlers for 4 addresses")
      (is (js/isFinite score) "affine chain: score is finite")
      (is (< score 0) "affine chain: score is negative")
      ;; Check x0 posterior after obs0
      (let [x0-belief (get-in s3 [:auto-kalman-beliefs :x0])
            x0-mean (mx/item (:mean x0-belief))
            x0-var (mx/item (:var x0-belief))]
        (is (h/close? 1.6 x0-mean 0.01) "affine chain: x0 posterior mean")
        (is (h/close? 0.2 x0-var 0.01) "affine chain: x0 posterior var"))
      ;; Verify batch vs interleaved
      (let [[_ t1] ((get handlers :x0) state :x0 x0-d)
            [_ t2] ((get handlers :obs0) t1 :obs0 o0-d)
            [_ t3] ((get handlers :x1) t2 :x1 x1-d)
            [_ t4] ((get handlers :obs1) t3 :obs1 o1-d)]
        (is (h/close? score (mx/item (:score t4)) 1e-6) "affine chain: interleaved matches batch")))))

(deftest auto-kalman-unconstrained-obs
  (testing "Unconstrained observation fallthrough"
    (let [chain {:steps [{:latent :z0
                          :observations [:y0]
                          :obs-dep-types [{:type :direct}]
                          :transition nil
                          :noise-std nil
                          :next-latent nil}]
                 :latent-addrs [:z0]
                 :obs-addrs [:y0]}
          handlers (auto/make-auto-kalman-handlers chain)
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
      (is (nil? result) "unconstrained obs returns nil (fallthrough)"))))

(deftest auto-kalman-degenerate-chain
  (testing "Single-step degenerate chain"
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
      (is (h/close? expected-ll (mx/item (:score s2)) 0.001) "degenerate chain: marginal LL")
      (is (h/close? 2.4 (mx/item (:mean (get-in s2 [:auto-kalman-beliefs :z]))) 0.01)
          "degenerate chain: posterior mean")
      (is (h/close? 0.8 (mx/item (:var (get-in s2 [:auto-kalman-beliefs :z]))) 0.01)
          "degenerate chain: posterior var"))))

(deftest build-auto-kalman-handlers-multi-chain
  (testing "build-auto-kalman-handlers merges multiple chains"
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
      (is (= 8 (count handlers)) "multi-chain: 8 handlers")
      (is (contains? handlers :a0) "multi-chain: has :a0")
      (is (contains? handlers :c1) "multi-chain: has :c1")
      (is (contains? handlers :d0) "multi-chain: has :d0"))))

(cljs.test/run-tests)
