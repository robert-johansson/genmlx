(ns genmlx.rewrite-test
  "Tests for algebraic graph rewriting engine (WP-5).
   Gate 6: Graph rewriting correctness on conjugate models."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.rewrite :as rw]
            [genmlx.conjugacy :as conj]
            [genmlx.affine :as aff]
            [genmlx.dep-graph :as dep-graph]
            [genmlx.schema :as schema]
            [genmlx.mlx :as mx]
            [genmlx.choicemap :as cm]
            [genmlx.dist :as dist]
            [genmlx.inference.auto-analytical :as auto]))

;; =========================================================================
;; Section 1: ConjugacyRule
;; =========================================================================

(deftest conjugacy-rule-test
  (testing "Normal-Normal conjugacy rule"
    (let [s (schema/extract-schema '([x]
              (let [mu (trace :mu (dist/gaussian 0 10))]
                (trace :y (dist/gaussian mu 1))
                mu)))
          pairs (conj/detect-conjugate-pairs s)
          graph (dep-graph/build-dep-graph s)
          rules (rw/generate-rewrite-rules s pairs)]
      (is (>= (count rules) 1) "generates at least 1 rule")
      (let [conj-rule (first (filter #(instance? rw/ConjugacyRule %) rules))]
        (is (some? conj-rule) "conjugacy rule exists")
        (is (rw/-applicable? conj-rule graph s) "conjugacy rule applicable")
        (is (= :mu (:prior-addr conj-rule)) "conjugacy rule prior")
        (is (= [:y] (:obs-addrs conj-rule)) "conjugacy rule obs")
        (is (= :normal-normal (:family conj-rule)) "conjugacy rule family")
        (let [result (rw/-apply conj-rule graph s nil)]
          (is (some? result) "apply returns result")
          (is (< (count (:nodes (:graph' result)))
                 (count (:nodes graph)))
              "graph' has fewer nodes")
          (is (not (contains? (:nodes (:graph' result)) :mu))
              "mu eliminated from nodes")
          (is (seq (:handlers result)) "handlers generated")
          (is (contains? (:eliminated result) :mu) "mu in eliminated set")
          (is (.includes (:description result) "mu") "description mentions mu")))))

  (testing "not applicable when prior missing from graph"
    (let [s (schema/extract-schema '([x]
              (let [mu (trace :mu (dist/gaussian 0 10))]
                (trace :y (dist/gaussian mu 1))
                mu)))
          pairs (conj/detect-conjugate-pairs s)
          graph (dep-graph/build-dep-graph s)
          graph' (update graph :nodes disj :mu)
          rule (rw/->ConjugacyRule :normal-normal :mu [:y])]
      (is (not (rw/-applicable? rule graph' s))
          "not applicable when prior missing")))

  (testing "multiple observations grouped into one rule"
    (let [s (schema/extract-schema '([x]
              (let [mu (trace :mu (dist/gaussian 0 10))]
                (trace :y1 (dist/gaussian mu 1))
                (trace :y2 (dist/gaussian mu 2))
                mu)))
          pairs (conj/detect-conjugate-pairs s)
          rules (rw/generate-rewrite-rules s pairs)
          conj-rules (filter #(instance? rw/ConjugacyRule %) rules)]
      (is (= 1 (count conj-rules)) "one grouped rule for mu")
      (is (= 2 (count (:obs-addrs (first conj-rules)))) "rule has 2 obs"))))

;; =========================================================================
;; Section 2: KalmanRule
;; =========================================================================

(deftest kalman-rule-test
  (testing "Kalman chain detection and application"
    (let [s (schema/extract-schema '([x]
              (let [z0 (trace :z0 (dist/gaussian 0 1))
                    z1 (trace :z1 (dist/gaussian z0 0.5))
                    z2 (trace :z2 (dist/gaussian z1 0.5))]
                (trace :y0 (dist/gaussian z0 0.3))
                (trace :y1 (dist/gaussian z1 0.3))
                (trace :y2 (dist/gaussian z2 0.3))
                z2)))
          pairs (conj/detect-conjugate-pairs s)
          chains (aff/detect-kalman-chains pairs)
          graph (dep-graph/build-dep-graph s)
          rules (rw/generate-rewrite-rules s pairs)]
      (is (some #(instance? rw/KalmanRule %) rules) "has kalman rule")
      (let [k-rule (first (filter #(instance? rw/KalmanRule %) rules))]
        (is (rw/-applicable? k-rule graph s) "kalman rule applicable")
        (let [result (rw/-apply k-rule graph s nil)]
          (is (every? #(not (contains? (:nodes (:graph' result)) %))
                      [:z0 :z1 :z2])
              "kalman: eliminates latent nodes")
          (is (seq (:handlers result)) "kalman: handlers generated")
          (is (contains? (:eliminated result) :z0) "kalman: z0 in eliminated")
          (is (contains? (:eliminated result) :z1) "kalman: z1 in eliminated")
          (is (contains? (:eliminated result) :z2) "kalman: z2 in eliminated"))))))

;; =========================================================================
;; Section 3: RaoBlackwellRule
;; =========================================================================

(deftest rao-blackwell-rule-test
  (testing "shared prior with conjugate and non-conjugate children"
    (let [s (schema/extract-schema '([x]
              (let [mu (trace :mu (dist/gaussian 0 10))]
                (trace :y (dist/gaussian mu 1))
                (trace :z (dist/cauchy mu 1))
                mu)))
          pairs (conj/detect-conjugate-pairs s)
          graph (dep-graph/build-dep-graph s)
          rules (rw/generate-rewrite-rules s pairs)]
      (let [rb-rules (filter #(instance? rw/RaoBlackwellRule %) rules)]
        (is (>= (count rb-rules) 1) "RB rule generated for shared prior")
        (when (seq rb-rules)
          (let [rb (first rb-rules)]
            (is (= :mu (:prior-addr rb)) "RB prior")
            (is (rw/-applicable? rb graph s) "RB applicable")
            (let [result (rw/-apply rb graph s nil)]
              (is (= (count (:nodes graph))
                     (count (:nodes (:graph' result))))
                  "RB: graph nodes unchanged")
              (is (empty? (:eliminated result)) "RB: nothing eliminated")
              (is (contains? (:handlers result) :mu)
                  "RB: handler generated for prior")
              (is (.includes (:description result) "Rao-Blackwell")
                  "RB: description mentions Rao-Blackwell"))))))))

;; =========================================================================
;; Section 4: apply-rewrites engine
;; =========================================================================

(deftest apply-rewrites-test
  (testing "single rule application"
    (let [s (schema/extract-schema '([x]
              (let [mu (trace :mu (dist/gaussian 0 10))]
                (trace :y (dist/gaussian mu 1))
                mu)))
          pairs (conj/detect-conjugate-pairs s)
          graph (dep-graph/build-dep-graph s)
          rules (rw/generate-rewrite-rules s pairs)
          result (rw/apply-rewrites graph s nil rules)]
      (is (contains? (:eliminated result) :mu) "single rule: mu eliminated")
      (is (seq (:handlers result)) "single rule: handlers present")
      (is (seq (:rewrite-log result)) "single rule: rewrite log non-empty")
      (is (some? (:residual-graph result)) "single rule: residual graph exists")))

  (testing "no applicable rules"
    (let [s (schema/extract-schema '([x]
              (let [a (trace :a (dist/uniform 0 1))
                    b (trace :b (dist/cauchy 0 1))]
                (mx/add a b))))
          pairs (conj/detect-conjugate-pairs s)
          graph (dep-graph/build-dep-graph s)
          rules (rw/generate-rewrite-rules s pairs)
          result (rw/apply-rewrites graph s nil rules)]
      (is (empty? (:eliminated result)) "no rules: nothing eliminated")
      (is (empty? (:handlers result)) "no rules: no handlers")
      (is (empty? (:rewrite-log result)) "no rules: empty rewrite log")
      (is (= (:nodes graph) (:nodes (:residual-graph result)))
          "no rules: graph unchanged")))

  (testing "multiple rules — progressive elimination"
    (let [s (schema/extract-schema '([x]
              (let [mu (trace :mu (dist/gaussian 0 10))
                    p  (trace :p (dist/beta-dist 1 1))]
                (trace :y1 (dist/gaussian mu 1))
                (trace :y2 (dist/gaussian mu 2))
                (trace :coin (dist/bernoulli p))
                mu)))
          pairs (conj/detect-conjugate-pairs s)
          graph (dep-graph/build-dep-graph s)
          rules (rw/generate-rewrite-rules s pairs)
          result (rw/apply-rewrites graph s nil rules)]
      (is (contains? (:eliminated result) :mu) "multi: mu eliminated")
      (is (contains? (:eliminated result) :p) "multi: p eliminated")
      (is (>= (count (:rewrite-log result)) 2) "multi: 2+ rewrites applied")
      (is (< (count (:nodes (:residual-graph result)))
             (count (:nodes graph)))
          "multi: residual has fewer nodes"))))

;; =========================================================================
;; Section 5: generate-rewrite-rules
;; =========================================================================

(deftest generate-rewrite-rules-test
  (testing "pure conjugate model"
    (let [s (schema/extract-schema '([x]
              (let [mu (trace :mu (dist/gaussian 0 10))]
                (trace :y (dist/gaussian mu 1))
                mu)))
          pairs (conj/detect-conjugate-pairs s)
          rules (rw/generate-rewrite-rules s pairs)]
      (is (seq rules) "pure conjugate: rules generated")
      (is (some #(instance? rw/ConjugacyRule %) rules)
          "pure conjugate: has ConjugacyRule")))

  (testing "Kalman chain model"
    (let [s (schema/extract-schema '([x]
              (let [z0 (trace :z0 (dist/gaussian 0 1))
                    z1 (trace :z1 (dist/gaussian z0 0.5))]
                (trace :y0 (dist/gaussian z0 0.3))
                (trace :y1 (dist/gaussian z1 0.3))
                z1)))
          pairs (conj/detect-conjugate-pairs s)
          rules (rw/generate-rewrite-rules s pairs)]
      (is (some #(instance? rw/KalmanRule %) rules)
          "kalman: KalmanRule generated")))

  (testing "no conjugate pairs"
    (let [s (schema/extract-schema '([x]
              (let [a (trace :a (dist/uniform 0 1))]
                a)))
          pairs (conj/detect-conjugate-pairs s)
          rules (rw/generate-rewrite-rules s pairs)]
      (is (empty? rules) "no pairs: empty rules")))

  (testing "priority: Kalman before conjugacy"
    (let [s (schema/extract-schema '([x]
              (let [z0 (trace :z0 (dist/gaussian 0 1))
                    z1 (trace :z1 (dist/gaussian z0 0.5))
                    mu (trace :mu (dist/gaussian 0 10))]
                (trace :y0 (dist/gaussian z0 0.3))
                (trace :y1 (dist/gaussian z1 0.3))
                (trace :obs (dist/gaussian mu 1))
                z1)))
          pairs (conj/detect-conjugate-pairs s)
          rules (rw/generate-rewrite-rules s pairs)]
      (is (instance? rw/KalmanRule (first rules))
          "priority: first rule is Kalman"))))

;; =========================================================================
;; Section 6: build-analytical-plan (integration)
;; =========================================================================

(deftest build-analytical-plan-test
  (testing "full pipeline"
    (let [s (schema/extract-schema '([x]
              (let [mu (trace :mu (dist/gaussian 0 10))]
                (trace :y1 (dist/gaussian mu 1))
                (trace :y2 (dist/gaussian mu 2))
                mu)))
          s-conj (conj/augment-schema-with-conjugacy s)
          plan (rw/build-analytical-plan s-conj)]
      (is (some? (:rewrite-result plan)) "plan: has rewrite result")
      (is (fn? (:auto-transition plan)) "plan: has auto-transition")
      (is (some? (:stats plan)) "plan: stats present")
      (is (= 3 (get-in plan [:stats :total-sites])) "plan: 3 total sites")
      (is (>= (get-in plan [:stats :eliminated]) 1) "plan: 1+ eliminated")
      (is (< (get-in plan [:stats :residual])
             (get-in plan [:stats :total-sites]))
          "plan: residual < total")))

  (testing "no conjugate pairs gives nil auto-transition"
    (let [s (schema/extract-schema '([x]
              (let [a (trace :a (dist/uniform 0 1))]
                a)))
          s-conj (conj/augment-schema-with-conjugacy s)
          plan (rw/build-analytical-plan s-conj)]
      (is (nil? (:auto-transition plan)) "no pairs: auto-transition is nil")
      (is (= 0 (get-in plan [:stats :eliminated])) "no pairs: 0 eliminated"))))

;; =========================================================================
;; Section 7: Handler correctness (Gate 6 core)
;; =========================================================================

(deftest gate6-handler-correctness-test
  (testing "marginal LL correct for Normal-Normal"
    (let [s (schema/extract-schema '([x]
              (let [mu (trace :mu (dist/gaussian 0 10))]
                (trace :y (dist/gaussian mu 1))
                mu)))
          s-conj (conj/augment-schema-with-conjugacy s)
          plan (rw/build-analytical-plan s-conj)
          transition (:auto-transition plan)
          constraints (-> (cm/choicemap) (cm/set-value :y (mx/scalar 3.0)))
          state {:choices (cm/choicemap)
                 :constraints constraints
                 :score (mx/scalar 0.0)
                 :weight (mx/scalar 0.0)
                 :auto-posteriors {}}
          mu-dist (dist/gaussian (mx/scalar 0.0) (mx/scalar 10.0))
          y-dist (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0))
          [_ s1] (transition state :mu mu-dist)
          [_ s2] (transition s1 :y y-dist)
          score (mx/item (:score s2))
          expected-ll (* -0.5 (+ 1.8378770664093453
                                 (js/Math.log 101.0)
                                 (/ 9.0 101.0)))]
      (is (h/close? expected-ll score 0.001) "gate6: marginal LL correct")
      (is (h/close? score (mx/item (:weight s2)) 1e-8) "gate6: weight = score"))))

;; =========================================================================
;; Section 8: Edge cases
;; =========================================================================

(deftest edge-cases-test
  (testing "fully analytical: mu eliminated"
    (let [s (schema/extract-schema '([x]
              (let [mu (trace :mu (dist/gaussian 0 10))]
                (trace :y (dist/gaussian mu 1))
                mu)))
          s-conj (conj/augment-schema-with-conjugacy s)
          plan (rw/build-analytical-plan s-conj)]
      (is (contains? (get-in plan [:rewrite-result :eliminated]) :mu)
          "fully analytical: mu eliminated")))

  (testing "empty schema (no trace sites)"
    (let [s (schema/extract-schema '([x] (mx/add x 1)))
          s-conj (conj/augment-schema-with-conjugacy s)
          plan (rw/build-analytical-plan s-conj)]
      (is (= 0 (get-in plan [:stats :total-sites])) "empty: 0 sites")
      (is (nil? (:auto-transition plan)) "empty: no transition")))

  (testing "Beta-Bernoulli elimination"
    (let [s (schema/extract-schema '([x]
              (let [p (trace :p (dist/beta-dist 2 3))]
                (trace :coin (dist/bernoulli p))
                p)))
          s-conj (conj/augment-schema-with-conjugacy s)
          plan (rw/build-analytical-plan s-conj)]
      (is (contains? (get-in plan [:rewrite-result :eliminated]) :p)
          "beta-bernoulli: p eliminated")
      (is (fn? (:auto-transition plan))
          "beta-bernoulli: has transition"))))

(cljs.test/run-tests)
