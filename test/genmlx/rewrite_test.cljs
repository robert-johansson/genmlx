(ns genmlx.rewrite-test
  "Tests for algebraic graph rewriting engine (WP-5).
   Gate 6: Graph rewriting correctness on conjugate models."
  (:require [genmlx.rewrite :as rw]
            [genmlx.conjugacy :as conj]
            [genmlx.affine :as aff]
            [genmlx.dep-graph :as dep-graph]
            [genmlx.schema :as schema]
            [genmlx.mlx :as mx]
            [genmlx.choicemap :as cm]
            [genmlx.dist :as dist]
            [genmlx.inference.auto-analytical :as auto]))

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

(defn assert-close [desc expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass-count inc)
          (println (str "  PASS: " desc)))
      (do (swap! fail-count inc)
          (println (str "  FAIL: " desc " — expected " expected " got " actual " (diff " diff ")"))))))

;; =========================================================================
;; Section 1: ConjugacyRule
;; =========================================================================

(println "\n=== Section 1: ConjugacyRule ===")

;; Build a simple Normal-Normal schema: mu ~ N(0,10), y ~ N(mu,1)
(let [s (schema/extract-schema '([x]
          (let [mu (trace :mu (dist/gaussian 0 10))]
            (trace :y (dist/gaussian mu 1))
            mu)))
      pairs (conj/detect-conjugate-pairs s)
      graph (dep-graph/build-dep-graph s)
      rules (rw/generate-rewrite-rules s pairs)]

  (assert-true "generates at least 1 rule" (>= (count rules) 1))

  ;; Find the conjugacy rule
  (let [conj-rule (first (filter #(instance? rw/ConjugacyRule %) rules))]
    (assert-true "conjugacy rule exists" (some? conj-rule))
    (assert-true "conjugacy rule applicable" (rw/-applicable? conj-rule graph s))
    (assert-equal "conjugacy rule prior" :mu (:prior-addr conj-rule))
    (assert-equal "conjugacy rule obs" [:y] (:obs-addrs conj-rule))
    (assert-equal "conjugacy rule family" :normal-normal (:family conj-rule))

    ;; Apply the rule
    (let [result (rw/-apply conj-rule graph s nil)]
      (assert-true "apply returns result" (some? result))
      (assert-true "graph' has fewer nodes" (< (count (:nodes (:graph' result)))
                                                (count (:nodes graph))))
      (assert-true "mu eliminated from nodes" (not (contains? (:nodes (:graph' result)) :mu)))
      (assert-true "handlers generated" (seq (:handlers result)))
      (assert-true "mu in eliminated set" (contains? (:eliminated result) :mu))
      (assert-true "description mentions mu" (.includes (:description result) "mu")))))

;; Not applicable when prior not in graph
(let [s (schema/extract-schema '([x]
          (let [mu (trace :mu (dist/gaussian 0 10))]
            (trace :y (dist/gaussian mu 1))
            mu)))
      pairs (conj/detect-conjugate-pairs s)
      graph (dep-graph/build-dep-graph s)
      ;; Remove :mu from graph nodes
      graph' (update graph :nodes disj :mu)
      rule (rw/->ConjugacyRule :normal-normal :mu [:y])]
  (assert-true "not applicable when prior missing" (not (rw/-applicable? rule graph' s))))

;; Multiple observations of same prior
(let [s (schema/extract-schema '([x]
          (let [mu (trace :mu (dist/gaussian 0 10))]
            (trace :y1 (dist/gaussian mu 1))
            (trace :y2 (dist/gaussian mu 2))
            mu)))
      pairs (conj/detect-conjugate-pairs s)
      rules (rw/generate-rewrite-rules s pairs)
      conj-rules (filter #(instance? rw/ConjugacyRule %) rules)]
  ;; Should group multiple obs into one rule
  (assert-equal "one grouped rule for mu" 1 (count conj-rules))
  (assert-equal "rule has 2 obs" 2 (count (:obs-addrs (first conj-rules)))))

;; =========================================================================
;; Section 2: KalmanRule
;; =========================================================================

(println "\n=== Section 2: KalmanRule ===")

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

  (assert-true "has kalman rule" (some #(instance? rw/KalmanRule %) rules))

  (let [k-rule (first (filter #(instance? rw/KalmanRule %) rules))]
    (assert-true "kalman rule applicable" (rw/-applicable? k-rule graph s))

    (let [result (rw/-apply k-rule graph s nil)]
      (assert-true "kalman: eliminates latent nodes"
                   (every? #(not (contains? (:nodes (:graph' result)) %))
                           [:z0 :z1 :z2]))
      (assert-true "kalman: handlers generated" (seq (:handlers result)))
      (assert-true "kalman: z0 in eliminated" (contains? (:eliminated result) :z0))
      (assert-true "kalman: z1 in eliminated" (contains? (:eliminated result) :z1))
      (assert-true "kalman: z2 in eliminated" (contains? (:eliminated result) :z2)))))

;; =========================================================================
;; Section 3: RaoBlackwellRule
;; =========================================================================

(println "\n=== Section 3: RaoBlackwellRule ===")

;; Shared prior: mu has conjugate obs (y) AND non-conjugate child (z)
(let [s (schema/extract-schema '([x]
          (let [mu (trace :mu (dist/gaussian 0 10))]
            (trace :y (dist/gaussian mu 1))
            (trace :z (dist/cauchy mu 1))
            mu)))
      pairs (conj/detect-conjugate-pairs s)
      graph (dep-graph/build-dep-graph s)
      rules (rw/generate-rewrite-rules s pairs)]

  ;; Should detect shared-prior situation
  (let [rb-rules (filter #(instance? rw/RaoBlackwellRule %) rules)]
    (assert-true "RB rule generated for shared prior" (>= (count rb-rules) 1))
    (when (seq rb-rules)
      (let [rb (first rb-rules)]
        (assert-equal "RB prior" :mu (:prior-addr rb))
        (assert-true "RB applicable" (rw/-applicable? rb graph s))
        (let [result (rw/-apply rb graph s nil)]
          ;; RB doesn't eliminate — graph unchanged
          (assert-equal "RB: graph nodes unchanged"
                        (count (:nodes graph))
                        (count (:nodes (:graph' result))))
          (assert-true "RB: nothing eliminated" (empty? (:eliminated result)))
          (assert-true "RB: handler generated for prior" (contains? (:handlers result) :mu))
          (assert-true "RB: description mentions Rao-Blackwell"
                       (.includes (:description result) "Rao-Blackwell")))))))

;; =========================================================================
;; Section 4: apply-rewrites engine
;; =========================================================================

(println "\n=== Section 4: apply-rewrites engine ===")

;; Single rule application
(let [s (schema/extract-schema '([x]
          (let [mu (trace :mu (dist/gaussian 0 10))]
            (trace :y (dist/gaussian mu 1))
            mu)))
      pairs (conj/detect-conjugate-pairs s)
      graph (dep-graph/build-dep-graph s)
      rules (rw/generate-rewrite-rules s pairs)
      result (rw/apply-rewrites graph s nil rules)]
  (assert-true "single rule: mu eliminated" (contains? (:eliminated result) :mu))
  (assert-true "single rule: handlers present" (seq (:handlers result)))
  (assert-true "single rule: rewrite log non-empty" (seq (:rewrite-log result)))
  (assert-true "single rule: residual graph exists" (some? (:residual-graph result))))

;; No applicable rules
(let [s (schema/extract-schema '([x]
          (let [a (trace :a (dist/uniform 0 1))
                b (trace :b (dist/cauchy 0 1))]
            (mx/add a b))))
      pairs (conj/detect-conjugate-pairs s)
      graph (dep-graph/build-dep-graph s)
      rules (rw/generate-rewrite-rules s pairs)
      result (rw/apply-rewrites graph s nil rules)]
  (assert-true "no rules: nothing eliminated" (empty? (:eliminated result)))
  (assert-true "no rules: no handlers" (empty? (:handlers result)))
  (assert-true "no rules: empty rewrite log" (empty? (:rewrite-log result)))
  (assert-equal "no rules: graph unchanged" (:nodes graph) (:nodes (:residual-graph result))))

;; Multiple rules — progressive elimination
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
  (assert-true "multi: mu eliminated" (contains? (:eliminated result) :mu))
  (assert-true "multi: p eliminated" (contains? (:eliminated result) :p))
  (assert-true "multi: 2+ rewrites applied" (>= (count (:rewrite-log result)) 2))
  (assert-true "multi: residual has fewer nodes"
               (< (count (:nodes (:residual-graph result)))
                  (count (:nodes graph)))))

;; =========================================================================
;; Section 5: generate-rewrite-rules
;; =========================================================================

(println "\n=== Section 5: generate-rewrite-rules ===")

;; Pure conjugate model
(let [s (schema/extract-schema '([x]
          (let [mu (trace :mu (dist/gaussian 0 10))]
            (trace :y (dist/gaussian mu 1))
            mu)))
      pairs (conj/detect-conjugate-pairs s)
      rules (rw/generate-rewrite-rules s pairs)]
  (assert-true "pure conjugate: rules generated" (seq rules))
  (assert-true "pure conjugate: has ConjugacyRule"
               (some #(instance? rw/ConjugacyRule %) rules)))

;; Kalman chain model
(let [s (schema/extract-schema '([x]
          (let [z0 (trace :z0 (dist/gaussian 0 1))
                z1 (trace :z1 (dist/gaussian z0 0.5))]
            (trace :y0 (dist/gaussian z0 0.3))
            (trace :y1 (dist/gaussian z1 0.3))
            z1)))
      pairs (conj/detect-conjugate-pairs s)
      rules (rw/generate-rewrite-rules s pairs)]
  (assert-true "kalman: KalmanRule generated"
               (some #(instance? rw/KalmanRule %) rules)))

;; No conjugate pairs
(let [s (schema/extract-schema '([x]
          (let [a (trace :a (dist/uniform 0 1))]
            a)))
      pairs (conj/detect-conjugate-pairs s)
      rules (rw/generate-rewrite-rules s pairs)]
  (assert-true "no pairs: empty rules" (empty? rules)))

;; Priority: Kalman rules come before conjugacy rules
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
  (assert-true "priority: first rule is Kalman"
               (instance? rw/KalmanRule (first rules))))

;; =========================================================================
;; Section 6: build-analytical-plan (integration)
;; =========================================================================

(println "\n=== Section 6: build-analytical-plan ===")

;; Full pipeline: schema → detection → graph → rewrite → handlers
(let [s (schema/extract-schema '([x]
          (let [mu (trace :mu (dist/gaussian 0 10))]
            (trace :y1 (dist/gaussian mu 1))
            (trace :y2 (dist/gaussian mu 2))
            mu)))
      s-conj (conj/augment-schema-with-conjugacy s)
      plan (rw/build-analytical-plan s-conj)]
  (assert-true "plan: has rewrite result" (some? (:rewrite-result plan)))
  (assert-true "plan: has auto-transition" (fn? (:auto-transition plan)))
  (assert-true "plan: stats present" (some? (:stats plan)))
  (assert-equal "plan: 3 total sites" 3 (get-in plan [:stats :total-sites]))
  (assert-true "plan: 1+ eliminated" (>= (get-in plan [:stats :eliminated]) 1))
  (assert-true "plan: residual < total"
               (< (get-in plan [:stats :residual])
                  (get-in plan [:stats :total-sites]))))

;; No conjugate pairs → nil auto-transition
(let [s (schema/extract-schema '([x]
          (let [a (trace :a (dist/uniform 0 1))]
            a)))
      s-conj (conj/augment-schema-with-conjugacy s)
      plan (rw/build-analytical-plan s-conj)]
  (assert-true "no pairs: auto-transition is nil" (nil? (:auto-transition plan)))
  (assert-equal "no pairs: 0 eliminated" 0 (get-in plan [:stats :eliminated])))

;; =========================================================================
;; Section 7: Handler correctness (Gate 6 core)
;; =========================================================================

(println "\n=== Section 7: Gate 6 — Handler correctness ===")

;; Verify auto-transition produces correct marginal LL
;; mu ~ N(0, 10), y ~ N(mu, 1), observed y=3.0
;; Marginal: y ~ N(0, 101), ll = N(3; 0, 101)
(let [s (schema/extract-schema '([x]
          (let [mu (trace :mu (dist/gaussian 0 10))]
            (trace :y (dist/gaussian mu 1))
            mu)))
      s-conj (conj/augment-schema-with-conjugacy s)
      plan (rw/build-analytical-plan s-conj)
      transition (:auto-transition plan)
      ;; Simulate: create state, run mu, run y
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
      ;; Expected: N(3; 0, 101) = -0.5*(log(2pi) + log(101) + 9/101)
      expected-ll (* -0.5 (+ 1.8378770664093453
                             (js/Math.log 101.0)
                             (/ 9.0 101.0)))]
  (assert-close "gate6: marginal LL correct" expected-ll score 0.001)
  (assert-close "gate6: weight = score" score (mx/item (:weight s2)) 1e-8))

;; =========================================================================
;; Section 8: Edge cases
;; =========================================================================

(println "\n=== Section 8: Edge cases ===")

;; All sites eliminable → fully analytical
(let [s (schema/extract-schema '([x]
          (let [mu (trace :mu (dist/gaussian 0 10))]
            (trace :y (dist/gaussian mu 1))
            mu)))
      s-conj (conj/augment-schema-with-conjugacy s)
      plan (rw/build-analytical-plan s-conj)]
  ;; mu eliminated, y is obs (constrained) — residual may have y but no sampling needed
  (assert-true "fully analytical: mu eliminated"
               (contains? (get-in plan [:rewrite-result :eliminated]) :mu)))

;; Empty schema (no trace sites)
(let [s (schema/extract-schema '([x] (mx/add x 1)))
      s-conj (conj/augment-schema-with-conjugacy s)
      plan (rw/build-analytical-plan s-conj)]
  (assert-equal "empty: 0 sites" 0 (get-in plan [:stats :total-sites]))
  (assert-true "empty: no transition" (nil? (:auto-transition plan))))

;; Beta-Bernoulli elimination
(let [s (schema/extract-schema '([x]
          (let [p (trace :p (dist/beta-dist 2 3))]
            (trace :coin (dist/bernoulli p))
            p)))
      s-conj (conj/augment-schema-with-conjugacy s)
      plan (rw/build-analytical-plan s-conj)]
  (assert-true "beta-bernoulli: p eliminated"
               (contains? (get-in plan [:rewrite-result :eliminated]) :p))
  (assert-true "beta-bernoulli: has transition" (fn? (:auto-transition plan))))

;; =========================================================================
;; Summary
;; =========================================================================

(println (str "\n== RESULTS: " @pass-count " passed, " @fail-count " failed =="))
