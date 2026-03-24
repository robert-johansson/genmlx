(ns genmlx.gfi-combinator-invariants-test
  "GFI combinator invariants: score additivity, importance weight identity,
   Switch structural update, and regenerate weight correctness.
   Sections 3.1-3.4 of CORRECTNESS_PLAN.md."
  (:require [cljs.test :refer [deftest is testing run-tests]]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.combinators :as comb]
            [genmlx.selection :as sel]
            [genmlx.trace :as tr]
            [genmlx.test-helpers :as th])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Private helpers
;; ---------------------------------------------------------------------------

(defn- realize
  "mx/eval! then mx/item -> JS number."
  [x]
  (mx/eval! x) (mx/item x))

(defn- logsumexp
  "Numerically stable log(sum(exp(xs))) for a seq of numbers."
  [xs]
  (let [max-x (apply max xs)]
    (+ max-x (js/Math.log (reduce + (map #(js/Math.exp (- % max-x)) xs))))))

(defn- is-estimate
  "Compute IS estimate exp(logsumexp(log-weights) - log(N))."
  [log-weights]
  (js/Math.exp (- (logsumexp log-weights) (js/Math.log (count log-weights)))))

(defn- choice-val
  "Extract JS number from choicemap at addr path."
  [choices addr-path]
  (realize (cm/get-choice choices addr-path)))

(defn- strip-compiled
  "Remove compiled paths and analytical plans from a DynamicGF,
   forcing the handler (interpreted) execution path.
   Re-wraps with auto-key for PRNG threading."
  [gf]
  (dyn/auto-key
    (dyn/->DynamicGF
      (:body-fn gf) (:source gf)
      (dissoc (:schema gf)
              :compiled-simulate :compiled-generate
              :compiled-update :compiled-assess
              :compiled-project :compiled-regenerate
              :analytical-plan :auto-transition
              :auto-regenerate-transition :auto-regenerate-handlers
              :auto-handlers))))

;; ---------------------------------------------------------------------------
;; Models M1-M4: reused from contract_verification_test patterns
;; ---------------------------------------------------------------------------

;; M1: Map kernel — y ~ N(x, 1)
(def map-kernel
  (dyn/auto-key (gen [x] (let [y (trace :y (dist/gaussian x 1))]
                            (mx/eval! y) (mx/item y)))))
(def map-model (comb/map-combinator map-kernel))

;; M2: Unfold kernel — y ~ N(state, 1), returns y as next state
(def unfold-kernel
  (dyn/auto-key (gen [t state] (let [y (trace :y (dist/gaussian state 1))]
                                  (mx/eval! y) (mx/item y)))))
(def unfold-model (comb/unfold-combinator unfold-kernel))

;; M3: Switch branches
(def branch-a (dyn/auto-key (gen [] (trace :x (dist/gaussian 0 1)))))
(def branch-b (dyn/auto-key (gen [] (trace :x (dist/gaussian 5 2)))))
(def switch-model (comb/switch-combinator branch-a branch-b))

;; M4: Scan kernel — y ~ N(carry, 1), returns [y, y]
(def scan-kernel
  (dyn/auto-key (gen [carry x] (let [y (trace :y (dist/gaussian carry 1))]
                                  (mx/eval! y)
                                  [(mx/item y) (mx/item y)]))))
(def scan-model (comb/scan-combinator scan-kernel))

;; ---------------------------------------------------------------------------
;; Models M5-M10: new for sections 3.2-3.4
;; ---------------------------------------------------------------------------

;; M5: Normal-Normal conjugate — mu ~ N(0, 10), y ~ N(mu, 1)
;; Analytical: log p(y=3) = -3.2710532, p(y=3) = 0.037966
(def nn-single-obs
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/gaussian 0 10))]
        (trace :y (dist/gaussian mu 1))
        mu))))

;; M6: Beta-Bernoulli — p ~ Beta(2, 2), x_i ~ Bernoulli(p), 10 obs k=8
;; Analytical (ordered): p(y) = B(10,4)/B(2,2) = 0.002098
(def beta-bern
  (dyn/auto-key
    (gen []
      (let [p (trace :p (dist/beta-dist 2 2))]
        (doseq [j (range 10)]
          (trace (keyword (str "x" j)) (dist/bernoulli p)))
        p))))

;; M7: Switch with asymmetric branches
(def branch-2site
  (dyn/auto-key (gen [] (trace :x (dist/gaussian 0 1))
                        (trace :y (dist/gaussian 0 1)))))
(def branch-1site
  (dyn/auto-key (gen [] (trace :z (dist/gaussian 3 1)))))
(def switch-asymmetric (comb/switch-combinator branch-2site branch-1site))

;; M8: Independent two-site — x ~ N(0,1), y ~ N(0,1)
(def independent-model
  (dyn/auto-key (gen [] (let [x (trace :x (dist/gaussian 0 1))
                              y (trace :y (dist/gaussian 0 1))]
                          (mx/eval! x y) [x y]))))

;; M9: Dependent two-site — x ~ N(0,1), y ~ N(x, 1)
(def dependent-model
  (dyn/auto-key (gen [] (let [x (trace :x (dist/gaussian 0 1))]
                          (mx/eval! x)
                          (let [y (trace :y (dist/gaussian (mx/item x) 1))]
                            (mx/eval! y) [x y])))))

;; M10: Map kernel with internal dependency — z ~ N(x,1), y ~ N(z,1)
(def dep-map-kernel
  (dyn/auto-key (gen [x] (let [z (trace :z (dist/gaussian x 1))]
                            (mx/eval! z)
                            (let [y (trace :y (dist/gaussian (mx/item z) 1))]
                              (mx/eval! y) (mx/item z))))))
(def dep-map-model (comb/map-combinator dep-map-kernel))

;; ---------------------------------------------------------------------------
;; Constraints
;; ---------------------------------------------------------------------------

(def nn-constraints
  "Constrain y=3 for the Normal-Normal model."
  (cm/choicemap :y (mx/scalar 3.0)))

(def bb-constraints
  "8 successes (x0-x7=1), 2 failures (x8-x9=0) for Beta-Bernoulli."
  (reduce (fn [cm j]
            (cm/set-choice cm [(keyword (str "x" j))]
                           (mx/scalar (if (< j 8) 1.0 0.0))))
          cm/EMPTY
          (range 10)))

;; =========================================================================
;; Section 3.1 — Score Under Combinators
;; =========================================================================

(deftest map-score-additivity
  (testing "Map score = sum of per-element assess weights (20 trials)"
    (let [inputs [1.0 2.0 3.0]]
      (doseq [_ (range 20)]
        (let [tr (p/simulate map-model [inputs])
              trace-score (realize (:score tr))
              {:keys [choices]} tr
              sum-weights (->> (range 3)
                               (map (fn [i]
                                      (let [{:keys [weight]} (p/assess map-kernel
                                                               [(nth inputs i)]
                                                               (cm/get-submap choices i))]
                                        (realize weight))))
                               (reduce +))]
          ;; Assertion #1
          (is (th/close? trace-score sum-weights 1e-4)
              (str "map score " trace-score " = sum " sum-weights)))))))

(deftest unfold-score-additivity
  (testing "Unfold score = sum of per-step assess weights (20 trials)"
    (let [init-state 0.0]
      (doseq [_ (range 20)]
        (let [tr (p/simulate unfold-model [3 init-state])
              trace-score (realize (:score tr))
              {:keys [choices retval]} tr
              states (into [init-state] retval)
              sum-weights (->> (range 3)
                               (map (fn [t]
                                      (let [{:keys [weight]} (p/assess unfold-kernel
                                                               [t (nth states t)]
                                                               (cm/get-submap choices t))]
                                        (realize weight))))
                               (reduce +))]
          ;; Assertion #2
          (is (th/close? trace-score sum-weights 1e-4)
              (str "unfold score " trace-score " = sum " sum-weights)))))))

(deftest switch-score-identity
  (testing "Switch score = branch assess weight (20 trials each branch)"
    ;; Assertion #3 — branch 0
    (doseq [_ (range 20)]
      (let [tr (p/simulate switch-model [0])
            trace-score (realize (:score tr))
            assess-weight (-> (p/assess branch-a [] (:choices tr)) :weight realize)]
        (is (th/close? trace-score assess-weight 1e-5)
            (str "switch branch-0 score " trace-score " = assess " assess-weight))))
    ;; Assertion #4 — branch 1
    (doseq [_ (range 20)]
      (let [tr (p/simulate switch-model [1])
            trace-score (realize (:score tr))
            assess-weight (-> (p/assess branch-b [] (:choices tr)) :weight realize)]
        (is (th/close? trace-score assess-weight 1e-5)
            (str "switch branch-1 score " trace-score " = assess " assess-weight))))))

(deftest scan-score-additivity
  (testing "Scan score = sum of per-step assess weights (20 trials)"
    (let [init-carry 0.0
          inputs [1 2 3]]
      (doseq [_ (range 20)]
        (let [tr (p/simulate scan-model [init-carry inputs])
              trace-score (realize (:score tr))
              {:keys [choices]} tr
              step-carries (:genmlx.combinators/step-carries (meta tr))
              carries (into [init-carry] step-carries)
              sum-weights (->> (range 3)
                               (map (fn [t]
                                      (let [{:keys [weight]} (p/assess scan-kernel
                                                               [(nth carries t) (nth inputs t)]
                                                               (cm/get-submap choices t))]
                                        (realize weight))))
                               (reduce +))]
          ;; Assertion #5
          (is (th/close? trace-score sum-weights 1e-4)
              (str "scan score " trace-score " = sum " sum-weights)))))))

;; =========================================================================
;; Section 3.2 — Importance Weight Identity
;; =========================================================================

(deftest normal-normal-importance-weight-identity
  (testing "E[exp(weight)] = p(y) for Normal-Normal model via z-test"
    ;; Strip compiled paths to force handler-based prior sampling.
    ;; The L3 analytical handler computes exact marginal likelihood,
    ;; making generate return a deterministic weight (zero variance).
    (let [analytical-py 0.037966
          nn-handler (strip-compiled nn-single-obs)
          n-per-batch 500
          m-batches 20
          batch-estimates
          (mapv (fn [_]
                  (let [log-ws (mapv (fn [_]
                                       (-> (p/generate nn-handler [] nn-constraints)
                                           :weight realize))
                                     (range n-per-batch))]
                    ;; Assertion #6 — all finite
                    (is (every? js/isFinite log-ws)
                        "all Normal-Normal log-weights are finite")
                    (is-estimate log-ws)))
                (range m-batches))]
      ;; Assertion #7 — z-test
      (is (th/z-test-passes? analytical-py batch-estimates 3.5)
          (str "Normal-Normal IS: mean " (th/sample-mean batch-estimates)
               " vs analytical " analytical-py)))))

(deftest beta-bernoulli-importance-weight-identity
  (testing "E[exp(weight)] = p(y) for Beta-Bernoulli model via z-test"
    ;; Analytical p(y) for ORDERED observations (10 individual Bernoulli sites):
    ;; p(y) = B(a+k, b+n-k) / B(a, b) = B(10,4) / B(2,2) = 0.002098
    ;; The spec value 0.09441 includes C(10,8)=45 for unordered observations,
    ;; but our model traces each x_i individually so the sequence is ordered.
    (let [analytical-py 0.002098
          n-per-batch 500
          m-batches 20
          batch-estimates
          (mapv (fn [_]
                  (let [log-ws (mapv (fn [_]
                                       (-> (p/generate beta-bern [] bb-constraints)
                                           :weight realize))
                                     (range n-per-batch))]
                    ;; Assertion #8 — all finite
                    (is (every? js/isFinite log-ws)
                        "all Beta-Bernoulli log-weights are finite")
                    (is-estimate log-ws)))
                (range m-batches))]
      ;; Assertion #9 — z-test
      (is (th/z-test-passes? analytical-py batch-estimates 3.5)
          (str "Beta-Bernoulli IS: mean " (th/sample-mean batch-estimates)
               " vs analytical " analytical-py)))))

(deftest map-combinator-importance-weight-identity
  (testing "Map combinator IS weight: fully constrained deterministic check"
    (let [inputs [1.0 2.0 3.0]
          ;; Constrain all components: y_i = x_i + 0.5
          constraints (reduce (fn [cm i]
                                (cm/set-choice cm [i :y]
                                               (mx/scalar (+ (nth inputs i) 0.5))))
                              cm/EMPTY (range 3))
          ;; Analytical: sum of log p(y_i | x_i, 1) for y_i = x_i + 0.5
          analytical-lp (reduce + (map #(th/gaussian-lp (+ % 0.5) % 1.0) inputs))
          log-ws (mapv (fn [_]
                         (-> (p/generate map-model [inputs] constraints)
                             :weight realize))
                       (range 20))]
      ;; Assertion #10 — all finite
      (is (every? js/isFinite log-ws)
          "all Map log-weights are finite")
      ;; Assertion #11 — all weights equal the analytical value (deterministic)
      (is (every? #(th/close? analytical-lp % 1e-4) log-ws)
          (str "Map IS: all weights equal analytical " analytical-lp)))))

;; =========================================================================
;; Section 3.3 — Switch Structural Update
;; =========================================================================

(deftest switch-cross-branch-update-with-constraint
  (testing "Switch update: branch 0 -> branch 1 with constraint"
    (let [old-trace (p/simulate switch-model [0])
          old-score (realize (:score old-trace))
          modified-trace (with-meta
                           (tr/make-trace (assoc old-trace :args [1]))
                           (meta old-trace))
          constraints (cm/choicemap :x (mx/scalar 4.5))
          {:keys [trace weight discard]} (p/update switch-model modified-trace constraints)
          new-score (realize (:score trace))
          new-x (choice-val (:choices trace) [:x])]
      ;; Assertion #12 — weight = new_score - old_score
      (is (th/close? (realize weight) (- new-score old-score) 1e-4)
          "cross-branch weight = new_score - old_score")
      ;; Assertion #13 — discard contains old branch choices
      (is (some? (cm/get-value (cm/get-submap discard :x)))
          "discard contains old :x")
      ;; Assertion #14 — new trace :x = 4.5
      (is (th/close? 4.5 new-x 1e-6)
          "new trace :x = constrained value 4.5")
      ;; Assertion #15 — new trace score = log p(4.5 | N(5, 2))
      (is (th/close? (th/gaussian-lp 4.5 5 2) new-score 1e-4)
          (str "new score " new-score " = analytical " (th/gaussian-lp 4.5 5 2))))))

(deftest switch-same-branch-update
  (testing "Switch update: same branch with new constraint"
    (let [old-trace (p/simulate switch-model [0])
          old-score (realize (:score old-trace))
          old-x (choice-val (:choices old-trace) [:x])
          constraints (cm/choicemap :x (mx/scalar -0.5))
          {:keys [trace weight discard]} (p/update switch-model old-trace constraints)
          new-score (realize (:score trace))
          new-x (choice-val (:choices trace) [:x])]
      ;; Assertion #16 — weight = new_score - old_score
      (is (th/close? (realize weight) (- new-score old-score) 1e-4)
          "same-branch weight = new_score - old_score")
      ;; Assertion #17 — discard has old :x value
      (is (some? (cm/get-value (cm/get-submap discard :x)))
          "discard contains old :x value")
      ;; Assertion #18 — new trace :x = -0.5
      (is (th/close? -0.5 new-x 1e-6)
          "new trace :x = -0.5"))))

(deftest switch-cross-branch-update-no-constraints
  (testing "Switch update: branch change with no constraints (prior sample)"
    (doseq [_ (range 20)]
      (let [old-trace (p/simulate switch-model [0])
            old-score (realize (:score old-trace))
            modified-trace (with-meta
                             (tr/make-trace (assoc old-trace :args [1]))
                             (meta old-trace))
            {:keys [trace weight discard]} (p/update switch-model modified-trace cm/EMPTY)
            new-score (realize (:score trace))]
        ;; Assertion #19 — weight = new_score - old_score
        (is (th/close? (realize weight) (- new-score old-score) 1e-4)
            "cross-branch no-constraint weight = new_score - old_score")
        ;; Assertion #20 — discard contains old branch choices
        (is (some? (cm/get-value (cm/get-submap discard :x)))
            "discard contains old branch choices")))))

(deftest switch-asymmetric-branch-update
  (testing "Switch update: branches with different numbers of trace sites"
    (let [old-trace (p/simulate switch-asymmetric [0])
          old-score (realize (:score old-trace))
          modified-trace (with-meta
                           (tr/make-trace (assoc old-trace :args [1]))
                           (meta old-trace))
          constraints (cm/choicemap :z (mx/scalar 3.5))
          {:keys [trace weight discard]} (p/update switch-asymmetric modified-trace constraints)
          new-score (realize (:score trace))]
      ;; Assertion #21 — discard contains both :x and :y from old branch
      (is (and (some? (cm/get-value (cm/get-submap discard :x)))
               (some? (cm/get-value (cm/get-submap discard :y))))
          "discard contains both :x and :y from old branch 0")
      ;; Assertion #22 — weight = new_score - old_score
      (is (th/close? (realize weight) (- new-score old-score) 1e-4)
          "asymmetric branch weight = new_score - old_score")
      ;; Assertion #23 — new trace has :z = 3.5
      (is (th/close? 3.5 (choice-val (:choices trace) [:z]) 1e-6)
          "new trace :z = constrained value 3.5"))))

;; =========================================================================
;; Section 3.4 — Regenerate Under Combinators
;; =========================================================================

(deftest regenerate-full-selection-independent
  (testing "Full selection on independent model: weight = 0 (20 trials)"
    (doseq [_ (range 20)]
      (let [tr (p/simulate independent-model [])
            w (-> (p/regenerate independent-model tr sel/all) :weight realize)]
        ;; Assertion #24
        (is (th/close? 0.0 w 1e-6)
            (str "independent full-selection weight = " w ", expected 0"))))))

(deftest regenerate-full-selection-dependent
  (testing "Full selection on dependent model: weight = downstream correction (20 trials)"
    (doseq [_ (range 20)]
      (let [tr (p/simulate dependent-model [])
            x-old (choice-val (:choices tr) [:x])
            y-old (choice-val (:choices tr) [:y])
            {:keys [trace weight]} (p/regenerate dependent-model tr sel/all)
            x-new (choice-val (:choices trace) [:x])
            actual (realize weight)
            expected (- (th/gaussian-lp y-old x-new 1)
                        (th/gaussian-lp y-old x-old 1))]
        ;; Assertion #25
        (is (th/close? expected actual 1e-4)
            (str "dependent full-selection weight " actual " = expected " expected))))))

(deftest regenerate-partial-selection-dependent
  (testing "Partial selection {:x} on dependent model: downstream weight (20 trials)"
    (doseq [_ (range 20)]
      (let [tr (p/simulate dependent-model [])
            x-old (choice-val (:choices tr) [:x])
            y-old (choice-val (:choices tr) [:y])
            {:keys [trace weight]} (p/regenerate dependent-model tr (sel/select :x))
            x-new (choice-val (:choices trace) [:x])
            y-after (choice-val (:choices trace) [:y])
            actual (realize weight)
            expected (- (th/gaussian-lp y-old x-new 1)
                        (th/gaussian-lp y-old x-old 1))]
        ;; Assertion #26
        (is (th/close? expected actual 1e-5)
            (str "partial-selection weight " actual " = expected " expected))
        ;; Assertion #27 — y unchanged
        (is (th/close? y-old y-after 1e-7)
            "y value unchanged after regenerating only :x")))))

(deftest map-regenerate-single-site-kernel
  (testing "Map regenerate: single-site kernel, select component 1 (10 trials)"
    (let [inputs [1.0 2.0 3.0]
          selection (sel/hierarchical 1 sel/all)]
      (doseq [_ (range 10)]
        (let [tr (p/simulate map-model [inputs])
              y0-before (choice-val (:choices tr) [0 :y])
              y2-before (choice-val (:choices tr) [2 :y])
              {:keys [trace weight]} (p/regenerate map-model tr selection)
              w (realize weight)
              y0-after (choice-val (:choices trace) [0 :y])
              y2-after (choice-val (:choices trace) [2 :y])]
          ;; Assertion #28 — weight = 0 (single independent site)
          (is (th/close? 0.0 w 1e-6)
              (str "map single-site weight = " w ", expected 0"))
          ;; Assertion #29 — unselected components unchanged
          (is (and (th/close? y0-before y0-after 1e-7)
                   (th/close? y2-before y2-after 1e-7))
              "unselected components 0 and 2 unchanged"))))))

(deftest map-regenerate-dependent-kernel
  (testing "Map regenerate: dependent kernel, select :z at component 1 (10 trials)"
    (let [inputs [0.0 0.0 0.0]
          selection (sel/hierarchical 1 (sel/select :z))]
      (doseq [_ (range 10)]
        (let [tr (p/simulate dep-map-model [inputs])
              z-old (choice-val (:choices tr) [1 :z])
              y-old (choice-val (:choices tr) [1 :y])
              c0-z-before (choice-val (:choices tr) [0 :z])
              c2-z-before (choice-val (:choices tr) [2 :z])
              {:keys [trace weight]} (p/regenerate dep-map-model tr selection)
              z-new (choice-val (:choices trace) [1 :z])
              c0-z-after (choice-val (:choices trace) [0 :z])
              c2-z-after (choice-val (:choices trace) [2 :z])
              actual (realize weight)
              expected (- (th/gaussian-lp y-old z-new 1)
                          (th/gaussian-lp y-old z-old 1))]
          ;; Assertion #30
          (is (th/close? expected actual 1e-4)
              (str "map dep-kernel weight " actual " = expected " expected))
          ;; Assertion #31 — unselected components unchanged
          (is (and (th/close? c0-z-before c0-z-after 1e-7)
                   (th/close? c2-z-before c2-z-after 1e-7))
              "unselected components 0 and 2 unchanged"))))))

(deftest unfold-regenerate-step0
  (testing "Unfold regenerate: select step 0, downstream step 1 affected (10 trials)"
    (let [init-state 0.0
          selection (sel/hierarchical 0 sel/all)]
      (doseq [_ (range 10)]
        (let [tr (p/simulate unfold-model [3 init-state])
              {:keys [choices]} tr
              y0-old (choice-val choices [0 :y])
              y1-old (choice-val choices [1 :y])
              y2-old (choice-val choices [2 :y])
              {:keys [trace weight]} (p/regenerate unfold-model tr selection)
              y0-new (choice-val (:choices trace) [0 :y])
              y1-after (choice-val (:choices trace) [1 :y])
              y2-after (choice-val (:choices trace) [2 :y])
              actual (realize weight)
              ;; Step 0: weight = 0 (single site, no intra-kernel dependency)
              ;; Step 1: weight = log p(y1 | y0_new) - log p(y1 | y0_old)
              ;; Step 2: weight = 0 (y1 unchanged => state entering step 2 same)
              expected (- (th/gaussian-lp y1-old y0-new 1)
                          (th/gaussian-lp y1-old y0-old 1))]
          ;; Assertion #32 — total weight matches downstream formula
          (is (th/close? expected actual 1e-4)
              (str "unfold regen weight " actual " = expected " expected))
          ;; Assertion #33 — step 2 state unchanged when step 1 replayed
          (is (th/close? y2-old y2-after 1e-7)
              "step 2 value unchanged (step 1 replayed with same value)"))))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(run-tests)
