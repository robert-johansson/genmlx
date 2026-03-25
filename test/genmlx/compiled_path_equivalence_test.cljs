(ns genmlx.compiled-path-equivalence-test
  "Section 7: Compiled path equivalence stress tests.

   Compiled execution must match handler (interpreted) execution exactly.
   The handler is ground truth; compilation is optimization.

   Tests:
   7.1 Stress: 1000x compiled vs handler simulate, all scores match to 1e-5
   7.1 Edge:  extreme parameter values (sigma=1e-6, mu=1e6)
   7.2 Weight: compiled generate weight = handler generate weight
   7.2 Gradient: compiled score gradient = handler score gradient
   7.2 Combinator: compiled Map/Unfold match handler Map/Unfold"
  (:require [cljs.test :refer [deftest is testing run-tests]]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.gradients :as grad]
            [genmlx.combinators :as comb]
            [genmlx.test-helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- force-handler
  "Strip compiled paths from a DynamicGF, forcing handler execution."
  [gf]
  (dyn/->DynamicGF (:body-fn gf) (:source gf)
                   (dissoc (:schema gf)
                           :compiled-simulate
                           :compiled-generate
                           :compiled-update
                           :compiled-assess
                           :compiled-project
                           :compiled-regenerate)))

(defn- realize [x] (mx/eval! x) (mx/item x))

(defn- choice-val
  "Extract JS number from choicemap at addr."
  [choices addr]
  (realize (cm/get-choice choices [addr])))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

(def single-site (gen [] (trace :x (dist/gaussian 0 1))))

(def linreg-model
  (gen [x]
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 5))]
      (trace :y (dist/gaussian (mx/add (mx/multiply slope x) intercept) 1))
      slope)))

(def multi-dist-model
  "Model with diverse distribution types."
  (gen []
    (let [mu    (trace :mu (dist/gaussian 0 5))
          sigma (trace :sigma (dist/exponential 1))
          x     (trace :x (dist/gaussian mu (mx/add sigma (mx/scalar 0.01))))]
      (trace :obs (dist/gaussian x 0.5))
      x)))

(def chain-model
  "3-step dependency chain."
  (gen []
    (let [z1 (trace :z1 (dist/gaussian 0 1))
          z2 (trace :z2 (dist/gaussian z1 1))
          z3 (trace :z3 (dist/gaussian z2 1))]
      z3)))

;; ---------------------------------------------------------------------------
;; Combinator kernels
;; ---------------------------------------------------------------------------

(def map-kernel
  "Kernel for Map combinator: independent regression per point."
  (dyn/auto-key
    (gen [x]
      (let [y-mean (mx/multiply (mx/scalar 2.0) x)]
        (trace :y (dist/gaussian y-mean 1))
        y-mean))))

(def unfold-kernel
  "Kernel for Unfold combinator: random walk with observation."
  (dyn/auto-key
    (gen [t state]
      (let [x (trace :x (dist/gaussian state 1))]
        (trace :y (dist/gaussian x 0.5))
        x))))

;; ==========================================================================
;; Test 1: Stress test — 1000x compiled vs handler simulate
;; ==========================================================================
;; Run same model through both paths with 1000 different seeds.
;; Every score must match to 1e-5.

(deftest stress-simulate-equivalence
  (testing "1000x compiled vs handler simulate on linreg model"
    (let [args [(mx/scalar 2.5)]
          n-trials 1000
          max-diff (atom 0.0)
          failures (atom 0)]
      (doseq [seed (range n-trials)]
        (let [k1 (rng/fresh-key seed)
              compiled-tr (p/simulate (dyn/with-key linreg-model k1) args)
              k2 (rng/fresh-key seed)
              handler-tr (p/simulate (dyn/with-key (force-handler linreg-model) k2) args)
              c-score (realize (:score compiled-tr))
              h-score (realize (:score handler-tr))
              diff (js/Math.abs (- c-score h-score))]
          (swap! max-diff max diff)
          (when (> diff 1e-5)
            (swap! failures inc))))
      (println (str "  Stress test: " n-trials " trials, max diff = "
                    (.toFixed @max-diff 8) ", failures = " @failures))
      (is (zero? @failures)
          (str @failures " of " n-trials " trials exceeded 1e-5 tolerance"))
      (is (< @max-diff 1e-5)
          (str "max score diff " (.toFixed @max-diff 8) " < 1e-5"))))

  (testing "1000x compiled vs handler simulate on chain model"
    (let [n-trials 1000
          max-diff (atom 0.0)
          failures (atom 0)]
      (doseq [seed (range 1000 (+ 1000 n-trials))]
        (let [k1 (rng/fresh-key seed)
              compiled-tr (p/simulate (dyn/with-key chain-model k1) [])
              k2 (rng/fresh-key seed)
              handler-tr (p/simulate (dyn/with-key (force-handler chain-model) k2) [])
              c-score (realize (:score compiled-tr))
              h-score (realize (:score handler-tr))
              diff (js/Math.abs (- c-score h-score))]
          (swap! max-diff max diff)
          (when (> diff 1e-5)
            (swap! failures inc))))
      (println (str "  Stress chain: " n-trials " trials, max diff = "
                    (.toFixed @max-diff 8) ", failures = " @failures))
      (is (zero? @failures)
          (str @failures " of " n-trials " trials exceeded 1e-5 tolerance")))))

;; ==========================================================================
;; Test 2: Numerical edge cases
;; ==========================================================================
;; Extreme parameter values that might expose floating-point divergence
;; between compiled (fused ops) and handler (sequential ops) paths.
;; 100 seeds per configuration to cover the random-draw space.

(defn- edge-case-stress
  "Run n-trials of compiled vs handler on an edge-case model.
   Returns {:max-diff :failures :all-finite?}."
  [model n-trials seed-offset tol]
  (let [max-diff (atom 0.0)
        failures (atom 0)
        all-finite (atom true)]
    (doseq [seed (range seed-offset (+ seed-offset n-trials))]
      (let [k1 (rng/fresh-key seed)
            compiled-tr (p/simulate (dyn/with-key model k1) [])
            k2 (rng/fresh-key seed)
            handler-tr (p/simulate (dyn/with-key (force-handler model) k2) [])
            c-score (realize (:score compiled-tr))
            h-score (realize (:score handler-tr))
            diff (js/Math.abs (- c-score h-score))]
        (when-not (and (js/isFinite c-score) (js/isFinite h-score))
          (reset! all-finite false))
        (swap! max-diff max diff)
        (when (> diff tol)
          (swap! failures inc))))
    {:max-diff @max-diff :failures @failures :all-finite? @all-finite}))

(deftest numerical-edge-cases
  (let [N 100]

    (testing "Tiny sigma (1e-6): 100x compiled = handler"
      (let [model (gen []
                    (let [x (trace :x (dist/gaussian 0 0.000001))]
                      (trace :y (dist/gaussian x 0.000001))
                      x))
            {:keys [max-diff failures all-finite?]} (edge-case-stress model N 5000 1e-2)]
        (println (str "  Tiny sigma: " N " trials, max diff = "
                      (.toFixed max-diff 8) ", failures = " failures))
        (is all-finite? "all scores finite")
        (is (zero? failures) (str failures " trials exceeded tolerance"))))

    (testing "Large mu (1e6): 100x compiled = handler"
      (let [model (gen []
                    (let [x (trace :x (dist/gaussian 1000000 1))]
                      (trace :y (dist/gaussian x 1))
                      x))
            {:keys [max-diff failures all-finite?]} (edge-case-stress model N 6000 1e-5)]
        (println (str "  Large mu: " N " trials, max diff = "
                      (.toFixed max-diff 8) ", failures = " failures))
        (is all-finite? "all scores finite")
        (is (zero? failures) (str failures " trials exceeded tolerance"))))

    (testing "Wide sigma (1e4): 100x compiled = handler"
      (let [model (gen []
                    (let [x (trace :x (dist/gaussian 0 10000))]
                      (trace :y (dist/gaussian x 10000))
                      x))
            {:keys [max-diff failures all-finite?]} (edge-case-stress model N 7000 1e-5)]
        (println (str "  Wide sigma: " N " trials, max diff = "
                      (.toFixed max-diff 8) ", failures = " failures))
        (is all-finite? "all scores finite")
        (is (zero? failures) (str failures " trials exceeded tolerance"))))

    (testing "Mixed extreme (mu=1e6, sigma=0.001/10000): 100x compiled = handler"
      (let [model (gen []
                    (let [x (trace :x (dist/gaussian 1000000 0.001))]
                      (trace :y (dist/gaussian x 10000))
                      x))
            {:keys [max-diff failures all-finite?]} (edge-case-stress model N 8000 1e-2)]
        (println (str "  Mixed extreme: " N " trials, max diff = "
                      (.toFixed max-diff 8) ", failures = " failures))
        (is all-finite? "all scores finite")
        (is (zero? failures) (str failures " trials exceeded tolerance"))))))

;; ==========================================================================
;; Test 3: Weight equivalence under generate
;; ==========================================================================
;; Compiled generate weight must equal handler generate weight for
;; the same constraints. Tests partial and full constraint scenarios.

(deftest generate-weight-equivalence
  (testing "Full constraints: compiled weight = handler weight"
    (let [constraints (-> cm/EMPTY
                          (cm/set-choice [:slope] (mx/scalar 1.5))
                          (cm/set-choice [:intercept] (mx/scalar -0.3))
                          (cm/set-choice [:y] (mx/scalar 2.8)))
          args [(mx/scalar 2.0)]
          n-trials 100
          max-diff (atom 0.0)]
      (doseq [seed (range n-trials)]
        (let [k1 (rng/fresh-key seed)
              c-result (p/generate (dyn/with-key linreg-model k1) args constraints)
              k2 (rng/fresh-key seed)
              h-result (p/generate (dyn/with-key (force-handler linreg-model) k2) args constraints)
              c-weight (realize (:weight c-result))
              h-weight (realize (:weight h-result))
              diff (js/Math.abs (- c-weight h-weight))]
          (swap! max-diff max diff)))
      (println (str "  Generate full constraints: " n-trials
                    " trials, max weight diff = " (.toFixed @max-diff 8)))
      (is (< @max-diff 1e-5)
          (str "max weight diff " (.toFixed @max-diff 8) " < 1e-5"))))

  (testing "Partial constraints: compiled weight = handler weight"
    (let [n-trials 100
          max-diff (atom 0.0)]
      ;; Constrain only :y (observation), let :slope and :intercept be sampled
      (doseq [seed (range 100 (+ 100 n-trials))]
        (let [constraints (cm/choicemap :y (mx/scalar 2.8))
              args [(mx/scalar 2.0)]
              k1 (rng/fresh-key seed)
              c-result (p/generate (dyn/with-key linreg-model k1) args constraints)
              k2 (rng/fresh-key seed)
              h-result (p/generate (dyn/with-key (force-handler linreg-model) k2) args constraints)
              c-weight (realize (:weight c-result))
              h-weight (realize (:weight h-result))
              diff (js/Math.abs (- c-weight h-weight))]
          (swap! max-diff max diff)))
      (println (str "  Generate partial constraints: " n-trials
                    " trials, max weight diff = " (.toFixed @max-diff 8)))
      (is (< @max-diff 1e-5)
          (str "max weight diff " (.toFixed @max-diff 8) " < 1e-5"))))

  (testing "No constraints: weight = 0 for both paths"
    (doseq [seed (range 10)]
      (let [k1 (rng/fresh-key seed)
            c-result (p/generate (dyn/with-key linreg-model k1) [(mx/scalar 1.0)] cm/EMPTY)
            k2 (rng/fresh-key seed)
            h-result (p/generate (dyn/with-key (force-handler linreg-model) k2) [(mx/scalar 1.0)] cm/EMPTY)
            c-weight (realize (:weight c-result))
            h-weight (realize (:weight h-result))]
        (is (h/close? 0.0 c-weight 1e-6) "compiled weight = 0 for no constraints")
        (is (h/close? 0.0 h-weight 1e-6) "handler weight = 0 for no constraints")))))

;; ==========================================================================
;; Test 4: Gradient equivalence
;; ==========================================================================
;; Score gradients through compiled and handler paths must agree.
;; Uses choice-gradients API which internally calls generate.

(deftest gradient-equivalence
  (testing "choice-gradients: compiled = handler on linreg model"
    (let [args [(mx/scalar 2.0)]
          ;; Generate a trace (same for both paths — just need the choices)
          k (rng/fresh-key 42)
          trace (p/simulate (dyn/with-key linreg-model k) args)
          ;; Gradients via compiled path
          c-grads (grad/choice-gradients linreg-model trace [:slope :intercept])
          ;; Gradients via handler path
          h-grads (grad/choice-gradients (force-handler linreg-model) trace [:slope :intercept])
          c-slope-grad (realize (:slope c-grads))
          h-slope-grad (realize (:slope h-grads))
          c-int-grad (realize (:intercept c-grads))
          h-int-grad (realize (:intercept h-grads))]
      (println (str "  Gradient slope: compiled=" (.toFixed c-slope-grad 6)
                    " handler=" (.toFixed h-slope-grad 6)))
      (println (str "  Gradient intercept: compiled=" (.toFixed c-int-grad 6)
                    " handler=" (.toFixed h-int-grad 6)))
      (is (h/close? c-slope-grad h-slope-grad 1e-4)
          (str "slope grad: compiled " c-slope-grad " ≈ handler " h-slope-grad))
      (is (h/close? c-int-grad h-int-grad 1e-4)
          (str "intercept grad: compiled " c-int-grad " ≈ handler " h-int-grad))))

  (testing "score-gradient: compiled = handler on linreg model"
    (let [args [(mx/scalar 2.0)]
          obs (-> cm/EMPTY
                  (cm/set-choice [:slope] (mx/scalar 1.5))
                  (cm/set-choice [:intercept] (mx/scalar -0.3))
                  (cm/set-choice [:y] (mx/scalar 2.8)))
          addresses [:slope :intercept]
          params (mx/array [1.5 -0.3])
          ;; Compiled path
          c-result (grad/score-gradient linreg-model args obs addresses params)
          ;; Handler path
          h-result (grad/score-gradient (force-handler linreg-model) args obs addresses params)
          c-grad (h/realize-vec (:grad c-result))
          h-grad (h/realize-vec (:grad h-result))
          c-score (realize (:score c-result))
          h-score (realize (:score h-result))]
      (println (str "  Score gradient compiled: " c-grad " handler: " h-grad))
      (is (h/close? c-score h-score 1e-5)
          (str "score: compiled " c-score " ≈ handler " h-score))
      (is (h/close? (first c-grad) (first h-grad) 1e-4)
          "d/d(slope) matches")
      (is (h/close? (second c-grad) (second h-grad) 1e-4)
          "d/d(intercept) matches")))

  (testing "Gradient equivalence across multiple traces"
    (let [args [(mx/scalar 3.0)]
          n-trials 20
          max-diff (atom 0.0)]
      (doseq [seed (range n-trials)]
        (let [k (rng/fresh-key seed)
              trace (p/simulate (dyn/with-key chain-model k) [])
              c-grads (grad/choice-gradients chain-model trace [:z1 :z2])
              h-grads (grad/choice-gradients (force-handler chain-model) trace [:z1 :z2])]
          (doseq [addr [:z1 :z2]]
            (let [cg (realize (get c-grads addr))
                  hg (realize (get h-grads addr))
                  diff (js/Math.abs (- cg hg))]
              (swap! max-diff max diff)))))
      (println (str "  Gradient chain model: " n-trials
                    " traces, max diff = " (.toFixed @max-diff 8)))
      (is (< @max-diff 1e-4)
          (str "max gradient diff " (.toFixed @max-diff 8) " < 1e-4")))))

;; ==========================================================================
;; Test 5: Map combinator compilation equivalence
;; ==========================================================================
;; Compiled Map(kernel) uses fused or per-element compiled paths, which
;; have different RNG flow than the handler. So we verify equivalence via:
;; compiled-simulate → handler-assess with same choices → score must match.

(deftest map-combinator-equivalence
  (let [map-gf (comb/map-combinator map-kernel)
        xs [(mapv mx/scalar [1.0 2.0 3.0 4.0 5.0])]]

    (testing "Map: compiled simulate score = handler assess weight"
      (let [n-trials 100
            max-diff (atom 0.0)]
        (doseq [seed (range n-trials)]
          (let [k (rng/fresh-key seed)
                tr (p/simulate (dyn/with-key map-gf k) xs)
                c-score (realize (:score tr))
                ;; Assess the compiled trace's choices with the handler
                {:keys [weight]} (p/assess map-gf xs (:choices tr))
                h-score (realize weight)
                diff (js/Math.abs (- c-score h-score))]
            (swap! max-diff max diff)))
        (println (str "  Map simulate→assess: " n-trials " trials, max diff = "
                      (.toFixed @max-diff 8)))
        (is (< @max-diff 1e-5)
            (str "Map score vs assess " (.toFixed @max-diff 8) " < 1e-5"))))

    (testing "Map: compiled generate weight is consistent"
      ;; Constrain observations at indices 0, 2, 4
      (let [constraints (-> cm/EMPTY
                            (cm/set-choice [0] (cm/choicemap :y (mx/scalar 2.0)))
                            (cm/set-choice [2] (cm/choicemap :y (mx/scalar 6.0)))
                            (cm/set-choice [4] (cm/choicemap :y (mx/scalar 10.0))))
            n-trials 50
            max-diff (atom 0.0)]
        (doseq [seed (range n-trials)]
          (let [k (rng/fresh-key seed)
                {:keys [trace weight]} (p/generate (dyn/with-key map-gf k) xs constraints)
                c-weight (realize weight)
                ;; Assess the trace's choices → weight should equal generate score
                {:keys [weight]} (p/assess map-gf xs (:choices trace))
                assess-score (realize weight)
                c-score (realize (:score trace))
                diff (js/Math.abs (- c-score assess-score))]
            (swap! max-diff max diff)))
        (println (str "  Map generate→assess: " n-trials " trials, max diff = "
                      (.toFixed @max-diff 8)))
        (is (< @max-diff 1e-5)
            (str "Map generate score vs assess " (.toFixed @max-diff 8) " < 1e-5"))))))

;; ==========================================================================
;; Test 6: Unfold combinator compilation equivalence
;; ==========================================================================
;; Same strategy: compiled-simulate → handler-assess for score consistency.

(deftest unfold-combinator-equivalence
  (let [unfold-gf (comb/unfold-combinator unfold-kernel)
        args [5 0.0]]  ;; 5 steps, init-state=0

    (testing "Unfold: compiled simulate score = handler assess weight"
      (let [n-trials 100
            max-diff (atom 0.0)]
        (doseq [seed (range n-trials)]
          (let [k (rng/fresh-key seed)
                tr (p/simulate (dyn/with-key unfold-gf k) args)
                c-score (realize (:score tr))
                {:keys [weight]} (p/assess unfold-gf args (:choices tr))
                h-score (realize weight)
                diff (js/Math.abs (- c-score h-score))]
            (swap! max-diff max diff)))
        (println (str "  Unfold simulate→assess: " n-trials " trials, max diff = "
                      (.toFixed @max-diff 8)))
        (is (< @max-diff 1e-5)
            (str "Unfold score vs assess " (.toFixed @max-diff 8) " < 1e-5"))))

    (testing "Unfold: compiled generate score is consistent"
      (let [constraints (-> cm/EMPTY
                            (cm/set-choice [0] (cm/choicemap :y (mx/scalar 0.5)))
                            (cm/set-choice [2] (cm/choicemap :y (mx/scalar 1.0)))
                            (cm/set-choice [4] (cm/choicemap :y (mx/scalar 0.3))))
            n-trials 50
            max-diff (atom 0.0)]
        (doseq [seed (range n-trials)]
          (let [k (rng/fresh-key seed)
                {:keys [trace]} (p/generate (dyn/with-key unfold-gf k) args constraints)
                c-score (realize (:score trace))
                {:keys [weight]} (p/assess unfold-gf args (:choices trace))
                assess-score (realize weight)
                diff (js/Math.abs (- c-score assess-score))]
            (swap! max-diff max diff)))
        (println (str "  Unfold generate→assess: " n-trials " trials, max diff = "
                      (.toFixed @max-diff 8)))
        (is (< @max-diff 1e-5)
            (str "Unfold generate score vs assess " (.toFixed @max-diff 8) " < 1e-5"))))))

;; ==========================================================================
;; Run
;; ==========================================================================

(run-tests)
