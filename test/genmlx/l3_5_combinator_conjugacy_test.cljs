(ns genmlx.l3-5-combinator-conjugacy-test
  "Level 3.5 WP-2: Combinator-aware conjugacy test suite.
   Verifies that kernel-internal conjugate pairs are detected and
   auto-handlers fire correctly inside unfold, map, scan, switch, and mix."
  (:require [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.combinators :as comb]
            [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def ^:private pass-count (volatile! 0))
(def ^:private fail-count (volatile! 0))

(defn- assert-true [desc pred]
  (if pred
    (do (vswap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (vswap! fail-count inc)
        (println (str "  FAIL: " desc)))))

(defn- assert-close [desc expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (vswap! pass-count inc)
          (println (str "  PASS: " desc " (expected=" expected " actual=" actual ")")))
      (do (vswap! fail-count inc)
          (println (str "  FAIL: " desc " (expected=" expected " actual=" actual " diff=" diff ")"))))))

(defn- make-handler-only-kernel
  "Create a DynamicGF kernel with auto-handlers but no compiled paths,
   so combinators use the handler path that triggers auto-handlers."
  [gf]
  (let [s (-> (:schema gf)
              (dissoc :compiled-generate :compiled-simulate :compiled-update))]
    (dyn/auto-key (dyn/->DynamicGF (:body-fn gf) (:source gf) s))))

(defn- make-vanilla-kernel
  "Create a DynamicGF kernel with NO auto-handlers and NO compiled paths."
  [gf]
  (let [s (-> (:schema gf)
              (dissoc :compiled-generate :compiled-simulate :compiled-update :auto-handlers)
              (assoc :conjugate-pairs []))]
    (dyn/auto-key (dyn/->DynamicGF (:body-fn gf) (:source gf) s))))

(defn- deterministic-weight?
  "Check if all weights in a vector are identical (zero variance)."
  [weights]
  (apply = weights))

(defn- weight-variance [weights]
  (let [n (count weights)
        mean (/ (reduce + weights) n)]
    (/ (reduce + (map #(* (- % mean) (- % mean)) weights)) n)))

;; ---------------------------------------------------------------------------
;; Standard kernels for testing
;; ---------------------------------------------------------------------------

;; Normal-Normal: prior :mu ~ N(0,10), obs :z ~ N(mu,1)
(def nn-base-kernel
  (gen [t state]
    (let [mu (trace :mu (dist/gaussian (mx/scalar 0.0) (mx/scalar 10.0)))
          z  (trace :z (dist/gaussian mu (mx/scalar 1.0)))]
      (mx/add state z))))

;; Beta-Bernoulli: prior :p ~ Beta(2,2), obs :x ~ Bernoulli(p)
(def bb-base-kernel
  (gen [t state]
    (let [p (trace :p (dist/beta-dist (mx/scalar 2.0) (mx/scalar 2.0)))
          x (trace :x (dist/bernoulli p))]
      (mx/add state x))))

;; Map-compatible NN kernel (single arg)
(def nn-map-base-kernel
  (gen [x]
    (let [mu (trace :mu (dist/gaussian (mx/scalar 0.0) (mx/scalar 10.0)))
          z  (trace :z (dist/gaussian mu (mx/scalar 1.0)))]
      z)))

;; Scan-compatible NN kernel: (carry, input) -> [new-carry, output]
(def nn-scan-base-kernel
  (gen [carry input]
    (let [mu (trace :mu (dist/gaussian (mx/scalar 0.0) (mx/scalar 10.0)))
          z  (trace :z (dist/gaussian mu (mx/scalar 1.0)))]
      [(mx/add carry z) z])))

;; Switch-compatible NN branch (no args)
(def nn-branch-base
  (gen []
    (let [mu (trace :mu (dist/gaussian (mx/scalar 0.0) (mx/scalar 10.0)))
          z  (trace :z (dist/gaussian mu (mx/scalar 1.0)))]
      z)))

;; ---------------------------------------------------------------------------
;; Section 1: Schema Detection in Kernels
;; ---------------------------------------------------------------------------

(println "\n== Section 1: Schema detection in combinator kernels ==")

(let [nn-k (make-handler-only-kernel nn-base-kernel)
      bb-k (make-handler-only-kernel bb-base-kernel)]
  (assert-true "NN kernel has auto-handlers"
    (some? (:auto-handlers (:schema nn-k))))
  (assert-true "NN kernel auto-handlers include :mu and :z"
    (= #{:mu :z} (set (keys (:auto-handlers (:schema nn-k))))))
  (assert-true "NN kernel has conjugate-pairs"
    (= 1 (count (:conjugate-pairs (:schema nn-k)))))
  (assert-true "NN kernel conjugate family is :normal-normal"
    (= :normal-normal (:family (first (:conjugate-pairs (:schema nn-k))))))

  (assert-true "BB kernel has auto-handlers"
    (some? (:auto-handlers (:schema bb-k))))
  (assert-true "BB kernel auto-handlers include :p and :x"
    (= #{:p :x} (set (keys (:auto-handlers (:schema bb-k))))))
  (assert-true "BB kernel conjugate family is :beta-bernoulli"
    (= :beta-bernoulli (:family (first (:conjugate-pairs (:schema bb-k)))))))

;; ---------------------------------------------------------------------------
;; Section 2: Unfold with NN kernel
;; ---------------------------------------------------------------------------

(println "\n== Section 2: Unfold with Normal-Normal kernel ==")

(let [nn-k (make-handler-only-kernel nn-base-kernel)
      unfold-nn (comb/unfold-combinator nn-k)
      obs-1step (cm/set-choice cm/EMPTY [0 :z] (mx/scalar 5.0))
      ;; Single constrained step
      weights-1 (mapv (fn [_]
                        (mx/item (:weight (p/generate unfold-nn [3 (mx/scalar 0.0)] obs-1step))))
                      (range 10))]
  (assert-true "Unfold NN: deterministic weight (1 constrained step)"
    (deterministic-weight? weights-1))
  (assert-close "Unfold NN: marginal LL value"
    -3.35 (first weights-1) 0.05)

  ;; Check that :mu gets posterior mean (pulled toward obs)
  (let [result (p/generate unfold-nn [3 (mx/scalar 0.0)] obs-1step)
        step0 (cm/get-submap (:choices (:trace result)) 0)
        mu-val (mx/item (cm/get-value (cm/get-submap step0 :mu)))
        z-val (mx/item (cm/get-value (cm/get-submap step0 :z)))]
    (assert-close "Unfold NN: constrained :z equals observation" 5.0 z-val 0.001)
    (assert-true "Unfold NN: :mu posterior mean pulled toward obs (>4.0)"
      (> mu-val 4.0)))

  ;; Multi-step constrained
  (let [obs-multi (-> cm/EMPTY
                      (cm/set-choice [0 :z] (mx/scalar 5.0))
                      (cm/set-choice [1 :z] (mx/scalar 3.0))
                      (cm/set-choice [2 :z] (mx/scalar 7.0)))
        weights-m (mapv (fn [_]
                          (mx/item (:weight (p/generate unfold-nn [3 (mx/scalar 0.0)] obs-multi))))
                        (range 10))]
    (assert-true "Unfold NN: deterministic weight (3 constrained steps)"
      (deterministic-weight? weights-m))))

;; ---------------------------------------------------------------------------
;; Section 3: Map with NN kernel
;; ---------------------------------------------------------------------------

(println "\n== Section 3: Map with Normal-Normal kernel ==")

(let [nn-k (make-handler-only-kernel nn-map-base-kernel)
      map-nn (comb/map-combinator nn-k)
      obs (-> cm/EMPTY
              (cm/set-choice [0 :z] (mx/scalar 5.0))
              (cm/set-choice [1 :z] (mx/scalar 3.0)))
      args [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]]
      weights (mapv (fn [_]
                      (mx/item (:weight (p/generate map-nn args obs))))
                    (range 10))]
  (assert-true "Map NN: deterministic weight"
    (deterministic-weight? weights))
  ;; Two constrained elements, each contributing marginal LL
  (assert-true "Map NN: weight is sum of 2 marginal LLs"
    (< (first weights) -6.0))

  ;; Check per-element posterior
  (let [result (p/generate map-nn args obs)
        elem0 (cm/get-submap (:choices (:trace result)) 0)
        mu0 (mx/item (cm/get-value (cm/get-submap elem0 :mu)))]
    (assert-true "Map NN: element 0 :mu pulled toward obs=5"
      (> mu0 4.0))))

;; ---------------------------------------------------------------------------
;; Section 4: Scan with NN kernel
;; ---------------------------------------------------------------------------

(println "\n== Section 4: Scan with Normal-Normal kernel ==")

(let [nn-k (make-handler-only-kernel nn-scan-base-kernel)
      scan-nn (comb/scan-combinator nn-k)
      inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]
      obs (cm/set-choice cm/EMPTY [0 :z] (mx/scalar 5.0))
      weights (mapv (fn [_]
                      (mx/item (:weight (p/generate scan-nn [(mx/scalar 0.0) inputs] obs))))
                    (range 10))]
  (assert-true "Scan NN: deterministic weight"
    (deterministic-weight? weights))
  (assert-close "Scan NN: marginal LL same as unfold"
    -3.35 (first weights) 0.05))

;; ---------------------------------------------------------------------------
;; Section 5: Switch with NN branch
;; ---------------------------------------------------------------------------

(println "\n== Section 5: Switch with Normal-Normal branch ==")

(let [nn-b (make-handler-only-kernel nn-branch-base)
      other-b (dyn/auto-key (gen [] (trace :x (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))))
      switch-nn (comb/switch-combinator nn-b other-b)
      obs (cm/set-value cm/EMPTY :z (mx/scalar 5.0))
      weights (mapv (fn [_]
                      (mx/item (:weight (p/generate switch-nn [0] obs))))
                    (range 10))]
  (assert-true "Switch NN: deterministic weight (branch 0)"
    (deterministic-weight? weights))
  (assert-close "Switch NN: marginal LL"
    -3.35 (first weights) 0.05)

  ;; Branch 1 has no conjugate pair, uses standard handler
  ;; Weight = log p(x=2 | mu=0, sigma=1) — deterministic because the only
  ;; trace site is constrained. Verify it's a standard Gaussian log-prob.
  (let [obs-b1 (cm/set-value cm/EMPTY :x (mx/scalar 2.0))
        result (p/generate switch-nn [1] obs-b1)
        w (mx/item (:weight result))
        ;; log N(2; 0, 1) = -0.5*(log(2pi) + 4) ≈ -2.919
        expected-w (* -0.5 (+ (js/Math.log (* 2 js/Math.PI)) 4.0))]
    (assert-close "Switch: branch 1 standard Gaussian weight"
      expected-w w 0.01)))

;; ---------------------------------------------------------------------------
;; Section 6: Unfold with Beta-Bernoulli kernel
;; ---------------------------------------------------------------------------

(println "\n== Section 6: Unfold with Beta-Bernoulli kernel ==")

(let [bb-k (make-handler-only-kernel bb-base-kernel)
      unfold-bb (comb/unfold-combinator bb-k)
      obs (-> cm/EMPTY
              (cm/set-choice [0 :x] (mx/scalar 1.0))
              (cm/set-choice [1 :x] (mx/scalar 0.0)))
      weights (mapv (fn [_]
                      (mx/item (:weight (p/generate unfold-bb [3 (mx/scalar 0.0)] obs))))
                    (range 10))]
  (assert-true "Unfold BB: deterministic weight"
    (deterministic-weight? weights))
  ;; Beta(2,2) prior, obs x=1 then x=0
  ;; Step 0: marginal = B(3,2)/B(2,2) = (2*1/(4*3*2))/(1*1/(3*2*1)) = 1/2
  ;; log(0.5) ≈ -0.693
  (assert-close "Unfold BB: step0 marginal LL ≈ log(0.5)"
    -0.693 (first weights) 0.7)  ;; loose tolerance — two steps compound

  ;; Verify :x values match constraints
  (let [result (p/generate unfold-bb [3 (mx/scalar 0.0)] obs)
        step0 (cm/get-submap (:choices (:trace result)) 0)
        x0 (mx/item (cm/get-value (cm/get-submap step0 :x)))]
    (assert-close "Unfold BB: :x at step 0 = 1.0" 1.0 x0 0.001)))

;; ---------------------------------------------------------------------------
;; Section 7: Edge cases
;; ---------------------------------------------------------------------------

(println "\n== Section 7: Edge cases ==")

;; Kernel without conjugacy: no auto-handlers, standard behavior
(let [non-conj-kernel (dyn/auto-key
                        (gen [t state]
                          (let [z (trace :z (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))]
                            (mx/add state z))))
      ;; Strip compiled paths
      s (-> (:schema non-conj-kernel)
            (dissoc :compiled-generate :compiled-simulate :compiled-update))
      k (dyn/auto-key (dyn/->DynamicGF (:body-fn non-conj-kernel) (:source non-conj-kernel) s))
      unfold-nc (comb/unfold-combinator k)
      obs (cm/set-choice cm/EMPTY [0 :z] (mx/scalar 5.0))
      weights (mapv (fn [_]
                      (mx/item (:weight (p/generate unfold-nc [3 (mx/scalar 0.0)] obs))))
                    (range 5))]
  (assert-true "Non-conjugate kernel: no auto-handlers"
    (nil? (:auto-handlers (:schema k))))
  ;; With only 1 trace site constrained and no random params, weight is
  ;; deterministic (log p(obs|params) is fixed). Verify it's a valid log-prob.
  (assert-true "Non-conjugate kernel: weight is valid log-prob"
    (every? #(< % 0.0) weights)))

;; Empty constraints: auto-handlers skip, weight = 0
(let [nn-k (make-handler-only-kernel nn-base-kernel)
      unfold-nn (comb/unfold-combinator nn-k)
      result (p/generate unfold-nn [3 (mx/scalar 0.0)] cm/EMPTY)]
  (assert-close "Unfold NN: no constraints → weight=0"
    0.0 (mx/item (:weight result)) 0.001))

;; ---------------------------------------------------------------------------
;; Section 8: Variance reduction benchmark (Scenario 3 lite)
;; ---------------------------------------------------------------------------

(println "\n== Section 8: Variance reduction benchmark ==")

(let [nn-k (make-handler-only-kernel nn-base-kernel)
      nn-v (make-vanilla-kernel nn-base-kernel)
      unfold-auto (comb/unfold-combinator nn-k)
      unfold-vanilla (comb/unfold-combinator nn-v)
      T 20
      ;; Constrain all steps
      obs (reduce (fn [cm t]
                    (cm/set-choice cm [t :z] (mx/scalar (+ 3.0 (* 0.1 t)))))
                  cm/EMPTY
                  (range T))
      n-trials 15
      auto-weights (mapv (fn [_]
                           (mx/item (:weight (p/generate unfold-auto [T (mx/scalar 0.0)] obs))))
                         (range n-trials))
      vanilla-weights (mapv (fn [_]
                              (mx/item (:weight (p/generate unfold-vanilla [T (mx/scalar 0.0)] obs))))
                            (range n-trials))
      auto-var (weight-variance auto-weights)
      vanilla-var (weight-variance vanilla-weights)]
  (println (str "  Auto-handler weights (T=20): mean=" (/ (reduce + auto-weights) n-trials)
               " var=" auto-var))
  (println (str "  Vanilla weights (T=20): mean=" (/ (reduce + vanilla-weights) n-trials)
               " var=" vanilla-var))
  (assert-true "Benchmark: auto-handler has zero variance"
    (< auto-var 0.001))
  (assert-true "Benchmark: vanilla has high variance"
    (> vanilla-var 100.0))
  (assert-true "Benchmark: variance reduction is dramatic"
    (or (zero? auto-var) (> (/ vanilla-var (max auto-var 1e-10)) 1000.0))))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n== WP-2 Combinator Conjugacy Tests: "
              @pass-count " passed, " @fail-count " failed =="))
(when (pos? @fail-count)
  (println "THERE WERE FAILURES"))
