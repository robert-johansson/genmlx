(ns genmlx.auto-wiring-test
  "WP-2 Gate 2: Auto-wiring tests — verify that DynamicGF automatically
   detects conjugate pairs and uses analytical handlers in p/generate.
   Side-by-side comparison: auto-analytical vs manual analytical vs standard."
  (:require [genmlx.gen :refer [gen]]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]
            [genmlx.conjugacy :as conjugacy]
            [genmlx.inference.auto-analytical :as auto-analytical]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn- assert-true [desc pred]
  (if pred
    (do (swap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " desc)))))

(defn- assert-close [desc expected actual tol]
  (let [diff (Math/abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass-count inc)
          (println (str "  PASS: " desc " (diff=" (.toExponential diff 2) ")")))
      (do (swap! fail-count inc)
          (println (str "  FAIL: " desc " expected=" expected " actual=" actual
                        " diff=" (.toExponential diff 2) " tol=" tol))))))

;; ---------------------------------------------------------------------------
;; Section 1: Schema augmentation in make-gen-fn
;; ---------------------------------------------------------------------------

(println "\n=== Section 1: Schema Auto-Augmentation ===")

;; Normal-Normal model
(def nn-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 2))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      mu)))

(let [schema (:schema nn-model)]
  (assert-true "NN: has-conjugate?" (:has-conjugate? schema))
  (assert-true "NN: 2 conjugate pairs" (= 2 (count (:conjugate-pairs schema))))
  (assert-true "NN: auto-handlers present" (map? (:auto-handlers schema)))
  (assert-true "NN: handler for :mu" (fn? (get (:auto-handlers schema) :mu)))
  (assert-true "NN: handler for :y1" (fn? (get (:auto-handlers schema) :y1)))
  (assert-true "NN: handler for :y2" (fn? (get (:auto-handlers schema) :y2))))

;; Beta-Bernoulli model
(def bb-model
  (gen []
    (let [p (trace :p (dist/beta-dist 2 3))]
      (trace :x1 (dist/bernoulli p))
      (trace :x2 (dist/bernoulli p))
      p)))

(let [schema (:schema bb-model)]
  (assert-true "BB: has-conjugate?" (:has-conjugate? schema))
  (assert-true "BB: family is :beta-bernoulli"
               (every? #(= :beta-bernoulli (:family %)) (:conjugate-pairs schema)))
  (assert-true "BB: auto-handlers for :p :x1 :x2"
               (= #{:p :x1 :x2} (set (keys (:auto-handlers schema))))))

;; Gamma-Poisson model
(def gp-model
  (gen []
    (let [rate (trace :rate (dist/gamma-dist 3 2))]
      (trace :c1 (dist/poisson rate))
      rate)))

(let [schema (:schema gp-model)]
  (assert-true "GP: has-conjugate?" (:has-conjugate? schema))
  (assert-true "GP: family is :gamma-poisson"
               (= :gamma-poisson (-> schema :conjugate-pairs first :family)))
  (assert-true "GP: auto-handlers for :rate :c1"
               (= #{:rate :c1} (set (keys (:auto-handlers schema))))))

;; Gamma-Exponential model
(def ge-model
  (gen []
    (let [rate (trace :rate (dist/gamma-dist 2 1))]
      (trace :t1 (dist/exponential rate))
      rate)))

(let [schema (:schema ge-model)]
  (assert-true "GE: has-conjugate?" (:has-conjugate? schema))
  (assert-true "GE: family is :gamma-exponential"
               (= :gamma-exponential (-> schema :conjugate-pairs first :family))))

;; Non-conjugate model — should NOT get auto-handlers
(def non-conj-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 1))]
      (trace :x (dist/bernoulli (mx/sigmoid mu)))
      mu)))

(let [schema (:schema non-conj-model)]
  (assert-true "non-conj: no conjugate pairs" (not (:has-conjugate? schema)))
  (assert-true "non-conj: no auto-handlers" (nil? (:auto-handlers schema))))

;; ---------------------------------------------------------------------------
;; Section 2: Gate 2 — Auto-Analytical Generate (NN)
;; ---------------------------------------------------------------------------

(println "\n=== Section 2: Gate 2 — Normal-Normal Auto Generate ===")

(let [model (dyn/auto-key nn-model)
      constraints (-> cm/EMPTY
                      (cm/set-value :y1 (mx/scalar 1.0))
                      (cm/set-value :y2 (mx/scalar 2.0)))
      {:keys [trace weight]} (p/generate model [] constraints)
      w (mx/item weight)
      s (mx/item (:score trace))
      ;; Manual marginal LL: N(0,4) prior, N(mu,1) obs
      ;; y1 marginal: N(0, 5), y2 marginal: N(post_mean, post_var + 1)
      log-2pi 1.8378770664093453
      ll-1 (* -0.5 (+ log-2pi (Math/log 5.0) (/ 1.0 5.0)))
      ;; Post after y1: var=4/5, mean=4/5
      post-var-1 0.8
      post-mean-1 0.8
      marg-var-2 (+ post-var-1 1.0)
      diff-2 (- 2.0 post-mean-1)
      ll-2 (* -0.5 (+ log-2pi (Math/log marg-var-2) (/ (* diff-2 diff-2) marg-var-2)))
      expected (+ ll-1 ll-2)]
  (assert-close "NN-auto: weight matches marginal LL" expected w 1e-5)
  (assert-close "NN-auto: score = weight" w s 1e-8)
  (assert-true "NN-auto: trace has :mu" (cm/has-value? (cm/get-submap (:choices trace) :mu)))
  (assert-true "NN-auto: trace has :y1" (cm/has-value? (cm/get-submap (:choices trace) :y1)))
  (assert-true "NN-auto: trace has :y2" (cm/has-value? (cm/get-submap (:choices trace) :y2)))
  (assert-close "NN-auto: y1 = 1.0" 1.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :y1))) 1e-8)
  (assert-close "NN-auto: y2 = 2.0" 2.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :y2))) 1e-8))

;; ---------------------------------------------------------------------------
;; Section 3: Gate 2 — Auto-Analytical Generate (BB)
;; ---------------------------------------------------------------------------

(println "\n=== Section 3: Gate 2 — Beta-Bernoulli Auto Generate ===")

(let [model (dyn/auto-key bb-model)
      constraints (-> cm/EMPTY
                      (cm/set-value :x1 (mx/scalar 1.0))
                      (cm/set-value :x2 (mx/scalar 0.0)))
      {:keys [trace weight]} (p/generate model [] constraints)
      w (mx/item weight)
      ;; Manual: Beta(2,3), x1=1 → ll=log(2/5), x2=0 → ll=log(3/6)
      expected (+ (Math/log (/ 2.0 5.0)) (Math/log (/ 3.0 6.0)))]
  (assert-close "BB-auto: weight matches marginal LL" expected w 1e-5)
  (assert-close "BB-auto: x1 = 1.0" 1.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x1))) 1e-8)
  (assert-close "BB-auto: x2 = 0.0" 0.0 (mx/item (cm/get-value (cm/get-submap (:choices trace) :x2))) 1e-8))

;; ---------------------------------------------------------------------------
;; Section 4: Gate 2 — Auto-Analytical Generate (GP)
;; ---------------------------------------------------------------------------

(println "\n=== Section 4: Gate 2 — Gamma-Poisson Auto Generate ===")

(let [model (dyn/auto-key gp-model)
      constraints (-> cm/EMPTY (cm/set-value :c1 (mx/scalar 5.0)))
      {:keys [trace weight]} (p/generate model [] constraints)
      w (mx/item weight)
      ;; Manual: Gamma(3,2), k=5
      a 3.0 b 2.0 k 5.0
      expected (+ (- (mx/item (mx/lgamma (mx/scalar (+ a k))))
                     (mx/item (mx/lgamma (mx/scalar a)))
                     (mx/item (mx/lgamma (mx/scalar (+ k 1.0)))))
                  (* a (- (Math/log b) (Math/log (+ b 1.0))))
                  (* k (- 0 (Math/log (+ b 1.0)))))]
  (assert-close "GP-auto: weight matches marginal LL" expected w 1e-5))

;; ---------------------------------------------------------------------------
;; Section 5: Gate 2 — Auto-Analytical Generate (GE)
;; ---------------------------------------------------------------------------

(println "\n=== Section 5: Gate 2 — Gamma-Exponential Auto Generate ===")

(let [model (dyn/auto-key ge-model)
      constraints (-> cm/EMPTY (cm/set-value :t1 (mx/scalar 0.5)))
      {:keys [trace weight]} (p/generate model [] constraints)
      w (mx/item weight)
      ;; Manual: Gamma(2,1), x=0.5
      ;; Lomax: log(shape) + shape*log(rate) - (shape+1)*log(rate+x)
      a 2.0 b 1.0 x 0.5
      expected (+ (Math/log a) (* a (Math/log b))
                  (- (* (+ a 1.0) (Math/log (+ b x)))))]
  (assert-close "GE-auto: weight matches marginal LL" expected w 1e-5))

;; ---------------------------------------------------------------------------
;; Section 6: Fallthrough — unconstrained obs use standard handler
;; ---------------------------------------------------------------------------

(println "\n=== Section 6: Fallthrough Behavior ===")

;; No obs constrained → standard handler (weight ≠ score for unconstrained sites)
(let [model (dyn/auto-key nn-model)
      {:keys [trace weight]} (p/generate model [] cm/EMPTY)
      w (mx/item weight)
      s (mx/item (:score trace))]
  (assert-close "fallthrough: weight = 0 (no constraints)" 0.0 w 1e-8)
  (assert-true "fallthrough: score < 0 (has log-probs)" (< s 0)))

;; Only prior constrained, no obs → standard handler
(let [model (dyn/auto-key nn-model)
      constraints (-> cm/EMPTY (cm/set-value :mu (mx/scalar 0.5)))
      {:keys [trace weight]} (p/generate model [] constraints)
      w (mx/item weight)
      s (mx/item (:score trace))]
  (assert-true "prior-only: weight ≠ 0 (prior constrained)" (not= 0.0 w))
  (assert-true "prior-only: weight ≠ score" (> (Math/abs (- w s)) 0.01)))

;; Partial obs constrained — only some obs are analytical
(def nn-model-3obs
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 2))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      (trace :y3 (dist/gaussian mu 1))
      mu)))

(let [model (dyn/auto-key nn-model-3obs)
      ;; Only constrain y1 and y3, leave y2 unconstrained
      constraints (-> cm/EMPTY
                      (cm/set-value :y1 (mx/scalar 1.0))
                      (cm/set-value :y3 (mx/scalar 2.0)))
      {:keys [trace weight]} (p/generate model [] constraints)]
  (assert-true "partial-obs: trace has all sites"
               (and (cm/has-value? (cm/get-submap (:choices trace) :mu))
                    (cm/has-value? (cm/get-submap (:choices trace) :y1))
                    (cm/has-value? (cm/get-submap (:choices trace) :y2))
                    (cm/has-value? (cm/get-submap (:choices trace) :y3))))
  (assert-close "partial-obs: y1 = 1.0" 1.0
                (mx/item (cm/get-value (cm/get-submap (:choices trace) :y1))) 1e-8)
  (assert-close "partial-obs: y3 = 2.0" 2.0
                (mx/item (cm/get-value (cm/get-submap (:choices trace) :y3))) 1e-8))

;; ---------------------------------------------------------------------------
;; Section 7: Multi-observation consistency
;; ---------------------------------------------------------------------------

(println "\n=== Section 7: Multi-Observation ===")

;; 5-obs Normal-Normal
(def nn-5obs
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 2))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      (trace :y3 (dist/gaussian mu 1))
      (trace :y4 (dist/gaussian mu 1))
      (trace :y5 (dist/gaussian mu 1))
      mu)))

(let [model (dyn/auto-key nn-5obs)
      constraints (-> cm/EMPTY
                      (cm/set-value :y1 (mx/scalar 1.0))
                      (cm/set-value :y2 (mx/scalar 1.5))
                      (cm/set-value :y3 (mx/scalar 0.5))
                      (cm/set-value :y4 (mx/scalar 2.0))
                      (cm/set-value :y5 (mx/scalar 1.2)))
      {:keys [trace weight]} (p/generate model [] constraints)
      w (mx/item weight)]
  (assert-true "5-obs NN: weight is finite" (js/isFinite w))
  (assert-true "5-obs NN: weight < 0" (< w 0))
  (assert-true "5-obs NN: all obs in trace"
               (every? #(cm/has-value? (cm/get-submap (:choices trace) %))
                       [:y1 :y2 :y3 :y4 :y5])))

;; ---------------------------------------------------------------------------
;; Section 8: Mixed conjugate families in same model
;; ---------------------------------------------------------------------------

(println "\n=== Section 8: Mixed Families ===")

(def mixed-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 2))
          p  (trace :p (dist/beta-dist 2 3))]
      (trace :y (dist/gaussian mu 1))
      (trace :x (dist/bernoulli p))
      [mu p])))

(let [schema (:schema mixed-model)]
  (assert-true "mixed: has-conjugate?" (:has-conjugate? schema))
  (assert-true "mixed: 2 pairs (NN + BB)" (= 2 (count (:conjugate-pairs schema))))
  (assert-true "mixed: handlers for all 4 addrs"
               (= #{:mu :y :p :x} (set (keys (:auto-handlers schema))))))

(let [model (dyn/auto-key mixed-model)
      constraints (-> cm/EMPTY
                      (cm/set-value :y (mx/scalar 1.0))
                      (cm/set-value :x (mx/scalar 1.0)))
      {:keys [trace weight]} (p/generate model [] constraints)
      w (mx/item weight)
      ;; NN marginal: N(0,5), y=1 → ll = -0.5*(log(2pi) + log(5) + 1/5)
      ;; BB marginal: Beta(2,3), x=1 → ll = log(2/5)
      log-2pi 1.8378770664093453
      nn-ll (* -0.5 (+ log-2pi (Math/log 5.0) (/ 1.0 5.0)))
      bb-ll (Math/log (/ 2.0 5.0))
      expected (+ nn-ll bb-ll)]
  (assert-close "mixed: weight matches sum of marginal LLs" expected w 1e-5))

;; ---------------------------------------------------------------------------
;; Results
;; ---------------------------------------------------------------------------

(println (str "\n=== RESULTS: " @pass-count "/" (+ @pass-count @fail-count)
              " passed, " @fail-count " failed ==="))
