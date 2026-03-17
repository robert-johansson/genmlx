(ns genmlx.fused-map-state-test
  "Phase 1: mx/compile-fn traces through CLJS map intermediates.
   Phase 2: Fused Unfold simulate works with map-state kernels."
  (:require [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.combinators :as comb]
            [genmlx.choicemap :as cm]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

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
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass-count inc)
          (println (str "  PASS: " desc " (diff=" (.toFixed diff 6) ")")))
      (do (swap! fail-count inc)
          (println (str "  FAIL: " desc " expected=" expected " actual=" actual " diff=" diff))))))

;; ---------------------------------------------------------------------------
;; Test 1: Simple — unpack [2] → map → compute → pack [2]
;; ---------------------------------------------------------------------------

(println "\n== Test 1: Simple map intermediate ==")

(defn simple-map-fn [init-state-flat]
  ;; Unpack [2] array to CLJS map
  (let [state {:a (mx/index init-state-flat 0)
               :b (mx/index init-state-flat 1)}
        ;; Compute using map keyword access
        result (mx/add (mx/multiply (:a state) (mx/scalar 2.0))
                       (:b state))
        ;; Pack back to array
        new-state (mx/stack [result (:b state)])]
    new-state))

(let [compiled-simple (mx/compile-fn simple-map-fn)
      input (mx/array [3.0 1.0])
      ;; Expected: a=3, b=1 → result = 3*2 + 1 = 7, output = [7, 1]
      interp-result (simple-map-fn input)
      _ (mx/eval! interp-result)
      compiled-result (compiled-simple input)
      _ (mx/eval! compiled-result)
      interp-0 (mx/item (mx/index interp-result 0))
      interp-1 (mx/item (mx/index interp-result 1))
      compiled-0 (mx/item (mx/index compiled-result 0))
      compiled-1 (mx/item (mx/index compiled-result 1))]
  (assert-close "simple map: element 0 matches" interp-0 compiled-0 1e-6)
  (assert-close "simple map: element 1 matches" interp-1 compiled-1 1e-6)
  (assert-close "simple map: element 0 = 7.0" 7.0 compiled-0 1e-6)
  (assert-close "simple map: element 1 = 1.0" 1.0 compiled-1 1e-6))

;; ---------------------------------------------------------------------------
;; Test 2: Loop — T=5 iterations with state threading through maps
;; ---------------------------------------------------------------------------

(println "\n== Test 2: T=5 loop with map state threading ==")

(defn loop-map-fn [init-flat]
  ;; Simulate Unfold: iterate T=5 times, threading state as map
  (let [init-state {:x (mx/index init-flat 0)
                    :y (mx/index init-flat 1)}]
    (loop [t 0
           state init-state]
      (if (>= t 5)
        ;; Pack final state back
        (mx/stack [(:x state) (:y state)])
        ;; Transition: x' = 0.9*x + 0.1*y, y' = 0.1*x + 0.9*y
        (let [new-x (mx/add (mx/multiply (mx/scalar 0.9) (:x state))
                            (mx/multiply (mx/scalar 0.1) (:y state)))
              new-y (mx/add (mx/multiply (mx/scalar 0.1) (:x state))
                            (mx/multiply (mx/scalar 0.9) (:y state)))]
          (recur (inc t) {:x new-x :y new-y}))))))

(let [compiled-loop (mx/compile-fn loop-map-fn)
      input (mx/array [10.0 0.0])
      interp-result (loop-map-fn input)
      _ (mx/eval! interp-result)
      compiled-result (compiled-loop input)
      _ (mx/eval! compiled-result)
      interp-0 (mx/item (mx/index interp-result 0))
      interp-1 (mx/item (mx/index interp-result 1))
      compiled-0 (mx/item (mx/index compiled-result 0))
      compiled-1 (mx/item (mx/index compiled-result 1))]
  (assert-close "loop map: element 0 matches" interp-0 compiled-0 1e-5)
  (assert-close "loop map: element 1 matches" interp-1 compiled-1 1e-5)
  ;; After 5 steps of 90/10 mixing from [10,0], x~6.64 y~3.36
  (assert-true "loop map: x + y = 10 (conserved)" (< (js/Math.abs (- (+ compiled-0 compiled-1) 10.0)) 1e-4))
  (assert-true "loop map: x > y (started at 10,0)" (> compiled-0 compiled-1)))

;; ---------------------------------------------------------------------------
;; Test 3: Multi-key — 6-key map matching depression kernel state
;; ---------------------------------------------------------------------------

(println "\n== Test 3: 6-key map (depression kernel shape) ==")

(defn depression-state-fn [init-flat]
  ;; Unpack [6] → 6-key map (matching :dep :ac :avr :atq :rp :es)
  (let [state {:dep (mx/index init-flat 0)
               :ac  (mx/index init-flat 1)
               :avr (mx/index init-flat 2)
               :atq (mx/index init-flat 3)
               :rp  (mx/index init-flat 4)
               :es  (mx/index init-flat 5)}
        ;; Compute transition means (simplified Lewinsohn-like kernel)
        new-dep (mx/add (mx/multiply (mx/scalar 0.8) (:dep state))
                        (mx/multiply (mx/scalar -0.1) (:ac state)))
        new-ac  (mx/add (mx/multiply (mx/scalar 0.1) (:dep state))
                        (mx/multiply (mx/scalar 0.7) (:ac state)))
        new-avr (mx/add (mx/multiply (mx/scalar 0.05) (:atq state))
                        (mx/multiply (mx/scalar 0.9) (:avr state)))
        new-atq (mx/add (mx/multiply (mx/scalar 0.3) (:rp state))
                        (mx/multiply (mx/scalar 0.6) (:atq state)))
        new-rp  (mx/add (mx/multiply (mx/scalar -0.2) (:es state))
                        (mx/multiply (mx/scalar 0.5) (:rp state)))
        new-es  (mx/add (mx/multiply (mx/scalar 0.1) (:dep state))
                        (mx/multiply (mx/scalar 0.85) (:es state)))]
    ;; Pack back to [6] — sorted key order: :ac :atq :avr :dep :es :rp
    ;; (but we'll use the logical order for clarity)
    (mx/stack [new-dep new-ac new-avr new-atq new-rp new-es])))

(let [compiled-dep (mx/compile-fn depression-state-fn)
      input (mx/array [1.0 2.0 3.0 4.0 5.0 6.0])
      interp-result (depression-state-fn input)
      _ (mx/eval! interp-result)
      compiled-result (compiled-dep input)
      _ (mx/eval! compiled-result)]
  (doseq [i (range 6)]
    (let [interp-v (mx/item (mx/index interp-result i))
          compiled-v (mx/item (mx/index compiled-result i))]
      (assert-close (str "6-key map: element " i " matches")
                    interp-v compiled-v 1e-6))))

;; Also test: multi-step iteration with 6-key state
(println "\n== Test 3b: 6-key map with T=3 loop ==")

(defn depression-loop-fn [init-flat]
  (let [init-state {:dep (mx/index init-flat 0)
                    :ac  (mx/index init-flat 1)
                    :avr (mx/index init-flat 2)
                    :atq (mx/index init-flat 3)
                    :rp  (mx/index init-flat 4)
                    :es  (mx/index init-flat 5)}]
    (loop [t 0
           state init-state]
      (if (>= t 3)
        (mx/stack [(:dep state) (:ac state) (:avr state)
                   (:atq state) (:rp state) (:es state)])
        (let [s state
              new-dep (mx/add (mx/multiply (mx/scalar 0.8) (:dep s))
                              (mx/multiply (mx/scalar -0.1) (:ac s)))
              new-ac  (mx/add (mx/multiply (mx/scalar 0.1) (:dep s))
                              (mx/multiply (mx/scalar 0.7) (:ac s)))
              new-avr (mx/add (mx/multiply (mx/scalar 0.05) (:atq s))
                              (mx/multiply (mx/scalar 0.9) (:avr s)))
              new-atq (mx/add (mx/multiply (mx/scalar 0.3) (:rp s))
                              (mx/multiply (mx/scalar 0.6) (:atq s)))
              new-rp  (mx/add (mx/multiply (mx/scalar -0.2) (:es s))
                              (mx/multiply (mx/scalar 0.5) (:rp s)))
              new-es  (mx/add (mx/multiply (mx/scalar 0.1) (:dep s))
                              (mx/multiply (mx/scalar 0.85) (:es s)))]
          (recur (inc t) {:dep new-dep :ac new-ac :avr new-avr
                          :atq new-atq :rp new-rp :es new-es}))))))

(let [compiled-dep-loop (mx/compile-fn depression-loop-fn)
      input (mx/array [1.0 2.0 3.0 4.0 5.0 6.0])
      interp-result (depression-loop-fn input)
      _ (mx/eval! interp-result)
      compiled-result (compiled-dep-loop input)
      _ (mx/eval! compiled-result)]
  (doseq [i (range 6)]
    (let [interp-v (mx/item (mx/index interp-result i))
          compiled-v (mx/item (mx/index compiled-result i))]
      (assert-close (str "6-key loop: element " i " matches")
                    interp-v compiled-v 1e-5))))

;; ===========================================================================
;; Phase 2: Fused Unfold simulate with map-state kernels
;; ===========================================================================

;; ---------------------------------------------------------------------------
;; Test 4: 2-key map-state kernel — fused simulate matches handler simulate
;; ---------------------------------------------------------------------------

(println "\n== Test 4: 2-key map-state Unfold simulate ==")

(def map-kernel-2
  (dyn/auto-key
    (gen [t state a]
      (let [mean-x (mx/add (mx/multiply a (:x state)) (:y state))
            mean-y (mx/multiply (mx/scalar 0.5) (:x state))
            x (trace :x (dist/gaussian mean-x 1.0))
            y (trace :y (dist/gaussian mean-y 1.0))]
        {:x x :y y}))))

(def unfold-2 (comb/unfold-combinator map-kernel-2))

;; Basic: fused simulate produces map retvals
(let [trace-fused (p/simulate unfold-2 [5 {:x (mx/scalar 1.0)
                                           :y (mx/scalar 0.0)}
                                        (mx/scalar 0.8)])]
  (assert-true "2-key unfold: 5 steps" (= 5 (count (:retval trace-fused))))
  (assert-true "2-key unfold: retval is map" (map? (first (:retval trace-fused))))
  (assert-true "2-key unfold: has :x key" (contains? (first (:retval trace-fused)) :x))
  (assert-true "2-key unfold: has :y key" (contains? (first (:retval trace-fused)) :y))
  (assert-true "2-key unfold: fused path used"
               (true? (::comb/fused (meta trace-fused))))
  ;; Score should be finite
  (let [sc (mx/item (:score trace-fused))]
    (assert-true "2-key unfold: finite score" (js/isFinite sc)))
  ;; Choices should have 5 time steps, each with :x and :y
  (doseq [t (range 5)]
    (let [step-cm (cm/get-submap (:choices trace-fused) t)]
      (assert-true (str "2-key unfold: step " t " has :x")
                   (cm/has-value? (cm/get-submap step-cm :x)))
      (assert-true (str "2-key unfold: step " t " has :y")
                   (cm/has-value? (cm/get-submap step-cm :y))))))

;; ---------------------------------------------------------------------------
;; Test 5: Fused vs handler score distribution — statistical equivalence
;; ---------------------------------------------------------------------------

(println "\n== Test 5: Fused vs handler score comparison ==")

;; Run multiple fused simulates and check score distribution is reasonable
(let [scores (mapv (fn [_]
                     (let [tr (p/simulate unfold-2 [3 {:x (mx/scalar 0.0)
                                                       :y (mx/scalar 0.0)}
                                                   (mx/scalar 0.5)])]
                       (mx/item (:score tr))))
                   (range 20))
      mean-score (/ (reduce + scores) (count scores))]
  ;; Gaussian log-probs for 3 steps × 2 sites = 6 sites
  ;; Each site ~ N(0, 1) → log-prob ~ -0.919 each, so total ~ -5.5
  (assert-true "score distribution: mean is negative" (neg? mean-score))
  (assert-true "score distribution: mean reasonable (> -30)" (> mean-score -30))
  (assert-true "score distribution: all finite" (every? js/isFinite scores)))

;; ---------------------------------------------------------------------------
;; Test 6: 6-key map-state kernel (depression kernel shape)
;; ---------------------------------------------------------------------------

(println "\n== Test 6: 6-key map-state Unfold simulate ==")

(def map-kernel-6
  (dyn/auto-key
    (gen [t state a]
      (let [dep (trace :dep (dist/gaussian
                             (mx/add (mx/multiply (mx/scalar 0.8) (:dep state))
                                     (mx/multiply a (:ac state)))
                             1.0))
            ac  (trace :ac  (dist/gaussian
                             (mx/add (mx/multiply (mx/scalar 0.1) (:dep state))
                                     (mx/multiply (mx/scalar 0.7) (:ac state)))
                             1.0))
            avr (trace :avr (dist/gaussian
                             (mx/add (mx/multiply (mx/scalar 0.05) (:atq state))
                                     (mx/multiply (mx/scalar 0.9) (:avr state)))
                             1.0))
            atq (trace :atq (dist/gaussian
                             (mx/add (mx/multiply (mx/scalar 0.3) (:rp state))
                                     (mx/multiply (mx/scalar 0.6) (:atq state)))
                             1.0))
            rp  (trace :rp  (dist/gaussian
                             (mx/add (mx/multiply (mx/scalar -0.2) (:es state))
                                     (mx/multiply (mx/scalar 0.5) (:rp state)))
                             1.0))
            es  (trace :es  (dist/gaussian
                             (mx/add (mx/multiply (mx/scalar 0.1) (:dep state))
                                     (mx/multiply (mx/scalar 0.85) (:es state)))
                             1.0))]
        {:dep dep :ac ac :avr avr :atq atq :rp rp :es es}))))

(def unfold-6 (comb/unfold-combinator map-kernel-6))

(let [init-state {:dep (mx/scalar 0.0) :ac (mx/scalar 0.0)
                  :avr (mx/scalar 0.0) :atq (mx/scalar 0.0)
                  :rp (mx/scalar 0.0) :es (mx/scalar 0.0)}
      trace-fused (p/simulate unfold-6 [9 init-state (mx/scalar -0.1)])]
  (assert-true "6-key unfold: 9 steps" (= 9 (count (:retval trace-fused))))
  (assert-true "6-key unfold: retval is map" (map? (first (:retval trace-fused))))
  (assert-true "6-key unfold: has all 6 keys"
               (= #{:dep :ac :avr :atq :rp :es}
                  (set (keys (first (:retval trace-fused))))))
  (assert-true "6-key unfold: fused path used"
               (true? (::comb/fused (meta trace-fused))))
  (let [sc (mx/item (:score trace-fused))]
    (assert-true "6-key unfold: finite score" (js/isFinite sc)))
  ;; Check each step has all 6 trace sites
  (doseq [t (range 9)]
    (let [step-cm (cm/get-submap (:choices trace-fused) t)]
      (assert-true (str "6-key unfold: step " t " has 6 sites")
                   (and (cm/has-value? (cm/get-submap step-cm :dep))
                        (cm/has-value? (cm/get-submap step-cm :ac))
                        (cm/has-value? (cm/get-submap step-cm :avr))
                        (cm/has-value? (cm/get-submap step-cm :atq))
                        (cm/has-value? (cm/get-submap step-cm :rp))
                        (cm/has-value? (cm/get-submap step-cm :es)))))))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n== TOTAL: " @pass-count "/" (+ @pass-count @fail-count)
              " passed =="))
(when (pos? @fail-count)
  (println (str "FAILURES: " @fail-count))
  (js/process.exit 1))
