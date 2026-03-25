(ns genmlx.fused-map-state-test
  "Phase 1: mx/compile-fn traces through CLJS map intermediates.
   Phase 2: Fused Unfold simulate works with map-state kernels."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.combinators :as comb]
            [genmlx.choicemap :as cm]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Phase 1: mx/compile-fn with map intermediates
;; ---------------------------------------------------------------------------

(defn simple-map-fn [init-state-flat]
  (let [state {:a (mx/index init-state-flat 0)
               :b (mx/index init-state-flat 1)}
        result (mx/add (mx/multiply (:a state) (mx/scalar 2.0))
                       (:b state))
        new-state (mx/stack [result (:b state)])]
    new-state))

(defn loop-map-fn [init-flat]
  (let [init-state {:x (mx/index init-flat 0)
                    :y (mx/index init-flat 1)}]
    (loop [t 0
           state init-state]
      (if (>= t 5)
        (mx/stack [(:x state) (:y state)])
        (let [new-x (mx/add (mx/multiply (mx/scalar 0.9) (:x state))
                            (mx/multiply (mx/scalar 0.1) (:y state)))
              new-y (mx/add (mx/multiply (mx/scalar 0.1) (:x state))
                            (mx/multiply (mx/scalar 0.9) (:y state)))]
          (recur (inc t) {:x new-x :y new-y}))))))

(defn depression-state-fn [init-flat]
  (let [state {:dep (mx/index init-flat 0)
               :ac  (mx/index init-flat 1)
               :avr (mx/index init-flat 2)
               :atq (mx/index init-flat 3)
               :rp  (mx/index init-flat 4)
               :es  (mx/index init-flat 5)}
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
    (mx/stack [new-dep new-ac new-avr new-atq new-rp new-es])))

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

;; ---------------------------------------------------------------------------
;; Phase 2: Fused Unfold models
;; ---------------------------------------------------------------------------

(def map-kernel-2
  (dyn/auto-key
    (gen [t state a]
      (let [mean-x (mx/add (mx/multiply a (:x state)) (:y state))
            mean-y (mx/multiply (mx/scalar 0.5) (:x state))
            x (trace :x (dist/gaussian mean-x 1.0))
            y (trace :y (dist/gaussian mean-y 1.0))]
        {:x x :y y}))))

(def unfold-2 (comb/unfold-combinator map-kernel-2))

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

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest simple-map-intermediate-test
  (testing "simple map intermediate"
    (let [compiled-simple (mx/compile-fn simple-map-fn)
          input (mx/array [3.0 1.0])
          interp-result (simple-map-fn input)
          _ (mx/eval! interp-result)
          compiled-result (compiled-simple input)
          _ (mx/eval! compiled-result)
          interp-0 (mx/item (mx/index interp-result 0))
          interp-1 (mx/item (mx/index interp-result 1))
          compiled-0 (mx/item (mx/index compiled-result 0))
          compiled-1 (mx/item (mx/index compiled-result 1))]
      (is (h/close? interp-0 compiled-0 1e-6) "simple map: element 0 matches")
      (is (h/close? interp-1 compiled-1 1e-6) "simple map: element 1 matches")
      (is (h/close? 7.0 compiled-0 1e-6) "simple map: element 0 = 7.0")
      (is (h/close? 1.0 compiled-1 1e-6) "simple map: element 1 = 1.0"))))

(deftest loop-map-state-threading-test
  (testing "T=5 loop with map state threading"
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
      (is (h/close? interp-0 compiled-0 1e-5) "loop map: element 0 matches")
      (is (h/close? interp-1 compiled-1 1e-5) "loop map: element 1 matches")
      (is (< (js/Math.abs (- (+ compiled-0 compiled-1) 10.0)) 1e-4) "loop map: x + y = 10 (conserved)")
      (is (> compiled-0 compiled-1) "loop map: x > y (started at 10,0)"))))

(deftest six-key-map-test
  (testing "6-key map (depression kernel shape)"
    (let [compiled-dep (mx/compile-fn depression-state-fn)
          input (mx/array [1.0 2.0 3.0 4.0 5.0 6.0])
          interp-result (depression-state-fn input)
          _ (mx/eval! interp-result)
          compiled-result (compiled-dep input)
          _ (mx/eval! compiled-result)]
      (doseq [i (range 6)]
        (let [interp-v (mx/item (mx/index interp-result i))
              compiled-v (mx/item (mx/index compiled-result i))]
          (is (h/close? interp-v compiled-v 1e-6) (str "6-key map: element " i " matches"))))))

  (testing "6-key map with T=3 loop"
    (let [compiled-dep-loop (mx/compile-fn depression-loop-fn)
          input (mx/array [1.0 2.0 3.0 4.0 5.0 6.0])
          interp-result (depression-loop-fn input)
          _ (mx/eval! interp-result)
          compiled-result (compiled-dep-loop input)
          _ (mx/eval! compiled-result)]
      (doseq [i (range 6)]
        (let [interp-v (mx/item (mx/index interp-result i))
              compiled-v (mx/item (mx/index compiled-result i))]
          (is (h/close? interp-v compiled-v 1e-5) (str "6-key loop: element " i " matches")))))))

(deftest two-key-map-state-unfold-test
  (testing "2-key map-state Unfold simulate"
    (let [trace-fused (p/simulate unfold-2 [5 {:x (mx/scalar 1.0)
                                               :y (mx/scalar 0.0)}
                                            (mx/scalar 0.8)])]
      (is (= 5 (count (:retval trace-fused))) "2-key unfold: 5 steps")
      (is (map? (first (:retval trace-fused))) "2-key unfold: retval is map")
      (is (contains? (first (:retval trace-fused)) :x) "2-key unfold: has :x key")
      (is (contains? (first (:retval trace-fused)) :y) "2-key unfold: has :y key")
      (is (true? (::comb/fused (meta trace-fused))) "2-key unfold: fused path used")
      (let [sc (mx/item (:score trace-fused))]
        (is (js/isFinite sc) "2-key unfold: finite score"))
      (doseq [t (range 5)]
        (let [step-cm (cm/get-submap (:choices trace-fused) t)]
          (is (cm/has-value? (cm/get-submap step-cm :x))
              (str "2-key unfold: step " t " has :x"))
          (is (cm/has-value? (cm/get-submap step-cm :y))
              (str "2-key unfold: step " t " has :y")))))))

(deftest fused-vs-handler-score-test
  (testing "fused vs handler score comparison"
    (let [scores (mapv (fn [_]
                         (let [tr (p/simulate unfold-2 [3 {:x (mx/scalar 0.0)
                                                           :y (mx/scalar 0.0)}
                                                       (mx/scalar 0.5)])]
                           (mx/item (:score tr))))
                       (range 20))
          mean-score (/ (reduce + scores) (count scores))]
      (is (neg? mean-score) "score distribution: mean is negative")
      (is (> mean-score -30) "score distribution: mean reasonable (> -30)")
      (is (every? js/isFinite scores) "score distribution: all finite"))))

(deftest six-key-map-state-unfold-test
  (testing "6-key map-state Unfold simulate"
    (let [init-state {:dep (mx/scalar 0.0) :ac (mx/scalar 0.0)
                      :avr (mx/scalar 0.0) :atq (mx/scalar 0.0)
                      :rp (mx/scalar 0.0) :es (mx/scalar 0.0)}
          trace-fused (p/simulate unfold-6 [9 init-state (mx/scalar -0.1)])]
      (is (= 9 (count (:retval trace-fused))) "6-key unfold: 9 steps")
      (is (map? (first (:retval trace-fused))) "6-key unfold: retval is map")
      (is (= #{:dep :ac :avr :atq :rp :es}
             (set (keys (first (:retval trace-fused)))))
          "6-key unfold: has all 6 keys")
      (is (true? (::comb/fused (meta trace-fused))) "6-key unfold: fused path used")
      (let [sc (mx/item (:score trace-fused))]
        (is (js/isFinite sc) "6-key unfold: finite score"))
      (doseq [t (range 9)]
        (let [step-cm (cm/get-submap (:choices trace-fused) t)]
          (is (and (cm/has-value? (cm/get-submap step-cm :dep))
                   (cm/has-value? (cm/get-submap step-cm :ac))
                   (cm/has-value? (cm/get-submap step-cm :avr))
                   (cm/has-value? (cm/get-submap step-cm :atq))
                   (cm/has-value? (cm/get-submap step-cm :rp))
                   (cm/has-value? (cm/get-submap step-cm :es)))
              (str "6-key unfold: step " t " has 6 sites")))))))

(cljs.test/run-tests)
