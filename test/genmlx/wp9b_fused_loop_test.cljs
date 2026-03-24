(ns genmlx.wp9b-fused-loop-test
  "WP-9B tests: fused unfold/scan simulate.
   Validates that fused loop execution (single mx/compile-fn dispatch for T steps)
   produces correct trace structure, scores, and state threading."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.compiled-ops :as compiled]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(defn force-handler
  "Strip ALL compiled paths from a gen-fn so it falls back to handler."
  [gf]
  (let [schema (dissoc (:schema gf)
                 :compiled-simulate :compiled-generate
                 :compiled-update :compiled-assess :compiled-project
                 :compiled-regenerate
                 :compiled-prefix :compiled-prefix-addrs
                 :compiled-prefix-generate :compiled-prefix-update
                 :compiled-prefix-assess :compiled-prefix-project
                 :compiled-prefix-regenerate)]
    (assoc gf :schema schema)))

;; ---------------------------------------------------------------------------
;; Test kernels
;; ---------------------------------------------------------------------------

(def k-simple
  (dyn/auto-key (gen [t state]
    (let [x (trace :x (dist/gaussian state 0.1))]
      x))))

(def k-2site
  (dyn/auto-key (gen [t state]
    (let [x (trace :x (dist/gaussian state 0.1))
          y (trace :y (dist/gaussian x 0.5))]
      x))))

(def k-delta
  (dyn/auto-key (gen [t state]
    (let [x (trace :x (dist/delta (mx/add state (mx/scalar 1.0))))]
      x))))

(def k-scan
  (dyn/auto-key (gen [carry input]
    (let [x (trace :x (dist/gaussian (mx/add carry input) 0.1))]
      [x x]))))

(def k-scan-delta
  (dyn/auto-key (gen [carry input]
    (let [x (trace :x (dist/delta (mx/add carry input)))]
      [x (mx/multiply x (mx/scalar 2.0))]))))

(def k-beta
  (dyn/auto-key (gen [t state]
    (let [x (trace :x (dist/beta-dist 2 5))]
      x))))

;; Map kernels
(def k-map
  (dyn/auto-key (gen [x]
    (let [y (trace :y (dist/gaussian x 1.0))]
      y))))

(def k-map-delta
  (dyn/auto-key (gen [x]
    (let [y (trace :y (dist/delta (mx/multiply x (mx/scalar 2.0))))]
      y))))

(def k-map-beta
  (dyn/auto-key (gen [x]
    (let [y (trace :y (dist/beta-dist 2 5))]
      y))))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest fusability-prerequisites-test
  (testing "Fusability prerequisites"
    (is (some? (compiled/get-compiled-simulate k-simple)) "k-simple has compiled-simulate")
    (is (nil? (compiled/get-compiled-simulate k-beta)) "k-beta lacks compiled-simulate (non-fusable)")
    (is (some? (compiled/get-compiled-simulate k-scan)) "k-scan has compiled-simulate")))

(deftest fused-unfold-t5-simple-test
  (testing "Fused unfold T=5 simple kernel"
    (let [unfold (comb/unfold-combinator k-simple)
          trace (p/simulate unfold [5 (mx/scalar 0.0)])]
      (mx/eval! (:score trace))
      (is (js/isFinite (mx/item (:score trace))) "score finite")
      (is (= 5 (count (:retval trace))) "retval count")
      (doseq [t (range 5)]
        (let [sub (cm/get-submap (:choices trace) t)]
          (is (cm/has-value? (cm/get-submap sub :x)) (str "step " t " has :x")))))))

(deftest fused-unfold-t10-test
  (testing "Fused unfold T=10"
    (let [unfold (comb/unfold-combinator k-simple)
          trace (p/simulate unfold [10 (mx/scalar 1.0)])]
      (mx/eval! (:score trace))
      (is (js/isFinite (mx/item (:score trace))) "T=10 score finite")
      (is (= 10 (count (:retval trace))) "T=10 retval count")
      (let [final (last (:retval trace))]
        (mx/eval! final)
        (is (js/isFinite (mx/item final)) "final state is finite")))))

(deftest fused-unfold-2site-test
  (testing "Fused unfold 2-site kernel"
    (let [unfold (comb/unfold-combinator k-2site)
          trace (p/simulate unfold [5 (mx/scalar 0.0)])]
      (mx/eval! (:score trace))
      (is (js/isFinite (mx/item (:score trace))) "2-site score finite")
      (doseq [t (range 5)]
        (let [sub (cm/get-submap (:choices trace) t)]
          (is (cm/has-value? (cm/get-submap sub :x)) (str "step " t " has :x"))
          (is (cm/has-value? (cm/get-submap sub :y)) (str "step " t " has :y")))))))

(deftest deterministic-kernel-equivalence-test
  (testing "Deterministic kernel equivalence"
    (let [unfold-c (comb/unfold-combinator k-delta)
          unfold-h (comb/unfold-combinator (force-handler k-delta))
          trace-c (p/simulate unfold-c [5 (mx/scalar 0.0)])
          trace-h (p/simulate unfold-h [5 (mx/scalar 0.0)])]
      (mx/eval! (:score trace-c))
      (mx/eval! (:score trace-h))
      (is (h/close? 0.0 (mx/item (:score trace-c)) 1e-6) "delta score compiled")
      (is (h/close? 0.0 (mx/item (:score trace-h)) 1e-6) "delta score handler")
      (doseq [t (range 5)]
        (let [sub-c (cm/get-submap (cm/get-submap (:choices trace-c) t) :x)
              sub-h (cm/get-submap (cm/get-submap (:choices trace-h) t) :x)
              val-c (mx/item (cm/get-value sub-c))
              val-h (mx/item (cm/get-value sub-h))]
          (is (h/close? val-h val-c 1e-6) (str "step " t " value match")))))))

(deftest state-threading-test
  (testing "State threading"
    (let [unfold (comb/unfold-combinator k-simple)
          trace (p/simulate unfold [3 (mx/scalar 5.0)])]
      (let [v0 (mx/item (cm/get-value (cm/get-submap (cm/get-submap (:choices trace) 0) :x)))
            v1 (mx/item (cm/get-value (cm/get-submap (cm/get-submap (:choices trace) 1) :x)))]
        (is (< (js/Math.abs (- v0 5.0)) 1.0) "step 0 near init (sigma=0.1)")
        (is (< (js/Math.abs (- v1 v0)) 1.0) "step 1 near step 0 (sigma=0.1)")))))

(deftest fused-scan-t5-test
  (testing "Fused scan T=5"
    (let [scan (comb/scan-combinator k-scan)
          inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)
                  (mx/scalar 4.0) (mx/scalar 5.0)]
          trace (p/simulate scan [(mx/scalar 0.0) inputs])]
      (mx/eval! (:score trace))
      (is (js/isFinite (mx/item (:score trace))) "scan score finite")
      (is (some? (:carry (:retval trace))) "has carry")
      (is (= 5 (count (:outputs (:retval trace)))) "outputs count")
      (doseq [t (range 5)]
        (is (cm/has-value? (cm/get-submap (cm/get-submap (:choices trace) t) :x))
            (str "scan step " t " has :x"))))))

(deftest deterministic-scan-carry-threading-test
  (testing "Deterministic scan carry threading"
    (let [scan-c (comb/scan-combinator k-scan-delta)
          scan-h (comb/scan-combinator (force-handler k-scan-delta))
          inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]
          trace-c (p/simulate scan-c [(mx/scalar 0.0) inputs])
          trace-h (p/simulate scan-h [(mx/scalar 0.0) inputs])]
      (mx/eval! (:score trace-c))
      (mx/eval! (:score trace-h))
      (let [carry-c (:carry (:retval trace-c))
            carry-h (:carry (:retval trace-h))]
        (mx/eval! carry-c) (mx/eval! carry-h)
        (is (h/close? 6.0 (mx/item carry-c) 1e-5) "scan final carry compiled")
        (is (h/close? 6.0 (mx/item carry-h) 1e-5) "scan final carry handler"))
      (let [outputs-c (:outputs (:retval trace-c))]
        (doseq [[i expected] [[0 2.0] [1 6.0] [2 12.0]]]
          (mx/eval! (nth outputs-c i))
          (is (h/close? expected (mx/item (nth outputs-c i)) 1e-5) (str "scan output " i)))))))

(deftest non-fusable-fallback-test
  (testing "Non-fusable fallback"
    (let [unfold (comb/unfold-combinator k-beta)
          trace (p/simulate unfold [3 (mx/scalar 0.5)])]
      (is (instance? tr/Trace trace) "beta fallback valid")
      (mx/eval! (:score trace))
      (is (js/isFinite (mx/item (:score trace))) "beta fallback score finite"))))

(deftest step-scores-metadata-test
  (testing "Step-scores metadata"
    (let [unfold (comb/unfold-combinator k-simple)
          trace (p/simulate unfold [5 (mx/scalar 0.0)])]
      (let [ss (::comb/step-scores (meta trace))]
        (is (some? ss) "has step-scores")
        (is (= 5 (count ss)) "step-scores count")
        (doseq [s ss] (mx/eval! s))
        (mx/eval! (:score trace))
        (is (h/close? (mx/item (:score trace))
                      (reduce + (map mx/item ss))
                      1e-5) "step-scores sum = total")))))

(deftest fused-path-detection-test
  (testing "Fused path detection"
    (is (compiled/fusable-kernel? k-simple) "k-simple is fusable")
    (is (not (compiled/fusable-kernel? k-beta)) "k-beta is NOT fusable")
    (let [unfold (comb/unfold-combinator k-simple)
          trace (p/simulate unfold [5 (mx/scalar 0.0)])]
      (is (::comb/fused (meta trace)) "unfold fused path used"))
    (let [scan (comb/scan-combinator k-scan)
          inputs [(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]
          trace (p/simulate scan [(mx/scalar 0.0) inputs])]
      (is (::comb/fused (meta trace)) "scan fused path used"))))

(deftest variable-t-test
  (testing "Variable T"
    (let [unfold (comb/unfold-combinator k-simple)
          trace5 (p/simulate unfold [5 (mx/scalar 0.0)])
          trace10 (p/simulate unfold [10 (mx/scalar 0.0)])]
      (mx/eval! (:score trace5))
      (mx/eval! (:score trace10))
      (is (js/isFinite (mx/item (:score trace5))) "T=5 valid")
      (is (js/isFinite (mx/item (:score trace10))) "T=10 valid")
      (is (= 5 (count (:retval trace5))) "T=5 retval count")
      (is (= 10 (count (:retval trace10))) "T=10 retval count"))))

(deftest fused-map-n5-test
  (testing "Fused map N=5"
    (let [mapped (comb/map-combinator k-map)
          args [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)
                 (mx/scalar 4.0) (mx/scalar 5.0)]]
          trace (p/simulate mapped args)]
      (mx/eval! (:score trace))
      (is (js/isFinite (mx/item (:score trace))) "map score finite")
      (is (= 5 (count (:retval trace))) "map retval count")
      (doseq [i (range 5)]
        (is (cm/has-value? (cm/get-submap (cm/get-submap (:choices trace) i) :y))
            (str "elem " i " has :y"))))))

(deftest fused-map-deterministic-equivalence-test
  (testing "Fused map deterministic equivalence"
    (let [mapped-c (comb/map-combinator k-map-delta)
          mapped-h (comb/map-combinator (force-handler k-map-delta))
          args [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]]
          trace-c (p/simulate mapped-c args)
          trace-h (p/simulate mapped-h args)]
      (mx/eval! (:score trace-c))
      (mx/eval! (:score trace-h))
      (is (h/close? 0.0 (mx/item (:score trace-c)) 1e-6) "delta map score compiled")
      (is (h/close? 0.0 (mx/item (:score trace-h)) 1e-6) "delta map score handler")
      (doseq [i (range 3)]
        (let [val-c (mx/item (cm/get-value (cm/get-submap (cm/get-submap (:choices trace-c) i) :y)))
              val-h (mx/item (cm/get-value (cm/get-submap (cm/get-submap (:choices trace-h) i) :y)))]
          (is (h/close? val-h val-c 1e-6) (str "elem " i " value match")))))))

(deftest fused-map-path-detection-test
  (testing "Fused map path detection"
    (is (compiled/fusable-kernel? k-map) "k-map is fusable")
    (is (not (compiled/fusable-kernel? k-map-beta)) "k-map-beta is NOT fusable")
    (let [mapped (comb/map-combinator k-map)
          trace (p/simulate mapped [[(mx/scalar 1.0) (mx/scalar 2.0)]])]
      (is (::comb/fused (meta trace)) "map fused path used"))))

(deftest non-fusable-map-fallback-test
  (testing "Non-fusable map fallback"
    (let [mapped (comb/map-combinator k-map-beta)
          trace (p/simulate mapped [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]])]
      (is (instance? tr/Trace trace) "beta map fallback valid")
      (mx/eval! (:score trace))
      (is (js/isFinite (mx/item (:score trace))) "beta map fallback score finite"))))

(deftest fused-map-element-scores-metadata-test
  (testing "Fused map element-scores metadata"
    (let [mapped (comb/map-combinator k-map)
          trace (p/simulate mapped [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]])]
      (let [es (::comb/element-scores (meta trace))]
        (is (some? es) "has element-scores")
        (is (= 3 (count es)) "element-scores count")
        (doseq [s es] (mx/eval! s))
        (mx/eval! (:score trace))
        (is (h/close? (mx/item (:score trace))
                      (reduce + (map mx/item es))
                      1e-5) "element-scores sum = total")))))

(cljs.test/run-tests)
