(ns genmlx.wp7-combinator-generate-test
  "WP-7 tests: combinator compiled generate.
   Validates that compiled generate paths for Map, Unfold, Scan, Switch, Mix
   produce identical traces, scores, and weights as handler paths."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.compiled-ops :as compiled]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(defn force-handler
  "Strip compiled paths from a gen-fn so it falls back to handler."
  [gf]
  (let [schema (dissoc (:schema gf) :compiled-simulate :compiled-generate
                       :compiled-prefix :compiled-prefix-addrs
                       :compiled-prefix-generate)]
    (assoc gf :schema schema)))

(defn compilable-generate?
  "Check if a gen-fn kernel has a compiled-generate function."
  [gf]
  (some? (compiled/get-compiled-generate gf)))

;; ---------------------------------------------------------------------------
;; Test kernels
;; ---------------------------------------------------------------------------

(def k-map
  (dyn/auto-key (gen [x]
    (let [y (trace :y (dist/gaussian x 1.0))]
      y))))

(def k-unfold
  (dyn/auto-key (gen [t state]
    (let [next (trace :x (dist/gaussian state 0.1))]
      next))))

(def k-scan
  (dyn/auto-key (gen [carry input]
    (let [x (trace :x (dist/gaussian (mx/add carry input) 0.1))]
      [x x]))))

(def k-switch
  (dyn/auto-key (gen []
    (let [x (trace :x (dist/gaussian 0 1))]
      x))))

(def k-switch-b
  (dyn/auto-key (gen []
    (let [x (trace :x (dist/gaussian 5 0.5))]
      x))))

;; Non-compilable kernel (beta-dist has no noise transform)
(def k-beta
  (dyn/auto-key (gen [x]
    (let [y (trace :y (dist/beta-dist 2 5))]
      y))))

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest prerequisites-test
  (testing "Prerequisites"
    (is (compilable-generate? k-map) "map kernel has compiled-generate")
    (is (compilable-generate? k-unfold) "unfold kernel has compiled-generate")
    (is (compilable-generate? k-scan) "scan kernel has compiled-generate")
    (is (compilable-generate? k-switch) "switch kernel has compiled-generate")
    (is (not (compilable-generate? k-beta)) "beta kernel NOT compilable")))

(deftest map-compiled-dispatch-test
  (testing "Map: compiled dispatch"
    (let [mapped (comb/map-combinator k-map)
          result (p/generate mapped [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]] cm/EMPTY)]
      (is (::comb/compiled-path (meta (:trace result))) "map compiled path used"))))

(deftest map-no-constraints-test
  (testing "Map: no constraints (weight=0)"
    (let [mapped (comb/map-combinator k-map)
          result (p/generate mapped [[(mx/scalar 1.0) (mx/scalar 2.0)]] cm/EMPTY)
          trace (:trace result)]
      (mx/eval! (:weight result) (:score trace))
      (is (h/close? 0.0 (h/realize (:weight result)) 1e-6) "map no-constraint weight=0")
      (is (instance? tr/Trace trace) "map trace valid")
      (is (= 2 (count (:retval trace))) "map retval count")
      (is (js/isFinite (mx/item (:score trace))) "map score finite"))))

(deftest map-all-constrained-test
  (testing "Map: all elements constrained"
    (let [mapped (comb/map-combinator k-map)
          obs (-> cm/EMPTY
                  (cm/set-choice [0 :y] (mx/scalar 1.5))
                  (cm/set-choice [1 :y] (mx/scalar 2.5)))
          result (p/generate mapped [[(mx/scalar 1.0) (mx/scalar 2.0)]] obs)
          mapped-h (comb/map-combinator (force-handler k-map))
          result-h (p/generate mapped-h [[(mx/scalar 1.0) (mx/scalar 2.0)]] obs)]
      (mx/eval! (:weight result) (:score (:trace result))
                (:weight result-h) (:score (:trace result-h)))
      (is (h/close? (mx/item (:weight result-h)) (h/realize (:weight result)) 1e-5)
          "map constrained weight matches handler")
      (is (h/close? (mx/item (:score (:trace result-h))) (h/realize (:score (:trace result))) 1e-5)
          "map constrained score matches handler")
      (is (h/close? 1.5 (h/realize (cm/get-choice (:choices (:trace result)) [0 :y])) 1e-6)
          "map constrained val[0]")
      (is (h/close? 2.5 (h/realize (cm/get-choice (:choices (:trace result)) [1 :y])) 1e-6)
          "map constrained val[1]"))))

(deftest map-partial-constraints-test
  (testing "Map: partial constraints"
    (let [mapped (comb/map-combinator k-map)
          obs (-> cm/EMPTY (cm/set-choice [1 :y] (mx/scalar 5.0)))
          result (p/generate mapped [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]] obs)
          trace (:trace result)]
      (mx/eval! (:weight result) (:score trace))
      (is (not= 0.0 (mx/item (:weight result))) "map partial weight nonzero")
      (is (h/close? 5.0 (h/realize (cm/get-choice (:choices trace) [1 :y])) 1e-6)
          "map partial constrained value")
      (is (= 3 (count (:retval trace))) "map partial retval count"))))

(deftest map-trace-structure-test
  (testing "Map: trace structure"
    (let [mapped (comb/map-combinator k-map)
          result (p/generate mapped [[(mx/scalar 1.0) (mx/scalar 2.0)]] cm/EMPTY)
          trace (:trace result)
          choices (:choices trace)]
      (is (mx/array? (cm/get-choice choices [0 :y])) "map choices[0 :y] present")
      (is (mx/array? (cm/get-choice choices [1 :y])) "map choices[1 :y] present"))))

(deftest map-non-compilable-fallback-test
  (testing "Map: non-compilable fallback"
    (let [mapped (comb/map-combinator k-beta)
          result (p/generate mapped [[(mx/scalar 1.0) (mx/scalar 2.0)]] cm/EMPTY)]
      (is (instance? tr/Trace (:trace result)) "map fallback trace valid")
      (is (not (::comb/compiled-path (meta (:trace result)))) "map fallback NOT compiled"))))

(deftest unfold-compiled-dispatch-test
  (testing "Unfold: compiled dispatch"
    (let [unfold (comb/unfold-combinator k-unfold)
          result (p/generate unfold [3 (mx/scalar 0.5)] cm/EMPTY)]
      (is (::comb/compiled-path (meta (:trace result))) "unfold compiled path used"))))

(deftest unfold-no-constraints-test
  (testing "Unfold: no constraints (weight=0)"
    (let [unfold (comb/unfold-combinator k-unfold)
          result (p/generate unfold [3 (mx/scalar 0.5)] cm/EMPTY)
          trace (:trace result)]
      (mx/eval! (:weight result) (:score trace))
      (is (h/close? 0.0 (h/realize (:weight result)) 1e-6) "unfold no-constraint weight=0")
      (is (= 3 (count (:retval trace))) "unfold retval count")
      (is (js/isFinite (mx/item (:score trace))) "unfold score finite"))))

(deftest unfold-all-constrained-test
  (testing "Unfold: all steps constrained"
    (let [unfold (comb/unfold-combinator k-unfold)
          obs (-> cm/EMPTY
                  (cm/set-choice [0 :x] (mx/scalar 0.6))
                  (cm/set-choice [1 :x] (mx/scalar 0.7))
                  (cm/set-choice [2 :x] (mx/scalar 0.8)))
          result (p/generate unfold [3 (mx/scalar 0.5)] obs)
          unfold-h (comb/unfold-combinator (force-handler k-unfold))
          result-h (p/generate unfold-h [3 (mx/scalar 0.5)] obs)]
      (mx/eval! (:weight result) (:score (:trace result))
                (:weight result-h) (:score (:trace result-h)))
      (is (h/close? (mx/item (:weight result-h)) (h/realize (:weight result)) 1e-5)
          "unfold constrained weight matches handler")
      (is (h/close? (mx/item (:score (:trace result-h))) (h/realize (:score (:trace result))) 1e-5)
          "unfold constrained score matches handler"))))

(deftest unfold-partial-constraints-test
  (testing "Unfold: partial constraints"
    (let [unfold (comb/unfold-combinator k-unfold)
          obs (-> cm/EMPTY (cm/set-choice [1 :x] (mx/scalar 0.7)))
          result (p/generate unfold [3 (mx/scalar 0.5)] obs)
          trace (:trace result)]
      (mx/eval! (:weight result) (:score trace))
      (is (not= 0.0 (mx/item (:weight result))) "unfold partial weight nonzero")
      (is (h/close? 0.7 (h/realize (cm/get-choice (:choices trace) [1 :x])) 1e-6)
          "unfold partial constrained value"))))

(deftest unfold-state-threading-test
  (testing "Unfold: state threading"
    (let [unfold (comb/unfold-combinator k-unfold)
          obs (-> cm/EMPTY
                  (cm/set-choice [0 :x] (mx/scalar 1.0))
                  (cm/set-choice [1 :x] (mx/scalar 2.0))
                  (cm/set-choice [2 :x] (mx/scalar 3.0)))
          result (p/generate unfold [3 (mx/scalar 0.5)] obs)
          trace (:trace result)
          states (:retval trace)]
      (mx/eval! (first states) (second states) (nth states 2))
      (is (h/close? 1.0 (h/realize (first states)) 1e-6) "unfold state[0] = constrained :x")
      (is (h/close? 2.0 (h/realize (second states)) 1e-6) "unfold state[1] = constrained :x")
      (is (h/close? 3.0 (h/realize (nth states 2)) 1e-6) "unfold state[2] = constrained :x"))))

(deftest unfold-trace-structure-test
  (testing "Unfold: trace structure"
    (let [unfold (comb/unfold-combinator k-unfold)
          result (p/generate unfold [3 (mx/scalar 0.5)] cm/EMPTY)
          choices (:choices (:trace result))]
      (is (mx/array? (cm/get-choice choices [0 :x])) "unfold choices[0 :x] present")
      (is (mx/array? (cm/get-choice choices [1 :x])) "unfold choices[1 :x] present")
      (is (mx/array? (cm/get-choice choices [2 :x])) "unfold choices[2 :x] present"))))

(deftest unfold-non-compilable-fallback-test
  (testing "Unfold: non-compilable fallback"
    (let [k (dyn/auto-key (gen [t state] (let [x (trace :x (dist/beta-dist 2 5))] x)))
          unfold (comb/unfold-combinator k)
          result (p/generate unfold [3 (mx/scalar 0.5)] cm/EMPTY)]
      (is (instance? tr/Trace (:trace result)) "unfold fallback valid")
      (is (not (::comb/compiled-path (meta (:trace result)))) "unfold fallback NOT compiled"))))

(deftest scan-compiled-dispatch-test
  (testing "Scan: compiled dispatch"
    (let [scan (comb/scan-combinator k-scan)
          inputs [(mx/scalar 0.1) (mx/scalar 0.2) (mx/scalar 0.3)]
          result (p/generate scan [(mx/scalar 0.0) inputs] cm/EMPTY)]
      (is (::comb/compiled-path (meta (:trace result))) "scan compiled path used"))))

(deftest scan-no-constraints-test
  (testing "Scan: no constraints (weight=0)"
    (let [scan (comb/scan-combinator k-scan)
          inputs [(mx/scalar 0.1) (mx/scalar 0.2) (mx/scalar 0.3)]
          result (p/generate scan [(mx/scalar 0.0) inputs] cm/EMPTY)
          trace (:trace result)]
      (mx/eval! (:weight result) (:score trace))
      (is (h/close? 0.0 (h/realize (:weight result)) 1e-6) "scan no-constraint weight=0")
      (is (= 3 (count (:outputs (:retval trace)))) "scan outputs count")
      (is (mx/array? (:carry (:retval trace))) "scan carry present")
      (is (js/isFinite (mx/item (:score trace))) "scan score finite"))))

(deftest scan-all-constrained-test
  (testing "Scan: all steps constrained"
    (let [scan (comb/scan-combinator k-scan)
          inputs [(mx/scalar 0.1) (mx/scalar 0.2) (mx/scalar 0.3)]
          obs (-> cm/EMPTY
                  (cm/set-choice [0 :x] (mx/scalar 0.15))
                  (cm/set-choice [1 :x] (mx/scalar 0.4))
                  (cm/set-choice [2 :x] (mx/scalar 0.8)))
          result (p/generate scan [(mx/scalar 0.0) inputs] obs)
          scan-h (comb/scan-combinator (force-handler k-scan))
          result-h (p/generate scan-h [(mx/scalar 0.0) inputs] obs)]
      (mx/eval! (:weight result) (:score (:trace result))
                (:weight result-h) (:score (:trace result-h)))
      (is (h/close? (mx/item (:weight result-h)) (h/realize (:weight result)) 1e-5)
          "scan constrained weight matches handler")
      (is (h/close? (mx/item (:score (:trace result-h))) (h/realize (:score (:trace result))) 1e-5)
          "scan constrained score matches handler"))))

(deftest scan-partial-constraints-test
  (testing "Scan: partial constraints"
    (let [scan (comb/scan-combinator k-scan)
          inputs [(mx/scalar 0.1) (mx/scalar 0.2) (mx/scalar 0.3)]
          obs (-> cm/EMPTY (cm/set-choice [1 :x] (mx/scalar 0.4)))
          result (p/generate scan [(mx/scalar 0.0) inputs] obs)
          trace (:trace result)]
      (mx/eval! (:weight result) (:score trace))
      (is (not= 0.0 (mx/item (:weight result))) "scan partial weight nonzero")
      (is (h/close? 0.4 (h/realize (cm/get-choice (:choices trace) [1 :x])) 1e-6)
          "scan partial constrained value"))))

(deftest scan-carry-threading-test
  (testing "Scan: carry threading"
    (let [scan (comb/scan-combinator k-scan)
          inputs [(mx/scalar 0.1) (mx/scalar 0.2) (mx/scalar 0.3)]
          obs (-> cm/EMPTY
                  (cm/set-choice [0 :x] (mx/scalar 0.5))
                  (cm/set-choice [1 :x] (mx/scalar 1.0))
                  (cm/set-choice [2 :x] (mx/scalar 1.5)))
          result (p/generate scan [(mx/scalar 0.0) inputs] obs)
          retval (:retval (:trace result))]
      (mx/eval! (:carry retval))
      (is (h/close? 1.5 (h/realize (:carry retval)) 1e-6) "scan final carry")
      (is (= 3 (count (:outputs retval))) "scan outputs count"))))

(deftest scan-trace-structure-test
  (testing "Scan: trace structure"
    (let [scan (comb/scan-combinator k-scan)
          inputs [(mx/scalar 0.1) (mx/scalar 0.2)]
          result (p/generate scan [(mx/scalar 0.0) inputs] cm/EMPTY)
          choices (:choices (:trace result))]
      (is (mx/array? (cm/get-choice choices [0 :x])) "scan choices[0 :x] present")
      (is (mx/array? (cm/get-choice choices [1 :x])) "scan choices[1 :x] present"))))

(deftest scan-non-compilable-fallback-test
  (testing "Scan: non-compilable fallback"
    (let [k (dyn/auto-key (gen [carry input]
              (let [x (trace :x (dist/beta-dist 2 5))] [x x])))
          scan (comb/scan-combinator k)
          inputs [(mx/scalar 0.1) (mx/scalar 0.2)]
          result (p/generate scan [(mx/scalar 0.0) inputs] cm/EMPTY)]
      (is (instance? tr/Trace (:trace result)) "scan fallback valid")
      (is (not (::comb/compiled-path (meta (:trace result)))) "scan fallback NOT compiled"))))

(deftest switch-compiled-dispatch-test
  (testing "Switch: compiled dispatch"
    (let [sw (comb/switch-combinator k-switch k-switch-b)
          result (p/generate sw [0] cm/EMPTY)]
      (is (::comb/compiled-path (meta (:trace result))) "switch compiled path used"))))

(deftest switch-no-constraints-test
  (testing "Switch: no constraints (weight=0)"
    (let [sw (comb/switch-combinator k-switch k-switch-b)
          result (p/generate sw [0] cm/EMPTY)
          trace (:trace result)]
      (mx/eval! (:weight result) (:score trace))
      (is (h/close? 0.0 (h/realize (:weight result)) 1e-6) "switch no-constraint weight=0")
      (is (js/isFinite (mx/item (:score trace))) "switch score finite"))))

(deftest switch-fully-constrained-test
  (testing "Switch: fully constrained"
    (let [sw (comb/switch-combinator k-switch k-switch-b)
          obs (-> cm/EMPTY (cm/set-value :x (mx/scalar 2.0)))
          result (p/generate sw [0] obs)
          sw-h (comb/switch-combinator (force-handler k-switch) (force-handler k-switch-b))
          result-h (p/generate sw-h [0] obs)]
      (mx/eval! (:weight result) (:score (:trace result))
                (:weight result-h) (:score (:trace result-h)))
      (is (h/close? (mx/item (:weight result-h)) (h/realize (:weight result)) 1e-5)
          "switch constrained weight matches handler")
      (is (h/close? (mx/item (:score (:trace result-h))) (h/realize (:score (:trace result))) 1e-5)
          "switch constrained score matches handler")
      (is (h/close? 2.0 (h/realize (cm/get-value (cm/get-submap (:choices (:trace result)) :x))) 1e-6)
          "switch constrained value"))))

(deftest switch-different-branches-test
  (testing "Switch: different branch indices"
    (let [sw (comb/switch-combinator k-switch k-switch-b)
          obs (-> cm/EMPTY (cm/set-value :x (mx/scalar 3.0)))
          r0 (p/generate sw [0] obs)
          r1 (p/generate sw [1] obs)]
      (mx/eval! (:weight r0) (:weight r1) (:score (:trace r0)) (:score (:trace r1)))
      (is (> (js/Math.abs (- (mx/item (:score (:trace r0)))
                             (mx/item (:score (:trace r1))))) 0.1)
          "switch branch scores differ"))))

(deftest switch-metadata-test
  (testing "Switch: metadata preserved"
    (let [sw (comb/switch-combinator k-switch k-switch-b)
          r0 (p/generate sw [0] cm/EMPTY)
          r1 (p/generate sw [1] cm/EMPTY)]
      (is (= 0 (::comb/switch-idx (meta (:trace r0)))) "switch-idx branch 0")
      (is (= 1 (::comb/switch-idx (meta (:trace r1)))) "switch-idx branch 1"))))

(deftest switch-non-compilable-fallback-test
  (testing "Switch: non-compilable fallback"
    (let [branch (dyn/auto-key (gen [] (let [x (trace :x (dist/beta-dist 2 5))] x)))
          sw (comb/switch-combinator branch)
          result (p/generate sw [0] cm/EMPTY)]
      (is (instance? tr/Trace (:trace result)) "switch fallback valid")
      (is (not (::comb/compiled-path (meta (:trace result)))) "switch fallback NOT compiled"))))

(deftest switch-mixed-branches-test
  (testing "Switch: mixed branches"
    (let [compilable (dyn/auto-key (gen [] (let [x (trace :x (dist/gaussian 0 1))] x)))
          non-comp (dyn/auto-key (gen [] (let [x (trace :x (dist/beta-dist 2 5))] x)))
          sw (comb/switch-combinator compilable non-comp)
          r0 (p/generate sw [0] cm/EMPTY)
          r1 (p/generate sw [1] cm/EMPTY)]
      (is (::comb/compiled-path (meta (:trace r0))) "switch mixed: compilable branch compiled")
      (is (not (::comb/compiled-path (meta (:trace r1)))) "switch mixed: non-compilable branch NOT compiled"))))

(deftest mix-compiled-dispatch-test
  (testing "Mix: compiled dispatch"
    (let [mix (comb/mix-combinator [k-switch k-switch-b]
                (fn [_] (mx/array [0.0 0.0])))
          result (p/generate mix [] cm/EMPTY)]
      (is (::comb/compiled-path (meta (:trace result))) "mix compiled path used"))))

(deftest mix-no-constraints-test
  (testing "Mix: no constraints"
    (let [mix (comb/mix-combinator [k-switch k-switch-b]
                (fn [_] (mx/array [0.0 0.0])))
          result (p/generate mix [] cm/EMPTY)
          trace (:trace result)]
      (mx/eval! (:weight result) (:score trace))
      (is (h/close? 0.0 (h/realize (:weight result)) 1e-6) "mix no-constraint weight=0")
      (is (js/isFinite (mx/item (:score trace))) "mix score finite")
      (is (mx/array? (cm/get-choice (:choices trace) [:component-idx])) "mix component-idx present"))))

(deftest mix-fully-constrained-test
  (testing "Mix: idx + component constrained matches handler"
    (let [mix (comb/mix-combinator [k-switch k-switch-b]
                (fn [_] (mx/array [0.0 0.0])))
          obs (-> cm/EMPTY
                  (cm/set-choice [:component-idx] (mx/scalar 0 mx/int32))
                  (cm/set-value :x (mx/scalar 2.0)))
          result (p/generate mix [] obs)
          mix-h (comb/mix-combinator [(force-handler k-switch) (force-handler k-switch-b)]
                  (fn [_] (mx/array [0.0 0.0])))
          result-h (p/generate mix-h [] obs)]
      (mx/eval! (:weight result) (:score (:trace result))
                (:weight result-h) (:score (:trace result-h)))
      (is (h/close? (mx/item (:weight result-h)) (h/realize (:weight result)) 1e-5)
          "mix fully-constrained weight matches handler")
      (is (h/close? (mx/item (:score (:trace result-h))) (h/realize (:score (:trace result))) 1e-5)
          "mix fully-constrained score matches handler"))))

(deftest mix-idx-and-component-constrained-test
  (testing "Mix: index + component constrained"
    (let [mix (comb/mix-combinator [k-switch k-switch-b]
                (fn [_] (mx/array [0.0 0.0])))
          obs (-> cm/EMPTY
                  (cm/set-choice [:component-idx] (mx/scalar 1 mx/int32))
                  (cm/set-value :x (mx/scalar 4.0)))
          result (p/generate mix [] obs)
          trace (:trace result)]
      (mx/eval! (:weight result) (:score trace))
      (is (not= 0.0 (mx/item (:weight result))) "mix idx+comp weight nonzero")
      (is (h/close? 4.0 (h/realize (cm/get-choice (:choices trace) [:x])) 1e-6) "mix constrained x")
      (is (= 1 (mx/item (cm/get-choice (:choices trace) [:component-idx]))) "mix constrained idx"))))

(deftest mix-idx-only-constrained-test
  (testing "Mix: index constrained only"
    (let [mix (comb/mix-combinator [k-switch k-switch-b]
                (fn [_] (mx/array [0.0 0.0])))
          obs (-> cm/EMPTY
                  (cm/set-choice [:component-idx] (mx/scalar 0 mx/int32)))
          result (p/generate mix [] obs)
          trace (:trace result)]
      (mx/eval! (:weight result) (:score trace))
      (is (not= 0.0 (mx/item (:weight result))) "mix idx-only weight nonzero")
      (is (= 0 (mx/item (cm/get-choice (:choices trace) [:component-idx]))) "mix idx-only component"))))

(deftest mix-non-compilable-fallback-test
  (testing "Mix: non-compilable fallback"
    (let [comp-beta (dyn/auto-key (gen [] (let [x (trace :x (dist/beta-dist 2 5))] x)))
          mix (comb/mix-combinator [comp-beta]
                (fn [_] (mx/array [0.0])))
          result (p/generate mix [] cm/EMPTY)]
      (is (instance? tr/Trace (:trace result)) "mix fallback valid")
      (is (not (::comb/compiled-path (meta (:trace result)))) "mix fallback NOT compiled"))))

(deftest metadata-map-element-scores-test
  (testing "Metadata: Map ::element-scores"
    (let [mapped (comb/map-combinator k-map)
          result (p/generate mapped [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]] cm/EMPTY)
          trace (:trace result)
          es (::comb/element-scores (meta trace))]
      (is (some? es) "map meta: ::element-scores present")
      (is (= 3 (count es)) "map meta: ::element-scores count")
      (mx/eval! (first es) (second es) (nth es 2))
      (is (js/isFinite (mx/item (first es))) "map meta: element-scores[0] finite")
      (is (js/isFinite (mx/item (second es))) "map meta: element-scores[1] finite")
      (is (js/isFinite (mx/item (nth es 2))) "map meta: element-scores[2] finite"))))

(deftest metadata-unfold-step-scores-test
  (testing "Metadata: Unfold ::step-scores"
    (let [unfold (comb/unfold-combinator k-unfold)
          result (p/generate unfold [3 (mx/scalar 0.5)] cm/EMPTY)
          trace (:trace result)
          ss (::comb/step-scores (meta trace))]
      (is (some? ss) "unfold meta: ::step-scores present")
      (is (= 3 (count ss)) "unfold meta: ::step-scores count")
      (mx/eval! (first ss) (second ss) (nth ss 2))
      (is (js/isFinite (mx/item (first ss))) "unfold meta: step-scores[0] finite")
      (is (js/isFinite (mx/item (nth ss 2))) "unfold meta: step-scores[2] finite"))))

(deftest metadata-scan-step-scores-and-carries-test
  (testing "Metadata: Scan ::step-scores + ::step-carries"
    (let [scan (comb/scan-combinator k-scan)
          inputs [(mx/scalar 0.1) (mx/scalar 0.2) (mx/scalar 0.3)]
          result (p/generate scan [(mx/scalar 0.0) inputs] cm/EMPTY)
          trace (:trace result)
          ss (::comb/step-scores (meta trace))
          sc (::comb/step-carries (meta trace))]
      (is (some? ss) "scan meta: ::step-scores present")
      (is (= 3 (count ss)) "scan meta: ::step-scores count")
      (is (some? sc) "scan meta: ::step-carries present")
      (is (= 3 (count sc)) "scan meta: ::step-carries count")
      (mx/eval! (first ss) (nth ss 2) (first sc) (nth sc 2))
      (is (js/isFinite (mx/item (first ss))) "scan meta: step-scores[0] finite")
      (is (mx/array? (first sc)) "scan meta: step-carries[0] is array"))))

(deftest metadata-switch-idx-test
  (testing "Metadata: Switch ::switch-idx"
    (let [sw (comb/switch-combinator k-switch k-switch-b)
          r0 (p/generate sw [0] cm/EMPTY)
          r1 (p/generate sw [1] cm/EMPTY)]
      (is (= 0 (::comb/switch-idx (meta (:trace r0)))) "switch meta: idx=0")
      (is (= 1 (::comb/switch-idx (meta (:trace r1)))) "switch meta: idx=1")
      (is (::comb/compiled-path (meta (:trace r0))) "switch meta: ::compiled-path on idx=0"))))

(deftest prng-consistency-test
  (testing "PRNG consistency: Map"
    (let [mapped (comb/map-combinator k-map)
          result (p/generate mapped [[(mx/scalar 1.0) (mx/scalar 2.0)]] cm/EMPTY)
          trace (:trace result)]
      (mx/eval! (:score trace) (:weight result))
      (is (js/isFinite (mx/item (:score trace))) "map prng: score finite")
      (is (h/close? 0.0 (h/realize (:weight result)) 1e-6) "map prng: weight=0")))

  (testing "PRNG consistency: Unfold"
    (let [unfold (comb/unfold-combinator k-unfold)
          result (p/generate unfold [3 (mx/scalar 0.5)] cm/EMPTY)
          trace (:trace result)]
      (mx/eval! (:score trace) (:weight result))
      (is (js/isFinite (mx/item (:score trace))) "unfold prng: score finite")
      (is (h/close? 0.0 (h/realize (:weight result)) 1e-6) "unfold prng: weight=0")))

  (testing "PRNG consistency: Scan"
    (let [scan (comb/scan-combinator k-scan)
          inputs [(mx/scalar 0.1) (mx/scalar 0.2)]
          result (p/generate scan [(mx/scalar 0.0) inputs] cm/EMPTY)
          trace (:trace result)]
      (mx/eval! (:score trace) (:weight result))
      (is (js/isFinite (mx/item (:score trace))) "scan prng: score finite")
      (is (h/close? 0.0 (h/realize (:weight result)) 1e-6) "scan prng: weight=0")))

  (testing "PRNG consistency: Switch"
    (let [sw (comb/switch-combinator k-switch)
          result (p/generate sw [0] cm/EMPTY)
          trace (:trace result)]
      (mx/eval! (:score trace) (:weight result))
      (is (js/isFinite (mx/item (:score trace))) "switch prng: score finite")
      (is (h/close? 0.0 (h/realize (:weight result)) 1e-6) "switch prng: weight=0"))))

(cljs.test/run-tests)
