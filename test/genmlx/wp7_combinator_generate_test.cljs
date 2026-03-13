(ns genmlx.wp7-combinator-generate-test
  "WP-7 tests: combinator compiled generate.
   Validates that compiled generate paths for Map, Unfold, Scan, Switch, Mix
   produce identical traces, scores, and weights as handler paths."
  (:require-macros [genmlx.gen :refer [gen]])
  (:require [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.compiled-ops :as compiled]
            [genmlx.combinators :as comb]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def pass-count (atom 0))
(def fail-count (atom 0))

(defn assert-true [msg pred]
  (if pred
    (do (swap! pass-count inc) (println (str "  PASS: " msg)))
    (do (swap! fail-count inc) (println (str "  FAIL: " msg)))))

(defn assert-close [msg expected actual tol]
  (let [e (if (number? expected) expected (mx/item expected))
        a (if (number? actual) actual (mx/item actual))
        diff (js/Math.abs (- e a))]
    (if (<= diff tol)
      (do (swap! pass-count inc) (println (str "  PASS: " msg)))
      (do (swap! fail-count inc)
          (println (str "  FAIL: " msg " expected=" e " actual=" a " diff=" diff))))))

(defn assert-equal [msg expected actual]
  (if (= expected actual)
    (do (swap! pass-count inc) (println (str "  PASS: " msg)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " msg " expected=" expected " actual=" actual)))))

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

(println "\n=== WP-7: Combinator Compiled Generate Tests ===\n")

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
;; Section 0: Prerequisites
;; ---------------------------------------------------------------------------

(println "-- Prerequisites --")
(assert-true "map kernel has compiled-generate" (compilable-generate? k-map))
(assert-true "unfold kernel has compiled-generate" (compilable-generate? k-unfold))
(assert-true "scan kernel has compiled-generate" (compilable-generate? k-scan))
(assert-true "switch kernel has compiled-generate" (compilable-generate? k-switch))
(assert-true "beta kernel NOT compilable" (not (compilable-generate? k-beta)))

;; ---------------------------------------------------------------------------
;; Section 1: Map Combinator
;; ---------------------------------------------------------------------------

(println "\n-- Map: compiled dispatch --")
(let [mapped (comb/map-combinator k-map)
      result (p/generate mapped [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]] cm/EMPTY)]
  (assert-true "map compiled path used"
    (::comb/compiled-path (meta (:trace result)))))

(println "\n-- Map: no constraints (weight=0) --")
(let [mapped (comb/map-combinator k-map)
      result (p/generate mapped [[(mx/scalar 1.0) (mx/scalar 2.0)]] cm/EMPTY)
      trace (:trace result)]
  (mx/eval! (:weight result) (:score trace))
  (assert-close "map no-constraint weight=0" 0.0 (:weight result) 1e-6)
  (assert-true "map trace valid" (instance? tr/Trace trace))
  (assert-equal "map retval count" 2 (count (:retval trace)))
  (assert-true "map score finite" (js/isFinite (mx/item (:score trace)))))

(println "\n-- Map: all elements constrained --")
(let [mapped (comb/map-combinator k-map)
      obs (-> cm/EMPTY
              (cm/set-choice [0 :y] (mx/scalar 1.5))
              (cm/set-choice [1 :y] (mx/scalar 2.5)))
      result (p/generate mapped [[(mx/scalar 1.0) (mx/scalar 2.0)]] obs)
      ;; Compare with handler
      mapped-h (comb/map-combinator (force-handler k-map))
      result-h (p/generate mapped-h [[(mx/scalar 1.0) (mx/scalar 2.0)]] obs)]
  (mx/eval! (:weight result) (:score (:trace result))
            (:weight result-h) (:score (:trace result-h)))
  (assert-close "map constrained weight matches handler"
    (mx/item (:weight result-h)) (:weight result) 1e-5)
  (assert-close "map constrained score matches handler"
    (mx/item (:score (:trace result-h))) (:score (:trace result)) 1e-5)
  (assert-close "map constrained val[0]"
    1.5 (cm/get-choice (:choices (:trace result)) [0 :y]) 1e-6)
  (assert-close "map constrained val[1]"
    2.5 (cm/get-choice (:choices (:trace result)) [1 :y]) 1e-6))

(println "\n-- Map: partial constraints --")
(let [mapped (comb/map-combinator k-map)
      obs (-> cm/EMPTY (cm/set-choice [1 :y] (mx/scalar 5.0)))
      result (p/generate mapped [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]] obs)
      trace (:trace result)]
  (mx/eval! (:weight result) (:score trace))
  (assert-true "map partial weight nonzero" (not= 0.0 (mx/item (:weight result))))
  (assert-close "map partial constrained value" 5.0
    (cm/get-choice (:choices trace) [1 :y]) 1e-6)
  (assert-equal "map partial retval count" 3 (count (:retval trace))))

(println "\n-- Map: trace structure --")
(let [mapped (comb/map-combinator k-map)
      result (p/generate mapped [[(mx/scalar 1.0) (mx/scalar 2.0)]] cm/EMPTY)
      trace (:trace result)
      choices (:choices trace)]
  (assert-true "map choices[0 :y] present"
    (mx/array? (cm/get-choice choices [0 :y])))
  (assert-true "map choices[1 :y] present"
    (mx/array? (cm/get-choice choices [1 :y]))))

(println "\n-- Map: non-compilable fallback --")
(let [mapped (comb/map-combinator k-beta)
      result (p/generate mapped [[(mx/scalar 1.0) (mx/scalar 2.0)]] cm/EMPTY)]
  (assert-true "map fallback trace valid" (instance? tr/Trace (:trace result)))
  (assert-true "map fallback NOT compiled"
    (not (::comb/compiled-path (meta (:trace result))))))

;; ---------------------------------------------------------------------------
;; Section 2: Unfold Combinator
;; ---------------------------------------------------------------------------

(println "\n-- Unfold: compiled dispatch --")
(let [unfold (comb/unfold-combinator k-unfold)
      result (p/generate unfold [3 (mx/scalar 0.5)] cm/EMPTY)]
  (assert-true "unfold compiled path used"
    (::comb/compiled-path (meta (:trace result)))))

(println "\n-- Unfold: no constraints (weight=0) --")
(let [unfold (comb/unfold-combinator k-unfold)
      result (p/generate unfold [3 (mx/scalar 0.5)] cm/EMPTY)
      trace (:trace result)]
  (mx/eval! (:weight result) (:score trace))
  (assert-close "unfold no-constraint weight=0" 0.0 (:weight result) 1e-6)
  (assert-equal "unfold retval count" 3 (count (:retval trace)))
  (assert-true "unfold score finite" (js/isFinite (mx/item (:score trace)))))

(println "\n-- Unfold: all steps constrained --")
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
  (assert-close "unfold constrained weight matches handler"
    (mx/item (:weight result-h)) (:weight result) 1e-5)
  (assert-close "unfold constrained score matches handler"
    (mx/item (:score (:trace result-h))) (:score (:trace result)) 1e-5))

(println "\n-- Unfold: partial constraints --")
(let [unfold (comb/unfold-combinator k-unfold)
      obs (-> cm/EMPTY (cm/set-choice [1 :x] (mx/scalar 0.7)))
      result (p/generate unfold [3 (mx/scalar 0.5)] obs)
      trace (:trace result)]
  (mx/eval! (:weight result) (:score trace))
  (assert-true "unfold partial weight nonzero" (not= 0.0 (mx/item (:weight result))))
  (assert-close "unfold partial constrained value" 0.7
    (cm/get-choice (:choices trace) [1 :x]) 1e-6))

(println "\n-- Unfold: state threading --")
(let [unfold (comb/unfold-combinator k-unfold)
      obs (-> cm/EMPTY
              (cm/set-choice [0 :x] (mx/scalar 1.0))
              (cm/set-choice [1 :x] (mx/scalar 2.0))
              (cm/set-choice [2 :x] (mx/scalar 3.0)))
      result (p/generate unfold [3 (mx/scalar 0.5)] obs)
      trace (:trace result)
      states (:retval trace)]
  (mx/eval! (first states) (second states) (nth states 2))
  ;; Kernel returns the sampled value as new state
  (assert-close "unfold state[0] = constrained :x" 1.0 (first states) 1e-6)
  (assert-close "unfold state[1] = constrained :x" 2.0 (second states) 1e-6)
  (assert-close "unfold state[2] = constrained :x" 3.0 (nth states 2) 1e-6))

(println "\n-- Unfold: trace structure --")
(let [unfold (comb/unfold-combinator k-unfold)
      result (p/generate unfold [3 (mx/scalar 0.5)] cm/EMPTY)
      choices (:choices (:trace result))]
  (assert-true "unfold choices[0 :x] present"
    (mx/array? (cm/get-choice choices [0 :x])))
  (assert-true "unfold choices[1 :x] present"
    (mx/array? (cm/get-choice choices [1 :x])))
  (assert-true "unfold choices[2 :x] present"
    (mx/array? (cm/get-choice choices [2 :x]))))

(println "\n-- Unfold: non-compilable fallback --")
(let [k (dyn/auto-key (gen [t state] (let [x (trace :x (dist/beta-dist 2 5))] x)))
      unfold (comb/unfold-combinator k)
      result (p/generate unfold [3 (mx/scalar 0.5)] cm/EMPTY)]
  (assert-true "unfold fallback valid" (instance? tr/Trace (:trace result)))
  (assert-true "unfold fallback NOT compiled"
    (not (::comb/compiled-path (meta (:trace result))))))

;; ---------------------------------------------------------------------------
;; Section 3: Scan Combinator
;; ---------------------------------------------------------------------------

(println "\n-- Scan: compiled dispatch --")
(let [scan (comb/scan-combinator k-scan)
      inputs [(mx/scalar 0.1) (mx/scalar 0.2) (mx/scalar 0.3)]
      result (p/generate scan [(mx/scalar 0.0) inputs] cm/EMPTY)]
  (assert-true "scan compiled path used"
    (::comb/compiled-path (meta (:trace result)))))

(println "\n-- Scan: no constraints (weight=0) --")
(let [scan (comb/scan-combinator k-scan)
      inputs [(mx/scalar 0.1) (mx/scalar 0.2) (mx/scalar 0.3)]
      result (p/generate scan [(mx/scalar 0.0) inputs] cm/EMPTY)
      trace (:trace result)]
  (mx/eval! (:weight result) (:score trace))
  (assert-close "scan no-constraint weight=0" 0.0 (:weight result) 1e-6)
  (assert-equal "scan outputs count" 3 (count (:outputs (:retval trace))))
  (assert-true "scan carry present" (mx/array? (:carry (:retval trace))))
  (assert-true "scan score finite" (js/isFinite (mx/item (:score trace)))))

(println "\n-- Scan: all steps constrained --")
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
  (assert-close "scan constrained weight matches handler"
    (mx/item (:weight result-h)) (:weight result) 1e-5)
  (assert-close "scan constrained score matches handler"
    (mx/item (:score (:trace result-h))) (:score (:trace result)) 1e-5))

(println "\n-- Scan: partial constraints --")
(let [scan (comb/scan-combinator k-scan)
      inputs [(mx/scalar 0.1) (mx/scalar 0.2) (mx/scalar 0.3)]
      obs (-> cm/EMPTY (cm/set-choice [1 :x] (mx/scalar 0.4)))
      result (p/generate scan [(mx/scalar 0.0) inputs] obs)
      trace (:trace result)]
  (mx/eval! (:weight result) (:score trace))
  (assert-true "scan partial weight nonzero" (not= 0.0 (mx/item (:weight result))))
  (assert-close "scan partial constrained value" 0.4
    (cm/get-choice (:choices trace) [1 :x]) 1e-6))

(println "\n-- Scan: carry threading --")
(let [scan (comb/scan-combinator k-scan)
      inputs [(mx/scalar 0.1) (mx/scalar 0.2) (mx/scalar 0.3)]
      obs (-> cm/EMPTY
              (cm/set-choice [0 :x] (mx/scalar 0.5))
              (cm/set-choice [1 :x] (mx/scalar 1.0))
              (cm/set-choice [2 :x] (mx/scalar 1.5)))
      result (p/generate scan [(mx/scalar 0.0) inputs] obs)
      retval (:retval (:trace result))]
  ;; Kernel: [carry input] -> (trace :x (gaussian (+ carry input) 0.1)) -> [x x]
  ;; So carry = x at each step (retval is [new-carry output])
  (mx/eval! (:carry retval))
  (assert-close "scan final carry" 1.5 (:carry retval) 1e-6)
  (assert-equal "scan outputs count" 3 (count (:outputs retval))))

(println "\n-- Scan: trace structure --")
(let [scan (comb/scan-combinator k-scan)
      inputs [(mx/scalar 0.1) (mx/scalar 0.2)]
      result (p/generate scan [(mx/scalar 0.0) inputs] cm/EMPTY)
      choices (:choices (:trace result))]
  (assert-true "scan choices[0 :x] present"
    (mx/array? (cm/get-choice choices [0 :x])))
  (assert-true "scan choices[1 :x] present"
    (mx/array? (cm/get-choice choices [1 :x]))))

(println "\n-- Scan: non-compilable fallback --")
(let [k (dyn/auto-key (gen [carry input]
          (let [x (trace :x (dist/beta-dist 2 5))] [x x])))
      scan (comb/scan-combinator k)
      inputs [(mx/scalar 0.1) (mx/scalar 0.2)]
      result (p/generate scan [(mx/scalar 0.0) inputs] cm/EMPTY)]
  (assert-true "scan fallback valid" (instance? tr/Trace (:trace result)))
  (assert-true "scan fallback NOT compiled"
    (not (::comb/compiled-path (meta (:trace result))))))

;; ---------------------------------------------------------------------------
;; Section 4: Switch Combinator
;; ---------------------------------------------------------------------------

(println "\n-- Switch: compiled dispatch --")
(let [sw (comb/switch-combinator k-switch k-switch-b)
      result (p/generate sw [0] cm/EMPTY)]
  (assert-true "switch compiled path used"
    (::comb/compiled-path (meta (:trace result)))))

(println "\n-- Switch: no constraints (weight=0) --")
(let [sw (comb/switch-combinator k-switch k-switch-b)
      result (p/generate sw [0] cm/EMPTY)
      trace (:trace result)]
  (mx/eval! (:weight result) (:score trace))
  (assert-close "switch no-constraint weight=0" 0.0 (:weight result) 1e-6)
  (assert-true "switch score finite" (js/isFinite (mx/item (:score trace)))))

(println "\n-- Switch: fully constrained --")
(let [sw (comb/switch-combinator k-switch k-switch-b)
      obs (-> cm/EMPTY (cm/set-value :x (mx/scalar 2.0)))
      result (p/generate sw [0] obs)
      sw-h (comb/switch-combinator (force-handler k-switch) (force-handler k-switch-b))
      result-h (p/generate sw-h [0] obs)]
  (mx/eval! (:weight result) (:score (:trace result))
            (:weight result-h) (:score (:trace result-h)))
  (assert-close "switch constrained weight matches handler"
    (mx/item (:weight result-h)) (:weight result) 1e-5)
  (assert-close "switch constrained score matches handler"
    (mx/item (:score (:trace result-h))) (:score (:trace result)) 1e-5)
  (assert-close "switch constrained value" 2.0
    (cm/get-value (cm/get-submap (:choices (:trace result)) :x)) 1e-6))

(println "\n-- Switch: different branch indices --")
(let [sw (comb/switch-combinator k-switch k-switch-b)
      obs (-> cm/EMPTY (cm/set-value :x (mx/scalar 3.0)))
      r0 (p/generate sw [0] obs)
      r1 (p/generate sw [1] obs)]
  (mx/eval! (:weight r0) (:weight r1) (:score (:trace r0)) (:score (:trace r1)))
  ;; Branch 0: gaussian(0,1), Branch 1: gaussian(5,0.5) — different scores for x=3
  (assert-true "switch branch scores differ"
    (> (js/Math.abs (- (mx/item (:score (:trace r0)))
                       (mx/item (:score (:trace r1))))) 0.1)))

(println "\n-- Switch: metadata preserved --")
(let [sw (comb/switch-combinator k-switch k-switch-b)
      r0 (p/generate sw [0] cm/EMPTY)
      r1 (p/generate sw [1] cm/EMPTY)]
  (assert-equal "switch-idx branch 0" 0 (::comb/switch-idx (meta (:trace r0))))
  (assert-equal "switch-idx branch 1" 1 (::comb/switch-idx (meta (:trace r1)))))

(println "\n-- Switch: non-compilable fallback --")
(let [branch (dyn/auto-key (gen [] (let [x (trace :x (dist/beta-dist 2 5))] x)))
      sw (comb/switch-combinator branch)
      result (p/generate sw [0] cm/EMPTY)]
  (assert-true "switch fallback valid" (instance? tr/Trace (:trace result)))
  (assert-true "switch fallback NOT compiled"
    (not (::comb/compiled-path (meta (:trace result))))))

(println "\n-- Switch: mixed branches --")
(let [compilable (dyn/auto-key (gen [] (let [x (trace :x (dist/gaussian 0 1))] x)))
      non-comp (dyn/auto-key (gen [] (let [x (trace :x (dist/beta-dist 2 5))] x)))
      sw (comb/switch-combinator compilable non-comp)
      r0 (p/generate sw [0] cm/EMPTY)
      r1 (p/generate sw [1] cm/EMPTY)]
  (assert-true "switch mixed: compilable branch compiled"
    (::comb/compiled-path (meta (:trace r0))))
  (assert-true "switch mixed: non-compilable branch NOT compiled"
    (not (::comb/compiled-path (meta (:trace r1))))))

;; ---------------------------------------------------------------------------
;; Section 5: Mix Combinator
;; ---------------------------------------------------------------------------

(println "\n-- Mix: compiled dispatch --")
(let [mix (comb/mix-combinator [k-switch k-switch-b]
            (fn [_] (mx/array [0.0 0.0])))
      result (p/generate mix [] cm/EMPTY)]
  (assert-true "mix compiled path used"
    (::comb/compiled-path (meta (:trace result)))))

(println "\n-- Mix: no constraints --")
(let [mix (comb/mix-combinator [k-switch k-switch-b]
            (fn [_] (mx/array [0.0 0.0])))
      result (p/generate mix [] cm/EMPTY)
      trace (:trace result)]
  (mx/eval! (:weight result) (:score trace))
  ;; Weight should be 0 for unconstrained (both idx and component)
  (assert-close "mix no-constraint weight=0" 0.0 (:weight result) 1e-6)
  (assert-true "mix score finite" (js/isFinite (mx/item (:score trace))))
  ;; component-idx should be in choices
  (assert-true "mix component-idx present"
    (mx/array? (cm/get-choice (:choices trace) [:component-idx]))))

(println "\n-- Mix: idx + component constrained matches handler --")
(let [mix (comb/mix-combinator [k-switch k-switch-b]
            (fn [_] (mx/array [0.0 0.0])))
      ;; Constrain both idx and x to eliminate PRNG differences
      obs (-> cm/EMPTY
              (cm/set-choice [:component-idx] (mx/scalar 0 mx/int32))
              (cm/set-value :x (mx/scalar 2.0)))
      result (p/generate mix [] obs)
      mix-h (comb/mix-combinator [(force-handler k-switch) (force-handler k-switch-b)]
              (fn [_] (mx/array [0.0 0.0])))
      result-h (p/generate mix-h [] obs)]
  (mx/eval! (:weight result) (:score (:trace result))
            (:weight result-h) (:score (:trace result-h)))
  (assert-close "mix fully-constrained weight matches handler"
    (mx/item (:weight result-h)) (:weight result) 1e-5)
  (assert-close "mix fully-constrained score matches handler"
    (mx/item (:score (:trace result-h))) (:score (:trace result)) 1e-5))

(println "\n-- Mix: index + component constrained --")
(let [mix (comb/mix-combinator [k-switch k-switch-b]
            (fn [_] (mx/array [0.0 0.0])))
      obs (-> cm/EMPTY
              (cm/set-choice [:component-idx] (mx/scalar 1 mx/int32))
              (cm/set-value :x (mx/scalar 4.0)))
      result (p/generate mix [] obs)
      trace (:trace result)]
  (mx/eval! (:weight result) (:score trace))
  (assert-true "mix idx+comp weight nonzero" (not= 0.0 (mx/item (:weight result))))
  (assert-close "mix constrained x" 4.0
    (cm/get-choice (:choices trace) [:x]) 1e-6)
  (assert-equal "mix constrained idx" 1
    (mx/item (cm/get-choice (:choices trace) [:component-idx]))))

(println "\n-- Mix: index constrained only --")
(let [mix (comb/mix-combinator [k-switch k-switch-b]
            (fn [_] (mx/array [0.0 0.0])))
      obs (-> cm/EMPTY
              (cm/set-choice [:component-idx] (mx/scalar 0 mx/int32)))
      result (p/generate mix [] obs)
      trace (:trace result)]
  (mx/eval! (:weight result) (:score trace))
  (assert-true "mix idx-only weight nonzero" (not= 0.0 (mx/item (:weight result))))
  (assert-equal "mix idx-only component" 0
    (mx/item (cm/get-choice (:choices trace) [:component-idx]))))

(println "\n-- Mix: non-compilable fallback --")
(let [comp-beta (dyn/auto-key (gen [] (let [x (trace :x (dist/beta-dist 2 5))] x)))
      mix (comb/mix-combinator [comp-beta]
            (fn [_] (mx/array [0.0])))
      result (p/generate mix [] cm/EMPTY)]
  (assert-true "mix fallback valid" (instance? tr/Trace (:trace result)))
  (assert-true "mix fallback NOT compiled"
    (not (::comb/compiled-path (meta (:trace result))))))

;; ---------------------------------------------------------------------------
;; Section 6: Metadata (WP-8 readiness)
;; ---------------------------------------------------------------------------
;; Compiled generate must set the same metadata as handler paths.
;; update/regenerate rely on ::element-scores, ::step-scores, ::step-carries.

(println "\n-- Metadata: Map ::element-scores --")
(let [mapped (comb/map-combinator k-map)
      result (p/generate mapped [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]] cm/EMPTY)
      trace (:trace result)
      es (::comb/element-scores (meta trace))]
  (assert-true "map meta: ::element-scores present" (some? es))
  (assert-equal "map meta: ::element-scores count" 3 (count es))
  (mx/eval! (first es) (second es) (nth es 2))
  (assert-true "map meta: element-scores[0] finite" (js/isFinite (mx/item (first es))))
  (assert-true "map meta: element-scores[1] finite" (js/isFinite (mx/item (second es))))
  (assert-true "map meta: element-scores[2] finite" (js/isFinite (mx/item (nth es 2)))))

(println "\n-- Metadata: Unfold ::step-scores --")
(let [unfold (comb/unfold-combinator k-unfold)
      result (p/generate unfold [3 (mx/scalar 0.5)] cm/EMPTY)
      trace (:trace result)
      ss (::comb/step-scores (meta trace))]
  (assert-true "unfold meta: ::step-scores present" (some? ss))
  (assert-equal "unfold meta: ::step-scores count" 3 (count ss))
  (mx/eval! (first ss) (second ss) (nth ss 2))
  (assert-true "unfold meta: step-scores[0] finite" (js/isFinite (mx/item (first ss))))
  (assert-true "unfold meta: step-scores[2] finite" (js/isFinite (mx/item (nth ss 2)))))

(println "\n-- Metadata: Scan ::step-scores + ::step-carries --")
(let [scan (comb/scan-combinator k-scan)
      inputs [(mx/scalar 0.1) (mx/scalar 0.2) (mx/scalar 0.3)]
      result (p/generate scan [(mx/scalar 0.0) inputs] cm/EMPTY)
      trace (:trace result)
      ss (::comb/step-scores (meta trace))
      sc (::comb/step-carries (meta trace))]
  (assert-true "scan meta: ::step-scores present" (some? ss))
  (assert-equal "scan meta: ::step-scores count" 3 (count ss))
  (assert-true "scan meta: ::step-carries present" (some? sc))
  (assert-equal "scan meta: ::step-carries count" 3 (count sc))
  (mx/eval! (first ss) (nth ss 2) (first sc) (nth sc 2))
  (assert-true "scan meta: step-scores[0] finite" (js/isFinite (mx/item (first ss))))
  (assert-true "scan meta: step-carries[0] is array" (mx/array? (first sc))))

(println "\n-- Metadata: Switch ::switch-idx --")
(let [sw (comb/switch-combinator k-switch k-switch-b)
      r0 (p/generate sw [0] cm/EMPTY)
      r1 (p/generate sw [1] cm/EMPTY)]
  (assert-equal "switch meta: idx=0" 0 (::comb/switch-idx (meta (:trace r0))))
  (assert-equal "switch meta: idx=1" 1 (::comb/switch-idx (meta (:trace r1))))
  (assert-true "switch meta: ::compiled-path on idx=0"
    (::comb/compiled-path (meta (:trace r0)))))

;; ---------------------------------------------------------------------------
;; Section 7: PRNG consistency (generate(empty) ≈ simulate structure)
;; ---------------------------------------------------------------------------

(println "\n-- PRNG consistency: Map --")
(let [mapped (comb/map-combinator k-map)
      result (p/generate mapped [[(mx/scalar 1.0) (mx/scalar 2.0)]] cm/EMPTY)
      trace (:trace result)]
  (mx/eval! (:score trace) (:weight result))
  (assert-true "map prng: score finite" (js/isFinite (mx/item (:score trace))))
  (assert-close "map prng: weight=0" 0.0 (:weight result) 1e-6))

(println "\n-- PRNG consistency: Unfold --")
(let [unfold (comb/unfold-combinator k-unfold)
      result (p/generate unfold [3 (mx/scalar 0.5)] cm/EMPTY)
      trace (:trace result)]
  (mx/eval! (:score trace) (:weight result))
  (assert-true "unfold prng: score finite" (js/isFinite (mx/item (:score trace))))
  (assert-close "unfold prng: weight=0" 0.0 (:weight result) 1e-6))

(println "\n-- PRNG consistency: Scan --")
(let [scan (comb/scan-combinator k-scan)
      inputs [(mx/scalar 0.1) (mx/scalar 0.2)]
      result (p/generate scan [(mx/scalar 0.0) inputs] cm/EMPTY)
      trace (:trace result)]
  (mx/eval! (:score trace) (:weight result))
  (assert-true "scan prng: score finite" (js/isFinite (mx/item (:score trace))))
  (assert-close "scan prng: weight=0" 0.0 (:weight result) 1e-6))

(println "\n-- PRNG consistency: Switch --")
(let [sw (comb/switch-combinator k-switch)
      result (p/generate sw [0] cm/EMPTY)
      trace (:trace result)]
  (mx/eval! (:score trace) (:weight result))
  (assert-true "switch prng: score finite" (js/isFinite (mx/item (:score trace))))
  (assert-close "switch prng: weight=0" 0.0 (:weight result) 1e-6))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== WP-7 Results: " @pass-count "/" (+ @pass-count @fail-count)
              " passed ==="))
(when (pos? @fail-count)
  (println (str "  " @fail-count " FAILURES")))
