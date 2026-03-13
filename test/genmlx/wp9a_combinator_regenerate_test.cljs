(ns genmlx.wp9a-combinator-regenerate-test
  "WP-9A tests: combinator compiled regenerate.
   Validates that compiled regenerate paths for Map, Unfold, Scan, Switch, Mix
   produce correct weights, scores, and trace structure.
   Also validates the weight bug fix: regenerate(none) must return weight=0.
   Note: for simple single-site kernels, weight=0 for ALL selections
   (the proposal IS the prior, so proposal_ratio cancels)."
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
            [genmlx.selection :as sel]
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

(println "\n=== WP-9A: Combinator Compiled Regenerate Tests ===\n")

;; ---------------------------------------------------------------------------
;; Test kernels (same as WP-7/WP-8)
;; ---------------------------------------------------------------------------

(def k-map
  (dyn/auto-key (gen [x]
    (let [y (trace :y (dist/gaussian x 1.0))]
      y))))

(def k-unfold
  (dyn/auto-key (gen [t state]
    (let [x (trace :x (dist/gaussian state 0.1))]
      x))))

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

;; Non-compilable kernels (beta-dist has no noise transform)
(def k-beta
  (dyn/auto-key (gen [x]
    (let [y (trace :y (dist/beta-dist 2 5))]
      y))))

(def k-beta-scan
  (dyn/auto-key (gen [carry input]
    (let [y (trace :y (dist/beta-dist 2 5))]
      [y y]))))

(def k-beta-switch
  (dyn/auto-key (gen []
    (let [y (trace :y (dist/beta-dist 2 5))]
      y))))

;; ===========================================================================
;; Section 0: Prerequisites
;; ===========================================================================

(println "-- 0. Prerequisites --")
(assert-true "map kernel has compiled-regenerate"
             (some? (compiled/get-compiled-regenerate k-map)))
(assert-true "unfold kernel has compiled-regenerate"
             (some? (compiled/get-compiled-regenerate k-unfold)))
(assert-true "scan kernel has compiled-regenerate"
             (some? (compiled/get-compiled-regenerate k-scan)))
(assert-true "switch kernel has compiled-regenerate"
             (some? (compiled/get-compiled-regenerate k-switch)))
(assert-true "beta kernel NOT compilable-regenerate"
             (not (compiled/get-compiled-regenerate k-beta)))

;; ===========================================================================
;; Section 1: MAP COMBINATOR REGENERATE
;; ===========================================================================

(println "\n-- 1. Map: compiled regenerate --")

(let [map-c (comb/map-combinator k-map)
      map-h (comb/map-combinator (force-handler k-map))
      inputs [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]]
      init-constraints (-> cm/EMPTY
                           (cm/set-choice [0 :y] (mx/scalar 1.5))
                           (cm/set-choice [1 :y] (mx/scalar 2.5))
                           (cm/set-choice [2 :y] (mx/scalar 3.5)))
      trace-c (:trace (p/generate map-c inputs init-constraints))
      trace-h (:trace (p/generate map-h inputs init-constraints))]

  ;; 1a. Select none — weight must be 0
  (let [result-c (p/regenerate map-c trace-c sel/none)
        result-h (p/regenerate map-h trace-h sel/none)]
    (mx/eval! (:weight result-c) (:weight result-h))
    (assert-close "map regenerate(none) weight=0" 0.0 (:weight result-c) 1e-6)
    (assert-close "map regenerate(none) weight matches handler"
                  (:weight result-h) (:weight result-c) 1e-5)
    (assert-close "map regenerate(none) score unchanged"
                  (:score trace-c) (:score (:trace result-c)) 1e-6))

  ;; 1b. Select all — weight=0 for single-site kernels, score is finite
  (let [result-c (p/regenerate map-c trace-c sel/all)]
    (mx/eval! (:weight result-c) (:score (:trace result-c)))
    (assert-close "map regenerate(all) weight=0" 0.0 (:weight result-c) 1e-5)
    (assert-true "map regenerate(all) score finite"
                 (js/isFinite (mx/item (:score (:trace result-c))))))

  ;; 1c. Select partial (element 1 only) — weight=0
  (let [sel-1 (sel/hierarchical 1 sel/all)
        result-c (p/regenerate map-c trace-c sel-1)]
    (mx/eval! (:weight result-c))
    (assert-close "map regenerate(partial) weight=0" 0.0 (:weight result-c) 1e-5))

  ;; 1d. Metadata
  (let [result (p/regenerate map-c trace-c sel/all)
        meta-t (meta (:trace result))]
    (assert-true "map regenerate meta: ::compiled-path"
                 (::comb/compiled-path meta-t))
    (assert-true "map regenerate meta: ::element-scores present"
                 (some? (::comb/element-scores meta-t)))))

;; ===========================================================================
;; Section 2: UNFOLD COMBINATOR REGENERATE
;; ===========================================================================

(println "\n-- 2. Unfold: compiled regenerate --")

(let [unfold-c (comb/unfold-combinator k-unfold)
      unfold-h (comb/unfold-combinator (force-handler k-unfold))
      args [3 (mx/scalar 1.0)]
      init-constraints (-> cm/EMPTY
                           (cm/set-choice [0 :x] (mx/scalar 1.1))
                           (cm/set-choice [1 :x] (mx/scalar 1.2))
                           (cm/set-choice [2 :x] (mx/scalar 1.3)))
      trace-c (:trace (p/generate unfold-c args init-constraints))
      trace-h (:trace (p/generate unfold-h args init-constraints))]

  ;; 2a. Select none — weight must be 0 (the bug test!)
  (let [result-c (p/regenerate unfold-c trace-c sel/none)
        result-h (p/regenerate unfold-h trace-h sel/none)]
    (mx/eval! (:weight result-c) (:weight result-h))
    (assert-close "unfold regenerate(none) weight=0" 0.0 (:weight result-c) 1e-6)
    (assert-close "unfold regenerate(none) weight matches handler"
                  (:weight result-h) (:weight result-c) 1e-5)
    (assert-close "unfold regenerate(none) score unchanged"
                  (:score trace-c) (:score (:trace result-c)) 1e-6))

  ;; 2b. Select all — weight finite (not 0: state threading changes dist-args)
  (let [result-c (p/regenerate unfold-c trace-c sel/all)]
    (mx/eval! (:weight result-c) (:score (:trace result-c)))
    (assert-true "unfold regenerate(all) weight finite"
                 (js/isFinite (mx/item (:weight result-c))))
    (assert-true "unfold regenerate(all) score finite"
                 (js/isFinite (mx/item (:score (:trace result-c))))))

  ;; 2c. Select partial (step 1 only) — weight finite
  (let [sel-1 (sel/hierarchical 1 sel/all)
        result-c (p/regenerate unfold-c trace-c sel-1)]
    (mx/eval! (:weight result-c))
    (assert-true "unfold regenerate(partial) weight finite"
                 (js/isFinite (mx/item (:weight result-c)))))

  ;; 2d. State threading — regenerated values become state for subsequent steps
  (let [result (p/regenerate unfold-c trace-c sel/all)
        states (:retval (:trace result))]
    (assert-equal "unfold regenerate states count" 3 (count states)))

  ;; 2e. Metadata
  (let [result (p/regenerate unfold-c trace-c sel/all)
        meta-t (meta (:trace result))]
    (assert-true "unfold regenerate meta: ::compiled-path"
                 (::comb/compiled-path meta-t))
    (assert-true "unfold regenerate meta: ::step-scores present"
                 (some? (::comb/step-scores meta-t)))
    (assert-equal "unfold regenerate meta: step-scores count" 3
                  (count (::comb/step-scores meta-t)))))

;; ===========================================================================
;; Section 3: SCAN COMBINATOR REGENERATE
;; ===========================================================================

(println "\n-- 3. Scan: compiled regenerate --")

(let [scan-c (comb/scan-combinator k-scan)
      scan-h (comb/scan-combinator (force-handler k-scan))
      inputs [(mx/scalar 0.0) [(mx/scalar 0.1) (mx/scalar 0.2) (mx/scalar 0.3)]]
      init-constraints (-> cm/EMPTY
                           (cm/set-choice [0 :x] (mx/scalar 0.15))
                           (cm/set-choice [1 :x] (mx/scalar 0.4))
                           (cm/set-choice [2 :x] (mx/scalar 0.8)))
      trace-c (:trace (p/generate scan-c inputs init-constraints))
      trace-h (:trace (p/generate scan-h inputs init-constraints))]

  ;; 3a. Select none — weight must be 0 (bug test!)
  (let [result-c (p/regenerate scan-c trace-c sel/none)
        result-h (p/regenerate scan-h trace-h sel/none)]
    (mx/eval! (:weight result-c) (:weight result-h))
    (assert-close "scan regenerate(none) weight=0" 0.0 (:weight result-c) 1e-6)
    (assert-close "scan regenerate(none) weight matches handler"
                  (:weight result-h) (:weight result-c) 1e-5)
    (assert-close "scan regenerate(none) score unchanged"
                  (:score trace-c) (:score (:trace result-c)) 1e-6))

  ;; 3b. Select all — weight finite (not 0: carry threading changes dist-args)
  (let [result-c (p/regenerate scan-c trace-c sel/all)]
    (mx/eval! (:weight result-c) (:score (:trace result-c)))
    (assert-true "scan regenerate(all) weight finite"
                 (js/isFinite (mx/item (:weight result-c))))
    (assert-true "scan regenerate(all) score finite"
                 (js/isFinite (mx/item (:score (:trace result-c))))))

  ;; 3c. Select partial (step 1 only) — weight finite
  (let [sel-1 (sel/hierarchical 1 sel/all)
        result-c (p/regenerate scan-c trace-c sel-1)]
    (mx/eval! (:weight result-c))
    (assert-true "scan regenerate(partial) weight finite"
                 (js/isFinite (mx/item (:weight result-c)))))

  ;; 3d. Carry threading
  (let [result (p/regenerate scan-c trace-c sel/all)
        retval (:retval (:trace result))]
    (assert-true "scan regenerate carry present" (some? (:carry retval)))
    (assert-equal "scan regenerate outputs count" 3 (count (:outputs retval))))

  ;; 3e. Metadata
  (let [result (p/regenerate scan-c trace-c sel/all)
        meta-t (meta (:trace result))]
    (assert-true "scan regenerate meta: ::compiled-path"
                 (::comb/compiled-path meta-t))
    (assert-true "scan regenerate meta: ::step-scores present"
                 (some? (::comb/step-scores meta-t)))
    (assert-equal "scan regenerate meta: step-scores count" 3
                  (count (::comb/step-scores meta-t)))))

;; ===========================================================================
;; Section 4: SWITCH COMBINATOR REGENERATE
;; ===========================================================================

(println "\n-- 4. Switch: compiled regenerate --")

(let [switch-c (comb/switch-combinator k-switch k-switch-b)
      switch-h (comb/switch-combinator (force-handler k-switch)
                                        (force-handler k-switch-b))
      ;; Create trace on branch 0
      init-obs-0 (-> cm/EMPTY (cm/set-value :x (mx/scalar 2.0)))
      args-0 [0]
      trace-c (:trace (p/generate switch-c args-0 init-obs-0))
      trace-h (:trace (p/generate switch-h args-0 init-obs-0))]

  ;; 4a. Select none — weight=0
  (let [result-c (p/regenerate switch-c trace-c sel/none)
        result-h (p/regenerate switch-h trace-h sel/none)]
    (mx/eval! (:weight result-c) (:weight result-h))
    (assert-close "switch regenerate(none) weight=0" 0.0 (:weight result-c) 1e-6)
    (assert-close "switch regenerate(none) weight matches handler"
                  (:weight result-h) (:weight result-c) 1e-5))

  ;; 4b. Select all — weight=0, score finite
  (let [result-c (p/regenerate switch-c trace-c sel/all)]
    (mx/eval! (:weight result-c) (:score (:trace result-c)))
    (assert-close "switch regenerate(all) weight=0" 0.0 (:weight result-c) 1e-5)
    (assert-true "switch regenerate(all) score finite"
                 (js/isFinite (mx/item (:score (:trace result-c))))))

  ;; 4c. Metadata
  (let [result (p/regenerate switch-c trace-c sel/all)
        meta-t (meta (:trace result))]
    (assert-true "switch regenerate meta: ::compiled-path"
                 (::comb/compiled-path meta-t))
    (assert-equal "switch regenerate meta: ::switch-idx" 0
                  (::comb/switch-idx meta-t))))

;; ===========================================================================
;; Section 5: MIX COMBINATOR REGENERATE
;; ===========================================================================

(println "\n-- 5. Mix: compiled regenerate --")

(let [mix-c (comb/mix-combinator
              [k-switch k-switch-b]
              (fn [_args] (mx/array [0.0 0.0])))
      mix-h (comb/mix-combinator
              [(force-handler k-switch) (force-handler k-switch-b)]
              (fn [_args] (mx/array [0.0 0.0])))
      args []
      ;; Create trace on component 0
      init-obs (-> cm/EMPTY
                   (cm/set-choice [:component-idx] (mx/scalar 0 mx/int32))
                   (cm/set-value :x (mx/scalar 2.0)))
      trace-c (:trace (p/generate mix-c args init-obs))
      trace-h (:trace (p/generate mix-h args init-obs))]

  ;; 5a. Same component, select none — weight=0
  (let [result-c (p/regenerate mix-c trace-c sel/none)
        result-h (p/regenerate mix-h trace-h sel/none)]
    (mx/eval! (:weight result-c) (:weight result-h))
    (assert-close "mix regenerate(none) weight=0" 0.0 (:weight result-c) 1e-6)
    (assert-close "mix regenerate(none) weight matches handler"
                  (:weight result-h) (:weight result-c) 1e-5))

  ;; 5b. Same component, select inner (not idx) — weight=0, score finite
  (let [inner-sel (sel/select #{:x})
        result-c (p/regenerate mix-c trace-c inner-sel)]
    (mx/eval! (:weight result-c) (:score (:trace result-c)))
    (assert-close "mix regenerate(inner) weight=0" 0.0 (:weight result-c) 1e-5)
    (assert-true "mix regenerate(inner) score finite"
                 (js/isFinite (mx/item (:score (:trace result-c))))))

  ;; 5c. Metadata
  (let [inner-sel (sel/select #{:x})
        result (p/regenerate mix-c trace-c inner-sel)]
    (assert-true "mix regenerate meta: ::compiled-path"
                 (::comb/compiled-path (meta (:trace result))))))

;; ===========================================================================
;; Section 6: Cross-operation consistency + fallback
;; ===========================================================================

(println "\n-- 6. Cross-op consistency + fallback --")

;; 6a. Unfold regenerate(none) weight=0 with different key
(let [unfold-c (comb/unfold-combinator k-unfold)
      args [3 (mx/scalar 1.0)]
      init (-> cm/EMPTY
               (cm/set-choice [0 :x] (mx/scalar 1.1))
               (cm/set-choice [1 :x] (mx/scalar 1.2))
               (cm/set-choice [2 :x] (mx/scalar 1.3)))
      trace (:trace (p/generate unfold-c args init))
      result (p/regenerate unfold-c trace sel/none)]
  (mx/eval! (:weight result))
  (assert-close "cross-op: unfold regenerate(none) weight=0"
                0.0 (:weight result) 1e-6))

;; 6b. Non-compilable fallback: map with beta kernel
(let [map-beta (comb/map-combinator k-beta)
      inputs [[(mx/scalar 1.0) (mx/scalar 2.0)]]
      init (-> cm/EMPTY
               (cm/set-choice [0 :y] (mx/scalar 0.3))
               (cm/set-choice [1 :y] (mx/scalar 0.5)))
      trace (:trace (p/generate map-beta inputs init))
      result (p/regenerate map-beta trace sel/none)]
  (mx/eval! (:weight result))
  (assert-close "fallback: map beta regenerate(none) weight=0"
                0.0 (:weight result) 1e-6)
  (assert-true "fallback: map beta no compiled-path"
               (not (::comb/compiled-path (meta (:trace result))))))

;; 6c. Non-compilable fallback: unfold with beta kernel
(let [unfold-beta (comb/unfold-combinator k-beta)
      args [2 (mx/scalar 0.5)]
      init (-> cm/EMPTY
               (cm/set-choice [0 :y] (mx/scalar 0.3))
               (cm/set-choice [1 :y] (mx/scalar 0.5)))
      trace (:trace (p/generate unfold-beta args init))
      result (p/regenerate unfold-beta trace sel/none)]
  (mx/eval! (:weight result))
  (assert-close "fallback: unfold beta regenerate(none) weight=0"
                0.0 (:weight result) 1e-6))

;; 6d. Non-compilable fallback: scan with beta kernel
(let [scan-beta (comb/scan-combinator k-beta-scan)
      inputs [(mx/scalar 0.5) [(mx/scalar 0.1) (mx/scalar 0.2)]]
      init (-> cm/EMPTY
               (cm/set-choice [0 :y] (mx/scalar 0.3))
               (cm/set-choice [1 :y] (mx/scalar 0.5)))
      trace (:trace (p/generate scan-beta inputs init))
      result (p/regenerate scan-beta trace sel/none)]
  (mx/eval! (:weight result))
  (assert-close "fallback: scan beta regenerate(none) weight=0"
                0.0 (:weight result) 1e-6))

;; 6e. Non-compilable fallback: switch with beta kernel
(let [switch-beta (comb/switch-combinator k-beta-switch k-switch)
      init (-> cm/EMPTY (cm/set-value :y (mx/scalar 0.4)))
      trace (:trace (p/generate switch-beta [0] init))
      result (p/regenerate switch-beta trace sel/none)]
  (mx/eval! (:weight result))
  (assert-close "fallback: switch beta regenerate(none) weight=0"
                0.0 (:weight result) 1e-6))

;; ===========================================================================
;; Section 7: Multi-site kernel (non-trivial proposal ratio)
;; ===========================================================================

(println "\n-- 7. Multi-site kernel --")

;; 2-site dependent kernel: y depends on x → selecting x changes y's lp
(def k-dep
  (dyn/auto-key (gen [mu]
    (let [x (trace :x (dist/gaussian mu 1.0))
          y (trace :y (dist/gaussian x 0.5))]
      y))))

;; Map with multi-site kernel
(let [map-c (comb/map-combinator k-dep)
      map-h (comb/map-combinator (force-handler k-dep))
      inputs [[(mx/scalar 1.0) (mx/scalar 2.0)]]
      init (-> cm/EMPTY
               (cm/set-choice [0 :x] (mx/scalar 1.2))
               (cm/set-choice [0 :y] (mx/scalar 1.3))
               (cm/set-choice [1 :x] (mx/scalar 2.1))
               (cm/set-choice [1 :y] (mx/scalar 2.2)))
      trace-c (:trace (p/generate map-c inputs init))
      trace-h (:trace (p/generate map-h inputs init))
      ;; Select only :x in element 0 → y's lp changes → weight != 0
      sel-x0 (sel/hierarchical 0 (sel/select #{:x}))]

  (let [result-c (p/regenerate map-c trace-c sel/none)
        result-h (p/regenerate map-h trace-h sel/none)]
    (mx/eval! (:weight result-c) (:weight result-h))
    (assert-close "multi-site map regenerate(none) weight=0"
                  0.0 (:weight result-c) 1e-6)
    (assert-close "multi-site map regenerate(none) matches handler"
                  (:weight result-h) (:weight result-c) 1e-5))

  (let [result-c (p/regenerate map-c trace-c sel-x0)]
    (mx/eval! (:weight result-c))
    (assert-true "multi-site map select(:x) weight finite"
                 (js/isFinite (mx/item (:weight result-c))))))

;; Mix with multi-site kernel (tests the fallback weight fix)
(let [mix-c (comb/mix-combinator
              [k-dep k-switch-b]
              (fn [_args] (mx/array [0.0 0.0])))
      mix-h (comb/mix-combinator
              [(force-handler k-dep) (force-handler k-switch-b)]
              (fn [_args] (mx/array [0.0 0.0])))
      args [(mx/scalar 1.0)]
      init-obs (-> cm/EMPTY
                   (cm/set-choice [:component-idx] (mx/scalar 0 mx/int32))
                   (cm/set-value :x (mx/scalar 1.2))
                   (cm/set-value :y (mx/scalar 1.3)))
      trace-c (:trace (p/generate mix-c args init-obs))
      trace-h (:trace (p/generate mix-h args init-obs))]

  (let [result-c (p/regenerate mix-c trace-c sel/none)
        result-h (p/regenerate mix-h trace-h sel/none)]
    (mx/eval! (:weight result-c) (:weight result-h))
    (assert-close "multi-site mix regenerate(none) weight=0"
                  0.0 (:weight result-c) 1e-6)
    (assert-close "multi-site mix regenerate(none) matches handler"
                  (:weight result-h) (:weight result-c) 1e-5)))

;; ===========================================================================
;; Section 8: Round-trip regenerate (MCMC chain scenario)
;; ===========================================================================

(println "\n-- 8. Round-trip regenerate --")

;; Regenerate a trace that was itself produced by compiled regenerate
(let [unfold-c (comb/unfold-combinator k-unfold)
      args [3 (mx/scalar 1.0)]
      init (-> cm/EMPTY
               (cm/set-choice [0 :x] (mx/scalar 1.1))
               (cm/set-choice [1 :x] (mx/scalar 1.2))
               (cm/set-choice [2 :x] (mx/scalar 1.3)))
      trace-0 (:trace (p/generate unfold-c args init))
      ;; First regenerate
      result-1 (p/regenerate unfold-c trace-0 sel/all)
      trace-1 (:trace result-1)
      ;; Second regenerate on the result of the first
      result-2 (p/regenerate unfold-c trace-1 sel/all)]
  (mx/eval! (:weight result-1) (:weight result-2)
            (:score trace-1) (:score (:trace result-2)))
  (assert-true "round-trip: first regenerate weight finite"
               (js/isFinite (mx/item (:weight result-1))))
  (assert-true "round-trip: second regenerate weight finite"
               (js/isFinite (mx/item (:weight result-2))))
  (assert-true "round-trip: second trace score finite"
               (js/isFinite (mx/item (:score (:trace result-2)))))
  (assert-true "round-trip: step-scores metadata preserved"
               (= 3 (count (::comb/step-scores (meta (:trace result-2)))))))

;; Same for Map
(let [map-c (comb/map-combinator k-map)
      inputs [[(mx/scalar 1.0) (mx/scalar 2.0)]]
      init (-> cm/EMPTY
               (cm/set-choice [0 :y] (mx/scalar 1.5))
               (cm/set-choice [1 :y] (mx/scalar 2.5)))
      trace-0 (:trace (p/generate map-c inputs init))
      result-1 (p/regenerate map-c trace-0 sel/all)
      result-2 (p/regenerate map-c (:trace result-1) sel/all)]
  (mx/eval! (:weight result-2) (:score (:trace result-2)))
  (assert-true "round-trip map: second regenerate score finite"
               (js/isFinite (mx/item (:score (:trace result-2)))))
  (assert-true "round-trip map: element-scores metadata preserved"
               (= 2 (count (::comb/element-scores (meta (:trace result-2)))))))

;; ===========================================================================
;; Section 9: Switch with idx=1 (second branch)
;; ===========================================================================

(println "\n-- 9. Switch idx=1 --")

(let [switch-c (comb/switch-combinator k-switch k-switch-b)
      switch-h (comb/switch-combinator (force-handler k-switch)
                                        (force-handler k-switch-b))
      init-obs-1 (-> cm/EMPTY (cm/set-value :x (mx/scalar 5.5)))
      trace-c (:trace (p/generate switch-c [1] init-obs-1))
      trace-h (:trace (p/generate switch-h [1] init-obs-1))]

  (let [result-c (p/regenerate switch-c trace-c sel/none)
        result-h (p/regenerate switch-h trace-h sel/none)]
    (mx/eval! (:weight result-c) (:weight result-h))
    (assert-close "switch idx=1 regenerate(none) weight=0"
                  0.0 (:weight result-c) 1e-6)
    (assert-close "switch idx=1 regenerate(none) matches handler"
                  (:weight result-h) (:weight result-c) 1e-5))

  (let [result-c (p/regenerate switch-c trace-c sel/all)]
    (mx/eval! (:weight result-c))
    (assert-close "switch idx=1 regenerate(all) weight=0"
                  0.0 (:weight result-c) 1e-5)
    (assert-equal "switch idx=1 meta: ::switch-idx"
                  1 (::comb/switch-idx (meta (:trace result-c))))))

;; ===========================================================================
;; Summary
;; ===========================================================================

(println (str "\n=== WP-9A Results: " @pass-count "/" (+ @pass-count @fail-count)
              " passed ==="))
(when (pos? @fail-count)
  (println (str "  " @fail-count " FAILURES")))
