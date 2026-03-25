(ns genmlx.wp9a-combinator-regenerate-test
  "WP-9A tests: combinator compiled regenerate.
   Validates that compiled regenerate paths for Map, Unfold, Scan, Switch, Mix
   produce correct weights, scores, and trace structure."
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
            [genmlx.selection :as sel]
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

;; ---------------------------------------------------------------------------
;; Tests
;; ---------------------------------------------------------------------------

(deftest prerequisites-test
  (testing "Prerequisites"
    (is (some? (compiled/get-compiled-regenerate k-map)) "map kernel has compiled-regenerate")
    (is (some? (compiled/get-compiled-regenerate k-unfold)) "unfold kernel has compiled-regenerate")
    (is (some? (compiled/get-compiled-regenerate k-scan)) "scan kernel has compiled-regenerate")
    (is (some? (compiled/get-compiled-regenerate k-switch)) "switch kernel has compiled-regenerate")
    (is (not (compiled/get-compiled-regenerate k-beta)) "beta kernel NOT compilable-regenerate")))

(deftest map-regenerate-test
  (let [map-c (comb/map-combinator k-map)
        map-h (comb/map-combinator (force-handler k-map))
        inputs [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]]
        init-constraints (-> cm/EMPTY
                             (cm/set-choice [0 :y] (mx/scalar 1.5))
                             (cm/set-choice [1 :y] (mx/scalar 2.5))
                             (cm/set-choice [2 :y] (mx/scalar 3.5)))
        trace-c (:trace (p/generate map-c inputs init-constraints))
        trace-h (:trace (p/generate map-h inputs init-constraints))]

    (testing "Map: select none -- weight=0"
      (let [result-c (p/regenerate map-c trace-c sel/none)
            result-h (p/regenerate map-h trace-h sel/none)]
        (mx/eval! (:weight result-c) (:weight result-h))
        (is (h/close? 0.0 (h/realize (:weight result-c)) 1e-6) "map regenerate(none) weight=0")
        (is (h/close? (h/realize (:weight result-h)) (h/realize (:weight result-c)) 1e-5)
            "map regenerate(none) weight matches handler")
        (is (h/close? (h/realize (:score trace-c)) (h/realize (:score (:trace result-c))) 1e-6)
            "map regenerate(none) score unchanged")))

    (testing "Map: select all -- weight=0 for single-site kernels"
      (let [result-c (p/regenerate map-c trace-c sel/all)]
        (mx/eval! (:weight result-c) (:score (:trace result-c)))
        (is (h/close? 0.0 (h/realize (:weight result-c)) 1e-5) "map regenerate(all) weight=0")
        (is (js/isFinite (mx/item (:score (:trace result-c)))) "map regenerate(all) score finite")))

    (testing "Map: select partial (element 1 only)"
      (let [sel-1 (sel/hierarchical 1 sel/all)
            result-c (p/regenerate map-c trace-c sel-1)]
        (mx/eval! (:weight result-c))
        (is (h/close? 0.0 (h/realize (:weight result-c)) 1e-5) "map regenerate(partial) weight=0")))

    (testing "Map: metadata"
      (let [result (p/regenerate map-c trace-c sel/all)
            meta-t (meta (:trace result))]
        (is (::comb/compiled-path meta-t) "map regenerate meta: ::compiled-path")
        (is (some? (::comb/element-scores meta-t)) "map regenerate meta: ::element-scores present")))))

(deftest unfold-regenerate-test
  (let [unfold-c (comb/unfold-combinator k-unfold)
        unfold-h (comb/unfold-combinator (force-handler k-unfold))
        args [3 (mx/scalar 1.0)]
        init-constraints (-> cm/EMPTY
                             (cm/set-choice [0 :x] (mx/scalar 1.1))
                             (cm/set-choice [1 :x] (mx/scalar 1.2))
                             (cm/set-choice [2 :x] (mx/scalar 1.3)))
        trace-c (:trace (p/generate unfold-c args init-constraints))
        trace-h (:trace (p/generate unfold-h args init-constraints))]

    (testing "Unfold: select none -- weight=0"
      (let [result-c (p/regenerate unfold-c trace-c sel/none)
            result-h (p/regenerate unfold-h trace-h sel/none)]
        (mx/eval! (:weight result-c) (:weight result-h))
        (is (h/close? 0.0 (h/realize (:weight result-c)) 1e-6) "unfold regenerate(none) weight=0")
        (is (h/close? (h/realize (:weight result-h)) (h/realize (:weight result-c)) 1e-5)
            "unfold regenerate(none) weight matches handler")
        (is (h/close? (h/realize (:score trace-c)) (h/realize (:score (:trace result-c))) 1e-6)
            "unfold regenerate(none) score unchanged")))

    (testing "Unfold: select all -- weight finite"
      (let [result-c (p/regenerate unfold-c trace-c sel/all)]
        (mx/eval! (:weight result-c) (:score (:trace result-c)))
        (is (js/isFinite (mx/item (:weight result-c))) "unfold regenerate(all) weight finite")
        (is (js/isFinite (mx/item (:score (:trace result-c)))) "unfold regenerate(all) score finite")))

    (testing "Unfold: select partial (step 1 only)"
      (let [sel-1 (sel/hierarchical 1 sel/all)
            result-c (p/regenerate unfold-c trace-c sel-1)]
        (mx/eval! (:weight result-c))
        (is (js/isFinite (mx/item (:weight result-c))) "unfold regenerate(partial) weight finite")))

    (testing "Unfold: state threading"
      (let [result (p/regenerate unfold-c trace-c sel/all)
            states (:retval (:trace result))]
        (is (= 3 (count states)) "unfold regenerate states count")))

    (testing "Unfold: metadata"
      (let [result (p/regenerate unfold-c trace-c sel/all)
            meta-t (meta (:trace result))]
        (is (::comb/compiled-path meta-t) "unfold regenerate meta: ::compiled-path")
        (is (some? (::comb/step-scores meta-t)) "unfold regenerate meta: ::step-scores present")
        (is (= 3 (count (::comb/step-scores meta-t))) "unfold regenerate meta: step-scores count")))))

(deftest scan-regenerate-test
  (let [scan-c (comb/scan-combinator k-scan)
        scan-h (comb/scan-combinator (force-handler k-scan))
        inputs [(mx/scalar 0.0) [(mx/scalar 0.1) (mx/scalar 0.2) (mx/scalar 0.3)]]
        init-constraints (-> cm/EMPTY
                             (cm/set-choice [0 :x] (mx/scalar 0.15))
                             (cm/set-choice [1 :x] (mx/scalar 0.4))
                             (cm/set-choice [2 :x] (mx/scalar 0.8)))
        trace-c (:trace (p/generate scan-c inputs init-constraints))
        trace-h (:trace (p/generate scan-h inputs init-constraints))]

    (testing "Scan: select none -- weight=0"
      (let [result-c (p/regenerate scan-c trace-c sel/none)
            result-h (p/regenerate scan-h trace-h sel/none)]
        (mx/eval! (:weight result-c) (:weight result-h))
        (is (h/close? 0.0 (h/realize (:weight result-c)) 1e-6) "scan regenerate(none) weight=0")
        (is (h/close? (h/realize (:weight result-h)) (h/realize (:weight result-c)) 1e-5)
            "scan regenerate(none) weight matches handler")
        (is (h/close? (h/realize (:score trace-c)) (h/realize (:score (:trace result-c))) 1e-6)
            "scan regenerate(none) score unchanged")))

    (testing "Scan: select all -- weight finite"
      (let [result-c (p/regenerate scan-c trace-c sel/all)]
        (mx/eval! (:weight result-c) (:score (:trace result-c)))
        (is (js/isFinite (mx/item (:weight result-c))) "scan regenerate(all) weight finite")
        (is (js/isFinite (mx/item (:score (:trace result-c)))) "scan regenerate(all) score finite")))

    (testing "Scan: select partial (step 1 only)"
      (let [sel-1 (sel/hierarchical 1 sel/all)
            result-c (p/regenerate scan-c trace-c sel-1)]
        (mx/eval! (:weight result-c))
        (is (js/isFinite (mx/item (:weight result-c))) "scan regenerate(partial) weight finite")))

    (testing "Scan: carry threading"
      (let [result (p/regenerate scan-c trace-c sel/all)
            retval (:retval (:trace result))]
        (is (some? (:carry retval)) "scan regenerate carry present")
        (is (= 3 (count (:outputs retval))) "scan regenerate outputs count")))

    (testing "Scan: metadata"
      (let [result (p/regenerate scan-c trace-c sel/all)
            meta-t (meta (:trace result))]
        (is (::comb/compiled-path meta-t) "scan regenerate meta: ::compiled-path")
        (is (some? (::comb/step-scores meta-t)) "scan regenerate meta: ::step-scores present")
        (is (= 3 (count (::comb/step-scores meta-t))) "scan regenerate meta: step-scores count")))))

(deftest switch-regenerate-test
  (let [switch-c (comb/switch-combinator k-switch k-switch-b)
        switch-h (comb/switch-combinator (force-handler k-switch)
                                          (force-handler k-switch-b))
        init-obs-0 (-> cm/EMPTY (cm/set-value :x (mx/scalar 2.0)))
        args-0 [0]
        trace-c (:trace (p/generate switch-c args-0 init-obs-0))
        trace-h (:trace (p/generate switch-h args-0 init-obs-0))]

    (testing "Switch: select none -- weight=0"
      (let [result-c (p/regenerate switch-c trace-c sel/none)
            result-h (p/regenerate switch-h trace-h sel/none)]
        (mx/eval! (:weight result-c) (:weight result-h))
        (is (h/close? 0.0 (h/realize (:weight result-c)) 1e-6) "switch regenerate(none) weight=0")
        (is (h/close? (h/realize (:weight result-h)) (h/realize (:weight result-c)) 1e-5)
            "switch regenerate(none) weight matches handler")))

    (testing "Switch: select all -- weight=0, score finite"
      (let [result-c (p/regenerate switch-c trace-c sel/all)]
        (mx/eval! (:weight result-c) (:score (:trace result-c)))
        (is (h/close? 0.0 (h/realize (:weight result-c)) 1e-5) "switch regenerate(all) weight=0")
        (is (js/isFinite (mx/item (:score (:trace result-c)))) "switch regenerate(all) score finite")))

    (testing "Switch: metadata"
      (let [result (p/regenerate switch-c trace-c sel/all)
            meta-t (meta (:trace result))]
        (is (::comb/compiled-path meta-t) "switch regenerate meta: ::compiled-path")
        (is (= 0 (::comb/switch-idx meta-t)) "switch regenerate meta: ::switch-idx")))))

(deftest mix-regenerate-test
  (let [mix-c (comb/mix-combinator
                [k-switch k-switch-b]
                (fn [_args] (mx/array [0.0 0.0])))
        mix-h (comb/mix-combinator
                [(force-handler k-switch) (force-handler k-switch-b)]
                (fn [_args] (mx/array [0.0 0.0])))
        args []
        init-obs (-> cm/EMPTY
                     (cm/set-choice [:component-idx] (mx/scalar 0 mx/int32))
                     (cm/set-value :x (mx/scalar 2.0)))
        trace-c (:trace (p/generate mix-c args init-obs))
        trace-h (:trace (p/generate mix-h args init-obs))]

    (testing "Mix: same component, select none -- weight=0"
      (let [result-c (p/regenerate mix-c trace-c sel/none)
            result-h (p/regenerate mix-h trace-h sel/none)]
        (mx/eval! (:weight result-c) (:weight result-h))
        (is (h/close? 0.0 (h/realize (:weight result-c)) 1e-6) "mix regenerate(none) weight=0")
        (is (h/close? (h/realize (:weight result-h)) (h/realize (:weight result-c)) 1e-5)
            "mix regenerate(none) weight matches handler")))

    (testing "Mix: same component, select inner -- weight=0, score finite"
      (let [inner-sel (sel/select #{:x})
            result-c (p/regenerate mix-c trace-c inner-sel)]
        (mx/eval! (:weight result-c) (:score (:trace result-c)))
        (is (h/close? 0.0 (h/realize (:weight result-c)) 1e-5) "mix regenerate(inner) weight=0")
        (is (js/isFinite (mx/item (:score (:trace result-c)))) "mix regenerate(inner) score finite")))

    (testing "Mix: metadata"
      (let [inner-sel (sel/select #{:x})
            result (p/regenerate mix-c trace-c inner-sel)]
        (is (::comb/compiled-path (meta (:trace result))) "mix regenerate meta: ::compiled-path")))))

(deftest cross-op-consistency-test
  (testing "Cross-op: unfold regenerate(none) weight=0"
    (let [unfold-c (comb/unfold-combinator k-unfold)
          args [3 (mx/scalar 1.0)]
          init (-> cm/EMPTY
                   (cm/set-choice [0 :x] (mx/scalar 1.1))
                   (cm/set-choice [1 :x] (mx/scalar 1.2))
                   (cm/set-choice [2 :x] (mx/scalar 1.3)))
          trace (:trace (p/generate unfold-c args init))
          result (p/regenerate unfold-c trace sel/none)]
      (mx/eval! (:weight result))
      (is (h/close? 0.0 (h/realize (:weight result)) 1e-6)
          "cross-op: unfold regenerate(none) weight=0")))

  (testing "Fallback: map beta regenerate(none)"
    (let [map-beta (comb/map-combinator k-beta)
          inputs [[(mx/scalar 1.0) (mx/scalar 2.0)]]
          init (-> cm/EMPTY
                   (cm/set-choice [0 :y] (mx/scalar 0.3))
                   (cm/set-choice [1 :y] (mx/scalar 0.5)))
          trace (:trace (p/generate map-beta inputs init))
          result (p/regenerate map-beta trace sel/none)]
      (mx/eval! (:weight result))
      (is (h/close? 0.0 (h/realize (:weight result)) 1e-6) "fallback: map beta regenerate(none) weight=0")
      (is (not (::comb/compiled-path (meta (:trace result)))) "fallback: map beta no compiled-path")))

  (testing "Fallback: unfold beta regenerate(none)"
    (let [unfold-beta (comb/unfold-combinator k-beta)
          args [2 (mx/scalar 0.5)]
          init (-> cm/EMPTY
                   (cm/set-choice [0 :y] (mx/scalar 0.3))
                   (cm/set-choice [1 :y] (mx/scalar 0.5)))
          trace (:trace (p/generate unfold-beta args init))
          result (p/regenerate unfold-beta trace sel/none)]
      (mx/eval! (:weight result))
      (is (h/close? 0.0 (h/realize (:weight result)) 1e-6) "fallback: unfold beta regenerate(none) weight=0")))

  (testing "Fallback: scan beta regenerate(none)"
    (let [scan-beta (comb/scan-combinator k-beta-scan)
          inputs [(mx/scalar 0.5) [(mx/scalar 0.1) (mx/scalar 0.2)]]
          init (-> cm/EMPTY
                   (cm/set-choice [0 :y] (mx/scalar 0.3))
                   (cm/set-choice [1 :y] (mx/scalar 0.5)))
          trace (:trace (p/generate scan-beta inputs init))
          result (p/regenerate scan-beta trace sel/none)]
      (mx/eval! (:weight result))
      (is (h/close? 0.0 (h/realize (:weight result)) 1e-6) "fallback: scan beta regenerate(none) weight=0")))

  (testing "Fallback: switch beta regenerate(none)"
    (let [switch-beta (comb/switch-combinator k-beta-switch k-switch)
          init (-> cm/EMPTY (cm/set-value :y (mx/scalar 0.4)))
          trace (:trace (p/generate switch-beta [0] init))
          result (p/regenerate switch-beta trace sel/none)]
      (mx/eval! (:weight result))
      (is (h/close? 0.0 (h/realize (:weight result)) 1e-6) "fallback: switch beta regenerate(none) weight=0"))))

(deftest multi-site-kernel-test
  (let [k-dep (dyn/auto-key (gen [mu]
                (let [x (trace :x (dist/gaussian mu 1.0))
                      y (trace :y (dist/gaussian x 0.5))]
                  y)))]

    (testing "Multi-site: Map regenerate(none)"
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
            sel-x0 (sel/hierarchical 0 (sel/select #{:x}))]
        (let [result-c (p/regenerate map-c trace-c sel/none)
              result-h (p/regenerate map-h trace-h sel/none)]
          (mx/eval! (:weight result-c) (:weight result-h))
          (is (h/close? 0.0 (h/realize (:weight result-c)) 1e-6)
              "multi-site map regenerate(none) weight=0")
          (is (h/close? (h/realize (:weight result-h)) (h/realize (:weight result-c)) 1e-5)
              "multi-site map regenerate(none) matches handler"))
        (let [result-c (p/regenerate map-c trace-c sel-x0)]
          (mx/eval! (:weight result-c))
          (is (js/isFinite (mx/item (:weight result-c))) "multi-site map select(:x) weight finite"))))

    (testing "Multi-site: Mix regenerate(none)"
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
          (is (h/close? 0.0 (h/realize (:weight result-c)) 1e-6)
              "multi-site mix regenerate(none) weight=0")
          (is (h/close? (h/realize (:weight result-h)) (h/realize (:weight result-c)) 1e-5)
              "multi-site mix regenerate(none) matches handler"))))))

(deftest round-trip-regenerate-test
  (testing "Round-trip: Unfold"
    (let [unfold-c (comb/unfold-combinator k-unfold)
          args [3 (mx/scalar 1.0)]
          init (-> cm/EMPTY
                   (cm/set-choice [0 :x] (mx/scalar 1.1))
                   (cm/set-choice [1 :x] (mx/scalar 1.2))
                   (cm/set-choice [2 :x] (mx/scalar 1.3)))
          trace-0 (:trace (p/generate unfold-c args init))
          result-1 (p/regenerate unfold-c trace-0 sel/all)
          trace-1 (:trace result-1)
          result-2 (p/regenerate unfold-c trace-1 sel/all)]
      (mx/eval! (:weight result-1) (:weight result-2)
                (:score trace-1) (:score (:trace result-2)))
      (is (js/isFinite (mx/item (:weight result-1))) "round-trip: first regenerate weight finite")
      (is (js/isFinite (mx/item (:weight result-2))) "round-trip: second regenerate weight finite")
      (is (js/isFinite (mx/item (:score (:trace result-2)))) "round-trip: second trace score finite")
      (is (= 3 (count (::comb/step-scores (meta (:trace result-2)))))
          "round-trip: step-scores metadata preserved")))

  (testing "Round-trip: Map"
    (let [map-c (comb/map-combinator k-map)
          inputs [[(mx/scalar 1.0) (mx/scalar 2.0)]]
          init (-> cm/EMPTY
                   (cm/set-choice [0 :y] (mx/scalar 1.5))
                   (cm/set-choice [1 :y] (mx/scalar 2.5)))
          trace-0 (:trace (p/generate map-c inputs init))
          result-1 (p/regenerate map-c trace-0 sel/all)
          result-2 (p/regenerate map-c (:trace result-1) sel/all)]
      (mx/eval! (:weight result-2) (:score (:trace result-2)))
      (is (js/isFinite (mx/item (:score (:trace result-2))))
          "round-trip map: second regenerate score finite")
      (is (= 2 (count (::comb/element-scores (meta (:trace result-2)))))
          "round-trip map: element-scores metadata preserved"))))

(deftest switch-idx1-test
  (let [switch-c (comb/switch-combinator k-switch k-switch-b)
        switch-h (comb/switch-combinator (force-handler k-switch)
                                          (force-handler k-switch-b))
        init-obs-1 (-> cm/EMPTY (cm/set-value :x (mx/scalar 5.5)))
        trace-c (:trace (p/generate switch-c [1] init-obs-1))
        trace-h (:trace (p/generate switch-h [1] init-obs-1))]

    (testing "Switch idx=1: select none"
      (let [result-c (p/regenerate switch-c trace-c sel/none)
            result-h (p/regenerate switch-h trace-h sel/none)]
        (mx/eval! (:weight result-c) (:weight result-h))
        (is (h/close? 0.0 (h/realize (:weight result-c)) 1e-6)
            "switch idx=1 regenerate(none) weight=0")
        (is (h/close? (h/realize (:weight result-h)) (h/realize (:weight result-c)) 1e-5)
            "switch idx=1 regenerate(none) matches handler")))

    (testing "Switch idx=1: select all"
      (let [result-c (p/regenerate switch-c trace-c sel/all)]
        (mx/eval! (:weight result-c))
        (is (h/close? 0.0 (h/realize (:weight result-c)) 1e-5) "switch idx=1 regenerate(all) weight=0")
        (is (= 1 (::comb/switch-idx (meta (:trace result-c)))) "switch idx=1 meta: ::switch-idx")))))

(cljs.test/run-tests)
