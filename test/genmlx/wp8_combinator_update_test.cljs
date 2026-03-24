(ns genmlx.wp8-combinator-update-test
  "WP-8 tests: combinator compiled update.
   Validates that compiled update paths for Map, Unfold, Scan, Switch, Mix
   produce identical traces, scores, weights, and discards as handler paths."
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
  "Strip ALL compiled paths from a gen-fn so it falls back to handler."
  [gf]
  (let [schema (dissoc (:schema gf) :compiled-simulate :compiled-generate
                       :compiled-update
                       :compiled-prefix :compiled-prefix-addrs
                       :compiled-prefix-generate :compiled-prefix-update)]
    (assoc gf :schema schema)))

(defn compilable-update?
  "Check if a gen-fn kernel has a compiled-update function."
  [gf]
  (some? (compiled/get-compiled-update gf)))

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
    (is (compilable-update? k-map) "map kernel has compiled-update")
    (is (compilable-update? k-unfold) "unfold kernel has compiled-update")
    (is (compilable-update? k-scan) "scan kernel has compiled-update")
    (is (compilable-update? k-switch) "switch kernel has compiled-update")
    (is (not (compilable-update? k-beta)) "beta kernel NOT compilable-update")))

(deftest map-update-test
  (let [map-gf (comb/map-combinator k-map)
        map-h  (comb/map-combinator (force-handler k-map))
        inputs [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]]
        init-constraints (-> cm/EMPTY
                             (cm/set-choice [0 :y] (mx/scalar 1.5))
                             (cm/set-choice [1 :y] (mx/scalar 2.5))
                             (cm/set-choice [2 :y] (mx/scalar 3.5)))
        trace   (:trace (p/generate map-gf inputs init-constraints))
        trace-h (:trace (p/generate map-h  inputs init-constraints))]

    (testing "Map: no constraints -- weight=0, choices unchanged"
      (let [result   (p/update map-gf trace cm/EMPTY)
            result-h (p/update map-h  trace-h cm/EMPTY)]
        (mx/eval! (:weight result) (:weight result-h))
        (is (h/close? 0.0 (h/realize (:weight result)) 1e-6) "map no-constraint weight=0")
        (is (h/close? (h/realize (:score trace)) (h/realize (:score (:trace result))) 1e-6)
            "map no-constraint score unchanged")
        (is (h/close? (h/realize (:weight result-h)) (h/realize (:weight result)) 1e-5)
            "map no-constraint weight matches handler")))

    (testing "Map: single element constrained"
      (let [new-obs  (-> cm/EMPTY (cm/set-choice [1 :y] (mx/scalar 5.0)))
            result   (p/update map-gf trace new-obs)
            result-h (p/update map-h  trace-h new-obs)]
        (mx/eval! (:weight result) (:weight result-h)
                  (:score (:trace result)) (:score (:trace result-h)))
        (is (h/close? (h/realize (:weight result-h)) (h/realize (:weight result)) 1e-5)
            "map single-constraint weight matches handler")
        (is (h/close? (h/realize (:score (:trace result-h))) (h/realize (:score (:trace result))) 1e-5)
            "map single-constraint score matches handler")
        (let [v (cm/get-choice (:choices (:trace result)) [1 :y])]
          (mx/eval! v)
          (is (h/close? 5.0 (h/realize v) 1e-6) "map constrained value applied"))
        (let [v0 (cm/get-choice (:choices (:trace result)) [0 :y])]
          (mx/eval! v0)
          (is (h/close? 1.5 (h/realize v0) 1e-6) "map unconstrained[0] kept"))
        (let [d (cm/get-choice (:discard result) [1 :y])]
          (mx/eval! d)
          (is (h/close? 2.5 (h/realize d) 1e-6) "map discard[1] = old value"))))

    (testing "Map: all elements constrained"
      (let [new-obs  (-> cm/EMPTY
                         (cm/set-choice [0 :y] (mx/scalar 10.0))
                         (cm/set-choice [1 :y] (mx/scalar 20.0))
                         (cm/set-choice [2 :y] (mx/scalar 30.0)))
            result   (p/update map-gf trace new-obs)
            result-h (p/update map-h  trace-h new-obs)]
        (mx/eval! (:weight result) (:weight result-h)
                  (:score (:trace result)) (:score (:trace result-h)))
        (is (h/close? (h/realize (:weight result-h)) (h/realize (:weight result)) 1e-5)
            "map all-constrained weight matches handler")
        (is (h/close? (h/realize (:score (:trace result-h))) (h/realize (:score (:trace result))) 1e-5)
            "map all-constrained score matches handler")))

    (testing "Map: metadata"
      (let [new-obs (-> cm/EMPTY (cm/set-choice [0 :y] (mx/scalar 10.0)))
            result  (p/update map-gf trace new-obs)
            meta-t  (meta (:trace result))
            es      (::comb/element-scores meta-t)]
        (is (::comb/compiled-path meta-t) "map meta: ::compiled-path")
        (is (some? es) "map meta: ::element-scores present")
        (is (= 3 (count es)) "map meta: element-scores count")))

    (testing "Map: non-compilable fallback"
      (let [map-beta (comb/map-combinator k-beta)
            inputs-b [[(mx/scalar 1.0) (mx/scalar 2.0)]]
            init-b   (-> cm/EMPTY
                         (cm/set-choice [0 :y] (mx/scalar 0.3))
                         (cm/set-choice [1 :y] (mx/scalar 0.5)))
            trace-b  (:trace (p/generate map-beta inputs-b init-b))
            result-b (p/update map-beta trace-b cm/EMPTY)]
        (mx/eval! (:weight result-b))
        (is (h/close? 0.0 (h/realize (:weight result-b)) 1e-6) "map beta fallback weight=0")
        (is (not (::comb/compiled-path (meta (:trace result-b)))) "map beta fallback: no compiled-path")))))

(deftest unfold-update-test
  (let [unfold-gf (comb/unfold-combinator k-unfold)
        unfold-h  (comb/unfold-combinator (force-handler k-unfold))
        args      [3 (mx/scalar 1.0)]
        init-constraints (-> cm/EMPTY
                             (cm/set-choice [0 :x] (mx/scalar 1.1))
                             (cm/set-choice [1 :x] (mx/scalar 1.2))
                             (cm/set-choice [2 :x] (mx/scalar 1.3)))
        trace   (:trace (p/generate unfold-gf args init-constraints))
        trace-h (:trace (p/generate unfold-h  args init-constraints))]

    (testing "Unfold: no constraints -- weight=0"
      (let [result   (p/update unfold-gf trace cm/EMPTY)
            result-h (p/update unfold-h  trace-h cm/EMPTY)]
        (mx/eval! (:weight result) (:weight result-h))
        (is (h/close? 0.0 (h/realize (:weight result)) 1e-6) "unfold no-constraint weight=0")
        (is (h/close? (h/realize (:weight result-h)) (h/realize (:weight result)) 1e-5)
            "unfold no-constraint weight matches handler")))

    (testing "Unfold: single step constrained"
      (let [new-obs  (-> cm/EMPTY (cm/set-choice [1 :x] (mx/scalar 2.0)))
            result   (p/update unfold-gf trace new-obs)
            result-h (p/update unfold-h  trace-h new-obs)]
        (mx/eval! (:weight result) (:weight result-h)
                  (:score (:trace result)) (:score (:trace result-h)))
        (is (h/close? (h/realize (:weight result-h)) (h/realize (:weight result)) 1e-5)
            "unfold single-constraint weight matches handler")
        (is (h/close? (h/realize (:score (:trace result-h))) (h/realize (:score (:trace result))) 1e-5)
            "unfold single-constraint score matches handler")
        (let [v (cm/get-choice (:choices (:trace result)) [1 :x])]
          (mx/eval! v)
          (is (h/close? 2.0 (h/realize v) 1e-6) "unfold constrained step[1] applied"))
        (let [v0 (cm/get-choice (:choices (:trace result)) [0 :x])]
          (mx/eval! v0)
          (is (h/close? 1.1 (h/realize v0) 1e-6) "unfold step[0] kept (prefix-skip)"))
        (let [d (cm/get-choice (:discard result) [1 :x])]
          (mx/eval! d)
          (is (h/close? 1.2 (h/realize d) 1e-6) "unfold discard[1] = old value"))))

    (testing "Unfold: all steps constrained"
      (let [new-obs  (-> cm/EMPTY
                         (cm/set-choice [0 :x] (mx/scalar 5.0))
                         (cm/set-choice [1 :x] (mx/scalar 6.0))
                         (cm/set-choice [2 :x] (mx/scalar 7.0)))
            result   (p/update unfold-gf trace new-obs)
            result-h (p/update unfold-h  trace-h new-obs)]
        (mx/eval! (:weight result) (:weight result-h)
                  (:score (:trace result)) (:score (:trace result-h)))
        (is (h/close? (h/realize (:weight result-h)) (h/realize (:weight result)) 1e-5)
            "unfold all-constrained weight matches handler")
        (is (h/close? (h/realize (:score (:trace result-h))) (h/realize (:score (:trace result))) 1e-5)
            "unfold all-constrained score matches handler")))

    (testing "Unfold: state threading"
      (let [new-obs  (-> cm/EMPTY (cm/set-choice [0 :x] (mx/scalar 5.0)))
            result   (p/update unfold-gf trace new-obs)
            states   (:retval (:trace result))]
        (mx/eval! (first states))
        (is (h/close? 5.0 (h/realize (first states)) 1e-6) "unfold state[0] = constrained value")))

    (testing "Unfold: prefix-skip"
      (let [new-obs  (-> cm/EMPTY (cm/set-choice [2 :x] (mx/scalar 9.0)))
            result   (p/update unfold-gf trace new-obs)]
        (let [v0 (cm/get-choice (:choices (:trace result)) [0 :x])
              v1 (cm/get-choice (:choices (:trace result)) [1 :x])]
          (mx/eval! v0 v1)
          (is (h/close? 1.1 (h/realize v0) 1e-6) "unfold prefix-skip step[0] kept")
          (is (h/close? 1.2 (h/realize v1) 1e-6) "unfold prefix-skip step[1] kept"))))

    (testing "Unfold: metadata"
      (let [new-obs (-> cm/EMPTY (cm/set-choice [1 :x] (mx/scalar 2.0)))
            result  (p/update unfold-gf trace new-obs)
            meta-t  (meta (:trace result))
            ss      (::comb/step-scores meta-t)]
        (is (::comb/compiled-path meta-t) "unfold meta: ::compiled-path")
        (is (some? ss) "unfold meta: ::step-scores present")
        (is (= 3 (count ss)) "unfold meta: step-scores count")))

    (testing "Unfold: non-compilable fallback"
      (let [unfold-beta (comb/unfold-combinator k-beta)
            args-b [2 (mx/scalar 0.5)]
            init-b (-> cm/EMPTY
                       (cm/set-choice [0 :y] (mx/scalar 0.3))
                       (cm/set-choice [1 :y] (mx/scalar 0.5)))
            trace-b  (:trace (p/generate unfold-beta args-b init-b))
            result-b (p/update unfold-beta trace-b cm/EMPTY)]
        (mx/eval! (:weight result-b))
        (is (h/close? 0.0 (h/realize (:weight result-b)) 1e-6) "unfold beta fallback weight=0")
        (is (not (::comb/compiled-path (meta (:trace result-b)))) "unfold beta fallback: no compiled-path")))))

(deftest scan-update-test
  (let [scan-gf (comb/scan-combinator k-scan)
        scan-h  (comb/scan-combinator (force-handler k-scan))
        inputs  [(mx/scalar 0.0) [(mx/scalar 0.1) (mx/scalar 0.2) (mx/scalar 0.3)]]
        init-constraints (-> cm/EMPTY
                             (cm/set-choice [0 :x] (mx/scalar 0.15))
                             (cm/set-choice [1 :x] (mx/scalar 0.4))
                             (cm/set-choice [2 :x] (mx/scalar 0.8)))
        trace   (:trace (p/generate scan-gf inputs init-constraints))
        trace-h (:trace (p/generate scan-h  inputs init-constraints))]

    (testing "Scan: no constraints -- weight=0"
      (let [result   (p/update scan-gf trace cm/EMPTY)
            result-h (p/update scan-h  trace-h cm/EMPTY)]
        (mx/eval! (:weight result) (:weight result-h))
        (is (h/close? 0.0 (h/realize (:weight result)) 1e-6) "scan no-constraint weight=0")
        (is (h/close? (h/realize (:weight result-h)) (h/realize (:weight result)) 1e-5)
            "scan no-constraint weight matches handler")))

    (testing "Scan: single step constrained"
      (let [new-obs  (-> cm/EMPTY (cm/set-choice [1 :x] (mx/scalar 1.0)))
            result   (p/update scan-gf trace new-obs)
            result-h (p/update scan-h  trace-h new-obs)]
        (mx/eval! (:weight result) (:weight result-h)
                  (:score (:trace result)) (:score (:trace result-h)))
        (is (h/close? (h/realize (:weight result-h)) (h/realize (:weight result)) 1e-5)
            "scan single-constraint weight matches handler")
        (is (h/close? (h/realize (:score (:trace result-h))) (h/realize (:score (:trace result))) 1e-5)
            "scan single-constraint score matches handler")
        (let [v (cm/get-choice (:choices (:trace result)) [1 :x])]
          (mx/eval! v)
          (is (h/close? 1.0 (h/realize v) 1e-6) "scan constrained step[1] applied"))
        (let [d (cm/get-choice (:discard result) [1 :x])]
          (mx/eval! d)
          (is (h/close? 0.4 (h/realize d) 1e-6) "scan discard[1] = old value"))))

    (testing "Scan: all steps constrained"
      (let [new-obs  (-> cm/EMPTY
                         (cm/set-choice [0 :x] (mx/scalar 0.5))
                         (cm/set-choice [1 :x] (mx/scalar 1.5))
                         (cm/set-choice [2 :x] (mx/scalar 2.5)))
            result   (p/update scan-gf trace new-obs)
            result-h (p/update scan-h  trace-h new-obs)]
        (mx/eval! (:weight result) (:weight result-h)
                  (:score (:trace result)) (:score (:trace result-h)))
        (is (h/close? (h/realize (:weight result-h)) (h/realize (:weight result)) 1e-5)
            "scan all-constrained weight matches handler")
        (is (h/close? (h/realize (:score (:trace result-h))) (h/realize (:score (:trace result))) 1e-5)
            "scan all-constrained score matches handler")))

    (testing "Scan: carry threading"
      (let [new-obs  (-> cm/EMPTY
                         (cm/set-choice [0 :x] (mx/scalar 0.5))
                         (cm/set-choice [1 :x] (mx/scalar 1.5))
                         (cm/set-choice [2 :x] (mx/scalar 2.5)))
            result   (p/update scan-gf trace new-obs)
            retval   (:retval (:trace result))]
        (mx/eval! (:carry retval))
        (is (h/close? 2.5 (h/realize (:carry retval)) 1e-6) "scan final carry = last constrained")
        (is (= 3 (count (:outputs retval))) "scan outputs count")))

    (testing "Scan: prefix-skip"
      (let [new-obs  (-> cm/EMPTY (cm/set-choice [2 :x] (mx/scalar 9.0)))
            result   (p/update scan-gf trace new-obs)]
        (let [v0 (cm/get-choice (:choices (:trace result)) [0 :x])
              v1 (cm/get-choice (:choices (:trace result)) [1 :x])]
          (mx/eval! v0 v1)
          (is (h/close? 0.15 (h/realize v0) 1e-6) "scan prefix-skip step[0] kept")
          (is (h/close? 0.4 (h/realize v1) 1e-6) "scan prefix-skip step[1] kept"))))

    (testing "Scan: metadata"
      (let [new-obs (-> cm/EMPTY (cm/set-choice [1 :x] (mx/scalar 1.0)))
            result  (p/update scan-gf trace new-obs)
            meta-t  (meta (:trace result))
            ss      (::comb/step-scores meta-t)
            sc      (::comb/step-carries meta-t)]
        (is (::comb/compiled-path meta-t) "scan meta: ::compiled-path")
        (is (some? ss) "scan meta: ::step-scores present")
        (is (= 3 (count ss)) "scan meta: step-scores count")
        (is (some? sc) "scan meta: ::step-carries present")
        (is (= 3 (count sc)) "scan meta: step-carries count")))

    (testing "Scan: non-compilable fallback"
      (let [scan-beta (comb/scan-combinator k-beta-scan)
            inputs-b  [(mx/scalar 0.5) [(mx/scalar 0.1) (mx/scalar 0.2)]]
            init-b    (-> cm/EMPTY
                          (cm/set-choice [0 :y] (mx/scalar 0.3))
                          (cm/set-choice [1 :y] (mx/scalar 0.5)))
            trace-b   (:trace (p/generate scan-beta inputs-b init-b))
            result-b  (p/update scan-beta trace-b cm/EMPTY)]
        (mx/eval! (:weight result-b))
        (is (h/close? 0.0 (h/realize (:weight result-b)) 1e-6) "scan beta fallback weight=0")
        (is (not (::comb/compiled-path (meta (:trace result-b)))) "scan beta fallback: no compiled-path")))))

(deftest switch-update-test
  (let [switch-gf (comb/switch-combinator k-switch k-switch-b)
        switch-h  (comb/switch-combinator (force-handler k-switch)
                                           (force-handler k-switch-b))
        init-obs-0 (-> cm/EMPTY (cm/set-value :x (mx/scalar 2.0)))
        args-0     [0]
        trace-0    (:trace (p/generate switch-gf args-0 init-obs-0))
        trace-h-0  (:trace (p/generate switch-h  args-0 init-obs-0))]

    (testing "Switch: same branch, no constraints -- weight=0"
      (let [result   (p/update switch-gf trace-0 cm/EMPTY)
            result-h (p/update switch-h  trace-h-0 cm/EMPTY)]
        (mx/eval! (:weight result) (:weight result-h))
        (is (h/close? 0.0 (h/realize (:weight result)) 1e-6)
            "switch same-branch no-constraint weight=0")
        (is (h/close? (h/realize (:weight result-h)) (h/realize (:weight result)) 1e-5)
            "switch same-branch no-constraint weight matches handler")))

    (testing "Switch: same branch, constrained"
      (let [new-obs  (-> cm/EMPTY (cm/set-value :x (mx/scalar 0.5)))
            result   (p/update switch-gf trace-0 new-obs)
            result-h (p/update switch-h  trace-h-0 new-obs)]
        (mx/eval! (:weight result) (:weight result-h)
                  (:score (:trace result)) (:score (:trace result-h)))
        (is (h/close? (h/realize (:weight result-h)) (h/realize (:weight result)) 1e-5)
            "switch same-branch constrained weight matches handler")
        (is (h/close? (h/realize (:score (:trace result-h))) (h/realize (:score (:trace result))) 1e-5)
            "switch same-branch constrained score matches handler")
        (let [v (cm/get-choice (:choices (:trace result)) [:x])]
          (mx/eval! v)
          (is (h/close? 0.5 (h/realize v) 1e-6) "switch constrained value applied"))
        (let [d (cm/get-choice (:discard result) [:x])]
          (mx/eval! d)
          (is (h/close? 2.0 (h/realize d) 1e-6) "switch discard = old value"))))

    (testing "Switch: metadata on same-branch update"
      (let [new-obs (-> cm/EMPTY (cm/set-value :x (mx/scalar 0.5)))
            result  (p/update switch-gf trace-0 new-obs)
            meta-t  (meta (:trace result))]
        (is (::comb/compiled-path meta-t) "switch meta: ::compiled-path")
        (is (= 0 (::comb/switch-idx meta-t)) "switch meta: ::switch-idx")))

    (testing "Switch: non-compilable branch fallback"
      (let [switch-beta (comb/switch-combinator k-beta-switch k-switch)
            init-beta   (-> cm/EMPTY (cm/set-value :y (mx/scalar 0.4)))
            trace-beta  (:trace (p/generate switch-beta [0] init-beta))
            result-beta (p/update switch-beta trace-beta cm/EMPTY)]
        (mx/eval! (:weight result-beta))
        (is (h/close? 0.0 (h/realize (:weight result-beta)) 1e-6) "switch beta fallback weight=0")
        (is (not (::comb/compiled-path (meta (:trace result-beta)))) "switch beta fallback: no compiled-path")))))

(deftest mix-update-test
  (let [mix-gf  (comb/mix-combinator
                  [k-switch k-switch-b]
                  (fn [_args] (mx/array [0.0 0.0])))
        mix-h   (comb/mix-combinator
                  [(force-handler k-switch) (force-handler k-switch-b)]
                  (fn [_args] (mx/array [0.0 0.0])))
        args    []
        init-obs (-> cm/EMPTY
                     (cm/set-choice [:component-idx] (mx/scalar 0 mx/int32))
                     (cm/set-value :x (mx/scalar 2.0)))
        trace   (:trace (p/generate mix-gf args init-obs))
        trace-h (:trace (p/generate mix-h  args init-obs))]

    (testing "Mix: same component, no constraints -- weight=0"
      (let [result   (p/update mix-gf trace cm/EMPTY)
            result-h (p/update mix-h  trace-h cm/EMPTY)]
        (mx/eval! (:weight result) (:weight result-h))
        (is (h/close? 0.0 (h/realize (:weight result)) 1e-6)
            "mix same-component no-constraint weight=0")
        (is (h/close? (h/realize (:weight result-h)) (h/realize (:weight result)) 1e-5)
            "mix same-component no-constraint weight matches handler")))

    (testing "Mix: same component, inner site constrained"
      (let [new-obs  (-> cm/EMPTY (cm/set-value :x (mx/scalar 0.5)))
            result   (p/update mix-gf trace new-obs)
            result-h (p/update mix-h  trace-h new-obs)]
        (mx/eval! (:weight result) (:weight result-h)
                  (:score (:trace result)) (:score (:trace result-h)))
        (is (h/close? (h/realize (:weight result-h)) (h/realize (:weight result)) 1e-5)
            "mix same-component constrained weight matches handler")
        (is (h/close? (h/realize (:score (:trace result-h))) (h/realize (:score (:trace result))) 1e-5)
            "mix same-component constrained score matches handler")
        (let [v (cm/get-choice (:choices (:trace result)) [:x])]
          (mx/eval! v)
          (is (h/close? 0.5 (h/realize v) 1e-6) "mix constrained value applied"))
        (let [idx (cm/get-choice (:choices (:trace result)) [:component-idx])]
          (mx/eval! idx)
          (is (= 0 (int (mx/item idx))) "mix component-idx preserved"))))

    (testing "Mix: metadata on same-component update"
      (let [new-obs (-> cm/EMPTY (cm/set-value :x (mx/scalar 0.5)))
            result  (p/update mix-gf trace new-obs)
            meta-t  (meta (:trace result))]
        (is (::comb/compiled-path meta-t) "mix meta: ::compiled-path")))

    (testing "Mix: non-compilable component fallback"
      (let [mix-beta (comb/mix-combinator
                       [k-beta-switch k-switch]
                       (fn [_args] (mx/array [0.0 0.0])))
            init-beta (-> cm/EMPTY
                          (cm/set-choice [:component-idx] (mx/scalar 0 mx/int32))
                          (cm/set-value :y (mx/scalar 0.4)))
            trace-beta (:trace (p/generate mix-beta args init-beta))
            result-beta (p/update mix-beta trace-beta cm/EMPTY)]
        (mx/eval! (:weight result-beta))
        (is (h/close? 0.0 (h/realize (:weight result-beta)) 1e-6) "mix beta fallback weight=0")
        (is (not (::comb/compiled-path (meta (:trace result-beta)))) "mix beta fallback: no compiled-path")))))

(deftest discard-structure-test
  (testing "Discard structure: Map"
    (let [map-gf (comb/map-combinator k-map)
          inputs [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]]
          init   (-> cm/EMPTY
                     (cm/set-choice [0 :y] (mx/scalar 1.5))
                     (cm/set-choice [1 :y] (mx/scalar 2.5))
                     (cm/set-choice [2 :y] (mx/scalar 3.5)))
          trace  (:trace (p/generate map-gf inputs init))
          result (p/update map-gf trace (-> cm/EMPTY (cm/set-choice [1 :y] (mx/scalar 9.0))))]
      (let [d1 (cm/get-submap (:discard result) 1)]
        (is (and (some? d1) (cm/has-value? (cm/get-submap d1 :y))) "map discard has element 1"))
      (is (= (cm/get-submap (:discard result) 0) cm/EMPTY) "map discard missing element 0")
      (is (= (cm/get-submap (:discard result) 2) cm/EMPTY) "map discard missing element 2")))

  (testing "Discard structure: Unfold"
    (let [unfold-gf (comb/unfold-combinator k-unfold)
          args      [3 (mx/scalar 1.0)]
          init      (-> cm/EMPTY
                        (cm/set-choice [0 :x] (mx/scalar 1.1))
                        (cm/set-choice [1 :x] (mx/scalar 1.2))
                        (cm/set-choice [2 :x] (mx/scalar 1.3)))
          trace  (:trace (p/generate unfold-gf args init))
          result (p/update unfold-gf trace (-> cm/EMPTY (cm/set-choice [2 :x] (mx/scalar 9.0))))]
      (let [d2 (cm/get-choice (:discard result) [2 :x])]
        (mx/eval! d2)
        (is (h/close? 1.3 (h/realize d2) 1e-6) "unfold discard step[2] present")))))

(cljs.test/run-tests)
