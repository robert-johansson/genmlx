(ns genmlx.wp8-combinator-update-test
  "WP-8 tests: combinator compiled update.
   Validates that compiled update paths for Map, Unfold, Scan, Switch, Mix
   produce identical traces, scores, weights, and discards as handler paths."
  (:require-macros [genmlx.gen :refer [gen]])
  (:require [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.compiled :as compiled]
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
  (let [schema (dissoc (:schema gf) :compiled-simulate :compiled-generate
                       :compiled-update
                       :compiled-prefix :compiled-prefix-addrs
                       :compiled-prefix-generate :compiled-prefix-update)]
    (assoc gf :schema schema)))

(defn compilable-update?
  "Check if a gen-fn kernel has a compiled-update function."
  [gf]
  (some? (compiled/get-compiled-update gf)))

(defn make-trace-via-generate
  "Create a deterministic trace via handler generate with full constraints."
  [gf args constraints]
  (let [gf-h (force-handler gf)
        {:keys [trace]} (p/generate (dyn/with-key gf-h (rng/fresh-key 1)) args constraints)]
    trace))

(defn make-combinator-trace
  "Create a deterministic combinator trace via generate with constraints.
   Uses force-handler on the kernel to ensure handler path."
  [make-combinator-fn kernel args constraints]
  (let [kernel-h (force-handler kernel)
        combinator (make-combinator-fn kernel-h)
        {:keys [trace]} (p/generate combinator args constraints)]
    trace))

(println "\n=== WP-8: Combinator Compiled Update Tests ===\n")

;; ---------------------------------------------------------------------------
;; Test kernels (same as WP-7)
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

;; Scan-compatible non-compilable kernel
(def k-beta-scan
  (dyn/auto-key (gen [carry input]
    (let [y (trace :y (dist/beta-dist 2 5))]
      [y y]))))

;; Switch-compatible non-compilable kernel (no args)
(def k-beta-switch
  (dyn/auto-key (gen []
    (let [y (trace :y (dist/beta-dist 2 5))]
      y))))

;; ---------------------------------------------------------------------------
;; Section 0: Prerequisites
;; ---------------------------------------------------------------------------

(println "-- 0. Prerequisites --")
(assert-true "map kernel has compiled-update" (compilable-update? k-map))
(assert-true "unfold kernel has compiled-update" (compilable-update? k-unfold))
(assert-true "scan kernel has compiled-update" (compilable-update? k-scan))
(assert-true "switch kernel has compiled-update" (compilable-update? k-switch))
(assert-true "beta kernel NOT compilable-update" (not (compilable-update? k-beta)))

;; ===========================================================================
;; Section 1: MAP COMBINATOR UPDATE
;; ===========================================================================

(println "\n-- 1. Map: compiled update --")

(let [map-gf (comb/map-combinator k-map)
      map-h  (comb/map-combinator (force-handler k-map))
      inputs [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]]
      ;; Create deterministic starting trace with all elements constrained
      init-constraints (-> cm/EMPTY
                           (cm/set-choice [0 :y] (mx/scalar 1.5))
                           (cm/set-choice [1 :y] (mx/scalar 2.5))
                           (cm/set-choice [2 :y] (mx/scalar 3.5)))
      trace   (:trace (p/generate map-gf inputs init-constraints))
      trace-h (:trace (p/generate map-h  inputs init-constraints))]

  ;; 1a. No constraints — weight=0, choices unchanged
  (let [result   (p/update map-gf trace cm/EMPTY)
        result-h (p/update map-h  trace-h cm/EMPTY)]
    (mx/eval! (:weight result) (:weight result-h))
    (assert-close "map no-constraint weight=0" 0.0 (:weight result) 1e-6)
    (assert-close "map no-constraint score unchanged"
                  (:score trace) (:score (:trace result)) 1e-6)
    (assert-close "map no-constraint weight matches handler"
                  (:weight result-h) (:weight result) 1e-5))

  ;; 1b. Single element constrained
  (let [new-obs  (-> cm/EMPTY (cm/set-choice [1 :y] (mx/scalar 5.0)))
        result   (p/update map-gf trace new-obs)
        result-h (p/update map-h  trace-h new-obs)]
    (mx/eval! (:weight result) (:weight result-h)
              (:score (:trace result)) (:score (:trace result-h)))
    (assert-close "map single-constraint weight matches handler"
                  (:weight result-h) (:weight result) 1e-5)
    (assert-close "map single-constraint score matches handler"
                  (:score (:trace result-h)) (:score (:trace result)) 1e-5)
    ;; Constrained value applied
    (let [v (cm/get-choice (:choices (:trace result)) [1 :y])]
      (mx/eval! v)
      (assert-close "map constrained value applied" 5.0 v 1e-6))
    ;; Unconstrained values kept
    (let [v0 (cm/get-choice (:choices (:trace result)) [0 :y])]
      (mx/eval! v0)
      (assert-close "map unconstrained[0] kept" 1.5 v0 1e-6))
    ;; Discard contains old value of constrained site
    (let [d (cm/get-choice (:discard result) [1 :y])]
      (mx/eval! d)
      (assert-close "map discard[1] = old value" 2.5 d 1e-6)))

  ;; 1c. All elements constrained
  (let [new-obs  (-> cm/EMPTY
                     (cm/set-choice [0 :y] (mx/scalar 10.0))
                     (cm/set-choice [1 :y] (mx/scalar 20.0))
                     (cm/set-choice [2 :y] (mx/scalar 30.0)))
        result   (p/update map-gf trace new-obs)
        result-h (p/update map-h  trace-h new-obs)]
    (mx/eval! (:weight result) (:weight result-h)
              (:score (:trace result)) (:score (:trace result-h)))
    (assert-close "map all-constrained weight matches handler"
                  (:weight result-h) (:weight result) 1e-5)
    (assert-close "map all-constrained score matches handler"
                  (:score (:trace result-h)) (:score (:trace result)) 1e-5))

  ;; 1d. Metadata
  (let [new-obs (-> cm/EMPTY (cm/set-choice [0 :y] (mx/scalar 10.0)))
        result  (p/update map-gf trace new-obs)
        meta-t  (meta (:trace result))
        es      (::comb/element-scores meta-t)]
    (assert-true "map meta: ::compiled-path"
                 (::comb/compiled-path meta-t))
    (assert-true "map meta: ::element-scores present" (some? es))
    (assert-equal "map meta: element-scores count" 3 (count es)))

  ;; 1e. Non-compilable fallback
  (let [map-beta (comb/map-combinator k-beta)
        inputs-b [[(mx/scalar 1.0) (mx/scalar 2.0)]]
        init-b   (-> cm/EMPTY
                     (cm/set-choice [0 :y] (mx/scalar 0.3))
                     (cm/set-choice [1 :y] (mx/scalar 0.5)))
        trace-b  (:trace (p/generate map-beta inputs-b init-b))
        result-b (p/update map-beta trace-b cm/EMPTY)]
    (mx/eval! (:weight result-b))
    (assert-close "map beta fallback weight=0" 0.0 (:weight result-b) 1e-6)
    (assert-true "map beta fallback: no compiled-path"
                 (not (::comb/compiled-path (meta (:trace result-b)))))))

;; ===========================================================================
;; Section 2: UNFOLD COMBINATOR UPDATE
;; ===========================================================================

(println "\n-- 2. Unfold: compiled update --")

(let [unfold-gf (comb/unfold-combinator k-unfold)
      unfold-h  (comb/unfold-combinator (force-handler k-unfold))
      args      [3 (mx/scalar 1.0)]
      init-constraints (-> cm/EMPTY
                           (cm/set-choice [0 :x] (mx/scalar 1.1))
                           (cm/set-choice [1 :x] (mx/scalar 1.2))
                           (cm/set-choice [2 :x] (mx/scalar 1.3)))
      trace   (:trace (p/generate unfold-gf args init-constraints))
      trace-h (:trace (p/generate unfold-h  args init-constraints))]

  ;; 2a. No constraints — weight=0
  (let [result   (p/update unfold-gf trace cm/EMPTY)
        result-h (p/update unfold-h  trace-h cm/EMPTY)]
    (mx/eval! (:weight result) (:weight result-h))
    (assert-close "unfold no-constraint weight=0" 0.0 (:weight result) 1e-6)
    (assert-close "unfold no-constraint weight matches handler"
                  (:weight result-h) (:weight result) 1e-5))

  ;; 2b. Single step constrained (step 1)
  (let [new-obs  (-> cm/EMPTY (cm/set-choice [1 :x] (mx/scalar 2.0)))
        result   (p/update unfold-gf trace new-obs)
        result-h (p/update unfold-h  trace-h new-obs)]
    (mx/eval! (:weight result) (:weight result-h)
              (:score (:trace result)) (:score (:trace result-h)))
    (assert-close "unfold single-constraint weight matches handler"
                  (:weight result-h) (:weight result) 1e-5)
    (assert-close "unfold single-constraint score matches handler"
                  (:score (:trace result-h)) (:score (:trace result)) 1e-5)
    ;; Constrained value applied
    (let [v (cm/get-choice (:choices (:trace result)) [1 :x])]
      (mx/eval! v)
      (assert-close "unfold constrained step[1] applied" 2.0 v 1e-6))
    ;; Step 0 kept (prefix-skip)
    (let [v0 (cm/get-choice (:choices (:trace result)) [0 :x])]
      (mx/eval! v0)
      (assert-close "unfold step[0] kept (prefix-skip)" 1.1 v0 1e-6))
    ;; Discard contains old value
    (let [d (cm/get-choice (:discard result) [1 :x])]
      (mx/eval! d)
      (assert-close "unfold discard[1] = old value" 1.2 d 1e-6)))

  ;; 2c. All steps constrained
  (let [new-obs  (-> cm/EMPTY
                     (cm/set-choice [0 :x] (mx/scalar 5.0))
                     (cm/set-choice [1 :x] (mx/scalar 6.0))
                     (cm/set-choice [2 :x] (mx/scalar 7.0)))
        result   (p/update unfold-gf trace new-obs)
        result-h (p/update unfold-h  trace-h new-obs)]
    (mx/eval! (:weight result) (:weight result-h)
              (:score (:trace result)) (:score (:trace result-h)))
    (assert-close "unfold all-constrained weight matches handler"
                  (:weight result-h) (:weight result) 1e-5)
    (assert-close "unfold all-constrained score matches handler"
                  (:score (:trace result-h)) (:score (:trace result)) 1e-5))

  ;; 2d. State threading — constrained values become state for subsequent steps
  (let [new-obs  (-> cm/EMPTY (cm/set-choice [0 :x] (mx/scalar 5.0)))
        result   (p/update unfold-gf trace new-obs)
        states   (:retval (:trace result))]
    (mx/eval! (first states))
    (assert-close "unfold state[0] = constrained value" 5.0 (first states) 1e-6))

  ;; 2e. Prefix-skip: constrain only last step — first two reused
  (let [new-obs  (-> cm/EMPTY (cm/set-choice [2 :x] (mx/scalar 9.0)))
        result   (p/update unfold-gf trace new-obs)]
    (let [v0 (cm/get-choice (:choices (:trace result)) [0 :x])
          v1 (cm/get-choice (:choices (:trace result)) [1 :x])]
      (mx/eval! v0 v1)
      (assert-close "unfold prefix-skip step[0] kept" 1.1 v0 1e-6)
      (assert-close "unfold prefix-skip step[1] kept" 1.2 v1 1e-6)))

  ;; 2f. Metadata
  (let [new-obs (-> cm/EMPTY (cm/set-choice [1 :x] (mx/scalar 2.0)))
        result  (p/update unfold-gf trace new-obs)
        meta-t  (meta (:trace result))
        ss      (::comb/step-scores meta-t)]
    (assert-true "unfold meta: ::compiled-path"
                 (::comb/compiled-path meta-t))
    (assert-true "unfold meta: ::step-scores present" (some? ss))
    (assert-equal "unfold meta: step-scores count" 3 (count ss)))

  ;; 2g. Non-compilable fallback (k-beta has [x] sig, unfold calls with [t state])
  (let [unfold-beta (comb/unfold-combinator k-beta)
        args-b [2 (mx/scalar 0.5)]
        init-b (-> cm/EMPTY
                   (cm/set-choice [0 :y] (mx/scalar 0.3))
                   (cm/set-choice [1 :y] (mx/scalar 0.5)))
        trace-b  (:trace (p/generate unfold-beta args-b init-b))
        result-b (p/update unfold-beta trace-b cm/EMPTY)]
    (mx/eval! (:weight result-b))
    (assert-close "unfold beta fallback weight=0" 0.0 (:weight result-b) 1e-6)
    (assert-true "unfold beta fallback: no compiled-path"
                 (not (::comb/compiled-path (meta (:trace result-b)))))))

;; ===========================================================================
;; Section 3: SCAN COMBINATOR UPDATE
;; ===========================================================================

(println "\n-- 3. Scan: compiled update --")

(let [scan-gf (comb/scan-combinator k-scan)
      scan-h  (comb/scan-combinator (force-handler k-scan))
      inputs  [(mx/scalar 0.0) [(mx/scalar 0.1) (mx/scalar 0.2) (mx/scalar 0.3)]]
      init-constraints (-> cm/EMPTY
                           (cm/set-choice [0 :x] (mx/scalar 0.15))
                           (cm/set-choice [1 :x] (mx/scalar 0.4))
                           (cm/set-choice [2 :x] (mx/scalar 0.8)))
      trace   (:trace (p/generate scan-gf inputs init-constraints))
      trace-h (:trace (p/generate scan-h  inputs init-constraints))]

  ;; 3a. No constraints — weight=0
  (let [result   (p/update scan-gf trace cm/EMPTY)
        result-h (p/update scan-h  trace-h cm/EMPTY)]
    (mx/eval! (:weight result) (:weight result-h))
    (assert-close "scan no-constraint weight=0" 0.0 (:weight result) 1e-6)
    (assert-close "scan no-constraint weight matches handler"
                  (:weight result-h) (:weight result) 1e-5))

  ;; 3b. Single step constrained
  (let [new-obs  (-> cm/EMPTY (cm/set-choice [1 :x] (mx/scalar 1.0)))
        result   (p/update scan-gf trace new-obs)
        result-h (p/update scan-h  trace-h new-obs)]
    (mx/eval! (:weight result) (:weight result-h)
              (:score (:trace result)) (:score (:trace result-h)))
    (assert-close "scan single-constraint weight matches handler"
                  (:weight result-h) (:weight result) 1e-5)
    (assert-close "scan single-constraint score matches handler"
                  (:score (:trace result-h)) (:score (:trace result)) 1e-5)
    ;; Constrained value
    (let [v (cm/get-choice (:choices (:trace result)) [1 :x])]
      (mx/eval! v)
      (assert-close "scan constrained step[1] applied" 1.0 v 1e-6))
    ;; Discard
    (let [d (cm/get-choice (:discard result) [1 :x])]
      (mx/eval! d)
      (assert-close "scan discard[1] = old value" 0.4 d 1e-6)))

  ;; 3c. All steps constrained
  (let [new-obs  (-> cm/EMPTY
                     (cm/set-choice [0 :x] (mx/scalar 0.5))
                     (cm/set-choice [1 :x] (mx/scalar 1.5))
                     (cm/set-choice [2 :x] (mx/scalar 2.5)))
        result   (p/update scan-gf trace new-obs)
        result-h (p/update scan-h  trace-h new-obs)]
    (mx/eval! (:weight result) (:weight result-h)
              (:score (:trace result)) (:score (:trace result-h)))
    (assert-close "scan all-constrained weight matches handler"
                  (:weight result-h) (:weight result) 1e-5)
    (assert-close "scan all-constrained score matches handler"
                  (:score (:trace result-h)) (:score (:trace result)) 1e-5))

  ;; 3d. Carry threading
  (let [new-obs  (-> cm/EMPTY
                     (cm/set-choice [0 :x] (mx/scalar 0.5))
                     (cm/set-choice [1 :x] (mx/scalar 1.5))
                     (cm/set-choice [2 :x] (mx/scalar 2.5)))
        result   (p/update scan-gf trace new-obs)
        retval   (:retval (:trace result))]
    (mx/eval! (:carry retval))
    (assert-close "scan final carry = last constrained" 2.5 (:carry retval) 1e-6)
    (assert-equal "scan outputs count" 3 (count (:outputs retval))))

  ;; 3e. Prefix-skip: constrain only last step
  (let [new-obs  (-> cm/EMPTY (cm/set-choice [2 :x] (mx/scalar 9.0)))
        result   (p/update scan-gf trace new-obs)]
    (let [v0 (cm/get-choice (:choices (:trace result)) [0 :x])
          v1 (cm/get-choice (:choices (:trace result)) [1 :x])]
      (mx/eval! v0 v1)
      (assert-close "scan prefix-skip step[0] kept" 0.15 v0 1e-6)
      (assert-close "scan prefix-skip step[1] kept" 0.4 v1 1e-6)))

  ;; 3f. Metadata
  (let [new-obs (-> cm/EMPTY (cm/set-choice [1 :x] (mx/scalar 1.0)))
        result  (p/update scan-gf trace new-obs)
        meta-t  (meta (:trace result))
        ss      (::comb/step-scores meta-t)
        sc      (::comb/step-carries meta-t)]
    (assert-true "scan meta: ::compiled-path"
                 (::comb/compiled-path meta-t))
    (assert-true "scan meta: ::step-scores present" (some? ss))
    (assert-equal "scan meta: step-scores count" 3 (count ss))
    (assert-true "scan meta: ::step-carries present" (some? sc))
    (assert-equal "scan meta: step-carries count" 3 (count sc)))

  ;; 3g. Non-compilable fallback (k-beta-scan returns [y y])
  (let [scan-beta (comb/scan-combinator k-beta-scan)
        inputs-b  [(mx/scalar 0.5) [(mx/scalar 0.1) (mx/scalar 0.2)]]
        init-b    (-> cm/EMPTY
                      (cm/set-choice [0 :y] (mx/scalar 0.3))
                      (cm/set-choice [1 :y] (mx/scalar 0.5)))
        trace-b   (:trace (p/generate scan-beta inputs-b init-b))
        result-b  (p/update scan-beta trace-b cm/EMPTY)]
    (mx/eval! (:weight result-b))
    (assert-close "scan beta fallback weight=0" 0.0 (:weight result-b) 1e-6)
    (assert-true "scan beta fallback: no compiled-path"
                 (not (::comb/compiled-path (meta (:trace result-b)))))))

;; ===========================================================================
;; Section 4: SWITCH COMBINATOR UPDATE
;; ===========================================================================

(println "\n-- 4. Switch: compiled update --")

(let [switch-gf (comb/switch-combinator k-switch k-switch-b)
      switch-h  (comb/switch-combinator (force-handler k-switch)
                                         (force-handler k-switch-b))
      ;; Create trace on branch 0
      init-obs-0 (-> cm/EMPTY (cm/set-value :x (mx/scalar 2.0)))
      args-0     [0]
      trace-0    (:trace (p/generate switch-gf args-0 init-obs-0))
      trace-h-0  (:trace (p/generate switch-h  args-0 init-obs-0))]

  ;; 4a. Same branch, no constraints — weight=0
  (let [result   (p/update switch-gf trace-0 cm/EMPTY)
        result-h (p/update switch-h  trace-h-0 cm/EMPTY)]
    (mx/eval! (:weight result) (:weight result-h))
    (assert-close "switch same-branch no-constraint weight=0"
                  0.0 (:weight result) 1e-6)
    (assert-close "switch same-branch no-constraint weight matches handler"
                  (:weight result-h) (:weight result) 1e-5))

  ;; 4b. Same branch, constrained
  (let [new-obs  (-> cm/EMPTY (cm/set-value :x (mx/scalar 0.5)))
        result   (p/update switch-gf trace-0 new-obs)
        result-h (p/update switch-h  trace-h-0 new-obs)]
    (mx/eval! (:weight result) (:weight result-h)
              (:score (:trace result)) (:score (:trace result-h)))
    (assert-close "switch same-branch constrained weight matches handler"
                  (:weight result-h) (:weight result) 1e-5)
    (assert-close "switch same-branch constrained score matches handler"
                  (:score (:trace result-h)) (:score (:trace result)) 1e-5)
    ;; Constrained value applied
    (let [v (cm/get-choice (:choices (:trace result)) [:x])]
      (mx/eval! v)
      (assert-close "switch constrained value applied" 0.5 v 1e-6))
    ;; Discard contains old value
    (let [d (cm/get-choice (:discard result) [:x])]
      (mx/eval! d)
      (assert-close "switch discard = old value" 2.0 d 1e-6)))

  ;; 4c. Metadata on same-branch update
  (let [new-obs (-> cm/EMPTY (cm/set-value :x (mx/scalar 0.5)))
        result  (p/update switch-gf trace-0 new-obs)
        meta-t  (meta (:trace result))]
    (assert-true "switch meta: ::compiled-path"
                 (::comb/compiled-path meta-t))
    (assert-equal "switch meta: ::switch-idx" 0
                  (::comb/switch-idx meta-t)))

  ;; 4d. Non-compilable branch fallback
  (let [switch-beta (comb/switch-combinator k-beta-switch k-switch)
        init-beta   (-> cm/EMPTY (cm/set-value :y (mx/scalar 0.4)))
        trace-beta  (:trace (p/generate switch-beta [0] init-beta))
        result-beta (p/update switch-beta trace-beta cm/EMPTY)]
    (mx/eval! (:weight result-beta))
    (assert-close "switch beta fallback weight=0" 0.0 (:weight result-beta) 1e-6)
    (assert-true "switch beta fallback: no compiled-path"
                 (not (::comb/compiled-path (meta (:trace result-beta)))))))

;; ===========================================================================
;; Section 5: MIX COMBINATOR UPDATE
;; ===========================================================================

(println "\n-- 5. Mix: compiled update --")

(let [mix-gf  (comb/mix-combinator
                [k-switch k-switch-b]
                (fn [_args] (mx/array [0.0 0.0])))
      mix-h   (comb/mix-combinator
                [(force-handler k-switch) (force-handler k-switch-b)]
                (fn [_args] (mx/array [0.0 0.0])))
      args    []
      ;; Create trace on component 0
      init-obs (-> cm/EMPTY
                   (cm/set-choice [:component-idx] (mx/scalar 0 mx/int32))
                   (cm/set-value :x (mx/scalar 2.0)))
      trace   (:trace (p/generate mix-gf args init-obs))
      trace-h (:trace (p/generate mix-h  args init-obs))]

  ;; 5a. Same component, no constraints — weight=0
  (let [result   (p/update mix-gf trace cm/EMPTY)
        result-h (p/update mix-h  trace-h cm/EMPTY)]
    (mx/eval! (:weight result) (:weight result-h))
    (assert-close "mix same-component no-constraint weight=0"
                  0.0 (:weight result) 1e-6)
    (assert-close "mix same-component no-constraint weight matches handler"
                  (:weight result-h) (:weight result) 1e-5))

  ;; 5b. Same component, inner site constrained
  (let [new-obs  (-> cm/EMPTY (cm/set-value :x (mx/scalar 0.5)))
        result   (p/update mix-gf trace new-obs)
        result-h (p/update mix-h  trace-h new-obs)]
    (mx/eval! (:weight result) (:weight result-h)
              (:score (:trace result)) (:score (:trace result-h)))
    (assert-close "mix same-component constrained weight matches handler"
                  (:weight result-h) (:weight result) 1e-5)
    (assert-close "mix same-component constrained score matches handler"
                  (:score (:trace result-h)) (:score (:trace result)) 1e-5)
    ;; Constrained value applied
    (let [v (cm/get-choice (:choices (:trace result)) [:x])]
      (mx/eval! v)
      (assert-close "mix constrained value applied" 0.5 v 1e-6))
    ;; Component index preserved
    (let [idx (cm/get-choice (:choices (:trace result)) [:component-idx])]
      (mx/eval! idx)
      (assert-equal "mix component-idx preserved" 0 (int (mx/item idx)))))

  ;; 5c. Metadata on same-component update
  (let [new-obs (-> cm/EMPTY (cm/set-value :x (mx/scalar 0.5)))
        result  (p/update mix-gf trace new-obs)
        meta-t  (meta (:trace result))]
    (assert-true "mix meta: ::compiled-path"
                 (::comb/compiled-path meta-t)))

  ;; 5d. Non-compilable component fallback
  (let [mix-beta (comb/mix-combinator
                   [k-beta-switch k-switch]
                   (fn [_args] (mx/array [0.0 0.0])))
        init-beta (-> cm/EMPTY
                      (cm/set-choice [:component-idx] (mx/scalar 0 mx/int32))
                      (cm/set-value :y (mx/scalar 0.4)))
        trace-beta (:trace (p/generate mix-beta args init-beta))
        result-beta (p/update mix-beta trace-beta cm/EMPTY)]
    (mx/eval! (:weight result-beta))
    (assert-close "mix beta fallback weight=0" 0.0 (:weight result-beta) 1e-6)
    (assert-true "mix beta fallback: no compiled-path"
                 (not (::comb/compiled-path (meta (:trace result-beta)))))))

;; ===========================================================================
;; Section 6: Cross-combinator discard structure
;; ===========================================================================

(println "\n-- 6. Discard structure validation --")

;; Map: discard only for constrained elements
(let [map-gf (comb/map-combinator k-map)
      inputs [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]]
      init   (-> cm/EMPTY
                 (cm/set-choice [0 :y] (mx/scalar 1.5))
                 (cm/set-choice [1 :y] (mx/scalar 2.5))
                 (cm/set-choice [2 :y] (mx/scalar 3.5)))
      trace  (:trace (p/generate map-gf inputs init))
      ;; Only constrain element 1
      result (p/update map-gf trace (-> cm/EMPTY (cm/set-choice [1 :y] (mx/scalar 9.0))))]
  (let [d1 (cm/get-submap (:discard result) 1)]
    (assert-true "map discard has element 1"
                 (and (some? d1) (cm/has-value? (cm/get-submap d1 :y)))))
  (assert-true "map discard missing element 0"
               (= (cm/get-submap (:discard result) 0) cm/EMPTY))
  (assert-true "map discard missing element 2"
               (= (cm/get-submap (:discard result) 2) cm/EMPTY)))

;; Unfold: discard only for re-executed steps
(let [unfold-gf (comb/unfold-combinator k-unfold)
      args      [3 (mx/scalar 1.0)]
      init      (-> cm/EMPTY
                    (cm/set-choice [0 :x] (mx/scalar 1.1))
                    (cm/set-choice [1 :x] (mx/scalar 1.2))
                    (cm/set-choice [2 :x] (mx/scalar 1.3)))
      trace  (:trace (p/generate unfold-gf args init))
      ;; Constrain step 2 only — steps 0,1 should not appear in discard
      result (p/update unfold-gf trace (-> cm/EMPTY (cm/set-choice [2 :x] (mx/scalar 9.0))))]
  (let [d2 (cm/get-choice (:discard result) [2 :x])]
    (mx/eval! d2)
    (assert-close "unfold discard step[2] present" 1.3 d2 1e-6)))

;; ===========================================================================
;; Summary
;; ===========================================================================

(println (str "\n=== WP-8 Results: " @pass-count "/" (+ @pass-count @fail-count)
              " passed ==="))
(when (pos? @fail-count)
  (println (str "  " @fail-count " FAILURES")))
