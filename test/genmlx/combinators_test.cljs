(ns genmlx.combinators-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.diff :as diff]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert= [msg expected actual]
  (if (= expected actual)
    (println "  PASS:" msg)
    (do (println "  FAIL:" msg)
        (println "    expected:" (pr-str expected))
        (println "    actual:  " (pr-str actual)))))

(println "\n=== Combinator Tests ===\n")

;; Map combinator
(println "-- Map combinator --")
(let [kernel (gen [x]
               (let [y (dyn/trace :y (dist/gaussian x 1))]
                 (mx/eval! y)
                 (mx/item y)))
      mapped (comb/map-combinator kernel)
      trace (p/simulate mapped [[1.0 2.0 3.0]])]
  (assert-true "map returns trace" (instance? tr/Trace trace))
  (assert= "map returns 3 values" 3 (count (:retval trace)))
  (assert-true "map retvals are numbers" (every? number? (:retval trace))))

;; Map combinator with generate
(println "\n-- Map combinator generate --")
(let [kernel (gen [x]
               (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                 (mx/eval! y)
                 (mx/item y)))
      mapped (comb/map-combinator kernel)
      constraints (cm/set-choice cm/EMPTY [0] (cm/choicemap :y (mx/scalar 1.5)))
      {:keys [trace weight]} (p/generate mapped [[1.0 2.0]] constraints)]
  (assert-true "map generate returns trace" (instance? tr/Trace trace))
  (mx/eval! weight)
  (assert-true "map generate has weight" (number? (mx/item weight))))

;; Unfold combinator
(println "\n-- Unfold combinator --")
(let [step (gen [t state]
             (let [next (dyn/trace :x (dist/gaussian state 0.1))]
               (mx/eval! next)
               (mx/item next)))
      unfold (comb/unfold-combinator step)
      trace (p/simulate unfold [5 0.0])]
  (assert-true "unfold returns trace" (instance? tr/Trace trace))
  (assert= "unfold returns 5 states" 5 (count (:retval trace)))
  (assert-true "unfold retvals are numbers" (every? number? (:retval trace))))

;; Switch combinator
(println "\n-- Switch combinator --")
(let [branch0 (gen []
                (let [x (dyn/trace :x (dist/gaussian 0 1))]
                  (mx/eval! x)
                  (mx/item x)))
      branch1 (gen []
                (let [x (dyn/trace :x (dist/gaussian 10 1))]
                  (mx/eval! x)
                  (mx/item x)))
      sw (comb/switch-combinator branch0 branch1)
      trace0 (p/simulate sw [0])
      trace1 (p/simulate sw [1])]
  (assert-true "switch branch 0 returns trace" (instance? tr/Trace trace0))
  (assert-true "switch branch 1 returns trace" (instance? tr/Trace trace1))
  (assert-true "branch 0 value near 0" (< (js/Math.abs (:retval trace0)) 5))
  (assert-true "branch 1 value near 10" (< (js/Math.abs (- (:retval trace1) 10)) 5)))

;; Mask combinator
(println "\n-- Mask combinator --")
(let [inner (gen [x]
              (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                (mx/eval! y)
                (mx/item y)))
      masked (comb/mask-combinator inner)
      ;; Active mask
      trace-active (p/simulate masked [true 5.0])
      ;; Inactive mask
      trace-inactive (p/simulate masked [false 5.0])]
  (assert-true "mask active returns trace" (instance? tr/Trace trace-active))
  (assert-true "mask active has choices" (not= (:choices trace-active) cm/EMPTY))
  (assert-true "mask inactive has empty choices" (= (:choices trace-inactive) cm/EMPTY))
  (assert-true "mask inactive retval is nil" (nil? (:retval trace-inactive))))

;; Mask combinator update (active)
(println "\n-- Mask update (active) --")
(let [inner (gen [x]
              (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                (mx/eval! y)
                (mx/item y)))
      masked (comb/mask-combinator inner)
      obs (cm/choicemap :y (mx/scalar 5.0))
      {:keys [trace]} (p/generate masked [true 5.0] obs)
      ;; Update with new constraint
      new-obs (cm/choicemap :y (mx/scalar 5.5))
      {:keys [trace weight discard]} (p/update masked trace new-obs)]
  (mx/eval! weight)
  (assert-true "mask update active returns trace" (instance? tr/Trace trace))
  (assert-true "mask update active has weight" (number? (mx/item weight)))
  (assert-true "mask update active has non-zero weight" (not= 0.0 (mx/item weight)))
  (assert-true "mask update active has discard" (not (nil? discard)))
  (let [new-val (cm/get-choice (:choices trace) [:y])]
    (mx/eval! new-val)
    (assert-true "mask update active updated value" (< (js/Math.abs (- (mx/item new-val) 5.5)) 0.01))))

;; Mask combinator update (inactive)
(println "\n-- Mask update (inactive) --")
(let [inner (gen [x]
              (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                (mx/eval! y)
                (mx/item y)))
      masked (comb/mask-combinator inner)
      trace (p/simulate masked [false 5.0])
      {:keys [trace weight discard]} (p/update masked trace cm/EMPTY)]
  (mx/eval! weight)
  (assert-true "mask update inactive returns same trace" (instance? tr/Trace trace))
  (assert-true "mask update inactive zero weight" (= 0.0 (mx/item weight)))
  (assert-true "mask update inactive empty discard" (= discard cm/EMPTY)))

;; Mask combinator regenerate (active)
(println "\n-- Mask regenerate (active) --")
(let [inner (gen [x]
              (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                (mx/eval! y)
                (mx/item y)))
      masked (comb/mask-combinator inner)
      obs (cm/choicemap :y (mx/scalar 5.0))
      {:keys [trace]} (p/generate masked [true 5.0] obs)
      ;; Regenerate the :y address
      {:keys [trace weight]} (p/regenerate masked trace (sel/select :y))]
  (mx/eval! weight)
  (assert-true "mask regenerate active returns trace" (instance? tr/Trace trace))
  (assert-true "mask regenerate active has weight" (number? (mx/item weight)))
  (assert-true "mask regenerate active has choices" (not= (:choices trace) cm/EMPTY)))

;; Mask combinator regenerate (inactive)
(println "\n-- Mask regenerate (inactive) --")
(let [inner (gen [x]
              (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                (mx/eval! y)
                (mx/item y)))
      masked (comb/mask-combinator inner)
      trace (p/simulate masked [false 5.0])
      {:keys [trace weight]} (p/regenerate masked trace (sel/select :y))]
  (mx/eval! weight)
  (assert-true "mask regenerate inactive returns same trace" (instance? tr/Trace trace))
  (assert-true "mask regenerate inactive zero weight" (= 0.0 (mx/item weight))))

;; Mask combinator update-with-diffs (no change fast path)
(println "\n-- Mask update-with-diffs --")
(let [inner (gen [x]
              (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                (mx/eval! y)
                (mx/item y)))
      masked (comb/mask-combinator inner)
      obs (cm/choicemap :y (mx/scalar 5.0))
      {:keys [trace]} (p/generate masked [true 5.0] obs)
      ;; No-change fast path
      {:keys [trace weight discard]} (p/update-with-diffs masked trace cm/EMPTY diff/no-change)]
  (mx/eval! weight)
  (assert-true "mask update-with-diffs no-change returns same trace" (identical? trace trace))
  (assert-true "mask update-with-diffs no-change zero weight" (= 0.0 (mx/item weight)))
  (assert-true "mask update-with-diffs no-change empty discard" (= discard cm/EMPTY)))

;; Contramap update-with-diffs (no change fast path)
(println "\n-- Contramap update-with-diffs --")
(let [inner (gen [x]
              (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                (mx/eval! y)
                (mx/item y)))
      cmapped (comb/contramap-gf inner identity)
      obs (cm/choicemap :y (mx/scalar 3.0))
      {:keys [trace]} (p/generate cmapped [3.0] obs)
      {:keys [trace weight discard]} (p/update-with-diffs cmapped trace cm/EMPTY diff/no-change)]
  (mx/eval! weight)
  (assert-true "contramap update-with-diffs no-change zero weight" (= 0.0 (mx/item weight)))
  (assert-true "contramap update-with-diffs no-change empty discard" (= discard cm/EMPTY)))

;; MapRetval update-with-diffs (no change fast path)
(println "\n-- MapRetval update-with-diffs --")
(let [inner (gen [x]
              (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                (mx/eval! y)
                (mx/item y)))
      mr (comb/map-retval inner (fn [v] (* v 2)))
      obs (cm/choicemap :y (mx/scalar 3.0))
      {:keys [trace]} (p/generate mr [3.0] obs)
      {:keys [trace weight discard]} (p/update-with-diffs mr trace cm/EMPTY diff/no-change)]
  (mx/eval! weight)
  (assert-true "map-retval update-with-diffs no-change zero weight" (= 0.0 (mx/item weight)))
  (assert-true "map-retval update-with-diffs no-change empty discard" (= discard cm/EMPTY)))

(println "\nAll combinator tests complete.")
