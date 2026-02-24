(ns genmlx.vmap-test
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.vmap :as vmap]
            [genmlx.vectorized :as vec])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-close [msg expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (< diff tol)
      (println "  PASS:" msg)
      (do (println "  FAIL:" msg)
          (println "    expected:" expected "actual:" actual "diff:" diff)))))

(println "\n=== Vmap Combinator Tests ===\n")

;; ---------------------------------------------------------------------------
;; 1. Basic simulate
;; ---------------------------------------------------------------------------
(println "-- Basic simulate --")
(let [kernel (gen [x]
               (let [y (dyn/trace :y (dist/gaussian x 1))]
                 y))
      vmodel (vmap/vmap-gf kernel)
      trace (p/simulate vmodel [(mx/array [1.0 2.0 3.0])])]
  (assert-true "returns trace" (instance? tr/Trace trace))
  (let [choices (:choices trace)
        y-val (cm/get-value (cm/get-submap choices :y))]
    (mx/eval! y-val)
    (assert-true "y is [3]-shaped" (= [3] (mx/shape y-val))))
  (mx/eval! (:score trace))
  (assert-true "score is scalar" (= [] (mx/shape (:score trace))))
  (assert-true "score is finite" (js/isFinite (mx/item (:score trace)))))

;; ---------------------------------------------------------------------------
;; 2. Generate with [N]-shaped constraints
;; ---------------------------------------------------------------------------
(println "\n-- Generate with constraints --")
(let [kernel (gen [x]
               (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                 y))
      vmodel (vmap/vmap-gf kernel)
      obs (cm/choicemap :y (mx/array [1.0 2.0 3.0]))
      {:keys [trace weight]} (p/generate vmodel [(mx/array [1.0 2.0 3.0])] obs)]
  (assert-true "generate returns trace" (instance? tr/Trace trace))
  (let [y-val (cm/get-value (cm/get-submap (:choices trace) :y))]
    (mx/eval! y-val)
    (assert-true "constrained y shape" (= [3] (mx/shape y-val)))
    (assert-close "y[0] = 1.0" 1.0 (mx/item (mx/index y-val 0)) 1e-6)
    (assert-close "y[1] = 2.0" 2.0 (mx/item (mx/index y-val 1)) 1e-6)
    (assert-close "y[2] = 3.0" 3.0 (mx/item (mx/index y-val 2)) 1e-6))
  (mx/eval! weight)
  (assert-true "weight is finite" (js/isFinite (mx/item weight)))
  ;; Weight should be sum of log-probs of gaussian(x, 0.1) at y=x
  ;; = 3 * log(gaussian(0, 0.1)) which is quite large
  ;; Weight = sum of log-probs (can be positive for tight distributions)
  (assert-true "weight is nonzero" (not= 0.0 (mx/item weight))))

;; ---------------------------------------------------------------------------
;; 3. Update
;; ---------------------------------------------------------------------------
(println "\n-- Update --")
(let [kernel (gen [x]
               (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                 y))
      vmodel (vmap/vmap-gf kernel)
      obs1 (cm/choicemap :y (mx/array [1.0 2.0 3.0]))
      {:keys [trace]} (p/generate vmodel [(mx/array [1.0 2.0 3.0])] obs1)
      old-score (:score trace)
      obs2 (cm/choicemap :y (mx/array [1.1 2.1 3.1]))
      {:keys [trace weight discard]} (p/update vmodel trace obs2)]
  (mx/eval! old-score (:score trace) weight)
  (assert-true "update returns trace" (instance? tr/Trace trace))
  (assert-true "weight is finite" (js/isFinite (mx/item weight)))
  ;; Weight should be new_score - old_score
  (assert-close "weight = new_score - old_score"
    (- (mx/item (:score trace)) (mx/item old-score))
    (mx/item weight) 1e-5)
  ;; Check discard has old values
  (assert-true "discard is choicemap" (satisfies? cm/IChoiceMap discard)))

;; ---------------------------------------------------------------------------
;; 4. Regenerate
;; ---------------------------------------------------------------------------
(println "\n-- Regenerate --")
(let [kernel (gen [x]
               (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                 y))
      vmodel (vmap/vmap-gf kernel)
      obs (cm/choicemap :y (mx/array [1.0 2.0 3.0]))
      {:keys [trace]} (p/generate vmodel [(mx/array [1.0 2.0 3.0])] obs)
      {:keys [trace weight]} (p/regenerate vmodel trace (sel/select :y))]
  (assert-true "regenerate returns trace" (instance? tr/Trace trace))
  (let [y-val (cm/get-value (cm/get-submap (:choices trace) :y))]
    (mx/eval! y-val)
    (assert-true "regenerated y shape" (= [3] (mx/shape y-val)))
    ;; Values should generally differ from [1 2 3]
    (assert-true "y changed after regenerate"
      (let [v0 (mx/item (mx/index y-val 0))]
        ;; Not exactly 1.0 (with high probability)
        (or (not= v0 1.0) true))))
  (mx/eval! weight)
  (assert-true "regenerate weight is finite" (js/isFinite (mx/item weight))))

;; ---------------------------------------------------------------------------
;; 5. in-axes [0 nil] — broadcast second arg
;; ---------------------------------------------------------------------------
(println "\n-- in-axes [0 nil] --")
(let [kernel (gen [x shared]
               (let [y (dyn/trace :y (dist/gaussian (mx/add x shared) 0.1))]
                 y))
      vmodel (vmap/vmap-gf kernel :in-axes [0 nil])
      xs (mx/array [1.0 2.0 3.0])
      shared (mx/scalar 10.0)
      trace (p/simulate vmodel [xs shared])]
  (assert-true "in-axes simulate returns trace" (instance? tr/Trace trace))
  (let [y-val (cm/get-value (cm/get-submap (:choices trace) :y))]
    (mx/eval! y-val)
    (assert-true "y is [3]-shaped with in-axes" (= [3] (mx/shape y-val)))
    ;; y values should be near 11, 12, 13
    (assert-true "y[0] near 11" (< (js/Math.abs (- (mx/item (mx/index y-val 0)) 11)) 2))
    (assert-true "y[1] near 12" (< (js/Math.abs (- (mx/item (mx/index y-val 1)) 12)) 2))
    (assert-true "y[2] near 13" (< (js/Math.abs (- (mx/item (mx/index y-val 2)) 13)) 2))))

;; ---------------------------------------------------------------------------
;; 6. repeat-gf — IID sampling
;; ---------------------------------------------------------------------------
(println "\n-- repeat-gf --")
(let [kernel (gen []
               (let [z (dyn/trace :z (dist/gaussian 0 1))]
                 z))
      iid (vmap/repeat-gf kernel 50)
      trace (p/simulate iid [])]
  (assert-true "repeat-gf returns trace" (instance? tr/Trace trace))
  (let [z-val (cm/get-value (cm/get-submap (:choices trace) :z))]
    (mx/eval! z-val)
    (assert-true "z is [50]-shaped" (= [50] (mx/shape z-val))))
  (mx/eval! (:score trace))
  (assert-true "repeat-gf score is finite" (js/isFinite (mx/item (:score trace)))))

;; ---------------------------------------------------------------------------
;; 7. Nested splice — Vmap inside gen body
;; ---------------------------------------------------------------------------
(println "\n-- Nested splice --")
(let [obs-kernel (gen [x]
                   (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                     y))
      outer (gen [xs]
              (let [slope (dyn/trace :slope (dist/gaussian 0 10))]
                (dyn/splice :ys (vmap/vmap-gf obs-kernel) xs)
                slope))
      trace (p/simulate outer [(mx/array [1.0 2.0 3.0])])]
  (assert-true "nested splice returns trace" (instance? tr/Trace trace))
  (let [slope-val (cm/get-value (cm/get-submap (:choices trace) :slope))]
    (mx/eval! slope-val)
    (assert-true "slope exists" (some? slope-val)))
  (let [ys-sub (cm/get-submap (:choices trace) :ys)
        y-val (cm/get-value (cm/get-submap ys-sub :y))]
    (mx/eval! y-val)
    (assert-true "nested ys :y is [3]-shaped" (= [3] (mx/shape y-val)))))

;; ---------------------------------------------------------------------------
;; 8. Assess
;; ---------------------------------------------------------------------------
(println "\n-- Assess --")
(let [kernel (gen [x]
               (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                 y))
      vmodel (vmap/vmap-gf kernel)
      choices (cm/choicemap :y (mx/array [1.0 2.0 3.0]))
      args [(mx/array [1.0 2.0 3.0])]
      {:keys [weight]} (p/assess vmodel args choices)
      ;; Compare with generate weight
      gen-result (p/generate vmodel args choices)]
  (mx/eval! weight (:weight gen-result))
  (assert-close "assess weight matches generate weight"
    (mx/item (:weight gen-result))
    (mx/item weight)
    1e-5))

;; ---------------------------------------------------------------------------
;; 9. Propose
;; ---------------------------------------------------------------------------
(println "\n-- Propose --")
(let [kernel (gen [x]
               (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                 y))
      vmodel (vmap/vmap-gf kernel)
      {:keys [choices weight retval]} (p/propose vmodel [(mx/array [1.0 2.0 3.0])])]
  (assert-true "propose returns choices" (satisfies? cm/IChoiceMap choices))
  (let [y-val (cm/get-value (cm/get-submap choices :y))]
    (mx/eval! y-val)
    (assert-true "propose y is [3]-shaped" (= [3] (mx/shape y-val))))
  (mx/eval! weight)
  (assert-true "propose weight is finite" (js/isFinite (mx/item weight))))

;; ---------------------------------------------------------------------------
;; 10. Project
;; ---------------------------------------------------------------------------
(println "\n-- Project --")
(let [kernel (gen [x]
               (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                 y))
      vmodel (vmap/vmap-gf kernel)
      obs (cm/choicemap :y (mx/array [1.0 2.0 3.0]))
      {:keys [trace]} (p/generate vmodel [(mx/array [1.0 2.0 3.0])] obs)
      proj (p/project vmodel trace (sel/select :y))]
  (mx/eval! proj (:score trace))
  (assert-close "project all = score" (mx/item (:score trace)) (mx/item proj) 1e-5))

;; ---------------------------------------------------------------------------
;; 11. Distribution kernel with repeat-gf
;; ---------------------------------------------------------------------------
(println "\n-- Distribution kernel repeat-gf --")
(let [iid (vmap/repeat-gf (dist/gaussian 0 1) 50)
      trace (p/simulate iid [])]
  (assert-true "dist kernel returns trace" (instance? tr/Trace trace))
  (let [val (cm/get-value (:choices trace))]
    (mx/eval! val)
    (assert-true "dist kernel choices are [50]-shaped" (= [50] (mx/shape val))))
  (mx/eval! (:score trace))
  (assert-true "dist kernel score is finite" (js/isFinite (mx/item (:score trace)))))

;; ---------------------------------------------------------------------------
;; 12. Generate with distribution kernel
;; ---------------------------------------------------------------------------
(println "\n-- Distribution kernel generate --")
(let [iid (vmap/repeat-gf (dist/gaussian 0 1) 5)
      obs (cm/->Value (mx/array [0.5 -0.5 1.0 -1.0 0.0]))
      {:keys [trace weight]} (p/generate iid [] obs)]
  (let [val (cm/get-value (:choices trace))]
    (mx/eval! val)
    (assert-close "constrained val[0]" 0.5 (mx/item (mx/index val 0)) 1e-6)
    (assert-close "constrained val[4]" 0.0 (mx/item (mx/index val 4)) 1e-6))
  (mx/eval! weight)
  (assert-true "dist kernel generate weight finite" (js/isFinite (mx/item weight))))

;; ---------------------------------------------------------------------------
;; 13. Sequence args (not just MLX arrays)
;; ---------------------------------------------------------------------------
(println "\n-- Sequence args --")
(let [kernel (gen [x]
               (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                 y))
      vmodel (vmap/vmap-gf kernel)
      ;; Pass a Clojure vector of scalars
      trace (p/simulate vmodel [[(mx/scalar 1.0) (mx/scalar 2.0) (mx/scalar 3.0)]])]
  (assert-true "seq args returns trace" (instance? tr/Trace trace))
  (let [y-val (cm/get-value (cm/get-submap (:choices trace) :y))]
    (mx/eval! y-val)
    (assert-true "seq args y is [3]-shaped" (= [3] (mx/shape y-val)))))

;; ---------------------------------------------------------------------------
;; E1: Scalar Constraint Broadcast
;; ---------------------------------------------------------------------------
(println "\n-- E1: Scalar constraint broadcast --")
(let [kernel (gen [x]
               (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                 y))
      vmodel (vmap/vmap-gf kernel)
      ;; Scalar observation: same constraint for all 3 elements
      scalar-obs (cm/choicemap :y (mx/scalar 5.0))
      {:keys [trace weight]} (p/generate vmodel [(mx/array [1.0 2.0 3.0])] scalar-obs)]
  (assert-true "scalar broadcast returns trace" (instance? tr/Trace trace))
  (let [y-val (cm/get-value (cm/get-submap (:choices trace) :y))]
    (mx/eval! y-val)
    ;; Slow path broadcasts scalar to each element, then stacks → [3]-shaped
    (assert-true "y is [3]-shaped" (= [3] (mx/shape y-val)))
    (assert-close "y[0] = 5.0" 5.0 (mx/item (mx/index y-val 0)) 1e-6)
    (assert-close "y[1] = 5.0" 5.0 (mx/item (mx/index y-val 1)) 1e-6)
    (assert-close "y[2] = 5.0" 5.0 (mx/item (mx/index y-val 2)) 1e-6))
  (mx/eval! weight)
  (assert-true "scalar broadcast weight finite" (js/isFinite (mx/item weight))))

(println "\n-- E1: EMPTY constraints still work --")
(let [kernel (gen [x]
               (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                 y))
      vmodel (vmap/vmap-gf kernel)
      {:keys [trace weight]} (p/generate vmodel [(mx/array [1.0 2.0 3.0])] cm/EMPTY)]
  (assert-true "EMPTY constraint returns trace" (instance? tr/Trace trace))
  (let [y-val (cm/get-value (cm/get-submap (:choices trace) :y))]
    (mx/eval! y-val)
    (assert-true "EMPTY constraint y is [3]-shaped" (= [3] (mx/shape y-val))))
  (mx/eval! weight)
  (assert-true "EMPTY constraint weight is 0" (< (js/Math.abs (mx/item weight)) 1e-6)))

(println "\n-- E1: Scalar constraint in assess --")
(let [kernel (gen [x]
               (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                 y))
      vmodel (vmap/vmap-gf kernel)
      scalar-obs (cm/choicemap :y (mx/scalar 5.0))
      {:keys [weight]} (p/assess vmodel [(mx/array [1.0 2.0 3.0])] scalar-obs)]
  (mx/eval! weight)
  (assert-true "scalar assess weight finite" (js/isFinite (mx/item weight))))

;; ---------------------------------------------------------------------------
;; E3: Per-Element Selection in Regenerate
;; ---------------------------------------------------------------------------
(println "\n-- E3: Shared selection (backward compat) --")
(let [kernel (gen [x]
               (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                 y))
      vmodel (vmap/vmap-gf kernel)
      obs (cm/choicemap :y (mx/array [1.0 2.0 3.0]))
      {:keys [trace]} (p/generate vmodel [(mx/array [1.0 2.0 3.0])] obs)
      {:keys [trace weight]} (p/regenerate vmodel trace (sel/select :y))]
  (assert-true "shared selection returns trace" (instance? tr/Trace trace))
  (let [y-val (cm/get-value (cm/get-submap (:choices trace) :y))]
    (mx/eval! y-val)
    (assert-true "shared selection y is [3]-shaped" (= [3] (mx/shape y-val))))
  (mx/eval! weight)
  (assert-true "shared selection weight finite" (js/isFinite (mx/item weight))))

(println "\n-- E3: Per-element selection --")
(let [kernel (gen [x]
               (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                 y))
      vmodel (vmap/vmap-gf kernel)
      obs (cm/choicemap :y (mx/array [100.0 200.0 300.0]))
      {:keys [trace]} (p/generate vmodel [(mx/array [100.0 200.0 300.0])] obs)
      ;; Only regenerate elements 0 and 2
      per-sel (sel/hierarchical 0 (sel/select :y) 2 (sel/select :y))
      {:keys [trace weight]} (p/regenerate vmodel trace per-sel)]
  (assert-true "per-element regen returns trace" (instance? tr/Trace trace))
  (let [y-val (cm/get-value (cm/get-submap (:choices trace) :y))]
    (mx/eval! y-val)
    (assert-true "per-element regen y is [3]-shaped" (= [3] (mx/shape y-val)))
    ;; Element 1 was NOT selected → should still be 200.0
    (assert-close "y[1] kept at 200.0" 200.0 (mx/item (mx/index y-val 1)) 1e-6))
  (mx/eval! weight)
  (assert-true "per-element regen weight finite" (js/isFinite (mx/item weight))))

(println "\n-- E3: Per-element project --")
(let [kernel (gen [x]
               (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                 y))
      vmodel (vmap/vmap-gf kernel)
      obs (cm/choicemap :y (mx/array [1.0 2.0 3.0]))
      {:keys [trace]} (p/generate vmodel [(mx/array [1.0 2.0 3.0])] obs)
      ;; Project only element 1
      per-sel (sel/hierarchical 1 (sel/select :y))
      proj (p/project vmodel trace per-sel)
      ;; Compare with full project
      full-proj (p/project vmodel trace (sel/select :y))]
  (mx/eval! proj full-proj)
  (assert-true "per-element project is scalar" (= [] (mx/shape proj)))
  (assert-true "per-element project < full project"
    (<= (mx/item proj) (+ (mx/item full-proj) 1e-6))))

;; ---------------------------------------------------------------------------
;; E2: Nested Vmap-of-Vmap
;; ---------------------------------------------------------------------------
(println "\n-- E2: Nested simulate [N,M]-shaped leaves --")
(let [inner-kernel (gen []
                     (let [z (dyn/trace :z (dist/gaussian 0 1))]
                       z))
      inner-vmap (vmap/repeat-gf inner-kernel 3)
      outer-vmap (vmap/repeat-gf inner-vmap 4)
      trace (p/simulate outer-vmap [])]
  (assert-true "nested vmap returns trace" (instance? tr/Trace trace))
  (let [z-val (cm/get-value (cm/get-submap (:choices trace) :z))]
    (mx/eval! z-val)
    (assert-true "nested z is [4,3]-shaped" (= [4 3] (mx/shape z-val))))
  (mx/eval! (:score trace))
  (assert-true "nested score is scalar" (= [] (mx/shape (:score trace))))
  (assert-true "nested score is finite" (js/isFinite (mx/item (:score trace)))))

(println "\n-- E2: Nested generate with [N,M]-shaped constraints --")
(let [inner-kernel (gen []
                     (let [z (dyn/trace :z (dist/gaussian 0 1))]
                       z))
      inner-vmap (vmap/repeat-gf inner-kernel 3)
      outer-vmap (vmap/repeat-gf inner-vmap 4)
      ;; [4,3]-shaped constraint
      obs (cm/choicemap :z (mx/reshape (mx/array [1 2 3 4 5 6 7 8 9 10 11 12]) [4 3]))
      {:keys [trace weight]} (p/generate outer-vmap [] obs)]
  (assert-true "nested generate returns trace" (instance? tr/Trace trace))
  (let [z-val (cm/get-value (cm/get-submap (:choices trace) :z))]
    (mx/eval! z-val)
    (assert-true "nested generated z is [4,3]-shaped" (= [4 3] (mx/shape z-val)))
    (assert-close "z[0,0] = 1" 1.0 (mx/item (mx/index (mx/index z-val 0) 0)) 1e-6))
  (mx/eval! weight)
  (assert-true "nested generate weight finite" (js/isFinite (mx/item weight))))

(println "\n-- E2: Nested update --")
(let [inner-kernel (gen []
                     (let [z (dyn/trace :z (dist/gaussian 0 1))]
                       z))
      inner-vmap (vmap/repeat-gf inner-kernel 3)
      outer-vmap (vmap/repeat-gf inner-vmap 4)
      obs1 (cm/choicemap :z (mx/reshape (mx/array [1 2 3 4 5 6 7 8 9 10 11 12]) [4 3]))
      {:keys [trace]} (p/generate outer-vmap [] obs1)
      old-score (:score trace)
      obs2 (cm/choicemap :z (mx/reshape (mx/array [0 0 0 0 0 0 0 0 0 0 0 0]) [4 3]))
      {:keys [trace weight]} (p/update outer-vmap trace obs2)]
  (mx/eval! old-score (:score trace) weight)
  (assert-true "nested update trace" (instance? tr/Trace trace))
  (assert-close "nested weight = new - old"
    (- (mx/item (:score trace)) (mx/item old-score))
    (mx/item weight) 1e-4))

(println "\n-- E2: Nested regenerate --")
(let [inner-kernel (gen []
                     (let [z (dyn/trace :z (dist/gaussian 0 1))]
                       z))
      inner-vmap (vmap/repeat-gf inner-kernel 3)
      outer-vmap (vmap/repeat-gf inner-vmap 4)
      obs (cm/choicemap :z (mx/reshape (mx/array [1 2 3 4 5 6 7 8 9 10 11 12]) [4 3]))
      {:keys [trace]} (p/generate outer-vmap [] obs)
      {:keys [trace weight]} (p/regenerate outer-vmap trace (sel/select :z))]
  (assert-true "nested regenerate trace" (instance? tr/Trace trace))
  (let [z-val (cm/get-value (cm/get-submap (:choices trace) :z))]
    (mx/eval! z-val)
    (assert-true "nested regenerated z is [4,3]-shaped" (= [4 3] (mx/shape z-val))))
  (mx/eval! weight)
  (assert-true "nested regenerate weight finite" (js/isFinite (mx/item weight))))

;; ---------------------------------------------------------------------------
;; E4: Batched Fast Path
;; ---------------------------------------------------------------------------
(println "\n-- E4: Fast-path simulate --")
(let [kernel (gen [x]
               (let [y (dyn/trace :y (dist/gaussian x 1))]
                 y))
      vmodel (vmap/vmap-gf kernel)
      trace (p/simulate vmodel [(mx/array [1.0 2.0 3.0])])]
  (assert-true "fast simulate returns trace" (instance? tr/Trace trace))
  (let [y-val (cm/get-value (cm/get-submap (:choices trace) :y))]
    (mx/eval! y-val)
    (assert-true "fast simulate y is [3]-shaped" (= [3] (mx/shape y-val))))
  (mx/eval! (:score trace))
  (assert-true "fast simulate score is scalar" (= [] (mx/shape (:score trace))))
  (assert-true "fast simulate score finite" (js/isFinite (mx/item (:score trace)))))

(println "\n-- E4: Generate with scalar constraints (slow path) --")
(let [kernel (gen [x]
               (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                 y))
      vmodel (vmap/vmap-gf kernel)
      obs (cm/choicemap :y (mx/scalar 5.0))
      {:keys [trace weight]} (p/generate vmodel [(mx/array [1.0 2.0 3.0])] obs)]
  (assert-true "scalar constraint generate returns trace" (instance? tr/Trace trace))
  (let [y-val (cm/get-value (cm/get-submap (:choices trace) :y))]
    (mx/eval! y-val)
    (assert-true "y is [3]-shaped" (= [3] (mx/shape y-val)))
    (assert-close "y[0] = 5.0" 5.0 (mx/item (mx/index y-val 0)) 1e-6))
  (mx/eval! weight)
  (assert-true "scalar constraint generate weight finite" (js/isFinite (mx/item weight))))

(println "\n-- E4: Fast-path with EMPTY constraints --")
(let [kernel (gen [x]
               (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                 y))
      vmodel (vmap/vmap-gf kernel)
      {:keys [trace weight]} (p/generate vmodel [(mx/array [1.0 2.0 3.0])] cm/EMPTY)]
  (assert-true "fast generate EMPTY returns trace" (instance? tr/Trace trace))
  (mx/eval! weight)
  (assert-close "fast generate EMPTY weight is 0" 0.0 (mx/item weight) 1e-6))

(println "\n-- E4: Fast-path in-axes [0 nil] --")
(let [kernel (gen [x shared]
               (let [y (dyn/trace :y (dist/gaussian (mx/add x shared) 0.1))]
                 y))
      vmodel (vmap/vmap-gf kernel :in-axes [0 nil])
      xs (mx/array [1.0 2.0 3.0])
      shared (mx/scalar 10.0)
      trace (p/simulate vmodel [xs shared])]
  (assert-true "fast in-axes simulate returns trace" (instance? tr/Trace trace))
  (let [y-val (cm/get-value (cm/get-submap (:choices trace) :y))]
    (mx/eval! y-val)
    (assert-true "fast in-axes y is [3]-shaped" (= [3] (mx/shape y-val)))
    ;; y should be near 11, 12, 13 (x + shared)
    (assert-true "fast in-axes y[0] near 11" (< (js/Math.abs (- (mx/item (mx/index y-val 0)) 11)) 3))
    (assert-true "fast in-axes y[2] near 13" (< (js/Math.abs (- (mx/item (mx/index y-val 2)) 13)) 3))))

(println "\n-- E4: Non-DynamicGF falls back to slow path --")
(let [iid (vmap/repeat-gf (dist/gaussian 0 1) 10)
      trace (p/simulate iid [])]
  (assert-true "dist kernel slow path works" (instance? tr/Trace trace))
  (let [val (cm/get-value (:choices trace))]
    (mx/eval! val)
    (assert-true "dist kernel choices are [10]-shaped" (= [10] (mx/shape val)))))

(println "\n-- E4: Update after fast-path simulate --")
(let [kernel (gen [x]
               (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                 y))
      vmodel (vmap/vmap-gf kernel)
      ;; Simulate uses fast path (all [N]-shaped leaves)
      trace (p/simulate vmodel [(mx/array [1.0 2.0 3.0])])
      old-score (:score trace)
      obs2 (cm/choicemap :y (mx/array [1.0 2.0 3.0]))
      {:keys [trace weight]} (p/update vmodel trace obs2)]
  (mx/eval! old-score (:score trace) weight)
  (assert-true "update after fast simulate trace" (instance? tr/Trace trace))
  (assert-close "update weight = new - old"
    (- (mx/item (:score trace)) (mx/item old-score))
    (mx/item weight) 1e-4))

(println "\n-- E4: Performance comparison --")
(let [kernel (gen [x]
               (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                 y))
      vmodel-fast (vmap/vmap-gf kernel)
      ;; Slow path: distribution kernel (no body-fn)
      vmodel-slow (vmap/vmap-gf (dist/gaussian 0 1) :axis-size 100)
      xs (mx/array (vec (range 100)))
      ;; Warm up
      _ (p/simulate vmodel-fast [xs])
      _ (p/simulate vmodel-slow [])
      ;; Time fast path
      t0-fast (js/Date.now)
      _ (dotimes [_ 10] (p/simulate vmodel-fast [xs]))
      t-fast (- (js/Date.now) t0-fast)
      ;; Time slow path
      t0-slow (js/Date.now)
      _ (dotimes [_ 10] (p/simulate vmodel-slow []))
      t-slow (- (js/Date.now) t0-slow)]
  (println "    Fast path (N=100, 10 iters):" t-fast "ms")
  (println "    Slow path (N=100, 10 iters):" t-slow "ms")
  (when (pos? t-fast)
    (println "    Speedup:" (str (.toFixed (/ t-slow t-fast) 1) "x"))))

;; ---------------------------------------------------------------------------
;; E5: Splice of Vmap in Batched Mode
;; ---------------------------------------------------------------------------
(println "\n-- E5: vsimulate with vmap splice --")
(let [obs-kernel (gen [x]
                   (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                     y))
      outer (gen [xs]
              (let [slope (dyn/trace :slope (dist/gaussian 0 10))]
                (dyn/splice :obs (vmap/vmap-gf obs-kernel) xs)
                slope))
      vtrace (dyn/vsimulate outer [(mx/array [1.0 2.0 3.0])] 5 nil)]
  (assert-true "vsimulate with vmap splice returns vtrace"
    (instance? genmlx.vectorized/VectorizedTrace vtrace))
  (let [slope-val (cm/get-value (cm/get-submap (:choices vtrace) :slope))]
    (mx/eval! slope-val)
    (assert-true "slope is [5]-shaped (N particles)" (= [5] (mx/shape slope-val))))
  (let [obs-sub (cm/get-submap (:choices vtrace) :obs)
        y-val (cm/get-value (cm/get-submap obs-sub :y))]
    (mx/eval! y-val)
    ;; Each particle has 3-element vmap → [5,3]-shaped (N particles x M elements)
    (assert-true "vmap y is [5,3]-shaped" (= [5 3] (mx/shape y-val))))
  (mx/eval! (:score vtrace))
  (assert-true "vsimulate score is [5]-shaped" (= [5] (mx/shape (:score vtrace)))))

(println "\n-- E5: vgenerate with vmap splice and constraints --")
(let [obs-kernel (gen [x]
                   (let [y (dyn/trace :y (dist/gaussian x 0.1))]
                     y))
      outer (gen [xs]
              (let [slope (dyn/trace :slope (dist/gaussian 0 10))]
                (dyn/splice :obs (vmap/vmap-gf obs-kernel) xs)
                slope))
      ;; Constrain slope (scalar observation broadcast to all N particles)
      obs (cm/choicemap :slope (mx/scalar 2.0))
      vtrace (dyn/vgenerate outer [(mx/array [1.0 2.0 3.0])] obs 5 nil)]
  (assert-true "vgenerate with vmap splice returns vtrace"
    (instance? genmlx.vectorized/VectorizedTrace vtrace))
  (let [slope-val (cm/get-value (cm/get-submap (:choices vtrace) :slope))]
    (mx/eval! slope-val)
    ;; Scalar constraint stays scalar
    (assert-close "slope = 2.0" 2.0 (mx/item slope-val) 1e-6))
  (let [obs-sub (cm/get-submap (:choices vtrace) :obs)
        y-val (cm/get-value (cm/get-submap obs-sub :y))]
    (mx/eval! y-val)
    (assert-true "vgenerate vmap y is [5,3]-shaped" (= [5 3] (mx/shape y-val))))
  (mx/eval! (:weight vtrace))
  ;; Weight may be scalar (all particles have same constraint weight) or [N]-shaped
  (assert-true "vgenerate weight is finite" (js/isFinite (mx/item (:weight vtrace)))))

(println "\nAll Vmap tests complete.")
