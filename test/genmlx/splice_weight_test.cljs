;; @tier fast core
(ns genmlx.splice-weight-test
  "Weight composition through splices and combinator descent (genmlx-ovop).

   Thesis/Gen.jl semantics: a child GFI call returns its FINAL weight
   (for regenerate, the MH weight w_child = dS_child - pr_child). The
   parent regenerate handler accumulates a proposal-ratio correction in
   :weight and computes W = (S' - S) - correction at the end. Merging a
   child's final weight directly into the correction accumulator is a
   convention mix-up; the correct contribution is dS_child - w_child.

   Canonical counterexample: x ~ N(0,10) at :x, spliced child
   z ~ N(x,1). Regenerating :z draws from its conditional prior with
   unchanged parents, so the exact weight is 0 — same as the flat
   (handler-derived) equivalent model."
  (:require-macros [genmlx.gen :refer [gen]])
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.combinators :as comb]
            [genmlx.vmap :as vmap]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]))

(def sub-gf
  (gen [mu] (trace :z (dist/gaussian mu 1))))

(def spliced-model
  (gen []
    (let [x (trace :x (dist/gaussian 0 10))]
      (splice :sub sub-gf x))))

(def flat-model
  (gen []
    (let [x (trace :x (dist/gaussian 0 10))]
      (trace :z (dist/gaussian x 1)))))

(defn- regen-weight [model seed selection]
  (let [t (p/simulate (dyn/with-key model (rng/fresh-key seed)) [])
        {:keys [weight]} (p/regenerate (dyn/with-key model (rng/fresh-key (inc seed)))
                                       t selection)]
    (h/realize weight)))

(deftest regen-through-splice-prior-resample-weight-zero
  (testing "flat reference: regenerating a leaf from its conditional prior => weight 0"
    (is (h/close? 0.0 (regen-weight flat-model 11 (sel/select :z)) 1e-3)))
  (testing "spliced: same model through a splice must also give weight 0"
    (is (h/close? 0.0 (regen-weight spliced-model 11
                                    (sel/hierarchical :sub (sel/select :z)))
                  1e-3))))

(def nested-sub-gf
  (gen [mu] (splice :inner sub-gf mu)))

(def nested-spliced-model
  (gen []
    (let [x (trace :x (dist/gaussian 0 10))]
      (splice :sub nested-sub-gf x))))

(deftest regen-through-nested-splice-weight-zero
  (testing "two-level splice: prior resample of leaf => weight 0"
    (is (h/close? 0.0 (regen-weight nested-spliced-model 17
                                    (sel/hierarchical
                                     :sub (sel/hierarchical
                                           :inner (sel/select :z))))
                  1e-3))))

(def spliced-model-downstream
  (gen []
    (let [x (trace :x (dist/gaussian 0 10))
          z (splice :sub sub-gf x)]
      (trace :y (dist/gaussian z 1)))))

(deftest regen-through-splice-downstream-dependency
  (testing "downstream :y depends on spliced :z => weight = lp_y(y;z') - lp_y(y;z)"
    (let [t (p/simulate (dyn/with-key spliced-model-downstream (rng/fresh-key 33)) [])
          old-z (cm/get-choice (:choices t) [:sub :z])
          y (cm/get-choice (:choices t) [:y])
          {:keys [trace weight]} (p/regenerate
                                  (dyn/with-key spliced-model-downstream (rng/fresh-key 34))
                                  t (sel/hierarchical :sub (sel/select :z)))
          new-z (cm/get-choice (:choices trace) [:sub :z])
          expected (- (h/realize (dc/dist-log-prob (dist/gaussian new-z 1) y))
                      (h/realize (dc/dist-log-prob (dist/gaussian old-z 1) y)))]
      (is (h/close? expected (h/realize weight) 1e-3)
          "weight captures only the retained downstream lp change"))))

(deftest project-through-splice-convention
  (testing "project through a splice = lp of the selected child site"
    (let [t (p/simulate (dyn/with-key spliced-model (rng/fresh-key 55)) [])
          x (cm/get-choice (:choices t) [:x])
          z (cm/get-choice (:choices t) [:sub :z])
          proj (p/project (dyn/with-key spliced-model (rng/fresh-key 56))
                          t (sel/hierarchical :sub (sel/select :z)))
          expected (h/realize (dc/dist-log-prob (dist/gaussian x 1) z))]
      (is (h/close? expected (h/realize proj) 1e-3)))))

(deftest update-through-splice-fixed-structure
  (testing "constraining the spliced child :z: weight = full lp delta (no fresh choices)"
    (let [t (p/simulate (dyn/with-key spliced-model-downstream (rng/fresh-key 77)) [])
          x (cm/get-choice (:choices t) [:x])
          old-z (cm/get-choice (:choices t) [:sub :z])
          y (cm/get-choice (:choices t) [:y])
          new-z (mx/scalar 0.25)
          constraints (cm/set-choice cm/EMPTY [:sub :z] new-z)
          {:keys [weight]} (p/update (dyn/with-key spliced-model-downstream (rng/fresh-key 78))
                                     t constraints)
          lp (fn [d v] (h/realize (dc/dist-log-prob d v)))
          expected (+ (- (lp (dist/gaussian x 1) new-z)
                         (lp (dist/gaussian x 1) old-z))
                      (- (lp (dist/gaussian new-z 1) y)
                         (lp (dist/gaussian old-z 1) y)))]
      (is (h/close? expected (h/realize weight) 1e-3)))))

;; ---------------------------------------------------------------------------
;; Vmap combinator regenerate (same convention-mix class)
;; ---------------------------------------------------------------------------

(def two-site-kernel
  (gen []
    (let [a (trace :a (dist/gaussian 0 1))]
      (trace :b (dist/gaussian a 1)))))

(deftest vmap-regenerate-weight-composition
  (testing "vmap regenerate: weight = sum of per-element MH weights"
    (let [vgf (vmap/vmap-gf (dyn/auto-key two-site-kernel) :axis-size 3)
          t (p/simulate vgf [])
          {:keys [trace weight]} (p/regenerate vgf t (sel/select :a))
          lp (fn [mu v] (h/realize (dc/dist-log-prob (dist/gaussian mu 1) v)))
          ;; vmap traces hold stacked [N]-shaped leaves at the kernel addresses
          old-as (cm/get-choice (:choices t) [:a])
          new-as (cm/get-choice (:choices trace) [:a])
          bs (cm/get-choice (:choices t) [:b])
          ;; expected: per element, retained :b re-scored under resampled :a
          expected (reduce + (for [i (range 3)]
                               (let [old-a (mx/index old-as i)
                                     new-a (mx/index new-as i)
                                     b (mx/index bs i)]
                                 (- (lp new-a b) (lp old-a b)))))]
      (is (h/close? expected (h/realize weight) 1e-3)))))

;; ---------------------------------------------------------------------------
;; Batched (vectorized) splice paths
;; ---------------------------------------------------------------------------

(deftest vregenerate-through-spliced-dynamic-gf
  (testing "batched regen through spliced DynamicGF: prior resample => all-zero weights"
    (let [n 8
          vt (dyn/vsimulate spliced-model [] n (rng/fresh-key 101))
          {:keys [weight]} (dyn/vregenerate spliced-model vt
                                            (sel/hierarchical :sub (sel/select :z))
                                            (rng/fresh-key 102))]
      (mx/eval! weight)
      (is (= [n] (mx/shape weight)))
      (is (h/close? 0.0 (mx/realize (mx/amax (mx/abs weight))) 1e-3)
          "every particle weight is 0"))))

(def map-spliced-model
  ;; Splices a Map combinator (non-DynamicGF) so batched execution takes
  ;; the combinator-batched-fallback path.
  (let [mgf (comb/map-combinator (dyn/auto-key sub-gf))]
    (gen []
      (splice :m mgf [(mx/scalar 0.0) (mx/scalar 0.0)]))))

(deftest vregenerate-through-spliced-combinator-fallback
  (testing "combinator-batched-fallback regen: prior resample => all-zero weights"
    (let [n 4
          vt (dyn/vsimulate map-spliced-model [] n (rng/fresh-key 111))
          {:keys [weight]} (dyn/vregenerate map-spliced-model vt
                                            (sel/hierarchical :m sel/all)
                                            (rng/fresh-key 112))]
      (mx/eval! weight)
      (is (= [n] (mx/shape weight)))
      (is (h/close? 0.0 (mx/realize (mx/amax (mx/abs weight))) 1e-3)
          "every particle weight is 0"))))

(cljs.test/run-tests)
