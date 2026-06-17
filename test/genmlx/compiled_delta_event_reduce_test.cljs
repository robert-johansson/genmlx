;; @tier fast core
(ns genmlx.compiled-delta-event-reduce-test
  "genmlx-kftc: the COMPILED noise-transform delta :log-prob (compiled.cljs)
   compared value to point mass elementwise and returned a [T] mask with NO
   event-axis reduction, while the handler delta (dist.cljs, genmlx-exw9)
   reduces the trailing event axes to a JOINT scalar (0 iff ALL elements match,
   else -Inf). For a vector point mass (dist/delta <tensor>) — a legal,
   compilable static site — the compiled path therefore diverged from the
   handler ground truth two ways:

     * MATCH:    weight broadcast to [T] instead of a joint scalar.
     * NO-MATCH: components that happen to match read FINITE while the joint
                 must be -Inf (e.g. compiled [-0.9, -0.9, -Inf] vs handler -Inf).

   The compiled path is OPTIMIZATION; the handler is GROUND TRUTH. These tests
   pin compiled == handler (value AND shape) for vector deltas across
   assess/generate/project, and that the scalar delta path is unchanged. Each
   model is asserted to actually reach :L1-M2 so the comparison is not vacuous
   (the compiled path must really be engaged)."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.gfi :as gfi]
            [genmlx.inspect :as insp])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn- num [x] (mx/eval! x) (mx/item x))
(defn- shp [x] (mx/shape x))
(defn- close? [a b] (or (and (not (js/isFinite a)) (not (js/isFinite b)) (= (< a 0) (< b 0)))
                        (< (js/Math.abs (- a b)) 1e-4)))

;; A static, L1-M2-compilable model with a VECTOR point-mass site (:d) and an
;; ordinary gaussian (:a) so the delta log-prob is summed into a real scalar
;; accumulator that the [T] mask would otherwise broadcast.
(def point (mx/array [1.0 2.0 3.0]))
(def vec-model0 (gen [] (let [a (trace :a (dist/gaussian 0 1))]
                          (trace :d (dist/delta point))
                          a)))
(def vec-model (dyn/auto-key vec-model0))

(def scalar-model0 (gen [] (let [a (trace :a (dist/gaussian 0 1))]
                             (trace :d (dist/delta (mx/scalar 5.0)))
                             a)))
(def scalar-model (dyn/auto-key scalar-model0))

(deftest models-actually-compile
  ;; Guard against a vacuous test: if these did not reach :L1-M2 the compiled
  ;; and handler paths would be identical and the regression would be untested.
  (is (= :L1-M2 (:compilation (insp/inspect vec-model0))) "vector-delta model is L1-M2")
  (is (= :L1-M2 (:compilation (insp/inspect scalar-model0))) "scalar-delta model is L1-M2"))

(deftest assess-vector-delta-matches-handler
  (testing "matching vector delta: compiled weight == handler (joint scalar)"
    (let [cmatch (cm/choicemap :a (mx/scalar 0.0) :d (mx/array [1.0 2.0 3.0]))
          rc (p/assess vec-model [] cmatch)
          rh (p/assess (gfi/strip-compiled vec-model) [] cmatch)]
      (is (= [] (shp (:weight rc))) "compiled weight is a joint scalar, not [T]")
      (is (= (shp (:weight rh)) (shp (:weight rc))) "compiled shape == handler shape")
      (is (close? (num (:weight rc)) (num (:weight rh))) "compiled value == handler value")))
  (testing "NON-matching vector delta: compiled is joint -Inf, not a partial finite mask"
    (let [cnomatch (cm/choicemap :a (mx/scalar 0.0) :d (mx/array [1.0 2.0 9.0]))
          rc (p/assess vec-model [] cnomatch)
          rh (p/assess (gfi/strip-compiled vec-model) [] cnomatch)]
      (is (= [] (shp (:weight rc))) "compiled weight is a joint scalar, not [T]")
      (is (not (js/isFinite (num (:weight rc)))) "compiled joint is -Inf (one component mismatched)")
      (is (close? (num (:weight rc)) (num (:weight rh))) "compiled -Inf == handler -Inf"))))

(deftest generate-vector-delta-matches-handler
  (testing "fully-constrained generate weight == handler, scalar, both cases"
    (doseq [[label dv finite?] [["match"   (mx/array [1.0 2.0 3.0]) true]
                                ["nomatch" (mx/array [1.0 9.0 3.0]) false]]]
      (let [c (cm/choicemap :a (mx/scalar 0.25) :d dv)
            rc (p/generate vec-model [] c)
            rh (p/generate (gfi/strip-compiled vec-model) [] c)]
        (is (= [] (shp (:weight rc))) (str label ": compiled generate weight scalar"))
        (is (= (shp (:weight rh)) (shp (:weight rc))) (str label ": compiled shape == handler shape"))
        (is (= finite? (js/isFinite (num (:weight rc)))) (str label ": finiteness matches expectation"))
        (is (close? (num (:weight rc)) (num (:weight rh))) (str label ": compiled value == handler value"))))))

(deftest project-vector-delta-matches-handler
  (testing "project on :d (delta value == point, always matches) is a joint scalar 0"
    (let [tr  (p/simulate vec-model [])
          trh (p/simulate (gfi/strip-compiled vec-model) [])
          pc  (p/project vec-model tr (sel/select :d))
          ph  (p/project (gfi/strip-compiled vec-model) trh (sel/select :d))]
      (is (= [] (shp pc)) "compiled project of vector delta is a joint scalar, not [T]")
      (is (= (shp ph) (shp pc)) "compiled project shape == handler project shape")
      (is (close? (num pc) 0.0) "matching delta projects to 0")
      (is (close? (num pc) (num ph)) "compiled project == handler project"))))

(deftest scalar-delta-unchanged
  (testing "scalar point mass: compiled == handler, value and shape, both cases"
    (doseq [[label dv finite?] [["match"   (mx/scalar 5.0) true]
                                ["nomatch" (mx/scalar 6.0) false]]]
      (let [c (cm/choicemap :a (mx/scalar 0.0) :d dv)
            rc (p/assess scalar-model [] c)
            rh (p/assess (gfi/strip-compiled scalar-model) [] c)]
        (is (= (shp (:weight rh)) (shp (:weight rc))) (str label ": scalar shape unchanged"))
        (is (= finite? (js/isFinite (num (:weight rc)))) (str label ": scalar finiteness expected"))
        (is (close? (num (:weight rc)) (num (:weight rh))) (str label ": scalar value == handler"))))))

(cljs.test/run-tests)
