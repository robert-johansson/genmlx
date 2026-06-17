;; @tier fast
(ns genmlx.update-discard-branch-test
  "Test that update includes deleted addresses in discard when model switches branches.
   Gen.jl semantics: discard must contain all old addresses not in new trace."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.trace :as tr]
            [genmlx.gen :refer [gen]]))

;; -------------------------------------------------------------------------
;; Branch model: coin flip -> heads or tails branch
;; -------------------------------------------------------------------------

(def branch-model
  (gen [_]
    (let [coin (trace :coin (dist/bernoulli 0.5))]
      (if (pos? (mx/item coin))
        (trace :heads (dist/gaussian 0 1))
        (trace :tails (dist/gaussian 0 1))))))

(def branch-model-k (dyn/auto-key branch-model))

(deftest heads-to-tails-switch
  (testing "Heads -> Tails branch switch"
    (let [heads-trace (:trace (p/generate branch-model-k [nil]
                                           (cm/choicemap :coin (mx/scalar 1.0)
                                                         :heads (mx/scalar 2.5))))
          heads-choices (:choices heads-trace)]
      (is (cm/has-value? (cm/get-submap heads-choices :coin)) "heads trace has :coin")
      (is (cm/has-value? (cm/get-submap heads-choices :heads)) "heads trace has :heads")
      (is (not (cm/has-value? (cm/get-submap heads-choices :tails))) "heads trace does NOT have :tails")

      (let [update-result (p/update branch-model-k heads-trace
                                    (cm/choicemap :coin (mx/scalar 0.0)
                                                  :tails (mx/scalar 3.7)))
            new-trace (:trace update-result)
            discard (:discard update-result)
            new-choices (:choices new-trace)]
        (is (cm/has-value? (cm/get-submap new-choices :coin)) "new trace has :coin")
        (is (cm/has-value? (cm/get-submap new-choices :tails)) "new trace has :tails")
        (is (not (cm/has-value? (cm/get-submap new-choices :heads))) "new trace does NOT have :heads")
        (is (cm/has-value? (cm/get-submap discard :coin)) "discard has :coin")
        (is (h/close? 1.0 (mx/item (cm/get-value (cm/get-submap discard :coin))) 1e-6) "discard :coin value is 1.0")
        (is (cm/has-value? (cm/get-submap discard :heads)) "discard has :heads (deleted address)")
        (is (h/close? 2.5 (mx/item (cm/get-value (cm/get-submap discard :heads))) 1e-6) "discard :heads value is 2.5")
        (is (not (cm/has-value? (cm/get-submap discard :tails))) "discard does NOT have :tails")))))

(deftest tails-to-heads-switch
  (testing "Tails -> Heads branch switch"
    (let [tails-trace (:trace (p/generate branch-model-k [nil]
                                           (cm/choicemap :coin (mx/scalar 0.0)
                                                         :tails (mx/scalar -1.3))))
          update-result (p/update branch-model-k tails-trace
                                  (cm/choicemap :coin (mx/scalar 1.0)
                                                :heads (mx/scalar 4.2)))
          discard (:discard update-result)
          new-choices (:choices (:trace update-result))]
      (is (cm/has-value? (cm/get-submap new-choices :heads)) "new trace has :heads")
      (is (not (cm/has-value? (cm/get-submap new-choices :tails))) "new trace does NOT have :tails")
      (is (cm/has-value? (cm/get-submap discard :coin)) "discard has :coin")
      (is (cm/has-value? (cm/get-submap discard :tails)) "discard has :tails (deleted address)")
      (is (h/close? -1.3 (mx/item (cm/get-value (cm/get-submap discard :tails))) 1e-6) "discard :tails value is -1.3"))))

(deftest same-branch-no-switch
  (testing "Same branch, no switch"
    (let [heads-trace (:trace (p/generate branch-model-k [nil]
                                           (cm/choicemap :coin (mx/scalar 1.0)
                                                         :heads (mx/scalar 2.5))))
          update-result (p/update branch-model-k heads-trace
                                  (cm/choicemap :heads (mx/scalar 9.9)))
          discard (:discard update-result)
          new-choices (:choices (:trace update-result))]
      (is (cm/has-value? (cm/get-submap new-choices :coin)) "new trace still has :coin")
      (is (cm/has-value? (cm/get-submap new-choices :heads)) "new trace has :heads with new value")
      (is (h/close? 9.9 (mx/item (cm/get-value (cm/get-submap new-choices :heads))) 1e-6) ":heads updated to 9.9")
      (is (cm/has-value? (cm/get-submap discard :heads)) "discard has :heads (value changed)")
      (is (h/close? 2.5 (mx/item (cm/get-value (cm/get-submap discard :heads))) 1e-6) "discard :heads old value 2.5")
      (is (not (cm/has-value? (cm/get-submap discard :tails))) "discard does NOT have :tails (never existed)"))))

(deftest round-trip-recovery
  (testing "Round-trip — update with discard recovers original trace"
    (let [heads-trace (:trace (p/generate branch-model-k [nil]
                                           (cm/choicemap :coin (mx/scalar 1.0)
                                                         :heads (mx/scalar 2.5))))
          old-score (:score heads-trace)
          fwd (p/update branch-model-k heads-trace
                         (cm/choicemap :coin (mx/scalar 0.0)
                                       :tails (mx/scalar 3.7)))
          tails-trace (:trace fwd)
          fwd-discard (:discard fwd)
          rev (p/update branch-model-k tails-trace fwd-discard)
          recovered-trace (:trace rev)
          recovered-choices (:choices recovered-trace)]
      (is (cm/has-value? (cm/get-submap recovered-choices :coin)) "recovered trace has :coin")
      (is (h/close? 1.0 (mx/item (cm/get-value (cm/get-submap recovered-choices :coin))) 1e-6) "recovered :coin = 1.0")
      (is (cm/has-value? (cm/get-submap recovered-choices :heads)) "recovered trace has :heads")
      (is (h/close? 2.5 (mx/item (cm/get-value (cm/get-submap recovered-choices :heads))) 1e-6) "recovered :heads = 2.5")
      (is (not (cm/has-value? (cm/get-submap recovered-choices :tails))) "recovered trace does NOT have :tails")
      (is (h/close? (mx/item old-score) (mx/item (:score recovered-trace)) 1e-4) "recovered score matches original"))))

(deftest weight-symmetry
  (testing "fwd_weight + rev_weight ~ 0"
    (let [heads-trace (:trace (p/generate branch-model-k [nil]
                                           (cm/choicemap :coin (mx/scalar 1.0)
                                                         :heads (mx/scalar 2.5))))
          fwd (p/update branch-model-k heads-trace
                         (cm/choicemap :coin (mx/scalar 0.0)
                                       :tails (mx/scalar 3.7)))
          rev (p/update branch-model-k (:trace fwd) (:discard fwd))
          total (+ (mx/item (:weight fwd)) (mx/item (:weight rev)))]
      (is (h/close? 0.0 total 1e-4) "fwd_weight + rev_weight ~ 0"))))

;; -------------------------------------------------------------------------
;; Batched analogue (genmlx-6v3h): vupdate*/vupdate-args must add deleted old
;; addresses to the discard on a structure-shrinking move, like the scalar path.
;; The batch branches on a HOST ARG (per-particle branching / mx/item is not
;; allowed in a vectorized body), so old args visit :a and new args visit :b.
;; -------------------------------------------------------------------------

(def arg-branch-model
  (gen [flag]
    (if flag
      (trace :a (dist/gaussian 0 1))
      (trace :b (dist/gaussian 5 1)))))

(defn- leaf-vec [choices addr]
  (mx/->clj (cm/get-value (cm/get-submap choices addr))))

(deftest batched-arg-branch-discard
  (testing "structure-shrinking vupdate-args: deleted :a appears in the discard with its [N] values"
    (let [n 4
          vt (dyn/vsimulate arg-branch-model [true] n (rng/fresh-key 1))
          r  (dyn/vupdate-args arg-branch-model vt [false] cm/EMPTY (rng/fresh-key 2))
          discard (:discard r)
          new-choices (:choices (:vtrace r))]
      (is (cm/has-value? (cm/get-submap (:choices vt) :a)) "old trace has :a")
      (is (cm/has-value? (cm/get-submap new-choices :b)) "new trace has :b")
      (is (not (cm/has-value? (cm/get-submap new-choices :a))) "new trace dropped :a")
      (is (cm/has-value? (cm/get-submap discard :a)) "discard includes the deleted :a")
      (is (= [n] (mx/shape (cm/get-value (cm/get-submap discard :a)))) "discarded :a is [N]-shaped")
      (is (= (leaf-vec (:choices vt) :a) (leaf-vec discard :a))
          "discarded :a values == the original per-particle :a values (round-trip recoverable)")
      (is (= [n] (mx/shape (:weight r))) "weight is [N]-shaped")
      (is (every? js/isFinite (mx/->clj (:weight r))) "weight is finite (unaffected by the discard fix)")))
  (testing "non-shrinking overwrite vupdate: discard carries the overwritten address only"
    (let [n 4
          plain (gen [] (trace :x (dist/gaussian 0 1)))
          vt (dyn/vsimulate plain [] n (rng/fresh-key 3))
          r  (dyn/vupdate plain vt (cm/choicemap :x (mx/zeros [n])) (rng/fresh-key 4))]
      (is (cm/has-value? (cm/get-submap (:discard r) :x)) "overwritten :x is in the discard")
      (is (= '(:x) (keys (:m (:discard r)))) "no spurious deleted addresses added"))))

(cljs.test/run-tests)
