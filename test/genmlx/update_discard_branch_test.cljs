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

(cljs.test/run-tests)
