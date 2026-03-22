(ns genmlx.update-discard-branch-test
  "Test that update includes deleted addresses in discard when model switches branches.
   Gen.jl semantics: discard must contain all old addresses not in new trace."
  (:require [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.trace :as tr]
            [genmlx.gen :refer [gen]]))

(defn assert-true [desc v]
  (if v
    (println (str "  PASS: " desc))
    (println (str "  FAIL: " desc))))

(defn assert-close [desc expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (println (str "  PASS: " desc " (" actual " ~ " expected ")"))
      (println (str "  FAIL: " desc " expected=" expected " actual=" actual " diff=" diff)))))

;; -------------------------------------------------------------------------
;; Branch model: coin flip → heads or tails branch
;; -------------------------------------------------------------------------

(def branch-model
  (gen [_]
    (let [coin (trace :coin (dist/bernoulli 0.5))]
      (if (pos? (mx/item coin))
        (trace :heads (dist/gaussian 0 1))
        (trace :tails (dist/gaussian 0 1))))))

(def branch-model-k (dyn/auto-key branch-model))

;; -------------------------------------------------------------------------
;; Test 1: Switching from heads to tails — discard must include :heads
;; -------------------------------------------------------------------------

(println "\n-- Test 1: Heads → Tails branch switch --")

(let [;; Generate on the heads branch
      heads-trace (:trace (p/generate branch-model-k [nil]
                                       (cm/choicemap :coin (mx/scalar 1.0)
                                                     :heads (mx/scalar 2.5))))
      ;; Verify heads trace has :coin and :heads
      heads-choices (:choices heads-trace)
      _ (assert-true "heads trace has :coin"
                     (cm/has-value? (cm/get-submap heads-choices :coin)))
      _ (assert-true "heads trace has :heads"
                     (cm/has-value? (cm/get-submap heads-choices :heads)))
      _ (assert-true "heads trace does NOT have :tails"
                     (not (cm/has-value? (cm/get-submap heads-choices :tails))))

      ;; Update: switch to tails branch
      update-result (p/update branch-model-k heads-trace
                              (cm/choicemap :coin (mx/scalar 0.0)
                                            :tails (mx/scalar 3.7)))
      new-trace (:trace update-result)
      discard (:discard update-result)
      new-choices (:choices new-trace)]

  ;; New trace should have :coin and :tails, NOT :heads
  (assert-true "new trace has :coin"
               (cm/has-value? (cm/get-submap new-choices :coin)))
  (assert-true "new trace has :tails"
               (cm/has-value? (cm/get-submap new-choices :tails)))
  (assert-true "new trace does NOT have :heads"
               (not (cm/has-value? (cm/get-submap new-choices :heads))))

  ;; Discard MUST contain :coin (explicitly constrained, old value discarded)
  (assert-true "discard has :coin"
               (cm/has-value? (cm/get-submap discard :coin)))
  (assert-close "discard :coin value is 1.0"
                1.0 (mx/item (cm/get-value (cm/get-submap discard :coin))) 1e-6)

  ;; Discard MUST contain :heads (deleted address — the bug fix)
  (assert-true "discard has :heads (deleted address)"
               (cm/has-value? (cm/get-submap discard :heads)))
  (assert-close "discard :heads value is 2.5"
                2.5 (mx/item (cm/get-value (cm/get-submap discard :heads))) 1e-6)

  ;; Discard should NOT contain :tails (new address, not deleted)
  (assert-true "discard does NOT have :tails"
               (not (cm/has-value? (cm/get-submap discard :tails)))))

;; -------------------------------------------------------------------------
;; Test 2: Switching from tails to heads — discard must include :tails
;; -------------------------------------------------------------------------

(println "\n-- Test 2: Tails → Heads branch switch --")

(let [;; Generate on the tails branch
      tails-trace (:trace (p/generate branch-model-k [nil]
                                       (cm/choicemap :coin (mx/scalar 0.0)
                                                     :tails (mx/scalar -1.3))))
      ;; Update: switch to heads branch
      update-result (p/update branch-model-k tails-trace
                              (cm/choicemap :coin (mx/scalar 1.0)
                                            :heads (mx/scalar 4.2)))
      discard (:discard update-result)
      new-choices (:choices (:trace update-result))]

  (assert-true "new trace has :heads"
               (cm/has-value? (cm/get-submap new-choices :heads)))
  (assert-true "new trace does NOT have :tails"
               (not (cm/has-value? (cm/get-submap new-choices :tails))))
  (assert-true "discard has :coin"
               (cm/has-value? (cm/get-submap discard :coin)))
  (assert-true "discard has :tails (deleted address)"
               (cm/has-value? (cm/get-submap discard :tails)))
  (assert-close "discard :tails value is -1.3"
                -1.3 (mx/item (cm/get-value (cm/get-submap discard :tails))) 1e-6))

;; -------------------------------------------------------------------------
;; Test 3: Same branch (no switch) — discard should NOT contain branch addr
;; -------------------------------------------------------------------------

(println "\n-- Test 3: Same branch, no switch --")

(let [;; Generate on heads branch
      heads-trace (:trace (p/generate branch-model-k [nil]
                                       (cm/choicemap :coin (mx/scalar 1.0)
                                                     :heads (mx/scalar 2.5))))
      ;; Update: stay on heads, change :heads value
      update-result (p/update branch-model-k heads-trace
                              (cm/choicemap :heads (mx/scalar 9.9)))
      discard (:discard update-result)
      new-choices (:choices (:trace update-result))]

  (assert-true "new trace still has :coin"
               (cm/has-value? (cm/get-submap new-choices :coin)))
  (assert-true "new trace has :heads with new value"
               (cm/has-value? (cm/get-submap new-choices :heads)))
  (assert-close ":heads updated to 9.9"
                9.9 (mx/item (cm/get-value (cm/get-submap new-choices :heads))) 1e-6)
  ;; Discard should have old :heads value, NOT :coin (unchanged)
  (assert-true "discard has :heads (value changed)"
               (cm/has-value? (cm/get-submap discard :heads)))
  (assert-close "discard :heads old value 2.5"
                2.5 (mx/item (cm/get-value (cm/get-submap discard :heads))) 1e-6)
  ;; No branch switch, so no deleted addresses
  (assert-true "discard does NOT have :tails (never existed)"
               (not (cm/has-value? (cm/get-submap discard :tails)))))

;; -------------------------------------------------------------------------
;; Test 4: Round-trip — update with discard recovers original trace
;; -------------------------------------------------------------------------

(println "\n-- Test 4: Round-trip recovery --")

(let [;; Start on heads
      heads-trace (:trace (p/generate branch-model-k [nil]
                                       (cm/choicemap :coin (mx/scalar 1.0)
                                                     :heads (mx/scalar 2.5))))
      old-score (:score heads-trace)

      ;; Switch to tails
      fwd (p/update branch-model-k heads-trace
                     (cm/choicemap :coin (mx/scalar 0.0)
                                   :tails (mx/scalar 3.7)))
      tails-trace (:trace fwd)
      fwd-discard (:discard fwd)

      ;; Reverse: use the discard as constraints to switch back
      rev (p/update branch-model-k tails-trace fwd-discard)
      recovered-trace (:trace rev)
      recovered-choices (:choices recovered-trace)]

  ;; Recovered trace should match original
  (assert-true "recovered trace has :coin"
               (cm/has-value? (cm/get-submap recovered-choices :coin)))
  (assert-close "recovered :coin = 1.0"
                1.0 (mx/item (cm/get-value (cm/get-submap recovered-choices :coin))) 1e-6)
  (assert-true "recovered trace has :heads"
               (cm/has-value? (cm/get-submap recovered-choices :heads)))
  (assert-close "recovered :heads = 2.5"
                2.5 (mx/item (cm/get-value (cm/get-submap recovered-choices :heads))) 1e-6)
  (assert-true "recovered trace does NOT have :tails"
               (not (cm/has-value? (cm/get-submap recovered-choices :tails))))
  ;; Score should match original
  (assert-close "recovered score matches original"
                (mx/item old-score) (mx/item (:score recovered-trace)) 1e-4))

;; -------------------------------------------------------------------------
;; Test 5: Weight symmetry — fwd_weight + rev_weight ≈ 0
;; -------------------------------------------------------------------------

(println "\n-- Test 5: Weight symmetry --")

(let [;; Start on heads
      heads-trace (:trace (p/generate branch-model-k [nil]
                                       (cm/choicemap :coin (mx/scalar 1.0)
                                                     :heads (mx/scalar 2.5))))
      ;; Switch to tails
      fwd (p/update branch-model-k heads-trace
                     (cm/choicemap :coin (mx/scalar 0.0)
                                   :tails (mx/scalar 3.7)))
      ;; Reverse
      rev (p/update branch-model-k (:trace fwd) (:discard fwd))
      ;; Weight symmetry: fwd + rev should equal 0
      total (+ (mx/item (:weight fwd)) (mx/item (:weight rev)))]

  (assert-close "fwd_weight + rev_weight ≈ 0" 0.0 total 1e-4))

(println "\n-- All update discard branch tests complete --")
