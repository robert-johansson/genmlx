(ns genmlx.fused-vi-test
  "WP-2: Fused VI composition tests.
   Tests the :vi path of fused-learn in a single VI call to stay under
   the cumulative iteration threshold that triggers a known MLX/nbb segfault
   in fresh processes (both Bun and Node.js). See memory/bug_bun_segfault_vmap.md.
   Full convergence testing is done via REPL where no limit exists."
  (:require [genmlx.mlx :as mx]
            [genmlx.choicemap :as cm]
            [genmlx.dist :as dist]
            [genmlx.inference.compiled-optimizer :as co])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def ^:dynamic *pass-count* (atom 0))
(def ^:dynamic *fail-count* (atom 0))

(defn assert-true [msg pred]
  (if pred
    (do (swap! *pass-count* inc)
        (println (str "  PASS: " msg)))
    (do (swap! *fail-count* inc)
        (println (str "  FAIL: " msg)))))

;; ---------------------------------------------------------------------------
;; Test model
;; ---------------------------------------------------------------------------

(def simple-model
  (gen []
       (let [mu (trace :mu (dist/gaussian 0 10))
             sigma (trace :sigma (dist/gaussian 1 0.1))]
         (trace :y (dist/gaussian mu (mx/abs sigma)))
         mu)))

;; ---------------------------------------------------------------------------
;; Single fused-learn :vi call — all assertions in one shot
;; ---------------------------------------------------------------------------
;; NOTE: Cumulative VI iterations in a fresh nbb process trigger a segfault
;; (MLX/nbb interaction, not Bun). We use ONE call with 20 iterations to
;; verify the full :vi dispatch path without hitting the threshold.

(println "\n=== WP-2 Fused VI Tests ===")

(let [obs (cm/choicemap {:y (mx/scalar 5.0)})
      result (co/fused-learn simple-model [] obs [:mu :sigma] :vi
                             {:iterations 20 :lr 0.01})]

  ;; Dispatch metadata
  (assert-true ":compilation-level is :vi"
               (= :vi (:compilation-level result)))
  (assert-true ":method is :vi"
               (= :vi (:method result)))

  ;; VI output structure
  (assert-true "returns :mu"
               (some? (:mu result)))
  (assert-true "returns :sigma"
               (some? (:sigma result)))
  (assert-true "returns :elbo-history"
               (some? (:elbo-history result)))
  (assert-true "returns :sample-fn"
               (fn? (:sample-fn result)))

  ;; ELBO history
  (assert-true "elbo-history is non-empty"
               (pos? (count (:elbo-history result))))

  ;; sample-fn works
  (let [samples ((:sample-fn result) 5)]
    (assert-true "sample-fn produces 5 samples"
                 (= 5 (count samples)))))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== WP-2 Fused VI Test Summary ==="))
(println (str "  PASS: " @*pass-count*))
(println (str "  FAIL: " @*fail-count*))
(println (str "  TOTAL: " (+ @*pass-count* @*fail-count*)))
(when (pos? @*fail-count*)
  (println "  *** FAILURES DETECTED ***"))
(when (zero? @*fail-count*)
  (println "  All tests passed!"))
