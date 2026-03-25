(ns genmlx.fused-vi-test
  "WP-2: Fused VI composition tests.
   Tests the :vi path of fused-learn in a single VI call to stay under
   the cumulative iteration threshold that triggers a known MLX/nbb segfault
   in fresh processes (both Bun and Node.js). See memory/bug_bun_segfault_vmap.md.
   Full convergence testing is done via REPL where no limit exists."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.choicemap :as cm]
            [genmlx.dist :as dist]
            [genmlx.inference.compiled-optimizer :as co])
  (:require-macros [genmlx.gen :refer [gen]]))

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
;; Tests
;; ---------------------------------------------------------------------------

(deftest fused-vi-dispatch-test
  (testing "single fused-learn :vi call"
    (let [obs (cm/choicemap {:y (mx/scalar 5.0)})
          result (co/fused-learn simple-model [] obs [:mu :sigma] :vi
                                 {:iterations 20 :lr 0.01})]
      ;; Dispatch metadata
      (is (= :vi (:compilation-level result)) ":compilation-level is :vi")
      (is (= :vi (:method result)) ":method is :vi")
      ;; VI output structure
      (is (some? (:mu result)) "returns :mu")
      (is (some? (:sigma result)) "returns :sigma")
      (is (some? (:elbo-history result)) "returns :elbo-history")
      (is (fn? (:sample-fn result)) "returns :sample-fn")
      ;; ELBO history
      (is (pos? (count (:elbo-history result))) "elbo-history is non-empty")
      ;; sample-fn works
      (let [samples ((:sample-fn result) 5)]
        (is (= 5 (count samples)) "sample-fn produces 5 samples")))))

(cljs.test/run-tests)
