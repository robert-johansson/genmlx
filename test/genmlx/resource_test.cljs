;; @tier medium
(ns genmlx.resource-test
  "Stress tests for Metal resource management.
   Verifies that inference loops don't leak Metal buffers over extended runs."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.kernel :as kern])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Simple 5-site Gaussian model
(def model
  (gen [xs]
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
                   (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                          intercept) 1)))
      slope)))

(def xs [1.0 2.0 3.0])
(def observations
  (reduce (fn [cm [j y]]
            (cm/set-choice cm [(keyword (str "y" j))] (mx/scalar y)))
          cm/EMPTY
          (map-indexed vector [2.5 5.1 7.3])))

(deftest mh-stress-test
  (testing "MH stress — 500 iterations shouldn't grow memory linearly"
    (let [_ (mx/clear-cache!)
          _ (mx/reset-peak-memory!)
          {:keys [trace]} (p/generate (dyn/auto-key model) [xs] observations)
          _ (mx/eval! (:score trace))
          _ (mcmc/mh {:samples 50 :selection (sel/select :slope :intercept)}
                     model [xs] observations)
          mem-50 (mx/get-active-memory)
          _ (mcmc/mh {:samples 500 :selection (sel/select :slope :intercept)}
                     model [xs] observations)
          mem-500 (mx/get-active-memory)]
      (is (or (< mem-500 (* 5 (max mem-50 1024)))
              (< mem-500 (* 10 1024 1024)))
          "memory bounded (500 iters < 5x 50 iters)"))))

(deftest is-stress-test
  (testing "IS stress — 200 samples complete without crash"
    (let [_ (mx/clear-cache!)
          result (is/importance-sampling {:samples 200} model [xs] observations)]
      (is (= 200 (count (:traces result))) "IS completed 200 samples")
      (is (= 200 (count (:log-weights result))) "IS has weights"))))

(deftest collect-samples-resource-test
  (testing "collect-samples with array-heavy step-fn"
    (let [_ (mx/clear-cache!)
          step-fn (fn [state _key]
                    (let [a (mx/add state (mx/scalar 0.1))
                          b (mx/multiply a (mx/scalar 0.99))
                          c (mx/add b (rng/normal (rng/fresh-key) [10]))
                          d (mx/sum c)]
                      (mx/eval! d)
                      {:state d :accepted? true}))
          results (kern/collect-samples
                    {:samples 200 :burn 50}
                    step-fn
                    mx/item
                    (mx/scalar 0.0))]
      (is (= 200 (count results)) "collect-samples completed 200 samples")
      (let [active-mem (mx/get-active-memory)]
        (is (< active-mem (* 50 1024 1024)) "active memory bounded")))))

(deftest clear-cache-effect-test
  (testing "clear-cache! releases the free-buffer pool"
    ;; What is under test, stated precisely: clear-cache! drops MLX's pool of
    ;; freed-but-retained buffers. It says nothing about ACTIVE memory, which is
    ;; why the pool has to be filled deliberately rather than assumed.
    ;;
    ;; genmlx-6e5i: this used to be (<= cache-after cache-before) straight after
    ;; a plain allocation loop. That loop puts nothing IN the pool -- the array
    ;; wrappers are still live, so `active` grows while `cache` stays ~0 -- and
    ;; it only passed because incidental finalization usually landed some buffer
    ;; there first. Under 4-way parallel GPU load it did not: cache-before read
    ;; 0, cache-after read 48 (one buffer finalized between the two samples),
    ;; and a true invariant failed as 48 <= 0. Widening the comparison with
    ;; slack would have hidden that. The repair is to ESTABLISH the precondition
    ;; and to make the signal large enough that a stray 48-byte buffer cannot
    ;; decide the outcome.
    (let [;; Buffers must be held LIVE and released together. Allocating and
          ;; dropping one at a time parks nothing: the allocator simply hands
          ;; back the same buffer each iteration, so the pool never grows (that
          ;; is why the old loop left it at ~0 in the first place).
          fill! (fn [] (let [held (mapv (fn [_] (doto (rng/normal (rng/fresh-key) [250000])
                                                  mx/eval!))
                                        (range 50))]
                         (count held)))]
      ;; Two rounds, because finalization lags by one: round 2's allocation is
      ;; what collects round 1's wrappers, and jsc-cleanup! then drains those
      ;; finalizers WITHOUT dropping the pool (force-gc! would clear it,
      ;; defeating the point). Measured on sm_120: this parks ~50 MB of 1 MB
      ;; buffers in the pool, ~10^6x the ambient noise that broke the old
      ;; assertion.
      (fill!)
      (mx/jsc-cleanup!)
      (mx/clear-cache!)
      (fill!)
      (mx/jsc-cleanup!)
      (let [cache-before (mx/get-cache-memory)
            _ (mx/clear-cache!)
            cache-after (mx/get-cache-memory)]
        (is (> cache-before (* 4 1024 1024))
            (str "precondition: pool holds > 4 MB before the clear (got "
                 cache-before " bytes)"))
        (is (< cache-after (quot cache-before 100))
            (str "clear-cache! releases the pool (" cache-before " -> "
                 cache-after " bytes)"))))))

(cljs.test/run-tests)
