(ns genmlx.kernel-combinator-test
  "Tests for kernel combinators: chain, cycle-kernels, mix-kernels, seed."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.kernel :as kern])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-close [msg expected actual tolerance]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tolerance)
      (println "  PASS:" msg)
      (do (println "  FAIL:" msg)
          (println "    expected:" expected "+/-" tolerance)
          (println "    actual:  " actual)))))

(println "\n=== Kernel Combinator Tests ===\n")

;; Shared model: x ~ N(0, 10), obs_i ~ N(x, 1) for i=0..4, all obs=3.0
;; Posterior: x ~ N(3, ~0.45)
(def model
  (gen []
    (let [mu (dyn/trace :mu (dist/gaussian 0 10))]
      (mx/eval! mu)
      (let [mu-val (mx/item mu)]
        (doseq [i (range 5)]
          (dyn/trace (keyword (str "obs" i))
                     (dist/gaussian mu-val 1)))
        mu-val))))

(def observations
  (reduce (fn [cm i]
            (cm/set-choice cm [(keyword (str "obs" i))]
                           (mx/scalar 3.0)))
          cm/EMPTY (range 5)))

(defn extract-mu-mean [traces]
  (let [mu-vals (mapv (fn [t]
                        (mx/realize (cm/get-value (cm/get-submap (:choices t) :mu))))
                      traces)]
    (/ (reduce + mu-vals) (count mu-vals))))

;; ---------------------------------------------------------------------------
;; 1. chain — sequential composition of kernels
;; ---------------------------------------------------------------------------

(println "-- chain: sequential kernel composition --")
(let [{:keys [trace]} (p/generate model [] observations)
      k (kern/chain (kern/mh-kernel (sel/select :mu))
                    (kern/mh-kernel (sel/select :mu)))
      traces (kern/run-kernel {:samples 100 :burn 50} k trace)
      mu-mean (extract-mu-mean traces)
      ar (:acceptance-rate (meta traces))]
  (assert-true "chain: 100 samples" (= 100 (count traces)))
  (assert-close "chain: posterior mu near 3" 3.0 mu-mean 1.0)
  (assert-true "chain: acceptance rate > 0" (> ar 0)))

;; ---------------------------------------------------------------------------
;; 2. cycle-kernels — round-robin cycling
;; ---------------------------------------------------------------------------

(println "\n-- cycle-kernels: round-robin cycling --")
(let [{:keys [trace]} (p/generate model [] observations)
      k (kern/cycle-kernels 6 [(kern/mh-kernel (sel/select :mu))])
      traces (kern/run-kernel {:samples 100 :burn 50} k trace)
      mu-mean (extract-mu-mean traces)
      ar (:acceptance-rate (meta traces))]
  (assert-true "cycle-kernels: 100 samples" (= 100 (count traces)))
  (assert-close "cycle-kernels: posterior mu near 3" 3.0 mu-mean 1.0)
  (assert-true "cycle-kernels: acceptance rate > 0" (> ar 0)))

;; ---------------------------------------------------------------------------
;; 3. mix-kernels — random mixture of kernels
;; ---------------------------------------------------------------------------

(println "\n-- mix-kernels: single kernel weight 1.0 --")
(let [{:keys [trace]} (p/generate model [] observations)
      k (kern/mix-kernels [[(kern/mh-kernel (sel/select :mu)) 1.0]])
      traces (kern/run-kernel {:samples 100 :burn 50} k trace)
      mu-mean (extract-mu-mean traces)]
  (assert-true "mix-kernels(1): 100 samples" (= 100 (count traces)))
  (assert-close "mix-kernels(1): posterior mu near 3" 3.0 mu-mean 1.0))

(println "\n-- mix-kernels: two kernels weighted --")
(let [{:keys [trace]} (p/generate model [] observations)
      sel (sel/select :mu)
      k (kern/mix-kernels [[(kern/mh-kernel sel) 0.7]
                           [(kern/mh-kernel sel) 0.3]])
      traces (kern/run-kernel {:samples 100 :burn 50} k trace)
      mu-mean (extract-mu-mean traces)]
  (assert-true "mix-kernels(2): 100 samples" (= 100 (count traces)))
  (assert-close "mix-kernels(2): posterior mu near 3" 3.0 mu-mean 1.0))

;; ---------------------------------------------------------------------------
;; 4. seed — fixed PRNG key for accept/reject
;; ---------------------------------------------------------------------------

(println "\n-- seed: kernel with fixed key converges --")
(let [{:keys [trace]} (p/generate model [] observations)
      fixed-key (rng/fresh-key)
      k (kern/seed (kern/mh-kernel (sel/select :mu)) fixed-key)
      ;; seed ignores the step-key from run-kernel, using fixed-key instead
      ;; Verify it still functions as a valid kernel and converges
      traces (kern/run-kernel {:samples 100 :burn 50} k trace)
      mu-mean (extract-mu-mean traces)]
  (assert-true "seed: 100 samples" (= 100 (count traces)))
  (assert-close "seed: posterior mu near 3" 3.0 mu-mean 1.0))

;; seed with update-kernel (deterministic): calling twice yields identical result
(println "\n-- seed: deterministic with update-kernel --")
(let [{:keys [trace]} (p/generate model [] observations)
      constraints (cm/choicemap :mu (mx/scalar 2.5))
      k (kern/seed (kern/update-kernel constraints) (rng/fresh-key))
      traces1 (kern/run-kernel {:samples 1 :burn 0} k trace)
      traces2 (kern/run-kernel {:samples 1 :burn 0} k trace)
      mu1 (mx/realize (cm/get-value (cm/get-submap (:choices (first traces1)) :mu)))
      mu2 (mx/realize (cm/get-value (cm/get-submap (:choices (first traces2)) :mu)))]
  (assert-close "seed(update): identical result" mu1 mu2 1e-6))

(println "\nAll kernel combinator tests complete.")
