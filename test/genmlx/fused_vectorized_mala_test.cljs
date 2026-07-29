;; @tier medium
(ns genmlx.fused-vectorized-mala-test
  "fused-vectorized-mala (genmlx-zebd): the whole N-chain MALA sweep as one
   fused graph on captured replay — batched vgenerate init, [T,N,D] noise,
   zero host syncs in the chain.

   Pins: (1) shapes + same-key determinism (captured replay is bit-stable);
   (2) statistical agreement with the eager vectorized-mala at the same eps
   (different PRNG stream layouts, so bands not bits): pooled acceptance
   and pooled slope-tail against the analytic posterior; (3) :chain-fn
   reuse across calls with fresh keys."
  (:require [cljs.test :as t :refer [deftest is testing]]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.gen :refer [gen]]
            [genmlx.choicemap :as cm]
            [genmlx.inference.mcmc :as mcmc]))

(def xs [0.0 1.0 2.0 3.0 4.0])
(def ys [1.1 2.9 5.2 6.8 9.1])   ;; slope ~2, intercept ~1

(def model
  (dyn/auto-key
    (gen [xs]
      (let [slope     (trace :slope (dist/gaussian 0.0 5.0))
            intercept (trace :intercept (dist/gaussian 0.0 5.0))]
        (doseq [[j x] (map-indexed vector xs)]
          (trace (keyword (str "y" j))
                 (dist/gaussian (mx/add (mx/multiply slope x) intercept) 1.0)))
        slope))))

(def hmodel (dyn/strip-analytical-path model))

(def obs
  (apply cm/choicemap
         (mapcat (fn [[j y]] [(keyword (str "y" j)) (mx/scalar y)])
                 (map-indexed vector ys))))

(defn- run-fused [opts k]
  (mcmc/fused-vectorized-mala
   (merge {:burn 0 :thin 1 :step-size 0.08
           :addresses [:slope :intercept] :key k :device :gpu}
          opts)
   hmodel [xs] obs))

(defn- max-abs-diff [a b]
  (mx/item (mx/amax (mx/abs (mx/subtract a b)))))

(deftest fused-vectorized-mala-shapes-and-determinism
  (testing "shapes and same-key bit-stability"
    (let [k  (rng/fresh-key 41)
          r1 (run-fused {:samples 50 :n-chains 16} k)
          r2 (run-fused {:samples 50 :n-chains 16 :chain-fn (:chain-fn r1)} k)]
      (is (= [50 16 2] (vec (mx/shape (:samples r1)))) "samples [S,N,D]")
      (is (= [16 2] (vec (mx/shape (:final-params r1)))) "final [N,D]")
      (is (<= 0.0 (:acceptance-rate r1) 1.0) "acceptance in [0,1]")
      (is (zero? (max-abs-diff (:samples r1) (:samples r2)))
          "same key + reused chain-fn -> identical samples")
      (is (== (:acceptance-rate r1) (:acceptance-rate r2))
          "same key -> identical acceptance"))))

(deftest fused-vectorized-mala-statistical-agreement
  (testing "agrees statistically with eager vectorized-mala at the same eps"
    (let [n 64, s 400
          rf (run-fused {:samples s :n-chains n} (rng/fresh-key 42))
          re (mcmc/vectorized-mala
              {:samples s :burn 0 :thin 1 :step-size 0.08
               :addresses [:slope :intercept] :n-chains n
               :key (rng/fresh-key 43) :device :gpu}
              hmodel [xs] obs)
          acc-e (:acceptance-rate (meta re))
          ;; pooled slope tail, fused: mean of samples[S/2:, :, 0]
          tail-f (mx/item
                  (mx/mean (mx/slice-nd (:samples rf)
                                        [(quot s 2) 0 0] [s n 1])))
          ;; pooled slope tail, eager: samples are a vector of [N,D] JS arrays
          tail-e (let [tail (drop (quot s 2) re)
                       ms (map (fn [snap]
                                 (let [rows (js->clj snap)]
                                   (/ (reduce + (map first rows)) (count rows))))
                               tail)]
                   (/ (reduce + ms) (count ms)))]
      (is (< (js/Math.abs (- (:acceptance-rate rf) acc-e)) 0.12)
          (str "pooled acceptance close: fused " (:acceptance-rate rf)
               " vs eager " acc-e))
      (is (< (js/Math.abs (- tail-f tail-e)) 0.35)
          (str "pooled slope tails close: fused " tail-f " vs eager " tail-e))
      (is (< (js/Math.abs (- tail-f 2.0)) 0.5)
          (str "fused pooled tail near true slope: " tail-f)))))

(deftest fused-vectorized-mala-chain-fn-reuse
  (testing "reused chain-fn with a fresh key gives a fresh chain"
    (let [r1 (run-fused {:samples 30 :n-chains 8} (rng/fresh-key 51))
          r2 (run-fused {:samples 30 :n-chains 8 :chain-fn (:chain-fn r1)}
                        (rng/fresh-key 52))]
      (is (= (vec (mx/shape (:samples r1))) (vec (mx/shape (:samples r2))))
          "same shapes across reuse")
      (is (pos? (max-abs-diff (:samples r1) (:samples r2)))
          "different keys -> different chains"))))

(t/run-tests)
