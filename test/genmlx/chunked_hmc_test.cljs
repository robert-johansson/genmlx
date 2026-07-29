;; @tier medium
(ns genmlx.chunked-hmc-test
  "Chunked captured HMC chain (genmlx-dys7): past the single-graph fused
   limit, fused-hmc decomposes the chain into persist-gate-sized chunks on
   captured replay instead of dropping to block-compiled HMC.

   Pins: (1) the chunked chain equals the single-graph fused chain given
   the SAME noise tensors — including across burn/thin chunk seams (both
   arms drive the private builders directly, fused_mcmc_test-style, with
   sizes that make the chunk plan multi-chunk/multi-shape under the real
   persist gate); (2) the public too-large path engages chunking on CUDA,
   tracks a REAL acceptance rate (the genmlx-d62h block-path nil is gone
   on this path), and returns a reusable :chain-fn. CUDA-only: on Metal
   the too-large fallback is still block-compiled, so this self-gates to
   a fast negative contract (parallel_stress_test pattern)."
  (:require [cljs.test :as t :refer [deftest is testing]]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.gen :refer [gen]]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.util :as u]))

(def ^:private mfhbc @(resolve 'genmlx.inference.mcmc/make-fused-hmc-burn-and-collect))
(def ^:private make-chunked @(resolve 'genmlx.inference.mcmc/make-chunked-hmc-runner))

(def cuda? (not (mx/metal-is-available?)))

;; Small linreg posterior — same family as the parity rows, cheap score.
(def xs [0.0 1.0 2.0 3.0])
(def ys [0.1 1.9 4.2 5.8])

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

(defn- max-abs-diff [a b]
  (mx/item (mx/amax (mx/abs (mx/subtract a b)))))

(defn- both-arms
  "Drive the fused single-graph builder and the chunked runner with the
   SAME start point and noise; return [fused-result chunked-result], each
   #js [final-q samples accept-count]."
  [{:keys [burn samples thin leapfrog step-size seed]}]
  (let [{:keys [trace]} (p/generate (dyn/with-key hmodel (rng/fresh-key seed))
                                    [xs] obs)
        {:keys [score-fn init-params n-params]}
        (u/prepare-mcmc-score hmodel [xs] obs [:slope :intercept] trace)
        neg-U (fn [q] (mx/negative (score-fn q)))
        grad-neg-U (mx/grad neg-U)
        eps (mx/scalar step-size)
        half-eps (mx/scalar (* 0.5 step-size))
        half (mx/scalar 0.5)
        total (+ burn (* thin samples))
        [k1 k2] (rng/split (rng/fresh-key (+ seed 100)))
        momentum (rng/normal k1 [total n-params])
        uniforms (rng/uniform k2 [total])
        _ (mx/materialize! momentum uniforms)
        fused-fn (mfhbc burn samples thin neg-U grad-neg-U
                        eps half-eps half n-params leapfrog)
        chunk-fn (make-chunked burn samples thin neg-U grad-neg-U
                               eps half-eps half n-params leapfrog)
        rf (fused-fn init-params momentum uniforms)
        _ (mx/materialize! (aget rf 0) (aget rf 1) (aget rf 2))
        rc (chunk-fn init-params momentum uniforms)
        _ (mx/materialize! (aget rc 0) (aget rc 1) (aget rc 2))]
    [rf rc]))

(deftest chunked-equals-fused-simple
  (when cuda?
    (testing "chunked == fused, burn 0 thin 1, multi-chunk plan (L=64: chunks of 62+38 steps)"
      (let [[rf rc] (both-arms {:burn 0 :samples 100 :thin 1
                                :leapfrog 64 :step-size 0.02 :seed 11})]
        (is (= [100 2] (vec (mx/shape (aget rc 1)))) "sample matrix shape")
        (is (< (max-abs-diff (aget rf 1) (aget rc 1)) 1e-4)
            "samples equal to kernel-fusion rounding")
        (is (< (max-abs-diff (aget rf 0) (aget rc 0)) 1e-4)
            "final params equal")
        (is (< (max-abs-diff (aget rf 2) (aget rc 2)) 0.5)
            "accept counts equal (integer-valued)")))))

(deftest chunked-equals-fused-burn-thin-seams
  (when cuda?
    (testing "burn/thin cadence preserved across burn->sample chunk seam"
      (let [[rf rc] (both-arms {:burn 37 :samples 20 :thin 3
                                :leapfrog 64 :step-size 0.02 :seed 23})]
        (is (= [20 2] (vec (mx/shape (aget rc 1)))) "thinned sample count")
        (is (< (max-abs-diff (aget rf 1) (aget rc 1)) 1e-4)
            "burn+thin chunked samples equal fused")))))

(deftest chunked-engages-past-fused-limit
  (if-not cuda?
    (testing "Metal negative contract: fused path unaffected by the chunked branch"
      (let [r (mcmc/fused-hmc {:samples 2 :burn 0 :thin 1 :leapfrog-steps 4
                               :step-size 0.1 :addresses [:slope :intercept]
                               :key (rng/fresh-key 5) :device :gpu}
                              hmodel [xs] obs)]
        (is (some? (:samples r)) "small fused path still runs on Metal")))
    (testing "public too-large path rides chunked: real acceptance + reusable :chain-fn"
      ;; S x L = 300 x 64 = 19200 > the 16000 CUDA fused limit — engages
      ;; chunking through the PUBLIC entry.
      (let [opts {:samples 300 :burn 0 :thin 1 :leapfrog-steps 64
                  :step-size 0.02 :addresses [:slope :intercept] :device :gpu}
            r1 (mcmc/fused-hmc (assoc opts :key (rng/fresh-key 31))
                               hmodel [xs] obs)
            r2 (mcmc/fused-hmc (assoc opts :key (rng/fresh-key 32)
                                      :chain-fn (:chain-fn r1))
                               hmodel [xs] obs)]
        (is (= [300 2] (vec (mx/shape (:samples r1)))) "full sample matrix")
        (is (some? (:acceptance-rate r1)) "acceptance tracked past the fused limit")
        (is (<= 0.2 (:acceptance-rate r1) 1.0) "acceptance in a sane band")
        (is (some? (:chain-fn r1)) "chunked runner returned as :chain-fn")
        (is (pos? (max-abs-diff (:samples r1) (:samples r2)))
            "reused :chain-fn with a different key gives a different chain")))))

(t/run-tests)
