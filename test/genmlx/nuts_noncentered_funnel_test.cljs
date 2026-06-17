;; @tier medium
(ns genmlx.nuts-noncentered-funnel-test
  "Positive control for genmlx-gp5i.

   The centered Neal funnel (paper_bench_funnel: v ~ N(0,3), x_i ~ N(0,exp(v/2)))
   gives NUTS a high R-hat (~2.2) at low ESS. That is NOT a NUTS bug — it is the
   canonical pathological GEOMETRY (the neck at negative v has exponentially small
   x-scale). An independent textbook NUTS reproduces the same bias, and the GenMLX
   U-turn (inv-mass-multiply / genmlx-o78h), slice, and energy-error divergence
   logic are all correct.

   This test pins the positive control: on the NON-CENTERED reparameterization of
   the SAME funnel — v ~ N(0,3), xr_i ~ N(0,1), so the SAMPLED latents are
   independent and well-conditioned (no funnel geometry) — NUTS converges cleanly
   (multi-chain R-hat on v near 1, v marginal recovers N(0,3)). If this ever
   regresses, the NUTS sampler itself is broken; the centered-funnel R-hat is not
   a regression signal."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.diagnostics :as diag]
            [cljs.test :refer [deftest is testing run-tests]]
            [genmlx.test-helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

(def D 5)

(def noncentered-funnel
  ;; Non-centered Neal's funnel: the funnel transform x_i = xr_i * exp(v/2) is
  ;; deterministic, so the SAMPLED space (v, xr_i) carries no funnel geometry.
  (dyn/auto-key
    (gen [d]
      (let [v (trace :v (dist/gaussian 0 3))]
        (doseq [i (range d)]
          (trace (keyword (str "xr" i)) (dist/gaussian 0 1)))
        v))))

(def addresses (vec (cons :v (map #(keyword (str "xr" %)) (range D)))))

(defn- v-chain
  "One NUTS chain; returns the :v samples as a vector of MLX scalars."
  [seed]
  (let [samples (mcmc/nuts {:samples 300 :burn 200 :addresses addresses
                            :adapt-step-size true :adapt-metric true
                            :target-accept 0.8 :key (rng/fresh-key seed)}
                           noncentered-funnel [D] cm/EMPTY)
        chain (mapv #(mx/scalar (nth % 0)) samples)]
    (mx/clear-cache!) (mx/force-gc!)
    chain))

(deftest nuts-converges-on-noncentered-funnel
  (testing "NUTS converges on the well-conditioned non-centered funnel (algorithm is sound)"
    (let [chains (mapv v-chain [42 43 44 45])
          rhat (diag/r-hat chains)
          all (mapcat #(map mx/item %) chains)
          v-mean (/ (reduce + all) (count all))
          v-std (let [m v-mean
                      n (count all)]
                  (js/Math.sqrt (/ (reduce + (map #(let [d (- % m)] (* d d)) all)) n)))]
      (println (str "  non-centered funnel: R-hat(v)=" (.toFixed rhat 3)
                    "  v-mean=" (.toFixed v-mean 3) " (truth 0)"
                    "  v-std=" (.toFixed v-std 3) " (truth 3)"))
      (is (< rhat 1.1)
          "multi-chain R-hat on v is near 1 — NUTS mixes on the well-conditioned reparameterization")
      (is (h/close? 0.0 v-mean 0.7)
          "v marginal mean recovers ~0 (no bias when geometry is well-conditioned)")
      (is (< 2.2 v-std 3.8)
          "v marginal std recovers ~3 (N(0,3) prior is the marginal — no observations)"))))

(run-tests)
