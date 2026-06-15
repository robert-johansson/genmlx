;; @tier fast core
(ns genmlx.mcmc-elim-filter-test
  "Pins the two parameterizations prepare-mcmc-score gives an analytically-
   eliminated model (bean genmlx-10z1).

   prepare-mcmc-score filters eliminated addresses out of the MCMC param
   vector (L3.5 Rao-Blackwell: score = analytical marginal over the rest).
   But the TENSOR-NATIVE score path derives its latent index from the
   schema/source and ignores the filtered address list entirely — so on
   every static model that tensor-compiles (all current eliminated SBC
   models), MCMC samples the FULL JOINT parameterization and the filter is
   bypassed. Both parameterizations are valid targets; they are different
   methods. The filter is live only on the GFI fallback (e.g. a conjugate
   pair whose prior has no noise transform, like beta-bernoulli).

   These tests pin which path each model class takes, so a dispatch change
   that silently flips a model between joint and Rao-Blackwellized MCMC
   (or zeroes out a param vector) fails here first."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.util :as u])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Static gaussian-gaussian pairs: eliminated AND tensor-compilable.
(def two-g (dyn/auto-key (gen [] (let [a (trace :a (dist/gaussian 0 2))
                                       b (trace :b (dist/gaussian 0 2))]
                                   (trace :obs-a (dist/gaussian a 1))
                                   (trace :obs-b (dist/gaussian b 1)) [a b]))))

;; Beta-bernoulli: eliminated but beta has no noise transform, so the
;; tensor-native compile declines and the GFI fallback (filter live) runs.
(def bb (dyn/auto-key (gen [] (let [p (trace :p (dist/beta-dist 2 2))]
                                (trace :obs (dist/bernoulli p)) p))))

(def obs-2g (-> cm/EMPTY
                (cm/set-choice [:obs-a] (mx/scalar 1.0))
                (cm/set-choice [:obs-b] (mx/scalar -1.0))))
(def obs-bb (cm/set-choice cm/EMPTY [:obs] (mx/scalar 1.0)))

(deftest both-models-are-eliminated  ;; preconditions: the class is active
  (is (= #{:a :b} (u/get-eliminated-addresses two-g)))
  (is (= #{:p} (u/get-eliminated-addresses bb))))

(deftest tensor-native-path-samples-the-full-joint
  ;; The filter empties the address list (#{:a :b} eliminates everything),
  ;; but the tensor-native score derives latents from the schema, so MCMC
  ;; still samples BOTH params — this is why SBC cmh/hmc on eliminated
  ;; models calibrate with all addresses present.
  (let [{:keys [trace]} (p/generate two-g [] obs-2g)
        r (u/prepare-mcmc-score two-g [] obs-2g [:a :b] trace)]
    (is (:tensor-native? r) "static model takes the tensor-native score")
    (is (= 2 (:n-params r)) "full joint: both eliminated params sampled")
    (is (= #{:a :b} (set (keys (:latent-index r))))
        "latent index covers the schema's latents, not the filtered list")))

(deftest gfi-fallback-applies-the-filter
  ;; beta-bernoulli declines tensor-native -> the filter is LIVE: with its
  ;; only latent eliminated the param vector is empty. MCMC over zero
  ;; params is meaningless (the posterior is fully analytical) — this test
  ;; pins the current contract so the n-params=0 surface is explicit.
  (let [{:keys [trace]} (p/generate bb [] obs-bb)
        r (u/prepare-mcmc-score bb [] obs-bb [:p] trace)]
    (is (not (:tensor-native? r)) "beta prior has no noise transform")
    (is (= 0 (:n-params r))
        "filter removed the eliminated latent from the param vector")))

(deftest extract-params-empty-latent-index-names-its-cause
  ;; genmlx-nytl: when a tensor-native score exposes ZERO sampling params
  ;; (every selected latent analytically eliminated), extract-params-by-index
  ;; must name the cause, not crash with the cryptic native error
  ;; 'stack requires at least one array' (mx/stack on []).
  (let [{:keys [trace]} (p/generate two-g [] obs-2g)]
    (is (thrown-with-msg? js/Error #"no free latent addresses to sample"
          (u/extract-params-by-index trace {}))
        "empty latent-index throws a clear :no-mcmc-latents ex-info")
    (is (= [2] (vec (mx/shape (u/extract-params-by-index trace {:a 0 :b 1}))))
        "non-empty latent-index still extracts normally (no regression)")))

(deftest pdm2-observed-only-latent-throws-but-free-latent-samples
  ;; genmlx-pdm2: the gen_jl_speed_test "single_gaussian" compiled-MH benchmark
  ;; observed its ONLY latent (model :x, constraint {:x 0.5}), leaving zero free
  ;; latents — so compiled-MH (correctly) threw :no-mcmc-latents. The bean's
  ;; premise (linreg/many are "fully eliminated") was inverted: those are
  ;; non-static loop models that sample fine; single_gaussian was the real crash,
  ;; an observed-sole-latent collision. The fix replaces it with a 1-latent
  ;; normal-normal (latent :mu, SEPARATE observed :y). This pins both halves.
  (testing "sole latent is observed -> :no-mcmc-latents (the documented contract)"
    (let [m (dyn/auto-key (gen [] (trace :x (dist/gaussian 0 1))))
          obs (cm/choicemap :x (mx/scalar 0.5))
          {:keys [trace]} (p/generate m [] obs)]
      (is (thrown-with-msg? js/Error #"no free latent addresses to sample"
            (u/prepare-mcmc-score m [] obs [:x] trace))
          "selecting the sole observed latent exposes zero sampling params")))
  (testing "free latent (normal-normal :mu, observed :y) -> samples, no throw"
    (let [m (dyn/auto-key (gen [] (let [mu (trace :mu (dist/gaussian 0 1))]
                                    (trace :y (dist/gaussian mu 1))
                                    mu)))
          obs (cm/choicemap :y (mx/scalar 0.5))
          {:keys [trace]} (p/generate m [] obs)
          r (u/prepare-mcmc-score m [] obs [:mu] trace)
          samples (mcmc/compiled-mh {:samples 10 :addresses [:mu]} m [] obs)]
      (is (= 1 (:n-params r)) "the :mu latent is a free sampling parameter")
      (is (= 10 (count samples)) "compiled-MH returns the requested samples")
      (is (= 1 (count (first samples))) "each sample is the 1-D :mu latent"))))

(cljs.test/run-tests)

