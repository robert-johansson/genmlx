;; @tier fast core
(ns genmlx.trace-mh-elim-test
  "Regression guard for trace-MH on analytically-eliminated models
   (bean genmlx-540f).

   On a static-conjugate model the L3 analytical generate pins eliminated
   latents at their DETERMINISTIC posterior mean and intercepts regenerate
   via :auto-regenerate-transition. Unstripped trace-MH chains therefore
   anchored at the posterior mean instead of sampling it: mini-SBC chi2(a)
   = 290.9 vs crit 21.67, draw sd 0.40 vs the analytic 0.894. The fix
   strips the analytical path in the trace-MH entry points
   (kern/mh-kernel, mcmc/mh-step, mcmc/mh), mirroring smc.

   These tests check the chain's marginal moments against the closed-form
   posterior — an independent oracle, never the path under test."
  (:require [cljs.test :refer [deftest is]]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.kernel :as kern]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Same shape as sbc_test's two-gaussians: two independent gaussian-gaussian
;; conjugate pairs, statically eliminable (#{:a :b}).
(def model
  (dyn/auto-key (gen []
                     (let [a (trace :a (dist/gaussian 0 2))
                           b (trace :b (dist/gaussian 0 2))]
                       (trace :obs-a (dist/gaussian a 1))
                       (trace :obs-b (dist/gaussian b 1))
                       [a b]))))

;; Closed-form posterior for a ~ N(0, 2^2), obs ~ N(a, 1^2), y observed:
;; mean = y * 4/5, sd = sqrt(4/5).
(def y-a 1.0)
(def y-b -1.0)
(def post-mean-a (* y-a 0.8))
(def post-sd (js/Math.sqrt 0.8))

(def obs (-> cm/EMPTY
             (cm/set-choice [:obs-a] (mx/scalar y-a))
             (cm/set-choice [:obs-b] (mx/scalar y-b))))

(defn- a-value [trace]
  (let [v (cm/get-value (cm/get-submap (:choices trace) :a))]
    (mx/eval! v)
    (mx/item v)))

(defn- mean [xs] (/ (reduce + xs) (count xs)))
(defn- sd [xs]
  (let [m (mean xs)]
    (js/Math.sqrt (/ (reduce + (map #(let [d (- % m)] (* d d)) xs))
                     (dec (count xs))))))

(deftest model-is-eliminated  ;; precondition: the class under test is active
  (is (= #{:a :b} (get-in (:schema model) [:analytical-plan :rewrite-result :eliminated]))
      "two-gaussians must be statically eliminated or this guard tests nothing")
  (is (some? (get-in (:schema model) [:auto-regenerate-transition]))
      "the regenerate interception must be present on the unstripped model"))

;; 400 post-burn draws: sd estimate has ~5% relative error at tau~1-2, so a
;; [0.6, 1.2] band separates the analytic 0.894 cleanly from the broken 0.40.
(deftest mh-cycle-samples-the-posterior  ;; the sbc mh-cycle path
  (let [kernel (kern/cycle-kernels [(kern/mh-kernel (sel/select :a))
                                    (kern/mh-kernel (sel/select :b))])
        {:keys [trace]} (p/generate model [] obs)
        traces (kern/run-kernel {:samples 400 :burn 200 :key (rng/fresh-key 540)}
                                kernel trace)
        draws (mapv a-value traces)
        m (mean draws) s (sd draws)]
    (is (< (js/Math.abs (- m post-mean-a)) 0.25)
        (str "chain mean " m " near analytic posterior mean " post-mean-a))
    (is (< 0.6 s 1.2)
        (str "chain sd " s " near analytic posterior sd " post-sd
             " (0.40 = mean-anchored regression)"))))

(deftest mcmc-mh-samples-the-posterior  ;; the mcmc/mh entry point
  (let [traces (mcmc/mh {:samples 400 :burn 200 :key (rng/fresh-key 541)}
                        model [] obs)
        draws (mapv a-value traces)
        m (mean draws) s (sd draws)]
    (is (< (js/Math.abs (- m post-mean-a)) 0.25)
        (str "chain mean " m " near analytic posterior mean " post-mean-a))
    (is (< 0.6 s 1.2)
        (str "chain sd " s " near analytic posterior sd " post-sd))))

(cljs.test/run-tests)
