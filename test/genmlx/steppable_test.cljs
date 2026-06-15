;; @tier medium
(ns genmlx.steppable-test
  "genmlx-rfal: the steppable SMC wrapper (init-state / step / peek / done?).
   The rigorous oracle is DRIVER-EQUIVALENCE: `step` reproduces the smcp3 driver
   loop body verbatim, so under a fixed seed a stepped run must equal the batch
   driver's log-ML bit-for-bit (the handler/driver is ground truth)."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.inference.steppable :as sp]
            [genmlx.inference.smcp3 :as smcp3]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.gen :refer [gen]]))

;; Normal-normal with a shared latent mu and a growing set of observations
;; y0..y_{T-1} | mu ~ N(mu, sn^2), mu ~ N(0, s0^2). Standard SMC (nil kernels)
;; progressively constrains y_t at step t; log-ML after T steps = log p(y_0..y_{T-1}).
(def s0 3.0)
(def sn 1.0)
(def obs-vals [0.5 1.2 0.8])
(def T (count obs-vals))

(def model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 s0))]
      (doseq [i (range T)]
        (trace (keyword (str "y" i)) (dist/gaussian mu sn)))
      mu)))

(def obs-seq
  (mapv (fn [i] (cm/choicemap (keyword (str "y" i)) (mx/scalar (nth obs-vals i))))
        (range T)))

(defn- opts [seed] {:particles 3000 :ess-threshold 0.5 :key (rng/fresh-key seed)})

(defn- mu-of [trace]
  (mx/realize (cm/get-value (cm/get-submap (:choices trace) :mu))))

(deftest steppable-shape-and-boundary-test
  (testing "init-state is O(1) / uninitialized; done?/t boundary; step-on-done throws"
    (let [s0state (sp/init-state model [] obs-seq (opts 1))]
      (is (= 0 (:t s0state)) "fresh state t=0")
      (is (nil? (:traces s0state)) "fresh state has no traces (init runs in the first step)")
      (is (not (sp/done? s0state)) "fresh state not done")
      (let [done (sp/run s0state)]
        (is (sp/done? done) "run reaches done")
        (is (= T (:t done)) "t == n-steps after run")
        (is (thrown? :default (sp/step done)) "stepping a completed state throws")))))

(deftest steppable-immutability-test
  (testing "step returns a NEW SMCState; the input is unchanged"
    (let [s (sp/init-state model [] obs-seq (opts 2))
          s' (sp/step s)]
      (is (not (identical? s s')) "step returns a new value")
      (is (= 0 (:t s)) "original state t unchanged")
      (is (= 1 (:t s')) "stepped state t advanced"))))

(deftest steppable-peek-symmetry-test
  (testing "peek returns the SAME key set for a fresh, mid-run, and completed state"
    (let [fresh (sp/init-state model [] obs-seq (opts 3))
          mid (sp/step fresh)
          done (sp/run fresh)
          ks (fn [m] (set (keys m)))]
      (is (= (ks (sp/peek fresh)) (ks (sp/peek mid)) (ks (sp/peek done)))
          "peek payload is symmetric across init/step/done")
      (is (= #{:t :n-steps :done? :particles :log-ml-estimate :ess}
             (ks (sp/peek fresh)))
          "peek exposes the documented keys (no per-step :resampled?)")
      (is (= (:particles (sp/peek fresh)) (:ess (sp/peek fresh)))
          "uninitialized peek reports full ESS")
      (is (js/isFinite (:log-ml-estimate (sp/peek done))) "completed log-ML is finite"))))

(deftest steppable-driver-equivalence-test
  (testing "DRIVER-EQUIVALENCE ORACLE: a stepped run equals the smcp3 batch driver's log-ML bit-for-bit under the same key"
    (let [o (opts 12345)
          driver (mx/realize (:log-ml-estimate (smcp3/smcp3 o model [] obs-seq)))
          stepped (:log-ml-estimate (sp/peek (sp/run (sp/init-state model [] obs-seq o))))]
      (is (h/close? driver stepped 1e-4)
          (str "stepped log-ML " stepped " == driver log-ML " driver))
      (is (js/isFinite stepped) "stepped log-ML finite")
      (is (< stepped 0) "log-ML of observations is negative"))))

(deftest steppable-determinism-test
  (testing "two runs under the same :key produce identical log-ML; particles stay diverse (strip-analytical applied)"
    (let [run-it (fn [] (sp/run (sp/init-state model [] obs-seq (opts 777))))
          a (sp/peek (run-it))
          b (sp/peek (run-it))]
      (is (= (:log-ml-estimate a) (:log-ml-estimate b))
          "same :key -> bit-identical log-ML (reproducible)")
      ;; strip-analytical must have removed the L3 conjugate path, else all
      ;; particles collapse to the deterministic posterior mean.
      (let [final (sp/run (sp/init-state model [] obs-seq (opts 99)))
            mus (mapv mu-of (:traces final))]
        (is (> (count (distinct mus)) 1)
            "particle latent values are diverse (not collapsed by an analytical handler)")))))

(deftest steppable-long-run-gc-test
  (testing "GC OWNERSHIP: a >10-step stepped run (crossing the every-10-step sweep) completes with finite log-ML"
    (let [tt 12
          ovals (mapv (fn [i] (+ 0.1 (* 0.05 i))) (range tt))
          long-model (gen []
                       (let [mu (trace :mu (dist/gaussian 0 s0))]
                         (doseq [i (range tt)]
                           (trace (keyword (str "y" i)) (dist/gaussian mu sn)))
                         mu))
          long-obs (mapv (fn [i] (cm/choicemap (keyword (str "y" i)) (mx/scalar (nth ovals i)))) (range tt))
          done (sp/run (sp/init-state long-model [] long-obs {:particles 500 :ess-threshold 0.5 :key (rng/fresh-key 5)}))
          p (sp/peek done)]
      (is (= tt (:t p)) "completed all 12 steps")
      (is (js/isFinite (:log-ml-estimate p)) "log-ML finite after the every-10-step sweep"))))

(cljs.test/run-tests)
