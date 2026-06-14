;; @tier medium
(ns genmlx.prng-hygiene-test
  "Deterministic-seed laws for the genmlx-njaq audit cluster: a seeded run
   must be bit-reproducible, and key streams for distinct purposes must be
   disjoint. smc/mh-custom/involutive-mh FAIL these pre-fix (their inner
   operations ran on fresh entropy); nuts/elliptical-slice/importance-
   resampling were deterministic-but-correlated, so their tests are
   regression guards pinning the property going forward."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.inference.smc :as smc]
            [genmlx.inference.importance :as imp]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ^:private xy-model
  (gen []
    (let [x (trace :x (dist/gaussian 0 1))]
      (trace :y0 (dist/gaussian x 0.5))
      (trace :y1 (dist/gaussian x 0.5))
      x)))

(defn- x-of [trace]
  (h/realize (cm/get-choice (:choices trace) [:x])))

;; ---------------------------------------------------------------------------
;; SMC: seeded run is bit-reproducible (pre-fix: init + update unkeyed)
;; ---------------------------------------------------------------------------

(deftest smc-seeded-determinism
  (testing "same :key → identical log-ML and identical particles"
    (let [obs-seq [(cm/choicemap :y0 (mx/scalar 1.0))
                   (cm/choicemap :y1 (mx/scalar 1.2))]
          run #(smc/smc {:particles 20 :key (rng/fresh-key 17)}
                        xy-model [] obs-seq)
          r1 (run)
          r2 (run)]
      (is (= (h/realize (:log-ml-estimate r1))
             (h/realize (:log-ml-estimate r2)))
          "log-ML bit-identical under the same seed")
      (is (= (mapv x-of (:traces r1)) (mapv x-of (:traces r2)))
          "particle :x values bit-identical under the same seed")
      (is (not= (mapv x-of (:traces r1))
                (mapv x-of (:traces (smc/smc {:particles 20
                                              :key (rng/fresh-key 18)}
                                             xy-model [] obs-seq))))
          "different seed → different particles"))))

;; ---------------------------------------------------------------------------
;; mh-custom: seeded chain is bit-reproducible (pre-fix: propose/update unkeyed)
;; ---------------------------------------------------------------------------

(def ^:private x-proposal
  (gen [_current-choices]
    (trace :x (dist/gaussian 0 1.5))))

(deftest mh-custom-seeded-determinism
  (testing "same :key → identical custom-proposal MH chain"
    (let [obs (cm/choicemap :y0 (mx/scalar 1.0) :y1 (mx/scalar 1.2))
          run #(mapv x-of (mcmc/mh-custom {:samples 8 :proposal-gf x-proposal
                                           :key (rng/fresh-key 23)}
                                          xy-model [] obs))]
      (is (= (run) (run)) "chains bit-identical under the same seed"))))

;; ---------------------------------------------------------------------------
;; involutive-mh: seeded chain is bit-reproducible (pre-fix: propose unkeyed)
;; ---------------------------------------------------------------------------

(deftest involutive-mh-seeded-determinism
  (testing "same :key → identical involutive MH chain"
    ;; Involution: reflect :x around the auxiliary midpoint m — an
    ;; involution in (trace, aux) jointly: applying it twice is identity.
    (let [aux-proposal (dyn/auto-key
                        (gen [_current-choices]
                          (trace :m (dist/gaussian 0 1))))
          involution (fn [trace-cm aux-cm]
                       (let [x (cm/get-choice trace-cm [:x])
                             m (cm/get-choice aux-cm [:m])
                             x' (mx/subtract (mx/multiply (mx/scalar 2.0) m) x)]
                         [(cm/set-choice trace-cm [:x] x')
                          aux-cm]))
          obs (cm/choicemap :y0 (mx/scalar 1.0) :y1 (mx/scalar 1.2))
          run #(mapv x-of (mcmc/involutive-mh {:samples 8
                                               :proposal-gf aux-proposal
                                               :involution involution
                                               :key (rng/fresh-key 29)}
                                              xy-model [] obs))]
      (is (= (run) (run)) "chains bit-identical under the same seed"))))

;; ---------------------------------------------------------------------------
;; Regression guards: paths that were deterministic-but-correlated pre-fix
;; ---------------------------------------------------------------------------

(deftest nuts-seeded-determinism
  (testing "same :key → identical NUTS samples (guards the disjoint
            tree/accept key streams)"
    (let [obs (cm/choicemap :y0 (mx/scalar 1.0) :y1 (mx/scalar 1.2))
          run #(mcmc/nuts {:samples 5 :burn 5 :adapt-step-size false
                           :step-size 0.2 :addresses [:x] :compile? false
                           :device :cpu :key (rng/fresh-key 31)}
                          xy-model [] obs)]
      (is (= (run) (run)) "NUTS samples bit-identical under the same seed"))))

(deftest elliptical-slice-seeded-determinism
  (testing "same :key → identical elliptical slice chain (guards the
            shrink-loop key being disjoint from k1/k2/k3)"
    (let [obs (cm/choicemap :y0 (mx/scalar 1.0) :y1 (mx/scalar 1.2))
          run #(mapv x-of (mcmc/elliptical-slice {:samples 6 :selection [:x]
                                                  :prior-std 1.0
                                                  :key (rng/fresh-key 37)}
                                                 xy-model [] obs))]
      (is (= (run) (run)) "chains bit-identical under the same seed"))))

;; ---------------------------------------------------------------------------
;; cSMC: seeded run is bit-reproducible (genmlx-g5ys: the init-step generate
;; leaked auto-key entropy — fixed by threading per-particle keys like smc).
;; This is a POSITIVE reproducibility test: it FAILED pre-fix (csmc gave
;; different log-ML/particles every run under a fixed :key) and PASSES post-fix.
;; ---------------------------------------------------------------------------

(deftest csmc-seeded-determinism
  (testing "same :key → identical cSMC log-ML and retained particles"
    (let [obs-seq [(cm/choicemap :y0 (mx/scalar 1.0))
                   (cm/choicemap :y1 (mx/scalar 1.2))]
          ref-trace (p/simulate (dyn/with-key xy-model (rng/fresh-key 5)) [])
          run #(smc/csmc {:particles 16 :key (rng/fresh-key 17)
                          :rejuvenation-steps 2}
                         xy-model [] obs-seq ref-trace)
          r1 (run)
          r2 (run)]
      (is (= (h/realize (:log-ml-estimate r1))
             (h/realize (:log-ml-estimate r2)))
          "log-ML bit-identical under the same seed")
      (is (= (mapv x-of (:traces r1)) (mapv x-of (:traces r2)))
          "particle :x values bit-identical under the same seed")
      (is (not= (mapv x-of (:traces r1))
                (mapv x-of (:traces (smc/csmc {:particles 16
                                               :key (rng/fresh-key 18)
                                               :rejuvenation-steps 2}
                                              xy-model [] obs-seq ref-trace))))
          "different seed → different particles"))))

(deftest importance-resampling-seeded-determinism
  (testing "same :key → identical resampled traces (guards the
            generate/resample key streams being disjoint)"
    (let [obs (cm/choicemap :y0 (mx/scalar 1.0) :y1 (mx/scalar 1.2))
          run #(mapv x-of (imp/importance-resampling
                           {:samples 10 :particles 30
                            :key (rng/fresh-key 41)}
                           xy-model [] obs))]
      (is (= (run) (run)) "resampled traces bit-identical under the same seed"))))

(cljs.test/run-tests)
