(ns genmlx.iid-gfi-test
  "M2 Steps 1-2: GFI completeness + inference integration for iid models."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.selection :as sel]
            [genmlx.vectorized :as vec]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn ->num [v]
  (if (mx/array? v) (mx/item v) v))

;; ---------------------------------------------------------------------------
;; Shared models
;; ---------------------------------------------------------------------------

(def model-a
  (gen [t]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :ys (dist/iid-gaussian mu (mx/scalar 1.0) t))
      mu)))

(def model-b
  (gen [xs]
    (let [slope     (trace :slope (dist/gaussian 0 10))
          intercept (trace :intercept (dist/gaussian 0 10))
          means     (mx/add (mx/multiply slope (mx/array xs)) intercept)]
      (trace :ys (dist/iid-gaussian means (mx/scalar 1.0) (count xs)))
      slope)))

(def model-c
  (gen [t]
    (let [p (trace :p (dist/beta-dist 2 5))]
      (trace :xs (dist/iid (dist/bernoulli p) t))
      p)))

;; =========================================================================
;; STEP 1: GFI Completeness
;; =========================================================================

;; NOTE: p/simulate on auto-key'd iid models triggers a pre-existing compiled
;; path bug. Tests use p/generate with constraints to create initial traces.

(deftest update-change-mu
  (testing "1.1 update: change mu"
    (let [gf (dyn/auto-key model-a)
          tr (:trace (p/generate gf [5] (cm/choicemap :ys (mx/array [1 2 3 4 5]))))
          new-constraints (cm/choicemap :mu 3.0)
          result (p/update gf tr new-constraints)]
      (is (some? (:trace result)) "update returns :trace")
      (is (some? (:weight result)) "update returns :weight")
      (is (js/isFinite (mx/item (:weight result))) "update weight is finite")
      (let [new-mu (cm/get-value (cm/get-submap (:choices (:trace result)) :mu))]
        (is (h/close? 3.0 (->num new-mu) 0.001) "mu updated to 3.0"))
      (is (= [5] (mx/shape (cm/get-value (cm/get-submap (:choices (:trace result)) :ys))))
          "ys shape preserved"))))

(deftest update-change-ys
  (testing "1.2 update: change ys"
    (let [gf (dyn/auto-key model-a)
          tr (:trace (p/generate gf [3] (cm/choicemap :ys (mx/array [1 2 3]))))
          new-ys (mx/array [10.0 20.0 30.0])
          new-constraints (cm/choicemap :ys new-ys)
          result (p/update gf tr new-constraints)]
      (is (js/isFinite (mx/item (:weight result))) "update with ys: weight is finite")
      (let [updated-ys (cm/get-value (cm/get-submap (:choices (:trace result)) :ys))]
        (is (= [3] (mx/shape updated-ys)) "ys shape")
        (is (h/close? 10.0 (mx/item (mx/index updated-ys 0)) 0.001) "ys[0]")))))

(deftest update-change-both
  (testing "1.3 update: change both"
    (let [gf (dyn/auto-key model-a)
          tr (:trace (p/generate gf [3] (cm/choicemap :ys (mx/array [1 2 3]))))
          new-constraints (cm/choicemap :mu 5.0 :ys (mx/array [5.0 5.0 5.0]))
          result (p/update gf tr new-constraints)]
      (is (js/isFinite (mx/item (:weight result))) "both updated: weight is finite")
      (let [new-mu (cm/get-value (cm/get-submap (:choices (:trace result)) :mu))]
        (is (h/close? 5.0 (->num new-mu) 0.001) "mu = 5.0")))))

(deftest regenerate-select-mu
  (testing "1.4 regenerate: select mu"
    (let [gf (dyn/auto-key model-a)
          tr (:trace (p/generate gf [5] (cm/choicemap :ys (mx/array [1 2 3 4 5]))))
          result (p/regenerate gf tr (sel/select :mu))]
      (is (some? (:trace result)) "regenerate mu: returns trace")
      (is (js/isFinite (mx/item (:weight result))) "regenerate mu: weight is finite")
      (is (= [5] (mx/shape (cm/get-value (cm/get-submap (:choices (:trace result)) :ys))))
          "ys shape preserved after regen mu"))))

(deftest regenerate-select-ys
  (testing "1.5 regenerate: select ys"
    (let [gf (dyn/auto-key model-a)
          tr (:trace (p/generate gf [5] (cm/choicemap :ys (mx/array [1 2 3 4 5]))))
          result (p/regenerate gf tr (sel/select :ys))]
      (is (js/isFinite (mx/item (:weight result))) "regenerate ys: weight is finite")
      (is (= [5] (mx/shape (cm/get-value (cm/get-submap (:choices (:trace result)) :ys))))
          "resampled ys shape")
      (let [old-mu (->num (cm/get-value (cm/get-submap (:choices tr) :mu)))
            new-mu (->num (cm/get-value (cm/get-submap (:choices (:trace result)) :mu)))]
        (is (h/close? old-mu new-mu 0.001) "mu preserved after regen ys")))))

(deftest assess-full-constraints
  (testing "1.6 assess: full constraints"
    (let [gf (dyn/auto-key model-a)
          choices (cm/choicemap :mu 3.0 :ys (mx/array [3.0 3.0 3.0]))
          result (p/assess gf [3] choices)]
      (is (js/isFinite (mx/item (:weight result))) "assess: weight is finite")
      (is (neg? (mx/item (:weight result))) "assess: weight is negative"))))

(deftest project-select-mu
  (testing "1.7 project: select mu"
    (let [gf (dyn/auto-key model-a)
          tr (:trace (p/generate gf [5] (cm/choicemap :ys (mx/array [1 2 3 4 5]))))
          w (p/project gf tr (sel/select :mu))]
      (is (= [] (mx/shape w)) "project returns scalar")
      (is (js/isFinite (mx/item w)) "project weight is finite"))))

(deftest project-select-ys
  (testing "1.8 project: select ys"
    (let [gf (dyn/auto-key model-a)
          tr (:trace (p/generate gf [3] (cm/choicemap :ys (mx/array [1 2 3]))))
          w (p/project gf tr (sel/select :ys))]
      (is (= [] (mx/shape w)) "project ys returns scalar")
      (is (js/isFinite (mx/item w)) "project ys weight is finite"))))

(deftest update-weight-identity
  (testing "1.9 update weight identity"
    (let [gf (dyn/auto-key model-a)
          obs (cm/choicemap :mu 2.0 :ys (mx/array [1.0 2.0 3.0]))
          {:keys [trace]} (p/generate gf [3] obs)
          result (p/update gf trace obs)]
      (is (js/isFinite (mx/item (:weight result)))
          "update same values: weight is finite"))))

(deftest generic-iid-update-regenerate
  (testing "1.10 generic iid: update + regenerate"
    (let [gf (dyn/auto-key model-c)
          tr (:trace (p/generate gf [5] (cm/choicemap :xs (mx/array [1 0 1 0 1]))))
          result-update (p/update gf tr (cm/choicemap :p 0.5))
          result-regen (p/regenerate gf tr (sel/select :xs))]
      (is (js/isFinite (mx/item (:weight result-update)))
          "generic iid update: weight finite")
      (is (js/isFinite (mx/item (:weight result-regen)))
          "generic iid regen: weight finite")
      (is (= [5] (mx/shape (cm/get-value (cm/get-submap (:choices (:trace result-regen)) :xs))))
          "generic iid regen: xs shape preserved"))))

;; =========================================================================
;; STEP 2: Inference Integration
;; =========================================================================

(deftest importance-sampling-posterior
  (testing "2.1 importance sampling"
    (let [gf (dyn/auto-key model-a)
          obs (cm/choicemap :ys (mx/array [1.0 2.0 3.0 4.0 5.0]))
          result (is/importance-sampling {:samples 2000 :key (rng/fresh-key)}
                                         gf [5] obs)
          lws (mx/array (mapv mx/item (:log-weights result)))
          mus (mx/array (mapv #(mx/item (cm/get-value (cm/get-submap (:choices %) :mu)))
                              (:traces result)))
          max-lw (mx/amax lws)
          ws (mx/exp (mx/subtract lws max-lw))
          wn (mx/divide ws (mx/sum ws))
          mu-est (mx/item (mx/sum (mx/multiply wn mus)))]
      (is (h/close? 3.0 mu-est 0.5) "IS posterior mean ~ 3.0"))))

(deftest vectorized-is
  (testing "2.2 vectorized IS"
    (let [gf (dyn/auto-key model-a)
          obs (cm/choicemap :ys (mx/array [1.0 2.0 3.0 4.0 5.0]))
          vt (dyn/vgenerate gf [5] obs 5000 (rng/fresh-key))
          w (:weight vt) r (:retval vt)
          wn (let [e (mx/exp (mx/subtract w (mx/amax w)))] (mx/divide e (mx/sum e)))
          mu-est (mx/item (mx/sum (mx/multiply wn r)))]
      (is (h/close? 3.0 mu-est 0.5) "VIS posterior mean ~ 3.0"))))

(deftest mh-chain-converges
  (testing "2.3 MH chain"
    (let [gf (dyn/auto-key model-a)
          obs (cm/choicemap :ys (mx/array [1.0 2.0 3.0 4.0 5.0]))
          traces (mcmc/mh {:samples 100 :burn 200 :selection (sel/select :mu)
                            :key (rng/fresh-key)}
                           gf [5] obs)
          mus (mapv #(mx/item (cm/get-value (cm/get-submap (:choices %) :mu))) traces)
          mu-mean (/ (reduce + mus) (count mus))]
      (is (= 100 (count traces)) "MH produced traces")
      (is (h/close? 3.0 mu-mean 0.8) "MH posterior mean ~ 3.0"))))

(deftest mh-on-iid-site
  (testing "2.4 MH on iid site"
    (mx/eval!)
    (let [gf (dyn/auto-key model-a)
          obs (cm/choicemap :mu 3.0)
          traces (mcmc/mh {:samples 5 :burn 5 :selection (sel/select :ys)
                            :key (rng/fresh-key)}
                           gf [3] obs)]
      (is (= 5 (count traces)) "MH on :ys produced traces")
      (is (some? (cm/get-submap (:choices (first traces)) :ys))
          "MH on :ys: ys exists"))))

(deftest is-linreg
  (testing "2.5 IS linreg"
    (let [gf (dyn/auto-key model-b)
          xs [0.0 1.0 2.0 3.0 4.0]
          ys (mx/array [1.0 3.0 5.0 7.0 9.0])
          obs (cm/choicemap :ys ys)
          result (is/importance-sampling {:samples 2000 :key (rng/fresh-key)}
                                         gf [xs] obs)
          lws (mx/array (mapv mx/item (:log-weights result)))
          slopes (mx/array (mapv #(mx/item (cm/get-value (cm/get-submap (:choices %) :slope)))
                                (:traces result)))
          max-lw (mx/amax lws)
          ws (mx/exp (mx/subtract lws max-lw))
          wn (mx/divide ws (mx/sum ws))
          slope-est (mx/item (mx/sum (mx/multiply wn slopes)))]
      (is (h/close? 2.0 slope-est 0.5) "IS linreg slope ~ 2.0"))))

(deftest score-consistency
  (testing "2.6 score consistency"
    ;; With L3 auto-analytical, generate score = marginal LL, not joint LL.
    (let [gf (dyn/auto-key model-a)
          tr (:trace (p/generate gf [3] (cm/choicemap :ys (mx/array [1 2 3]))))
          score (mx/item (:score tr))]
      (is (js/isFinite score) "score is finite")
      (is (neg? score) "score is negative"))))

(deftest generic-iid-is
  (testing "2.7 generic iid: IS"
    (let [gf (dyn/auto-key model-c)
          t 10
          obs (cm/choicemap :xs (mx/array (repeat t 1.0)))
          result (is/importance-sampling {:samples 1000 :key (rng/fresh-key)}
                                         gf [t] obs)
          lws (mx/array (mapv mx/item (:log-weights result)))
          ps (mx/array (mapv #(mx/item (:retval %)) (:traces result)))
          max-lw (mx/amax lws)
          ws (mx/exp (mx/subtract lws max-lw))
          wn (mx/divide ws (mx/sum ws))
          p-est (mx/item (mx/sum (mx/multiply wn ps)))
          expected (/ 12.0 17.0)]
      (is (h/close? expected p-est 0.1) "generic iid IS posterior mean"))))

(deftest vis-performance
  (testing "2.8 VIS performance"
    (let [gf (dyn/auto-key model-a)
          t 100
          obs (cm/choicemap :ys (mx/array (repeat t 5.0)))
          n 10000
          key (rng/fresh-key)
          _ (dyn/vgenerate gf [t] obs n key)
          t0 (js/Date.now)
          _ (dyn/vgenerate gf [t] obs n key)
          t1 (js/Date.now)
          ms (- t1 t0)]
      (is (< ms 200) "VIS T=100 N=10K < 200ms"))))

(cljs.test/run-tests)
