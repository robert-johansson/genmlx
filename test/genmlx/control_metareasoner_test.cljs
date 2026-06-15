;; @tier medium
(ns genmlx.control-metareasoner-test
  "genmlx-nrkq: the metareasoner (control = agents pointed at computation).
   decision-value oracles, the policy as a generative function, and the myopic
   VOC controller driven over the rfal steppable by the world.proc scheduler."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.control.meta-mdp :as ctrl]
            [genmlx.control.decision-value :as dv]
            [genmlx.inference.steppable :as sp]
            [genmlx.world.proc :as proc]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic]
            [genmlx.gen :refer [gen]]))

;; --------------------------------------------------------------------------
;; decision-value (pure, fast)
;; --------------------------------------------------------------------------
(deftest decision-value-test
  (testing "neg-bayes-risk = -weighted-variance (independent hand oracle)"
    (let [wl {:probs [0.5 0.5] :values [1.0 3.0]}]   ; mean 2, var 1
      (is (h/close? 2.0 (dv/weighted-mean wl) 1e-9))
      (is (h/close? 1.0 (dv/weighted-variance wl) 1e-9))
      (is (h/close? -1.0 (dv/neg-bayes-risk wl) 1e-9))))
  (testing "max-eu = max_a sum_i p_i u(v_i,a) (independent hand oracle)"
    (let [wl {:probs [0.5 0.5] :values [0.0 10.0]}
          util (fn [v a] (if (= a :high) v (- 5 v)))]
      ;; eu-high = 5, eu-low = 0 -> max 5
      (is (h/close? 5.0 (dv/max-eu wl util [:high :low]) 1e-9))))
  (testing "assert-downstream! rejects sampler diagnostics (ESS / log-ML)"
    (is (thrown? :default (dv/assert-downstream! {:ess 0.5 :log-ml-estimate -3.0})))
    (is (= {:eu [1.0 0.0]} (dv/assert-downstream! {:eu [1.0 0.0]})) "a real meta-state passes through")))

;; --------------------------------------------------------------------------
;; the policy is a generative function over the meta-action
;; --------------------------------------------------------------------------
(deftest policy-is-a-gf-test
  (testing "p/simulate on the policy yields a Trace with a :meta-action; ##Inf is argmax"
    (let [c (ctrl/make-metareasoner {:alpha ##Inf :latent-addr :mu})
          tr (p/simulate (:policy c) [{:eu [1.0 0.0]}])]
      (is (some? (cm/get-value (cm/get-submap (:choices tr) :meta-action))) "policy traces :meta-action")
      ;; argmax over [continue=1.0, stop=0.0] -> :continue ; over [-1, 0] -> :stop
      (is (= :continue ((:act c) {:eu [1.0 0.0]})) "positive VOC -> continue (argmax)")
      (is (= :stop ((:act c) {:eu [-1.0 0.0]})) "negative VOC -> stop (argmax)")))
  (testing "the return shape mirrors an agent (params/policy/act/decision-value)"
    (let [c (ctrl/make-metareasoner {:latent-addr :mu})]
      (is (every? #(contains? c %) [:params :policy :act :decision-value :control])))))

(deftest switch-method-fence-test
  (testing "switch-method is deferred for v1.0"
    (is (not (contains? (set ctrl/actions) :switch-method)) ":switch-method absent from the v1.0 action set")
    (is (thrown? :default (ctrl/switch-method-translate {})) "the translation stub throws :not-implemented")))

;; --------------------------------------------------------------------------
;; the VOC controller driven over the rfal steppable by world.proc
;; --------------------------------------------------------------------------
(def s0 3.0)
(def sn 1.0)
(def obs-vals [1.0 1.5 0.5 1.2 0.8 1.1])
(def T (count obs-vals))

(def model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 s0))]
      (doseq [i (range T)]
        (trace (keyword (str "y" i)) (dist/gaussian mu sn)))
      mu)))

(def obs-seq
  (mapv (fn [i] (cm/choicemap (keyword (str "y" i)) (mx/scalar (nth obs-vals i)))) (range T)))

;; exact normal-normal posterior mean over mu given all T obs
(def exact-mu
  (let [prec (+ (/ 1.0 (* s0 s0)) (/ T (* sn sn)))
        num (/ (reduce + obs-vals) (* sn sn))]
    (/ num prec)))

(defn- base-steppable [seed]
  (let [o {:particles 3000 :ess-threshold 0.5 :key (rng/fresh-key seed)}]
    {:init (fn [] (sp/init-state model [] obs-seq o))
     :step sp/step
     :done? sp/done?
     :best (fn [s] (dv/weighted-mean (dv/weighted-latent s :mu)))}))

(deftest voc-controller-test
  (testing "lambda=0 (compute free): never stops early, folds all data, decision ~ exact posterior mean"
    (let [c (ctrl/make-metareasoner {:alpha ##Inf :lambda 0.0 :latent-addr :mu})
          steppable ((:control c) (base-steppable 7))
          r (proc/with-deadline (:init steppable) (:step steppable) (:done? steppable) (:best steppable)
              {:budget-ms 120000 :chunk 1 :gc-every 1})]
      (is (>= (:control-steps (:state r)) (dec T))
          "folded (nearly) all data when compute is free (hysteresis tolerates noise)")
      ;; The decision moved from the prior (mean 0) toward the data/posterior.
      ;; (Tight estimator accuracy vs the exact mean is the gdtq microbench's job,
      ;; with seeds + CIs; the growing-obs SMC here degenerates without
      ;; rejuvenation, so use a loose band + a directional check.)
      (is (< (js/Math.abs (- (:best r) exact-mu))
             (js/Math.abs (- (:best r) 0.0)))
          (str "decision " (:best r) " is closer to the exact posterior mean "
               exact-mu " than to the prior mean 0"))
      (is (h/close? exact-mu (:best r) 0.4)
          (str "controller decision " (:best r) " ~ exact posterior mean " exact-mu))))
  (testing "large lambda (compute costly): VOC-stops BEFORE folding all data"
    (let [c (ctrl/make-metareasoner {:alpha ##Inf :lambda 1.0 :latent-addr :mu})
          steppable ((:control c) (base-steppable 7))
          r (proc/with-deadline (:init steppable) (:step steppable) (:done? steppable) (:best steppable)
              {:budget-ms 120000 :chunk 1 :gc-every 1})]
      (is (true? (:stopped? (:state r))) "controller VOC-stopped (the value gain did not justify the compute)")
      (is (< (:control-steps (:state r)) T) "stopped before folding all data")
      (is (>= (:control-steps (:state r)) 1) "took at least one step (anytime)"))))

(cljs.test/run-tests)
