(ns genmlx.auto-analytical-test
  "Tests for Level 3 address-based analytical handlers (WP-1).
   Verifies marginal LL, posterior, weight/score accounting against conjugate.cljs."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.inference.auto-analytical :as aa]
            [genmlx.inference.conjugate :as conjugate]
            [genmlx.conjugacy :as conj]
            [genmlx.handler :as handler]
            [genmlx.runtime :as rt]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.choicemap :as cm]
            [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]))

;; ---------------------------------------------------------------------------
;; Section 1: Update step functions match conjugate.cljs
;; ---------------------------------------------------------------------------

(deftest nn-update-step-matches-conjugate
  (testing "NN update step"
    (let [prior {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
          obs (mx/scalar 3.0)
          obs-var (mx/scalar 1.0)
          ours (aa/nn-update-step prior obs obs-var)
          ref (conjugate/nn-update prior obs obs-var (mx/scalar 1.0))]
      (mx/eval!)
      (is (h/close? (mx/item (:ll ref)) (mx/item (:ll ours)) 1e-10) "NN-step: ll matches")
      (is (h/close? (mx/item (:mean (:posterior ref))) (mx/item (:mean ours)) 1e-10) "NN-step: mean matches")
      (is (h/close? (mx/item (:var (:posterior ref))) (mx/item (:var ours)) 1e-10) "NN-step: var matches"))))

(deftest bb-update-step-matches-conjugate
  (testing "BB update step"
    (let [prior {:alpha (mx/scalar 2.0) :beta (mx/scalar 5.0)}
          obs (mx/scalar 1.0)
          ours (aa/bb-update-step prior obs)
          ref (conjugate/bb-update prior obs (mx/scalar 1.0))]
      (mx/eval!)
      (is (h/close? (mx/item (:ll ref)) (mx/item (:ll ours)) 1e-10) "BB-step: ll matches")
      (is (h/close? (mx/item (:alpha (:posterior ref))) (mx/item (:alpha ours)) 1e-10) "BB-step: alpha matches")
      (is (h/close? (mx/item (:beta (:posterior ref))) (mx/item (:beta ours)) 1e-10) "BB-step: beta matches"))))

(deftest gp-update-step-matches-conjugate
  (testing "GP update step"
    (let [prior {:shape (mx/scalar 2.0) :rate (mx/scalar 1.0)}
          obs (mx/scalar 3.0)
          ours (aa/gp-update-step prior obs)
          ref (conjugate/gp-update prior obs (mx/scalar 1.0))]
      (mx/eval!)
      (is (h/close? (mx/item (:ll ref)) (mx/item (:ll ours)) 1e-10) "GP-step: ll matches")
      (is (h/close? (mx/item (:shape (:posterior ref))) (mx/item (:shape ours)) 1e-10) "GP-step: shape matches")
      (is (h/close? (mx/item (:rate (:posterior ref))) (mx/item (:rate ours)) 1e-10) "GP-step: rate matches"))))

(deftest nn-sequential-updates
  (testing "NN sequential updates (5 observations)"
    (let [obs-vals [3.0 5.0 1.0 4.0 2.0]
          final-ours (reduce
                       (fn [post obs]
                         (let [r (aa/nn-update-step post (mx/scalar obs) (mx/scalar 1.0))]
                           {:mean (:mean r) :var (:var r) :ll (mx/add (:ll post (mx/scalar 0.0)) (:ll r))}))
                       {:mean (mx/scalar 0.0) :var (mx/scalar 100.0) :ll (mx/scalar 0.0)}
                       obs-vals)
          final-ref (reduce
                      (fn [state obs]
                        (let [r (conjugate/nn-update (:posterior state) (mx/scalar obs) (mx/scalar 1.0) (mx/scalar 1.0))]
                          {:posterior (:posterior r) :ll (mx/add (:ll state) (:ll r))}))
                      {:posterior {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)} :ll (mx/scalar 0.0)}
                      obs-vals)]
      (mx/eval!)
      (is (h/close? (mx/item (:ll final-ref)) (mx/item (:ll final-ours)) 1e-6) "NN-5obs: total ll matches")
      (is (h/close? (mx/item (:mean (:posterior final-ref))) (mx/item (:mean final-ours)) 1e-6) "NN-5obs: final mean matches")
      (is (h/close? (mx/item (:var (:posterior final-ref))) (mx/item (:var final-ours)) 1e-6) "NN-5obs: final var matches"))))

;; ---------------------------------------------------------------------------
;; Section 2: make-address-dispatch
;; ---------------------------------------------------------------------------

(deftest make-address-dispatch-test
  (testing "dispatches to intercepted handlers"
    (let [handlers {:a (fn [state addr dist] [:intercepted-a state])
                    :b (fn [state addr dist] [:intercepted-b state])}
          base (fn [state addr dist] [:base state])
          dispatch (aa/make-address-dispatch base handlers)]
      (is (= :intercepted-a (first (dispatch {} :a nil))) "dispatch :a intercepted")
      (is (= :intercepted-b (first (dispatch {} :b nil))) "dispatch :b intercepted")
      (is (= :base (first (dispatch {} :c nil))) "dispatch :c falls through")))

  (testing "nil-return fallthrough"
    (let [handlers {:a (fn [state addr dist] nil)}
          base (fn [state addr dist] [:base state])
          dispatch (aa/make-address-dispatch base handlers)]
      (is (= :base (first (dispatch {} :a nil))) "nil return falls through"))))

;; ---------------------------------------------------------------------------
;; Section 3: NN handler via transitions
;; ---------------------------------------------------------------------------

(deftest nn-handler-single-observation
  (testing "NN handler via transitions — single observation"
    (let [handlers (aa/build-auto-handlers
                     [{:prior-addr :mu :obs-addr :y :family :normal-normal}])
          transition (aa/make-address-dispatch handler/generate-transition handlers)
          constraints (-> cm/EMPTY (cm/set-value :y (mx/scalar 3.0)))
          init {:choices cm/EMPTY :score (mx/scalar 0.0) :weight (mx/scalar 0.0)
                :key (rng/fresh-key 42) :constraints constraints :auto-posteriors {}}
          [mu s1] (transition init :mu (dist/gaussian (mx/scalar 0.0) (mx/scalar 10.0)))
          [y s2] (transition s1 :y (dist/gaussian mu (mx/scalar 1.0)))
          ref (conjugate/nn-update {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
                                   (mx/scalar 3.0) (mx/scalar 1.0) (mx/scalar 1.0))]
      (mx/eval!)
      (is (h/close? (mx/item (:ll ref)) (mx/item (:weight s2)) 1e-10) "NN-1obs: weight = marginal LL")
      (is (h/close? (mx/item (:ll ref)) (mx/item (:score s2)) 1e-10) "NN-1obs: score = marginal LL")
      (is (h/close? (mx/item (:mean (:posterior ref)))
                    (mx/item (get-in s2 [:auto-posteriors :mu :mean])) 1e-10) "NN-1obs: posterior mean")
      (is (h/close? (mx/item (:var (:posterior ref)))
                    (mx/item (get-in s2 [:auto-posteriors :mu :var])) 1e-10) "NN-1obs: posterior var")
      (is (h/close? (mx/item (:mean (:posterior ref)))
                    (mx/item (cm/get-value (cm/get-submap (:choices s2) :mu))) 1e-10) "NN-1obs: choices :mu = posterior mean")
      (is (h/close? 3.0 (mx/item (cm/get-value (cm/get-submap (:choices s2) :y))) 1e-10) "NN-1obs: choices :y = 3.0"))))

(deftest nn-handler-three-observations
  (testing "NN handler via transitions — three observations"
    (let [handlers (aa/build-auto-handlers
                     [{:prior-addr :mu :obs-addr :y1 :family :normal-normal}
                      {:prior-addr :mu :obs-addr :y2 :family :normal-normal}
                      {:prior-addr :mu :obs-addr :y3 :family :normal-normal}])
          transition (aa/make-address-dispatch handler/generate-transition handlers)
          constraints (-> cm/EMPTY
                        (cm/set-value :y1 (mx/scalar 3.0))
                        (cm/set-value :y2 (mx/scalar 5.0))
                        (cm/set-value :y3 (mx/scalar 1.0)))
          init {:choices cm/EMPTY :score (mx/scalar 0.0) :weight (mx/scalar 0.0)
                :key (rng/fresh-key 42) :constraints constraints :auto-posteriors {}}
          [mu s1] (transition init :mu (dist/gaussian (mx/scalar 0.0) (mx/scalar 10.0)))
          [_ s2] (transition s1 :y1 (dist/gaussian mu (mx/scalar 1.0)))
          [_ s3] (transition s2 :y2 (dist/gaussian mu (mx/scalar 1.0)))
          [_ s4] (transition s3 :y3 (dist/gaussian mu (mx/scalar 1.0)))
          r1 (conjugate/nn-update {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
                                  (mx/scalar 3.0) (mx/scalar 1.0) (mx/scalar 1.0))
          r2 (conjugate/nn-update (:posterior r1) (mx/scalar 5.0) (mx/scalar 1.0) (mx/scalar 1.0))
          r3 (conjugate/nn-update (:posterior r2) (mx/scalar 1.0) (mx/scalar 1.0) (mx/scalar 1.0))
          ref-ll (+ (mx/item (:ll r1)) (mx/item (:ll r2)) (mx/item (:ll r3)))]
      (mx/eval!)
      (is (h/close? ref-ll (mx/item (:weight s4)) 1e-6) "NN-3obs: total weight matches ref")
      (is (h/close? (mx/item (:mean (:posterior r3)))
                    (mx/item (get-in s4 [:auto-posteriors :mu :mean])) 1e-6) "NN-3obs: posterior mean"))))

;; ---------------------------------------------------------------------------
;; Section 4: BB Handler
;; ---------------------------------------------------------------------------

(deftest bb-handler-single-observation
  (testing "BB handler — single observation"
    (let [handlers (aa/build-auto-handlers
                     [{:prior-addr :p :obs-addr :x :family :beta-bernoulli}])
          transition (aa/make-address-dispatch handler/generate-transition handlers)
          constraints (-> cm/EMPTY (cm/set-value :x (mx/scalar 1.0)))
          init {:choices cm/EMPTY :score (mx/scalar 0.0) :weight (mx/scalar 0.0)
                :key (rng/fresh-key 42) :constraints constraints :auto-posteriors {}}
          [p s1] (transition init :p (dist/beta-dist (mx/scalar 2.0) (mx/scalar 5.0)))
          [x s2] (transition s1 :x (dist/bernoulli p))
          ref (conjugate/bb-update {:alpha (mx/scalar 2.0) :beta (mx/scalar 5.0)}
                                   (mx/scalar 1.0) (mx/scalar 1.0))]
      (mx/eval!)
      (is (h/close? (mx/item (:ll ref)) (mx/item (:weight s2)) 1e-10) "BB-1obs: weight = marginal LL")
      (is (h/close? (mx/item (:alpha (:posterior ref)))
                    (mx/item (get-in s2 [:auto-posteriors :p :alpha])) 1e-10) "BB-1obs: posterior alpha")
      (is (h/close? (mx/item (:beta (:posterior ref)))
                    (mx/item (get-in s2 [:auto-posteriors :p :beta])) 1e-10) "BB-1obs: posterior beta"))))

(deftest bb-handler-three-observations
  (testing "BB handler — three observations"
    (let [handlers (aa/build-auto-handlers
                     [{:prior-addr :p :obs-addr :x1 :family :beta-bernoulli}
                      {:prior-addr :p :obs-addr :x2 :family :beta-bernoulli}
                      {:prior-addr :p :obs-addr :x3 :family :beta-bernoulli}])
          transition (aa/make-address-dispatch handler/generate-transition handlers)
          constraints (-> cm/EMPTY
                        (cm/set-value :x1 (mx/scalar 1.0))
                        (cm/set-value :x2 (mx/scalar 0.0))
                        (cm/set-value :x3 (mx/scalar 1.0)))
          init {:choices cm/EMPTY :score (mx/scalar 0.0) :weight (mx/scalar 0.0)
                :key (rng/fresh-key 42) :constraints constraints :auto-posteriors {}}
          [p s1] (transition init :p (dist/beta-dist (mx/scalar 2.0) (mx/scalar 5.0)))
          [_ s2] (transition s1 :x1 (dist/bernoulli p))
          [_ s3] (transition s2 :x2 (dist/bernoulli p))
          [_ s4] (transition s3 :x3 (dist/bernoulli p))
          r1 (conjugate/bb-update {:alpha (mx/scalar 2.0) :beta (mx/scalar 5.0)}
                                  (mx/scalar 1.0) (mx/scalar 1.0))
          r2 (conjugate/bb-update (:posterior r1) (mx/scalar 0.0) (mx/scalar 1.0))
          r3 (conjugate/bb-update (:posterior r2) (mx/scalar 1.0) (mx/scalar 1.0))
          ref-ll (+ (mx/item (:ll r1)) (mx/item (:ll r2)) (mx/item (:ll r3)))]
      (mx/eval!)
      (is (h/close? ref-ll (mx/item (:weight s4)) 1e-6) "BB-3obs: total weight matches ref"))))

;; ---------------------------------------------------------------------------
;; Section 5: GP Handler
;; ---------------------------------------------------------------------------

(deftest gp-handler-single-observation
  (testing "GP handler — single observation"
    (let [handlers (aa/build-auto-handlers
                     [{:prior-addr :rate :obs-addr :count :family :gamma-poisson}])
          transition (aa/make-address-dispatch handler/generate-transition handlers)
          constraints (-> cm/EMPTY (cm/set-value :count (mx/scalar 3.0)))
          init {:choices cm/EMPTY :score (mx/scalar 0.0) :weight (mx/scalar 0.0)
                :key (rng/fresh-key 42) :constraints constraints :auto-posteriors {}}
          [rate s1] (transition init :rate (dist/gamma-dist (mx/scalar 2.0) (mx/scalar 1.0)))
          [cnt s2] (transition s1 :count (dist/poisson rate))
          ref (conjugate/gp-update {:shape (mx/scalar 2.0) :rate (mx/scalar 1.0)}
                                   (mx/scalar 3.0) (mx/scalar 1.0))]
      (mx/eval!)
      (is (h/close? (mx/item (:ll ref)) (mx/item (:weight s2)) 1e-10) "GP-1obs: weight = marginal LL")
      (is (h/close? (mx/item (:shape (:posterior ref)))
                    (mx/item (get-in s2 [:auto-posteriors :rate :shape])) 1e-10) "GP-1obs: posterior shape")
      (is (h/close? (mx/item (:rate (:posterior ref)))
                    (mx/item (get-in s2 [:auto-posteriors :rate :rate])) 1e-10) "GP-1obs: posterior rate"))))

;; ---------------------------------------------------------------------------
;; Section 6: GE Handler
;; ---------------------------------------------------------------------------

(deftest ge-handler-single-observation
  (testing "GE handler — single observation"
    (let [handlers (aa/build-auto-handlers
                     [{:prior-addr :rate :obs-addr :x :family :gamma-exponential}])
          transition (aa/make-address-dispatch handler/generate-transition handlers)
          constraints (-> cm/EMPTY (cm/set-value :x (mx/scalar 2.5)))
          init {:choices cm/EMPTY :score (mx/scalar 0.0) :weight (mx/scalar 0.0)
                :key (rng/fresh-key 42) :constraints constraints :auto-posteriors {}}
          [rate s1] (transition init :rate (dist/gamma-dist (mx/scalar 3.0) (mx/scalar 2.0)))
          [x s2] (transition s1 :x (dist/exponential rate))
          ref-ll (+ (js/Math.log 3.0) (* 3.0 (js/Math.log 2.0))
                   (* -4.0 (js/Math.log 4.5)))]
      (mx/eval!)
      (is (h/close? ref-ll (mx/item (:weight s2)) 1e-6) "GE-1obs: weight = marginal LL")
      (is (h/close? 4.0 (mx/item (get-in s2 [:auto-posteriors :rate :shape])) 1e-10) "GE-1obs: posterior shape")
      (is (h/close? 4.5 (mx/item (get-in s2 [:auto-posteriors :rate :rate])) 1e-10) "GE-1obs: posterior rate"))))

;; ---------------------------------------------------------------------------
;; Section 7: Fallthrough behavior
;; ---------------------------------------------------------------------------

(deftest fallthrough-behavior
  (testing "unconstrained obs falls through to standard generate"
    (let [handlers (aa/build-auto-handlers
                     [{:prior-addr :mu :obs-addr :y :family :normal-normal}])
          transition (aa/make-address-dispatch handler/generate-transition handlers)
          init {:choices cm/EMPTY :score (mx/scalar 0.0) :weight (mx/scalar 0.0)
                :key (rng/fresh-key 42) :constraints cm/EMPTY :auto-posteriors {}}
          [mu s1] (transition init :mu (dist/gaussian (mx/scalar 0.0) (mx/scalar 10.0)))
          [y s2] (transition s1 :y (dist/gaussian mu (mx/scalar 1.0)))]
      (mx/eval!)
      (is (h/close? 0.0 (mx/item (:weight s2)) 1e-10) "fallthrough: weight = 0 (no constraint)")
      (is (some? y) "fallthrough: y was sampled (not nil)")))

  (testing "non-intercepted address uses base handler"
    (let [handlers (aa/build-auto-handlers
                     [{:prior-addr :mu :obs-addr :y :family :normal-normal}])
          transition (aa/make-address-dispatch handler/generate-transition handlers)
          constraints (-> cm/EMPTY (cm/set-value :z (mx/scalar 5.0)))
          init {:choices cm/EMPTY :score (mx/scalar 0.0) :weight (mx/scalar 0.0)
                :key (rng/fresh-key 42) :constraints constraints :auto-posteriors {}}
          [z s1] (transition init :z (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))]
      (mx/eval!)
      (is (h/close? 5.0 (mx/item z) 1e-10) "non-intercepted: z = 5.0 (constrained)")
      (is (not= 0.0 (mx/item (:weight s1))) "non-intercepted: weight nonzero"))))

;; ---------------------------------------------------------------------------
;; Section 8: Mixed model (NN + BB in same model)
;; ---------------------------------------------------------------------------

(deftest mixed-handlers
  (testing "NN + BB in same model"
    (let [handlers (aa/build-auto-handlers
                     [{:prior-addr :mu :obs-addr :y :family :normal-normal}
                      {:prior-addr :p :obs-addr :x :family :beta-bernoulli}])
          transition (aa/make-address-dispatch handler/generate-transition handlers)
          constraints (-> cm/EMPTY
                        (cm/set-value :y (mx/scalar 3.0))
                        (cm/set-value :x (mx/scalar 1.0)))
          init {:choices cm/EMPTY :score (mx/scalar 0.0) :weight (mx/scalar 0.0)
                :key (rng/fresh-key 42) :constraints constraints :auto-posteriors {}}
          [mu s1] (transition init :mu (dist/gaussian (mx/scalar 0.0) (mx/scalar 10.0)))
          [p s2] (transition s1 :p (dist/beta-dist (mx/scalar 2.0) (mx/scalar 5.0)))
          [y s3] (transition s2 :y (dist/gaussian mu (mx/scalar 1.0)))
          [x s4] (transition s3 :x (dist/bernoulli p))
          nn-ref (conjugate/nn-update {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
                                      (mx/scalar 3.0) (mx/scalar 1.0) (mx/scalar 1.0))
          bb-ref (conjugate/bb-update {:alpha (mx/scalar 2.0) :beta (mx/scalar 5.0)}
                                      (mx/scalar 1.0) (mx/scalar 1.0))
          ref-total (+ (mx/item (:ll nn-ref)) (mx/item (:ll bb-ref)))]
      (mx/eval!)
      (is (h/close? ref-total (mx/item (:weight s4)) 1e-6) "mixed: total weight = NN-ll + BB-ll")
      (is (contains? (:auto-posteriors s4) :mu) "mixed: has mu posterior")
      (is (contains? (:auto-posteriors s4) :p) "mixed: has p posterior"))))

;; ---------------------------------------------------------------------------
;; Section 9: Mixed conjugate + non-conjugate sites
;; ---------------------------------------------------------------------------

(deftest conjugate-plus-standard-sites
  (testing "conjugate + standard sites"
    (let [handlers (aa/build-auto-handlers
                     [{:prior-addr :mu :obs-addr :y :family :normal-normal}])
          transition (aa/make-address-dispatch handler/generate-transition handlers)
          constraints (-> cm/EMPTY
                        (cm/set-value :y (mx/scalar 3.0))
                        (cm/set-value :z (mx/scalar 7.0)))
          init {:choices cm/EMPTY :score (mx/scalar 0.0) :weight (mx/scalar 0.0)
                :key (rng/fresh-key 42) :constraints constraints :auto-posteriors {}}
          [mu s1] (transition init :mu (dist/gaussian (mx/scalar 0.0) (mx/scalar 10.0)))
          [y s2] (transition s1 :y (dist/gaussian mu (mx/scalar 1.0)))
          [z s3] (transition s2 :z (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))
          nn-ref (conjugate/nn-update {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
                                      (mx/scalar 3.0) (mx/scalar 1.0) (mx/scalar 1.0))
          z-lp (mx/item (dc/dist-log-prob (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)) (mx/scalar 7.0)))
          expected-weight (+ (mx/item (:ll nn-ref)) z-lp)]
      (mx/eval!)
      (is (h/close? expected-weight (mx/item (:weight s3)) 1e-6) "mixed-std: weight = NN-ll + z-logprob"))))

;; ---------------------------------------------------------------------------
;; Section 10: run-handler integration
;; ---------------------------------------------------------------------------

(deftest run-handler-integration
  (testing "run-handler integration"
    (let [model (dyn/auto-key
                  (gen [sigma]
                    (let [mu (trace :mu (dist/gaussian 0 10))]
                      (trace :y (dist/gaussian mu sigma))
                      mu)))
          pairs (conj/detect-conjugate-pairs (:schema model))
          handlers (aa/build-auto-handlers pairs)
          transition (aa/make-address-dispatch handler/generate-transition handlers)
          constraints (-> cm/EMPTY (cm/set-value :y (mx/scalar 3.0)))
          init {:choices cm/EMPTY :score (mx/scalar 0.0) :weight (mx/scalar 0.0)
                :key (rng/fresh-key 42) :constraints constraints :auto-posteriors {}}
          result (rt/run-handler transition init
                   (fn [rt] (apply (:body-fn model) rt [(mx/scalar 1.0)])))
          ref (conjugate/nn-update {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
                                   (mx/scalar 3.0) (mx/scalar 1.0) (mx/scalar 1.0))]
      (mx/eval!)
      (is (h/close? (mx/item (:ll ref)) (mx/item (:weight result)) 1e-10) "run-handler: weight matches")
      (is (h/close? (mx/item (:ll ref)) (mx/item (:score result)) 1e-10) "run-handler: score matches")
      (is (some? (:retval result)) "run-handler: has retval")
      (is (h/close? 3.0 (mx/item (cm/get-value (cm/get-submap (:choices result) :y))) 1e-10) "run-handler: choices :y = 3.0")
      (is (h/close? (mx/item (:mean (:posterior ref)))
                    (mx/item (cm/get-value (cm/get-submap (:choices result) :mu))) 1e-10) "run-handler: choices :mu = posterior mean"))))

;; ---------------------------------------------------------------------------
;; Section 11: build-auto-handlers
;; ---------------------------------------------------------------------------

(deftest build-auto-handlers-test
  (testing "build-auto-handlers with multiple pairs"
    (let [pairs [{:prior-addr :mu :obs-addr :y1 :family :normal-normal}
                 {:prior-addr :mu :obs-addr :y2 :family :normal-normal}
                 {:prior-addr :p :obs-addr :x :family :beta-bernoulli}]
          handlers (aa/build-auto-handlers pairs)]
      (is (contains? handlers :mu) "build: has :mu handler")
      (is (contains? handlers :y1) "build: has :y1 handler")
      (is (contains? handlers :y2) "build: has :y2 handler")
      (is (contains? handlers :p) "build: has :p handler")
      (is (contains? handlers :x) "build: has :x handler")
      (is (= 5 (count handlers)) "build: 5 handlers total")))

  (testing "empty pairs produces empty handlers"
    (let [handlers (aa/build-auto-handlers [])]
      (is (= 0 (count handlers)) "build empty: 0 handlers"))))

;; ---------------------------------------------------------------------------
;; Section 12: some-conjugate-obs-constrained?
;; ---------------------------------------------------------------------------

(deftest some-conjugate-obs-constrained-test
  (testing "some-conjugate-obs-constrained?"
    (let [pairs [{:prior-addr :mu :obs-addr :y :family :normal-normal}]]
      (is (aa/some-conjugate-obs-constrained? pairs
            (-> cm/EMPTY (cm/set-value :y (mx/scalar 3.0)))) "obs constrained: true")
      (is (not (aa/some-conjugate-obs-constrained? pairs cm/EMPTY)) "obs NOT constrained: false")
      (is (not (aa/some-conjugate-obs-constrained? pairs
                 (-> cm/EMPTY (cm/set-value :z (mx/scalar 3.0))))) "wrong addr constrained: false"))))

(cljs.test/run-tests)
