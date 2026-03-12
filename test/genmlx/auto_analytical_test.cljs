(ns genmlx.auto-analytical-test
  "Tests for Level 3 address-based analytical handlers (WP-1).
   Verifies marginal LL, posterior, weight/score accounting against conjugate.cljs."
  (:require [genmlx.inference.auto-analytical :as aa]
            [genmlx.inference.conjugate :as conjugate]
            [genmlx.conjugacy :as conj]
            [genmlx.handler :as h]
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
;; Test helpers
;; ---------------------------------------------------------------------------

(def ^:private pass-count (volatile! 0))
(def ^:private fail-count (volatile! 0))

(defn- assert-true [desc pred]
  (if pred
    (do (vswap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (vswap! fail-count inc)
        (println (str "  FAIL: " desc)))))

(defn- assert-equal [desc expected actual]
  (if (= expected actual)
    (do (vswap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (vswap! fail-count inc)
        (println (str "  FAIL: " desc " — expected " expected ", got " actual)))))

(defn- assert-close [desc expected actual tol]
  (let [e (if (number? expected) expected (mx/item expected))
        a (if (number? actual) actual (mx/item actual))
        diff (js/Math.abs (- e a))]
    (if (<= diff tol)
      (do (vswap! pass-count inc)
          (println (str "  PASS: " desc " (diff=" (.toExponential diff 2) ")")))
      (do (vswap! fail-count inc)
          (println (str "  FAIL: " desc " — expected " e ", got " a " (diff=" diff ")"))))))

;; ---------------------------------------------------------------------------
;; Section 1: Update step functions match conjugate.cljs
;; ---------------------------------------------------------------------------

(println "\n=== Section 1: Update Steps Match conjugate.cljs ===")

;; NN update step
(let [prior {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
      obs (mx/scalar 3.0)
      obs-var (mx/scalar 1.0)
      ours (aa/nn-update-step prior obs obs-var)
      ref (conjugate/nn-update prior obs obs-var (mx/scalar 1.0))]
  (mx/eval!)
  (assert-close "NN-step: ll matches" (mx/item (:ll ref)) (mx/item (:ll ours)) 1e-10)
  (assert-close "NN-step: mean matches" (mx/item (:mean (:posterior ref))) (mx/item (:mean ours)) 1e-10)
  (assert-close "NN-step: var matches" (mx/item (:var (:posterior ref))) (mx/item (:var ours)) 1e-10))

;; BB update step
(let [prior {:alpha (mx/scalar 2.0) :beta (mx/scalar 5.0)}
      obs (mx/scalar 1.0)
      ours (aa/bb-update-step prior obs)
      ref (conjugate/bb-update prior obs (mx/scalar 1.0))]
  (mx/eval!)
  (assert-close "BB-step: ll matches" (mx/item (:ll ref)) (mx/item (:ll ours)) 1e-10)
  (assert-close "BB-step: alpha matches" (mx/item (:alpha (:posterior ref))) (mx/item (:alpha ours)) 1e-10)
  (assert-close "BB-step: beta matches" (mx/item (:beta (:posterior ref))) (mx/item (:beta ours)) 1e-10))

;; GP update step
(let [prior {:shape (mx/scalar 2.0) :rate (mx/scalar 1.0)}
      obs (mx/scalar 3.0)
      ours (aa/gp-update-step prior obs)
      ref (conjugate/gp-update prior obs (mx/scalar 1.0))]
  (mx/eval!)
  (assert-close "GP-step: ll matches" (mx/item (:ll ref)) (mx/item (:ll ours)) 1e-10)
  (assert-close "GP-step: shape matches" (mx/item (:shape (:posterior ref))) (mx/item (:shape ours)) 1e-10)
  (assert-close "GP-step: rate matches" (mx/item (:rate (:posterior ref))) (mx/item (:rate ours)) 1e-10))

;; NN sequential updates (5 observations)
(let [obs-vals [3.0 5.0 1.0 4.0 2.0]
      ;; Our path
      final-ours (reduce
                   (fn [post obs]
                     (let [r (aa/nn-update-step post (mx/scalar obs) (mx/scalar 1.0))]
                       {:mean (:mean r) :var (:var r) :ll (mx/add (:ll post (mx/scalar 0.0)) (:ll r))}))
                   {:mean (mx/scalar 0.0) :var (mx/scalar 100.0) :ll (mx/scalar 0.0)}
                   obs-vals)
      ;; Reference path
      final-ref (reduce
                  (fn [state obs]
                    (let [r (conjugate/nn-update (:posterior state) (mx/scalar obs) (mx/scalar 1.0) (mx/scalar 1.0))]
                      {:posterior (:posterior r) :ll (mx/add (:ll state) (:ll r))}))
                  {:posterior {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)} :ll (mx/scalar 0.0)}
                  obs-vals)]
  (mx/eval!)
  (assert-close "NN-5obs: total ll matches" (mx/item (:ll final-ref)) (mx/item (:ll final-ours)) 1e-6)
  (assert-close "NN-5obs: final mean matches" (mx/item (:mean (:posterior final-ref))) (mx/item (:mean final-ours)) 1e-6)
  (assert-close "NN-5obs: final var matches" (mx/item (:var (:posterior final-ref))) (mx/item (:var final-ours)) 1e-6))

;; ---------------------------------------------------------------------------
;; Section 2: make-address-dispatch
;; ---------------------------------------------------------------------------

(println "\n=== Section 2: make-address-dispatch ===")

(let [handlers {:a (fn [state addr dist] [:intercepted-a state])
                :b (fn [state addr dist] [:intercepted-b state])}
      base (fn [state addr dist] [:base state])
      dispatch (aa/make-address-dispatch base handlers)]

  (assert-equal "dispatch :a intercepted" :intercepted-a (first (dispatch {} :a nil)))
  (assert-equal "dispatch :b intercepted" :intercepted-b (first (dispatch {} :b nil)))
  (assert-equal "dispatch :c falls through" :base (first (dispatch {} :c nil))))

;; Nil-return fallthrough
(let [handlers {:a (fn [state addr dist] nil)}  ;; returns nil = fallthrough
      base (fn [state addr dist] [:base state])
      dispatch (aa/make-address-dispatch base handlers)]
  (assert-equal "nil return falls through" :base (first (dispatch {} :a nil))))

;; ---------------------------------------------------------------------------
;; Section 3: NN handler via transitions
;; ---------------------------------------------------------------------------

(println "\n=== Section 3: NN Handler via Transitions ===")

;; Single observation
(let [handlers (aa/build-auto-handlers
                 [{:prior-addr :mu :obs-addr :y :family :normal-normal}])
      transition (aa/make-address-dispatch h/generate-transition handlers)
      constraints (-> cm/EMPTY (cm/set-value :y (mx/scalar 3.0)))
      init {:choices cm/EMPTY :score (mx/scalar 0.0) :weight (mx/scalar 0.0)
            :key (rng/fresh-key 42) :constraints constraints :auto-posteriors {}}
      [mu s1] (transition init :mu (dist/gaussian (mx/scalar 0.0) (mx/scalar 10.0)))
      [y s2] (transition s1 :y (dist/gaussian mu (mx/scalar 1.0)))
      ref (conjugate/nn-update {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
                               (mx/scalar 3.0) (mx/scalar 1.0) (mx/scalar 1.0))]
  (mx/eval!)
  (assert-close "NN-1obs: weight = marginal LL" (mx/item (:ll ref)) (mx/item (:weight s2)) 1e-10)
  (assert-close "NN-1obs: score = marginal LL" (mx/item (:ll ref)) (mx/item (:score s2)) 1e-10)
  (assert-close "NN-1obs: posterior mean" (mx/item (:mean (:posterior ref)))
                (mx/item (get-in s2 [:auto-posteriors :mu :mean])) 1e-10)
  (assert-close "NN-1obs: posterior var" (mx/item (:var (:posterior ref)))
                (mx/item (get-in s2 [:auto-posteriors :mu :var])) 1e-10)
  (assert-close "NN-1obs: choices :mu = posterior mean"
                (mx/item (:mean (:posterior ref)))
                (mx/item (cm/get-value (cm/get-submap (:choices s2) :mu))) 1e-10)
  (assert-close "NN-1obs: choices :y = 3.0"
                3.0 (mx/item (cm/get-value (cm/get-submap (:choices s2) :y))) 1e-10))

;; Three observations
(let [handlers (aa/build-auto-handlers
                 [{:prior-addr :mu :obs-addr :y1 :family :normal-normal}
                  {:prior-addr :mu :obs-addr :y2 :family :normal-normal}
                  {:prior-addr :mu :obs-addr :y3 :family :normal-normal}])
      transition (aa/make-address-dispatch h/generate-transition handlers)
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
      ;; Reference
      r1 (conjugate/nn-update {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
                              (mx/scalar 3.0) (mx/scalar 1.0) (mx/scalar 1.0))
      r2 (conjugate/nn-update (:posterior r1) (mx/scalar 5.0) (mx/scalar 1.0) (mx/scalar 1.0))
      r3 (conjugate/nn-update (:posterior r2) (mx/scalar 1.0) (mx/scalar 1.0) (mx/scalar 1.0))
      ref-ll (+ (mx/item (:ll r1)) (mx/item (:ll r2)) (mx/item (:ll r3)))]
  (mx/eval!)
  (assert-close "NN-3obs: total weight matches ref" ref-ll (mx/item (:weight s4)) 1e-6)
  (assert-close "NN-3obs: posterior mean" (mx/item (:mean (:posterior r3)))
                (mx/item (get-in s4 [:auto-posteriors :mu :mean])) 1e-6))

;; ---------------------------------------------------------------------------
;; Section 4: BB Handler
;; ---------------------------------------------------------------------------

(println "\n=== Section 4: BB Handler ===")

(let [handlers (aa/build-auto-handlers
                 [{:prior-addr :p :obs-addr :x :family :beta-bernoulli}])
      transition (aa/make-address-dispatch h/generate-transition handlers)
      constraints (-> cm/EMPTY (cm/set-value :x (mx/scalar 1.0)))
      init {:choices cm/EMPTY :score (mx/scalar 0.0) :weight (mx/scalar 0.0)
            :key (rng/fresh-key 42) :constraints constraints :auto-posteriors {}}
      [p s1] (transition init :p (dist/beta-dist (mx/scalar 2.0) (mx/scalar 5.0)))
      [x s2] (transition s1 :x (dist/bernoulli p))
      ref (conjugate/bb-update {:alpha (mx/scalar 2.0) :beta (mx/scalar 5.0)}
                               (mx/scalar 1.0) (mx/scalar 1.0))]
  (mx/eval!)
  (assert-close "BB-1obs: weight = marginal LL" (mx/item (:ll ref)) (mx/item (:weight s2)) 1e-10)
  (assert-close "BB-1obs: posterior alpha" (mx/item (:alpha (:posterior ref)))
                (mx/item (get-in s2 [:auto-posteriors :p :alpha])) 1e-10)
  (assert-close "BB-1obs: posterior beta" (mx/item (:beta (:posterior ref)))
                (mx/item (get-in s2 [:auto-posteriors :p :beta])) 1e-10))

;; BB with 3 observations
(let [handlers (aa/build-auto-handlers
                 [{:prior-addr :p :obs-addr :x1 :family :beta-bernoulli}
                  {:prior-addr :p :obs-addr :x2 :family :beta-bernoulli}
                  {:prior-addr :p :obs-addr :x3 :family :beta-bernoulli}])
      transition (aa/make-address-dispatch h/generate-transition handlers)
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
      ;; Reference
      r1 (conjugate/bb-update {:alpha (mx/scalar 2.0) :beta (mx/scalar 5.0)}
                              (mx/scalar 1.0) (mx/scalar 1.0))
      r2 (conjugate/bb-update (:posterior r1) (mx/scalar 0.0) (mx/scalar 1.0))
      r3 (conjugate/bb-update (:posterior r2) (mx/scalar 1.0) (mx/scalar 1.0))
      ref-ll (+ (mx/item (:ll r1)) (mx/item (:ll r2)) (mx/item (:ll r3)))]
  (mx/eval!)
  (assert-close "BB-3obs: total weight matches ref" ref-ll (mx/item (:weight s4)) 1e-6))

;; ---------------------------------------------------------------------------
;; Section 5: GP Handler
;; ---------------------------------------------------------------------------

(println "\n=== Section 5: GP Handler ===")

(let [handlers (aa/build-auto-handlers
                 [{:prior-addr :rate :obs-addr :count :family :gamma-poisson}])
      transition (aa/make-address-dispatch h/generate-transition handlers)
      constraints (-> cm/EMPTY (cm/set-value :count (mx/scalar 3.0)))
      init {:choices cm/EMPTY :score (mx/scalar 0.0) :weight (mx/scalar 0.0)
            :key (rng/fresh-key 42) :constraints constraints :auto-posteriors {}}
      [rate s1] (transition init :rate (dist/gamma-dist (mx/scalar 2.0) (mx/scalar 1.0)))
      [cnt s2] (transition s1 :count (dist/poisson rate))
      ref (conjugate/gp-update {:shape (mx/scalar 2.0) :rate (mx/scalar 1.0)}
                               (mx/scalar 3.0) (mx/scalar 1.0))]
  (mx/eval!)
  (assert-close "GP-1obs: weight = marginal LL" (mx/item (:ll ref)) (mx/item (:weight s2)) 1e-10)
  (assert-close "GP-1obs: posterior shape" (mx/item (:shape (:posterior ref)))
                (mx/item (get-in s2 [:auto-posteriors :rate :shape])) 1e-10)
  (assert-close "GP-1obs: posterior rate" (mx/item (:rate (:posterior ref)))
                (mx/item (get-in s2 [:auto-posteriors :rate :rate])) 1e-10))

;; ---------------------------------------------------------------------------
;; Section 6: GE Handler
;; ---------------------------------------------------------------------------

(println "\n=== Section 6: GE Handler ===")

(let [handlers (aa/build-auto-handlers
                 [{:prior-addr :rate :obs-addr :x :family :gamma-exponential}])
      transition (aa/make-address-dispatch h/generate-transition handlers)
      constraints (-> cm/EMPTY (cm/set-value :x (mx/scalar 2.5)))
      init {:choices cm/EMPTY :score (mx/scalar 0.0) :weight (mx/scalar 0.0)
            :key (rng/fresh-key 42) :constraints constraints :auto-posteriors {}}
      [rate s1] (transition init :rate (dist/gamma-dist (mx/scalar 3.0) (mx/scalar 2.0)))
      [x s2] (transition s1 :x (dist/exponential rate))
      ;; Compute reference marginal LL manually
      ;; Lomax: log(shape) + shape*log(rate) - (shape+1)*log(rate+x)
      ref-ll (+ (js/Math.log 3.0) (* 3.0 (js/Math.log 2.0))
               (* -4.0 (js/Math.log 4.5)))]
  (mx/eval!)
  (assert-close "GE-1obs: weight = marginal LL" ref-ll (mx/item (:weight s2)) 1e-6)
  (assert-close "GE-1obs: posterior shape" 4.0
                (mx/item (get-in s2 [:auto-posteriors :rate :shape])) 1e-10)
  (assert-close "GE-1obs: posterior rate" 4.5
                (mx/item (get-in s2 [:auto-posteriors :rate :rate])) 1e-10))

;; ---------------------------------------------------------------------------
;; Section 7: Fallthrough behavior
;; ---------------------------------------------------------------------------

(println "\n=== Section 7: Fallthrough Behavior ===")

;; Unconstrained obs → nil → falls through to standard generate
(let [handlers (aa/build-auto-handlers
                 [{:prior-addr :mu :obs-addr :y :family :normal-normal}])
      transition (aa/make-address-dispatch h/generate-transition handlers)
      init {:choices cm/EMPTY :score (mx/scalar 0.0) :weight (mx/scalar 0.0)
            :key (rng/fresh-key 42) :constraints cm/EMPTY :auto-posteriors {}}
      [mu s1] (transition init :mu (dist/gaussian (mx/scalar 0.0) (mx/scalar 10.0)))
      [y s2] (transition s1 :y (dist/gaussian mu (mx/scalar 1.0)))]
  (mx/eval!)
  (assert-close "fallthrough: weight = 0 (no constraint)" 0.0 (mx/item (:weight s2)) 1e-10)
  (assert-true "fallthrough: y was sampled (not nil)" (some? y)))

;; Non-intercepted address → base handler
(let [handlers (aa/build-auto-handlers
                 [{:prior-addr :mu :obs-addr :y :family :normal-normal}])
      transition (aa/make-address-dispatch h/generate-transition handlers)
      constraints (-> cm/EMPTY (cm/set-value :z (mx/scalar 5.0)))
      init {:choices cm/EMPTY :score (mx/scalar 0.0) :weight (mx/scalar 0.0)
            :key (rng/fresh-key 42) :constraints constraints :auto-posteriors {}}
      [z s1] (transition init :z (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))]
  (mx/eval!)
  (assert-close "non-intercepted: z = 5.0 (constrained)" 5.0 (mx/item z) 1e-10)
  (assert-true "non-intercepted: weight nonzero" (not= 0.0 (mx/item (:weight s1)))))

;; ---------------------------------------------------------------------------
;; Section 8: Mixed model (NN + BB in same model)
;; ---------------------------------------------------------------------------

(println "\n=== Section 8: Mixed Handlers ===")

(let [handlers (aa/build-auto-handlers
                 [{:prior-addr :mu :obs-addr :y :family :normal-normal}
                  {:prior-addr :p :obs-addr :x :family :beta-bernoulli}])
      transition (aa/make-address-dispatch h/generate-transition handlers)
      constraints (-> cm/EMPTY
                    (cm/set-value :y (mx/scalar 3.0))
                    (cm/set-value :x (mx/scalar 1.0)))
      init {:choices cm/EMPTY :score (mx/scalar 0.0) :weight (mx/scalar 0.0)
            :key (rng/fresh-key 42) :constraints constraints :auto-posteriors {}}
      [mu s1] (transition init :mu (dist/gaussian (mx/scalar 0.0) (mx/scalar 10.0)))
      [p s2] (transition s1 :p (dist/beta-dist (mx/scalar 2.0) (mx/scalar 5.0)))
      [y s3] (transition s2 :y (dist/gaussian mu (mx/scalar 1.0)))
      [x s4] (transition s3 :x (dist/bernoulli p))
      ;; Reference
      nn-ref (conjugate/nn-update {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
                                  (mx/scalar 3.0) (mx/scalar 1.0) (mx/scalar 1.0))
      bb-ref (conjugate/bb-update {:alpha (mx/scalar 2.0) :beta (mx/scalar 5.0)}
                                  (mx/scalar 1.0) (mx/scalar 1.0))
      ref-total (+ (mx/item (:ll nn-ref)) (mx/item (:ll bb-ref)))]
  (mx/eval!)
  (assert-close "mixed: total weight = NN-ll + BB-ll" ref-total (mx/item (:weight s4)) 1e-6)
  (assert-true "mixed: has mu posterior" (contains? (:auto-posteriors s4) :mu))
  (assert-true "mixed: has p posterior" (contains? (:auto-posteriors s4) :p)))

;; ---------------------------------------------------------------------------
;; Section 9: Mixed conjugate + non-conjugate sites
;; ---------------------------------------------------------------------------

(println "\n=== Section 9: Conjugate + Standard Sites ===")

(let [;; Only :mu → :y is conjugate; :z is standard
      handlers (aa/build-auto-handlers
                 [{:prior-addr :mu :obs-addr :y :family :normal-normal}])
      transition (aa/make-address-dispatch h/generate-transition handlers)
      constraints (-> cm/EMPTY
                    (cm/set-value :y (mx/scalar 3.0))
                    (cm/set-value :z (mx/scalar 7.0)))
      init {:choices cm/EMPTY :score (mx/scalar 0.0) :weight (mx/scalar 0.0)
            :key (rng/fresh-key 42) :constraints constraints :auto-posteriors {}}
      [mu s1] (transition init :mu (dist/gaussian (mx/scalar 0.0) (mx/scalar 10.0)))
      [y s2] (transition s1 :y (dist/gaussian mu (mx/scalar 1.0)))
      [z s3] (transition s2 :z (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)))
      ;; Weight should be NN-marginal-LL + standard log-prob for :z
      nn-ref (conjugate/nn-update {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
                                  (mx/scalar 3.0) (mx/scalar 1.0) (mx/scalar 1.0))
      z-lp (mx/item (dc/dist-log-prob (dist/gaussian (mx/scalar 0.0) (mx/scalar 1.0)) (mx/scalar 7.0)))
      expected-weight (+ (mx/item (:ll nn-ref)) z-lp)]
  (mx/eval!)
  (assert-close "mixed-std: weight = NN-ll + z-logprob" expected-weight (mx/item (:weight s3)) 1e-6))

;; ---------------------------------------------------------------------------
;; Section 10: run-handler integration
;; ---------------------------------------------------------------------------

(println "\n=== Section 10: run-handler Integration ===")

(let [model (dyn/auto-key
              (gen [sigma]
                (let [mu (trace :mu (dist/gaussian 0 10))]
                  (trace :y (dist/gaussian mu sigma))
                  mu)))
      pairs (conj/detect-conjugate-pairs (:schema model))
      handlers (aa/build-auto-handlers pairs)
      transition (aa/make-address-dispatch h/generate-transition handlers)
      constraints (-> cm/EMPTY (cm/set-value :y (mx/scalar 3.0)))
      init {:choices cm/EMPTY :score (mx/scalar 0.0) :weight (mx/scalar 0.0)
            :key (rng/fresh-key 42) :constraints constraints :auto-posteriors {}}
      result (rt/run-handler transition init
               (fn [rt] (apply (:body-fn model) rt [(mx/scalar 1.0)])))
      ref (conjugate/nn-update {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
                               (mx/scalar 3.0) (mx/scalar 1.0) (mx/scalar 1.0))]
  (mx/eval!)
  (assert-close "run-handler: weight matches" (mx/item (:ll ref)) (mx/item (:weight result)) 1e-10)
  (assert-close "run-handler: score matches" (mx/item (:ll ref)) (mx/item (:score result)) 1e-10)
  (assert-true "run-handler: has retval" (some? (:retval result)))
  (assert-close "run-handler: choices :y = 3.0"
                3.0 (mx/item (cm/get-value (cm/get-submap (:choices result) :y))) 1e-10)
  (assert-close "run-handler: choices :mu = posterior mean"
                (mx/item (:mean (:posterior ref)))
                (mx/item (cm/get-value (cm/get-submap (:choices result) :mu))) 1e-10))

;; ---------------------------------------------------------------------------
;; Section 11: build-auto-handlers
;; ---------------------------------------------------------------------------

(println "\n=== Section 11: build-auto-handlers ===")

(let [pairs [{:prior-addr :mu :obs-addr :y1 :family :normal-normal}
             {:prior-addr :mu :obs-addr :y2 :family :normal-normal}
             {:prior-addr :p :obs-addr :x :family :beta-bernoulli}]
      handlers (aa/build-auto-handlers pairs)]
  (assert-true "build: has :mu handler" (contains? handlers :mu))
  (assert-true "build: has :y1 handler" (contains? handlers :y1))
  (assert-true "build: has :y2 handler" (contains? handlers :y2))
  (assert-true "build: has :p handler" (contains? handlers :p))
  (assert-true "build: has :x handler" (contains? handlers :x))
  (assert-equal "build: 5 handlers total" 5 (count handlers)))

;; Empty pairs → empty handlers
(let [handlers (aa/build-auto-handlers [])]
  (assert-equal "build empty: 0 handlers" 0 (count handlers)))

;; ---------------------------------------------------------------------------
;; Section 12: some-conjugate-obs-constrained?
;; ---------------------------------------------------------------------------

(println "\n=== Section 12: some-conjugate-obs-constrained? ===")

(let [pairs [{:prior-addr :mu :obs-addr :y :family :normal-normal}]]
  (assert-true "obs constrained: true"
    (aa/some-conjugate-obs-constrained? pairs
      (-> cm/EMPTY (cm/set-value :y (mx/scalar 3.0)))))
  (assert-true "obs NOT constrained: false"
    (not (aa/some-conjugate-obs-constrained? pairs cm/EMPTY)))
  (assert-true "wrong addr constrained: false"
    (not (aa/some-conjugate-obs-constrained? pairs
           (-> cm/EMPTY (cm/set-value :z (mx/scalar 3.0)))))))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== RESULTS: " @pass-count "/" (+ @pass-count @fail-count)
              " passed, " @fail-count " failed ==="))
