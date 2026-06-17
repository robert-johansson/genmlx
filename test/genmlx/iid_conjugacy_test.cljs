;; @tier medium
(ns genmlx.iid-conjugacy-test
  "M2 Step 4: Conjugacy + auto-analytical for iid-gaussian."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.inference.auto-analytical :as aa]
            [genmlx.conjugacy :as conj]
            [genmlx.handler :as handler]
            [genmlx.runtime :as rt]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; 0. INDEPENDENT oracles for the marginal log-evidence (genmlx-ke9i)
;;
;; The marginal evidence of T iid observations with a SHARED mu is the joint
;; multivariate normal  N(ys; m0*1, sigma^2 I + tau^2 11^T)  — NOT a product of
;; independent per-point marginals. These two oracles are derived/computed here
;; by DIFFERENT methods and are independent of auto-analytical/nn-iid-update-step
;; (the function under test). Asserting the handler against the function under
;; test was the original circularity that let the wrong-math bug pass.
;; ---------------------------------------------------------------------------

(defn oracle-marginal-closed
  "Closed-form (matrix-determinant lemma) shared-mu joint marginal log-evidence."
  [ys m0 tau2 s2]
  (let [T (count ys)
        d (map #(- % m0) ys)
        sum-d (reduce + d)
        sum-d2 (reduce + (map #(* % %) d))
        denom (+ s2 (* T tau2))
        logdet (+ (* (dec T) (js/Math.log s2)) (js/Math.log denom))
        quad (/ (- sum-d2 (* (/ tau2 denom) (* sum-d sum-d))) s2)]
    (* -0.5 (+ (* T (js/Math.log (* 2 js/Math.PI))) logdet quad))))

(defn oracle-marginal-quad
  "Independent METHOD: marginalise mu by numerical integration over a fine grid.
   p(ys) = ∫ N(mu; m0, tau2) * prod_i N(y_i; mu, s2) dmu."
  [ys m0 tau2 s2]
  (let [tau (js/Math.sqrt tau2)
        lo (- m0 (* 8 tau)) hi (+ m0 (* 8 tau))
        n 40000
        dx (/ (- hi lo) n)
        l2ps2 (js/Math.log (* 2 js/Math.PI s2))
        lpn (js/Math.log (js/Math.sqrt (* 2 js/Math.PI tau2)))]
    (loop [k 0 acc 0.0]
      (if (> k n)
        (js/Math.log (* dx acc))
        (let [mu (+ lo (* k dx))
              lp (- (- (/ (* (- mu m0) (- mu m0)) (* 2 tau2))) lpn)
              ll (reduce (fn [a y] (+ a (* -0.5 (+ l2ps2 (/ (* (- y mu) (- y mu)) s2))))) 0.0 ys)]
          (recur (inc k) (+ acc (js/Math.exp (+ lp ll)))))))))

(deftest oracle-self-consistency
  (testing "the two independent oracles agree (so they are trustworthy ground truth)"
    (let [ys [1.0 2.0 3.0 4.0 5.0]]
      (is (h/close? (oracle-marginal-closed ys 0 100 1)
                    (oracle-marginal-quad ys 0 100 1) 1e-3)
          "closed-form == numerical quadrature")
      (is (h/close? -12.748 (oracle-marginal-closed ys 0 100 1) 1e-2)
          "anchor: shared-mu joint marginal ~ -12.748 (NOT -16.40 sum-of-independent)"))))

;; ---------------------------------------------------------------------------
;; 1. Conjugacy table entry
;; ---------------------------------------------------------------------------

(deftest conjugacy-table-entry
  (testing "Conjugacy table: :gaussian + :iid-gaussian"
    (let [entry (get conj/conjugacy-table [:gaussian :iid-gaussian])]
      (is (some? entry) "entry exists")
      (is (= :normal-iid-normal (:family entry)) "family")
      (is (= 0 (:natural-param-idx entry)) "natural-param-idx")
      (is (= :mu (:prior-mean-key entry)) "prior-mean-key")
      (is (= :sigma (:prior-std-key entry)) "prior-std-key")
      (is (= :mu (:obs-mean-key entry)) "obs-mean-key")
      (is (= :sigma (:obs-noise-key entry)) "obs-noise-key"))))

;; ---------------------------------------------------------------------------
;; 2. Conjugate pair detection on iid model
;; ---------------------------------------------------------------------------

(def iid-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :ys (dist/iid-gaussian mu (mx/scalar 1.0) 5))
      mu)))

(deftest conjugate-pair-detection
  (testing "Conjugate pair detection"
    (let [pairs (conj/detect-conjugate-pairs (:schema iid-model))]
      (is (= 1 (count pairs)) "1 conjugate pair")
      (let [pair (first pairs)]
        (is (= :mu (:prior-addr pair)) "prior-addr")
        (is (= :ys (:obs-addr pair)) "obs-addr")
        (is (= :normal-iid-normal (:family pair)) "family")
        (is (= :direct (get-in pair [:dependency-type :type])) "dep-type direct"))))

  (testing "Augmented schema"
    (let [aug (conj/augment-schema-with-conjugacy (:schema iid-model))]
      (is (:has-conjugate? aug) "has-conjugate?")
      (is (= 1 (count (:conjugate-pairs aug))) "conjugate-pairs count"))))

;; ---------------------------------------------------------------------------
;; 3. nn-iid-update-step math correctness
;; ---------------------------------------------------------------------------

(deftest nn-iid-update-step-math
  (testing "nn-iid-update-step basic"
    (let [prior {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
          obs (mx/array [1.0 2.0 3.0 4.0 5.0])
          obs-var (mx/scalar 1.0)
          result (aa/nn-iid-update-step prior obs obs-var)]
      (mx/eval!)
      (is (h/close? 2.994 (mx/item (:mean result)) 0.01) "posterior mean ~ 2.994")
      (is (h/close? 0.1996 (mx/item (:var result)) 0.01) "posterior var ~ 0.1996")
      ;; :ll is the SHARED-MU joint marginal — pinned against INDEPENDENT oracles
      ;; (never against nn-iid-update-step itself — that was the ke9i circularity).
      (is (h/close? -12.748 (mx/item (:ll result)) 1e-2)
          ":ll ~ -12.748 (shared-mu joint; NOT -16.40 sum-of-independent)")
      (is (h/close? (oracle-marginal-closed [1.0 2.0 3.0 4.0 5.0] 0 100 1)
                    (mx/item (:ll result)) 1e-2)
          ":ll == closed-form oracle")
      (is (h/close? (oracle-marginal-quad [1.0 2.0 3.0 4.0 5.0] 0 100 1)
                    (mx/item (:ll result)) 1e-2)
          ":ll == numerical-quadrature oracle")))

  (testing "T=1 matches nn-update-step"
    (let [prior {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
          obs-single (mx/array [3.0])
          obs-var (mx/scalar 1.0)
          iid-result (aa/nn-iid-update-step prior obs-single obs-var)
          scalar-result (aa/nn-update-step prior (mx/scalar 3.0) obs-var)]
      (mx/eval!)
      (is (h/close? (mx/item (:mean scalar-result)) (mx/item (:mean iid-result)) 1e-6)
          "T=1: mean matches nn-update")
      (is (h/close? (mx/item (:var scalar-result)) (mx/item (:var iid-result)) 1e-6)
          "T=1: var matches nn-update")
      (is (h/close? (mx/item (:ll scalar-result)) (mx/item (:ll iid-result)) 1e-6)
          "T=1: ll matches nn-update")))

  (testing "Large T: posterior tight around sample mean"
    (let [prior {:mean (mx/scalar 0.0) :var (mx/scalar 100.0)}
          obs (mx/array (vec (repeat 100 5.0)))
          obs-var (mx/scalar 1.0)
          result (aa/nn-iid-update-step prior obs obs-var)]
      (mx/eval!)
      (is (h/close? 5.0 (mx/item (:mean result)) 0.01) "T=100: posterior mean ~ 5.0")
      (is (h/close? 0.01 (mx/item (:var result)) 0.001) "T=100: posterior var ~ 0.01"))))

;; ---------------------------------------------------------------------------
;; 4. Handler integration: build-auto-handlers with iid pair
;; ---------------------------------------------------------------------------

(deftest handler-integration
  (testing "Handler integration"
    (let [pairs [{:prior-addr :mu :obs-addr :ys :family :normal-iid-normal}]
          handlers (aa/build-auto-handlers pairs)]
      (is (contains? handlers :mu) "has :mu handler")
      (is (contains? handlers :ys) "has :ys handler")
      (is (= 2 (count handlers)) "2 handlers total"))))

;; ---------------------------------------------------------------------------
;; 5. run-handler with auto-analytical transition (iid-gaussian obs)
;; ---------------------------------------------------------------------------

(deftest run-handler-iid-gaussian
  (testing "run-handler + iid-gaussian"
    (let [model (dyn/auto-key
                  (gen []
                    (let [mu (trace :mu (dist/gaussian 0 10))]
                      (trace :ys (dist/iid-gaussian mu (mx/scalar 1.0) 5))
                      mu)))
          pairs (conj/detect-conjugate-pairs (:schema model))
          handlers (aa/build-auto-handlers pairs)
          transition (aa/make-address-dispatch handler/generate-transition handlers)
          obs-data (mx/array [1.0 2.0 3.0 4.0 5.0])
          constraints (-> cm/EMPTY (cm/set-value :ys obs-data))
          init {:choices cm/EMPTY :score (mx/scalar 0.0) :weight (mx/scalar 0.0)
                :key (rng/fresh-key 42) :constraints constraints :auto-posteriors {}}
          result (rt/run-handler transition init
                   (fn [rt] (apply (:body-fn model) rt [])))]
      (mx/eval!)
      (is (js/isFinite (mx/item (:weight result))) "weight is finite")
      (is (neg? (mx/item (:weight result))) "weight is negative")
      (is (js/isFinite (mx/item (:score result))) "score is finite")
      (is (some? (cm/get-submap (:choices result) :mu)) "choices has :mu")
      (is (some? (cm/get-submap (:choices result) :ys)) "choices has :ys")
      (let [mu-val (mx/item (cm/get-value (cm/get-submap (:choices result) :mu)))
            oracle-ll (oracle-marginal-closed [1.0 2.0 3.0 4.0 5.0] 0 100 1)]
        (mx/eval!)
        (is (h/close? 2.994 mu-val 1e-2) "mu = posterior mean ~ 2.994")
        (is (h/close? oracle-ll (mx/item (:weight result)) 1e-2) "weight = marginal LL (independent oracle)")
        (is (h/close? oracle-ll (mx/item (:score result)) 1e-2) "score = marginal LL (independent oracle)")))))

;; ---------------------------------------------------------------------------
;; 6. End-to-end: p/generate with auto-analytical elimination
;; ---------------------------------------------------------------------------

(def iid-model-e2e
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :ys (dist/iid-gaussian mu (mx/scalar 1.0) 5))
      mu)))

(deftest generate-end-to-end
  (testing "p/generate end-to-end"
    (let [gf (dyn/auto-key iid-model-e2e)
          obs (cm/choicemap :ys (mx/array [1.0 2.0 3.0 4.0 5.0]))
          result (p/generate gf [] obs)]
      (mx/eval!)
      (is (js/isFinite (mx/item (:weight result))) "e2e: weight is finite")
      (is (neg? (mx/item (:weight result))) "e2e: weight is negative")
      (let [oracle-ll (oracle-marginal-closed [1.0 2.0 3.0 4.0 5.0] 0 100 1)]
        (is (h/close? oracle-ll (mx/item (:weight result)) 1e-2)
            "e2e: weight = marginal LL (independent oracle, NOT the fn under test)")))))

;; ---------------------------------------------------------------------------
;; 7. Multi-obs: iid-gaussian + scalar gaussian on same prior
;; ---------------------------------------------------------------------------

(def mixed-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :ys (dist/iid-gaussian mu (mx/scalar 1.0) 3))
      (trace :y-extra (dist/gaussian mu 1))
      mu)))

(deftest mixed-iid-scalar-obs
  (testing "Mixed iid + scalar obs"
    (let [pairs (conj/detect-conjugate-pairs (:schema mixed-model))]
      (is (= 2 (count pairs)) "mixed: 2 pairs detected")
      (let [families (set (map :family pairs))]
        (is (contains? families :normal-iid-normal) "mixed: has normal-iid-normal")
        (is (contains? families :normal-normal) "mixed: has normal-normal")))))

;; ---------------------------------------------------------------------------
;; 8. Regenerate handlers for iid-gaussian
;; ---------------------------------------------------------------------------

(deftest regenerate-handlers
  (testing "Regenerate handlers"
    (let [pairs [{:prior-addr :mu :obs-addr :ys :family :normal-iid-normal}]
          handlers (aa/build-regenerate-handlers pairs)]
      (is (contains? handlers :mu) "regen: has :mu handler")
      (is (contains? handlers :ys) "regen: has :ys handler")
      (is (= 2 (count handlers)) "regen: 2 handlers total"))))

;; ---------------------------------------------------------------------------
;; 9. Variance reduction: auto-analytical should reduce IS weight variance
;; ---------------------------------------------------------------------------

(def var-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 5))]
      (trace :ys (dist/iid-gaussian mu (mx/scalar 1.0) 10))
      mu)))

(deftest variance-reduction
  (testing "Variance reduction"
    (let [gf (dyn/auto-key var-model)
          obs (cm/choicemap :ys (mx/array [2.0 2.1 1.9 2.0 2.1 1.9 2.0 2.1 1.9 2.0]))
          weights (vec (for [_ (range 50)]
                         (mx/item (:weight (p/generate gf [] obs)))))
          mean-w (/ (reduce + weights) (count weights))
          var-w (/ (reduce + (map #(* (- % mean-w) (- % mean-w)) weights)) (count weights))]
      (is (h/close? 0.0 var-w 1e-6) "variance reduction: weight variance ~ 0")
      (is (every? #(< (js/Math.abs (- % (first weights))) 1e-6) weights)
          "variance reduction: all weights equal"))))

;; ---------------------------------------------------------------------------
;; 10. Score accounting: score = weight for fully constrained model
;; ---------------------------------------------------------------------------

(deftest score-weight-accounting
  (testing "Score = weight accounting"
    (let [gf (dyn/auto-key iid-model-e2e)
          obs (cm/choicemap :ys (mx/array [1.0 2.0 3.0 4.0 5.0]))
          results (for [_ (range 10)]
                    (let [r (p/generate gf [] obs)
                          tr (:trace r)]
                      {:weight (mx/item (:weight r))
                       :score (mx/item (:score tr))}))]
      (doseq [r results]
        (is (h/close? (:weight r) (:score r) 1e-6) "score ~ weight")))))

;; ---------------------------------------------------------------------------
;; 11. Heteroscedastic [T]-sigma iid-gaussian (genmlx-symr)
;;
;; nn-iid-update-step hard-codes the HOMOSCEDASTIC normal-iid-normal closed form
;; (treats obs-var as a SCALAR s2). dist/iid-gaussian also accepts a per-element
;; [T] sigma, whose correct marginal is the DIFFERENT MVN
;;   N(y; m0*1, diag(sigma_i^2) + tau2 11^T).
;; Before the fix a [T]-sigma model was routed to L3 analytical elimination and
;; silently mis-scored — the ll came out [T]-shaped (not the joint scalar),
;; flowing into :score/:weight. The fix is defense-in-depth: a static gate in
;; detect-conjugate-pairs declines a provably-vector sigma, and a runtime
;; backstop in the :normal-iid-normal update-step throws {:analytical/bail true}
;; for ANY non-scalar obs-var, re-routing to the handler joint path which scores
;; the per-element sigma correctly (dist-log-prob :iid-gaussian).
;;
;; INDEPENDENT oracles (ke9i discipline): the correct heteroscedastic marginal,
;; computed two ways that do NOT touch the function under test.
;; ---------------------------------------------------------------------------

(defn oracle-marginal-het-closed
  "Closed-form heteroscedastic shared-mu marginal log-evidence via the
   matrix-determinant lemma / Sherman-Morrison rank-1 update of
   Sigma = diag(s2s) + tau2 11^T. Reduces to oracle-marginal-closed when all
   s2s are equal (verified in het-oracle-self-consistency)."
  [ys m0 tau2 s2s]
  (let [T (count ys)
        d (map #(- % m0) ys)
        w (map #(/ 1.0 %) s2s)                 ; precisions 1/s_i^2
        A (+ 1.0 (* tau2 (reduce + w)))        ; 1 + tau2 1^T D^{-1} 1
        sum-dw (reduce + (map * d w))
        sum-d2w (reduce + (map (fn [di wi] (* di di wi)) d w))
        logdet (+ (reduce + (map js/Math.log s2s)) (js/Math.log A))
        quad (- sum-d2w (/ (* tau2 sum-dw sum-dw) A))]
    (* -0.5 (+ (* T (js/Math.log (* 2 js/Math.PI))) logdet quad))))

(defn oracle-marginal-het-quad
  "Independent METHOD: numerically marginalise mu for heteroscedastic s_i^2.
   p(ys) = ∫ N(mu; m0, tau2) * prod_i N(y_i; mu, s2s_i) dmu."
  [ys m0 tau2 s2s]
  (let [tau (js/Math.sqrt tau2)
        lo (- m0 (* 8 tau)) hi (+ m0 (* 8 tau))
        n 40000 dx (/ (- hi lo) n)
        lpn (js/Math.log (js/Math.sqrt (* 2 js/Math.PI tau2)))]
    (loop [k 0 acc 0.0]
      (if (> k n)
        (js/Math.log (* dx acc))
        (let [mu (+ lo (* k dx))
              lp (- (- (/ (* (- mu m0) (- mu m0)) (* 2 tau2))) lpn)
              ll (reduce (fn [a [y s2]]
                           (+ a (* -0.5 (+ (js/Math.log (* 2 js/Math.PI s2))
                                           (/ (* (- y mu) (- y mu)) s2)))))
                         0.0 (map vector ys s2s))]
          (recur (inc k) (+ acc (js/Math.exp (+ lp ll)))))))))

(deftest het-oracle-self-consistency
  (testing "the heteroscedastic oracle reduces to the homoscedastic one for equal s2s"
    (let [ys [1.0 2.0 3.0 4.0 5.0]]
      (is (h/close? (oracle-marginal-closed ys 0 100 1)
                    (oracle-marginal-het-closed ys 0 100 [1 1 1 1 1]) 1e-6)
          "het closed-form == homoscedastic closed-form at uniform s2s")))
  (testing "the two independent heteroscedastic methods agree (trustworthy ground truth)"
    (let [ys [0.5 1.5 -0.5 2.0 0.0]
          s2s [1.0 4.0 0.25 1.96 0.49]]
      (is (h/close? (oracle-marginal-het-closed ys 0 100 s2s)
                    (oracle-marginal-het-quad ys 0 100 s2s) 1e-3)
          "het closed-form == numerical-quadrature marginal"))))

;; ys = [0.5 1.5 -0.5 2.0 0.0], sigma = [1 2 0.5 1.4 0.7] -> s2s = [1 4 0.25 1.96 0.49]
(def het-sigma (mx/array [1.0 2.0 0.5 1.4 0.7]))
(def het-ys (mx/array [0.5 1.5 -0.5 2.0 0.0]))
;; INLINE (mx/array [...]) literal so the static gate can prove the [T] shape;
;; a def'd-symbol or arg sigma is static-blind and is the runtime backstop's job.
(def het-model
  (gen [] (let [mu (trace :mu (dist/gaussian 0 10))]
            (trace :ys (dist/iid-gaussian mu (mx/array [1.0 2.0 0.5 1.4 0.7]) 5))
            mu)))
(def sca-model
  (gen [] (let [mu (trace :mu (dist/gaussian 0 10))]
            (trace :ys (dist/iid-gaussian mu (mx/scalar 1.0) 5))
            mu)))
;; sigma supplied as a model ARG — the static gate cannot prove its shape, so
;; the RUNTIME backstop is the load-bearing guard for this case.
(def arg-model
  (gen [sig] (let [mu (trace :mu (dist/gaussian 0 10))]
               (trace :ys (dist/iid-gaussian mu sig 5))
               mu)))

(deftest heteroscedastic-detection-declined
  (testing "a provably-vector [T] sigma is declined by static detection"
    (is (empty? (conj/detect-conjugate-pairs (:schema het-model)))
        "het [T]-sigma: NO conjugate pair detected"))
  (testing "a scalar sigma is still detected as conjugate (path unchanged)"
    (is (= 1 (count (conj/detect-conjugate-pairs (:schema sca-model))))
        "scalar sigma: still :normal-iid-normal conjugate"))
  (testing "an arg-supplied sigma is NOT declined statically (runtime must guard it)"
    (is (= 1 (count (conj/detect-conjugate-pairs (:schema arg-model))))
        "arg sigma: static gate is blind, detection still fires")))

(deftest heteroscedastic-runtime-bails-to-scalar
  (testing "[T]-sigma literal: detection-declined -> handler scores a JOINT SCALAR weight"
    (let [gf (dyn/auto-key het-model)
          w  (:weight (p/generate gf [] (cm/choicemap :ys het-ys)))]
      (is (= [] (mx/shape w)) "het generate weight is a joint scalar, not [T]")
      (is (js/isFinite (mx/item w)) "het generate weight is finite")))
  (testing "[T]-sigma via model arg: runtime backstop bails -> joint SCALAR weight"
    (let [gf (dyn/auto-key arg-model)
          w  (:weight (p/generate gf [het-sigma] (cm/choicemap :ys het-ys)))]
      (is (= [] (mx/shape w)) "arg [T]-sigma generate weight is a joint scalar, not [T]")
      (is (js/isFinite (mx/item w)) "arg [T]-sigma generate weight is finite"))))

(deftest scalar-sigma-analytical-still-exact
  ;; The supported (scalar-sigma) analytical path must STILL equal the closed-form
  ;; MVN marginal exactly — this is the "analytical == closed-form MVN oracle"
  ;; guarantee, verified against an INDEPENDENT oracle (not nn-iid-update-step).
  (testing "scalar-sigma generate weight == independent MVN marginal oracle"
    (let [gf (dyn/auto-key sca-model)
          w  (mx/item (:weight (p/generate gf [] (cm/choicemap :ys het-ys))))
          ys [0.5 1.5 -0.5 2.0 0.0]]
      (is (h/close? w (oracle-marginal-closed ys 0 100 1) 1e-3)
          "scalar-sigma weight == closed-form oracle")
      (is (h/close? w (oracle-marginal-het-quad ys 0 100 [1 1 1 1 1]) 1e-3)
          "scalar-sigma weight == numerical-quadrature oracle")))
  ;; And the scalar arg path stays analytical-exact too (runtime sees scalar obs-var).
  (testing "scalar-sigma via model arg stays analytical and exact"
    (let [gf (dyn/auto-key arg-model)
          w  (mx/item (:weight (p/generate gf [(mx/scalar 1.0)] (cm/choicemap :ys het-ys))))
          ys [0.5 1.5 -0.5 2.0 0.0]]
      (is (h/close? w (oracle-marginal-closed ys 0 100 1) 1e-3)
          "scalar arg-sigma weight == closed-form oracle"))))

(cljs.test/run-tests)
