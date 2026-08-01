;; @tier medium
(ns genmlx.encapsulated-test
  "genmlx-qbaa: encapsulated randomness (Cusumano-Towner 2020 thesis §4.5).

   Verifies that a generative function whose realized score is an unbiased
   density ESTIMATOR xi(x, tau, omega) (not the exact density) behaves
   correctly under the full GFI, that omega is resampled/reused correctly so
   identity operations cost weight 0, and that pseudo-marginal MCMC built on
   it targets the EXACT posterior.

   INDEPENDENT ORACLE discipline ([[feedback-independent-oracle-tests]]): every
   numeric expectation is a closed form derived by hand below (`o-log-gauss`,
   `o-mixture-logp`, `o-conv-logp`, the Normal-Normal posterior 24/13 & 4/13),
   never computed through the function under test."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.encapsulated :as enc]
            [genmlx.dynamic :as dyn]
            [genmlx.diff :as diff]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.gfi :as gfi]
            [genmlx.serialize :as ser]))

;; ---------------------------------------------------------------------------
;; Independent JS oracles (NEVER call the implementation)
;; ---------------------------------------------------------------------------

(defn- o-log-gauss [x mu sigma]
  (- (* -0.5 (js/Math.log (* 2 js/Math.PI)))
     (js/Math.log sigma)
     (* 0.5 (js/Math.pow (/ (- x mu) sigma) 2))))

(defn- o-logsumexp [xs]
  (let [m (reduce max xs)]
    (+ m (js/Math.log (reduce + (map #(js/Math.exp (- % m)) xs))))))

(defn- o-mixture-logp [y weights means sigmas]
  (let [wsum (reduce + weights)
        lpi (mapv #(js/Math.log (/ % wsum)) weights)]
    (o-logsumexp (mapv (fn [lp mu s] (+ lp (o-log-gauss y mu s)))
                       lpi means sigmas))))

(defn- o-conv-logp
  "Marginal of y_i = z + noise, z~N(theta,tau), noise~N(0,sigma): N(theta, sqrt(tau^2+sigma^2))."
  [ys theta tau sigma]
  (let [s (js/Math.sqrt (+ (* tau tau) (* sigma sigma)))]
    (reduce + (map #(o-log-gauss % theta s) ys))))

(defn- mean [xs] (/ (reduce + xs) (double (count xs))))
(defn- variance [xs]
  (let [m (mean xs)] (/ (reduce + (map #(* (- % m) (- % m)) xs)) (double (count xs)))))
(defn- it [a] (mx/item a))

;; ===========================================================================
;; 1. The omega field
;; ===========================================================================

(deftest omega-field-roundtrips
  (testing "Trace gains an optional omega field, default nil"
    (let [t0 (tr/make-trace {:gen-fn :g :args [] :choices cm/EMPTY :retval nil
                             :score (mx/scalar 0.0)})]
      (is (nil? (:omega t0)) "ordinary trace has nil omega")
      (is (= 6 (count (keys (into {} t0)))) "Trace now has 6 fields"))
    (let [om (mx/array [1.0 2.0 3.0])
          t (tr/make-trace {:gen-fn :g :args [] :choices cm/EMPTY :retval nil
                            :score (mx/scalar 0.0) :omega om})]
      (is (identical? om (:omega t)) "omega round-trips through make-trace")))
  (testing "encapsulated simulate stores omega and tags :joint"
    (let [{:keys [gf]} (enc/mixture-density {:weights [0.3 0.5 0.2] :means [-2 0 3]
                                             :sigmas [1 0.5 2] :k 16})
          t (p/simulate (dyn/with-key gf (rng/fresh-key 1)) [])]
      (is (some? (:omega t)) "encapsulated trace carries omega")
      (is (= :joint (tr/score-type t)) "encapsulated traces are :joint (see ns docstring)")
      (is (js/Number.isFinite (it (:score t))) "score is a finite log-xi"))))

;; ===========================================================================
;; 2. Estimator exactness vs independent oracle  (the EXACT marginal helper)
;; ===========================================================================

(deftest exact-marginal-matches-oracle
  (testing "mixture exact-log-density == hand oracle"
    (let [{:keys [exact-log-density]} (enc/mixture-density
                                       {:weights [0.3 0.5 0.2] :means [-2 0 3]
                                        :sigmas [1 0.5 2] :k 8})]
      (is (< (js/Math.abs (- (it (exact-log-density [] (mx/scalar 0.0)))
                             (o-mixture-logp 0.0 [0.3 0.5 0.2] [-2 0 3] [1 0.5 2]))) 1e-4)
          "logp(0) matches oracle -0.848418718")
      (is (< (js/Math.abs (- (it (exact-log-density [] (mx/scalar 1.0)))
                             (o-mixture-logp 1.0 [0.3 0.5 0.2] [-2 0 3] [1 0.5 2]))) 1e-4)
          "logp(1) matches oracle -2.531776980")))
  (testing "marginalized-gaussian exact-log-density == convolution oracle"
    (let [{:keys [exact-log-density marginal-sigma]} (enc/marginalized-gaussian
                                                      {:n 3 :tau 2.0 :sigma 0.5 :k 8})
          y (mx/array [1.5 1.0 2.0])]
      (is (< (js/Math.abs (- marginal-sigma (js/Math.sqrt 4.25))) 1e-6) "S=sqrt(4.25)")
      (is (< (js/Math.abs (- (it (exact-log-density [(mx/scalar 1.0)] y))
                             (o-conv-logp [1.5 1.0 2.0] 1.0 2.0 0.5))) 1e-4)
          "convolution marginal matches oracle"))))

;; ===========================================================================
;; 3. Eq 4.3 — unbiasedness of the density estimator:  E_omega[xi] = p(tau;x)
;; ===========================================================================

(defn- mc-estimates
  "R independent realized log-xi values for `gf` at value v under args, via assess."
  [gf args v addr R seed0]
  (let [obs (cm/set-value cm/EMPTY addr v)]
    (mapv (fn [i]
            (it (:weight (p/assess (dyn/with-key gf (rng/fresh-key (+ seed0 i)))
                                   args obs))))
          (range R))))

(deftest eq43-estimator-unbiased
  (testing "mixture: MC-average of xi over omega converges to the exact density"
    (let [w [0.3 0.5 0.2] mu [-2 0 3] sg [1 0.5 2]
          {:keys [gf]} (enc/mixture-density {:weights w :means mu :sigmas sg :k 64})
          y 0.0
          exact-logp (o-mixture-logp y w mu sg)
          R 600
          log-xis (mc-estimates gf [] (mx/scalar y) :y R 1000)
          xis (mapv js/Math.exp log-xis)
          mc-density (mean xis)
          ;; derived tolerance band 4*sd/sqrt(R) (math-verifier guidance)
          band (* 4.0 (/ (js/Math.sqrt (variance xis)) (js/Math.sqrt R)))]
      (is (< (js/Math.abs (- mc-density (js/Math.exp exact-logp))) (max band 1e-3))
          (str "E[xi]=" mc-density " ~ p(y)=" (js/Math.exp exact-logp) " band=" band))))
  (testing "marginalized-gaussian: MC-average of xi converges to the convolution marginal"
    (let [{:keys [gf]} (enc/marginalized-gaussian {:n 1 :tau 1.0 :sigma 1.0 :k 64})
          theta (mx/scalar 0.5) y (mx/array [1.3])
          exact-logp (o-conv-logp [1.3] 0.5 1.0 1.0)
          R 600
          log-xis (mc-estimates gf [theta] y :y R 5000)
          xis (mapv js/Math.exp log-xis)
          mc-density (mean xis)
          band (* 4.0 (/ (js/Math.sqrt (variance xis)) (js/Math.sqrt R)))]
      (is (< (js/Math.abs (- mc-density (js/Math.exp exact-logp))) (max band 1e-3))
          (str "E[xi]=" mc-density " ~ p(y)=" (js/Math.exp exact-logp))))))

(deftest eq43-variance-decreases-with-K
  (testing "more importance samples => lower estimator variance (same unbiased mean)"
    (let [w [0.3 0.5 0.2] mu [-2 0 3] sg [1 0.5 2]
          mk (fn [k] (:gf (enc/mixture-density {:weights w :means mu :sigmas sg :k k})))
          xis-k4  (mapv js/Math.exp (mc-estimates (mk 4)  [] (mx/scalar 0.0) :y 300 2000))
          xis-k64 (mapv js/Math.exp (mc-estimates (mk 64) [] (mx/scalar 0.0) :y 300 3000))]
      (is (< (variance xis-k64) (variance xis-k4))
          (str "Var(K=64)=" (variance xis-k64) " < Var(K=4)=" (variance xis-k4))))))

(deftest eq43-unbalanced-mixture
  (testing "unbiasedness holds for a heavily-unbalanced mixture (rare-component regime)"
    (let [w [0.95 0.05] mu [0 4] sg [0.5 0.5]
          {:keys [gf]} (enc/mixture-density {:weights w :means mu :sigmas sg :k 512})
          y 4.0  ;; rare-component-dominated: stresses the importance estimator + K-adequacy
          exact (js/Math.exp (o-mixture-logp y w mu sg))
          R 600
          xis (mapv js/Math.exp (mc-estimates gf [] (mx/scalar y) :y R 8000))
          mc (mean xis)
          band (* 4.0 (/ (js/Math.sqrt (variance xis)) (js/Math.sqrt R)))]
      (is (< (js/Math.abs (- mc exact)) (max band 1e-3))
          (str "E[xi]=" mc " ~ p(y)=" exact " band=" band " (unbalanced [0.95,0.05] at rare mode)")))))

;; ===========================================================================
;; 4. Eq 4.4 — the reciprocal is NOT 1/p for the naive estimator (Jensen guard)
;; ===========================================================================

(deftest eq44-naive-reciprocal-is-biased
  (testing "E[1/xi] > 1/p(y) for a finite-variance estimator (Jensen) — never invert xi"
    (let [w [0.3 0.5 0.2] mu [-2 0 3] sg [1 0.5 2]
          {:keys [gf]} (enc/mixture-density {:weights w :means mu :sigmas sg :k 4})
          y 0.0
          p (js/Math.exp (o-mixture-logp y w mu sg))
          log-xis (mc-estimates gf [] (mx/scalar y) :y 800 7000)
          mc-recip (mean (mapv #(/ 1.0 (js/Math.exp %)) log-xis))]
      (is (> mc-recip (/ 1.0 p))
          (str "E[1/xi]=" mc-recip " >= 1/p=" (/ 1.0 p)
               " (Jensen: inverting xi over-estimates 1/p — must use a "
               "meta-inference proposal for reciprocal unbiasedness)")))))

;; ===========================================================================
;; 5. Full GFI op semantics
;; ===========================================================================

(deftest gfi-op-semantics
  (let [{:keys [gf]} (enc/marginalized-gaussian {:n 2 :tau 1.0 :sigma 1.0 :k 32})
        y  (mx/array [1.0 2.0])
        y2 (mx/array [0.5 2.5])
        theta (mx/scalar 0.3)
        obs (cm/set-value cm/EMPTY :y y)]
    (testing "generate: fully-constrained weight == trace score (= log xi)"
      (let [{:keys [trace weight]} (p/generate (dyn/with-key gf (rng/fresh-key 11)) [theta] obs)]
        (is (< (js/Math.abs (- (it weight) (it (:score trace)))) 1e-6)
            "generate weight equals stored score")
        (is (some? (:omega trace)) "generate stores omega")))
    (testing "generate empty constraints: weight = log xi - log q(v)  (genmlx-evab)"
      ;; NOT 0. The value is drawn from the internal proposal q, and the trace's
      ;; score is an ESTIMATE xi != q(v), so generate's score/weight tie
      ;; (score - weight = log q(what was proposed)) makes the weight the
      ;; estimator/proposal ratio. Oracle: q for marginalized-gaussian is the
      ;; hand-derived convolution marginal o-conv-logp (tau=1, sigma=1 => S=sqrt 2).
      (let [{:keys [trace weight]} (p/generate (dyn/with-key gf (rng/fresh-key 12))
                                               [theta] cm/EMPTY)
            v (mx/realize-clj (cm/get-value (cm/get-submap (:choices trace) :y)))
            log-q (o-conv-logp v (it theta) 1.0 1.0)]
        (is (< (js/Math.abs (- (it weight) (- (it (:score trace)) log-q))) 1e-4)
            (str "weight " (it weight) " = log xi " (it (:score trace))
                 " - log q(v) " log-q))
        (is (not= 0.0 (it weight))
            "a genuine estimator makes xi != q(v), so the weight is NOT 0")))
    (testing "assess: weight is a finite log-xi; retval is the value"
      (let [{:keys [retval weight]} (p/assess (dyn/with-key gf (rng/fresh-key 13)) [theta] obs)]
        (is (js/Number.isFinite (it weight)))
        (is (= (mx/realize-clj y) (mx/realize-clj retval))
            "observed value vector round-trips as retval")))
    (testing "project: all-selected == score, none == 0"
      (let [t (:trace (p/generate (dyn/with-key gf (rng/fresh-key 14)) [theta] obs))]
        (is (< (js/Math.abs (- (it (p/project gf t (sel/select :y))) (it (:score t)))) 1e-9)
            "project of the observed addr == score")
        (is (= 0.0 (it (p/project gf t sel/none))) "project of nothing == 0")))
    (testing "update identity (same value) => weight EXACTLY 0, omega reused"
      (let [t (:trace (p/generate (dyn/with-key gf (rng/fresh-key 15)) [theta] obs))
            {:keys [trace weight]} (p/update gf t obs)]
        (is (= 0.0 (it weight)) "identity update weight is exactly 0")
        (is (identical? (:omega t) (:omega trace)) "omega reused (same object)")))
    (testing "update changed value => weight == log xi' - old score, discard old"
      (let [t (:trace (p/generate (dyn/with-key gf (rng/fresh-key 16)) [theta] obs))
            {:keys [trace weight discard]} (p/update (dyn/with-key gf (rng/fresh-key 17))
                                                     t (cm/set-value cm/EMPTY :y y2))]
        (is (< (js/Math.abs (- (it weight) (- (it (:score trace)) (it (:score t))))) 1e-5)
            "weight = new score - old score")
        (is (= (mx/realize-clj y) (mx/realize-clj (cm/get-value (cm/get-submap discard :y))))
            "discard holds the old value")
        (is (not (identical? (:omega t) (:omega trace))) "omega resampled on a genuine move")))
    (testing "update-with-args: no-change & empty => weight 0; arg change => move"
      (let [t (:trace (p/generate (dyn/with-key gf (rng/fresh-key 18)) [theta] obs))
            {w0 :weight} (p/update-with-args gf t [theta] diff/no-change cm/EMPTY)
            {trace :trace w1 :weight} (p/update-with-args (dyn/with-key gf (rng/fresh-key 19))
                                                          t [(mx/scalar 0.9)] :unknown cm/EMPTY)]
        (is (= 0.0 (it w0)) "no-change update-with-args is exactly 0")
        (is (< (js/Math.abs (- (it w1) (- (it (:score trace)) (it (:score t))))) 1e-5)
            "arg-change weight = log xi'(theta') - old score")
        (is (= [(it (mx/scalar 0.9))] (mapv it (:args trace))) "args updated to theta'")))
    (testing "regenerate: empty selection => 0; selected => (log xi'-log xi) - (log q(y')-log q(y))"
      ;; genmlx-evab: the weight is NOT the bare estimator ratio. regenerate
      ;; proposes y' from the internal proposal q itself, so the q ratio must be
      ;; divided back out. log q here is the hand-derived convolution marginal
      ;; (o-conv-logp), NEVER the implementation.
      (let [t (:trace (p/generate (dyn/with-key gf (rng/fresh-key 20)) [theta] obs))
            {w0 :weight} (p/regenerate gf t sel/none)
            {trace :trace w1 :weight} (p/regenerate (dyn/with-key gf (rng/fresh-key 21))
                                                    t (sel/select :y))
            y-old (mx/realize-clj (cm/get-value (cm/get-submap (:choices t) :y)))
            y-new (mx/realize-clj (cm/get-value (cm/get-submap (:choices trace) :y)))
            th (it theta)
            expect (- (- (it (:score trace)) (it (:score t)))
                      (- (o-conv-logp y-new th 1.0 1.0) (o-conv-logp y-old th 1.0 1.0)))]
        (is (= 0.0 (it w0)) "unselected regenerate weight 0")
        (is (not= y-old y-new) "selected regenerate proposes a fresh value")
        (is (< (js/Math.abs (- (it w1) expect)) 1e-4)
            (str "weight " (it w1) " = (log xi'-log xi) - (log q(y')-log q(y)) = " expect))
        ;; and it is NOT the bare estimator ratio the pre-genmlx-evab code returned
        (is (> (js/Math.abs (- (it w1) (- (it (:score trace)) (it (:score t))))) 1e-3)
            "the q ratio is genuinely subtracted (not accidentally ~0)")))))

;; ===========================================================================
;; 5b. Robustness & edge cases (adversarial-review hardening)
;; ===========================================================================

(deftest encapsulated-robustness
  (testing "update-with-args at x'=x with a conservative :unknown argdiff is a no-op"
    ;; The argdiff is a trusted hint; :unknown at genuinely-unchanged args must
    ;; NOT pay spurious estimator noise — it reduces to update (weight 0).
    (let [{:keys [gf]} (enc/marginalized-gaussian {:n 2 :tau 1.0 :sigma 1.0 :k 16})
          theta (mx/scalar 0.4) y (mx/array [1.0 2.0])
          t (:trace (p/generate (dyn/with-key gf (rng/fresh-key 60)) [theta]
                                (cm/set-value cm/EMPTY :y y)))
          {w :weight tr :trace} (p/update-with-args (dyn/with-key gf (rng/fresh-key 61))
                                                    t [theta] :unknown cm/EMPTY)]
      (is (= 0.0 (it w)) ":unknown at unchanged args is exactly 0")
      (is (identical? (:omega t) (:omega tr)) "omega reused (no spurious resample)")))
  (testing "robust to a hand-built trace with nil omega (uses the stored score as old xi)"
    (let [{:keys [gf]} (enc/marginalized-gaussian {:n 1 :tau 1.0 :sigma 1.0 :k 16})
          theta (mx/scalar 0.2) y (mx/array [1.0])
          t0 (:trace (p/generate (dyn/with-key gf (rng/fresh-key 50)) [theta]
                                 (cm/set-value cm/EMPTY :y y)))
          t (assoc t0 :omega nil)]          ;; mimic a deserialized / hand-built trace
      (is (nil? (:omega t)) "precondition: omega cleared")
      (let [{trace :trace w :weight} (p/update (dyn/with-key gf (rng/fresh-key 51))
                                               t (cm/set-value cm/EMPTY :y (mx/array [1.7])))]
        (is (some? (:omega trace)) "update draws fresh omega")
        ;; update does not propose the value, so no q ratio: w = log xi' - log xi
        (is (< (js/Math.abs (- (it w) (- (it (:score trace)) (it (:score t))))) 1e-5)
            "update weight = log xi' - log xi (stored score is the old xi)"))
      (let [{trace :trace w :weight} (p/regenerate (dyn/with-key gf (rng/fresh-key 52))
                                                   t (sel/select :y))
            y-old (mx/realize-clj (cm/get-value (cm/get-submap (:choices t) :y)))
            y-new (mx/realize-clj (cm/get-value (cm/get-submap (:choices trace) :y)))
            th (it theta)
            expect (- (- (it (:score trace)) (it (:score t)))
                      (- (o-conv-logp y-new th 1.0 1.0) (o-conv-logp y-old th 1.0 1.0)))]
        (is (some? (:omega trace)) "regenerate draws fresh omega")
        ;; genmlx-evab: correctness, not merely finiteness — the proposal ratio
        ;; is divided out even when the stored omega was missing.
        (is (< (js/Math.abs (- (it w) expect)) 1e-4)
            (str "weight " (it w) " = (log xi'-log xi) - (log q(y')-log q(y)) = " expect)))))
  (testing "EncapsulatedGF satisfies the generic project laws (via the law framework)"
    (let [g (dyn/auto-key (:gf (enc/marginalized-gaussian {:n 1 :tau 1.0 :sigma 1.0 :k 8})))]
      (doseq [law-name [:project-all-equals-score :project-none-equals-zero]]
        (is (:pass? (gfi/check-law law-name g [(mx/scalar 0.0)]))
            (str law-name " holds on EncapsulatedGF")))))
  (testing "pseudo-marginal-mh rejects a non-encapsulated gf"
    (is (thrown? js/Error
                 (enc/pseudo-marginal-mh {:enc-gf {:not :encapsulated} :y (mx/array [1.0])
                                          :theta0 0.0 :log-prior (fn [_] 0.0)
                                          :step 0.5 :samples 1})))))

;; ===========================================================================
;; 5c. genmlx-evab — regenerate/generate divide out the INTERNAL PROPOSAL q
;;
;; The two ops that propose the observed value themselves (regenerate on the
;; selected address, generate with the address unconstrained) must divide the
;; proposal density back out:
;;   regenerate: W = (log xi' - log xi) - (log q(v') - log q(v))
;;   generate  : W =  log xi           -  log q(v)
;; Returning the bare estimator ratio makes MH on the address target ~p(v)^2.
;; ===========================================================================

(def ^:private LOG2PI (js/Math.log (* 2 js/Math.PI)))

(defn- mx-log-gauss [v mu sigma]
  (mx/subtract
   (mx/subtract (mx/scalar (* -0.5 LOG2PI)) (mx/scalar (js/Math.log sigma)))
   (mx/multiply (mx/scalar 0.5) (mx/square (mx/divide (mx/subtract v (mx/scalar mu))
                                                      (mx/scalar sigma))))))

(defn- degenerate-enc
  "An EncapsulatedGF whose estimator IGNORES omega and returns the EXACT density
   of the very distribution :sample-value draws from (xi = q = N(0,1)). It is
   therefore observationally an ordinary exact-density GF whose internal
   proposal is the prior, so the ordinary GFI laws must hold EXACTLY:
   regenerate on the selected address costs 0, unconstrained generate costs 0.
   `extra` lets a caller drop :log-proposal-density to test the throw."
  ([] (degenerate-enc {:log-proposal-density (fn [_args v] (mx-log-gauss v 0.0 1.0))}))
  ([extra]
   (enc/encapsulated
    (merge {:addr :v
            :sample-value (fn [k _args] (rng/normal k []))
            :sample-omega (fn [_k _args _v] (mx/scalar 0.0))
            :log-density-estimate (fn [_args v _om] (mx-log-gauss v 0.0 1.0))}
           extra))))

(deftest degenerate-estimator-obeys-exact-density-laws
  (testing "xi = q exactly => regenerate on the selected address costs EXACTLY 0"
    ;; This is the confirm-or-close oracle of genmlx-evab. With the bare
    ;; estimator ratio it returns log p(v') - log p(v) (measured 0.2612 on
    ;; seeds 7/8), i.e. the acceptance of an MH move that targets p(v)^2.
    (let [g (degenerate-enc)
          t (p/simulate (dyn/with-key g (rng/fresh-key 7)) [])
          {tr :trace w :weight} (p/regenerate (dyn/with-key g (rng/fresh-key 8))
                                              t (sel/select :v))
          v  (it (cm/get-value (cm/get-submap (:choices t) :v)))
          v' (it (cm/get-value (cm/get-submap (:choices tr) :v)))]
      (is (not= v v') "precondition: the proposal actually moved the value")
      (is (> (js/Math.abs (- (o-log-gauss v' 0.0 1.0) (o-log-gauss v 0.0 1.0))) 0.05)
          "precondition: log p(v') - log p(v) is far from 0, so 0 is not vacuous")
      (is (< (js/Math.abs (it w)) 1e-5)
          (str "regenerate weight " (it w) " must be 0 (exact-density law); the "
               "undivided ratio would be "
               (- (o-log-gauss v' 0.0 1.0) (o-log-gauss v 0.0 1.0))))))
  (testing "xi = q exactly => unconstrained generate costs EXACTLY 0"
    (let [g (degenerate-enc)
          {trace :trace w :weight} (p/generate (dyn/with-key g (rng/fresh-key 9))
                                               [] cm/EMPTY)]
      (is (< (js/Math.abs (it w)) 1e-5)
          (str "unconstrained generate weight " (it w) " must be 0 when xi = q"))))
  (testing "without :log-proposal-density the two PROPOSING ops throw (no plausible-but-wrong weight)"
    (let [g (degenerate-enc {})
          t (p/simulate (dyn/with-key g (rng/fresh-key 2)) [])]
      (is (thrown? js/Error (p/regenerate (dyn/with-key g (rng/fresh-key 3))
                                          t (sel/select :v)))
          "regenerate on the selected address throws")
      (is (thrown? js/Error (p/generate (dyn/with-key g (rng/fresh-key 4)) [] cm/EMPTY))
          "unconstrained generate throws")
      ;; every op that does NOT propose a value still works: a black-box sampler
      ;; with an unknown density remains usable for pseudo-marginal MCMC.
      (let [obs (cm/set-value cm/EMPTY :v (mx/scalar 0.25))]
        (is (< (js/Math.abs (- (it (:weight (p/generate (dyn/with-key g (rng/fresh-key 6))
                                                        [] obs)))
                               (o-log-gauss 0.25 0.0 1.0))) 1e-5)
            "constrained generate = log xi (no q involved)")
        (is (js/Number.isFinite (it (:weight (p/update (dyn/with-key g (rng/fresh-key 7))
                                                       t obs))))
            "update works")
        (is (= 0.0 (it (:weight (p/regenerate g t sel/none)))) "unselected regenerate works")
        (is (js/Number.isFinite (it (p/project g t (sel/select :v)))) "project works")))))

(deftest regenerate-mh-targets-p-not-p-squared
  ;; THE consequence test. An independent MH kernel built out of p/regenerate
  ;; (propose v' from the internal proposal, accept with min(1, exp(weight)))
  ;; must leave the GF's own density p invariant. Here p = q = N(0.7, 1) and the
  ;; estimator is genuinely stochastic: xi = p(v) * exp(a Z - a^2/2), Z~N(0,1),
  ;; so E[xi] = p(v) (Eq 4.3) while xi != p(v) in every realization.
  ;;   correct weight (log xi'-log xi) - (log q(v')-log q(v))  => stationary N(0.7, 1)
  ;;   bare ratio      log xi'-log xi                          => stationary ∝ p*q = p^2
  ;;                                                              = N(0.7, 0.5)
  ;; The VARIANCE separates them by a factor of 2. Oracle: the conjugate closed
  ;; forms 1.0 and 0.5, derived by hand (product of two N(0.7,1) densities).
  (let [a 0.4
        mu 0.7
        g (enc/encapsulated
           {:addr :v
            :sample-value (fn [k _args] (mx/add (mx/scalar mu) (rng/normal k [])))
            :sample-omega (fn [k _args _v] (rng/normal k []))
            :log-density-estimate
            (fn [_args v z] (mx/add (mx-log-gauss v mu 1.0)
                                    (mx/subtract (mx/multiply (mx/scalar a) z)
                                                 (mx/scalar (* 0.5 a a)))))
            :log-proposal-density (fn [_args v] (mx-log-gauss v mu 1.0))})
        n-samples 4000
        n-burn 500
        samples
        (loop [t (p/simulate (dyn/with-key g (rng/fresh-key 771)) [])
               i 0
               rk (rng/fresh-key 772)
               acc (transient [])]
          (if (>= i (+ n-burn n-samples))
            (persistent! acc)
            (let [[k1 k2 rk'] (rng/split-n rk 3)
                  {tr :trace w :weight} (p/regenerate (dyn/with-key g k1) t (sel/select :v))
                  accept? (< (js/Math.log (it (rng/uniform k2 []))) (it w))
                  nt (if accept? tr t)]
              (recur nt (inc i) rk'
                     (if (>= i n-burn)
                       (conj! acc (it (cm/get-value (cm/get-submap (:choices nt) :v))))
                       acc)))))
        m (mean samples)
        v (variance samples)]
    (testing "the regenerate-MH chain is stationary for p, not for p^2"
      (is (< (js/Math.abs (- m mu)) 0.08) (str "chain mean " m " ~ " mu))
      (is (< (js/Math.abs (- v 1.0)) 0.13)
          (str "chain variance " v " ~ 1.0 (p); the undivided-ratio kernel gives "
               "0.5 (p^2)"))
      (is (> v 0.75) "variance is nowhere near the p^2 value 0.5"))))

;; ===========================================================================
;; 6. Pseudo-marginal MCMC stationarity (THE headline)
;; ===========================================================================

(defn- normal-logprior [m0 s0]
  (fn [th] (o-log-gauss th m0 s0)))

(defn- exact-marginal-mh
  "Vanilla RW-MH on theta using the EXACT marginal likelihood (no estimator).
   The control that pseudo-marginal MH must match."
  [exact-log-density y log-prior theta0 step samples burn key]
  (let [rk (rng/ensure-key key)]
    (loop [theta theta0 i 0 rk rk acc (transient [])]
      (if (>= i (+ burn samples))
        (persistent! acc)
        (let [[kp ka rk'] (rng/split-n rk 3)
              theta' (+ theta (* step (it (rng/normal kp []))))
              ll  (it (exact-log-density [(mx/scalar theta)]  y))
              ll' (it (exact-log-density [(mx/scalar theta')] y))
              la (+ (- ll' ll) (- (log-prior theta') (log-prior theta)))
              accept? (< (js/Math.log (it (rng/uniform ka []))) la)
              nt (if accept? theta' theta)]
          (recur nt (inc i) rk' (if (>= i burn) (conj! acc nt) acc)))))))

(deftest pseudo-marginal-stationarity
  ;; Normal-Normal conjugate: theta ~ N(0,2); y_i ~ N(theta, S=1) via tau=0.6,sigma=0.8.
  ;; data = [1,2,3] => posterior N(24/13, 4/13) = N(1.846154, 0.307692).  [oracle]
  (let [{:keys [gf exact-log-density]} (enc/marginalized-gaussian {:n 3 :tau 0.6 :sigma 0.8 :k 8})
        y (mx/array [1.0 2.0 3.0])
        log-prior (normal-logprior 0.0 2.0)
        post-mean (/ 24.0 13.0)
        post-var  (/ 4.0 13.0)]
    (testing "PM-MH posterior mean & variance match the exact conjugate posterior"
      (let [{:keys [samples accept-rate]}
            (enc/pseudo-marginal-mh {:enc-gf gf :y y :theta0 0.0 :log-prior log-prior
                                     :step 0.7 :samples 8000 :burn 2000
                                     :key (rng/fresh-key 4242)})
            m (mean samples) v (variance samples)]
        (is (> accept-rate 0.2) (str "chain mixes (accept-rate " accept-rate ")"))
        (is (< (js/Math.abs (- m post-mean)) 0.06)
            (str "PM-MH mean " m " ~ 24/13=" post-mean " (estimated likelihood, K=8)"))
        (is (< (js/Math.abs (- v post-var)) 0.06)
            (str "PM-MH var " v " ~ 4/13=" post-var))))
    (testing "PM-MH (estimated likelihood) matches exact-marginal MH (closed form)"
      (let [exact-samples (exact-marginal-mh exact-log-density y log-prior
                                             0.0 0.7 8000 2000 (rng/fresh-key 99))
            {:keys [samples]} (enc/pseudo-marginal-mh
                               {:enc-gf gf :y y :theta0 0.0 :log-prior log-prior
                                :step 0.7 :samples 8000 :burn 2000 :key (rng/fresh-key 7)})]
        (is (< (js/Math.abs (- (mean samples) (mean exact-samples))) 0.06)
            (str "PM mean " (mean samples) " ~ exact-MH mean " (mean exact-samples)))))))

;; ===========================================================================
;; 7. GFI laws registered
;; ===========================================================================

(deftest gfi-laws-registered
  (let [names (set (map :name gfi/laws))]
    (is (contains? names :encapsulated-estimator-unbiased) "Eq 4.3 law present")
    (is (contains? names :encapsulated-identity-update-zero) "identity-zero law present")
    (is (contains? names :pseudo-marginal-stationarity) "pseudo-marginal law present")
    (is (contains? names :encapsulated-regenerate-divides-proposal)
        "regenerate proposal-ratio law present (genmlx-evab)")))

(deftest gfi-laws-hold
  (testing "the four §4.5 laws pass when run directly"
    (doseq [law-name [:encapsulated-estimator-unbiased
                      :encapsulated-identity-update-zero
                      :pseudo-marginal-stationarity
                      :encapsulated-regenerate-divides-proposal]]
      (let [{:keys [pass? error]} (gfi/check-law law-name
                                                 (dyn/auto-key
                                                  (:gf (enc/mixture-density
                                                        {:weights [0.5 0.5] :means [0 1]
                                                         :sigmas [1 1] :k 8})))
                                                 [])]
        (is pass? (str law-name " holds" (when error (str " — ERROR: " error))))))))

(cljs.test/run-tests)
