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
    (testing "generate empty constraints == simulate, weight 0"
      (let [{:keys [weight]} (p/generate (dyn/with-key gf (rng/fresh-key 12)) [theta] cm/EMPTY)]
        (is (= 0.0 (it weight)) "unconstrained generate has weight 0")))
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
    (testing "regenerate: empty selection => weight 0; selected => fresh value + finite weight"
      (let [t (:trace (p/generate (dyn/with-key gf (rng/fresh-key 20)) [theta] obs))
            {w0 :weight} (p/regenerate gf t sel/none)
            {trace :trace w1 :weight} (p/regenerate (dyn/with-key gf (rng/fresh-key 21))
                                                    t (sel/select :y))]
        (is (= 0.0 (it w0)) "unselected regenerate weight 0")
        (is (js/Number.isFinite (it w1)) "selected regenerate weight finite")
        (is (not (identical? (cm/get-value (cm/get-submap (:choices t) :y))
                             (cm/get-value (cm/get-submap (:choices trace) :y))))
            "selected regenerate proposes a fresh value")))))

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
        (is (js/Number.isFinite (it w)) "weight finite"))
      (let [{trace :trace w :weight} (p/regenerate (dyn/with-key gf (rng/fresh-key 52))
                                                   t (sel/select :y))]
        (is (some? (:omega trace)) "regenerate draws fresh omega")
        (is (js/Number.isFinite (it w)) "weight finite"))))
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
    (is (contains? names :pseudo-marginal-stationarity) "pseudo-marginal law present")))

(deftest gfi-laws-hold
  (testing "the three §4.5 laws pass when run directly"
    (doseq [law-name [:encapsulated-estimator-unbiased
                      :encapsulated-identity-update-zero
                      :pseudo-marginal-stationarity]]
      (let [{:keys [pass? error]} (gfi/check-law law-name
                                                 (dyn/auto-key
                                                  (:gf (enc/mixture-density
                                                        {:weights [0.5 0.5] :means [0 1]
                                                         :sigmas [1 1] :k 8})))
                                                 [])]
        (is pass? (str law-name " holds" (when error (str " — ERROR: " error))))))))

(cljs.test/run-tests)
