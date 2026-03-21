(ns ipomdp
  "I-POMDP Investment Game — multi-agent reasoning with hidden types.

   Two agents (investor, trustee) play a trust game. The investor sends
   a fraction of their endowment (tripled by the experimenter), and the
   trustee returns some fraction. Each has a hidden 'guilt' parameter
   that penalizes inequitable outcomes.

   The investor reasons about the trustee's likely response (theory of
   mind), and an observer can infer an agent's guilt from their actions
   (inverse I-POMDP).

   Inspired by Berg et al. (1995) and Gmytrasiewicz & Doshi (2005).

   Domains:
     guilt:  0=selfish (g=0.0), 1=guilt-averse (g=0.7)
     action: 0=low fraction (0.2), 1=high fraction (0.8)"
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.inference.exact :as exact]
            [genmlx.inference.kernel :as kern])
  (:require-macros [genmlx.gen :refer [gen]]))

;; =========================================================================
;; Game parameters
;; =========================================================================

(def fractions (mx/array #js [0.2 0.8]))
(def guilts (mx/array #js [0.0 0.7]))
(def multiplier 3.0)

(mx/eval! fractions guilts)

(defn payout-investor [fi ft]
  (let [fiv (mx/idx fractions fi)
        ftv (mx/idx fractions ft)]
    (mx/add (mx/subtract (mx/scalar 1.0) fiv)
            (mx/multiply (mx/scalar multiplier) (mx/multiply fiv ftv)))))

(defn payout-trustee [fi ft]
  (let [fiv (mx/idx fractions fi)
        ftv (mx/idx fractions ft)]
    (mx/multiply (mx/scalar multiplier)
      (mx/multiply fiv (mx/subtract (mx/scalar 1.0) ftv)))))

;; =========================================================================
;; Level 0: Trustee Q-values (no theory of mind)
;; =========================================================================

(defn compute-trustee-q0
  "Trustee's Q-value without modeling the investor.
   Q(tg, fi, ft) = payout - guilt * max(my_payout - their_payout, 0).
   Returns [2, 2, 2] tensor: [guilt, fi-received, ft-choice].
   Vectorized: computes all 8 entries in one pass via broadcasting."
  []
  (let [;; fraction values: fi-axis [2,1], ft-axis [1,2]
        fiv (mx/reshape fractions [2 1])   ;; [2,1] — investor fraction
        ftv (mx/reshape fractions [1 2])   ;; [1,2] — trustee fraction
        ;; payouts: [2, 2] over (fi, ft) via broadcasting
        pt (mx/multiply (mx/scalar multiplier)
             (mx/multiply fiv (mx/subtract (mx/scalar 1.0) ftv)))
        pi (mx/add (mx/subtract (mx/scalar 1.0) fiv)
                   (mx/multiply (mx/scalar multiplier) (mx/multiply fiv ftv)))
        ;; guilt: [2, 1, 1] for broadcasting over (fi, ft)
        g  (mx/reshape guilts [2 1 1])
        ;; inequity and Q: broadcast to [2, 2, 2] = [guilt, fi, ft]
        inequity (mx/maximum (mx/subtract pt pi) (mx/scalar 0.0))
        t (mx/subtract pt (mx/multiply g inequity))]
    (mx/eval! t)
    t))

;; =========================================================================
;; Level 1: Trustee response model (gen function, exact enumerable)
;; =========================================================================

(defn trustee-response
  "Trustee's response given investor's offer.
   Traces :guilt (hidden type) and :ft (action).
   Uses softmax(beta * Q0) for action selection."
  [trustee-q0 fi-val beta]
  (let [q-for-fi (mx/multiply (mx/scalar beta)
                   (mx/idx trustee-q0 fi-val 1))
        _ (mx/eval! q-for-fi)]
    (gen []
      (let [g (trace :guilt (dist/weighted [1.0 1.0]))
            ft (trace :ft (dist/categorical (mx/take-idx q-for-fi g 0)))]
        ft))))

;; =========================================================================
;; Level 1: Investor model (models trustee via exact enumeration)
;; =========================================================================

(defn compute-investor-q1
  "Investor's Q-values at level 1, modeling the trustee.
   For each (investor-guilt, fi), compute expected payoff given
   trustee's level-0 response distribution.
   Returns [2, 2] tensor: [guilt, fi].
   Vectorized: inner loop replaced by tensor dot product over ft."
  [trustee-q0 beta]
  (let [compute-q
        (fn [fi-val]
          (let [joint (exact/exact-joint (trustee-response trustee-q0 fi-val beta) [] nil)
                m (exact/joint-marginal (:log-probs joint) (:axes joint) #{:ft})
                p-ft (mx/exp (:log-probs m))   ;; [2] over ft
                ;; payoffs for this fi-val over all ft: [2]
                pi (mx/add (mx/subtract (mx/scalar 1.0)
                             (mx/idx fractions fi-val))
                           (mx/multiply (mx/scalar multiplier)
                             (mx/multiply (mx/idx fractions fi-val) fractions)))
                pt (mx/multiply (mx/scalar multiplier)
                     (mx/multiply (mx/idx fractions fi-val)
                       (mx/subtract (mx/scalar 1.0) fractions)))
                ;; inequity from investor's perspective: max(pi - pt, 0), shape [2]
                ineq (mx/maximum (mx/subtract pi pt) (mx/scalar 0.0))
                ;; guilt [2,1] x ineq [1,2] -> [2,2] (guilt, ft)
                g (mx/reshape guilts [2 1])
                ;; investor Q per (guilt, ft): pi - g * ineq, shape [2,2]
                q-per-ft (mx/subtract pi (mx/multiply g ineq))
                ;; expected Q = sum_ft [p_ft * q_per_ft], shape [2]
                _ (mx/eval! p-ft q-per-ft)
                eq (mx/sum (mx/multiply p-ft q-per-ft) [1])]
            eq))
        eq-lo (compute-q 0)   ;; [2] over guilt
        eq-hi (compute-q 1)   ;; [2] over guilt
        ;; stack: [2, 2] = [guilt, fi]
        t (mx/stack [eq-lo eq-hi] 1)]
    (mx/eval! t)
    t))

(defn investor-model
  "Investor at level 1: models trustee's response.
   Traces :guilt (hidden type) and :fi (action).
   Action weights = softmax(beta * Q1[guilt, fi])."
  [investor-q1 beta]
  (let [logits (mx/multiply (mx/scalar beta) investor-q1)
        _ (mx/eval! logits)]
    (gen []
      (let [ig (trace :guilt (dist/weighted [1.0 1.0]))
            fi (trace :fi (dist/categorical (mx/take-idx logits ig 0)))]
        fi))))

;; =========================================================================
;; Continuous guilt + sequential Bayesian updating
;; =========================================================================

(defn- expected-q
  "Investor's expected Q-value for a given continuous guilt and action fi.
   Computes E_ft[payout - guilt * inequity] using the trustee's
   response distribution (precomputed from level-0 discrete model)."
  [guilt fi-val p-ft]
  (reduce (fn [acc ft]
            (let [pi (payout-investor fi-val ft)
                  pt (payout-trustee fi-val ft)
                  _ (mx/eval! pi pt)
                  ineq (max (- (mx/item pi) (mx/item pt)) 0.0)
                  p (mx/item (mx/idx p-ft ft))]
              (mx/add acc (mx/scalar (* p (- (mx/item pi) (* (mx/item guilt) ineq)))))))
          (mx/scalar 0.0)
          [0 1]))

(defn sequential-trust-model
  "Observer watches an investor play N rounds of the trust game.
   The investor has a continuous guilt parameter (not discrete).

   guilt ~ N(0, 1) transformed via softplus to ensure positivity.
   Each round: investor chooses fi ~ softmax(beta * Q(guilt, fi)).
   Observer sees all actions and infers guilt via MCMC.

   Traces: :guilt (continuous), :r0..:r4 (categorical actions)."
  [trustee-q0 beta n-rounds]
  (let [p-fts (mapv (fn [fi-val]
                      (let [model (trustee-response trustee-q0 fi-val beta)
                            joint (exact/exact-joint model [] nil)
                            m (exact/joint-marginal (:log-probs joint) (:axes joint) #{:ft})
                            probs (mx/exp (:log-probs m))
                            _ (mx/eval! probs)]
                        probs))
                    [0 1])]
    (gen []
      (let [guilt-raw (trace :guilt (dist/gaussian 0.0 1.0))
            guilt (mx/log (mx/add (mx/exp guilt-raw) (mx/scalar 1.0)))
            q-lo (expected-q guilt 0 (nth p-fts 0))
            q-hi (expected-q guilt 1 (nth p-fts 1))
            logits (mx/multiply (mx/scalar beta) (mx/stack #js [q-lo q-hi]))]
        (when (> n-rounds 0) (trace :r0 (dist/categorical logits)))
        (when (> n-rounds 1) (trace :r1 (dist/categorical logits)))
        (when (> n-rounds 2) (trace :r2 (dist/categorical logits)))
        (when (> n-rounds 3) (trace :r3 (dist/categorical logits)))
        (when (> n-rounds 4) (trace :r4 (dist/categorical logits)))))))


;; =========================================================================
;; Verification
;; =========================================================================

(println "\n== I-POMDP Investment Game Example ==")
(println "   Multi-agent trust with theory of mind + inverse inference\n")

(def pass-count (atom 0))
(def fail-count (atom 0))

(defn check [label pred]
  (if pred
    (do (swap! pass-count inc) (println (str "  PASS: " label)))
    (do (swap! fail-count inc) (println (str "  FAIL: " label)))))

(defn check-close [label expected actual tol]
  (if (<= (abs (- expected actual)) tol)
    (do (swap! pass-count inc) (println (str "  PASS: " label)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " label " (expected " expected " got " actual ")")))))

;; -------------------------------------------------------------------------
;; Setup
;; -------------------------------------------------------------------------

(def beta 5.0)
(def tq0 (compute-trustee-q0))
(def iq1 (compute-investor-q1 tq0 beta))

;; -------------------------------------------------------------------------
;; 1. Trustee level-0 response
;; -------------------------------------------------------------------------

(println "-- 1. Trustee level-0 response --")

(let [model (trustee-response tq0 0 beta)
      joint (exact/exact-joint model [] nil)
      m (exact/joint-marginal (:log-probs joint) (:axes joint) #{:ft})
      probs (mx/exp (:log-probs m))
      _ (mx/eval! probs)
      p-lo (mx/item (mx/idx probs 0))]
  (println (str "   P(return low | got low offer)  = " (.toFixed p-lo 4)))
  (check-close "P(return low | got low) ~ 0.858" 0.858 p-lo 0.01))

(let [model (trustee-response tq0 1 beta)
      joint (exact/exact-joint model [] nil)
      m (exact/joint-marginal (:log-probs joint) (:axes joint) #{:ft})
      probs (mx/exp (:log-probs m))
      _ (mx/eval! probs)
      p-lo (mx/item (mx/idx probs 0))]
  (println (str "   P(return low | got high offer) = " (.toFixed p-lo 4)))
  (check-close "P(return low | got high) ~ 0.973" 0.973 p-lo 0.01))

;; -------------------------------------------------------------------------
;; 2. Investor level-1 policy
;; -------------------------------------------------------------------------

(println "\n-- 2. Investor level-1 policy --")

(let [model (investor-model iq1 beta)
      p-lo-selfish (exact/pr model :fi 0 :given :guilt 0)
      p-lo-guilty  (exact/pr model :fi 0 :given :guilt 1)]
  (println (str "   Selfish investor: P(low offer)       = " (.toFixed p-lo-selfish 4)))
  (check-close "selfish: P(low offer) ~ 0.7786" 0.7786 p-lo-selfish 1e-3)
  (println (str "   Guilt-averse investor: P(low offer)  = " (.toFixed p-lo-guilty 4)))
  (check-close "guilt-averse: P(low offer) ~ 0.3817" 0.3817 p-lo-guilty 1e-3))

;; -------------------------------------------------------------------------
;; 3. Theory of mind reversal
;; -------------------------------------------------------------------------

(println "\n-- 3. Theory of mind reversal --")
(println "   Level-0 investor (naive) prefers high offers.")
(println "   Level-1 investor (strategic) switches to low offers.")

(let [q0-lo (/ (+ (mx/item (payout-investor 0 0))
                  (mx/item (payout-investor 0 1))) 2)
      q0-hi (/ (+ (mx/item (payout-investor 1 0))
                  (mx/item (payout-investor 1 1))) 2)
      q1-g0 (.tolist (mx/idx iq1 0))]
  (println (str "   Level-0 Q(low)=" (.toFixed q0-lo 2) " Q(high)=" (.toFixed q0-hi 2)
               " -> prefers " (if (> q0-hi q0-lo) "HIGH" "LOW")))
  (println (str "   Level-1 Q(low)=" (.toFixed (aget q1-g0 0) 2) " Q(high)=" (.toFixed (aget q1-g0 1) 2)
               " -> prefers " (if (> (aget q1-g0 0) (aget q1-g0 1)) "LOW" "HIGH")))
  (check "level 0: prefers high offer" (> q0-hi q0-lo))
  (check "level 1 selfish: prefers low offer" (> (aget q1-g0 0) (aget q1-g0 1))))

;; -------------------------------------------------------------------------
;; 4. Inverse I-POMDP: infer guilt from action
;; -------------------------------------------------------------------------

(println "\n-- 4. Inverse I-POMDP: infer guilt from observed action --")

(let [model (investor-model iq1 beta)
      p-selfish-lo (exact/pr model :guilt 0 :given :fi 0)
      p-selfish-hi (exact/pr model :guilt 0 :given :fi 1)]
  (println (str "   Observe LOW offer  -> P(selfish) = " (.toFixed p-selfish-lo 4)))
  (println (str "   Observe HIGH offer -> P(selfish) = " (.toFixed p-selfish-hi 4)))
  (check "low offer -> P(selfish) > 0.5" (> p-selfish-lo 0.5))
  (check "high offer -> P(selfish) < 0.5" (< p-selfish-hi 0.5))
  (check "low offer -> more selfish than high offer" (> p-selfish-lo p-selfish-hi)))

;; -------------------------------------------------------------------------
;; 5. GFI compliance
;; -------------------------------------------------------------------------

(println "\n-- 5. GFI compliance --")

(let [model (dyn/auto-key (investor-model iq1 beta))
      tr (p/simulate model [])
      _ (mx/eval! (:retval tr) (:score tr))
      {:keys [weight]} (p/generate model []
                         (cm/choicemap :fi (mx/scalar 1 mx/int32)))
      _ (mx/eval! weight)]
  (check "simulate produces trace" (some? (:retval tr)))
  (check "generate weight is finite" (js/isFinite (mx/item weight))))

;; -------------------------------------------------------------------------
;; 6. Continuous guilt + sequential Bayesian updating (MCMC)
;; -------------------------------------------------------------------------

(println "\n-- 6. Continuous guilt + sequential Bayesian updating --")
(println "   Observer watches investor play 5 rounds, infers continuous guilt via MCMC.")

(let [model-5 (sequential-trust-model tq0 beta 5)
      infer (fn [obs]
              (let [cm (reduce (fn [cm [i a]]
                                 (cm/merge-cm cm (cm/choicemap (keyword (str "r" i))
                                                               (mx/scalar a mx/int32))))
                               cm/EMPTY (map-indexed vector obs))
                    {:keys [trace]} (p/generate (dyn/auto-key model-5) [] cm)
                    _ (mx/eval! (:score trace))
                    kernel (kern/random-walk {:guilt 0.1})
                    traces (kern/run-kernel {:samples 400 :burn 200} kernel trace)
                    raws (mapv #(mx/item (cm/get-choice (:choices %) [:guilt])) traces)
                    gs (mapv #(js/Math.log (+ 1 (js/Math.exp %))) raws)]
                (/ (reduce + gs) (count gs))))
      g-generous (infer [1 1 1 1 1])
      g-selfish (infer [0 0 0 0 0])
      g-2h (infer [1 1])
      g-2h3l (infer [1 1 0 0 0])]
  (println (str "   5 generous actions -> guilt = " (.toFixed g-generous 3)))
  (println (str "   5 selfish actions  -> guilt = " (.toFixed g-selfish 3)))
  (println (str "   2 generous         -> guilt = " (.toFixed g-2h 3)))
  (println (str "   2 generous + 3 selfish -> guilt = " (.toFixed g-2h3l 3)))
  (check "generous -> higher guilt than selfish" (> g-generous g-selfish))
  (check "generous -> guilt > 0.4" (> g-generous 0.4))
  (check "selfish -> guilt < 0.7" (< g-selfish 0.7))
  (check "mixed: guilt drops as selfish actions accumulate" (> g-2h g-2h3l)))

;; =========================================================================
;; Summary
;; =========================================================================

(println (str "\n== Results: " @pass-count " passed, " @fail-count " failed =="))
(when (pos? @fail-count) (js/process.exit 1))
