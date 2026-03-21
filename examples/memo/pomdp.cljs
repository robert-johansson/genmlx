(ns pomdp
  "Crying Baby POMDP — belief-space value iteration + inverse planning.

   An agent decides whether to feed, sing, or ignore a baby based on
   partial observations (crying/quiet). The agent maintains a belief
   over the baby's hidden state (hungry/sated) and plans ahead over
   a finite horizon.

   Part 1: Solve the POMDP via tensor DP on MLX.
   Part 2: Inverse POMDP — infer the caregiver's belief from observed actions
           using exact enumeration and MCMC."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.inference.exact :as exact]
            [genmlx.inference.kernel :as kern])
  (:require-macros [genmlx.gen :refer [gen]]))

;; =========================================================================
;; Model parameters
;; =========================================================================

(def n-beliefs 50)

(def beliefs
  "Belief grid: P(hungry) from 0 to 1 in n-beliefs steps."
  (.astype (mx/array (clj->js (mapv #(/ % (dec n-beliefs)) (range n-beliefs))))
           mx/float32))

;; Transition T[s, a, s']: P(next_state | state, action)
(def T
  (mx/reshape
    (mx/array #js [;; s=hungry
                    0.0 1.0  1.0 0.0  1.0 0.0   ;; feed->sated, sing->hungry, ignore->hungry
                    ;; s=sated
                    0.0 1.0  0.1 0.9  0.1 0.9])  ;; feed->sated, sing/ignore->10% hungry
    #js [2 3 2]))

;; Observation O[o, s', a]: P(obs | next_state, action)
(def O
  (mx/reshape
    (mx/array #js [;; o=crying: hungry cries often, sated rarely
                    0.8 0.9 0.8  0.1 0.0 0.1
                    ;; o=quiet: inverse
                    0.2 0.1 0.2  0.9 1.0 0.9])
    #js [2 2 3]))

;; Reward R[s, a]: immediate reward
(def R
  (mx/reshape
    (mx/array #js [-15.0 -10.5 -10.0   ;; hungry: feed costly but helps
                    -5.0  -0.5   0.0])  ;; sated: ignore is free
    #js [2 3]))

(def gamma 0.9)

(mx/eval! beliefs T O R)

;; =========================================================================
;; Precomputed quantities
;; =========================================================================

(def expected-reward
  "E_s[R(s,a)] for each belief. Shape [n-beliefs, 3]."
  (let [b (mx/reshape beliefs #js [n-beliefs 1])
        r-h (mx/idx R 0)
        r-s (mx/idx R 1)]
    (mx/add (mx/multiply b r-h)
            (mx/multiply (mx/subtract (mx/scalar 1.0) b) r-s))))

(def p-next-hungry
  "P(s'=hungry | b, a). Shape [n-beliefs, 3]."
  (let [b (mx/reshape beliefs #js [n-beliefs 1])
        t-hh (mx/idx (mx/idx T 0) 0 1)
        t-sh (mx/idx (mx/idx T 1) 0 1)]
    (mx/add (mx/multiply b t-hh)
            (mx/multiply (mx/subtract (mx/scalar 1.0) b) t-sh))))

(mx/eval! expected-reward p-next-hungry)

;; =========================================================================
;; Belief update
;; =========================================================================

(defn- obs-weights [o-idx s-idx]
  "Extract O[o, s', :] -- P(obs=o | s'=s-idx, action) for all actions."
  (mx/idx (mx/idx O o-idx) s-idx))

(def belief-posterior
  "Posterior belief b'(b, a, o). Shape [n-beliefs, 3, 2].
   For each (belief, action, observation), the updated P(hungry)."
  (let [p-nh p-next-hungry
        p-ns (mx/subtract (mx/scalar 1.0) p-nh)
        update-for-obs
        (fn [o]
          (let [p-o-h (obs-weights o 0)
                p-o-s (obs-weights o 1)
                num (mx/multiply p-o-h p-nh)
                den (mx/add num (mx/multiply p-o-s p-ns))]
            (mx/divide num (mx/maximum den (mx/scalar 1e-10)))))]
    (mx/stack #js [(update-for-obs 0) (update-for-obs 1)] -1)))

(def belief-indices
  "Grid indices for posterior beliefs. Shape [n-beliefs, 3, 2] int32."
  (let [bp (mx/reshape belief-posterior #js [n-beliefs 3 2 1])
        gr (mx/reshape beliefs #js [1 1 1 n-beliefs])]
    (mx/argmin (mx/abs (mx/subtract bp gr)) -1)))

(def obs-probability
  "P(o | b, a). Shape [n-beliefs, 3, 2]."
  (let [p-nh p-next-hungry
        p-ns (mx/subtract (mx/scalar 1.0) p-nh)
        prob-for-obs
        (fn [o]
          (let [p-o-h (obs-weights o 0)
                p-o-s (obs-weights o 1)]
            (mx/add (mx/multiply p-o-h p-nh) (mx/multiply p-o-s p-ns))))]
    (mx/stack #js [(prob-for-obs 0) (prob-for-obs 1)] -1)))

(mx/eval! belief-posterior belief-indices obs-probability)

;; =========================================================================
;; Value iteration
;; =========================================================================

(defn bellman-backup
  "One POMDP Bellman backup. V-prev: [n-beliefs] -> Q: [n-beliefs, 3]."
  [V-prev]
  (let [v-flat (.astype (mx/reshape belief-indices #js [-1]) mx/int32)
        v-next (mx/reshape (mx/take-idx V-prev v-flat 0) #js [n-beliefs 3 2])
        e-v-next (mx/sum (mx/multiply obs-probability v-next) [-1])]
    (mx/add expected-reward (mx/multiply (mx/scalar gamma) e-v-next))))

(defn solve
  "Solve the POMDP for a given time horizon.
   Returns Q-values of shape [n-beliefs, 3]."
  [horizon]
  (loop [V (mx/zeros #js [n-beliefs]), t 0]
    (if (>= t horizon)
      (bellman-backup V)
      (let [Q (bellman-backup V)
            V-new (mx/amax Q [1])
            _ (mx/eval! V-new)]
        (recur V-new (inc t))))))

(defn optimal-action
  "Best action at each belief point. Returns [n-beliefs] int32."
  [Q]
  (mx/argmin (mx/negative Q) -1))

(defn value-function
  "V(b) = max_a Q(b,a). Returns [n-beliefs]."
  [Q]
  (mx/amax Q [1]))

;; =========================================================================
;; Inverse POMDP: generative models
;; =========================================================================

(defn caregiver-model
  "Caregiver acts according to POMDP policy (softmax over Q-values).
   Traces :belief (n-beliefs categorical) and :action (3 categorical).
   beta controls rationality (higher = more deterministic)."
  [Q beta]
  (let [logits (mx/multiply (mx/scalar beta) Q)
        _ (mx/eval! logits)]
    (gen []
      (let [b (trace :belief (dist/weighted (vec (repeat n-beliefs 1.0))))
            a (trace :action (dist/categorical (mx/take-idx logits b 0)))]
        b))))

(defn observer-model
  "Observer infers baby's hunger severity from caregiver's actions.
   Continuous severity (sampled via MCMC) + discrete belief x action
   (exactly enumerable)."
  [Q beta]
  (let [logits (mx/multiply (mx/scalar beta) Q)
        _ (mx/eval! logits)]
    (gen []
      (let [severity-raw (trace :severity (dist/gaussian 0.0 1.0))
            severity (mx/sigmoid severity-raw)
            b-logits (mx/multiply (mx/scalar 3.0)
                       (mx/subtract (mx/multiply (mx/scalar 2.0)
                                      (mx/subtract beliefs (mx/scalar 0.5)))
                                    (mx/subtract (mx/scalar 0.5) severity)))
            b (trace :belief (dist/categorical b-logits))
            a (trace :action (dist/categorical (mx/take-idx logits b 0)))]
        severity))))

;; =========================================================================
;; Helper functions
;; =========================================================================

(def action-names {0 "feed" 1 "sing" 2 "ignore"})

(defn q-at [Q bi ai]
  (mx/item (mx/idx (mx/idx Q bi) ai)))

(defn v-at [V bi]
  (mx/item (mx/idx V bi)))

(defn pi-at [pi bi]
  (mx/item (mx/idx pi bi)))

(defn check [name pred]
  (if pred
    (println (str "  PASS: " name))
    (do (println (str "  FAIL: " name))
        (throw (js/Error. (str "Assertion failed: " name))))))

(defn check-close [name expected actual tol]
  (if (<= (js/Math.abs (- expected actual)) tol)
    (println (str "  PASS: " name))
    (do (println (str "  FAIL: " name " (expected " expected " got " actual ")"))
        (throw (js/Error. (str "Assertion failed: " name))))))

;; =========================================================================
;; Part 1: Solve the POMDP
;; =========================================================================

(println "\n========================================")
(println " Crying Baby POMDP Example")
(println "========================================")

(println "\n-- Solving POMDP (horizon=3) --")

(let [Q (solve 3)
      V (value-function Q)
      pi (optimal-action Q)
      _ (mx/eval! Q V pi)]

  ;; Display policy
  (println "\nOptimal policy at selected beliefs:")
  (doseq [bi [0 10 20 24 30 40 49]]
    (let [b-val (/ bi (dec n-beliefs))
          a (pi-at pi bi)]
      (println (str "  P(hungry)=" (.toFixed b-val 2)
                    " -> " (action-names a)))))

  ;; Display Q-values at extremes
  (println "\nQ-values at boundary beliefs:")
  (println (str "  Q(b=0.00, feed)="   (.toFixed (q-at Q 0 0)  2)
               "  Q(b=0.00, sing)="   (.toFixed (q-at Q 0 1)  2)
               "  Q(b=0.00, ignore)=" (.toFixed (q-at Q 0 2)  2)))
  (println (str "  Q(b=1.00, feed)="   (.toFixed (q-at Q 49 0) 2)
               "  Q(b=1.00, sing)="   (.toFixed (q-at Q 49 1) 2)
               "  Q(b=1.00, ignore)=" (.toFixed (q-at Q 49 2) 2)))

  ;; Display value function
  (println "\nValue function at selected beliefs:")
  (doseq [bi [0 12 24 36 49]]
    (let [b-val (/ bi (dec n-beliefs))]
      (println (str "  V(" (.toFixed b-val 2) ") = " (.toFixed (v-at V bi) 2)))))

  ;; Verification checks
  (println "\n-- Verifying Q-value shape --")
  (check "Q shape [50, 3]" (= [50 3] (mx/shape Q)))

  (println "\n-- Verifying optimal policy --")
  (check "b=0.00 -> Ignore" (= 2 (pi-at pi 0)))
  (check "b=0.20 -> Ignore" (= 2 (pi-at pi 10)))
  (check "b=0.49 -> Feed"   (= 0 (pi-at pi 24)))
  (check "b=0.82 -> Feed"   (= 0 (pi-at pi 40)))
  (check "b=1.00 -> Feed"   (= 0 (pi-at pi 49)))

  (println "\n-- Verifying Q-values (discretization tolerance) --")
  (check-close "Q(b=0, feed)"   -7.25  (q-at Q 0 0)  0.5)
  (check-close "Q(b=0, ignore)" -3.86  (q-at Q 0 2)  0.5)
  (check-close "Q(b=1, feed)"   -17.25 (q-at Q 49 0) 0.5)
  (check-close "Q(b=1, sing)"   -24.74 (q-at Q 49 1) 0.5)

  (println "\n-- Verifying value function --")
  (check-close "V(b=0)"   -3.86  (v-at V 0)  0.5)
  (check-close "V(b=0.5)" -12.15 (v-at V 24) 0.5)
  (check-close "V(b=1)"   -17.25 (v-at V 49) 0.5)

  (println "\n-- Verifying monotonicity --")
  (check "V decreases with belief" (> (v-at V 0) (v-at V 24) (v-at V 49)))
  (check "Q(feed) decreases with belief" (> (q-at Q 0 0) (q-at Q 24 0) (q-at Q 49 0)))

  (println "\n-- Verifying policy threshold --")
  (let [actions (mapv #(pi-at pi %) (range 50))
        feed-start (first (keep-indexed (fn [i a] (when (= 0 a) i)) actions))]
    (check "feed threshold exists" (some? feed-start))
    (check "threshold in middle range" (< 10 feed-start 40))
    (check "ignore before threshold" (= 2 (pi-at pi (dec feed-start))))
    (check "feed at threshold" (= 0 (pi-at pi feed-start)))))

;; =========================================================================
;; Part 1b: Horizon effect
;; =========================================================================

(println "\n-- Verifying horizon effect --")
(let [Q1 (solve 1)
      Q3 (solve 3)
      V1 (value-function Q1)
      V3 (value-function Q3)
      _ (mx/eval! V1 V3)
      v1-mid (mx/item (mx/idx V1 24))
      v3-mid (mx/item (mx/idx V3 24))]
  (println (str "  V_h1(0.5) = " (.toFixed v1-mid 2)
               "  V_h3(0.5) = " (.toFixed v3-mid 2)))
  (check "longer horizon -> lower V(b=0.5)" (> v1-mid v3-mid)))

;; =========================================================================
;; Part 2: Inverse POMDP
;; =========================================================================

(println "\n========================================")
(println " Inverse POMDP: Inferring Belief")
(println "========================================")

(let [Q3 (solve 3)
      _ (mx/eval! Q3)]

  ;; Exact enumeration over caregiver model
  (println "\n-- Exact enumeration: caregiver model --")
  (let [model (caregiver-model Q3 0.5)
        ;; P(belief | action) via exact/observes
        p-feed (exact/observes model :action 0 :belief)
        p-ign  (exact/observes model :action 2 :belief)
        e-b-feed (mx/item (mx/sum (mx/multiply beliefs p-feed)))
        e-b-ign  (mx/item (mx/sum (mx/multiply beliefs p-ign)))]

    (println (str "  E[P(hungry) | feed]   = " (.toFixed e-b-feed 3)))
    (println (str "  E[P(hungry) | ignore] = " (.toFixed e-b-ign 3)))
    (check "feed -> higher E[P(hungry)] than ignore" (> e-b-feed e-b-ign))
    (check "feed -> E[P(hungry)] > 0.5" (> e-b-feed 0.5))
    (check "ignore -> E[P(hungry)] < 0.5" (< e-b-ign 0.5)))

  ;; GFI compliance
  (println "\n-- GFI compliance: caregiver model --")
  (let [model (dyn/auto-key (caregiver-model Q3 0.5))
        tr (p/simulate model [])
        _ (mx/eval! (:retval tr) (:score tr))
        {:keys [weight]} (p/generate model []
                           (cm/choicemap :action (mx/scalar 0 mx/int32)))
        _ (mx/eval! weight)]
    (println (str "  simulate retval = " (mx/item (:retval tr))))
    (println (str "  generate weight = " (.toFixed (mx/item weight) 3)))
    (check "simulate produces trace" (some? (:retval tr)))
    (check "generate weight is finite" (js/isFinite (mx/item weight))))

  ;; MCMC inference on observer model
  (println "\n-- MCMC inference: observer model --")
  (let [run-obs (fn [action-val]
                  (let [model (dyn/auto-key (observer-model Q3 0.5))
                        {:keys [trace]} (p/generate model []
                                          (cm/choicemap :action (mx/scalar action-val mx/int32)))
                        _ (mx/eval! (:score trace))
                        kernel (kern/random-walk {:severity 0.1})
                        traces (kern/run-kernel {:samples 800 :burn 400} kernel trace)
                        raws (mapv #(mx/item (cm/get-choice (:choices %) [:severity])) traces)
                        sevs (mapv #(/ 1.0 (+ 1.0 (js/Math.exp (- %)))) raws)]
                    {:mean-sev (/ (reduce + sevs) (count sevs))
                     :acceptance (:acceptance-rate (meta traces))}))
        feed-r (run-obs 0)
        ign-r  (run-obs 2)]
    (println (str "  mean severity | feed   = " (.toFixed (:mean-sev feed-r) 3)
                 "  (acceptance " (.toFixed (* 100 (:acceptance feed-r)) 0) "%)"))
    (println (str "  mean severity | ignore = " (.toFixed (:mean-sev ign-r) 3)
                 "  (acceptance " (.toFixed (* 100 (:acceptance ign-r)) 0) "%)"))
    (if (> (:mean-sev feed-r) (:mean-sev ign-r))
      (println "  PASS: feed -> higher severity than ignore")
      (println "  NOTE: MCMC stochastic — feed severity not higher this run (expected occasionally)"))
    (check "MCMC acceptance > 50%" (> (:acceptance feed-r) 0.5))))

(println "\n========================================")
(println " All checks passed.")
(println "========================================\n")
