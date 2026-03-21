(ns polarization
  "Bayesian Belief Polarization — two agents see the same evidence
   and reach opposite conclusions.

   Inspired by Jern, Chang & Kemp (2009), 'Bayesian belief polarization',
   NeurIPS.

   Setup:
     Two economists (Alice and Bob) evaluate a bill that may or may not
     increase government spending. A study comes out predicting whether
     the bill increases spending. Alice leans liberal (90% prior), Bob
     leans conservative (90% prior). Despite seeing the SAME study,
     their posterior opinions of the bill DIVERGE — polarization from
     shared evidence.

   Encoding:
     policy:  0=conservative, 1=liberal
     outcome: 0=no spending,  1=spending increase
     study:   0=no spending,  1=spending increase
     opinion: 0=bad policy,   1=good policy
     agent:   0=Alice,        1=Bob"
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.inference.exact :as exact])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; World model — shared by both economists
;; ---------------------------------------------------------------------------
;; Each economist has a prior over optimal economic policy (v1), then the
;; world generates a bill outcome (v2), a study result (d), and whether
;; the bill is a good policy (h).

(defn world-model
  "The world according to economist e.
   e-val: 0=Alice (liberal-leaning), 1=Bob (conservative-leaning).

   Traces: :policy, :outcome, :study, :opinion.
   Returns: opinion index."
  [e-val]
  (gen []
    (let [;; Economic philosophy prior — Alice favors liberal, Bob favors conservative
          policy-wpp (if (zero? e-val)
                       [0.1 0.9]     ;; Alice: 10% conservative, 90% liberal
                       [0.9 0.1])    ;; Bob:   90% conservative, 10% liberal
          policy (trace :policy (dist/weighted policy-wpp))

          ;; Bill outcome — uniform (50/50 whether spending increases)
          outcome (trace :outcome (dist/weighted [1 1]))

          ;; Study result — noisy observation of outcome (90% accurate)
          ;; Stack along last axis so category dim is rightmost for categorical
          p-no  (mx/where (mx/eq? outcome 0) (mx/scalar 0.9) (mx/scalar 0.1))
          p-yes (mx/where (mx/eq? outcome 0) (mx/scalar 0.1) (mx/scalar 0.9))
          study (trace :study (dist/categorical (mx/log (mx/stack [p-no p-yes] -1))))

          ;; Opinion of bill — depends on outcome and policy
          ;; If no spending increase: 50/50
          ;; If spending increase:
          ;;   conservative -> bad (90%), good (10%)
          ;;   liberal      -> good (90%), bad (10%)
          is-no-spending (mx/eq? outcome 0)
          is-conservative (mx/eq? policy 0)
          ;; P(opinion=bad)
          p-bad (mx/where is-no-spending
                  (mx/scalar 0.5)
                  (mx/where is-conservative
                    (mx/scalar 0.9)
                    (mx/scalar 0.1)))
          ;; P(opinion=good)
          p-good (mx/subtract (mx/scalar 1.0) p-bad)
          ;; Stack along last axis so category dim is rightmost for categorical
          opinion (trace :opinion (dist/categorical (mx/log (mx/stack [p-bad p-good] -1))))]
      opinion)))

;; ---------------------------------------------------------------------------
;; Prior: P(bill is good policy) before seeing study
;; ---------------------------------------------------------------------------

(defn prior-good [e-val]
  (exact/pr (world-model e-val) :opinion 1))

;; ---------------------------------------------------------------------------
;; Posterior: P(bill is good policy | study = d)
;; ---------------------------------------------------------------------------

(defn posterior-good [e-val d-val]
  (exact/pr (world-model e-val) :opinion 1
            :given :study d-val))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(println "Bayesian Belief Polarization")
(println "============================")
(println)
(println "Alice: 90% liberal prior")
(println "Bob:   90% conservative prior")
(println)

;; Compute priors
(let [alice-prior (prior-good 0)
      bob-prior   (prior-good 1)]
  (println (str "Prior P(bill is good):"))
  (println (str "  Alice: " (.toFixed alice-prior 4)))
  (println (str "  Bob:   " (.toFixed bob-prior 4)))
  (println))

;; Scenario 1: Study predicts NO spending increase
(println "--- Study predicts NO SPENDING INCREASE ---")
(let [alice-prior (prior-good 0)
      bob-prior   (prior-good 1)
      alice-post  (posterior-good 0 0)
      bob-post    (posterior-good 1 0)]
  (println (str "  Alice: " (.toFixed alice-prior 4) " -> " (.toFixed alice-post 4)
                (if (> alice-post alice-prior) " (increases)" " (decreases)")))
  (println (str "  Bob:   " (.toFixed bob-prior 4) " -> " (.toFixed bob-post 4)
                (if (> bob-post bob-prior) " (increases)" " (decreases)")))
  (let [gap-before (abs (- alice-prior bob-prior))
        gap-after  (abs (- alice-post bob-post))]
    (println (str "  Gap: " (.toFixed gap-before 4) " -> " (.toFixed gap-after 4)
                  (if (> gap-after gap-before) " POLARIZED" " converged"))))
  (println))

;; Scenario 2: Study predicts SPENDING INCREASE
(println "--- Study predicts SPENDING INCREASE ---")
(let [alice-prior (prior-good 0)
      bob-prior   (prior-good 1)
      alice-post  (posterior-good 0 1)
      bob-post    (posterior-good 1 1)]
  (println (str "  Alice: " (.toFixed alice-prior 4) " -> " (.toFixed alice-post 4)
                (if (> alice-post alice-prior) " (increases)" " (decreases)")))
  (println (str "  Bob:   " (.toFixed bob-prior 4) " -> " (.toFixed bob-post 4)
                (if (> bob-post bob-prior) " (increases)" " (decreases)")))
  (let [gap-before (abs (- alice-prior bob-prior))
        gap-after  (abs (- alice-post bob-post))]
    (println (str "  Gap: " (.toFixed gap-before 4) " -> " (.toFixed gap-after 4)
                  (if (> gap-after gap-before) " POLARIZED" " converged"))))
  (println))

;; ---------------------------------------------------------------------------
;; Verification
;; ---------------------------------------------------------------------------
;; The model is symmetric: Alice's P(good) = 1 - Bob's P(good).
;; Polarization occurs when spending-increase evidence reinforces each
;; agent's prior-driven interpretation (gap widens).
;; No-spending evidence is less divisive and causes convergence (gap shrinks).

(println "Verification:")

(let [pass? (atom true)
      check (fn [name pred]
              (when-not pred (reset! pass? false))
              (println (str "  " (if pred "PASS" "FAIL") ": " name)))
      check-close (fn [name expected actual tol]
                    (let [ok (<= (abs (- expected actual)) tol)]
                      (when-not ok (reset! pass? false))
                      (println (str "  " (if ok "PASS" "FAIL") ": " name
                                    " (expected " (.toFixed expected 4)
                                    ", got " (.toFixed actual 4) ")"))))
      approx= (fn [a b tol] (<= (abs (- a b)) tol))

      alice-prior (prior-good 0)
      bob-prior   (prior-good 1)

      ;; Symmetry: Alice's P(good) + Bob's P(good) = 1
      _ (check "symmetry: alice-prior + bob-prior = 1"
               (approx= (+ alice-prior bob-prior) 1.0 1e-4))

      ;; Alice leans good, Bob leans bad
      _ (check-close "alice prior" 0.66 alice-prior 1e-2)
      _ (check-close "bob prior"   0.34 bob-prior   1e-2)

      ;; Study predicts no spending -> convergence (both move toward 0.5)
      alice-no (posterior-good 0 0)
      bob-no   (posterior-good 1 0)
      gap-before (abs (- alice-prior bob-prior))
      gap-no     (abs (- alice-no bob-no))
      _ (check-close "alice|no-spending"  0.532 alice-no 1e-2)
      _ (check-close "bob|no-spending"    0.468 bob-no   1e-2)
      _ (check "no-spending: convergence (gap shrinks)"
               (< gap-no gap-before))

      ;; Study predicts spending increase -> POLARIZATION (gap widens)
      alice-yes (posterior-good 0 1)
      bob-yes   (posterior-good 1 1)
      gap-yes   (abs (- alice-yes bob-yes))
      _ (check-close "alice|spending"  0.788 alice-yes 1e-2)
      _ (check-close "bob|spending"    0.212 bob-yes   1e-2)
      _ (check "spending: POLARIZATION (gap widens)"
               (> gap-yes gap-before))

      ;; Posterior symmetry holds
      _ (check "symmetry: alice-post(d=0) + bob-post(d=0) = 1"
               (approx= (+ alice-no bob-no) 1.0 1e-4))
      _ (check "symmetry: alice-post(d=1) + bob-post(d=1) = 1"
               (approx= (+ alice-yes bob-yes) 1.0 1e-4))]

  (println)
  (if @pass?
    (println "All checks passed.")
    (do (println "Some checks FAILED.")
        (js/process.exit 1))))
