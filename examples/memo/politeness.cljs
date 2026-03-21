(ns politeness
  "Polite speech — a speaker balances informativeness and social courtesy.

   Inspired by: Yoon, Tessler, Goodman & Frank (2020).
   Polite speech emerges from competing social goals. Open Mind, 4, 71-87.

   States (how good the performance really was):
     0=terrible, 1=bad, 2=good, 3=amazing   (4 hearts)
   Utterances:
     0=terrible, 1=not_amazing, 2=not_terrible, 3=amazing

   The literal semantics L(w,s) are soft: each utterance has graded
   applicability to each state. The listener uses Bayesian inference
   to recover the state from the utterance. The speaker chooses an
   utterance to optimize a weighted combination of:
     U_inf = log P_L(s_true | w)   (informativeness — listener learns truth)
     U_soc = E_L[V(s)]             (social — listener feels good)
   with mixing weight phi: high phi = informative, low phi = polite."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.inference.exact :as exact])
  (:require-macros [genmlx.gen :refer [gen]]))

;; =========================================================================
;; Domain setup
;; =========================================================================

;; States: 0=terrible, 1=bad, 2=good, 3=amazing
(def n-states 4)

;; Utterances: 0=terrible, 1=not_amazing, 2=not_terrible, 3=amazing
(def n-utterances 4)
(def utterance-names ["terrible" "not_amazing" "not_terrible" "amazing"])
(def state-names ["0 hearts" "1 heart" "2 hearts" "3 hearts"])

;; Literal semantics L(w, s) — soft truth values
;; Row = utterance, col = state
;; "terrible" applies strongly to state 0, weakly to 1
;; "not amazing" applies to 0,1,2 (everything except 3)
;; "not terrible" applies to 1,2,3 (everything except 0)
;; "amazing" applies weakly to 2, strongly to 3
(def literal-semantics
  (.astype (mx/array #js [#js [1.0 0.5 0.0 0.0]    ;; terrible
                          #js [1.0 1.0 1.0 0.0]    ;; not_amazing
                          #js [0.0 1.0 1.0 1.0]    ;; not_terrible
                          #js [0.0 0.0 0.5 1.0]])  ;; amazing
           mx/float32))

;; V(s) = s — the listener's value is linear in the state
(def values (mx/array #js [0.0 1.0 2.0 3.0]))

(mx/eval! literal-semantics values)

;; =========================================================================
;; Literal listener (L0)
;; =========================================================================

(defn literal-listener-model
  "L0: listener hears utterance w, infers state s.
   P_L0(s | w) proportional to L(w,s) * P(s).
   Uniform state prior, soft literal semantics."
  [w]
  (gen []
    (let [s (trace :s (dist/weighted [1 1 1 1]))
          ;; Score by literal semantics L(w, s)
          sem (mx/idx (mx/idx literal-semantics w) s)]
      (trace :valid (dist/bernoulli sem))
      s)))

(defn literal-listener-table
  "Compute P_L0(s | w) for a given utterance w.
   Returns [4] probability tensor over states."
  [w]
  (let [joint (exact/exact-joint (literal-listener-model w) []
                (cm/choicemap :valid (mx/scalar 1)))
        m (exact/joint-marginal (:log-probs joint) (:axes joint) #{:s})
        probs (mx/exp (:log-probs m))]
    (mx/eval! probs)
    probs))

;; Precompute L0 table: [4 utterances, 4 states]
(def L0-table
  (let [rows (mapv literal-listener-table (range n-utterances))
        stacked (mx/stack (clj->js rows))]
    (mx/eval! stacked)
    stacked))

;; =========================================================================
;; Speaker (S1) — balances informativeness and social utility
;; =========================================================================

(defn compute-speaker-utility
  "For each (state, utterance) pair, compute the speaker's utility:
     U(w; s, phi) = phi * log P_L0(s | w) + (1-phi) * E_L0[V(s') | w]

   Returns [4, 4] tensor: row=state, col=utterance."
  [alpha phi]
  (let [;; U_inf(w, s) = log P_L0(s | w) — [4 utt, 4 state]
        log-L0 (mx/log (mx/maximum L0-table (mx/scalar 1e-30)))
        ;; E_L0[V | w] = sum_s' P_L0(s' | w) * V(s') — [4] per utterance
        E-value (mx/matmul L0-table (mx/reshape values #js [4 1]))  ;; [4, 1]
        E-value (mx/squeeze E-value)  ;; [4]
        ;; Build utility tensor [state, utterance]
        ;; For each state s and utterance w:
        ;;   U = phi * log_L0[w, s] + (1-phi) * E_value[w]
        phi-mx (mx/scalar phi)
        one-minus-phi (mx/scalar (- 1.0 phi))
        ;; log_L0 is [utt, state], transpose to [state, utt]
        inf-term (mx/multiply phi-mx (mx/transpose log-L0))
        ;; E-value is [utt], broadcast across states
        soc-term (mx/multiply one-minus-phi E-value)
        utility (mx/add inf-term soc-term)
        ;; Speaker chooses w ~ softmax(alpha * U)
        logits (mx/multiply (mx/scalar alpha) utility)]
    (mx/eval! logits)
    logits))

(defn speaker-model
  "S1: speaker knows the true state, chooses an utterance.
   Returns a gen function parameterized by alpha and phi."
  [alpha phi]
  (let [logits (compute-speaker-utility alpha phi)]
    (gen []
      (let [s (trace :s (dist/weighted [1 1 1 1]))
            w (trace :w (dist/categorical (mx/idx logits s)))]
        w))))

;; =========================================================================
;; Compute speaker distributions
;; =========================================================================

(defn speaker-table
  "Compute P_S1(w | s) for all states.
   Returns [4 states, 4 utterances] probability tensor."
  [alpha phi]
  (let [model (speaker-model alpha phi)
        joint (exact/exact-joint model [] nil)
        table (exact/extract-table joint :s)]
    (mx/eval! table)
    table))

;; =========================================================================
;; Display and verification
;; =========================================================================

(println "Polite Speech — exact enumeration")
(println "  Yoon, Tessler, Goodman & Frank (2020)\n")

(defn print-speaker-table [alpha phi]
  (let [table (speaker-table alpha phi)]
    (println (str "  Speaker (alpha=" alpha ", phi=" phi "):"))
    (println "  state         terrible  not_amaz  not_terr  amazing")
    (doseq [s (range n-states)]
      (let [row (mx/idx table s)
            _ (mx/eval! row)
            ps (mapv #(mx/item (mx/idx row %)) (range n-utterances))]
        (println (str "  " (nth state-names s) "      "
                      (.toFixed (nth ps 0) 4) "    "
                      (.toFixed (nth ps 1) 4) "    "
                      (.toFixed (nth ps 2) 4) "    "
                      (.toFixed (nth ps 3) 4)))))
    (println)
    table))

;; ---- Informative speaker (phi=0.9) ----

(println "-- Informative speaker (phi=0.9) --")
(def informative-table (print-speaker-table 3.0 0.9))

;; ---- Polite speaker (phi=0.1) ----

(println "-- Polite speaker (phi=0.1) --")
(def polite-table (print-speaker-table 3.0 0.1))

;; =========================================================================
;; Verification
;; =========================================================================

(println "-- Verification --")

(def pass-count (atom 0))
(def fail-count (atom 0))

(defn check [label pred]
  (if pred
    (do (swap! pass-count inc) (println (str "  PASS: " label)))
    (do (swap! fail-count inc) (println (str "  FAIL: " label)))))

(defn check-close [label expected actual tol]
  (if (<= (abs (- expected actual)) tol)
    (do (swap! pass-count inc)
        (println (str "  PASS: " label " (" (.toFixed actual 4) " ~ " expected ")")))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " label " (expected " expected " got " (.toFixed actual 4) ")")))))

(defn get-prob [table s w]
  (let [v (mx/idx (mx/idx table s) w)]
    (mx/eval! v)
    (mx/item v)))

;; 1. L0 sanity: P_L0(s=0 | "terrible") should be high
(let [p (mx/item (mx/idx (literal-listener-table 0) 0))]
  (check "L0: 'terrible' -> state 0 most likely" (> p 0.5)))

;; 2. L0: "amazing" -> state 3 most likely
(let [p (mx/item (mx/idx (literal-listener-table 3) 3))]
  (check "L0: 'amazing' -> state 3 most likely" (> p 0.5)))

;; 3. Informative speaker: state 0 -> "terrible" preferred
(let [p-terrible (get-prob informative-table 0 0)
      p-not-amaz (get-prob informative-table 0 1)]
  (check "informative: state 0 -> 'terrible' > 'not_amazing'"
         (> p-terrible p-not-amaz)))

;; 4. Informative speaker: state 3 -> "amazing" preferred
(let [p-amazing (get-prob informative-table 3 3)
      p-not-terr (get-prob informative-table 3 2)]
  (check "informative: state 3 -> 'amazing' > 'not_terrible'"
         (> p-amazing p-not-terr)))

;; 5. Polite speaker: state 0 -> shifts toward "not terrible" (euphemism)
(let [p-terrible-polite (get-prob polite-table 0 0)
      p-terrible-inform (get-prob informative-table 0 0)]
  (check "polite: state 0 -> less 'terrible' than informative"
         (< p-terrible-polite p-terrible-inform)))

;; 6. Polite speaker: state 0 -> more "not terrible" than informative
(let [p-nt-polite (get-prob polite-table 0 2)
      p-nt-inform (get-prob informative-table 0 2)]
  (check "polite: state 0 -> more 'not_terrible' (euphemism)"
         (> p-nt-polite p-nt-inform)))

;; 7. Polite speaker: state 3 -> "amazing" still dominant
;;    (being nice aligns with being truthful for good states)
(let [p-amazing (get-prob polite-table 3 3)]
  (check "polite: state 3 -> 'amazing' still strong" (> p-amazing 0.3)))

;; 8. White lie test: polite speaker with bad state (0) uses
;;    socially positive utterances more than informative speaker
(let [positive-polite (+ (get-prob polite-table 0 2) (get-prob polite-table 0 3))
      positive-inform (+ (get-prob informative-table 0 2) (get-prob informative-table 0 3))]
  (check "polite: white lies — more positive utterances for bad state"
         (> positive-polite positive-inform)))

;; 9. Rows sum to 1 (proper distributions)
(doseq [s (range n-states)]
  (let [row-sum (reduce + (map #(get-prob informative-table s %) (range n-utterances)))]
    (check-close (str "informative row " s " sums to 1") 1.0 row-sum 1e-4)))

;; 10. Phi tradeoff: as phi increases, speaker becomes more informative
;;     about state 0 (higher P("terrible" | s=0))
(let [p-terr-01 (get-prob (speaker-table 3.0 0.1) 0 0)
      p-terr-05 (get-prob (speaker-table 3.0 0.5) 0 0)
      p-terr-09 (get-prob (speaker-table 3.0 0.9) 0 0)]
  (check "phi tradeoff: P('terrible'|s=0) increases with phi"
         (and (< p-terr-01 p-terr-05) (< p-terr-05 p-terr-09))))

;; =========================================================================
;; Pragmatic listener (L1) — infers state from polite speaker
;; =========================================================================

(println "\n-- Pragmatic listener (L1) --")

(defn pragmatic-listener-model
  "L1: listener hears utterance w, infers state s.
   Uses S1 as the generative model of speech production."
  [alpha phi w]
  (let [logits (compute-speaker-utility alpha phi)]
    (gen []
      (let [s (trace :s (dist/weighted [1 1 1 1]))
            ;; Speaker would have chosen w with probability proportional to
            ;; softmax(alpha * U(w; s, phi)), but we condition on observed w
            speaker-logits (mx/idx logits s)
            _ (trace :w (dist/categorical speaker-logits))]
        s))))

(defn pragmatic-listener-probs
  "P_L1(s | w, phi) — what the pragmatic listener infers."
  [alpha phi w]
  (let [model (pragmatic-listener-model alpha phi w)
        r (exact/exact-posterior model [] (cm/choicemap :w (mx/scalar w mx/int32)))]
    (mapv #(get-in r [:marginals :s %]) (range n-states))))

(println "  Hearing 'not_terrible' (w=2):")
(let [probs-inform (pragmatic-listener-probs 3.0 0.9 2)
      probs-polite (pragmatic-listener-probs 3.0 0.1 2)]
  (println (str "    Informative speaker: " (mapv #(.toFixed % 3) probs-inform)))
  (println (str "    Polite speaker:      " (mapv #(.toFixed % 3) probs-polite)))
  ;; With a polite speaker, "not terrible" shifts toward lower states
  ;; because it might be a white lie
  (let [E-inform (reduce + (map * probs-inform [0 1 2 3]))
        E-polite (reduce + (map * probs-polite [0 1 2 3]))]
    (println (str "    E[state | informative] = " (.toFixed E-inform 3)))
    (println (str "    E[state | polite]      = " (.toFixed E-polite 3)))
    (check "L1: 'not_terrible' from polite speaker -> lower expected state"
           (< E-polite E-inform))))

(println (str "\nHearing 'amazing' (w=3):"))
(let [probs-inform (pragmatic-listener-probs 3.0 0.9 3)
      probs-polite (pragmatic-listener-probs 3.0 0.1 3)]
  (println (str "    Informative speaker: " (mapv #(.toFixed % 3) probs-inform)))
  (println (str "    Polite speaker:      " (mapv #(.toFixed % 3) probs-polite)))
  (let [E-inform (reduce + (map * probs-inform [0 1 2 3]))
        E-polite (reduce + (map * probs-polite [0 1 2 3]))]
    (println (str "    E[state | informative] = " (.toFixed E-inform 3)))
    (println (str "    E[state | polite]      = " (.toFixed E-polite 3)))
    (check "L1: 'amazing' from polite speaker -> lower expected state"
           (< E-polite E-inform))))

;; =========================================================================
;; Summary
;; =========================================================================

(println (str "\n== Results: " @pass-count " passed, " @fail-count " failed =="))
(when (pos? @fail-count) (js/process.exit 1))
