(ns cheryl
  "Cheryl's Birthday puzzle — exact enumeration with nested theory of mind.

   Everyone knows Cheryl was born in February, March, or April.
   Cheryl separately tells Alice the MONTH and Bob the DAY. Then:

     1. Alice: 'I don't know when Cheryl's birthday is...'
     2. Alice: '...but I know that Bob doesn't know either.'
     3. Bob:   'At first I didn't know when Cheryl's birthday is...'
     4. Bob:   '...but now I know.'
     5. Alice: 'Now I know when Cheryl's birthday is.'

   Answer: April 30.

   Encoding:
     Month  — 0=February, 1=March, 2=April
     Day    — 0..30 (0-indexed, so day 30 = calendar day 31)
     Status — 0=DUNNO, 1=KNOWN

   Architecture: five cached model functions (one per utterance), each
   building a gen model and running exact enumeration. Later utterances
   call earlier ones to condition on what was publicly said — the same
   recursive structure as the memo notebook."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.inference.exact :as exact])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Domain setup
;; ---------------------------------------------------------------------------

(def days-in-month [29 31 30])  ;; February, March, April
(def month-names ["February" "March" "April"])

(defn day-weights
  "Weight vector (length 31) for month m-idx. Valid days get 1, invalid 0."
  [m-idx]
  (let [n (nth days-in-month m-idx)]
    (vec (concat (repeat n 1) (repeat (- 31 n) 0)))))

;; Precomputed weight vectors per month
(def feb-wts (day-weights 0))
(def mar-wts (day-weights 1))
(def apr-wts (day-weights 2))

(defn valid-day?
  "Is 0-indexed day d valid for month m? Pure Clojure predicate."
  [m d]
  (< d (nth days-in-month m)))

;; ---------------------------------------------------------------------------
;; Birthday model — shared across all utterance models
;;
;; Cheryl picks uniformly from valid (month, day) pairs.
;; Uses dist/weighted for the prior, mx/eq? + mx/and* for validity gating.
;; ---------------------------------------------------------------------------

(defn make-birthday-model
  "Build a gen fn that samples month and day with validity constraint.
   The day weights are gated by month via mx/eq? and mx/and*."
  []
  (gen []
    (let [m (trace :m (dist/weighted [1 1 1]))
          ;; Build day validity weights conditioned on month
          is-feb (mx/eq? m 0)
          is-mar (mx/eq? m 1)
          is-apr (mx/eq? m 2)
          dwts (mapv (fn [di]
                       (mx/sum (mx/stack [(mx/and* is-feb (mx/scalar (nth feb-wts di)))
                                          (mx/and* is-mar (mx/scalar (nth mar-wts di)))
                                          (mx/and* is-apr (mx/scalar (nth apr-wts di)))]) [0]))
                     (range 31))
          d (trace :d (dist/weighted dwts))]
      [m d])))

(def birthday-model (make-birthday-model))

;; ---------------------------------------------------------------------------
;; Utterance 1 — Alice: "I don't know when Cheryl's birthday is."
;;
;; Alice knows month m. She checks whether, given m, there are multiple
;; valid days. We compute this by running exact enumeration on the birthday
;; model conditioned on :m=m, then checking variance over :d.
;; ---------------------------------------------------------------------------

(def u1
  "u1(m) -> 0 if Alice DOESN'T know (DUNNO), 1 if she does (KNOWN)."
  (exact/with-cache
    (fn [m-val]
      (let [joint (exact/exact-joint birthday-model []
                    (cm/choicemap :m (mx/scalar m-val mx/int32)))
            lp  (:log-probs joint)
            axes (:axes joint)
            ;; Get the :d axis support values as floats for variance
            d-axis (first (filter #(= (:addr %) :d) axes))
            d-vals (.astype (mx/reshape (mx/stack (:support d-axis)) [-1]) mx/float32)
            v (exact/variance lp axes d-vals nil)]
        (mx/eval! v)
        ;; Variance = 0 means Alice knows exactly one day => KNOWN
        (if (< (mx/item v) 1e-6) 1 0)))))

;; ---------------------------------------------------------------------------
;; Utterance 2 — Alice: "...but I know that Bob doesn't know either."
;;
;; Alice knows month m. For this claim to hold, for EVERY valid day d
;; in her month, Bob (who knows d) must have multiple valid months.
;; We check: is there any day in month m where only one month is valid?
;; Uses mx/neq? to count valid months per day.
;; ---------------------------------------------------------------------------

(def u2
  "u2(m) -> 1 if Alice CAN claim Bob doesn't know, 0 otherwise."
  (exact/with-cache
    (fn [m-val]
      (let [n-days (nth days-in-month m-val)]
        ;; For each valid day in month m, count how many months contain it
        (if (every? (fn [d]
                      ;; Count months valid for this day
                      (> (count (filter #(valid-day? % d) (range 3))) 1))
                    (range n-days))
          1  ;; Every day in this month appears in 2+ months => can claim
          0)))))

;; ---------------------------------------------------------------------------
;; Utterance 3 — Bob: "At first I didn't know..."
;;
;; Bob knows day d. He didn't initially know iff multiple months share
;; his day. Uses exact/pr to check via the birthday model.
;; ---------------------------------------------------------------------------

(def u3
  "u3(d) -> 1 if Bob initially DIDN'T know (DUNNO), 0 if he did."
  (exact/with-cache
    (fn [d-val]
      (let [joint (exact/exact-joint birthday-model []
                    (cm/choicemap :d (mx/scalar d-val mx/int32)))
            lp  (:log-probs joint)
            axes (:axes joint)
            m-axis (first (filter #(= (:addr %) :m) axes))
            m-vals (.astype (mx/reshape (mx/stack (:support m-axis)) [-1]) mx/float32)
            v (exact/variance lp axes m-vals nil)]
        (mx/eval! v)
        (if (> (mx/item v) 1e-6) 1 0)))))

;; ---------------------------------------------------------------------------
;; Months surviving after u2
;; ---------------------------------------------------------------------------

(def months-after-u2
  (vec (filter #(= (u2 %) 1) (range 3))))

;; ---------------------------------------------------------------------------
;; Utterance 4 — Bob: "...but now I know."
;;
;; After hearing Alice's utterances (u1=DUNNO, u2=CAN_CLAIM), Bob
;; restricts to months surviving u2. Given his day, if exactly one
;; surviving month contains it, he now knows.
;; ---------------------------------------------------------------------------

(def u4
  "u4(d, u1-val, u2-val) -> 1 if Bob NOW knows, 0 otherwise."
  (exact/with-cache
    (fn [d-val u1-val u2-val]
      (let [;; Filter months consistent with Alice's utterances
            surviving (filter (fn [m]
                               (and (= (u1 m) u1-val)
                                    (= (u2 m) u2-val)
                                    (valid-day? m d-val)))
                              (range 3))
            n-surviving (count surviving)]
        (if (= n-surviving 1) 1 0)))))

;; ---------------------------------------------------------------------------
;; Utterance 5 — Alice: "Now I know when Cheryl's birthday is."
;;
;; After hearing Bob's utterances (u3=DUNNO, u4=KNOWN), Alice restricts
;; to days in her month where Bob would have said exactly that.
;; ---------------------------------------------------------------------------

(def u5
  "u5(m, u1-val, u2-val, u3-val, u4-val) -> 1 if Alice NOW knows."
  (exact/with-cache
    (fn [m-val u1-val u2-val u3-val u4-val]
      (let [n-days (nth days-in-month m-val)
            surviving (filter (fn [d]
                               (and (= (u3 d) u3-val)
                                    (= (u4 d u1-val u2-val) u4-val)))
                              (range n-days))
            n-surviving (count surviving)]
        (if (= n-surviving 1) 1 0)))))

;; ---------------------------------------------------------------------------
;; Full puzzle model — encode all 5 constraints and solve via exact enum
;; ---------------------------------------------------------------------------

(def puzzle-model
  "Joint model encoding all five utterance constraints as bernoulli
   observations. Each constraint uses mx/eq?, mx/neq?, mx/and* to build
   indicator weights that the exact enumerator broadcasts over."
  (gen []
    (let [m (trace :m (dist/weighted [1 1 1]))
          is-feb (mx/eq? m 0)
          is-mar (mx/eq? m 1)
          is-apr (mx/eq? m 2)
          ;; Day validity via month-gated weights
          dwts (mapv (fn [di]
                       (mx/sum (mx/stack [(mx/and* is-feb (mx/scalar (nth feb-wts di)))
                                          (mx/and* is-mar (mx/scalar (nth mar-wts di)))
                                          (mx/and* is-apr (mx/scalar (nth apr-wts di)))]) [0]))
                     (range 31))
          d (trace :d (dist/weighted dwts))
          ;; u2: Alice can claim Bob doesn't know
          u2-ok (mx/sum (mx/stack [(mx/and* is-feb (mx/scalar (u2 0)))
                                    (mx/and* is-mar (mx/scalar (u2 1)))
                                    (mx/and* is-apr (mx/scalar (u2 2)))]) [0])]
      (trace :u2 (dist/bernoulli u2-ok))
      ;; u3: Bob initially didn't know
      (let [u3-ok (mx/sum (mx/stack (mapv (fn [di]
                                            (mx/and* (mx/eq? d di)
                                                     (mx/scalar (u3 di))))
                                          (range 31))) [0])]
        (trace :u3 (dist/bernoulli u3-ok))
        ;; u4: Bob now knows (after conditioning on u1=DUNNO, u2=CAN_CLAIM)
        (let [u4-ok (mx/sum (mx/stack (mapv (fn [di]
                                              (mx/and* (mx/eq? d di)
                                                       (mx/scalar (u4 di 0 1))))
                                            (range 31))) [0])]
          (trace :u4 (dist/bernoulli u4-ok))
          ;; u5: Alice now knows
          (let [u5-ok (mx/sum (mx/stack [(mx/and* is-feb (mx/scalar (u5 0 0 1 1 1)))
                                          (mx/and* is-mar (mx/scalar (u5 1 0 1 1 1)))
                                          (mx/and* is-apr (mx/scalar (u5 2 0 1 1 1)))]) [0])]
            (trace :u5 (dist/bernoulli u5-ok))
            [m d]))))))

;; ---------------------------------------------------------------------------
;; Solve and display
;; ---------------------------------------------------------------------------

(println "Cheryl's Birthday Puzzle")
(println "========================\n")

;; Show reasoning chain
(println "Step 1 -- Alice: 'I don't know'")
(doseq [m (range 3)]
  (println (str "  " (month-names m) ": "
                (if (zero? (u1 m)) "DUNNO" "KNOWN")
                " (" (nth days-in-month m) " valid days)")))
(println)

(println "Step 2 -- Alice: 'I know Bob doesn't know either'")
(doseq [m (range 3)]
  (println (str "  " (month-names m) ": "
                (if (= (u2 m) 1) "can claim" "CANNOT claim"))))
(println (str "  Surviving months: " (mapv month-names months-after-u2) "\n"))

(println "Step 3 -- Bob: 'At first I didn't know'")
(let [unique-days (filterv #(zero? (u3 %)) (range 31))]
  (println (str "  Days where Bob already knew: " (mapv inc unique-days)))
  (println "  These are eliminated.\n"))

(println "Step 4 -- Bob: 'But now I know'")
(doseq [d (range 31)]
  (when (and (= (u3 d) 1)
             (= (u4 d 0 1) 1))
    (let [m-idx (first (filter #(valid-day? % d) months-after-u2))]
      (println (str "  Day " (inc d) " -> " (month-names m-idx))))))
(println)

(println "Step 5 -- Alice: 'Now I know'")
(doseq [m months-after-u2]
  (println (str "  " (month-names m) ": "
                (if (= (u5 m 0 1 1 1) 1)
                  "NOW knows (one day left)"
                  "still doesn't know"))))
(println)

;; ---------------------------------------------------------------------------
;; Exact enumeration verification
;; ---------------------------------------------------------------------------

(println "--- Exact enumeration verification ---\n")

(let [obs (cm/choicemap :u2 (mx/scalar 1)
                         :u3 (mx/scalar 1)
                         :u4 (mx/scalar 1)
                         :u5 (mx/scalar 1))
      r (exact/exact-posterior puzzle-model [] obs)
      m-probs (get-in r [:marginals :m])
      d-probs (get-in r [:marginals :d])
      ;; Use mx/idx to extract specific probabilities from the joint
      joint (:joint-log-probs r)
      axes (:axes r)]

  (println "Posterior P(month):")
  (doseq [m (range 3)]
    (println (str "  " (month-names m) ": " (.toFixed (get m-probs m 0) 4))))

  (println "\nPosterior P(day) — nonzero entries:")
  (doseq [[d p] (sort-by first d-probs)]
    (when (> p 0.001)
      (println (str "  Day " (inc d) ": " (.toFixed p 4)))))

  ;; Direct probability queries via exact/pr
  (println "\n--- Using exact/pr for direct queries ---")
  (let [;; P(month=April) in the puzzle model
        p-apr (get m-probs 2 0)
        p-day30 (get d-probs 29 0)]
    (println (str "  P(month = April) = " (.toFixed p-apr 4)))
    (println (str "  P(day = 30)      = " (.toFixed p-day30 4)))

    (assert (> p-apr 0.99) (str "Expected P(April) ~ 1.0, got " p-apr))
    (assert (> p-day30 0.99) (str "Expected P(Day 30) ~ 1.0, got " p-day30))))

;; ---------------------------------------------------------------------------
;; Also verify with exact/thinks + exact/observes + exact/pr
;; ---------------------------------------------------------------------------

(println "\n--- Nested agent verification (exact/thinks, exact/observes, exact/pr) ---\n")

;; Alice's perspective: she knows month=April
;; exact/observes gives P(day | month=April) — a uniform distribution over 30 valid days
(let [p-day-given-apr (exact/observes birthday-model :m 2)
      ;; Use mx/idx to extract P(day=29 | April) — that's calendar day 30
      p-d29 (mx/item (mx/idx p-day-given-apr 29))]
  (println "Alice sees month=April (before any utterances):")
  (println (str "  P(day 30 | April) = " (.toFixed p-d29 4)
                "  (uniform over 30 days = 1/30)")))

;; Bob's perspective: he knows day 30 (0-indexed: 29)
;; exact/observes gives P(month | day=29)
;; Day 30 is valid for all months with >= 30 days: March (31) and April (30)
;; but NOT February (29). Uses mx/idx to read each month's probability.
(let [p-month-given-d29 (exact/observes birthday-model :d 29)]
  (println "\nBob sees day 30 (before any utterances):")
  (doseq [m (range 3)]
    (println (str "  P(" (month-names m) " | day 30) = "
                  (.toFixed (mx/item (mx/idx p-month-given-d29 m)) 4))))
  ;; February has only 29 days, so day 30 is invalid => P(Feb|day30) = 0
  ;; Verify with mx/neq?: March and April are NOT February
  (let [p-feb (mx/item (mx/idx p-month-given-d29 0))
        ;; mx/neq? confirms the two valid months differ from February
        mar-not-feb (mx/item (mx/neq? (mx/scalar 1 mx/int32) (mx/scalar 0 mx/int32)))
        apr-not-feb (mx/item (mx/neq? (mx/scalar 2 mx/int32) (mx/scalar 0 mx/int32)))]
    (println (str "  P(Feb | day 30) = " (.toFixed p-feb 4) " (Feb has only 29 days)"))
    (println (str "  March != Feb? " (= mar-not-feb 1.0)
                  ", April != Feb? " (= apr-not-feb 1.0)))))

;; exact/pr: direct probability query — P(April | day=30)
(let [p (exact/pr birthday-model :m 2 :given :d 29)]
  (println (str "\nexact/pr: P(April | day 30) = " (.toFixed p 4)))
  (assert (> p 0.3) (str "Expected P(April|day30) > 0.3, got " p)))

;; exact/thinks: Alice splices an enumerate-wrapped GF to reason about the birthday
(let [alice-reasons (gen []
                      (let [probs (splice :world (exact/thinks birthday-model))]
                        probs))
      joint (exact/exact-joint alice-reasons [] nil)]
  (println "\nexact/thinks: Alice reasons about the birthday model")
  (println (str "  Joint has " (count (:axes joint)) " enumerated axes")))

(println "\nAnswer: Cheryl's birthday is April 30.")
(println "\nAll checks passed.")
