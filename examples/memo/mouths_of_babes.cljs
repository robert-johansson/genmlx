(ns mouths-of-babes
  "From the Mouths of Babes — toddler word production as pragmatic reasoning.

   Inspired by Washington et al. (2024), 'From the mouths of babes:
   Toddlers' early word production favors information in common ground.'

   A baby sees two objects: a FIXED event (common ground with parent)
   and a NOVEL event (known only to the baby). The baby chooses an
   utterance to help the parent identify the novel object.

   Vocab:  Ball=0  Bear=1  Duck=2  Milk=3  Shoe=4

   The baby models the parent as a pragmatic listener who:
     1. Has a uniform prior over what the scene objects are
     2. Observes which object is the fixed event (common ground)
     3. Models the baby as choosing a referent from {fixed, novel}
        then producing a noisy utterance matching the referent
     4. After hearing the utterance, guesses the baby's referent

   Two goals:
     INFORM:  baby maximizes P(parent guesses novel event)
     CONNECT: baby maximizes P(parent guesses baby's intended referent)

   Computed via exact enumeration — no sampling."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.inference.exact :as exact])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Domain: 5 words/objects
;; ---------------------------------------------------------------------------

(def n-vocab 5)
(def vocab-names ["Ball" "Bear" "Duck" "Milk" "Shoe"])
(def uniform-5 (vec (repeat n-vocab 1.0)))

;; ---------------------------------------------------------------------------
;; Parent's mental model
;;
;; The parent thinks the scene has a fixed and novel event (both uniform
;; over Vocab). The parent observes the fixed event. The parent models
;; the baby as: choosing a referent from {fixed, novel}, then producing
;; an utterance with wpp = (utterance == referent) + 0.25.
;;
;; The parent's referent is a Vocab index (0..4), not a binary choice.
;; ---------------------------------------------------------------------------

(defn parent-model
  "Parent's generative model of the scene and baby's behavior.
   The parent reasons about what the baby might be referring to.

   Traces:
     :fixed-e  — fixed event (uniform prior, will be conditioned)
     :novel-e  — novel event (uniform prior, unknown to parent)
     :referent — baby's intended referent (a Vocab item, must be fixed or novel)
     :utterance — baby's produced word (noisy version of referent)"
  []
  (gen []
    (let [fixed-e (trace :fixed-e (dist/weighted uniform-5))
          novel-e (trace :novel-e (dist/weighted uniform-5))
          ;; Baby chooses referent from {fixed-e, novel-e}
          ;; wpp(r) = 1 if r == fixed-e or r == novel-e, else 0
          ref-wpp (mapv (fn [r]
                          (let [r-s (mx/scalar r mx/int32)]
                            (mx/add (mx/eq? r-s fixed-e)
                                    (mx/eq? r-s novel-e))))
                        (range n-vocab))
          referent (trace :referent (dist/weighted ref-wpp))
          ;; Baby's utterance: wpp(u) = (u == referent) + 0.25
          utt-wpp (mapv (fn [u]
                          (mx/add (mx/eq? referent (mx/scalar u mx/int32))
                                  (mx/scalar 0.25)))
                        (range n-vocab))
          utterance (trace :utterance (dist/weighted utt-wpp))]
      referent)))

;; ---------------------------------------------------------------------------
;; EU computation
;;
;; The baby knows fixed-event and novel-event. For each candidate utterance u,
;; the baby computes:
;;
;;   EU_inform(u) = P(parent guesses novel-event | parent saw fixed-event,
;;                                                  parent heard u)
;;
;; The parent's "guess" is just their posterior over baby.referent.
;; So EU_inform(u) = P(referent == novel-event | fixed-e == f, utterance == u)
;;                    computed from the parent's model.
;;
;; Similarly for connect:
;;   EU_connect(ref, u) = P(referent == ref | fixed-e == f, utterance == u)
;; ---------------------------------------------------------------------------

(defn compute-parent-posterior
  "Compute the parent's posterior P(referent = v | fixed-e = f, utterance = u)
   for all combinations. Returns a 3D nested vector [f][u][v] -> probability."
  []
  (let [model (parent-model)
        joint (exact/exact-joint model [] nil)]
    ;; For each (fixed-val, utterance-val), condition and get P(referent)
    (mapv (fn [f]
            (let [cf (exact/condition-on (:log-probs joint) (:axes joint)
                                         :fixed-e f)]
              (mapv (fn [u]
                      (let [cfu (exact/condition-on (:log-probs cf) (:axes cf)
                                                    :utterance u)
                            m (exact/joint-marginal (:log-probs cfu) (:axes cfu)
                                                    #{:referent})
                            p (mx/exp (:log-probs m))
                            _ (mx/eval! p)]
                        (mapv #(mx/item (mx/idx p %)) (range n-vocab))))
                    (range n-vocab))))
          (range n-vocab))))

;; Precompute the parent's posterior table (used by both goals)
(def parent-posterior (compute-parent-posterior))

;; ---------------------------------------------------------------------------
;; Inform goal
;; ---------------------------------------------------------------------------

(defn inform-eu
  "EU_inform(u) = P(referent == novel-event | fixed-e = f, utterance = u)
   from the parent's perspective. This is how well utterance u helps the
   parent identify the novel event."
  [fixed-val novel-val u]
  (get-in parent-posterior [fixed-val u novel-val]))

(defn compute-inform-probs
  "P(baby says u) under the inform goal.
   P(u) ~ exp(beta * EU_inform(u)), then normalized."
  [fixed-val novel-val beta]
  (let [eus (mapv #(inform-eu fixed-val novel-val %) (range n-vocab))
        weights (mapv #(js/Math.exp (* beta %)) eus)
        total (reduce + weights)]
    (mapv #(/ % total) weights)))

;; ---------------------------------------------------------------------------
;; Connect goal
;; ---------------------------------------------------------------------------

(defn connect-eu
  "EU_connect(ref, u) = P(referent == ref | fixed-e = f, utterance = u)
   from the parent's perspective. This is how well utterance u helps the
   parent guess the baby's intended referent."
  [fixed-val ref-val u]
  (get-in parent-posterior [fixed-val u ref-val]))

(defn compute-connect-probs
  "P(baby says u) under the connect goal.
   Baby sequentially chooses referent then utterance from ALL of Vocab.
   Both use wpp=exp(beta * EU_connect).

   Sequential semantics (matching memo):
     1. For each ref, compute E_u[EU(ref,u)] under optimal u-choice
     2. P(ref) ~ exp(beta * E_u[EU(ref,u)])
     3. P(u|ref) ~ exp(beta * EU(ref,u))
     4. P(u) = sum_ref P(ref) * P(u|ref)"
  [fixed-val _novel-val beta]
  (let [;; Step 1 & 3: for each ref, compute P(u|ref) and expected EU
        ref-data
        (mapv (fn [ref-val]
                (let [;; EU(ref, u) for each utterance
                      eus (mapv #(connect-eu fixed-val ref-val %) (range n-vocab))
                      ;; P(u|ref) ~ exp(beta * EU(ref,u))
                      weights (mapv #(js/Math.exp (* beta %)) eus)
                      w-total (reduce + weights)
                      p-u-given-ref (mapv #(/ % w-total) weights)
                      ;; E_u[EU(ref,u)] under optimal u-choice
                      eu-over-u (reduce + (map * p-u-given-ref eus))]
                  {:p-u-given-ref p-u-given-ref :eu-over-u eu-over-u}))
              (range n-vocab))
        ;; Step 2: P(ref) ~ exp(beta * EU_over_u(ref))
        ref-weights (mapv #(js/Math.exp (* beta (:eu-over-u %))) ref-data)
        ref-total (reduce + ref-weights)
        p-ref (mapv #(/ % ref-total) ref-weights)
        ;; Step 4: P(u) = sum_ref P(ref) * P(u|ref)
        utt-probs (mapv (fn [u]
                          (reduce + (map-indexed
                                      (fn [r rd]
                                        (* (nth p-ref r) (nth (:p-u-given-ref rd) u)))
                                      ref-data)))
                        (range n-vocab))]
    utt-probs))

;; ---------------------------------------------------------------------------
;; Run and display
;; ---------------------------------------------------------------------------

(println "From the Mouths of Babes")
(println "========================")
(println)
(println "A baby sees two objects: one familiar (common ground with parent)")
(println "and one novel (only baby knows). The baby chooses a word to help")
(println "the parent identify the novel object.")
(println)
(println "Vocab: Ball=0  Bear=1  Duck=2  Milk=3  Shoe=4")
(println)

(def beta 2.0)

(defn print-scenario [fixed-val novel-val]
  (let [fixed-name (nth vocab-names fixed-val)
        novel-name (nth vocab-names novel-val)
        inform-p (compute-inform-probs fixed-val novel-val beta)
        connect-p (compute-connect-probs fixed-val novel-val beta)]
    (println (str "Fixed=" fixed-name " (known to both), Novel=" novel-name " (known to baby)"))
    (println "  Goal: INFORM (maximize P(parent identifies novel))")
    (doseq [u (range n-vocab)]
      (println (str "    P(says " (nth vocab-names u) ") = " (.toFixed (nth inform-p u) 4)
                    (when (= u novel-val) "  <- novel word")
                    (when (= u fixed-val) "  <- fixed word"))))
    (println "  Goal: CONNECT (maximize P(parent guesses baby's referent))")
    (doseq [u (range n-vocab)]
      (println (str "    P(says " (nth vocab-names u) ") = " (.toFixed (nth connect-p u) 4)
                    (when (= u novel-val) "  <- novel word")
                    (when (= u fixed-val) "  <- fixed word"))))
    (println)
    {:inform inform-p :connect connect-p}))

;; Scenario 1: Fixed=Ball, Novel=Bear
(println "-- Scenario 1 --")
(def s1 (print-scenario 0 1))

;; Scenario 2: Fixed=Ball, Novel=Duck
(println "-- Scenario 2 --")
(def s2 (print-scenario 0 2))

;; Scenario 3: Fixed=Milk, Novel=Shoe
(println "-- Scenario 3 --")
(def s3 (print-scenario 3 4))

;; ---------------------------------------------------------------------------
;; Verification against memo reference values (beta=2.0)
;; ---------------------------------------------------------------------------

(println "Verification:")

(let [pass? (atom true)
      check (fn [name ok]
              (when-not ok (reset! pass? false))
              (println (str "  " (if ok "PASS" "FAIL") ": " name)))
      close (fn [name expected actual tol]
              (let [ok (< (abs (- expected actual)) tol)]
                (when-not ok (reset! pass? false))
                (println (str "  " (if ok "PASS" "FAIL") ": " name
                              " (expected " (.toFixed expected 4)
                              ", got " (.toFixed actual 4) ")"))))]

  ;; memo reference: inform(fixed=Ball, novel=Bear, beta=2)
  ;;   [0.1616, 0.3112, 0.1757, 0.1757, 0.1757]
  (close "inform s1: P(Ball)"  0.1616 (nth (:inform s1) 0) 1e-3)
  (close "inform s1: P(Bear)"  0.3112 (nth (:inform s1) 1) 1e-3)
  (close "inform s1: P(Duck)"  0.1757 (nth (:inform s1) 2) 1e-3)
  (close "inform s1: P(Milk)"  0.1757 (nth (:inform s1) 3) 1e-3)
  (close "inform s1: P(Shoe)"  0.1757 (nth (:inform s1) 4) 1e-3)

  ;; memo reference: connect(fixed=Ball, novel=Bear, beta=2)
  ;;   [0.2456, 0.1886, 0.1886, 0.1886, 0.1886]
  (close "connect s1: P(Ball)" 0.2456 (nth (:connect s1) 0) 1e-3)
  (close "connect s1: P(Bear)" 0.1886 (nth (:connect s1) 1) 1e-3)

  ;; Structural checks
  (let [p (:inform s1)]
    (check "inform: novel word is most probable"
           (= (nth p 1) (apply max p)))
    (check "inform: fixed word is least probable"
           (= (nth p 0) (apply min p))))

  (let [p (:connect s1)]
    (check "connect: fixed word is most probable"
           (= (nth p 0) (apply max p))))

  ;; Probabilities sum to 1
  (close "inform probs sum to 1"  1.0 (reduce + (:inform s1))  1e-6)
  (close "connect probs sum to 1" 1.0 (reduce + (:connect s1)) 1e-6)

  ;; Symmetry
  (close "symmetry: inform permutation-invariant"
         (nth (:inform s1) 1)
         (nth (:inform (let [ip (compute-inform-probs 1 0 beta)
                             cp (compute-connect-probs 1 0 beta)]
                         {:inform ip :connect cp})) 0)
         1e-6)

  ;; Scenario 3: same structure as s1 with different labels
  (close "inform s3: P(Shoe)" 0.3112 (nth (:inform s3) 4) 1e-3)
  (close "inform s3: P(Milk)" 0.1616 (nth (:inform s3) 3) 1e-3)

  (println)
  (if @pass?
    (println "All checks passed.")
    (do (println "Some checks FAILED.")
        (js/process.exit 1))))
