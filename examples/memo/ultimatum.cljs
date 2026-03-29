(ns ultimatum
  "Ultimatum Game — softmax-rational offerer and receiver.

   An offerer is endowed with $1. They propose a split (discretized 0-1).
   The receiver can accept (both get their share) or reject (nobody gets
   anything). The receiver uses softmax over their payout; the offerer
   reasons about the receiver's likely response via exact enumeration.

   Result: the rational offerer offers the minimum amount where the
   receiver still reliably accepts. In practice humans offer closer
   to 50%, suggesting reasoning about fairness — not just utility.

   Reference: Guth, Schmittberger & Schwarze (1982). An experimental
   analysis of ultimatum bargaining."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.inference.exact :as exact])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Domain setup
;; ---------------------------------------------------------------------------

;; 21 proposals: 0.00, 0.05, 0.10, ..., 1.00
(def n-proposals 21)
(def proposals (mx/divide (mx/arange 0 n-proposals 1) (mx/scalar (dec n-proposals))))
(mx/eval! proposals)

;; Rationality parameter (higher = more rational, less noisy)
(def beta 50.0)

;; ---------------------------------------------------------------------------
;; Receiver model
;; ---------------------------------------------------------------------------
;; Receiver knows the proposal and softmax-chooses accept (0) or reject (1)
;; weighted by exp(beta * payout).
;;   payout(accept) = prop, payout(reject) = 0

(defn make-receiver
  "Build a receiver model for a specific proposal value.
   Chooses accept (0) or reject (1) weighted by exp(beta * payout)."
  [prop-val]
  (gen []
    (let [dec-val (trace :dec (dist/weighted
                                [(js/Math.exp (* beta (mx/item prop-val)))
                                 (js/Math.exp (* beta 0))]))]
      dec-val)))

;; ---------------------------------------------------------------------------
;; Offerer model — reasons about receiver via thinks
;; ---------------------------------------------------------------------------
;; The offerer THINKS about the receiver's response for each candidate
;; proposal. exact/thinks wraps the receiver for exact enumeration, returning
;; the full probability table. exact/pr extracts P(dec=0) = P(accept).
;;
;; Expected utility:
;;   EU(prop) = P(accept | prop) * (1 - prop)
;;
;; The offerer softmax-chooses weighted by exp(beta * EU).

(defn make-offerer
  "Build the offerer model. Uses exact/thinks to imagine the receiver's
   response, exact/pr to extract accept probability."
  []
  (let [;; For each proposal, imagine the receiver's response
        p-accepts (mapv (fn [i]
                          (let [p (mx/squeeze (mx/idx proposals i))]
                            ;; exact/pr: P(dec = 0) from receiver model
                            (exact/pr (make-receiver p) :dec 0)))
                        (range n-proposals))
        ;; EU(prop_i) = P(accept | prop_i) * (1 - prop_i)
        eus (mx/stack (mapv (fn [i]
                              (let [p-val (mx/item (mx/squeeze (mx/idx proposals i)))
                                    eu (* (nth p-accepts i) (- 1.0 p-val))]
                                (mx/scalar eu)))
                            (range n-proposals)))
        ;; Softmax weights: exp(beta * EU)
        weights (mx/exp (mx/multiply (mx/scalar beta) eus))]
    (mx/eval! weights)
    (gen []
      (let [prop-idx (trace :prop (dist/categorical-weights weights))]
        prop-idx))))

;; ---------------------------------------------------------------------------
;; Alternative: offerer with splice + thinks inside gen body
;; ---------------------------------------------------------------------------
;; This version uses splice + exact/thinks directly inside the gen body,
;; which is the idiomatic GenMLX pattern for theory-of-mind reasoning.
;; The offerer's gen body splices each receiver sub-model via exact enumeration.

(def offerer-thinks
  (let [receivers (mapv (fn [i] (exact/thinks (make-receiver (mx/squeeze (mx/idx proposals i)))))
                        (range n-proposals))]
    (gen []
      (let [;; Think about receiver's response for each proposal
            eus (mx/stack
                  (mapv (fn [i]
                          (let [;; splice the enumerate-wrapped GF — returns [P(accept), P(reject)]
                                recv-probs (splice (keyword (str "recv" i))
                                                   (nth receivers i))
                                p-accept (mx/squeeze (mx/idx recv-probs 0))
                                p-val    (mx/item (mx/squeeze (mx/idx proposals i)))
                                eu       (mx/multiply p-accept (mx/scalar (- 1.0 p-val)))]
                            eu))
                        (range n-proposals)))
            prop-idx (trace :prop (dist/categorical-weights
                                    (mx/exp (mx/multiply (mx/scalar beta) eus))))]
        prop-idx))))

;; ---------------------------------------------------------------------------
;; Compute and display results
;; ---------------------------------------------------------------------------

(println "Ultimatum Game")
(println "==============")
(println)

;; Receiver table
(println "Receiver's P(accept | proposal):")
(println "  prop    P(accept)  P(reject)")
(doseq [i (range n-proposals)]
  (let [p (mx/squeeze (mx/idx proposals i))
        p-acc (exact/pr (make-receiver p) :dec 0)
        p-rej (exact/pr (make-receiver p) :dec 1)]
    (println (str "  " (.toFixed (mx/item p) 2) "     "
                  (.toFixed p-acc 4) "     "
                  (.toFixed p-rej 4)))))

;; Offerer distribution (using splice+thinks version)
(println)
(println "Offerer's P(propose) — reasoning about receiver via thinks:")
(println "  prop    P(propose)")

(let [r (exact/exact-posterior offerer-thinks [] nil)
      marginals (:marginals r)]
  (def offerer-marginals marginals)
  (doseq [i (range n-proposals)]
    (let [p (mx/item (mx/squeeze (mx/idx proposals i)))
          prob (get-in marginals [:prop i])]
      (println (str "  " (.toFixed p 2) "     " (.toFixed prob 6))))))

(println)
(println "The rational offerer offers the minimum amount where the receiver")
(println "still reliably accepts — maximizing (1 - prop) * P(accept | prop).")

;; ---------------------------------------------------------------------------
;; Verification
;; ---------------------------------------------------------------------------

(println)
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

  ;; Receiver checks
  (close "P(accept | prop=0.0) = 0.5" 0.5
         (exact/pr (make-receiver (mx/scalar 0.0)) :dec 0) 1e-4)
  (close "P(reject | prop=0.0) = 0.5" 0.5
         (exact/pr (make-receiver (mx/scalar 0.0)) :dec 1) 1e-4)
  (check "P(accept | prop=0.5) > 0.99"
         (> (exact/pr (make-receiver (mx/scalar 0.5)) :dec 0) 0.99))

  ;; Offerer checks
  (let [probs (mapv #(get-in offerer-marginals [:prop %]) (range n-proposals))
        mode-idx (first (apply max-key second (map-indexed vector probs)))
        mode-prop (mx/item (mx/squeeze (mx/idx proposals mode-idx)))]
    ;; Mode should be low (offerer exploits receiver's rationality)
    (check "offerer mode is a low proposal (< 0.25)"
           (< mode-prop 0.25))
    ;; Positive offer should beat zero offer
    (check "offerer prefers positive offer over zero"
           (> (nth probs 1) (nth probs 0)))
    ;; P(propose) should decrease with higher proposals
    (check "P(prop=0.10) > P(prop=0.25)"
           (> (nth probs 2) (nth probs 5)))
    ;; High proposals should be near zero
    (check "P(prop=0.50) < 0.001"
           (< (nth probs 10) 0.001)))

  ;; Verify splice+thinks matches exact/pr computation
  (let [offerer-pr (make-offerer)
        r-pr (exact/exact-posterior offerer-pr [] nil)
        r-thinks (exact/exact-posterior offerer-thinks [] nil)]
    (check "splice+thinks matches exact/pr offerer"
           (< (abs (- (get-in (:marginals r-pr) [:prop 2])
                      (get-in (:marginals r-thinks) [:prop 2])))
              1e-5)))

  (println)
  (if @pass?
    (println "All checks passed.")
    (do (println "Some checks FAILED.")
        (js/process.exit 1))))
