(ns scalar-implicature
  "Scalar Implicature — pragmatic reasoning about 'some' vs 'all'.

   Domain: N = 0..100 (how many), U = {NONE=0, SOME=1, ALL=2}

   Meaning function:
     NONE: true iff n = 0
     SOME: true iff n > 0
     ALL:  true iff n = 100

   A naive literal listener interprets each utterance by filtering:
     P_L0(n | u) = meaning(n, u) / sum_n meaning(n, u)

   A pragmatic speaker chooses utterances weighted by how well the
   naive listener would recover the true n:
     P_S1(u | n) ~ P_L0(n | u)

   The pragmatic listener (L1) inverts the speaker via Bayes' rule:
     P_L1(n | u) ~ P_S1(u | n) * P(n)

   Result (scalar implicature):
     'some' implies 'not none' AND 'not all' — even though 'some'
     literally includes n=100. The speaker would say 'all' if n=100,
     so saying 'some' implicates n < 100."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.inference.exact :as exact])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Setup: meaning table, naive listener, speaker logits
;; ---------------------------------------------------------------------------

(def nn 100)
(def n-vals (.astype (mx/arange 0 (inc nn) 1) mx/int32))

;; Meaning table [101, 3]: rows=n, cols=utterance
(def meaning-tbl
  (let [none-c (mx/eq? n-vals 0)
        some-c (mx/gt? n-vals 0)
        all-c  (mx/eq? n-vals nn)]
    (mx/stack #js [none-c some-c all-c] -1)))

;; Naive listener: P(n|u) = meaning(n,u) / sum_n meaning(n,u)
(def naive-table (mx/divide meaning-tbl (mx/sum meaning-tbl [0] true)))

;; Speaker logits: log(P_naive(n|u)), guarded for zeros
(def spk-logits (mx/log (mx/maximum naive-table (mx/scalar 1e-30))))

;; Uniform prior over n
(def uniform-logits (mx/zeros #js [101]))

(mx/eval! meaning-tbl naive-table spk-logits)

;; ---------------------------------------------------------------------------
;; Pragmatic listener via exact enumeration
;; ---------------------------------------------------------------------------

;; Speaker model: chooses n uniformly, then u weighted by P_naive(n|u)
;; Joint: P(n, u) ~ P(n) * P_S1(u | n)
(def speaker
  (gen []
    (let [n (trace :n (dist/categorical uniform-logits))
          u (trace :u (dist/categorical spk-logits))]
      n)))

(def joint (exact/exact-joint speaker [] nil))
(def table (exact/extract-table joint :u))
(mx/eval! table)

;; table shape: [3, 101] — row = utterance, col = n
;; Each row is P(n | utterance) for the pragmatic listener

;; ---------------------------------------------------------------------------
;; Display results
;; ---------------------------------------------------------------------------

(println "Scalar Implicature")
(println "==================")
(println)
(println "Pragmatic listener: P(n | utterance)")
(println)

;; NONE
(let [none-row (mx/idx table 0)]
  (println "NONE (u=0):")
  (println (str "  P(n=0 | NONE) = " (.toFixed (mx/item (mx/idx none-row 0)) 4)))
  (println (str "  P(n=1 | NONE) = " (.toFixed (mx/item (mx/idx none-row 1)) 6)))
  (println (str "  P(n=50| NONE) = " (.toFixed (mx/item (mx/idx none-row 50)) 6)))
  (println))

;; ALL
(let [all-row (mx/idx table 2)]
  (println "ALL (u=2):")
  (println (str "  P(n=100| ALL) = " (.toFixed (mx/item (mx/idx all-row nn)) 4)))
  (println (str "  P(n=99 | ALL) = " (.toFixed (mx/item (mx/idx all-row 99)) 6)))
  (println (str "  P(n=50 | ALL) = " (.toFixed (mx/item (mx/idx all-row 50)) 6)))
  (println))

;; SOME — the interesting case
(let [some-row (mx/idx table 1)
      _ (mx/eval! some-row)
      p-n0   (mx/item (mx/idx some-row 0))
      p-n1   (mx/item (mx/idx some-row 1))
      p-n50  (mx/item (mx/idx some-row 50))
      p-n99  (mx/item (mx/idx some-row 99))
      p-n100 (mx/item (mx/idx some-row nn))]
  (println "SOME (u=1) — scalar implicature:")
  (println (str "  P(n=0  | SOME) = " (.toFixed p-n0 6) "  (implicature: not none)"))
  (println (str "  P(n=1  | SOME) = " (.toFixed p-n1 4)))
  (println (str "  P(n=50 | SOME) = " (.toFixed p-n50 4) "  (uniform over 1..99)"))
  (println (str "  P(n=99 | SOME) = " (.toFixed p-n99 4)))
  (println (str "  P(n=100| SOME) = " (.toFixed p-n100 6) "  (implicature: not all)"))
  (println)
  (println "Key insight: Saying 'some' implicates 'not none' AND 'not all'.")
  (println "The speaker would say 'all' if n=100 and 'none' if n=0,")
  (println "so 'some' gets its meaning partly by exclusion."))

;; ---------------------------------------------------------------------------
;; Verification
;; ---------------------------------------------------------------------------

(println)
(println "Verification:")

(let [some-row (mx/idx table 1)
      none-row (mx/idx table 0)
      all-row  (mx/idx table 2)
      _ (mx/eval! some-row none-row all-row)
      pass? (atom true)
      check (fn [name ok]
              (when-not ok (reset! pass? false))
              (println (str "  " (if ok "PASS" "FAIL") ": " name)))
      close (fn [name expected actual tol]
              (let [ok (< (abs (- expected actual)) tol)]
                (when-not ok (reset! pass? false))
                (println (str "  " (if ok "PASS" "FAIL") ": " name
                              " (expected " (.toFixed expected 4)
                              ", got " (.toFixed actual 4) ")"))))]
  ;; Table shape
  (check "table shape [3, 101]" (= [3 101] (mx/shape table)))
  ;; Row sums
  (close "NONE row sums to 1" 1.0 (mx/item (mx/sum none-row)) 1e-4)
  (close "SOME row sums to 1" 1.0 (mx/item (mx/sum some-row)) 1e-4)
  (close "ALL row sums to 1"  1.0 (mx/item (mx/sum all-row))  1e-4)
  ;; NONE -> n=0
  (close "P(n=0|NONE) = 1" 1.0 (mx/item (mx/idx none-row 0)) 1e-5)
  (close "P(n=1|NONE) = 0" 0.0 (mx/item (mx/idx none-row 1)) 1e-5)
  ;; ALL -> n=100
  (close "P(n=100|ALL) = 1" 1.0 (mx/item (mx/idx all-row nn)) 1e-5)
  (close "P(n=99|ALL) = 0"  0.0 (mx/item (mx/idx all-row 99)) 1e-5)
  ;; SOME implicature
  (close "P(n=0|SOME) ~ 0 (not none)" 0.0 (mx/item (mx/idx some-row 0)) 1e-5)
  (close "P(n=50|SOME) ~ 0.0101" 0.0101 (mx/item (mx/idx some-row 50)) 1e-3)
  (close "P(n=100|SOME) ~ 0.0001 (not all)" 0.0001 (mx/item (mx/idx some-row nn)) 1e-3)
  (check "SOME: middle >> boundary" (> (mx/item (mx/idx some-row 50))
                                       (* 10 (mx/item (mx/idx some-row nn)))))
  (println)
  (if @pass?
    (println "All checks passed.")
    (do (println "Some checks FAILED.")
        (js/process.exit 1))))
