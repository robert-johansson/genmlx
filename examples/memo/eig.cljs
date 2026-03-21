(ns eig
  "Expected Information Gain (EIG) for optimal question selection.

   Setup: Two fair dice (1..6 each). We can ask one yes/no question
   about the sum before seeing the dice. Which question tells us
   the most about the outcome?

   Questions:
     Q0: Is the sum = 7?
     Q1: Is the sum > 6?
     Q2: Is the sum even?
     Q3: Is the sum prime?

   EIG = H(answer) — the entropy of the answer distribution. A question
   that splits outcomes evenly (50/50 yes/no) has EIG = ln(2) = 0.6931.

   Result: 'Is the sum even?' achieves EIG = ln(2), a perfect bisection
   of the outcome space. It is the best question to ask."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.inference.exact :as exact])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Question functions: sum (2..12) -> float 0/1
;; ---------------------------------------------------------------------------

(def q-fns
  [;; Q0: Is the sum exactly 7?
   (fn [s] (mx/eq? s 7))
   ;; Q1: Is the sum > 6?
   (fn [s] (mx/gt? s 6))
   ;; Q2: Is the sum even?
   (fn [s]
     (.astype (mx/take-idx (mx/array #js [1 0 1 0 1 0 1 0 1 0 1 0 1])
                           (.astype s mx/int32) 0) mx/float32))
   ;; Q3: Is the sum prime?
   (fn [s]
     (.astype (mx/take-idx (mx/array #js [0 0 1 1 0 1 0 1 0 0 0 1 0])
                           (.astype s mx/int32) 0) mx/float32))])

(def q-names ["sum = 7?" "sum > 6?" "sum even?" "sum prime?"])

;; ---------------------------------------------------------------------------
;; Model: two dice + question answer
;; ---------------------------------------------------------------------------

(defn make-model [q-fn]
  (gen []
    (let [nr (trace :nr (dist/weighted (vec (repeat 6 1.0))))
          nb (trace :nb (dist/weighted (vec (repeat 6 1.0))))
          s  (mx/add (mx/add nr nb) (mx/scalar 2 mx/int32))
          yes (q-fn s)
          wpp (mx/stack #js [(mx/subtract (mx/scalar 1.0) yes) yes] -1)
          a  (trace :a (dist/categorical
                         (mx/log (mx/maximum wpp (mx/scalar 1e-30)))))]
      s)))

;; ---------------------------------------------------------------------------
;; Compute EIG for each question
;; ---------------------------------------------------------------------------

(def eig-vals
  (mapv (fn [q-fn]
          (let [joint (exact/exact-joint (make-model q-fn) [] nil)
                h (exact/entropy (:log-probs joint) (:axes joint) #{:a})]
            (mx/eval! h)
            (mx/item h)))
        q-fns))

;; ---------------------------------------------------------------------------
;; Display results
;; ---------------------------------------------------------------------------

(println "Expected Information Gain (EIG)")
(println "===============================")
(println)
(println "Two fair dice. Which yes/no question about the sum is most informative?")
(println)

(doseq [[i name eig] (map vector (range) q-names eig-vals)]
  (println (str "  Q" i ": " name (apply str (repeat (- 14 (count name)) " "))
                "EIG = " (.toFixed eig 4) " nats"
                (when (= i 2) "  <- ln(2), perfect bisection"))))

(println)

(let [best-idx (.indexOf eig-vals (apply max eig-vals))]
  (println (str "Best question: Q" best-idx " (" (nth q-names best-idx) ")"
                " with EIG = " (.toFixed (nth eig-vals best-idx) 4)
                " = ln(2) = " (.toFixed (js/Math.log 2) 4)))
  (println)
  (println "Why? Exactly half of all 36 dice outcomes have even sums,")
  (println "so the answer is always 50/50 — maximum entropy."))

;; ---------------------------------------------------------------------------
;; Verification
;; ---------------------------------------------------------------------------

(println)
(println "Verification:")

(let [pass? (atom true)
      close (fn [name expected actual tol]
              (let [ok (< (abs (- expected actual)) tol)]
                (when-not ok (reset! pass? false))
                (println (str "  " (if ok "PASS" "FAIL") ": " name
                              " (expected " (.toFixed expected 4)
                              ", got " (.toFixed actual 4) ")"))))
      check (fn [name ok]
              (when-not ok (reset! pass? false))
              (println (str "  " (if ok "PASS" "FAIL") ": " name)))]
  ;; EIG values
  (close "EIG(sum=7?)"  0.4506 (nth eig-vals 0) 1e-3)
  (close "EIG(sum>6?)"  0.6792 (nth eig-vals 1) 1e-3)
  (close "EIG(even?)"   0.6931 (nth eig-vals 2) 1e-3)
  (close "EIG(prime?)"  0.6792 (nth eig-vals 3) 1e-3)
  ;; Best question
  (let [best-idx (.indexOf eig-vals (apply max eig-vals))]
    (check "best question is Q2 (even?)" (= 2 best-idx)))
  ;; EIG(even?) = ln(2)
  (close "EIG(even?) = ln(2)" (js/Math.log 2) (nth eig-vals 2) 1e-4)
  ;; Ordering
  (check "even? > sum>6?" (> (nth eig-vals 2) (nth eig-vals 1)))
  (check "sum>6? > sum=7?" (> (nth eig-vals 1) (nth eig-vals 0)))
  (check "sum>6? = prime?" (< (abs (- (nth eig-vals 1) (nth eig-vals 3))) 1e-4))
  (println)
  (if @pass?
    (println "All checks passed.")
    (do (println "Some checks FAILED.")
        (js/process.exit 1))))
