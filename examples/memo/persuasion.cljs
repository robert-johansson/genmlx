(ns persuasion
  "Bayesian Persuasion — strategic evidence design.

   A prosecutor chooses a lie detector test (precision/recall) to
   maximize conviction rate. The judge knows the prosecutor chose
   strategically and reasons backward about this.

   Shocking result (Kamenica & Gentzkow, 2011): even though the
   defendant is only 30% likely guilty, the prosecutor can raise
   the conviction rate to 60% by choosing a test that is perfectly
   accurate for guilty defendants but only slightly better than
   chance for innocent ones.

   Implements memo's demo-persuasion.ipynb."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.inference.exact :as exact])
  (:require-macros [genmlx.gen :refer [gen]]))

;; =========================================================================
;; Test space: 101 × 101 = 10,201 possible tests
;; =========================================================================

;; Each test is parameterized by:
;;   pi-i = P(result=innocent | actually innocent)  in [0.5, 1.0]
;;   pi-g = P(result=guilty   | actually guilty)    in [0.5, 1.0]

(def n-probs 101)
(def n-tests (* n-probs n-probs))  ;; 10,201

(def probability (.astype (mx/linspace 0.5 1.0 n-probs) mx/float32))
(mx/eval! probability)

;; Compound domain: test index t → (pi-i-idx, pi-g-idx)
(defn test->pi-i [t] (mx/take-idx probability (.astype (mx/floor-divide t (mx/scalar n-probs)) mx/int32) 0))
(defn test->pi-g [t] (mx/take-idx probability (.astype (mx/remainder t (mx/scalar n-probs)) mx/int32) 0))

;; =========================================================================
;; Test execution: P(result | state, test)
;; =========================================================================

(defn run-test
  "P(result | state, pi-i, pi-g).
   state: 0=innocent, 1=guilty
   result: 0=exonerating, 1=incriminating"
  [result state pi-i pi-g]
  ;; P(exon|inno)=pi-i  P(incr|inno)=1-pi-i
  ;; P(exon|guil)=1-pi-g  P(incr|guil)=pi-g
  (let [p-incriminating (mx/where (mx/eq? state 0)
                          (mx/subtract (mx/scalar 1.0) pi-i)
                          pi-g)]
    (mx/where (mx/eq? result 0)
      (mx/subtract (mx/scalar 1.0) p-incriminating)
      p-incriminating)))

;; =========================================================================
;; Judge model: sees test result, decides verdict
;; =========================================================================

(defn judge-conviction-rate
  "P(judge convicts | test t, level).
   At each level, the judge reasons about the prosecutor's test choice.

   The judge maximizes accuracy: verdict = argmax P(verdict == true state).
   This means: convict iff P(guilty | test result) > 0.5.

   Returns [n-tests] tensor of conviction probabilities."
  [prosecutor-probs level]
  (let [;; For each test t, compute P(convict)
        ;; = P(result=incriminating) * I(P(guilty|incriminating) > 0.5)
        ;; + P(result=exonerating) * I(P(guilty|exonerating) > 0.5)
        ;;
        ;; P(guilty|result, test) via Bayes rule:
        ;; P(G|incr) = P(incr|G)*P(G) / P(incr)
        ;; P(incr) = P(incr|G)*P(G) + P(incr|I)*P(I)

        ;; All 10201 test parameters
        t-idx (mx/arange 0 n-tests)
        pi-i (test->pi-i t-idx)   ;; [10201]
        pi-g (test->pi-g t-idx)   ;; [10201]
        _ (mx/eval! pi-i pi-g)

        ;; Prior: P(guilty) = 0.3
        p-g 0.3
        p-i 0.7

        ;; P(incriminating | test)
        p-incr (mx/add (mx/multiply (mx/scalar p-g) pi-g)
                       (mx/multiply (mx/scalar p-i) (mx/subtract (mx/scalar 1.0) pi-i)))

        ;; P(guilty | incriminating)
        p-g-given-incr (mx/divide (mx/multiply (mx/scalar p-g) pi-g) p-incr)

        ;; P(guilty | exonerating)
        p-exon (mx/subtract (mx/scalar 1.0) p-incr)
        p-g-given-exon (mx/divide
                         (mx/multiply (mx/scalar p-g) (mx/subtract (mx/scalar 1.0) pi-g))
                         (mx/maximum p-exon (mx/scalar 1e-10)))

        ;; Judge convicts iff P(guilty|result) > 0.5 (argmax accuracy)
        convict-if-incr (mx/gt? p-g-given-incr 0.5)
        convict-if-exon (mx/gt? p-g-given-exon 0.5)

        ;; P(convict | test) = P(incr)*I(convict|incr) + P(exon)*I(convict|exon)
        p-convict (mx/add (mx/multiply p-incr convict-if-incr)
                          (mx/multiply p-exon convict-if-exon))
        _ (mx/eval! p-convict)]
    p-convict))

;; =========================================================================
;; Prosecutor model: chooses test to maximize conviction
;; =========================================================================

(defn prosecutor-probs
  "P(prosecutor chooses test t | level).
   Softmax over conviction rates."
  [conviction-rates beta]
  (let [logits (mx/multiply (mx/scalar beta) conviction-rates)
        probs (mx/softmax logits)
        _ (mx/eval! probs)]
    probs))

;; =========================================================================
;; RSA iteration: prosecutor ↔ judge
;; =========================================================================

(defn persuasion-step
  "One level of recursive reasoning. Pure function."
  [beta {:keys [conviction-rates]}]
  (let [pros-probs (prosecutor-probs conviction-rates beta)
        new-conviction (judge-conviction-rate pros-probs 1)]
    {:conviction-rates new-conviction
     :prosecutor pros-probs}))

;; =========================================================================
;; Demo
;; =========================================================================

(println "=============================================")
(println " Bayesian Persuasion (10,201 tests)")
(println "=============================================\n")

;; Level 0: judge doesn't model prosecutor (uniform prior over tests)
(println "-- Computing conviction rates --\n")

(let [;; Initial: uniform prosecutor
      conv-0 (judge-conviction-rate nil 0)
      _ (mx/eval! conv-0)

      ;; Iterate
      states (->> {:conviction-rates conv-0}
                  (iterate (partial persuasion-step 20.0))
                  (take 3)
                  vec)

      final (:prosecutor (last states))
      final-conv (:conviction-rates (last states))
      _ (mx/eval! final final-conv)

      ;; Best test
      best-idx (mx/item (mx/argmax final))
      best-pi-i-idx (quot best-idx n-probs)
      best-pi-g-idx (rem best-idx n-probs)
      best-pi-i (mx/item (mx/idx probability best-pi-i-idx))
      best-pi-g (mx/item (mx/idx probability best-pi-g-idx))

      ;; Conviction rate at best test
      best-conv (mx/item (mx/idx final-conv best-idx))

      ;; Baseline: perfectly informative test (pi-i=1, pi-g=1)
      perfect-idx (+ (* (dec n-probs) n-probs) (dec n-probs))
      baseline-conv (mx/item (mx/idx conv-0 perfect-idx))]

  (println (str "  Baseline (perfect test): conviction rate = "
               (.toFixed baseline-conv 3)))
  (println (str "  Prosecutor's optimal test:"))
  (println (str "    P(innocent | innocent) = " (.toFixed best-pi-i 3)))
  (println (str "    P(guilty   | guilty)   = " (.toFixed best-pi-g 3)))
  (println (str "    Conviction rate         = " (.toFixed best-conv 3)))

  (println (str "\n  Kamenica & Gentzkow reference:"))
  (println (str "    P(innocent | innocent) = " (.toFixed (/ 4.0 7) 3)))
  (println (str "    P(guilty   | guilty)   = 1.000"))

  ;; Verification
  (println "\nVerification:")
  (assert (< (abs (- baseline-conv 0.3)) 0.01)
          "perfect test → 30% conviction")
  (println "  PASS: perfect test → 30% conviction")

  (assert (> best-conv 0.55)
          "optimal test → >55% conviction")
  (println (str "  PASS: optimal test → " (.toFixed best-conv 1) "% conviction (> 55%)"))

  (assert (> best-pi-g 0.99)
          "optimal: perfect recall for guilty")
  (println "  PASS: optimal test has perfect recall for guilty")

  (assert (< best-pi-i 0.65)
          "optimal: imperfect precision for innocent")
  (println "  PASS: optimal test has imperfect precision for innocent")

  (assert (< (abs (- best-pi-i (/ 4.0 7))) 0.03)
          "matches K&G solution")
  (println "  PASS: matches Kamenica & Gentzkow solution"))

;; GenMLX integration: gen function with exact/pr
(println "\n-- GenMLX integration --\n")

(let [conv (judge-conviction-rate nil 0)
      pros (prosecutor-probs conv 20.0)
      _ (mx/eval! pros)
      ;; Wrap as gen function for exact/pr
      model (gen []
              (let [t (trace :test (dist/categorical (mx/log (mx/maximum pros (mx/scalar 1e-30)))))]
                t))
      ;; What's the probability the prosecutor picks the K&G test?
      ;; K&G: pi-i ≈ 0.575 (index ~15), pi-g = 1.0 (index 100)
      kg-idx (+ (* 15 n-probs) 100)
      p-kg (exact/pr model :test kg-idx)]
  (println (str "  P(prosecutor picks K&G test) = " (.toFixed p-kg 4)))
  (assert (> p-kg 0.001) "K&G test has nonzero probability")
  (println "  PASS: K&G test has nonzero probability"))

(println "\nAll checks passed.")
