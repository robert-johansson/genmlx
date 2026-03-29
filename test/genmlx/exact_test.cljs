(ns genmlx.exact-test
  "Tests for exact tensor enumeration engine (genmlx.inference.exact).
   Phase 1: single-model enumeration, conditioning, post-processing.
   Phase 2: multi-agent theory-of-mind (Monty Hall, bernoulli trick, cross-verify).
   Phase 3: Exact combinator (splice, nested exact, caching).
   Phase 4: enumerate (full GFI), mixed inference, ADEV."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.exact :as exact]
            [genmlx.inference.enumerate :as enum]
            [genmlx.protocols :as p]
            [genmlx.selection :as sel]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn- approx= [a b tol]
  (<= (abs (- a b)) tol))

(defn- check [name pred]
  (if pred
    (do (swap! pass-count inc) (println (str "  PASS: " name)))
    (do (swap! fail-count inc) (println (str "  FAIL: " name)))))

(defn- check-close [name expected actual tol]
  (let [ok (approx= expected actual tol)]
    (when-not ok
      (println (str "    expected " expected " got " actual " (tol " tol ")")))
    (check name ok)))

;; ---------------------------------------------------------------------------
;; Phase 1 tests: single model enumeration
;; ---------------------------------------------------------------------------

(println "\n== exact_test.cljs — Phase 1: Exact Tensor Enumeration ==\n")

;; -- Test 1: Fair coin --
(println "\n-- 1. Fair coin --")
(let [model (gen [] (trace :coin (dist/bernoulli 0.5)))
      r (exact/exact-posterior model [] nil)]
  (check-close "P(coin=0)" 0.5 (get-in r [:marginals :coin 0]) 1e-5)
  (check-close "P(coin=1)" 0.5 (get-in r [:marginals :coin 1]) 1e-5)
  (check-close "log-ml" 0.0 (:log-ml r) 1e-5))

;; -- Test 2: Biased coin --
(println "\n-- 2. Biased coin --")
(let [model (gen [] (trace :coin (dist/bernoulli 0.7)))
      r (exact/exact-posterior model [] nil)]
  (check-close "P(coin=0)" 0.3 (get-in r [:marginals :coin 0]) 1e-5)
  (check-close "P(coin=1)" 0.7 (get-in r [:marginals :coin 1]) 1e-5))

;; -- Test 3: Two independent coins —
(println "\n-- 3. Two independent coins --")
(let [model (gen []
              (let [a (trace :a (dist/bernoulli 0.7))
                    b (trace :b (dist/bernoulli 0.4))]
                [a b]))
      r (exact/exact-posterior model [] nil)]
  (check "joint shape [2,2]" (= [2 2] (mx/shape (:joint-log-probs r))))
  (check-close "P(a=0)" 0.3 (get-in r [:marginals :a 0]) 1e-5)
  (check-close "P(a=1)" 0.7 (get-in r [:marginals :a 1]) 1e-5)
  (check-close "P(b=0)" 0.6 (get-in r [:marginals :b 0]) 1e-5)
  (check-close "P(b=1)" 0.4 (get-in r [:marginals :b 1]) 1e-5))

;; -- Test 4: Dependent sites + Bayesian conditioning --
(println "\n-- 4. Bayesian conditioning (noisy coin) --")
(let [model (gen []
              (let [coin (trace :coin (dist/bernoulli 0.5))
                    p (mx/where coin (mx/scalar 0.9) (mx/scalar 0.1))
                    obs (trace :obs (dist/bernoulli p))]
                coin))
      r (exact/exact-posterior model [] (cm/choicemap :obs (mx/scalar 1)))]
  ;; P(coin=1|obs=1) = 0.9*0.5 / (0.9*0.5 + 0.1*0.5) = 0.9
  (check-close "P(coin=0|obs=1)" 0.1 (get-in r [:marginals :coin 0]) 1e-5)
  (check-close "P(coin=1|obs=1)" 0.9 (get-in r [:marginals :coin 1]) 1e-5)
  (check-close "log P(obs=1)" (js/Math.log 0.5) (:log-ml r) 1e-5))

;; -- Test 5: Categorical (3 values) --
(println "\n-- 5. Categorical distribution --")
(let [model (gen []
              (trace :x (dist/weighted [0.2 0.5 0.3])))
      r (exact/exact-posterior model [] nil)]
  (check-close "P(x=0)" 0.2 (get-in r [:marginals :x 0]) 1e-5)
  (check-close "P(x=1)" 0.5 (get-in r [:marginals :x 1]) 1e-5)
  (check-close "P(x=2)" 0.3 (get-in r [:marginals :x 2]) 1e-5))

;; -- Test 6: Categorical + conditioning --
(println "\n-- 6. Categorical + Bayesian conditioning --")
(let [model (gen []
              (let [x (trace :x (dist/weighted [0.2 0.5 0.3]))
                    p (mx/where (mx/eq? x 0)
                                (mx/scalar 0.8)
                                (mx/where (mx/eq? x 1)
                                          (mx/scalar 0.3)
                                          (mx/scalar 0.5)))
                    obs (trace :obs (dist/bernoulli p))]
                x))
      r (exact/exact-posterior model [] (cm/choicemap :obs (mx/scalar 1)))]
  ;; P(x=0|obs=1) = 0.8*0.2/0.46, P(x=1|obs=1) = 0.3*0.5/0.46, P(x=2|obs=1) = 0.5*0.3/0.46
  (check-close "P(x=0|obs=1)" (/ 0.16 0.46) (get-in r [:marginals :x 0]) 1e-4)
  (check-close "P(x=1|obs=1)" (/ 0.15 0.46) (get-in r [:marginals :x 1]) 1e-4)
  (check-close "P(x=2|obs=1)" (/ 0.15 0.46) (get-in r [:marginals :x 2]) 1e-4))

;; -- Test 7: Three-variable chain --
(println "\n-- 7. Three-variable chain --")
(let [model (gen []
              (let [a (trace :a (dist/bernoulli 0.6))
                    b (trace :b (dist/bernoulli (mx/where a (mx/scalar 0.8) (mx/scalar 0.3))))
                    c (trace :c (dist/bernoulli (mx/where b (mx/scalar 0.7) (mx/scalar 0.2))))]
                c))
      r-uncond (exact/exact-posterior model [] nil)
      r-cond (exact/exact-posterior model [] (cm/choicemap :c (mx/scalar 1)))]
  (check "uncond joint shape [2,2,2]" (= [2 2 2] (mx/shape (:joint-log-probs r-uncond))))
  (check-close "P(a=1)" 0.6 (get-in r-uncond [:marginals :a 1]) 1e-5)
  ;; P(b=1) = 0.8*0.6 + 0.3*0.4 = 0.6
  (check-close "P(b=1)" 0.6 (get-in r-uncond [:marginals :b 1]) 1e-5)
  ;; P(c=1) = 0.7*0.6 + 0.2*0.4 = 0.5
  (check-close "P(c=1)" 0.5 (get-in r-uncond [:marginals :c 1]) 1e-5)
  ;; Conditioned: P(b=1|c=1) = 0.84
  (check-close "P(b=1|c=1)" 0.84 (get-in r-cond [:marginals :b 1]) 1e-4)
  ;; P(a=1|c=1) = 0.72
  (check-close "P(a=1|c=1)" 0.72 (get-in r-cond [:marginals :a 1]) 1e-4))

;; -- Test 8: Model with arguments --
(println "\n-- 8. Model with arguments --")
(let [model (gen [p-prior]
              (let [coin (trace :coin (dist/bernoulli p-prior))
                    obs-p (mx/where coin (mx/scalar 0.9) (mx/scalar 0.1))
                    obs (trace :obs (dist/bernoulli obs-p))]
                coin))
      r1 (exact/exact-posterior model [0.5] (cm/choicemap :obs (mx/scalar 1)))
      r2 (exact/exact-posterior model [0.8] (cm/choicemap :obs (mx/scalar 1)))]
  (check-close "prior=0.5: P(coin=1|obs=1)" 0.9 (get-in r1 [:marginals :coin 1]) 1e-5)
  ;; prior=0.8: P(coin=1|obs=1) = 0.9*0.8/(0.9*0.8+0.1*0.2) = 0.72/0.74 ≈ 0.973
  (check-close "prior=0.8: P(coin=1|obs=1)" (/ 0.72 0.74) (get-in r2 [:marginals :coin 1]) 1e-4))

;; -- Test 9: Cross-verify vs Cartesian enumerator --
(println "\n-- 9. Cross-verify vs enumerate-marginals --")
(let [model (gen []
              (let [x (trace :x (dist/weighted [0.2 0.5 0.3]))
                    p (mx/where (mx/eq? x 0)
                                (mx/scalar 0.8)
                                (mx/where (mx/eq? x 1)
                                          (mx/scalar 0.3)
                                          (mx/scalar 0.5)))
                    obs (trace :obs (dist/bernoulli p))]
                x))
      obs (cm/choicemap :obs (mx/scalar 1))
      ep (exact/exact-posterior model [] obs)
      em (enum/enumerate-marginals
           (dyn/auto-key model) [] obs
           {:x [(mx/scalar 0 mx/int32) (mx/scalar 1 mx/int32) (mx/scalar 2 mx/int32)]})]
  (doseq [[v p] (:x em)]
    (check-close (str "x=" v " matches Cartesian")
                 p (get-in ep [:marginals :x v]) 1e-5)))

;; -- Test 10: exact-joint API --
(println "\n-- 10. exact-joint API --")
(let [model (gen []
              (let [a (trace :a (dist/bernoulli 0.7))
                    b (trace :b (dist/bernoulli 0.4))]
                [a b]))
      r (exact/exact-joint model [] nil)]
  (check "probs shape [2,2]" (= [2 2] (mx/shape (:probs r))))
  (check-close "probs sum to 1" 1.0 (mx/item (mx/sum (:probs r))) 1e-5)
  (check-close "log-ml = 0 (no obs)" 0.0 (:log-ml r) 1e-5))

;; -- Test 11: exact-marginal-likelihood --
(println "\n-- 11. exact-marginal-likelihood --")
(let [model (gen []
              (let [coin (trace :coin (dist/bernoulli 0.5))
                    p (mx/where coin (mx/scalar 0.9) (mx/scalar 0.1))
                    obs (trace :obs (dist/bernoulli p))]
                coin))
      ml (exact/exact-marginal-likelihood model [] (cm/choicemap :obs (mx/scalar 1)))]
  (check-close "log P(obs=1) = log(0.5)" (js/Math.log 0.5) ml 1e-5))

;; -- Test 12: Entropy --
(println "\n-- 12. Entropy --")
(let [model (gen [] (trace :coin (dist/bernoulli 0.5)))
      r (exact/exact-joint model [] nil)
      h (exact/entropy (:log-probs r) (:axes r) #{:coin})]
  (mx/eval! h)
  ;; H(Bernoulli(0.5)) = log(2) ≈ 0.6931
  (check-close "H(fair coin)" (js/Math.log 2) (mx/item h) 1e-4))

;; -- Test 13: Expectation and Variance --
(println "\n-- 13. Expectation and Variance --")
(let [model (gen []
              (trace :die (dist/weighted [0.1 0.2 0.3 0.2 0.1 0.1])))
      r (exact/exact-joint model [] nil)
      vals (mx/array #js [0 1 2 3 4 5])
      e-val (exact/expectation (:log-probs r) (:axes r) vals nil)
      v-val (exact/variance (:log-probs r) (:axes r) vals nil)]
  (mx/eval! e-val v-val)
  ;; E = 0*0.1+1*0.2+2*0.3+3*0.2+4*0.1+5*0.1 = 2.3
  (check-close "E[die]" 2.3 (mx/item e-val) 1e-4)
  ;; Var = E[X^2] - E[X]^2 = 7.3 - 5.29 = 2.01
  (check-close "Var[die]" 2.01 (mx/item v-val) 1e-3))

;; -- Test 14: Multiple observations --
(println "\n-- 14. Multiple observations --")
(let [model (gen []
              (let [coin (trace :coin (dist/bernoulli 0.5))
                    obs1 (trace :obs1 (dist/bernoulli (mx/where coin (mx/scalar 0.9) (mx/scalar 0.1))))
                    obs2 (trace :obs2 (dist/bernoulli (mx/where coin (mx/scalar 0.8) (mx/scalar 0.2))))]
                coin))
      r (exact/exact-posterior model []
          (cm/merge-cm (cm/choicemap :obs1 (mx/scalar 1))
                       (cm/choicemap :obs2 (mx/scalar 1))))]
  ;; P(coin=1|obs1=1,obs2=1) = 0.9*0.8*0.5 / (0.9*0.8*0.5 + 0.1*0.2*0.5)
  ;;   = 0.36 / (0.36 + 0.01) = 0.36/0.37 ≈ 0.9730
  (check-close "P(coin=1|obs1=1,obs2=1)" (/ 0.36 0.37) (get-in r [:marginals :coin 1]) 1e-4))

;; -- Test 15: Discrete uniform distribution --
(println "\n-- 15. Discrete uniform distribution --")
(let [model (gen [] (trace :x (dist/discrete-uniform 0 4)))
      r (exact/exact-posterior model [] nil)]
  (check "5 support values" (= 5 (count (get-in r [:marginals :x]))))
  (doseq [v (range 5)]
    (check-close (str "P(x=" v ") = 0.2") 0.2 (get-in r [:marginals :x v]) 1e-5)))

;; -- Test 16: Discrete uniform + conditioning --
(println "\n-- 16. Discrete uniform + conditioning --")
(let [model (gen []
              (let [x (trace :x (dist/discrete-uniform 1 5))
                    p (mx/divide (.astype x mx/float32) (mx/scalar 6.0))
                    obs (trace :obs (dist/bernoulli p))]
                x))
      r (exact/exact-posterior model [] (cm/choicemap :obs (mx/scalar 1)))]
  ;; P(x=k|obs=1) = (k/6) / sum(j/6, j=1..5) = k/15
  (doseq [k (range 1 6)]
    (check-close (str "P(x=" k "|obs=1) = " k "/15")
                 (/ k 15.0) (get-in r [:marginals :x k]) 1e-4)))

;; -- Test 17: Delta (deterministic function of traced value) --
(println "\n-- 17. Delta distribution --")
(let [model (gen []
              (let [x (trace :x (dist/bernoulli 0.6))
                    y (trace :y (dist/delta (mx/multiply x (mx/scalar 2.0))))]
                y))
      r (exact/exact-posterior model [] nil)]
  ;; Delta doesn't add an axis — only :x is free
  (check "only 1 free axis" (= 1 (count (:axes r))))
  (check-close "P(x=0)" 0.4 (get-in r [:marginals :x 0]) 1e-5)
  (check-close "P(x=1)" 0.6 (get-in r [:marginals :x 1]) 1e-5))

;; -- Test 18: Support stored in axes metadata --
(println "\n-- 18. Support in axes metadata --")
(let [model (gen []
              (trace :x (dist/weighted [0.2 0.5 0.3])))
      r (exact/exact-posterior model [] nil)
      ax (first (:axes r))]
  (check "support stored" (some? (:support ax)))
  (check "support has 3 values" (= 3 (count (:support ax))))
  (check "support values are MLX scalars" (mx/array? (first (:support ax)))))

;; ===========================================================================
;; Phase 2: Multi-Agent Theory-of-Mind
;; ===========================================================================

(println "\n== Phase 2: Multi-Agent Theory-of-Mind ==\n")

;; -- Test 19: Bernoulli conditioning trick --
(println "\n-- 19. Bernoulli conditioning trick --")
(let [model (gen []
              (let [s (trace :s (dist/weighted [1/3 1/3 1/3]))
                    mask (mx/neq? s 2)
                    _ (trace :cond (dist/bernoulli mask))]
                s))
      obs (exact/observe-constraint nil [:cond])
      r (exact/exact-posterior model [] obs)]
  ;; memo ref: [0.5, 0.5, 0.0]
  (check-close "P(s=0|s!=2)" 0.5 (get-in r [:marginals :s 0]) 1e-5)
  (check-close "P(s=1|s!=2)" 0.5 (get-in r [:marginals :s 1]) 1e-5)
  (check-close "P(s=2|s!=2)" 0.0 (get-in r [:marginals :s 2]) 1e-5))

;; -- Test 20: Hard evidence (state == 0) --
(println "\n-- 20. Hard evidence (state == 0) --")
(let [model (gen []
              (let [s (trace :s (dist/weighted [1/3 1/3 1/3]))
                    mask (mx/eq? s 0)
                    _ (trace :cond (dist/bernoulli mask))]
                s))
      obs (exact/observe-constraint nil [:cond])
      r (exact/exact-posterior model [] obs)]
  ;; memo ref: [1.0, 0.0, 0.0]
  (check-close "P(s=0|s==0)" 1.0 (get-in r [:marginals :s 0]) 1e-5)
  (check-close "P(s=1|s==0)" 0.0 (get-in r [:marginals :s 1]) 1e-5)
  (check-close "P(s=2|s==0)" 0.0 (get-in r [:marginals :s 2]) 1e-5))

;; Shared Monty Hall model (used in tests 21, 22, 28)
(def ^:private monty-hall-model
  (gen [pick-idx]
    (let [prize (trace :prize (dist/weighted [1/3 1/3 1/3]))
          pick (mx/scalar pick-idx mx/int32)
          v0 (mx/and* (mx/neq? prize 0) (mx/neq? pick 0))
          v1 (mx/and* (mx/neq? prize 1) (mx/neq? pick 1))
          v2 (mx/and* (mx/neq? prize 2) (mx/neq? pick 2))
          logits (mx/transpose (mx/log (mx/stack [v0 v1 v2])))
          reveal (trace :reveal (dist/categorical logits))]
      prize)))

;; -- Test 21: Monty Hall Problem --
(println "\n-- 21. Monty Hall --")
(let [model monty-hall-model
      ;; pick=0, reveal=2 — memo ref: [1/3, 2/3, 0]
      r1 (exact/exact-posterior model [0] (cm/choicemap :reveal (mx/scalar 2 mx/int32)))
      ;; pick=0, reveal=1 — memo ref: [1/3, 0, 2/3]
      r2 (exact/exact-posterior model [0] (cm/choicemap :reveal (mx/scalar 1 mx/int32)))
      ;; pick=1, reveal=2 — memo ref: [2/3, 1/3, 0]
      r3 (exact/exact-posterior model [1] (cm/choicemap :reveal (mx/scalar 2 mx/int32)))]
  (check-close "pick=0,rev=2: P(prize=0)" (/ 1.0 3) (get-in r1 [:marginals :prize 0]) 1e-5)
  (check-close "pick=0,rev=2: P(prize=1)" (/ 2.0 3) (get-in r1 [:marginals :prize 1]) 1e-5)
  (check-close "pick=0,rev=2: P(prize=2)" 0.0 (get-in r1 [:marginals :prize 2]) 1e-5)
  (check-close "pick=0,rev=1: P(prize=2)" (/ 2.0 3) (get-in r2 [:marginals :prize 2]) 1e-5)
  (check-close "pick=1,rev=2: P(prize=0)" (/ 2.0 3) (get-in r3 [:marginals :prize 0]) 1e-5))

;; -- Test 22: condition-on post-processing --
(println "\n-- 22. condition-on post-processing --")
(let [;; Full joint (no constraints) then condition via post-processing
      r (exact/exact-joint monty-hall-model [0] nil)
      c (exact/condition-on (:log-probs r) (:axes r) :reveal 2)
      probs (mx/exp (:log-probs c))
      _ (mx/eval! probs)]
  (check "condition-on removes reveal axis" (= 1 (count (:axes c))))
  (check "remaining axis is :prize" (= :prize (:addr (first (:axes c)))))
  (check-close "P(prize=0|rev=2)" (/ 1.0 3) (mx/item (mx/slice probs 0 1)) 1e-5)
  (check-close "P(prize=1|rev=2)" (/ 2.0 3) (mx/item (mx/slice probs 1 2)) 1e-5))

;; -- Test 23: joint-marginal --
(println "\n-- 23. joint-marginal --")
(let [model (gen []
              (let [a (trace :a (dist/bernoulli 0.7))
                    b (trace :b (dist/bernoulli 0.4))
                    c (trace :c (dist/bernoulli 0.5))]
                [a b c]))
      r (exact/exact-joint model [] nil)
      ;; Keep only :a and :c, marginalize :b
      m (exact/joint-marginal (:log-probs r) (:axes r) #{:a :c})
      probs (mx/exp (:log-probs m))
      _ (mx/eval! probs)]
  (check "joint-marginal shape" (= 2 (count (mx/shape probs))))
  (check "keeps 2 axes" (= 2 (count (:axes m))))
  ;; P(a=1,c=1) = 0.7 * 0.5 = 0.35
  (check-close "probs sum to 1" 1.0 (mx/item (mx/sum probs)) 1e-5))

;; -- Test 24: conditional-marginal --
(println "\n-- 24. conditional-marginal --")
(let [model (gen []
              (let [a (trace :a (dist/bernoulli 0.6))
                    b (trace :b (dist/bernoulli (mx/where a (mx/scalar 0.8) (mx/scalar 0.3))))
                    c (trace :c (dist/bernoulli (mx/where b (mx/scalar 0.7) (mx/scalar 0.2))))]
                c))
      r (exact/exact-joint model [] nil)
      ;; P(a | c=1) — should match Phase 1 test
      cm-lp (exact/conditional-marginal (:log-probs r) (:axes r) :a :c 1)
      probs (mx/exp cm-lp)
      _ (mx/eval! probs)]
  (check-close "P(a=0|c=1)" 0.28 (mx/item (mx/slice probs 0 1)) 1e-3)
  (check-close "P(a=1|c=1)" 0.72 (mx/item (mx/slice probs 1 2)) 1e-3))

;; -- Test 25: Namespaced addresses + agent-marginal --
(println "\n-- 25. Namespaced addresses + agent-marginal --")
(let [model (gen []
              (let [prize (trace :monty/prize (dist/weighted [1/3 1/3 1/3]))
                    action (trace :monty/action (dist/bernoulli 0.5))
                    guess (trace :player/guess (dist/weighted [1/3 1/3 1/3]))]
                [prize guess]))
      r (exact/exact-joint model [] nil)
      m (exact/agent-marginal (:log-probs r) (:axes r) "monty")]
  (check "agent-marginal keeps monty axes" (= 2 (count (:axes m))))
  (check "all axes are monty/*"
         (every? #(= "monty" (namespace (:addr %))) (:axes m))))

;; -- Test 26: observe-constraint helper --
(println "\n-- 26. observe-constraint helper --")
(let [obs (exact/observe-constraint nil [:c1 :c2 :c3])]
  (check "c1 constrained" (cm/has-value? (cm/get-submap obs :c1)))
  (check "c2 constrained" (cm/has-value? (cm/get-submap obs :c2)))
  (check "c3 constrained" (cm/has-value? (cm/get-submap obs :c3)))
  (check "c1 value is 1" (= 1.0 (mx/item (cm/get-value (cm/get-submap obs :c1))))))

;; -- Test 27: Noisy coin (multi-agent via bernoulli trick) --
(println "\n-- 27. Noisy coin (multi-agent reasoning) --")
(let [model (gen []
              (let [coin (trace :coin (dist/bernoulli 0.5))
                    obs (trace :obs (dist/bernoulli (mx/where coin (mx/scalar 0.9) (mx/scalar 0.1))))
                    ;; Condition on obs=1 via bernoulli trick (alternative to constraints)
                    mask (mx/eq? obs 1)
                    _ (trace :cond (dist/bernoulli mask))]
                coin))
      obs (exact/observe-constraint nil [:cond])
      r (exact/exact-posterior model [] obs)]
  ;; memo ref: [0.1, 0.9]
  (check-close "P(coin=0|obs=1)" 0.1 (get-in r [:marginals :coin 0]) 1e-4)
  (check-close "P(coin=1|obs=1)" 0.9 (get-in r [:marginals :coin 1]) 1e-4))

;; -- Test 28: Cross-verify Monty Hall vs Cartesian enumerator --
(println "\n-- 28. Cross-verify Monty Hall vs Cartesian --")
(let [;; Tensor enumeration
      ep (exact/exact-posterior monty-hall-model [0] (cm/choicemap :reveal (mx/scalar 2 mx/int32)))
      ;; Cartesian enumeration
      em (enum/enumerate-marginals
           (dyn/auto-key monty-hall-model) [0]
           (cm/choicemap :reveal (mx/scalar 2 mx/int32))
           {:prize [(mx/scalar 0 mx/int32) (mx/scalar 1 mx/int32) (mx/scalar 2 mx/int32)]})]
  (doseq [[v p] (:prize em)]
    (check-close (str "prize=" v " matches Cartesian") p (get-in ep [:marginals :prize v]) 1e-5)))

;; -- Test 29: Chained condition-on (exercises squeeze + dim remapping) --
(println "\n-- 29. Chained condition-on --")
(let [model (gen []
              (let [a (trace :a (dist/bernoulli 0.6))
                    b (trace :b (dist/bernoulli (mx/where a (mx/scalar 0.8) (mx/scalar 0.3))))
                    c (trace :c (dist/bernoulli (mx/where b (mx/scalar 0.7) (mx/scalar 0.2))))]
                c))
      r (exact/exact-joint model [] nil)
      ;; Condition on c=1, then condition on b=1
      c1 (exact/condition-on (:log-probs r) (:axes r) :c 1)
      c2 (exact/condition-on (:log-probs c1) (:axes c1) :b 1)
      probs (mx/exp (:log-probs c2))
      _ (mx/eval! probs)]
  (check "3→2→1 axes" (= 1 (count (:axes c2))))
  (check "remaining is :a" (= :a (:addr (first (:axes c2)))))
  ;; P(a=1|b=1,c=1) = P(c=1|b=1)*P(b=1|a=1)*P(a=1) / Z
  ;; = 0.7 * 0.8 * 0.6 / (0.7*0.8*0.6 + 0.7*0.3*0.4)
  ;; = 0.336 / (0.336 + 0.084) = 0.336/0.42 = 0.8
  (check-close "P(a=0|b=1,c=1)" 0.2 (mx/item (mx/slice probs 0 1)) 1e-4)
  (check-close "P(a=1|b=1,c=1)" 0.8 (mx/item (mx/slice probs 1 2)) 1e-4))

;; -- Test 30: Chained joint-marginal → condition-on --
(println "\n-- 30. joint-marginal then condition-on --")
(let [model (gen []
              (let [a (trace :a (dist/bernoulli 0.6))
                    b (trace :b (dist/bernoulli 0.4))
                    c (trace :c (dist/bernoulli 0.5))]
                [a b c]))
      r (exact/exact-joint model [] nil)
      ;; Keep :a and :b, marginalize :c
      m (exact/joint-marginal (:log-probs r) (:axes r) #{:a :b})
      ;; Then condition on :b = 1
      c (exact/condition-on (:log-probs m) (:axes m) :b 1)
      probs (mx/exp (:log-probs c))
      _ (mx/eval! probs)]
  (check "joint-marginal→condition-on produces 1 axis" (= 1 (count (:axes c))))
  ;; a and b are independent, so P(a=1|b=1) = P(a=1) = 0.6
  (check-close "P(a=0|b=1)" 0.4 (mx/item (mx/slice probs 0 1)) 1e-4)
  (check-close "P(a=1|b=1)" 0.6 (mx/item (mx/slice probs 1 2)) 1e-4))

;; ===========================================================================
;; Phase 3: Exact Combinator — Compositional Exact Inference
;; ===========================================================================

(println "\n== Phase 3: Exact Combinator ==\n")

;; -- Test 31: Basic splice + Exact --
(println "\n-- 31. Basic splice + Exact --")
(let [sub-model (gen [] (trace :coin (dist/bernoulli 0.7)))
      parent (gen []
               (let [probs (splice :agent (exact/Exact sub-model))]
                 probs))
      r (exact/exact-joint parent [] nil)]
  (mx/eval! (:retval r))
  ;; Splice returns probability tensor [0.3, 0.7]
  (check-close "P(coin=0) via splice" 0.3 (mx/item (mx/slice (:retval r) 0 1)) 1e-5)
  (check-close "P(coin=1) via splice" 0.7 (mx/item (mx/slice (:retval r) 1 2)) 1e-5)
  (check-close "log-ml = 0 (no obs)" 0.0 (:log-ml r) 1e-5))

;; -- Test 32: Splice + Exact with constraints --
(println "\n-- 32. Splice + Exact with constraints --")
(let [sub-model (gen []
                  (let [coin (trace :coin (dist/bernoulli 0.5))
                        p (mx/where coin (mx/scalar 0.9) (mx/scalar 0.1))
                        obs (trace :obs (dist/bernoulli p))]
                    coin))
      parent (gen []
               (let [probs (splice :agent (exact/Exact sub-model))]
                 probs))
      ;; Constraint on sub-model's :obs via hierarchical addressing
      r (exact/exact-joint parent []
          (cm/choicemap :agent (cm/choicemap :obs (mx/scalar 1))))]
  (mx/eval! (:retval r))
  ;; P(coin | obs=1) = [0.1, 0.9]
  (check-close "P(coin=0|obs=1) via splice" 0.1 (mx/item (mx/slice (:retval r) 0 1)) 1e-5)
  (check-close "P(coin=1|obs=1) via splice" 0.9 (mx/item (mx/slice (:retval r) 1 2)) 1e-5)
  ;; log-ml = log(0.5) since P(obs=1) = 0.5
  (check-close "log-ml = log(0.5)" (js/Math.log 0.5) (:log-ml r) 1e-5))

;; -- Test 33: Parent enumerates + Exact splice --
(println "\n-- 33. Parent enumerates + Exact splice --")
(let [sub-model (gen [] (trace :action (dist/weighted [0.2 0.5 0.3])))
      parent (gen []
               (let [world (trace :world (dist/bernoulli 0.6))
                     agent-probs (splice :agent (exact/Exact sub-model))]
                 ;; agent-probs independent of world → marginals unchanged
                 world))
      r (exact/exact-posterior parent [] nil)]
  (check-close "P(world=0)" 0.4 (get-in r [:marginals :world 0]) 1e-5)
  (check-close "P(world=1)" 0.6 (get-in r [:marginals :world 1]) 1e-5))

;; -- Test 34: categorical-argmax --
(println "\n-- 34. categorical-argmax --")
(let [d (exact/categorical-argmax (mx/array #js [3.0 7.0 2.0]))]
  (check-close "argmax picks index 1" 1.0
               (js/Math.exp (mx/item (dc/dist-log-prob d (mx/scalar 1 mx/int32)))) 1e-5)
  (check-close "non-argmax is 0" 0.0
               (js/Math.exp (mx/item (dc/dist-log-prob d (mx/scalar 0 mx/int32)))) 1e-5))

;; ===========================================================================
;; Phase 4: enumerate — Full GFI via Handler Substitution
;; ===========================================================================

(println "\n== Phase 4: enumerate — Full GFI via Handler Substitution ==\n")

;; -- Test 35: enumerate simulate --
(println "\n-- 35. enumerate simulate --")
(let [model (gen [] (trace :x (dist/weighted [0.2 0.5 0.3])))
      egf (exact/thinks model)
      trace (p/simulate egf [])]
  (check "simulate: choices empty" (= cm/EMPTY (:choices trace)))
  (check-close "simulate: score = 0" 0.0 (mx/item (:score trace)) 1e-5)
  ;; retval = probability tensor [0.2, 0.5, 0.3]
  (let [probs (:retval trace)]
    (check "simulate: retval shape [3]" (= [3] (mx/shape probs)))
    (check-close "simulate: P(x=1)" 0.5 (mx/item (mx/slice probs 1 2)) 1e-5)))

;; -- Test 36: enumerate generate --
(println "\n-- 36. enumerate generate --")
(let [model (gen []
              (let [coin (trace :coin (dist/bernoulli 0.5))
                    p (mx/where coin (mx/scalar 0.9) (mx/scalar 0.1))
                    obs (trace :obs (dist/bernoulli p))]
                coin))
      egf (exact/thinks model)
      {:keys [trace weight]} (p/generate egf [] (cm/choicemap :obs (mx/scalar 1)))]
  ;; P(coin | obs=1) = [0.1, 0.9], weight = log(0.5)
  (check-close "generate: P(coin=0|obs=1)" 0.1
               (mx/item (mx/slice (:retval trace) 0 1)) 1e-5)
  (check-close "generate: P(coin=1|obs=1)" 0.9
               (mx/item (mx/slice (:retval trace) 1 2)) 1e-5)
  (check-close "generate: weight = log(0.5)"
               (js/Math.log 0.5) (mx/item weight) 1e-5)
  (check-close "generate: score = log-ml"
               (mx/item weight) (mx/item (:score trace)) 1e-5))

;; -- Test 37: enumerate update --
(println "\n-- 37. enumerate update --")
(let [model (gen []
              (let [coin (trace :coin (dist/bernoulli 0.5))
                    p (mx/where coin (mx/scalar 0.9) (mx/scalar 0.1))
                    obs (trace :obs (dist/bernoulli p))]
                coin))
      egf (exact/thinks model)
      ;; First: generate with obs=1
      {:keys [trace]} (p/generate egf [] (cm/choicemap :obs (mx/scalar 1)))
      ;; Update: change to obs=0
      {:keys [trace weight]} (p/update egf trace (cm/choicemap :obs (mx/scalar 0)))]
  ;; P(coin | obs=0) = [0.9, 0.1]
  (check-close "update: P(coin=0|obs=0)" 0.9
               (mx/item (mx/slice (:retval trace) 0 1)) 1e-5)
  (check-close "update: P(coin=1|obs=0)" 0.1
               (mx/item (mx/slice (:retval trace) 1 2)) 1e-5)
  ;; Weight = delta log-ml = log(0.5) - log(0.5) = 0
  (check-close "update: weight = 0 (same log-ml)" 0.0 (mx/item weight) 1e-5))

;; -- Test 38: enumerate assess --
(println "\n-- 38. enumerate assess --")
(let [model (gen []
              (let [coin (trace :coin (dist/bernoulli 0.5))
                    obs (trace :obs (dist/bernoulli (mx/where coin (mx/scalar 0.9) (mx/scalar 0.1))))]
                coin))
      egf (exact/thinks model)
      {:keys [weight]} (p/assess egf []
                         (cm/merge-cm (cm/choicemap :coin (mx/scalar 1))
                                      (cm/choicemap :obs (mx/scalar 1))))]
  ;; P(coin=1, obs=1) = 0.5 * 0.9 = 0.45 → log ≈ -0.799
  (check-close "assess: log P(coin=1,obs=1)" (js/Math.log 0.45) (mx/item weight) 1e-4))

;; -- Test 39: enumerate via splice --
(println "\n-- 39. enumerate via splice --")
(let [sub-model (gen [] (trace :coin (dist/bernoulli 0.7)))
      parent (gen []
               (let [probs (splice :agent (exact/thinks sub-model))]
                 probs))
      r (exact/exact-joint parent [] nil)]
  (mx/eval! (:retval r))
  (check-close "enumerate splice: P(coin=0)" 0.3
               (mx/item (mx/slice (:retval r) 0 1)) 1e-5)
  (check-close "enumerate splice: P(coin=1)" 0.7
               (mx/item (mx/slice (:retval r) 1 2)) 1e-5))

;; -- Test 40: enumerate matches Exact annotation --
(println "\n-- 40. enumerate matches Exact annotation --")
(let [model (gen []
              (let [coin (trace :coin (dist/bernoulli 0.5))
                    p (mx/where coin (mx/scalar 0.9) (mx/scalar 0.1))
                    obs (trace :obs (dist/bernoulli p))]
                coin))
      ;; Via Exact annotation
      parent-annot (gen []
                     (splice :agent (exact/Exact model)))
      r-annot (exact/exact-joint parent-annot []
                (cm/choicemap :agent (cm/choicemap :obs (mx/scalar 1))))
      ;; Via enumerate
      parent-gf (gen []
                  (splice :agent (exact/thinks model)))
      r-gf (exact/exact-joint parent-gf []
              (cm/choicemap :agent (cm/choicemap :obs (mx/scalar 1))))]
  (mx/eval! (:retval r-annot) (:retval r-gf))
  ;; Both should produce identical results
  (let [diff (mx/sum (mx/abs (mx/subtract (:retval r-annot) (:retval r-gf))))]
    (mx/eval! diff)
    (check-close "enumerate == Exact annotation" 0.0 (mx/item diff) 1e-5)))

;; -- Test 41: Mixed inference (MCMC + enumerate) --
(println "\n-- 41. Mixed inference (MCMC + enumerate) --")
(let [;; Model: continuous mu + discrete agent (exactly marginalized)
      mixed (gen []
              (let [mu (trace :mu (dist/gaussian 0 5))
                    agent-probs (splice :agent (exact/thinks
                                  (gen [] (trace :choice (dist/bernoulli 0.6)))))
                    ;; effective = mu + 2 * P(choice=1)
                    p1 (mx/idx agent-probs 1)
                    effective (mx/add mu (mx/multiply (mx/scalar 2.0) p1))]
                (trace :obs (dist/gaussian effective 1))))
      obs (cm/choicemap :obs (mx/scalar 5.0))
      {:keys [trace]} (p/generate (dyn/auto-key mixed) [] obs)
      _ (mx/eval! (:score trace))
      selection (sel/select :mu)
      ;; Run 300 MH steps
      final (reduce
              (fn [{:keys [trace mus]} _]
                (let [{new-trace :trace w :weight}
                      (p/regenerate (dyn/auto-key mixed) trace selection)
                      _ (mx/eval! (:score new-trace) w)
                      accept? (or (>= (mx/item w) 0)
                                  (> (mx/item w) (js/Math.log (js/Math.random))))
                      use-trace (if accept? new-trace trace)
                      mu-val (mx/item (cm/get-choice (:choices use-trace) [:mu]))]
                  {:trace use-trace :mus (conj mus mu-val)}))
              {:trace trace :mus []}
              (range 300))
      posterior-mus (subvec (:mus final) 100)
      mean-mu (/ (reduce + posterior-mus) (count posterior-mus))]
  ;; effective = mu + 1.2, obs = 5 → posterior mu ≈ 3.8
  (check-close "MCMC posterior mu near 3.8" 3.8 mean-mu 1.0))

;; -- Test 42: Gradient descent through enumerate (ADEV) --
(println "\n-- 42. Gradient descent through enumerate --")
(let [;; RSA beta fitting: gradient descent on beta to match human data
      ;; Target: "green" → 64% green_circle (Qing & Franke)
      target-green (mx/array #js [0.36 0.64 0.0])
      rsa-den (mx/array #js [#js [1 0 1 0] #js [1 0 0 1] #js [0 1 0 1]])
      uniform-3 (mx/log (mx/array #js [1/3 1/3 1/3]))
      loss-fn (fn [beta]
                (let [m0 (gen [] (let [r (trace :r (dist/categorical uniform-3))
                                       u (trace :u (dist/categorical-weights
                                                     (.astype rsa-den mx/float32)))] r))
                      j0 (exact/exact-joint m0 [] nil)
                      L0 (exact/extract-table j0 :u)
                      m1 (gen [] (let [r (trace :r (dist/categorical uniform-3))
                                       wpp (mx/multiply (.astype rsa-den mx/float32)
                                             (mx/exp (mx/multiply beta (mx/transpose L0))))
                                       u (trace :u (dist/categorical-weights wpp))] r))
                      j1 (exact/exact-joint m1 [] nil)
                      L1 (exact/extract-table j1 :u)
                      pred (mx/idx L1 0)
                      diff (mx/subtract pred target-green)]
                  (mx/sum (mx/multiply diff diff))))
      grad-fn (mx/grad loss-fn)
      ;; 50 gradient steps
      result (reduce
               (fn [{:keys [beta]} _]
                 (let [g (grad-fn beta) _ (mx/eval! g)
                       nb (mx/subtract beta (mx/multiply (mx/scalar 0.5) g))
                       _ (mx/eval! nb)]
                   {:beta nb}))
               {:beta (mx/scalar 1.0)}
               (range 50))
      final-loss (loss-fn (:beta result))]
  (mx/eval! final-loss)
  ;; Loss should decrease significantly from initial
  (let [initial-loss (loss-fn (mx/scalar 1.0))]
    (mx/eval! initial-loss)
    (check "grad descent reduces loss"
           (< (mx/item final-loss) (* 0.5 (mx/item initial-loss))))))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n== Results: " @pass-count " passed, " @fail-count " failed =="))
(when (pos? @fail-count) (js/process.exit 1))
