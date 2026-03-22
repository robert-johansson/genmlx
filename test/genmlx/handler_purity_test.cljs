(ns genmlx.handler-purity-test
  "Phase 3.1: Handler purity verification.
   Tests verify determinism via the GFI protocol (which handles PRNG properly),
   immutability of Clojure data structures, and idempotent generate behavior.

   Note: MLX PRNG keys are mutable arrays. The handler's rng/split mutates the
   key in-place, so raw handler transitions cannot be called twice with the
   'same' key object. Determinism is tested at the protocol level where
   rng/seed! creates fresh key state for each invocation."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.handler :as handler]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.test-helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ^:private LOG-2PI (js/Math.log (* 2 js/Math.PI)))

;; ---------------------------------------------------------------------------
;; Test models
;; ---------------------------------------------------------------------------

(def single-gaussian
  (dyn/auto-key
    (gen []
      (trace :x (dist/gaussian 0 1)))))

(def two-gaussians
  (dyn/auto-key
    (gen []
      (let [x (trace :x (dist/gaussian 0 1))
            y (trace :y (dist/gaussian 0 1))]
        (mx/add x y)))))

;; ---------------------------------------------------------------------------
;; 3.1 Purity: generate with constraints is fully deterministic
;; ---------------------------------------------------------------------------
;; When all addresses are constrained, the output is deterministic regardless
;; of PRNG state — no randomness is involved.

(deftest generate-constrained-is-deterministic
  (testing "generate with constraints is deterministic (no PRNG involved)"
    (let [d (dist/gaussian 0 1)
          constraints (cm/choicemap :x (mx/scalar 1.5))
          ;; Use two different keys — shouldn't matter for constrained generate
          init1 {:key (rng/fresh-key 1) :choices cm/EMPTY :score (mx/scalar 0.0)
                 :weight (mx/scalar 0.0) :constraints constraints :executor nil}
          init2 {:key (rng/fresh-key 99) :choices cm/EMPTY :score (mx/scalar 0.0)
                 :weight (mx/scalar 0.0) :constraints constraints :executor nil}
          [v1 s1] (handler/generate-transition init1 :x d)
          [v2 s2] (handler/generate-transition init2 :x d)]
      (mx/eval! v1)
      (mx/eval! v2)
      (is (h/close? (mx/item v1) (mx/item v2) 1e-10)
          "constrained generate produces same value regardless of key")
      (mx/eval! (:score s1))
      (mx/eval! (:score s2))
      (is (h/close? (mx/item (:score s1)) (mx/item (:score s2)) 1e-10)
          "constrained generate produces same score regardless of key"))))

(deftest gen-generate-deterministic
  (testing "generate with full constraints is deterministic at protocol level"
    (let [constraints (cm/choicemap :x (mx/scalar 1.5))
          {:keys [trace weight] :as r1} (p/generate single-gaussian [] constraints)
          {:keys [trace weight] :as r2} (p/generate single-gaussian [] constraints)]
      (is (h/close? (h/realize (:weight r1)) (h/realize (:weight r2)) 1e-10)
          "same constraints → same weight")
      (is (h/close? (h/realize (:score (:trace r1))) (h/realize (:score (:trace r2))) 1e-10)
          "same constraints → same score"))))

(deftest assess-is-deterministic
  (testing "assess is fully deterministic (no randomness)"
    (let [choices (cm/choicemap :x (mx/scalar 1.0))
          r1 (p/assess single-gaussian [] choices)
          r2 (p/assess single-gaussian [] choices)]
      (is (h/close? (h/realize (:weight r1)) (h/realize (:weight r2)) 1e-10)
          "same choices → same weight"))))

;; ---------------------------------------------------------------------------
;; 3.1 Purity: generate values match constraints exactly
;; ---------------------------------------------------------------------------

(deftest generate-respects-constraints-exactly
  (testing "constrained value is exactly the constraint"
    (let [constraints (cm/choicemap :x (mx/scalar 1.5))
          {:keys [trace weight]} (p/generate single-gaussian [] constraints)]
      (is (h/close? 1.5
                    (h/realize (cm/get-value (cm/get-submap (:choices trace) :x)))
                    1e-6)
          "constrained value is exact")))

  (testing "weight is analytically correct"
    ;; weight = log N(1.5; 0, 1) = -0.5*log(2pi) - 0.5*1.5^2 = -2.04394
    (let [constraints (cm/choicemap :x (mx/scalar 1.5))
          {:keys [weight]} (p/generate single-gaussian [] constraints)]
      (is (h/close? -2.04394
                    (h/realize weight)
                    1e-4)
          "weight is analytically correct"))))

;; ---------------------------------------------------------------------------
;; 3.1 Purity: handler does not mutate Clojure data structures
;; ---------------------------------------------------------------------------
;; The score in the input state map is an immutable mx/scalar.
;; After the transition, the original state's score should still be 0.
;; Note: the key IS mutated (MLX arrays are mutable), but all other
;; Clojure values (choices, score) remain immutable.

(deftest handler-does-not-mutate-choices
  (testing "simulate-transition does not modify input choices"
    (let [init {:key (rng/fresh-key 42) :choices cm/EMPTY :score (mx/scalar 0.0) :executor nil}
          orig-choices (:choices init)
          _ (handler/simulate-transition init :x (dist/gaussian 0 1))]
      (is (= cm/EMPTY orig-choices)
          "input choices unchanged after transition"))))

(deftest handler-does-not-mutate-score
  (testing "simulate-transition returns new score, original score object preserved"
    ;; The score is an MLX scalar (immutable value type).
    ;; mx/add creates a new array, doesn't modify the original.
    (let [orig-score (mx/scalar 0.0)
          init {:key (rng/fresh-key 42) :choices cm/EMPTY :score orig-score :executor nil}
          [_ new-state] (handler/simulate-transition init :x (dist/gaussian 0 1))]
      (mx/eval! orig-score)
      (is (= 0.0 (mx/item orig-score))
          "original score unchanged")
      (mx/eval! (:score new-state))
      (is (not= 0.0 (mx/item (:score new-state)))
          "new state has different score"))))

(deftest update-does-not-mutate-discard
  (testing "update-transition creates new discard, doesn't modify input"
    (let [old-choices (cm/choicemap :x (mx/scalar 1.0))
          new-constraints (cm/choicemap :x (mx/scalar 2.0))
          orig-discard cm/EMPTY
          init {:key (rng/fresh-key 42) :choices cm/EMPTY :score (mx/scalar 0.0)
                :weight (mx/scalar 0.0) :constraints new-constraints
                :old-choices old-choices :discard orig-discard :executor nil}
          [_ new-state] (handler/update-transition init :x (dist/gaussian 0 1))]
      (is (= cm/EMPTY orig-discard)
          "original discard unchanged"))))

;; ---------------------------------------------------------------------------
;; 3.1 Purity: assess and generate agree
;; ---------------------------------------------------------------------------

(deftest assess-equals-generate-for-full-constraints
  (testing "assess weight = generate score when fully constrained"
    (let [choices (cm/choicemap :x (mx/scalar 0.5))
          {:keys [weight]} (p/assess single-gaussian [] choices)
          {:keys [trace]} (p/generate single-gaussian [] choices)]
      (is (h/close? (h/realize weight) (h/realize (:score trace)) 1e-6)
          "assess weight = generate score"))))

;; ---------------------------------------------------------------------------
;; 3.1 Purity: project is deterministic
;; ---------------------------------------------------------------------------

(deftest project-is-deterministic
  (testing "project produces same result for same trace"
    (let [constraints (cm/choicemap :x (mx/scalar 1.5))
          {:keys [trace]} (p/generate single-gaussian [] constraints)
          p1 (h/realize (p/project single-gaussian trace sel/all))
          p2 (h/realize (p/project single-gaussian trace sel/all))]
      (is (h/close? p1 p2 1e-10)
          "project is deterministic for same trace"))))

;; ==========================================================================
;; Run tests
;; ==========================================================================

(cljs.test/run-tests)
