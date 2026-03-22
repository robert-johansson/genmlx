(ns genmlx.branching-property-test
  "Property-based tests for branching models using test.check.
   Tests the measure-theoretic handling of changing support -- uniquely PP.

   Models with if/cond visit different trace sites per execution.
   This is common in real probabilistic programming (mixture models,
   model selection). These tests verify GFI invariants hold when
   the set of random addresses changes across executions."
  (:require [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [cljs.test :as t]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm])
  (:require-macros [genmlx.gen :refer [gen]]
                   [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- choice-val
  "Extract a JS number from a choicemap at addr, or nil."
  [choices addr]
  (let [sub (cm/get-submap choices addr)]
    (when (and sub (cm/has-value? sub))
      (let [v (cm/get-value sub)]
        (mx/eval! v)
        (mx/item v)))))

(defn- trace-score [trace]
  (mx/eval! (:score trace))
  (mx/item (:score trace)))

(defn- eval-weight [w]
  (mx/eval! w)
  (mx/item w))

(defn- finite? [x]
  (and (number? x) (js/isFinite x)))

(defn- close? [a b tol]
  (and (finite? a) (finite? b) (<= (js/Math.abs (- a b)) tol)))

;; ---------------------------------------------------------------------------
;; Branching model
;; ---------------------------------------------------------------------------
;; A model that branches on a discrete choice. Each execution visits either
;; :heads or :tails, never both. This is the canonical test case for
;; variable-support probabilistic programs.

(def branch-model
  (dyn/auto-key
    (gen []
      (let [coin (trace :coin (dist/bernoulli 0.5))]
        (if (> (mx/item coin) 0.5)
          (trace :heads (dist/gaussian 10 1))
          (trace :tails (dist/gaussian -10 1)))))))

;; ---------------------------------------------------------------------------
;; Generators (SCI-safe: gen/elements from pre-built pools)
;; ---------------------------------------------------------------------------

(def key-pool (mapv #(rng/fresh-key %) [1 2 3 7 13 42 99 123 200 255]))
(def gen-key (gen/elements key-pool))

;; ---------------------------------------------------------------------------
;; BR1. branching: simulate produces valid trace
;; ---------------------------------------------------------------------------
;; Law: branching models produce valid traces with consistent address sets.
;; Either :heads or :tails is present (never both), :coin is always present,
;; and the score is finite. This tests that the GFI correctly handles
;; data-dependent control flow where the set of random choices varies.

(defspec branching-simulate-score-is-finite 50
  (prop/for-all [_k gen-key]
    (let [tr (p/simulate branch-model [])
          s (trace-score tr)]
      (finite? s))))

(defspec branching-simulate-has-coin 50
  (prop/for-all [_k gen-key]
    (let [tr (p/simulate branch-model [])
          choices (:choices tr)]
      (some? (choice-val choices :coin)))))

(defspec branching-simulate-has-exactly-one-branch-address 50
  (prop/for-all [_k gen-key]
    (let [tr (p/simulate branch-model [])
          choices (:choices tr)
          has-heads (some? (choice-val choices :heads))
          has-tails (some? (choice-val choices :tails))]
      ;; Exclusive or: exactly one branch present
      (and (or has-heads has-tails)
           (not (and has-heads has-tails))))))

(defspec branching-coin-value-consistent-with-branch-taken 50
  (prop/for-all [_k gen-key]
    (let [tr (p/simulate branch-model [])
          choices (:choices tr)
          coin (choice-val choices :coin)
          has-heads (some? (choice-val choices :heads))
          has-tails (some? (choice-val choices :tails))]
      (if (> coin 0.5)
        has-heads
        has-tails))))

;; ---------------------------------------------------------------------------
;; BR2. branching: generate(full) weight = score
;; ---------------------------------------------------------------------------
;; Law: GFI importance weighting identity for branching models.
;; When all addresses are constrained, generate weight equals the trace score.
;; This must hold regardless of which branch the constraints specify --
;; the model must correctly route to the matching branch.

(defspec branching-generate-full-weight-equals-score 50
  (prop/for-all [_k gen-key]
    (let [;; Simulate to get a valid set of choices for one branch
          sim-tr (p/simulate branch-model [])
          sim-choices (:choices sim-tr)
          ;; Generate with those exact choices
          {:keys [trace weight]} (p/generate branch-model [] sim-choices)
          w (eval-weight weight)
          s (trace-score trace)]
      (close? w s 0.01))))

;; ---------------------------------------------------------------------------
;; BR3. branching: update preserves branch when condition unchanged
;; ---------------------------------------------------------------------------
;; Law: update identity holds when branch condition is unchanged.
;; Constraining all addresses to their current values yields weight ~ 0
;; and preserves the same branch address set.

(defspec branching-update-same-weight-equals-0 50
  (prop/for-all [_k gen-key]
    (let [tr (p/simulate branch-model [])
          {:keys [weight]} (p/update branch-model tr (:choices tr))
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(defspec branching-update-same-preserves-branch-address 50
  (prop/for-all [_k gen-key]
    (let [tr (p/simulate branch-model [])
          choices (:choices tr)
          orig-has-heads (some? (choice-val choices :heads))
          {:keys [trace]} (p/update branch-model tr choices)
          new-choices (:choices trace)
          new-has-heads (some? (choice-val new-choices :heads))]
      (= orig-has-heads new-has-heads))))

;; ---------------------------------------------------------------------------
;; BR4. branching: update changes branch (address set changes)
;; ---------------------------------------------------------------------------
;; Law: update correctly handles address set changes across branches.
;; When we flip the coin, the model visits the opposite branch. The new trace
;; must contain the new branch's address and NOT the old branch's address.
;; The weight must be finite and equal new_score - old_score.

(defspec branching-update-flip-coin-switches-branch-address 50
  (prop/for-all [_k gen-key]
    (let [tr (p/simulate branch-model [])
          choices (:choices tr)
          coin (choice-val choices :coin)
          orig-has-heads (some? (choice-val choices :heads))
          ;; Flip coin: if it was 1, set to 0; if 0, set to 1
          flipped (if (> coin 0.5) 0.0 1.0)
          constraint (cm/choicemap :coin (mx/scalar flipped))
          {:keys [trace]} (p/update branch-model tr constraint)
          new-choices (:choices trace)
          new-has-heads (some? (choice-val new-choices :heads))
          new-has-tails (some? (choice-val new-choices :tails))]
      ;; Branch must have flipped
      (and (not= orig-has-heads new-has-heads)
           ;; Exactly one branch present in new trace
           (or new-has-heads new-has-tails)
           (not (and new-has-heads new-has-tails))))))

(defspec branching-update-flip-coin-weight-is-finite 50
  (prop/for-all [_k gen-key]
    (let [tr (p/simulate branch-model [])
          coin (choice-val (:choices tr) :coin)
          flipped (if (> coin 0.5) 0.0 1.0)
          constraint (cm/choicemap :coin (mx/scalar flipped))
          {:keys [weight]} (p/update branch-model tr constraint)
          w (eval-weight weight)]
      (finite? w))))

(defspec branching-update-flip-coin-weight-equals-new-score-minus-old-score 50
  (prop/for-all [_k gen-key]
    (let [tr (p/simulate branch-model [])
          old-score (trace-score tr)
          coin (choice-val (:choices tr) :coin)
          flipped (if (> coin 0.5) 0.0 1.0)
          constraint (cm/choicemap :coin (mx/scalar flipped))
          {:keys [trace weight]} (p/update branch-model tr constraint)
          w (eval-weight weight)
          new-score (trace-score trace)]
      ;; Fundamental GFI update weight identity:
      ;; weight = new_score - old_score (for address-changing updates
      ;; where new addresses are freshly sampled)
      (close? w (- new-score old-score) 0.1))))

(defspec branching-update-flip-coin-discard-contains-old-coin 50
  (prop/for-all [_k gen-key]
    (let [tr (p/simulate branch-model [])
          coin (choice-val (:choices tr) :coin)
          flipped (if (> coin 0.5) 0.0 1.0)
          constraint (cm/choicemap :coin (mx/scalar flipped))
          {:keys [discard]} (p/update branch-model tr constraint)
          disc-m (cm/to-map discard)]
      ;; Discard must contain the old :coin value
      (and (contains? disc-m :coin)
           (close? coin (mx/item (get disc-m :coin)) 1e-6)))))

(t/run-tests)
