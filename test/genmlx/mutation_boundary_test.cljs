(ns genmlx.mutation-boundary-test
  "P1-7 M2: Property tests verifying that pure layers have no hidden mutation.
   Tests that the mutation boundary (volatile! in runtime.cljs) is properly
   scoped and that all layers above it are referentially transparent."
  (:require [cljs.test :as t :refer [deftest is testing]]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.handler :as hdlr]
            [genmlx.runtime :as rt]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.protocols :as p]
            [genmlx.gfi :as gfi]
            [genmlx.test-helpers :as th])
  (:require-macros [genmlx.gen :refer [gen]]))

;; =========================================================================
;; Test models — no auto-key, so we control PRNG deterministically
;; =========================================================================

(def simple-model
  (dyn/make-gen-fn
    (fn [rt x]
      (let [trace (.-trace rt)
            slope (trace :slope (dist/gaussian 0 10))
            intercept (trace :intercept (dist/gaussian 0 5))]
        (mx/add (mx/multiply slope (mx/scalar x)) intercept)))
    '([x]
      (let [slope (trace :slope (dist/gaussian 0 10))
            intercept (trace :intercept (dist/gaussian 0 5))]
        (mx/add (mx/multiply slope (mx/scalar x)) intercept)))))

(def sub-model
  (dyn/make-gen-fn
    (fn [rt]
      (let [trace (.-trace rt)]
        (trace :sub-x (dist/gaussian 0 1))))
    '([] (trace :sub-x (dist/gaussian 0 1)))))

(def splicing-model
  (dyn/make-gen-fn
    (fn [rt]
      (let [trace (.-trace rt)
            splice (.-splice rt)
            x (trace :x (dist/gaussian 0 1))
            sub-result (splice :sub sub-model [])]
        (mx/add x sub-result)))
    '([]
      (let [x (trace :x (dist/gaussian 0 1))
            sub-result (splice :sub sub-model [])]
        (mx/add x sub-result)))))

;; =========================================================================
;; Test 1: Determinism — same key produces identical traces
;; =========================================================================

(deftest determinism-with-fixed-key
  (testing "simulate with same key produces identical retval"
    (let [key (rng/fresh-key 42)
          m1 (dyn/with-key simple-model key)
          m2 (dyn/with-key simple-model key)
          t1 (p/simulate m1 [2.0])
          t2 (p/simulate m2 [2.0])
          r1 (th/realize (:retval t1))
          r2 (th/realize (:retval t2))]
      (is (th/close? r1 r2 1e-10)
          "same key must produce identical retval")))

  (testing "simulate with same key produces identical scores"
    (let [key (rng/fresh-key 42)
          t1 (p/simulate (dyn/with-key simple-model key) [2.0])
          t2 (p/simulate (dyn/with-key simple-model key) [2.0])
          s1 (th/realize (:score t1))
          s2 (th/realize (:score t2))]
      (is (th/close? s1 s2 1e-10)
          "same key must produce identical score")))

  (testing "simulate with same key produces identical choices"
    (let [key (rng/fresh-key 42)
          t1 (p/simulate (dyn/with-key simple-model key) [2.0])
          t2 (p/simulate (dyn/with-key simple-model key) [2.0])
          slope1 (th/realize (cm/get-value (cm/get-submap (:choices t1) :slope)))
          slope2 (th/realize (cm/get-value (cm/get-submap (:choices t2) :slope)))
          int1 (th/realize (cm/get-value (cm/get-submap (:choices t1) :intercept)))
          int2 (th/realize (cm/get-value (cm/get-submap (:choices t2) :intercept)))]
      (is (th/close? slope1 slope2 1e-10)
          "same key must produce identical :slope choice")
      (is (th/close? int1 int2 1e-10)
          "same key must produce identical :intercept choice")))

  (testing "different keys produce different traces"
    (let [t1 (p/simulate (dyn/with-key simple-model (rng/fresh-key 1)) [2.0])
          t2 (p/simulate (dyn/with-key simple-model (rng/fresh-key 2)) [2.0])
          r1 (th/realize (:retval t1))
          r2 (th/realize (:retval t2))]
      (is (not (th/close? r1 r2 1e-10))
          "different keys should produce different retvals"))))

;; =========================================================================
;; Test 2: Independence — no state leakage between calls
;; =========================================================================

(deftest no-state-leakage-between-calls
  (testing "interleaved model calls don't affect determinism"
    (let [key-a (rng/fresh-key 100)
          ;; First: simulate model A
          t-a1 (p/simulate (dyn/with-key simple-model key-a) [3.0])
          r-a1 (th/realize (:retval t-a1))
          ;; Interleave: simulate a different model many times with random keys
          _ (dotimes [_ 20]
              (p/simulate (dyn/auto-key sub-model) []))
          ;; Second: simulate model A again with same key
          t-a2 (p/simulate (dyn/with-key simple-model key-a) [3.0])
          r-a2 (th/realize (:retval t-a2))]
      (is (th/close? r-a1 r-a2 1e-10)
          "interleaved model calls must not affect results"))))

;; =========================================================================
;; Test 3: Handler transitions are pure functions
;; =========================================================================

(deftest handler-transitions-are-pure
  (testing "simulate transition is deterministic"
    (let [key (rng/fresh-key 300)
          d (dist/gaussian 0 1)
          make-state (fn []
                       {:key key :choices (cm/choicemap) :score (mx/scalar 0.0)
                        :executor nil})
          [v1 s1] (hdlr/simulate-transition (make-state) :x d)
          [v2 s2] (hdlr/simulate-transition (make-state) :x d)]
      (is (th/close? (th/realize v1) (th/realize v2) 1e-10)
          "same inputs must produce same sample value")
      (is (th/close? (th/realize (:score s1)) (th/realize (:score s2)) 1e-10)
          "same inputs must produce same score")))

  (testing "generate transition with constraint is deterministic"
    (let [key (rng/fresh-key 301)
          d (dist/gaussian 0 1)
          obs (cm/choicemap :x (mx/scalar 1.5))
          make-state (fn []
                       {:key key :choices (cm/choicemap) :score (mx/scalar 0.0)
                        :weight (mx/scalar 0.0) :constraints obs :executor nil})
          [v1 s1] (hdlr/generate-transition (make-state) :x d)
          [v2 s2] (hdlr/generate-transition (make-state) :x d)]
      (is (th/close? (th/realize v1) (th/realize v2) 1e-10)
          "constrained generate must return same value")
      (is (th/close? (th/realize (:weight s1)) (th/realize (:weight s2)) 1e-10)
          "constrained generate must produce same weight"))))

;; =========================================================================
;; Test 4: Gen-fn record is not mutated by GFI operations
;; =========================================================================

(deftest gfi-ops-dont-mutate-gen-fn
  (testing "simulate does not mutate gen-fn record fields"
    (let [m (dyn/with-key simple-model (rng/fresh-key 400))
          body-before (:body-fn m)
          source-before (:source m)
          schema-before (:schema m)
          _ (p/simulate m [1.0])
          _ (p/simulate m [2.0])
          _ (p/simulate m [3.0])]
      (is (identical? body-before (:body-fn m))
          "body-fn must be the same object")
      (is (= source-before (:source m))
          "source must not change")
      (is (= schema-before (:schema m))
          "schema must not change"))))

;; =========================================================================
;; Test 5: Auto-cleanup counters don't affect results
;; =========================================================================

(deftest cleanup-heuristics-dont-affect-results
  (testing "many GFI ops triggering cleanup still produce deterministic results"
    (let [key (rng/fresh-key 500)
          ;; Run enough ops to trigger both auto-cleanup! and gfi-cleanup!
          _ (dotimes [_ 100]
              (p/simulate (dyn/auto-key sub-model) []))
          ;; Now run with our fixed key — should be unaffected by cleanup state
          t1 (p/simulate (dyn/with-key simple-model key) [2.0])
          r1 (th/realize (:retval t1))
          ;; Run more ops to trigger cleanup again
          _ (dotimes [_ 100]
              (p/simulate (dyn/auto-key sub-model) []))
          ;; Same key again
          t2 (p/simulate (dyn/with-key simple-model key) [2.0])
          r2 (th/realize (:retval t2))]
      (is (th/close? r1 r2 1e-10)
          "cleanup heuristic state must not affect simulation results"))))

;; =========================================================================
;; Test 6: PRNG key splitting is purely functional
;; =========================================================================

(deftest prng-key-splitting-is-pure
  (testing "split is deterministic"
    (let [key (rng/fresh-key 600)
          [k1a k2a] (rng/split key)
          [k1b k2b] (rng/split key)
          v1a (mx/->clj k1a)
          v1b (mx/->clj k1b)
          v2a (mx/->clj k2a)
          v2b (mx/->clj k2b)]
      (is (= v1a v1b)
          "splitting same key must produce same first sub-key")
      (is (= v2a v2b)
          "splitting same key must produce same second sub-key")))

  (testing "split produces different sub-keys"
    (let [key (rng/fresh-key 601)
          [k1 k2] (rng/split key)]
      (is (not= (mx/->clj k1) (mx/->clj k2))
          "split sub-keys must differ from each other")))

  (testing "splitting does not mutate the original key"
    (let [key (rng/fresh-key 602)
          val-before (mx/->clj key)
          _ (rng/split key)
          _ (rng/split key)
          _ (rng/split-n key 10)
          val-after (mx/->clj key)]
      (is (= val-before val-after)
          "original key must not be mutated by splitting"))))

;; =========================================================================
;; Test 7: Splicing maintains isolation
;; =========================================================================

(deftest splice-maintains-isolation
  (testing "splicing model is deterministic with fixed key"
    (let [key (rng/fresh-key 700)
          t1 (p/simulate (dyn/with-key splicing-model key) [])
          t2 (p/simulate (dyn/with-key splicing-model key) [])
          r1 (th/realize (:retval t1))
          r2 (th/realize (:retval t2))]
      (is (th/close? r1 r2 1e-10)
          "splicing model must be deterministic")))

  (testing "splice choices are properly nested"
    (let [key (rng/fresh-key 701)
          t (p/simulate (dyn/with-key splicing-model key) [])
          choices (:choices t)
          parent-x (cm/get-submap choices :x)
          sub-choices (cm/get-submap choices :sub)]
      (is (cm/has-value? parent-x)
          "parent choice :x must exist")
      (is (some? sub-choices)
          "sub-model namespace :sub must exist"))))

;; =========================================================================
;; Test 8: Remaining handler transitions are pure (update, regenerate,
;;         assess, project)
;; =========================================================================

(deftest update-transition-is-pure
  (testing "update transition with new constraint is deterministic"
    (let [d (dist/gaussian 0 1)
          key (rng/fresh-key 800)
          make-state (fn []
                       {:key key :choices (cm/choicemap) :score (mx/scalar 0.0)
                        :weight (mx/scalar 0.0)
                        :constraints (cm/choicemap :x (mx/scalar 2.0))
                        :old-choices (cm/choicemap :x (mx/scalar 1.0))
                        :discard (cm/choicemap)
                        :executor nil})
          [v1 s1] (hdlr/update-transition (make-state) :x d)
          [v2 s2] (hdlr/update-transition (make-state) :x d)]
      (is (th/close? (th/realize v1) (th/realize v2) 1e-10)
          "update with constraint must return same value")
      (is (th/close? (th/realize (:weight s1)) (th/realize (:weight s2)) 1e-10)
          "update must produce same weight"))))

(deftest assess-transition-is-pure
  (testing "assess transition is deterministic (no randomness)"
    (let [d (dist/gaussian 0 1)
          make-state (fn []
                       {:choices (cm/choicemap) :score (mx/scalar 0.0)
                        :weight (mx/scalar 0.0)
                        :constraints (cm/choicemap :x (mx/scalar 1.5))
                        :executor nil})
          [v1 s1] (hdlr/assess-transition (make-state) :x d)
          [v2 s2] (hdlr/assess-transition (make-state) :x d)]
      (is (th/close? (th/realize v1) (th/realize v2) 1e-10)
          "assess must return same constrained value")
      (is (th/close? (th/realize (:weight s1)) (th/realize (:weight s2)) 1e-10)
          "assess must produce same weight"))))

(deftest regenerate-transition-is-pure
  (testing "regenerate transition with selection is deterministic"
    (let [d (dist/gaussian 0 1)
          key (rng/fresh-key 802)
          make-state (fn []
                       {:key key :choices (cm/choicemap) :score (mx/scalar 0.0)
                        :weight (mx/scalar 0.0)
                        :old-choices (cm/choicemap :x (mx/scalar 1.0))
                        :selection (sel/select :x)
                        :executor nil})
          [v1 s1] (hdlr/regenerate-transition (make-state) :x d)
          [v2 s2] (hdlr/regenerate-transition (make-state) :x d)]
      (is (th/close? (th/realize v1) (th/realize v2) 1e-10)
          "regenerate must produce same resampled value")
      (is (th/close? (th/realize (:weight s1)) (th/realize (:weight s2)) 1e-10)
          "regenerate must produce same weight"))))

(deftest project-transition-is-pure
  (testing "project transition replays deterministically"
    (let [d (dist/gaussian 0 1)
          make-state (fn []
                       {:choices (cm/choicemap) :score (mx/scalar 0.0)
                        :weight (mx/scalar 0.0)
                        :old-choices (cm/choicemap :x (mx/scalar 1.0))
                        :selection (sel/select :x)
                        :executor nil})
          [v1 s1] (hdlr/project-transition (make-state) :x d)
          [v2 s2] (hdlr/project-transition (make-state) :x d)]
      (is (th/close? (th/realize v1) (th/realize v2) 1e-10)
          "project must replay same value")
      (is (th/close? (th/realize (:weight s1)) (th/realize (:weight s2)) 1e-10)
          "project must produce same weight"))))

;; =========================================================================
;; Test 9: Compiled paths produce identical traces to handler paths
;; =========================================================================

(def static-model
  (dyn/make-gen-fn
    (fn [rt]
      (let [trace (.-trace rt)
            x (trace :x (dist/gaussian 0 1))]
        (trace :y (dist/gaussian x 0.5))))
    '([]
      (let [x (trace :x (dist/gaussian 0 1))]
        (trace :y (dist/gaussian x 0.5))))))

(deftest compiled-path-equals-handler-path
  (testing "compiled simulate score equals handler assess weight"
    (let [compiled? (some? (:compiled-simulate (:schema static-model)))]
      (when compiled?
        (let [key (rng/fresh-key 900)
              ;; Simulate via compiled path
              compiled-trace (p/simulate (dyn/with-key static-model key) [])
              compiled-score (th/realize (:score compiled-trace))
              compiled-choices (:choices compiled-trace)
              ;; Assess same choices via handler path (strip-compiled)
              handler-model (dyn/with-key (gfi/strip-compiled static-model) (rng/fresh-key 901))
              {:keys [weight]} (p/assess handler-model [] compiled-choices)
              handler-score (th/realize weight)]
          (is (th/finite? compiled-score)
              "compiled score must be finite")
          (is (th/finite? handler-score)
              "handler score must be finite")
          (is (th/close? compiled-score handler-score 1e-4)
              "compiled simulate score must equal handler assess weight"))))))

;; =========================================================================
;; Run
;; =========================================================================

(t/run-tests)
