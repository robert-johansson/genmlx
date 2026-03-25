(ns genmlx.error-message-test
  "Tests for helpful error messages (11.2)."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.selection :as sel]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; 1. Distribution parameter validation
;; ---------------------------------------------------------------------------

(deftest distribution-parameter-validation
  (testing "gaussian: negative sigma throws"
    (is (thrown-with-msg? js/Error #"sigma must be positive"
          (dist/gaussian 0 -1))
        "gaussian: negative sigma throws"))

  (testing "gaussian: zero sigma throws"
    (is (thrown-with-msg? js/Error #"sigma must be positive"
          (dist/gaussian 0 0))
        "gaussian: zero sigma throws"))

  (testing "uniform: lo >= hi throws"
    (is (thrown-with-msg? js/Error #"lo must be less than hi"
          (dist/uniform 5 2))
        "uniform: lo >= hi throws"))

  (testing "uniform: lo == hi throws"
    (is (thrown-with-msg? js/Error #"lo must be less than hi"
          (dist/uniform 3 3))
        "uniform: lo == hi throws"))

  (testing "beta-dist: negative alpha throws"
    (is (thrown-with-msg? js/Error #"alpha must be positive"
          (dist/beta-dist -1 2))
        "beta-dist: negative alpha throws"))

  (testing "beta-dist: negative beta throws"
    (is (thrown-with-msg? js/Error #"beta must be positive"
          (dist/beta-dist 2 -1))
        "beta-dist: negative beta throws"))

  (testing "gamma-dist: negative shape throws"
    (is (thrown-with-msg? js/Error #"shape must be positive"
          (dist/gamma-dist -1 1))
        "gamma-dist: negative shape throws"))

  (testing "gamma-dist: negative rate throws"
    (is (thrown-with-msg? js/Error #"rate must be positive"
          (dist/gamma-dist 2 -1))
        "gamma-dist: negative rate throws"))

  (testing "exponential: negative rate throws"
    (is (thrown-with-msg? js/Error #"rate must be positive"
          (dist/exponential -1))
        "exponential: negative rate throws"))

  (testing "valid distributions construct without error"
    (let [g (dist/gaussian 0 1)
          u (dist/uniform 0 1)
          b (dist/beta-dist 2 3)
          gm (dist/gamma-dist 2 1)
          e (dist/exponential 2)]
      (is (and g u b gm e) "valid distributions construct without error")))

  (testing "MLX array params accepted"
    (let [g (dist/gaussian 0 (mx/scalar 1))]
      (is (some? g) "MLX array params accepted (no eager check)"))))

;; ---------------------------------------------------------------------------
;; 2. ChoiceMap "Not a leaf" error with sub-addresses
;; ---------------------------------------------------------------------------

(deftest choicemap-leaf-error
  (testing "get-value on Node shows sub-addresses"
    (let [cm-node (cm/choicemap :x 1.0 :y 2.0 :z 3.0)]
      (is (thrown? js/Error (cm/-get-value cm-node))
          "get-value on Node throws"))))

;; ---------------------------------------------------------------------------
;; 3. Unused constraint detection in generate/update
;; ---------------------------------------------------------------------------

(def simple-model
  (dyn/auto-key (gen [x]
    (let [slope (trace :slope (dist/gaussian 0 10))]
      (trace :obs (dist/gaussian (mx/multiply slope (mx/scalar x)) 1))
      slope))))

(deftest unused-constraint-detection
  (testing "generate detects unused constraint :typo"
    (let [result (p/generate simple-model [2.0]
                   (cm/choicemap :typo 5.0 :obs 3.0))]
      (is (some? (:unused-constraints result))
          "generate returns unused-constraints")
      (is (contains? (:unused-constraints result) :typo)
          "unused-constraints includes :typo")))

  (testing "generate with correct constraints has no unused"
    (let [result (p/generate simple-model [2.0]
                   (cm/choicemap :obs 3.0))]
      (is (nil? (:unused-constraints result))
          "generate with correct constraints has no unused")))

  (testing "update with wrong-addr completes without error"
    (let [trace (:trace (p/generate simple-model [2.0]
                          (cm/choicemap :obs 3.0)))
          result (p/update simple-model trace
                   (cm/choicemap :wrong-addr 99.0))]
      (is (some? (:trace result))
          "update with unknown address completes"))))

;; ---------------------------------------------------------------------------
;; 4. Batched eval!/item -- model still runs
;; ---------------------------------------------------------------------------

(def eval-in-model
  (dyn/auto-key (gen []
    (let [x (trace :x (dist/gaussian 0 1))]
      (mx/eval! x)
      x))))

(deftest batched-eval-runs
  (testing "vsimulate with eval! in model body completes"
    (let [key (rng/fresh-key)
          vt (dyn/vsimulate eval-in-model [] 10 key)]
      (is (some? vt) "vsimulate completes despite eval! in body")))

  (testing "scalar simulate works normally"
    (let [tr (p/simulate eval-in-model [])]
      (is (some? (:choices tr)) "scalar simulate works"))))

;; ---------------------------------------------------------------------------
;; 5. Regenerate with non-existent address
;; ---------------------------------------------------------------------------

(def regen-model
  (dyn/auto-key (gen []
    (trace :x (dist/gaussian 0 1)))))

(deftest regenerate-nil-error
  (testing "regenerate with missing old choices throws"
    (let [empty-trace (tr/make-trace {:gen-fn regen-model :args []
                                       :choices cm/EMPTY
                                       :retval nil :score (mx/scalar 0.0)})]
      (is (thrown? js/Error
            (p/regenerate regen-model empty-trace (sel/select :other)))
          "regenerate with missing old choices throws"))))

;; ---------------------------------------------------------------------------
;; 6. Batched log-prob for discrete distributions (7.9)
;; ---------------------------------------------------------------------------

(def poisson-model
  (gen []
    (trace :x (dist/poisson (mx/scalar 3.0)))))

(def neg-binom-model
  (gen []
    (trace :x (dist/neg-binomial (mx/scalar 5.0) (mx/scalar 0.4)))))

(def binomial-model
  (gen []
    (trace :x (dist/binomial (mx/scalar 10) (mx/scalar 0.3)))))

(def piecewise-model
  (gen []
    (trace :x (dist/piecewise-uniform
                    (mx/array [0.0 1.0 2.0 3.0])
                    (mx/array [1.0 2.0 1.0])))))

(deftest batched-discrete-log-prob
  (testing "vsimulate poisson produces [5]-shaped output"
    (let [key (rng/fresh-key)
          vt (dyn/vsimulate poisson-model [] 5 key)
          choices (:choices vt)
          x-vals (cm/get-value (cm/get-submap choices :x))]
      (is (= [5] (mx/shape x-vals))
          "vsimulate poisson produces [5]-shaped output")))

  (testing "vsimulate neg-binomial produces [5]-shaped output"
    (let [key (rng/fresh-key)
          vt (dyn/vsimulate neg-binom-model [] 5 key)
          choices (:choices vt)
          x-vals (cm/get-value (cm/get-submap choices :x))]
      (is (= [5] (mx/shape x-vals))
          "vsimulate neg-binomial produces [5]-shaped output")))

  (testing "vsimulate binomial produces [5]-shaped output"
    (let [key (rng/fresh-key)
          vt (dyn/vsimulate binomial-model [] 5 key)
          choices (:choices vt)
          x-vals (cm/get-value (cm/get-submap choices :x))]
      (is (= [5] (mx/shape x-vals))
          "vsimulate binomial produces [5]-shaped output")))

  (testing "vsimulate piecewise-uniform produces [5]-shaped output"
    (let [key (rng/fresh-key)
          vt (dyn/vsimulate piecewise-model [] 5 key)
          choices (:choices vt)
          x-vals (cm/get-value (cm/get-submap choices :x))]
      (is (= [5] (mx/shape x-vals))
          "vsimulate piecewise-uniform produces [5]-shaped output"))))

(cljs.test/run-tests)
