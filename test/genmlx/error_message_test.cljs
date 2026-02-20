(ns genmlx.error-message-test
  "Tests for helpful error messages (11.2)."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.selection :as sel]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-throws [msg f expected-substring]
  (try
    (f)
    (println "  FAIL:" msg "- expected exception but none thrown")
    (catch :default e
      (let [emsg (ex-message e)]
        (if (and emsg (.includes emsg expected-substring))
          (println "  PASS:" msg)
          (do (println "  FAIL:" msg)
              (println "    expected substring:" expected-substring)
              (println "    actual message:    " emsg)))))))

(defn with-warn-capture
  "Execute f with console.warn output captured via stderr interception. Returns atom of strings."
  [f]
  (let [captured (atom [])
        orig-write (.-write (.-stderr js/process))]
    (set! (.-write (.-stderr js/process))
      (fn [chunk & args]
        (swap! captured conj (str chunk))
        (.apply orig-write (.-stderr js/process) (into-array (cons chunk args)))))
    (try
      (f)
      (finally
        (set! (.-write (.-stderr js/process)) orig-write)))
    captured))

(println "\n=== Error Message Tests (11.2) ===")

;; ---------------------------------------------------------------------------
;; 1. Distribution parameter validation
;; ---------------------------------------------------------------------------

(println "\n-- Distribution parameter validation --")

(assert-throws "gaussian: negative sigma throws"
  #(dist/gaussian 0 -1)
  "sigma must be positive")

(assert-throws "gaussian: zero sigma throws"
  #(dist/gaussian 0 0)
  "sigma must be positive")

(assert-throws "uniform: lo >= hi throws"
  #(dist/uniform 5 2)
  "lo must be less than hi")

(assert-throws "uniform: lo == hi throws"
  #(dist/uniform 3 3)
  "lo must be less than hi")

(assert-throws "beta-dist: negative alpha throws"
  #(dist/beta-dist -1 2)
  "alpha must be positive")

(assert-throws "beta-dist: negative beta throws"
  #(dist/beta-dist 2 -1)
  "beta must be positive")

(assert-throws "gamma-dist: negative shape throws"
  #(dist/gamma-dist -1 1)
  "shape must be positive")

(assert-throws "gamma-dist: negative rate throws"
  #(dist/gamma-dist 2 -1)
  "rate must be positive")

(assert-throws "exponential: negative rate throws"
  #(dist/exponential -1)
  "rate must be positive")

;; Valid constructions should not throw
(let [g (dist/gaussian 0 1)
      u (dist/uniform 0 1)
      b (dist/beta-dist 2 3)
      gm (dist/gamma-dist 2 1)
      e (dist/exponential 2)]
  (assert-true "valid distributions construct without error"
    (and g u b gm e)))

;; MLX arrays bypass validation (can't check lazily)
(let [g (dist/gaussian 0 (mx/scalar 1))]
  (assert-true "MLX array params accepted (no eager check)" (some? g)))

;; ---------------------------------------------------------------------------
;; 2. ChoiceMap "Not a leaf" error with sub-addresses
;; ---------------------------------------------------------------------------

(println "\n-- ChoiceMap leaf error --")

(let [cm-node (cm/choicemap :x 1.0 :y 2.0 :z 3.0)]
  (assert-throws "get-value on Node shows sub-addresses"
    #(cm/-get-value cm-node)
    "Available sub-addresses"))

;; ---------------------------------------------------------------------------
;; 3. Unused constraint warnings in generate
;; ---------------------------------------------------------------------------

(println "\n-- Unused constraint warnings --")

(def simple-model
  (gen [x]
    (let [slope (dyn/trace :slope (dist/gaussian 0 10))]
      (dyn/trace :obs (dist/gaussian (mx/multiply slope (mx/scalar x)) 1))
      slope)))

;; Generate with a typo constraint
(let [warnings (with-warn-capture
                 #(p/generate simple-model [2.0]
                    (cm/choicemap :typo 5.0 :obs 3.0)))]
  (assert-true "generate warns about unused constraint :typo"
    (some #(.includes % ":typo") @warnings))
  (assert-true "warning mentions trace addresses"
    (some #(.includes % "Trace addresses") @warnings)))

;; Generate with correct constraints should NOT warn
(let [warnings (with-warn-capture
                 #(p/generate simple-model [2.0]
                    (cm/choicemap :obs 3.0)))]
  (assert-true "generate with correct constraints does not warn"
    (empty? @warnings)))

;; Update with unused constraints
(let [trace (:trace (p/generate simple-model [2.0]
                      (cm/choicemap :obs 3.0)))
      warnings (with-warn-capture
                 #(p/update simple-model trace
                    (cm/choicemap :wrong-addr 99.0)))]
  (assert-true "update warns about unused constraint :wrong-addr"
    (some #(.includes % ":wrong-addr") @warnings)))

;; ---------------------------------------------------------------------------
;; 4. Batched eval!/item warning
;; ---------------------------------------------------------------------------

(println "\n-- Batched eval!/item warning --")

(def eval-in-model
  (gen []
    (let [x (dyn/trace :x (dist/gaussian 0 1))]
      (mx/eval! x)
      x)))

(let [key (rng/fresh-key)
      warnings (with-warn-capture
                 #(dyn/vsimulate eval-in-model [] 10 key))]
  (assert-true "vsimulate warns about mx/eval! in batched mode"
    (some #(.includes % "batched execution") @warnings)))

;; Same model in scalar mode should NOT warn
(let [warnings (with-warn-capture
                 #(p/simulate eval-in-model []))]
  (assert-true "scalar simulate does not warn about mx/eval!"
    (empty? @warnings)))

;; ---------------------------------------------------------------------------
;; 5. Regenerate with non-existent address
;; ---------------------------------------------------------------------------

(println "\n-- Regenerate nil error --")

(def regen-model
  (gen []
    (dyn/trace :x (dist/gaussian 0 1))))

;; Create a trace with empty choices, then regenerate
;; The model visits :x, which is NOT selected and NOT in old-choices → nil error
(let [empty-trace (tr/make-trace {:gen-fn regen-model :args []
                                   :choices cm/EMPTY
                                   :retval nil :score (mx/scalar 0.0)})]
  (assert-throws "regenerate with missing old choices throws"
    #(p/regenerate regen-model empty-trace (sel/select :other))
    "not found in previous trace"))

;; ---------------------------------------------------------------------------
;; 6. Batched log-prob for discrete distributions (7.9)
;; ---------------------------------------------------------------------------

(println "\n-- Batched discrete log-prob (7.9) --")

(def poisson-model
  (gen []
    (dyn/trace :x (dist/poisson (mx/scalar 3.0)))))

(let [key (rng/fresh-key)
      vt (dyn/vsimulate poisson-model [] 5 key)
      choices (:choices vt)
      x-vals (cm/get-value (cm/get-submap choices :x))]
  (assert-true "vsimulate poisson produces [5]-shaped output"
    (= [5] (mx/shape x-vals))))

(def neg-binom-model
  (gen []
    (dyn/trace :x (dist/neg-binomial (mx/scalar 5.0) (mx/scalar 0.4)))))

(let [key (rng/fresh-key)
      vt (dyn/vsimulate neg-binom-model [] 5 key)
      choices (:choices vt)
      x-vals (cm/get-value (cm/get-submap choices :x))]
  (assert-true "vsimulate neg-binomial produces [5]-shaped output"
    (= [5] (mx/shape x-vals))))

(def binomial-model
  (gen []
    (dyn/trace :x (dist/binomial (mx/scalar 10) (mx/scalar 0.3)))))

(let [key (rng/fresh-key)
      vt (dyn/vsimulate binomial-model [] 5 key)
      choices (:choices vt)
      x-vals (cm/get-value (cm/get-submap choices :x))]
  (assert-true "vsimulate binomial produces [5]-shaped output"
    (= [5] (mx/shape x-vals))))

(def piecewise-model
  (gen []
    (dyn/trace :x (dist/piecewise-uniform
                    (mx/array [0.0 1.0 2.0 3.0])
                    (mx/array [1.0 2.0 1.0])))))

(let [key (rng/fresh-key)
      vt (dyn/vsimulate piecewise-model [] 5 key)
      choices (:choices vt)
      x-vals (cm/get-value (cm/get-submap choices :x))]
  (assert-true "vsimulate piecewise-uniform produces [5]-shaped output"
    (= [5] (mx/shape x-vals))))

(println "\n=== All error message tests complete ===")
