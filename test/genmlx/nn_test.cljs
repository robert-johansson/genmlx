(ns genmlx.nn-test
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.nn :as nn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn assert-true [msg actual]
  (if actual
    (println "  PASS:" msg)
    (println "  FAIL:" msg "- expected truthy")))

(defn assert-close [msg expected actual tolerance]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tolerance)
      (println "  PASS:" msg)
      (do (println "  FAIL:" msg)
          (println "    expected:" expected "+/-" tolerance)
          (println "    actual:  " actual)))))

(println "\n=== Neural Network Integration Tests ===\n")

;; ---------------------------------------------------------------------------
;; Layer constructors
;; ---------------------------------------------------------------------------

(println "-- Layer constructors --")

(let [lin (nn/linear 3 1)]
  (assert-true "Linear is a Module instance"
               (instance? (.-Module mx/nn-mod) lin))
  (assert-true "Linear has forward method"
               (fn? (.-forward lin))))

(let [seq-mod (nn/sequential [(nn/linear 4 8) (nn/relu) (nn/linear 8 1)])]
  (assert-true "Sequential is a Module instance"
               (instance? (.-Module mx/nn-mod) seq-mod))
  (assert-true "Sequential has forward method"
               (fn? (.-forward seq-mod))))

(let [r (nn/relu)
      g (nn/gelu)
      t (nn/tanh-act)
      s (nn/sigmoid-act)]
  (assert-true "ReLU created" (some? r))
  (assert-true "GELU created" (some? g))
  (assert-true "Tanh created" (some? t))
  (assert-true "Sigmoid created" (some? s)))

(let [ln (nn/layer-norm 8)]
  (assert-true "LayerNorm created"
               (instance? (.-Module mx/nn-mod) ln)))

(let [emb (nn/embedding 100 32)]
  (assert-true "Embedding created"
               (instance? (.-Module mx/nn-mod) emb)))

;; ---------------------------------------------------------------------------
;; Forward pass
;; ---------------------------------------------------------------------------

(println "\n-- Forward pass --")

(let [lin (nn/linear 3 1)
      x (rng/normal (rng/fresh-key) [3])
      y (.forward lin x)]
  (mx/eval! y)
  (assert-true "Linear(3,1) forward produces [1]-shaped output"
               (= [1] (mx/shape y))))

(let [mlp (nn/sequential [(nn/linear 4 8) (nn/relu) (nn/linear 8 2)])
      x (rng/normal (rng/fresh-key) [4])
      y (.forward mlp x)]
  (mx/eval! y)
  (assert-true "Sequential MLP forward produces [2]-shaped output"
               (= [2] (mx/shape y))))

;; Batched input
(let [lin (nn/linear 3 1)
      x (rng/normal (rng/fresh-key) [5 3])
      y (.forward lin x)]
  (mx/eval! y)
  (assert-true "Linear(3,1) batched [5,3] -> [5,1]"
               (= [5 1] (mx/shape y))))

;; ---------------------------------------------------------------------------
;; NeuralNetGF simulate
;; ---------------------------------------------------------------------------

(println "\n-- NeuralNetGF simulate --")

(let [lin (nn/linear 3 1)
      gf (nn/nn->gen-fn lin)
      x (rng/normal (rng/fresh-key) [3])
      trace (p/simulate gf [x])]
  (mx/eval! (:retval trace) (:score trace))
  (assert-true "simulate retval shape = [1]"
               (= [1] (mx/shape (:retval trace))))
  (assert-close "simulate score = 0"
                0.0 (mx/item (:score trace)) 1e-6)
  (assert-true "simulate choices are empty"
               (= (:choices trace) cm/EMPTY)))

;; ---------------------------------------------------------------------------
;; NeuralNetGF generate
;; ---------------------------------------------------------------------------

(println "\n-- NeuralNetGF generate --")

(let [lin (nn/linear 3 1)
      gf (nn/nn->gen-fn lin)
      x (rng/normal (rng/fresh-key) [3])
      {:keys [trace weight]} (p/generate gf [x] cm/EMPTY)]
  (mx/eval! (:retval trace) weight)
  (assert-true "generate retval shape = [1]"
               (= [1] (mx/shape (:retval trace))))
  (assert-close "generate weight = 0"
                0.0 (mx/item weight) 1e-6))

;; ---------------------------------------------------------------------------
;; NeuralNetGF assess / propose
;; ---------------------------------------------------------------------------

(println "\n-- NeuralNetGF assess / propose --")

(let [lin (nn/linear 3 1)
      gf (nn/nn->gen-fn lin)
      x (rng/normal (rng/fresh-key) [3])]

  (let [{:keys [retval weight]} (p/assess gf [x] cm/EMPTY)]
    (mx/eval! retval weight)
    (assert-true "assess retval shape = [1]"
                 (= [1] (mx/shape retval)))
    (assert-close "assess weight = 0"
                  0.0 (mx/item weight) 1e-6))

  (let [{:keys [choices weight retval]} (p/propose gf [x])]
    (mx/eval! retval weight)
    (assert-true "propose retval shape = [1]"
                 (= [1] (mx/shape retval)))
    (assert-close "propose weight = 0"
                  0.0 (mx/item weight) 1e-6)
    (assert-true "propose choices are empty"
                 (= choices cm/EMPTY))))

;; ---------------------------------------------------------------------------
;; has-argument-grads
;; ---------------------------------------------------------------------------

(println "\n-- has-argument-grads --")

(let [gf (nn/nn->gen-fn (nn/linear 3 1))]
  (assert-true "NeuralNetGF has-argument-grads = [true]"
               (= [true] (p/has-argument-grads gf))))

;; ---------------------------------------------------------------------------
;; Splice into model
;; ---------------------------------------------------------------------------

(println "\n-- Splice into model --")

(let [net (nn/linear 3 1)
      net-gf (nn/nn->gen-fn net)
      model (gen [x]
              (let [mu (dyn/splice :net net-gf x)]
                (dyn/trace :y (dist/gaussian mu 1))
                mu))
      x (rng/normal (rng/fresh-key) [3])
      ;; Compute expected output
      expected (.forward net x)
      _ (mx/eval! expected)
      obs (cm/choicemap :y expected)
      {:keys [trace weight]} (p/generate model [x] obs)]
  (mx/eval! (:retval trace) weight)
  ;; retval should match module forward
  (assert-close "spliced model retval matches module forward"
                (mx/item expected) (mx/item (:retval trace)) 1e-5)
  ;; weight = log gaussian(y=mu; mu, 1) = -0.5*log(2*pi)
  (assert-close "generate weight = log gaussian(mu; mu, 1)"
                -0.9189 (mx/item weight) 0.01))

;; ---------------------------------------------------------------------------
;; Gradient flow — nn.valueAndGrad
;; ---------------------------------------------------------------------------

(println "\n-- Gradient flow --")

(let [lin (nn/linear 3 1)
      loss-fn (fn [x]
                (let [y (.forward lin x)]
                  (mx/sum (mx/square y))))
      vg (nn/value-and-grad lin loss-fn)
      x (rng/normal (rng/fresh-key) [3])
      [loss grads] (vg x)]
  (mx/eval! loss)
  (assert-true "loss is a scalar"
               (= [] (mx/shape loss)))
  (assert-true "loss is non-negative"
               (>= (mx/item loss) 0.0))
  (assert-true "grads is an object (parameter tree)"
               (object? grads))
  ;; Check that weight gradients exist and are non-zero
  (let [w-grad (.-weight grads)]
    (mx/eval! w-grad)
    (assert-true "weight grad exists"
                 (some? w-grad))
    (assert-true "weight grad has correct shape"
                 (= [1 3] (mx/shape w-grad)))
    (assert-true "weight grad is non-zero"
                 (> (mx/item (mx/sum (mx/abs w-grad))) 1e-10))))

;; ---------------------------------------------------------------------------
;; Training convergence: fit y = 2x + 1
;; ---------------------------------------------------------------------------

(println "\n-- Training convergence --")

(let [lin (nn/linear 1 1)
      opt (nn/optimizer :adam 0.01)
      loss-fn (fn [x]
                (let [y-pred (.forward lin x)
                      y-true (mx/add (mx/multiply (mx/scalar 2.0) x)
                                     (mx/scalar 1.0))]
                  (mx/mean (mx/square (mx/subtract y-pred y-true)))))
      vg (nn/value-and-grad lin loss-fn)]

  ;; Train for 200 steps
  (dotimes [_ 200]
    (let [x (mx/subtract (mx/multiply (rng/uniform (rng/fresh-key) [10 1]) (mx/scalar 2.0)) (mx/scalar 1.0))]
      (nn/step! lin opt vg x)))

  ;; Check final parameters
  (let [w (.-weight lin)
        b (.-bias lin)]
    (mx/eval! w b)
    (let [w-val (mx/item (mx/squeeze w))
          b-val (mx/item (mx/squeeze b))]
      (println "    weight:" w-val "(target: 2.0)")
      (println "    bias:  " b-val "(target: 1.0)")
      (assert-close "weight converges to ~2.0" 2.0 w-val 0.1)
      (assert-close "bias converges to ~1.0" 1.0 b-val 0.1))))

(println "\n=== All Neural Network Tests Complete ===")
