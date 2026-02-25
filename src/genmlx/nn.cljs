(ns genmlx.nn
  "Neural network integration — wrap MLX nn.Module as generative functions.
   Provides constructors for common layers, a bridge from nn.Module to the GFI,
   and training utilities using MLX's native nn infrastructure."
  (:require [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]))

;; ---------------------------------------------------------------------------
;; Layer constructors
;; ---------------------------------------------------------------------------

(defn linear
  "Create an nn.Linear layer: y = x @ W^T + b."
  [in-dims out-dims & {:keys [bias] :or {bias true}}]
  (new (.-Linear mx/nn-mod) in-dims out-dims bias))

(defn sequential
  "Create an nn.Sequential module from a vector of layers."
  [layers]
  (js/Reflect.construct (.-Sequential mx/nn-mod) (to-array layers)))

(defn relu [] (new (.-ReLU mx/nn-mod)))
(defn gelu [] (new (.-GELU mx/nn-mod)))
(defn tanh-act [] (new (.-Tanh mx/nn-mod)))
(defn sigmoid-act [] (new (.-Sigmoid mx/nn-mod)))

(defn layer-norm [dims] (new (.-LayerNorm mx/nn-mod) dims))
(defn embedding [num-embeddings dims] (new (.-Embedding mx/nn-mod) num-embeddings dims))
(defn dropout [p] (new (.-Dropout mx/nn-mod) p))

;; ---------------------------------------------------------------------------
;; NeuralNetGF — wraps nn.Module as a deterministic generative function
;; ---------------------------------------------------------------------------

(defrecord NeuralNetGF [module]
  p/IGenerativeFunction
  (simulate [this args]
    (let [retval (.forward module (first args))]
      (tr/make-trace {:gen-fn this :args args
                      :choices cm/EMPTY :retval retval
                      :score (mx/scalar 0.0)})))

  p/IGenerate
  (generate [this args constraints]
    {:trace (p/simulate this args) :weight (mx/scalar 0.0)})

  p/IAssess
  (assess [this args choices]
    {:retval (.forward module (first args)) :weight (mx/scalar 0.0)})

  p/IPropose
  (propose [this args]
    {:choices cm/EMPTY :weight (mx/scalar 0.0)
     :retval (.forward module (first args))})

  p/IUpdate
  (update [this trace constraints]
    {:trace (p/simulate this (:args trace))
     :weight (mx/scalar 0.0)
     :discard cm/EMPTY})

  p/IRegenerate
  (regenerate [this trace selection]
    {:trace (p/simulate this (:args trace))
     :weight (mx/scalar 0.0)})

  p/IProject
  (project [this trace selection]
    (mx/scalar 0.0))

  p/IHasArgumentGrads
  (has-argument-grads [_] [true]))

;; ---------------------------------------------------------------------------
;; Bridge
;; ---------------------------------------------------------------------------

(defn nn->gen-fn
  "Wrap an MLX nn.Module as a deterministic generative function.
   The module's forward pass becomes the GF body."
  [module]
  (->NeuralNetGF module))

;; ---------------------------------------------------------------------------
;; Training utilities
;; ---------------------------------------------------------------------------

(defn value-and-grad
  "Create a function that computes loss and gradients w.r.t. module parameters.
   Uses MLX's native nn.valueAndGrad.
   loss-fn: (fn [& inputs] -> MLX scalar loss)
   Returns: (fn [& inputs] -> [loss, grad-tree])"
  [module loss-fn]
  (let [vg (.valueAndGrad mx/nn-mod module loss-fn)]
    (fn [& inputs]
      (let [result (apply vg inputs)]
        [(aget result 0) (aget result 1)]))))

(defn optimizer
  "Create a native MLX optimizer.
   type: :adam, :sgd, :adamw
   lr: learning rate"
  [type lr]
  (case type
    :adam  (new (.-Adam mx/optim-mod) lr)
    :sgd   (new (.-SGD mx/optim-mod) lr)
    :adamw (new (.-AdamW mx/optim-mod) lr)))

(defn step!
  "One training step: compute loss+grads, update module parameters.
   Returns the loss value (JS number)."
  [module optim vg-fn & inputs]
  (let [[loss grads] (apply vg-fn inputs)]
    (.update optim module grads)
    (mx/eval! module)
    (mx/eval! loss)
    (mx/item loss)))
