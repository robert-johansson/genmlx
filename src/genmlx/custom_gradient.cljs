(ns genmlx.custom-gradient
  "CustomGradientGF — a deterministic generative function with optional
   user-supplied gradient. Wraps a differentiable forward computation
   (e.g. a neural network, custom loss) as a generative function.

   No random choices, score = 0. Gradients flow through forward-fn
   via MLX autograd when used inside models."
  (:require [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]))

(defrecord CustomGradientGF [forward-fn gradient-fn arg-grads]
  p/IGenerativeFunction
  (simulate [this args]
    (let [retval (apply forward-fn args)]
      (tr/make-trace {:gen-fn this :args args
                      :choices cm/EMPTY
                      :retval retval
                      :score (mx/scalar 0.0)})))

  p/IGenerate
  (generate [this args constraints]
    ;; Deterministic: no stochastic choices to constrain
    {:trace (p/simulate this args)
     :weight (mx/scalar 0.0)})

  p/IAssess
  (assess [this args choices]
    {:retval (apply forward-fn args)
     :weight (mx/scalar 0.0)})

  p/IPropose
  (propose [this args]
    {:choices cm/EMPTY
     :weight (mx/scalar 0.0)
     :retval (apply forward-fn args)})

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
  (has-argument-grads [_] arg-grads))

(defn custom-gradient-gf
  "Create a CustomGradientGF from a config map.
   :forward            — (fn [& args] -> retval), the forward computation
   :gradient           — optional (fn [args retval cotangent] -> arg-grads-vec)
   :has-argument-grads — vector of booleans per argument position"
  [{:keys [forward gradient has-argument-grads]}]
  (->CustomGradientGF forward gradient has-argument-grads))

(defn accepts-arg-grads?
  "Returns true if gf implements IHasArgumentGrads and has non-nil grads."
  [gf]
  (and (satisfies? p/IHasArgumentGrads gf)
       (some? (p/has-argument-grads gf))))
