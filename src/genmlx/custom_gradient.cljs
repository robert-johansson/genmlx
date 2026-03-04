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

(defn- wrap-with-custom-grad
  "Wrap forward-fn so MLX autograd uses gradient-fn for backward pass.
   Uses stop-gradient trick: output = sg(f(x)) + surrogate - sg(surrogate)
   where surrogate is a linear function with the desired gradient."
  [forward-fn gradient-fn]
  (fn [& args]
    (let [fwd (apply forward-fn args)
          grads (gradient-fn (vec args) fwd (mx/scalar 1.0))
          surrogate (reduce mx/add (mx/scalar 0.0)
                      (map (fn [g a] (mx/multiply (mx/stop-gradient g) a))
                           grads args))]
      (mx/add (mx/stop-gradient fwd)
              (mx/subtract surrogate (mx/stop-gradient surrogate))))))

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
  (->CustomGradientGF
    (if gradient (wrap-with-custom-grad forward gradient) forward)
    gradient
    has-argument-grads))

(defn accepts-arg-grads?
  "Returns true if gf implements IHasArgumentGrads and has non-nil grads."
  [gf]
  (and (satisfies? p/IHasArgumentGrads gf)
       (some? (p/has-argument-grads gf))))
