(ns genmlx.gradients
  "Per-choice gradients from traces.
   Extract gradients of log p(trace) w.r.t. individual continuous choices.
   Foundation for gradient-based learning, parameter training, and VI."
  (:require [genmlx.mlx :as mx]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]))

(defn choice-gradients
  "Compute gradients of the model's log-probability w.r.t. specified choices.
   model: generative function
   trace: current trace
   addresses: vector of choice addresses to differentiate w.r.t.

   Returns a map of {:address -> MLX-gradient-array}.

   The gradient is d(log p(all choices | args)) / d(choice at address).
   Uses MLX's grad through the generate interface."
  [model trace addresses]
  (let [args (:args trace)
        choices (:choices trace)
        ;; Build a score function parameterized by the target choices
        indexed-addrs (mapv vector (range) addresses)
        ;; Extract current values
        current-vals (mapv #(mx/realize (cm/get-choice choices [%])) addresses)
        params (mx/array current-vals)
        ;; Score function: reconstruct choicemap with params, run generate
        score-fn (fn [params-arr]
                   (let [cm (reduce
                              (fn [cm [i addr]]
                                (cm/set-choice cm [addr] (mx/index params-arr i)))
                              choices
                              indexed-addrs)]
                     (:weight (p/generate model args cm))))
        ;; Compute gradient (compiled for faster execution)
        grad-fn (mx/compile-fn (mx/grad score-fn))
        grad-arr (grad-fn params)]
    (mx/eval! grad-arr)
    ;; Split gradient array into per-address map
    (into {}
      (map (fn [[i addr]]
             [addr (mx/index grad-arr i)])
           indexed-addrs))))

(defn score-gradient
  "Compute gradient of the model score w.r.t. a flat parameter array.
   Useful for gradient-based inference algorithms.

   model: generative function
   args: model arguments
   observations: fixed observations (ChoiceMap)
   addresses: vector of parameter addresses
   params: MLX array of parameter values

   Returns {:score MLX-scalar :grad MLX-array}."
  [model args observations addresses params]
  (let [indexed-addrs (mapv vector (range) addresses)
        score-fn (fn [p]
                   (let [cm (reduce
                              (fn [cm [i addr]]
                                (cm/set-choice cm [addr] (mx/index p i)))
                              observations
                              indexed-addrs)]
                     (:weight (p/generate model args cm))))
        vag (mx/compile-fn (mx/value-and-grad score-fn))
        [score grad] (vag params)]
    (mx/eval! score grad)
    {:score score :grad grad}))
