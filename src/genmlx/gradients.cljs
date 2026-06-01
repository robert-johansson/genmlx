(ns genmlx.gradients
  "Per-choice gradients from traces.
   Extract gradients of log p(trace) w.r.t. individual continuous choices.
   Foundation for gradient-based learning, parameter training, and VI."
  (:require [genmlx.mlx :as mx]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]))

(defn- params->weight
  "Set each indexed address on base-cm to the corresponding entry of params-arr,
   then return the generate weight under model/args. indexed-addrs is a vector
   of [i addr] pairs (index into params-arr)."
  [model args base-cm indexed-addrs params-arr]
  (let [acc (reduce
              (fn [acc [i addr]]
                (cm/set-choice acc [addr] (mx/index params-arr i)))
              base-cm
              indexed-addrs)]
    (:weight (p/generate model args acc))))

(defn choice-gradients
  "Compute gradients of the model's log-probability w.r.t. specified choices.
   model: generative function
   trace: current trace
   addresses: vector of choice addresses to differentiate w.r.t.

   Returns a map of {:address -> MLX-gradient-array}.

   The gradient is d(log p(all choices | args)) / d(choice at address).
   Uses MLX's grad through the generate interface."
  [model trace addresses]
  (let [model (dyn/auto-key model)
        args (:args trace)
        choices (:choices trace)]
    (run! (fn [addr]
            (when-not (cm/has-value? (cm/get-submap choices addr))
              (throw (ex-info (str "choice-gradients: address " (pr-str addr)
                                   " not found in trace choices")
                              {:address addr}))))
          addresses)
    (let [;; Build a score function parameterized by the target choices
          indexed-addrs (mapv vector (range) addresses)
          ;; Extract current values
          current-vals (mapv #(mx/realize (cm/get-choice choices [%])) addresses)
          params (mx/array current-vals)
          ;; Score function: reconstruct choicemap with params, run generate
          score-fn (fn [params-arr]
                     (params->weight model args choices indexed-addrs params-arr))
          ;; Compute gradient (uncompiled — compile-fn severs backward
          ;; pass when model body uses mx/eval!, returning silent zeros)
          grad-fn (mx/grad score-fn)
          grad-arr (grad-fn params)]
      (mx/materialize! grad-arr)
      ;; Split gradient array into per-address map
      (into {}
        (map (fn [[i addr]]
               [addr (mx/index grad-arr i)])
             indexed-addrs)))))

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
  (let [model (dyn/auto-key model)
        n-addr (count addresses)
        n-param (first (mx/shape params))]
    (when-not (= n-addr n-param)
      (throw (ex-info (str "score-gradient: addresses count (" n-addr
                           ") != params dimension (" n-param ")")
                      {:addresses-count n-addr
                       :params-shape (mx/shape params)})))
    (let [indexed-addrs (mapv vector (range) addresses)
          score-fn (fn [p]
                     (params->weight model args observations indexed-addrs p))
          vag (mx/value-and-grad score-fn)
          [score grad] (vag params)]
      (mx/materialize! score grad)
      {:score score :grad grad})))
