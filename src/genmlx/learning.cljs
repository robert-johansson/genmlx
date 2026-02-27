(ns genmlx.learning
  "Trainable parameters, parameter stores, optimizers, and learning algorithms.
   Includes wake-sleep learning for amortized inference."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.handler :as h]))

;; ---------------------------------------------------------------------------
;; Parameter Store
;; ---------------------------------------------------------------------------

(defn make-param-store
  "Create a functional parameter store.
   Initial params can be a map of {name -> MLX-array}."
  ([] {:params {} :version 0})
  ([init-params]
   {:params (into {} (map (fn [[k v]]
                            [k (if (mx/array? v) v (mx/scalar v))])
                          init-params))
    :version 0}))

(defn get-param
  "Get a parameter value from the store."
  [store name]
  (get-in store [:params name]))

(defn set-param
  "Set a parameter value in the store."
  [store name value]
  (-> store
      (assoc-in [:params name] (if (mx/array? value) value (mx/scalar value)))
      (update :version inc)))

(defn update-params
  "Apply a map of updates {name -> new-value} to the store."
  [store updates]
  (reduce (fn [s [k v]] (set-param s k v))
          store
          updates))

(defn param-names
  "List all parameter names in the store."
  [store]
  (keys (:params store)))

(defn params->array
  "Flatten all parameters into a single 1-D MLX array."
  [store names]
  (mx/array (mapv #(mx/realize (get-param store %)) names)))

(defn array->params
  "Unflatten a 1-D MLX array back into named parameters."
  [arr names]
  (into {} (map-indexed (fn [i name]
                          [name (mx/index arr i)])
                        names)))

;; ---------------------------------------------------------------------------
;; Optimizers
;; ---------------------------------------------------------------------------

(defn sgd-step
  "Stochastic gradient descent step.
   Returns updated parameter array."
  [params grad lr]
  (mx/subtract params (mx/multiply (mx/scalar lr) grad)))

(defn adam-init
  "Initialize Adam optimizer state."
  [params]
  {:m (mx/zeros (mx/shape params))
   :v (mx/zeros (mx/shape params))
   :t 0})

(defn adam-step
  "One Adam optimizer step. Returns [new-params new-state]."
  [params grad state
   {:keys [lr beta1 beta2 epsilon]
    :or {lr 0.001 beta1 0.9 beta2 0.999 epsilon 1e-8}}]
  (let [t (inc (:t state))
        m (mx/add (mx/multiply (mx/scalar beta1) (:m state))
                  (mx/multiply (mx/scalar (- 1.0 beta1)) grad))
        v (mx/add (mx/multiply (mx/scalar beta2) (:v state))
                  (mx/multiply (mx/scalar (- 1.0 beta2)) (mx/square grad)))
        m-hat (mx/divide m (mx/scalar (- 1.0 (js/Math.pow beta1 t))))
        v-hat (mx/divide v (mx/scalar (- 1.0 (js/Math.pow beta2 t))))
        update (mx/divide m-hat
                          (mx/add (mx/sqrt v-hat) (mx/scalar epsilon)))
        new-params (mx/subtract params (mx/multiply (mx/scalar lr) update))]
    (mx/eval! new-params m v)
    [new-params {:m m :v v :t t}]))

;; ---------------------------------------------------------------------------
;; Training loop
;; ---------------------------------------------------------------------------

(defn train
  "Generic training loop for parameter learning.

   opts:
     :iterations   - number of training steps
     :optimizer    - :sgd or :adam (default :adam)
     :lr           - learning rate (default 0.001)
     :callback     - fn called each step with {:iter :loss :params}
     :key          - PRNG key

   loss-grad-fn: (fn [params key] -> {:loss MLX-scalar :grad MLX-array})
   init-params: initial MLX parameter array

   Returns {:params final-params :loss-history [numbers...]}"
  [{:keys [iterations optimizer lr callback key]
    :or {iterations 1000 optimizer :adam lr 0.001}}
   loss-grad-fn init-params]
  (let [opt-state (when (= optimizer :adam) (adam-init init-params))]
    (loop [i 0 params init-params
           opt-st opt-state
           losses (transient [])
           rk key]
      (if (>= i iterations)
        {:params params :loss-history (persistent! losses)}
        (let [[step-key next-key] (rng/split-or-nils rk)
              {:keys [loss grad]} (loss-grad-fn params step-key)
              _ (mx/eval! loss grad)
              loss-val (mx/item loss)
              [new-params new-opt-st]
              (case optimizer
                :sgd [(sgd-step params grad lr) nil]
                :adam (adam-step params grad opt-st {:lr lr}))]
          (when callback
            (callback {:iter i :loss loss-val :params new-params}))
          (recur (inc i) new-params new-opt-st
                 (conj! losses loss-val) next-key))))))

;; ---------------------------------------------------------------------------
;; Param-store integration with GFI
;; ---------------------------------------------------------------------------

(defn simulate-with-params
  "Simulate a generative function with a param store bound.
   Parameters declared via dyn/param inside the model will read from the store."
  [model args param-store]
  (binding [h/*param-store* param-store]
    (p/simulate model args)))

(defn generate-with-params
  "Generate from a generative function with a param store bound."
  [model args constraints param-store]
  (binding [h/*param-store* param-store]
    (p/generate model args constraints)))

(defn make-param-loss-fn
  "Create a loss-gradient function for training model parameters.
   model: generative function using dyn/param
   args: model arguments
   observations: observed data (choice map)
   param-names-vec: vector of parameter names (keywords)

   Returns (fn [params-array key] -> {:loss :grad})
   where params-array is a flat MLX array matching param-names-vec."
  [model args observations param-names-vec]
  (fn [params-array key]
    (let [loss-fn (fn [p]
                    (let [store {:params (into {}
                                          (map-indexed
                                            (fn [i nm] [nm (mx/index p i)])
                                            param-names-vec))}]
                      (binding [h/*param-store* store]
                        (let [{:keys [weight]} (p/generate model args observations)]
                          ;; Negative log-joint (minimize)
                          (mx/negative weight)))))
          grad-fn (mx/grad loss-fn)
          loss (loss-fn params-array)
          grad (grad-fn params-array)]
      {:loss loss :grad grad})))

;; ---------------------------------------------------------------------------
;; Wake-Sleep Learning
;; ---------------------------------------------------------------------------

(defn wake-phase-loss
  "Wake phase: minimize KL(q||p) by optimizing guide parameters.
   Uses reparameterized samples from the guide.
   model: target generative function
   guide: guide/recognition generative function
   args: model arguments
   observations: observed data

   Returns (fn [guide-params key] -> {:loss :grad})"
  [model guide args observations guide-addresses]
  (fn [guide-params key]
    (let [indexed-addrs (mapv vector (range) guide-addresses)
          [k1 k2] (rng/split (rng/ensure-key key))
          loss-fn (fn [params]
                    (let [;; Build guide choicemap from params
                          guide-cm (reduce
                                     (fn [cm [i addr]]
                                       (cm/set-choice cm [addr] (mx/index params i)))
                                     cm/EMPTY indexed-addrs)
                          ;; Sample from guide
                          {:keys [trace weight]}
                          (binding [rng/*prng-key* (volatile! k1)]
                            (p/generate guide args guide-cm))
                          ;; Score under model with guide's choices + observations
                          guide-weight weight
                          model-cm (cm/merge-cm (:choices trace) observations)
                          {:keys [weight]}
                          (binding [rng/*prng-key* (volatile! k2)]
                            (p/generate model args model-cm))]
                      ;; Negative ELBO: -(log p(x,z) - log q(z|x))
                      (mx/negative (mx/subtract weight guide-weight))))
          grad-fn (mx/grad loss-fn)
          loss (loss-fn guide-params)
          grad (grad-fn guide-params)]
      {:loss loss :grad grad})))

(defn sleep-phase-loss
  "Sleep phase: minimize KL(p||q) by optimizing guide to match model's prior.
   Samples from the model prior, trains guide to reconstruct.
   model: target generative function
   guide: guide/recognition generative function
   args: model arguments

   Returns (fn [guide-params key] -> {:loss :grad})"
  [model guide args guide-addresses]
  (fn [guide-params key]
    (let [indexed-addrs (mapv vector (range) guide-addresses)
          [k1 k2] (rng/split (rng/ensure-key key))
          ;; Sample from model prior (simulate)
          model-trace (binding [rng/*prng-key* (volatile! k1)]
                        (p/simulate model args))
          model-choices (:choices model-trace)
          ;; Score guide on model's choices
          loss-fn (fn [params]
                    (let [guide-cm (reduce
                                     (fn [cm [i addr]]
                                       (cm/set-choice cm [addr] (mx/index params i)))
                                     cm/EMPTY indexed-addrs)
                          {:keys [weight]}
                          (binding [rng/*prng-key* (volatile! k2)]
                            (p/generate guide args
                                        (cm/merge-cm guide-cm model-choices)))]
                      ;; Negative log-likelihood of model choices under guide
                      (mx/negative weight)))
          grad-fn (mx/grad loss-fn)
          loss (loss-fn guide-params)
          grad (grad-fn guide-params)]
      {:loss loss :grad grad})))

(defn- discover-guide-addresses
  "Simulate the guide once to discover all trace addresses."
  [guide args]
  (let [trace (p/simulate guide args)
        addrs (cm/addresses (:choices trace))]
    (vec (distinct (map first addrs)))))

(defn wake-sleep
  "Wake-sleep learning for amortized inference.

   opts:
     :iterations       - number of wake-sleep cycles (default 1000)
     :wake-steps       - wake phase steps per cycle (default 1)
     :sleep-steps      - sleep phase steps per cycle (default 1)
     :lr               - learning rate (default 0.001)
     :callback         - fn called each cycle
     :key              - PRNG key

   model: target generative function
   guide: guide/recognition generative function
   args: model arguments
   observations: observed data
   guide-addresses: vector of guide parameter addresses (or nil to auto-discover)
   init-guide-params: initial guide parameter array (or nil for zero-init)

   Returns {:params final-guide-params :wake-losses :sleep-losses}"
  [{:keys [iterations wake-steps sleep-steps lr callback key]
    :or {iterations 1000 wake-steps 1 sleep-steps 1 lr 0.001}}
   model guide args observations guide-addresses init-guide-params]
  (let [;; Auto-discover guide addresses if not provided
        guide-addresses (or guide-addresses
                            (discover-guide-addresses guide args))
        init-guide-params (or init-guide-params
                              (mx/zeros [(count guide-addresses)]))
        wake-loss-fn (wake-phase-loss model guide args observations guide-addresses)
        sleep-loss-fn (sleep-phase-loss model guide args guide-addresses)
        opt-state (adam-init init-guide-params)]
    (loop [i 0 params init-guide-params
           opt-st opt-state
           wake-losses (transient [])
           sleep-losses (transient [])
           rk key]
      (if (>= i iterations)
        {:params params
         :wake-losses (persistent! wake-losses)
         :sleep-losses (persistent! sleep-losses)}
        (let [[wake-key sleep-key next-key] (rng/split-n-or-nils rk 3)
              ;; Wake phase
              [params' opt-st' wl]
              (loop [j 0 p params os opt-st losses [] wk wake-key]
                (if (>= j wake-steps)
                  [p os losses]
                  (let [[wk1 wk2] (rng/split-or-nils wk)
                        {:keys [loss grad]} (wake-loss-fn p wk1)
                        _ (mx/eval! loss grad)
                        [p' os'] (adam-step p grad os {:lr lr})]
                    (recur (inc j) p' os' (conj losses (mx/item loss)) wk2))))
              ;; Sleep phase
              [params'' opt-st'' sl]
              (loop [j 0 p params' os opt-st' losses [] sk sleep-key]
                (if (>= j sleep-steps)
                  [p os losses]
                  (let [[sk1 sk2] (rng/split-or-nils sk)
                        {:keys [loss grad]} (sleep-loss-fn p sk1)
                        _ (mx/eval! loss grad)
                        [p' os'] (adam-step p grad os {:lr lr})]
                    (recur (inc j) p' os' (conj losses (mx/item loss)) sk2))))]
          (when (zero? (mod i 50)) (mx/clear-cache!))
          (when callback
            (callback {:iter i :wake-loss (last wl) :sleep-loss (last sl)}))
          (recur (inc i) params'' opt-st''
                 (reduce conj! wake-losses wl)
                 (reduce conj! sleep-losses sl)
                 next-key))))))
