(ns genmlx.nn
  "Neural network layers as functional data.

   Each layer is a plain ClojureScript map:

     {:params  {:weight <MxArray>, :bias <MxArray>}
      :forward (fn [x] ...)
      :type    :linear}

   Layers compose via `sequential`. Parameters are flat maps with
   deterministic key order, directly usable with MxArray.valueAndGrad.

   Training uses `value-and-grad` (bridges to MLX autograd) and
   pure-function optimizers. `NeuralNetGF` wraps layers as
   deterministic generative functions for the GFI."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
))

;; Forward declarations for rebuild functions (defined after constructors).
(declare make-linear-rebuild make-layer-norm-rebuild make-embedding-rebuild
         make-sequential-rebuild rebuild-with-params)

;; ---------------------------------------------------------------------------
;; Layer constructors
;; ---------------------------------------------------------------------------

(defn linear
  "Create a Linear layer: y = x @ W^T + b.
   Kaiming uniform initialization: uniform(-k, k) where k = 1/sqrt(in-dims)."
  [in-dims out-dims & {:keys [bias] :or {bias true}}]
  (let [k       (/ 1.0 (js/Math.sqrt in-dims))
        key     (rng/fresh-key)
        [k1 k2] (rng/split key)
        s       (mx/scalar (* 2.0 k))
        weight  (mx/subtract (mx/multiply (rng/uniform k1 [out-dims in-dims]) s)
                              (mx/scalar k))
        bias-arr (when bias
                   (mx/subtract (mx/multiply (rng/uniform k2 [out-dims]) s)
                                (mx/scalar k)))
        params  (cond-> {:weight weight}
                  bias (assoc :bias bias-arr))]
    (mx/eval! weight)
    (when bias-arr (mx/eval! bias-arr))
    (let [rebuild-fn (make-linear-rebuild bias)]
      (assoc (rebuild-fn params) :type :linear))))

(defn sequential
  "Compose layers. Params are flattened with index-prefixed keys:
   :0/weight, :0/bias, :1/weight, etc."
  [layers]
  (let [all-params (into {}
                     (mapcat (fn [[i layer]]
                               (map (fn [[k v]]
                                      [(keyword (str i "/" (name k))) v])
                                    (:params layer)))
                             (map-indexed vector layers)))]
    {:params all-params
     :forward (fn [x] (reduce (fn [acc layer] ((:forward layer) acc)) x layers))
     :type :sequential
     :layers (vec layers)
     :rebuild (make-sequential-rebuild (vec layers))}))

;; ---------------------------------------------------------------------------
;; Activation layers (no parameters)
;; ---------------------------------------------------------------------------

(defn relu [] {:params {} :forward (fn [x] (mx/maximum x (mx/scalar 0.0))) :type :relu})
(defn gelu []
  {:params {}
   :forward (fn [x]
              (let [c (mx/scalar 0.044715)
                    s (mx/scalar (js/Math.sqrt (/ 2.0 js/Math.PI)))]
                (mx/multiply x
                  (mx/multiply (mx/scalar 0.5)
                    (mx/add (mx/scalar 1.0)
                      (mx/tanh (mx/multiply s
                                 (mx/add x (mx/multiply c
                                             (mx/multiply x (mx/multiply x x)))))))))))
   :type :gelu})
(defn tanh-act   [] {:params {} :forward mx/tanh    :type :tanh})
(defn sigmoid-act [] {:params {} :forward mx/sigmoid :type :sigmoid})

;; ---------------------------------------------------------------------------
;; Other layers
;; ---------------------------------------------------------------------------

(defn layer-norm
  "Layer normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta."
  [dims]
  (let [gamma (mx/ones [dims])
        beta  (mx/zeros [dims])
        rebuild-fn (make-layer-norm-rebuild)]
    (rebuild-fn {:gamma gamma :beta beta})))

(defn embedding
  "Lookup embedding: indices -> weight[indices]."
  [num-embeddings dims]
  (let [weight (mx/multiply (rng/normal (rng/fresh-key) [num-embeddings dims])
                            (mx/scalar 0.01))]
    (mx/eval! weight)
    ((make-embedding-rebuild) {:weight weight})))

(defn dropout
  "No-op layer. Stochasticity during forward passes is handled by the
   generative function framework, not by dropout masks."
  [_p]
  {:params {} :forward identity :type :dropout})

;; ---------------------------------------------------------------------------
;; Parameter utilities
;; ---------------------------------------------------------------------------

(defn param-keys
  "Parameter keys in deterministic sorted order."
  [layer]
  (vec (sort (keys (:params layer)))))

(defn parameters
  "Parameter arrays in deterministic sorted order."
  [layer]
  (mapv #(get (:params layer) %) (param-keys layer)))

;; ---------------------------------------------------------------------------
;; Rebuild layer with new parameter values
;;
;; Each layer stores a :rebuild function that reconstructs the forward
;; closure from new parameter arrays. This lets MLX autograd trace through
;; the computation: valueAndGrad passes parameter arrays as arguments,
;; rebuild creates a forward fn that uses them, and all MxArray operations
;; inside forward are part of the gradient computation graph.
;;
;; AUTOGRAD GOTCHA: Inside a valueAndGrad scope, MLX wraps parameter arrays
;; in traced wrappers. Iterating over a ClojureScript map's entries (via seq,
;; keep, map-on-map) can lose this traced identity. Always use direct `get`
;; lookups on the params map — never iterate entries to extract values.
;; ---------------------------------------------------------------------------

(defn- make-linear-rebuild [has-bias?]
  (fn rebuild [new-params]
    (let [w (:weight new-params)
          b (:bias new-params)]
      {:params new-params
       :forward (if (and has-bias? b)
                  (fn [x] (mx/add (mx/matmul x (mx/transpose w)) b))
                  (fn [x] (mx/matmul x (mx/transpose w))))
       :type :linear
       :rebuild rebuild})))

(defn- make-layer-norm-rebuild []
  (fn rebuild [new-params]
    (let [gamma (:gamma new-params)
          beta  (:beta new-params)]
      {:params new-params
       :forward (fn [x]
                  (let [mu (mx/mean x [-1])
                        v  (mx/variance x [-1])
                        xh (mx/divide (mx/subtract x mu)
                                      (mx/sqrt (mx/add v (mx/scalar 1e-5))))]
                    (mx/add (mx/multiply gamma xh) beta)))
       :type :layer-norm
       :rebuild rebuild})))

(defn- make-embedding-rebuild []
  (fn rebuild [new-params]
    (let [w (:weight new-params)]
      {:params new-params
       :forward (fn [indices] (mx/take-idx w indices 0))
       :type :embedding
       :rebuild rebuild})))

(defn rebuild-with-params
  "Create a new layer with updated parameters. Delegates to the layer's
   own :rebuild function, which reconstructs the forward closure to capture
   the new arrays so autograd can trace through them."
  [layer new-params]
  (if-let [rebuild-fn (:rebuild layer)]
    (rebuild-fn new-params)
    ;; Activation / dropout layers have no params and no :rebuild — return as-is
    layer))

(defn- make-sequential-rebuild [child-layers]
  (fn rebuild [new-params]
    (let [rebuilt (vec (map-indexed
                         (fn [i child]
                           (let [prefix (str i "/")
                                 ;; Look up each child's param keys via the parent's
                                 ;; prefixed keys. Uses direct `get` on new-params
                                 ;; to preserve traced array identity for autograd.
                                 child-param-keys (keys (:params child))
                                 child-params (reduce
                                                (fn [acc ck]
                                                  (let [pk (keyword (str prefix (name ck)))]
                                                    (if-let [v (get new-params pk)]
                                                      (assoc acc ck v)
                                                      acc)))
                                                {} child-param-keys)]
                             (if (empty? child-params)
                               child
                               (rebuild-with-params child child-params))))
                         child-layers))]
      {:params new-params
       :forward (fn [x] (reduce (fn [acc l] ((:forward l) acc)) x rebuilt))
       :type :sequential
       :layers rebuilt
       :rebuild (make-sequential-rebuild rebuilt)})))

;; ---------------------------------------------------------------------------
;; NeuralNetGF — wraps a layer as a deterministic generative function
;; ---------------------------------------------------------------------------

(defrecord NeuralNetGF [layer-ref]
  ;; layer-ref is an atom containing a layer map.
  p/IGenerativeFunction
  (simulate [this args]
    (let [layer @layer-ref
          retval ((:forward layer) (first args))]
      (tr/make-trace {:gen-fn this :args args
                      :choices cm/EMPTY :retval retval
                      :score (mx/scalar 0.0)})))

  p/IGenerate
  (generate [this args constraints]
    {:trace (p/simulate this args) :weight (mx/scalar 0.0)})

  p/IAssess
  (assess [this args choices]
    {:retval ((:forward @layer-ref) (first args)) :weight (mx/scalar 0.0)})

  p/IPropose
  (propose [this args]
    {:choices cm/EMPTY :weight (mx/scalar 0.0)
     :retval ((:forward @layer-ref) (first args))})

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
  "Wrap a layer (map or atom) as a deterministic generative function."
  [layer-or-atom]
  (let [ref (if (instance? Atom layer-or-atom)
              layer-or-atom
              (atom layer-or-atom))]
    (->NeuralNetGF ref)))

;; ---------------------------------------------------------------------------
;; Autograd bridge
;; ---------------------------------------------------------------------------

(defn value-and-grad
  "Create a function that computes loss and parameter gradients.

   layer-ref: atom containing a layer map
   loss-fn:   (fn [forward-fn & inputs] -> scalar MxArray)
              The forward-fn is the layer's forward function, rebuilt from
              the parameter arrays that autograd is tracking.

   Returns: (fn [& inputs] -> [scalar-loss param-grads-map])"
  [layer-ref loss-fn]
  (fn [& inputs]
    (let [layer       @layer-ref
          keys-sorted (param-keys layer)
          n-params    (count keys-sorted)
          ;; Close over inputs — only pass parameter MxArrays to valueAndGrad.
          ;; Inputs can be arbitrary values (vectors, maps, etc.), not just MxArrays.
          flat-fn     (fn [& param-args]
                        (let [new-params (zipmap keys-sorted (vec param-args))
                              new-layer  (rebuild-with-params layer new-params)]
                          (apply loss-fn (:forward new-layer) inputs)))
          param-arrays (mapv #(get (:params layer) %) keys-sorted)
          argnums     (vec (range n-params))
          vg-fn       (mx/value-and-grad flat-fn argnums)
          [loss grads] (apply vg-fn param-arrays)
          param-grads (zipmap keys-sorted
                              (if (sequential? grads) grads [grads]))]
      [loss param-grads])))

;; ---------------------------------------------------------------------------
;; Optimizers
;; ---------------------------------------------------------------------------

(defn- sgd-step [lr]
  (let [lr-s (mx/scalar lr)]
    (fn [params grads]
      (into {} (map (fn [[k v]]
                      [k (mx/subtract v (mx/multiply lr-s (get grads k)))])
                    params)))))

(defn- adam-step [lr {:keys [beta1 beta2 eps] :or {beta1 0.9 beta2 0.999 eps 1e-8}}]
  (let [state (atom {:m {} :v {} :t 0})]
    (fn [params grads]
      (let [{:keys [m v t]} (swap! state update :t inc)
            b1s  (mx/scalar beta1)
            b1cs (mx/scalar (- 1.0 beta1))
            b2s  (mx/scalar beta2)
            b2cs (mx/scalar (- 1.0 beta2))
            lrs  (mx/scalar lr)
            epss (mx/scalar eps)]
        (reduce (fn [acc [k p]]
                  (let [g  (get grads k)
                        mk (mx/add (mx/multiply b1s (get m k (mx/zeros (mx/shape p))))
                                   (mx/multiply b1cs g))
                        vk (mx/add (mx/multiply b2s (get v k (mx/zeros (mx/shape p))))
                                   (mx/multiply b2cs (mx/square g)))
                        mh (mx/divide mk (mx/scalar (- 1.0 (js/Math.pow beta1 t))))
                        vh (mx/divide vk (mx/scalar (- 1.0 (js/Math.pow beta2 t))))
                        np (mx/subtract p (mx/divide (mx/multiply lrs mh)
                                                     (mx/add (mx/sqrt vh) epss)))]
                    (swap! state assoc-in [:m k] mk)
                    (swap! state assoc-in [:v k] vk)
                    (assoc acc k np)))
                {} params)))))

(defn optimizer
  "Create an optimizer function: (fn [params grads] -> new-params).
   type: :sgd, :adam, or :adamw."
  [type lr & {:as opts}]
  (case type
    :sgd   (sgd-step lr)
    :adam  (adam-step lr (or opts {}))
    :adamw (let [wd  (or (:weight-decay opts) 0.01)
                 wds (mx/scalar (- 1.0 (* lr wd)))
                 inner (adam-step lr (or opts {}))]
             (fn [params grads]
               (inner (into {} (map (fn [[k v]] [k (mx/multiply v wds)]) params))
                      grads)))))

;; ---------------------------------------------------------------------------
;; Training
;; ---------------------------------------------------------------------------

(defn training-step!
  "One training step: compute loss+grads, apply optimizer, update layer atom.
   Returns scalar loss value (JS number)."
  [layer-ref opt vg-fn & inputs]
  (let [[loss grads] (apply vg-fn inputs)
        new-params   (opt (:params @layer-ref) grads)
        new-layer    (rebuild-with-params @layer-ref new-params)]
    ;; Materialize new params + loss before next iteration
    (apply mx/eval! (vals new-params))
    (mx/eval! loss)
    (reset! layer-ref new-layer)
    (mx/item loss)))
