(ns genmlx.inference.enumerate
  "Exact enumerative inference for models with finite discrete support.
   Computes exact posterior marginals, joint posteriors, and marginal likelihoods
   by exhaustive enumeration over all combinations of discrete addresses."
  (:require [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]))

;; ---------------------------------------------------------------------------
;; Cartesian product
;; ---------------------------------------------------------------------------

(defn- cartesian-product
  "Cartesian product of a sequence of sequences.
   (cartesian-product [[1 2] [3 4]]) => [[1 3] [1 4] [2 3] [2 4]]"
  [colls]
  (reduce (fn [acc coll]
            (for [prefix acc, v coll]
              (conj prefix v)))
          [[]]
          colls))

;; ---------------------------------------------------------------------------
;; Core enumeration
;; ---------------------------------------------------------------------------

(defn- enumerate-all
  "Enumerate all combinations and score each.
   addr-supports: map of {addr [values...]}
   opts: optional map with :max-combinations (default 10000)
   Returns vector of {:choices choicemap :log-weight MLX-scalar}."
  [model args observations addr-supports opts]
  (let [model (dyn/auto-key model)
        addrs (vec (keys addr-supports))
        supports (mapv #(get addr-supports %) addrs)
        combos (cartesian-product supports)
        n-combos (count combos)
        max-combos (or (:max-combinations opts) 10000)]
    (when (> n-combos max-combos)
      (throw (ex-info (str "Enumeration: Cartesian product has " n-combos
                           " combinations (max " max-combos "). Reduce support sizes or "
                           "use approximate inference.")
                      {:n-combinations n-combos
                       :max-combinations max-combos
                       :addr-supports (into {} (map (fn [[a s]] [a (count s)]) addr-supports))})))
    (mapv (fn [combo]
            (let [combo-cm (reduce (fn [acc [addr val]]
                                    (cm/merge-cm acc (cm/choicemap addr val)))
                                  (cm/choicemap)
                                  (map vector addrs combo))
                  full-cm (if observations (cm/merge-cm observations combo-cm) combo-cm)
                  {:keys [weight]} (p/generate model args full-cm)]
              {:choices combo-cm :log-weight weight}))
          combos)))

;; ---------------------------------------------------------------------------
;; Public API
;; ---------------------------------------------------------------------------

(defn enumerate-joint
  "Exact joint posterior over discrete addresses.
   model: generative function
   args: model arguments
   observations: choicemap of observed data (or nil)
   addr-supports: map of {addr [values...]}
   opts: optional map with :max-combinations (default 10000)

   Returns vector of {:choices choicemap :log-prob MLX-scalar :prob JS-number}
   sorted by descending probability."
  ([model args observations addr-supports]
   (enumerate-joint model args observations addr-supports nil))
  ([model args observations addr-supports opts]
  (let [entries (enumerate-all model args observations addr-supports opts)
        ;; Realize all weights and extract as JS numbers
        lw-vals (mapv (fn [{:keys [log-weight]}]
                        (mx/materialize! log-weight)
                        (mx/item log-weight))
                      entries)
        w-arr (mx/array (into-array lw-vals))
        log-z-val (mx/item (mx/logsumexp w-arr))]
    (->> (map-indexed
           (fn [i {:keys [choices]}]
             (let [lp (- (nth lw-vals i) log-z-val)]
               {:choices choices
                :log-prob (mx/scalar lp)
                :prob (js/Math.exp lp)}))
           entries)
         vec
         (sort-by :prob >)))))

(defn enumerate-marginals
  "Exact posterior marginals for discrete addresses.
   model: generative function
   args: model arguments
   observations: choicemap of observed data (or nil)
   addr-supports: map of {addr [values...]}
   opts: optional map with :max-combinations (default 10000)

   Returns map of {addr {value posterior-prob}} where probabilities
   are JS numbers summing to 1.0 for each address."
  ([model args observations addr-supports]
   (enumerate-marginals model args observations addr-supports nil))
  ([model args observations addr-supports opts]
  (let [joint (enumerate-joint model args observations addr-supports opts)
        addrs (keys addr-supports)]
    (into {}
      (map (fn [addr]
             [addr
              (let [marginal-map
                    (reduce (fn [acc {:keys [choices prob]}]
                              (let [v (cm/get-choice choices [addr])
                                    v-key (mx/item v)]
                                (update acc v-key (fnil + 0.0) prob)))
                            {}
                            joint)]
                marginal-map)])
           addrs)))))

(defn enumerate-marginal-likelihood
  "Exact marginal likelihood by summing over all discrete configurations.
   Returns MLX scalar log p(observations).

   model: generative function
   args: model arguments
   observations: choicemap of observed data (or nil)
   addr-supports: map of {addr [values...]}
   opts: optional map with :max-combinations (default 10000)"
  ([model args observations addr-supports]
   (enumerate-marginal-likelihood model args observations addr-supports nil))
  ([model args observations addr-supports opts]
  (let [entries (enumerate-all model args observations addr-supports opts)
        lw-vals (mapv (fn [{:keys [log-weight]}]
                        (mx/materialize! log-weight)
                        (mx/item log-weight))
                      entries)
        w-arr (mx/array (into-array lw-vals))]
    (mx/logsumexp w-arr))))
