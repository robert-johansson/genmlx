(ns genmlx.contracts
  "Data-driven GFI contract registry.
   Each contract is {:theorem string :check (fn [{:keys [model args trace]}] -> bool)}.
   Contracts verify that GFI operations satisfy their measure-theoretic invariants."
  (:require [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dynamic :as dyn]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- ev
  "Evaluate an MLX array and extract its scalar value."
  [x]
  (mx/eval! x)
  (mx/item x))

(defn- approx=
  "True if |a - b| <= tol."
  [a b tol]
  (<= (js/Math.abs (- a b)) tol))

(defn- finite?
  "True if x is a finite JS number."
  [x]
  (js/Number.isFinite x))

(defn- first-address
  "Return the first leaf address keyword from a choicemap, or nil."
  [choices]
  (let [addrs (cm/addresses choices)]
    (when (seq addrs)
      (first (first addrs)))))

(defn- path->selection
  "Convert an address path like [:x] or [:inner :z] to a proper selection.
   Single element paths use sel/select, multi-element paths use sel/hierarchical."
  [path]
  (if (= 1 (count path))
    (sel/select (first path))
    ;; Build hierarchical selection from inside out
    (reduce (fn [inner-sel addr]
              (sel/hierarchical addr inner-sel))
            (sel/select (last path))
            (reverse (butlast path)))))

;; ---------------------------------------------------------------------------
;; Contract registry
;; ---------------------------------------------------------------------------

(def contracts
  "Map of contract keyword -> {:theorem string, :check fn}.
   Each :check fn takes {:keys [model args trace]} and returns boolean."

  {:generate-weight-equals-score
   {:theorem "generate(model, args, trace.choices).weight ≈ trace.score when fully constrained"
    :check
    (fn [{:keys [model args trace]}]
      (let [{:keys [trace weight]} (p/generate model args (:choices trace))]
        (approx= (ev (:score trace)) (ev weight) 0.01)))}

   :update-empty-identity
   {:theorem "update(model, trace, trace.choices).weight ≈ 0 (no-op update)"
    :check
    (fn [{:keys [model trace]}]
      (let [{:keys [weight]} (p/update model trace (:choices trace))]
        (approx= 0.0 (ev weight) 0.01)))}

   :update-weight-correctness
   {:theorem "update(model, trace, constraint).weight ≈ new_score - old_score"
    :check
    (fn [{:keys [model trace]}]
      (let [addr (first-address (:choices trace))]
        (if (nil? addr)
          true ;; no addresses — vacuously true
          (let [old-score (ev (:score trace))
                constraint (cm/choicemap addr (mx/scalar 0.0))
                {:keys [trace weight]} (p/update model trace constraint)
                new-score (ev (:score trace))
                w (ev weight)]
            (approx= w (- new-score old-score) 0.05)))))}

   :update-round-trip
   {:theorem "update(trace, c) then update(trace', discard) recovers original values"
    :check
    (fn [{:keys [model trace]}]
      (let [addr (first-address (:choices trace))]
        (if (nil? addr)
          true
          (let [orig-val (ev (cm/get-value (cm/get-submap (:choices trace) addr)))
                constraint (cm/choicemap addr (mx/scalar 42.0))
                {:keys [trace discard]} (p/update model trace constraint)]
            (if (nil? discard)
              true
              (let [{:keys [trace]} (p/update model trace discard)
                    recovered (ev (cm/get-value (cm/get-submap (:choices trace) addr)))]
                (approx= orig-val recovered 0.01)))))))}

   :regenerate-empty-identity
   {:theorem "regenerate(model, trace, sel/none).weight ≈ 0, choices unchanged"
    :check
    (fn [{:keys [model trace]}]
      (let [addr (first-address (:choices trace))
            orig-val (when addr
                       (ev (cm/get-value (cm/get-submap (:choices trace) addr))))
            result (p/regenerate model trace sel/none)
            w (ev (:weight result))]
        (if (nil? addr)
          (approx= 0.0 w 0.01)
          (let [new-val (ev (cm/get-value (cm/get-submap (:choices (:trace result)) addr)))]
            (and (approx= 0.0 w 0.01)
                 (approx= orig-val new-val 0.01))))))}

   :project-all-equals-score
   {:theorem "project(model, trace, sel/all) ≈ trace.score"
    :check
    (fn [{:keys [model trace]}]
      (let [proj (ev (p/project model trace sel/all))
            score (ev (:score trace))]
        (approx= proj score 0.01)))}

   :project-none-equals-zero
   {:theorem "project(model, trace, sel/none) ≈ 0"
    :check
    (fn [{:keys [model trace]}]
      (let [proj (ev (p/project model trace sel/none))]
        (approx= 0.0 proj 0.01)))}

   :assess-equals-generate-score
   {:theorem "assess(model, args, choices).weight ≈ generate(model, args, choices).score"
    :check
    (fn [{:keys [model args trace]}]
      (let [choices (:choices trace)
            {:keys [weight]} (p/assess model args choices)
            gen-result (p/generate model args choices)]
        (approx= (ev weight) (ev (:score (:trace gen-result))) 0.01)))}

   :propose-generate-round-trip
   {:theorem "propose(model, args) produces choices; generate with those choices has finite weight"
    :check
    (fn [{:keys [model args]}]
      (let [{:keys [choices weight]} (p/propose model args)
            pw (ev weight)]
        (if-not (finite? pw)
          false
          (let [{:keys [weight]} (p/generate model args choices)
                gw (ev weight)]
            (finite? gw)))))}

   :score-decomposition
   {:theorem "Σ project(trace, {addr_i}) ≈ trace.score for all leaf addresses"
    :check
    (fn [{:keys [model trace]}]
      (let [addrs (cm/addresses (:choices trace))
            total-score (ev (:score trace))
            proj-sum (reduce
                       (fn [acc addr-path]
                         (let [sel (path->selection addr-path)
                               proj (ev (p/project model trace sel))]
                           (+ acc proj)))
                       0.0
                       addrs)]
        (approx= proj-sum total-score 0.05)))}

   :broadcast-equivalence
   {:theorem "vsimulate(model, args, N) produces finite scores with shape [N]"
    :check
    (fn [{:keys [model args]}]
      (let [n 10
            key (rng/fresh-key 42)
            vtrace (dyn/vsimulate model args n key)
            scores (:score vtrace)]
        (mx/eval! scores)
        (let [shape (mx/shape scores)]
          (and (= (vec shape) [n])
               (every? finite? (mx/->clj scores))))))}})
