(ns genmlx.tensor-trace
  "Tensor-backed trace and choicemap for Level 2 compiled inference.

   TensorTrace stores all latent site values in a single [K] MLX array,
   indexed by addr-index ({:slope 0, :intercept 1, ...}).

   TensorChoiceMap lazily unpacks the tensor into standard IChoiceMap
   protocol, enabling interop with handler-based inference.

   make-tensor-score builds a score function: [K]-tensor → scalar log-prob,
   bypassing GFI protocol entirely using L1 noise-transform log-prob closures."
  (:require [genmlx.mlx :as mx]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]))

;; =========================================================================
;; TensorChoiceMap: lazy view over [K] tensor
;; =========================================================================

(defrecord TensorChoiceMap [values addr-index]
  cm/IChoiceMap
  (-has-value? [_] false)
  (-get-value [_] (throw (ex-info "TensorChoiceMap is a node, not a leaf" {})))
  (-get-submap [_ addr]
    (if-let [idx (get addr-index addr)]
      (cm/->Value (mx/index values idx))
      cm/EMPTY))
  (-submaps [_]
    (map (fn [[addr idx]]
           [addr (cm/->Value (mx/index values idx))])
         addr-index)))

;; =========================================================================
;; TensorTrace: tensor-backed trace record
;; =========================================================================
;;
;; Uses same field names as Trace (:gen-fn :args :choices :retval :score)
;; so keyword access works identically. Extra fields :values and :addr-index
;; provide direct tensor access.

(defrecord TensorTrace [gen-fn args choices values addr-index score retval])

(defn make-tensor-trace
  "Create a TensorTrace. Builds TensorChoiceMap from values + addr-index."
  [{:keys [gen-fn args values addr-index score retval]}]
  (->TensorTrace gen-fn args
                 (->TensorChoiceMap values addr-index)
                 values addr-index score retval))

;; =========================================================================
;; addr-index construction
;; =========================================================================

(defn make-addr-index
  "Build address → tensor index mapping from schema's static trace-sites.
   Uses source order (same as L1 compiled paths in prepare-static-sites),
   NOT dep-order (which may reorder independent sites)."
  [schema]
  (let [static-sites (filterv :static? (:trace-sites schema))]
    (into {} (map-indexed (fn [i s] [(:addr s) i]) static-sites))))

(defn make-latent-addr-index
  "Build address → tensor index for latent sites only (excluding observed).
   Uses source order of static trace-sites, consistent with L1 compiled paths.
   observations: a ChoiceMap — addresses present in it are excluded."
  [schema observations]
  (let [obs-addrs (set (map first (cm/addresses observations)))
        static-sites (filterv :static? (:trace-sites schema))
        latent-sites (remove #(obs-addrs (:addr %)) static-sites)]
    (into {} (map-indexed (fn [i s] [(:addr s) i]) latent-sites))))

;; =========================================================================
;; Pack / Unpack utilities
;; =========================================================================

(defn pack-values
  "Pack {addr → MLX-scalar} map into [K] tensor using addr-index ordering."
  [values-map addr-index]
  (let [pairs (sort-by val addr-index)]
    (mx/stack (mapv (fn [[addr _]] (get values-map addr)) pairs))))

(defn unpack-values
  "Unpack [K] tensor into {addr → MLX-scalar} map."
  [values-tensor addr-index]
  (into {} (map (fn [[addr idx]]
                  [addr (mx/index values-tensor idx)])
                addr-index)))

(defn tensor-trace->trace
  "Convert TensorTrace to standard Trace (for interop)."
  [tt]
  (let [values-map (unpack-values (:values tt) (:addr-index tt))
        cm (cm/from-flat-map values-map)]
    (tr/make-trace {:gen-fn (:gen-fn tt)
                    :args (:args tt)
                    :choices cm
                    :retval (:retval tt)
                    :score (:score tt)})))

(defn trace->tensor-trace
  "Convert standard Trace to TensorTrace given addr-index.
   Extracts values at each address from the trace's choicemap."
  [trace addr-index]
  (let [choices (:choices trace)
        values-map (into {} (map (fn [[addr _]]
                                   [addr (cm/get-value (cm/get-submap choices addr))])
                                 addr-index))
        values-tensor (pack-values values-map addr-index)]
    (make-tensor-trace {:gen-fn (:gen-fn trace)
                        :args (:args trace)
                        :values values-tensor
                        :addr-index addr-index
                        :score (:score trace)
                        :retval (:retval trace)})))
