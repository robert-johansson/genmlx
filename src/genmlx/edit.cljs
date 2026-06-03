(ns genmlx.edit
  "The parametric edit interface — GenJAX's most distinctive contribution.
   Generalizes update/regenerate into a single parametric operation with
   typed EditRequests and backward requests for reversible kernels.

   Every trace mutation (update, regenerate, custom proposal) becomes an
   instance of edit. The backward request enables automatic computation of
   acceptance weights for reversible kernels — the foundation of SMCP3."
  (:require [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]))

;; ---------------------------------------------------------------------------
;; EditRequest types
;; ---------------------------------------------------------------------------

;; Equivalent to current update: change observed values
(defrecord ConstraintEdit [constraints])

;; Equivalent to current regenerate: resample selected addresses
(defrecord SelectionEdit [selection])

;; Forward proposal GF + backward proposal GF (for SMCP3):
;; forward-gf proposes new choices, backward-gf scores the reverse move
(defrecord ProposalEdit [forward-gf forward-args backward-gf backward-args])

;; Constructors
(defn constraint-edit
  "Create a ConstraintEdit (equivalent to update)."
  [constraints]
  (->ConstraintEdit constraints))

(defn selection-edit
  "Create a SelectionEdit (equivalent to regenerate)."
  [selection]
  (->SelectionEdit selection))

(defn proposal-edit
  "Create a ProposalEdit for SMCP3-style reversible kernels."
  ([forward-gf backward-gf]
   (->ProposalEdit forward-gf nil backward-gf nil))
  ([forward-gf forward-args backward-gf backward-args]
   (->ProposalEdit forward-gf forward-args backward-gf backward-args)))

;; ---------------------------------------------------------------------------
;; IEdit protocol
;; ---------------------------------------------------------------------------

(defprotocol IEdit
  (edit [gf trace edit-request]
    "Apply an edit request to a trace.
     Returns {:trace Trace :weight MLX-scalar :discard ChoiceMap
              :backward-request EditRequest}.
     The backward-request reverses the edit — this is what makes SMCP3 work."))

;; ---------------------------------------------------------------------------
;; Default implementation that delegates to existing GFI operations
;; ---------------------------------------------------------------------------

(defn- discard-of
  "The :discard choicemap from a GFI result, defaulting to the empty map."
  [result]
  (or (:discard result) cm/EMPTY))

;; ---------------------------------------------------------------------------
;; ProposalEdit weight arithmetic — backend-free until an MLX weight appears
;; ---------------------------------------------------------------------------
;;
;; Only ProposalEdit combines scores numerically. Deterministic / degenerate GFs
;; have plain-number weights and need no MLX; MLX-backed GFs have MxArray weights
;; and need the native scalar ops. We resolve genmlx.mlx LAZILY (through the
;; membrane, not @mlx-node directly) so that merely requiring genmlx.edit never
;; forces the GPU backend — ConstraintEdit, SelectionEdit, and every constructor
;; stay genuinely Layer-A pure and loadable without @mlx-node/core present. The
;; native mx/add and mx/subtract accept both MxArray and JS-number args, so they
;; also cover mixed weights once realized. See genmlx-5413.
(defonce ^:private mx-scalar
  (delay (require '[genmlx.mlx])
         {:add (resolve 'genmlx.mlx/add)
          :subtract (resolve 'genmlx.mlx/subtract)}))

(defn- w-add
  "Add two scores: numeric (+) for pure GFs, native mx/add for MxArray weights."
  [a b]
  (if (and (number? a) (number? b)) (+ a b) ((:add @mx-scalar) a b)))

(defn- w-sub
  "Subtract scores: numeric (-) for pure GFs, native mx/subtract for MxArrays."
  [a b]
  (if (and (number? a) (number? b)) (- a b) ((:subtract @mx-scalar) a b)))

(defmulti edit-dispatch
  "Generic edit implementation that dispatches based on EditRequest type."
  (fn [_gf _trace edit-request] (type edit-request)))

(defmethod edit-dispatch ConstraintEdit
  [gf trace edit-request]
  (let [{:keys [constraints]} edit-request
        result (p/update gf trace constraints)
        discard (discard-of result)]
    (assoc result
           :backward-request (->ConstraintEdit discard))))

(defmethod edit-dispatch SelectionEdit
  [gf trace edit-request]
  (let [{:keys [selection]} edit-request
        result (p/regenerate gf trace selection)]
    (assoc result
           :discard cm/EMPTY
           ;; Backward request: regenerate the same selection
           ;; (since regenerate is its own inverse in terms of the proposal)
           :backward-request (->SelectionEdit selection))))

(defmethod edit-dispatch ProposalEdit
  [gf trace edit-request]
  (let [{:keys [forward-gf forward-args backward-gf backward-args]} edit-request
        ;; 1. Run propose on forward GF
        fwd-args (or forward-args [(:choices trace)])
        fwd-result (p/propose forward-gf fwd-args)
        fwd-choices (:choices fwd-result)
        fwd-score (:weight fwd-result)
        ;; 2. Apply proposed choices to model via update
        update-result (p/update gf trace fwd-choices)
        new-trace (:trace update-result)
        update-weight (:weight update-result)
        ;; 3. Score backward proposal
        bwd-args (or backward-args [(:choices new-trace)])
        bwd-result (p/assess backward-gf bwd-args
                             (discard-of update-result))
        bwd-score (:weight bwd-result)
        ;; 4. Combined weight (backend-free for pure GFs; see w-add/w-sub)
        weight (w-add update-weight (w-sub bwd-score fwd-score))]
    {:trace new-trace
     :weight weight
     :discard (discard-of update-result)
     :backward-request (->ProposalEdit backward-gf backward-args
                                        forward-gf forward-args)}))

(defmethod edit-dispatch :default
  [_gf _trace edit-request]
  (throw (ex-info "Unknown EditRequest type" {:request edit-request})))
