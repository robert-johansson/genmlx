(ns genmlx.edit
  "The parametric edit interface — GenJAX's most distinctive contribution.
   Generalizes update/regenerate into a single parametric operation with
   typed EditRequests and backward requests for reversible kernels.

   Every trace mutation (update, regenerate, custom proposal) becomes an
   instance of edit. The backward request enables automatic computation of
   acceptance weights for reversible kernels — the foundation of SMCP3."
  (:require [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]))

;; ---------------------------------------------------------------------------
;; EditRequest types
;; ---------------------------------------------------------------------------

(defrecord ConstraintEdit [constraints]
  ;; Equivalent to current update: change observed values
  )

(defrecord SelectionEdit [selection]
  ;; Equivalent to current regenerate: resample selected addresses
  )

(defrecord ProposalEdit [forward-gf forward-args backward-gf backward-args]
  ;; Forward proposal GF + backward proposal GF (for SMCP3)
  ;; forward-gf proposes new choices, backward-gf scores the reverse move
  )

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

(defn edit-dispatch
  "Generic edit implementation that dispatches based on EditRequest type."
  [gf trace edit-request]
  (cond
    (instance? ConstraintEdit edit-request)
    (let [result (p/update gf trace (:constraints edit-request))
          discard (or (:discard result) cm/EMPTY)]
      (assoc result
             :backward-request (->ConstraintEdit discard)))

    (instance? SelectionEdit edit-request)
    (let [result (p/regenerate gf trace (:selection edit-request))
          ;; Backward request: regenerate the same selection
          ;; (since regenerate is its own inverse in terms of the proposal)
          ]
      (assoc result
             :discard cm/EMPTY
             :backward-request (->SelectionEdit (:selection edit-request))))

    (instance? ProposalEdit edit-request)
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
                               (or (:discard update-result) cm/EMPTY))
          bwd-score (:weight bwd-result)
          ;; 4. Combined weight
          weight (mx/add update-weight (mx/subtract bwd-score fwd-score))]
      {:trace new-trace
       :weight weight
       :discard (or (:discard update-result) cm/EMPTY)
       :backward-request (->ProposalEdit backward-gf backward-args
                                          forward-gf forward-args)})

    :else
    (throw (ex-info "Unknown EditRequest type" {:request edit-request}))))
