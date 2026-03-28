(ns genmlx.schemas
  "Malli schemas for GenMLX's data boundaries.

   Validates shapes at component seams — handler state initialization,
   sub-result merging, GFI return values, dispatcher transition-specs,
   and model schema extraction output. Never runs in the inference hot
   path. Gated by *validate?* dynamic var.

   See MALLI_INTEGRATION.md for the design rationale."
  (:require [malli.core :as m]
            [malli.util :as mu]
            [malli.error :as me]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]))

;; ---------------------------------------------------------------------------
;; Validation gate
;; ---------------------------------------------------------------------------

(def ^:dynamic *validate?*
  "When true, schema validation is active. Set to true during development
   and testing, false in production inference. Default: false."
  false)

(defn validated
  "Validate value against schema when *validate?* is true.
   No-op in production. Throws with humanized errors on failure."
  [schema value context]
  (when *validate?*
    (when-not (m/validate schema value)
      (throw (ex-info (str "Schema violation: " context)
                      {:errors (me/humanize (m/explain schema value))
                       :context context})))))

;; ---------------------------------------------------------------------------
;; Predicates for custom schema types
;; ---------------------------------------------------------------------------

(defn choicemap?
  "True if x is a GenMLX choice map (Node or Value)."
  [x]
  (or (instance? cm/Node x) (instance? cm/Value x)))

(defn trace?
  "True if x is a GenMLX Trace record."
  [x]
  (instance? tr/Trace x))

(defn mlx-array?
  "True if x is an MLX array (has .shape property)."
  [x]
  (and (some? x) (some? (.-shape x))))

;; ---------------------------------------------------------------------------
;; Handler state schemas
;; ---------------------------------------------------------------------------

(def BaseState
  "Keys common to all handler modes."
  [:map
   [:key some?]
   [:choices [:fn {:error/message "should be a ChoiceMap"} choicemap?]]
   [:score some?]
   [:executor {:optional true} fn?]])

(def SimulateState
  "Handler state for simulate mode."
  BaseState)

(def GenerateState
  "Handler state for generate mode. Adds :weight and :constraints."
  (mu/merge BaseState
    [:map
     [:weight some?]
     [:constraints [:fn {:error/message "should be a ChoiceMap"} choicemap?]]]))

(def AssessState
  "Handler state for assess mode. Same shape as generate."
  GenerateState)

(def UpdateState
  "Handler state for update mode. Adds :old-choices and :discard."
  (mu/merge GenerateState
    [:map
     [:old-choices [:fn {:error/message "should be a ChoiceMap"} choicemap?]]
     [:discard [:fn {:error/message "should be a ChoiceMap"} choicemap?]]]))

(def RegenerateState
  "Handler state for regenerate mode. Adds :old-choices and :selection."
  (mu/merge BaseState
    [:map
     [:weight some?]
     [:old-choices [:fn {:error/message "should be a ChoiceMap"} choicemap?]]
     [:selection some?]]))

(def ProjectState
  "Handler state for project mode."
  (mu/merge BaseState
    [:map
     [:weight some?]
     [:old-choices [:fn {:error/message "should be a ChoiceMap"} choicemap?]]
     [:selection some?]
     [:constraints [:fn {:error/message "should be a ChoiceMap"} choicemap?]]]))

;; ---------------------------------------------------------------------------
;; Sub-result schema (merge-sub-result input)
;; ---------------------------------------------------------------------------

(def SubResult
  "Shape of sub-generative-function results for merge-sub-result.
   :choices and :score are required. :weight and :discard are optional
   (present only in generate/update modes). :weight may be nil in
   simulate mode (combinators set it explicitly to nil)."
  [:map
   [:choices [:fn {:error/message "should be a ChoiceMap"} choicemap?]]
   [:score some?]
   [:retval {:optional true} any?]
   [:weight {:optional true} any?]
   [:discard {:optional true} [:or nil? [:fn {:error/message "should be a ChoiceMap"} choicemap?]]]
   [:splice-scores {:optional true} map?]])

;; ---------------------------------------------------------------------------
;; GFI return value schemas
;; ---------------------------------------------------------------------------

(def SimulateReturn
  "Return value of p/simulate: a Trace."
  [:fn {:error/message "should be a Trace"} trace?])

(def GenerateReturn
  "Return value of p/generate: {:trace Trace :weight mlx-scalar}."
  [:map
   [:trace [:fn {:error/message "should be a Trace"} trace?]]
   [:weight some?]
   [:unused-constraints {:optional true} set?]])

(def UpdateReturn
  "Return value of p/update: {:trace Trace :weight mlx-scalar :discard choicemap}."
  [:map
   [:trace [:fn {:error/message "should be a Trace"} trace?]]
   [:weight some?]
   [:discard [:fn {:error/message "should be a ChoiceMap"} choicemap?]]
   [:unused-constraints {:optional true} set?]])

(def RegenerateReturn
  "Return value of p/regenerate: {:trace Trace :weight mlx-scalar}."
  [:map
   [:trace [:fn {:error/message "should be a Trace"} trace?]]
   [:weight some?]])

(def AssessReturn
  "Return value of p/assess: {:retval any :weight mlx-scalar}."
  [:map
   [:retval some?]
   [:weight some?]])

(def ProposeReturn
  "Return value of p/propose: {:choices choicemap :weight mlx-scalar :retval any}."
  [:map
   [:choices [:fn {:error/message "should be a ChoiceMap"} choicemap?]]
   [:weight some?]
   [:retval some?]])

(def ProjectReturn
  "Return value of p/project: an mlx scalar."
  some?)

;; ---------------------------------------------------------------------------
;; Score type metadata
;; ---------------------------------------------------------------------------

(def ScoreType
  "Valid score encodings for transitions."
  [:enum :joint :marginal :collapsed :beam-marginal])

;; ---------------------------------------------------------------------------
;; Dispatcher transition-spec (for ARCHITECTURE.md refactoring)
;; ---------------------------------------------------------------------------

(def TransitionSpec
  "Shape returned by IDispatcher/resolve-transition.
   :run is a function (fn [gf args key opts] -> gfi-result)."
  [:map
   [:run fn?]
   [:score-type ScoreType]])

;; ---------------------------------------------------------------------------
;; Model schema (schema.cljs output)
;; ---------------------------------------------------------------------------

(def TraceSite
  "One trace site extracted from a gen body."
  [:map
   [:addr some?]
   [:addr-form some?]
   [:dist-type keyword?]
   [:dist-args vector?]
   [:deps set?]
   [:static? boolean?]])

(def SpliceSite
  "One splice site extracted from a gen body."
  [:map
   [:addr some?]
   [:gf-sym some?]
   [:deps set?]])

(def ModelSchema
  "Output of schema/extract-schema, augmented by conjugacy and compilation."
  [:map
   [:trace-sites [:vector TraceSite]]
   [:static? boolean?]
   [:dynamic-addresses? boolean?]
   [:has-branches? boolean?]
   ;; Optional keys added by various augmentation passes
   [:splice-sites {:optional true} vector?]
   [:param-sites {:optional true} vector?]
   [:loop-sites {:optional true} vector?]
   [:dep-order {:optional true} vector?]
   [:return-form {:optional true} some?]
   [:has-loops? {:optional true} boolean?]
   [:params {:optional true} some?]
   ;; Added by conjugacy detection
   [:conjugate-pairs {:optional true} [:or vector? map?]]
   [:has-conjugate? {:optional true} boolean?]
   [:analytical-plan {:optional true} some?]
   ;; Added by compilation
   [:compiled-simulate {:optional true} fn?]
   [:compiled-generate {:optional true} fn?]
   [:compiled-update {:optional true} fn?]
   [:compiled-regenerate {:optional true} fn?]
   [:compiled-assess {:optional true} fn?]
   [:compiled-project {:optional true} fn?]
   ;; Added by auto-analytical
   [:auto-handlers {:optional true} map?]
   [:auto-regenerate-transition {:optional true} fn?]
   [:auto-regenerate-handlers {:optional true} map?]])

;; ---------------------------------------------------------------------------
;; Conjugacy table entry
;; ---------------------------------------------------------------------------

(def ConjugacyEntry
  "One entry in the conjugacy table. All families have :family and
   :natural-param-idx. Additional keys are family-specific param accessors."
  [:map
   [:family keyword?]
   [:natural-param-idx int?]])

;; ---------------------------------------------------------------------------
;; Distribution record
;; ---------------------------------------------------------------------------

(def DistRecord
  "A GenMLX Distribution record: {:type keyword :params map}."
  [:map
   [:type keyword?]
   [:params map?]])
