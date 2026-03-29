(ns genmlx.schemas
  "Malli schemas for GenMLX's data boundaries.

   Defines shapes for handler state, sub-results, GFI return values,
   dispatcher transition-specs, and model schema extraction output.
   Used by genmlx.dev for runtime instrumentation via start!/stop!.

   Citations reference [T] Cusumano-Towner 2020 PhD thesis and
   [D] gen.dev/docs/stable/ref/core/gfi.

   See MALLI_INTEGRATION.md for the design rationale."
  (:require [malli.core :as m]
            [malli.util :as mu]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]))

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
  "[T] section 2.3.1 trace ADT internal state. Keys common to all handler modes."
  [:map
   [:key some?]
   [:choices [:fn {:error/message "should be a ChoiceMap"} choicemap?]]
   [:score some?]
   [:executor {:optional true} fn?]])

(def SimulateState
  "Handler state for simulate mode."
  BaseState)

(def GenerateState
  "[T] section 4.1.1 extended generate with internal proposal. Adds :weight and :constraints."
  (mu/merge BaseState
    [:map
     [:weight some?]
     [:constraints [:fn {:error/message "should be a ChoiceMap"} choicemap?]]]))

(def AssessState
  "Handler state for assess mode. Same shape as generate."
  GenerateState)

(def UpdateState
  "[T] section 2.3.1, Def 2.3.1 h_update. Adds :old-choices and :discard."
  (mu/merge GenerateState
    [:map
     [:old-choices [:fn {:error/message "should be a ChoiceMap"} choicemap?]]
     [:discard [:fn {:error/message "should be a ChoiceMap"} choicemap?]]]))

(def RegenerateState
  "[T] section 4.1.2 regenerate with internal proposal. Adds :old-choices and :selection."
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
  "[T] Def 2.1.16, section 2.3.1 SIMULATE. Return value of p/simulate: a Trace."
  [:fn {:error/message "should be a Trace"} trace?])

(def GenerateReturn
  "[T] section 2.3.1 GENERATE. Return value: {:trace Trace :weight MLX-scalar}."
  [:map
   [:trace [:fn {:error/message "should be a Trace"} trace?]]
   [:weight some?]
   [:unused-constraints {:optional true} set?]])

(def UpdateReturn
  "[T] section 2.3.1 UPDATE. Return value: {:trace Trace :weight MLX-scalar :discard ChoiceMap}."
  [:map
   [:trace [:fn {:error/message "should be a Trace"} trace?]]
   [:weight some?]
   [:discard [:fn {:error/message "should be a ChoiceMap"} choicemap?]]
   [:unused-constraints {:optional true} set?]])

(def RegenerateReturn
  "[T] section 4.1.2, Eq 4.1. Return value: {:trace Trace :weight MLX-scalar}."
  [:map
   [:trace [:fn {:error/message "should be a Trace"} trace?]]
   [:weight some?]])

(def AssessReturn
  "[T] section 2.3.1 LOGPDF extension. Return value: {:retval any :weight MLX-scalar}."
  [:map
   [:retval some?]
   [:weight some?]])

(def ProposeReturn
  "[D] propose. Return value: {:choices ChoiceMap :weight MLX-scalar :retval any}."
  [:map
   [:choices [:fn {:error/message "should be a ChoiceMap"} choicemap?]]
   [:weight some?]
   [:retval some?]])

(def ProjectReturn
  "[D] project. Return value: an MLX scalar."
  some?)

;; ---------------------------------------------------------------------------
;; Score type metadata
;; ---------------------------------------------------------------------------

(def ScoreType
  "Valid score encodings for transitions."
  [:enum :joint :marginal :collapsed :beam-marginal])

;; ---------------------------------------------------------------------------
;; Dispatcher transition-spec
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
  "[T] Fig 2-1, Addrs[[E]]. One trace site extracted from a gen body."
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
  "[T] section 2.2.2 denotational semantics. Output of schema/extract-schema,
   augmented by conjugacy and compilation."
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
   ;; Added by full compilation (L1-M2, L1-M4)
   [:compiled-simulate {:optional true} fn?]
   [:compiled-generate {:optional true} fn?]
   [:compiled-update {:optional true} fn?]
   [:compiled-regenerate {:optional true} fn?]
   [:compiled-assess {:optional true} fn?]
   [:compiled-project {:optional true} fn?]
   ;; Added by prefix compilation (L1-M3)
   [:compiled-prefix {:optional true} fn?]
   [:compiled-prefix-addrs {:optional true} vector?]
   [:compiled-prefix-generate {:optional true} fn?]
   [:compiled-prefix-update {:optional true} fn?]
   [:compiled-prefix-assess {:optional true} fn?]
   [:compiled-prefix-project {:optional true} fn?]
   [:compiled-prefix-regenerate {:optional true} fn?]
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
