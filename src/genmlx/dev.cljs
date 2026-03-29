(ns genmlx.dev
  "Development instrumentation for GenMLX.

   Provides start!/stop! for Malli schema validation at GFI boundaries.
   Replaces malli.dev.cljs/start! which requires Google Closure and the
   ClojureScript compiler — neither available under nbb.

   Usage (REPL or test preload):
     (require '[genmlx.dev :as dev])
     (dev/start!)        ;; validates all GFI inputs/outputs
     (dev/stop!)         ;; back to zero-cost dispatch

   All validation is off by default. Calling start! swaps two atoms:
   - genmlx.dynamic/dispatch-fn  (wraps GFI dispatch with output validation)
   - genmlx.runtime/validate-fn  (wraps handler init-state and sub-result checks)"
  (:require [clojure.string :as str]
            [malli.core :as m]
            [malli.error :as me]
            [genmlx.schemas :as schemas]
            [genmlx.dynamic :as dyn]
            [genmlx.runtime :as rt]))

;; ---------------------------------------------------------------------------
;; Precompiled validators — compiled once at load time, not per call.
;; ---------------------------------------------------------------------------

(def ^:private op->validator
  (into {} (map (fn [[op schema]] [op {:validate (m/validator schema)
                                       :explain  (m/explainer schema)
                                       :schema   schema}]))
        {:simulate   schemas/SimulateReturn
         :generate   schemas/GenerateReturn
         :update     schemas/UpdateReturn
         :regenerate schemas/RegenerateReturn
         :assess     schemas/AssessReturn
         :propose    schemas/ProposeReturn
         :project    schemas/ProjectReturn}))

(def ^:private key->validator
  (into {} (map (fn [[k schema]] [k {:validate (m/validator schema)
                                     :explain  (m/explainer schema)
                                     :schema   schema}]))
        {:base-state schemas/BaseState
         :sub-result schemas/SubResult}))

;; ---------------------------------------------------------------------------
;; Default reporter
;; ---------------------------------------------------------------------------

(defn- default-thrower
  [type {:keys [entry value context fn-name op schema-key] :as data}]
  (let [humanized (me/humanize ((:explain entry) value))
        label (or fn-name context (some-> schema-key name) (some-> op name))]
    (throw (ex-info (str "Schema violation in " label ": " type)
                    {:type type
                     :errors humanized
                     :label label
                     :data (dissoc data :value :entry)}))))

;; ---------------------------------------------------------------------------
;; Instrumentation
;; ---------------------------------------------------------------------------

(defn start!
  "Enable Malli schema validation on all GFI boundaries.
   Options:
     :report — (fn [type data]) error reporter (default: throws with humanized errors)
     :scope  — #{:output :input} validation scope (default: both)"
  [& [opts]]
  (let [report (or (:report opts) default-thrower)
        scope  (or (:scope opts) #{:output :input})

        validating
        (fn [gf op args key opts]
          (let [result (dyn/run-dispatched* gf op args key opts)]
            (when (:output scope)
              (when-let [entry (op->validator op)]
                (when-not ((:validate entry) result)
                  (report ::invalid-output
                    {:op op :entry entry :value result
                     :fn-name (str "DynamicGF/" (name op))}))))
            result))]

    (reset! dyn/dispatch-fn validating)

    (when (:input scope)
      (reset! rt/validate-fn
        (fn [schema-key value context]
          (when-let [entry (key->validator schema-key)]
            (when-not ((:validate entry) value)
              (report ::invalid-input
                {:schema-key schema-key :entry entry
                 :value value :context context}))))))

    (println "genmlx.dev: schema validation enabled"
             (str "(" (str/join ", " (map name scope)) ")"))
    :started))

(defn stop!
  "Disable all schema validation. Returns to zero-cost dispatch."
  []
  (reset! dyn/dispatch-fn dyn/run-dispatched*)
  (reset! rt/validate-fn (fn [_ _ _]))
  (println "genmlx.dev: schema validation disabled")
  :stopped)
