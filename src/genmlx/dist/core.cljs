(ns genmlx.dist.core
  "Foundation for distributions-as-data: single Distribution record,
   open multimethods, and GFI bridge functions."
  (:require [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.mlx :as mx]))

;; ---------------------------------------------------------------------------
;; Open multimethods — dispatch on (:type dist)
;; ---------------------------------------------------------------------------

(defmulti dist-sample   (fn [d _key] (:type d)))
(defmulti dist-log-prob (fn [d _value] (:type d)))
(defmulti dist-reparam  (fn [d _key] (:type d)))
(defmulti dist-support   (fn [d] (:type d)))

;; Defaults: helpful errors
(defmethod dist-reparam :default [d _]
  (throw (ex-info (str "Distribution " (:type d) " does not support reparameterized sampling")
                  {:type (:type d)})))

(defmethod dist-support :default [d]
  (throw (ex-info (str "Distribution " (:type d) " is not enumerable")
                  {:type (:type d)})))

;; ---------------------------------------------------------------------------
;; GFI bridge: distribution -> trace
;; ---------------------------------------------------------------------------

(defn dist-simulate [dist]
  (let [v  (dist-sample dist nil)
        lp (dist-log-prob dist v)]
    (tr/make-trace {:gen-fn dist :args [] :choices (cm/->Value v)
                    :retval v :score lp})))

(defn dist-generate [dist constraints]
  (if (cm/has-value? constraints)
    (let [v  (cm/get-value constraints)
          lp (dist-log-prob dist v)]
      {:trace (tr/make-trace {:gen-fn dist :args [] :choices (cm/->Value v)
                              :retval v :score lp})
       :weight lp})
    {:trace (dist-simulate dist) :weight (mx/scalar 0.0)}))

;; ---------------------------------------------------------------------------
;; THE single record for all distributions
;; ---------------------------------------------------------------------------

(defrecord Distribution [type params]
  p/IGenerativeFunction
  (simulate [this _] (dist-simulate this))

  p/IGenerate
  (generate [this _ constraints] (dist-generate this constraints)))
