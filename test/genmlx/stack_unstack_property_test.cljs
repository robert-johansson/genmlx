(ns genmlx.stack-unstack-property-test
  "Property-based tests for choicemap stack/unstack isomorphism.
   Verifies: round-trip identity, shape promotion to [N], shape
   demotion to scalar, and address structure preservation."
  (:require [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Test infrastructure
;; ---------------------------------------------------------------------------

(def ^:private pass-count (volatile! 0))
(def ^:private fail-count (volatile! 0))

(defn- report-result [name result]
  (if (:pass? result)
    (do (vswap! pass-count inc)
        (println "  PASS:" name (str "(" (:num-tests result) " trials)")))
    (do (vswap! fail-count inc)
        (println "  FAIL:" name)
        (println "    seed:" (:seed result))
        (when-let [s (get-in result [:shrunk :smallest])]
          (println "    shrunk:" s)))))

(defn- check [name prop & {:keys [num-tests] :or {num-tests 30}}]
  (let [result (tc/quick-check num-tests prop)]
    (report-result name result)))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- s [v] (mx/scalar (double v)))

(defn- close? [a b tol]
  (< (js/Math.abs (- a b)) tol))

(defn- leaf-values
  "Extract all leaf values from a choicemap as {path -> mx-value}."
  [cm-node]
  (reduce (fn [acc path]
            (assoc acc path (cm/get-choice cm-node path)))
          {}
          (cm/addresses cm-node)))

(defn- scalar-leaf?
  "True if value is a scalar MLX array (shape = [])."
  [v]
  (and (mx/array? v) (empty? (mx/shape v))))

;; ---------------------------------------------------------------------------
;; Pre-built pools (SCI shrink safety)
;; ---------------------------------------------------------------------------

(def simple-model
  (gen [] (trace :x (dist/gaussian (s 0) (s 1)))))

(def two-site-model
  (gen []
    (let [x (trace :x (dist/gaussian (s 0) (s 1)))]
      (trace :y (dist/gaussian x (s 1))))))

(def three-site-model
  (gen []
    (let [a (trace :a (dist/gaussian (s 0) (s 2)))
          b (trace :b (dist/gaussian a (s 1)))]
      (trace :c (dist/gaussian b (s 0.5))))))

(def model-pool [simple-model two-site-model three-site-model])
(def n-pool [3 5 10])

(def key-pool
  (let [root (rng/fresh-key)]
    (vec (rng/split-n root 5))))

;; ---------------------------------------------------------------------------
;; Helper: simulate N choicemaps from a model
;; ---------------------------------------------------------------------------

(defn- simulate-n-choices
  "Simulate N traces from model and extract their choicemaps."
  [model n base-key]
  (let [keys (rng/split-n base-key n)]
    (mapv (fn [k]
            (:choices (p/simulate (dyn/with-key model k) [])))
          keys)))

(println "\n=== Stack/Unstack Property Tests ===\n")

;; ---------------------------------------------------------------------------
;; E17.1: unstack(stack(cms)) = cms
;; Law: Stack and unstack are inverse operations — stacking N scalar
;;       choicemaps into one batched choicemap, then unstacking, recovers
;;       the original values at every address.
;; ---------------------------------------------------------------------------

(println "-- stack/unstack isomorphism --")

(check "unstack(stack(cms)) recovers original values"
  (prop/for-all [model (gen/elements model-pool)
                 n (gen/elements n-pool)
                 key (gen/elements key-pool)]
    (let [cms (simulate-n-choices model n key)
          stacked (cm/stack-choicemaps cms mx/stack)
          unstacked (cm/unstack-choicemap stacked n mx/index scalar-leaf?)
          ;; Compare each unstacked cm to original
          addrs (cm/addresses (first cms))]
      (every? true?
        (for [i (range n)
              path addrs]
          (let [orig-v (cm/get-choice (nth cms i) path)
                round-v (cm/get-choice (nth unstacked i) path)]
            (mx/eval! orig-v round-v)
            (close? (mx/item orig-v) (mx/item round-v) 1e-6))))))
  :num-tests 30)

;; ---------------------------------------------------------------------------
;; E17.2: stack — leaf shapes become [N]
;; Law: Stacking N scalar choicemaps promotes every leaf from shape []
;;       to shape [N] — the fundamental batching operation.
;; ---------------------------------------------------------------------------

(println "\n-- shape promotion --")

(check "stack: leaf shapes become [N]"
  (prop/for-all [model (gen/elements model-pool)
                 n (gen/elements n-pool)
                 key (gen/elements key-pool)]
    (let [cms (simulate-n-choices model n key)
          stacked (cm/stack-choicemaps cms mx/stack)
          addrs (cm/addresses stacked)]
      (every? true?
        (for [path addrs]
          (let [v (cm/get-choice stacked path)]
            (= [n] (vec (mx/shape v))))))))
  :num-tests 30)

;; ---------------------------------------------------------------------------
;; E17.3: unstack — leaf shapes become scalar
;; Law: Unstacking removes the batch dimension, recovering scalar leaves
;;       — the inverse of shape promotion.
;; ---------------------------------------------------------------------------

(println "\n-- shape demotion --")

(check "unstack: leaf shapes become scalar []"
  (prop/for-all [model (gen/elements model-pool)
                 n (gen/elements n-pool)
                 key (gen/elements key-pool)]
    (let [cms (simulate-n-choices model n key)
          stacked (cm/stack-choicemaps cms mx/stack)
          unstacked (cm/unstack-choicemap stacked n mx/index scalar-leaf?)]
      (every? true?
        (for [cm-i unstacked
              path (cm/addresses cm-i)]
          (let [v (cm/get-choice cm-i path)]
            (= [] (vec (mx/shape v))))))))
  :num-tests 30)

;; ---------------------------------------------------------------------------
;; E17.4: stack preserves addresses
;; Law: Stacking does not alter the tree structure — the set of address
;;       paths in the stacked choicemap equals the set in any constituent.
;; ---------------------------------------------------------------------------

(println "\n-- address preservation --")

(check "stack preserves address set"
  (prop/for-all [model (gen/elements model-pool)
                 n (gen/elements n-pool)
                 key (gen/elements key-pool)]
    (let [cms (simulate-n-choices model n key)
          stacked (cm/stack-choicemaps cms mx/stack)
          original-addrs (set (cm/addresses (first cms)))
          stacked-addrs (set (cm/addresses stacked))]
      (= original-addrs stacked-addrs)))
  :num-tests 30)

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== Stack/Unstack Property Tests Complete: "
              @pass-count " passed, " @fail-count " failed ==="))
(when (pos? @fail-count)
  (js/process.exit 1))
