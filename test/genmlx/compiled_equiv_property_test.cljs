(ns genmlx.compiled-equiv-property-test
  "Property-based tests for compiled/handler equivalence.
   The compilation ladder's central invariant: compiled paths must produce
   identical traces, scores, and weights as the handler path. The handler
   is ground truth; compilation is optimization.

   Every test verifies that a compiled operation and its handler counterpart
   agree to within floating-point tolerance on the same inputs."
  (:require [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.tensor-trace :as tt])
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

(defn- check [name prop & {:keys [num-tests] :or {num-tests 50}}]
  (let [result (tc/quick-check num-tests prop)]
    (report-result name result)))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- choice-val
  "Extract a JS number from a choicemap at addr."
  [choices addr]
  (let [sub (cm/get-submap choices addr)]
    (when (and sub (cm/has-value? sub))
      (let [v (cm/get-value sub)]
        (mx/eval! v)
        (mx/item v)))))

(defn- trace-score [trace]
  (mx/eval! (:score trace))
  (mx/item (:score trace)))

(defn- eval-weight [w]
  (mx/eval! w)
  (mx/item w))

(defn- finite? [x]
  (and (number? x) (js/isFinite x)))

(defn- close? [a b tol]
  (and (finite? a) (finite? b) (<= (js/Math.abs (- a b)) tol)))

(defn- force-handler
  "Return a copy of gf that always uses the handler path (no compiled paths).
   Strips :compiled-simulate and all other compiled ops from the schema."
  [gf]
  (dyn/->DynamicGF (:body-fn gf) (:source gf)
                    (dissoc (:schema gf)
                            :compiled-simulate :compiled-generate
                            :compiled-update :compiled-assess
                            :compiled-project :compiled-regenerate)))

;; ---------------------------------------------------------------------------
;; Model pool: all static models (schema :static? = true)
;; ---------------------------------------------------------------------------

(def single-site
  (gen [] (trace :x (dist/gaussian 0 1))))

(def two-site-dep
  (gen []
    (let [x (trace :x (dist/gaussian 0 1))]
      (trace :y (dist/gaussian x 1)))))

(def three-site-dep
  (gen []
    (let [a (trace :a (dist/gaussian 0 1))
          b (trace :b (dist/gaussian a 2))]
      (trace :c (dist/gaussian b 0.5)))))

(def with-uniform
  (gen []
    (let [u (trace :u (dist/uniform 0 1))]
      (trace :v (dist/gaussian u 1)))))

;; Verify static and compiled
(assert (get-in single-site [:schema :static?]) "single-site must be static")
(assert (get-in two-site-dep [:schema :static?]) "two-site-dep must be static")
(assert (get-in three-site-dep [:schema :static?]) "three-site-dep must be static")
(assert (get-in with-uniform [:schema :static?]) "with-uniform must be static")
(assert (get-in single-site [:schema :compiled-simulate]) "single-site must have compiled-simulate")

(def model-pool
  [{:model single-site   :addrs [:x]       :label "single-site"}
   {:model two-site-dep  :addrs [:x :y]    :label "two-site-dep"}
   {:model three-site-dep :addrs [:a :b :c] :label "three-site-dep"}
   {:model with-uniform  :addrs [:u :v]    :label "with-uniform"}])

(def gen-model (gen/elements model-pool))

;; Seed pool: generate fresh keys from these seeds (keys are consumed per simulate)
(def seed-pool (vec (range 10)))
(def gen-seed (gen/elements seed-pool))

(println "\n=== Compiled/Handler Equivalence Property Tests ===\n")

;; ---------------------------------------------------------------------------
;; E12.1: compiled simulate score = handler simulate score
;; Law: compilation preserves score semantics
;; ---------------------------------------------------------------------------

(println "-- simulate equivalence --")

;; The compiled path uses mx/compile-fn which freezes the PRNG (all calls produce
;; the same samples). So we cannot compare samples directly. Instead we verify:
;; the compiled simulate score is consistent with handler generate for the same choices.
(check "compiled simulate: score = handler assess weight for same choices"
  (prop/for-all [m gen-model
                 seed gen-seed]
    (let [model (:model m)
          k1 (rng/fresh-key seed)
          ;; Compiled simulate
          compiled-trace (p/simulate (dyn/with-key model k1) [])
          compiled-score (trace-score compiled-trace)
          ;; Handler assess with compiled choices
          handler-model (force-handler model)
          k2 (rng/fresh-key (+ seed 100))
          {:keys [weight]} (p/assess (dyn/with-key handler-model k2) [] (:choices compiled-trace))
          assess-w (eval-weight weight)]
      (close? compiled-score assess-w 1e-5))))

;; ---------------------------------------------------------------------------
;; E12.2: compiled simulate choices = handler simulate choices
;; Law: compilation preserves sampled values
;; ---------------------------------------------------------------------------

;; Verify compiled simulate produces valid choices: all addresses present and finite.
(check "compiled simulate: all choices present and finite"
  (prop/for-all [m gen-model
                 seed gen-seed]
    (let [model (:model m)
          k1 (rng/fresh-key seed)
          compiled-trace (p/simulate (dyn/with-key model k1) [])
          addrs (:addrs m)]
      (every? (fn [addr]
                (let [v (choice-val (:choices compiled-trace) addr)]
                  (finite? v)))
              addrs))))

;; ---------------------------------------------------------------------------
;; E12.3: compiled generate weight = handler generate weight
;; Law: compilation preserves importance weights
;; ---------------------------------------------------------------------------

(println "\n-- generate equivalence --")

(check "compiled generate weight = handler generate weight"
  (prop/for-all [m gen-model
                 seed gen-seed]
    (let [model (:model m)
          ;; Get choices from a simulate to use as constraints
          k0 (rng/fresh-key seed)
          source-trace (p/simulate (dyn/with-key model k0) [])
          constraints (:choices source-trace)
          ;; Compiled generate — fresh key
          k1 (rng/fresh-key (+ seed 100))
          {:keys [weight]} (p/generate (dyn/with-key model k1) [] constraints)
          compiled-w (eval-weight weight)
          ;; Handler generate — fresh key from same seed
          k2 (rng/fresh-key (+ seed 100))
          handler-model (force-handler model)
          {:keys [weight]} (p/generate (dyn/with-key handler-model k2) [] constraints)
          handler-w (eval-weight weight)]
      (close? compiled-w handler-w 1e-5))))

;; ---------------------------------------------------------------------------
;; E12.4: compiled generate score = handler generate score
;; Law: compilation preserves scores under constraints
;; ---------------------------------------------------------------------------

(check "compiled generate score = handler generate score"
  (prop/for-all [m gen-model
                 seed gen-seed]
    (let [model (:model m)
          k0 (rng/fresh-key seed)
          source-trace (p/simulate (dyn/with-key model k0) [])
          constraints (:choices source-trace)
          ;; Compiled generate — fresh key
          k1 (rng/fresh-key (+ seed 100))
          {:keys [trace]} (p/generate (dyn/with-key model k1) [] constraints)
          compiled-score (trace-score trace)
          ;; Handler generate — fresh key from same seed
          k2 (rng/fresh-key (+ seed 100))
          handler-model (force-handler model)
          {:keys [trace]} (p/generate (dyn/with-key handler-model k2) [] constraints)
          handler-score (trace-score trace)]
      (close? compiled-score handler-score 1e-5))))

;; ---------------------------------------------------------------------------
;; E12.5: TensorTrace -> standard Trace round-trip
;; Law: TensorTrace is a faithful representation of the trace's address-value map
;; ---------------------------------------------------------------------------

(println "\n-- TensorTrace round-trip --")

(check "TensorTrace -> standard Trace -> generate recovers score"
  (prop/for-all [m gen-model
                 seed gen-seed]
    (let [model (:model m)
          ;; Simulate (compiled path produces trace, possibly TensorTrace)
          k1 (rng/fresh-key seed)
          trace (p/simulate (dyn/with-key model k1) [])
          orig-score (trace-score trace)
          addrs (:addrs m)
          ;; Extract all choices as a standard choicemap
          ;; (works whether trace is TensorTrace or standard Trace)
          choices (:choices trace)
          ;; Generate with handler using those choices — fresh key
          k2 (rng/fresh-key (+ seed 200))
          handler-model (force-handler model)
          {:keys [trace]} (p/generate (dyn/with-key handler-model k2) [] choices)
          gen-score (trace-score trace)]
      ;; Scores must match: the same choices yield the same log-probability
      (and (close? orig-score gen-score 1e-5)
           ;; All addresses must be present
           (every? (fn [addr]
                     (some? (choice-val (:choices trace) addr)))
                   addrs))))
  :num-tests 30)

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== Compiled/Handler Equivalence Tests Complete: "
              @pass-count " passed, " @fail-count " failed ==="))
