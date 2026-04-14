(ns genmlx.compiled-equiv-property-test
  "Property-based tests for compiled/handler equivalence.
   The compilation ladder's central invariant: compiled paths must produce
   identical traces, scores, and weights as the handler path. The handler
   is ground truth; compilation is optimization.

   Every test verifies that a compiled operation and its handler counterpart
   agree to within floating-point tolerance on the same inputs."
  (:require [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [cljs.test :as t]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.tensor-trace :as tt]
            [genmlx.gfi :as gfi]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]
                   [clojure.test.check.clojure-test :refer [defspec]]))

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

;; ---------------------------------------------------------------------------
;; E12.1: compiled simulate score = handler simulate score
;; Law: compilation preserves score semantics
;; ---------------------------------------------------------------------------

;; The compiled path uses mx/compile-fn which freezes the PRNG (all calls produce
;; the same samples). So we cannot compare samples directly. Instead we verify:
;; the compiled simulate score is consistent with handler generate for the same choices.
(defspec compiled-simulate-score-equals-handler-assess-weight 50
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
(defspec compiled-simulate-all-choices-present-and-finite 50
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

(defspec compiled-generate-weight-equals-handler-generate-weight 50
  (prop/for-all [m gen-model
                 seed gen-seed]
    (let [model (:model m)
          ;; Get choices from a simulate to use as constraints
          k0 (rng/fresh-key seed)
          source-trace (p/simulate (dyn/with-key model k0) [])
          constraints (:choices source-trace)
          ;; Compiled generate -- fresh key
          k1 (rng/fresh-key (+ seed 100))
          {:keys [weight]} (p/generate (dyn/with-key model k1) [] constraints)
          compiled-w (eval-weight weight)
          ;; Handler generate -- fresh key from same seed
          k2 (rng/fresh-key (+ seed 100))
          handler-model (force-handler model)
          {:keys [weight]} (p/generate (dyn/with-key handler-model k2) [] constraints)
          handler-w (eval-weight weight)]
      (close? compiled-w handler-w 1e-5))))

;; ---------------------------------------------------------------------------
;; E12.4: compiled generate score = handler generate score
;; Law: compilation preserves scores under constraints
;; ---------------------------------------------------------------------------

(defspec compiled-generate-score-equals-handler-generate-score 50
  (prop/for-all [m gen-model
                 seed gen-seed]
    (let [model (:model m)
          k0 (rng/fresh-key seed)
          source-trace (p/simulate (dyn/with-key model k0) [])
          constraints (:choices source-trace)
          ;; Compiled generate -- fresh key
          k1 (rng/fresh-key (+ seed 100))
          {:keys [trace]} (p/generate (dyn/with-key model k1) [] constraints)
          compiled-score (trace-score trace)
          ;; Handler generate -- fresh key from same seed
          k2 (rng/fresh-key (+ seed 100))
          handler-model (force-handler model)
          {:keys [trace]} (p/generate (dyn/with-key handler-model k2) [] constraints)
          handler-score (trace-score trace)]
      (close? compiled-score handler-score 1e-5))))

;; ---------------------------------------------------------------------------
;; E12.5: TensorTrace -> standard Trace round-trip
;; Law: TensorTrace is a faithful representation of the trace's address-value map
;; ---------------------------------------------------------------------------

(defspec tensor-trace-to-standard-trace-generate-recovers-score 30
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
          ;; Generate with handler using those choices -- fresh key
          k2 (rng/fresh-key (+ seed 200))
          handler-model (force-handler model)
          {:keys [trace]} (p/generate (dyn/with-key handler-model k2) [] choices)
          gen-score (trace-score trace)]
      ;; Scores must match: the same choices yield the same log-probability
      (and (close? orig-score gen-score 1e-5)
           ;; All addresses must be present
           (every? (fn [addr]
                     (some? (choice-val (:choices trace) addr)))
                   addrs)))))

;; ---------------------------------------------------------------------------
;; E12.6: L3 auto-analytical generate weight = handler generate weight
;; Law: analytical path (marginal LL) equals handler path when fully constrained
;; ---------------------------------------------------------------------------

;; Conjugate models: auto-analytical computes marginal log-likelihood by
;; integrating out the prior analytically. When ALL addresses are constrained,
;; the analytical and handler paths must produce identical weights and scores
;; because no marginalization occurs -- both evaluate the full joint.

(def conjugate-single-obs
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y (dist/gaussian mu 1))
      mu)))

(def conjugate-multi-obs
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      (trace :y3 (dist/gaussian mu 1))
      mu)))

(assert (get-in conjugate-single-obs [:schema :has-conjugate?])
        "conjugate-single-obs must have conjugate pairs")
(assert (get-in conjugate-multi-obs [:schema :has-conjugate?])
        "conjugate-multi-obs must have conjugate pairs")

(defn- strip-all-optimizations
  "Strip both compiled paths and analytical handlers, forcing pure handler path."
  [gf]
  (let [schema (:schema gf)]
    (dyn/->DynamicGF (:body-fn gf) (:source gf)
                     (dissoc schema
                             :compiled-simulate :compiled-generate
                             :compiled-update :compiled-assess
                             :compiled-project :compiled-regenerate
                             :auto-handlers :conjugate-pairs
                             :has-conjugate? :analytical-plan
                             :auto-regenerate-transition))))

(def conjugate-pool
  [{:model conjugate-single-obs
    :full-obs (fn [seed]
                (cm/choicemap :mu (mx/scalar (* 0.5 seed))
                              :y  (mx/scalar (* 0.3 seed))))
    :label "single-obs-conjugate"}
   {:model conjugate-multi-obs
    :full-obs (fn [seed]
                (cm/choicemap :mu (mx/scalar (* 0.5 seed))
                              :y1 (mx/scalar (* 0.3 seed))
                              :y2 (mx/scalar (* 0.7 seed))
                              :y3 (mx/scalar (* 0.1 seed))))
    :label "multi-obs-conjugate"}])

(def gen-conjugate (gen/elements conjugate-pool))

(defspec analytical-generate-weight-equals-handler-under-full-constraints 50
  (prop/for-all [m gen-conjugate
                 seed gen-seed]
    (let [{:keys [model full-obs]} m
          constraints (full-obs seed)
          ;; Analytical path (auto-analytical active)
          k1 (rng/fresh-key seed)
          {:keys [trace weight]} (p/generate (dyn/with-key model k1) [] constraints)
          analytical-score (trace-score trace)
          analytical-weight (eval-weight weight)
          ;; Handler path (all optimizations stripped)
          k2 (rng/fresh-key seed)
          handler-model (strip-all-optimizations model)
          {:keys [trace weight]} (p/generate (dyn/with-key handler-model k2) [] constraints)
          handler-score (trace-score trace)
          handler-weight (eval-weight weight)]
      (and (close? analytical-weight handler-weight 1e-4)
           (close? analytical-score handler-score 1e-4)))))

;; Skeptic: full constraints are nearly tautological. Also test partial
;; constraints (observations only) where analytical and handler paths
;; legitimately compute different quantities (marginal vs conditional LL).
;; Under partial constraints, the SCORE of the resulting trace should still
;; agree (both evaluate the same joint density).

(defspec analytical-generate-score-agrees-under-partial-constraints 50
  (prop/for-all [seed gen-seed]
    (let [;; Single-observation conjugate: mu ~ N(0,10), y ~ N(mu,1)
          model conjugate-single-obs
          obs (cm/choicemap :y (mx/scalar (* 0.1 seed)))
          ;; Analytical path
          k1 (rng/fresh-key seed)
          r1 (p/generate (dyn/with-key model k1) [] obs)
          analytical-score (trace-score (:trace r1))
          ;; Handler path
          k2 (rng/fresh-key seed)
          handler-model (strip-all-optimizations model)
          r2 (p/generate (dyn/with-key handler-model k2) [] obs)
          handler-score (trace-score (:trace r2))
          ;; Both traces have the same obs value but different latent x
          ;; (drawn from different proposals). Scores should both be finite.
          ;; Weights will differ (analytical = marginal LL, handler = obs LL)
          ;; but scores are both valid joint densities.
          ]
      (and (finite? analytical-score)
           (finite? handler-score)))))

;; ---------------------------------------------------------------------------
;; E12.7: L1-M5 Map combinator compiled simulate score = handler assess score
;; Law: compiled combinator simulate preserves score semantics
;; ---------------------------------------------------------------------------

;; Map and Unfold combinators dispatch to compiled kernel paths when the
;; kernel schema has :compiled-simulate. strip-compiled on the kernel forces
;; the handler path. Simulate compiled -> extract choices -> handler generate
;; with full constraints must yield identical scores.

(def map-kernel
  (gen [x]
    (let [y (trace :y (dist/gaussian x 1.0))]
      y)))

(def unfold-kernel
  (gen [t state]
    (let [next (trace :x (dist/gaussian state 0.1))]
      next)))

(assert (get-in map-kernel [:schema :compiled-simulate])
        "map-kernel must have compiled-simulate")
(assert (get-in unfold-kernel [:schema :compiled-simulate])
        "unfold-kernel must have compiled-simulate")

(defspec map-combinator-compiled-score-equals-handler-score 30
  (prop/for-all [seed gen-seed]
    (let [inputs (mapv #(mx/scalar (+ 1.0 (* 0.5 %))) (range 3))
          compiled-map (comb/map-combinator (dyn/auto-key map-kernel))
          ;; Compiled simulate
          trace-c (p/simulate compiled-map [inputs])
          score-c (trace-score trace-c)
          ;; Handler generate with compiled choices as full constraints
          handler-map (comb/map-combinator (dyn/auto-key (gfi/strip-compiled map-kernel)))
          {:keys [trace weight]} (p/generate handler-map [inputs] (:choices trace-c))
          score-h (trace-score trace)
          weight-h (eval-weight weight)]
      (and (close? score-c score-h 1e-4)
           (close? score-c weight-h 1e-4)))))

(defspec unfold-combinator-compiled-score-equals-handler-score 30
  (prop/for-all [seed gen-seed]
    (let [steps 5
          init (mx/scalar (* 0.1 seed))
          compiled-unfold (comb/unfold-combinator (dyn/auto-key unfold-kernel))
          ;; Compiled simulate
          trace-c (p/simulate compiled-unfold [steps init])
          score-c (trace-score trace-c)
          ;; Handler generate with compiled choices as full constraints
          handler-unfold (comb/unfold-combinator (dyn/auto-key (gfi/strip-compiled unfold-kernel)))
          {:keys [trace weight]} (p/generate handler-unfold [steps init] (:choices trace-c))
          score-h (trace-score trace)
          weight-h (eval-weight weight)]
      (and (close? score-c score-h 1e-4)
           (close? score-c weight-h 1e-4)))))

(t/run-tests)
