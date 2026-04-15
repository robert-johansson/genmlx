(ns genmlx.gfi-law-checkers
  "Reusable GFI algebraic law checkers for property-based testing.
   Each function takes a generative function and verifies one algebraic
   law from Cusumano-Towner 2020 PhD thesis, Chapter 2.

   Every checker returns {:pass? bool :law \"name\" :detail \"...\"}
   so failures are informative. No checker throws."
  (:require [clojure.set :as set]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.test-helpers :as h]))

;; ---------------------------------------------------------------------------
;; Internal helpers
;; ---------------------------------------------------------------------------

(defn- result
  "Build a checker result map."
  [pass? law detail]
  {:pass? pass? :law law :detail detail})

(defn- safe-realize
  "Realize an MLX scalar, returning ##NaN on error."
  [x]
  (try (h/realize x) (catch :default _ ##NaN)))

;; ---------------------------------------------------------------------------
;; Law 1: simulate produces valid trace
;; [T] Def 2.1.16, Section 2.3.1 SIMULATE
;;
;; simulate(P, x) draws tau ~ p(.; x) and returns a trace with:
;;   - choices: the random choices tau
;;   - score: log p(tau; x), which must be finite for tau in supp(p)
;;   - all expected addresses present
;; ---------------------------------------------------------------------------

(defn check-simulate-produces-valid-trace
  "Law: simulate returns a trace with finite score and all expected addresses."
  [gf args addrs]
  (try
    (let [trace  (p/simulate gf args)
          score  (safe-realize (:score trace))
          choices (:choices trace)
          present (set (map first (cm/addresses choices)))
          missing (set/difference addrs present)]
      (result
       (and (h/finite? score)
            (some? choices)
            (empty? missing))
       "simulate-produces-valid-trace"
       (str "score=" score
            (when (seq missing) (str " missing-addrs=" missing)))))
    (catch :default e
      (result false "simulate-produces-valid-trace" (str e)))))

;; ---------------------------------------------------------------------------
;; Law 2: simulate score consistency
;; [T] Section 2.3.1, LOGPDF / get_score
;;
;; trace.score = log p(tau; x)
;; assess(P, x, tau).weight = log p(tau; x)
;; Therefore: trace.score = assess(P, x, trace.choices).weight
;;
;; Both compute the same joint log-density, just via different code paths.
;; Tolerance: 1e-4 (float32 accumulation across trace sites; both paths
;; sum the same log-prob terms, so error is bounded by N * eps_float32
;; where N = number of trace sites and eps ~ 1e-7).
;; ---------------------------------------------------------------------------

(defn check-simulate-score-law
  "Law: trace.score = assess(trace.choices).weight"
  [gf args]
  (try
    (let [trace   (p/simulate gf args)
          s       (safe-realize (:score trace))
          {:keys [weight]} (p/assess gf args (:choices trace))
          w       (safe-realize weight)]
      (result
       (and (h/finite? s) (h/finite? w) (h/close? s w 1e-4))
       "simulate-score"
       (str "score=" s " assess-weight=" w " diff=" (js/Math.abs (- s w)))))
    (catch :default e
      (result false "simulate-score" (str e)))))

;; ---------------------------------------------------------------------------
;; Law 3: generate with empty constraints has weight 0
;; [D] generate with empty constraints
;;
;; generate(P, x, {}) is equivalent to simulate: no constraints imposed,
;; so the importance weight = log(p_bar({})) where p_bar({}) = 1
;; (marginalizing over all choices integrates to 1 by normalization).
;; weight = log(1) = 0.0
;;
;; Tolerance: 1e-4 (the weight accumulator starts at 0 and no constraints
;; contribute; any deviation is pure floating-point noise).
;; ---------------------------------------------------------------------------

(defn check-generate-empty-weight
  "Law: generate(gf, args, EMPTY).weight = 0.0"
  [gf args]
  (try
    (let [{:keys [weight]} (p/generate gf args cm/EMPTY)
          w (safe-realize weight)]
      (result
       (and (h/finite? w) (h/close? 0.0 w 1e-4))
       "generate-empty-weight"
       (str "weight=" w)))
    (catch :default e
      (result false "generate-empty-weight" (str e)))))

;; ---------------------------------------------------------------------------
;; Law 4: generate with full constraints has weight = score
;; [T] Section 2.3.1 GENERATE
;;
;; When sigma fully specifies tau (every address constrained):
;;   generate(P, x, tau).weight = log p(tau; x) = trace.score
;;
;; Derivation: The importance weight for fully constrained generation is
;;   w = p(tau; x) / q(tau; x, sigma)
;; where q is the internal proposal. When sigma = tau, the proposal is
;; forced to produce exactly tau, so q(tau; x, tau) = 1 (delta measure),
;; giving w = p(tau; x), hence log w = log p(tau; x) = score.
;;
;; Tolerance: 1e-3 (simulate and generate traverse different code paths;
;; compiled vs handler dispatch can introduce float32 ordering differences
;; in the log-prob sum).
;; ---------------------------------------------------------------------------

(defn check-generate-full-weight
  "Law: generate(gf, args, trace.choices).weight = trace.score"
  [gf args]
  (try
    (let [trace (p/simulate gf args)
          s     (safe-realize (:score trace))
          {:keys [trace weight]} (p/generate gf args (:choices trace))
          w     (safe-realize weight)
          gs    (safe-realize (:score trace))]
      (result
       (and (h/finite? s) (h/finite? w) (h/finite? gs)
            (h/close? s w 1e-3)
            (h/close? gs w 1e-3))
       "generate-full-weight"
       (str "sim-score=" s " gen-weight=" w " gen-score=" gs)))
    (catch :default e
      (result false "generate-full-weight" (str e)))))

;; ---------------------------------------------------------------------------
;; Law 5: update with identical choices has weight 0
;; [T] Section 2.3.1 UPDATE, h_update(tau, tau) = (tau, {})
;;
;; update(P, t, t.choices).weight = log(p(tau; x) / p(tau; x)) = log(1) = 0
;;
;; The update weight is the log density ratio between the new and old traces.
;; When choices are identical, the ratio is 1.
;;
;; Tolerance: 1e-4 (the new score minus old score should cancel exactly
;; up to float32 rounding; any residual is from non-associativity of
;; floating-point addition).
;; ---------------------------------------------------------------------------

(defn check-update-identity-law
  "Law: update(gf, trace, same-choices).weight = 0.0"
  [gf args]
  (try
    (let [trace (p/simulate gf args)
          {:keys [weight]} (p/update gf trace (:choices trace))
          w     (safe-realize weight)]
      (result
       (and (h/finite? w) (h/close? 0.0 w 1e-4))
       "update-identity"
       (str "weight=" w)))
    (catch :default e
      (result false "update-identity" (str e)))))

;; ---------------------------------------------------------------------------
;; Law 5b: update density ratio
;; [T] Section 2.3.1 UPDATE, Proposition 2.3.1
;;
;; update(P, t1, t2.choices).weight = log p(t2; x) - log p(t1; x)
;;                                  = t2.score - t1.score
;;
;; When updating trace t1 to have choices from a different trace t2,
;; the weight is the log density ratio between the new and old traces.
;; This is the core importance weight property of the update operation.
;;
;; Tolerance: 1e-3 (involves two independent simulate calls plus the
;; update, three different code path traversals with float32 accumulation).
;; ---------------------------------------------------------------------------

(defn check-update-density-ratio
  "Law: update(gf, t1, t2.choices).weight = t2.score - t1.score"
  [gf args]
  (try
    (let [t1 (p/simulate gf args)
          t2 (p/simulate gf args)
          s1 (safe-realize (:score t1))
          s2 (safe-realize (:score t2))
          {:keys [weight]} (p/update gf t1 (:choices t2))
          w  (safe-realize weight)
          expected (- s2 s1)]
      (result
       (and (h/finite? s1) (h/finite? s2) (h/finite? w)
            (h/close? w expected 1e-3))
       "update-density-ratio"
       (str "weight=" w " expected=" expected " s1=" s1 " s2=" s2
            " diff=" (js/Math.abs (- w expected)))))
    (catch :default e
      (result false "update-density-ratio" (str e)))))

;; ---------------------------------------------------------------------------
;; Law 6: project(all) = score
;; [D] project, [T] Section 2.3.1
;;
;; project(P, t, all) = sum_{a in all addrs} log p(tau_a | parents)
;;                     = log p(tau; x)
;;                     = trace.score
;;
;; This is the definition of the joint density as a product (sum in log
;; space) of conditional densities at each trace site.
;;
;; Tolerance: 1e-4 (project and score compute the same sum of log-probs,
;; possibly in different traversal order; float32 non-associativity).
;; ---------------------------------------------------------------------------

(defn check-project-all-equals-score
  "Law: project(gf, trace, select-all) = trace.score"
  [gf args]
  (try
    (let [trace (p/simulate gf args)
          s     (safe-realize (:score trace))
          proj  (safe-realize (p/project gf trace sel/all))]
      (result
       (and (h/finite? s) (h/finite? proj) (h/close? proj s 1e-4))
       "project-all-equals-score"
       (str "score=" s " project-all=" proj " diff=" (js/Math.abs (- s proj)))))
    (catch :default e
      (result false "project-all-equals-score" (str e)))))

;; ---------------------------------------------------------------------------
;; Law 7: project(none) = 0
;; [D] project
;;
;; project(P, t, none) = sum over empty set of addresses = 0
;;
;; No addresses are selected, so no log-density terms are included.
;;
;; Tolerance: 1e-4 (should be exactly 0; any deviation is implementation
;; artifact in the empty-sum accumulator).
;; ---------------------------------------------------------------------------

(defn check-project-none-equals-zero
  "Law: project(gf, trace, select-none) = 0.0"
  [gf args]
  (try
    (let [trace (p/simulate gf args)
          proj  (safe-realize (p/project gf trace sel/none))]
      (result
       (and (h/finite? proj) (h/close? 0.0 proj 1e-4))
       "project-none-equals-zero"
       (str "project-none=" proj)))
    (catch :default e
      (result false "project-none-equals-zero" (str e)))))

;; ---------------------------------------------------------------------------
;; Law 8: project decomposition (score additivity)
;; [T] Section 2.3.1, score additivity over disjoint selections
;;
;; sum_i project(t, {addr_i}) = trace.score
;;
;; By the chain rule of probability:
;;   log p(tau; x) = sum_i log p(tau_i | tau_{<i}, x)
;;
;; Each project({addr_i}) returns log p(tau_i | tau_{<i}, x), the
;; log-density contribution of site i conditioned on its parents.
;; This holds for ALL models (independent or dependent) because the
;; chain rule is universal.
;;
;; Tolerance: 1e-3 (summing N float32 terms introduces O(N * eps)
;; error from non-associativity; for N ~ 10 sites, error ~ 1e-6,
;; but we use 1e-3 to allow for models with larger site counts
;; and more complex dependency structures).
;; ---------------------------------------------------------------------------

(defn check-project-decomposition
  "Law: sum of individual site projections = trace.score"
  [gf args addrs]
  (try
    (let [trace (p/simulate gf args)
          s     (safe-realize (:score trace))
          proj-sum (reduce
                    (fn [acc addr]
                      (let [proj (safe-realize
                                  (p/project gf trace (sel/select addr)))]
                        (+ acc proj)))
                    0.0
                    addrs)]
      (result
       (and (h/finite? s)
            (h/finite? proj-sum)
            (h/close? proj-sum s 1e-3))
       "project-decomposition"
       (str "score=" s " proj-sum=" proj-sum
            " diff=" (js/Math.abs (- s proj-sum)))))
    (catch :default e
      (result false "project-decomposition" (str e)))))

;; ---------------------------------------------------------------------------
;; Law 8b: project decomposition for hierarchical addresses
;; Same chain-rule law as above, but supports address paths like [0 :a]
;; from combinator traces. Builds hierarchical selections from the
;; actual address structure in the trace's choicemap.
;; ---------------------------------------------------------------------------

(defn- path->selection
  "Convert an address path to a selection.
   [:a]       -> (sel/select :a)
   [0 :a]     -> (sel/hierarchical 0 (sel/select :a))
   [0 :z :y]  -> (sel/hierarchical 0 (sel/hierarchical :z (sel/select :y)))"
  [path]
  (if (= 1 (count path))
    (sel/select (first path))
    (reduce (fn [inner addr] (sel/hierarchical addr inner))
            (sel/select (last path))
            (reverse (butlast path)))))

(defn check-project-decomposition-hierarchical
  "Law: sum of individual leaf projections = trace.score (hierarchical addresses)"
  [gf args]
  (try
    (let [trace (p/simulate gf args)
          s     (safe-realize (:score trace))
          paths (cm/addresses (:choices trace))
          proj-sum (reduce
                    (fn [acc path]
                      (let [proj (safe-realize
                                  (p/project gf trace (path->selection path)))]
                        (+ acc proj)))
                    0.0
                    paths)]
      (result
       (and (h/finite? s)
            (h/finite? proj-sum)
            (h/close? proj-sum s 1e-3))
       "project-decomposition-hierarchical"
       (str "score=" s " proj-sum=" proj-sum " n-paths=" (count paths)
            " diff=" (js/Math.abs (- s proj-sum)))))
    (catch :default e
      (result false "project-decomposition-hierarchical" (str e)))))

;; ---------------------------------------------------------------------------
;; Law 9: propose-generate consistency
;; [D] propose
;;
;; propose(P, x) returns {choices, weight, retval} where:
;;   weight = log p(choices; x)
;;
;; generate(P, x, propose.choices) with full constraints returns:
;;   weight = log p(choices; x)
;;
;; Both weights must agree because they compute the same joint density.
;;
;; Tolerance: 1e-3 (propose and generate may use different internal
;; execution paths; the weight is the same mathematical quantity
;; computed via two code paths).
;; ---------------------------------------------------------------------------

(defn check-propose-generate-consistency
  "Law: propose(gf, args).weight = generate(gf, args, propose.choices).weight"
  [gf args]
  (try
    (let [{:keys [choices weight]} (p/propose gf args)
          pw    (safe-realize weight)
          {:keys [weight]} (p/generate gf args choices)
          gw    (safe-realize weight)]
      (result
       (and (h/finite? pw) (h/finite? gw) (h/close? pw gw 1e-3))
       "propose-generate-consistency"
       (str "propose-weight=" pw " generate-weight=" gw
            " diff=" (js/Math.abs (- pw gw)))))
    (catch :default e
      (result false "propose-generate-consistency" (str e)))))

;; ---------------------------------------------------------------------------
;; Composite checker
;; ---------------------------------------------------------------------------

(defn check-all-laws
  "Run all GFI law checkers on the given generative function.
   Returns a vector of result maps, one per law."
  [gf args addrs]
  [(check-simulate-produces-valid-trace gf args addrs)
   (check-simulate-score-law gf args)
   (check-generate-empty-weight gf args)
   (check-generate-full-weight gf args)
   (check-update-identity-law gf args)
   (check-update-density-ratio gf args)
   (check-project-all-equals-score gf args)
   (check-project-none-equals-zero gf args)
   (check-project-decomposition gf args addrs)
   (check-propose-generate-consistency gf args)])
