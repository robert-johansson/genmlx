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
            [genmlx.diff :as diff]
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
  "Realize an MLX scalar, returning ##NaN on error.
   Handles plain numbers (e.g., delta distribution values) directly."
  [x]
  (if (number? x) x (try (h/realize x) (catch :default _ ##NaN))))

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

;; ===========================================================================
;; Gap checkers — thesis §8.0b gaps
;; ===========================================================================

;; ---------------------------------------------------------------------------
;; Gap 1: Partial-constraint GENERATE
;; [T] §2.3.1 p68, Algorithm 2 (p79)
;;
;; generate(P, x, σ_partial) where σ constrains a strict subset of addresses.
;; Constrained addresses use the constraint value; unconstrained are sampled.
;; The importance weight equals the sum of log-probs at constrained addresses:
;;
;;   weight = Σ_{a ∈ constrained} log p(τ_a | parents_a)
;;
;; This equals project(resulting-trace, constrained-selection) because project
;; computes the same sum of conditional log-densities at selected addresses.
;;
;; Tolerance: 1e-3 (generate and project may traverse different code paths
;; with float32 accumulation).
;; ---------------------------------------------------------------------------

(defn check-partial-constraint-weight
  "Law: generate(gf, args, partial-σ).weight = project(result-trace, constrained-sel)"
  [gf args constrained-addrs]
  (try
    (let [;; Simulate to get values for building partial constraints
          trace (p/simulate gf args)
          ;; Build partial choicemap with only constrained addresses
          partial-cm (reduce (fn [cm addr]
                               (let [sub (cm/get-submap (:choices trace) addr)]
                                 (if (cm/has-value? sub)
                                   (cm/set-value cm addr (cm/get-value sub))
                                   cm)))
                             cm/EMPTY
                             constrained-addrs)
          ;; Generate with partial constraints
          gen-result (p/generate gf args partial-cm)
          gen-trace (:trace gen-result)
          w (safe-realize (:weight gen-result))
          ;; Project the resulting trace at the constrained selection
          constrained-sel (sel/from-set constrained-addrs)
          proj (safe-realize (p/project gf gen-trace constrained-sel))
          ;; Verify constrained values are preserved
          values-ok? (every?
                       (fn [addr]
                         (let [gen-v (safe-realize (cm/get-value
                                                    (cm/get-submap (:choices gen-trace) addr)))
                               orig-v (safe-realize (cm/get-value
                                                      (cm/get-submap partial-cm addr)))]
                           (h/close? gen-v orig-v 1e-6)))
                       constrained-addrs)]
      (result
        (and values-ok?
             (h/finite? w)
             (h/finite? proj)
             (h/close? w proj 1e-3))
        "partial-constraint-weight"
        (str "weight=" w " project=" proj
             " diff=" (js/Math.abs (- w proj))
             " values-preserved=" values-ok?)))
    (catch :default e
      (result false "partial-constraint-weight" (str e)))))

;; ---------------------------------------------------------------------------
;; Gap 2: Argument gradients
;; [T] §2.3.1 p72, Eq 2.10: ∇_x g(x,τ) = ∇_x log p(τ;x) + J(x,τ)v
;;
;; The GRADIENT operation includes gradients w.r.t. model arguments x.
;; For a model with arg x and choices τ:
;;
;;   ∂/∂x log p(τ; x) via AD should match finite-difference approximation.
;;
;; AD: (mx/grad (fn [x-arr] (:weight (p/generate model [x-arr] τ))))(x)
;; FD: [weight(x+h) - weight(x-h)] / (2h)
;;
;; Tolerance: Combined absolute (5e-2) and relative (5e-2), same as
;; choice gradient tests. See gfi_gradient_test.cljs for derivation.
;; ---------------------------------------------------------------------------

(defn check-argument-gradient
  "Law: ∂/∂x_i score(τ; x) via AD matches symmetric FD."
  [gf args]
  (try
    (let [trace (p/simulate gf args)
          choices (:choices trace)
          args-v (vec args)
          h 1e-3
          per-arg
          (mapv
            (fn [i]
              (let [x-val (nth args-v i)
                    x-num (if (number? x-val) x-val (safe-realize x-val))
                    x-arr (mx/scalar x-num)
                    ;; AD gradient
                    score-fn (fn [x-a]
                               (:weight (p/generate gf (assoc args-v i x-a) choices)))
                    ad-g (safe-realize ((mx/grad score-fn) x-arr))
                    ;; Symmetric FD gradient
                    sp (safe-realize
                         (:weight (p/generate gf
                                    (assoc args-v i (mx/scalar (+ x-num h)))
                                    choices)))
                    sm (safe-realize
                         (:weight (p/generate gf
                                    (assoc args-v i (mx/scalar (- x-num h)))
                                    choices)))
                    fd-g (/ (- sp sm) (* 2.0 h))]
                {:i i :ad ad-g :fd fd-g}))
            (range (count args-v)))
          all-match?
          (every?
            (fn [{:keys [ad fd]}]
              (let [abs-diff (js/Math.abs (- ad fd))
                    max-mag (js/Math.max (js/Math.abs ad) (js/Math.abs fd))]
                (or ;; Both near zero
                    (and (< (js/Math.abs ad) 1e-6)
                         (< (js/Math.abs fd) 1e-6))
                    ;; Absolute match
                    (< abs-diff 5e-2)
                    ;; Relative match
                    (and (> max-mag 1e-6)
                         (< (/ abs-diff max-mag) 5e-2)))))
            per-arg)]
      (result all-match?
              "argument-gradient"
              (str (mapv (fn [{:keys [i ad fd]}]
                           (str "arg" i ":ad=" (.toFixed ad 6)
                                " fd=" (.toFixed fd 6)
                                " diff=" (.toFixed (js/Math.abs (- ad fd)) 6)))
                         per-arg))))
    (catch :default e
      (result false "argument-gradient" (str e)))))

;; ---------------------------------------------------------------------------
;; Gap 4: Arg-dependent density via assess
;; [T] §2.3.1 p71
;;
;; For models with arguments, changing x changes the density log p(τ; x).
;; Verify: assess(model, [x1], τ) and assess(model, [x2], τ) are both
;; internally consistent, and generate(model, [x2], τ).weight matches.
;;
;; This exercises the argument-dependent density computation through
;; two independent code paths (assess and generate with full constraints).
;; ---------------------------------------------------------------------------

(defn check-assess-arg-consistency
  "Law: assess(gf, [x2], τ).weight = generate(gf, [x2], τ).weight for changed args"
  [gf x1 x2]
  (try
    (let [trace (p/simulate gf [x1])
          choices (:choices trace)
          ;; Assess at x2
          w-assess (safe-realize (:weight (p/assess gf [x2] choices)))
          ;; Generate at x2 with full constraints
          gen-result (p/generate gf [x2] choices)
          w-gen (safe-realize (:weight gen-result))
          s-gen (safe-realize (:score (:trace gen-result)))]
      (result
        (and (h/finite? w-assess) (h/finite? w-gen) (h/finite? s-gen)
             ;; assess and generate agree on log p(τ; x2)
             (h/close? w-assess w-gen 1e-3)
             ;; generate(full).weight = generate.score
             (h/close? w-gen s-gen 1e-3))
        "assess-arg-consistency"
        (str "assess=" w-assess " gen-w=" w-gen " gen-s=" s-gen
             " assess-gen-diff=" (js/Math.abs (- w-assess w-gen)))))
    (catch :default e
      (result false "assess-arg-consistency" (str e)))))

;; ---------------------------------------------------------------------------
;; Gap 5: Argdiffs equivalence
;; [T] Def 2.3.2 (p70), §2.3.1 p71
;;
;; update-with-diffs(P, t, σ, :unknown) must produce identical results
;; to update(P, t, σ). The :unknown argdiff means "args may have changed"
;; and should trigger full recomputation (same as regular update).
;;
;; For DynamicGF, the implementation delegates to p/update when argdiffs
;; is not no-change. For combinators, the implementation may use different
;; internal paths but must produce the same external result.
;;
;; Tolerance: 1e-4 (both paths compute the same thing; any difference
;; is from non-determinism in float32 accumulation order).
;; ---------------------------------------------------------------------------

(defn check-argdiffs-equivalence
  "Law: update-with-diffs(gf, t, σ, :unknown) ≡ update(gf, t, σ)"
  [gf args]
  (try
    (let [t1 (p/simulate gf args)
          t2 (p/simulate gf args)
          ;; Regular update
          upd (p/update gf t1 (:choices t2))
          upd-w (safe-realize (:weight upd))
          upd-s (safe-realize (:score (:trace upd)))
          ;; update-with-diffs with :unknown
          uwd (p/update-with-diffs gf t1 (:choices t2) :unknown)
          uwd-w (safe-realize (:weight uwd))
          uwd-s (safe-realize (:score (:trace uwd)))]
      (result
        (and (h/finite? upd-w) (h/finite? uwd-w)
             (h/close? upd-w uwd-w 1e-4)
             (h/finite? upd-s) (h/finite? uwd-s)
             (h/close? upd-s uwd-s 1e-4))
        "argdiffs-equivalence"
        (str "upd-w=" upd-w " uwd-w=" uwd-w
             " w-diff=" (js/Math.abs (- upd-w uwd-w))
             " upd-s=" upd-s " uwd-s=" uwd-s)))
    (catch :default e
      (result false "argdiffs-equivalence" (str e)))))

;; ---------------------------------------------------------------------------
;; Gap 5b: Argdiffs no-change fast path
;; [T] Def 2.3.2
;;
;; update-with-diffs(P, t, EMPTY, no-change) should return:
;;   {:trace t :weight 0 :discard EMPTY}
;;
;; When arguments haven't changed and no new constraints are provided,
;; the trace is unchanged. This is the fast-path optimization that
;; DynamicGF and combinators use to skip re-execution.
;; ---------------------------------------------------------------------------

(defn check-argdiffs-no-change
  "Law: update-with-diffs(gf, t, EMPTY, no-change) → weight=0, discard=EMPTY"
  [gf args]
  (try
    (let [trace (p/simulate gf args)
          {:keys [weight discard]} (p/update-with-diffs gf trace cm/EMPTY diff/no-change)
          w (safe-realize weight)]
      (result
        (and (h/close? 0.0 w 1e-6)
             (= discard cm/EMPTY))
        "argdiffs-no-change"
        (str "weight=" w " discard-empty=" (= discard cm/EMPTY))))
    (catch :default e
      (result false "argdiffs-no-change" (str e)))))
