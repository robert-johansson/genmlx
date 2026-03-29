(ns genmlx.gfi-laws-test
  "Property-based verification of GFI algebraic laws.

   Tests the laws defined in genmlx.gfi across diverse model families:
   - Single-site continuous (gaussian, uniform, exponential, beta)
   - Multi-site independent
   - Chain models (dependent sites)
   - Mixed discrete/continuous
   - Branching models (different address sets per execution)
   - Models with arguments

   Each defspec maps 1:1 to a law in genmlx.gfi/laws, cited by
   name and thesis/docs reference."
  (:require [cljs.test :as t]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.gfi :as gfi]
            [genmlx.gradients :as grad]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.util :as u]
            [genmlx.learning :as learn]
            [genmlx.test-helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]
                   [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Helpers — delegate to test-helpers with property-test defaults
;; ---------------------------------------------------------------------------

(defn- ev
  "Realize MLX scalar. Alias for h/realize."
  [x] (h/realize x))

(defn- close?
  "Like h/close? but with finiteness guard and wider default tolerance
   appropriate for property-based tests across diverse model families."
  ([a b] (close? a b 0.05))
  ([a b tol] (and (h/finite? a) (h/finite? b) (h/close? a b tol))))

;; ---------------------------------------------------------------------------
;; Model families
;; ---------------------------------------------------------------------------

;; Family 1: Single-site continuous
(def single-gaussian
  {:model (dyn/auto-key (gen [] (trace :x (dist/gaussian 0 1))))
   :args []
   :label "single-gaussian"})

(def single-uniform
  {:model (dyn/auto-key (gen [] (trace :x (dist/uniform 0 1))))
   :args []
   :label "single-uniform"})

(def single-exponential
  {:model (dyn/auto-key (gen [] (trace :x (dist/exponential 1))))
   :args []
   :label "single-exponential"})

(def single-beta
  {:model (dyn/auto-key (gen [] (trace :x (dist/beta-dist 2 5))))
   :args []
   :label "single-beta"})

;; Family 2: Multi-site independent
(def two-independent
  {:model (dyn/auto-key (gen []
                             (let [x (trace :x (dist/gaussian 0 1))
                                   y (trace :y (dist/uniform 0 1))]
                               x)))
   :args []
   :label "two-independent"})

(def three-independent
  {:model (dyn/auto-key (gen []
                             (let [a (trace :a (dist/gaussian 0 2))
                                   b (trace :b (dist/exponential 1))
                                   c (trace :c (dist/uniform -1 1))]
                               a)))
   :args []
   :label "three-independent"})

;; Family 3: Chain models (dependent sites)
(def gaussian-chain
  {:model (dyn/auto-key (gen []
                             (let [x (trace :x (dist/gaussian 0 1))]
                               (trace :y (dist/gaussian x 0.5)))))
   :args []
   :label "gaussian-chain"})

(def three-chain
  {:model (dyn/auto-key (gen []
                             (let [a (trace :a (dist/gaussian 0 1))
                                   b (trace :b (dist/gaussian a 1))]
                               (trace :c (dist/gaussian (mx/add a b) 0.5)))))
   :args []
   :label "three-chain"})

;; Family 4: Mixed discrete/continuous
(def mixed-disc-cont
  {:model (dyn/auto-key (gen []
                             (let [b (trace :b (dist/bernoulli 0.5))
                                   x (trace :x (dist/gaussian 0 1))]
                               x)))
   :args []
   :label "mixed-disc-cont"})

;; Family 5: Branching models (stochastic control flow)
(def branching-model
  {:model (dyn/auto-key (gen []
                             (let [coin (trace :coin (dist/bernoulli 0.3))]
                               (mx/eval! coin)
                               (if (pos? (mx/item coin))
                                 (trace :heads (dist/gaussian 10 1))
                                 (trace :tails (dist/uniform 0 1))))))
   :args []
   :label "branching"})

;; Family 6: Models with arguments
(def linear-regression
  {:model (dyn/auto-key (gen [x-val]
                             (let [slope (trace :slope (dist/gaussian 0 5))
                                   intercept (trace :intercept (dist/gaussian 0 5))]
                               (trace :y (dist/gaussian (mx/add (mx/multiply slope x-val)
                                                                intercept) 1)))))
   :args [(mx/scalar 2.0)]
   :label "linear-regression"})

;; Family 6b: More models with arguments (gradient coverage)
(def single-arg-model
  {:model (dyn/auto-key (gen [mu]
                             (trace :x (dist/gaussian mu 1))))
   :args [(mx/scalar 3.0)]
   :label "single-arg"})

(def two-arg-model
  {:model (dyn/auto-key (gen [mu sigma]
                             (trace :x (dist/gaussian mu sigma))))
   :args [(mx/scalar 0.0) (mx/scalar 2.0)]
   :label "two-arg"})

;; Family 7: Splice models (compositional)
(def splice-inner
  (gen [] (trace :a (dist/gaussian 0 1))))

(def splice-dependent
  {:model (dyn/auto-key
           (gen []
                (let [a (splice :inner splice-inner)]
                  (trace :b (dist/gaussian a 1)))))
   :args []
   :label "splice-dependent"})

(def splice-independent
  {:model (dyn/auto-key
           (gen []
                (let [_a (splice :inner splice-inner)]
                  (trace :b (dist/gaussian 0 1)))))
   :args []
   :label "splice-independent"})

(def splice-pool [splice-dependent splice-independent])

;; ---------------------------------------------------------------------------
;; Model pool and generator
;; ---------------------------------------------------------------------------

(def model-pool
  "Diverse model families for comprehensive GFI law testing.
   Includes branching-model for structured-density coverage."
  [single-gaussian single-uniform single-exponential single-beta
   two-independent three-independent
   gaussian-chain three-chain
   mixed-disc-cont
   branching-model
   linear-regression single-arg-model two-arg-model
   splice-dependent splice-independent])

;; Branching models need special handling for some laws (update/regenerate
;; may change control flow), so we keep them in a separate pool
(def non-branching-pool
  "Models with fixed address sets (safe for all laws)."
  [single-gaussian single-uniform single-exponential single-beta
   two-independent three-independent
   gaussian-chain three-chain
   mixed-disc-cont
   linear-regression single-arg-model two-arg-model
   splice-dependent splice-independent])

;; Multi-site models with flat addresses only (for decomposition laws that
;; assume per-leaf projection decomposes additively — not valid through splices)
(def multi-site-pool
  (filter #(let [t (p/simulate (:model %) (:args %))
                 addrs (cm/addresses (:choices t))]
             (and (> (count addrs) 1)
                  (every? (fn [path] (= 1 (count path))) addrs)))
          non-branching-pool))

(def gen-model
  "Generator: pick a random model from the full pool."
  (gen/elements model-pool))

(def gen-nonbranching
  "Generator: pick a model with fixed address set."
  (gen/elements non-branching-pool))

(def gen-multisite
  "Generator: pick a model with 2+ trace sites."
  (gen/elements multi-site-pool))

(def gen-splice
  "Generator: pick a splice model for compositionality tests."
  (gen/elements splice-pool))

;; ---------------------------------------------------------------------------
;; SIMULATE laws [T] Def 2.1.16, §2.3.1
;; ---------------------------------------------------------------------------

(defspec law:simulate-produces-trace 100
  ;; [T] Def 2.1.16, §2.3.1 SIMULATE
  ;; simulate(P, x) returns trace with finite score and correct gen-fn
  (prop/for-all [m gen-model]
                (let [t (p/simulate (:model m) (:args m))]
                  (and (some? (:choices t))
                       (= (:gen-fn t) (:model m))
                       (h/finite? (ev (:score t)))))))

(defspec law:simulate-score-is-log-density 100
  ;; [T] §2.3.1 LOGPDF, [D] get_score
  ;; trace.score = assess(P, x, tau).weight
  (prop/for-all [m gen-model]
                (let [t (p/simulate (:model m) (:args m))
                      s (ev (:score t))
                      {:keys [weight]} (p/assess (:model m) (:args m) (:choices t))
                      w (ev weight)]
                  (close? s w 0.01))))

(defspec law:halts-with-probability-one 100
  ;; [T] Def 2.1.16 — simulate(P, x) terminates with probability 1
  (prop/for-all [m gen-model]
                (let [t (p/simulate (:model m) (:args m))]
                  (some? t))))

;; ---------------------------------------------------------------------------
;; GENERATE laws [T] §2.3.1
;; ---------------------------------------------------------------------------

(defspec law:generate-empty-is-simulate 100
  ;; [D] generate with empty constraints = simulate
  ;; weight should be 0
  (prop/for-all [m gen-model]
                (let [{:keys [weight]} (p/generate (:model m) (:args m) cm/EMPTY)]
                  (close? 0.0 (ev weight) 0.01))))

(defspec law:generate-full-weight-equals-score 100
  ;; [T] §2.3.1 GENERATE
  ;; Fully constrained: weight = trace.score = log p-bar(sigma)
  (prop/for-all [m gen-nonbranching]
                (let [t (p/simulate (:model m) (:args m))
                      {:keys [trace weight]} (p/generate (:model m) (:args m)
                                                         (:choices t))]
                  (close? (ev (:score trace)) (ev weight) 0.01))))

(defspec law:return-value-independence 100
  ;; [T] §2.3.1 — f(x,τ) depends only on x and τ
  (prop/for-all [m gen-nonbranching]
                (let [{:keys [model args]} m
                      t (p/simulate model args)
                      r1 (:trace (p/generate model args (:choices t)))
                      r2 (:trace (p/generate model args (:choices t)))]
                  (close? (ev (:retval r1)) (ev (:retval r2)) 1e-10))))

;; ---------------------------------------------------------------------------
;; ASSESS laws [D] assess
;; ---------------------------------------------------------------------------

(defspec law:assess-equals-generate-score 100
  ;; [D] assess(P, x, tau).weight = generate(P, x, tau).trace.score
  (prop/for-all [m gen-nonbranching]
                (let [t (p/simulate (:model m) (:args m))
                      {:keys [weight]} (p/assess (:model m) (:args m) (:choices t))
                      {:keys [trace]} (p/generate (:model m) (:args m) (:choices t))]
                  (close? (ev weight) (ev (:score trace)) 0.01))))

;; ---------------------------------------------------------------------------
;; PROPOSE laws [D] propose
;; ---------------------------------------------------------------------------

(defspec law:propose-weight-equals-generate 100
  ;; [D] propose(P,x).weight = generate(P,x, propose.choices).weight
  (prop/for-all [m gen-nonbranching]
                (let [{:keys [choices weight]} (p/propose (:model m) (:args m))
                      pw (ev weight)
                      {:keys [weight]} (p/generate (:model m) (:args m) choices)
                      gw (ev weight)]
                  (close? pw gw 0.01))))

;; ---------------------------------------------------------------------------
;; UPDATE laws [T] §2.3.1, Proposition 2.3.1
;; ---------------------------------------------------------------------------

(defspec law:update-identity 100
  ;; [T] h_update(tau, tau) = (tau, {})
  ;; update with same choices: weight = 0
  (prop/for-all [m gen-nonbranching]
                (let [t (p/simulate (:model m) (:args m))
                      {:keys [weight]} (p/update (:model m) t (:choices t))]
                  (close? 0.0 (ev weight) 0.01))))

(defspec law:update-density-ratio 100
  ;; [T] §2.3.1 UPDATE: weight = log(p(tau';x')/p(tau;x))
  (prop/for-all [m gen-nonbranching]
                (let [t1 (p/simulate (:model m) (:args m))
                      t2 (p/simulate (:model m) (:args m))
                      old-score (ev (:score t1))
                      {:keys [trace weight]} (p/update (:model m) t1 (:choices t2))
                      new-score (ev (:score trace))
                      w (ev weight)]
                  (close? w (- new-score old-score) 0.1))))

(defspec law:update-round-trip 100
  ;; [T] Proposition 2.3.1
  ;; update(update(t,sigma).trace, discard) recovers original
  (prop/for-all [m gen-nonbranching]
                (let [t (p/simulate (:model m) (:args m))
                      orig-score (ev (:score t))
                      ;; Update with a different trace's choices
                      t2 (p/simulate (:model m) (:args m))
                      {:keys [trace discard]} (p/update (:model m) t (:choices t2))
                      ;; Round-trip back
                      {:keys [trace]} (p/update (:model m) trace discard)
                      recovered-score (ev (:score trace))]
                  (close? orig-score recovered-score 0.05))))

(defspec law:update-discard-completeness 100
  ;; [T] §2.3.1 UPDATE — discard holds original values at overwritten addresses
  (prop/for-all [m gen-nonbranching]
                (let [{:keys [model args]} m
                      t1 (p/simulate model args)
                      t2 (p/simulate model args)
                      {:keys [discard]} (p/update model t1 (:choices t2))]
                  (every? (fn [path]
                            (close? (ev (cm/get-choice (:choices t1) path))
                                    (ev (cm/get-choice discard path))
                                    1e-6))
                          (cm/addresses discard)))))

;; ---------------------------------------------------------------------------
;; REGENERATE laws [D] regenerate
;; ---------------------------------------------------------------------------

(defspec law:regenerate-empty-identity 100
  ;; regenerate(t, none).weight = 0
  (prop/for-all [m gen-nonbranching]
                (let [t (p/simulate (:model m) (:args m))
                      {:keys [weight]} (p/regenerate (:model m) t sel/none)]
                  (close? 0.0 (ev weight) 0.01))))

(defspec law:regenerate-preserves-unselected 100
  ;; regenerate(t, S) preserves values at addresses NOT in S
  (prop/for-all [m gen-multisite]
                (let [t (p/simulate (:model m) (:args m))
                      addrs (cm/addresses (:choices t))
                      selected-addr (first (first addrs))
                      unselected (map first (rest addrs))
                      orig-vals (into {} (map (fn [a]
                                                [a (ev (cm/get-value
                                                        (cm/get-submap (:choices t) a)))])
                                              unselected))
                      {:keys [trace]} (p/regenerate (:model m) t
                                                    (sel/select selected-addr))]
                  (every? (fn [a]
                            (let [v (ev (cm/get-value (cm/get-submap (:choices trace) a)))]
                              (close? (get orig-vals a) v 1e-6)))
                          unselected))))

;; ---------------------------------------------------------------------------
;; PROJECT laws [D] project
;; ---------------------------------------------------------------------------

(defspec law:project-all-equals-score 100
  ;; project(t, all) = trace.score
  (prop/for-all [m gen-nonbranching]
                (let [t (p/simulate (:model m) (:args m))
                      proj (ev (p/project (:model m) t sel/all))
                      s (ev (:score t))]
                  (close? proj s 0.01))))

(defspec law:project-none-equals-zero 100
  ;; project(t, none) = 0
  (prop/for-all [m gen-nonbranching]
                (let [t (p/simulate (:model m) (:args m))
                      proj (ev (p/project (:model m) t sel/none))]
                  (close? 0.0 proj 0.01))))

(defspec law:score-decomposition 100
  ;; [T] project(t, S) + project(t, S^c) = trace.score
  (prop/for-all [m gen-multisite]
                (let [t (p/simulate (:model m) (:args m))
                      addrs (cm/addresses (:choices t))
                      first-addr (first (first addrs))
                      s (sel/select first-addr)
                      score (ev (:score t))
                      ps (ev (p/project (:model m) t s))
                      pcs (ev (p/project (:model m) t (sel/complement-sel s)))]
                  (close? score (+ ps pcs) 0.1))))

(defspec law:score-full-decomposition 100
  ;; Sum over all individual project = trace.score
  (prop/for-all [m gen-multisite]
                (let [t (p/simulate (:model m) (:args m))
                      addrs (cm/addresses (:choices t))
                      score (ev (:score t))
                      proj-sum (reduce
                                (fn [acc path]
                                  (+ acc (ev (p/project (:model m) t
                                                        (gfi/path->selection path)))))
                                0.0
                                addrs)]
                  (close? proj-sum score 0.05))))

;; ---------------------------------------------------------------------------
;; STRUCTURED DENSITY laws [T] Def 2.1.3, Prop 2.1.2, Prop 2.1.3
;; ---------------------------------------------------------------------------

(defspec law:structured-density 100
  ;; [T] Def 2.1.3 — structured density: address sets with nonzero probability
  ;; are determined by values. Group 20 simulates by address set; every group
  ;; must have all finite scores. For branching models this verifies every
  ;; execution path has positive density (P(miss both branches) < 0.001).
  (prop/for-all [m gen-model]
                (let [traces (repeatedly 20 #(p/simulate (:model m) (:args m)))
                      groups (group-by (fn [t] (set (cm/addresses (:choices t)))) traces)]
                  (every? (fn [[_addr-set ts]]
                            (every? (fn [t] (h/finite? (ev (:score t)))) ts))
                          groups))))

(defspec law:conditional-is-structured 100
  ;; [T] Prop 2.1.2 -- generate with partial constraints has finite score
  (prop/for-all [m gen-multisite]
                (let [t (p/simulate (:model m) (:args m))
                      addrs (cm/addresses (:choices t))
                      first-addr (first (first addrs))
                      val (cm/get-value (cm/get-submap (:choices t) first-addr))
                      partial (cm/choicemap first-addr val)
                      {:keys [trace weight]} (p/generate (:model m) (:args m) partial)]
                  (and (h/finite? (ev (:score trace)))
                       (h/finite? (ev weight))))))

(defspec law:generate-uniqueness 100
  ;; [T] Prop 2.1.3 -- full constraints produce deterministic trace
  ;; Score, retval, AND weight must all be identical
  (prop/for-all [m gen-nonbranching]
                (let [t (p/simulate (:model m) (:args m))
                      r1 (p/generate (:model m) (:args m) (:choices t))
                      r2 (p/generate (:model m) (:args m) (:choices t))]
                  (and (close? (ev (:score (:trace r1)))
                               (ev (:score (:trace r2))) 1e-10)
                       (close? (ev (:retval (:trace r1)))
                               (ev (:retval (:trace r2))) 1e-10)
                       (close? (ev (:weight r1))
                               (ev (:weight r2)) 1e-10)))))

;; ---------------------------------------------------------------------------
;; COMPOSITIONALITY laws [T] Proposition 2.3.2
;; ---------------------------------------------------------------------------

(defspec law:update-compositionality 100
  ;; [T] Proposition 2.3.2 -- splice weight decomposition
  ;; For P3 = P1;P2: w3 = w1 + w2 where w1 = inner score diff, w2 = outer-only diff
  ;; Verified by projecting onto inner/outer partitions independently
  (prop/for-all [m gen-splice]
                (let [{:keys [model args]} m
                      t1 (p/simulate model args)
                      t2 (p/simulate model args)
                      ;; Total update weight
                      {:keys [trace weight]} (p/update model t1 (:choices t2))
                      w3 (ev weight)
                      ;; Partition: inner (splice namespace) vs outer
                      inner-sel (sel/hierarchical :inner sel/all)
                      outer-sel (sel/complement-sel inner-sel)
                      ;; w1 = inner score difference via project
                      w1 (- (ev (p/project model trace inner-sel))
                            (ev (p/project model t1 inner-sel)))
                      ;; w2 = outer-only score difference via project
                      w2 (- (ev (p/project model trace outer-sel))
                            (ev (p/project model t1 outer-sel)))]
                  (close? w3 (+ w1 w2) 0.05))))

;; ---------------------------------------------------------------------------
;; CROSS-OPERATION consistency laws
;; ---------------------------------------------------------------------------

(defspec law:generate-assess-agreement 100
  ;; generate(P,x,sigma).trace.score = assess(P,x,sigma).weight
  ;; (for fully constrained sigma)
  (prop/for-all [m gen-nonbranching]
                (let [t (p/simulate (:model m) (:args m))
                      {:keys [trace]} (p/generate (:model m) (:args m) (:choices t))
                      gs (ev (:score trace))
                      {:keys [weight]} (p/assess (:model m) (:args m) (:choices t))
                      aw (ev weight)]
                  (close? gs aw 0.01))))

(defspec law:propose-assess-agreement 100
  ;; Cross-operation consistency: propose weight = assess weight
  ;; Complements law:propose-weight-equals-generate (propose vs generate)
  (prop/for-all [m gen-nonbranching]
                (let [{:keys [choices weight]} (p/propose (:model m) (:args m))
                      pw (ev weight)
                      {:keys [weight]} (p/assess (:model m) (:args m) choices)
                      aw (ev weight)]
                  (close? pw aw 0.01))))

;; ---------------------------------------------------------------------------
;; GRADIENT laws [T] Eq 2.12, §2.3.1
;; ---------------------------------------------------------------------------

;; Differentiable model pool: continuous distributions only (no bernoulli,
;; no branching). Discrete sites yield zero gradient for both analytical
;; and FD, so they pass trivially — but we restrict the pool to models
;; where gradients carry nontrivial information.
(def differentiable-pool
  [single-gaussian single-uniform single-exponential single-beta
   two-independent three-independent
   gaussian-chain three-chain
   linear-regression single-arg-model two-arg-model])

(def gen-differentiable
  "Generator: pick a model with only continuous trace sites."
  (gen/elements differentiable-pool))

(def models-with-args
  "Differentiable models that take arguments (non-vacuous gradient tests)."
  (filterv #(seq (:args %)) differentiable-pool))

(def gen-with-args
  "Generator: pick a differentiable model that has arguments."
  (gen/elements models-with-args))

(defspec law:gradient-choice-correctness 50
  ;; [T] Eq 2.12 — choice gradients match finite differences
  (prop/for-all [m gen-differentiable]
                (let [{:keys [model args]} m
                      t (p/simulate model args)
                      addrs (->> (:choices t) cm/addresses (mapv first))
                      grads (grad/choice-gradients model t addrs)
                      h 1e-3]
                  (every?
                   (fn [addr]
                     (let [v (ev (cm/get-value (cm/get-submap (:choices t) addr)))
                           choices-plus (cm/set-choice (:choices t) [addr]
                                                       (mx/scalar (+ v h)))
                           choices-minus (cm/set-choice (:choices t) [addr]
                                                        (mx/scalar (- v h)))
                           score-plus (-> (p/generate model args choices-plus)
                                          :trace :score ev)
                           score-minus (-> (p/generate model args choices-minus)
                                           :trace :score ev)
                           fd-grad (/ (- score-plus score-minus) (* 2 h))
                           analytical (ev (get grads addr))]
                       (close? analytical fd-grad 0.05)))
                   addrs))))

(defspec law:gradient-argument-correctness 50
  ;; [T] §2.3.1 — AD argument gradients match finite differences
  ;; Uses gen-with-args so every trial exercises a model with arguments
  (prop/for-all [m gen-with-args]
                (let [{:keys [model args]} m
                      t (p/simulate model args)
                      choices (:choices t)
                      args-v (vec args)
                      h 1e-3]
                  (every?
                   (fn [i]
                     (let [x-val (nth args-v i)
                            ;; AD gradient
                           score-fn (fn [x-arr]
                                      (:weight (p/generate model
                                                           (assoc args-v i x-arr)
                                                           choices)))
                           analytical (ev ((mx/grad score-fn) x-val))
                            ;; FD gradient
                           x-num (ev x-val)
                           sp (ev (:weight (p/generate model
                                                       (assoc args-v i (mx/scalar (+ x-num h)))
                                                       choices)))
                           sm (ev (:weight (p/generate model
                                                       (assoc args-v i (mx/scalar (- x-num h)))
                                                       choices)))
                           fd (/ (- sp sm) (* 2 h))]
                       (close? analytical fd 0.05)))
                   (range (count args-v))))))

;; ---------------------------------------------------------------------------
;; INFERENCE laws (importance sampling) [T] Alg 2, Eq 3.2-3.5
;; ---------------------------------------------------------------------------

(defspec law:is-weight-formula 100
  ;; [T] Alg 2, Eq 3.2 — IS weight = project(trace, obs_selection)
  (prop/for-all [m gen-multisite]
                (let [{:keys [model args]} m
                      t (p/simulate model args)
                      obs-addr (first (first (cm/addresses (:choices t))))
                      obs-val (cm/get-value (cm/get-submap (:choices t) obs-addr))
                      obs (cm/choicemap obs-addr obs-val)
                      {:keys [trace weight]} (p/generate model args obs)
                      w (ev weight)
                      expected (ev (p/project model trace (sel/select obs-addr)))]
                  (close? w expected 0.01))))

(defspec law:proposal-support-coverage 100
  ;; [T] Eq 3.3 — no -Inf weights means proposal covers model support
  (prop/for-all [m gen-multisite]
                (let [{:keys [model args]} m
                      t (p/simulate model args)
                      obs-addr (first (first (cm/addresses (:choices t))))
                      obs-val (cm/get-value (cm/get-submap (:choices t) obs-addr))
                      obs (cm/choicemap obs-addr obs-val)
                      weights (repeatedly 20
                                          #(ev (:weight (p/generate model args obs))))]
                  (every? h/finite? weights))))

(defspec law:log-ml-estimate-well-defined 50
  ;; [T] Eq 3.5 — IS log-ML estimate is finite (well-defined) for any model.
  ;; Convergence to analytical value tested in log-ml-convergence-analytical.
  (prop/for-all [m gen-multisite]
                (let [{:keys [model args]} m
                      t (p/simulate model args)
                      obs-addr (first (first (cm/addresses (:choices t))))
                      obs-val (cm/get-value (cm/get-submap (:choices t) obs-addr))
                      obs (cm/choicemap obs-addr obs-val)
                      weights (repeatedly 100
                                          #(ev (:weight (p/generate model args obs))))
                      max-w (apply max weights)
                      log-ml (+ max-w
                                (js/Math.log
                                 (/ (reduce + (map #(js/Math.exp (- % max-w)) weights))
                                    (count weights))))]
                  (h/finite? log-ml))))

(defn- logsumexp
  "Numerically stable log-sum-exp: log(sum(exp(xs)))."
  [xs]
  (let [max-x (apply max xs)]
    (+ max-x (js/Math.log (reduce + (map #(js/Math.exp (- % max-x)) xs))))))

(t/deftest log-ml-convergence-analytical
  (t/testing "IS log-ML converges to analytical value for Normal-Normal conjugate"
    (let [conj-model (dyn/auto-key
                      (gen []
                           (let [mu (trace :mu (dist/gaussian 0 2))]
                             (trace :y (dist/gaussian mu 1)))))
          obs (cm/choicemap :y (mx/scalar 1.5))
          ;; Analytical: log N(1.5; 0, sqrt(5))
          ;; = -0.5*log(2*pi) - 0.5*log(5) - 0.5*(2.25/5)
          analytical (- (- (* 0.5 (js/Math.log (* 2 js/Math.PI))))
                        (* 0.5 (js/Math.log 5))
                        (* 0.5 (/ 2.25 5)))
          n-samples 5000
          weights (repeatedly n-samples
                              #(ev (:weight (p/generate conj-model [] obs))))
          log-ml (- (logsumexp weights) (js/Math.log n-samples))]
      (t/is (close? log-ml analytical 0.1)
            (str "log-ML " log-ml " not close to analytical " analytical)))))

;; ---------------------------------------------------------------------------
;; MH + REGENERATE WEIGHT laws [T] Eq 4.1, Alg 5, §3.4.2
;; ---------------------------------------------------------------------------

(defspec law:regenerate-weight-formula 100
  ;; [T] Eq 4.1 — regenerate weight equals score change at unselected sites
  ;; w = (new_score - old_score) - (project(new, S) - project(old, S))
  ;; Verified for BOTH first and last leaf addresses.
  (prop/for-all [m gen-multisite]
                (let [{:keys [model args]} m
                      t (p/simulate model args)
                      addrs (cm/addresses (:choices t))
                      check-addr (fn [addr]
                                   (let [sel (sel/select addr)
                                         old-score (ev (:score t))
                                         old-proj-s (ev (p/project model t sel))
                                         {:keys [trace weight]}
                                         (p/regenerate model t sel)
                                         new-score (ev (:score trace))
                                         new-proj-s (ev (p/project model trace sel))
                                         w (ev weight)
                                         expected (- (- new-score old-score)
                                                     (- new-proj-s old-proj-s))]
                                     (close? w expected 0.01)))
                      first-addr (first (first addrs))
                      last-addr (first (last addrs))]
                  (and (check-addr first-addr)
                       (check-addr last-addr)))))

(defspec law:mh-acceptance-correctness 10
  ;; [T] Alg 5, §3.4.2 — MH with regenerate weight converges to correct posterior.
  ;; Uses fixed Normal-Normal conjugate model (independent of gen-nonbranching pool).
  ;; Prior: x ~ N(0,1), Likelihood: y ~ N(x, 0.5), observe y=2.
  ;; Posterior: x | y=2 ~ N(1.6, sqrt(0.2)). 10 trials (each runs 500 MH steps).
  (prop/for-all [_ (gen/return nil)]
                (let [mh-model (dyn/auto-key
                                (gen []
                                     (let [x (trace :x (dist/gaussian 0 1))]
                                       (trace :y (dist/gaussian x 0.5)))))
                      obs (cm/choicemap :y (mx/scalar 2.0))
                      init-trace (:trace (p/generate mh-model [] obs))
                      sel (sel/select :x)
                      samples (loop [t init-trace i 0 acc [] k (rng/fresh-key 7)]
                                (if (>= i 500)
                                  acc
                                  (let [[k1 k2] (rng/split k)
                                        {:keys [trace weight]}
                                        (p/regenerate mh-model t sel)
                                        w (ev weight)
                                        accept? (u/accept-mh? w k1)
                                        next-t (if accept? trace t)
                                        x-val (ev (cm/get-value
                                                   (cm/get-submap (:choices next-t) :x)))]
                                    (recur next-t (inc i)
                                           (if (>= i 100) (conj acc x-val) acc)
                                           k2))))
                      mean (/ (reduce + samples) (count samples))]
                  (close? mean 1.6 0.3))))

(defspec law:mh-proposal-reversibility 100
  ;; [T] §3.4.2 — regenerate-based MH is reversible: forward and reverse
  ;; moves both produce valid traces with finite weights
  (prop/for-all [m gen-multisite]
                (let [{:keys [model args]} m
                      t (p/simulate model args)
                      sel-addr (first (first (cm/addresses (:choices t))))
                      sel (sel/select sel-addr)
                      ;; Forward move
                      {fwd-trace :trace fwd-weight :weight}
                      (p/regenerate model t sel)
                      w-fwd (ev fwd-weight)
                      ;; Reverse move
                      {rev-trace :trace rev-weight :weight}
                      (p/regenerate model fwd-trace sel)
                      w-rev (ev rev-weight)]
                  (and (js/Number.isFinite w-fwd)
                       (js/Number.isFinite w-rev)
                       ;; Reverse trace preserves address set
                       (= (set (cm/addresses (:choices t)))
                          (set (cm/addresses (:choices rev-trace))))))))

(t/deftest regenerate-weight-chain-model-exact
  (t/testing "Regenerate weight on gaussian chain matches child log-prob change"
    ;; x ~ N(0,1), y ~ N(x, 0.5). Regenerate S={:x}.
    ;; weight should equal lp(y; x', 0.5) - lp(y; x, 0.5)
    ;; because y's distribution changed but y's value didn't.
    (let [n-trials 20
          results
          (for [_ (range n-trials)]
            (let [t (p/simulate (:model gaussian-chain) [])
                  old-x (-> (:choices t) (cm/get-submap :x) cm/get-value)
                  old-y (-> (:choices t) (cm/get-submap :y) cm/get-value)
                  {:keys [trace weight]} (p/regenerate (:model gaussian-chain)
                                                       t (sel/select :x))
                  new-x (-> (:choices trace) (cm/get-submap :x) cm/get-value)
                  new-y (-> (:choices trace) (cm/get-submap :y) cm/get-value)
                  w (ev weight)
                  ;; y must be unchanged (unselected)
                  _ (assert (close? (ev old-y) (ev new-y) 1e-8)
                            "y should be preserved")
                  ;; Independently compute the expected weight:
                  ;; change in child's log-prob due to parent change
                  old-child-lp (ev (dist/log-prob
                                    (dist/gaussian old-x 0.5) old-y))
                  new-child-lp (ev (dist/log-prob
                                    (dist/gaussian new-x 0.5) old-y))
                  expected (- new-child-lp old-child-lp)]
              (close? w expected 0.01)))]
      (t/is (every? true? results)
            (str "Regenerate weight mismatch in "
                 (count (remove true? results)) "/" n-trials " trials")))))

;; ---------------------------------------------------------------------------
;; INTERNAL PROPOSAL laws [T] Ch. 4 — Cusumano-Towner 2020
;; ---------------------------------------------------------------------------

;; --- Law #27: Internal Proposal Support [T] Def 4.1.1 ---
;; q(v; x, sigma) > 0 iff p(v|sigma; x) > 0
;;
;; For forward-sampling: q samples unconstrained addresses from their conditional
;; priors. Same support as the model by construction. Distinguished from
;; proposal-support-coverage (#12) by testing at SUPPORT BOUNDARIES specifically.

(t/deftest law:internal-proposal-support
  ;; [T] Def 4.1.1 — support equivalence at boundaries
  ;; Deterministic test: fixed models with known boundary values.
  (t/testing "in-support boundary values produce finite weights"
    (let [;; Uniform(0,1) at x=0 and x=1 (boundary of support)
          uni-model (dyn/auto-key
                      (gen [] (let [x (trace :x (dist/uniform 0 1))]
                                (trace :y (dist/gaussian x 1)))))
          w0 (ev (:weight (p/generate uni-model [] (cm/choicemap :x (mx/scalar 0.0)))))
          w1 (ev (:weight (p/generate uni-model [] (cm/choicemap :x (mx/scalar 1.0)))))]
      (t/is (h/finite? w0) "Uniform boundary x=0 should be finite")
      (t/is (close? w0 0.0 1e-6) "Uniform(0;0,1) log-prob = 0")
      (t/is (h/finite? w1) "Uniform boundary x=1 should be finite")
      (t/is (close? w1 0.0 1e-6) "Uniform(1;0,1) log-prob = 0"))
    (let [;; Beta(2,5) at x=0.01 and x=0.99 (near boundary)
          beta-model (dyn/auto-key
                       (gen [] (let [p (trace :p (dist/beta-dist 2 5))]
                                 (trace :x (dist/bernoulli p)))))
          w-lo (ev (:weight (p/generate beta-model [] (cm/choicemap :p (mx/scalar 0.01)))))
          w-hi (ev (:weight (p/generate beta-model [] (cm/choicemap :p (mx/scalar 0.99)))))]
      (t/is (h/finite? w-lo) "Beta near-boundary p=0.01 should be finite")
      (t/is (h/finite? w-hi) "Beta near-boundary p=0.99 should be finite"))
    (let [;; Exponential(1) at x=0 (boundary of support [0,inf))
          exp-model (dyn/auto-key
                      (gen [] (let [x (trace :x (dist/exponential 1))]
                                (trace :y (dist/gaussian x 1)))))
          w0 (ev (:weight (p/generate exp-model [] (cm/choicemap :x (mx/scalar 0.0)))))]
      (t/is (h/finite? w0) "Exponential boundary x=0 should be finite")
      (t/is (close? w0 0.0 1e-6) "Exp(0;1) log-prob = 0")))
  (t/testing "out-of-support values produce non-finite log-prob (reverse direction)"
    ;; Reverse direction of support equivalence: q(v;x,sigma) = 0 when
    ;; p(v|sigma;x) = 0. Verified at the distribution primitive level,
    ;; which defines the proposal's density function.
    (let [cases [[(dist/exponential 1) (mx/scalar -1.0) "Exp(1) at x=-1"]
                 [(dist/uniform 0 1) (mx/scalar 2.0) "Uniform(0,1) at x=2"]
                 [(dist/uniform 0 1) (mx/scalar -0.5) "Uniform(0,1) at x=-0.5"]]]
      (doseq [[d v label] cases]
        (let [lp (ev (dist/log-prob d v))]
          (t/is (not (h/finite? lp))
                (str label " should have non-finite log-prob, got " lp)))))))

;; --- Law #28: Extended Generate Weight [T] §4.1.1 ---
;; log w = log p(tau; x) / q(v; x, sigma)
;;       = sum_{a in sigma} log p(sigma[a] | parents(a))
;;
;; For forward-sampling, the weight equals the sum of log-probs at constrained
;; addresses only (the marginal likelihood of the constraints).
;; Distinguished from is-weight-formula (#11): tests with MULTIPLE constrained
;; addresses simultaneously.

(t/deftest law:generate-weight-is-marginal-likelihood
  (t/testing "weight = sum of log-probs at constrained addresses, multi-constraint"
    ;; Non-conjugate model: a ~ Uniform(0,1), b ~ Laplace(a, 1), c ~ Exp(1)
    ;; No conjugate pairs, so uses compiled/handler path (not analytical).
    (let [model (dyn/auto-key
                  (gen [] (let [a (trace :a (dist/uniform 0 1))
                                b (trace :b (dist/laplace a 1))]
                            (trace :c (dist/exponential 1)))))
          ;; Test 1: single constraint
          ;; Constrain a=0.5 only: weight = log Uniform(0.5; 0,1) = 0
          c1 (cm/choicemap :a (mx/scalar 0.5))
          w1 (ev (:weight (p/generate model [] c1)))
          ;; Test 2: two constraints
          ;; Constrain a=0.5 and c=0.5: weight = 0 + log Exp(0.5; 1)
          ;; Analytical: log Exp(0.5; 1) = log(1) - 1*0.5 = -0.5
          c2 (cm/choicemap :a (mx/scalar 0.5) :c (mx/scalar 0.5))
          {:keys [trace weight]} (p/generate model [] c2)
          w2 (ev weight)
          ;; Test 3: weight = project(trace, constrained selection)
          proj (ev (p/project model trace (sel/from-set #{:a :c})))]
      (t/is (close? w1 0.0 0.01) "single constraint: Uniform(0.5) = 0")
      (t/is (close? w2 -0.5 0.01) "multi constraint: 0 + (-0.5) = -0.5")
      (t/is (close? w2 proj 0.01) "weight = project for constrained selection"))))

;; --- Law #29: Forward Sampling Factorization [T] §4.1.3 ---
;; q(v; x, sigma) = product_i p({a_i->v[a_i]} | context)^[a_i not in sigma]
;;
;; For each unconstrained address, the proposal samples from the conditional
;; prior. Verified via: weight = score - sum(log-prob at unconstrained addrs).

(t/deftest law:forward-sampling-factorization
  (t/testing "weight = score - log q(unconstrained), Laplace chain model"
    ;; a ~ Uniform(0,1), b ~ Laplace(a, 1), c ~ Laplace(a+b, 0.5)
    ;; No conjugate pairs.
    (let [model (dyn/auto-key
                  (gen [] (let [a (trace :a (dist/uniform 0 1))
                                b (trace :b (dist/laplace a 1))]
                            (trace :c (dist/laplace (mx/add a b) 0.5)))))
          ;; Case 1: constrain c=3.0 only. a, b unconstrained.
          ;; weight = log Laplace(3; a+b, 0.5).
          ;; Factorization: weight = score - [lp(a) + lp(b|a)].
          ;;
          ;; Fixed analytical example (verified by hand):
          ;; a=0.5, b=1.0 => score = 0 + (-0.693-0.5) + (-3.0) = -4.193
          ;; log q = 0 + (-0.693-0.5) = -1.193
          ;; weight = -4.193 - (-1.193) = -3.0
          ;; Direct: log Laplace(3; 1.5, 0.5) = -log(1) - |3-1.5|/0.5 = -3.0
          results-c
          (for [seed (range 20)]
            (let [m (dyn/with-key model (rng/fresh-key seed))
                  {:keys [trace weight]} (p/generate m []
                                           (cm/choicemap :c (mx/scalar 3.0)))
                  w (ev weight)
                  s (ev (:score trace))
                  a (ev (cm/get-value (cm/get-submap (:choices trace) :a)))
                  b (ev (cm/get-value (cm/get-submap (:choices trace) :b)))
                  lp-a (ev (dist/log-prob (dist/uniform 0 1) (mx/scalar a)))
                  lp-b (ev (dist/log-prob (dist/laplace (mx/scalar a) 1)
                                           (mx/scalar b)))
                  log-q (+ lp-a lp-b)]
              (close? w (- s log-q) 0.01)))
          ;; Case 2: constrain a=0.7 and c=3.0. b unconstrained.
          ;; Factorization: weight = score - lp(b|a=0.7).
          results-ac
          (for [seed (range 20)]
            (let [m (dyn/with-key model (rng/fresh-key seed))
                  constraints (cm/choicemap :a (mx/scalar 0.7) :c (mx/scalar 3.0))
                  {:keys [trace weight]} (p/generate m [] constraints)
                  w (ev weight)
                  s (ev (:score trace))
                  b (ev (cm/get-value (cm/get-submap (:choices trace) :b)))
                  log-q (ev (dist/log-prob (dist/laplace (mx/scalar 0.7) 1)
                                            (mx/scalar b)))]
              (close? w (- s log-q) 0.01)))]
      (t/is (every? true? results-c)
            (str "Factorization failed (constrain c only) in "
                 (count (remove true? results-c)) "/20 trials"))
      (t/is (every? true? results-ac)
            (str "Factorization failed (constrain a+c) in "
                 (count (remove true? results-ac)) "/20 trials")))))

;; ---------------------------------------------------------------------------
;; Run the full law suite via gfi/verify (integration test)
;; ---------------------------------------------------------------------------

;; ---------------------------------------------------------------------------
;; DENOTATIONAL SEMANTICS laws [T] Chapter 2.2.2, Figure 2-1
;; Cusumano-Towner 2020 PhD thesis, §2.2.2
;;
;; The thesis defines three semantic functions for a toy modeling language:
;;   Addrs⟦E⟧          — set of addresses used by expression E
;;   Val⟦E⟧(σ)(τ)      — value of E given environment σ and choices τ
;;   Dist⟦E⟧(σ)(τ)     — probability distribution on choices for E
;;
;; The top-level denotation maps a gen definition to a generative function:
;;   ⟦@gen function(X₁,...,Xₙ) E end⟧ = (Rⁿ, R, λx,τ.Dist⟦E⟧, λx,τ.Val⟦E⟧)
;; ---------------------------------------------------------------------------

;; --- Law #38: Addrs⟦E⟧ correctness ---
;; GenMLX's schema/extract-schema walks the gen body source form at construction
;; time and extracts :trace-sites. For static models (no branches, no dynamic
;; address construction), the schema addresses must exactly equal the runtime
;; trace addresses produced by simulate.

(t/deftest law:addrs-correctness-two-site
  (t/testing "Addrs⟦E⟧: schema addresses = runtime addresses (2-site chain)"
    ;; Model: x ~ N(0,1), y ~ N(x, 0.5)
    ;; Addrs⟦E⟧ = {:x, :y}
    (let [model (:model gaussian-chain)
          schema (:schema model)
          schema-addrs (set (map :addr (:trace-sites schema)))
          trace (p/simulate model [])
          trace-addrs (set (map first (cm/addresses (:choices trace))))]
      (t/is (:static? schema)
            "gaussian-chain should be classified as static")
      (t/is (= schema-addrs trace-addrs)
            (str "Addrs mismatch: schema=" schema-addrs " trace=" trace-addrs)))))

(t/deftest law:addrs-correctness-three-site
  (t/testing "Addrs⟦E⟧: schema addresses = runtime addresses (3-site chain)"
    ;; Model: a ~ N(0,1), b ~ N(a,1), c ~ N(a+b, 0.5)
    ;; Addrs⟦E��� = {:a, :b, :c}
    (let [model (:model three-chain)
          schema (:schema model)
          schema-addrs (set (map :addr (:trace-sites schema)))
          trace (p/simulate model [])
          trace-addrs (set (map first (cm/addresses (:choices trace))))]
      (t/is (:static? schema)
            "three-chain should be classified as static")
      (t/is (= schema-addrs trace-addrs)
            (str "Addrs mismatch: schema=" schema-addrs " trace=" trace-addrs)))))

(t/deftest law:addrs-dependency-graph
  (t/testing "Addrs⟦E⟧: dependency structure matches semantic analysis"
    ;; For three-chain: :a has no deps, :b depends on :a, :c depends on {:a, :b}
    ;; This is the dependency structure that Addrs⟦E⟧ implicitly encodes:
    ;; each trace site's distribution depends on previously-traced values.
    (let [schema (:schema (:model three-chain))
          dep-map (into {} (map (fn [s] [(:addr s) (:deps s)])
                                (:trace-sites schema)))]
      (t/is (= #{} (get dep-map :a))
            ":a should have no dependencies")
      (t/is (= #{:a} (get dep-map :b))
            ":b should depend only on :a")
      (t/is (= #{:a :b} (get dep-map :c))
            ":c should depend on both :a and :b"))))

;; --- Law #39: Val⟦E⟧ determinism ---
;; Val⟦E⟧(σ)(τ) is a mathematical function: given the same environment (args)
;; and choices (τ), the return value is deterministic. This is the denotational
;; semantics perspective on law #8 (return-value-independence): the value
;; function f in the GFI tuple (X, Y, p, f) is well-defined as a mathematical
;; function from (X × T) → Y, not a stochastic procedure.

(t/deftest law:val-determinism-chain
  (t/testing "Val⟦E⟧: f(args, ��) is deterministic (chain model)"
    ;; Model B: (gen [] (let [x (trace :x ...)] (trace :y ...)))
    ;; f([], τ) = τ[:y] (return value is the last trace expression)
    (let [model (:model gaussian-chain)
          tau (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0))
          r1 (p/generate model [] tau)
          r2 (p/generate model [] tau)
          rv1 (ev (:retval (:trace r1)))
          rv2 (ev (:retval (:trace r2)))]
      ;; Tolerance: 1e-10 (algebraic identity — no randomness, rounding only)
      (t/is (close? rv1 rv2 1e-10)
            (str "Val not deterministic: " rv1 " vs " rv2))
      ;; Verify retval = τ[:y]
      (t/is (close? rv1 1.0 1e-6)
            (str "Expected retval=1.0 (τ[:y]), got " rv1)))))

(t/deftest law:val-determinism-retval-not-last-trace
  (t/testing "Val⟦E⟧: retval = body return value, not necessarily last trace"
    ;; Model A: (gen [] (let [x (trace :x ...) y (trace :y ...)] x))
    ;; f([], τ) = τ[:x] (body returns x, not y)
    (let [model (:model two-independent)
          tau (cm/choicemap :x (mx/scalar 3.14) :y (mx/scalar -2.0))
          r1 (p/generate model [] tau)
          r2 (p/generate model [] tau)
          rv1 (ev (:retval (:trace r1)))
          rv2 (ev (:retval (:trace r2)))]
      ;; Tolerance: 1e-10 (algebraic identity)
      (t/is (close? rv1 rv2 1e-10)
            (str "Val not deterministic: " rv1 " vs " rv2))
      ;; Verify retval = τ[:x], not τ[:y]
      (t/is (close? rv1 3.14 1e-4)
            (str "Expected retval=3.14 (τ[:x]), got " rv1)))))

;; --- Law #40: Dist⟦E⟧ analytical match ---
;; For static models, Dist⟦E⟧ defines the joint density on choices.
;; We derive analytical moments of the joint distribution and verify
;; that simulated traces match within statistically justified tolerances.
;;
;; Tolerance policy for moment-matching (N=5000 iid samples):
;;   SE(mean) = sqrt(Var / N)
;;   SE(variance) = sqrt(2 * Var^2 / N)  [Gaussian kurtosis = 3]
;;   SE(covariance) = sqrt((Var_x * Var_y + Cov_xy^2) / N)  [Isserlis]
;;   Tolerance = 3.5 * SE  (p(false positive) < 0.001 for Gaussian)

(def ^:private N-moment-samples 5000)

(defn- collect-samples
  "Simulate N traces and collect values at given addresses."
  [model args addrs n]
  (let [results (repeatedly n #(let [tr (p/simulate model args)]
                                 (into {} (map (fn [a]
                                                 [a (ev (cm/get-value
                                                         (cm/get-submap (:choices tr) a)))])
                                               addrs))))]
    ;; Transpose: {addr -> [values]}
    (into {} (map (fn [a] [a (mapv #(get % a) results)]) addrs))))

(defn- sample-mean [xs] (/ (reduce + xs) (count xs)))
(defn- sample-var [xs]
  (let [m (sample-mean xs)]
    (/ (reduce + (map #(* (- % m) (- % m)) xs)) (count xs))))
(defn- sample-cov [xs ys]
  (let [mx (sample-mean xs) my (sample-mean ys)]
    (/ (reduce + (map #(* (- %1 mx) (- %2 my)) xs ys)) (count xs))))

(t/deftest law:dist-model-A-independent-gaussians
  (t/testing "Dist⟦E⟧ Model A: x ~ N(0,1), y ~ N(0,2) independent"
    ;; Analytical derivation:
    ;;   p(τ) = N(τ[:x]; 0, 1) * N(τ[:y]; 0, 2)
    ;;   E[x] = 0, Var[x] = 1
    ;;   E[y] = 0, Var[y] = sigma^2 = 4
    ;;   Cov(x,y) = 0  (independent)
    (let [model (dyn/auto-key (gen [] (let [x (trace :x (dist/gaussian 0 1))
                                            y (trace :y (dist/gaussian 0 2))]
                                        x)))
          samples (collect-samples model [] [:x :y] N-moment-samples)
          xs (get samples :x)
          ys (get samples :y)
          ;; Tolerances (N=5000, z=3.5):
          ;;   SE(mean_x) = sqrt(1/5000) = 0.01414, tol = 0.050
          ;;   SE(mean_y) = sqrt(4/5000) = 0.02828, tol = 0.099
          ;;   SE(var_x) = sqrt(2/5000) = 0.0200, tol = 0.070
          ;;   SE(var_y) = sqrt(2*16/5000) = 0.0800, tol = 0.280
          ;;   SE(cov_xy) = sqrt(1*4/5000) = 0.0283, tol = 0.099  [independent: Isserlis simplifies]
          ]
      (t/is (close? (sample-mean xs) 0.0 0.050)
            (str "E[x]: expected 0, got " (sample-mean xs)))
      (t/is (close? (sample-mean ys) 0.0 0.099)
            (str "E[y]: expected 0, got " (sample-mean ys)))
      (t/is (close? (sample-var xs) 1.0 0.070)
            (str "Var[x]: expected 1, got " (sample-var xs)))
      (t/is (close? (sample-var ys) 4.0 0.280)
            (str "Var[y]: expected 4, got " (sample-var ys)))
      (t/is (close? (sample-cov xs ys) 0.0 0.099)
            (str "Cov(x,y): expected 0, got " (sample-cov xs ys))))))

(t/deftest law:dist-model-B-gaussian-chain
  (t/testing "Dist⟦E⟧ Model B: x ~ N(0,1), y|x ~ N(x, 0.5)"
    ;; Analytical derivation:
    ;;   y = x + eps, eps ~ N(0, 0.5) independent of x
    ;;   E[x] = 0, Var[x] = 1
    ;;   E[y] = E[x + eps] = 0
    ;;   Var[y] = Var[x] + Var[eps] = 1 + 0.25 = 1.25
    ;;   Cov(x,y) = Cov(x, x+eps) = Var(x) = 1
    ;;   Corr(x,y) = 1/sqrt(1 * 1.25) = 2/sqrt(5) = 0.89443
    (let [model (:model gaussian-chain)
          samples (collect-samples model [] [:x :y] N-moment-samples)
          xs (get samples :x) ys (get samples :y)
          ;; Tolerances (N=5000, z=3.5):
          ;;   SE(mean_y) = sqrt(1.25/5000) = 0.01581, tol = 0.055
          ;;   SE(var_y) = sqrt(2*1.5625/5000) = 0.0250, tol = 0.088
          ;;   SE(cov) = sqrt((1*1.25 + 1^2)/5000) = sqrt(2.25/5000) = 0.02121, tol = 0.074
          ;;   For corr: delta method, tol ~ 0.06
          obs-cov (sample-cov xs ys)
          obs-corr (/ obs-cov (js/Math.sqrt (* (sample-var xs) (sample-var ys))))]
      (t/is (close? (sample-mean xs) 0.0 0.050)
            (str "E[x]: expected 0, got " (sample-mean xs)))
      (t/is (close? (sample-mean ys) 0.0 0.055)
            (str "E[y]: expected 0, got " (sample-mean ys)))
      (t/is (close? (sample-var xs) 1.0 0.070)
            (str "Var[x]: expected 1, got " (sample-var xs)))
      (t/is (close? (sample-var ys) 1.25 0.088)
            (str "Var[y]: expected 1.25, got " (sample-var ys)))
      (t/is (close? obs-cov 1.0 0.074)
            (str "Cov(x,y): expected 1, got " obs-cov))
      (t/is (close? obs-corr (/ 2.0 (js/Math.sqrt 5.0)) 0.06)
            (str "Corr(x,y): expected 2/sqrt(5)=" (/ 2.0 (js/Math.sqrt 5.0))
                 ", got " obs-corr)))))

(t/deftest law:dist-model-C-three-site-chain
  (t/testing "Dist⟦E⟧ Model C: a ~ N(0,1), b|a ~ N(a,1), c|a,b ~ N(a+b, 0.5)"
    ;; Analytical derivation:
    ;;   b = a + eps1, eps1 ~ N(0,1)
    ;;   c = (a + b) + eps2, eps2 ~ N(0, 0.5)
    ;;
    ;;   E[a] = 0, Var[a] = 1
    ;;   E[b] = 0, Var[b] = Var[a] + Var[eps1] = 2
    ;;   Cov(a,b) = Cov(a, a+eps1) = Var(a) = 1
    ;;   Var[a+b] = Var[a] + Var[b] + 2*Cov(a,b) = 1 + 2 + 2 = 5
    ;;   E[c] = 0, Var[c] = Var[a+b] + Var[eps2] = 5 + 0.25 = 5.25
    ;;   Cov(a,c) = Cov(a, a+b+eps2) = Var(a) + Cov(a,b) = 1 + 1 = 2
    ;;   Cov(b,c) = Cov(b, a+b+eps2) = Cov(b,a) + Var(b) = 1 + 2 = 3
    (let [model (:model three-chain)
          samples (collect-samples model [] [:a :b :c] N-moment-samples)
          as (get samples :a) bs (get samples :b) cs (get samples :c)
          ;; Tolerances (N=5000, z=3.5):
          ;;   SE(mean_b) = sqrt(2/5000) = 0.0200, tol = 0.070
          ;;   SE(mean_c) = sqrt(5.25/5000) = 0.0324, tol = 0.113
          ;;   SE(var_b) = sqrt(2*4/5000) = 0.0400, tol = 0.140
          ;;   SE(var_c) = sqrt(2*27.5625/5000) = 0.1050, tol = 0.367
          ;;   SE(cov_ab) = sqrt((1*2+1)/5000) = 0.0245, tol = 0.086
          ;;   SE(cov_ac) = sqrt((1*5.25+4)/5000) = 0.0430, tol = 0.150
          ;;   SE(cov_bc) = sqrt((2*5.25+9)/5000) = 0.0625, tol = 0.219
          ]
      (t/is (close? (sample-mean as) 0.0 0.050)
            (str "E[a]: expected 0, got " (sample-mean as)))
      (t/is (close? (sample-mean bs) 0.0 0.070)
            (str "E[b]: expected 0, got " (sample-mean bs)))
      (t/is (close? (sample-mean cs) 0.0 0.113)
            (str "E[c]: expected 0, got " (sample-mean cs)))
      (t/is (close? (sample-var as) 1.0 0.070)
            (str "Var[a]: expected 1, got " (sample-var as)))
      (t/is (close? (sample-var bs) 2.0 0.140)
            (str "Var[b]: expected 2, got " (sample-var bs)))
      (t/is (close? (sample-var cs) 5.25 0.367)
            (str "Var[c]: expected 5.25, got " (sample-var cs)))
      (t/is (close? (sample-cov as bs) 1.0 0.086)
            (str "Cov(a,b): expected 1, got " (sample-cov as bs)))
      (t/is (close? (sample-cov as cs) 2.0 0.150)
            (str "Cov(a,c): expected 2, got " (sample-cov as cs)))
      (t/is (close? (sample-cov bs cs) 3.0 0.219)
            (str "Cov(b,c): expected 3, got " (sample-cov bs cs))))))

;; --- Law #41: Gen function denotation ---
;; ⟦gen body⟧ = P = (X, Y, p, f) — all four components verified simultaneously.
;;
;; For a concrete model, generate with full constraints and verify:
;;   - X: model accepts declared arguments
;;   - Y: retval type and value match f(args, τ)
;;   - p: score = log p(τ; x) matches analytical joint density
;;   - f: retval = f(x, τ) is deterministic function of (args, choices)

(t/deftest law:gen-denotation-model-B
  (t/testing "⟦gen⟧ = (X, Y, p, f): all four components for gaussian chain"
    ;; Model B: (gen [] (let [x (trace :x (dist/gaussian 0 1))]
    ;;                     (trace :y (dist/gaussian x 0.5))))
    ;; X = []
    ;; Y = R
    ;; p(τ) = N(τ[:x]; 0, 1) * N(��[:y]; τ[:x], 0.5)
    ;; f([], τ) = τ[:y]
    ;;
    ;; At τ = {:x 0.5, :y 1.0}:
    ;;
    ;; log N(0.5; 0, 1) = -0.5*log(2π) - log(1) - 0.5*(0.5/1)²
    ;;                   = -0.91893853 - 0 - 0.125
    ;;                   = -1.04393853
    ;;
    ;; log N(1.0; 0.5, 0.5) = -0.5*log(2π) - log(0.5) - 0.5*((1.0-0.5)/0.5)²
    ;;                       = -0.91893853 + 0.69314718 - 0.5
    ;;                       = -0.72579135
    ;;
    ;; score = -1.04393853 + -0.72579135 = -1.76972988
    ;; retval = τ[:y] = 1.0
    ;; Tolerance: 1e-4 (float32 accumulation in two log-prob computations + addition)
    (let [model (:model gaussian-chain)
          tau (cm/choicemap :x (mx/scalar 0.5) :y (mx/scalar 1.0))
          result (p/generate model [] tau)
          trace (:trace result)
          ;; Analytical score
          log-2pi-half (* 0.5 (js/Math.log (* 2.0 js/Math.PI)))
          lp-x (- (- log-2pi-half) (* 0.5 0.25)) ;; N(0.5; 0, 1)
          lp-y (- (- (- log-2pi-half) (js/Math.log 0.5)) ;; N(1.0; 0.5, 0.5)
                  (* 0.5 1.0)) ;; z=(1-0.5)/0.5=1, z²=1
          analytical-score (+ lp-x lp-y)]
      ;; Component X: model accepts [] arguments (no exception)
      (t/is (some? trace)
            "X: model should accept [] arguments")
      ;; Component Y: retval = f([], τ) = τ[:y] = 1.0
      (t/is (close? (ev (:retval trace)) 1.0 1e-6)
            (str "Y/f: expected retval=1.0, got " (ev (:retval trace))))
      ;; Component p: score = log p(τ; [])
      (t/is (close? (ev (:score trace)) analytical-score 1e-4)
            (str "p: expected score=" analytical-score
                 ", got " (ev (:score trace))))
      ;; Cross-check: assess agrees with generate
      (let [assess-w (ev (:weight (p/assess model [] tau)))]
        (t/is (close? assess-w (ev (:score trace)) 1e-6)
              (str "p cross-check: assess=" assess-w
                   " vs generate.score=" (ev (:score trace)))))
      ;; Weight = score (fully constrained, proposal = prior)
      (t/is (close? (ev (:weight result)) (ev (:score trace)) 1e-6)
            "weight should equal score for fully constrained generate"))))

(t/deftest law:gen-denotation-model-C
  (t/testing "⟦gen⟧ = (X, Y, p, f): all four components for three-site chain"
    ;; Model C: a ~ N(0,1), b|a ~ N(a,1), c|a,b ~ N(a+b, 0.5)
    ;; f([], τ) = τ[:c]  (last trace expression)
    ;;
    ;; At τ = {:a 1.0, :b 0.5, :c 2.0}:
    ;;
    ;; log N(1.0; 0, 1) = -0.5*log(2π) - 0.5*1.0 = -1.41893853
    ;; log N(0.5; 1.0, 1) = -0.5*log(2π) - 0.5*(0.5)² = -0.91894 - 0.125 = -1.04393853
    ;; log N(2.0; 1.5, 0.5) = -0.5*log(2π) - log(0.5) - 0.5*((2-1.5)/0.5)²
    ;;                       = -0.91894 + 0.69315 - 0.5 = -0.72579135
    ;; Total = -1.41894 + -1.04394 + -0.72579 = -3.18867
    ;; Tolerance: 1e-4 (float32 accumulation across 3 log-prob terms)
    (let [model (:model three-chain)
          tau (cm/choicemap :a (mx/scalar 1.0) :b (mx/scalar 0.5) :c (mx/scalar 2.0))
          result (p/generate model [] tau)
          trace (:trace result)
          log-2pi-half (* 0.5 (js/Math.log (* 2.0 js/Math.PI)))
          lp-a (- (- log-2pi-half) (* 0.5 1.0)) ;; N(1.0; 0, 1)
          lp-b (- (- log-2pi-half) (* 0.5 0.25)) ;; N(0.5; 1.0, 1): z=-0.5
          lp-c (- (- (- log-2pi-half) (js/Math.log 0.5)) ;; N(2.0; 1.5, 0.5): z=1.0
                  (* 0.5 1.0))
          analytical-score (+ lp-a lp-b lp-c)]
      ;; X: accepts [] arguments
      (t/is (some? trace) "X: model should accept [] arguments")
      ;; Y/f: retval = τ[:c] = 2.0
      (t/is (close? (ev (:retval trace)) 2.0 1e-6)
            (str "Y/f: expected retval=2.0, got " (ev (:retval trace))))
      ;; p: score matches analytical joint density
      (t/is (close? (ev (:score trace)) analytical-score 1e-4)
            (str "p: expected score=" analytical-score
                 ", got " (ev (:score trace))))
      ;; Cross-check via assess
      (let [assess-w (ev (:weight (p/assess model [] tau)))]
        (t/is (close? assess-w (ev (:score trace)) 1e-6)
              (str "p cross-check: assess=" assess-w
                   " vs generate.score=" (ev (:score trace))))))))

(t/deftest law:gen-denotation-with-args
  (t/testing "⟦gen⟧ = (X, Y, p, f): model with arguments"
    ;; linear-regression: (gen [x-val] ...)
    ;; X = [R] (one scalar argument)
    ;; At x-val=2.0, τ = {:slope 1.0, :intercept 0.5, :y 3.0}:
    ;;   slope ~ N(0, 5):     log N(1.0; 0, 5) = -0.5*log(2π) - log(5) - 0.5*(1/5)²
    ;;                       = -0.91894 - 1.60944 - 0.02 = -2.54838
    ;;   intercept ~ N(0, 5): log N(0.5; 0, 5) = -0.91894 - 1.60944 - 0.5*(0.1)²
    ;;                       = -0.91894 - 1.60944 - 0.005 = -2.53338
    ;;   y ~ N(slope*x + intercept, 1) = N(2.5, 1):
    ;;     log N(3.0; 2.5, 1) = -0.91894 - 0.5*(0.5)² = -0.91894 - 0.125 = -1.04394
    ;; Total = -2.54838 + -2.53338 + -1.04394 = -6.12569
    ;; Tolerance: 1e-4 (float32 across 3 terms)
    (let [model (:model linear-regression)
          x-val (mx/scalar 2.0)
          tau (cm/choicemap :slope (mx/scalar 1.0)
                            :intercept (mx/scalar 0.5)
                            :y (mx/scalar 3.0))
          result (p/generate model [x-val] tau)
          trace (:trace result)
          log-2pi-half (* 0.5 (js/Math.log (* 2.0 js/Math.PI)))
          lp-slope (- (- (- log-2pi-half) (js/Math.log 5.0))
                      (* 0.5 (/ (* 1.0 1.0) (* 5.0 5.0))))
          lp-intercept (- (- (- log-2pi-half) (js/Math.log 5.0))
                          (* 0.5 (/ (* 0.5 0.5) (* 5.0 5.0))))
          lp-y (- (- log-2pi-half)
                  (* 0.5 (* 0.5 0.5)))
          analytical-score (+ lp-slope lp-intercept lp-y)]
      ;; X: model accepts [x-val]
      (t/is (some? trace)
            "X: model should accept [x-val] arguments")
      ;; p: score matches analytical
      (t/is (close? (ev (:score trace)) analytical-score 1e-4)
            (str "p: expected score=" analytical-score
                 ", got " (ev (:score trace)))))))

;; ---------------------------------------------------------------------------
;; SELECTION MH + KERNEL COMPOSITION laws [T] §3.4, §4.1, Prop 3.4.1
;; ---------------------------------------------------------------------------

;; --- Law #30: Optimal Proposal Weight [T] §4.1.3 ---
;; For conjugate models, generate weight = log marginal likelihood.
;; Normal-Normal conjugate: mu ~ N(0,2), y ~ N(mu,1), observe y=3.
;; Analytical: log N(3; 0, sqrt(5)) = -2.62366

(t/deftest law:optimal-proposal-weight
  (t/testing "conjugate generate weight = log marginal likelihood"
    (let [model (dyn/auto-key
                  (gen [] (let [mu (trace :mu (dist/gaussian 0 2))]
                            (trace :y (dist/gaussian mu 1)))))
          obs (cm/choicemap :y (mx/scalar 3.0))
          analytical (- (* -0.5 (js/Math.log (* 2 js/Math.PI)))
                        (* 0.5 (js/Math.log 5.0))
                        (* 0.5 (/ 9.0 5.0)))
          n-trials 10
          results (for [_ (range n-trials)]
                    (let [{:keys [weight]} (p/generate model [] obs)
                          w (ev weight)]
                      (close? w analytical 1e-3)))]
      (t/is (every? true? results)
            (str "Optimal proposal weight failed in "
                 (count (remove true? results)) "/" n-trials " trials")))))

;; --- Law #31: Selection MH Correctness [T] Alg 15 ---
;; regenerate weight = (new_score - old_score) - (new_proj - old_proj)
;; Verified for first leaf address across multi-site models.

(defspec law:selection-mh-correctness 100
  ;; [T] Alg 15 — regenerate weight decomposes as score change at unselected sites
  (prop/for-all [m gen-multisite]
                (let [{:keys [model args]} m
                      t (p/simulate model args)
                      addrs (cm/addresses (:choices t))
                      sel-addr (first (first addrs))
                      sel (sel/select sel-addr)
                      {:keys [trace weight]} (p/regenerate model t sel)
                      w (ev weight)
                      new-score (ev (:score trace))
                      old-score (ev (:score t))
                      old-proj (ev (p/project model t sel))
                      new-proj (ev (p/project model trace sel))
                      expected (- (- new-score old-score)
                                  (- new-proj old-proj))]
                  (and (h/finite? w)
                       (close? w expected 0.01)))))

;; --- Law #32: Simulate Address Set Consistency [T] §4.1.4 ---
;; For static models, simulate always produces the same address set.

(defspec law:simulate-address-set-consistency 100
  ;; [T] §4.1.4, Alg 16 — prerequisite: consistent address structure
  (prop/for-all [m gen-nonbranching]
                (let [{:keys [model args]} m
                      t1 (p/simulate model args)
                      t2 (p/simulate model args)]
                  (and (h/finite? (ev (:score t1)))
                       (h/finite? (ev (:score t2)))
                       (= (set (cm/addresses (:choices t1)))
                          (set (cm/addresses (:choices t2))))))))

;; --- Law #16: Mixture Kernel Stationarity [T] Prop 3.4.1 ---
;; Mix of stationary kernels is stationary. Random-mix and deterministic-cycle
;; of MH kernels on different addresses converge to the same posterior.

(t/deftest law:mixture-kernel-stationarity
  (t/testing "mix-kernels and cycle-kernels both converge to analytical posterior"
    ;; x ~ N(0,1), y ~ N(0,1), obs ~ N(x+y, 0.5), observe obs=2
    ;; s = x+y: prior N(0, sqrt(2)), likelihood N(s, 0.5)
    ;; Posterior mean = 8/4.5 = 1.778
    (let [model (dyn/auto-key
                  (gen []
                    (let [x (trace :x (dist/gaussian 0 1))
                          y (trace :y (dist/gaussian 0 1))]
                      (trace :obs (dist/gaussian (mx/add x y) 0.5)))))
          obs (cm/choicemap :obs (mx/scalar 2.0))
          init (:trace (p/generate model [] obs))
          analytical-mean (/ 8.0 4.5)
          run-chain
          (fn [select-fn n-steps burn]
            (loop [t init i 0 acc [] k (rng/fresh-key 13)]
              (if (>= i n-steps)
                acc
                (let [[k1 k2] (rng/split k)
                      sel (select-fn i k1)
                      {:keys [trace weight]} (p/regenerate model t sel)
                      w (ev weight)
                      accept? (u/accept-mh? w k1)
                      next-t (if accept? trace t)
                      x-val (ev (cm/get-value (cm/get-submap (:choices next-t) :x)))
                      y-val (ev (cm/get-value (cm/get-submap (:choices next-t) :y)))]
                  (recur next-t (inc i)
                         (if (>= i burn)
                           (conj acc (+ x-val y-val))
                           acc)
                         k2)))))
          mix-samples (run-chain
                       (fn [_ k] (sel/select (if (pos? (mx/item (rng/bernoulli k 0.5 []))) :x :y)))
                       4000 1000)
          cycle-samples (run-chain
                         (fn [i _k] (sel/select (if (even? i) :x :y)))
                         4000 1000)
          mean-mix (/ (reduce + mix-samples) (count mix-samples))
          mean-cycle (/ (reduce + cycle-samples) (count cycle-samples))]
      (t/is (close? mean-mix analytical-mean 0.3)
            (str "Mix mean=" mean-mix " expected=" analytical-mean))
      (t/is (close? mean-cycle analytical-mean 0.3)
            (str "Cycle mean=" mean-cycle " expected=" analytical-mean)))))

;; ---------------------------------------------------------------------------
;; HMC ACCEPTANCE laws [T] Alg 6
;; ---------------------------------------------------------------------------

;; --- Law #52: HMC Acceptance Correctness [T] Alg 6, §3.4.3 ---
;; HMC acceptance probability: alpha = min{1, exp(H(q,p) - H(q',p'))}
;; Leapfrog approximately preserves the Hamiltonian, yielding high acceptance.
;; Verified on Normal-Normal conjugate: x ~ N(0,1), y ~ N(x, 0.5), y=2.
;; Posterior: x|y ~ N(1.6, sqrt(0.2)).

(t/deftest law:hmc-acceptance-correctness
  ;; Normal-Normal conjugate: x ~ N(0,1), y ~ N(x, 0.5), observe y=2.
  ;; Posterior: x|y ~ N(1.6, sqrt(0.2)). Run HMC once, check both moments.
  (let [hmc-model (dyn/auto-key
                   (gen []
                        (let [x (trace :x (dist/gaussian 0 1))]
                          (trace :y (dist/gaussian x 0.5)))))
        obs (cm/choicemap :y (mx/scalar 2.0))
        samples (mcmc/hmc {:samples 500 :burn 200 :step-size 0.05
                           :leapfrog-steps 10 :addresses [:x]
                           :compile? false :device :cpu}
                          hmc-model [] obs)
        x-vals (mapv first samples)
        mean-x (/ (reduce + x-vals) (count x-vals))
        var-x (/ (reduce + (map #(let [d (- % mean-x)] (* d d)) x-vals))
                 (dec (count x-vals)))]
    (t/testing "HMC converges to correct posterior mean"
      (t/is (close? mean-x 1.6 0.3)
            (str "HMC posterior mean=" mean-x " expected=1.6")))
    (t/testing "HMC posterior variance is approximately correct"
      (t/is (close? var-x 0.2 0.15)
            (str "HMC posterior var=" var-x " expected=0.2")))))

;; ---------------------------------------------------------------------------
;; PROPOSAL TRAINING laws [T] Eq 3.8-3.9
;; ---------------------------------------------------------------------------

;; --- Law #53: Proposal Training Objective [T] Eq 3.8-3.9 ---
;; max_theta E_{(sigma+rho)~p}[log q(sigma; rho, theta)]
;;   = min_theta E_rho[D_KL(p(.|rho) || q(.; rho, theta))]
;;
;; Training a parametric model via gradient descent on negative log-likelihood
;; decreases the loss and converges to the MLE. This is the variational
;; principle that underlies proposal training (wake phase).

(t/deftest law:proposal-training-objective
  ;; Model: x ~ N(mu_param, 1), observe x=3. MLE: mu -> 3.
  ;; Train once, verify three properties of the variational principle.
  (let [train-model (dyn/auto-key
                     (gen []
                          (let [mu (param :mu (mx/scalar 0.0))
                                x (trace :x (dist/gaussian mu 1))]
                            x)))
        obs (cm/choicemap :x (mx/scalar 3.0))
        loss-grad-fn (learn/make-param-loss-fn train-model [] obs [:mu])
        init-params (mx/array [0.0])
        {:keys [loss-history params]}
        (learn/train {:iterations 100 :optimizer :adam :lr 0.05
                      :key (rng/fresh-key 42)}
                     loss-grad-fn init-params)
        first-10 (take 10 loss-history)
        last-10 (take-last 10 loss-history)
        mean-first (/ (reduce + first-10) (count first-10))
        mean-last (/ (reduce + last-10) (count last-10))
        final-mu (mx/realize (mx/index params 0))
        final-loss (last loss-history)
        theoretical-min (* 0.5 (js/Math.log (* 2 js/Math.PI)))]
    (t/testing "gradient descent decreases negative log-likelihood"
      (t/is (< mean-last mean-first)
            (str "Loss should decrease: first-10-mean=" mean-first
                 " last-10-mean=" mean-last)))
    (t/testing "optimized parameter converges to MLE"
      (t/is (close? final-mu 3.0 0.5)
            (str "MLE mu=" final-mu " expected=3.0")))
    (t/testing "loss at convergence approaches theoretical minimum"
      ;; Theoretical minimum NLL for N(mu, 1) at MLE: 0.5*log(2*pi) = 0.919
      (t/is (close? final-loss theoretical-min 0.1)
            (str "Final loss=" final-loss " theoretical-min=" theoretical-min)))))

;; ---------------------------------------------------------------------------
;; COMPILED PATH EQUIVALENCE laws [T] Ch 5
;; Laws #47-50: Compiled execution paths must produce the same probability
;; density p(tau; x) as the handler (interpreter) path.
;; ---------------------------------------------------------------------------

;; Compiled model pool: static models with compiled paths (no splice, no branch)
(def compiled-pool
  "Models that have :compiled-simulate in their schema."
  (filterv #(some? (get-in % [:model :schema :compiled-simulate]))
           non-branching-pool))

(def gen-compiled
  "Generator: pick a model with compiled execution paths."
  (gen/elements compiled-pool))

;; Multi-site compiled models (needed for regenerate, which requires 2+ addresses)
(def compiled-multisite-pool
  (filterv #(let [t (p/simulate (:model %) (:args %))
                  addrs (cm/addresses (:choices t))]
              (and (> (count addrs) 1)
                   (every? (fn [path] (= 1 (count path))) addrs)))
           compiled-pool))

(def gen-compiled-multisite
  "Generator: pick a compiled model with 2+ trace sites."
  (gen/elements compiled-multisite-pool))

;; --- Law #47: Compiled Simulate Equivalence ---
;; compiled simulate score = handler assess weight for the same choices.

(defspec law:compiled-simulate-equivalence 50
  ;; [T] Ch 5 -- compiled simulate preserves p(tau; x)
  ;; Simulate via compiled path, then verify the score via handler assess
  ;; on the same choices. Avoids needing matched PRNG keys.
  (prop/for-all [m gen-compiled]
                (let [{:keys [model args]} m
                      compiled-trace (p/simulate model args)
                      compiled-score (ev (:score compiled-trace))
                      handler-model (dyn/auto-key (gfi/strip-compiled model))
                      {:keys [weight]} (p/assess handler-model args
                                                  (:choices compiled-trace))
                      handler-score (ev weight)]
                  (close? compiled-score handler-score 1e-4))))

;; --- Law #48: Compiled Generate Equivalence ---
;; Fully constrained generate via compiled and handler paths produces
;; identical scores, weights, and return values.

(defspec law:compiled-generate-equivalence 50
  ;; [T] Ch 5 -- compiled generate preserves scores and weights
  (prop/for-all [m gen-compiled]
                (let [{:keys [model args]} m
                      source-trace (p/simulate model args)
                      constraints (:choices source-trace)
                      ;; Compiled generate
                      {:keys [trace weight]} (p/generate model args constraints)
                      compiled-score (ev (:score trace))
                      compiled-weight (ev weight)
                      compiled-retval (ev (:retval trace))
                      ;; Handler generate
                      handler-model (dyn/auto-key (gfi/strip-compiled model))
                      {:keys [trace weight]} (p/generate handler-model args
                                                         constraints)
                      handler-score (ev (:score trace))
                      handler-weight (ev weight)
                      handler-retval (ev (:retval trace))]
                  (and (close? compiled-score handler-score 1e-4)
                       (close? compiled-weight handler-weight 1e-4)
                       (close? compiled-retval handler-retval 1e-4)))))

;; --- Law #49: Compiled Update Equivalence ---
;; Update with new constraints via compiled and handler paths produces
;; identical weights (density ratios) and new scores.

(defspec law:compiled-update-equivalence 50
  ;; [T] Ch 5 -- compiled update preserves density ratios
  (prop/for-all [m gen-compiled]
                (let [{:keys [model args]} m
                      t1 (p/simulate model args)
                      t2 (p/simulate model args)
                      ;; Compiled update
                      {:keys [trace weight]} (p/update model t1 (:choices t2))
                      compiled-weight (ev weight)
                      compiled-new-score (ev (:score trace))
                      ;; Handler update: reconstruct starting trace via generate
                      handler-model (dyn/auto-key (gfi/strip-compiled model))
                      handler-t1 (:trace (p/generate handler-model args
                                                     (:choices t1)))
                      {:keys [trace weight]} (p/update handler-model handler-t1
                                                       (:choices t2))
                      handler-weight (ev weight)
                      handler-new-score (ev (:score trace))]
                  (and (close? compiled-weight handler-weight 1e-4)
                       (close? compiled-new-score handler-new-score 1e-4)))))

;; --- Law #50: Compiled Regenerate Equivalence ---
;; Regenerated trace score equals handler-assessed density for the same
;; choices. Weight is finite. Unselected addresses are preserved.

(defspec law:compiled-regenerate-equivalence 50
  ;; [T] Ch 5 -- compiled regenerate preserves weight semantics
  (prop/for-all [m gen-compiled-multisite]
                (let [{:keys [model args]} m
                      t (p/simulate model args)
                      addrs (cm/addresses (:choices t))
                      sel-addr (first (first addrs))
                      sel (sel/select sel-addr)
                      ;; Compiled regenerate
                      {:keys [trace weight]} (p/regenerate model t sel)
                      compiled-score (ev (:score trace))
                      compiled-weight (ev weight)
                      ;; Verify score via handler assess on regenerated choices
                      handler-model (dyn/auto-key (gfi/strip-compiled model))
                      {:keys [weight]} (p/assess handler-model args
                                                  (:choices trace))
                      handler-score (ev weight)
                      ;; Verify unselected addresses preserved
                      unselected (map first (rest addrs))]
                  (and (close? compiled-score handler-score 1e-4)
                       (h/finite? compiled-weight)
                       (every? (fn [a]
                                 (close?
                                  (ev (cm/get-value (cm/get-submap (:choices t) a)))
                                  (ev (cm/get-value (cm/get-submap (:choices trace) a)))
                                  1e-6))
                               unselected)))))

;; ---------------------------------------------------------------------------
;; Integration tests
;; ---------------------------------------------------------------------------

(t/deftest gfi-verify-integration
  (t/testing "gfi/verify runs algebraic laws on gaussian-chain"
    ;; Excludes statistical/training laws (tested by their own deftests):
    ;; :mixture-kernel-stationarity, :hmc-acceptance-correctness,
    ;; :proposal-training-objective. The integration test verifies the
    ;; gfi/verify API on algebraic laws, not MCMC/training convergence.
    (let [slow-laws #{:mixture-kernel-stationarity
                      :hmc-acceptance-correctness
                      :proposal-training-objective}
          algebraic-laws (->> gfi/laws
                              (remove #(slow-laws (:name %)))
                              (mapv :name))
          report (gfi/verify (:model gaussian-chain) (:args gaussian-chain)
                             :law-names algebraic-laws
                             :n-trials 5)]
      (t/is (:all-pass? report)
            (str "GFI laws failed: "
                 (pr-str (filterv #(not (:pass? %)) (:results report))))))))

(t/deftest gfi-verify-branching
  (t/testing "gfi/verify core laws on branching model"
    (let [report (gfi/verify (:model branching-model) (:args branching-model)
                             :tags #{:simulate :core}
                             :n-trials 10)]
      ;; Some laws (update-round-trip, decomposition) may not apply
      ;; to branching models — just verify simulate/generate/assess
      (t/is (>= (:total-pass report) (* 3 10))
            (str "Too few core law passes on branching model: "
                 (:total-pass report))))))

(t/run-tests)
