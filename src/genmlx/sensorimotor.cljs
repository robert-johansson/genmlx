(ns genmlx.sensorimotor
  "Hammer-style sensorimotor learning in GenMLX.

   Each procedural implication ((precondition, operation) ⇒ consequent) is a
   trace site whose Bernoulli parameter is the Beta posterior mean of the
   implication's success rate. Truth-value revision is Beta-Bernoulli
   conjugate update with exponential temporal projection on pseudocounts.

   The world model is a `ConceptMemory` record holding `Implication` records
   under three indices (by primary key, by precondition, by consequent). The
   indices give O(1) goal-directed lookup, antecedent-matching lookup, and
   primary lookup. Both records are immutable values; index updates produce
   new memories. Zero Clojure atoms.

   The kernels are two gen functions: `consequent-kernel` (used by both
   phases) and `action-kernel` (operant only). Phase differences live in
   the orchestration: which goal is active, whether outcomes depend on
   actions.

   See ../genmlx-lab/dev/docs/PLAN_SENSORIMOTOR_NARS_GENMLX.md."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn])
  (:require-macros [genmlx.gen :refer [gen]]))

;; =============================================================================
;; Implication: a procedural hypothesis ((pkey, op-key) ⇒ K) with Beta posterior
;; =============================================================================

(defrecord Implication
  ;; Invariants:
  ;; - (precondition, operation, consequent) are set at induction time and
  ;;   never change. update-implication asserts this. To represent a NEW
  ;;   consequent for the same antecedent+operation, induce a new implication.
  ;; - priority is reserved for future attention scheduling (deferred to the
  ;;   cognitive architecture layer). Currently 0.0 default; not read by any
  ;;   inference logic in this milestone.
  [precondition operation consequent
   alpha beta last-time update-count priority])

(defn fresh-implication
  "Initial Beta(2, 1) for a freshly induced implication.
   One positive observation atop a Beta(1, 1) uniform prior.
   Mean = 2/3 ≈ 0.667 — moderate confidence in a single confirmation."
  [pkey op-key consequent t]
  (map->Implication
    {:precondition pkey :operation op-key :consequent consequent
     :alpha 2.0 :beta 1.0 :last-time t :update-count 1 :priority 0.0}))

(defn project-implication
  "Apply exponential decay to pseudocounts (α-1, β-1) since :last-time.
   Identity at Δt=0. Fixed point at α=β=1 (prior). Multiplicative composition:
   project-then-project = project-by-sum-of-Δt.

   See PLAN_SENSORIMOTOR_NARS_GENMLX.md §3.2 and math-verifier Claim 1."
  [{:keys [alpha beta last-time] :as impl} now decay-rate]
  (let [decay (Math/pow decay-rate (- now last-time))]
    (assoc impl
      :alpha     (+ 1.0 (* (- alpha 1.0) decay))
      :beta      (+ 1.0 (* (- beta  1.0) decay))
      :last-time now)))

(defn revise-positive
  "Add 1 to α — consequent matched."
  [impl now]
  (-> impl (update :alpha + 1.0) (update :update-count inc) (assoc :last-time now)))

(defn revise-negative
  "Add δ to β — anticipated consequent did not occur."
  [impl now anticip-neg-evidence]
  (-> impl (update :beta + anticip-neg-evidence) (update :update-count inc) (assoc :last-time now)))

(defn beta-mean
  "Posterior mean = α / (α + β). Used as Bernoulli parameter and desire score."
  [{:keys [alpha beta]}]
  (/ alpha (+ alpha beta)))

(defn evidence-mass
  "Total pseudocount mass beyond prior: α + β - 2."
  [{:keys [alpha beta]}]
  (- (+ alpha beta) 2.0))

;; =============================================================================
;; ConceptMemory: opinionated world-model structure
;; =============================================================================
;;
;; Three indices, all immutable Clojure data:
;;
;;   :by-key          {[pkey op-key] -> Implication}        — primary lookup
;;   :by-precondition {pkey          -> #{[pkey op-key]}}   — antecedent index
;;   :by-consequent   {consequent-K  -> #{[pkey op-key]}}   — goal-directed index
;;
;; Index queries are O(1) hash lookups + O(matches) iteration. The structure
;; is the *what* (concept identity, indexing); MLX tensors carry the *how
;; much* (rates, weights, decision logits) at the orchestration call sites.

(defrecord ConceptMemory [by-key by-precondition by-consequent])

(def empty-memory (->ConceptMemory {} {} {}))

(defn- index-add [index k v]
  (update index k (fnil conj #{}) v))

(defn add-implication
  "Insert a new implication, maintaining all three indices. Throws on
   duplicate key — callers that want no-op-on-duplicate semantics (e.g.,
   induce-from-trial) must guard with lookup-implication first."
  [memory impl]
  (let [k [(:precondition impl) (:operation impl)]]
    (assert (not (contains? (:by-key memory) k))
            (str "add-implication: key " k " already present; "
                 "use update-implication for revisions or check via lookup-implication first"))
    (-> memory
        (update :by-key          assoc k impl)
        (update :by-precondition index-add (:precondition impl) k)
        (update :by-consequent   index-add (:consequent impl)   k))))

(defn update-implication
  "Replace an existing implication's record under its key. Asserts the
   consequent has not changed — that's a record-level invariant
   (Implication consequents are immutable once induced). Skips index
   updates under that invariant."
  [memory impl]
  (let [k [(:precondition impl) (:operation impl)]
        old (get (:by-key memory) k)]
    (assert (or (nil? old) (= (:consequent old) (:consequent impl)))
            (str "update-implication: consequent must not change "
                 "(was " (:consequent old) ", now " (:consequent impl) ")"))
    (update memory :by-key assoc k impl)))

;; =============================================================================
;; Memory queries — O(1) via indices
;; =============================================================================

(defn lookup-implication
  "Find the implication for (pkey, op-key), or nil."
  [memory pkey op-key]
  (get (:by-key memory) [pkey op-key]))

(defn implications-with-consequent
  "All implications whose :consequent matches K. O(matches), not O(memory)."
  [memory K]
  (->> (get (:by-consequent memory) K #{})
       (keep #(get (:by-key memory) %))))

(defn implications-matching-antecedent
  "All implications whose :precondition matches pkey. O(matches)."
  [memory pkey]
  (->> (get (:by-precondition memory) pkey #{})
       (keep #(get (:by-key memory) %))))

;; =============================================================================
;; Projection over the whole memory
;; =============================================================================

(defn project-all
  "Project every implication in memory to time `now`.
   Indices are unchanged (only Beta params + last-time mutate per record)."
  [memory now decay-rate]
  (update memory :by-key
    update-vals #(project-implication % now decay-rate)))

;; =============================================================================
;; Induction
;; =============================================================================

(defn induce-from-trial
  "If no implication exists for (pkey, op-key), create one with α=2, β=1.
   Otherwise leave memory unchanged — subsequent observations of this
   (pkey, op-key) flow through revise-after-observation, not induction.

   This is the ONLY caller in the codebase that wants no-op-on-duplicate
   semantics; add-implication itself throws on duplicate."
  [memory pkey op-key consequent t]
  (if (lookup-implication memory pkey op-key)
    memory
    (add-implication memory (fresh-implication pkey op-key consequent t))))

;; =============================================================================
;; Revision (anticipation reckoning)
;; =============================================================================

(defn revise-after-observation
  "End-of-trial revision: project the implication for (pkey, op-key) to t,
   then update its Beta posterior. Positive evidence (+1 α) when the
   observed consequent matches the implication's predicted consequent;
   negative evidence (+δ β) otherwise.

   No-op when no implication exists for (pkey, op-key) — induction is
   the orchestration's job before this call."
  [memory pkey op-key observed-consequent t anticip-neg-evidence decay-rate]
  (if-let [impl (lookup-implication memory pkey op-key)]
    (let [projected (project-implication impl t decay-rate)
          revised   (if (= (:consequent projected) observed-consequent)
                      (revise-positive projected t)
                      (revise-negative projected t anticip-neg-evidence))]
      (update-implication memory revised))
    memory))

(defn observe
  "Per-trial memory update: 'observed (pkey, op-key) ⇒ K at t.'

   Counts the observation exactly once:
   - If implication exists: project to t + revise (positive on match, negative
     on consequent mismatch).
   - If implication does NOT exist: induce — fresh-implication's Beta(2,1)
     embodies 'Beta(1,1) prior + this single positive observation,' so
     no further revision fires.

   Use this from orchestration. The lower-level induce-from-trial and
   revise-after-observation are still public for testing the math-verifier
   sequence (which calls them as separate observations)."
  [memory pkey op-key observed-K t anticip-neg-evidence decay-rate]
  (if (lookup-implication memory pkey op-key)
    (revise-after-observation memory pkey op-key observed-K t
                              anticip-neg-evidence decay-rate)
    (induce-from-trial memory pkey op-key observed-K t)))

;; =============================================================================
;; Per-particle rate tensor: bridge from symbolic memory to MLX
;; =============================================================================

(defn particle-rates
  "Compute [N, K] rate tensor: per-particle, per-operation predicted success
   rate of (pkey ⇒ goal) under each candidate operation.

   For nil goal (pre-learning, no preference active): all 0.5.
   For missing implication: 0.5 (uniform prior).
   For an implication matching (pkey, op-key, goal): its Beta mean."
  [particle-memories pkey goal op-keys]
  (mx/array
    (mapv (fn [memory]
            (mapv (fn [op-key]
                    (let [impl (lookup-implication memory pkey op-key)]
                      (cond
                        (nil? goal)                 0.5
                        (nil? impl)                 0.5
                        (= (:consequent impl) goal) (beta-mean impl)
                        :else                       (- 1.0 (beta-mean impl)))))
                  op-keys))
          particle-memories)))

(defn particle-decision-logits
  "[N, K] logits from per-particle rates. Threshold-gated softmax temperature:
   below threshold → uniform (motor babbling); above → rates / temperature
   with temperature decreasing in max confidence."
  [rates {:keys [decision-threshold min-temperature]
          :or {decision-threshold 0.51 min-temperature 0.1}}]
  (let [max-flat         (mx/amax rates [-1])
        max-per-particle (mx/expand-dims max-flat 1)
        above-thresh     (mx/greater-equal max-per-particle
                                            (mx/scalar decision-threshold))
        temperature      (mx/maximum (mx/scalar min-temperature)
                                      (mx/subtract (mx/scalar 1.0) max-per-particle))
        active-logits    (mx/divide rates temperature)
        babble-logits    (mx/zeros (mx/shape rates))]
    (mx/where above-thresh active-logits babble-logits)))

;; =============================================================================
;; Systematic resampling on log-weights — reusable across phases
;; =============================================================================

(defn systematic-resample
  "Systematic-sampling indices from log-weights. Pure function: takes the
   log-weights and a uniform-in-[0,1) offset; returns N indices into the
   particle array (0..N-1).

   Caller threads PRNG via the key arg → uniform draw (no host-side rand)."
  [log-weights u]
  (let [n      (count log-weights)
        max-w  (apply max log-weights)
        shifted (mapv #(Math/exp (- % max-w)) log-weights)
        z      (apply + shifted)
        norm   (mapv #(/ % z) shifted)
        cumsum (vec (reductions + norm))
        pairs  (mapv vector cumsum (range n))]
    ;; Floating-point drift may put cumsum[n-1] just under 1.0; clamp the
    ;; final position by including the n-1 fallback only when no bucket
    ;; matches (rare; if it fires often, the weights are degenerate).
    (vec
      (for [i (range n)]
        (let [pos (/ (+ i u) n)]
          (or (some (fn [[c idx]] (when (<= pos c) idx)) pairs)
              (dec n)))))))

;; =============================================================================
;; The kernels: action-kernel + consequent-kernel
;; =============================================================================
;;
;; Two gen functions, complementary roles:
;;
;;   action-kernel       traces :operation. Used by operant only. Classical has
;;                        no action site; the orchestration skips action-kernel.
;;
;;   consequent-kernel   traces :expected-consequent. Used by BOTH phases
;;                        identically. The constraint on this trace site
;;                        produces the per-particle particle weight via
;;                        vgenerate's importance-weight mechanism.
;;
;; This factoring matches BRIDGE_RFT_GENMLX_CONDITIONING.md §5's split of
;; act and consequence into separate gen functions. Each kernel splices in
;; the percept gen function (action-as-perception preserved at both sites).

(def action-kernel
  "Per-trial action selection. Used by operant; not used in classical.

   Args (vgenerate-batched):
     percept-suite     — percept gen function (spliced for action-as-perception)
     retina            — trial config (shared across particles)
     decision-logits   — [N, K] desire-driven categorical logits

   Trace sites:
     :percept (splice)
     :operation        — [N]-shaped operation index per particle

   Returns: the operation index per particle."
  (dyn/auto-key
    (gen [t kernel-args]
      (let [_percept (splice :percept (:percept-suite kernel-args)
                              [(:retina kernel-args)])
            op-idx   (trace :operation
                            (dist/categorical (:decision-logits kernel-args)))]
        op-idx))))

(def consequent-kernel
  "Per-trial consequent prediction. Used identically by both phases.

   Args (vgenerate-batched):
     percept-suite     — percept gen function (spliced)
     retina            — trial config (shared)
     rates             — [N]-shaped success rate per particle (already gathered
                          for the chosen action in operant; just the antecedent's
                          predicted rate in classical)

   Trace sites:
     :percept (splice)
     :expected-consequent — Bernoulli([N]). Constrained externally per particle
                            to encode whether the actual outcome matched the
                            implication's predicted consequent (1.0 = match,
                            0.0 = mismatch). The constraint produces the
                            per-particle particle weight.

   Returns: nil (orchestration extracts weight from the vtrace)."
  (dyn/auto-key
    (gen [t kernel-args]
      (let [_percept (splice :percept (:percept-suite kernel-args)
                              [(:retina kernel-args)])
            _outcome (trace :expected-consequent
                            (dist/bernoulli (:rates kernel-args)))]
        nil))))

(defn gather-chosen-rates
  "Per-particle rate of the chosen operation: [N, K] rates × [N] op-idxs → [N]."
  [rates op-idxs]
  (mx/squeeze
    (mx/take-along-axis rates (mx/expand-dims op-idxs 1) 1)
    [1]))
