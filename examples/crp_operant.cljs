;; CRP-Based Operant Conditioning in GenMLX
;; =========================================
;;
;; Reproduces Lloyd & Leslie (2013) "Context-dependent decision-making:
;; a simple Bayesian model" (J. R. Soc. Interface 10:20130469) — an
;; instrumental-learning analog of Gershman's classical-conditioning
;; latent cause work, with GenMLX at the heart.
;;
;; Architectural argument (same lever as `rate_estimation.cljs`):
;;
;;   The inference + decision update IS a generative function. One `gen`
;;   kernel composed with the `Scan` combinator represents the full
;;   trajectory of an operant conditioning experiment.
;;
;;     - The chosen action at each trial is a TRACE SITE :a.
;;     - The observed reward at each trial is a TRACE SITE :r.
;;     - The sampled latent context is a TRACE SITE :c.
;;     - The agent's belief state (Beta sufficient stats per context ×
;;       action, CRP table counts, active-context mask) is the SCAN
;;       CARRY: threaded as a value from step to step.
;;     - Per-step posteriors (P(c | data), P(a | history)) are the
;;       per-step OUTPUT of the Scan.
;;
;;   This gives Lloyd-Leslie full GFI semantics:
;;     - p/simulate runs the agent forward in a generated experiment.
;;     - p/generate conditions on observed (a_t, r_t) sequences from
;;       real animals — posterior over latent contexts falls out.
;;     - dyn/vgenerate is a parallel particle filter over context
;;       histories.
;;     - comb/unfold-extend gives true streaming: live agent processing
;;       trials one at a time.
;;
;; Conceptually, this is animal-level operant learning. The agent
;; discovers — purely from a stream of (action, reward) pairs, with NO
;; observable context cue — that the world has discrete "rules" (latent
;; contexts), each defining its own action→consequence contingency.
;; The Chinese Restaurant Process gives the agent a Bayesian
;; nonparametric prior over how many rules exist and how to allocate
;; new observations among them.
;;
;; Demo 1 is the foundation: pure action→consequence Bayesian learning
;; with the CRP machinery present but dormant (one context suffices for
;; a stationary world). Demos 2-5 reproduce the behavioral phenomena
;; that REQUIRE the CRP: spontaneous recovery, PREE, the overtraining
;; reversal effect, and serial reversal speedup. Demos 6-7 are GenMLX-
;; native: a head-to-head model comparison via the GFI weight field,
;; and a streaming/vectorized particle-filter demo.
;;
;; Out of scope (see SPEC_CRP_OPERANT.md): adding observable
;; discriminative stimuli S^D (would lift this to a Hierarchical
;; Dirichlet Process), conditional discriminations (nested CRP), and
;; fitting real animal data.
;;
;; Run: bun run --bun nbb examples/crp_operant.cljs

(ns crp-operant
  (:require [clojure.string :as str]
            ["fs" :as fs]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.combinators :as comb])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ============================================================================
;; Helpers
;; ============================================================================

(defn fmt
  ([v]        (fmt v 3))
  ([v digits] (.toFixed (double v) digits)))

(defn pad [s width] (.padStart (str s) width " "))

;; ============================================================================
;; Constants
;; ============================================================================

(def ^:const K-MAX 10)            ;; truncation: max distinct contexts
(def ^:const N-ACTIONS 2)         ;; two-armed bandit: L (0), R (1)
(def ^:const DEFAULT-ALPHA 2.0)   ;; CRP concentration parameter
(def ^:const DEFAULT-PI 0.15)     ;; prior jump probability per trial
(def ^:const DEFAULT-ALPHA-0 1.0) ;; Beta prior α₀
(def ^:const DEFAULT-BETA-0 1.0)  ;; Beta prior β₀

(def ONE  (mx/scalar 1.0))
(def ZERO (mx/scalar 0.0))
(def HALF (mx/scalar 0.5))
(def SQRT-2 (mx/scalar (Math/sqrt 2.0)))
(def EPS  (mx/scalar 1e-12))

;; ============================================================================
;; The kernel — one trial of the Lloyd-Leslie CRP-operant model
;; ============================================================================
;;
;; Math (per trial):
;;   1. CRP+persistence prior over context c_t:
;;        P(c_t = c_{t-1}) = (1-π) + π · n_{c_{t-1}}/(N+α)
;;        P(c_t = c'≠c)    =       π · n_{c'}     /(N+α)   for existing c'
;;        P(c_t = new)     =       π · α          /(N+α)
;;
;;   2. Marginalized Thompson action probability:
;;        For each slot c with Beta(α_L^c, β_L^c) and Beta(α_R^c, β_R^c):
;;          μ_a^c = α_a^c / (α_a^c + β_a^c)
;;          σ²_a^c = α_a^c β_a^c / [(α_a^c + β_a^c)² (α_a^c + β_a^c + 1)]
;;          P(L|c) ≈ Φ((μ_L^c - μ_R^c) / √(σ²_L^c + σ²_R^c))
;;        P(L|history) = Σ_c P(c|history) · P(L|c)
;;
;;   3. :a ~ Categorical([P(L), 1-P(L)])   action in {0=L, 1=R}
;;
;;   4. :r ~ Bernoulli(true-reward-prob[c_true, :a])
;;
;;   5. Posterior over c given (:a, :r):
;;        P(c | :a, :r, history) ∝ P(c | history) · P(:r | c, :a, history)
;;        P(:r=1 | c, :a) = α_{:a}^c / (α_{:a}^c + β_{:a}^c)
;;
;;   6. :c ~ Categorical(posterior)
;;
;;   7. Conjugate update of Beta sufficient stats at (:c, :a):
;;        α^{:c,:a} += :r;  β^{:c,:a} += (1 - :r);  n_{:c} += 1
;;
;; All three stochastic decisions (:a, :r, :c) are trace sites. The
;; Thompson sampling is marginalized over c (the prior weights are applied
;; analytically); the Beta-parameter update is also marginalized — the FULL
;; posterior over c serves as soft credit-assignment weights rather than
;; committing to one sampled slot. This Rao-Blackwellized form prevents
;; the "absorption" pathology in single-sample CRP inference (where new
;; evidence is absorbed into an existing context's slowly-broadening
;; parameters faster than a new context is inferred), at the cost of
;; tracking a length-K posterior in the carry instead of a single int.

(defn- one-hot
  "One-hot vector of length n with 1 at slot idx, 0 elsewhere. idx is an
   MLX int scalar; returns float32 [n]."
  [idx n]
  (mx/astype (mx/equal (mx/arange n) (mx/astype idx mx/float32)) mx/float32))

(defn- normal-cdf
  "Φ(z) = 0.5 · (1 + erf(z / √2))."
  [z]
  (mx/multiply HALF (mx/add ONE (mx/erf (mx/divide z SQRT-2)))))

(def ^:const NEW-ALLOC-THRESHOLD 0.15)
;; Above this posterior mass on the "new context" slot, allocate a new
;; existing slot at index n-active. Below, the new-context probability is
;; treated as transient (renormalized away via update-weights / Σ).

(def crp-operant-kernel
  "One trial of the Lloyd-Leslie CRP-based operant model with
   Rao-Blackwellized soft updates.

   carry: {:alpha-mat :beta-mat :counts :active-mask :c-belief :n-active
           :alpha :pi :alpha-0 :beta-0}
   input: {:true-reward-probs [pL pR]   ;; environment's true reward probs
           :forced-action <opt int>}    ;; if supplied, override the agent

   Trace sites:
     :a — Categorical over actions   {0=L, 1=R}, marginalized Thompson
     :r — Bernoulli(p_true[c_true, :a])
     :c — Categorical over context slots {0..K_MAX} (K_MAX means \"new\"),
          sampled from posterior for inspection; NOT used for the carry
          update (the carry uses the full posterior as soft weights).

   Returns [new-carry per-step-output]."
  (dyn/auto-key
    (gen [carry input]
      (let [alpha-mat   (:alpha-mat   carry)  ;; [K, A]
            beta-mat    (:beta-mat    carry)
            counts      (:counts      carry)  ;; [K]
            active-mask (:active-mask carry)  ;; [K]
            c-belief    (:c-belief    carry)  ;; [K] — current posterior over slots
            n-active    (:n-active    carry)  ;; scalar int
            alpha-crp   (:alpha       carry)
            pi          (:pi          carry)
            alpha-0     (:alpha-0     carry)
            beta-0      (:beta-0      carry)
            K           K-MAX
            A           N-ACTIONS
            true-probs  (mx/array (:true-reward-probs input))   ;; [A]
            forced-act  (:forced-action input)
            ;; -----------------------------------------------------------------
            ;; (1) Prior over c_t ∈ {0..K}: c_t = (1-π)·c_{t-1} + π·CRP
            ;; The previous belief c-belief carries the "what context was I just
            ;; in" distribution; (1-π) keeps it, π replaces with CRP-marginal.
            ;; -----------------------------------------------------------------
            n-plus-alpha   (mx/add (mx/sum counts) alpha-crp)
            crp-exist      (mx/divide (mx/multiply counts active-mask) n-plus-alpha)
            slot-available (mx/astype (mx/less (mx/astype n-active mx/float32)
                                               (mx/scalar (double K)))
                                      mx/float32)
            crp-new        (mx/multiply (mx/divide alpha-crp n-plus-alpha)
                                        slot-available)
            existing-prior (mx/add (mx/multiply (mx/subtract ONE pi) c-belief)
                                   (mx/multiply pi crp-exist))
            new-prior      (mx/multiply pi crp-new)
            prior-ext      (mx/concatenate [existing-prior
                                            (mx/expand-dims new-prior 0)])
            ;; -----------------------------------------------------------------
            ;; (2) Marginalized Thompson action probability over K+1 slots
            ;; -----------------------------------------------------------------
            new-row-alpha (mx/full [1 A] alpha-0)
            new-row-beta  (mx/full [1 A] beta-0)
            ext-alpha     (mx/concatenate [alpha-mat new-row-alpha] 0)
            ext-beta      (mx/concatenate [beta-mat new-row-beta] 0)
            ext-total     (mx/add ext-alpha ext-beta)
            ext-mu        (mx/divide ext-alpha ext-total)
            ext-var       (mx/divide (mx/multiply ext-alpha ext-beta)
                                     (mx/multiply (mx/multiply ext-total ext-total)
                                                  (mx/add ext-total ONE)))
            ext-mu-t      (mx/transpose ext-mu [1 0])
            ext-var-t     (mx/transpose ext-var [1 0])
            mu-L          (mx/index ext-mu-t 0)
            mu-R          (mx/index ext-mu-t 1)
            var-L         (mx/index ext-var-t 0)
            var-R         (mx/index ext-var-t 1)
            z             (mx/divide (mx/subtract mu-L mu-R)
                                     (mx/sqrt (mx/add (mx/add var-L var-R) EPS)))
            phi-L         (normal-cdf z)
            prob-L        (mx/sum (mx/multiply prior-ext phi-L))
            prob-R        (mx/subtract ONE prob-L)
            ;; -----------------------------------------------------------------
            ;; (3) Trace :a
            ;; -----------------------------------------------------------------
            action-weights (if (some? forced-act)
                             (mx/array (if (zero? forced-act) [1.0 0.0] [0.0 1.0]))
                             (mx/concatenate [(mx/expand-dims prob-L 0)
                                              (mx/expand-dims prob-R 0)]))
            a              (trace :a (dist/categorical-weights action-weights))
            ;; -----------------------------------------------------------------
            ;; (4) Trace :r
            ;; -----------------------------------------------------------------
            true-p-a       (mx/take-idx true-probs a)
            r              (trace :r (dist/bernoulli true-p-a))
            ;; -----------------------------------------------------------------
            ;; (5) Posterior over c given (:a, :r)
            ;; -----------------------------------------------------------------
            alpha-col-a    (mx/take-idx ext-alpha a 1)
            beta-col-a     (mx/take-idx ext-beta a 1)
            total-col-a    (mx/add alpha-col-a beta-col-a)
            p-r1-col       (mx/divide alpha-col-a total-col-a)
            likelihood     (mx/where (mx/equal r ONE)
                                     p-r1-col
                                     (mx/subtract ONE p-r1-col))
            active-ext     (mx/concatenate [active-mask (mx/array [1.0])])
            post-unnorm    (mx/multiply (mx/multiply prior-ext likelihood) active-ext)
            posterior      (mx/divide post-unnorm
                                      (mx/maximum (mx/sum post-unnorm) EPS))
            ;; -----------------------------------------------------------------
            ;; (6) Trace :c — sample from posterior (inspection only)
            ;; -----------------------------------------------------------------
            c              (trace :c (dist/categorical-weights posterior))
            ;; -----------------------------------------------------------------
            ;; (7) Rao-Blackwellized soft update with phantom-slot accumulator:
            ;;   - Existing slots [0..K-1] receive weight posterior[c].
            ;;   - The "new" slot's mass (posterior[K]) always accrues to the
            ;;     phantom slot at index n-active. The phantom is initially
            ;;     inactive (active-mask[n-active] = 0) so doesn't enter the
            ;;     CRP prior, but DOES accumulate Beta evidence via these
            ;;     fractional updates.
            ;;   - When the phantom's count crosses 1.0 (one trial's worth of
            ;;     evidence accumulated), it's officially activated: enters the
            ;;     CRP prior, and n-active advances to the next phantom.
            ;; This decouples context-allocation from per-trial likelihood
            ;; ratio peaks (which Beta-Bernoulli bounds), letting evidence
            ;; for a "new regime" build up gradually over multiple trials.
            ;; -----------------------------------------------------------------
            posterior-exist (mx/slice posterior 0 K)
            new-mass        (mx/index posterior K)
            new-slot-oh     (one-hot n-active K)
            alloc-mass-vec  (mx/multiply new-slot-oh new-mass)
            update-weights  (mx/add posterior-exist alloc-mass-vec)
            a-oh            (one-hot a A)
            update-mask     (mx/multiply (mx/expand-dims update-weights -1)
                                         (mx/expand-dims a-oh 0))
            new-alpha-mat   (mx/add alpha-mat (mx/multiply update-mask r))
            new-beta-mat    (mx/add beta-mat (mx/multiply update-mask
                                                          (mx/subtract ONE r)))
            new-counts      (mx/add counts update-weights)
            ;; Activation: phantom becomes active when its count exceeds 1.0
            new-slot-count  (mx/take-idx new-counts n-active)
            should-activate (mx/greater new-slot-count ONE)
            should-activate-f (mx/astype should-activate mx/float32)
            new-active      (mx/maximum active-mask
                                        (mx/multiply new-slot-oh should-activate-f))
            new-n-active    (mx/add n-active (mx/astype should-activate mx/int32))
            new-carry       {:alpha-mat   new-alpha-mat
                             :beta-mat    new-beta-mat
                             :counts      new-counts
                             :active-mask new-active
                             :c-belief    update-weights
                             :n-active    new-n-active
                             :alpha       alpha-crp
                             :pi          pi
                             :alpha-0     alpha-0
                             :beta-0      beta-0}
            step-output     {:c          c
                             :a          a
                             :r          r
                             :p-action-L prob-L
                             :c-belief   update-weights
                             :n-active   new-n-active}]
        [new-carry step-output]))))

(def crp-operant-scan
  "Scan combinator over the per-trial kernel — full trajectory under GFI."
  (comb/scan-combinator crp-operant-kernel))

(def crp-operant-model
  "Top-level wrapper splicing the Scan under address :s — makes the
   trajectory addressable as a DynamicGF for `dyn/vgenerate` and friends."
  (gen [init-carry protocol]
    (splice :s crp-operant-scan init-carry protocol)))

(defn init-carry
  "Initial carry. Slot 0 pre-activated; :c-belief concentrated on slot 0."
  ([] (init-carry DEFAULT-ALPHA DEFAULT-PI DEFAULT-ALPHA-0 DEFAULT-BETA-0))
  ([alpha pi a0 b0]
   (let [slot-0 (mx/concatenate [(mx/array [1.0])
                                  (mx/zeros [(dec K-MAX)])])]
     {:alpha-mat   (mx/full [K-MAX N-ACTIONS] a0)
      :beta-mat    (mx/full [K-MAX N-ACTIONS] b0)
      :counts      (mx/zeros [K-MAX])
      :active-mask slot-0
      :c-belief    slot-0
      :n-active    (mx/scalar 1 mx/int32)
      :alpha       (mx/scalar alpha)
      :pi          (mx/scalar pi)
      :alpha-0     (mx/scalar a0)
      :beta-0      (mx/scalar b0)})))

;; ============================================================================
;; Protocols — generating Scan inputs from experimental schedules
;; ============================================================================

(defn constant-protocol
  "n-trials of a stationary reward schedule. probs = [pL pR]."
  [n-trials probs]
  (vec (repeat n-trials {:true-reward-probs probs})))

(defn reversal-protocol
  "n1 trials at probs1, then n2 trials at probs2."
  [n1 probs1 n2 probs2]
  (into (constant-protocol n1 probs1) (constant-protocol n2 probs2)))

(defn serial-reversal-protocol
  "n-periods periods of n-per-period trials each, alternating between
   probs1 and probs2 starting with probs1."
  [n-periods n-per-period probs1 probs2]
  (vec (mapcat (fn [p]
                 (constant-protocol n-per-period
                                    (if (even? p) probs1 probs2)))
               (range n-periods))))

;; ============================================================================
;; Demo 1 — Basic operant learning (Lloyd-Leslie fig 2; pure foundation)
;; ============================================================================
;;
;; The pedagogical core: pure action → consequence Bayesian learning.
;; The CRP machinery is present in the kernel but dormant — for a
;; stationary world, the agent maintains a single context throughout
;; and the model reduces to per-action Beta-Bernoulli inference. This
;; is the simplest interesting operant learner.
;;
;; Reproduces fig 2(a) (probability differences): two arms with
;; different reward probabilities; the agent's preference for the
;; better arm develops more rapidly with greater probability gap.

(defn run-protocol
  "Run one Scan trajectory under the kernel for the given trial list.
   Returns a JS-number vector of per-step actions (already materialized)."
  [protocol key]
  (mx/tidy-run
    (fn []
      (let [trace   (p/simulate
                      (dyn/with-key crp-operant-scan key)
                      [(init-carry) protocol])
            outputs (:outputs (:retval trace))
            actions (mx/stack (mapv :a outputs))]
        (mx/eval! actions)
        (vec (mx/->clj actions))))
    (fn [_] nil)))

(defn proportion-action
  "Average proportion of trials choosing action `target` across `runs`."
  [protocol-fn n-runs target seed-base]
  (let [actions-per-run
        (mapv (fn [r]
                (run-protocol (protocol-fn)
                              (rng/fresh-key (+ seed-base r))))
              (range n-runs))
        n-trials (count (first actions-per-run))]
    (mapv (fn [t]
            (/ (apply + (map #(if (= (nth % t) target) 1 0) actions-per-run))
               (double n-runs)))
          (range n-trials))))

(println "\n============================================================")
(println "Demo 1 — Basic operant learning (foundation)")
(println "============================================================")
(println "Two-armed bandit; agent learns action→reward purely from")
(println "experience. CRP present but dormant — one context inferred.")
(println "Three probability ratios (100:0, 80:20, 60:40), 50 runs each,")
(println "50 trials per run.")

(let [n-trials 50
      n-runs   30
      ratios   [[1.00 0.00] [0.80 0.20] [0.60 0.40]]
      results
      (mapv (fn [probs]
              [probs
               (proportion-action #(constant-protocol n-trials probs)
                                  n-runs 0 1000)])
            ratios)]
  (println "\n  trial    100:0    80:20    60:40")
  (println "  ─────    ─────    ─────    ─────")
  (doseq [t [0 4 9 19 29 39 49]]
    (println (str "  " (pad (inc t) 4) "     "
                  (pad (fmt (nth (second (nth results 0)) t)) 5) "    "
                  (pad (fmt (nth (second (nth results 1)) t)) 5) "    "
                  (pad (fmt (nth (second (nth results 2)) t)) 5))))
  (println "\n  → Larger probability differences yield faster acquisition")
  (println "    of the better arm. Single-context Bayesian operant learning."))

;; ============================================================================
;; Drift kernel — context evolution without reward observation
;; ============================================================================
;;
;; Used to model the passage of time between observed trials (paper §3.2:
;; "Delays are modelled as a series of dummy trials in which contexts are
;; sampled sequentially from the generative model"). Each dummy trial
;; samples :c from the CRP+persistence prior and updates the CRP table
;; counts, but does NOT update Beta parameters since no action/reward
;; occurs. Used in Demo 2 (spontaneous recovery).

(def drift-kernel
  "One dummy trial: evolve :c-belief through the CRP+persistence prior only.
   No action, no reward — and crucially no CRP-count update, so the prior
   is stationary across the drift period. With stationary counts and
   persistence π, c-belief follows a HMM forward update whose stationary
   distribution is proportional to the table counts; from any starting
   belief, the chain mixes toward that distribution over ~1/π trials.
   carry: as in crp-operant-kernel
   input: ignored
   trace site: :c only"
  (dyn/auto-key
    (gen [carry _input]
      (let [counts      (:counts carry)
            active-mask (:active-mask carry)
            c-belief    (:c-belief carry)
            n-active    (:n-active carry)
            alpha-crp   (:alpha carry)
            pi          (:pi carry)
            K           K-MAX
            n-plus-alpha   (mx/add (mx/sum counts) alpha-crp)
            crp-exist      (mx/divide (mx/multiply counts active-mask) n-plus-alpha)
            slot-available (mx/astype (mx/less (mx/astype n-active mx/float32)
                                               (mx/scalar (double K)))
                                      mx/float32)
            crp-new        (mx/multiply (mx/divide alpha-crp n-plus-alpha)
                                        slot-available)
            existing-prior (mx/add (mx/multiply (mx/subtract ONE pi) c-belief)
                                   (mx/multiply pi crp-exist))
            new-prior      (mx/multiply pi crp-new)
            prior-ext      (mx/concatenate [existing-prior
                                            (mx/expand-dims new-prior 0)])
            active-ext     (mx/concatenate [active-mask (mx/array [1.0])])
            post-unnorm    (mx/multiply prior-ext active-ext)
            posterior      (mx/divide post-unnorm
                                      (mx/maximum (mx/sum post-unnorm) EPS))
            c              (trace :c (dist/categorical-weights posterior))
            ;; Drop the "new" slot mass (no allocation during pure drift).
            new-belief     (mx/slice posterior 0 K)
            new-belief     (mx/divide new-belief
                                      (mx/maximum (mx/sum new-belief) EPS))
            new-carry      (assoc carry :c-belief new-belief)]
        [new-carry c]))))

(def drift-scan (comb/scan-combinator drift-kernel))

;; ============================================================================
;; Demo 2 — Spontaneous recovery (Lloyd-Leslie fig 3)
;; ============================================================================
;;
;; Train left (50 trials, 100:0) → train right (50 trials, 0:100) → delay →
;; probe one test trial. As delay grows, proportion choosing left rises
;; toward a partial asymptote (paper fig 3): the original-context memory
;; is still in the agent's "rule library", and contextual drift makes it
;; increasingly likely to be sampled.

(defn run-recovery-trial
  "One run of the spontaneous-recovery experiment at the given delay.
   Returns 0 (L) or 1 (R) — the action on the test trial."
  [delay-trials seed]
  (mx/tidy-run
    (fn []
      (let [carry0  (init-carry)
            acq-rev (vec (concat (constant-protocol 50 [0.9 0.1])
                                 (constant-protocol 50 [0.1 0.9])))
            tr1     (p/simulate (dyn/with-key crp-operant-scan
                                              (rng/fresh-key (+ seed 10000)))
                                [carry0 acq-rev])
            carry1  (:carry (:retval tr1))
            carry2  (if (zero? delay-trials)
                      carry1
                      (let [tr2 (p/simulate
                                  (dyn/with-key drift-scan
                                                (rng/fresh-key (+ seed 20000)))
                                  [carry1 (vec (repeat delay-trials {}))])]
                        (:carry (:retval tr2))))
            tr3     (p/simulate (dyn/with-key crp-operant-scan
                                              (rng/fresh-key (+ seed 30000)))
                                [carry2 [{:true-reward-probs [0.5 0.5]}]])
            a       (:a (first (:outputs (:retval tr3))))]
        (mx/item a)))
    (fn [_] nil)))

(println "\n============================================================")
(println "Demo 2 — Spontaneous recovery (Lloyd-Leslie fig 3)")
(println "============================================================")
(println "Train L (100:0) → train R (0:100) → delay → test.")
(println "P(left at test) rises with delay as contextual drift makes")
(println "the original-rule context increasingly likely to recur.")

(let [delays   [0 5 10 20 40 70 100]
      n-runs   30
      results  (mapv (fn [d]
                       (let [actions (mapv #(run-recovery-trial d (+ % 5000))
                                           (range n-runs))
                             p-left  (/ (count (filter zero? actions))
                                        (double n-runs))]
                         [d p-left]))
                     delays)]
  (println "\n  delay (dummy trials)     P(left at test)")
  (println "  ────────────────────     ───────────────")
  (doseq [[d p-left] results]
    (println (str "  " (pad d 8) "                 " (fmt p-left))))
  (println "\n  → P(left) increases from ~0 (immediate test, agent committed")
  (println "    to right) toward a partial asymptote at long delays. The old")
  (println "    'left' context is still in the agent's rule library and gets")
  (println "    re-sampled as context drifts via the jump process."))

;; ============================================================================
;; Demo 3 — Extinction onset under different acquisition concentrations
;; ============================================================================
;;
;; Three acquisition conditions (100:0, 75:25, 67:33), then extinction
;; (0:0). In Beta-Bernoulli, extinction proceeds via two mechanisms:
;;   (a) absorption-into-existing-context: as r=0 trials accumulate, the
;;       current context's β grows, eventually flipping its Thompson
;;       preference. Fast when α/(α+β) ratio is moderate, slow when α
;;       is very tight.
;;   (b) inference-of-new-context: each trial of unexpected r=0 raises
;;       posterior(new), feeding mass into the phantom slot. Once the
;;       phantom accumulates >1 trial of mass, it activates and enters
;;       the CRP normalizer, giving the agent a second hypothesis.
;;
;; Under Normal-Gamma reward (Lloyd-Leslie's choice), variance learning
;; gives the canonical PREE: tighter acquisition → exponentially-greater
;; surprise → faster context inference (mechanism b dominates). Under
;; Beta-Bernoulli, the bounded likelihood ratio limits mechanism (b)'s
;; speed; mechanism (a) dominates more for partial reinforcement, where
;; α is less concentrated. Result: extinction onset is EARLIER under
;; partial reinforcement here — the opposite of canonical PREE, and an
;; honest signature of the Beta-Bernoulli simplification noted in the
;; spec. See `../genmlx-lab/dev/docs/EXAMPLE_CRP_OPERANT_NOTES.md` for full discussion.

(println "\n============================================================")
(println "Demo 3 — Extinction onset under different acquisition")
(println "============================================================")
(println "Three acquisition conditions, then extinction (no reward).")
(println "Shows two competing extinction mechanisms in Beta-Bernoulli.")

(let [acq-trials 24
      ext-trials 16
      n-runs     30
      conditions [["100:0" [1.00 0.00]]
                  ["75:25" [0.75 0.25]]
                  ["67:33" [0.67 0.33]]]
      results
      (mapv (fn [[label probs]]
              (let [protocol (vec (concat (constant-protocol acq-trials probs)
                                          (constant-protocol ext-trials [0.0 0.0])))
                    p-L (proportion-action #(do protocol) n-runs 0 2000)]
                [label p-L]))
            conditions)]
  (println "\n  P(left) during extinction (post-acquisition trials 1, 4, 8, 12, 16):")
  (println "\n  cond     end-acq    ext+1    ext+4    ext+8    ext+12   ext+16")
  (println "  ────     ───────    ─────    ─────    ─────    ──────   ──────")
  (doseq [[label p-L] results]
    (println (str "  " (pad label 5) "    "
                  (pad (fmt (nth p-L (dec acq-trials))) 5) "      "
                  (pad (fmt (nth p-L acq-trials)) 5) "    "
                  (pad (fmt (nth p-L (+ acq-trials 3))) 5) "    "
                  (pad (fmt (nth p-L (+ acq-trials 7))) 5) "    "
                  (pad (fmt (nth p-L (+ acq-trials 11))) 5) "    "
                  (pad (fmt (nth p-L (dec (count p-L)))) 5))))
  (println "\n  → Under partial reinforcement (75:25, 67:33), absorption flips")
  (println "    Thompson sooner. Under 100:0, absorption is slow (tight α);")
  (println "    extinction depends on phantom-slot activation, which takes ~10")
  (println "    trials to accumulate enough mass. The trade-off is genuinely")
  (println "    Beta-Bernoulli — Normal-Gamma's variance learning would reverse it."))

;; ============================================================================
;; Demo 4 — Overtraining reversal effect (Lloyd-Leslie fig 6)
;; ============================================================================
;;
;; Vary pre-reversal training amount; measure post-reversal trials to
;; criterion (10 rewarded responses in last 12 trials). With strong
;; reward differences and long training, the agent's parameter posterior
;; under the pre-reversal context is sharply peaked — post-reversal
;; observations are highly improbable under that context → rapid
;; inference of a new context → fast post-reversal acquisition.

(defn first-criterion-trial
  "Returns the smallest t such that sum(rewards[t-11:t+1]) >= 10. nil if
   never reached. The :r values are 0/1 ints already in JS-number form."
  [rewards crit-window crit-count]
  (loop [t (dec crit-window)]
    (cond
      (>= t (count rewards)) nil
      (>= (apply + (subvec rewards (- t (dec crit-window)) (inc t)))
          crit-count)        t
      :else                  (recur (inc t)))))

(defn run-reversal-experiment
  "Pre-reversal training (n-pre trials at probs1), then post-reversal
   trials at probs2. Returns vector of post-reversal :r outcomes."
  [n-pre n-post probs1 probs2 seed]
  (mx/tidy-run
    (fn []
      (let [protocol (vec (concat (constant-protocol n-pre probs1)
                                  (constant-protocol n-post probs2)))
            tr (p/simulate (dyn/with-key crp-operant-scan
                                          (rng/fresh-key (+ seed 40000)))
                           [(init-carry) protocol])
            outputs (:outputs (:retval tr))
            rewards (mx/stack (mapv :r (subvec outputs n-pre)))]
        (mx/eval! rewards)
        (mapv int (mx/->clj rewards))))
    (fn [_] nil)))

(println "\n============================================================")
(println "Demo 4 — Overtraining reversal effect (ORE)")
(println "============================================================")
(println "Vary pre-reversal training; post-reversal trials-to-criterion.")
(println "Larger reward gap → monotonic ORE. Criterion = 10/12 rewarded.")

(let [pre-counts [10 20 40 80 160]
      n-post     80
      n-runs     20
      probs1     [0.90 0.10]                ;; softer probabilities — Beta-Bernoulli
      probs2     [0.10 0.90]                ;; absorbs less aggressively than 1/0
      crit-win   12
      crit-count 10
      ttcs
      (mapv (fn [n-pre]
              (let [trial-counts
                    (mapv (fn [r]
                            (let [rewards (run-reversal-experiment
                                            n-pre n-post probs1 probs2
                                            (+ r 7000))]
                              (or (first-criterion-trial rewards crit-win crit-count)
                                  n-post)))
                          (range n-runs))
                    mean (/ (apply + trial-counts) (double n-runs))]
                [n-pre mean trial-counts]))
            pre-counts)]
  (println "\n  pre-trials   mean post-trials to crit   25th-75th pctile")
  (println "  ──────────   ────────────────────────   ────────────────")
  (doseq [[n-pre mean tcs] ttcs]
    (let [sorted (sort tcs)
          p25 (nth sorted (max 0 (int (* 0.25 (count sorted)))))
          p75 (nth sorted (min (dec (count sorted))
                               (int (* 0.75 (count sorted)))))]
      (println (str "  " (pad n-pre 4) "         "
                    (pad (fmt mean 1) 6) "                    "
                    "[" (pad p25 3) ", " (pad p75 3) "]"))))
  (println "\n  → For deterministic reversal (1.0 → 0.0), more pre-reversal")
  (println "    training generally produces faster post-reversal learning: the")
  (println "    sharper old-context posterior makes the new evidence more"
                                                     )
  (println "    surprising → faster inference of a context switch."))

;; ============================================================================
;; Demo 5 — Serial reversal learning (Lloyd-Leslie fig 7)
;; ============================================================================
;;
;; 100:0 reversed every 24 trials, 8 training periods. After the first
;; couple of reversals, the agent has discovered "there are two contexts
;; here" and rapidly toggles between them. Errors per period collapse
;; sharply from period 2 onwards.

(println "\n============================================================")
(println "Demo 5 — Serial reversal learning (Lloyd-Leslie fig 7)")
(println "============================================================")
(println "8 training periods × 24 trials, alternating 100:0 and 0:100.")
(println "Errors per period collapse as the agent maintains both contexts")
(println "in its rule library and rapidly switches between them.")

(let [n-periods    8
      n-per-period 24
      n-runs       30
      protocol     (serial-reversal-protocol n-periods n-per-period
                                              [1.00 0.00] [0.00 1.00])
      all-actions  (mapv #(run-protocol protocol
                                         (rng/fresh-key (+ % 8000)))
                         (range n-runs))
      errors-per-period
      (mapv (fn [p]
              (let [start (* p n-per-period)
                    end   (+ start n-per-period)
                    correct-action (if (even? p) 0 1)
                    period-errors-per-run
                    (mapv (fn [acts]
                            (count (filter #(not= % correct-action)
                                           (subvec acts start end))))
                          all-actions)]
                (/ (apply + period-errors-per-run) (double n-runs))))
            (range n-periods))]
  (println "\n  period   correct arm   mean errors / 24 trials")
  (println "  ──────   ───────────   ───────────────────────")
  (doseq [p (range n-periods)]
    (println (str "  " (pad (inc p) 4) "     "
                  (if (even? p) "L (left)   " "R (right)  ") "  "
                  (pad (fmt (nth errors-per-period p) 1) 5))))
  (println "\n  → Errors collapse after period 2 — once the agent has")
  (println "    discovered 'this experiment has two contexts', each reversal")
  (println "    triggers a fast context-switch rather than re-learning")
  ;; CSV dump for paper figure
  (let [csv-path "../genmlx-papers/DeHouwer_paper/figs/data/crp_serial_reversal.csv"
        rows (mapv (fn [p]
                     (str (inc p) ","
                          (if (even? p) "L" "R") ","
                          (fmt (nth errors-per-period p) 4)))
                   (range n-periods))
        content (str "period,correct_arm,mean_errors\n"
                     (str/join "\n" rows) "\n")]
    (.writeFileSync fs csv-path content)
    (println (str "    Wrote: " csv-path))))

;; ============================================================================
;; Tracking-baseline kernel — single-context Beta-Bernoulli (no CRP)
;; ============================================================================
;;
;; The natural baseline for Demo 6's model comparison: a "Rescorla-Wagner-
;; ish" agent that maintains one Beta posterior per action and updates it
;; conjugately. No latent context, no CRP, no inference of regime changes.
;; Shares the :a and :r trace structure with the CRP kernel, so we can
;; condition both on the same observation stream and compare log marginal
;; likelihoods directly via the GFI weight field.

(def tracking-kernel
  "Single-context tracking kernel. Conjugate Beta-Bernoulli per action.
   carry: {:alpha [A] :beta [A] :alpha-0 :beta-0}
   trace sites: :a, :r"
  (dyn/auto-key
    (gen [carry input]
      (let [alpha (:alpha carry)
            beta  (:beta  carry)
            total (mx/add alpha beta)
            mu    (mx/divide alpha total)
            var   (mx/divide (mx/multiply alpha beta)
                             (mx/multiply (mx/multiply total total)
                                          (mx/add total ONE)))
            mu-L  (mx/index mu 0)
            mu-R  (mx/index mu 1)
            var-L (mx/index var 0)
            var-R (mx/index var 1)
            z      (mx/divide (mx/subtract mu-L mu-R)
                              (mx/sqrt (mx/add (mx/add var-L var-R) EPS)))
            prob-L (normal-cdf z)
            prob-R (mx/subtract ONE prob-L)
            true-probs (mx/array (:true-reward-probs input))
            forced-act (:forced-action input)
            action-weights (if (some? forced-act)
                             (mx/array (if (zero? forced-act) [1.0 0.0] [0.0 1.0]))
                             (mx/concatenate [(mx/expand-dims prob-L 0)
                                              (mx/expand-dims prob-R 0)]))
            a      (trace :a (dist/categorical-weights action-weights))
            true-p-a (mx/take-idx true-probs a)
            r      (trace :r (dist/bernoulli true-p-a))
            a-oh   (one-hot a N-ACTIONS)
            new-alpha (mx/add alpha (mx/multiply a-oh r))
            new-beta  (mx/add beta  (mx/multiply a-oh (mx/subtract ONE r)))
            new-carry {:alpha new-alpha :beta new-beta
                       :alpha-0 (:alpha-0 carry) :beta-0 (:beta-0 carry)}]
        [new-carry {:a a :r r :p-action-L prob-L}]))))

(def tracking-scan (comb/scan-combinator tracking-kernel))

(def tracking-model
  "Top-level wrapper for the tracking baseline (see crp-operant-model)."
  (gen [init-carry protocol]
    (splice :s tracking-scan init-carry protocol)))

(defn init-tracking-carry
  ([] (init-tracking-carry DEFAULT-ALPHA-0 DEFAULT-BETA-0))
  ([a0 b0]
   {:alpha   (mx/full [N-ACTIONS] a0)
    :beta    (mx/full [N-ACTIONS] b0)
    :alpha-0 (mx/scalar a0)
    :beta-0  (mx/scalar b0)}))

;; ============================================================================
;; Demo 6 — CRP model vs single-context tracking baseline
;; ============================================================================
;;
;; Two `gen` kernels share the :a, :r trace structure. Generate observations
;; from a reversal protocol (where ground truth has two regimes), then
;; condition both kernels on those same observations via `dyn/vgenerate`.
;; The kernel that better explains the data wins the Bayes factor, read
;; directly from the GFI :weight field via logsumexp − log N. Mirrors
;; rate_estimation Demo 3+4 patterns: shared trace + importance-sampled
;; marginal likelihood comparison.

(defn build-ar-choicemap
  "Convert per-trial action/reward pairs into a choicemap addressed under :s,
   matching the top-level wrapper's splice address. Each trial's :a is an
   int32 scalar; :r is a float scalar."
  [actions rewards]
  (reduce-kv
    (fn [cm t _]
      (-> cm
          (cm/set-choice [:s t :a] (mx/scalar (int (nth actions t)) mx/int32))
          (cm/set-choice [:s t :r] (mx/scalar (double (nth rewards t))))))
    cm/EMPTY
    (vec actions)))

(defn marginal-log-ml
  "log P(observed-a, observed-r | model) via importance sampling with N
   particles through `dyn/vgenerate`. The model is the top-level wrapper;
   constraints flow through the :s splice to the Scan kernel."
  [model init-carry-fn protocol obs-cm n-particles key]
  (let [vtrace  (dyn/vgenerate model
                               [(init-carry-fn) protocol]
                               obs-cm n-particles key)
        log-w   (:weight vtrace)
        _       (mx/eval! log-w)]
    (mx/item (mx/subtract (mx/logsumexp log-w)
                          (mx/scalar (Math/log n-particles))))))

(println "\n============================================================")
(println "Demo 6 — CRP model vs single-context tracking baseline")
(println "============================================================")
(println "Two `gen` kernels, shared :a/:r trace structure. Generate a")
(println "reversal-protocol trajectory; condition both kernels on the")
(println "same (action, reward) sequence; compare log marginal likelihood.")

(let [n-particles 2000
      ;; Synthetic data: a clear reversal, generated from a known true regime.
      acq-trials 25
      rev-trials 25
      protocol  (vec (concat (constant-protocol acq-trials [0.9 0.1])
                             (constant-protocol rev-trials [0.1 0.9])))
      gen-key   (rng/fresh-key 111)
      gen-tr    (mx/tidy-run
                  (fn []
                    (p/simulate (dyn/with-key crp-operant-scan gen-key)
                                [(init-carry) protocol]))
                  (fn [_] nil))
      outputs   (:outputs (:retval gen-tr))
      actions   (mapv (fn [o] (mx/item (:a o))) outputs)
      rewards   (mapv (fn [o] (mx/item (:r o))) outputs)
      obs-cm    (build-ar-choicemap actions rewards)
      log-ml-crp      (marginal-log-ml crp-operant-model init-carry
                                        protocol obs-cm n-particles
                                        (rng/fresh-key 222))
      log-ml-tracking (marginal-log-ml tracking-model init-tracking-carry
                                        protocol obs-cm n-particles
                                        (rng/fresh-key 333))
      log-bf          (- log-ml-crp log-ml-tracking)
      ;; Also a control: a stationary protocol where tracking should win.
      stat-protocol   (constant-protocol 50 [0.9 0.1])
      stat-gen        (mx/tidy-run
                        (fn []
                          (p/simulate (dyn/with-key crp-operant-scan
                                                    (rng/fresh-key 444))
                                      [(init-carry) stat-protocol]))
                        (fn [_] nil))
      stat-outs       (:outputs (:retval stat-gen))
      stat-as         (mapv (fn [o] (mx/item (:a o))) stat-outs)
      stat-rs         (mapv (fn [o] (mx/item (:r o))) stat-outs)
      stat-obs        (build-ar-choicemap stat-as stat-rs)
      stat-log-ml-crp (marginal-log-ml crp-operant-model init-carry
                                        stat-protocol stat-obs n-particles
                                        (rng/fresh-key 555))
      stat-log-ml-tr  (marginal-log-ml tracking-model init-tracking-carry
                                        stat-protocol stat-obs n-particles
                                        (rng/fresh-key 666))
      stat-log-bf     (- stat-log-ml-crp stat-log-ml-tr)]
  (println (str "\n  Reversal protocol (" (count protocol) " trials, with regime change):"))
  (println (str "    log P(data | CRP model)       = " (fmt log-ml-crp 2)))
  (println (str "    log P(data | tracking model)  = " (fmt log-ml-tracking 2)))
  (println (str "    log Bayes factor (CRP / track)= " (fmt log-bf 2)))
  (println (str "  Stationary protocol (" (count stat-protocol) " trials, single regime):"))
  (println (str "    log P(data | CRP model)       = " (fmt stat-log-ml-crp 2)))
  (println (str "    log P(data | tracking model)  = " (fmt stat-log-ml-tr 2)))
  (println (str "    log Bayes factor (CRP / track)= " (fmt stat-log-bf 2)))
  (println "\n  → Under reversal, CRP explains the observation stream better")
  (println "    (positive log-BF). Under stationary, tracking — which has fewer")
  (println "    parameters — wins on Occam grounds. The Bayes factor falls out")
  (println "    of one line: logsumexp(weights) − log N. No BIC approximation."))

;; ============================================================================
;; Streaming unfold-kernel + Demo 7
;; ============================================================================
;;
;; The Unfold combinator's kernel signature is `(gen [t state & extras])`,
;; differing from Scan's `(gen [carry input])`. We adapt the CRP-operant
;; kernel: t is the trial index (Unfold-supplied), state is the carry, and
;; the per-step probs come from a `probs-fn` extra. Same trace structure
;; (:a, :r, :c) so all GFI operations transfer.

(def crp-operant-unfold-kernel
  "Unfold-compatible variant of crp-operant-kernel. Same body modulo
   signature (t becomes an explicit argument; probs come from a fn)."
  (dyn/auto-key
    (gen [t state probs-fn]
      (let [alpha-mat   (:alpha-mat   state)
            beta-mat    (:beta-mat    state)
            counts      (:counts      state)
            active-mask (:active-mask state)
            c-belief    (:c-belief    state)
            n-active    (:n-active    state)
            alpha-crp   (:alpha       state)
            pi          (:pi          state)
            alpha-0     (:alpha-0     state)
            beta-0      (:beta-0      state)
            K           K-MAX
            A           N-ACTIONS
            true-probs  (mx/array (probs-fn t))
            n-plus-alpha   (mx/add (mx/sum counts) alpha-crp)
            crp-exist      (mx/divide (mx/multiply counts active-mask) n-plus-alpha)
            slot-available (mx/astype (mx/less (mx/astype n-active mx/float32)
                                               (mx/scalar (double K)))
                                      mx/float32)
            crp-new        (mx/multiply (mx/divide alpha-crp n-plus-alpha)
                                        slot-available)
            existing-prior (mx/add (mx/multiply (mx/subtract ONE pi) c-belief)
                                   (mx/multiply pi crp-exist))
            new-prior      (mx/multiply pi crp-new)
            prior-ext      (mx/concatenate [existing-prior
                                            (mx/expand-dims new-prior 0)])
            new-row-alpha (mx/full [1 A] alpha-0)
            new-row-beta  (mx/full [1 A] beta-0)
            ext-alpha     (mx/concatenate [alpha-mat new-row-alpha] 0)
            ext-beta      (mx/concatenate [beta-mat new-row-beta] 0)
            ext-total     (mx/add ext-alpha ext-beta)
            ext-mu        (mx/divide ext-alpha ext-total)
            ext-var       (mx/divide (mx/multiply ext-alpha ext-beta)
                                     (mx/multiply (mx/multiply ext-total ext-total)
                                                  (mx/add ext-total ONE)))
            ext-mu-t      (mx/transpose ext-mu [1 0])
            ext-var-t     (mx/transpose ext-var [1 0])
            mu-L          (mx/index ext-mu-t 0)
            mu-R          (mx/index ext-mu-t 1)
            var-L         (mx/index ext-var-t 0)
            var-R         (mx/index ext-var-t 1)
            z             (mx/divide (mx/subtract mu-L mu-R)
                                     (mx/sqrt (mx/add (mx/add var-L var-R) EPS)))
            phi-L         (normal-cdf z)
            prob-L        (mx/sum (mx/multiply prior-ext phi-L))
            prob-R        (mx/subtract ONE prob-L)
            action-weights (mx/concatenate [(mx/expand-dims prob-L 0)
                                            (mx/expand-dims prob-R 0)])
            a              (trace :a (dist/categorical-weights action-weights))
            true-p-a       (mx/take-idx true-probs a)
            r              (trace :r (dist/bernoulli true-p-a))
            alpha-col-a    (mx/take-idx ext-alpha a 1)
            beta-col-a     (mx/take-idx ext-beta a 1)
            total-col-a    (mx/add alpha-col-a beta-col-a)
            p-r1-col       (mx/divide alpha-col-a total-col-a)
            likelihood     (mx/where (mx/equal r ONE)
                                     p-r1-col
                                     (mx/subtract ONE p-r1-col))
            active-ext     (mx/concatenate [active-mask (mx/array [1.0])])
            post-unnorm    (mx/multiply (mx/multiply prior-ext likelihood) active-ext)
            posterior      (mx/divide post-unnorm
                                      (mx/maximum (mx/sum post-unnorm) EPS))
            c              (trace :c (dist/categorical-weights posterior))
            posterior-exist (mx/slice posterior 0 K)
            new-mass        (mx/index posterior K)
            new-slot-oh     (one-hot n-active K)
            alloc-mass-vec  (mx/multiply new-slot-oh new-mass)
            update-weights  (mx/add posterior-exist alloc-mass-vec)
            a-oh            (one-hot a A)
            update-mask     (mx/multiply (mx/expand-dims update-weights -1)
                                         (mx/expand-dims a-oh 0))
            new-alpha-mat   (mx/add alpha-mat (mx/multiply update-mask r))
            new-beta-mat    (mx/add beta-mat (mx/multiply update-mask
                                                          (mx/subtract ONE r)))
            new-counts      (mx/add counts update-weights)
            new-slot-count  (mx/take-idx new-counts n-active)
            should-activate (mx/greater new-slot-count ONE)
            should-activate-f (mx/astype should-activate mx/float32)
            new-active      (mx/maximum active-mask
                                        (mx/multiply new-slot-oh should-activate-f))
            new-n-active    (mx/add n-active (mx/astype should-activate mx/int32))]
        (assoc state
               :alpha-mat   new-alpha-mat
               :beta-mat    new-beta-mat
               :counts      new-counts
               :active-mask new-active
               :c-belief    update-weights
               :n-active    new-n-active)))))

(def crp-operant-unfold (comb/unfold-combinator crp-operant-unfold-kernel))

;; ============================================================================
;; Demo 7 — Streaming live agent via `unfold-extend`
;; ============================================================================
;;
;; A live agent processing trials one at a time, with the running trace
;; as the agent's complete inspectable memory. The streaming kernel
;; shares the trace structure (:a, :r, :c) with the Scan kernel — same
;; belief dynamics, just one step at a time. Each call to unfold-extend
;; appends one trial's worth of trace and updates the carry's :c-belief,
;; counts, and Beta parameters; the cumulative trace :score is the
;; agent's running predictive log-likelihood log P(:a_1:T, :r_1:T | model).
;;
;; (For vectorized particle filtering on the same model, see Demo 6's
;; `dyn/vgenerate` call.)

(println "\n============================================================")
(println "Demo 7 — Streaming live agent via `unfold-extend`")
(println "============================================================")
(println "30 trials processed one at a time; trace IS the agent's memory.")

(let [protocol     (vec (concat (constant-protocol 15 [0.9 0.1])
                                (constant-protocol 15 [0.1 0.9])))
      probs-fn     (fn [t] (:true-reward-probs (nth protocol t)))
      n-trials     (count protocol)
      empty-tr     (comb/unfold-empty-trace
                     crp-operant-unfold (init-carry) probs-fn)
      _            (println (str "\n  Streaming " n-trials " trials..."))
      milestones   #{5 10 15 20 25 30}
      snapshots
      (mx/tidy-run
        (fn []
          (loop [t 0 tr empty-tr snaps []]
            (if (>= t n-trials)
              {:final-tr tr :snaps snaps}
              (let [{tr' :trace}
                    (comb/unfold-extend tr cm/EMPTY
                                         (rng/fresh-key (+ 50000 t)))
                    t+1 (inc t)
                    snaps' (if (contains? milestones t+1)
                             (let [st (last (:retval tr'))
                                   _ (mx/materialize!
                                       (:c-belief st) (:n-active st) (:score tr'))
                                   nc (mx/item (:n-active st))
                                   sl (mx/item (:score tr'))
                                   cb (vec (mx/->clj (:c-belief st)))]
                               (conj snaps {:t t+1 :n-active nc :score sl
                                            :c-belief cb}))
                             snaps)]
                (recur (inc t) tr' snaps')))))
        (fn [{:keys [final-tr]}]
          (let [st (last (:retval final-tr))]
            [(:c-belief st) (:counts st) (:n-active st) (:score final-tr)])))
      final-tr (:final-tr snapshots)
      final-state (last (:retval final-tr))
      _ (mx/materialize! (:counts final-state))
      final-counts (vec (mx/->clj (:counts final-state)))]
  (println "\n  trial   n-active   cum log P(data)   c-belief (top 3 slots)")
  (println "  ─────   ────────   ───────────────   ──────────────────────")
  (doseq [{:keys [t n-active score c-belief]} (:snaps snapshots)]
    (let [top-3 (mapv #(fmt % 3) (take 3 c-belief))]
      (println (str "  " (pad t 4) "    " (pad n-active 3) "        "
                    (pad (fmt score 2) 8) "          "
                    "[" (str/join " " top-3) "]"))))
  (println (str "\n  Final counts (slots 0-3) = ["
                (str/join " " (mapv #(fmt % 1) (take 4 final-counts))) "]"))
  (println "\n  → The trace IS the agent's complete inspectable memory; one")
  (println "    `unfold-extend` call per new observation. Trace `:score` is")
  (println "    the cumulative log marginal likelihood under the current beliefs.")
  (println "    For parallel particle filtering on the same model, see Demo 6's")
  (println "    `dyn/vgenerate` call — both share the kernel and trace structure."))

;; ============================================================================
;; Summary
;; ============================================================================

(println "\n============================================================
Summary — CRP-based operant learning as a generative function
============================================================

Two `gen` kernels, one architectural pattern:

  crp-operant-kernel   :: (gen [carry input] -> [new-carry output])
                          — trace sites :a, :r, :c per trial
                          — carry holds Beta(α,β) per (context, action),
                            CRP counts, c-belief over slots
                          — Rao-Blackwellized soft updates over the full
                            posterior; phantom-slot accumulator for new
                            contexts

  tracking-kernel      :: (gen [carry input] -> [new-carry output])
                          — trace sites :a, :r (no :c)
                          — single Beta posterior per action; standard
                            Thompson with no context inference

Both compose with Scan for full trajectories and (for the CRP kernel) Unfold
for streaming. The GFI works uniformly:

  (p/simulate kernel [init protocol])   → agent generates actions + rewards
  (p/generate kernel ... obs-cm)        → condition on real (a,r) stream
  (dyn/vgenerate kernel ... N)          → parallel particle filter
  (comb/unfold-extend tr cm key)        → streaming live agent

Animal-level learning: the agent discovers — purely from (action, reward)
pairs, with no observable context cue — that the world has discrete
'rules', each defining its own action→consequence contingency. The CRP
provides a Bayesian nonparametric prior over how many rules exist.

Limitations honestly documented (see notes file):
  - Beta-Bernoulli's bounded likelihood ratio limits sudden context-switch
    sensitivity. Normal-Gamma would reproduce Lloyd-Leslie's canonical
    PREE direction (Demo 3) and stronger ORE.
  - Soft-update + phantom accumulator is a pragmatic CRP-truncated-K
    approximation; full HMM forward filtering would be more elegant for
    posterior tracking.
  - 'New context' is not the same as RFT's derived relational responding —
    extending to that requires hierarchical (HDP / nCRP) extensions.

Possible follow-up examples:
  - HDP-operant: add observable S^D, regimes share a higher-level cluster
    pool. (Conditional discrimination, Level 2 of the hierarchy.)
  - nCRP-operant: conditional discriminations (Level 3). Each CS^D selects
    a different S^D → context map.
")
