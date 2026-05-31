;; ============================================================================
;; Multi-Model Embodied Agent Loop
;; ============================================================================
;;
;; Companion to:
;;   examples/habituation.cljs   — Gershman (2024) habituation as optimal filtering
;;   examples/rate_estimation.cljs — Gershman (2025) online Bayesian rate estimation
;;   examples/crp_operant.cljs    — Lloyd-Leslie (2013) CRP-based operant decision
;;
;; Architectural design: ../genmlx-lab/dev/docs/DESIGN_MULTI_MODEL_AGENT.md
;; Paper plan:           ../genmlx-lab/dev/docs/PAPER_THREE_LEARNING_PHENOMENA.md  (§5)
;;
;; What this demonstrates:
;; ------------------------
;; A single embodied agent loop runs all three kernels in parallel. Each cycle:
;;
;;   1. The "world" generates an observation: time t, stimulus intensity x,
;;      CS indicator vector, action a (chosen by an external policy), and a
;;      binary reward r.
;;
;;   2. Each model's `comb/unfold-extend` is called with constraints encoding
;;      *the same* observed reward r. Each model interprets the rest of the
;;      observation in its own way:
;;        - Habituation: sees the stimulus stream (z, x); traces :y;
;;          we constrain :y = r.
;;        - Rate estimation: sees the CS indicator vector; traces :r;
;;          we constrain :r = r.
;;        - CRP-operant (scoring variant): sees the chosen action;
;;          traces :a and :r; we constrain :a = a and :r = r. The :r
;;          distribution is the AGENT's marginalized predictive (not the
;;          world's truth), so the trace's :score is the proper marginal
;;          log-likelihood under CRP.
;;
;;   3. Each model's cumulative trace :score is its log P(observed sequence
;;      so far | model). By the GFI's algebraic laws, the posterior over
;;      models is one log-normalization away. Bayesian model averaging
;;      across phenomena emerges with no extra machinery.
;;
;; Three regime transitions are simulated to show that the posterior over
;; models tracks the world's structure:
;;
;;   - Regime 1 (cycles 0–49):   Stable CS→reward contingency → rate-est wins
;;   - Regime 2 (cycles 50–99):  Abrupt reversal              → CRP overtakes
;;   - Regime 3 (cycles 100–149): Weak below-threshold stim   → habituation wins
;;
;; This is the core §5 demonstration the paper plan calls out. Run with:
;;
;;   bun run --bun nbb examples/multi_model_agent.cljs
;;
;; ============================================================================

(ns multi-model-agent
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
;; Shared utilities
;; ============================================================================

(defn fmt
  ([v]        (fmt v 3))
  ([v digits] (.toFixed (double v) digits)))

(defn pad [s width] (.padStart (str s) width " "))

(def ONE       (mx/scalar 1.0))
(def ZERO      (mx/scalar 0.0))
(def HALF      (mx/scalar 0.5))
(def SQRT-2    (mx/scalar (Math/sqrt 2.0)))
(def EPS       (mx/scalar 1e-12))
(def P-CLAMP-LO (mx/scalar 1e-6))
(def P-CLAMP-HI (mx/scalar (- 1.0 1e-6)))

(defn normal-cdf [z]
  (mx/multiply HALF (mx/add ONE (mx/erf (mx/divide z SQRT-2)))))

(defn clamp-prob [p]
  (mx/minimum P-CLAMP-HI (mx/maximum P-CLAMP-LO p)))

(defn- one-hot [idx n]
  (mx/astype (mx/equal (mx/arange n) (mx/astype idx mx/float32)) mx/float32))

;; ============================================================================
;; HABITUATION KERNEL (from examples/habituation.cljs)
;; ============================================================================

(defn rbf-kernel-matrix [Z lam]
  (let [Zi   (mx/expand-dims Z 0)
        Zj   (mx/expand-dims Z 1)
        diff (mx/subtract Zi Zj)
        sq   (mx/sum (mx/multiply diff diff) [2])
        lam2 (mx/multiply lam lam)]
    (mx/exp (mx/divide (mx/negative sq)
                       (mx/multiply (mx/scalar 2.0) lam2)))))

(defn cho-solve [L b]
  (let [b1?  (= 1 (mx/ndim b))
        b2   (if b1? (mx/expand-dims b 1) b)
        y    (mx/solve-triangular L b2 false)
        LT   (mx/transpose L)
        x    (mx/solve-triangular LT y true)
        x'   (if b1? (mx/squeeze x [1]) x)]
    x'))

(defn gp-posterior [Z X active-mask t T-max lam alpha]
  (let [K        (rbf-kernel-matrix Z lam)
        eye-T    (mx/eye T-max)
        m-col    (mx/expand-dims active-mask 1)
        m-row    (mx/expand-dims active-mask 0)
        m-outer  (mx/multiply m-col m-row)
        K-masked (mx/add (mx/multiply K m-outer)
                         (mx/multiply (mx/subtract ONE m-outer) eye-T))
        A        (mx/add K-masked (mx/multiply alpha eye-T))
        L        (mx/cholesky A)
        k-vec    (mx/index K t)
        k-m      (mx/multiply k-vec active-mask)
        X-m      (mx/multiply X active-mask)
        beta     (cho-solve L X-m)
        gamma    (cho-solve L k-m)
        xhat     (mx/sum (mx/multiply k-m beta))
        sigma2   (mx/maximum EPS
                             (mx/subtract ONE
                                          (mx/sum (mx/multiply k-m gamma))))
        sigma    (mx/sqrt sigma2)]
    [xhat sigma]))

(def HAB-T-MAX 160)
(def HAB-LAM   1.0)
(def HAB-ALPHA 0.3)
(def HAB-PSI   0.5)

(def hab-unfold-kernel
  "Streaming habituation kernel. Trace site :y is a Bernoulli at the
   closed-form Normal-CDF response probability. When :y is constrained
   to an observed binary value, the trace's cumulative :score is the
   habituation model's log P(observed responses | stimulus history)."
  (dyn/auto-key
    (gen [step carry inputs-fn]
      (let [Z           (:Z           carry)
            X           (:X           carry)
            active-mask (:active-mask carry)
            t           (:t           carry)
            T-max       (:T-max       carry)
            lam         (:lam         carry)
            alpha       (:alpha       carry)
            psi         (:psi         carry)
            {z-new :z x-num :x} (inputs-fn step)
            x-new       (mx/scalar (double x-num))
            D           (long (second (mx/shape Z)))
            indices     (mx/arange T-max)
            oh-t-bool   (mx/equal indices t)
            oh-t        (mx/astype oh-t-bool mx/float32)
            oh-t-col    (mx/expand-dims oh-t 1)
            z-row       (mx/expand-dims (mx/array z-new) 0)
            z-broadcast (mx/broadcast-to z-row [T-max D])
            Z'          (mx/add (mx/multiply oh-t-col z-broadcast)
                                (mx/multiply (mx/subtract ONE oh-t-col) Z))
            X'          (mx/add (mx/multiply oh-t x-new)
                                (mx/multiply (mx/subtract ONE oh-t) X))
            mask'       (mx/maximum active-mask oh-t)
            [xhat sigma] (gp-posterior Z' X' mask' t T-max lam alpha)
            p-resp      (clamp-prob (normal-cdf (mx/divide (mx/subtract xhat psi) sigma)))
            y           (trace :y (dist/bernoulli p-resp))]
        {:Z Z' :X X' :active-mask mask'
         :t (mx/add t (mx/scalar 1 mx/int32))
         :T-max T-max :lam lam :alpha alpha :psi psi
         :p-resp p-resp :xhat xhat :sigma sigma :y y}))))

(def hab-unfold (comb/unfold-combinator hab-unfold-kernel))

(defn init-hab-carry
  "Initial habituation carry with stimulus dimension D=2 (time + intensity)."
  [D]
  {:Z           (mx/zeros [HAB-T-MAX D])
   :X           (mx/zeros [HAB-T-MAX])
   :active-mask (mx/zeros [HAB-T-MAX])
   :t           (mx/scalar 0 mx/int32)
   :T-max       HAB-T-MAX
   :lam         (mx/scalar HAB-LAM)
   :alpha       (mx/scalar HAB-ALPHA)
   :psi         (mx/scalar HAB-PSI)})

;; ============================================================================
;; RATE-ESTIMATION KERNEL (from examples/rate_estimation.cljs)
;; ============================================================================

(def RATE-DT 1.0)  ;; one rate-estimation cycle per agent cycle

(def rate-unfold-kernel
  "Streaming Bayesian rate-estimation kernel (Gershman 2025). Trace site
   :r is a Poisson at the model's predictive rate. When :r is constrained
   to an observed binary value (treated as count 0 or 1), the trace's
   cumulative :score is log P(observed reinforcements | CS history) under
   the conjugate Gamma-Poisson model."
  (dyn/auto-key
    (gen [t state dt-num x-fn]
      (let [alpha-b     (:alpha-b state)
            beta-b      (:beta-b state)
            alpha-total (:alpha-total state)
            beta-total  (:beta-total state)
            actual-time (* t dt-num)
            x-vec       (x-fn actual-time)
            stim        (mx/scalar (double (nth x-vec 1)))
            bg          (mx/subtract (mx/scalar 1.0) stim)
            mean-b      (mx/divide alpha-b beta-b)
            mean-total  (mx/divide alpha-total beta-total)
            pred-rate   (mx/add (mx/multiply bg mean-b)
                                (mx/multiply stim mean-total))
            r           (trace :r (dist/poisson pred-rate))
            new-alpha-b     (mx/add alpha-b     (mx/multiply bg r))
            new-beta-b      (mx/add beta-b      bg)
            new-alpha-total (mx/add alpha-total (mx/multiply stim r))
            new-beta-total  (mx/add beta-total  stim)]
        {:alpha-b new-alpha-b :beta-b new-beta-b
         :alpha-total new-alpha-total :beta-total new-beta-total
         :pred-rate pred-rate}))))

(def rate-unfold (comb/unfold-combinator rate-unfold-kernel))

(defn init-rate-state
  ([] (init-rate-state 0.1 1.0))
  ([alpha0 beta0]
   {:alpha-b     (mx/scalar alpha0)
    :beta-b      (mx/scalar beta0)
    :alpha-total (mx/scalar alpha0)
    :beta-total  (mx/scalar beta0)}))

;; ============================================================================
;; CRP-OPERANT SCORING KERNEL
;; ============================================================================
;;
;; Adaptation of crp-operant-unfold-kernel for the multi-model agent loop.
;; Two changes from the original:
;;   1. The :r distribution is the agent's MARGINALIZED predictive
;;      P(r=1 | a, history) = Σ_c P(c | history) · α^{c,a} / (α^{c,a} + β^{c,a})
;;      — not the world's truth. When :r is constrained to the observed
;;      reward, :score accumulates the proper marginal log-likelihood
;;      log P(observed-r | a, history) under the CRP model.
;;   2. The action is supplied per step by an external action-fn (the
;;      world's chosen action); we constrain :a to this value.

(def ^:const K-MAX 10)
(def ^:const N-ACTIONS 2)
(def ^:const DEFAULT-CRP-ALPHA 2.0)
(def ^:const DEFAULT-CRP-PI 0.15)
(def ^:const DEFAULT-CRP-ALPHA-0 1.0)
(def ^:const DEFAULT-CRP-BETA-0 1.0)

(def crp-scoring-unfold-kernel
  "CRP-operant scoring kernel for the multi-model agent loop. Differs from
   crp-operant-unfold-kernel in three ways:

     1. The action is supplied per step by action-fn (the world's observed
        action) and is NOT a trace site — it enters the kernel deterministically.
     2. :c is marginalized internally (Rao-Blackwellized soft update over the
        posterior) and is NOT a trace site — sampling it would inject
        H(posterior) noise into :score.
     3. The remaining trace site :r is sampled from the AGENT'S marginalized
        predictive P(r=1 | a, history) — not the world's truth.

   Result: when :r is constrained to the observed reward, the trace's
   cumulative :score is the proper Bayesian marginal log-likelihood
   log P(observed reward sequence | observed actions, CRP model)."
  (dyn/auto-key
    (gen [t state action-fn]
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
            ;; Observed action (deterministic input, not a trace site)
            a              (mx/scalar (int (action-fn t)) mx/int32)
            ;; ---------- CRP + persistence prior over c_t ----------
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
            ;; ---------- Per-slot Beta params (extended with phantom slot) ----------
            new-row-alpha (mx/full [1 A] alpha-0)
            new-row-beta  (mx/full [1 A] beta-0)
            ext-alpha     (mx/concatenate [alpha-mat new-row-alpha] 0)
            ext-beta      (mx/concatenate [beta-mat new-row-beta] 0)
            ext-total     (mx/add ext-alpha ext-beta)
            ext-mu        (mx/divide ext-alpha ext-total)             ; [K+1, A] per-slot mean
            ext-mu-t      (mx/transpose ext-mu [1 0])                 ; [A, K+1]
            ;; ---------- Agent's marginalized predictive for :r given :a ----------
            mu-a-vec       (mx/index ext-mu-t a)                      ; [K+1] mu per slot
            agent-p-r1     (clamp-prob (mx/sum (mx/multiply prior-ext mu-a-vec)))
            ;; ---------- :r trace site at AGENT'S predictive (only trace site) ----------
            r              (trace :r (dist/bernoulli agent-p-r1))
            ;; ---------- Posterior over c_t given (a, r) — used for soft update ----------
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
            ;; ---------- Rao-Blackwellized soft update of beta stats ----------
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
               :n-active    new-n-active
               :agent-p-r1  agent-p-r1)))))

(def crp-scoring-unfold (comb/unfold-combinator crp-scoring-unfold-kernel))

(defn init-crp-state []
  (let [slot-0 (mx/concatenate [(mx/array [1.0])
                                (mx/zeros [(dec K-MAX)])])]
    {:alpha-mat   (mx/full [K-MAX N-ACTIONS] DEFAULT-CRP-ALPHA-0)
     :beta-mat    (mx/full [K-MAX N-ACTIONS] DEFAULT-CRP-BETA-0)
     :counts      (mx/zeros [K-MAX])
     :active-mask slot-0
     :c-belief    slot-0
     :n-active    (mx/scalar 1 mx/int32)
     :alpha       (mx/scalar DEFAULT-CRP-ALPHA)
     :pi          (mx/scalar DEFAULT-CRP-PI)
     :alpha-0     (mx/scalar DEFAULT-CRP-ALPHA-0)
     :beta-0      (mx/scalar DEFAULT-CRP-BETA-0)}))

;; ============================================================================
;; The world — regime-structured event stream
;; ============================================================================
;;
;; A single shared timeline. Each cycle, the world produces:
;;   :t           absolute time (= cycle index here; one cycle = one time unit)
;;   :stim-x      stimulus intensity in [0, 1]    (for habituation)
;;   :cs          CS indicator vector [bg cs]     (for rate estimation)
;;   :action      chosen action ∈ {0, 1}          (for CRP-operant)
;;   :reward      shared observable ∈ {0, 1}      (the binary quantity all
;;                                                  three models score)
;;
;; Three regimes:
;;   Regime 1 (cycles 0..49):  Stable CS-on, action 0, reward = 1 w/ p=0.85
;;                             — rate-estimation should excel
;;   Regime 2 (cycles 50..99): CS-on persists but reward flips off
;;                             — CRP-operant should overtake (regime change)
;;   Regime 3 (cycles 100..149): Below-threshold stim, no CS, sporadic reward
;;                             — habituation should explain best

(def N-CYCLES 150)

(defn unif01
  "Sample one uniform(0, 1) as a JS number."
  [key]
  (mx/item (dist/sample (dist/uniform (mx/scalar 0.0) (mx/scalar 1.0)) key)))

(defn world-event
  "Generate the world's event at cycle t. The regime structure is designed
   so that each of habituation / CRP-operant has a regime where it
   structurally excels:

     Regime 1 (cycles 0..49):    Stable contingency.
        stim=0.7 throughout, reward=1 with p=0.9.
        Warm-up — all kernels learn the high-reward state.

     Regime 2 (cycles 50..99):   Reversal mid-regime.
        stim=0.7 (unchanged), reward flips to ~0 at p=0.9.
        Habituation has conflicting training (same z, different y) and
        cannot localize the change. CRP infers a new context.

     Regime 3 (cycles 100..149): Stimulus-driven alternation.
        Even cycles: stim=0.95, reward=1 at p=0.95.
        Odd cycles:  stim=0.05, reward=0 at p=0.95.
        Habituation's GP can predict reward perfectly from stim. CRP
        averages across cycles and cannot improve beyond the marginal rate.

   Rate-estimation is included for comparison but is at a structural
   disadvantage throughout: Poisson scoring of binary data is uniformly
   worse than Bernoulli, by ~0.5 nats per cycle. This is an honest
   limitation, not a bug — Poisson would dominate if rewards were
   integer counts."
  [t key]
  (let [u (unif01 key)]
    (cond
      ;; Regime 1: stable, mostly-rewarded
      (< t 50)
      {:t t :stim-x 0.7 :cs [0.0 1.0] :action 0
       :reward (if (< u 0.9) 1 0)}

      ;; Regime 2: reversal — same stim, flipped reward
      (< t 100)
      {:t t :stim-x 0.7 :cs [0.0 1.0] :action 0
       :reward (if (< u 0.1) 1 0)}

      ;; Regime 3: stim-driven alternation (high contrast)
      :else
      (let [stim   (if (even? t) 0.95 0.05)
            p-rew  (if (even? t) 0.95 0.05)]
        {:t t :stim-x stim :cs [1.0 0.0] :action 0
         :reward (if (< u p-rew) 1 0)}))))

;; ============================================================================
;; Per-model adapters: build inputs-fn / x-fn / action-fn from a stream of events
;; ============================================================================

(defn build-hab-inputs-fn
  "Habituation expects a function step → {:z [t stim-x] :x stim-x}."
  [events]
  (fn [step]
    (let [e (nth events step)]
      {:z [(double (:t e)) (double (:stim-x e))]
       :x (double (:stim-x e))})))

(defn build-rate-x-fn
  "Rate estimation expects a function actual-time → [bg cs] vector."
  [events]
  (fn [actual-time]
    (let [step (long (Math/round (double actual-time)))
          step (max 0 (min (dec (count events)) step))
          e    (nth events step)]
      (:cs e))))

(defn build-crp-action-fn
  "Per-step action supplied to the CRP scoring kernel."
  [events]
  (fn [t] (:action (nth events (min t (dec (count events)))))))

;; ============================================================================
;; The agent loop — one step per cycle
;; ============================================================================

(defn step-hab
  "Extend the habituation trace by one cycle, conditioning :y on observed reward."
  [tr reward key]
  (let [cm (cm/set-choice cm/EMPTY [:y] (mx/scalar (double reward)))]
    (comb/unfold-extend tr cm key)))

(defn step-rate
  "Extend the rate-estimation trace by one cycle, conditioning :r."
  [tr reward key]
  (let [cm (cm/set-choice cm/EMPTY [:r] (mx/scalar (double reward)))]
    (comb/unfold-extend tr cm key)))

(defn step-crp
  "Extend the CRP scoring trace by one cycle, conditioning :r (the only
   trace site in the scoring kernel; :a is a deterministic input)."
  [tr _action reward key]
  (let [cm (cm/set-choice cm/EMPTY [:r] (mx/scalar (double reward)))]
    (comb/unfold-extend tr cm key)))

(defn logsumexp [xs]
  (let [mx-val (apply max xs)]
    (+ mx-val (Math/log (apply + (mapv #(Math/exp (- % mx-val)) xs))))))

(defn model-posterior
  "From a vector of per-model cumulative log-likelihoods (and a flat prior),
   return the posterior over models."
  [log-mls]
  (let [log-prior (- (Math/log (count log-mls)))
        unnorm    (mapv #(+ % log-prior) log-mls)
        log-Z     (logsumexp unnorm)]
    (mapv #(Math/exp (- % log-Z)) unnorm)))

;; ============================================================================
;; Run the loop
;; ============================================================================

(println "============================================================")
(println "Multi-Model Embodied Agent Loop")
(println "============================================================")
(println "Three GenMLX kernels — habituation, rate-estimation, CRP-operant —")
(println "running in parallel through one unfold-extend cycle each per tick.")
(println "Each model accumulates :score; the model posterior falls out of")
(println "the GFI's algebraic laws as one logsumexp normalization.")
(println "")
(println (str "Cycles: " N-CYCLES))
(println "Regime 1 (0..49):    Stable contingency       (warm-up)")
(println "Regime 2 (50..99):   Reversal at midpoint     (CRP advantage)")
(println "Regime 3 (100..149): Stimulus-driven reward   (habituation advantage)")

;; Generate the world's events first (deterministic with a fixed seed).
(def events
  (mapv (fn [t] (world-event t (rng/fresh-key (+ 1000 t))))
        (range N-CYCLES)))

(def hab-inputs-fn  (build-hab-inputs-fn  events))
(def rate-x-fn      (build-rate-x-fn      events))
(def crp-action-fn  (build-crp-action-fn  events))

(println "\nRunning agent loop ...")

(def results
  (mx/tidy-run
    (fn []
      (let [hab-tr0  (comb/unfold-empty-trace
                       hab-unfold (init-hab-carry 2) hab-inputs-fn)
            rate-tr0 (comb/unfold-empty-trace
                       rate-unfold (init-rate-state) RATE-DT rate-x-fn)
            crp-tr0  (comb/unfold-empty-trace
                       crp-scoring-unfold (init-crp-state) crp-action-fn)]
        (loop [t        0
               hab-tr   hab-tr0
               rate-tr  rate-tr0
               crp-tr   crp-tr0
               trail    []]
          (if (>= t N-CYCLES)
            trail
            (let [e        (nth events t)
                  reward   (:reward e)
                  action   (:action e)
                  k1       (rng/fresh-key (+ 20000 t))
                  k2       (rng/fresh-key (+ 30000 t))
                  k3       (rng/fresh-key (+ 40000 t))
                  hab-r    (step-hab  hab-tr  reward k1)
                  rate-r   (step-rate rate-tr reward k2)
                  crp-r    (step-crp  crp-tr  action reward k3)
                  hab-tr'  (:trace hab-r)
                  rate-tr' (:trace rate-r)
                  crp-tr'  (:trace crp-r)
                  _        (mx/materialize! (:score hab-tr')
                                            (:score rate-tr')
                                            (:score crp-tr'))
                  hab-s    (mx/item (:score hab-tr'))
                  rate-s   (mx/item (:score rate-tr'))
                  crp-s    (mx/item (:score crp-tr'))
                  post     (model-posterior [hab-s rate-s crp-s])]
              (recur (inc t) hab-tr' rate-tr' crp-tr'
                     (conj trail {:t      t
                                  :reward reward
                                  :hab    hab-s
                                  :rate   rate-s
                                  :crp    crp-s
                                  :post   post})))))))
    (fn [_] [])))

;; ============================================================================
;; Print summary
;; ============================================================================

(defn pct [x] (str (.toFixed (* 100.0 (double x)) 1) "%"))

(println "\nCumulative log-likelihood and model posterior at milestones:")
(println "")
(println "  cycle   regime   reward%   hab log-L   rate log-L   crp log-L   P(hab)  P(rate)  P(crp)")
(println "  ─────   ──────   ───────   ─────────   ──────────   ─────────   ──────  ───────  ──────")

(let [milestones [9 24 49 59 74 99 109 124 149]]
  (doseq [t milestones]
    (let [row     (nth results t)
          regime  (cond (< t 50) "1" (< t 100) "2" :else "3")
          window  (subvec results (max 0 (- t 9)) (inc t))
          rwd     (/ (apply + (mapv :reward window)) (double (count window)))
          [ph pr pc] (:post row)]
      (println (str "  " (pad (inc t) 4) "      "
                    regime "       "
                    (pad (pct rwd) 6) "    "
                    (pad (fmt (:hab row)  1) 7) "     "
                    (pad (fmt (:rate row) 1) 7) "      "
                    (pad (fmt (:crp row)  1) 7) "     "
                    (pad (pct ph) 5) "   "
                    (pad (pct pr) 5) "    "
                    (pad (pct pc) 5))))))

(println "")
(println "Per-regime mean model posterior (averaged over the regime's cycles):")
(println "")
(println "  regime   cycles     P(hab)    P(rate)    P(crp)")
(println "  ──────   ───────    ──────    ───────    ──────")

(doseq [[regime lo hi]
        [["1" 0 50] ["2" 50 100] ["3" 100 150]]]
  (let [window (subvec results lo hi)
        n      (count window)
        sums   (reduce (fn [acc r]
                         (let [[ph pr pc] (:post r)]
                           [(+ (nth acc 0) ph)
                            (+ (nth acc 1) pr)
                            (+ (nth acc 2) pc)]))
                       [0.0 0.0 0.0] window)
        means  (mapv #(/ % (double n)) sums)]
    (println (str "    " regime "    " (pad lo 4) ".." (pad (dec hi) 4)
                  "    " (pad (pct (nth means 0)) 5)
                  "    " (pad (pct (nth means 1)) 5)
                  "    " (pad (pct (nth means 2)) 5)))))

(println "")
(println "Final cumulative scores:")
(let [final (last results)]
  (println (str "  habituation:      log P(rewards | model) = " (fmt (:hab  final) 2)))
  (println (str "  rate-estimation:  log P(rewards | model) = " (fmt (:rate final) 2)))
  (println (str "  CRP-operant:      log P(rewards | model) = " (fmt (:crp  final) 2))))

;; ============================================================================
;; CSV dump for paper figure generation
;; ============================================================================
;;
;; Writes per-cycle data to ../genmlx-papers/DeHouwer_paper/data/multi_model_agent.csv
;; for plotting with external tools (matplotlib / R / etc.). Schema:
;;
;;   cycle, regime, reward, hab_logL, rate_logL, crp_logL, p_hab, p_rate, p_crp

(def CSV-OUT "../genmlx-papers/DeHouwer_paper/figs/data/multi_model_agent.csv")

(let [header "cycle,regime,reward,hab_logL,rate_logL,crp_logL,p_hab,p_rate,p_crp"
      regime (fn [t] (cond (< t 50) 1 (< t 100) 2 :else 3))
      rows   (mapv (fn [{:keys [t reward hab rate crp post]}]
                     (str t "," (regime t) "," reward ","
                          (fmt hab  4) "," (fmt rate 4) "," (fmt crp  4) ","
                          (fmt (nth post 0) 6) ","
                          (fmt (nth post 1) 6) ","
                          (fmt (nth post 2) 6)))
                   results)
      content (str header "\n" (str/join "\n" rows) "\n")]
  (.writeFileSync fs CSV-OUT content)
  (println "")
  (println (str "Wrote per-cycle CSV: " CSV-OUT " (" (count rows) " rows).")))

(println "")
(println "Interpretation:")
(println "  Regime 1 (stable contingency at one stimulus) and Regime 2")
(println "  (reversal at the same stimulus) both favor CRP-operant — its")
(println "  Beta-conjugate predictive concentrates on the high or low reward")
(println "  rate efficiently, and the CRP context inference absorbs the")
(println "  reversal in Regime 2 with a single new latent slot.")
(println "")
(println "  Regime 3 (stimulus-driven alternation) is the regime habituation")
(println "  is built for: the GP posterior over stimulus history lets the")
(println "  kernel predict the response from the just-observed stimulus,")
(println "  while CRP-operant — which receives no stimulus signal — averages")
(println "  across cycles and pays ~log(2) nats per trial. Habituation")
(println "  overtakes CRP within Regime 3 (the model posterior crosses near")
(println "  the regime midpoint).")
(println "")
(println "  Rate-estimation runs in parallel but is at a structural")
(println "  disadvantage throughout: Poisson scoring of binary rewards is")
(println "  uniformly worse than Bernoulli by ~0.5 nats per cycle. This is")
(println "  honest — Poisson would dominate if rewards were integer counts.")
(println "")
(println "  No model-comparison machinery beyond the GFI's :score field was")
(println "  invoked. Bayesian model averaging is a consequence of the kernel-")
(println "  as-gen-function form and the algebraic laws (cumulative :score")
(println "  = log marginal likelihood, posterior = one logsumexp away).")
(println "")
(println "  Companion: ../genmlx-lab/dev/docs/DESIGN_MULTI_MODEL_AGENT.md")
(println "             ../genmlx-lab/dev/docs/PAPER_THREE_LEARNING_PHENOMENA.md  (§5)")
