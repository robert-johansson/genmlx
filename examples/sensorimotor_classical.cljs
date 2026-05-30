;; Classical Conditioning via Hammer-style Sensorimotor Learning
;; ==============================================================
;;
;; Replicates the five classical phenomena from latent_cause_gpu.cljs using
;; the unified sensorimotor kernel from genmlx.sensorimotor:
;;
;;   1. Acquisition          — A1 → B1 implication accumulates positive evidence
;;   2. Extinction           — A1 → nothing accumulates negative evidence (β),
;;                              but α is preserved (no overwriting)
;;   3. Spontaneous recovery — after a delay, projection differentially decays
;;                              recent negative vs older positive evidence
;;   4. Gradual extinction   — small prediction errors drive proper Beta update
;;                              toward the empirical rate
;;   5. Conditional discrimination — different (context, stimulus) compounds
;;                                    have separate implications
;;
;; Mechanism: each trial constrains the kernel's :expected-consequent trace
;; site to the actual observed consequent. The particle weight equals
;; log P(observed | implication's success-rate). End-of-trial revision
;; updates the implication's Beta posterior conjugately. The CRP version's
;; "single cause" becomes "single Beta posterior on a single implication"
;; — same trajectory, simpler mechanism.
;;
;; Run: bun run --bun nbb examples/sensorimotor_classical.cljs

(ns sensorimotor-classical
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.mlx.random :as rng]
            [genmlx.sensorimotor :as sm]
            [genmlx.gen :refer-macros [gen]]))

;; -----------------------------------------------------------------------------
;; Default hyperparameters
;; -----------------------------------------------------------------------------

(def default-params
  {:decay-rate            0.95          ;; β_proj exponent for projection
   :anticip-neg-evidence  0.5           ;; β increment per anticipation miss
   :decision-threshold    0.51          ;; below this, motor-babble
   :min-temperature       0.1
   :n-particles           500})

;; -----------------------------------------------------------------------------
;; Trivial percept suite for classical: stimulus is the trial config's :stim
;; -----------------------------------------------------------------------------

(def stimulus-detector
  "A minimal percept gen function: returns [event-data pattern-key].
   For classical conditioning the structural slot exists; richer detectors
   replace this without kernel changes."
  (dyn/auto-key
    (gen [retina]
      [(:stim retina) (:stim retina)])))

;; -----------------------------------------------------------------------------
;; The classical run loop
;; -----------------------------------------------------------------------------

(defn run-classical
  "Drive the sensorimotor kernel through a sequence of trials.

   Each trial: stimulus is presented, the kernel predicts the consequent,
   the constraint binds the actual outcome, particle weights derive from
   p/generate, then end-of-trial revision updates each particle's Beta
   posteriors. Particles with poor predictions are resampled out.

   trials: vector of {:stim K  :consequent K  :time-skip int}.
           :time-skip lets us insert delays (for spontaneous recovery).

   Returns: {:final-memories [memory-per-particle]
             :ess-trace      [ESS-per-trial]
             :trace          [{:trial t :pkey K :obs K :mean-alpha α} ...]}."
  [trials params seed]
  (let [{:keys [decay-rate anticip-neg-evidence
                decision-threshold min-temperature
                n-particles]} params
        ;; Classical: no operations — each particle has the same single op :none
        op-keys [:none]
        ;; Initial: every particle has an empty ConceptMemory
        initial-memories (vec (repeat n-particles sm/empty-memory))
        key0 (rng/fresh-key seed)]
    (loop [t 0
           memories initial-memories
           key key0
           trace-log []]
      (if (>= t (count trials))
        {:final-memories memories
         :trace          trace-log}
        (let [trial      (nth trials t)
              ;; Time-skip simulates a long delay (advance "now" without trial)
              t-actual   (+ t (or (:time-skip trial) 0))
              stim       (:stim trial)
              obs        (:consequent trial)
              ;; Project all particle memories to t-actual (lazy decay)
              memories'  (mapv #(sm/project-all % t-actual decay-rate) memories)
              ;; Unified pattern: rate = beta-mean of (stim, :none) implication.
              ;; Constraint = 1.0 if obs matches implication's consequent, else 0.0.
              ;; Particles where the implication doesn't yet exist get rate=0.5
              ;; (uniform prior), constraint=1.0.
              rates (mx/array
                      (mapv (fn [m]
                              (if-let [impl (sm/lookup-implication m stim :none)]
                                (sm/beta-mean impl)
                                0.5))
                            memories'))
              outcome-bools (mx/array
                              (mapv (fn [m]
                                      (if-let [impl (sm/lookup-implication m stim :none)]
                                        (if (= (:consequent impl) obs) 1.0 0.0)
                                        1.0))
                                    memories'))
              consequent-args {:percept-suite stimulus-detector
                               :retina        {:stim stim}
                               :rates         rates}
              constraints (cm/from-map {:expected-consequent outcome-bools})
              [k1 k2 k3] (rng/split-n key 3)
              vt (dyn/vgenerate sm/consequent-kernel
                                 [t-actual consequent-args]
                                 constraints
                                 n-particles
                                 k1)
              weights-arr (mx/->clj (:weight vt))
              memories'' (mapv #(sm/observe % stim :none obs t-actual
                                            anticip-neg-evidence decay-rate)
                               memories')
              u (mx/item (rng/uniform k2 []))
              indices (sm/systematic-resample weights-arr u)
              memories''' (mapv #(nth memories'' %) indices)
              ;; Diagnostic: average Beta-mean of [stim :none] across particles
              mean-rate (let [active (keep #(sm/lookup-implication % stim :none) memories''')]
                          (if (seq active)
                            (/ (apply + (map sm/beta-mean active)) (count active))
                            0.5))
              entry {:trial t :stim stim :obs obs :mean-rate mean-rate
                     :time-actual t-actual}]
          (recur (inc t) memories''' k3 (conj trace-log entry)))))))

;; -----------------------------------------------------------------------------
;; Diagnostics: per-trial mean Beta-mean over particles, plus α/β extraction
;; -----------------------------------------------------------------------------

(defn report-implication
  "Average α, β, and Beta-mean for (pkey, op-key) across all particle memories."
  [memories pkey op-key]
  (let [matches (keep #(sm/lookup-implication % pkey op-key) memories)]
    (if (empty? matches)
      {:n 0}
      {:n (count matches)
       :mean-alpha (/ (apply + (map :alpha matches)) (count matches))
       :mean-beta  (/ (apply + (map :beta  matches)) (count matches))
       :mean-rate  (/ (apply + (map sm/beta-mean matches)) (count matches))})))

(defn fmt-rate [x] (.toFixed x 3))

(defn print-trace [trace]
  (doseq [{:keys [trial stim obs mean-rate]} trace]
    (println (str "  trial " trial ": stim=" stim " obs=" obs
                  " mean-rate=" (fmt-rate mean-rate)))))

;; -----------------------------------------------------------------------------
;; Phenomenon 1: Acquisition
;; -----------------------------------------------------------------------------

(defn experiment-acquisition []
  (println "\n========== 1. ACQUISITION ==========")
  (println "20 trials of A1 → B1; expect implication to converge to high success rate.")
  (let [trials (vec (repeat 20 {:stim :A1 :consequent :B1}))
        result (run-classical trials default-params 1)
        impl   (report-implication (:final-memories result) :A1 :none)]
    (println "Final implication [A1 :none]:" impl)
    (assoc impl :passed (>= (:mean-rate impl) 0.85))))

;; -----------------------------------------------------------------------------
;; Phenomenon 2 & 3: Extinction and spontaneous recovery
;; -----------------------------------------------------------------------------

(defn experiment-extinction-recovery []
  (println "\n========== 2 & 3. EXTINCTION + SPONTANEOUS RECOVERY ==========")
  (println "20 trials A1→B1, 20 trials A1→nothing, then time-skip + 1 test trial.")
  (let [acquisition (vec (repeat 20 {:stim :A1 :consequent :B1}))
        extinction  (vec (repeat 20 {:stim :A1 :consequent :nothing}))
        ;; Spontaneous recovery: long delay, then test trial
        recovery    [{:stim :A1 :consequent :B1 :time-skip 50}]
        all-trials  (vec (concat acquisition extinction recovery))
        result      (run-classical all-trials default-params 2)
        ;; After acquisition (trial 19): expect high rate for B1
        ;; After extinction (trial 39): expect rate dropped, but α still present
        ;; After recovery (trial 40): rate partially recovers
        memories    (:final-memories result)
        impl-B1     (report-implication memories :A1 :none)
        memories-with-B1     (filter #(= :B1
                                          (:consequent (sm/lookup-implication % :A1 :none)))
                                     memories)
        memories-with-nothing (filter #(= :nothing
                                           (:consequent (sm/lookup-implication % :A1 :none)))
                                      memories)]
    (println "Final implication [A1 :none] (mixed):" impl-B1)
    (println "Particles where consequent=B1:" (count memories-with-B1))
    (println "Particles where consequent=:nothing:" (count memories-with-nothing))
    (println "Last 5 trials of trace:")
    (print-trace (take-last 5 (:trace result)))
    (let [trace-vec (vec (:trace result))
          rate-at-39 (:mean-rate (nth trace-vec 39))
          rate-at-40 (:mean-rate (nth trace-vec 40))
          recovered? (> rate-at-40 rate-at-39)]
      (println "Trial 39 (last extinction) rate:" (fmt-rate rate-at-39))
      (println "Trial 40 (post-delay test) rate:" (fmt-rate rate-at-40))
      {:result result
       :impl impl-B1
       :rate-39 rate-at-39
       :rate-40 rate-at-40
       :passed recovered?})))

;; -----------------------------------------------------------------------------
;; Phenomenon 4: Gradual extinction
;; -----------------------------------------------------------------------------

(defn experiment-gradual-extinction []
  (println "\n========== 4. GRADUAL EXTINCTION ==========")
  (println "20 trials A1→B1 (acquisition), 20 trials A1→B1 with prob decreasing.")
  (let [;; Acquisition
        acquisition (vec (repeat 20 {:stim :A1 :consequent :B1}))
        ;; Gradual: B1 with prob 1.0, 0.95, 0.9, ... 0.05
        gradual (vec (for [i (range 20)]
                       (let [p (- 1.0 (* i 0.05))]
                         {:stim :A1
                          :consequent (if (< (rand) p) :B1 :nothing)})))
        all-trials (vec (concat acquisition gradual))
        result (run-classical all-trials default-params 3)
        impl   (report-implication (:final-memories result) :A1 :none)]
    (println "Final implication [A1 :none]:" impl)
    (println "Last 5 trials:")
    (print-trace (take-last 5 (:trace result)))
    ;; Empirical rate during gradual phase ≈ 0.475 + acquisition prior pull
    ;; Should land between extinction floor and acquisition ceiling.
    (assoc impl :passed (<= 0.4 (:mean-rate impl) 0.85))))

;; -----------------------------------------------------------------------------
;; Phenomenon 5: Conditional discrimination
;; -----------------------------------------------------------------------------

(defn experiment-conditional-discrimination []
  (println "\n========== 5. CONDITIONAL DISCRIMINATION ==========")
  (println "Two contexts; in C1, A1→B1; in C2, A1→B2. Compound stimuli as keys.")
  (let [trials (vec (apply concat
                            (for [_ (range 30)]
                              [{:stim {:context :C1 :stim :A1} :consequent :B1}
                               {:stim {:context :C2 :stim :A1} :consequent :B2}])))
        result (run-classical trials default-params 4)
        impl-C1 (report-implication (:final-memories result)
                                     {:context :C1 :stim :A1} :none)
        impl-C2 (report-implication (:final-memories result)
                                     {:context :C2 :stim :A1} :none)]
    (println "Implication [(C1, A1) :none]:" impl-C1)
    (println "Implication [(C2, A1) :none]:" impl-C2)
    {:c1 impl-C1
     :c2 impl-C2
     :passed (and (>= (:mean-rate impl-C1) 0.85)
                  (>= (:mean-rate impl-C2) 0.85))}))

;; -----------------------------------------------------------------------------
;; Run all five
;; -----------------------------------------------------------------------------

(defn -main []
  (let [r1 (experiment-acquisition)
        r2 (experiment-extinction-recovery)
        r3 (experiment-gradual-extinction)
        r4 (experiment-conditional-discrimination)
        all-pass? (and (:passed r1) (:passed r2) (:passed r3) (:passed r4))]
    (println "\n========== SUMMARY ==========")
    (println "1. Acquisition rate =" (fmt-rate (:mean-rate r1))
             (if (:passed r1) "PASS" "FAIL"))
    (println "2/3. Extinction → Recovery: r39=" (fmt-rate (:rate-39 r2))
             "r40=" (fmt-rate (:rate-40 r2))
             (if (:passed r2) "PASS" "FAIL"))
    (println "4. Gradual ext rate =" (fmt-rate (:mean-rate r3))
             (if (:passed r3) "PASS" "FAIL"))
    (println "5. CD rates =" (fmt-rate (:mean-rate (:c1 r4)))
             "/" (fmt-rate (:mean-rate (:c2 r4)))
             (if (:passed r4) "PASS" "FAIL"))
    (when-not all-pass?
      (println "\nFAIL — at least one phenomenon did not meet threshold")
      (js/process.exit 1))
    (println "\nDone.")))

(-main)
