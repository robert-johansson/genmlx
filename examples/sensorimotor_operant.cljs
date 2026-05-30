;; Operant Conditioning via Hammer-style Sensorimotor Learning
;; ============================================================
;;
;; Replicates the three operant phenomena from operant_conditioning.cljs
;; using the SAME unified sensorimotor kernel (genmlx.sensorimotor) — no
;; modifications. Key architectural claim: operant conditioning is
;; classical conditioning + action selection via deduction; both fall out
;; of the same kernel and the same Beta-Bernoulli machinery.
;;
;;   1. Simple discrimination — A1→press-left, A2→press-right reinforced
;;   2. Reversal               — after acquisition, swap which side reinforced
;;   3. Matching-to-sample (MTS) — sample × comparison-pair → which side
;;
;; Critical architectural test (this file MUST NOT contain mx/where reward
;; injection): per-particle weights are derived from p/generate constraints
;; on the kernel's :expected-consequent trace site. Each particle's chosen
;; action determines its outcome (via env response), and the weight is
;; log P(outcome | implication's success-rate). No hardcoded penalty.
;;
;; Run: bun run --bun nbb examples/sensorimotor_operant.cljs

(ns sensorimotor-operant
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.mlx.random :as rng]
            [genmlx.sensorimotor :as sm]
            [genmlx.gen :refer-macros [gen]]))

(def default-params
  {:decay-rate            0.95
   :anticip-neg-evidence  0.5
   :decision-threshold    0.51
   :min-temperature       0.1
   :n-particles           500
   :goal                  :reinforced
   :op-keys               [:left :right]})

(def stimulus-detector
  (dyn/auto-key
    (gen [retina]
      [(:stim retina) (:stim retina)])))

;; -----------------------------------------------------------------------------
;; The operant run loop — same kernel, but per-particle outcomes
;; -----------------------------------------------------------------------------

(defn run-operant
  "Drive the sensorimotor kernel through operant trials.

   Differences from classical:
     - Goal :reinforced is active throughout training.
     - The kernel chooses an operation via desire-driven categorical.
     - Each particle's outcome depends on its CHOSEN action — the
       environment-response function takes (trial-config, op-key) → :reinforced
       or :not-reinforced. This is per-particle: each particle observes its
       own consequent.
     - Resampling weights particles by how well their action's predicted
       success-rate matched the actual outcome.

   trials: vector of {:stim K  :env-fn (fn [op-key] -> :reinforced | :not-reinforced)
                      :testing? bool}.
           When :testing? is true, no revision happens (frozen memories).

   Returns: {:final-memories  [memory-per-particle]
             :trace           [{:trial t :stim K :ops [...] :rewards [...] :acc f} ...]}."
  [trials params seed]
  (let [{:keys [decay-rate anticip-neg-evidence
                decision-threshold min-temperature
                n-particles goal op-keys]} params
        initial-memories (vec (repeat n-particles sm/empty-memory))
        key0 (rng/fresh-key seed)]
    (loop [t 0
           memories initial-memories
           key key0
           trace-log []]
      (if (>= t (count trials))
        {:final-memories memories
         :trace          trace-log}
        (let [trial    (nth trials t)
              stim     (:stim trial)
              env-fn   (:env-fn trial)
              testing? (boolean (:testing? trial))
              ;; Project all particle memories
              memories' (mapv #(sm/project-all % t decay-rate) memories)
              ;; Per-particle [N, K] success-rate tensor for (stim, op) → goal
              rates           (sm/particle-rates memories' stim goal op-keys)
              decision-logits (sm/particle-decision-logits rates
                                                            {:decision-threshold decision-threshold
                                                             :min-temperature min-temperature})
              [k1 k2 k3 k4] (rng/split-n key 4)
              ;; Step 1: action-kernel samples per-particle operation
              action-args {:percept-suite   stimulus-detector
                           :retina          {:stim stim}
                           :decision-logits decision-logits}
              action-vt (dyn/vgenerate sm/action-kernel
                                        [t action-args]
                                        cm/EMPTY
                                        n-particles
                                        k1)
              op-idxs-tensor (:retval action-vt)
              op-idxs (mx/->clj op-idxs-tensor)
              ;; Step 2: environment resolves per-particle outcomes from chosen ops
              outcomes (mapv (fn [op-idx]
                               (env-fn (nth op-keys op-idx)))
                             op-idxs)
              ;; Step 3: consequent-kernel with per-particle constraint produces
              ;; weights via vgenerate's importance-weight mechanism. No host-side
              ;; log-prob computation. The chosen rate is gathered on GPU.
              chosen-rates (sm/gather-chosen-rates rates op-idxs-tensor)
              outcome-bools (mx/array
                              (mapv #(if (= % goal) 1.0 0.0) outcomes))
              consequent-args {:percept-suite stimulus-detector
                               :retina        {:stim stim}
                               :rates         chosen-rates}
              consequent-constraints (cm/from-map
                                       {:expected-consequent outcome-bools})
              consequent-vt (dyn/vgenerate sm/consequent-kernel
                                            [t consequent-args]
                                            consequent-constraints
                                            n-particles
                                            k2)
              proper-weights (mx/->clj (:weight consequent-vt))
              ;; End-of-trial revision (skip if testing)
              memories'' (if testing?
                           memories'
                           (mapv (fn [m particle-idx]
                                   (let [op-idx  (nth op-idxs particle-idx)
                                         op-key  (nth op-keys op-idx)
                                         outcome (nth outcomes particle-idx)]
                                     (sm/observe m stim op-key outcome t
                                                  anticip-neg-evidence decay-rate)))
                                 memories'
                                 (range n-particles)))
              memories'''
              (if testing?
                memories''
                (let [u (mx/item (rng/uniform k3 []))
                      indices (sm/systematic-resample proper-weights u)]
                  (mapv #(nth memories'' %) indices)))
              ;; Diagnostic: per-trial accuracy = fraction of correct actions
              correct-count (count (filter #(= % goal) outcomes))
              acc (/ correct-count (double n-particles))
              entry {:trial t :stim stim :acc acc
                     :ops-summary (frequencies (map #(nth op-keys %) op-idxs))
                     :outcomes-summary (frequencies outcomes)}]
          (recur (inc t) memories''' k4 (conj trace-log entry)))))))

;; -----------------------------------------------------------------------------
;; Diagnostics
;; -----------------------------------------------------------------------------

(defn fmt-rate [x] (.toFixed x 3))

(defn report-implication [memories pkey op-key]
  (let [matches (keep #(sm/lookup-implication % pkey op-key) memories)]
    (if (empty? matches)
      {:n 0}
      {:n (count matches)
       :mean-alpha (/ (apply + (map :alpha matches)) (count matches))
       :mean-beta  (/ (apply + (map :beta  matches)) (count matches))
       :mean-rate  (/ (apply + (map sm/beta-mean matches)) (count matches))
       :consequent (->> matches (map :consequent) frequencies)})))

(defn last-n-acc [trace n]
  (let [last-n (take-last n trace)]
    (/ (apply + (map :acc last-n)) (count last-n))))

(defn print-trace-summary [trace n-step]
  (doseq [entry (filter #(zero? (mod (:trial %) n-step)) trace)]
    (println (str "  trial " (:trial entry)
                  " stim=" (:stim entry)
                  " acc=" (fmt-rate (:acc entry))
                  " ops=" (:ops-summary entry)))))

;; -----------------------------------------------------------------------------
;; Phenomenon 1: Simple discrimination
;; -----------------------------------------------------------------------------

(defn experiment-simple-discrimination []
  (println "\n========== 1. SIMPLE DISCRIMINATION ==========")
  (println "A1 → press-left = reinforced; A2 → press-right = reinforced.")
  (println "Expect: ≥85% accuracy in last 20 trials.")
  (let [n-trials 100
        trials (vec (for [i (range n-trials)]
                      (let [stim (if (zero? (mod i 2)) :A1 :A2)]
                        {:stim stim
                         :env-fn (fn [op]
                                   (cond
                                     (and (= stim :A1) (= op :left))  :reinforced
                                     (and (= stim :A2) (= op :right)) :reinforced
                                     :else :not-reinforced))})))
        result (run-operant trials default-params 1)
        acc-last-20 (last-n-acc (:trace result) 20)
        impl-A1-left (report-implication (:final-memories result) :A1 :left)
        impl-A2-right (report-implication (:final-memories result) :A2 :right)]
    (println "Trace (every 10 trials):")
    (print-trace-summary (:trace result) 10)
    (println "Last-20 accuracy:" (fmt-rate acc-last-20))
    (println "Implication [A1 :left]:" impl-A1-left)
    (println "Implication [A2 :right]:" impl-A2-right)
    (let [;; The "wrong" implications: A1→right and A2→left should be low
          impl-A1-right (report-implication (:final-memories result) :A1 :right)
          impl-A2-left  (report-implication (:final-memories result) :A2 :left)]
      {:acc acc-last-20
       :passed (and (>= acc-last-20 0.85)
                    (>= (:mean-rate impl-A1-left)  0.7)
                    (>= (:mean-rate impl-A2-right) 0.7))})))

;; -----------------------------------------------------------------------------
;; Phenomenon 2: Reversal
;; -----------------------------------------------------------------------------

(defn experiment-reversal []
  (println "\n========== 2. REVERSAL ==========")
  (println "100 trials with A1→left, then 100 trials with A1→right (swapped).")
  (println "Expect: ≥75% accuracy in last 20 trials of reversal phase.")
  (let [training (vec (for [i (range 100)]
                        (let [stim (if (zero? (mod i 2)) :A1 :A2)]
                          {:stim stim
                           :env-fn (fn [op]
                                     (cond
                                       (and (= stim :A1) (= op :left))  :reinforced
                                       (and (= stim :A2) (= op :right)) :reinforced
                                       :else :not-reinforced))})))
        reversal (vec (for [i (range 100)]
                        (let [stim (if (zero? (mod i 2)) :A1 :A2)]
                          {:stim stim
                           :env-fn (fn [op]
                                     (cond
                                       (and (= stim :A1) (= op :right)) :reinforced
                                       (and (= stim :A2) (= op :left))  :reinforced
                                       :else :not-reinforced))})))
        all-trials (vec (concat training reversal))
        result (run-operant all-trials default-params 2)
        acc-train-last-20 (last-n-acc (take 100 (:trace result)) 20)
        acc-rev-last-20 (last-n-acc (drop 100 (:trace result)) 20)]
    (println "Training acc (last 20):" (fmt-rate acc-train-last-20))
    (println "Reversal acc (last 20):" (fmt-rate acc-rev-last-20))
    (println "Reversal trace (every 20 trials):")
    (print-trace-summary (drop 100 (:trace result)) 20)
    {:training-acc acc-train-last-20
     :reversal-acc acc-rev-last-20
     :passed (>= acc-rev-last-20 0.75)}))

;; -----------------------------------------------------------------------------
;; Phenomenon 3: Matching-to-sample
;; -----------------------------------------------------------------------------

(defn experiment-mts []
  (println "\n========== 3. MATCHING-TO-SAMPLE ==========")
  (println "Sample on a side, comparison stimuli left/right.")
  (println "If sample=A1, pressing matching side reinforced; if sample=A2, other side.")
  (println "Compound stimulus key includes (sample, left-comp, right-comp).")
  (println "Expect: ≥85% accuracy in last 20 trials.")
  (let [n-trials 200
        trials
        (vec (for [i (range n-trials)]
               (let [sample (rand-nth [:A1 :A2])
                     left-on-left? (rand-nth [true false])
                     left  (if left-on-left? :B1 :B2)
                     right (if left-on-left? :B2 :B1)
                     stim {:sample sample :left left :right right}
                     ;; Correct if pressed side matches sample (B1 ⇔ A1, B2 ⇔ A2)
                     correct-side (cond
                                    (and (= sample :A1) (= left :B1)) :left
                                    (and (= sample :A1) (= right :B1)) :right
                                    (and (= sample :A2) (= left :B2)) :left
                                    (and (= sample :A2) (= right :B2)) :right)]
                 {:stim stim
                  :env-fn (fn [op]
                            (if (= op correct-side)
                              :reinforced
                              :not-reinforced))})))
        result (run-operant trials default-params 3)
        acc-last-20 (last-n-acc (:trace result) 20)]
    (println "Trace (every 20 trials):")
    (print-trace-summary (:trace result) 20)
    (println "Last-20 accuracy:" (fmt-rate acc-last-20))
    (println "Number of distinct compound stimuli encountered:")
    (let [all-stims (set (mapv :stim (:trace result)))]
      (println "  total unique compound keys:" (count all-stims)))
    (println "Memory size (avg implications per particle):")
    (let [sizes (mapv #(count (:by-key %)) (:final-memories result))]
      (println "  min:" (apply min sizes) "max:" (apply max sizes)
               "mean:" (fmt-rate (/ (apply + sizes) (double (count sizes))))))
    {:acc acc-last-20 :passed (>= acc-last-20 0.85)}))

;; -----------------------------------------------------------------------------
;; Run all three
;; -----------------------------------------------------------------------------

(defn -main []
  (let [r1 (experiment-simple-discrimination)
        r2 (experiment-reversal)
        r3 (experiment-mts)
        all-pass? (and (:passed r1) (:passed r2) (:passed r3))]
    (println "\n========== SUMMARY ==========")
    (println "1. Simple disc acc:" (fmt-rate (:acc r1))
             (if (:passed r1) "PASS" "FAIL"))
    (println "2. Reversal acc:" (fmt-rate (:reversal-acc r2))
             (if (:passed r2) "PASS" "FAIL"))
    (println "3. MTS acc:" (fmt-rate (:acc r3))
             (if (:passed r3) "PASS" "FAIL"))
    (when-not all-pass?
      (println "\nFAIL — at least one phenomenon did not meet threshold")
      (js/process.exit 1))
    (println "\nDone.")))

(-main)
