;; Operant Conditioning with Latent Causes — GPU
;; ===============================================
;;
;; Extends the CRP latent cause model with action selection.
;; Replicates three experiments from Johansson (2024):
;;   1. Simple discrimination
;;   2. Changing contingencies
;;   3. Conditional discrimination (matching-to-sample)
;;
;; Each trial:
;;   stimulus → cause (CRP) → action (bernoulli) → outcome (deterministic)
;;
;; The kernel traces :cause and :action. The outcome is computed inside the
;; kernel from the contingency rule (closure). The M-step updates both
;; action-weights and outcome-weights per cause.
;;
;; The fold weights particles by how well each predicted the outcome:
;;   weight = log P(actual_outcome | model's predicted probability)
;;
;; During testing, no feedback: equal weights, no M-step.
;;
;; Run: bun run --bun nbb examples/operant_conditioning.cljs

(ns operant-conditioning
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.vectorized :as vec])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ============================================================================
;; CRP (same as latent_cause_gpu.cljs)
;; ============================================================================

(def K-MAX 10)

(defn temporal-weights [times t]
  (let [ct (nth times t)]
    (mx/array (mapv (fn [t'] (let [tau (- ct (nth times t'))]
                               (if (pos? tau) (/ 1.0 tau) 0.0)))
                    (range t)))))

(defn crp-logits [history n-causes alpha times t]
  (let [counts (if (pos? t)
                 (mx/sum (mx/multiply (mx/reshape (temporal-weights times t) [1 -1 1])
                                      history) [1])
                 (mx/zeros [1 K-MAX]))
        k-range (mx/astype (mx/arange 0 K-MAX 1) mx/int32)
        n-exp (mx/reshape n-causes [-1 1])]
    (mx/where (mx/less k-range n-exp)
              (mx/log (mx/maximum counts (mx/scalar 1e-10)))
              (mx/where (mx/equal k-range n-exp)
                        (mx/scalar (js/Math.log alpha))
                        (mx/scalar -100.0)))))

;; ============================================================================
;; One-hot masking
;; ============================================================================

(defn one-hot-mask [cause-ids]
  (mx/where (mx/equal (mx/expand-dims cause-ids 1)
                      (mx/astype (mx/arange 0 K-MAX 1) mx/int32))
            (mx/scalar 1.0) (mx/scalar 0.0)))

(defn select-by-cause [weights mask]
  (mx/sum (mx/multiply (mx/expand-dims mask 2) weights) [1]))

(defn scatter-update [weights mask delta]
  (mx/add weights (mx/multiply (mx/expand-dims mask 2)
                               (mx/expand-dims delta 1))))

;; ============================================================================
;; Operant Kernel
;; ============================================================================

(defn make-kernel
  "Build the per-trial operant kernel.

   The contingency rule (correct-actions, feedback-flags) and stimuli
   are captured in the closure.

   Carry state:
     :action-weights   [N, K, D]   — maps stimulus -> action logit per cause
     :outcome-weights  [N, K, D+1] — maps stimulus+action -> outcome logit per cause
     :history          [N, t, K]   — cause assignment history
     :n-causes         [N]
     :outcome-pred     [N]         — predicted P(reinforced), for fold to weight by

   Trace sites:
     :cause  — categorical from CRP (always unconstrained)
     :action — bernoulli from action policy (always unconstrained)"
  [stimuli times correct-actions feedback-flags
   {:keys [alpha eta] :or {alpha 0.15 eta 0.3}}]
  (let [D (last (mx/shape (first stimuli)))
        eta-s (mx/scalar eta)]
    (dyn/auto-key
     (gen [t state]
          (let [{:keys [action-weights outcome-weights history n-causes]} state
                stimulus (nth stimuli t)
                feedback? (nth feedback-flags t)

              ;; CRP -> cause
                cause-ids (trace :cause (dist/categorical
                                         (crp-logits history n-causes alpha times t)))
                mask (one-hot-mask cause-ids)

              ;; Action policy: P(right) = sigmoid(action_weights_k . stimulus)
                action-w (select-by-cause action-weights mask) ;; [N, D]
                action-logit (mx/sum (mx/multiply action-w stimulus) [-1]) ;; [N]
                action-prob (mx/sigmoid action-logit)
                action (trace :action (dist/bernoulli action-prob)) ;; [N] 0.0 or 1.0

              ;; Outcome prediction: P(reinforced) = sigmoid(outcome_w . [stim, action])
              ;; Split outcome-w [N,D+1] into stimulus part [N,D] and action part [N]
              ;; to avoid broadcast-to. Dot product = stim_part.stim + act_part * action
                outcome-w (select-by-cause outcome-weights mask) ;; [N, D+1]
                outcome-stim-logit (mx/sum (mx/multiply (mx/take-idx outcome-w
                                                                     (mx/astype (mx/arange 0 D 1) mx/int32) 1)
                                                        stimulus) [-1]) ;; [N]
                outcome-act-logit (mx/multiply (mx/take-idx outcome-w
                                                            (mx/scalar D mx/int32) 1)
                                               action) ;; [N]
                outcome-logit (mx/add outcome-stim-logit outcome-act-logit)
                outcome-pred (mx/sigmoid outcome-logit) ;; [N]

              ;; Outcome: always compute (needed for broadcasting to [N,...])
              ;; During testing, use dummy correct-action; eta=0 prevents learning
                correct-act (mx/scalar (nth correct-actions t))
                actual (mx/where (mx/equal action correct-act)
                                 (mx/scalar 1.0) (mx/scalar 0.0))
                effective-eta (if feedback? eta-s (mx/scalar 0.0))

              ;; M-step: always runs (eta=0 during testing → no weight change)
              ;; This ensures weights broadcast from [K,D] to [N,K,D] at step 0
              ;; Action weights: REINFORCE with signed reward
              ;; reinforced → push toward chosen action, not reinforced → push away
                reinforcement (mx/subtract (mx/multiply (mx/scalar 2.0) actual) (mx/scalar 1.0))
                action-error (mx/multiply reinforcement (mx/subtract action action-prob))
                action-delta (mx/multiply effective-eta
                                          (mx/multiply (mx/expand-dims action-error 1)
                                                       (mx/expand-dims stimulus 0)))
                new-action-weights (scatter-update action-weights mask action-delta)

                outcome-error (mx/subtract actual outcome-pred)
                stim-delta (mx/multiply (mx/expand-dims outcome-error 1)
                                        (mx/expand-dims stimulus 0))
                act-delta (mx/expand-dims (mx/multiply outcome-error action) 1)
                full-delta (mx/multiply effective-eta
                                        (mx/concatenate [stim-delta act-delta] 1))
                new-outcome-weights (scatter-update outcome-weights mask full-delta)

              ;; History
                new-row (mx/expand-dims mask 1)
                n-flat (mx/reshape (mx/astype n-causes mx/int32) [-1])]

            {:action-weights new-action-weights
             :outcome-weights new-outcome-weights
             :history (if history (mx/concatenate [history new-row] 1) new-row)
             :n-causes (mx/where (mx/equal cause-ids n-flat)
                                 (mx/add n-flat (mx/scalar 1 mx/int32))
                                 n-flat)
             :outcome-pred outcome-pred
             :actual-outcome actual})))))

;; ============================================================================
;; Inference fold
;; ============================================================================

(defn bernoulli-log-prob
  "log P(x | Bernoulli(p)), clamping p to [eps, 1-eps]."
  [p x]
  (let [p (mx/maximum (mx/minimum p (mx/scalar 0.999)) (mx/scalar 0.001))]
    (mx/add (mx/multiply x (mx/log p))
            (mx/multiply (mx/subtract (mx/scalar 1.0) x)
                         (mx/log (mx/subtract (mx/scalar 1.0) p))))))

(defn- resample-state [state indices]
  (into {} (map (fn [[k v]] [k (if v (mx/take-idx v indices) v)])) state))

(defn infer
  "Trial-by-trial operant particle filter.

   The fold weights particles by outcome prediction accuracy during training.
   During testing (no feedback), all particles have equal weight.

   Returns {:state :log-ml :actions :accuracies}."
  [kernel stimuli feedback-flags correct-actions
   {:keys [particles seed] :or {particles 300 seed 42}}]
  (let [N particles
        D (last (mx/shape (first stimuli)))
        init {:action-weights (mx/zeros [K-MAX D])
              :outcome-weights (mx/zeros [K-MAX (inc D)])
              :history nil
              :n-causes (mx/astype (mx/array [0]) mx/int32)
              :outcome-pred nil
              :actual-outcome nil}
        T (count stimuli)]
    (loop [t 0
           state init
           ml (mx/scalar 0.0)
           actions []
           accs []
           key (rng/ensure-key (rng/fresh-key seed))]
      (if (>= t T)
        {:state state :log-ml ml :actions actions :accuracies accs}
        (let [[vk rk nk] (rng/split-n key 3)
              feedback? (nth feedback-flags t)

              ;; All particles act freely (no constraints)
              vtrace (dyn/vgenerate kernel [t state] cm/EMPTY N vk)
              new-state (:retval vtrace)

              ;; Extract actions and compute accuracy
              chosen (cm/get-value (cm/get-submap (:choices vtrace) :action))
              correct (mx/scalar (nth correct-actions t))
              acc (mx/item (mx/mean (mx/where (mx/equal chosen correct)
                                              (mx/scalar 1.0) (mx/scalar 0.0))))

              ;; Per-particle weights: correct action → high weight, incorrect → low
              ;; This is the operant selection signal: feedback shapes which particles survive
              weights (if feedback?
                        (mx/where (mx/equal (:actual-outcome new-state) (mx/scalar 1.0))
                                  (mx/scalar 0.0) (mx/scalar -3.0))
                        (mx/zeros [N]))
              _ (mx/materialize! weights)
              ml-inc (mx/subtract (mx/logsumexp weights)
                                  (mx/scalar (js/Math.log N)))
              _ (mx/materialize! ml-inc)

              ;; Resample
              indices (vec/systematic-resample-indices weights N rk)
              resampled (resample-state new-state indices)
              _ (mx/materialize! (keep val resampled))

              _ (when (zero? (mod (inc t) 5))
                  (mx/sweep-dead-arrays!) (mx/clear-cache!))]

          (recur (inc t) resampled (mx/add ml ml-inc)
                 (conj actions chosen) (conj accs acc) nk))))))

;; ============================================================================
;; Trial generation
;; ============================================================================

(defn make-block
  "Generate one block of trials. Returns [{:stimulus :correct-action} ...]."
  [trial-types n-each]
  (let [trials (vec (apply concat (repeat n-each trial-types)))]
    (shuffle trials)))

(defn make-phase [trial-types n-blocks n-each feedback?]
  (let [blocks (vec (apply concat (map (fn [_] (make-block trial-types n-each)) (range n-blocks))))]
    {:stimuli (mapv :stimulus blocks)
     :correct-actions (mapv :correct-action blocks)
     :feedback-flags (vec (repeat (count blocks) feedback?))}))

(defn concat-phases [& phases]
  {:stimuli (vec (apply concat (map :stimuli phases)))
   :correct-actions (vec (apply concat (map :correct-actions phases)))
   :feedback-flags (vec (apply concat (map :feedback-flags phases)))})

;; ============================================================================
;; Display
;; ============================================================================

(defn block-accuracies
  "Compute per-block accuracy from per-trial accuracies."
  [accs block-size]
  (mapv (fn [block]
          (/ (reduce + block) (count block)))
        (partition block-size accs)))

(defn- sep [s]
  (println (str "\n" (apply str (repeat 70 "="))
                "\n " s "\n" (apply str (repeat 70 "=")))))

;; ============================================================================
;; Experiment 1: Simple Discrimination
;; ============================================================================

(sep "Experiment 1: Simple Discrimination")
(println "  A1 left: choose left. A1 right: choose right.")
(println "  Baseline (3 blocks) -> Training (3 blocks) -> Testing (3 blocks)")

(let [;; Stimulus encoding: [A1_left, A1_right, A2_left, A2_right]
      trial-types [{:stimulus (mx/array [1 0 0 1]) :correct-action 0.0} ;; A1 left, A2 right -> left
                   {:stimulus (mx/array [0 1 1 0]) :correct-action 1.0}] ;; A1 right, A2 left -> right
      baseline (make-phase trial-types 3 6 false)
      training (make-phase trial-types 3 6 true)
      testing (make-phase trial-types 3 6 false)
      design (concat-phases baseline training testing)
      times (vec (range 1 (inc (count (:stimuli design)))))
      kernel (make-kernel (:stimuli design) times
                          (:correct-actions design)
                          (:feedback-flags design) {})
      t0 (js/Date.now)
      result (infer kernel (:stimuli design)
                    (:feedback-flags design) (:correct-actions design)
                    {:particles 300})
      ms (- (js/Date.now) t0)
      block-accs (block-accuracies (:accuracies result) 12)]
  (println (str "\n  " ms "ms"))
  (println "\n  Block accuracies (12 trials/block):")
  (println "  Baseline:  " (mapv #(.toFixed (* % 100) 0) (subvec block-accs 0 3)))
  (println "  Training:  " (mapv #(.toFixed (* % 100) 0) (subvec block-accs 3 6)))
  (println "  Testing:   " (mapv #(.toFixed (* % 100) 0) (subvec block-accs 6 9)))
  (let [test-mean (/ (reduce + (subvec block-accs 6)) 3)]
    (println (str "\n  " (if (> test-mean 0.8) "PASS" "FAIL")
                  ": testing accuracy " (.toFixed (* test-mean 100) 1) "%"))))

;; ============================================================================
;; Experiment 2: Changing Contingencies
;; ============================================================================

(sep "Experiment 2: Changing Contingencies")
(println "  Phase 1: choose A1's position. Phase 2: choose A2's position (reversed).")

(let [trial-types-1 [{:stimulus (mx/array [1 0 0 1]) :correct-action 0.0}
                     {:stimulus (mx/array [0 1 1 0]) :correct-action 1.0}]
      trial-types-2 [{:stimulus (mx/array [1 0 0 1]) :correct-action 1.0} ;; reversed!
                     {:stimulus (mx/array [0 1 1 0]) :correct-action 0.0}]
      baseline (make-phase trial-types-1 2 6 false)
      training1 (make-phase trial-types-1 4 6 true)
      testing1 (make-phase trial-types-1 2 6 false)
      training2 (make-phase trial-types-2 6 6 true)
      testing2 (make-phase trial-types-2 2 6 false)
      design (concat-phases baseline training1 testing1 training2 testing2)
      times (vec (range 1 (inc (count (:stimuli design)))))
      kernel (make-kernel (:stimuli design) times
                          (:correct-actions design)
                          (:feedback-flags design) {})
      t0 (js/Date.now)
      result (infer kernel (:stimuli design)
                    (:feedback-flags design) (:correct-actions design)
                    {:particles 300})
      ms (- (js/Date.now) t0)
      accs (block-accuracies (:accuracies result) 12)
      n-base 2 n-tr1 4 n-te1 2 n-tr2 6 n-te2 2
      i0 0 i1 n-base i2 (+ i1 n-tr1) i3 (+ i2 n-te1) i4 (+ i3 n-tr2)]
  (println (str "\n  " ms "ms"))
  (println "\n  Block accuracies:")
  (println "  Baseline:   " (mapv #(.toFixed (* % 100) 0) (subvec accs i0 i1)))
  (println "  Training 1: " (mapv #(.toFixed (* % 100) 0) (subvec accs i1 i2)))
  (println "  Testing 1:  " (mapv #(.toFixed (* % 100) 0) (subvec accs i2 i3)))
  (println "  Training 2: " (mapv #(.toFixed (* % 100) 0) (subvec accs i3 i4)))
  (println "  Testing 2:  " (mapv #(.toFixed (* % 100) 0) (subvec accs i4)))
  (let [te2-mean (/ (reduce + (subvec accs i4)) n-te2)]
    (println (str "\n  " (if (> te2-mean 0.7) "PASS" "FAIL")
                  ": testing 2 accuracy " (.toFixed (* te2-mean 100) 1) "%"))))

;; ============================================================================
;; Experiment 3: Conditional Discrimination (MTS)
;; ============================================================================

(sep "Experiment 3: Conditional Discrimination (Matching-to-Sample)")
(println "  Sample A1: choose B1. Sample A2: choose B2.")

(let [;; Features: [A1_s, A2_s, B1_l, B1_r, B2_l, B2_r, A1×B1_l, A1×B1_r, A2×B2_l, A2×B2_r]
      ;; Interaction features match NARS's compound hypotheses
      trial-types [{:stimulus (mx/array [1 0 1 0 0 1  1 0 0 0]) :correct-action 0.0} ;; A1, B1 left
                   {:stimulus (mx/array [1 0 0 1 1 0  0 1 0 0]) :correct-action 1.0} ;; A1, B1 right
                   {:stimulus (mx/array [0 1 1 0 0 1  0 0 0 1]) :correct-action 1.0} ;; A2, B2 right
                   {:stimulus (mx/array [0 1 0 1 1 0  0 0 1 0]) :correct-action 0.0}] ;; A2, B2 left
      baseline (make-phase trial-types 3 3 false)
      training (make-phase trial-types 12 3 true)
      testing (make-phase trial-types 3 3 false)
      design (concat-phases baseline training testing)
      times (vec (range 1 (inc (count (:stimuli design)))))
      kernel (make-kernel (:stimuli design) times
                          (:correct-actions design)
                          (:feedback-flags design)
                          {:alpha 0.3 :eta 0.3})
      t0 (js/Date.now)
      result (infer kernel (:stimuli design)
                    (:feedback-flags design) (:correct-actions design)
                    {:particles 500})
      ms (- (js/Date.now) t0)
      accs (block-accuracies (:accuracies result) 12)]
  (println (str "\n  " ms "ms"))
  (println "\n  Block accuracies:")
  (println "  Baseline: " (mapv #(.toFixed (* % 100) 0) (subvec accs 0 3)))
  (println "  Training: " (mapv #(.toFixed (* % 100) 0) (subvec accs 3 15)))
  (println "  Testing:  " (mapv #(.toFixed (* % 100) 0) (subvec accs 15)))
  (let [test-mean (/ (reduce + (subvec accs 15)) 3)]
    (println (str "\n  " (if (> test-mean 0.7) "PASS" "FAIL")
                  ": testing accuracy " (.toFixed (* test-mean 100) 1) "%"))))

;; ============================================================================

(println (str "\n" (apply str (repeat 70 "="))
              "\n Done.\n" (apply str (repeat 70 "="))))
(.exit js/process 0)
