;; Latent Cause Theory of Conditioning
;; ====================================
;;
;; Gershman's latent cause model (2010, 2017) in GenMLX.
;;
;; The Chinese Restaurant Process looks exotic, but in GenMLX it reduces to:
;;
;;   (trace :cause (dist/categorical (crp-logits state)))
;;
;; An ordinary categorical trace site whose parameters are a pure function
;; of the organism's learning state. The CRP's "infinite capacity" is just
;; a "new cause" slot with weight alpha. The temporal kernel modulates
;; recency. All of this is logits computation — the Gen operation is a
;; standard categorical.
;;
;; The Unfold combinator threads the organism's state trial by trial:
;;
;;   trial-kernel: (carry, trial-index) -> carry'
;;
;; Each trial, the kernel does six things:
;;   1. Compute CRP logits from carry          (pure function of state)
;;   2. Trace cause from categorical           (Gen operation — random choice)
;;   3. Predict outcome from cause weights     (pure: w_k . stimulus)
;;   4. Trace outcome from Gaussian            (Gen operation — constrained by observation)
;;   5. Update cause weights by error          (pure: w + eta * x * (y - w.x))
;;   6. Record assignment in carry             (pure: conj to history)
;;
;; Only steps 2 and 4 are random. Everything else is pure computation on
;; the carry state — immutable Clojure maps holding MLX arrays.
;;
;; SMC (smc-unfold) maintains particles = different cause-assignment
;; hypotheses. Each trial: extend all particles, weight by prediction
;; accuracy, resample. The posterior over cause sequences emerges from
;; the particle distribution.
;;
;; Run: bun run --bun nbb examples/latent_cause_conditioning.cljs

(ns latent-cause-conditioning
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.inference.smc :as smc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ============================================================================
;; CRP Prior
;; ============================================================================

(def K-MAX 10)

(defn temporal-weight
  "Power-law recency kernel: K(tau) = 1/tau.
   Recent assignments weigh more than distant ones."
  [current-time event-time]
  (let [tau (- current-time event-time)]
    (if (pos? tau) (/ 1.0 tau) 0.0)))

(defn cause-count
  "Temporal-weighted count of assignments to cause k."
  [assignments k current-time]
  (transduce (comp (filter #(= (:cause-id %) k))
                   (map #(temporal-weight current-time (:time %))))
             + 0.0 assignments))

(defn n-active-causes [assignments]
  (if (empty? assignments) 0
    (inc (apply max (map :cause-id assignments)))))

(defn crp-logits
  "CRP prior as a fixed-size log-probability vector.
   Existing causes get log(temporal-weighted count).
   The next unused slot gets log(alpha) — the 'new cause' option.
   All remaining slots get -100 (effectively zero)."
  [assignments alpha current-time]
  (let [n (n-active-causes assignments)]
    (->> (range K-MAX)
         (mapv (fn [k]
                 (cond
                   (< k n) (js/Math.log (max (cause-count assignments k current-time) 1e-10))
                   (= k n) (js/Math.log alpha)
                   :else   -100.0)))
         mx/array)))

;; ============================================================================
;; Trial Kernel
;; ============================================================================

(def trial-kernel
  "Unfold kernel: one trial of the latent cause model.

   Carry state is a Clojure map:
     :cause-weights  {cause-id -> MLX weight vector}
     :assignments    [{:cause-id int :time number} ...]
     :stimuli        [MLX stimulus per trial]
     :times          [explicit time per trial]
     :alpha, :eta, :sigma, :n-features

   Two trace sites:
     :cause   — which latent cause is active (categorical from CRP)
     :outcome — observed consequence (gaussian from cause prediction)"
  (dyn/auto-key
    (gen [t state]
      (let [{:keys [cause-weights assignments stimuli times
                    alpha eta sigma n-features]} state
            stimulus    (nth stimuli t)
            current-time (nth times t)

            ;; 1-2. CRP prior -> sample cause
            cause-id    (trace :cause (dist/categorical
                                       (crp-logits assignments alpha current-time)))
            cause-int   (int (mx/item cause-id))

            ;; 3. Predict from cause weights
            weights     (get cause-weights cause-int (mx/zeros [n-features]))
            prediction  (mx/sum (mx/multiply weights stimulus))

            ;; 4. Outcome (constrained by observation during inference)
            outcome     (trace :outcome (dist/gaussian prediction (mx/scalar sigma)))

            ;; 5. M-step: adjust weights toward observed outcome
            error       (mx/subtract outcome prediction)
            new-weights (mx/add weights
                               (mx/multiply (mx/scalar eta)
                                            (mx/multiply stimulus error)))]

        ;; 6. Updated carry
        (-> state
            (assoc-in [:cause-weights cause-int] new-weights)
            (update :assignments conj {:cause-id cause-int
                                       :time current-time}))))))

;; ============================================================================
;; Running and Analysis
;; ============================================================================

(defn make-state
  "Initial carry: empty history, no learned weights."
  [stimuli times {:keys [alpha eta sigma] :or {alpha 0.15 eta 0.1 sigma 0.3}}]
  {:cause-weights {}
   :assignments   []
   :stimuli       stimuli
   :times         times
   :alpha         alpha
   :eta           eta
   :sigma         sigma
   :n-features    (last (mx/shape (first stimuli)))})

(defn run-model
  "Run the latent cause model via SMC particle filtering.
   Returns {:log-ml :traces :final-ess}."
  [stimuli times outcomes & {:keys [particles alpha eta sigma seed]
                              :or {particles 200 alpha 0.15 eta 0.1
                                   sigma 0.3 seed 42}}]
  (smc/smc-unfold {:particles particles :key (rng/fresh-key seed)}
                  trial-kernel
                  (make-state stimuli times {:alpha alpha :eta eta :sigma sigma})
                  (mapv #(cm/set-value cm/EMPTY :outcome (mx/scalar %)) outcomes)))

(defn cause-posteriors
  "Posterior probability of each cause at each timestep,
   estimated from particle frequencies."
  [traces n-steps]
  (let [n (count traces)]
    (mapv (fn [t]
            (let [ids (mapv #(int (mx/item (cm/get-choice (:choices %) [t :cause])))
                            traces)]
              (into (sorted-map)
                    (map (fn [[k v]] [k (/ v n)]))
                    (frequencies ids))))
          (range n-steps))))

(defn final-weights
  "Average cause weights across particles at the final step."
  [traces]
  (let [states (mapv #(last (:retval %)) traces)
        all-causes (into (sorted-set) (mapcat #(keys (:cause-weights %))) states)]
    (into (sorted-map)
          (keep (fn [k]
                  (let [ws (keep #(get (:cause-weights %) k) states)]
                    (when (seq ws)
                      [k {:w (mx/mean (mx/stack (vec ws)) 0)
                          :n (count ws)}]))))
          all-causes)))

;; ============================================================================
;; Display Helpers
;; ============================================================================

(defn- fmt-weight [w]
  (if (= [1] (mx/shape w))
    (.toFixed (mx/item w) 3)
    (str (mapv #(.toFixed % 3) (mx/->clj w)))))

(defn- print-posteriors [posteriors & {:keys [every] :or {every 5}}]
  (doseq [t (range (count posteriors))]
    (when (or (zero? (mod t every)) (= t (dec (count posteriors))))
      (let [probs (nth posteriors t)]
        (print (str "  t=" (if (< t 10) (str " " t) t)))
        (doseq [[k p] probs]
          (print (str "  c" k "=" (.toFixed (* p 100) 0) "%")))
        (println)))))

(defn- sep [title]
  (println (str "\n" (apply str (repeat 65 "="))
               "\n " title
               "\n" (apply str (repeat 65 "=")))))

;; ============================================================================
;; Experiment 1: Basic Acquisition
;; ============================================================================

(sep "Experiment 1: Basic Acquisition (20 trials, A -> reward)")

(let [n       20
      stim    (vec (repeat n (mx/array [1.0])))
      times   (vec (range 1 (inc n)))
      result  (run-model stim times (vec (repeat n 1.0)) :particles 300)
      posts   (cause-posteriors (:traces result) n)
      weights (final-weights (:traces result))]
  (println (str "  log-ML: " (.toFixed (mx/item (:log-ml result)) 2)))
  (println "\n  Cause posteriors over time:")
  (print-posteriors posts :every 4)
  (println "\n  Final cause weights:")
  (doseq [[k {:keys [w n]}] weights]
    (println (str "    cause " k ": w=" (fmt-weight w) " (" n " particles)")))
  (let [c0 (get (last posts) 0 0)]
    (println (str "\n  " (if (> c0 0.6) "PASS" "FAIL")
                  ": cause 0 at " (.toFixed (* c0 100) 0) "%"))))

;; ============================================================================
;; Experiment 2: Extinction
;; ============================================================================

(sep "Experiment 2: Extinction (20 acquisition + 20 extinction)")

(let [n-acq 20, n-ext 20, n (+ n-acq n-ext)
      stim    (vec (repeat n (mx/array [1.0])))
      times   (vec (range 1 (inc n)))
      outcomes (vec (concat (repeat n-acq 1.0) (repeat n-ext 0.0)))
      result  (run-model stim times outcomes :particles 300)
      posts   (cause-posteriors (:traces result) n)
      weights (final-weights (:traces result))]
  (println (str "  log-ML: " (.toFixed (mx/item (:log-ml result)) 2)))
  (println "\n  Acquisition phase:")
  (print-posteriors (subvec posts 0 n-acq) :every 5)
  (println "\n  Extinction phase:")
  (print-posteriors (subvec posts n-acq) :every 5)
  (println "\n  Final cause weights:")
  (doseq [[k {:keys [w n]}] weights]
    (println (str "    cause " k ": w=" (fmt-weight w) " (" n " particles)")))
  (let [final (last posts)
        sig   (count (filter #(> (val %) 0.05) final))]
    (println (str "\n  " (if (>= sig 2) "PASS" "FAIL")
                  ": " sig " significant causes (acquisition weights preserved at "
                  (fmt-weight (:w (get weights 0))) ")"))))

;; ============================================================================
;; Experiment 3: Spontaneous Recovery
;; ============================================================================

(sep "Experiment 3: Spontaneous Recovery (acq + ext + delay + test)")

(let [n-acq 20, n-ext 20
      stim    (vec (concat (repeat (+ n-acq n-ext) (mx/array [1.0]))
                           [(mx/array [1.0])]))
      outcomes (vec (concat (repeat n-acq 1.0) (repeat n-ext 0.0) [1.0]))
      n       (count stim)
      ;; With delay: test at time 200 (long gap after extinction ends at t=40)
      times-delay    (vec (concat (range 1 (inc (+ n-acq n-ext))) [200]))
      ;; Without delay: test at time 41 (immediate)
      times-no-delay (vec (range 1 (inc n)))
      r-delay    (run-model stim times-delay outcomes :particles 300 :seed 42)
      r-no-delay (run-model stim times-no-delay outcomes :particles 300 :seed 42)
      p-delay    (last (cause-posteriors (:traces r-delay) n))
      p-no-delay (last (cause-posteriors (:traces r-no-delay) n))]
  (println "\n  Test trial cause posteriors:")
  (println "    Without delay (time=41):")
  (doseq [[k p] p-no-delay]
    (println (str "      cause " k ": " (.toFixed (* p 100) 1) "%")))
  (println "    With delay (time=200):")
  (doseq [[k p] p-delay]
    (println (str "      cause " k ": " (.toFixed (* p 100) 1) "%")))
  (let [ml-d  (mx/item (:log-ml r-delay))
        ml-nd (mx/item (:log-ml r-no-delay))
        c0-d  (get p-delay 0 0)
        c0-nd (get p-no-delay 0 0)]
    (println (str "\n  log-ML: delay=" (.toFixed ml-d 2)
                  "  no-delay=" (.toFixed ml-nd 2)))
    (println (str "  " (if (> c0-d c0-nd) "PASS" "NOTE")
                  ": acquisition cause " (.toFixed (* c0-nd 100) 1)
                  "% -> " (.toFixed (* c0-d 100) 1) "% with delay"))))

;; ============================================================================
;; Experiment 4: Gradual vs Abrupt Extinction
;; ============================================================================

(sep "Experiment 4: Gradual vs Abrupt Extinction")

(let [n-acq 20, n-ext 20, n (+ n-acq n-ext)
      stim  (vec (repeat n (mx/array [1.0])))
      times (vec (range 1 (inc n)))
      ;; Gradual: outcome ramps smoothly from 1.0 to 0.0
      gradual (vec (concat (repeat n-acq 1.0)
                           (mapv #(- 1.0 (/ (inc %) (inc n-ext)))
                                 (range n-ext))))
      abrupt  (vec (concat (repeat n-acq 1.0) (repeat n-ext 0.0)))
      r-grad (run-model stim times gradual :particles 300 :seed 123)
      r-abr  (run-model stim times abrupt  :particles 300 :seed 123)
      p-grad (cause-posteriors (:traces r-grad) n)
      p-abr  (cause-posteriors (:traces r-abr) n)
      c0-grad-end (get (last p-grad) 0 0)
      c0-abr-end  (get (last p-abr) 0 0)
      w-grad (final-weights (:traces r-grad))
      w-abr  (final-weights (:traces r-abr))]
  (println "\n  First extinction trial — cause 0 (acquisition cause):")
  (println (str "    Gradual: " (.toFixed (* (get (nth p-grad n-acq) 0 0) 100) 1) "%"))
  (println (str "    Abrupt:  " (.toFixed (* (get (nth p-abr n-acq) 0 0) 100) 1) "%"))
  (println "\n  Last trial — cause 0:")
  (println (str "    Gradual: " (.toFixed (* c0-grad-end 100) 1) "%"
               "  w=" (fmt-weight (:w (get w-grad 0))) " (adapted smoothly)"))
  (println (str "    Abrupt:  " (.toFixed (* c0-abr-end 100) 1) "%"
               "  w=" (fmt-weight (:w (get w-abr 0))) " (preserved from acquisition)"))
  (println (str "\n  " (if (> c0-grad-end c0-abr-end) "PASS" "NOTE")
                ": gradual maintained cause 0 (" (.toFixed (* c0-grad-end 100) 0)
                "% vs " (.toFixed (* c0-abr-end 100) 0) "%)")))

;; ============================================================================
;; Experiment 5: Conditional Discrimination
;; ============================================================================

(sep "Experiment 5: Conditional Discrimination (two contexts)")

(let [;; 4 features: [context1 context2 stimA stimB]
      c1a (mx/array [1 0 1 0])  c1b (mx/array [1 0 0 1])
      c2a (mx/array [0 1 1 0])  c2b (mx/array [0 1 0 1])
      n-blocks 10
      stim     (vec (apply concat (repeat n-blocks [c1a c1b c2a c2b])))
      outcomes (vec (apply concat (repeat n-blocks [1.0 0.0 0.0 1.0])))
      n        (count stim)
      times    (vec (range 1 (inc n)))
      result   (run-model stim times outcomes :particles 300 :eta 0.15)
      posts    (cause-posteriors (:traces result) n)
      weights  (final-weights (:traces result))]
  (println (str "  " n " trials (" n-blocks " blocks x 4 trial types)"))
  (println (str "  log-ML: " (.toFixed (mx/item (:log-ml result)) 2)))
  (println "\n  Final block cause posteriors:")
  (let [labels ["C1+A->1" "C1+B->0" "C2+A->0" "C2+B->1"]]
    (doseq [[i label] (map-indexed vector labels)]
      (let [probs (nth posts (+ (- n 4) i))]
        (print (str "    " label ": "))
        (doseq [[k p] probs]
          (print (str "c" k "=" (.toFixed (* p 100) 0) "% ")))
        (println))))
  (println "\n  Cause weight vectors (avg across particles):")
  (doseq [[k {:keys [w n]}] weights]
    (println (str "    cause " k ": " (fmt-weight w) " (" n " particles)")))
  (let [nc (count (filter #(> (:n (val %)) 100) weights))]
    (println (str "\n  " (if (>= nc 2) "PASS" "NOTE")
                  ": " nc " well-supported causes"))))

;; ============================================================================

(println (str "\n" (apply str (repeat 65 "="))
              "\n All experiments complete."
              "\n" (apply str (repeat 65 "="))))

(.exit js/process 0)
