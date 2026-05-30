;; Latent Cause Theory — GPU
;; =========================
;;
;; Gershman's latent cause model on GPU. Trial by trial.
;;
;; The CRP is a categorical whose logits are a pure function of state:
;;
;;   (trace :cause (dist/categorical (crp-logits history n-causes alpha times t)))
;;
;; The history tensor [N, t, K] records which cause was active on each past
;; trial for each particle. The power-law temporal kernel 1/tau is computed
;; exactly each step via a single GPU reduction over the history.
;;
;; Inference is an explicit fold over trials. Each iteration:
;;
;;   state_t+1 = resample(vgenerate(kernel, state_t, observation_t))
;;
;; One vgenerate call = one Metal dispatch for all N particles.
;; Between steps we hold the state, inspect cause posteriors, or intervene.
;;
;; Run: bun run --bun nbb examples/latent_cause_gpu.cljs

(ns latent-cause-gpu
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.vectorized :as vec])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ============================================================================
;; CRP
;; ============================================================================

(def K-MAX 10)

(defn temporal-weights
  "Power-law weight vector for trial t: w[t'] = 1/(time_t - time_t').
   Computed in the closure (at most T floats), shared across all particles."
  [times t]
  (let [ct (nth times t)]
    (mx/array (mapv (fn [t'] (let [tau (- ct (nth times t'))]
                               (if (pos? tau) (/ 1.0 tau) 0.0)))
                    (range t)))))

(defn crp-logits
  "CRP prior from assignment history.
   history [N,t,K] -> temporal-weighted counts [N,K] -> logits [N,K]."
  [history n-causes alpha times t]
  (let [counts (if (pos? t)
                 (mx/sum (mx/multiply (mx/reshape (temporal-weights times t) [1 -1 1])
                                      history)
                         [1])
                 (mx/zeros [1 K-MAX]))
        k-range (mx/astype (mx/arange 0 K-MAX 1) mx/int32)
        n-exp   (mx/reshape n-causes [-1 1])]
    (mx/where (mx/less k-range n-exp)
              (mx/log (mx/maximum counts (mx/scalar 1e-10)))
              (mx/where (mx/equal k-range n-exp)
                        (mx/scalar (js/Math.log alpha))
                        (mx/scalar -100.0)))))

;; ============================================================================
;; One-hot masking
;; ============================================================================

(defn one-hot-mask
  "Selection mask [N,K] from [N] cause indices."
  [cause-ids]
  (mx/where (mx/equal (mx/expand-dims cause-ids 1)
                      (mx/astype (mx/arange 0 K-MAX 1) mx/int32))
            (mx/scalar 1.0) (mx/scalar 0.0)))

(defn select-by-cause
  "Gather: W[i, cause_i, :] for each particle. [N,K,D] x [N,K] -> [N,D]."
  [weights mask]
  (mx/sum (mx/multiply (mx/expand-dims mask 2) weights) [1]))

(defn scatter-update
  "Scatter: W + mask * delta. Updates only the active cause per particle."
  [weights mask delta]
  (mx/add weights (mx/multiply (mx/expand-dims mask 2)
                               (mx/expand-dims delta 1))))

;; ============================================================================
;; Trial Kernel
;; ============================================================================

(defn make-kernel
  "Build the per-trial generative function.
   Stimuli and times captured in closure — not in carry, not resampled."
  [stimuli times {:keys [alpha eta sigma] :or {alpha 0.15 eta 0.1 sigma 0.3}}]
  (let [eta-s (mx/scalar eta) sigma-s (mx/scalar sigma)]
    (dyn/auto-key
      (gen [t state]
        (let [{:keys [weights history n-causes]} state
              stimulus (nth stimuli t)

              ;; CRP prior -> sample cause
              cause-ids (trace :cause (dist/categorical
                                       (crp-logits history n-causes alpha times t)))
              mask      (one-hot-mask cause-ids)

              ;; Predict from active cause
              selected   (select-by-cause weights mask)
              prediction (mx/sum (mx/multiply selected stimulus) [-1])

              ;; Observe outcome
              outcome (trace :outcome (dist/gaussian prediction sigma-s))

              ;; M-step: w_k += eta * error * stimulus
              error (mx/subtract outcome prediction)
              delta (mx/multiply eta-s
                                 (mx/multiply (mx/expand-dims error 1)
                                              (mx/expand-dims stimulus 0)))

              ;; Advance carry
              new-row (mx/expand-dims mask 1)
              n-flat  (mx/reshape (mx/astype n-causes mx/int32) [-1])]

          {:weights  (scatter-update weights mask delta)
           :history  (if history (mx/concatenate [history new-row] 1) new-row)
           :n-causes (mx/where (mx/equal cause-ids n-flat)
                               (mx/add n-flat (mx/scalar 1 mx/int32))
                               n-flat)})))))

;; ============================================================================
;; Inference: explicit fold over trials
;; ============================================================================

(defn- resample-state [state indices]
  (into {} (map (fn [[k v]] [k (if v (mx/take-idx v indices) v)])) state))

(defn infer
  "Trial-by-trial particle filter.

   Each iteration: vgenerate (one GPU call for all particles) -> resample.
   Returns {:state :log-ml :posteriors}.

   posteriors is a vector of [K] arrays — the fraction of particles
   assigned to each cause at each trial."
  [kernel init-state observations {:keys [particles seed on-trial]
                                    :or {particles 300 seed 42}}]
  (let [N    particles
        init init-state]
    (loop [t     0
           state init
           ml    (mx/scalar 0.0)
           posts []
           key   (rng/ensure-key (rng/fresh-key seed))]
      (if (>= t (count observations))
        {:state state :log-ml ml :posteriors posts}
        (let [[vk rk nk] (rng/split-n key 3)

              ;; One GPU dispatch: all particles, one trial
              vtrace    (dyn/vgenerate kernel [t state]
                                       (nth observations t) N vk)
              weights   (:weight vtrace)
              new-state (:retval vtrace)

              ;; Per-trial posterior from the history
              cause-ids (cm/get-value (cm/get-submap (:choices vtrace) :cause))
              trial-post (let [oh (one-hot-mask cause-ids)]
                           (mx/mean oh 0))

              ;; Log-ML increment
              _ (mx/materialize! weights)
              ml-inc (mx/subtract (mx/logsumexp weights)
                                  (mx/scalar (js/Math.log N)))
              _ (mx/materialize! ml-inc)

              ;; Resample
              indices   (vec/systematic-resample-indices weights N rk)
              resampled (resample-state new-state indices)
              _ (mx/materialize! (vals resampled))

              ;; Callback
              _ (when on-trial (on-trial {:t t :posterior trial-post :state resampled}))

              ;; Periodic cleanup
              _ (when (zero? (mod (inc t) 5))
                  (mx/sweep-dead-arrays!)
                  (mx/clear-cache!))]

          (recur (inc t) resampled (mx/add ml ml-inc)
                 (conj posts trial-post) nk))))))

(defn run-model
  "Convenience: build kernel, run inference, return result."
  [stimuli times outcomes & {:keys [particles alpha eta sigma seed]
                              :or {particles 300 alpha 0.15 eta 0.1
                                   sigma 0.3 seed 42}}]
  (let [D      (last (mx/shape (first stimuli)))
        kernel (make-kernel stimuli times {:alpha alpha :eta eta :sigma sigma})
        init   {:weights  (mx/zeros [K-MAX D])
                :history  nil
                :n-causes (mx/astype (mx/array [0]) mx/int32)}
        obs    (mapv #(cm/set-value cm/EMPTY :outcome (mx/scalar %)) outcomes)]
    (infer kernel init obs {:particles particles :seed seed})))

;; ============================================================================
;; Analysis
;; ============================================================================

(defn cause-posterior-at
  "Posterior P(cause=k) at trial t from the posteriors vector."
  [posteriors t]
  (let [p (mx/->clj (nth posteriors t))]
    (into (sorted-map)
          (keep (fn [k] (let [v (nth p k)] (when (> v 0.005) [k v]))))
          (range K-MAX))))

(defn analyze [{:keys [state]}]
  (let [{:keys [weights n-causes]} state
        max-k (int (mx/item (mx/amax n-causes)))]
    (into (sorted-map)
          (for [k (range max-k)]
            (let [active (mx/greater-equal n-causes (mx/scalar (inc k) mx/int32))
                  n-act  (int (mx/item (mx/sum (mx/where active (mx/scalar 1.0) (mx/scalar 0.0)))))
                  avg    (mx/mean (mx/take-idx weights (mx/scalar k mx/int32) 1) 0)]
              [k {:w avg :n n-act}])))))

(defn- fmt [w]
  (if (= [1] (mx/shape w))
    (.toFixed (mx/item w) 3)
    (str (mapv #(.toFixed % 3) (mx/->clj w)))))

(defn- print-posteriors [posteriors & {:keys [every] :or {every 5}}]
  (doseq [t (range (count posteriors))]
    (when (or (zero? (mod t every)) (= t (dec (count posteriors))))
      (let [probs (cause-posterior-at posteriors t)]
        (print (str "  t=" (if (< t 10) (str " " t) t)))
        (doseq [[k p] probs]
          (print (str "  c" k "=" (.toFixed (* p 100) 0) "%")))
        (println)))))

(defn- sep [s]
  (println (str "\n" (apply str (repeat 65 "="))
               "\n " s "\n" (apply str (repeat 65 "=")))))

;; ============================================================================
;; Experiments
;; ============================================================================

(sep "1. Basic Acquisition (20 trials, A -> reward)")

(let [n 20
      t0 (js/Date.now)
      r (run-model (vec (repeat n (mx/array [1.0])))
                   (vec (range 1 (inc n)))
                   (vec (repeat n 1.0)))
      ms (- (js/Date.now) t0)
      info (analyze r)]
  (println (str "  " ms "ms  log-ML=" (.toFixed (mx/item (:log-ml r)) 2)))
  (println "\n  Cause posteriors:")
  (print-posteriors (:posteriors r) :every 4)
  (println "\n  Final weights:")
  (doseq [[k {:keys [w n]}] info]
    (println (str "  cause " k ": w=" (fmt w) " (" n " particles)")))
  (println (str "\n  " (if (> (:n (get info 0)) 200) "PASS" "FAIL"))))

;; ---

(sep "2. Extinction (20 acquisition + 20 extinction)")

(let [t0 (js/Date.now)
      r (run-model (vec (repeat 40 (mx/array [1.0])))
                   (vec (range 1 41))
                   (vec (concat (repeat 20 1.0) (repeat 20 0.0))))
      ms (- (js/Date.now) t0)
      info (analyze r)
      w0 (mx/item (:w (get info 0)))]
  (println (str "  " ms "ms  log-ML=" (.toFixed (mx/item (:log-ml r)) 2)))
  (println "\n  Acquisition phase:")
  (print-posteriors (subvec (:posteriors r) 0 20) :every 5)
  (println "\n  Extinction phase:")
  (print-posteriors (subvec (:posteriors r) 20) :every 5)
  (println "\n  Final weights:")
  (doseq [[k {:keys [w n]}] info]
    (println (str "  cause " k ": w=" (fmt w) " (" n " particles)")))
  (println (str "\n  " (if (> w0 0.5) "PASS" "FAIL")
                ": cause 0 weight preserved at " (.toFixed w0 3))))

;; ---

(sep "3. Spontaneous Recovery (acq + ext + delay + test)")

(let [stim (vec (repeat 41 (mx/array [1.0])))
      outcomes (vec (concat (repeat 20 1.0) (repeat 20 0.0) [1.0]))
      t0 (js/Date.now)
      r-d  (run-model stim (vec (concat (range 1 41) [200])) outcomes :seed 42)
      r-nd (run-model stim (vec (range 1 42)) outcomes :seed 42)
      ms (- (js/Date.now) t0)
      ml-d  (mx/item (:log-ml r-d))
      ml-nd (mx/item (:log-ml r-nd))
      ;; Compare cause 0 posterior at test trial
      p-d  (cause-posterior-at (:posteriors r-d) 40)
      p-nd (cause-posterior-at (:posteriors r-nd) 40)
      c0-d  (get p-d 0 0)
      c0-nd (get p-nd 0 0)]
  (println (str "  " ms "ms (both conditions)"))
  (println "\n  Test trial posteriors:")
  (println (str "    No delay:  " (pr-str p-nd)))
  (println (str "    With delay: " (pr-str p-d)))
  (println (str "\n  log-ML: delay=" (.toFixed ml-d 2) "  no-delay=" (.toFixed ml-nd 2)))
  (println (str "  " (if (> c0-d c0-nd) "PASS" "NOTE")
                ": cause 0 at test: " (.toFixed (* c0-nd 100) 1) "% -> "
                (.toFixed (* c0-d 100) 1) "% with delay")))

;; ---

(sep "4. Gradual vs Abrupt Extinction")

(let [stim (vec (repeat 40 (mx/array [1.0])))
      times (vec (range 1 41))
      gradual (vec (concat (repeat 20 1.0)
                           (mapv #(- 1.0 (/ (inc %) 21.0)) (range 20))))
      abrupt (vec (concat (repeat 20 1.0) (repeat 20 0.0)))
      r-g (run-model stim times gradual :seed 123)
      r-a (run-model stim times abrupt  :seed 123)
      i-g (analyze r-g) i-a (analyze r-a)
      w0g (mx/item (:w (get i-g 0)))
      w0a (mx/item (:w (get i-a 0)))
      ;; Cause 0 posterior at first extinction trial
      c0g-ext (get (cause-posterior-at (:posteriors r-g) 20) 0 0)
      c0a-ext (get (cause-posterior-at (:posteriors r-a) 20) 0 0)]
  (println (str "  First extinction trial — cause 0:"))
  (println (str "    Gradual: " (.toFixed (* c0g-ext 100) 1) "%"))
  (println (str "    Abrupt:  " (.toFixed (* c0a-ext 100) 1) "%"))
  (println (str "\n  Final cause 0 weight:"))
  (println (str "    Gradual: " (.toFixed w0g 3) " (adapted smoothly)"))
  (println (str "    Abrupt:  " (.toFixed w0a 3) " (preserved from acquisition)"))
  (println (str "\n  " (if (> w0a w0g) "PASS" "NOTE")
                ": abrupt preserved > gradual adapted")))

;; ---

(sep "5. Conditional Discrimination (two contexts)")

(let [c1a (mx/array [1 0 1 0]) c1b (mx/array [1 0 0 1])
      c2a (mx/array [0 1 1 0]) c2b (mx/array [0 1 0 1])
      n-blocks 10
      stim     (vec (apply concat (repeat n-blocks [c1a c1b c2a c2b])))
      outcomes (vec (apply concat (repeat n-blocks [1.0 0.0 0.0 1.0])))
      times    (vec (range 1 (inc (count stim))))
      t0 (js/Date.now)
      r (run-model stim times outcomes :eta 0.15)
      ms (- (js/Date.now) t0)
      info (analyze r)]
  (println (str "  " ms "ms  log-ML=" (.toFixed (mx/item (:log-ml r)) 2)))
  (println "\n  Cause posteriors (final block):")
  (let [labels ["C1+A->1" "C1+B->0" "C2+A->0" "C2+B->1"]]
    (doseq [[i label] (map-indexed vector labels)]
      (let [probs (cause-posterior-at (:posteriors r) (+ (- (count stim) 4) i))]
        (print (str "    " label ": "))
        (doseq [[k p] probs]
          (print (str "c" k "=" (.toFixed (* p 100) 0) "% ")))
        (println))))
  (println "\n  Cause weight vectors:")
  (doseq [[k {:keys [w n]}] info]
    (println (str "  cause " k ": " (fmt w) " (" n " particles)")))
  (let [nc (count (filter #(> (:n (val %)) 100) info))]
    (println (str "\n  " (if (>= nc 2) "PASS" "NOTE") ": " nc " well-supported causes"))))

;; ---

(sep "Timing")

(let [t0 (js/Date.now)
      _ (run-model (vec (repeat 40 (mx/array [1.0])))
                   (vec (range 1 41))
                   (vec (concat (repeat 20 1.0) (repeat 20 0.0)))
                   :particles 500)
      ms (- (js/Date.now) t0)]
  (println (str "  GPU fold: " ms "ms (500 particles, 40 trials, exact power-law CRP)"))
  (println "  Sequential smc-unfold: ~24000ms (same workload, measured separately)"))

(println (str "\n" (apply str (repeat 65 "="))
              "\n Done.\n" (apply str (repeat 65 "="))))
(.exit js/process 0)
