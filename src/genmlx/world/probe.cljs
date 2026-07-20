(ns genmlx.world.probe
  "Pure statistics for the GRPO probe-set eval (genmlx-lkt0).

   WHY: the 2026-07-20 night's learning verdict rested on comparing single
   K=8 group means across epochs — per-pair SEM(diff) 0.3–0.44, paired
   t=-0.74: an instrument that cannot resolve shifts under ~0.25 mean
   reward, far above what a 50-step run can plausibly produce. The probe
   fixes the POWER: a fixed prompt set scored with N rollouts per prompt
   (generation-only, no weight update) before and after training. At N=64
   and P=6 prompts the paired-delta MDE lands around 0.06–0.08 — matched to
   plausible effect sizes.

   This namespace is the PURE half: probe-set selection, per-prompt
   summaries, pooled statistics, and the paired pre/post delta. The
   effectful half (train/generate-batch rollouts, JSONL logging) lives in
   scripts/grpo_sessions.cljs, which feeds completions + rewards in and
   writes the records out. No MLX, no engine, no I/O here."
  (:require [genmlx.llm.toolcall :as tc]))

(defn probe-indices
  "K distinct indices evenly spaced over [0, n-prompts) — deterministic, so
   the probe set is FIXED for a given kept-prompt corpus (spans sessions
   instead of clustering on the first one). k > n-prompts degrades to all
   indices."
  [n-prompts k]
  (let [k (max 0 (min k n-prompts))]
    (vec (distinct (map #(quot (* % n-prompts) k) (range k))))))

(defn- mean [xs] (when (seq xs) (/ (reduce + 0.0 xs) (count xs))))

(defn- sample-sd
  "Sample standard deviation (n-1); nil below 2 observations."
  [xs]
  (when (>= (count xs) 2)
    (let [m (mean xs)]
      (js/Math.sqrt (/ (reduce + 0.0 (map #(let [d (- % m)] (* d d)) xs))
                       (dec (count xs)))))))

(defn summarize-prompt
  "One probe prompt's rollout summary. `key` identifies the prompt across
   passes (e.g. \"<session-id>/<turn-index>\"); `rewards` the N per-rollout
   rewards; `completions` the N texts (truncated-frac + mean-chars come
   from them). opts {:floor -1.0} sets the floor value counted by
   :floored-frac."
  ([key rewards completions] (summarize-prompt key rewards completions {}))
  ([key rewards completions {:keys [floor] :or {floor -1.0}}]
   (let [n (count rewards)]
     {:key key
      :n n
      :reward-mean (mean rewards)
      :reward-std (sample-sd rewards)
      :floored-frac (when (pos? n)
                      (/ (count (filter #(== % floor) rewards)) n))
      :truncated-frac (when (pos? n)
                        (/ (count (filter #(:truncated?
                                            (tc/strip-truncated-tail %))
                                          completions))
                           n))
      :mean-chars (mean (map count completions))})))

(defn pool
  "Pool per-prompt summaries into a pass-level statistic. The SE is
   CLUSTERED at the prompt level (sd of per-prompt means / sqrt P) — the
   rollouts within a prompt share its difficulty, so pooling raw rollouts
   would understate the error (the night's lesson: prompt difficulty spans
   -0.29..+0.76)."
  [summaries]
  (let [means (mapv :reward-mean summaries)
        p (count means)]
    {:n-prompts p
     :n-rollouts (reduce + 0 (map :n summaries))
     :mean (mean means)
     :se (when-let [sd (sample-sd means)] (/ sd (js/Math.sqrt p)))
     :floored-frac (mean (keep :floored-frac summaries))
     :truncated-frac (mean (keep :truncated-frac summaries))
     :mean-chars (mean (keep :mean-chars summaries))}))

(defn pass-record
  "The JSONL-shaped record for one probe pass. `label` is \"pre\", \"post\",
   or \"step-N\"; `step` the training step count at probe time."
  [label step summaries]
  {:probe label
   :step step
   :prompts summaries
   :pooled (pool summaries)})

(defn delta
  "Paired pre/post comparison over the prompts present in BOTH passes
   (matched by :key). Returns {:per-prompt [{:key :delta}] :mean :sem :t
   :n-pairs}; :sem/:t nil below 2 pairs. t = mean paired delta / SEM — the
   number the night's verdict lacked the power to compute meaningfully."
  [pre-summaries post-summaries]
  (let [pre-by (into {} (map (juxt :key identity)) pre-summaries)
        pairs (into []
                    (keep (fn [post]
                            (when-let [pre (pre-by (:key post))]
                              {:key (:key post)
                               :delta (- (:reward-mean post)
                                         (:reward-mean pre))})))
                    post-summaries)
        ds (mapv :delta pairs)
        m (mean ds)
        sem (when-let [sd (sample-sd ds)]
              (/ sd (js/Math.sqrt (count ds))))]
    {:per-prompt pairs
     :n-pairs (count pairs)
     :mean m
     :sem sem
     :t (when (and m sem (pos? sem)) (/ m sem))}))
