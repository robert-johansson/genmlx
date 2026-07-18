(ns genmlx.world.session-grpo
  "pi sessions -> GRPO training prompts (genmlx-rlho, L4): the
   trajectory->group converter of the overnight-self-improvement epic
   (genmlx-z92i). PURE — no GPU, no model, no train.cljs dependency; the
   output is exactly the prompt shape world.train/train-step! marshals
   (vectors of {:role :content} chat maps, tool fields riding along).

   The GRPO engine is ON-POLICY: train-step! regenerates group-size
   completions per prompt itself, so a converted session contributes its
   DECISION POINTS (the context each assistant turn was decoded from),
   not its recorded completions. The recorded completion rides along for
   reward shaping and provenance.

   REWARD SEAM: each prompt vector carries CLJS metadata
   {:session-grpo/meta {:session-id :cwd :path :turn-index :completion}}.
   train-step!'s reward-fn receives (prompt completion-text) with the
   very prompt value from the batch, so (meta prompt) recovers provenance
   without any side channel — the oracle keys its scoring off that.

   Modes: :terminal (default — one point per session, the LAST decision
   point; the epic's terminal-reward-per-trial starting mode) or :all
   (every assistant turn = multi-step credit, the follow-up)."
  (:require [clojure.string :as str]
            [genmlx.llm.pi-session :as ps]
            ["fs" :as fs]
            ["path" :as path]))

(defn- prompt-msg
  "Strip a converted message to the train-step! chat shape: role/content
   plus the tool fields ->chat-msg passes through. :images never survive
   here — image handling is decided upstream (typed error or skip)."
  [m]
  (select-keys m [:role :content :toolCalls :toolCallId :isError]))

(defn decision-points
  "One point per assistant turn of a path->messages vector:
   {:index k :prompt [msg...] :completion msg} — :prompt is the context
   the turn was decoded from (messages before k), :completion the
   recorded assistant message. Prompt messages are train-step!-shaped."
  [messages]
  (mapv (fn [k]
          {:index      k
           :prompt     (mapv prompt-msg (subvec messages 0 k))
           :completion (prompt-msg (nth messages k))})
        (ps/assistant-indices messages)))

(defn terminal-points
  "The last decision point only (terminal-reward-per-trial), as a
   zero-or-one element vector."
  [messages]
  (let [pts (decision-points messages)]
    (if (seq pts) [(peek pts)] [])))

(defn- attach-meta
  "Stamp provenance metadata onto a point's prompt vector (the reward
   seam) and return the finished {:prompt :meta} entry."
  [point header file]
  (let [m {:session-id (:id header)
           :cwd        (:cwd header)
           :path       file
           :turn-index (:index point)
           :completion (:completion point)}]
    {:prompt (with-meta (:prompt point) {:session-grpo/meta m})
     :meta   m}))

(defn prompt-meta
  "Recover a prompt's provenance inside a reward-fn: (prompt-meta prompt)
   -> the {:session-id :cwd :path :turn-index :completion} map, or nil."
  [prompt]
  (:session-grpo/meta (meta prompt)))

(defn session-file->points
  "Convert one session file into prompt points.
   opts: :mode :terminal (default) | :all
         :system-prompt — prepended system message (pi holds it outside
           the session file; pass the DEPLOYED one for render parity)
         :skip-images? — false (default): an image-bearing prompt is a
           typed error :images-unsupported; true: image-bearing points
           are dropped and counted.
   Returns {:points [{:prompt :meta}...] :skipped n}."
  ([file] (session-file->points file {}))
  ([file {:keys [mode system-prompt skip-images?] :or {mode :terminal}}]
   (let [session  (ps/read-session file)
         messages (ps/path->messages (ps/leaf-path session)
                                     {:system-prompt system-prompt})
         points   (case mode
                    :terminal (terminal-points messages)
                    :all      (decision-points messages)
                    (throw (ex-info (str "session-grpo: unknown mode " (pr-str mode))
                                    {:genmlx/error :unknown-mode :mode mode})))
         imaged?  (fn [{:keys [index]}]
                    (some :images (subvec messages 0 index)))
         with-img (filterv imaged? points)]
     (when (and (seq with-img) (not skip-images?))
       (throw (ex-info "session-grpo: image-bearing prompt — the training engine is text-only; pass :skip-images? true to drop these points"
                       {:genmlx/error :images-unsupported :file file
                        :turns (mapv :index with-img)})))
     {:points  (mapv #(attach-meta % (:header session) file)
                     (remove (set with-img) points))
      :skipped (count with-img)})))

(defn sessions->prompts
  "Convert a session DIRECTORY (every *.jsonl inside) or an explicit
   vector of file paths into a flat training-prompt batch:
   {:prompts [prompt...] :points [{:prompt :meta}...] :skipped n
    :failed [{:file :error}...]}.
   :prompts is ready for train-step!; each prompt carries the reward-seam
   metadata (see prompt-meta). opts as session-file->points, plus
   :on-error :throw (default) | :skip — :skip records unparseable or
   compacted files under :failed instead of throwing."
  ([src] (sessions->prompts src {}))
  ([src {:keys [on-error] :or {on-error :throw} :as opts}]
   (let [files (if (string? src)
                 (->> (fs/readdirSync src)
                      (filter #(str/ends-with? % ".jsonl"))
                      (mapv #(path/join src %))
                      sort
                      vec)
                 (vec src))
         init  {:points [] :skipped 0 :failed []}
         acc   (reduce
                (fn [acc file]
                  (try
                    (let [{:keys [points skipped]}
                          (session-file->points file opts)]
                      (-> acc
                          (update :points into points)
                          (update :skipped + skipped)))
                    (catch :default e
                      (if (= :skip on-error)
                        (update acc :failed conj
                                {:file file :error (ex-message e)})
                        (throw e)))))
                init files)]
     (assoc acc :prompts (mapv :prompt (:points acc))))))
