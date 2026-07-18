(ns genmlx.world.session-reward
  "Reward builders for GRPO-on-sessions (genmlx-qhy4, L4): pure
   (fn [prompt completion-text] -> number) closures for world.train
   train-step!, computed from the session data itself — no oracle keys,
   no GPU, no I/O at reward time.

   THE REWARD SEAM: prompts converted by genmlx.world.session-grpo carry
   provenance as CLJS metadata ({:session-id :turn-index :completion ...}
   via prompt-meta), and train-step! calls the reward-fn with the very
   prompt value from the batch — so a reward can key off the administered
   session without any side channel. match-administered-reward is the
   built-in demonstration of that seam; the maithri2 downstream oracle
   replaces it later through the SAME contract (they own answer keys; we
   consume scores).

   EXTERNAL ORACLE CONTRACT (resolve-reward): a plugin file's LAST FORM
   must be (fn [ctx] -> (fn [prompt completion] -> number)) where ctx =
   {:points [...] :toolset [...] :opts {...}}. The returned closure must
   be PURE and return FINITE numbers — an infinite/NaN reward poisons the
   whole GRPO group (the world.train-reward floor lesson)."
  (:require [genmlx.llm.toolcall :as tc]
            [genmlx.world.session-grpo :as sg]
            [nbb.core :as nbb]
            [promesa.core :as p]))

(defn observed-toolset
  "Derive the toolset actually exercised by `points` (session-grpo output):
   every tool call in the prompt contexts and administered completions
   contributes its name + argument names. Returns the parse-tool-calls
   declared-check shape [{:name n :params [{:name p}...]}], name-sorted —
   an honest stand-in when the deployed tool declarations are not passed."
  [points]
  (let [calls (mapcat (fn [{:keys [prompt meta]}]
                        (concat (mapcat :toolCalls prompt)
                                (:toolCalls (:completion meta))))
                      points)
        by-name (reduce (fn [acc {:keys [name arguments]}]
                          (update acc name (fnil into #{})
                                  (keys (js->clj (js/JSON.parse
                                                  (or arguments "{}"))))))
                        {} calls)]
    (->> (sort-by key by-name)
         (mapv (fn [[n ps]]
                 {:name n :params (mapv (fn [p] {:name p}) (sort ps))})))))

(defn tool-format-reward
  "Well-formedness reward over the qwen3_xml tool-call dialect: any parse
   error or (when `toolset` is non-empty) undeclared call -> :reward-floor
   (default -1.0); well-formed declared calls -> 1.0; a call-free
   completion -> :no-call-reward (default 0.0 — prose is not malformed,
   just not a tool action)."
  ([toolset] (tool-format-reward toolset {}))
  ([toolset {:keys [reward-floor no-call-reward]
             :or {reward-floor -1.0 no-call-reward 0.0}}]
   (fn [_prompt completion]
     (let [{:keys [calls errors]} (tc/parse-tool-calls
                                   (str completion)
                                   (when (seq toolset) toolset))]
       (cond
         (seq errors) reward-floor
         (seq calls)  1.0
         :else        no-call-reward)))))

(defn- norm-args
  "String-normalize an argument map for order/type-insensitive comparison."
  [m]
  (into {} (map (fn [[k v]] [(str k) (str v)])) m))

(defn- administered-calls [completion-msg]
  (mapv (fn [{:keys [name arguments]}]
          {:name name
           :args (norm-args (js->clj (js/JSON.parse (or arguments "{}"))))})
        (:toolCalls completion-msg)))

(defn match-administered-reward
  "The provenance-seam reward: score a REGENERATED completion against the
   action the child actually took in the administered session (recovered
   via session-grpo/prompt-meta — no side channel). Additive components,
   default weights {:parse 0.25 :names 0.35 :args 0.4}:
     :parse — the completion parses cleanly (no malformed blocks)
     :names — the multiset of called tool names matches the administered
     :args  — the full {name args} set matches (string-normalized)
   A call-free administered turn awards :names/:args to call-free regens.
   Parse errors or a prompt without provenance metadata -> :reward-floor
   (default -1.0). Everything finite by construction."
  ([] (match-administered-reward {}))
  ([{:keys [reward-floor weights]
     :or {reward-floor -1.0
          weights {:parse 0.25 :names 0.35 :args 0.4}}}]
   (fn [prompt completion]
     (let [prov (sg/prompt-meta prompt)]
       (if-not prov
         reward-floor
         (let [admin (administered-calls (:completion prov))
               {:keys [calls errors]} (tc/parse-tool-calls (str completion))
               regen (mapv (fn [{:keys [name args]}]
                             {:name name :args (norm-args args)})
                           calls)]
           (if (seq errors)
             reward-floor
             (+ (:parse weights)
                (if (= (frequencies (map :name admin))
                       (frequencies (map :name regen)))
                  (:names weights) 0.0)
                (if (= (set admin) (set regen))
                  (:args weights) 0.0)))))))))

(defn resolve-reward
  "Resolve a reward SPEC into a PROMISE of a reward-fn (plugin loading is
   an I/O boundary):
     \"tool-format\"         -> tool-format-reward over ctx :toolset
     \"match-administered\"  -> match-administered-reward
     anything else           -> a plugin FILE path, loaded via
                                nbb.core/load-file; its LAST FORM must be
                                (fn [ctx] -> reward-fn) — the
                                external-oracle contract (ns docstring).
   ctx = {:points :toolset :opts}."
  [spec {:keys [toolset opts] :as ctx}]
  (case spec
    "tool-format"        (p/resolved (tool-format-reward toolset (or opts {})))
    "match-administered" (p/resolved (match-administered-reward (or opts {})))
    (p/let [make (nbb/load-file spec)]
      (when-not (fn? make)
        (throw (ex-info (str "session-reward: plugin " spec
                             " did not evaluate to a function")
                        {:genmlx/error :bad-reward-plugin :path spec})))
      (let [f (make ctx)]
        (when-not (fn? f)
          (throw (ex-info (str "session-reward: plugin " spec
                               " returned a non-function reward")
                          {:genmlx/error :bad-reward-plugin :path spec})))
        f))))
