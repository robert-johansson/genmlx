(ns session-scores
  "L3-C1 measurement instrument (genmlx-opwh): per-turn assess on an
   administered pi session — 'what probability did the model assign to the
   move it actually made'. Replays the session JSONL through the same
   renderer the provider used and scores every assistant turn in one
   owned-branch walk (backend/forward-branch-scores).

   Env:
     SESSION       path to a pi session .jsonl (required)
     MODEL_DIR     checkpoint dir (default ~/.cache/models/qwen3.5-0.8b-mlx-bf16
                   — a SMOKE default; for administration parity pass the
                   model that ran the session, e.g. the ornith 35b)
     SYSTEM_PROMPT the deployed system prompt (optional; scores are
                   conditioned differently without it — reported)
     PER_TOKEN=1   include per-token logprobs in the JSON output
     SCORES_OUT    write a JSON report here (optional; NOT `OUT` — the
                   guard script owns that name and would swallow it)

   Image-bearing sessions are refused (VLM replay = follow-up).

   Run (from repo root, guarded on Thor):
     SESSION=~/.mlx-node/agent/sessions/<dir>/<file>.jsonl \\
       scripts/guarded-run.sh session-scores \\
       bunx --bun nbb@1.4.208 scripts/session_scores.cljs"
  (:require [genmlx.llm.backend :as llm]
            [genmlx.llm.pi-assess :as pa]
            [genmlx.llm.pi-session :as ps]
            [promesa.core :as pr]
            ["fs" :as fs]
            ["os" :as os]
            ["path" :as path]))

(def session-file (.. js/process -env -SESSION))
(def model-dir (or (.. js/process -env -MODEL_DIR)
                   (path/join (os/homedir) ".cache" "models"
                              "qwen3.5-0.8b-mlx-bf16")))
(def system-prompt (.. js/process -env -SYSTEM_PROMPT))
(def per-token? (= "1" (.. js/process -env -PER_TOKEN)))
(def out-file (.. js/process -env -SCORES_OUT))

(when-not session-file
  (println "SESSION env var required (path to a pi session .jsonl)")
  (js/process.exit 2))

(defn- fmt [x] (.toFixed x 3))

(-> (pr/let [_ (do (println "session:" session-file)
                   (println "model:  " model-dir)
                   (when-not system-prompt
                     (println "note:    no SYSTEM_PROMPT — scores are"
                              "conditioned on a system-free render"))
                   nil)
             mm (llm/load-model model-dir {:cljs-forward? true})
             session (pr/resolved (ps/read-session session-file))
             msgs (pr/resolved
                   (ps/path->messages (ps/leaf-path session)
                                      {:system-prompt system-prompt}))
             scores (pa/session-scores mm msgs {})
             texts (pr/all (mapv #(llm/decode
                                   (:tokenizer mm)
                                   (js/Uint32Array.from (into-array (:tokens %))))
                                 scores))]
      (println "\nsession" (get-in session [:header :id])
               "—" (count msgs) "messages,"
               (count scores) "assistant turns\n")
      (println (str "turn  tokens  cached  logprob    lp/tok  parity  text"))
      (doseq [[s text] (map vector scores texts)]
        (println (str (.padEnd (str "  " (:index s)) 6)
                      (.padEnd (str (:n-tokens s)) 8)
                      (.padEnd (str (:cached s)) 8)
                      (.padEnd (fmt (:logprob s)) 11)
                      (.padEnd (fmt (/ (:logprob s) (max 1 (:n-tokens s)))) 8)
                      (.padEnd (str (:parity? s)) 8)
                      (pr-str (subs text 0 (min 48 (count text)))))))
      (let [total (reduce + 0.0 (map :logprob scores))
            n     (reduce + 0 (map :n-tokens scores))]
        (println (str "\ntotal: " (fmt total) " nats over " n
                      " action tokens (" (fmt (/ total (max 1 n))) "/token)"))
        (when out-file
          (fs/writeFileSync
           out-file
           (js/JSON.stringify
            (clj->js {:session (get-in session [:header :id])
                      :file session-file
                      :model model-dir
                      :system-prompt? (boolean system-prompt)
                      :total-logprob total
                      :action-tokens n
                      :turns (mapv (fn [s text]
                                     (cond-> {:index (:index s)
                                              :logprob (:logprob s)
                                              :n-tokens (:n-tokens s)
                                              :cached (:cached s)
                                              :parity (:parity? s)
                                              :text text}
                                       per-token?
                                       (assoc :per-token (:per-token s))))
                                   scores texts)})
            nil 2))
          (println "wrote" out-file))))
    (pr/catch (fn [e]
                (println "ERROR:" (or (ex-message e) (str e)))
                (when-let [d (ex-data e)]
                  (println "  " (pr-str (dissoc d :sci.impl/callstack))))
                (js/process.exit 1))))
