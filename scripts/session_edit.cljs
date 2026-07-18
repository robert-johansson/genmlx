(ns session-edit
  "L3-C2 editing instrument (genmlx-5v23): resample an administered pi
   session turn from a chosen boundary and write the counterfactual back
   into pi's session tree ('step 3 was wrong, resample from there').

   Env:
     SESSION       path to a pi session .jsonl (required)
     TURN          messages index of the assistant turn to edit (required;
                   the :index column scripts/session_scores.cljs prints)
     BOUNDARY      \"tool-call:J\" | \"token:I\" | unset = whole turn
     TEMPERATURE   sampling temperature (default 0 = greedy)
     MAX_NEW       max resampled tokens (default 256)
     MODEL_DIR     checkpoint dir (default the 0.8b smoke model; pass the
                   administering model for real counterfactuals)
     SYSTEM_PROMPT the deployed system prompt (optional)
     OUT_DIR       directory for the forked file (default: the source's
                   dir — pass a scratch dir to keep smoke forks out of
                   the live agent session list)
     IN_PLACE=1    append the branch to the source file instead of forking
     EDIT_OUT      write a JSON report here (optional; NOT `OUT` — the
                   guard script owns that name)

   Default mode writes a FORKED session file next to the source (header
   parentSession = source); the administered artifact is never mutated.
   The fork loads in pi (--resume it) with the edit as the live leaf.

   Run (from repo root, guarded on Thor):
     SESSION=... TURN=2 BOUNDARY=tool-call:0 \\
       scripts/guarded-run.sh session-edit \\
       bunx --bun nbb@1.4.208 scripts/session_edit.cljs"
  (:require [clojure.string :as str]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.pi-edit :as pe]
            [genmlx.llm.pi-session :as ps]
            [promesa.core :as pr]
            ["fs" :as fs]
            ["os" :as os]
            ["path" :as path]))

(def session-file (.. js/process -env -SESSION))
(def turn (some-> (.. js/process -env -TURN) js/parseInt))
(def boundary
  (when-let [b (.. js/process -env -BOUNDARY)]
    (let [[kind n] (str/split b #":")]
      (case kind
        "tool-call" {:tool-call (js/parseInt n)}
        "token"     {:token (js/parseInt n)}
        (do (println "unknown BOUNDARY" (pr-str b)
                     "— use tool-call:J | token:I")
            (js/process.exit 2))))))
(def model-dir (or (.. js/process -env -MODEL_DIR)
                   (path/join (os/homedir) ".cache" "models"
                              "qwen3.5-0.8b-mlx-bf16")))
(def opts
  (cond-> {:system-prompt (.. js/process -env -SYSTEM_PROMPT)
           :max-new (or (some-> (.. js/process -env -MAX_NEW) js/parseInt) 256)}
    (.. js/process -env -TEMPERATURE)
    (assoc :temperature (js/parseFloat (.. js/process -env -TEMPERATURE)))
    (.. js/process -env -OUT_DIR)
    (assoc :out-dir (.. js/process -env -OUT_DIR))
    (= "1" (.. js/process -env -IN_PLACE))
    (assoc :in-place? true)))
(def out-file (.. js/process -env -EDIT_OUT))

(when (or (not session-file) (nil? turn) (js/isNaN turn))
  (println "SESSION and TURN env vars required")
  (js/process.exit 2))

(defn- fmt [x] (.toFixed x 3))

(-> (pr/let [_ (do (println "session: " session-file)
                   (println "turn:    " turn "| boundary:"
                            (or (pr-str boundary) "whole turn"))
                   (println "model:   " model-dir)
                   nil)
             mm (llm/load-model model-dir {:cljs-forward? true})
             messages (pr/resolved
                       (ps/path->messages
                        (ps/leaf-path (ps/read-session session-file))
                        {:system-prompt (:system-prompt opts)}))
             result (pe/edit-session! mm session-file turn boundary opts)]
      (let [t (:turn result)]
        (println "\n-- original turn --")
        (println (pr-str (:content (nth messages turn))))
        (println "\n-- edited turn (" (:finish-reason t) ","
                 (count (:kept-tokens t)) "kept +"
                 (count (:sampled-tokens t)) "sampled, suffix"
                 (fmt (:suffix-logprob t)) "nats ) --")
        (println (pr-str (:text t)))
        (when (seq (:tool-calls t))
          (println "tool calls:" (pr-str (:tool-calls t))))
        (println "\nwrote branch to:" (:file result)
                 "\nleaf:" (:leaf-id result))
        (when out-file
          (fs/writeFileSync
           out-file
           (js/JSON.stringify
            (clj->js {:source session-file
                      :edited-file (:file result)
                      :leaf-id (:leaf-id result)
                      :turn turn
                      :boundary boundary
                      :boundary-token (:boundary-token t)
                      :kept (count (:kept-tokens t))
                      :sampled (count (:sampled-tokens t))
                      :suffix-logprob (:suffix-logprob t)
                      :finish-reason (:finish-reason t)
                      :text (:text t)
                      :original (:content (nth messages turn))})
            nil 2))
          (println "wrote" out-file))))
    (pr/catch (fn [e]
                (println "ERROR:" (or (ex-message e) (str e)))
                (when-let [d (ex-data e)]
                  (println "  " (pr-str (dissoc d :sci.impl/callstack))))
                (js/process.exit 1))))
