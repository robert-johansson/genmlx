(ns genmlx-tool-worker
  "L5-A child worker (genmlx-wdx0): GenMLX ops behind the pi tool bridge.
   Single-shot protocol — one JSON request on STDIN, exactly one marker
   line on stdout:

     GENMLX_RESULT:{...}

   (nbb / addon startup noise safely precedes the marker; the extension
   parses the LAST marker line.) Every failure is {ok:false, error} —
   this process never exits nonzero for a WELL-FORMED op failure, only
   for protocol-level breakage.

   Ops (genmlx.llm.msa-score is the engine):
     {op \"eval-model\",  code}                            -> validity + schema
     {op \"score-model\", code, observations, nParticles?} -> log-ML + method
     {op \"rank-models\", candidates, observations, nParticles?} -> ranking

   JSON cannot carry ##-Inf, so logMl is null when non-finite plus a
   `finite` flag — score-model*'s ##-Inf error channel preserved
   honestly. Observation keys arrive as JSON strings -> keyword addrs.

   SIGTRAP discipline: this worker loads @genmlx/core in ITS OWN process;
   the agent process (either provider) never does both. GPU-light: SCI
   eval + tiny scalar graphs; no LLM checkpoint ever loads here.

   Run: echo '{\"op\":\"eval-model\",\"code\":\"...\"}' | \\
          bun run --bun nbb scripts/genmlx_tool_worker.cljs"
  (:require [genmlx.llm.msa-score :as score]
            [promesa.core :as p]))

(defn- finite? [x] (and (number? x) (js/isFinite x)))

(defn- schema-summary [gf]
  (let [s (:schema gf)]
    {:traceSites (mapv (fn [site]
                         {:addr (str (:addr site))
                          :distType (some-> (:dist-type site) name)})
                       (:trace-sites s))
     :static (boolean (:static? s))
     :conjugate (boolean (seq (:conjugate-pairs s)))}))

(defn- ->observations [obs-js]
  (when obs-js
    (into {}
          (map (fn [k] [(keyword k) (unchecked-get obs-js k)]))
          (js-keys obs-js))))

(defn- eval-op [req]
  (let [code (.-code req)]
    (if-not (string? code)
      {:ok false :error "eval-model: string `code` required"}
      (if-let [gf (score/eval-model code)]
        {:ok true :valid true :schema (schema-summary gf)}
        {:ok true :valid false
         :error "code did not evaluate to a (fn [trace] ...) model"}))))

(defn- score-one [code observations n-particles]
  (let [gf (score/eval-model code)]
    (if-not gf
      {:valid false :logMl nil :finite false :method nil}
      (let [{:keys [log-ml method]}
            (score/score-model* gf observations {:n-particles n-particles})]
        {:valid true
         :logMl (when (finite? log-ml) log-ml)
         :finite (finite? log-ml)
         :method (some-> method name)}))))

(defn- score-op [req]
  (let [code (.-code req)
        obs  (->observations (.-observations req))
        n    (or (.-nParticles req) 50)]
    (cond
      (not (string? code)) {:ok false :error "score-model: string `code` required"}
      (empty? obs)         {:ok false :error "score-model: non-empty `observations` required"}
      :else (assoc (score-one code obs n) :ok true))))

(defn- rank-op [req]
  (let [cands (.-candidates req)
        obs   (->observations (.-observations req))
        n     (or (.-nParticles req) 50)]
    (cond
      (or (nil? cands) (zero? (.-length cands)))
      {:ok false :error "rank-models: non-empty `candidates` required"}
      (empty? obs)
      {:ok false :error "rank-models: non-empty `observations` required"}
      :else
      {:ok true
       :ranking
       (->> (map-indexed (fn [i code]
                           (assoc (if (string? code)
                                    (score-one code obs n)
                                    {:valid false :logMl nil :finite false
                                     :method nil})
                                  :index i))
                         (array-seq cands))
            ;; best first; invalid / -Inf last
            (sort-by (fn [r] (- (if (:finite r) (:logMl r) ##-Inf))))
            vec)})))

(defn- dispatch [req]
  (case (.-op req)
    "eval-model"  (eval-op req)
    "score-model" (score-op req)
    "rank-models" (rank-op req)
    {:ok false :error (str "unknown op " (pr-str (.-op req))
                           " — supported: eval-model, score-model, rank-models")}))

(defn- respond! [m]
  (println (str "GENMLX_RESULT:" (js/JSON.stringify (clj->js m)))))

(-> (p/create
     (fn [resolve _]
       (let [buf (volatile! "")]
         (.setEncoding js/process.stdin "utf8")
         (.on js/process.stdin "data" (fn [chunk] (vswap! buf str chunk)))
         (.on js/process.stdin "end" (fn [] (resolve @buf))))))
    (p/then
     (fn [text]
       (let [req (try (js/JSON.parse text)
                      (catch :default e
                        {:parse-error (ex-message e)}))]
         (if (:parse-error req)
           (respond! {:ok false :error (str "malformed request JSON: "
                                            (:parse-error req))})
           (respond! (try (dispatch req)
                          (catch :default e
                            {:ok false :error (str (or (ex-message e) e))})))))))
    (p/catch
     (fn [e]
       (respond! {:ok false :error (str (or (ex-message e) e))}))))
