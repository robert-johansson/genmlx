(ns examples.vlm-grid-battery
  "Generalization battery for the vision → synthesis → recognize → solve pipeline.

   Tests four maze variants that stress different layers:
     A_rotated_hiking   — same shape as the demo, named cells + agent rotated
                          (does synthesis depend on positions or structure?)
     B_cheap_rewards    — same layout, different reward values
                          (does the LLM read prompted values or memorize?)
     C_small_topology   — 4×4, two interior walls instead of cross
                          (does the recognizer handle different topologies?)
     D_agent_top_right  — same hiking layout, agent at (0,4)
                          (does the solver re-route from arbitrary starts?)

   For each variant the script measures:
     vision-acc:    cells correctly classified vs ground truth
     parse:         did synthesized code parse?
     eval:          did the SCI sandbox accept it?
     round-trip:    do reconstructed labels match input labels?
     recognized:    did the recognizer materialize a non-trivial MDP?
     solved:        did value iteration converge to a finite V*(start)?
     policy-sane:   does the greedy rollout end on the highest-reward goal?

   Prereq:  python3 dev/render_gridworld_battery.py

   Run: bun run --bun nbb examples/vlm_grid_battery.cljs"
  (:require [genmlx.llm.vision :as vision]
            ["@mlx-node/lm" :as mlx-lm]
            ["node:fs/promises" :as fs]
            [edamame.core :as eda]
            [sci.core :as sci]
            [clojure.string :as str]
            [promesa.core :as pr]))

(def model-path
  (str (.-HOME js/process.env) "/.cache/models/Qwen3.6-35B-A3B-4bit"))

(def battery-base
  (str (.-HOME js/process.env) "/code/genmlx/dev/grid_battery"))

(def variants
  ["A_rotated_hiking"
   "B_cheap_rewards"
   "C_small_topology"
   "D_agent_top_right"])

(def cell-types
  [{:label "empty" :description "a blank white cell with thin gray borders"}
   {:label "wall"  :description "a solid dark/black cell"}
   {:label "agent" :description "a blue circle with the letter \"A\""}
   {:label "west"  :description "an orange cell with text \"West\""}
   {:label "hill"  :description "a yellow cell with text \"Hill\""}
   {:label "east"  :description "a green cell with text \"East\""}])

(def system-prompt
  "You are a probabilistic programming assistant. Output only valid ClojureScript code, no commentary.")

;; ---------------------------------------------------------------------------
;; Spec loading
;; ---------------------------------------------------------------------------

(defn load-spec
  "Read the variant's spec.json and return {:rows :cols :truth :rewards :start}."
  [variant]
  (pr/let [raw (fs/readFile (str battery-base "/" variant "/spec.json") "utf-8")
           j (js->clj (.parse js/JSON raw) :keywordize-keys true)]
    {:rows    (get-in j [:spec :rows])
     :cols    (get-in j [:spec :cols])
     :truth   (mapv (fn [row] (mapv #(if (string? %) (str/lower-case %) %) row))
                    (:truth_labels j))
     :rewards (into {} (map (fn [[k v]] [(str/lower-case (name k)) v])
                            (get-in j [:spec :rewards])))
     :start   (get-in j [:spec :agent])}))

(defn read-cell [variant]
  (fn [r c]
    (pr/let [buf (fs/readFile (str battery-base "/" variant
                                   "/cells/cell_r" r "_c" c ".png"))]
      (js/Uint8Array. (.-buffer buf) (.-byteOffset buf) (.-byteLength buf)))))

;; ---------------------------------------------------------------------------
;; Vision: classify all cells (uses an existing ChatSession we hand in)
;; ---------------------------------------------------------------------------

(defn classify-variant [session variant rows cols]
  (vision/classify-grid session rows cols (read-cell variant) cell-types))

(defn count-mismatches [a b]
  (count (for [r (range (count a))
               c (range (count (first a)))
               :when (not= (get-in a [r c]) (get-in b [r c]))]
           1)))

;; ---------------------------------------------------------------------------
;; Synthesis: parametric prompt
;; ---------------------------------------------------------------------------

(defn format-grid [grid]
  (let [w 7
        pad (fn [s] (str s (apply str (repeat (max 0 (- w (count s))) " "))))]
    (str/join "\n"
              (for [r (range (count grid))]
                (str "  ROW " r ": "
                     (str/join " | " (map pad (nth grid r))))))))

(defn build-prompt [labels rewards]
  (let [present (set (flatten labels))
        rew-lines (str/join "\n"
                            (for [[label v] rewards
                                  :when (contains? present (str/lower-case (name label)))]
                              (str "  " (name label) " cell: " (if (pos? v) "+" "") v)))
        rows (count labels)
        cols (count (first labels))]
    (str
     "You are given a " rows "x" cols " gridworld maze, classified cell-by-cell. The labels are:\n\n"
     (format-grid labels) "\n\n"
     "Reward convention:\n" rew-lines "\n"
     "  walls cannot be entered.\n"
     "  empty cells: -0.1 step cost.\n\n"
     "Write a self-contained GenMLX program with EXACTLY these top-level forms in this order:\n\n"
     "1. (def maze {...}) — a map with these keys:\n"
     "     :rows         the number of rows (an integer)\n"
     "     :cols         the number of columns (an integer)\n"
     "     :walls        a SET of [row col] vectors for wall positions\n"
     "     :rewards      a MAP of [row col] -> reward number for the reward cells\n"
     "     :start        the [row col] vector for the agent's starting cell\n\n"
     "2. (def actions [:up :down :left :right])\n\n"
     "3. (defn move [pos action] ...) — given a position and an action keyword,\n"
     "   return the new position. Out-of-bounds or wall moves leave pos unchanged.\n\n"
     "Use only standard ClojureScript core (no genmlx imports needed for this stage).\n"
     "Use coordinates [row col]. Row 0 is top, column 0 is left.\n"
     "Output ONLY the three forms — no comments, no namespace, no markdown.")))

(defn synthesize-text
  "Text-only generation through the same ChatSession used for vision.
   Resets first so synthesis doesn't inherit the cell-classification history.
   Returns a promise of the response text."
  [session prompt {:keys [max-tokens]
                    :or {max-tokens 700}}]
  (pr/let [_ (.reset session)
           result (.send session prompt
                          #js {:config #js {:maxNewTokens max-tokens
                                            :temperature 0
                                            :reasoningEffort "none"
                                            :repetitionPenalty 1.0}})]
    (or (.-text result) "")))

(defn extract-code [text]
  (let [t (str/trim (or text ""))]
    (cond
      (str/blank? t) ""
      (re-find #"```" t)
      (let [m (re-find #"```(?:clojure|cljs|clj|clojurescript)?\s*\n?([\s\S]*?)```" t)]
        (if m (str/trim (nth m 1)) t))
      (str/starts-with? t "(") t
      :else (let [i (str/index-of t "(")] (if i (subs t i) "")))))

(defn parse-cljs [code]
  (try {:ok? true :forms (eda/parse-string code {:all true})}
       (catch :default e {:ok? false :err (.-message e)})))

(defn evaluate-program [code]
  (let [augmented (str code "\n\n{:maze maze :actions actions :move move}")]
    (sci/eval-string augmented {})))

;; ---------------------------------------------------------------------------
;; Round-trip: maze data → labels grid (rewards keyed by value, agent by start)
;; ---------------------------------------------------------------------------

(defn maze->labels [{:keys [rows cols walls rewards start]} reward-name-by-value]
  (let [walls-set   (set (map vec walls))
        rewards-map (into {} (map (fn [[k v]] [(vec k) v]) rewards))
        start-vec   (vec start)]
    (vec
     (for [r (range rows)]
       (vec
        (for [c (range cols)]
          (let [pos [r c]]
            (cond
              (= pos start-vec)            "agent"
              (contains? walls-set pos)    "wall"
              (contains? rewards-map pos)
              (or (reward-name-by-value (rewards-map pos))
                  (str "?reward=" (rewards-map pos)))
              :else                         "empty"))))))))

;; ---------------------------------------------------------------------------
;; Recognizer + solver (copied from vlm_grid_synthesis.cljs)
;; ---------------------------------------------------------------------------

(defn recognize-mdp
  ([synth] (recognize-mdp synth -0.1))
  ([{:keys [maze actions move]} step-cost]
   (let [{:keys [rows cols walls rewards start]} maze
         walls-set (set (map vec walls))
         rewards-map (into {} (map (fn [[k v]] [(vec k) v]) rewards))
         all-cells (vec (for [r (range rows) c (range cols)] [r c]))
         states (vec (remove walls-set all-cells))
         state->idx (into {} (map-indexed (fn [i s] [s i])) states)
         absorbing (into #{} (filter rewards-map states))
         n-s (count states)
         n-a (count actions)
         T (vec (for [s states]
                  (vec (for [a actions]
                         (if (absorbing s)
                           (state->idx s)
                           (let [s' (vec (move s a))]
                             (state->idx s' (state->idx s))))))))
         R (vec (for [s states]
                  (vec (for [a actions]
                         (if (absorbing s)
                           0.0
                           (let [s' (vec (move s a))]
                             (+ step-cost (double (get rewards-map s' 0)))))))))]
     {:n-states n-s :n-actions n-a
      :states states :state->idx state->idx
      :T T :R R :absorbing absorbing
      :step-cost step-cost
      :rows rows :cols cols
      :walls walls-set
      :rewards-map rewards-map
      :actions (vec actions)
      :start (vec start)})))

(defn value-iterate
  ([mdp] (value-iterate mdp {}))
  ([{:keys [T R n-states n-actions]} {:keys [gamma tol max-iters]
                                      :or {gamma 0.95 tol 1.0e-9 max-iters 2000}}]
   (loop [V (vec (repeat n-states 0.0))
          iter 0]
     (let [V-new (vec (for [s (range n-states)]
                        (apply max (for [a (range n-actions)]
                                     (let [s' (get-in T [s a])
                                           r (get-in R [s a])]
                                       (+ r (* gamma (V s'))))))))
           delta (apply max (map (fn [a b] (Math/abs (- a b))) V V-new))]
       (if (or (< delta tol) (>= iter max-iters))
         (let [Q (vec (for [s (range n-states)]
                        (vec (for [a (range n-actions)]
                               (let [s' (get-in T [s a])
                                     r (get-in R [s a])]
                                 (+ r (* gamma (V-new s'))))))))
               policy (vec (for [s (range n-states)]
                             (apply max-key #(get-in Q [s %]) (range n-actions))))]
           {:V V-new :Q Q :policy policy :iters iter :delta delta})
         (recur V-new (inc iter)))))))

(defn rollout-greedy
  "Walk greedy policy from start; return {:steps [(pos, action)...] :total-r :final-pos}.
   Stops when reaching an absorbing state or after max-steps."
  [{:keys [state->idx absorbing actions T R move start]} policy & {:keys [max-steps]
                                                                   :or {max-steps 30}}]
  (loop [pos (vec start)
         steps []
         total-r 0.0
         n 0]
    (let [s-idx (state->idx pos)]
      (cond
        (absorbing pos)            {:steps steps :total-r total-r :final-pos pos}
        (or (>= n max-steps)
            (nil? s-idx))          {:steps steps :total-r total-r :final-pos pos
                                    :truncated? true}
        :else
        (let [a-idx (nth policy s-idx)
              a (nth actions a-idx)
              next-pos (vec (move pos a))
              r (get-in R [s-idx a-idx])]
          (recur next-pos (conj steps [pos a]) (+ total-r r) (inc n)))))))

;; ---------------------------------------------------------------------------
;; Per-variant evaluation
;; ---------------------------------------------------------------------------

(defn now-ms [] (.now js/performance))

(defn pad [s n]
  (let [s (str s)
        diff (- n (count s))]
    (if (pos? diff) (str s (apply str (repeat diff " "))) s)))

(defn print-grid [title grid]
  (println (str "  --- " title " ---"))
  (doseq [row grid]
    (println (str "    " (str/join " | " (map #(pad % 7) row))))))

(defn evaluate-variant [session variant]
  (pr/let [t-start (now-ms)
           {:keys [rows cols truth rewards start]} (load-spec variant)
           _ (println (str "\n══════ " variant " (" rows "×" cols ") ══════"))
           _ (print-grid "Truth labels" truth)

           ;; Stage 1: vision (per-cell probe)
           _ (println (str "  [vision] classifying " (* rows cols) " cells..."))
           t-vis (now-ms)
           {:keys [labels]} (classify-variant session variant rows cols)
           vis-acc-mis (count-mismatches labels truth)
           vision-acc (- (* rows cols) vis-acc-mis)
           _ (println (str "    vision: " vision-acc "/" (* rows cols)
                           "  (" (.toFixed (- (now-ms) t-vis) 0) "ms)"))

           ;; Stage 3: synthesis (text-only LLM gen)
           _ (println "  [synthesis] generating GenMLX code...")
           t-syn (now-ms)
           text (synthesize-text session
                                  (str system-prompt "\n\n" (build-prompt labels rewards))
                                  {:max-tokens 700})
           code (extract-code text)
           parsed (parse-cljs code)
           _ (println (str "    synthesis: " (if (:ok? parsed) "parse=✓" "parse=✗")
                           "  (" (.toFixed (- (now-ms) t-syn) 0) "ms)"))]

    (let [eval-result (try (evaluate-program code)
                           (catch :default e
                             (println (str "    eval=✗  err: " (.-message e)))
                             nil))]
      (if (nil? eval-result)
        {:variant variant :vision-acc vision-acc :n-cells (* rows cols)
         :code code :parse? (:ok? parsed) :eval? false}
        (let [{:keys [maze actions move]} eval-result
              ;; Round-trip
              reward-name-by-value (into {} (map (fn [[lbl v]] [v (name lbl)])
                                                  rewards))
              reconstructed (maze->labels maze reward-name-by-value)
              n-rt-mis (count-mismatches reconstructed labels)

              ;; Stage 4-5: recognize + solve
              t-rec (now-ms)
              mdp (recognize-mdp {:maze maze :actions actions :move move})
              {:keys [V policy iters]} (value-iterate mdp)
              rec-ms (- (now-ms) t-rec)

              ;; Sanity: rollout from start, check final cell
              start-vec (vec (:start mdp))
              s0 ((:state->idx mdp) start-vec)
              v-start (when s0 (nth V s0))
              {:keys [final-pos total-r truncated?]}
                 (rollout-greedy (assoc mdp :move move) policy)
              final-cell-reward (get (:rewards-map mdp) final-pos)
              best-reward (when (seq (:rewards-map mdp))
                            (apply max (vals (:rewards-map mdp))))
              policy-sane? (and (not truncated?)
                                (= final-cell-reward best-reward))]

          (println (str "    eval=✓  round-trip=" n-rt-mis "/" (* rows cols)
                        "  recognize+vi=" (.toFixed rec-ms 0) "ms ("
                        iters " iters)"))
          (println (str "    V*(start)=" (.toFixed v-start 4)
                        "  rollout: " start-vec " → " final-pos
                        " (reward=" final-cell-reward ")"
                        "  total=" (.toFixed total-r 4)))
          (println (str "    policy-sane?=" (if policy-sane? "✓" "✗")
                        "  total-time=" (.toFixed (- (now-ms) t-start) 0) "ms"))

          {:variant variant :vision-acc vision-acc :n-cells (* rows cols)
           :code code :parse? (:ok? parsed) :eval? true
           :round-trip-mismatches n-rt-mis
           :recognized? (pos? (:n-states mdp))
           :solved? (number? v-start)
           :policy-sane? policy-sane?
           :v-start v-start
           :total-reward total-r
           :iters iters
           :final-pos final-pos})))))

(defn print-summary [results]
  (println "\n╔══════════════════════════════════════════════════════════════════╗")
  (println "║                       BATTERY SUMMARY                           ║")
  (println "╚══════════════════════════════════════════════════════════════════╝\n")
  (println "  Variant            vision  parse  eval  round-trip  solved  sane")
  (println "  -----              ------  -----  ----  ----------  ------  ----")
  (doseq [{:keys [variant vision-acc n-cells parse? eval?
                  round-trip-mismatches solved? policy-sane?]} results]
    (println (str "  "
                  (pad variant 18)
                  " " (pad (str vision-acc "/" n-cells) 7)
                  " " (if parse? "✓     " "✗     ")
                  " " (if eval? "✓    " "✗    ")
                  " " (if (and round-trip-mismatches (zero? round-trip-mismatches))
                        "✓ 0/n     "
                        (if round-trip-mismatches
                          (str "✗ " round-trip-mismatches "/" n-cells "   ")
                          "—         "))
                  " " (if solved? "✓     " "—     ")
                  " " (if policy-sane? "✓" "✗"))))
  (println)
  (let [n (count results)
        n-pass (count (filter #(and (:eval? %) (zero? (:round-trip-mismatches % 1))
                                    (:policy-sane? %)) results))]
    (println (str "  Net: " n-pass "/" n " variants completely passing"))))

;; ---------------------------------------------------------------------------
;; Drive
;; ---------------------------------------------------------------------------

(defn run! []
  (println "Loading session (single handle for both vision and synthesis)...")
  (pr/let [t0 (now-ms)
           session (.loadSession mlx-lm model-path)
           _ (println (str "  load=" (.toFixed (- (now-ms) t0) 0) "ms"))

           results (reduce (fn [acc-promise variant]
                             (pr/let [acc acc-promise
                                      r (evaluate-variant session variant)]
                               (conj acc r)))
                           (pr/resolved [])
                           variants)]
    (print-summary results)
    (println (str "\n  Total: " (.toFixed (- (now-ms) t0) 0) "ms"))))

(run!)
