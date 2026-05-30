(ns examples.vlm-grid-synthesis
  "Stages 3–5 of the maze pipeline: synthesize, recognize, solve.

     Stage 1 (vision):     image → labels             [examples/vlm_grid_gf.cljs]
     Stage 2 (gfi):        labels → trace + score     [examples/vlm_grid_gf.cljs]
     Stage 3 (synthesis):  labels → ClojureScript     [HERE]
     Stage 4 (recognizer): code → finite-tabular MDP  [HERE]
     Stage 5 (solver):     MDP → policy + value       [HERE]

   This is the structural-recognizer pattern from the linear-regression case
   (`dev/docs/INVESTIGATION_STRUCTURAL_RECOGNIZER.md`) applied to MDPs:

     - probe the LLM-synthesized `move` function over every (state, action)
       pair to materialize the deterministic transition tensor T;
     - read the reward map directly from the synthesized data;
     - mark named-reward cells as absorbing (terminal in the agentmodels
       sense — once you reach West/Hill/East the episode ends);
     - run value iteration on the resulting finite tabular MDP.

   The trick is the same one that worked for linear regression: don't trust
   the LLM's coordinates; trust its program structure. We never inspect the
   AST after parsing — we just *run* `move` with controlled inputs.

   Run: bun run --bun nbb examples/vlm_grid_synthesis.cljs"
  (:require [genmlx.llm.backend :as llm]
            [edamame.core :as eda]
            [sci.core :as sci]
            [clojure.string :as str]
            [promesa.core :as pr]))

(def model-path
  (str (.-HOME js/process.env) "/.cache/models/Qwen3.6-35B-A3B-4bit"))

(def system-prompt
  "You are a probabilistic programming assistant. Output only valid ClojureScript code, no commentary.")

;; ---------------------------------------------------------------------------
;; The labels from stage 1+2 (verified 25/25 against ground truth)
;; ---------------------------------------------------------------------------

(def labels
  [["empty" "empty" "empty" "empty" "empty"]
   ["empty" "wall" "empty" "wall" "empty"]
   ["west" "empty" "hill" "empty" "east"]
   ["empty" "wall" "empty" "wall" "empty"]
   ["empty" "empty" "agent" "empty" "empty"]])

;; ---------------------------------------------------------------------------
;; Reward convention encoded in the prompt: West=+5, Hill=-10, East=+10
;; ---------------------------------------------------------------------------

(def reward-table {"west" 5 "hill" -10 "east" 10})

(defn format-grid
  "Render the labels as a compact ASCII table for the prompt."
  [grid]
  (let [w 7
        pad (fn [s] (str s (apply str (repeat (max 0 (- w (count s))) " "))))]
    (str/join "\n"
              (for [r (range (count grid))]
                (str "  ROW " r ": "
                     (str/join " | " (map pad (nth grid r))))))))

(defn build-prompt [grid]
  (str
   "You are given a 5x5 gridworld maze, classified cell-by-cell. The labels are:\n\n"
   (format-grid grid) "\n\n"
   "Reward convention:\n"
   "  west cell: +5  (small reward)\n"
   "  hill cell: -10 (penalty — agent should avoid)\n"
   "  east cell: +10 (large reward — best goal)\n"
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
   "Output ONLY the three forms — no comments, no namespace, no markdown."))

;; ---------------------------------------------------------------------------
;; Code extraction & parsing (same helpers as qwen36_structural)
;; ---------------------------------------------------------------------------

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

;; ---------------------------------------------------------------------------
;; Sandbox eval — vars (def, defn) + standard core, no IO
;; ---------------------------------------------------------------------------

(defn evaluate-program
  "Run the synthesized code in a sandboxed SCI env and return a map of the
   bound vars. Appends a trailing return-expression so a single eval-string
   call yields the values in one shot."
  [code]
  (let [augmented (str code "\n\n{:maze maze :actions actions :move move}")]
    (sci/eval-string augmented {})))

;; ---------------------------------------------------------------------------
;; Round-trip: rebuild labels grid from the evaluated maze data
;; ---------------------------------------------------------------------------

(defn maze->labels [{:keys [rows cols walls rewards start]}]
  (let [walls-set (set (map vec walls))
        rewards-map (into {} (map (fn [[k v]] [(vec k) v]) rewards))
        ;; rebuild reverse lookup of reward → label name
        reward->name (into {} (map (fn [[k v]] [v k])) reward-table)
        start-vec (vec start)]
    (vec
     (for [r (range rows)]
       (vec
        (for [c (range cols)]
          (let [pos [r c]]
            (cond
              (= pos start-vec) "agent"
              (contains? walls-set pos) "wall"
              (contains? rewards-map pos) (or (reward->name (rewards-map pos))
                                              (str "?reward=" (rewards-map pos)))
              :else "empty"))))))))

(defn count-mismatches [a b]
  (count (for [r (range (count a))
               c (range (count (first a)))
               :when (not= (get-in a [r c]) (get-in b [r c]))]
           1)))

(defn pad [s n]
  (let [s (str s)
        diff (- n (count s))]
    (if (pos? diff) (str s (apply str (repeat diff " "))) s)))

(defn print-grid [title grid]
  (println (str "--- " title " ---"))
  (doseq [row grid]
    (println (str "  " (str/join " | " (map #(pad % 7) row)))))
  (println))

(defn now-ms [] (.now js/performance))
(defn fmt-ms [t0] (str (.toFixed (- (now-ms) t0) 0) "ms"))

;; ---------------------------------------------------------------------------
;; Stage 4: structural recognizer — probe the synthesized program as an MDP.
;; ---------------------------------------------------------------------------

(defn recognize-mdp
  "Probe the synthesized {:maze :actions :move} to extract a finite-tabular
   MDP: state space (non-wall cells), deterministic transition tensor T, and
   reward tensor R. Named-reward cells are marked absorbing (terminal — once
   you reach West/Hill/East the episode ends, matching agentmodels semantics).

   Returns {:n-states :n-actions :states :state->idx :T :R :absorbing
            :step-cost :rows :cols :walls :rewards-map :actions :start}."
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
      :T T :R R
      :absorbing absorbing
      :step-cost step-cost
      :rows rows :cols cols
      :walls walls-set
      :rewards-map rewards-map
      :actions (vec actions)
      :start (vec start)})))

;; ---------------------------------------------------------------------------
;; Stage 5: synchronous value iteration over the recognized MDP.
;; ---------------------------------------------------------------------------

(defn value-iterate
  "Run value iteration to convergence. Returns {:V :Q :policy :iters :delta}."
  ([mdp] (value-iterate mdp {}))
  ([{:keys [T R n-states n-actions]} {:keys [gamma tol max-iters]
                                      :or {gamma 0.95 tol 1.0e-9 max-iters 2000}}]
   (loop [V (vec (repeat n-states 0.0))
          iter 0]
     (let [V-new (vec (for [s (range n-states)]
                        (apply max
                               (for [a (range n-actions)]
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

;; ---------------------------------------------------------------------------
;; Pretty-print policy + value
;; ---------------------------------------------------------------------------

(defn arrow-for [action]
  (case action :up "↑" :down "↓" :left "←" :right "→" "?"))

(defn print-policy [{:keys [rows cols state->idx walls actions absorbing rewards-map]}
                    policy]
  (println "--- Optimal policy (G=goal-positive, X=goal-negative, █=wall) ---")
  (doseq [r (range rows)]
    (println
     (str "  "
          (str/join "   "
                    (for [c (range cols)]
                      (let [pos [r c]]
                        (cond
                          (walls pos) "█"
                          (absorbing pos)
                          (let [v (get rewards-map pos)]
                            (cond (pos? v) "G"
                                  (neg? v) "X"
                                  :else "·"))
                          :else
                          (if-let [s-idx (state->idx pos)]
                            (arrow-for (nth actions (nth policy s-idx)))
                            "?"))))))))
  (println))

(defn pad-num [v]
  (let [s (.toFixed v 2)]
    (if (= "-" (subs s 0 1)) s (str " " s))))

(defn print-value [{:keys [rows cols state->idx walls]} V]
  (println "--- Value function V(s) ---")
  (doseq [r (range rows)]
    (println
     (str "  "
          (str/join " "
                    (for [c (range cols)]
                      (let [pos [r c]]
                        (cond
                          (walls pos) "  ##  "
                          :else
                          (if-let [s-idx (state->idx pos)]
                            (pad-num (nth V s-idx))
                            " ?? "))))))))
  (println))

;; ---------------------------------------------------------------------------
;; Drive
;; ---------------------------------------------------------------------------

(defn run! []
  (let [t0 (now-ms)]
    (print-grid "Input labels (from VLM)" labels)

    (println "[1] Loading model (text-only generation, no image this turn)...")
    (pr/let [m (llm/load-model model-path)]
      (println (str "    type=" (:type m) "  load=" (fmt-ms t0)))

      (let [t1 (now-ms)
            prompt (build-prompt labels)]
        (println "\n[2] Synthesizing maze code...")
        (pr/let [text (llm/generate-text-raw
                       m prompt
                       {:max-tokens 600 :temperature 0 :system-prompt system-prompt})]
          (println (str "    gen=" (fmt-ms t1)))
          (let [code (extract-code text)]
            (println "\n--- Synthesized code ---")
            (doseq [line (str/split code #"\n")] (println (str "  │ " line)))
            (println)

            (let [{:keys [ok? err]} (parse-cljs code)]
              (println (str "[3] Parse: " (if ok? "✓" (str "✗ " err)))))

            (println "[4] Evaluate in SCI sandbox...")
            (let [{:keys [maze actions move]}
                  (try (evaluate-program code)
                       (catch :default e
                         (println (str "    eval failed: " (.-message e)))
                         {:maze nil}))]
              (when maze
                (println (str "    maze keys: " (vec (keys maze))))
                (println (str "    walls:     " (sort (:walls maze))))
                (println (str "    rewards:   " (into (sorted-map-by compare) (:rewards maze))))
                (println (str "    start:     " (:start maze)))
                (println (str "    actions:   " actions))

                (println "\n[5] Round-trip: reconstruct labels from maze data...")
                (let [reconstructed (maze->labels maze)
                      n-mismatch (count-mismatches reconstructed labels)]
                  (print-grid "Reconstructed from synthesized code" reconstructed)
                  (println (str "    mismatches vs input: " n-mismatch "/25"))
                  (println (if (zero? n-mismatch)
                             "    ✓ Round trip clean — synthesis preserves the maze"
                             "    ✗ Synthesis introduced errors — see grid above"))

                  (println "\n[6] Smoke-test the move function...")
                  (try
                    (let [start (vec (:start maze))
                          test-cases [[start :up]
                                      [start :down]
                                      [start :left]
                                      [start :right]
                                      [[1 0] :right] ; into a wall at (1,1)
                                      [[0 0] :up] ; out of bounds
                                      [[0 4] :right]]] ; out of bounds
                      (doseq [[pos a] test-cases]
                        (let [next-pos (move pos a)]
                          (println (str "    move " pos " " a " -> " next-pos)))))
                    (catch :default e
                      (println (str "    move failed: " (.-message e))))))

                (println "\n[7] Stage 4 — recognize as finite-tabular MDP...")
                (let [t-rec (now-ms)
                      mdp (recognize-mdp {:maze maze :actions actions :move move})
                      _ (println (str "    states=" (:n-states mdp)
                                      "  actions=" (:n-actions mdp)
                                      "  absorbing=" (count (:absorbing mdp))
                                      "  step-cost=" (:step-cost mdp)
                                      "  recognize=" (fmt-ms t-rec)))

                      _ (println "\n[8] Stage 5 — value iteration (γ=0.95, tol=1e-9)...")
                      t-vi (now-ms)
                      {:keys [V policy iters delta]} (value-iterate mdp)
                      _ (println (str "    converged in " iters " iters"
                                      "  Δfinal=" (.toExponential delta 2)
                                      "  vi=" (fmt-ms t-vi)))]
                  (println)
                  (print-value mdp V)
                  (print-policy mdp policy)

                  (let [start-pos (:start mdp)
                        s0 ((:state->idx mdp) start-pos)
                        E0 (nth V s0)]
                    (println (str "[9] Expected return from start " start-pos
                                  ": V*(s0) = " (.toFixed E0 4)))

                    ;; Roll out the optimal greedy trajectory
                    (println "\n[10] Greedy rollout from start (max 30 steps):")
                    (loop [pos start-pos
                           steps []
                           total-r 0.0
                           n 0]
                      (let [s-idx ((:state->idx mdp) pos)
                            absorbed? ((:absorbing mdp) pos)]
                        (cond
                          absorbed?
                          (do (println (str "    "
                                            (str/join " → " (conj steps (str pos " [absorbed]")))))
                              (println (str "    total reward: " (.toFixed total-r 4)
                                            "  steps: " n)))
                          (or (>= n 30) (nil? s-idx))
                          (println (str "    truncated at " pos " after " n " steps"))
                          :else
                          (let [a-idx (nth policy s-idx)
                                a (nth (:actions mdp) a-idx)
                                next-pos (vec (move pos a))
                                r (get-in (:R mdp) [s-idx a-idx])]
                            (recur next-pos
                                   (conj steps (str pos " " (arrow-for a)))
                                   (+ total-r r)
                                   (inc n))))))))))

            (println (str "\nTotal: " (fmt-ms t0)))))))))

(run!)
