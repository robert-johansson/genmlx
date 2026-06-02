(ns agentmodels.gridworld
  "Gridworld MDP environment — a human-readable grid literal compiled to the MLX
   tensors an MDP planner needs: the transition tensor T:[S,A,S'], the reward
   tensor R:[S,A], and the terminal mask term:[S].

   GenMLX-native by design. The discrete *geometry* (which cell is a wall, where
   each action lands) is tiny and is computed in plain, obvious host-side CLJS —
   no `mx/clip` int-bounds dance (that path regressed; see bean genmlx-aonv).
   The heavy *numeric* work (value iteration) is pure MLX and lives in agent.cljs.
   That is the right purity split: host-side combinatorics, GPU-side linear
   algebra. (examples/memo/mdp.cljs is a memo→genmlx conversion consulted only to
   confirm which mx ops exist — not an idiomatic template.)

   Coordinate convention (screen coordinates, fixed once here so renderers never
   need to know it): a grid literal is a vector of rows, TOP row first. For cell
   (x, y) — x = column, y = row from the top — the dense state index is

       s = x + W*y

   and actions move :left -x, :right +x, :up -y (toward the top), :down +y.

   Cell literal vocabulary: `:empty`, `:wall`, or any other keyword names a
   terminal cell (e.g. :A, :B), whose utility is looked up in the :utilities map."
  (:require [genmlx.mlx :as mx]))

;; Action 0..3. y increases DOWNWARD (screen coords), so :up is -y.
(def action-deltas [[-1 0] [1 0] [0 -1] [0 1]])
(def action-kw     [:left :right :up :down])

(defn next-state
  "Deterministic next state from state `idx` under action `a` (0..3). Clamps to
   grid bounds with plain integer min/max, and stays put if the target is a wall."
  [W H walls idx a]
  (let [x (mod idx W), y (quot idx W)
        [dx dy] (action-deltas a)
        nx (max 0 (min (dec W) (+ x dx)))
        ny (max 0 (min (dec H) (+ y dy)))
        n  (+ nx (* W ny))]
    (if (contains? walls n) idx n)))

(defn parse-grid
  "Parse a grid literal into geometry:
   {:W :H :S :walls #{idx} :terminals {idx -> kw}}."
  [grid]
  (let [H (count grid)
        W (count (first grid))
        cells (for [y (range H) x (range W)]
                [(+ x (* W y)) (nth (nth grid y) x)])]
    {:W W :H H :S (* W H)
     :walls     (into #{} (for [[idx c] cells :when (= c :wall)] idx))
     :terminals (into {}  (for [[idx c] cells
                                :when (and (keyword? c) (not (#{:empty :wall} c)))]
                            [idx c]))}))

(defn build-mdp
  "Compile {:grid :utilities :start :gamma} into the MDP tensors.

   `utilities` maps each terminal keyword to its reward and may carry a
   `:timeCost` applied on every (s,a) (agentmodels' time cost). Returns

     {:W :H :S :A
      :T  [S,A,S'] float32 (deterministic one-hot rows)
      :R  [S,A]    float32 (utility(s) + timeCost)
      :term [S]    float32 (1.0 at terminal cells)
      :terminals {idx->kw} :walls #{idx} :start-idx int
      :ns-fn (fn [s a] -> s') :action-kw [...] :gamma}"
  [{:keys [grid utilities start gamma] :or {utilities {} gamma 1.0}}]
  (let [{:keys [W H S walls terminals]} (parse-grid grid)
        A          (count action-deltas)
        time-cost  (get utilities :timeCost 0.0)
        ns-fn      (fn [s a] (next-state W H walls s a))
        ;; geometry (host-side, pure CLJS) -> [S,A] table of next-state indices
        ns-rows    (vec (for [s (range S)] (vec (for [a (range A)] (ns-fn s a)))))
        ns-table   (mx/array (clj->js ns-rows) mx/int32)               ; [S,A]
        ;; T[s,a,s'] = 1 iff s' == ns-table[s,a]  (one-hot via broadcasting)
        sp         (mx/reshape (mx/astype (mx/arange S) mx/int32) #js [1 1 S])
        nsx        (mx/reshape ns-table #js [S A 1])
        T          (mx/astype (mx/eq? sp nsx) mx/float32)              ; [S,A,S']
        util       (fn [s] (+ (get utilities (get terminals s) 0.0) time-cost))
        R          (mx/array (clj->js (vec (for [s (range S)] (vec (repeat A (util s))))))
                             mx/float32)                              ; [S,A]
        term       (mx/array (clj->js (vec (for [s (range S)]
                                             (if (contains? terminals s) 1.0 0.0))))
                             mx/float32)                              ; [S]
        [sx sy]    (or start [0 0])
        start-idx  (+ sx (* W sy))]
    (mx/eval! T R term)
    {:W W :H H :S S :A A
     :T T :R R :term term
     :terminals terminals :walls walls :start-idx start-idx
     :ns-fn ns-fn :action-kw action-kw :gamma gamma}))
