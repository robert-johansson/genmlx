(ns genmlx.agents.pacman
  "Pac-Man maze substrate for the GenMLX Agents book — the single environment every
   chapter reuses, compiled through genmlx.agents.gridworld/build-mdp into the MDP
   tensors a planner needs.

   The whole book speaks ONE visual language: a Pac-Man maze drawn as ASCII art
   (the CS188 `.lay` convention), where

     % / #   wall              (impassable)
     (space) open corridor     (floor)
     .       pellet            terminal cache, small reward
     o       power pellet      terminal cache, large reward
     F       fruit             terminal cache, highest reward
     P       Pac-Man start     (floor; position recorded, not a terminal)
     G       ghost seed        (floor; POMDP latent, NOT in the planner state)

   THE LOAD-BEARING DESIGN DECISION. Pellets/power/fruit are *terminal-utility
   caches* — exactly agentmodels.org's restaurant pattern — NOT a consumed food
   *set* carried in the state. And ghosts are POMDP *latent* state (a belief over
   ghost cells, handled by genmlx.agents.pomdp), NOT added to the planner's state
   vector. Both choices keep the planner state the single dense cell index

       s = x + W*y      (screen coords: x = column, y = row from the top)

   so every genmlx.agents planner (agent / biased_planners / pomdp / inverse /
   differentiable) consumes a Pac-Man maze unchanged, and dense value iteration
   never faces the food-set/ghost-count state explosion that classic Pac-Man has.

   Coordinate + action conventions are inherited verbatim from gridworld.cljs
   (:left -x, :right +x, :up -y, :down +y). Three concerns live here, each a thin
   layer over existing genmlx.agents code: (1) the ASCII maze front-end + build-mdp
   compile; (2) one-line bindings of the four agent constructors over Pac-Man
   worlds (so a chapter writes (pac/mdp-agent), not an incantation); (3) render-data
   adapters producing the genmlx.agents.presentation Frame / PosteriorBars shapes.
   The heavy numeric work (value iteration, belief filtering) lives below, unchanged."
  (:require [genmlx.agents.gridworld :as gw]
            [genmlx.agents.agent :as agent]
            [genmlx.agents.biased-planners :as bp]
            [genmlx.agents.pomdp :as pomdp]
            [genmlx.agents.pomdp-env :as penv]
            [genmlx.agents.presentation :as present]
            [genmlx.mlx :as mx]))

;; ===========================================================================
;; Vocabulary: glyphs, scoring, and the legend (one source of truth)
;; ===========================================================================

(def pacman-scoring
  "Default cache utilities (CS188-flavoured) plus the per-step time cost. Pellets,
   power pellets, and fruit are terminal-utility caches; `:timeCost` is charged on
   every step, so a rational Pac-Man trades reward against the distance to it."
  {:pellet 10.0 :power 50.0 :fruit 100.0 :timeCost -1.0})

(def glyph->cell
  "ASCII maze glyph -> gridworld cell keyword. `%`/`#` are walls; `.`/`o`/`F` are
   terminal caches; everything Pac-Man or a ghost stands on is open floor (`P` and
   `G` are start/ghost MARKERS — their positions are extracted by parse-ascii and
   the cell itself compiles to :empty, because neither is a terminal)."
  {\% :wall, \# :wall, \space :empty,
   \. :pellet, \o :power, \F :fruit,
   \P :empty, \G :empty})

(def legend
  "The maze vocabulary in one place: every cell role with its ASCII glyph, the
   gridworld cell it compiles to, its default reward (from pacman-scoring), and its
   meaning. Renderers and the book read this so the visual language is defined once."
  [{:role :wall   :glyph \%     :cell :wall   :reward nil                  :desc "maze wall (impassable)"}
   {:role :floor  :glyph \space :cell :empty  :reward nil                  :desc "open corridor"}
   {:role :pellet :glyph \.     :cell :pellet :reward (:pellet pacman-scoring) :desc "pellet — small reward, terminal"}
   {:role :power  :glyph \o     :cell :power  :reward (:power pacman-scoring)  :desc "power pellet — large reward, terminal"}
   {:role :fruit  :glyph \F     :cell :fruit  :reward (:fruit pacman-scoring)  :desc "bonus fruit — highest reward, terminal"}
   {:role :pacman :glyph \P     :cell :empty  :reward nil                  :desc "Pac-Man start (compiles to floor)"}
   {:role :ghost  :glyph \G     :cell :empty  :reward nil                  :desc "ghost seed — POMDP latent, not planner state"}])

;; ===========================================================================
;; ASCII front-end
;; ===========================================================================

(defn parse-ascii
  "Parse an ASCII maze (a vector of equal-length strings, top row first) into

     {:grid <vector of keyword rows> :start [x y] :ghosts [[x y] ...]}

   suitable for gridworld/build-mdp. `P` marks Pac-Man's start (the first one wins;
   defaults to [0 0] if absent) and each `G` marks a latent ghost seed; both glyphs
   compile to :empty floor."
  [rows]
  (let [W (count (first rows))]
    (assert (apply = (map count rows)) "every maze row must be the same width")
    (reduce
      (fn [acc y]
        (let [row (nth rows y)]
          (-> acc
              (update :grid conj (mapv glyph->cell row))
              (update :start (fn [s] (or s (some (fn [x] (when (= \P (nth row x)) [x y]))
                                                 (range W)))))
              (update :ghosts into (for [x (range W) :when (= \G (nth row x))] [x y])))))
      {:grid [] :start nil :ghosts []}
      (range (count rows)))))

;; ===========================================================================
;; MDP builder
;; ===========================================================================

(defn pacman-mdp
  "Compile a Pac-Man maze into MDP tensors via gridworld/build-mdp. Give it either
   an `:ascii` maze (vector of strings) or a ready `:grid` literal, plus optional
   `:utilities` (defaults to pacman-scoring), `:start`, `:gamma`, `:noise`.

   Returns build-mdp's full map (`:W :H :S :A :T :R :term :terminals :walls
   :start-idx :ns-fn :action-kw :gamma :noise`) augmented with `:ascii`, the
   extracted `:ghosts`, and the `:legend`."
  [{:keys [ascii grid utilities start gamma noise]
    :or   {utilities pacman-scoring gamma 1.0 noise 0.0}}]
  (let [{pg :grid pstart :start ghosts :ghosts} (when ascii (parse-ascii ascii))
        grid (or grid pg)
        mdp  (gw/build-mdp {:grid grid :utilities utilities
                            :start (or start pstart [0 0])
                            :gamma gamma :noise noise})]
    (assoc mdp :ascii ascii :ghosts (vec ghosts) :legend legend)))

;; ===========================================================================
;; Canonical mazes (the book's running examples)
;; ===========================================================================

(def classic-maze
  "The canonical book maze: a 9x7 ring with four reachable caches at the corners —
   a pellet (10) top-left, a power pellet (50) top-right, fruit (100) bottom-left,
   and a pellet (10) bottom-right — with Pac-Man starting at the centre. All four
   caches are equidistant (5 steps from start), so an optimal Pac-Man's choice is
   driven purely by VALUE: it heads for the fruit."
  ["%%%%%%%%%"
   "%.     o%"
   "% %%%%% %"
   "%   P   %"
   "% %%%%% %"
   "%F     .%"
   "%%%%%%%%%"])

(def corridor
  "Ch 3a line-world, Pac-Man flavour: a 1-D hallway from Pac-Man (left) to a single
   power pellet (right). Up/down clamp to a stay (one row high). The minimal,
   hand-traceable teaching MDP."
  ["P     o"])

(def two-caches
  "The donut-North/South setup for inverse planning: a vertical corridor with a
   small pellet (10) to the north and a power pellet (50) to the south, Pac-Man
   between them. The caches are equidistant, so the very first step reveals the
   preference — the engine of the IRL chapter."
  ["%.%"
   "% %"
   "%P%"
   "% %"
   "%o%"])

(def haunted-maze
  "Hidden-state maze for the POMDP chapter: Pac-Man must reach the rewarding cache
   but cannot tell which of the two — the pellet (north-west) or the power pellet
   (north-east) — actually pays until it stands on the SIGNPOST floor cell (idx 12),
   the 'which corridor is safe?' reveal. The signpost is plain floor (NOT a cache),
   so Pac-Man can sense there and still act on what it learns."
  ["%%%%%"
   "%. o%"
   "%   %"
   "% P %"
   "%%%%%"])

(def haunted-signpost
  "The floor cell of `haunted-maze` whose observation reveals the true world (idx 12)."
  12)

;; ===========================================================================
;; E1.2 — agent bindings: the four genmlx.agents constructors over Pac-Man worlds
;; ===========================================================================
;; Thin convenience wrappers. Each returns its constructor's CONTRACTS.md shape
;; UNCHANGED — the binding only supplies a Pac-Man maze and sensible defaults, so a
;; book chapter reads (pac/mdp-agent) instead of a build-mdp + make-*-agent dance.

(defn mdp-agent
  "Optimal (alpha ##Inf) or soft (finite alpha) MDP Pac-Man over a maze. Returns the
   agent/make-mdp-agent contract {:mdp :Q :V :policy :act :expected-utility :params}.
   Opts: :maze (ascii, default classic-maze) :utilities :noise :alpha :gamma :n-iters."
  ([] (mdp-agent {}))
  ([{:keys [maze utilities noise alpha gamma n-iters]
     :or   {maze classic-maze utilities pacman-scoring noise 0.0
            alpha ##Inf gamma 1.0 n-iters 30}}]
   (agent/make-mdp-agent {:mdp   (pacman-mdp {:ascii maze :utilities utilities
                                              :gamma gamma :noise noise})
                          :alpha alpha :gamma gamma :n-iters n-iters})))

(defn biased-agent
  "Time-inconsistent (hyperbolic) Pac-Man. `bias` = {:discount k :bias
   :naive|:sophisticated :reward-myopic-bound C}. Returns the
   biased-planners/make-biased-mdp-agent contract (the MDP shape minus :Q/:V).
   At :discount 0 it recovers the unbiased agent. Opts as `mdp-agent`."
  ([bias] (biased-agent bias {}))
  ([bias {:keys [maze utilities noise alpha gamma n-iters]
          :or   {maze classic-maze utilities pacman-scoring noise 0.0
                 alpha ##Inf gamma 1.0 n-iters 30}}]
   (bp/make-biased-mdp-agent {:mdp   (pacman-mdp {:ascii maze :utilities utilities
                                                  :gamma gamma :noise noise})
                              :alpha alpha :gamma gamma :n-iters n-iters}
                             bias)))

(defn pacman-pomdp
  "Hidden-state Pac-Man POMDP env (the make-pomdp-agent / simulate-pomdp bundle):
   the latent world is which cache rewards, revealed only at the `:signpost` floor
   cell. Wraps pomdp-env/restaurant-gridworld for a Pac-Man maze.
   Opts: :maze (default haunted-maze) :goals (default [:pellet :power]) :signpost
         :true-world :start."
  ([] (pacman-pomdp {}))
  ([{:keys [maze goals signpost true-world start]
     :or   {maze haunted-maze goals [:pellet :power] signpost haunted-signpost
            true-world :power}}]
   (let [{pg :grid pstart :start} (parse-ascii maze)]
     (-> (penv/restaurant-gridworld {:grid pg :goals goals :signpost signpost
                                     :true-world true-world :start (or start pstart)})
         (assoc :start (or start pstart))))))

(defn pomdp-agent
  "Build a QMDP POMDP agent for a hidden-state Pac-Man env (from `pacman-pomdp`).
   Returns the 10-key pomdp/make-pomdp-agent contract. Opts: :alpha :noise :gamma
   :n-iters."
  ([] (pomdp-agent (pacman-pomdp) {}))
  ([env] (pomdp-agent env {}))
  ([env {:keys [alpha noise gamma n-iters]
         :or   {alpha 2.0 noise 0.0 gamma 1.0 n-iters 30}}]
   (pomdp/make-pomdp-agent (merge env {:alpha alpha :noise noise :gamma gamma :n-iters n-iters}))))

(defn bandit-env
  "Pac-Man bandit env — re-narrating K corridors as Bernoulli fruit-spawners, where
   corridor i spawns fruit with probability `fruit-rates`[i]. (A deliberate
   re-storying: a bandit is not spatial; the maze is motivation, not a literal MDP.)
   Wraps pomdp-env/bandit-pomdp; returns the make-bandit-agent / simulate-bandit
   bundle. Opts: :prior (per-arm Beta) :horizon."
  ([fruit-rates] (bandit-env fruit-rates {}))
  ([fruit-rates {:keys [prior horizon] :or {horizon 30}}]
   (penv/bandit-pomdp {:thetas fruit-rates :prior prior :horizon horizon})))

(defn bandit-agent
  "Build a multi-armed bandit Pac-Man (corridors-as-arms). Returns the
   pomdp/make-bandit-agent contract {:act :update-belief :arm-values :params}.
   Opts: :strategy (:thompson | :softmax) :alpha."
  ([] (bandit-agent {}))
  ([{:keys [strategy alpha] :or {strategy :thompson alpha 1.0}}]
   (pomdp/make-bandit-agent {:strategy strategy :alpha alpha})))

;; ===========================================================================
;; E1.3 — render-data adapters on the presentation.cljs seam
;; ===========================================================================
;; Producers of the genmlx.agents.presentation DATA shapes (Frame / PosteriorBars)
;; for Pac-Man worlds. Unlike the generic present/state->frame (which collapses all
;; terminals to a single :goal role), `frame` keeps each cache's TRUE role
;; (:pellet / :power / :fruit) and tags Pac-Man (:pacman) and the latent ghost
;; (:ghost), so a sprite renderer (the E2 Canvas2D layer) can draw each distinctly.
;; The Frame shape is byte-identical to presentation's, so present/render-frame-text
;; and every other consumer keep working. No cs188-cljs import.

(def ^:private cache-glyph
  "Cache cell keyword -> its display glyph (read from the legend, one source of truth)."
  (into {} (for [{:keys [cell glyph]} legend :when (#{:pellet :power :fruit} cell)]
             [cell (str glyph)])))

(defn- norm01 [v lo hi]
  (when (and v lo hi (not= hi lo)) (max 0.0 (min 1.0 (/ (- v lo) (- hi lo))))))

(defn frame
  "A Pac-Man presentation Frame — {:W :H :cells [{:glyph :role :value?}] :meta
   {:step :action}} (the genmlx.agents.presentation contract, with roles extended to
   :pacman / :pellet / :power / :fruit / :ghost) — picturing the agent at `agent-idx`
   on a pacman-mdp. Opts: :step :action :vs/:vlo/:vhi (pre-extracted value function
   to shade empty floor) :path (visited set) :ghost-idx (latent / belief-peak ghost)."
  [{:keys [W H walls terminals]} agent-idx
   {:keys [step action vs vlo vhi path ghost-idx] :or {path #{}}}]
  {:W W :H H
   :meta {:step step :action action}
   :cells (vec (for [y (range H) x (range W)
                     :let [idx (+ x (* W y))]]
                 (cond
                   (contains? walls idx)     {:glyph "█" :role :wall}
                   (= idx agent-idx)         {:glyph "ᗧ" :role :pacman}
                   (= idx ghost-idx)         {:glyph "ᗣ" :role :ghost}
                   (contains? terminals idx) (let [kw (terminals idx)]
                                               {:glyph (get cache-glyph kw (subs (name kw) 0 1))
                                                :role  kw})
                   (contains? path idx)      {:glyph "∘" :role :path}
                   :else                     {:glyph "·" :role :empty
                                              :value (when vs (norm01 (nth vs idx) vlo vhi))})))})

(defn trajectory
  "A Pac-Man Trajectory ([Frame]) from a rollout {:states :actions} on a pacman-mdp.
   Frame i shows Pac-Man at states[i] with earlier states drawn as :path. Options:
   `:V` ([S] MLX value fn) shades empties — the lone seam crossing (mx/->clj V);
   `:ghost-idx` overlays the ghost on every frame."
  [{:keys [action-kw] :as mdp} {:keys [states actions]} & [{:keys [V ghost-idx]}]]
  (let [vs  (when V (vec (mx/->clj V)))
        vlo (when vs (reduce min vs))
        vhi (when vs (reduce max vs))]
    (vec (map-indexed
           (fn [i s]
             (frame mdp s {:step      i
                           :action    (when (< i (count actions)) (nth action-kw (nth actions i)))
                           :vs vs :vlo vlo :vhi vhi
                           :path      (set (take i states))
                           :ghost-idx ghost-idx}))
           states))))

(defn belief->bars
  "PosteriorBars from a Pac-Man belief / goal posterior {world -> prob} (delegates to
   presentation/dist->bars). `highlight` marks the true world's bar. Bandit beliefs
   ({:arms [[a b]...]}) use presentation/bandit-bars directly."
  [title belief & [highlight]]
  (present/dist->bars title belief highlight))
