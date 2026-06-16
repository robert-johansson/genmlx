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
   (:left -x, :right +x, :up -y, :down +y). This namespace is pure data + a thin
   ASCII front-end over build-mdp; the heavy numeric work lives in agent.cljs."
  (:require [genmlx.agents.gridworld :as gw]))

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
