# The Pac-Man Maze: A Legend

Every chapter draws on one environment, compiled from ASCII art by
`genmlx.agents.pacman`. The vocabulary is fixed here so no chapter has to redefine
it.

| Glyph | Cell | Role | Reward | Meaning |
|-------|------|------|--------|---------|
| `%` / `#` | `:wall` | wall | — | impassable |
| (space) | `:empty` | floor | — | open corridor |
| `.` | `:pellet` | pellet | +10 | small reward, terminal cache |
| `o` | `:power` | power pellet | +50 | large reward, terminal cache |
| `F` | `:fruit` | fruit | +100 | highest reward, terminal cache |
| `P` | `:empty` | Pac-Man start | — | compiles to floor; start position |
| `G` | `:empty` | ghost seed | — | POMDP *latent*, never planner state |

Every step costs **−1** (the `:timeCost`), so a rational Pac-Man trades reward
against distance.

## Coordinates and actions

A maze is a vector of rows, top row first. For a cell at column `x`, row `y`, the
dense state index is `s = x + W·y`. Actions are `:left` (−x), `:right` (+x), `:up`
(−y, toward the top), and `:down` (+y).

## The load-bearing decision

Pellets, power pellets, and fruit are **terminal-utility caches** — the agentmodels
"restaurant" pattern — not a consumed food *set* carried in the state. Ghosts are
**POMDP latent state** — a belief over which cell they occupy — not added to the
planner's state vector. Both choices keep the planner's state the single dense
index `s = x + W·y`, so dense value iteration never faces the food-set / ghost-count
explosion that classic Pac-Man has. This is what lets the whole `genmlx.agents`
library apply to a Pac-Man maze unchanged.

## The canonical mazes

- **`classic-maze`** — a 9×7 ring with four caches at the corners and Pac-Man at the
  centre; all four are equidistant, so an optimal Pac-Man's choice is pure value.
- **`corridor`** — a 1-D hallway to a single power pellet (the minimal teaching MDP).
- **`two-caches`** — a vertical corridor with a pellet north and a power pellet south
  (the inverse-planning setup: the first step reveals the preference).
- **`haunted-maze`** — a hidden-state maze whose true rewarding cache is revealed only
  at a signpost floor cell (the POMDP chapter).
