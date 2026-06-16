;; @tier slow
;; Headless tests for the canonical Pac-Man maze substrate (genmlx.agents.pacman):
;; the book's single shared environment. Pure data + asserts (no Ink/terminal /
;; renderer). Confirms (1) the ASCII front-end + build-mdp produce VALID MDP
;; tensors (proper distributions, correct terminals/walls/start), and (2) the
;; substrate is consumed UNCHANGED by the existing MDP planner (agent.cljs) — an
;; optimal Pac-Man routes to the highest-value reachable cache. Value iteration is
;; RNG-free, so every route below is reproducible.
;; Run: bun run --bun nbb test/genmlx/agents_pacman_test.cljs

(ns genmlx.agents-pacman-test
  (:require [genmlx.agents.pacman :as pac]
            [genmlx.agents.agent :as agent]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]))

(def passed (volatile! 0))
(def failed (volatile! 0))
(defn assert-true [msg c]
  (if c (do (vswap! passed inc) (println " PASS" msg))
        (do (vswap! failed inc) (println " FAIL" msg))))
(defn assert-equal [msg expected actual]
  (if (= expected actual) (do (vswap! passed inc) (println " PASS" msg "  =" (pr-str actual)))
      (do (vswap! failed inc) (println " FAIL" msg "  expected:" (pr-str expected) "  got:" (pr-str actual)))))
(defn assert-close [msg expected actual tol]
  (if (< (Math/abs (- expected actual)) tol)
    (do (vswap! passed inc) (println " PASS" msg "  ~=" actual))
    (do (vswap! failed inc) (println " FAIL" msg "  expected ~" expected "  got:" actual))))

(defn- greedy-path
  "Follow the agent's argmax-Q action under deterministic geometry (ns-fn) from
   `start` to a terminal — the agent's INTENDED route, independent of env noise."
  [ag start]
  (let [Qm (mx/->clj (:Q ag)) A (:A (:mdp ag)) nsf (:ns-fn (:mdp ag))
        terms (set (keys (:terminals (:mdp ag))))]
    (loop [s start path [start] n 0]
      (if (or (terms s) (> n 40)) path
          (let [a (apply max-key #(nth (nth Qm s) %) (range A)) s' (nsf s a)]
            (recur s' (conj path s') (inc n)))))))

;; --- ASCII front-end + vocabulary -------------------------------------------
(println "\n-- pacman: ASCII parsing + vocabulary --")
(let [{:keys [grid start ghosts]} (pac/parse-ascii pac/classic-maze)]
  (assert-equal "classic-maze parses to 7 rows" 7 (count grid))
  (assert-equal "classic-maze rows are 9 wide" 9 (count (first grid)))
  (assert-equal "P -> start at centre [4 3]" [4 3] start)
  (assert-equal "no ghost seeds in classic-maze" [] ghosts)
  (assert-equal "top-left glyph . -> :pellet" :pellet (get-in grid [1 1]))
  (assert-equal "top-right glyph o -> :power"  :power  (get-in grid [1 7]))
  (assert-equal "bottom-left glyph F -> :fruit" :fruit (get-in grid [5 1]))
  (assert-equal "border is wall" :wall (get-in grid [0 0]))
  (assert-true  "P compiles to floor, not a terminal" (= :empty (get-in grid [3 4]))))

(assert-equal "scoring: pellet < power < fruit and a time cost"
              [10.0 50.0 100.0 -1.0]
              [(:pellet pac/pacman-scoring) (:power pac/pacman-scoring)
               (:fruit pac/pacman-scoring) (:timeCost pac/pacman-scoring)])
(assert-equal "legend covers all 7 roles" 7 (count pac/legend))

;; --- build-mdp produces valid tensors ---------------------------------------
(println "\n-- pacman: classic-maze MDP tensors --")
(def classic (pac/pacman-mdp {:ascii pac/classic-maze}))
(let [{:keys [W H S A T R term terminals start-idx]} classic]
  (assert-equal "classic W=9 H=7 S=63 A=4" [9 7 63 4] [W H S A])
  (assert-equal "start-idx = 4 + 9*3 = 31" 31 start-idx)
  (assert-equal "four terminal caches at the corners"
                {10 :pellet 16 :power 46 :fruit 52 :pellet} terminals)
  (assert-equal "T shape [63 4 63]" [63 4 63] (mx/shape T))
  (assert-equal "R shape [63 4]"    [63 4]    (mx/shape R))
  (assert-equal "term shape [63]"   [63]      (mx/shape term))
  (assert-close "every T row is a distribution (sum = S*A = 252)"
                252.0 (mx/item (mx/sum T)) 1e-2)
  (assert-close "term mask marks exactly the 4 caches" 4.0 (mx/item (mx/sum term)) 1e-3)
  (assert-close "fruit cell reward = 100 + timeCost = 99"
                99.0 (mx/item (mx/idx (mx/idx R 46) 0)) 1e-3))

;; --- the planner consumes the substrate UNCHANGED ---------------------------
(println "\n-- pacman: optimal Pac-Man routes to the fruit (highest value) --")
(let [ag (agent/make-mdp-agent {:mdp classic :alpha ##Inf :gamma 1.0 :n-iters 30})
      path (greedy-path ag 31)
      {:keys [states]} (agent/simulate-mdp ag 31 30)
      V (mapv #(mx/item (mx/idx (:V ag) %)) (range 63))]
  (println "   intended route:" path)
  (assert-equal "greedy route ends at the fruit (idx 46)" 46 (last path))
  (assert-true  "value is maximal at a cache cell"
                (contains? #{10 16 46 52} (apply max-key #(nth V %) (range 63))))
  (assert-equal "a deterministic rollout also reaches the fruit" 46 (last states)))

;; --- corridor: the minimal line-world ---------------------------------------
(println "\n-- pacman: corridor (Ch 3a line-world) --")
(def corr (pac/pacman-mdp {:ascii pac/corridor}))
(let [ag (agent/make-mdp-agent {:mdp corr :alpha ##Inf :gamma 1.0 :n-iters 12})
      {:keys [states actions]} (agent/simulate-mdp ag (:start-idx corr) 12)]
  (assert-equal "corridor S = 7"                 7 (:S corr))
  (assert-equal "corridor terminal {6 :power}"   {6 :power} (:terminals corr))
  (assert-equal "corridor rollout walks 0 -> 6"  [0 1 2 3 4 5 6] states)
  (assert-true  "corridor rollout is all :right" (every? #{:right} (mapv (:action-kw corr) actions))))

;; --- two-caches: the inverse-planning setup ---------------------------------
(println "\n-- pacman: two-caches (donut N/S) — optimal heads to the power pellet --")
(def caches (pac/pacman-mdp {:ascii pac/two-caches}))
(let [ag (agent/make-mdp-agent {:mdp caches :alpha ##Inf :gamma 1.0 :n-iters 12})
      path (greedy-path ag (:start-idx caches))]
  (assert-equal "two-caches W=3 H=5 S=15"  [3 5 15] [(:W caches) (:H caches) (:S caches)])
  (assert-equal "start between the caches (idx 7)" 7 (:start-idx caches))
  (assert-equal "caches: pellet north (1), power south (13)"
                {1 :pellet 13 :power} (:terminals caches))
  (assert-equal "optimal Pac-Man goes SOUTH to the power pellet (idx 13)" 13 (last path)))

;; ===========================================================================
;; E1.2 — agent bindings over Pac-Man worlds
;; ===========================================================================
(println "\n-- E1.2: MDP agent binding + GFI (an agent IS a generative function) --")
(def mag (pac/mdp-agent {:maze pac/classic-maze :alpha ##Inf :n-iters 30}))
(assert-true "mdp-agent returns the make-mdp-agent contract"
             (and (some? (:policy mag)) (some? (:Q mag)) (some? (:V mag))
                  (fn? (:act mag)) (fn? (:expected-utility mag)) (map? (:params mag))))
(let [tr (p/simulate (dyn/auto-key (:policy mag)) [31])]
  (assert-true "p/simulate on the Pac-Man policy yields an :action choice + a score"
               (and (some? (cm/get-choice (:choices tr) [:action])) (some? (:score tr)))))
(let [w (mx/item (:weight (p/assess (dyn/auto-key (:policy mag)) [31] (cm/choicemap :action 0))))]
  (assert-true "p/assess on the Pac-Man policy yields a finite log-weight" (js/isFinite w)))
(assert-true "mdp-agent :act picks a legal action int (0..3)"
             (let [a ((:act mag) 31)] (and (integer? a) (<= 0 a 3))))

(println "\n-- E1.2: biased (hyperbolic) agent binding --")
(def bag0 (pac/biased-agent {:discount 0.0 :bias :sophisticated} {:maze pac/classic-maze :alpha 1.0 :n-iters 12}))
(def mag1 (pac/mdp-agent {:maze pac/classic-maze :alpha 1.0 :n-iters 12}))
(assert-true "biased-agent returns the MDP shape MINUS :Q/:V"
             (and (some? (:policy bag0)) (fn? (:expected-utility bag0))
                  (nil? (:Q bag0)) (nil? (:V bag0))))
(let [eub (:expected-utility bag0) eum (:expected-utility mag1)
      S (:S (:mdp mag1)) A (:A (:mdp mag1))
      err (apply max (for [s (range S) a (range A)] (Math/abs (- (eub s a) (eum s a)))))]
  (assert-true "biased agent at discount 0 recovers the unbiased agent (max err < 1e-4)" (< err 1e-4)))

(println "\n-- E1.2: POMDP agent binding (hidden cache, signpost reveal) --")
(def penv-pac (pac/pacman-pomdp {:true-world :power}))
(def pag (pac/pomdp-agent penv-pac {:alpha 2.0 :n-iters 30}))
(assert-true "pomdp-agent returns the 10-key make-pomdp-agent contract"
             (every? pag [:worlds :world-agents :prior :observe :belief-Q :update-belief
                          :update-belief-tensor :act :expected-utility :params]))
(assert-equal "two latent worlds {:pellet :power}" #{:pellet :power} (set (:worlds pag)))
(let [b0 (:prior pag)]
  (assert-true "nil observation leaves the belief unchanged (identity)"
               (= b0 ((:update-belief pag) b0 5 nil)))
  (assert-close "belief snaps to :power at the signpost"
                1.0 (double (get ((:update-belief pag) b0 pac/haunted-signpost :power) :power 0.0)) 1e-6)
  (assert-true ":act on a belief+state returns a legal action int"
               (let [a ((:act pag) b0 (:start-idx penv-pac))] (and (integer? a) (<= 0 a 3)))))

(println "\n-- E1.2: bandit agent binding (corridors as fruit-spawners) --")
(let [env (pac/bandit-env [0.2 0.5 0.9])]
  (assert-equal "bandit-env: 3 corridors -> 3 arms" 3 (:arms env))
  (assert-equal "bandit-env: the best corridor is the 0.9 spawner (idx 2)" 2 (:true-best env)))
(def bandit (pac/bandit-agent {:strategy :thompson :alpha 1.0}))
(assert-true "bandit-agent returns {:act :update-belief :arm-values :params}, NO :policy"
             (and (fn? (:act bandit)) (fn? (:update-belief bandit))
                  (fn? (:arm-values bandit)) (nil? (:policy bandit))))
(let [b      {:arms [[1.0 1.0] [1.0 1.0] [1.0 1.0]]}
      means0 ((:arm-values bandit) b)
      means1 ((:arm-values bandit) ((:update-belief bandit) b 0 1))]
  (assert-true "arm-values: uniform Beta(1,1) priors -> all means 0.5"
               (every? #(< (Math/abs (- 0.5 %)) 1e-9) means0))
  (assert-close "update-belief: a success on arm 0 raises its mean to 2/3"
                (/ 2.0 3.0) (nth means1 0) 1e-9)
  (assert-true "update-belief: the other arms' means stay at 0.5"
               (every? #(< (Math/abs (- 0.5 (nth means1 %))) 1e-9) [1 2])))

;; ===========================================================================
;; E1.3 — render-data adapters on the presentation seam
;; ===========================================================================
(println "\n-- E1.3: Frame producer (Pac-Man roles) + render parity --")
(def cframe (pac/frame classic 31 {:step 0}))
(assert-equal "frame dims match the maze (W=9 H=7, 63 cells)"
              [9 7 63] [(:W cframe) (:H cframe) (count (:cells cframe))])
(assert-equal "agent cell (idx 31) has role :pacman" :pacman (:role (nth (:cells cframe) 31)))
(assert-equal "fruit cache (idx 46) keeps role :fruit" :fruit (:role (nth (:cells cframe) 46)))
(assert-equal "power cache (idx 16) keeps role :power" :power (:role (nth (:cells cframe) 16)))
(assert-equal "border cell (idx 0) has role :wall"     :wall  (:role (nth (:cells cframe) 0)))
(assert-true  "render parity: every frame cache role == the mdp terminals"
              (= (:terminals classic)
                 (into {} (for [[idx _] (:terminals classic)] [idx (:role (nth (:cells cframe) idx))]))))

(let [ag   (pac/mdp-agent {:maze pac/classic-maze :alpha ##Inf :n-iters 30})
      roll (agent/simulate-mdp ag 31 30)
      traj (pac/trajectory classic roll {:V (:V ag)})]
  (assert-equal "trajectory has one frame per rollout state" (count (:states roll)) (count traj))
  (assert-equal "first frame: Pac-Man at the start (idx 31)" :pacman (:role (nth (:cells (first traj)) 31)))
  (assert-equal "last frame: Pac-Man arrives on the fruit (idx 46)" :pacman (:role (nth (:cells (last traj)) 46)))
  (assert-true  "path grows along the trajectory"
                (<= (count (filter #(= :path (:role %)) (:cells (second traj))))
                    (count (filter #(= :path (:role %)) (:cells (last traj)))))))

(println "\n-- E1.3: PosteriorBars from a belief --")
(let [bars (pac/belief->bars "cache belief" {:pellet 0.7 :power 0.3} :power)]
  (assert-equal "belief->bars: one bar per world" 2 (count (:bars bars)))
  (assert-close "belief->bars: weights sum to 1" 1.0 (reduce + (map :weight (:bars bars))) 1e-9)
  (assert-true  "belief->bars: the true world is highlighted" (boolean (some :highlight (:bars bars)))))

(println (str "\n== Results: " @passed " passed, " @failed " failed =="))
(when (pos? @failed) (js/process.exit 1))
