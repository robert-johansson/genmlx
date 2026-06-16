;; Stage 1 of the GenMLX Agents figure pipeline.
;;
;; Runs the genmlx.agents Pac-Man substrate (E1) to compute REAL rollouts /
;; value functions / posteriors, and serialises each as the presentation data
;; shapes (Frame / Trajectory / PosteriorBars / LineChart) to JSON in
;; docs/agents/render/data/. Stage 2 (build-figures.mjs) renders these to PNG/GIF.
;;
;; Everything here is deterministic: value iteration is RNG-free and the rollouts
;; are taken at alpha = ##Inf / noise = 0 (argmax policy, one-hot transition), so
;; re-running produces byte-identical figures.
;;
;; Run via bin/agents-book-figures (from the repo root).

(ns figures
  (:require [genmlx.agents.pacman :as pac]
            [genmlx.agents.agent :as agent]
            [genmlx.agents.inverse :as inv]
            [agentmodels.biased-inverse :as bi]
            [genmlx.mlx :as mx]
            ["fs" :as fs]))

(def data-dir "docs/agents/render/data")
(fs/mkdirSync data-dir #js {:recursive true})

(def manifest (atom []))
(defn emit!
  "Write one figure's data to data/<id>.json and record it in the manifest."
  [id kind data opts]
  (fs/writeFileSync (str data-dir "/" id ".json") (js/JSON.stringify (clj->js data)))
  (swap! manifest conj {:id id :kind kind :opts opts})
  (println "  data:" id (str "(" (name kind) ")")))

(println "\n== stage 1: computing figure data ==")

;; --- classic maze: value-shaded frame + the optimal rollout (GIF) ------------
(def classic (pac/pacman-mdp {:ascii pac/classic-maze}))
(def cag     (agent/make-mdp-agent {:mdp classic :alpha ##Inf :gamma 1.0 :n-iters 30}))
(let [vs    (vec (mx/->clj (:V cag)))
      vlo   (reduce min vs)
      vhi   (reduce max vs)
      start (:start-idx classic)]
  (emit! "classic-value" :frame
         (pac/frame classic start {:vs vs :vlo vlo :vhi vhi :step 0})
         {:cell 40})
  (emit! "classic-rollout" :trajectory
         (pac/trajectory classic (agent/simulate-mdp cag start 30) {:V (:V cag)})
         {:cell 40 :delay 380}))

;; --- corridor: the minimal line-world rollout (GIF) --------------------------
(def corr    (pac/pacman-mdp {:ascii pac/corridor}))
(def corr-ag (agent/make-mdp-agent {:mdp corr :alpha ##Inf :n-iters 12}))
(emit! "corridor-rollout" :trajectory
       (pac/trajectory corr (agent/simulate-mdp corr-ag (:start-idx corr) 12) {})
       {:cell 44 :delay 320})

;; --- inverse planning: a real goal posterior (bars) --------------------------
;; Watch one step SOUTH on two-caches (state 7, action :down=3); infer the favourite.
(let [grid (:grid (pac/parse-ascii pac/two-caches))
      gas  (inv/goal-agents {:grid grid :goals [:pellet :power] :alpha 2.0})
      post (last (inv/posterior-sequence gas {:pellet 0.5 :power 0.5} [[7 3]]))]
  (emit! "cache-posterior" :bars
         (pac/belief->bars "P(favourite cache | one step south)" post :power)
         {:width 470}))

;; --- value-iteration convergence (line) --------------------------------------
(let [start (:start-idx classic)
      pts   (mapv (fn [n] (mx/item (mx/idx (:V (agent/value-iteration classic ##Inf n)) start)))
                  (range 1 13))]
  (emit! "vi-convergence" :lines
         {:title "value iteration: V(start) vs sweeps" :xlabel "sweep" :ylabel "V(start)"
          :series [{:label "V(start)" :points pts}]}
         {:width 480 :height 300}))

;; --- ch01: the geometric distribution as a bar chart -------------------------
;; P(n) = 0.5^(n+1) (trials until the first head) — the geometric pmf the ch01
;; example samples; shown truncated to n=0..6.
(emit! "ch01-geometric" :bars
       {:title "P(n) — geometric: ghost-blocked steps before a turn"
        :bars  (vec (for [k (range 7)] {:label (str k) :weight (Math/pow 0.5 (inc k))}))}
       {:width 470})

;; --- ch02: softmax policy — P(best action) vs rationality alpha (line) -------
(let [eu     [3.0 1.0 0.5 0.0]
      alphas [0.0 0.25 0.5 1.0 2.0 4.0 8.0]
      p-best (fn [a] (let [es (mapv #(Math/exp (* a %)) eu)] (/ (first es) (reduce + es))))]
  (emit! "ch02-softmax" :lines
         {:title "softmax policy: P(best action) vs rationality α" :xlabel "α (sweep)" :ylabel "P(a*)"
          :series [{:label "P(best)" :points (mapv p-best alphas)}]}
         {:width 480 :height 300}))

;; --- ch02: a 4-way junction, floor shaded by V(s) (frame) --------------------
(def junction-maze
  ["%%.%%"
   "%% %%"
   "o P F"
   "%% %%"
   "%%.%%"])
(def jmdp (pac/pacman-mdp {:ascii junction-maze}))
(def jag  (agent/make-mdp-agent {:mdp jmdp :alpha ##Inf :gamma 1.0 :n-iters 20}))
(let [vs (vec (mx/->clj (:V jag))) vlo (reduce min vs) vhi (reduce max vs)]
  (emit! "ch02-junction" :frame
         (pac/frame jmdp (:start-idx jmdp) {:vs vs :vlo vlo :vhi vhi :step 0})
         {:cell 48}))

;; ===========================================================================
;; E5 — advanced chapters
;; ===========================================================================

;; --- ch05 POMDP: the haunted maze + the belief snap at the signpost ----------
(def hmdp (pac/pacman-mdp {:ascii pac/haunted-maze}))
(emit! "ch05-haunted" :frame
       (pac/frame hmdp (:start-idx hmdp) {:step 0})
       {:cell 52})
(let [env  (pac/pacman-pomdp {:true-world :power})
      pag  (pac/pomdp-agent env {:alpha 2.0 :n-iters 30})
      post ((:update-belief pag) (:prior pag) pac/haunted-signpost :power)]
  (emit! "ch05-belief" :bars
         (pac/belief->bars "belief over the rewarding cache, after the signpost" post :power)
         {:width 480}))

;; --- ch06 bandits: corridor fruit-rate posteriors (Beta means) ---------------
(let [bandit (pac/bandit-agent {:strategy :thompson})
      pulls  [[2 1] [2 1] [2 1] [2 1] [0 0] [0 0] [1 1]]   ; corridor 2 spawns fruit, 0 is dry
      belief (reduce (fn [b [arm r]] ((:update-belief bandit) b arm r))
                     {:arms [[1.0 1.0] [1.0 1.0] [1.0 1.0]]} pulls)
      means  ((:arm-values bandit) belief)]
  (emit! "ch06-arm-posteriors" :bars
         {:title "corridor fruit-rate posteriors (Beta means) after learning"
          :bars  (vec (map-indexed (fn [i m] (cond-> {:label (str "corridor " i) :weight m}
                                               (= i 2) (assoc :highlight true)))
                                   means))}
         {:width 480}))

;; --- ch07 IRL: the observed step on two-caches (value-shaded) ----------------
(let [tc   (pac/pacman-mdp {:ascii pac/two-caches})
      ag   (agent/make-mdp-agent {:mdp tc :alpha ##Inf :gamma 1.0 :n-iters 12})
      vs   (vec (mx/->clj (:V ag))) vlo (reduce min vs) vhi (reduce max vs)
      roll (agent/simulate-mdp ag (:start-idx tc) 12)
      mid  (nth (:states roll) 1)]                          ; one step south, toward power
  (emit! "ch07-twocaches" :frame
         (pac/frame tc mid {:vs vs :vlo vlo :vhi vhi :path #{(:start-idx tc)} :step 1 :action :down})
         {:cell 56}))

;; --- ch09 biases: exponential vs hyperbolic discounting (line) ---------------
(let [ts (range 0 11)]
  (emit! "ch09-discount-curves" :lines
         {:title "discounting a delayed reward: exponential vs hyperbolic" :xlabel "delay t" :ylabel "weight D(t)"
          :series [{:label "exponential 1/2^t"   :points (mapv #(Math/pow 0.5 %) ts)}
                   {:label "hyperbolic 1/(1+2t)" :points (mapv #(/ 1.0 (+ 1.0 (* 2.0 %))) ts)}]}
         {:width 500 :height 300}))

;; --- ch13 joint inference: P(bias | one safe step) — a real posterior --------
(let [post (bi/bias-posterior {:mdp (bi/temptation-mdp) :alpha ##Inf :discount 1.0
                               :n-iters 10 :states [0] :actions [1]})]
  (emit! "ch13-bias-posterior" :bars
         (pac/belief->bars "P(bias | Pac-Man took the safe route once)" post :sophisticated)
         {:width 480}))

;; --- manifest ---------------------------------------------------------------
(fs/writeFileSync (str data-dir "/manifest.json")
                  (js/JSON.stringify (clj->js {:figures @manifest})))
(println "wrote manifest:" (count @manifest) "figures")
