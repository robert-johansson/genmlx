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

;; --- manifest ---------------------------------------------------------------
(fs/writeFileSync (str data-dir "/manifest.json")
                  (js/JSON.stringify (clj->js {:figures @manifest})))
(println "wrote manifest:" (count @manifest) "figures")
