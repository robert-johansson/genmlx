(ns gallery
  "agentmodels TUI gallery — a launcher menu over self-registering demos.

   Mirrors the proven examples/genmlx-tui idiom: state in one r/atom, plain
   reagent components, a single (render (r/as-element [app])) mount. Keyboard is
   handled by a small Node stdin listener (an async event that swap!s state) so
   we lean only on primitives this repo has already proven, not React hooks.

   A demo registers an entry {:id :title :view :on-key :enter :leave}; adding a
   new demo is one entry, no nav code to touch. View primitives live in views.cljs."
  (:require ["ink" :refer [render Text Box Newline]]
            [reagent.core :as r]
            [ch3-demo :as ch3]
            [ch3c-demo :as ch3c]
            [ch3d-demo :as ch3d]
            [ch5-demo :as ch5]))

;; -- Demo registry ----------------------------------------------------------
(def demos
  [{:id :ch3 :title "Ch 3: Gridworld MDP"
    :view ch3/view :on-key ch3/on-key :enter ch3/enter! :leave ch3/leave!}
   {:id :ch3c :title "Ch 3c: POMDP belief filtering"
    :view ch3c/view :on-key ch3c/on-key :enter ch3c/enter! :leave ch3c/leave!}
   {:id :ch3d :title "Ch 3d: Bandits (posterior sampling)"
    :view ch3d/view :on-key ch3d/on-key :enter ch3d/enter! :leave ch3d/leave!}
   {:id :ch5 :title "Ch 5: Inverse goal inference"
    :view ch5/view :on-key ch5/on-key :enter ch5/enter! :leave ch5/leave!}])

(defonce app-state (r/atom {:screen :menu :sel 0}))

(defn- demo-by-id [id] (first (filter #(= (:id %) id) demos)))

;; -- Keyboard ---------------------------------------------------------------
;; stdin delivers raw escape sequences; build the control strings from char
;; codes so this source stays pure ASCII. ESC=27, Ctrl-C=3.
(def ^:private esc (js/String.fromCharCode 27))
(def ^:private ctrl-c (js/String.fromCharCode 3))

(defn- normalize [s]
  (condp = s
    (str esc "[A") :up
    (str esc "[B") :down
    (str esc "[C") :right
    (str esc "[D") :left
    esc            :escape
    ctrl-c         :ctrl-c
    "\r" :enter   "\n" :enter   " " :space
    "q"  :quit    "Q"  :quit
    "r"  :replay  "R"  :replay
    "n"  :noise   "N"  :noise
    "t"  :toggle  "T"  :toggle
    "+"  :plus    "="  :plus
    "-"  :minus   "_"  :minus
    :other))

(defn- exit! [] (js/process.exit 0))

(defn- handle-key [k]
  (let [{:keys [screen sel]} @app-state]
    (cond
      (= k :ctrl-c) (exit!)
      (= screen :menu)
      (case k
        :up    (swap! app-state update :sel #(mod (dec %) (count demos)))
        :down  (swap! app-state update :sel #(mod (inc %) (count demos)))
        :enter (let [d (nth demos sel)]
                 (when (:enter d) ((:enter d)))
                 (swap! app-state assoc :screen (:id d)))
        :quit  (exit!)
        nil)
      :else
      (let [d (demo-by-id screen)]
        (if (#{:escape :quit} k)
          (do (when (:leave d) ((:leave d))) (swap! app-state assoc :screen :menu))
          (when (:on-key d) ((:on-key d) k)))))))

(defn- setup-keys! []
  (let [stdin js/process.stdin]
    (when (.-setRawMode stdin) (.setRawMode stdin true))
    (.resume stdin)
    (.setEncoding stdin "utf8")
    (.on stdin "data" (fn [d] (handle-key (normalize d))))))

;; -- Components -------------------------------------------------------------
(defn- menu []
  (let [sel (:sel @app-state)]
    [:> Box {:flexDirection "column" :padding 1}
     [:> Text {:bold true :color "cyan"} " GenMLX · agentmodels gallery "]
     [:> Newline]
     (for [[i d] (map-indexed vector demos)]
       ^{:key (:id d)}
       [:> Text {:color (if (= i sel) "greenBright" "white")}
        (str (if (= i sel) " > " "   ") (:title d))])
     [:> Newline]
     [:> Text {:dimColor true} " up/down select   enter open   q quit"]]))

(defn- app []
  (if (= :menu (:screen @app-state))
    [menu]
    [(:view (demo-by-id (:screen @app-state)))]))

;; -- Main -------------------------------------------------------------------
(render (r/as-element [app]))
(setup-keys!)
