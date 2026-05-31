;; ============================================================================
;; GenMLX VLM — perceiving ARC-AGI-3 scenes (ink + nbb slide deck)
;; ============================================================================
;; Each scene slide renders a real ARC-AGI-3 frame as colored blocks in the
;; terminal (LEFT) and, on demand, calls the qwen3.5-4b VLM live to DESCRIBE the
;; scene, then PARSES that description into structured facts (RIGHT).
;;
;;   describe (VLM, live)  →  free-form scene description  →  parse → {facts}
;;
;; The VLM is given the rendered PNG (../genmlx-lab/dev/arc_frames/<game>_deck.png); the
;; terminal render is drawn from the same frame JSON with the same palette, so
;; what you see is what the model saw.
;;
;; Run (from repo root, or via run.sh):
;;   OS_ACTIVITY_MODE=disable NODE_PATH=examples/genmlx-tui/node_modules:node_modules \
;;     nbb examples/genmlx-vlm-deck/deck.cljs
;;   Keys: ← / → move · r / Enter ask the VLM · q quit
;;
;; Headless self-test: DECK_SELFTEST=1 (grids only) | =full (also runs the VLM)
;; ============================================================================

(ns genmlx-vlm-deck.deck
  (:require [genmlx.mlx :as mx]
            [genmlx.llm.vision :as vision]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [clojure.string :as str]
            [promesa.core :as pr]
            [reagent.core :as r]
            ["ink" :refer [render Text Box Newline useInput useApp]]
            ["ink-spinner$default" :as Spinner]))

(def fs (js/require "fs"))
(def FRAMES "../genmlx-lab/dev/arc_frames")
(def VLM-NAME "qwen3.5-4b-mlx-bf16")
(def VLM-PATH (str (.-HOME js/process.env) "/.cache/models/" VLM-NAME))

;; ARC-AGI-3 16-colour palette (recovered from the rendered PNGs; 0 = white bg).
(def palette
  ["#FFFFFF" "#CCCCCC" "#999999" "#666666" "#333333" "#000000" "#E53AA3" "#FF851B"
   "#F93C31" "#1E93FF" "#921231" "#FFDC00" "#B10DC9" "#39CCCC" "#4FCC30" "#7FDBFF"])

;; ----------------------------------------------------------------------------
;; helpers
;; ----------------------------------------------------------------------------
(defn png-bytes [path]
  (let [b (.readFileSync fs path)]
    (js/Uint8Array. (.-buffer b) (.-byteOffset b) (.-byteLength b))))

(defn load-grid
  "Load frame `idx` of game `g` from its JSON as a 2D vector of ints (0-15)."
  [g idx]
  (let [data (js/JSON.parse (.readFileSync fs (str FRAMES "/" g ".json") "utf8"))]
    (js->clj (aget data "steps" idx "frame" 0))))

(defn downsample [grid f]
  (let [h (count grid) w (count (nth grid 0))]
    (vec (for [r (range 0 h f)] (vec (for [c (range 0 w f)] (get-in grid [r c])))))))

;; ----------------------------------------------------------------------------
;; Parse a VLM description into structured facts — "in ways we want".
;; ----------------------------------------------------------------------------
(def ^:private color-words
  ["magenta" "pink" "cyan" "teal" "turquoise" "yellow" "red" "blue" "green"
   "orange" "purple" "violet" "grey" "gray" "black" "white" "maroon" "brown" "azure"])

(defn- present?
  "True if `kw` (a regex fragment) appears in `t` in at least one clause that is
   not negated — so \"no visible walls\" reads as absent, not present."
  [t kw]
  (boolean (some (fn [w] (not (re-find #"(?i)\bno\b|\bnot\b|\bnone\b|without|n't" w)))
                 (re-seq (re-pattern (str "(?i)[\\s\\S]{0,18}\\b" kw)) t))))

(defn parse-desc [t]
  (let [size (when-let [m (re-find #"(\d+)\s*(?:[x×]|by)\s*(\d+)" t)]
               (str (nth m 1) "×" (nth m 2)))
        ;; word-boundary match so "coloured" doesn't register as "red"
        colors (vec (distinct (filter #(re-find (re-pattern (str "(?i)\\b" % "\\b")) t) color-words)))
        coords (when-let [m (re-find #"(?i)(?:player|agent)[\s\S]{0,60}?row\s*(\d+)[\s\S]{0,25}?col(?:umn)?\s*(\d+)" t)]
                 (str "row " (nth m 1) ", col " (nth m 2)))
        player-present (or (present? t "player") (present? t "agent"))]
    {:layout (or size "unstructured")
     :colors colors
     :regions (count (re-seq #"(?i)\d+\s*[x×]\s*\d+\b" t))
     :player (cond coords coords player-present "present (no coords)" :else "none")
     :walls? (present? t "walls?")
     :goal?  (or (present? t "goal") (present? t "target"))
     :border? (or (present? t "border") (present? t "outline"))}))

;; ----------------------------------------------------------------------------
;; VLM call (async I/O)
;; ----------------------------------------------------------------------------
(def describe-prompt
  (str "This is one frame from an ARC-AGI-3 grid puzzle game — a grid of coloured "
       "cells on a white background. Describe the scene concretely and concisely: "
       "the distinct coloured shapes or regions (their colour, rough size, and "
       "location), anything that looks like a movable player or agent, and any "
       "walls, borders, or goal markers you can see."))

(defn describe-and-parse [session game]
  (pr/let [_ (.reset session)
           bytes (png-bytes (str FRAMES "/" game "_deck.png"))
           res (.send session describe-prompt
                      #js {:images #js [bytes]
                           :config #js {:maxNewTokens 230 :reasoningEffort "none" :repetitionPenalty 1.0}})
           text (str/trim (or (.-text res) ""))]
    {:desc text :parsed (parse-desc text)}))

;; ----------------------------------------------------------------------------
;; Perception as inference: a per-cell reading becomes a scored GFI trace.
;; (vision/make-grid-gf + labels->constraints + p/generate — the same plumbing
;;  vision.cljs uses; here on a coarse grid with a sparsity prior so the score
;;  discriminates real scenes from noise.)
;; ----------------------------------------------------------------------------
(defn coarse-of
  "Downsample a grid to K×K, taking the most common (mode) colour per block."
  [grid K]
  (let [h (count grid) w (count (first grid)) bh (quot h K) bw (quot w K)]
    (vec (for [br (range K)]
           (vec (for [bc (range K)]
                  (let [fr (frequencies (for [r (range (* br bh) (* (inc br) bh))
                                              c (range (* bc bw) (* (inc bc) bw))]
                                          (get-in grid [r c])))]
                    (key (apply max-key val fr)))))))))

(defn- idx-char [i] (if (zero? i) "·" (nth "123456789ABCDEF" (dec i))))
(defn- f1 [x] (if (number? x) (.toFixed x 1) (str x)))

(defn infer-fig [grid]
  (let [K 8
        coarse  (coarse-of grid K)
        present (vec (sort (distinct (apply concat coarse))))
        ctypes  (mapv str present)
        ;; the scene's own dominant colour is its "background"
        bg-color (key (apply max-key val (frequencies (apply concat coarse))))
        bg-idx  (or (first (keep-indexed (fn [i v] (when (= v (str bg-color)) i)) ctypes)) 0)
        ncol    (count ctypes)
        cell-logits (vec (for [i (range ncol)] (if (= i bg-idx) 1.5 0.0))) ; prefer dominant colour
        prior   (vec (repeat K (vec (repeat K cell-logits))))
        gf      (vision/make-grid-gf K K ctypes prior)
        score   (fn [lbls] (mx/item (:weight (p/generate gf []
                            (cm/from-map (vision/labels->constraints lbls ctypes))))))
        labels  (mapv (fn [row] (mapv str row)) coarse)
        w-real  (score labels)
        rand-l  (vec (for [_ (range K)] (vec (for [_ (range K)] (str (rand-nth present))))))
        w-rand  (score rand-l)]
    {:lines (into ["Perceived scene → 8×8 coarse grid (mode colour per block):"]
                  (concat
                   (mapv (fn [row] (str "    " (apply str (map idx-char row)))) coarse)
                   [""
                    (str "make-grid-gf → 8×8 categorical cells over " ncol " colours, with a")
                    "prior that expects a scene to concentrate on its dominant colour."
                    ""
                    "labels->constraints + p/generate → the reading is now a scored TRACE:"
                    (str "  log P(real perceived scene) = " (f1 w-real))
                    (str "  log P(random grid)          = " (f1 w-rand))
                    (str "  → the real scene is " (f1 (- w-real w-rand)) " nats more probable than noise")
                    ""
                    "→ Perception is GFI evidence: the SAME p/generate that scores any"
                    "  GenMLX model now scores what was seen. Swap the prior to encode"
                    "  core-knowledge and the score discriminates real scenes from noise."]))}))

;; ----------------------------------------------------------------------------
;; STATE  (grids + metadata preloaded; VLM loaded async)
;; ----------------------------------------------------------------------------
(def scenes
  [{:game "sk48" :frame 0  :note "10 colours; a player-like marker among blocks"}
   {:game "g50t" :frame 18 :note "6 colours; mid-game, structured regions"}
   {:game "re86" :frame 1  :note "7 colours; sparse early frame"}
   {:game "bp35" :frame 1  :note "8 colours; can reach GAME_OVER"}])

(defonce state (r/atom {:idx 0 :vlm nil :vlm-status :idle
                        :running false :output nil :error nil :history []
                        :cols 80 :rows 24 :grids {} :meta {}}))

(defn preload! []
  (let [summary (js->clj (js/JSON.parse (.readFileSync fs (str FRAMES "/summary.json") "utf8")))
        grids (into {} (for [{:keys [game frame]} scenes] [game (load-grid game frame)]))]
    (swap! state assoc :grids grids :meta summary)))

;; ----------------------------------------------------------------------------
;; SLIDES
;; ----------------------------------------------------------------------------
(def slides
  (into
   [{:kind :title
     :title "GenMLX VLM — perceiving ARC-AGI-3 scenes"
     :body ["A vision-language model is just another generative function."
            "Here it PERCEIVES: each scene is shown live, the VLM describes it,"
            "and we parse that description into structured facts we control."
            ""
            "→ / ←  move      r  ask the VLM      q  quit"]}
    {:kind :text
     :feature "The idea"
     :title "VLM-as-perception, then parse"
     :body ["ARC-AGI-3 drops an agent into a novel game with no rules: each"
            "observation is a 64×64 grid of 16 colours. Step one for any agent"
            "is to SEE the scene."
            ""
            "We render a real frame as coloured blocks (left on the next slides),"
            "hand the same image to the qwen3.5-4b VLM, and ask for a description."
            "Then we PARSE that free-form text into structured facts — grid size,"
            "colours, a player position, walls/goal — exactly as our downstream"
            "code wants them."
            ""
            "The VLM is loaded once at startup (badge in the header). Press r on a"
            "scene to call it live (~15s)."]}]
   (concat
    (map (fn [{:keys [game frame note]}]
           {:kind :scene :game game :frame frame
            :feature (str "Scene · " game)
            :title (str "What does the VLM see in " game "?")
            :note note})
         scenes)
    [{:kind :infer :game "sk48"
      :feature "Perception as inference"
      :title "The VLM's reading becomes a scored GFI trace"
      :note "make-grid-gf + labels->constraints + p/generate"}
     {:kind :end
      :title "fin"
      :body ["A VLM is a generative function that perceives; its description is"
             "just text we can parse into whatever structure we need next —"
             "the first step of an ARC-AGI-3 agent built on the GFI."
             ""
             "frames: ../genmlx-lab/dev/arc_frames/   ·   deck: examples/genmlx-vlm-deck/run.sh"]}])))

;; ----------------------------------------------------------------------------
;; ACTIONS
;; ----------------------------------------------------------------------------
(defn nav! [d sym]
  (swap! state (fn [s]
                 (let [j (max 0 (min (dec (count slides)) (+ (:idx s) d)))]
                   (-> s (assoc :idx j :output nil :error nil :running false)
                       (update :history (fnil conj []) {:from (:idx s) :to j :key sym}))))))

(defn run-current! []
  (let [s @state slide (nth slides (:idx s))]
    (when (not (:running s))
      (case (:kind slide)
        :scene (if-let [vlm (:vlm s)]
                 (do (swap! state assoc :running true :output nil :error nil)
                     (-> (describe-and-parse vlm (:game slide))
                         (.then  (fn [r] (swap! state assoc :running false :output r) (mx/force-gc!)))
                         (.catch (fn [e] (swap! state assoc :running false :error (str (.-message e))) (mx/force-gc!)))))
                 (swap! state assoc :error (str VLM-NAME " still loading — wait for ✓")))
        :infer (do (swap! state assoc :running true :output nil :error nil)
                   (js/setTimeout
                    (fn [] (try (swap! state assoc :running false
                                       :output (infer-fig (get (:grids @state) (:game slide))))
                                (mx/force-gc!)
                                (catch :default e
                                  (swap! state assoc :running false :error (str (.-message e))) (mx/force-gc!))))
                    30))
        nil))))

(defn load-vlm! []
  (swap! state assoc :vlm-status :loading)
  (-> (vision/load-vlm VLM-PATH)
      (.then  (fn [sess] (swap! state assoc :vlm sess :vlm-status :ready)))
      (.catch (fn [e] (swap! state assoc :vlm-status :error :error (str "VLM: " (.-message e)))))))

;; ----------------------------------------------------------------------------
;; FULLSCREEN + noise suppression
;; ----------------------------------------------------------------------------
(defn term-cols [] (or (.-columns js/process.stdout) 80))
(defn term-rows [] (or (.-rows js/process.stdout) 24))
(defn silence-metal-noise! []
  (let [orig (.bind (.-write js/process.stdout) js/process.stdout)]
    (set! (.-write js/process.stdout)
          (fn [chunk & args]
            (if (and (string? chunk) (re-find #"Context leak|CoreAnalytics" chunk))
              true (apply orig chunk args))))))
(defn enter-fullscreen! [] (.write js/process.stdout "[?1049h[2J[H"))
(defn leave-fullscreen! [] (.write js/process.stdout "[?1049l"))

;; ----------------------------------------------------------------------------
;; COMPONENTS
;; ----------------------------------------------------------------------------
(defn dots [idx n] (apply str (map (fn [i] (cond (= i idx) "●" (< i idx) "•" :else "·")) (range n))))
(defn vlm-badge [s] (case (:vlm-status s) :loading "loading…" :ready "ready ✓" :error "error ✗" "idle"))
(defn txt-rows
  ([color lines] (txt-rows color "truncate-end" lines))
  ([color wrap lines]
   (into [:> Box {:flexDirection "column"}]
         (map-indexed (fn [i l] ^{:key i} [:> Text {:color color :wrap wrap} (if (str/blank? l) " " l)]) lines))))

(defn grid-view
  "Render a 2D colour-index grid as coloured half-blocks (▀: fg=top cell,
   bg=bottom cell). Downsampled 2× and run-length-compressed per row."
  [grid]
  (let [g (downsample grid 2)
        nrows (count g) ncols (count (first g))]
    (into [:> Box {:flexDirection "column"}]
          (for [cr (range 0 nrows 2)]
            (let [top (nth g cr)
                  bot (get g (inc cr) (vec (repeat ncols 0)))
                  runs (reduce (fn [acc c]
                                 (let [k [(nth top c) (nth bot c)] lr (peek acc)]
                                   (if (and lr (= (:k lr) k))
                                     (conj (pop acc) (update lr :n inc))
                                     (conj acc {:k k :n 1}))))
                               [] (range ncols))]
              ^{:key cr}
              (into [:> Box {:flexDirection "row"}]
                    (map-indexed
                     (fn [i {[tc bc] :k n :n}]
                       ^{:key i}
                       [:> Text {:color (nth palette tc) :backgroundColor (nth palette bc)}
                        (apply str (repeat n "▀"))])
                     runs)))))))

(defn scene-meta-lines [s game]
  (let [m (get (:meta s) game)]
    (if m
      [(str "game " (get m "game_id"))
       (str "grid " (first (get m "grid_widths")) "×" (first (get m "grid_heights"))
            " · " (get m "num_colors") " colours")
       (str "actions " (str/join " " (get m "actions_available")))
       (str "states " (str/join " " (get m "states_seen")))]
      ["(metadata unavailable)"])))

(defn scene-panel [s slide]
  (cond
    (:running s) [:> Box [:> Spinner {:type "dots"}]
                  [:> Text {:color "yellow"} (if (= :infer (:kind slide)) " running inference…" " asking the VLM…")]]
    (:error s)   [:> Text {:color "red" :wrap "wrap"} (str "✗ " (:error s))]
    (:output s)  (let [o (:output s)]
                   (if (:desc o)
                     [:> Box {:flexDirection "column"}
                      [:> Text {:color "blue" :bold true} "VLM description (live):"]
                      [:> Box {:marginTop 0} [:> Text {:color "green" :wrap "wrap"} (:desc o)]]
                      [:> Box {:marginTop 1} [:> Text {:color "cyan" :bold true} "── parsed facts (what WE extract) ──"]]
                      (txt-rows "cyanBright"
                                [(str "VLM layout : " (:layout (:parsed o)) "   (true grid is 64×64)")
                                 (str "colours    : " (str/join ", " (:colors (:parsed o))))
                                 (str "player     : " (:player (:parsed o)))
                                 (str "walls : " (:walls? (:parsed o)) "    goal : " (:goal? (:parsed o))
                                      "    border : " (:border? (:parsed o)))])]
                     (txt-rows "green" "wrap" (:lines o))))
    :else [:> Text {:color "blueBright" :wrap "wrap"}
           (if (= :infer (:kind slide))
             "▶ press r to score this scene as a GFI trace"
             "▶ press r to ask the VLM to describe this scene")]))

(defn scene-view [s slide]
  (let [grid (get (:grids s) (:game slide))]
    [:> Box {:flexGrow 1 :flexDirection "column"}
     (when (:feature slide) [:> Text {:color "magenta" :bold true} (str "◆ " (:feature slide))])
     [:> Text {:bold true :color "white" :wrap "wrap"} (:title slide)]
     [:> Box {:flexGrow 1 :flexDirection "row" :marginTop 1}
      [:> Box {:flexDirection "column" :width "46%" :paddingRight 2}
       (if grid (grid-view grid) [:> Text {:color "gray"} "(grid unavailable)"])
       [:> Box {:marginTop 1 :flexDirection "column"} (txt-rows "gray" (scene-meta-lines s (:game slide)))]]
      [:> Box {:flexGrow 1 :flexDirection "column" :borderStyle "round" :borderColor "blue" :paddingX 1}
       (scene-panel s slide)]]]))

(defn text-view [slide]
  [:> Box {:flexGrow 1 :flexDirection "column" :marginTop 1}
   (when (:feature slide) [:> Text {:color "magenta" :bold true} (str "◆ " (:feature slide))])
   [:> Box {:marginTop 1} [:> Text {:bold true :color "white" :wrap "wrap"} (:title slide)]]
   [:> Box {:marginTop 1} (txt-rows "gray" "wrap" (:body slide))]])

(defn title-view [s slide]
  [:> Box {:width (:cols s) :height (:rows s) :flexDirection "column"
           :justifyContent "center" :alignItems "center"}
   [:> Text {:bold true :color "cyan" :wrap "wrap"} (:title slide)]
   [:> Newline]
   (into [:> Box {:flexDirection "column" :alignItems "center"}]
         (map-indexed (fn [i l] ^{:key i} [:> Text {:color "gray"} (if (str/blank? l) " " l)]) (:body slide)))
   [:> Newline]
   [:> Text {:color "cyan"} (dots (:idx s) (count slides))]])

(defn app []
  (let [s @state api (useApp) slide (nth slides (:idx s))]
    (useInput
     (fn [input key]
       (cond
         (or (.-rightArrow key) (= input " ") (= input "n") (= input "l")) (nav! 1 "→")
         (or (.-leftArrow key) (= input "p") (= input "h"))               (nav! -1 "←")
         (or (= input "r") (.-return key))                                (run-current!)
         (or (= input "q") (and (.-ctrl key) (= input "c")))
         (do (leave-fullscreen!) (when api (.exit api)) (js/process.exit 0))
         :else nil)))
    (if (#{:title :end} (:kind slide))
      (title-view s slide)
      [:> Box {:flexDirection "column" :width (:cols s) :height (:rows s) :paddingX 1 :paddingY 1}
       [:> Box {:justifyContent "space-between" :width (- (:cols s) 4)}
        [:> Text {:bold true :color "cyan"} " GenMLX VLM · ARC-AGI-3 perception"]
        [:> Text {:color "gray"} (str "slide " (inc (:idx s)) "/" (count slides) "   "
                                      VLM-NAME ": " (vlm-badge s) "  " (dots (:idx s) (count slides)))]]
       (if (#{:scene :infer} (:kind slide)) (scene-view s slide) (text-view slide))
       [:> Text {:color "gray"} " ← → move · r ask the VLM · q quit"]])))

;; ----------------------------------------------------------------------------
;; MAIN
;; ----------------------------------------------------------------------------
(defn ascii-preview [grid]
  ;; selftest only: one char per downsampled cell ('.' = background)
  (let [g (downsample grid 4)]
    (mapv (fn [row] (apply str (map (fn [v] (if (zero? v) "." (str (mod v 10)))) row))) g)))

(defn run-selftest! []
  (println "=== VLM DECK SELFTEST ===")
  (preload!)
  (doseq [{:keys [game frame]} scenes]
    (let [grid (get (:grids @state) game)]
      (println (str "\n-- " game " (frame " frame ") " (count grid) "×" (count (first grid)) " --"))
      (doseq [l (take 16 (ascii-preview grid))] (println "   " l))))
  (println "\n-- perception-as-inference (sk48) --")
  (doseq [l (:lines (infer-fig (get (:grids @state) "sk48")))] (println "   " l))
  (mx/force-gc!)
  (if (= "full" (.. js/process -env -DECK_SELFTEST))
    (-> (vision/load-vlm VLM-PATH)
        (.then (fn [sess]
                 (println "\n=== VLM loaded — describing each scene ===")
                 (reduce (fn [p {:keys [game]}]
                           (pr/let [_ p
                                    {:keys [desc parsed]} (describe-and-parse sess game)]
                             (println (str "\n#### " game " ####"))
                             (println desc)
                             (println "PARSED:" (pr-str parsed))
                             (mx/force-gc!)))
                         (pr/resolved nil) scenes)))
        (.then (fn [_] (println "\nSELFTEST-OK") (js/process.exit 0)))
        (.catch (fn [e] (println "SELFTEST-ERROR:" (.-message e)) (js/process.exit 1))))
    (do (println "\nSELFTEST-OK (grids only; DECK_SELFTEST=full also runs the VLM)")
        (js/process.exit 0))))

(if (some? (.. js/process -env -DECK_SELFTEST))
  (run-selftest!)
  (do
    (preload!)
    (swap! state assoc :cols (term-cols) :rows (term-rows))
    (.on js/process.stdout "resize" (fn [] (swap! state assoc :cols (term-cols) :rows (term-rows))))
    (silence-metal-noise!)
    (enter-fullscreen!)
    (.on js/process "exit" leave-fullscreen!)
    (render (r/as-element [:f> app]))
    (load-vlm!)))
