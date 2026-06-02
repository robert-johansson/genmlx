(ns agentmodels.presentation
  "THE RENDER-AGNOSTIC SEAM.

   Pure CLJS data producers + plain-text renderers. NO Ink, NO React, NO reagent
   — this namespace stays on the assert-test side of the line. Inference below it
   produces these immutable data shapes; the ASCII renderers here and the Ink view
   primitives in examples/agentmodels-tui/ consume the *identical* data. The same
   bytes that an assert test renders to text drive the live terminal.

   DATA CONTRACT
   -------------
   Frame  — one gridworld picture:
     {:W int :H int
      :cells [ {:glyph str :role kw :value (optional float 0..1)} ... ]
                ;; row-major, count == W*H, index = x + W*y
                ;; role ∈ #{:agent :wall :goal :path :empty}
      :meta  {:step int :action (kw|nil)}}
   Trajectory — [Frame], one per rollout step.
   PosteriorBars — {:title str :bars [ {:label str :weight float
                                        :highlight (optional bool)} ... ]}  (weights sum to 1)."
  (:require [genmlx.mlx :as mx]
            [clojure.string :as str]))

;; ---------------------------------------------------------------------------
;; env + trajectory -> [Frame]   (pure producers; the ONE seam crossing is the
;; single (mx/->clj V) below, turning the MLX value function into JS numbers)
;; ---------------------------------------------------------------------------

(defn- terminal-glyph [kw] (subs (str/upper-case (name kw)) 0 1)) ; :A -> "A"

(defn- norm01 [v lo hi]
  (when (and v lo hi (not= hi lo))
    (max 0.0 (min 1.0 (/ (- v lo) (- hi lo))))))

(defn state->frame
  "Pure: one Frame picturing the agent at `agent-idx`. `path` (set of visited
   indices) renders as :path; `vs`/`vlo`/`vhi` (pre-extracted JS value function)
   shade empty cells."
  [{:keys [W H walls terminals]} agent-idx
   {:keys [step action vs vlo vhi path] :or {path #{}}}]
  {:W W :H H
   :meta {:step step :action action}
   :cells (vec (for [y (range H) x (range W)
                     :let [idx (+ x (* W y))]]
                 (cond
                   (contains? walls idx)     {:glyph "█" :role :wall}
                   (= idx agent-idx)         {:glyph "@" :role :agent}
                   (contains? terminals idx) {:glyph (terminal-glyph (terminals idx)) :role :goal}
                   (contains? path idx)      {:glyph "∘" :role :path}
                   :else                     {:glyph "·" :role :empty
                                              :value (when vs (norm01 (nth vs idx) vlo vhi))})))})

(defn env->trajectory
  "Pure: a Trajectory ([Frame]) from a rollout {:states :actions}. Frame i shows
   the agent at states[i] with earlier states drawn as :path. Optional value
   array `V` ([S] MLX) shades empties; (mx/->clj V) is the lone seam crossing."
  [{:keys [action-kw] :as mdp} {:keys [states actions]} & [V]]
  (let [vs  (when V (vec (mx/->clj V)))
        vlo (when vs (reduce min vs))
        vhi (when vs (reduce max vs))]
    (vec (map-indexed
           (fn [i s]
             (state->frame mdp s
                           {:step i
                            :action (when (< i (count actions)) (nth action-kw (nth actions i)))
                            :vs vs :vlo vlo :vhi vhi
                            :path (set (take i states))}))
           states))))

;; ---------------------------------------------------------------------------
;; marginals -> PosteriorBars   (consumes exact/exact-posterior :marginals shape)
;; ---------------------------------------------------------------------------

(defn marginals->bars
  "PosteriorBars from an exact/exact-posterior {:marginals {addr {value prob}}}
   map: pick address `addr`, one bar per value."
  [marginals addr title]
  {:title title
   :bars  (->> (get marginals addr)
               (map (fn [[v p]] {:label (str v) :weight p}))
               (sort-by :label)
               vec)})

(defn dist->bars
  "PosteriorBars from a plain {value -> probability} map (e.g. a goal posterior).
   Optional `highlight` marks the bar for that value with :highlight true (e.g.
   the known true goal), which the view renders distinctly."
  [title m & [highlight]]
  {:title title
   :bars  (->> m
               (map (fn [[v p]]
                      (cond-> {:label (if (keyword? v) (name v) (str v)) :weight p}
                        (= v highlight) (assoc :highlight true))))
               (sort-by :label)
               vec)})

;; ---------------------------------------------------------------------------
;; text renderers (consume the data shapes above)
;; ---------------------------------------------------------------------------

(defn render-frame-text
  "ASCII render of a Frame: a header line + rows of glyphs, top row first."
  [{:keys [W H cells meta]}]
  (str/join
    "\n"
    (cons (str "step " (:step meta) (when (:action meta) (str "  →" (name (:action meta)))))
          (for [y (range H)]
            (str/join " " (for [x (range W)] (:glyph (nth cells (+ x (* W y))))))))))

(defn- bar-glyphs [weight width] (apply str (repeat (Math/round (* weight width)) "█")))

(defn render-bars-text
  "ASCII bar chart of a PosteriorBars."
  [{:keys [title bars]} & [width]]
  (let [w   (or width 24)
        lab (apply max 1 (map (comp count :label) bars))]
    (str/join
      "\n"
      (cons title
            (for [{:keys [label weight]} bars]
              (str (str/join (repeat (- lab (count label)) " ")) label " │"
                   (bar-glyphs weight w) " " (.toFixed weight 3)))))))
