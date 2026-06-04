(ns views
  "Reusable Ink view primitives for the agentmodels gallery. ABOVE THE SEAM:
   every component is a pure function of the render-agnostic data shapes defined
   in genmlx.agents.presentation (Frame, PosteriorBars) plus reagent atoms. They
   know nothing about MDPs, MLX, or inference — only how to turn data into
   colored cells. Re-rendering is the proven genmlx-tui model: an r/atom changes,
   reagent re-renders these plain components.

   grid-view        Frame        -> column of colored rows
   bars-view        PosteriorBars-> labeled horizontal bars
   frames-view      [Frame] + idx-atom -> grid-view of the current frame
   trajectory-player [Frame] thunk -> a stateful player that OWNS frame
                    stepping (space), replay (r), and autoplay
   status-bar       map          -> a one-line title/status strip"
  (:require ["ink" :refer [Text Box]]
            [reagent.core :as r]))

(def role->color {:agent "greenBright" :wall "gray" :goal "yellowBright"
                  :path "cyan" :empty "white"})

(defn- hex2 [n] (let [s (.toString (max 0 (min 255 (int n))) 16)]
                  (if (= 1 (count s)) (str "0" s) s)))

(defn- value->bg
  "Subtle value-function shading for empty cells: brighter green = higher value."
  [v]
  (when v (str "#00" (hex2 (+ 24 (* 150 v))) "00")))

(defn grid-view
  "Frame -> a column of rows of colored cells. role -> foreground color;
   optional :value -> background shade on empty cells."
  [{:keys [W H cells]}]
  [:> Box {:flexDirection "column"}
   (for [y (range H)]
     ^{:key y}
     [:> Box {}
      (for [x (range W)
            :let [{:keys [glyph role value]} (nth cells (+ x (* W y)))]]
        ^{:key x}
        [:> Text (cond-> {:color (role->color role) :bold (= role :agent)}
                   (and value (= role :empty)) (assoc :backgroundColor (value->bg value)))
         (str " " glyph " ")])])])

(defn frames-view
  "[Frame] + an index r/atom -> the current frame + a step line. Derefing the
   atom makes this re-render as the index advances (proven reagent reactivity).
   An optional `status-fn` (fn [i n] -> string) appends extra text to the step
   line. Robust to an empty/shrunk frame list (clamps the index)."
  [frames idx & [status-fn]]
  (let [n (max 1 (count frames))
        i (min @idx (dec n))
        f (when (seq frames) (nth frames i))]
    [:> Box {:flexDirection "column"}
     (when f [grid-view f])
     [:> Text {:dimColor true}
      (str "step " i "/" (dec n)
           (when-let [a (:action (:meta f))] (str "   →" (name a)))
           (when status-fn (str "   " (status-fn i n))))]]))

(defn make-trajectory-player
  "Construct a stateful trajectory player that OWNS frame stepping. Returns a map
   {:view :on-key :start! :stop! :replay! :step! :index} closing over a private
   frame-index r/atom; rendering delegates to frames-view, so this stays above
   the seam — it knows nothing of how frames are produced. `frames-fn` is a thunk
   returning the current [Frame], which makes upstream resampling transparent
   (call :replay! after a resample to rewind to frame 0).

     :on-key  :space advances one frame, :replay (r) rewinds to frame 0
     :start!  begin autoplay (no-op if already running or :interval-ms is nil)
     :stop!   halt autoplay
     :view    a reagent component rendering the current frame + step line

   opts:
     :frames-fn   thunk -> current [Frame]                       (required)
     :interval-ms autoplay period in ms, default 450; nil disables autoplay
     :status-fn   (fn [i n] -> string) extra step-line text       (optional)"
  [{:keys [frames-fn interval-ms status-fn] :or {interval-ms 450}}]
  (let [idx     (r/atom 0)
        timer   (atom nil)
        n-now   #(max 1 (count (frames-fn)))
        step!   (fn [] (swap! idx #(min (dec (n-now)) (inc %))))
        replay! (fn [] (reset! idx 0))]
    {:index   idx
     :step!   step!
     :replay! replay!
     :view    (fn [] [frames-view (frames-fn) idx status-fn])
     :on-key  (fn [k] (case k :space (step!) :replay (replay!) nil))
     :start!  (fn [] (when (and interval-ms (nil? @timer))
                       (reset! timer (js/setInterval step! interval-ms))))
     :stop!   (fn [] (when-let [t @timer]
                       (js/clearInterval t)
                       (reset! timer nil)))}))

(defn bars-view
  "PosteriorBars -> labeled horizontal bars; widths normalized to `width` cells.
   A bar carrying :highlight true (e.g. the known true goal) is drawn in green,
   bold, with a trailing marker — so the eye can check the posterior against truth."
  [{:keys [title bars]} & [width]]
  (let [w (or width 20)]
    [:> Box {:flexDirection "column"}
     [:> Text {:bold true} title]
     (for [{:keys [label weight highlight]} bars]
       ^{:key label}
       [:> Box {}
        [:> Text {:color (if highlight "greenBright" "gray") :bold (boolean highlight)}
         (str label " ")]
        [:> Text {:color (if highlight "greenBright" "cyanBright")}
         (apply str (repeat (Math/round (* weight w)) "█"))]
        [:> Text (if highlight {:color "green" :bold true} {:dimColor true})
         (str " " (.toFixed weight 3) (when highlight "  ◄ true"))]])]))

(defn status-bar
  "A bordered one-line status strip: a title plus arbitrary right-hand status."
  [{:keys [title status]}]
  [:> Box {:borderStyle "round" :borderColor "cyan" :paddingX 1}
   [:> Text {:bold true :color "cyan"} (str " " title " ")]
   (when status [:> Text {:color "gray"} (str "  " status)])])
