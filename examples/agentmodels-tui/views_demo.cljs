(ns views-demo
  "Headless-ish smoke for the view primitives: renders grid-view, status-bar and
   bars-view against HAND-WRITTEN sample data (no inference, no keyboard), then
   unmounts and exits. Proves the view layer is a pure function of seam data and
   that the reagent -> Ink mount works, independently of the models.

   Run: bun run --bun nbb examples/agentmodels-tui/views_demo.cljs (from repo root
   via run.sh's classpath), or ./run.sh views (see README)."
  (:require ["ink" :refer [render Box Newline]]
            [reagent.core :as r]
            [views]))

;; A hand-written Frame (no MDP involved): 3x3, agent + wall + two goals + a path.
(def sample-frame
  {:W 3 :H 3 :meta {:step 2 :action :right}
   :cells [{:glyph "A" :role :goal}                 {:glyph "∘" :role :path}  {:glyph "·" :role :empty :value 0.6}
           {:glyph "·" :role :empty :value 0.2}     {:glyph "█" :role :wall}  {:glyph "@" :role :agent}
           {:glyph "·" :role :empty :value 0.1}     {:glyph "·" :role :empty :value 0.4} {:glyph "B" :role :goal}]})

(def sample-bars
  {:title "P(goal)" :bars [{:label "A" :weight 0.27}
                           {:label "B" :weight 0.73 :highlight true}]})

(defn app []
  [:> Box {:flexDirection "column" :padding 1}
   [views/status-bar {:title "views_demo" :status "hand-written sample data"}]
   [:> Newline]
   ;; the Ch 5 side-by-side layout: grid left, posterior bars right
   [:> Box {:flexDirection "row"}
    [:> Box {:marginRight 4} [views/grid-view sample-frame]]
    [:> Box {} [views/bars-view sample-bars 18]]]])

(render (r/as-element [app]))
;; render one frame, then exit cleanly in a non-TTY shell
(js/setTimeout (fn [] (js/process.exit 0)) 300)
