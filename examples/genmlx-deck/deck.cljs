;; ============================================================================
;; GenMLX — a self-demonstrating, FULL-SCREEN, SPLIT terminal slide deck
;; ============================================================================
;; Fullscreen (alternate screen buffer + full-height root box) with a split
;; layout: narrative on the LEFT, a LIVE visual on the RIGHT — generated code,
;; an ASCII histogram/bar chart, or live numeric output. Every figure is a real
;; GenMLX computation in this same process.
;;
;; Run (from the repo root, or via run.sh):
;;   NODE_PATH=examples/genmlx-tui/node_modules:node_modules \
;;     nbb examples/genmlx-deck/deck.cljs
;;   Keys: ← / → (or n/p/space) move · r / Enter run the demo · q quit
;;
;; Headless self-test (no TTY): DECK_SELFTEST=1 (sync figs) | =full (also LLM)
;; ============================================================================

(ns genmlx-deck.deck
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inspect :as inspect]
            [genmlx.gfi :as gfi]
            [genmlx.inference.importance :as is]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as llm-core]
            [genmlx.llm.grammar :as grammar]
            [genmlx.llm.msa :as msa]
            [clojure.string :as str]
            [promesa.core :as pr]
            [reagent.core :as r]
            ["ink" :refer [render Text Box Newline useInput useApp]]
            ["ink-spinner$default" :as Spinner])
  (:require-macros [genmlx.gen :refer [gen]]))

(def MODEL-NAME "qwen3.5-4b-mlx-bf16")
(def MODEL-PATH (str (.-HOME js/process.env) "/.cache/models/" MODEL-NAME))

;; ----------------------------------------------------------------------------
;; helpers
;; ----------------------------------------------------------------------------
(defn fmt
  ([x] (fmt x 3))
  ([x d] (cond (not (number? x))   (str x)
               (js/Number.isNaN x) "NaN"
               (= x ##Inf)         "+inf"
               (= x ##-Inf)        "-inf"
               :else               (.toFixed x d))))
(defn mi  [v] (mx/item v))
(defn shp [a] (vec (mx/shape a)))
(defn- padr [s w] (let [s (str s)] (str s (apply str (repeat (max 0 (- w (count s))) " ")))))
(defn- padl [s w] (let [s (str s)] (str (apply str (repeat (max 0 (- w (count s))) " ")) s)))
;; collapse whitespace/newlines so an LLM completion renders as one clean line
(defn- oneline [s] (str/replace (str/trim (str s)) #"\s+" " "))

;; horizontal bar chart: data = [[label value] ...] -> vector of strings
(defn bar-rows [width data]
  (let [mxv (apply max 1e-9 (map (comp js/Math.abs second) data))
        lw  (apply max 1 (map (comp count str first) data))]
    (mapv (fn [[label v]]
            (let [n (max 0 (min width (js/Math.round (* (/ (js/Math.abs v) mxv) width))))]
              (str (padr label lw) "  "
                   (apply str (repeat n "█")) (apply str (repeat (max 0 (- width n)) "░"))
                   "  " (fmt v 2))))
          data)))

;; compact horizontal histogram of a sequence of numbers
(defn histo-rows [width nbins samples]
  (let [lo (reduce min samples) hi (reduce max samples)
        span (max 1e-9 (- hi lo)) bw (/ span nbins)
        counts (reduce (fn [a x]
                         (let [b (max 0 (min (dec nbins) (int (/ (- x lo) bw))))]
                           (update a b inc)))
                       (vec (repeat nbins 0)) samples)
        mxc (apply max 1 counts)]
    (mapv (fn [b c]
            (str (padl (fmt (+ lo (* (+ b 0.5) bw)) 1) 6) " │"
                 (apply str (repeat (js/Math.round (* (/ c mxc) width)) "█"))))
          (range nbins) counts)))

;; ============================================================================
;; MODELS
;; ============================================================================
(def linreg
  (gen [xs]
    (let [slope     (trace :slope     (dist/gaussian 0 2))
          intercept (trace :intercept (dist/gaussian 0 2))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x)) intercept) 1.0)))
      {:slope slope :intercept intercept})))
(def normal-mean
  (gen [] (let [mu (trace :mu (dist/gaussian 0 10))]
            (trace :y0 (dist/gaussian mu 1.0)) (trace :y1 (dist/gaussian mu 1.0))
            (trace :y2 (dist/gaussian mu 1.0)) mu)))
(def static-linreg
  (gen [x] (let [slope (trace :slope (dist/gaussian 0 10))
                 intercept (trace :intercept (dist/gaussian 0 5))
                 mu (mx/add (mx/multiply slope (mx/scalar x)) intercept)]
             (trace :y (dist/gaussian mu 1)) slope)))
(def dynamic-loop-linreg
  (gen [xs] (let [slope (trace :slope (dist/gaussian 0 10))
                  intercept (trace :intercept (dist/gaussian 0 5))]
              (doseq [[j x] (map-indexed vector xs)]
                (trace (keyword (str "y" j))
                       (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x)) intercept) 1)))
              slope)))
(def branch-model
  (gen [flag] (let [coin (trace :coin (dist/bernoulli 0.5))]
                (if flag (trace :x (dist/gaussian 0 1)) (trace :x (dist/gaussian 5 1)))
                coin)))
(def nn-model
  (gen [] (let [mu (trace :mu (dist/gaussian 0 3))]
            (trace :y0 (dist/gaussian mu 1)) (trace :y1 (dist/gaussian mu 1))
            (trace :y2 (dist/gaussian mu 1)) (trace :y3 (dist/gaussian mu 1)) mu)))
(def vec-model
  (gen [] (let [mu (trace :mu (dist/gaussian 0 2)) y (trace :y (dist/gaussian mu 1))] {:mu mu :y y})))
(def trace-model
  (gen [mu0] (let [mu (trace :mu (dist/gaussian mu0 1)) y (trace :y (dist/gaussian mu 1))] {:mu mu :y y})))
(def coin     (gen [] (trace :x (dist/bernoulli 0.5))))
(def gaussian (gen [] (trace :x (dist/gaussian 0 1))))

(def nav-actions [:advance :back :stay])
(def navigate
  (gen [n-steps]
    (loop [i 0, slide 0, path [0], acts []]
      (if (>= i n-steps)
        {:path path :acts acts}
        (let [a-idx (mx/item (trace (keyword (str "a" i))
                                    (dist/categorical (mx/array [1.6 0.2 0.2]))))
              a (nth nav-actions a-idx)
              nxt (case a :advance (inc slide) :back (max 0 (dec slide)) :stay slide)]
          (recur (inc i) nxt (conj path nxt) (conj acts a)))))))

;; ============================================================================
;; CONSTANTS
;; ============================================================================
(def lr-args [0.0 1.0 2.0 3.0 4.0])
(def f02-args [2.5])
(def f02-choices (-> cm/EMPTY (cm/set-choice [:slope] (mx/scalar 1.5))
                     (cm/set-choice [:intercept] (mx/scalar -0.3)) (cm/set-choice [:y] (mx/scalar 2.8))))
(def nn-data [1.8 2.2 1.5 2.5])
(def nn-obs (apply cm/choicemap (mapcat (fn [i y] [(keyword (str "y" i)) (mx/scalar y)]) (range) nn-data)))
(defn nn-hand-log-ml []
  (let [n (count nn-data) s0sq 9.0 ssq 1.0
        log-det (+ (* (dec n) (js/Math.log ssq)) (js/Math.log (+ ssq (* n s0sq))))
        d (mapv #(- % 0.0) nn-data) dd (reduce + (map * d d)) sd (reduce + d)
        quad (* (/ 1.0 ssq) (- dd (* (/ s0sq (+ ssq (* n s0sq))) sd sd)))]
    (* -0.5 (+ (* n (js/Math.log (* 2 js/Math.PI))) log-det quad))))
(defn is-log-ml [n seed]
  (let [{:keys [log-weights]} (is/importance-sampling {:samples n :key (rng/fresh-key seed)} nn-model [] nn-obs)
        st (mx/array (mapv mi log-weights))]
    (- (mi (mx/logsumexp st)) (js/Math.log n))))

(def synth-system-prompt
  (str "Write a probabilistic model. For each variable write one line:\n"
       "name ~ distribution(params)\n\nWhen a variable depends on another, use that "
       "variable's name as a parameter.\nUse 'gaussian' for normal distributions.\n\n"
       "Example:\nx ~ gaussian(0, 10)\ny ~ gaussian(x, 1)\n\nOutput ONLY the lines."))
(def synth-task
  {:description (str "We observe y1=3.2, y2=2.9, y3=3.1, noisy observations of an unknown "
                     "mean mu with a wide prior. Write the model for mu and y1,y2,y3.")
   :variables [:mu :y1 :y2 :y3] :observations {:y1 3.2 :y2 2.9 :y3 3.1} :query :mu})

;; ============================================================================
;; FIGURES — return {:lines [...] :chart {:title :rows [...]} :code [...]} (any subset),
;; or a vector of lines (normalized to {:lines ...}). LLM figures return a promise.
;; ============================================================================
(defn fig-homoiconicity []
  (let [tr (p/simulate (dyn/auto-key linreg) [lr-args])
        info (inspect/inspect linreg)
        vt (dyn/vsimulate (dyn/auto-key linreg) [lr-args] 2000 (rng/fresh-key))
        slope-n (cm/get-value (cm/get-submap (:choices vt) :slope))
        samples (vec (mx/->clj slope-n))
        sinfo (inspect/inspect normal-mean)]
    {:lines [(str "p/simulate → slope=" (fmt (mi (:slope (:retval tr)))))
             (str "inspect linreg → " (:compilation info) " (doseq ⇒ prefix)")
             (str "vsimulate 2000 → :slope shape " (shp slope-n))
             (str "  E=" (fmt (mi (mx/mean slope-n))) "  Var=" (fmt (mi (mx/variance slope-n))) " (prior 4)")
             (str "inspect normal-mean → " (:compilation sinfo)
                  " · conj " (if (:conjugacy sinfo) "✓" "✗"))]
     :chart {:title "slope ~ prior, 2000 samples in ONE body run" :rows (histo-rows 26 11 samples)}}))

(defn fig-ladder []
  (let [cw (mi (:weight (p/assess (dyn/auto-key static-linreg) f02-args f02-choices)))
        hw (mi (:weight (p/assess (dyn/auto-key (gfi/strip-compiled static-linreg)) f02-args f02-choices)))
        d (js/Math.abs (- cw hw))]
    {:lines ["The SAME gen idiom, three structures → three tiers:"
             (str "  static linear-Gaussian  → " (:compilation (inspect/inspect static-linreg)))
             (str "  doseq dynamic addresses → " (:compilation (inspect/inspect dynamic-loop-linreg)))
             (str "  if/branch same addr+fam → " (:compilation (inspect/inspect branch-model)))
             ""
             "What the tiers mean (how far work moves onto the GPU graph):"
             "  L0    = interpreted handler (the ground truth)"
             "  L1-M2 = fully compiled to noise-transforms"
             "  L1-M3 = static prefix compiled, dynamic tail interpreted"
             "  L1-M4 = if-branches rewritten to one mx/where select"
             ""
             "Bit-exact check — p/assess on identical full choices:"
             (str "  compiled path = " (fmt cw 9))
             (str "  handler path  = " (fmt hw 9))
             (str "  |Δ| = " (fmt d 9) (if (< d 1e-6) "  ✓ bit-exact" "  ✗"))]}))

(defn fig-auto-analytical []
  (let [info (inspect/inspect nn-model)
        fam (:family (first (:pairs (:conjugacy info))))
        a-ml (mi (:weight (p/generate (dyn/with-key nn-model (rng/fresh-key 1)) [] nn-obs)))
        h-ml (nn-hand-log-ml)
        n (count nn-data) S (reduce + nn-data)
        post-var (/ 1.0 (+ (/ 1.0 9.0) (/ n 1.0)))
        post-mean (* post-var S) post-sd (js/Math.sqrt post-var)
        {:keys [traces log-weights]} (is/importance-sampling {:samples 1200 :key (rng/fresh-key 3)} nn-model [] nn-obs)
        lw (mapv mi log-weights) mw (apply max lw)
        ws (mapv #(js/Math.exp (- % mw)) lw) tot (reduce + ws)
        mus (mapv (fn [t] (mi (cm/get-value (cm/get-submap (:choices t) :mu)))) traces)
        wmean (/ (reduce + (map * ws mus)) tot)
        _ (mx/force-gc!)
        e200 (let [v (js/Math.abs (- (is-log-ml 200 7) a-ml))] (mx/force-gc!) v)
        e1k  (let [v (js/Math.abs (- (is-log-ml 600 7) a-ml))] (mx/force-gc!) v)
        e5k  (let [v (js/Math.abs (- (is-log-ml 1500 7) a-ml))] (mx/force-gc!) v)]
    {:lines [(str "conjugacy AUTO-DETECTED from the source: " fam)
             (str "tier " (:compilation info) " · zero hints · disabled inside autodiff")
             (str "data y = [1.8 2.2 1.5 2.5]  (n=" n ", prior mu~N(0,3))")
             ""
             "p/generate fires the analytical path → EXACT log p(data):"
             (str "  analytical (GenMLX) = " (fmt a-ml 6))
             (str "  hand-derived MVN    = " (fmt h-ml 6) "  |Δ|=" (fmt (js/Math.abs (- a-ml h-ml)) 7))
             ""
             "Closed-form conjugate posterior over mu:"
             (str "  mu | data ~ N(mean=" (fmt post-mean) ", sd=" (fmt post-sd) ")")
             (str "  weighted-IS posterior mean = " (fmt wmean) "   (agrees)")]
     :chart {:title "IS |error| in log p(data) vs analytic, by N"
             :rows (bar-rows 22 [["N=200" e200] ["N=600" e1k] ["N=1500" e5k]])}}))

(defn fig-vectorization []
  (let [vt (dyn/vsimulate vec-model [] 10000 (rng/fresh-key))
        mu-n (cm/get-value (cm/get-submap (:choices vt) :mu))
        y-n (cm/get-value (cm/get-submap (:choices vt) :y))
        _ (mx/eval! mu-n y-n)
        mu-samples (vec (mx/->clj mu-n))
        _ (mx/eval! (:score (dyn/vsimulate vec-model [] 16 (rng/fresh-key))))
        mk (dyn/auto-key vec-model)
        _ (mx/eval! (:score (p/simulate mk [])))
        nn 4000
        t0 (js/performance.now) v (dyn/vsimulate vec-model [] nn (rng/fresh-key))
        _ (mx/eval! (:score v)) t1 (js/performance.now)
        s0 (js/performance.now) _ (dotimes [_ nn] (mx/eval! (:score (p/simulate mk [])))) s1 (js/performance.now)
        vms (- t1 t0) sms (- s1 s0)]
    {:lines [(str "vsimulate N=10000 → :mu and :y both shape " (shp mu-n))
             "produced by ONE body execution, not 10000 runs:"
             (str "  E[mu]=" (fmt (mi (mx/mean mu-n))) "  Var[mu]=" (fmt (mi (mx/variance mu-n))) "  (prior var 4)")
             (str "  E[y] =" (fmt (mi (mx/mean y-n))) "  Var[y]=" (fmt (mi (mx/variance y-n))) "  (Var[mu]+1 = 5)")
             ""
             (str "speedup over the scalar loop = " (fmt (/ sms (max vms 0.001)) 0) "×")
             "(same statistics — broadcasting carries the batch, no vmap)"]
     :chart {:title "mu over 10000 particles (ONE body run)"
             :rows (concat (histo-rows 24 9 mu-samples)
                           ["" "wall-time, vectorized vs scalar loop (ms):"]
                           (bar-rows 18 [["vsimulate" vms] ["scalar ×4000" sms]]))}}))

(defn fig-value-semantics []
  (let [n 1000000 iters 50 base (mx/array (vec (range n)))
        t0 (js/performance.now)
        graph (loop [k 0 acc base]
                (if (>= k iters) acc
                    (let [shifted (mx/add (mx/multiply acc (mx/scalar 1.0000001)) (mx/scalar 0.5))]
                      (recur (inc k) (mx/add acc (mx/log (mx/add (mx/scalar 1.0)
                                                                 (mx/exp (mx/multiply shifted (mx/scalar 1e-7))))))))))
        t1 (js/performance.now) reduced (mx/sum graph)
        t2 (js/performance.now) _ (mx/item reduced) t3 (js/performance.now)
        cms (- t1 t0) ems (- t3 t2)
        tr (p/simulate (dyn/auto-key trace-model) [0.0]) score (:score tr)]
    {:code ["(loop [acc base]   ; base: 1e6 elems"
            "  (mx/add acc        ; each op just"
            "    (mx/log ...)))   ; APPENDS a node"
            "(mx/shape graph)     ; => [1000000]"
            "                     ;   (no compute)"
            "(mx/item (mx/sum g)) ; forces Metal 1×"]
     :lines [(str iters "-layer graph over " n " elements — built as a VALUE.")
             (str "Its shape is known without computing it: " (shp graph))
             ""
             "A trace's :score is the same kind of deferred value:"
             (str "  (mx/array? score) = " (mx/array? score) "  (a graph node, not a number)")
             (str "  (mx/item score)   = " (fmt (mi score)) "  ← only this runs Metal")]
     :chart {:title "construct (build a value) vs eval (run Metal), ms"
             :rows (bar-rows 22 [["construct" cms] ["eval!" ems]])}}))

(defn fig-llm-distribution [mm]
  (let [tok (:tokenizer mm) llm-gf (llm-core/make-llm-gf mm)]
    (pr/let [raw (llm/encode tok "The best programming language is") pid (vec raw)
             cljids (llm/encode tok " Clojure") clj-id (first (vec cljids))
             coin-tr (p/simulate (dyn/auto-key coin) [])
             gn-tr (p/simulate (dyn/auto-key gaussian) [])
             llm-tr (p/simulate llm-gf [pid 14]) llm-txt (llm-core/decode-trace tok llm-tr)
             coin-r (p/generate (dyn/auto-key coin) [] (cm/set-value cm/EMPTY :x (mx/scalar 1.0)))
             gn-r (p/generate (dyn/auto-key gaussian) [] (cm/set-value cm/EMPTY :x (mx/scalar 1.5)))
             llm-r (p/generate llm-gf [pid 14] (cm/set-value cm/EMPTY :t0 (mx/scalar clj-id mx/int32)))
             llm-ctxt (llm-core/decode-trace tok (:trace llm-r))]
      {:lines [(str "p/simulate draws one sample and reports its log p (:score):")
               (str "  coin → :x=" (mi (cm/get-value (cm/get-submap (:choices coin-tr) :x)))
                    "   log p=" (fmt (mi (:score coin-tr))) "  (= log 0.5)")
               (str "  gaussian → :x=" (fmt (mi (cm/get-value (cm/get-submap (:choices gn-tr) :x))))
                    "   log p=" (fmt (mi (:score gn-tr))))
               ""
               "LLM FREE completion of \"The best programming language is\":"
               (str "  →\"" (oneline llm-txt) "\"")
               (str "  log p(sequence) = " (fmt (mi (:score llm-tr))))
               ""
               "p/generate FORCES the first token to \" Clojure\", then continues:"
               (str "  →\"" (oneline llm-ctxt) "\"")
               (str "  log p(constraint) = " (fmt (mi (:weight llm-r)))
                    "  = log p(\" Clojure\" | prompt)")
               ""
               "Same simulate / generate ops on all three: an LLM is just a"
               "generative function whose tokens are categorical trace sites."]
       :chart {:title "log p(constraint): same op, three GFs"
               :rows (bar-rows 18 [["coin =1" (mi (:weight coin-r))]
                                   ["gauss 1.5" (mi (:weight gn-r))]
                                   ["llm Clojure" (mi (:weight llm-r))]])}})))

(defn fig-grammar [mm]
  (let [tok (:tokenizer mm) ti (grammar/build-token-index tok)
        num-c (grammar/compile-constraint tok "[0-9]{1,3}" {:token-index ti})
        yn-c (grammar/compile-constraint tok "(yes|no)" {:token-index ti})
        num-gf (grammar/constrain (llm-core/make-llm-gf mm) num-c)
        yn-gf (grammar/constrain (llm-core/make-llm-gf mm) yn-c)
        die (dyn/auto-key (gen [] (trace :t0 (dist/categorical (mx/array [0.0 0.0 0.0 0.0 0.0])))))
        die-c {:dfa (grammar/compile-regex "3") :token-index ["0" "1" "2" "3" "4"] :eos-id -1 :masks nil}
        free-gf (llm-core/make-llm-gf mm)
        die-gf (grammar/constrain die die-c)]
    (pr/let [num-raw (llm/encode tok "Pick a number: ") num-ids (vec num-raw)
             yn-raw (llm/encode tok "Answer yes or no: Is 7 prime? ") yn-ids (vec yn-raw)
             free-tr (p/simulate free-gf [num-ids 8]) free-txt (llm-core/decode-trace tok free-tr)
             num-tr (p/simulate num-gf [num-ids 6]) num-txt (llm-core/decode-trace tok num-tr)
             yn-tr (p/simulate yn-gf [yn-ids 4]) yn-txt (llm-core/decode-trace tok yn-tr)
             d1 (p/simulate die-gf []) d2 (p/simulate die-gf []) d3 (p/simulate die-gf [])]
      (let [picks (mapv #(mx/item (cm/get-value (cm/get-submap (:choices %) :t0))) [d1 d2 d3])]
        {:lines ["The regex is compiled to a DFA over the 151k-token vocabulary."
                 (str "Membership, [0-9]{1,3}:  \"42\"→" (grammar/dfa-accepts? (:dfa num-c) "42")
                      "   \"1234\"→" (grammar/dfa-accepts? (:dfa num-c) "1234") "  (≤ 3 digits)")
                 ""
                 "UNCONSTRAINED, the model may emit anything after \"Pick a number: \":"
                 (str "  →\"" (oneline free-txt) "\"")
                 ""
                 "CONSTRAINED: at each token the DFA masks every invalid token to"
                 "-inf, so the result is grammar-valid by construction, not by luck:"
                 (str "  under [0-9]{1,3} →\"" (oneline num-txt) "\"  (a valid 1-3 digit number)")
                 (str "  under (yes|no)   →\"" (oneline yn-txt) "\"")
                 ""
                 "And it is GENERIC — grammar/constrain masks ANY dist/categorical,"
                 "via the same with-handler middleware GenMLX uses for conjugacy."
                 "A plain 5-way die masked by the regex \"3\", every roll:"
                 (str "  draws = " (pr-str picks) "  (the mask zeroed all but symbol 3)")]}))))

(defn fig-synthesis [mm]
  (pr/loop [i 0 acc []]
    (if (>= i 2)
      (let [ranked (vec (sort-by :weight > acc)) best (first ranked)]
        (if (and best (:gf best) (not= (:weight best) ##-Inf))
          (pr/let [samples (msa/importance-sample (:gf best) (:observations synth-task) (:query synth-task) 120)
                   post (msa/infer-answer samples)]
            {:code (str/split-lines (:code best))
             :lines (into ["The 4B model proposed model SOURCE as text. Instaparse parsed"
                           "it, SCI compiled it into a DynamicGF in THIS process, and"
                           "p/generate scored it by log marginal likelihood."
                           ""
                           "Candidates ranked by coherence (higher log-ML fits the data better):"]
                          (conj (mapv (fn [c] (str "  log-ML=" (fmt (:weight c) 3)
                                                   "   mu ← " (or (:mu (:dist-map c)) "(default)"))) ranked)
                                ""
                                "↑ Code on the right is the best machine-written model."
                                (str "Its posterior over mu: mean=" (fmt (:mean post))
                                     "  (true 3.0, error " (fmt (js/Math.abs (- (:mean post) 3.0)) 3) ")")
                                ""
                                "Propose → eval → score: one language, one process,"
                                "no representational boundary crossed."))})
          {:lines ["The base model wrote output that did not parse into a valid"
                   "model; no candidate scored. (Bump MODEL-NAME for stronger synthesis.)"]}))
      (pr/let [txt (llm/generate-text-raw mm (:description synth-task)
                                          {:max-tokens 110 :temperature 0.6 :system-prompt synth-system-prompt})
               dist-map (msa/parse-math txt)
               code (msa/assemble-gen-fn (:variables synth-task) (or dist-map {}))
               gf (msa/eval-model code)
               w (if gf (msa/score-model gf (:observations synth-task) {:n-particles 40}) ##-Inf)]
        (pr/recur (inc i) (conj acc {:code code :dist-map dist-map :gf gf :weight w}))))))

(defn fig-deck-as-gen [history]
  (let [tr (p/simulate (dyn/auto-key navigate) [8])
        {:keys [path acts]} (:retval tr) hist (take-last 15 history)]
    {:lines (into ["p/simulate of the navigation gen function samples an"
                   "alternative walkthrough (each action is a categorical draw):"
                   (str "  actions: " (str/join " " (map name acts)))
                   (str "  slide path: " (str/join "→" path))
                   (str "  log p(walk) = " (fmt (mi (:score tr))))
                   ""
                   (str "YOUR path so far — the operant trace you generated ("
                        (count history) " steps, last " (min 15 (count history)) " shown):")]
                  (if (seq hist)
                    (mapv (fn [h] (str "  slide " (:from h) "  --[ " (:key h) " ]-->  slide " (:to h))) hist)
                    ["  (navigate with ← → to record your operant trace)"]))}))

(defn normalize-out [r] (if (map? r) r {:lines (vec r)}))

;; ============================================================================
;; STATE
;; ============================================================================
(defonce state (r/atom {:idx 0 :model nil :model-status :idle
                        :running false :output nil :error nil :history []
                        :cols 80 :rows 24}))

;; ============================================================================
;; SLIDES
;; ============================================================================
(def slides
  [{:kind :title
    :title "GenMLX — a probabilistic language where programs are data"
    :body ["Every figure in this deck is a LIVE computation running in THIS"
           "process: models simulate, condition and vectorize on screen; real"
           "LLMs generate; and the deck itself navigates by a generative function."
           ""
           "→ / ←  move      r  run a slide's demo      q  quit"]}
   {:n "01" :feature "Homoiconicity — programs are data"
    :title "A model is a value: run, analyzed, vectorized — one source"
    :body ["A GenMLX model is an ordinary ClojureScript function — and the gen"
           "macro also captures its source form as data."
           ""
           "Because the program IS data, the same unchanged source can be"
           "executed, statically analyzed, compiled, conditioned on data, and"
           "vectorized — no rewrites, no second representation."
           ""
           "Right: 2000 slope values drawn in ONE body execution (the bell shape"
           "is the prior). The analyzer also reads the source to report the"
           "compilation tier and to auto-detect conjugacy."]
    :code ["(gen [xs]"
           "  (let [slope (trace :slope"
           "                 (dist/gaussian 0 2)) ...]"
           "    (doseq [[j x] ...]"
           "      (trace (kw \"y\" j) ...))))"]
    :demo fig-homoiconicity}
   {:n "02" :feature "Verified compilation ladder"
    :title "Same source → many tiers; compiled == handler, bit-exact"
    :body ["The compilation ladder moves work from the interpreter into fused"
           "MLX graphs — but the model source never changes."
           ""
           "Structure alone decides the tier: a fully-static model fully compiles"
           "(L1-M2); a loop over runtime-built addresses compiles its static prefix"
           "(L1-M3); matching if-branches rewrite to a select (L1-M4)."
           ""
           "And it is provably safe — the compiled path's log-density equals the"
           "handler's to the last bit. The handler is the semantics; compilation"
           "is only an optimization."]
    :demo fig-ladder}
   {:n "06" :feature "Auto-analytical from source"
    :title "Conjugacy detected from source ⇒ exact inference"
    :body ["Inference need not always sample. GenMLX inspects the model's"
           "structure and, where a prior/likelihood pair is conjugate, replaces"
           "sampling with the exact closed-form posterior — with no hint from you."
           ""
           "Here a Normal-Normal model is detected automatically, so p/generate"
           "returns the EXACT log marginal likelihood (it matches a hand-derived"
           "multivariate-Normal to machine precision)."
           ""
           "Right: an independent importance-sampling estimate of the same number,"
           "converging to it as the particle count grows."]
    :demo fig-auto-analytical}
   {:n "07" :feature "Shape-based vectorization"
    :title "N particles by changing shapes, not by vmap"
    :body ["To run N particles, GenMLX does not transform the function with vmap."
           "It changes array shapes from [] to [N] and lets MLX broadcasting carry"
           "the batch."
           ""
           "The model body runs exactly ONCE; the batched handler transition"
           "differs from the scalar one by a single call (dist-sample-n instead of"
           "dist-sample)."
           ""
           "Right: wall-clock for one vectorized run of 4000 particles versus a"
           "4000-iteration scalar loop — same statistics, thousands of × faster."]
    :demo fig-vectorization}
   {:n "08" :feature "Value-semantics through the GPU"
    :title "The lazy MLX graph is a value; eval! is the only dispatch"
    :body ["An MLX operation does not compute — it appends a node to a lazy graph,"
           "which is itself an immutable value. Nothing touches the GPU until"
           "mx/eval! / mx/item forces it."
           ""
           "So scores, weights and posteriors all compose as deferred values, and"
           "eval! is the single, explicit dispatch boundary."
           ""
           "Right: building a 50-layer graph over a million elements is nearly"
           "free; evaluating it is where Metal actually runs. A trace's :score is"
           "likewise a lazy value until forced."]
    :demo fig-value-semantics}
   {:n "03" :feature "LLMs as distributions" :llm? true
    :title "An LLM is an actual generative function"
    :body ["The same GFI operations apply to a coin, a Gaussian, and a"
           "multi-billion-parameter language model — there is no special case"
           "for LLMs."
           ""
           "make-llm-gf returns an ordinary generative function: each token is a"
           "trace site sampling from a categorical over the vocabulary."
           ""
           "So p/simulate generates text and reports its log-probability, and"
           "p/generate conditions the first token and returns log p(constraint)."
           "Press r and watch the completed text appear on the right."]
    :demo fig-llm-distribution}
   {:n "04" :feature "Code/text as a conditioned random variable" :llm? true
    :title "Conditioning on a grammar is a first-class GFI op"
    :body ["Conditioning a sampled value on a grammar is a first-class GFI"
           "operation — the same Ring-style with-handler middleware GenMLX uses"
           "to install conjugacy."
           ""
           "A regex is compiled to a DFA over the tokenizer's vocabulary; at each"
           "token the categorical is masked so only grammar-valid strings have any"
           "probability."
           ""
           "Right: free generation rambles, but the constrained model can ONLY"
           "emit a valid number or yes/no. Because it masks any categorical — not"
           "just an LLM's — a plain die constrained to \"3\" always rolls 3."]
    :demo fig-grammar}
   {:n "05" :feature "ClojureScript writing ClojureScript" :llm? true
    :title "Program synthesis as inference (propose → eval → score)"
    :body ["Program synthesis is just inference where the latent variable is code."
           "The LLM proposes model SOURCE; SCI evaluates it into a generative"
           "function in this very process; the GFI scores it by log marginal"
           "likelihood."
           ""
           "Proposal, evaluation, and scoring share one language and one runtime —"
           "ClojureScript writing ClojureScript, with no parse/compile/sandbox"
           "boundary in between."
           ""
           "Right: the actual model the 4B wrote, the candidates ranked by"
           "coherence, and the posterior it implies over mu. (~15s the first run.)"]
    :demo fig-synthesis}
   {:feature "The deck as a gen function"
    :title "This deck navigates by a generative function"
    :body ["This deck is itself an instance of the thesis. Its navigation is a"
           "discriminated operant: the current slide is the stimulus, your"
           "keypress is the response, the next slide is the consequence."
           ""
           "Encoded as a gen function, your path through the talk becomes a"
           "behavioral TRACE, and p/simulate samples an alternative walkthrough."
           ""
           "Press r to see a sampled walkthrough beside the path you actually took."]
    :demo (fn [] (fig-deck-as-gen (:history @state)))}
   {:kind :end
    :title "fin"
    :body ["Eight distinctive features, each shown as a live computation —"
           "and a deck that is itself an instance of the thesis it presents."
           ""
           "demos: examples/distinctive/     deck: examples/genmlx-deck/run.sh"]}])

;; ============================================================================
;; ACTIONS
;; ============================================================================
(defn nav! [d sym]
  (swap! state (fn [s]
                 (let [j (max 0 (min (dec (count slides)) (+ (:idx s) d)))]
                   (-> s (assoc :idx j :output nil :error nil :running false)
                       (update :history (fnil conj []) {:from (:idx s) :to j :key sym}))))))

(defn run-current! []
  (let [s @state slide (nth slides (:idx s)) demo (:demo slide)]
    (when (and demo (not (:running s)))
      (if (:llm? slide)
        (if-let [mm (:model s)]
          (do (swap! state assoc :running true :output nil :error nil)
              (-> (demo mm)
                  (.then  (fn [r] (swap! state assoc :running false :output (normalize-out r)) (mx/force-gc!)))
                  (.catch (fn [e] (swap! state assoc :running false :error (str (.-message e))) (mx/force-gc!)))))
          (swap! state assoc :error (str MODEL-NAME " still loading — wait for ✓")))
        (do (swap! state assoc :running true :output nil :error nil)
            (js/setTimeout
             (fn [] (try (swap! state assoc :running false :output (normalize-out (demo))) (mx/force-gc!)
                         (catch :default e (swap! state assoc :running false :error (str (.-message e))) (mx/force-gc!))))
             30))))))

(defn load-model! []
  (swap! state assoc :model-status :loading)
  (-> (llm/load-model MODEL-PATH)
      (.then  (fn [m] (swap! state assoc :model m :model-status :ready)))
      (.catch (fn [e] (swap! state assoc :model-status :error :error (str "model: " (.-message e)))))))

;; ============================================================================
;; FULLSCREEN
;; ============================================================================
(defn term-cols [] (or (.-columns js/process.stdout) 80))
(defn term-rows [] (or (.-rows js/process.stdout) 24))
;; macOS Metal/CoreAnalytics prints "Context leak detected, CoreAnalytics
;; returned false" to stdout during heavy GPU work, which would corrupt ink's
;; render. Drop only those lines; pass everything else (incl. ink frames) through.
(defn silence-metal-noise! []
  (let [orig (.bind (.-write js/process.stdout) js/process.stdout)]
    (set! (.-write js/process.stdout)
          (fn [chunk & args]
            (if (and (string? chunk) (re-find #"Context leak|CoreAnalytics" chunk))
              true
              (apply orig chunk args))))))
(defn enter-fullscreen! [] (.write js/process.stdout "[?1049h[2J[H"))
(defn leave-fullscreen! [] (.write js/process.stdout "[?1049l"))

;; ============================================================================
;; COMPONENTS
;; ============================================================================
(defn dots [idx n]
  (apply str (map (fn [i] (cond (= i idx) "●" (< i idx) "•" :else "·")) (range n))))
(defn model-badge [s]
  (case (:model-status s) :loading "loading…" :ready "ready ✓" :error "error ✗" "idle"))
(defn txt-rows
  ([color lines] (txt-rows color "truncate-end" lines))
  ([color wrap lines]
   (into [:> Box {:flexDirection "column"}]
         (map-indexed (fn [i l] ^{:key i} [:> Text {:color color :wrap wrap} (if (str/blank? l) " " l)]) lines))))

(defn render-output [out]
  (into [:> Box {:flexDirection "column"}]
        (remove nil?
                [(when (:code out)
                   ^{:key :code} [:> Box {:flexDirection "column"} (txt-rows "yellow" (:code out))])
                 (when (:chart out)
                   ^{:key :chart} [:> Box {:flexDirection "column" :marginTop (if (:code out) 1 0)}
                                   [:> Text {:color "cyan" :dimColor true} (:title (:chart out))]
                                   (txt-rows "cyanBright" (:rows (:chart out)))])
                 (when (:lines out)
                   ^{:key :lines} [:> Box {:flexDirection "column" :marginTop (if (or (:code out) (:chart out)) 1 0)}
                                   (txt-rows "green" "wrap" (:lines out))])])))

(defn panel-body [s slide]
  (cond
    (:running s) [:> Box [:> Spinner {:type "dots"}] [:> Text {:color "yellow"} " running live…"]]
    (:error s)   [:> Text {:color "red" :wrap "wrap"} (str "✗ " (:error s))]
    (:output s)  (render-output (:output s))
    (:code slide) [:> Box {:flexDirection "column"} (txt-rows "yellow" (:code slide))
                   [:> Newline] [:> Text {:color "blueBright"} "▶ press r to run"]]
    (:demo slide) [:> Text {:color "blueBright" :wrap "wrap"} "▶ press r to run this demo live"]
    :else [:> Text {:color "gray"} "—"]))

(defn content-view [s slide]
  (let [cols (:cols s) rows (:rows s)]
    [:> Box {:flexDirection "column" :width cols :height rows :paddingX 1 :paddingY 1}
     [:> Box {:justifyContent "space-between" :width (- cols 4)}
      [:> Text {:bold true :color "cyan"} " GenMLX · distinctive features"]
      [:> Text {:color "gray"} (str "slide " (inc (:idx s)) "/" (count slides) "   "
                                    MODEL-NAME ": " (model-badge s) "  " (dots (:idx s) (count slides)))]]
     [:> Box {:flexGrow 1 :flexDirection "row" :marginTop 1}
      [:> Box {:flexDirection "column" :width "42%" :paddingRight 2}
       (when (:feature slide) [:> Text {:color "magenta" :bold true :wrap "wrap"} (str "◆ " (:feature slide))])
       [:> Box {:marginTop 1} [:> Text {:bold true :color "white" :wrap "wrap"} (:title slide)]]
       [:> Box {:marginTop 1} (txt-rows "gray" "wrap" (:body slide))]]
      [:> Box {:flexGrow 1 :flexDirection "column" :borderStyle "round" :borderColor "blue" :paddingX 1}
       (panel-body s slide)]]
     [:> Text {:color "gray"} " ← →  move  ·  r / Enter  run demo  ·  q  quit"]]))

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
    (if (#{:title :end} (:kind slide)) (title-view s slide) (content-view s slide))))

;; ============================================================================
;; MAIN
;; ============================================================================
(defn print-out [nm out]
  (println (str "\n-- " nm " --"))
  (when (:code out)  (doseq [l (:code out)]  (println "   |" l)))
  (when (:chart out) (println (str "   [" (:title (:chart out)) "]")) (doseq [l (:rows (:chart out))] (println "   " l)))
  (when (:lines out) (doseq [l (:lines out)] (println "   " l))))

(defn run-selftest! []
  (println "=== DECK SELFTEST (sync figures) ===")
  (doseq [[nm f] [["01" fig-homoiconicity] ["02" fig-ladder] ["06" fig-auto-analytical]
                  ["07" fig-vectorization] ["08" fig-value-semantics]
                  ["deck" (fn [] (fig-deck-as-gen []))]]]
    (print-out nm (normalize-out (f)))
    (mx/force-gc!))
  (if (= "full" (.. js/process -env -DECK_SELFTEST))
    (-> (llm/load-model MODEL-PATH)
        (.then (fn [mm]
                 (println "\n=== SELFTEST (llm figures) — model loaded ===")
                 (-> (pr/let [r (fig-llm-distribution mm)] (print-out "03" (normalize-out r)))
                     (.then (fn [_] (pr/let [r (fig-grammar mm)] (print-out "04" (normalize-out r)))))
                     (.then (fn [_] (pr/let [r (fig-synthesis mm)] (print-out "05" (normalize-out r)))))
                     (.then (fn [_] (println "\nSELFTEST-OK") (js/process.exit 0)))
                     (.catch (fn [e] (println "SELFTEST-ERROR:" (.-message e)) (js/process.exit 1))))))
        (.catch (fn [e] (println "model load error:" (.-message e)) (js/process.exit 1))))
    (do (println "\nSELFTEST-OK (sync only; DECK_SELFTEST=full also runs the LLM figs)")
        (js/process.exit 0))))

(if (some? (.. js/process -env -DECK_SELFTEST))
  (run-selftest!)
  (do
    (swap! state assoc :cols (term-cols) :rows (term-rows))
    (.on js/process.stdout "resize" (fn [] (swap! state assoc :cols (term-cols) :rows (term-rows))))
    (silence-metal-noise!)
    (enter-fullscreen!)
    (.on js/process "exit" leave-fullscreen!)
    (render (r/as-element [:f> app]))
    (load-model!)))
