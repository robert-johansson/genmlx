(ns genmlx.program
  "Two-level GFI: structural inference over probabilistic programs.

   Outer level: a distribution over model structures (which causal edges exist).
   Inner level: for each structure, SCI compiles a gen function and p/generate
   scores it against data. The log marginal likelihood integrates out parameters.

   The key loop:
     edge structure → build source → SCI compile → DynamicGF → p/generate → score
   All in one process, all on the same GPU."
  (:require [sci.core :as sci]
            [clojure.string :as str]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.codegen :as codegen]
            [promesa.core :as pr])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ============================================================
;; SCI evaluation context
;; ============================================================

(def sci-opts
  "SCI options for evaluating generated transition model code.
   Exposes GenMLX distributions and MLX operations."
  {:namespaces
   {'dist {'gaussian dist/gaussian
           'exponential dist/exponential
           'uniform dist/uniform}
    'mx {'add mx/add
         'subtract mx/subtract
         'multiply mx/multiply
         'divide mx/divide
         'scalar mx/scalar
         'abs mx/abs
         'exp mx/exp
         'log mx/log}}})

;; ============================================================
;; Synthetic data generation
;; ============================================================

(defn- randn
  "Sample from N(0,1) via Box-Muller transform."
  []
  (let [u1 (js/Math.random)
        u2 (js/Math.random)]
    (* (js/Math.sqrt (* -2 (js/Math.log u1)))
       (js/Math.cos (* 2 js/Math.PI u2)))))

(defn generate-synthetic-data
  "Generate synthetic 2-variable time series with known causal structure.

   dgp: data-generating process specification
     :n-individuals  number of independent time series (default 50)
     :n-steps        number of time points per individual (default 10)
     :ar-x           AR(1) coefficient for X (default 0.8)
     :ar-y           AR(1) coefficient for Y (default 0.5)
     :beta-xy        cross-effect X→Y (default 0, set to e.g. -0.3)
     :beta-yx        cross-effect Y→X (default 0)
     :sigma-x        noise std for X (default 1.0)
     :sigma-y        noise std for Y (default 2.0)
     :x0-mean        initial X mean (default 5)
     :x0-std         initial X std (default 2)
     :y0-mean        initial Y mean (default 20)
     :y0-std         initial Y std (default 5)

   Returns a vector of individuals, each a vector of {:x :y} maps."
  [dgp]
  (let [{:keys [n-individuals n-steps ar-x ar-y beta-xy beta-yx
                sigma-x sigma-y x0-mean x0-std y0-mean y0-std]
         :or {n-individuals 50 n-steps 10
              ar-x 0.8 ar-y 0.5 beta-xy 0 beta-yx 0
              sigma-x 1.0 sigma-y 2.0
              x0-mean 5 x0-std 2 y0-mean 20 y0-std 5}} dgp]
    (vec
      (for [_ (range n-individuals)]
        (loop [t 0
               x (+ x0-mean (* x0-std (randn)))
               y (+ y0-mean (* y0-std (randn)))
               series [{:x x :y y}]]
          (if (>= t (dec n-steps))
            series
            (let [x-next (+ (* ar-x x) (* beta-yx y) (* sigma-x (randn)))
                  y-next (+ (* ar-y y) (* beta-xy x) (* sigma-y (randn)))]
              (recur (inc t) x-next y-next
                     (conj series {:x x-next :y y-next})))))))))

(defn extract-transitions
  "Extract transition pairs from time series data.
   Returns a flat vector of {:x-prev :y-prev :x-next :y-next} maps
   across all individuals."
  [data]
  (vec
    (for [individual data
          [prev nxt] (partition 2 1 individual)]
      {:x-prev (:x prev) :y-prev (:y prev)
       :x-next (:x nxt)  :y-next (:y nxt)})))

;; ============================================================
;; Model source construction
;; ============================================================

(defn- mean-expr
  "Build the mean expression string for a target variable's transition.
   target: :x or :y
   sources: set of source variables with active edges into target"
  [target sources]
  (let [ar-term (str "(mx/multiply ar-" (name target) " " (name target) "-prev)")]
    (if (empty? sources)
      ar-term
      (reduce (fn [acc src]
                (str "(mx/add " acc
                     " (mx/multiply beta-" (name src) "->" (name target)
                     " " (name src) "-prev))"))
              ar-term
              sources))))

(defn- param-bindings
  "Build the let-binding strings for model parameters.
   Returns a vector of binding strings."
  [var-names edges]
  (let [ar-bindings (mapv (fn [v]
                            (str "ar-" (name v)
                                 " (trace :ar-" (name v)
                                 " (dist/gaussian 0.5 0.15))"))
                          var-names)
        cross-bindings (vec
                         (for [target var-names
                               source var-names
                               :when (and (not= source target)
                                          (get edges [(name source) (name target)]))]
                           (str "beta-" (name source) "->" (name target)
                                " (trace :beta-" (name source) "->" (name target)
                                " (dist/gaussian 0 0.3))")))
        sigma-bindings (mapv (fn [v]
                               (str "sigma-" (name v)
                                    " (trace :sigma-" (name v)
                                    " (dist/exponential 1))"))
                             var-names)]
    (into [] (concat ar-bindings cross-bindings sigma-bindings))))

(defn build-transition-source
  "Build a transition model source string from variable names and edge structure.

   var-names: [:x :y]
   edges:     {[\"x\" \"y\"] true, [\"y\" \"x\"] false}
              keys are [source target] pairs

   Returns a string: (fn [trace transitions] ...) that when evaluated by SCI
   produces a function. The function traces parameters once, then traces
   observations for each transition in the data."
  [var-names edges]
  (let [bindings (param-bindings var-names edges)
        binding-str (str/join "\n        " bindings)
        body-lines
        (mapv (fn [v]
                (let [sources (set (for [src var-names
                                        :when (and (not= src v)
                                                   (get edges [(name src) (name v)]))]
                                    src))
                      mexpr (mean-expr v sources)]
                  (str "(trace (keyword (str \"" (name v) "\" i))\n"
                       "             (dist/gaussian " mexpr " sigma-" (name v) "))")))
              var-names)
        body-str (str/join "\n      " body-lines)
        destructure-keys (str/join " " (map #(str (name %) "-prev") var-names))]
    (str "(fn [trace transitions]\n"
         "  (let [" binding-str "]\n"
         "    (doseq [[i {:keys [" destructure-keys "]}] (map-indexed vector transitions)]\n"
         "      " body-str ")))")))

;; ============================================================
;; Model compilation
;; ============================================================

(defn compile-model
  "Compile a transition model source string into a DynamicGF.

   The source should evaluate to (fn [trace transitions] ...).
   Returns a DynamicGF that takes [transitions] as arguments,
   or nil on compilation failure."
  [source]
  (try
    (let [f (sci/eval-string source sci-opts)]
      (when (fn? f)
        (dyn/auto-key (gen [transitions] (f trace transitions)))))
    (catch :default _ nil)))

;; ============================================================
;; Scoring
;; ============================================================

(defn build-constraints
  "Build a choicemap constraining all observed transitions.
   Each transition produces two constrained trace sites:
   :x0, :y0, :x1, :y1, ..."
  [transitions var-names]
  (let [pairs (for [[i t] (map-indexed vector transitions)
                    v var-names]
                [(keyword (str (name v) i))
                 (mx/scalar (get t (keyword (str (name v) "-next"))))])]
    (apply cm/choicemap (mapcat identity pairs))))

(defn- log-sum-exp
  "Numerically stable log(sum(exp(xs)))."
  [xs]
  (let [max-x (apply max xs)]
    (if (= max-x ##-Inf)
      ##-Inf
      (+ max-x
         (js/Math.log
           (reduce + (map #(js/Math.exp (- % max-x)) xs)))))))

(defn log-mean-exp
  "Numerically stable log(mean(exp(xs))) = log-sum-exp(xs) - log(n)."
  [xs]
  (- (log-sum-exp xs) (js/Math.log (count xs))))

(defn score-model
  "Estimate the log marginal likelihood of a model against data.

   Runs n-particles importance sampling passes via p/generate.
   Each pass samples parameters from their priors and scores all
   transitions. Returns the log-mean-exp of the importance weights.

   gf:           compiled DynamicGF from compile-model
   transitions:  vector of transition maps from extract-transitions
   var-names:    [:x :y]
   opts:
     :n-particles  number of importance samples (default 50)"
  ([gf transitions var-names] (score-model gf transitions var-names {}))
  ([gf transitions var-names opts]
   (let [{:keys [n-particles] :or {n-particles 50}} opts
         constraints (build-constraints transitions var-names)
         weights (loop [i 0, ws []]
                   (if (>= i n-particles)
                     ws
                     (let [w (try
                               (let [{:keys [weight]} (p/generate gf [transitions] constraints)
                                     v (mx/item weight)]
                                 v)
                               (catch :default _ ##-Inf))]
                       (when (zero? (mod (inc i) 10))
                         (mx/force-gc!))
                       (recur (inc i) (conj ws w)))))]
     (mx/force-gc!)
     (log-mean-exp weights))))

;; ============================================================
;; Structure enumeration
;; ============================================================

(defn enumerate-2var-structures
  "Enumerate all 4 possible edge structures for 2 variables.
   Returns a vector of {:name :edges :description} maps."
  [var-a var-b]
  (let [a (name var-a) b (name var-b)]
    [{:name (str a "->" b)
      :edges {[a b] true  [b a] false}
      :description (str a "(t) causes " b "(t+1)")}
     {:name (str b "->" a)
      :edges {[a b] false [b a] true}
      :description (str b "(t) causes " a "(t+1)")}
     {:name "both"
      :edges {[a b] true  [b a] true}
      :description "bidirectional"}
     {:name "neither"
      :edges {[a b] false [b a] false}
      :description "independent AR(1)"}]))

;; ============================================================
;; Posterior computation
;; ============================================================

(defn compute-posterior
  "Compute posterior probabilities from log marginal likelihoods.
   Assumes uniform prior over structures.

   scored: [{:name :log-ml ...} ...]
   Returns the same maps with :posterior added."
  [scored]
  (let [log-mls (mapv :log-ml scored)
        log-norm (log-sum-exp log-mls)]
    (mapv (fn [s]
            (assoc s :posterior
                   (js/Math.exp (- (:log-ml s) log-norm))))
          scored)))

;; ============================================================
;; End-to-end model comparison
;; ============================================================

(defn compare-structures
  "Score all candidate structures against data and compute posteriors.

   var-names:    [:x :y]
   transitions:  from extract-transitions
   opts:
     :n-particles  importance samples per model (default 200)
     :structures   custom structures (default: enumerate-2var-structures)

   Returns a vector of {:name :edges :description :log-ml :posterior :source},
   sorted by posterior probability descending."
  ([var-names transitions] (compare-structures var-names transitions {}))
  ([var-names transitions opts]
   (let [structures (or (:structures opts)
                        (enumerate-2var-structures (first var-names) (second var-names)))
         scored (vec
                  (for [s structures]
                    (let [source (build-transition-source var-names (:edges s))
                          gf (compile-model source)]
                      (println (str "  scoring: " (:name s) "..."))
                      (if gf
                        (let [lml (score-model gf transitions var-names opts)]
                          (println (str "    log-ML: " (.toFixed lml 2)))
                          (mx/force-gc!)
                          (assoc s :log-ml lml :source source))
                        (do
                          (println "    FAILED to compile")
                          (assoc s :log-ml ##-Inf :source source))))))]
     (->> (compute-posterior scored)
          (sort-by :posterior >)
          vec))))

;; ============================================================
;; FIM scaffold: structure with holes for LLM to fill
;; ============================================================

(def ^:private hole-marker "<<<HOLE>>>")

(defn build-scaffold
  "Build a transition model template with holes for mean expressions.

   All possible parameters (AR + cross-effects for every pair) are
   included in the let bindings. The mean expression for each variable
   is replaced with <<<HOLE>>>. FIM fills the holes.

   var-names: [:x :y]
   Returns the template string with <<<HOLE>>> markers."
  [var-names]
  (let [n (count var-names)
        ar-bindings (mapv (fn [v]
                            (str "ar-" (name v) " (trace :ar-" (name v)
                                 " (dist/gaussian 0.5 0.3))"))
                          var-names)
        cross-bindings (vec
                         (for [src var-names, tgt var-names
                               :when (not= src tgt)]
                           (str "beta-" (name src) "->" (name tgt)
                                " (trace :beta-" (name src) "->" (name tgt)
                                " (dist/gaussian 0 0.3))")))
        sigma-bindings (mapv (fn [v]
                               (str "sigma-" (name v)
                                    " (trace :sigma-" (name v)
                                    " (dist/exponential 1))"))
                             var-names)
        all-bindings (into [] (concat ar-bindings cross-bindings sigma-bindings))
        binding-str (str/join "\n        " all-bindings)
        destructure-keys (str/join " " (map #(str (name %) "-prev") var-names))
        hint ";; mean: use mx/multiply, mx/add with ar-VAR, VAR-prev, optionally beta-SRC->TGT\n      ;; e.g. (mx/add (mx/multiply ar-y y-prev) (mx/multiply beta-x->y x-prev))\n      "
        trace-lines
        (mapv (fn [v]
                (str hint
                     "(trace (keyword (str \"" (name v) "\" i))\n"
                     "             (dist/gaussian " hole-marker
                     " sigma-" (name v) "))"))
              var-names)
        body-str (str/join "\n      " trace-lines)]
    (str "(fn [trace transitions]\n"
         "  (let [" binding-str "]\n"
         "    (doseq [[i {:keys [" destructure-keys "]}] (map-indexed vector transitions)]\n"
         "      " body-str ")))")))

(defn scaffold-holes
  "Find all hole positions in a scaffold template.
   Returns a vector of {:index :prefix :suffix} for each <<<HOLE>>>."
  [scaffold]
  (loop [pos 0, holes []]
    (let [idx (str/index-of scaffold hole-marker pos)]
      (if (nil? idx)
        holes
        (recur (+ idx (count hole-marker))
               (conj holes
                     {:index (count holes)
                      :prefix (subs scaffold 0 idx)
                      :suffix (subs scaffold (+ idx (count hole-marker)))}))))))

(defn fill-scaffold
  "Replace <<<HOLE>>> markers in scaffold with provided expressions.
   fills: vector of expression strings, one per hole."
  [scaffold fills]
  (reduce (fn [s fill]
            (str/replace-first s hole-marker fill))
          scaffold
          fills))

(defn fim-prompt
  "Build a FIM prompt string from prefix and suffix.
   Format: <|fim_prefix|>PREFIX<|fim_suffix|>SUFFIX<|fim_middle|>"
  [prefix suffix]
  (str "<|fim_prefix|>" prefix "<|fim_suffix|>" suffix "<|fim_middle|>"))

;; ============================================================
;; Chat-based hole filling: LLM proposes transition equations
;; ============================================================

(def ^:private fill-system-prompt
  "You are a ClojureScript code assistant. Output ONLY valid ClojureScript code.")

(defn- fill-prompt
  "Build a chat prompt asking the LLM to fill a scaffold hole."
  [var-name var-names]
  (str "Write a ClojureScript expression for the mean of `" (name var-name)
       "` in a time series transition model. "
       "The expression should use `mx/multiply` and `mx/add` to combine "
       "autoregressive terms like `(mx/multiply ar-" (name var-name) " " (name var-name) "-prev)` "
       "with optional cross-effects like `(mx/multiply beta-x->y x-prev)`. "
       "Available bindings: "
       (str/join ", " (concat
                        (map #(str "ar-" (name %)) var-names)
                        (map #(str (name %) "-prev") var-names)
                        (for [s var-names, t var-names :when (not= s t)]
                          (str "beta-" (name s) "->" (name t)))))
       ". Example: (mx/add (mx/multiply ar-y y-prev) (mx/multiply beta-x->y x-prev))"))

(defn fill-hole-chat
  "Fill a scaffold hole using the chat API.

   Prompts the model to generate a mean expression for a specific variable.
   Validates the result with edamame. Returns the expression string or nil.

   model-map:  {:model :tokenizer :type} from llm/load-model
   var-name:   which variable's mean expression to generate (:x or :y)
   var-names:  all variables [:x :y]
   opts:
     :temperature  sampling temperature (default 0.3)
     :max-tokens   max tokens (default 80)"
  ([model-map var-name var-names] (fill-hole-chat model-map var-name var-names {}))
  ([model-map var-name var-names opts]
   (let [{:keys [temperature max-tokens] :or {temperature 0.3 max-tokens 80}} opts
         prompt (fill-prompt var-name var-names)]
     (pr/let [text (llm/generate-text-raw model-map prompt
                     {:max-tokens max-tokens
                      :system-prompt fill-system-prompt})
              extracted (codegen/extract-code text)
              expr (str/trim extracted)]
       (when (and (seq expr) (= :complete (codegen/prefix-status expr)))
         expr)))))

(defn fill-scaffold-chat
  "Fill all holes in a scaffold using chat-based generation.

   For each variable, prompts the model to generate the mean expression.
   Returns the completed source string, or nil if any fill fails validation.

   model-map:  {:model :tokenizer :type}
   var-names:  [:x :y]
   opts:       passed to fill-hole-chat"
  ([model-map var-names] (fill-scaffold-chat model-map var-names {}))
  ([model-map var-names opts]
   (let [scaffold (build-scaffold var-names)]
     (pr/loop [i 0, current scaffold, vnames var-names]
       (if (empty? vnames)
         current
         (pr/let [expr (fill-hole-chat model-map (first vnames) var-names opts)]
           (if expr
             (pr/recur (inc i) (fill-scaffold current [expr]) (rest vnames))
             nil)))))))

(defn generate-candidates
  "Generate N candidate models by filling scaffold holes via chat LLM.

   For each candidate:
   1. Chat model proposes mean expressions for each variable
   2. Reader validates syntax
   3. SCI compiles the result

   model-map:  {:model :tokenizer :type}
   var-names:  [:x :y]
   n:          number of candidates to generate
   opts:
     :temperature  sampling temperature (default 0.5 for diversity)
     + all fill-hole-chat opts

   Returns a promise of [{:source :gf :expressions [...]} ...].
   Failed compilations included with :gf nil."
  ([model-map var-names n] (generate-candidates model-map var-names n {}))
  ([model-map var-names n opts]
   (let [opts (update opts :temperature #(or % 0.5))]
     (pr/loop [i 0, results []]
       (if (>= i n)
         results
         (pr/let [source (fill-scaffold-chat model-map var-names opts)]
           (let [gf (when source (compile-model source))]
             (pr/recur (inc i)
                       (conj results {:source source
                                      :gf gf})))))))))
