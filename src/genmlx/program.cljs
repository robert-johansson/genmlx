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
      :x-next (:x nxt) :y-next (:y nxt)})))

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
;; Analytical scoring (Bayesian linear regression)
;; ============================================================

(defn- dot [a b]
  (let [n (count a)]
    (loop [i 0, acc 0.0]
      (if (>= i n) acc
          (recur (inc i) (+ acc (* (nth a i) (nth b i))))))))

(defn- vsub [a b] (mapv - a b))

(defn- mat-lu-solve
  "Solve Ax = b via Gaussian elimination with partial pivoting. A is p*p, b is p-vector."
  [A b]
  (let [p (count b)
        aug (vec (for [i (range p)]
                   (conj (vec (nth A i)) (nth b i))))
        aug (loop [k 0, aug aug]
              (if (>= k p) aug
                  (let [pivot-row (apply max-key
                                         (fn [i] (js/Math.abs (get-in aug [i k])))
                                         (range k p))
                        aug (if (= pivot-row k) aug
                                (let [tmp (nth aug k)]
                                  (-> aug (assoc k (nth aug pivot-row))
                                      (assoc pivot-row tmp))))
                        pivot-val (get-in aug [k k])]
                    (if (< (js/Math.abs pivot-val) 1e-15)
                      aug
                      (recur (inc k)
                             (reduce (fn [aug i]
                                       (let [factor (/ (get-in aug [i k]) pivot-val)]
                                         (reduce (fn [aug j]
                                                   (assoc-in aug [i j]
                                                             (- (get-in aug [i j])
                                                                (* factor (get-in aug [k j])))))
                                                 aug
                                                 (range k (inc p)))))
                                     aug
                                     (range (inc k) p)))))))]
    (loop [k (dec p), x (vec (repeat p 0.0))]
      (if (< k 0) x
          (let [sum (reduce + (map (fn [j] (* (get-in aug [k j]) (nth x j)))
                                   (range (inc k) p)))
                val (/ (- (get-in aug [k p]) sum) (get-in aug [k k]))]
            (recur (dec k) (assoc x k val)))))))

(defn- mat-log-det
  "Log absolute determinant of a p*p matrix via Gaussian elimination."
  [A]
  (let [p (count A)]
    (loop [k 0, aug (mapv vec A), log-det 0.0, sign 1]
      (if (>= k p)
        (+ log-det (js/Math.log (js/Math.abs sign)))
        (let [pivot-row (apply max-key
                               (fn [i] (js/Math.abs (get-in aug [i k])))
                               (range k p))
              swapped? (not= pivot-row k)
              aug (if swapped?
                    (let [tmp (nth aug k)]
                      (-> aug (assoc k (nth aug pivot-row)) (assoc pivot-row tmp)))
                    aug)
              sign (if swapped? (- sign) sign)
              pivot-val (get-in aug [k k])]
          (if (< (js/Math.abs pivot-val) 1e-15)
            ##-Inf
            (recur (inc k)
                   (reduce (fn [aug i]
                             (let [factor (/ (get-in aug [i k]) pivot-val)]
                               (reduce (fn [aug j]
                                         (assoc-in aug [i j]
                                                   (- (get-in aug [i j])
                                                      (* factor (get-in aug [k j])))))
                                       aug
                                       (range (inc k) p))))
                           aug
                           (range (inc k) p))
                   (+ log-det (js/Math.log (js/Math.abs pivot-val)))
                   (if (neg? pivot-val) (- sign) sign))))))))

(defn- log-ml-general
  "Analytical log-ML for one variable's regression with p predictors.
   Uses Gaussian elimination for general p*p matrices."
  [y X-cols sigma-sq]
  (let [n (count y)
        p (count X-cols)
        m0 (mapv :prior-mean X-cols)
        s0-diag (mapv :prior-var X-cols)
        Xm0 (reduce (fn [acc j]
                      (let [xj (:values (nth X-cols j))
                            mj (nth m0 j)]
                        (mapv + acc (mapv #(* mj %) xj))))
                    (vec (repeat n 0.0))
                    (range p))
        r (vsub y Xm0)
        rr (dot r r)
        gram (vec (for [i (range p)]
                    (vec (for [j (range p)]
                           (dot (:values (nth X-cols i))
                                (:values (nth X-cols j)))))))
        Xtr (mapv (fn [j] (dot (:values (nth X-cols j)) r)) (range p))
        M (vec (for [i (range p)]
                 (vec (for [j (range p)]
                        (+ (get-in gram [i j])
                           (if (= i j) (/ sigma-sq (nth s0-diag i)) 0))))))
        Minv-Xtr (mat-lu-solve M Xtr)
        quad-correction (reduce + (map * Xtr Minv-Xtr))
        A (vec (for [i (range p)]
                 (vec (for [j (range p)]
                        (+ (if (= i j) 1.0 0.0)
                           (/ (* (nth s0-diag i) (get-in gram [i j])) sigma-sq))))))
        log-det-A (mat-log-det A)
        quad (/ (- rr quad-correction) sigma-sq)]
    (+ (* -0.5 n (js/Math.log (* 2.0 js/Math.PI sigma-sq)))
       (* -0.5 log-det-A)
       (* -0.5 quad))))

(defn- extract-regression-data
  "Build regression components for one variable's transition equation.
   Returns {:y [observations] :cols [{:values :prior-mean :prior-var} ...]}"
  [transitions target var-names edges
   {:keys [ar-prior-mean ar-prior-std beta-prior-mean beta-prior-std]
    :or {ar-prior-mean 0.5 ar-prior-std 0.15
         beta-prior-mean 0.0 beta-prior-std 3.0}}]
  (let [y (mapv #(get % (keyword (str (name target) "-next"))) transitions)
        ar-col {:values (mapv #(get % (keyword (str (name target) "-prev"))) transitions)
                :prior-mean ar-prior-mean
                :prior-var (* ar-prior-std ar-prior-std)}
        cross-cols (vec
                    (for [src var-names
                          :when (and (not= src target)
                                     (get edges [(name src) (name target)]))]
                      {:values (mapv #(get % (keyword (str (name src) "-prev"))) transitions)
                       :prior-mean beta-prior-mean
                       :prior-var (* beta-prior-std beta-prior-std)}))]
    {:y y :cols (into [ar-col] cross-cols)}))

(defn- estimate-sigma-sq
  "Estimate noise variance via OLS residuals for one variable."
  [{:keys [y cols]}]
  (let [n (count y)
        p (count cols)]
    (if (<= n p)
      1.0
      (if (= p 1)
        (let [x (:values (first cols))
              xx (dot x x)
              xy (dot x y)
              beta-hat (/ xy xx)
              residuals (mapv (fn [yi xi] (- yi (* beta-hat xi))) y x)]
          (/ (dot residuals residuals) n))
        (let [x1 (:values (first cols))
              x2 (:values (second cols))
              g00 (dot x1 x1) g01 (dot x1 x2) g11 (dot x2 x2)
              det-g (- (* g00 g11) (* g01 g01))
              xy1 (dot x1 y) xy2 (dot x2 y)
              b1 (/ (- (* g11 xy1) (* g01 xy2)) det-g)
              b2 (/ (- (* g00 xy2) (* g01 xy1)) det-g)
              residuals (mapv (fn [yi x1i x2i] (- yi (* b1 x1i) (* b2 x2i)))
                              y x1 x2)]
          (/ (dot residuals residuals) n))))))

(defn- log-ml-variable
  "Analytical log marginal likelihood for one variable's transition.
   Integrates out beta: y ~ N(X*beta, sigma^2*I), beta ~ N(m0, S0)."
  [{:keys [y cols]} sigma-sq]
  (let [n (count y)
        p (count cols)
        m0 (mapv :prior-mean cols)
        s0 (mapv :prior-var cols)
        Xm0 (reduce (fn [acc j]
                      (let [xj (:values (nth cols j))
                            mj (nth m0 j)]
                        (mapv + acc (mapv #(* mj %) xj))))
                    (vec (repeat n 0.0))
                    (range p))
        r (vsub y Xm0)
        rr (dot r r)
        gram (vec (for [i (range p)]
                    (vec (for [j (range p)]
                           (dot (:values (nth cols i))
                                (:values (nth cols j)))))))
        Xtr (mapv (fn [j] (dot (:values (nth cols j)) r)) (range p))]
    (if (= p 1)
      (let [xx (get-in gram [0 0])
            xr (nth Xtr 0)
            tau-sq (nth s0 0)
            m-val (+ xx (/ sigma-sq tau-sq))
            log-det (js/Math.log (+ 1.0 (/ (* tau-sq xx) sigma-sq)))
            quad (/ (- rr (/ (* xr xr) m-val)) sigma-sq)]
        (+ (* -0.5 n (js/Math.log (* 2.0 js/Math.PI sigma-sq)))
           (* -0.5 log-det)
           (* -0.5 quad)))
      (let [m00 (+ (get-in gram [0 0]) (/ sigma-sq (nth s0 0)))
            m01 (get-in gram [0 1])
            m11 (+ (get-in gram [1 1]) (/ sigma-sq (nth s0 1)))
            det-M (- (* m00 m11) (* m01 m01))
            xr0 (nth Xtr 0) xr1 (nth Xtr 1)
            Minv-xr-0 (/ (- (* m11 xr0) (* m01 xr1)) det-M)
            Minv-xr-1 (/ (- (* m00 xr1) (* m01 xr0)) det-M)
            quad-correction (+ (* xr0 Minv-xr-0) (* xr1 Minv-xr-1))
            a00 (+ 1.0 (/ (* (nth s0 0) (get-in gram [0 0])) sigma-sq))
            a01 (/ (* (nth s0 0) (get-in gram [0 1])) sigma-sq)
            a10 (/ (* (nth s0 1) (get-in gram [1 0])) sigma-sq)
            a11 (+ 1.0 (/ (* (nth s0 1) (get-in gram [1 1])) sigma-sq))
            det-A (- (* a00 a11) (* a01 a10))
            log-det (js/Math.log det-A)
            quad (/ (- rr quad-correction) sigma-sq)]
        (+ (* -0.5 n (js/Math.log (* 2.0 js/Math.PI sigma-sq)))
           (* -0.5 log-det)
           (* -0.5 quad))))))

(defn score-model-analytical
  "Exact log marginal likelihood for Gaussian linear transition models.

   Integrates out regression coefficients analytically using conjugate
   Gaussian prior. Sigma is either provided or estimated via OLS.
   Deterministic — no sampling, exact answer given data.

   transitions:  from extract-transitions
   var-names:    [:x :y]
   edges:        {[\"x\" \"y\"] true, [\"y\" \"x\"] false}
   opts:
     :sigma-x          known noise std for x (estimated from OLS if omitted)
     :sigma-y          known noise std for y (estimated from OLS if omitted)
     :ar-prior-mean    prior mean for AR coefficients (default 0.5)
     :ar-prior-std     prior std for AR coefficients (default 0.15)
     :beta-prior-mean  prior mean for cross-effects (default 0.0)
     :beta-prior-std   prior std for cross-effects (default 3.0)"
  ([transitions var-names edges]
   (score-model-analytical transitions var-names edges {}))
  ([transitions var-names edges opts]
   (reduce
    (fn [total v]
      (let [reg (extract-regression-data transitions v var-names edges opts)
            sigma-key (keyword (str "sigma-" (name v)))
            known-sigma (get opts sigma-key)
            sigma-sq (if known-sigma
                       (* known-sigma known-sigma)
                       (estimate-sigma-sq reg))]
        (+ total (log-ml-variable reg sigma-sq))))
    0.0
    var-names)))

;; ============================================================
;; Structure enumeration
;; ============================================================

(defn enumerate-2var-structures
  "Enumerate all 4 possible edge structures for 2 variables.
   Returns a vector of {:name :edges :description} maps."
  [var-a var-b]
  (let [a (name var-a) b (name var-b)]
    [{:name (str a "->" b)
      :edges {[a b] true [b a] false}
      :description (str a "(t) causes " b "(t+1)")}
     {:name (str b "->" a)
      :edges {[a b] false [b a] true}
      :description (str b "(t) causes " a "(t+1)")}
     {:name "both"
      :edges {[a b] true [b a] true}
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
     :scoring     :analytical (default) or :importance
     :n-particles  importance samples per model (default 200, only for :importance)
     :structures   custom structures (default: enumerate-2var-structures)
     + all score-model-analytical opts (sigma, prior widths)

   Returns a vector of {:name :edges :description :log-ml :posterior :source},
   sorted by posterior probability descending."
  ([var-names transitions] (compare-structures var-names transitions {}))
  ([var-names transitions opts]
   (let [scoring (or (:scoring opts) :analytical)
         structures (or (:structures opts)
                        (enumerate-2var-structures (first var-names) (second var-names)))
         scored (vec
                 (for [s structures]
                   (let [source (build-transition-source var-names (:edges s))]
                     (println (str "  scoring: " (:name s) "..."))
                     (if (= scoring :analytical)
                       (let [lml (score-model-analytical transitions var-names (:edges s) opts)]
                         (println (str "    log-ML: " (.toFixed lml 2)))
                         (assoc s :log-ml lml :source source))
                       (let [gf (compile-model source)]
                         (if gf
                           (let [lml (score-model gf transitions var-names opts)]
                             (println (str "    log-ML: " (.toFixed lml 2)))
                             (mx/force-gc!)
                             (assoc s :log-ml lml :source source))
                           (do
                             (println "    FAILED to compile")
                             (assoc s :log-ml ##-Inf :source source))))))))]
     (->> (compute-posterior scored)
          (sort-by :posterior >)
          vec))))

;; ============================================================
;; K-variable structure discovery
;; ============================================================

(defn generate-kvar-data
  "Generate K-variable panel data with known causal structure.

   var-names:  [:sleep :exercise :mood]
   dgp:
     :n-individuals  (default 50)
     :n-steps        (default 10)
     :ar             {var -> coefficient}  (default 0.5 for all)
     :cross          {[src tgt] -> coefficient}  e.g. {[:exercise :mood] 0.4}
     :sigma          {var -> noise-std}  (default 1.0 for all)
     :init-mean      {var -> mean}  (default 0 for all)
     :init-std       {var -> std}   (default 2 for all)

   Returns a vector of individuals, each a vector of {var -> value} maps."
  [var-names dgp]
  (let [{:keys [n-individuals n-steps ar cross sigma init-mean init-std]
         :or {n-individuals 50 n-steps 10}} dgp
        ar (merge (zipmap var-names (repeat 0.5)) ar)
        cross (or cross {})
        sigma (merge (zipmap var-names (repeat 1.0)) sigma)
        init-mean (merge (zipmap var-names (repeat 0.0)) init-mean)
        init-std (merge (zipmap var-names (repeat 2.0)) init-std)]
    (vec
     (for [_ (range n-individuals)]
       (let [init (zipmap var-names
                          (map #(+ (get init-mean %) (* (get init-std %) (randn))) var-names))]
         (loop [t 0, prev init, series [init]]
           (if (>= t (dec n-steps))
             series
             (let [nxt (zipmap var-names
                               (map (fn [v]
                                      (+ (* (get ar v) (get prev v))
                                         (reduce + (map (fn [[[src tgt] coeff]]
                                                          (if (= tgt v) (* coeff (get prev src)) 0))
                                                        cross))
                                         (* (get sigma v) (randn))))
                                    var-names))]
               (recur (inc t) nxt (conj series nxt))))))))))

(defn extract-kvar-transitions
  "Extract transition pairs from K-variable panel data.
   Returns [{:prev {:sleep 1.2 ...} :next {:sleep 0.8 ...}} ...]"
  [data]
  (vec (for [individual data
             [prev nxt] (partition 2 1 individual)]
         {:prev prev :next nxt})))

(defn enumerate-edges
  "All possible directed edges between K variables (excluding self-loops)."
  [var-names]
  (vec (for [src var-names, tgt var-names :when (not= src tgt)]
         [src tgt])))

(defn enumerate-all-structures
  "Enumerate all 2^(K*(K-1)) directed graphs for K variables.
   Returns [{:edges #{[:a :b] ...} :name \"a->b, c->a\"} ...]"
  [var-names]
  (let [all-edges (enumerate-edges var-names)
        n-edges (count all-edges)]
    (mapv (fn [i]
            (let [active (set (keep-indexed
                               (fn [j edge]
                                 (when (bit-test i j) edge))
                               all-edges))
                  nm (if (empty? active)
                       "independent"
                       (str/join ", "
                                 (map (fn [[s t]] (str (name s) "->" (name t)))
                                      (sort-by str active))))]
              {:edges active :name nm :index i}))
          (range (bit-shift-left 1 n-edges)))))

(defn score-all-structures
  "Score all 2^(K*(K-1)) directed graphs with precomputed sufficient statistics.

   Precomputes all pairwise dot products once, then assembles the Gram submatrix
   for each (variable, structure) pair. For K=3 (64 structures): ~18ms.
   For K=4 (4096 structures): ~1s.

   transitions:  from extract-kvar-transitions
   var-names:    [:sleep :exercise :mood]
   opts:
     :sigma          {var -> known-noise-std}
     :ar-prior-mean  (default 0.5)
     :ar-prior-std   (default 0.15)
     :beta-prior-std (default 3.0)

   Returns the structures augmented with :log-ml."
  ([transitions var-names] (score-all-structures transitions var-names {}))
  ([transitions var-names opts]
   (let [{:keys [ar-prior-mean ar-prior-std beta-prior-std]
          :or {ar-prior-mean 0.5 ar-prior-std 0.15 beta-prior-std 3.0}} opts
         ar-var (* ar-prior-std ar-prior-std)
         beta-var (* beta-prior-std beta-prior-std)
         n (count transitions)
         known-sigma (:sigma opts)
         col-vecs (into {}
                        (map (fn [v] [v (mapv #(get-in % [:prev v]) transitions)])
                             var-names))
         obs-vecs (into {}
                        (map (fn [v] [v (mapv #(get-in % [:next v]) transitions)])
                             var-names))
         all-dots (into {}
                        (for [a var-names, b var-names]
                          [[a b] (dot (get col-vecs a) (get col-vecs b))]))
         xty-dots (into {}
                        (for [pred var-names, tgt var-names]
                          [[pred tgt] (dot (get col-vecs pred) (get obs-vecs tgt))]))
         yty (into {} (map (fn [v] [v (dot (get obs-vecs v) (get obs-vecs v))])
                           var-names))
         all-structs (enumerate-all-structures var-names)]
     (mapv
      (fn [structure]
        (let [edges (:edges structure)
              lml (reduce
                   (fn [total target]
                     (let [sources (vec (filter #(contains? edges [% target]) var-names))
                           preds (into [target] sources)
                           p (count preds)
                           m0 (into [ar-prior-mean] (repeat (count sources) 0.0))
                           s0 (into [ar-var] (repeat (count sources) beta-var))
                           gram (vec (for [i (range p)]
                                       (vec (for [j (range p)]
                                              (get all-dots [(nth preds i) (nth preds j)])))))
                           Xty (mapv (fn [j] (get xty-dots [(nth preds j) target])) (range p))
                           gram-m0 (mapv (fn [i]
                                           (reduce + (map (fn [j] (* (get-in gram [i j]) (nth m0 j)))
                                                          (range p))))
                                         (range p))
                           Xtr (mapv - Xty gram-m0)
                           m0Xty (reduce + (map * m0 Xty))
                           m0Gram-m0 (reduce + (map * m0 gram-m0))
                           rr (+ (get yty target) (* -2.0 m0Xty) m0Gram-m0)
                           sigma-sq (if-let [s (get known-sigma target)]
                                      (* s s)
                                      (if (<= n p)
                                        1.0
                                        (let [beta-hat (mat-lu-solve gram Xty)
                                              Xbeta-y (reduce + (map * beta-hat Xty))
                                              rss (- (get yty target) Xbeta-y)]
                                          (/ (max rss 0.001) n))))
                           M (vec (for [i (range p)]
                                    (vec (for [j (range p)]
                                           (+ (get-in gram [i j])
                                              (if (= i j) (/ sigma-sq (nth s0 i)) 0))))))
                           Minv-Xtr (mat-lu-solve M Xtr)
                           quad-corr (reduce + (map * Xtr Minv-Xtr))
                           A (vec (for [i (range p)]
                                    (vec (for [j (range p)]
                                           (+ (if (= i j) 1.0 0.0)
                                              (/ (* (nth s0 i) (get-in gram [i j])) sigma-sq))))))
                           log-det-A (mat-log-det A)
                           quad (/ (- rr quad-corr) sigma-sq)]
                       (+ total
                          (* -0.5 n (js/Math.log (* 2.0 js/Math.PI sigma-sq)))
                          (* -0.5 log-det-A)
                          (* -0.5 quad))))
                   0.0
                   var-names)]
          (assoc structure :log-ml lml)))
      all-structs))))

(defn edge-marginals
  "Compute marginal posterior probability for each possible directed edge.
   Takes scored structures (with :log-ml) and returns {[src tgt] -> probability}."
  [var-names scored]
  (let [log-mls (mapv :log-ml scored)
        log-norm (log-sum-exp log-mls)]
    (into {}
          (map (fn [edge]
                 [edge (reduce + (keep (fn [s]
                                         (when (contains? (:edges s) edge)
                                           (js/Math.exp (- (:log-ml s) log-norm))))
                                       scored))])
               (enumerate-edges var-names)))))

(defn discover-structure
  "End-to-end K-variable structure discovery.

   var-names:    [:sleep :exercise :mood]
   transitions:  from extract-kvar-transitions
   opts:         passed to score-all-structures

   Returns {:ranked     [{:name :edges :log-ml :posterior} ...] (sorted)
            :marginals  {[:exercise :mood] 0.99 ...}
            :best       the top-ranked structure
            :elapsed-ms scoring time}"
  ([var-names transitions] (discover-structure var-names transitions {}))
  ([var-names transitions opts]
   (let [t0 (js/Date.now)
         scored (score-all-structures transitions var-names opts)
         elapsed (- (js/Date.now) t0)
         log-mls (mapv :log-ml scored)
         log-norm (log-sum-exp log-mls)
         ranked (->> scored
                     (map #(assoc % :posterior (js/Math.exp (- (:log-ml %) log-norm))))
                     (sort-by :posterior >)
                     vec)]
     {:ranked ranked
      :marginals (edge-marginals var-names scored)
      :best (first ranked)
      :elapsed-ms elapsed})))

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
