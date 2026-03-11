(ns genmlx.affine
  "Affine expression analysis for Level 3 auto-Kalman detection.

   Walks schema dist-arg source forms to determine whether an expression
   is affine in a target trace address: expr = coefficient * target + offset.

   Conservative: false negatives (missing affine) are safe; false positives
   (wrong affine) would produce incorrect posteriors.

   Level 3 — WP-3: Affine Expression Analysis."
  (:require [clojure.set :as set]))

;; =========================================================================
;; Affine result constructors
;; =========================================================================

(defn- affine-constant
  "Expression is a constant (does not depend on target)."
  [expr]
  {:affine? true :has-target? false :coefficient 0 :offset expr})

(defn- affine-target
  "Expression IS the target variable."
  []
  {:affine? true :has-target? true :coefficient 1 :offset 0})

(defn- affine-linear
  "Expression is coefficient * target + offset."
  [coefficient offset]
  {:affine? true :has-target? true :coefficient coefficient :offset offset})

(defn- not-affine
  "Expression is nonlinear in target."
  []
  {:affine? false})

;; =========================================================================
;; Core analysis
;; =========================================================================

(declare analyze-affine)

(defn- analyze-affine-add
  "Analyze (mx/add a b) or (+ a b) for affine structure.
   affine + affine = affine: (a1*x + b1) + (a2*x + b2) = (a1+a2)*x + (b1+b2)"
  [args target-sym env]
  (let [results (mapv #(analyze-affine % target-sym env) args)]
    (if (every? :affine? results)
      (let [target-results (filter :has-target? results)
            const-results (filter (complement :has-target?) results)]
        (cond
          ;; No target dependency — pure constant addition
          (empty? target-results)
          (affine-constant (cons 'mx/add (map :offset results)))

          ;; One or more target-dependent terms: sum coefficients and offsets
          :else
          (let [coeff (if (= 1 (count target-results))
                        (:coefficient (first target-results))
                        (cons 'mx/add (map :coefficient target-results)))
                ;; Collect all offsets
                all-offsets (keep (fn [r]
                                   (when (and (:offset r) (not= 0 (:offset r)))
                                     (:offset r)))
                                  results)
                offset (cond
                         (empty? all-offsets) 0
                         (= 1 (count all-offsets)) (first all-offsets)
                         :else (cons 'mx/add all-offsets))]
            (affine-linear coeff offset))))
      (not-affine))))

(defn- analyze-affine-subtract
  "Analyze (mx/subtract a b) for affine structure.
   a - b = a + (-b)"
  [args target-sym env]
  (if (= 2 (count args))
    (let [a (analyze-affine (first args) target-sym env)
          b (analyze-affine (second args) target-sym env)]
      (if (and (:affine? a) (:affine? b))
        (cond
          ;; Neither depends on target
          (and (not (:has-target? a)) (not (:has-target? b)))
          (affine-constant (list 'mx/subtract (:offset a) (:offset b)))

          ;; Only a depends on target: a*x + a_off - b_off
          (and (:has-target? a) (not (:has-target? b)))
          (let [offset (if (and (= 0 (:offset a)) (= 0 (:offset b)))
                         0
                         (list 'mx/subtract
                               (if (= 0 (:offset a)) 0 (:offset a))
                               (if (= 0 (:offset b)) 0 (:offset b))))]
            (affine-linear (:coefficient a) offset))

          ;; Only b depends on target: -b*x + a_off - b_off
          (and (not (:has-target? a)) (:has-target? b))
          (let [neg-coeff (if (= 1 (:coefficient b))
                            -1
                            (list 'mx/negate (:coefficient b)))
                offset (if (= 0 (:offset b))
                         (:offset a)
                         (list 'mx/subtract (:offset a) (:offset b)))]
            (affine-linear neg-coeff offset))

          ;; Both depend on target: (a-b)*x + (a_off - b_off)
          :else
          (let [coeff (list 'mx/subtract (:coefficient a) (:coefficient b))
                offset (if (and (= 0 (:offset a)) (= 0 (:offset b)))
                         0
                         (list 'mx/subtract (:offset a) (:offset b)))]
            (affine-linear coeff offset)))
        (not-affine)))
    (not-affine)))

(defn- analyze-affine-multiply
  "Analyze (mx/multiply a b) for affine structure.
   constant * affine = affine: c * (a*x + b) = (c*a)*x + (c*b)
   affine * affine with both having target = NONLINEAR (quadratic)"
  [args target-sym env]
  (if (= 2 (count args))
    (let [a (analyze-affine (first args) target-sym env)
          b (analyze-affine (second args) target-sym env)]
      (if (and (:affine? a) (:affine? b))
        (cond
          ;; Neither depends on target — constant * constant
          (and (not (:has-target? a)) (not (:has-target? b)))
          (affine-constant (list 'mx/multiply (:offset a) (:offset b)))

          ;; Both depend on target — quadratic, not affine
          (and (:has-target? a) (:has-target? b))
          (not-affine)

          ;; One constant, one target-dependent
          :else
          (let [[const-r target-r] (if (:has-target? a) [b a] [a b])
                c (:offset const-r)
                coeff (if (= 1 (:coefficient target-r))
                        c
                        (list 'mx/multiply c (:coefficient target-r)))
                offset (if (= 0 (:offset target-r))
                         0
                         (list 'mx/multiply c (:offset target-r)))]
            (affine-linear coeff offset)))
        (not-affine)))
    (not-affine)))

(defn- analyze-affine-divide
  "Analyze (mx/divide a b) for affine structure.
   affine / constant = affine: (a*x + b) / c = (a/c)*x + (b/c)
   anything / target-dependent = NONLINEAR"
  [args target-sym env]
  (if (= 2 (count args))
    (let [a (analyze-affine (first args) target-sym env)
          b (analyze-affine (second args) target-sym env)]
      (if (and (:affine? a) (:affine? b))
        (cond
          ;; Dividing by target-dependent expression — nonlinear
          (:has-target? b)
          (not-affine)

          ;; Neither depends on target
          (not (:has-target? a))
          (affine-constant (list 'mx/divide (:offset a) (:offset b)))

          ;; Numerator depends on target, denominator constant
          :else
          (let [c (:offset b)
                coeff (if (= 1 (:coefficient a))
                        (list 'mx/divide 1 c)
                        (list 'mx/divide (:coefficient a) c))
                offset (if (= 0 (:offset a))
                         0
                         (list 'mx/divide (:offset a) c))]
            (affine-linear coeff offset)))
        (not-affine)))
    (not-affine)))

(defn- analyze-affine-negate
  "Analyze (mx/negate x) for affine structure.
   negate(a*x + b) = (-a)*x + (-b)"
  [args target-sym env]
  (if (= 1 (count args))
    (let [a (analyze-affine (first args) target-sym env)]
      (if (:affine? a)
        (if (:has-target? a)
          (affine-linear
            (if (= 1 (:coefficient a)) -1 (list 'mx/negate (:coefficient a)))
            (if (= 0 (:offset a)) 0 (list 'mx/negate (:offset a))))
          (affine-constant (list 'mx/negate (:offset a))))
        (not-affine)))
    (not-affine)))

(defn- analyze-affine-call
  "Analyze a function call for affine structure."
  [expr target-sym env]
  (let [op (first expr)
        op-name (name op)
        args (rest expr)]
    (case op-name
      ("add" "+")       (analyze-affine-add (vec args) target-sym env)
      ("subtract" "-")  (analyze-affine-subtract (vec args) target-sym env)
      ("multiply" "*")  (analyze-affine-multiply (vec args) target-sym env)
      ("divide" "/")    (analyze-affine-divide (vec args) target-sym env)
      ("negate")        (analyze-affine-negate (vec args) target-sym env)
      ("scalar")        (analyze-affine (first args) target-sym env)
      ;; Everything else is nonlinear (conservative)
      (not-affine))))

(defn analyze-affine
  "Analyze whether a source expression is affine in a target symbol.

   Returns:
   - {:affine? true :has-target? true :coefficient form :offset form}
     when expr = coefficient * target + offset
   - {:affine? true :has-target? false :coefficient 0 :offset form}
     when expr is constant (no target dependency)
   - {:affine? false}
     when expr is nonlinear in target

   target-sym: the symbol name (e.g., 'mu) to analyze affinity with respect to.
   env: schema binding environment mapping symbols to trace address deps.

   Examples:
   - mu                           → coeff=1, offset=0
   - (mx/multiply 2 mu)           → coeff=2, offset=0
   - (mx/add mu 3)                → coeff=1, offset=3
   - (mx/add (mx/multiply s mu) b) → coeff=s, offset=b
   - (mx/exp mu)                  → not affine
   - (mx/multiply mu mu)          → not affine (quadratic)"
  [expr target-sym env]
  (cond
    ;; Target symbol itself
    (and (symbol? expr) (= expr target-sym))
    (affine-target)

    ;; Other symbol — check if it depends on target via env
    (symbol? expr)
    (let [sym-deps (get env expr)
          ;; env maps symbols to sets of keywords (trace addrs)
          ;; target-sym is a symbol, so check keyword version too
          target-kw (keyword (name target-sym))]
      (if (and sym-deps (or (contains? sym-deps target-sym)
                            (contains? sym-deps target-kw)))
        ;; Depends on target through env — conservatively nonlinear
        ;; (we don't have the binding expression to analyze)
        (not-affine)
        ;; Independent of target — treat as constant
        (affine-constant expr)))

    ;; Numeric literal
    (number? expr)
    (affine-constant expr)

    ;; Other literals (keywords, strings, booleans, nil)
    (or (keyword? expr) (string? expr) (nil? expr) (boolean? expr))
    (affine-constant expr)

    ;; Function call
    (and (seq? expr) (seq expr) (symbol? (first expr)))
    (analyze-affine-call expr target-sym env)

    ;; Everything else — conservatively nonlinear
    :else
    (not-affine)))

;; =========================================================================
;; Integration with conjugacy detection
;; =========================================================================

(defn classify-affine-dependency
  "Classify the dependency of an obs site on a prior site.
   Analyzes the natural parameter dist-arg for affine structure.

   prior-addr: keyword address of the prior trace site
   obs-site: schema trace site map for the observation
   natural-param-idx: which dist-arg position is the natural parameter

   Returns:
   - {:type :direct} if prior value used directly (coefficient=1, offset=0)
   - {:type :affine :coefficient form :offset form} if affine
   - {:type :nonlinear} if nonlinear or unknown"
  [prior-addr obs-site natural-param-idx]
  (let [dist-args (:dist-args obs-site)
        natural-arg (when (and dist-args (< natural-param-idx (count dist-args)))
                      (nth dist-args natural-param-idx))
        ;; The target symbol should match the prior address name
        target-sym (symbol (name prior-addr))
        ;; Build env from obs-site deps (symbols that map to trace addrs)
        ;; We need to know which symbols in the expression are trace-dependent
        env (into {} (map (fn [dep] [(symbol (name dep)) #{dep}]))
                     (:deps obs-site))
        result (when natural-arg
                 (analyze-affine natural-arg target-sym env))]
    (cond
      (nil? result)
      {:type :nonlinear}

      (not (:affine? result))
      {:type :nonlinear}

      (not (:has-target? result))
      {:type :nonlinear}  ;; doesn't actually depend on prior

      ;; Direct: coefficient=1, offset=0
      (and (= 1 (:coefficient result)) (= 0 (:offset result)))
      {:type :direct}

      ;; Affine
      :else
      {:type :affine
       :coefficient (:coefficient result)
       :offset (:offset result)})))

;; =========================================================================
;; Kalman chain detection
;; =========================================================================

(defn detect-kalman-chains
  "Detect linear-Gaussian chains in a schema suitable for Kalman filtering.

   A Kalman chain is a sequence of gaussian trace sites z0 → z1 → ... → zT
   where each z_{t+1} ~ N(a*z_t + b, noise) (affine or direct dependency),
   with observation sites y_t ~ N(z_t, obs_noise) at each step.

   Uses conjugate pairs (from conjugacy detection) to find chains.
   A chain edge connects prior → obs where the obs is ALSO a prior
   in another pair (i.e., an intermediate latent node).

   Returns vector of chain descriptors, each:
   {:steps [{:latent addr
             :observations [addr ...]
             :transition {:type :affine/:direct :coefficient :offset}
             :next-latent addr-or-nil} ...]
    :latent-addrs [z0 z1 z2 ...]
    :obs-addrs [y0 y1 y2 ...]}"
  [conjugate-pairs]
  (when (seq conjugate-pairs)
    (let [;; Only gaussian-gaussian pairs form Kalman structure
          nn-pairs (filter #(= :normal-normal (:family %)) conjugate-pairs)
          ;; Index by prior
          by-prior (group-by :prior-addr nn-pairs)
          ;; Addresses that appear as priors
          prior-set (set (map :prior-addr nn-pairs))

          ;; Follow chains from each root
          follow-chain
          (fn [root]
            (loop [current root
                   steps []]
              (let [successors (get by-prior current [])
                    ;; Chain successor: obs that is itself a prior of another pair
                    chain-next (first (filter #(contains? prior-set (:obs-addr %)) successors))
                    ;; Observations: non-chain successors
                    obs (filterv #(not= (:obs-addr %) (:obs-addr chain-next)) successors)
                    step {:latent current
                          :observations (mapv :obs-addr obs)
                          :obs-dep-types (mapv :dependency-type obs)
                          :transition (when chain-next (:dependency-type chain-next))
                          :noise-std (when chain-next
                                       (let [args (:dist-args (:obs-site chain-next))]
                                         (when (>= (count args) 2) (second args))))
                          :next-latent (when chain-next (:obs-addr chain-next))}
                    steps' (conj steps step)]
                (if chain-next
                  (recur (:obs-addr chain-next) steps')
                  steps'))))

          ;; Find chain roots: priors that are not obs of any NN pair
          obs-set (set (map :obs-addr nn-pairs))
          chain-roots (set/difference prior-set obs-set)

          chains (for [root chain-roots
                       :let [steps (follow-chain root)]
                       ;; Must have at least 2 latent nodes to be a chain
                       :when (>= (count steps) 2)]
                   {:steps steps
                    :latent-addrs (mapv :latent steps)
                    :obs-addrs (into [] (mapcat :observations) steps)})]
      (vec chains))))
