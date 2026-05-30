(ns examples.qwen36-structural
  "Structural recognizer for LLM-synthesized Bayesian linear regression.

   The LLM proposes (fn [trace xs] (let [a ... b ... c ...] (mapv …))).
   Instead of running HMC, we walk the parsed form, recognize the shape
   ‘Bayesian linear regression in features φ(x)’, build the design matrix
   by probe-evaluation, verify linearity numerically, and short-circuit
   to the closed-form Gaussian posterior.

   No tuning. No sampler. Milliseconds instead of seconds.
   Falls through cleanly to a :type :unknown if the LLM's program isn't
   in this family.

   Run: bun run --bun nbb examples/qwen36_structural.cljs"
  (:require [genmlx.llm.backend :as llm]
            [edamame.core :as eda]
            [sci.core :as sci]
            [clojure.string :as str]
            [promesa.core :as pr]))

(def model-path
  (str (.-HOME js/process.env) "/.cache/models/Qwen3.6-35B-A3B-4bit"))

(def system-prompt
  "You are a probabilistic programming assistant. Output only valid ClojureScript code.")

;; ---------------------------------------------------------------------------
;; Two datasets to test on. Same prompt template, just different shape.
;; ---------------------------------------------------------------------------

(def linear-task
  {:label  "linear (truth y = 2x + 1)"
   :xs     [-3.0 -2.0 -1.0 0.0 1.0 2.0 3.0]
   :ys     [-5.16 -3.04 -0.86 1.21 3.05 4.95 7.13]
   :truth  {'slope 2.0 'intercept 1.0}
   :prompt-shape
   (str "Write a probabilistic program of one argument [trace xs] that models the\n"
        "relationship as a noisy LINEAR function:  y = slope·x + intercept + ε.\n\n"
        "Priors:    slope, intercept ~ Gaussian(0, 10), at addresses :slope, :intercept\n"
        "Noise std: 0.5\n\n"
        "Per-observation trace addresses MUST include the index, e.g.\n"
        "  (keyword (str \"y\" i))   — produces :y0, :y1, :y2, …\n\n"
        "Use this shape:\n"
        "  (mapv (fn [i]\n"
        "          (let [x (nth xs i)]\n"
        "            (trace (keyword (str \"y\" i))\n"
        "                   (dist/gaussian <slope*x + intercept via mx ops> 0.5))))\n"
        "        (range (count xs)))\n\n"
        "Output ONLY the (fn [trace xs] ...) form.")})

(def cubic-task
  {:label  "cubic (truth y = 0.1·x³ + 0.5·x² + 1·x + 2)"
   :xs     [-3.0 -2.0 -1.0 0.0 1.0 2.0 3.0]
   ;; True means at xs: [0.8, 1.2, 1.4, 2, 3.6, 6.8, 12.2]
   :ys     [0.71 1.32 1.18 2.04 3.85 6.92 12.05]
   :truth  {'a 0.1 'b 0.5 'c 1.0 'd 2.0}
   :prompt-shape
   (str "Write a probabilistic program of one argument [trace xs] that models the\n"
        "relationship as a noisy CUBIC function:  y = a·x³ + b·x² + c·x + d + ε.\n\n"
        "Priors:    a, b, c, d ~ Gaussian(0, 10), at addresses :a, :b, :c, :d\n"
        "Noise std: 0.5\n\n"
        "Per-observation trace addresses MUST include the index, e.g.\n"
        "  (keyword (str \"y\" i))\n\n"
        "Use this shape:\n"
        "  (mapv (fn [i]\n"
        "          (let [x  (nth xs i)\n"
        "                x² (* x x)\n"
        "                x³ (* x² x)]\n"
        "            (trace (keyword (str \"y\" i))\n"
        "                   (dist/gaussian <a*x³ + b*x² + c*x + d via mx ops> 0.5))))\n"
        "        (range (count xs)))\n\n"
        "Output ONLY the (fn [trace xs] ...) form.")})

(def quadratic-task
  {:label  "quadratic (truth y = 0.5·x² + 1.5·x + 2)"
   :xs     [-3.0 -2.0 -1.0 0.0 1.0 2.0 3.0]
   :ys     [1.65 0.84 1.32 2.18 4.05 6.78 11.14]
   :truth  {'a 0.5 'b 1.5 'c 2.0}
   :prompt-shape
   (str "Write a probabilistic program of one argument [trace xs] that models the\n"
        "relationship as a noisy QUADRATIC function:  y = a·x² + b·x + c + ε.\n\n"
        "Priors:    a, b, c ~ Gaussian(0, 10), at addresses :a, :b, :c\n"
        "Noise std: 0.5\n\n"
        "Per-observation trace addresses MUST include the index, e.g.\n"
        "  (keyword (str \"y\" i))\n\n"
        "Use this shape:\n"
        "  (mapv (fn [i]\n"
        "          (let [x  (nth xs i)\n"
        "                x² (* x x)]\n"
        "            (trace (keyword (str \"y\" i))\n"
        "                   (dist/gaussian <a*x² + b*x + c via mx ops> 0.5))))\n"
        "        (range (count xs)))\n\n"
        "Output ONLY the (fn [trace xs] ...) form.")})

(defn format-data-table [xs ys]
  (str "  x: [" (str/join ", " (map #(.toFixed % 2) xs)) "]\n"
       "  y: [" (str/join ", " (map #(.toFixed % 2) ys)) "]"))

(def few-shot
  (str
   "Probabilistic programs are written as `(fn [trace ...args] body)` where\n"
   "`(trace <addr> <dist>)` samples and records under <addr>.\n"
   "Distributions: dist/gaussian (mu sigma), dist/uniform (lo hi),\n"
   "dist/bernoulli (p), dist/beta (alpha beta), dist/gamma (shape rate).\n"
   "For arithmetic on traced (MLX) values, use mx/add, mx/multiply, mx/scalar.\n\n"

   "Example — N IID Gaussian observations sharing one mean:\n"
   "(fn [trace n]\n"
   "  (let [mu (trace :mu (dist/gaussian 0 10))]\n"
   "    (mapv (fn [i] (trace (keyword (str \"obs\" i)) (dist/gaussian mu 1)))\n"
   "          (range n))))\n"))

(defn build-prompt [task]
  (str few-shot
       "\nNow I have these (x, y) data points:\n"
       (format-data-table (:xs task) (:ys task)) "\n\n"
       (:prompt-shape task)))

;; ===========================================================================
;; Math SCI env: re-eval the LLM's program with mx/* swapped for arithmetic
;; and dist/gaussian collapsed to its mean. Then the program becomes a
;; deterministic mean-evaluator, and we can probe parameters by passing a
;; closure-based `trace` that overrides specific addresses.
;; ===========================================================================

(def math-sci-env
  {:namespaces
   {'dist {'gaussian (fn [μ _σ] μ)}
    'mx   {'add      +
           'multiply *
           'subtract -
           'divide   /
           'scalar   identity
           'item     identity}}})

(defn build-math-evaluator
  "Eval `code` in the math env, then wrap so callers pass `overrides` (a map
   from param address → numeric value). Returns (fn [overrides xs] → [mean_i])."
  [code]
  (let [f (sci/eval-string code math-sci-env)]
    (fn [overrides xs]
      (let [trace-fn (fn [addr v] (get overrides addr v))]
        (f trace-fn xs)))))

;; ===========================================================================
;; Recognizer: walk the parsed form, find param defs and observation gaussian.
;; ===========================================================================

(defn extract-param-defs
  "Find top-level let bindings of shape  <sym> (trace <kw-addr> (dist/gaussian _ <std>)).
   Returns vector of {:symbol s :addr a :prior-std σ}. Order matters — these become
   the columns of the design matrix."
  [parsed]
  (when (and (seq? parsed) (= 'fn (first parsed)))
    (let [body (drop 2 parsed)
          let-form (first body)]
      (when (and (seq? let-form) (= 'let (first let-form)))
        (let [bindings (second let-form)]
          (->> (partition 2 bindings)
               (keep (fn [[sym rhs]]
                       (when (and (seq? rhs) (= 'trace (first rhs)))
                         (let [addr (nth rhs 1)
                               dist-call (nth rhs 2)]
                           (when (and (keyword? addr)
                                      (seq? dist-call)
                                      (= 'dist/gaussian (first dist-call))
                                      (number? (nth dist-call 2)))
                             {:symbol sym
                              :addr addr
                              :prior-std (nth dist-call 2)})))))
               vec))))))

(defn find-observation-trace
  "Recursively find the first (trace <computed-addr> (dist/gaussian _ σ)) call.
   `<computed-addr>` is a function call (e.g. `(keyword (str \"y\" i))`), as
   opposed to a literal keyword (which marks parameter trace sites).
   Returns {:noise-std σ} or nil."
  [form]
  (cond
    (and (seq? form) (= 'trace (first form))
         (>= (count form) 3))
    (let [addr (nth form 1)
          dist (nth form 2)]
      (if (and (not (keyword? addr))
               (seq? dist) (= 'dist/gaussian (first dist))
               (number? (nth dist 2)))
        {:noise-std (nth dist 2)}
        (some find-observation-trace form)))
    (sequential? form) (some find-observation-trace form)
    :else nil))

;; ===========================================================================
;; Probe evaluation → design matrix
;; ===========================================================================

(defn design-matrix
  "For each param, set its override to 1.0 and others to 0.0; the evaluator
   returns the n means; that is the column φ_·j of the design matrix.
   k evaluations total. Returns n×k cljs vector-of-vectors."
  [evaluator param-addrs xs]
  (let [zeros   (zipmap param-addrs (repeat 0.0))
        cols    (mapv (fn [pa]
                        (vec (evaluator (assoc zeros pa 1.0) xs)))
                      param-addrs)
        n       (count xs)]
    (mapv (fn [i] (mapv #(nth % i) cols)) (range n))))

(defn linearity-ok?
  "Set all params to 2.0; the evaluator's means must equal 2 · row-sum(X)."
  [evaluator param-addrs xs X]
  (let [actual    (vec (evaluator (zipmap param-addrs (repeat 2.0)) xs))
        predicted (mapv (fn [row] (* 2.0 (reduce + row))) X)
        max-err   (apply max (mapv #(Math/abs (- %1 %2)) actual predicted))]
    {:ok? (< max-err 1e-9) :max-err max-err}))

;; ===========================================================================
;; Closed-form Gaussian posterior for the recognized linear model
;; ===========================================================================

(defn matrix-inverse-general
  "Gauss-Jordan elimination with partial pivoting. Works for any k×k.
   Returns the inverse or nil on singular input."
  [m]
  (let [n (count m)
        ;; Augmented matrix [m | I]: each row is (row-i ++ unit-vector-i)
        aug0 (vec (map-indexed
                   (fn [i row]
                     (vec (concat row
                                  (map (fn [j] (if (= i j) 1.0 0.0)) (range n)))))
                   m))
        result
        (loop [aug aug0, col 0]
          (if (>= col n)
            aug
            (let [pivot-row (apply max-key
                                   #(Math/abs (get-in aug [% col]))
                                   (range col n))
                  pivot-val (get-in aug [pivot-row col])]
              (if (< (Math/abs pivot-val) 1e-12)
                nil  ; singular
                (let [aug (if (= pivot-row col)
                            aug
                            (assoc aug col (nth aug pivot-row)
                                       pivot-row (nth aug col)))
                      aug (assoc aug col
                                 (mapv #(/ % pivot-val) (nth aug col)))
                      aug (reduce
                           (fn [a r]
                             (if (= r col) a
                                 (let [factor (get-in a [r col])]
                                   (assoc a r
                                          (mapv #(- %1 (* factor %2))
                                                (nth a r) (nth a col))))))
                           aug
                           (range n))]
                  (recur aug (inc col)))))))]
    (when result (mapv #(subvec % n (* 2 n)) result))))

(defn matrix-inverse
  "Cofactor inverse for k ∈ {1, 2, 3} (cheap and exact); Gauss-Jordan for k ≥ 4."
  [m]
  (case (count m)
    1 [[(/ 1.0 (get-in m [0 0]))]]
    2 (let [[[a b] [c d]] m
            det (- (* a d) (* b c))]
        [[(/ d det)       (- (/ b det))]
         [(- (/ c det))   (/ a det)]])
    3 (let [[[a b c] [d e f] [g h i]] m
            det (+ (* a (- (* e i) (* f h)))
                   (- (* b (- (* d i) (* f g))))
                   (* c (- (* d h) (* e g))))]
        (mapv (fn [row] (mapv #(/ % det) row))
              [[(- (* e i) (* f h))  (- (* c h) (* b i))  (- (* b f) (* c e))]
               [(- (* f g) (* d i))  (- (* a i) (* c g))  (- (* c d) (* a f))]
               [(- (* d h) (* e g))  (- (* b g) (* a h))  (- (* a e) (* b d))]]))
    (matrix-inverse-general m)))

(defn matrix-mul
  "Multiply k×m by m×n → k×n."
  [A B]
  (let [k (count A), m (count (first A)), n (count (first B))]
    (mapv (fn [i]
            (mapv (fn [j]
                    (reduce + (map (fn [t] (* (get-in A [i t]) (get-in B [t j])))
                                   (range m))))
                  (range n)))
          (range k))))

(defn matvec-mul
  "Multiply k×n matrix by n-vector → k-vector."
  [A v]
  (mapv (fn [row] (reduce + (map * row v))) A))

(defn transpose [M] (apply mapv vector M))

(defn closed-form-posterior
  "X is n×k design, ys is n-vec, prior-stds is k-vec, noise-std is scalar.
   Returns {:means [μ_1..k] :stds [σ_1..k] :covariance [[..]]} or nil if k > 3."
  [X ys prior-stds noise-std]
  (let [k    (count prior-stds)
        σ²   (* noise-std noise-std)
        Xᵀ   (transpose X)
        XᵀX  (matrix-mul Xᵀ X)
        ;; Precision = XᵀX/σ² + diag(1/τ²)
        prec (mapv (fn [i]
                     (mapv (fn [j]
                             (let [base (/ (get-in XᵀX [i j]) σ²)]
                               (if (= i j)
                                 (+ base (/ 1.0 (* (nth prior-stds i)
                                                   (nth prior-stds i))))
                                 base)))
                           (range k)))
                   (range k))
        cov  (matrix-inverse prec)]
    (when cov
      (let [Xᵀy  (matvec-mul Xᵀ ys)
            b    (mapv #(/ % σ²) Xᵀy)
            μ    (matvec-mul cov b)
            σs   (mapv (fn [i] (Math/sqrt (get-in cov [i i]))) (range k))]
        {:means μ :stds σs :covariance cov}))))

;; ===========================================================================
;; Top-level analysis
;; ===========================================================================

(defn analyze-as-bayesian-linear
  "Try to recognize `code` as Bayesian linear regression in features φ(x).
   On success: {:type :bayesian-linear :params [...] :design-matrix [...]
                :prior-stds [...] :noise-std σ :posterior {...}}.
   Otherwise:  {:type :unknown :reason \"…\"}."
  [code parsed-form xs ys]
  (let [params (extract-param-defs parsed-form)]
    (cond
      (empty? params)
      {:type :unknown :reason "no Gaussian parameter definitions found in top-level let"}

      :else
      (let [obs (find-observation-trace parsed-form)]
        (cond
          (nil? obs)
          {:type :unknown :reason "no observation gaussian found inside the loop"}

          :else
          (let [evaluator   (build-math-evaluator code)
                param-addrs (mapv :addr params)
                X           (design-matrix evaluator param-addrs xs)
                lin         (linearity-ok? evaluator param-addrs xs X)]
            (cond
              (not (:ok? lin))
              {:type :unknown
               :reason (str "model is not linear in parameters; max numerical error = "
                            (.toExponential (:max-err lin) 2))}

              :else
              (let [prior-stds (mapv :prior-std params)
                    noise-std  (:noise-std obs)
                    post       (closed-form-posterior X ys prior-stds noise-std)]
                (if post
                  {:type :bayesian-linear
                   :params params
                   :design-matrix X
                   :prior-stds prior-stds
                   :noise-std noise-std
                   :posterior post}
                  {:type :unknown
                   :reason "matrix inversion failed (singular precision matrix)"})))))))))

;; ===========================================================================
;; Helpers (synthesis side, copied from previous probes)
;; ===========================================================================

(defn extract-code [text]
  (let [t (str/trim (or text ""))]
    (cond
      (str/blank? t) ""
      (re-find #"```" t)
      (let [m (re-find #"```(?:clojure|cljs|clj|clojurescript)?\s*\n?([\s\S]*?)```" t)]
        (if m (str/trim (nth m 1)) t))
      (str/starts-with? t "(") t
      :else (let [i (str/index-of t "(")] (if i (subs t i) "")))))

(defn parse-cljs [code]
  (try {:ok? true :form (eda/parse-string code {:all true})}
       (catch :default e {:ok? false :err (.-message e)})))

(defn now-ms [] (.now js/performance))
(defn fmt-ms [t0] (str (.toFixed (- (now-ms) t0) 0) "ms"))
(defn pad-string [s n]
  (str s (apply str (repeat (max 0 (- n (count s))) " "))))

;; ===========================================================================
;; Per-task pipeline
;; ===========================================================================

(defn run-task [m task]
  (println "\n────────────────────────────────────────────────────────────────────")
  (println " TASK:" (:label task))
  (println "────────────────────────────────────────────────────────────────────")
  (pr/let
    [_    (println "[a] Asking the LLM to propose a program…")
     t1   (now-ms)
     text (llm/generate-text-raw
           m (build-prompt task)
           {:max-tokens 350 :temperature 0 :system-prompt system-prompt})
     code (extract-code text)
     _    (println "    gen time:" (fmt-ms t1))
     _    (println "    code:")
     _    (doseq [line (str/split code #"\n")] (println "    │" line))

     parse (parse-cljs code)
     _     (println "[b] Parse:" (if (:ok? parse) "✓" (str "✗ " (:err parse))))

     _    (println "[c] Structural recognition…")
     t2   (now-ms)
     analysis (when (:ok? parse)
                (try (analyze-as-bayesian-linear code (:form parse) (:xs task) (:ys task))
                     (catch :default e
                       {:type :error :reason (.-message e)})))
     _    (println "    recognition time:" (fmt-ms t2))]

    (case (or (:type analysis) :no-analysis)
      :bayesian-linear
      (let [post   (:posterior analysis)
            params (:params analysis)]
        (println "[d] ✓ Recognized as Bayesian linear regression")
        (println "    parameters extracted:")
        (doseq [p params]
          (println (str "      "
                        (pad-string (name (:symbol p)) 12)
                        "addr=" (:addr p)
                        "  prior=N(0," (:prior-std p) ")")))
        (println "    noise std :" (:noise-std analysis))
        (println "    design matrix (n×k):")
        (doseq [[i row] (map-indexed vector (:design-matrix analysis))]
          (println "      x_" i ":" (mapv #(.toFixed % 4) row)))
        (println "[e] Closed-form posterior:")
        (doseq [[i p] (map-indexed vector params)]
          (let [μ (nth (:means post) i)
                σ (nth (:stds post) i)
                truth-val (get (:truth task) (:symbol p))
                err (when truth-val (Math/abs (- μ truth-val)))]
            (println (str "      "
                          (pad-string (name (:symbol p)) 12)
                          "= " (.toFixed μ 4)
                          " ± " (.toFixed σ 4)
                          (when truth-val
                            (str "    truth = " truth-val
                                 "    Δ = " (.toFixed err 4)))))))
        (println (str "[f] Total inference time: " (fmt-ms t2)
                      " (vs ~7-50s with HMC)")))

      :unknown
      (do (println "[d] ✗ NOT recognized as Bayesian linear regression")
          (println "    reason:" (:reason analysis))
          (println "    would fall through to HMC here"))

      :error
      (println "[d] error during recognition:" (:reason analysis))

      :no-analysis
      (println "[d] skipped (parse failed)"))))

;; ===========================================================================
;; Run
;; ===========================================================================

(println "============================================================")
(println " Qwen3.6-35B-A3B-4bit  —  Structural recognizer (Bayesian linear regression)")
(println "============================================================")

(pr/let
  [_   (println "\nLoading model…")
   t0  (now-ms)
   m   (llm/load-model model-path)
   _   (println "  type:" (:type m) "  load:" (fmt-ms t0))

   _   (run-task m linear-task)
   _   (run-task m quadratic-task)
   _   (run-task m cubic-task)

   _   (println "\n============================================================")
   _   (println " Both tasks: structure recognized → closed-form posterior")
   _   (println " No HMC. No tuning. No anisotropy headache.")
   _   (println "============================================================\n")]
  nil)
