(ns examples.qwen36-regression
  "Cognition as program synthesis, on real data.

   Setup: ground truth y = 2x + 1 + N(0, 0.5), seven (x, y) points.
   We hand the data to the LLM as text, ask it to write a probabilistic
   program that explains the relationship, then run inference on the
   synthesized program to see if the latents recover the true parameters.

   Pipeline:
     show data → LLM → cljs gen function → parse → SCI eval → wrap
       → p/simulate (sanity)
       → p/generate × N constraining :y0..:y6 to observations
       → self-normalized weighted posterior on (slope, intercept)
       → compare to ground truth.

   This is the canonical first probe of 'cognition as program synthesis':
   the agent (the LLM) proposes the program; inference grades it against
   data; the trace records the explanation.

   Run: bun run --bun nbb examples/qwen36_regression.cljs"
  (:require [genmlx.llm.backend :as llm]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.mcmc :as mcmc]
            [edamame.core :as eda]
            [sci.core :as sci]
            [clojure.string :as str]
            [promesa.core :as pr])
  (:require-macros [genmlx.gen :refer [gen]]))

(def model-path
  (str (.-HOME js/process.env) "/.cache/models/Qwen3.6-35B-A3B-4bit"))

;; ---------------------------------------------------------------------------
;; Ground truth and data.
;; The ys were generated offline as true-slope*x + true-intercept + N(0, 0.5)
;; and frozen here so the demo is fully reproducible across runs.
;; ---------------------------------------------------------------------------

(def true-a    0.5)   ; coefficient on x²
(def true-b    1.5)   ; coefficient on x
(def true-c    2.0)   ; intercept
(def noise-std 0.5)

(def xs [-3.0 -2.0 -1.0 0.0 1.0 2.0 3.0])
;; ys = 0.5·x² + 1.5·x + 2 + N(0, 0.5).  True means: [2, 1, 1, 2, 4, 7, 11].
(def ys [1.65 0.84 1.32 2.18 4.05 6.78 11.14])

(def n-points (count xs))

;; HMC settings (vectorized — N parallel chains)
(def n-chains          50)
(def samples-per-chain 20)
(def n-hmc-samples     (* n-chains samples-per-chain))
(def n-burn            200)   ; quadratic posterior is anisotropic, needs more burn

;; Step-size / leapfrog sweep, retuned for the quadratic problem.
;; Analytical posterior stds: σ_a≈0.055, σ_b≈0.095, σ_c≈0.289 — so the
;; tightest direction (a) sets the ceiling on usable ε. Need ε ≲ σ_a / 2.
(def hmc-configs
  [{:label "ε=0.005 L=40"             :step-size 0.005 :leapfrog-steps 40}
   {:label "ε=0.01  L=20"             :step-size 0.01  :leapfrog-steps 20}
   {:label "ε=0.02  L=20"             :step-size 0.02  :leapfrog-steps 20}
   {:label "ε=0.03  L=15"             :step-size 0.03  :leapfrog-steps 15}
   {:label "ε=0.05  L=10  (linear ε)" :step-size 0.05  :leapfrog-steps 10}])

;; Prior std on slope and intercept (per the prompt we send to the LLM)
(def prior-std 10.0)

;; ---------------------------------------------------------------------------
;; SCI environment
;; ---------------------------------------------------------------------------

(def sci-env
  {:namespaces
   {'dist {'gaussian    dist/gaussian
           'uniform     dist/uniform
           'bernoulli   dist/bernoulli
           'beta        dist/beta-dist
           'gamma       dist/gamma-dist
           'exponential dist/exponential}
    'mx   {'add      mx/add
           'subtract mx/subtract
           'multiply mx/multiply
           'divide   mx/divide
           'scalar   mx/scalar
           'item     mx/item}}})

;; ---------------------------------------------------------------------------
;; Prompt
;; ---------------------------------------------------------------------------

(defn format-data-table [xs ys]
  (str "  x: [" (str/join ", " (map #(.toFixed % 2) xs)) "]\n"
       "  y: [" (str/join ", " (map #(.toFixed % 2) ys)) "]"))

(def system-prompt
  "You are a probabilistic programming assistant. Output only valid ClojureScript code.")

(defn build-prompt [xs ys]
  (str
   "Probabilistic programs are written as `(fn [trace ...args] body)` where\n"
   "`(trace <addr> <dist>)` samples and records under <addr>.\n"
   "Distributions: dist/gaussian (mu sigma), dist/uniform (lo hi),\n"
   "dist/bernoulli (p), dist/beta (alpha beta), dist/gamma (shape rate),\n"
   "dist/exponential (rate).\n"
   "For arithmetic on traced (MLX) values, use mx/add, mx/multiply, mx/scalar.\n\n"

   "Example — N IID Gaussian observations sharing one mean:\n"
   "(fn [trace n]\n"
   "  (let [mu (trace :mu (dist/gaussian 0 10))]\n"
   "    (mapv (fn [i] (trace (keyword (str \"obs\" i)) (dist/gaussian mu 1)))\n"
   "          (range n))))\n\n"

   "Now I have these (x, y) data points:\n"
   (format-data-table xs ys) "\n\n"

   "Write a probabilistic program of one argument [trace xs] that models the\n"
   "relationship as a noisy QUADRATIC function:  y = a·x² + b·x + c + ε.\n\n"
   "Priors:    a, b, c ~ Gaussian(0, 10), each at addresses :a, :b, :c\n"
   "Noise std: " (.toFixed noise-std 2) "\n\n"
   "Per-observation trace addresses MUST include the index, e.g.\n"
   "  (keyword (str \"y\" i))   — produces :y0, :y1, :y2, …\n"
   "Do NOT use a bare :y keyword (every trace site needs a UNIQUE address,\n"
   "otherwise later observations overwrite earlier ones).\n\n"
   "Use this exact iteration shape:\n"
   "  (mapv (fn [i]\n"
   "          (let [x  (nth xs i)\n"
   "                x² (* x x)]                    ; JS multiplication, x is a number\n"
   "            (trace (keyword (str \"y\" i))\n"
   "                   (dist/gaussian <a*x² + b*x + c via mx ops> "
                                        (.toFixed noise-std 2) "))))\n"
   "        (range (count xs)))\n\n"
   "Output ONLY the (fn [trace xs] ...) form."))

;; ---------------------------------------------------------------------------
;; Inline helpers (same as previous probes)
;; ---------------------------------------------------------------------------

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

(defn sci-eval-fn [code]
  (try
    (let [v (sci/eval-string code sci-env)]
      (cond
        (fn? v) {:fn v}
        (and (var? v) (fn? (deref v))) {:fn (deref v)}
        :else {:err (str "value is not a function: " (pr-str v))}))
    (catch :default e {:err (.-message e)})))

(defn wrap-as-gf [model-fn]
  (dyn/auto-key (gen [xs] (model-fn trace xs))))

(defn ys->constraints
  "Constrain :y0..:yN to the observed numeric values (as MLX float scalars)."
  [ys]
  (reduce (fn [cm [i v]]
            (cm/set-choice cm
                           [(keyword (str "y" i))]
                           (mx/scalar v)))
          cm/EMPTY
          (map-indexed vector ys)))

(defn log-sum-exp [xs]
  (let [m (apply max xs)]
    (+ m (Math/log (reduce + (map #(Math/exp (- % m)) xs))))))

(defn now-ms [] (.now js/performance))
(defn fmt-ms [t0] (str (.toFixed (- (now-ms) t0) 0) "ms"))
(defn pad [s n] (str s (apply str (repeat (max 0 (- n (count s))) " "))))

;; ---------------------------------------------------------------------------
;; HMC posterior summary. HMC samples are unweighted draws from the posterior
;; (after burn-in), so plain mean/std over the samples is the right estimator.
;; ---------------------------------------------------------------------------

(defn unwrap-mlx [v] (cond-> v (mx/array? v) mx/item))

(defn extract-samples-vec
  "vectorized-hmc returns a vector of [n-chains, n-params] nested vectors —
   one entry per *sample*, with all chains' values along the first dim.
   Flatten across (sample, chain) and extract parameter `idx`."
  [samples idx]
  (vec (for [sample samples, chain sample] (nth chain idx))))

;; Order of parameters in the HMC sample arrays matches the :addresses vector
(def param-idx {:a 0 :b 1 :c 2})

(defn mean+std [xs]
  (let [n (count xs)
        mean (/ (reduce + xs) n)
        var (/ (reduce + (map #(Math/pow (- % mean) 2) xs)) n)]
    {:mean mean :std (Math/sqrt var)}))

;; ---------------------------------------------------------------------------
;; Closed-form Bayesian quadratic regression posterior.
;;
;;   priors:      a, b, c  ~ N(0, τ²) independently
;;   likelihood:  y_i ~ N(a·x_i² + b·x_i + c, σ²)
;;
;; Build X with rows [x_i², x_i, 1].  Posterior on β = [a, b, c]:
;;   Σ_post = ( XᵀX / σ²  +  I / τ² )⁻¹       (3×3 matrix)
;;   μ_post = Σ_post  ·  Xᵀy / σ²
;; Computed via MLX matrix ops.
;; ---------------------------------------------------------------------------

;; Closed-form Bayesian quadratic regression by hand on a 3×3 system,
;; using cofactors so we don't have to call mx/inv (which appears to
;; crash on this nbb/mlx-node combo for small dense matrices).
;;
;; Build precision P = XᵀX / σ² + I / τ² where X has rows [x², x, 1].
;; Then μ = P⁻¹ · Xᵀy / σ², and σ_i = √(P⁻¹)_ii.
(defn analytical-bayesian-quadreg [xs ys σ τ]
  (let [σ² (* σ σ)
        τ² (* τ τ)
        n  (count xs)
        Σx⁴ (reduce + (map #(let [x² (* % %)] (* x² x²)) xs))
        Σx³ (reduce + (map #(* % % %) xs))
        Σx² (reduce + (map #(* % %) xs))
        Σx  (reduce + xs)
        Σx²y (reduce + (map (fn [x y] (* x x y)) xs ys))
        Σxy  (reduce + (map * xs ys))
        Σy   (reduce + ys)
        ;; Precision P (symmetric 3×3):
        ;;   [Σx⁴/σ²+1/τ²,  Σx³/σ²,         Σx²/σ²]
        ;;   [Σx³/σ²,       Σx²/σ²+1/τ²,    Σx /σ²]
        ;;   [Σx²/σ²,       Σx /σ²,         n  /σ²+1/τ²]
        p11 (+ (/ Σx⁴ σ²) (/ 1.0 τ²))
        p12 (/ Σx³ σ²)
        p13 (/ Σx² σ²)
        p22 (+ (/ Σx² σ²) (/ 1.0 τ²))
        p23 (/ Σx  σ²)
        p33 (+ (/ n   σ²) (/ 1.0 τ²))
        ;; Determinant by cofactor expansion (P is symmetric)
        det (- (* p11 (- (* p22 p33) (* p23 p23)))
               (- (* p12 (- (* p12 p33) (* p23 p13)))
                  (* p13 (- (* p12 p23) (* p22 p13)))))
        ;; Cofactor matrix entries (symmetric)
        c11 (- (* p22 p33) (* p23 p23))
        c12 (- (- (* p12 p33) (* p23 p13)))
        c13 (- (* p12 p23) (* p22 p13))
        c22 (- (* p11 p33) (* p13 p13))
        c23 (- (- (* p11 p23) (* p12 p13)))
        c33 (- (* p11 p22) (* p12 p12))
        ;; P⁻¹ entries
        i11 (/ c11 det) i12 (/ c12 det) i13 (/ c13 det)
        i22 (/ c22 det) i23 (/ c23 det)
        i33 (/ c33 det)
        ;; Xᵀy / σ²
        b1 (/ Σx²y σ²) b2 (/ Σxy σ²) b3 (/ Σy σ²)
        ;; μ = P⁻¹ · Xᵀy/σ²
        μa (+ (* i11 b1) (* i12 b2) (* i13 b3))
        μb (+ (* i12 b1) (* i22 b2) (* i23 b3))
        μc (+ (* i13 b1) (* i23 b2) (* i33 b3))]
    {:a-mean μa :a-std (Math/sqrt i11)
     :b-mean μb :b-std (Math/sqrt i22)
     :c-mean μc :c-std (Math/sqrt i33)}))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(println "============================================================")
(println " Qwen3.6-35B-A3B-4bit  —  Cognition as program synthesis on (x,y) data")
(println "============================================================")
(println " ground truth: y =" true-a "x² +" true-b "x +" true-c "+ N(0," noise-std ")")
(println " data:")
(doseq [[x y] (map vector xs ys)]
  (println (str "    x = " (.toFixed x 2) "    y = " (.toFixed y 2))))
(println " vectorized HMC :" n-chains "chains ×" samples-per-chain "samples = "
         n-hmc-samples "total per config (" n-burn "burn)")
(println " sweeping" (count hmc-configs) "(step-size, leapfrog) configs")
(println "============================================================")

(pr/let
  [_   (println "\n[1] Loading model")
   t0  (now-ms)
   m   (llm/load-model model-path)
   _   (println "    type:" (:type m) "  load:" (fmt-ms t0))

   _   (println "\n[2] Asking the LLM to propose a program for the data…")
   t1  (now-ms)
   text (llm/generate-text-raw
         m (build-prompt xs ys)
         {:max-tokens 350
          :temperature 0
          :system-prompt system-prompt})
   _   (println "    gen time:" (fmt-ms t1))
   code (extract-code text)
   _   (println "    code (LLM-emitted):")
   _   (doseq [line (str/split code #"\n")] (println "    │" line))

   parse (parse-cljs code)
   _   (println "    parse:" (if (:ok? parse) "✓" (str "✗ " (:err parse))))

   _   (println "\n[3] SCI eval & wrap")
   evald (when (:ok? parse) (sci-eval-fn code))
   _   (cond
         (nil? evald) (println "    skipped")
         (:err evald) (println "    err:" (:err evald))
         :else (println "    ✓"))

   gf  (when (:fn evald) (wrap-as-gf (:fn evald)))
   _   (println "\n[4] Sanity simulate (LLM's program runs end-to-end?)")
   sim (when gf (try (p/simulate gf [xs])
                     (catch :default e {:err (.-message e)})))
   sim-ok? (and sim (not (:err sim)))
   _   (when sim-ok?
         (let [paths (cm/addresses (:choices sim))]
           (println "    trace sites:" (count paths))
           (println "    addresses :" paths)
           (println "    log-prior :" (.toFixed (mx/item (:score sim)) 4))))
   _   (when (:err sim) (println "    err:" (:err sim)))

   ;; Validate the LLM gave each observation a unique address. Without this,
   ;; HMC tries to compute gradients on a malformed trace structure and the
   ;; MLX layer crashes with SIGTRAP.
   expected-sites (+ 3 n-points)            ; :a :b :c + :y0 .. :y(n-1)
   structure-valid? (when sim-ok?
                      (= expected-sites (count (cm/addresses (:choices sim)))))
   _   (when sim-ok?
         (println "    structure :" (if structure-valid? "✓" "✗")
                  (str "(expected " expected-sites " sites, got "
                       (count (cm/addresses (:choices sim))) ")")))
   _   (when (and sim-ok? (not structure-valid?))
         (println "    Aborting — LLM did not give each observation a unique address."))

   ;; -------- Constrain y0..yN, run vectorized HMC over a sweep --------
   _   (println "\n[5] Constraining observations and sweeping HMC configurations")
   _   (println "    -- building constraints --")
   constraints (ys->constraints ys)
   _   (println "    -- computing analytical posterior --")
   analytical (analytical-bayesian-quadreg xs ys noise-std prior-std)
   _   (println "    -- analytical done --")
   _   (println "    analytical a (x²) :" (.toFixed (:a-mean analytical) 4)
                "± " (.toFixed (:a-std analytical) 4)
                "  (truth" true-a ")")
   _   (println "    analytical b (x ) :" (.toFixed (:b-mean analytical) 4)
                "± " (.toFixed (:b-std analytical) 4)
                "  (truth" true-b ")")
   _   (println "    analytical c (1 ) :" (.toFixed (:c-mean analytical) 4)
                "± " (.toFixed (:c-std analytical) 4)
                "  (truth" true-c ")")

   sweep-results
   (when (and gf structure-valid?)
     (mapv (fn [cfg]
             (let [t2 (now-ms)
                   r  (try (mcmc/vectorized-hmc
                            {:samples       samples-per-chain
                             :n-chains      n-chains
                             :burn          n-burn
                             :step-size     (:step-size cfg)
                             :leapfrog-steps (:leapfrog-steps cfg)
                             :addresses     [:a :b :c]
                             :device        :gpu}
                            gf [xs] constraints)
                           (catch :default e {:err (.-message e)}))
                   ms (- (now-ms) t2)]
               (if (:err r)
                 (assoc cfg :status :err :err (:err r) :time-ms ms)
                 (let [a-vals (extract-samples-vec r 0)
                       b-vals (extract-samples-vec r 1)
                       c-vals (extract-samples-vec r 2)
                       sa     (mean+std a-vals)
                       sb     (mean+std b-vals)
                       sc     (mean+std c-vals)
                       acc    (or (:acceptance-rate (meta r)) 0.0)]
                   (assoc cfg
                          :status :ok :time-ms ms :accept acc
                          :a sa :b sb :c sc
                          :a-Δ (Math/abs (- (:mean sa) (:a-mean analytical)))
                          :b-Δ (Math/abs (- (:mean sb) (:b-mean analytical)))
                          :c-Δ (Math/abs (- (:mean sc) (:c-mean analytical))))))))
           hmc-configs))

   ;; -------- Per-config detail --------
   _   (doseq [r sweep-results]
         (println (str "\n  ─── " (:label r) " ───"))
         (case (:status r)
           :ok  (do
                  (println "    time   :" (.toFixed (:time-ms r) 0) "ms"
                           "  accept:" (.toFixed (* 100.0 (:accept r)) 1) "%")
                  (println (str "    a (x²)  " (.toFixed (:mean (:a r)) 4)
                                " ± " (.toFixed (:std (:a r)) 4)
                                "   Δ: " (.toFixed (:a-Δ r) 4)))
                  (println (str "    b (x )  " (.toFixed (:mean (:b r)) 4)
                                " ± " (.toFixed (:std (:b r)) 4)
                                "   Δ: " (.toFixed (:b-Δ r) 4)))
                  (println (str "    c (1 )  " (.toFixed (:mean (:c r)) 4)
                                " ± " (.toFixed (:std (:c r)) 4)
                                "   Δ: " (.toFixed (:c-Δ r) 4))))
           :err (println "    ERROR :" (:err r))))

   ;; -------- Comparison table --------
   _   (println "\n[6] Sweep summary")
   _   (println (str "    " (pad "config" 30) (pad "time" 10) (pad "accept" 9)
                     (pad "a-mean" 11) (pad "a-Δ" 9)
                     (pad "b-mean" 11) (pad "b-Δ" 9)
                     (pad "c-mean" 11) (pad "c-Δ" 9)))
   _   (println "    ───────────────────────────────────────────────────────────────────────────────────────────────────────")
   _   (doseq [r sweep-results]
         (when (= :ok (:status r))
           (println (str "    "
                         (pad (:label r) 30)
                         (pad (str (.toFixed (:time-ms r) 0) "ms") 10)
                         (pad (str (.toFixed (* 100.0 (:accept r)) 1) "%") 9)
                         (pad (.toFixed (:mean (:a r)) 4) 11)
                         (pad (.toFixed (:a-Δ r) 4) 9)
                         (pad (.toFixed (:mean (:b r)) 4) 11)
                         (pad (.toFixed (:b-Δ r) 4) 9)
                         (pad (.toFixed (:mean (:c r)) 4) 11)
                         (pad (.toFixed (:c-Δ r) 4) 9)))))

   _   (println "\n============================================================\n")]
  nil)
