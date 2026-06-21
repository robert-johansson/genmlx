(ns genmlx.llm.msa-score
  "Native-free model eval + scoring spine for the Model Synthesis Architecture
   (extracted from genmlx.llm.msa, genmlx-ugkv — same move as genmlx.codegen.eval
   vs genmlx.llm.codegen, genmlx-t246).

   These helpers turn a probabilistic-program CODE STRING into a GenMLX DynamicGF
   (SCI eval in a sandbox exposing dist/* and mx/*) and SCORE it against
   observations — returning Bayesian model evidence (exact analytical marginal for
   conjugate/eliminable models, importance-sampling log-mean-exp otherwise). They
   depend ONLY on the pure GenMLX core (mlx, dist, dynamic, protocols, choicemap,
   method-selection) + sci/reader — NOT on genmlx.llm.backend, which ESM-imports the
   native LM addon. So the Phase-1 reward path (genmlx.world.train-reward) can score
   generated programs WITHOUT loading the policy LLM at all — making reward purity a
   load-time guarantee, not just a convention.

   genmlx.llm.msa re-exports every public name here, so existing callers are
   unchanged."
  (:require [sci.core :as sci]
            [cljs.reader :as reader]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.method-selection :as ms])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ============================================================
;; SCI evaluation context
;; ============================================================

(def msa-sci-opts
  "SCI options exposing GenMLX distributions and MLX ops to eval'd code.
   Code evaluated with these opts can reference dist/gaussian, mx/add, etc."
  {:namespaces
   {'dist {'gaussian dist/gaussian
           'normal dist/gaussian
           'uniform dist/uniform
           'bernoulli dist/bernoulli
           ;; Canonical genmlx.dist names — these are the symbols whose name the
           ;; schema walker extracts as :dist-type, so they must match the
           ;; conjugacy table keys (:beta-dist, :gamma-dist). The short aliases
           ;; (beta, gamma) are kept for LLM-friendliness and resolve to the
           ;; same distribution; code->source-form rewrites them to canonical.
           'beta-dist dist/beta-dist
           'gamma-dist dist/gamma-dist
           'beta dist/beta-dist
           'gamma dist/gamma-dist
           'exponential dist/exponential
           'poisson dist/poisson
           'categorical dist/categorical
           'dirichlet dist/dirichlet
           'multivariate-normal dist/multivariate-normal
           'delta dist/delta}
    'mx {'add mx/add
         'subtract mx/subtract
         'multiply mx/multiply
         'divide mx/divide
         'scalar mx/scalar
         'item mx/item
         'exp mx/exp
         'log mx/log
         'sqrt mx/sqrt
         'abs mx/abs}}})

(defn eval-model-fn
  "Evaluate a code string in the MSA SCI context.
   Expects the string to produce a (fn [trace] ...) value.
   Returns the function, or throws on syntax/eval errors."
  [code-str]
  (sci/eval-string code-str msa-sci-opts))

(def ^:private canonical-dist-syms
  "Rewrite map from short dist aliases (as the LLM/parser emit them) to the
   canonical genmlx.dist symbol names. The schema walker derives :dist-type from
   the symbol name, and the conjugacy table keys on the canonical names, so the
   source form handed to make-gen-fn must use them for conjugacy to fire."
  {'dist/beta 'dist/beta-dist
   'dist/gamma 'dist/gamma-dist})

(defn- normalize-dist-syms
  "Recursively rewrite dist constructor symbols in a source form to their
   canonical genmlx.dist names (see canonical-dist-syms)."
  [form]
  (cond
    (seq? form)    (map normalize-dist-syms form)
    (vector? form) (mapv normalize-dist-syms form)
    (map? form)    (into {} (map (fn [[k v]] [(normalize-dist-syms k)
                                              (normalize-dist-syms v)]))
                         form)
    (symbol? form) (get canonical-dist-syms form form)
    :else form))

(defn code->source-form
  "Read a (fn [params] body...) code string and rewrite it into a gen source
   form ([] body...) with canonical dist symbols. This is what make-gen-fn walks
   to extract a faithful schema (real keyword trace-sites), so L1 compilation and
   L3 conjugacy detection fire on synthesized models exactly as they do on
   hand-written (gen ...) models.

   The body is unchanged — `trace` stays as the gen-runtime local. Model args
   are [] (synthesized models are zero-argument). Returns nil if the string does
   not read as an (fn [..] ..) form, so callers can fall back to the opaque
   wrapper without losing the model."
  [code-str]
  (try
    (let [form (reader/read-string code-str)]
      (when (and (seq? form)
                 (symbol? (first form))
                 (contains? #{"fn" "fn*"} (name (first form)))
                 (vector? (second form)))
        (normalize-dist-syms (list* [] (drop 2 form)))))
    (catch :default _ nil)))

(defn wrap-model
  "Wrap a (fn [trace] body) into a zero-argument DynamicGF.

   With a source form (([] body...) — see code->source-form), make-gen-fn
   extracts a faithful schema: real keyword trace-sites that drive L1 compilation
   and L3 conjugacy detection. Without one, falls back to the opaque
   (gen [] (model-fn trace)) wrapper, whose schema sees no trace-sites.

   Execution is identical either way: the SCI closure model-fn is what runs,
   invoked with the gen-runtime's trace closure."
  ([model-fn]
   (dyn/auto-key (gen [] (model-fn trace))))
  ([model-fn source-form]
   (if source-form
     (dyn/auto-key (dyn/make-gen-fn (fn [rt] (model-fn (.-trace rt))) source-form))
     (wrap-model model-fn))))

(defn eval-model
  "Evaluate a model code string and wrap into a DynamicGF.
   Combines eval-model-fn and wrap-model in one step, deriving a faithful source
   form from the code so conjugacy/compilation fire.
   Returns the DynamicGF, or nil on failure."
  [code-str]
  (try
    (let [f (eval-model-fn code-str)]
      (when (fn? f)
        (wrap-model f (code->source-form code-str))))
    (catch :default _ nil)))

;; ============================================================
;; Model scoring (Bayesian model evidence)
;; ============================================================

(defn observations->choicemap
  "Convert an observations map {:addr value ...} to a ChoiceMap. Scalar values
   are wrapped as mx/scalar; sequential values (vector/matrix observations, e.g.
   a multivariate-normal site) as mx/array."
  [observations]
  (apply cm/choicemap
         (mapcat (fn [[k v]]
                   [k (if (sequential? v) (mx/array v) (mx/scalar v))])
                 observations)))

(defn- log-sum-exp
  "Numerically stable log(sum(exp(xs)))."
  [xs]
  (let [max-x (apply max xs)]
    (if (= max-x ##-Inf)
      ##-Inf
      (+ max-x
         (js/Math.log
          (reduce + (map #(js/Math.exp (- % max-x)) xs)))))))

(defn- score-exact
  "Exact marginal log-evidence for a conjugate/eliminable model: the weight of a
   single analytical p/generate IS the marginal likelihood log p(obs) once the
   latents are eliminated. Mirrors fit's :exact branch (fit.cljs run-method)."
  [gf obs-cm]
  (let [{:keys [weight]} (p/generate gf [] obs-cm)]
    (mx/materialize! weight)
    (mx/item weight)))

(defn- score-is
  "Importance-sampling marginal estimate: log-mean-exp of N p/generate weights."
  [gf obs-cm n-particles]
  (let [weights (loop [i 0, ws []]
                  (if (>= i n-particles)
                    ws
                    (let [w (try
                              (mx/item (:weight (p/generate gf [] obs-cm)))
                              (catch :default _ ##-Inf))]
                      (when (zero? (mod (inc i) 10))
                        (mx/force-gc!))
                      (recur (inc i) (conj ws w)))))]
    (mx/force-gc!)
    (- (log-sum-exp weights) (js/Math.log (count weights)))))

(defn score-model*
  "Score a gen function against observations, returning {:log-ml :method}.

   Routes via method selection: conjugate / fully-eliminable models get EXACT
   analytical marginal evidence (:method :exact or :kalman — a single
   p/generate); everything else falls back to importance sampling (:method
   :handler-is, :smc, :hmc, ... — labeled as whatever was selected, scored by IS).
   Returns {:log-ml ##-Inf :method nil} on nil gf or any error.

   gf:           a DynamicGF (from eval-model or wrap-model)
   observations: {:addr value ...} map
   opts:
     :n-particles  number of importance samples for the IS fallback (default 50)"
  ([gf observations] (score-model* gf observations {}))
  ([gf observations opts]
   (if (nil? gf)
     {:log-ml ##-Inf :method nil}
     (let [{:keys [n-particles] :or {n-particles 50}} opts]
       (try
         (let [obs-cm (observations->choicemap observations)
               method (:method (ms/select-method gf obs-cm))
               ;; :exact / :kalman is a SINGLE analytical p/generate weight — valid
               ;; only when latents were actually eliminated. An opaque-fallback
               ;; model (genmlx-sndo) has an EMPTY trace-sites schema, which
               ;; method-selection labels :exact (count-trace-sites==0); scoring
               ;; it by one joint draw is NOT the marginal log p(obs). Require real
               ;; sites; otherwise score by IS, labeled honestly as :handler-is.
               eliminable? (pos? (count (:trace-sites (:schema gf))))]
           (if (and (#{:exact :kalman} method) eliminable?)
             {:log-ml (score-exact gf obs-cm) :method method}
             {:log-ml (score-is gf obs-cm n-particles)
              :method (if (#{:exact :kalman} method) :handler-is method)}))
         (catch :default _ {:log-ml ##-Inf :method nil}))))))

(defn score-model
  "Score a gen function against observations, returning a log-ML number.

   Conjugate / fully-eliminable models are scored by EXACT analytical marginal
   evidence; everything else falls back to importance sampling (log-mean-exp of N
   weights). Returns ##-Inf on nil gf or any error. See score-model* for the
   variant that also reports which method was used.

   gf:           a DynamicGF (from eval-model or wrap-model)
   observations: {:addr value ...} map
   opts:
     :n-particles  number of importance samples for the IS fallback (default 50)"
  ([gf observations] (score-model gf observations {}))
  ([gf observations opts] (:log-ml (score-model* gf observations opts))))
