(ns genmlx.llm.msa
  "Model Synthesis Architecture — synthesize probabilistic programs from
   task descriptions using an LLM, then run inference.

   Given a natural language task description + observations, generates N
   candidate gen functions via either:
     - Template mode (fine-tuned model + regex parsing), or
     - Knowledge mode (base model + Instaparse grammar parsing)
   Scores each against observations via the GFI, and runs importance
   sampling on the best model to compute a posterior.

   Sections:
     9.1  SCI evaluation context  (msa-sci-opts, eval-model-fn, wrap-model, eval-model)
     9.2  Template prompt building (build-prompt)
     9.3  LLM output parsing       (parse-dist-lines)
     9.3b Grammar-based parsing    (math-spec-grammar, normalize-llm, parse-math)
     9.4  Gen function assembly    (assemble-gen-fn)
     9.5  Generation with no-think (generate-candidate)
     9.5b Knowledge generation     (generate-knowledge-candidate)
     9.6  Model scoring            (score-model)
     9.7  Synthesize and rank      (synthesize-and-rank)
     9.8  Importance sampling      (importance-sample, infer-answer)
     9.9  End-to-end pipeline      (msa)"
  (:require [sci.core :as sci]
            [clojure.string :as str]
            [cljs.reader :as reader]
            [instaparse.core :as insta]
            ["@mlx-node/lm" :refer [ChatSession]]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.method-selection :as ms]
            [promesa.core :as pr])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ============================================================
;; 9.1 SCI evaluation context
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
;; 9.2 Template prompt building
;; ============================================================

(def ^:private msa-system-prompt
  "You fill in distribution expressions for probabilistic model variables. Output ONLY the variable assignments, one per line. Use S-expression syntax.")

(def ^:private template-example
  "a = (dist/gaussian 5 2)\nb = (dist/gaussian a 1)")

(defn build-prompt
  "Build a template prompt for the LLM from a task specification.
   Asks the LLM to fill in one (dist/...) expression per variable.

   task: {:description string, :variables [keyword ...]}"
  [{:keys [variables description]}]
  (str "Model: " description "\n\n"
       "For each variable, write the ClojureScript distribution call.\n"
       "Use S-expression syntax: (dist/gaussian mean std), (dist/bernoulli p), etc.\n"
       "You can reference earlier variables by name.\n"
       "For arithmetic on variables use: (mx/divide a b), (mx/multiply a b), (mx/add a b).\n\n"
       "Example format:\n"
       template-example "\n\n"
       "Now fill in:\n"
       (str/join "\n" (map #(str (name %) " = ???") variables))))

;; ============================================================
;; 9.3 LLM output parsing
;; ============================================================

(defn parse-dist-lines
  "Parse LLM output into a map of {variable-keyword -> dist-expression-string}.
   Looks for lines matching 'varname = (dist/...)' or 'varname : (dist/...)'.
   Variables not found in the output are omitted from the result.

   text:      raw LLM output string
   variables: ordered vector of variable keywords, e.g. [:x :y]"
  [text variables]
  (let [lines (str/split-lines (str/trim text))]
    (reduce
     (fn [acc var-kw]
       (let [var-name (name var-kw)]
         (if-let [line (some #(when (str/starts-with? (str/trim %) var-name) %)
                             lines)]
           (let [expr (-> line
                          (str/replace (re-pattern (str "^\\s*" var-name "\\s*[=:]\\s*")) "")
                          str/trim
                          (str/replace #"[,;]\s*$" ""))]
             (assoc acc var-kw expr))
           acc)))
     {}
     variables)))

;; ============================================================
;; 9.3b Grammar-based parsing (Instaparse)
;; ============================================================

(def math-spec-grammar
  "Instaparse grammar for math-notation model specs.
   Parses lines like: name ~ gaussian(param, param)
   Supports +, -, *, / arithmetic with variable references."
  (insta/parser
   "specs = (line <nl>)* line
     <nl> = #'\\n'
     line = <bullet?> name <ws> <sep> <ws> dist-expr
     <bullet> = #'[-*•]\\s*'
     <sep> = '~' | '='
     <ws> = #'\\s*'
     dist-expr = dist-name <'('> args <')'>
     dist-name = 'gaussian' | 'uniform' | 'bernoulli' | 'exponential'
     args = arg (<','> <ws> arg)*
     <arg> = number | ref | binop
     binop = <'('> arg <ws> op <ws> arg <')'> | arg <ws> op <ws> arg
     op = '+' | '-' | '*' | '/'
     ref = #'[a-zA-Z][a-zA-Z0-9_-]*'
     name = #'[a-zA-Z][a-zA-Z0-9_-]*'
     number = #'-?[0-9]+(\\.[0-9]+)?'"))

(def ^:private math->sexpr
  "Instaparse transform map: math notation parse tree -> S-expression strings."
  {:number identity
   :ref identity
   :name identity
   :dist-name identity
   :op identity
   :binop (fn [a op b]
            (let [mx-op (case op "+" "mx/add" "-" "mx/subtract"
                              "*" "mx/multiply" "/" "mx/divide")]
              (str "(" mx-op " " a " " b ")")))
   :args (fn [& args] (str/join " " args))
   :dist-expr (fn [dname args] (str "(dist/" dname " " args ")"))
   :line (fn [n expr] [(keyword (str/lower-case n)) expr])
   :specs (fn [& lines] (into {} lines))})

(defn normalize-llm
  "Normalize LLM math output for parsing.
   Lowercases distribution names, strips keyword=value prefixes,
   removes empty lines and trailing periods."
  [text]
  (-> (str/trim text)
      (str/replace #"(?i)(Gaussian|Uniform|Bernoulli|Exponential)"
                   #(str/lower-case (first %)))
      (str/replace #"(?:mean|std|loc|scale|p|prob|low|high)\s*=\s*" "")
      (str/replace #"\n\s*\n" "\n")
      (str/replace #"\.\s*$" "")))

(defn parse-math
  "Parse math-notation model specs via Instaparse.
   Input: 'x ~ gaussian(0, 10)\\ny ~ gaussian(x, 1)'
   Output: {:x '(dist/gaussian 0 10)' :y '(dist/gaussian x 1)'}
   Returns nil on parse failure."
  [text]
  (let [tree (insta/parse math-spec-grammar (normalize-llm text))]
    (when-not (insta/failure? tree)
      (insta/transform math->sexpr tree))))

;; ============================================================
;; 9.4 Gen function assembly
;; ============================================================

(defn assemble-gen-fn
  "Build a (fn [trace] ...) code string from variables and parsed dist expressions.
   Each variable becomes a let-binding: varname (trace :varname dist-expr).
   The return value is a map of all variables.

   variables: [:x :y ...]
   dist-map:  {:x \"(dist/gaussian 0 10)\" :y \"(dist/gaussian x 1)\"}

   Variables missing from dist-map default to (dist/gaussian 0 1)."
  [variables dist-map]
  (let [bindings (->> variables
                      (map (fn [v]
                             (let [expr (get dist-map v "(dist/gaussian 0 1)")]
                               (str (name v) " (trace :" (name v) " " expr ")"))))
                      (str/join "\n        "))
        ret-map (->> variables
                     (map (fn [v] (str ":" (name v) " " (name v))))
                     (str/join " "))]
    (str "(fn [trace]\n"
         "  (let [" bindings "]\n"
         "    {" ret-map "}))")))

;; ============================================================
;; 9.5 Generation with no-think
;; ============================================================

(defn- normalize-defn->fn
  "Normalize (defn name [...] ...) and (defn- name [...] ...) to (fn [...] ...).
   The fine-tuned model sometimes produces defn forms."
  [text]
  (-> (str/trim text)
      (str/replace #"\(defn-?\s+\S+\s+\[" "(fn [")
      (str/replace #"\(defn-?\s+\[" "(fn [")))

(defn generate-candidate
  "Generate a single model candidate from the LLM.
   Uses .chat with enableThinking=false for direct output.

   model-map: {:model :tokenizer :type} from llm/load-model
   task:      {:description :variables ...}
   opts:      :max-tokens (default 150), :temperature (default 0.5)

   Returns a promise of {:code :dist-map :variables}."
  ([model-map task] (generate-candidate model-map task {}))
  ([model-map task opts]
   (let [{:keys [max-tokens temperature]
          :or {max-tokens 150 temperature 0.5}} opts
         {:keys [variables]} task
         prompt (build-prompt task)]
     (pr/let [sess (ChatSession. (:model model-map)
                                (clj->js {:system msa-system-prompt
                                          :maxNewTokens max-tokens
                                          :temperature temperature}))
              result (.send sess prompt)
              text (normalize-defn->fn (.-text result))
              dist-map (parse-dist-lines text variables)
              code (assemble-gen-fn variables dist-map)]
       {:code code
        :dist-map dist-map
        :variables variables}))))

;; ============================================================
;; 9.5b Knowledge-based generation (base model + Instaparse)
;; ============================================================

(def ^:private knowledge-system-prompt
  "Write a probabilistic model. For each variable write:
name ~ distribution(params)

IMPORTANT: when a variable depends on another, use that variable name as a parameter.

Example - y depends on x:
x ~ gaussian(0, 10)
y ~ gaussian(x, 1)

Output ONLY the lines. No explanation.")

(defn- build-knowledge-prompt
  "Build a prompt for the base model to output math-notation model specs."
  [{:keys [description]}]
  description)

(defn generate-knowledge-candidate
  "Generate a model candidate using the base model + Instaparse grammar.
   The base model outputs math notation, Instaparse parses it into
   a dist-map, and assemble-gen-fn builds the code.

   model-map: {:model :tokenizer :type} — base (non-fine-tuned) model
   task:      {:description :variables :observations ...}
   opts:      :max-tokens (default 120), :temperature (default 0.5)

   Returns a promise of {:code :dist-map :variables}."
  ([model-map task] (generate-knowledge-candidate model-map task {}))
  ([model-map task opts]
   (let [{:keys [max-tokens temperature]
          :or {max-tokens 120 temperature 0.5}} opts
         {:keys [variables]} task]
     (pr/let [sess (ChatSession. (:model model-map)
                                (clj->js {:system knowledge-system-prompt
                                          :maxNewTokens max-tokens
                                          :temperature temperature}))
              result (.send sess (build-knowledge-prompt task))
              text (.-text result)
              dist-map (or (parse-math text) {})]
       {:code (assemble-gen-fn variables dist-map)
        :dist-map dist-map
        :variables variables}))))

;; ============================================================
;; 9.6 Model scoring
;; ============================================================

(defn- observations->choicemap
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
               method (:method (ms/select-method gf obs-cm))]
           (if (#{:exact :kalman} method)
             {:log-ml (score-exact gf obs-cm) :method method}
             {:log-ml (score-is gf obs-cm n-particles) :method method}))
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

;; ============================================================
;; 9.7 Synthesize and rank
;; ============================================================

(defn synthesize-and-rank
  "Generate N candidate models, eval each, score against observations,
   and return sorted by weight (best first).

   model-map: {:model :tokenizer :type} from llm/load-model
   task:      {:description :variables :observations ...}
   opts:      :n (default 10), :temperature (default 0.5), :max-tokens (default 150)
              :mode — :template (default, fine-tuned model + regex parsing)
                      :knowledge (base model + Instaparse grammar)

   Returns a promise of a vector of {:code :gf :weight :score-method :dist-map},
   sorted by weight descending. :weight is the log-ML (exact marginal for
   conjugate models, IS estimate otherwise); :score-method labels which path
   produced it. Failed candidates are included with :gf nil, :weight ##-Inf,
   :score-method nil."
  ([model-map task] (synthesize-and-rank model-map task {}))
  ([model-map task opts]
   (let [{:keys [n mode] :or {n 10 mode :template}} opts
         {:keys [observations]} task
         gen-fn (if (= mode :knowledge)
                  generate-knowledge-candidate
                  generate-candidate)]
     (pr/loop [i 0, results []]
       (if (>= i n)
         (->> results
              (sort-by :weight >)
              vec)
         (pr/let [{:keys [code dist-map]} (gen-fn model-map task opts)
                  gf (eval-model code)
                  {:keys [log-ml method]} (if gf
                                            (score-model* gf observations)
                                            {:log-ml ##-Inf :method nil})]
           (pr/recur (inc i)
                     (conj results
                           {:code code
                            :gf gf
                            :weight log-ml
                            :score-method method
                            :dist-map dist-map}))))))))

;; ============================================================
;; 9.8 Importance sampling posterior
;; ============================================================

(defn importance-sample
  "Run N particles of importance sampling on a gen function.
   Each particle calls p/generate with the observations, producing
   a weighted sample from the posterior.

   gf:           a DynamicGF
   observations: {:addr value ...}
   query:        keyword — which trace site to extract
   n:            number of particles

   Returns {:values [numbers] :log-weights [numbers] :query keyword}."
  [gf observations query n]
  (let [obs-cm (observations->choicemap observations)
        {:keys [values log-weights]}
        (reduce (fn [acc _]
                  (let [{:keys [trace weight]} (p/generate gf [] obs-cm)
                        ;; A synthesized candidate may not trace `query` at all
                        ;; (or trace it as a non-leaf). Extracting then yields nil
                        ;; and mx/item throws. Record NaN instead — infer-answer
                        ;; drops non-finite values, so one bad site can't NaN the
                        ;; whole posterior.
                        v (try
                            (mx/item (cm/get-value (cm/get-submap (:choices trace) query)))
                            (catch :default _ js/NaN))]
                    (-> acc
                        (update :values conj v)
                        (update :log-weights conj (mx/item weight)))))
                {:values [] :log-weights []}
                (range n))]
    {:values values :log-weights log-weights :query query}))

(defn- exp-normalize
  "Exponentiate log-weights after subtracting their max, for numerically
   stable normalization. Returns the vector of unnormalized weights."
  [log-weights]
  (let [max-w (apply max log-weights)]
    (mapv #(js/Math.exp (- % max-w)) log-weights)))

(defn infer-answer
  "Compute posterior mean and variance from importance sampling results.
   Normalizes log-weights via log-sum-exp for numerical stability.

   Robust to degenerate inputs: a particle is only usable when BOTH its value
   and its log-weight are finite. A non-finite value (e.g. a candidate that
   divides by a near-zero sample) or a non-finite weight (an impossible
   observation gives -Inf) would otherwise poison the reduction and produce a
   NaN mean. Such particles are dropped before normalization. When no usable
   particle remains, the posterior is degenerate — fall back to the unweighted
   mean of any finite values, else 0.0. The result is always finite.

   samples: {:values [...] :log-weights [...] :query keyword}
            as returned by importance-sample

   Returns {:mean number, :variance number, :ess number, :query keyword}."
  [{:keys [values log-weights query]}]
  (let [pairs (filterv (fn [[v w]] (and (js/isFinite v) (js/isFinite w)))
                       (mapv vector values log-weights))]
    (if (seq pairs)
      ;; ≥1 usable particle: standard self-normalized importance estimate.
      ;; exp-normalize subtracts the max, so the largest weight is exactly 1 and
      ;; total ≥ 1 — no zero-division, and every term is finite.
      (let [vs (mapv first pairs)
            unnorm (exp-normalize (mapv second pairs))
            total (reduce + unnorm)
            probs (mapv #(/ % total) unnorm)
            mean (reduce + (map * vs probs))
            variance (reduce + (map (fn [v p] (* p (* (- v mean) (- v mean)))) vs probs))
            ess (/ (* total total) (reduce + (map * unnorm unnorm)))]
        {:mean mean :variance variance :ess ess :query query})
      ;; No usable particle — degenerate posterior. Unweighted mean of finite
      ;; values if any survive, else 0.0. Never NaN.
      (let [vs (filterv js/isFinite values)
            n (count vs)]
        (if (pos? n)
          (let [mean (/ (reduce + vs) n)
                variance (/ (reduce + (map #(* (- % mean) (- % mean)) vs)) n)]
            {:mean mean :variance variance :ess (double n) :query query})
          {:mean 0.0 :variance 0.0 :ess 0.0 :query query})))))

;; ============================================================
;; 9.9 End-to-end pipeline
;; ============================================================

(defn msa
  "End-to-end Model Synthesis Architecture pipeline.

   1. Generates N candidate probabilistic programs from the task description
   2. Evaluates and scores each against observations
   3. Picks the best-scoring model
   4. Runs importance sampling to compute the posterior over the query variable

   model-map: {:model :tokenizer :type} from llm/load-model
   task:      {:name        string
               :description string
               :variables   [keyword ...]
               :observations {:addr value ...}
               :query       keyword}
   opts:      :n           candidates to generate (default 10)
              :particles   importance sampling particles (default 200)
              :temperature LLM temperature (default 0.5)
              :max-tokens  LLM max tokens (default 150)
              :mode        :template (fine-tuned model) or :knowledge (base model + Instaparse)

   Returns a promise of
     {:model     {:code :gf :weight :dist-map}
      :posterior {:mean :variance :ess :query}
      :candidates [all scored candidates, best first]}"
  ([model-map task] (msa model-map task {}))
  ([model-map task opts]
   (let [{:keys [particles] :or {particles 200}} opts
         {:keys [observations query]} task]
     (pr/let [candidates (synthesize-and-rank model-map task opts)
              best (first candidates)]
       (if-not (:gf best)
         {:model best
          :posterior nil
          :candidates candidates}
         (let [samples (importance-sample (:gf best) observations query particles)
               posterior (infer-answer samples)]
           {:model best
            :posterior posterior
            :candidates candidates}))))))
