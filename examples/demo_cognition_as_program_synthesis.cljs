;; ============================================================================
;; DEMO 3 — Cognition as program synthesis under coherence
;; ============================================================================
;;
;; Companion code for §6 of the Machine-Psychology-with-GenMLX paper, in
;; response to Griffiths et al. 2026 ("Whither symbols in the era of advanced
;; neural networks?", Trends in Cognitive Sciences).
;;
;; THE STRUCTURAL CLAIM
;; --------------------
;; Cognition, in the unified framework, is **program synthesis under
;; coherence**.  An organism (or any canonical-form machine) is presented
;; with experience.  It proposes hypotheses about the generative process
;; that produced that experience.  It scores each hypothesis by how well
;; it explains the experience.  It selects the highest-scoring hypothesis.
;; That selection is itself a canonical-form object available for further
;; cognition.  The recursive structure — gen functions generating gen
;; functions, scored by gen-function-evaluation — is the AARR closure.
;;
;; This unifies several characterisations from different traditions:
;;   - Operant psychology / RFT: Arbitrarily Applicable Relational
;;     Responding (AARR) — the highest-order operant capacity.
;;   - Wang's NARS: intelligence as adaptation under AIKR via inference
;;     rule application + term hierarchy.
;;   - Probabilistic programming: program synthesis as Bayesian inference
;;     over generative models.
;;
;; In our framework, all three characterisations name the SAME loop, made
;; operational in the canonical form.  This demo runs that loop in
;; working code.
;;
;; THE DEMO
;; --------
;; A small inference task: 3 observations are drawn from a hidden
;; generative process (y_i ~ Gaussian(μ_true, 1) with μ_true = 3.0).
;; Qwen3.6-35B-A3B-4bit is asked to PROPOSE generative models.  Each
;; proposal is parsed (via Instaparse grammar), assembled into a gen
;; function, evaluated (via SCI), and scored against the observations
;; (via importance sampling under p/generate).  The best candidate is
;; selected by log-marginal-likelihood.  Importance sampling on the best
;; candidate produces the posterior over μ.
;;
;; Because Qwen3 / Qwen3.5 / Qwen3.6 use a "thinking" mode by default
;; that produces internal-reasoning tokens stripped from visible output,
;; we use `generate-text-raw` (which forces empty think tags so the
;; model goes straight to the answer).  We also alias `normal` →
;; `gaussian` because the model occasionally uses statistics terminology
;; instead of the project's distribution name.
;;
;; WHY THIS ANSWERS GRIFFITHS' FINAL QUESTION
;; ------------------------------------------
;; Griffiths et al. close their paper with: "to what extent does
;; behavioural alignment necessitate mechanistic alignment?"  In our
;; framework, when cognition is program synthesis under coherence, the
;; question dissolves: the cognitive theory IS the canonical-form code
;; IS the artifact selected by coherence.  Behavioural alignment IS
;; mechanistic alignment, because there is no separate "mental model"
;; inferred from behaviour — there is only the gen function that
;; scored highest.
;;
;; Run:
;;   bun run --bun nbb examples/demo_cognition_as_program_synthesis.cljs
;;
;; (Requires the Qwen3.6-35B-A3B-4bit model at ~/.cache/models/.)
;; ============================================================================

(ns demo-cognition-as-program-synthesis
  (:require [genmlx.mlx :as mx]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.msa :as msa]
            [clojure.string :as str]
            [promesa.core :as pr]))

;; ----------------------------------------------------------------------------
;; Configuration
;; ----------------------------------------------------------------------------

(def MODEL-NAME "Qwen3.6-35B-A3B-4bit")
(def MODEL-DIR  (str (.-HOME js/process.env) "/.cache/models/" MODEL-NAME))

(def N-CANDIDATES 5)
(def MAX-TOKENS 200)
(def BASE-TEMPERATURE 0.7)
(def SCORING-PARTICLES 100)
(def POSTERIOR-PARTICLES 500)

;; ----------------------------------------------------------------------------
;; The hidden generative process
;; ----------------------------------------------------------------------------

(def MU-TRUE 3.0)
(def SIGMA-TRUE 1.0)

(def OBSERVATIONS
  {:y1 3.2
   :y2 2.9
   :y3 3.1})

(def task
  {:variables    [:mu :y1 :y2 :y3]
   :observations OBSERVATIONS
   :query        :mu})

;; ----------------------------------------------------------------------------
;; LLM prompting — knowledge mode + think-skip
;; ----------------------------------------------------------------------------

(def synthesis-system-prompt
  (str "Write a probabilistic model. For each variable write:\n"
       "name ~ distribution(params)\n\n"
       "IMPORTANT: when a variable depends on another, use that variable name as a parameter.\n"
       "Use 'gaussian' (NOT 'normal') for normal distributions.\n\n"
       "Example - y depends on x:\n"
       "x ~ gaussian(0, 10)\n"
       "y ~ gaussian(x, 1)\n\n"
       "Output ONLY the lines. No explanation."))

(def task-description
  (str "We observe three measurements: y1=3.2, y2=2.9, y3=3.1. "
       "These are noisy observations of an unknown underlying mean "
       "μ. The variable μ has a wide prior. Each observation y_i is "
       "drawn from a distribution centred on μ with some noise. "
       "Write the probabilistic model."))

(defn normalise-llm-output
  "Pre-process LLM output: alias `normal` → `gaussian` so the
   Instaparse grammar accepts it."
  [text]
  (-> text
      (str/replace #"\bnormal\b" "gaussian")
      (str/replace #"\bnorm\b"   "gaussian")))

(defn generate-candidate
  "Generate one candidate model via the LLM.  Returns a promise of
   {:raw text, :dist-map parsed-map, :code assembled-code, :gf nil-or-DynamicGF}."
  [model-map seed]
  (pr/let [raw (llm/generate-text-raw
                 model-map task-description
                 {:system-prompt synthesis-system-prompt
                  :max-tokens MAX-TOKENS
                  :temperature BASE-TEMPERATURE
                  :seed seed})
           text     (normalise-llm-output raw)
           dist-map (or (msa/parse-math text) {})
           code     (msa/assemble-gen-fn (:variables task) dist-map)
           gf       (msa/eval-model code)]
    {:raw raw
     :text text
     :dist-map dist-map
     :code code
     :gf gf
     :proper-parse? (seq dist-map)}))

(defn score-candidate
  "Score one candidate against the observations via importance sampling.
   Returns the candidate map with :weight added (##-Inf if scoring fails)."
  [candidate]
  (let [w (if (and (:gf candidate) (:proper-parse? candidate))
            (msa/score-model (:gf candidate)
                             (:observations task)
                             {:n-particles SCORING-PARTICLES})
            ##-Inf)]
    (assoc candidate :weight w)))

;; ----------------------------------------------------------------------------
;; Helpers
;; ----------------------------------------------------------------------------

(defn fmt
  ([x] (fmt x 3))
  ([x digits]
   (cond
     (= x ##-Inf) "-∞"
     (= x ##Inf)  "+∞"
     (number? x)  (.toFixed x digits)
     :else        (str x))))

(defn print-section [title]
  (println)
  (println (str "════════════════════════════════════════"
                "════════════════════════════════════════"))
  (println (str "  " title))
  (println (str "════════════════════════════════════════"
                "════════════════════════════════════════")))

(defn print-subsection [title]
  (println)
  (println (str "── " title " ──")))

;; ----------------------------------------------------------------------------
;; Phase 1 — Show the task and ground truth
;; ----------------------------------------------------------------------------

(defn phase-1-task []
  (print-section "PHASE 1 — The cognitive task")
  (println "  Hidden generative process (unknown to the system):")
  (println (str "    μ_true = " MU-TRUE))
  (println (str "    σ_true = " SIGMA-TRUE))
  (println (str "    y_i ~ Gaussian(μ_true, σ_true)"))
  (println)
  (println "  Observations (these are what the system sees):")
  (doseq [[k v] OBSERVATIONS]
    (println (str "    " (name k) " = " v)))
  (println)
  (println "  The cognitive task:")
  (println "    1. PROPOSE generative models for the observations.")
  (println "    2. SCORE each by coherence (log-marginal-likelihood).")
  (println "    3. SELECT by coherence.")
  (println "    4. INFER the posterior over the latent μ via importance")
  (println "       sampling on the selected model.")
  (println)
  (println "  None of this requires special 'meta-learning' or 'symbolic'")
  (println "  machinery.  It is one application of the canonical form:")
  (println "  gen functions generating gen functions, scored by")
  (println "  gen-function-evaluation."))

;; ----------------------------------------------------------------------------
;; Phase 2 — LLM proposes candidates; pipeline scores and ranks them
;; ----------------------------------------------------------------------------

(defn phase-2-synthesis [model-map]
  (print-section "PHASE 2 — LLM proposes candidates; coherence ranks them")
  (println (str "  Asking " MODEL-NAME " to propose " N-CANDIDATES
                " candidate models."))
  (println "  Each candidate must specify a generative distribution for")
  (println "  μ and for each y_i.  Knowledge-mode prompting + Instaparse")
  (println "  grammar parsing extracts each candidate as a runnable")
  (println "  ClojureScript gen function.  We use generate-text-raw with")
  (println "  empty <think></think> tags to skip the model's reasoning")
  (println "  phase and go straight to the answer.")
  (println)
  (println "  Each candidate is scored via importance sampling under")
  (println "  p/generate, with the observations as constraints.  The weight")
  (println "  returned IS the log-marginal-likelihood of the observations")
  (println "  under that candidate.  This is COHERENCE in our framework.")
  (println)
  (println "  Generating + scoring candidates ...")
  (println)

  (pr/loop [i 0, candidates []]
    (if (>= i N-CANDIDATES)
      (let [scored (->> candidates
                        (map score-candidate)
                        (sort-by :weight >)
                        vec)]
        (print-subsection "All candidates, ranked by coherence (best first)")
        (doseq [[idx c] (map-indexed vector scored)]
          (let [tag (cond
                      (not (:proper-parse? c)) "[FAILED to parse]"
                      (= (:weight c) ##-Inf)   "[FAILED to score]"
                      :else                    (str "log-ML = " (fmt (:weight c))))]
            (println (str "  [" (inc idx) "] " tag))
            (if (:proper-parse? c)
              (doseq [[k v] (:dist-map c)]
                (println (str "        " (name k) " = " v)))
              (do
                (println "        Raw LLM output:")
                (doseq [line (str/split-lines (or (:raw c) ""))]
                  (println (str "          | " line)))))))
        scored)
      (pr/let [_ (println (str "    [" (inc i) "/" N-CANDIDATES "] generating ..."))
               c (generate-candidate model-map (+ 42 i))
               _ (println (str "          → " (if (:proper-parse? c)
                                                 "parsed OK"
                                                 "(no parseable lines)")))]
        (pr/recur (inc i) (conj candidates c))))))

;; ----------------------------------------------------------------------------
;; Phase 3 — The selection: pick best by coherence
;; ----------------------------------------------------------------------------

(defn phase-3-selection [candidates]
  (print-section "PHASE 3 — Selection by coherence")
  (let [valid (filter #(and (:proper-parse? %) (not= (:weight %) ##-Inf))
                      candidates)
        best (first valid)]
    (if-not best
      (do
        (println "  No candidate parsed and scored successfully.")
        (println "  Try re-running with different seeds or adjusting the prompt.")
        nil)
      (do
        (println "  The selected hypothesis (highest log-ML):")
        (println)
        (doseq [[k v] (:dist-map best)]
          (println (str "    " (name k) " = " v)))
        (println)
        (println (str "  Coherence: log-ML = " (fmt (:weight best))))
        (println)
        (println "  Selection by coherence is the same operation that the")
        (println "  framework uses everywhere it makes a choice between")
        (println "  competing canonical-form objects: log-marginal-likelihood")
        (println "  from p/generate.  No separate 'model selection' machinery")
        (println "  is required.")
        best))))

;; ----------------------------------------------------------------------------
;; Phase 4 — Posterior inference on the selected hypothesis
;; ----------------------------------------------------------------------------

(defn phase-4-posterior [best-candidate]
  (print-section "PHASE 4 — Posterior inference under the selected hypothesis")
  (println "  Run importance sampling on the selected gen function with")
  (println "  the observations as constraints.  This produces a posterior")
  (println "  distribution over the query variable μ.")
  (println)
  (println "  This is the SAME operation as Phase 2's scoring step, but")
  (println "  now we keep the samples (not just the marginal likelihood)")
  (println "  to produce a posterior estimate.  It is one further application")
  (println "  of the canonical form.")
  (println)
  (let [{:keys [gf]} best-candidate
        samples (msa/importance-sample gf OBSERVATIONS :mu POSTERIOR-PARTICLES)
        post    (msa/infer-answer samples)]
    (println (str "  Posterior over μ (importance sampling, "
                  POSTERIOR-PARTICLES " particles):"))
    (println (str "    mean    = " (fmt (:mean post))))
    (println (str "    stddev  = " (fmt (Math/sqrt (:variance post)))))
    (println (str "    ESS     = " (fmt (:ess post) 1)))
    (println)
    (println (str "  Ground truth: μ_true = " MU-TRUE))
    (println (str "  Posterior mean is "
                  (fmt (Math/abs (- (:mean post) MU-TRUE)))
                  " away from the truth."))))

;; ----------------------------------------------------------------------------
;; Closing reflection
;; ----------------------------------------------------------------------------

(defn closing-remarks []
  (print-section "WHAT THIS DEMONSTRATES")
  (println "
  The loop just executed:

    1. The system was given experience (3 observations).
    2. The LLM proposed multiple candidate gen functions as
       hypotheses about the generative process.
    3. Each candidate was scored by coherence (log-marginal-
       likelihood from p/generate against the observations).
    4. The candidate with highest coherence was selected.
    5. Importance sampling on the selected candidate produced
       the posterior over the latent variable.

  Each step is one application of the canonical form (the GFI):
  proposal generation is gen-function-construction (assembled
  from LLM output via SCI); coherence scoring is p/generate;
  selection is comparison of weights; posterior inference is
  importance sampling.  No new machinery is introduced at any
  step.  The whole loop is one expression of the framework's
  basic operations, applied recursively.

  THIS IS AARR, OPERATIONALISED.  The recursive structure
  (gen functions generating gen functions, selected by gen-
  function-evaluation) is the closure that operant psychology /
  RFT identify as the highest-order operant capacity, that
  Wang identifies as intelligence under AIKR.  All three
  characterisations name the same loop.  In our framework, that
  loop is implementable in working code and runnable on a
  laptop.

  THIS ANSWERS GRIFFITHS' FINAL QUESTION CONSTRUCTIVELY.  They
  ask: 'to what extent does behavioural alignment necessitate
  mechanistic alignment?'  When cognition is program synthesis
  under coherence, the question dissolves: the cognitive theory
  IS the canonical-form code (the gen function the LLM proposes
  and the system selects).  Behavioural alignment IS mechanistic
  alignment, because the model that explains the data IS the
  program that generates the data IS the artifact selected by
  coherence.  There is no separate 'mental model' inferred from
  behaviour — there is only the gen function that scored highest.

  THE FRAMEWORK SUPPORTS THIS LOOP NATIVELY.  No special
  symbolic-vs-subsymbolic vocabulary is required.  No special
  meta-learning capacity is required.  The canonical form IS the
  substrate; AIKR IS the operational discipline; the operant /
  RFT empirical literature IS the theoretical scaffold; the
  whole loop is one expression of the unified framework.

  See PAPER_MACHINE_PSYCHOLOGY.md §6 for the discussion.
"))

;; ----------------------------------------------------------------------------
;; Main async entry point
;; ----------------------------------------------------------------------------

(println "Loading model:" MODEL-NAME)
(println "  (this can take a moment for the 35B-A3B variant)")

(pr/let [model-map (llm/load-model MODEL-DIR)
         _         (println "Model loaded.\n")
         _         (phase-1-task)
         candidates (phase-2-synthesis model-map)
         best       (phase-3-selection candidates)
         _          (when best (phase-4-posterior best))
         _          (closing-remarks)]
  (println "\nDone."))
