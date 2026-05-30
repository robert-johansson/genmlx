;; ============================================================================
;; DISTINCTIVE DEMO 05 — Program synthesis as inference
;;                       ("ClojureScript writing ClojureScript")
;; ============================================================================
;;
;; THE DISTINCTIVE FEATURE
;; -----------------------
;; A 0.6B LLM PROPOSES probabilistic-program source. That source is EVALUATED
;; in-process via SCI into a real generative function. That gen function is
;; SCORED against data by the GFI itself — the p/generate importance weight,
;; which for a fully-constrained model is an estimate of the log marginal
;; likelihood of the observations.
;;
;; Three stages, ZERO representational boundary crossed:
;;
;;     PROPOSE  (LLM-as-GF)   ──►  ClojureScript source string
;;        │
;;     EVALUATE (SCI)         ──►  a DynamicGF, in this same process
;;        │
;;     SCORE    (p/generate)  ──►  log marginal likelihood (a number)
;;
;; Proposal, evaluation, and scoring all live in one runtime, one
;; representation. The same GFI that scores a hand-written model scores a
;; machine-written one. Cognition = program synthesis under coherence.
;;
;; THE TASK
;; --------
;; Three noisy observations of an unknown mean (true μ ≈ 3.0):
;;     y1 = 3.2, y2 = 2.9, y3 = 3.1
;; The LLM must propose a generative model: a prior over μ and a likelihood
;; for each y_i. We synthesise a few candidates, score each by coherence,
;; pick the best, then read off the posterior over μ.
;;
;; NOTE: the 0.6B base model is weak. The DEMO is that the loop runs end to
;; end — propose → eval → score — not that the tiny model writes a great
;; model. We narrate whatever candidates/scores result.
;;
;; Run:
;;   bun run --bun nbb examples/distinctive/05_program_synthesis.cljs
;; ============================================================================

(ns distinctive.05-program-synthesis
  (:require [genmlx.mlx :as mx]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.msa :as msa]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [promesa.core :as pr]))

;; ----------------------------------------------------------------------------
;; Configuration — small model, small counts, fast
;; ----------------------------------------------------------------------------

;; A capable base model writes far better probabilistic programs than a 0.6B.
;; qwen3.5-4b is a good quality/speed balance; bump to Qwen3.6-35B-A3B-4bit for
;; the strongest synthesis (slower to load).
(def MODEL-NAME "qwen3.5-4b-mlx-bf16")
(def MODEL-DIR (str (.-HOME js/process.env) "/.cache/models/" MODEL-NAME))

(def N-CANDIDATES 3)
(def MAX-TOKENS   120)
(def TEMPERATURE  0.6)

;; System prompt for the PROPOSE stage. We drive generation with
;; llm/generate-text-raw, which injects think-skip tokens and hard-caps the
;; token count — fast and reliable for thinking-mode models (the ChatSession
;; path can ramble for minutes on a thinking model).
(def synth-system-prompt
  (str "Write a probabilistic model. For each variable write one line:\n"
       "name ~ distribution(params)\n\n"
       "When a variable depends on another, use that variable's name as a parameter.\n"
       "Use 'gaussian' for normal distributions.\n\n"
       "Example:\nx ~ gaussian(0, 10)\ny ~ gaussian(x, 1)\n\n"
       "Output ONLY the lines, nothing else."))

(def MU-TRUE 3.0)

;; The inference task. :knowledge mode reads :description and asks the base
;; model to write "name ~ distribution(params)" lines, parsed by Instaparse.
(def task
  {:description (str "We observe three measurements: y1=3.2, y2=2.9, y3=3.1. "
                     "These are noisy observations of an unknown underlying "
                     "mean mu. mu has a wide prior. Each observation y_i is "
                     "drawn from a gaussian centred on mu with some noise. "
                     "Write the probabilistic model.")
   :variables    [:mu :y1 :y2 :y3]
   :observations {:y1 3.2 :y2 2.9 :y3 3.1}
   :query        :mu})

;; ----------------------------------------------------------------------------
;; Formatting helpers
;; ----------------------------------------------------------------------------

(defn fmt
  ([x] (fmt x 3))
  ([x d]
   (cond
     (= x ##-Inf) "-inf"
     (= x ##Inf)  "+inf"
     (and (number? x) (js/Number.isNaN x)) "NaN"
     (number? x)  (.toFixed x d)
     :else        (str x))))

(defn rule []
  (println (apply str (repeat 76 "="))))

;; ----------------------------------------------------------------------------
;; A reference: hand-written model scored by the SAME GFI, so the LLM's
;; candidates are graded on the same yardstick the framework uses everywhere.
;; ----------------------------------------------------------------------------

(def reference-code
  "(fn [trace]
     (let [mu (trace :mu (dist/gaussian 0 10))
           y1 (trace :y1 (dist/gaussian mu 1))
           y2 (trace :y2 (dist/gaussian mu 1))
           y3 (trace :y3 (dist/gaussian mu 1))]
       {:mu mu :y1 y1 :y2 y2 :y3 y3}))")

;; ----------------------------------------------------------------------------
;; Main pipeline
;; ----------------------------------------------------------------------------

(println (str "Loading model: " MODEL-NAME " ..."))

(pr/let
 [t-load0 (js/performance.now)
  model-map (llm/load-model MODEL-DIR)
  t-load1 (js/performance.now)
  _ (println (str "Model loaded in " (fmt (- t-load1 t-load0) 0) " ms.\n"))

  ;; ----------------------------------------------------------------------
  _ (rule)
  _ (println "  STAGE 0 — The task, and the GFI as the universal yardstick")
  _ (rule)
  _ (println "  Observations (noisy draws of an unknown mean, true mu ~ 3.0):")
  _ (doseq [[k v] (:observations task)]
      (println (str "    " (name k) " = " v)))
  _ (println)
  _ (println "  Before the LLM proposes anything, show that the GFI scores")
  _ (println "  a HAND-WRITTEN model the same way it will score a")
  _ (println "  MACHINE-WRITTEN one. This is the reference model:")
  _ (println)
  _ (doseq [line (clojure.string/split-lines reference-code)]
      (println (str "    " (clojure.string/triml line))))
  ref-gf (msa/eval-model reference-code)
  ref-w  (msa/score-model ref-gf (:observations task) {:n-particles 50})
  _ (println)
  _ (println (str "  GFI score of reference model (log marginal likelihood)"
                  " = " (fmt ref-w)))
  _ (println "  (eval-model -> SCI -> DynamicGF; score-model -> p/generate.)")

  ;; ----------------------------------------------------------------------
  _ (println)
  _ (rule)
  _ (println "  STAGE 1+2+3 — PROPOSE (LLM) -> EVALUATE (SCI) -> SCORE (GFI)")
  _ (rule)
  _ (println (str "  Asking " MODEL-NAME " for " N-CANDIDATES
                  " candidate models (generate-text-raw, think-skipped)."))
  _ (println "  Each candidate's text is parsed by Instaparse, assembled into")
  _ (println "  ClojureScript source, eval'd by SCI into a DynamicGF, and")
  _ (println "  scored by p/generate against the observations. One process,")
  _ (println "  one representation, one yardstick.")
  _ (println)

  t-syn0 (js/performance.now)
  ;; PROPOSE (generate-text-raw) -> EVALUATE (SCI: parse-math -> assemble -> eval-model)
  ;; -> SCORE (GFI: score-model = p/generate log-ML). Ranked by coherence (log-ML).
  raw-candidates
  (pr/loop [i 0 acc []]
    (if (>= i N-CANDIDATES)
      acc
      (pr/let [txt      (llm/generate-text-raw model-map (:description task)
                                               {:max-tokens MAX-TOKENS
                                                :temperature TEMPERATURE
                                                :system-prompt synth-system-prompt})
               dist-map (msa/parse-math txt)
               code     (msa/assemble-gen-fn (:variables task) (or dist-map {}))
               gf       (msa/eval-model code)
               w        (if gf (msa/score-model gf (:observations task) {:n-particles 50})
                                ##-Inf)]
        (pr/recur (inc i) (conj acc {:code code :dist-map dist-map :gf gf :weight w})))))
  candidates (vec (sort-by :weight > raw-candidates))
  t-syn1 (js/performance.now)
  _ (println (str "  Synthesised + scored " (count candidates)
                  " candidates in " (fmt (- t-syn1 t-syn0) 0) " ms.\n"))

  _ (doseq [[i c] (map-indexed vector candidates)]
      (let [scored? (and (:gf c) (not= (:weight c) ##-Inf))]
        (println (str "  -- candidate [" i "]  log-ML = "
                      (fmt (:weight c)) (when-not scored? "  (no parse / score)")))
        (println "     SYNTHESIZED ClojureScript source (LLM -> SCI):")
        (doseq [line (clojure.string/split-lines (:code c))]
          (println (str "       " line)))
        (when (seq (:dist-map c))
          (println (str "     parsed dist-map: " (pr-str (:dist-map c)))))
        (println)))

  best (first candidates)
  _ (rule)
  _ (println "  SELECTION — best candidate by coherence (highest log-ML)")
  _ (rule)
  _ (if (and best (:gf best) (not= (:weight best) ##-Inf))
      (do
        (println (str "  best log-ML = " (fmt (:weight best))
                      "   (reference = " (fmt ref-w) ")"))
        (println "  best candidate source:")
        (doseq [line (clojure.string/split-lines (:code best))]
          (println (str "    " line))))
      (println "  No candidate parsed+scored; the 0.6B model wrote weak output."))

  ;; ----------------------------------------------------------------------
  ;; STAGE 4 — Posterior on the selected hypothesis (same GFI op, keep samples)
  ;; ----------------------------------------------------------------------
  _ (println)
  _ (rule)
  _ (println "  POSTERIOR — importance sampling on the selected gen function")
  _ (rule)
  _ (if (and best (:gf best) (not= (:weight best) ##-Inf))
      (let [samples (msa/importance-sample (:gf best) (:observations task)
                                           (:query task) 150)
            post    (msa/infer-answer samples)]
        (println (str "  Posterior over mu (150 particles, machine-written model):"))
        (println (str "    mean   = " (fmt (:mean post))))
        (println (str "    stddev = " (fmt (js/Math.sqrt (:variance post)))))
        (println (str "    ESS    = " (fmt (:ess post) 1)))
        (println (str "  true mu = " MU-TRUE
                      "  ->  |posterior mean - truth| = "
                      (fmt (js/Math.abs (- (:mean post) MU-TRUE))))))
      ;; Fall back to the reference model so the posterior stage always
      ;; demonstrates end-to-end: a scored gen function -> posterior.
      (let [samples (msa/importance-sample ref-gf (:observations task)
                                           (:query task) 150)
            post    (msa/infer-answer samples)]
        (println "  (0.6B candidates too weak to score; showing the SAME")
        (println "   posterior step on the reference gen function, to prove")
        (println "   the loop's scoring->posterior tail runs end to end.)")
        (println (str "  Posterior over mu (150 particles, reference model):"))
        (println (str "    mean   = " (fmt (:mean post))))
        (println (str "    stddev = " (fmt (js/Math.sqrt (:variance post)))))
        (println (str "    ESS    = " (fmt (:ess post) 1)))
        (println (str "  true mu = " MU-TRUE
                      "  ->  |posterior mean - truth| = "
                      (fmt (js/Math.abs (- (:mean post) MU-TRUE)))))))

  _ (println)
  _ (rule)
  _ (println "  WHAT THIS DEMONSTRATES")
  _ (rule)
  _ (println "  The synthesis loop crossed NO representational boundary:")
  _ (println "    PROPOSE  : the LLM (itself a generative function) emitted")
  _ (println "               ClojureScript source describing a model.")
  _ (println "    EVALUATE : SCI turned that source into a DynamicGF in this")
  _ (println "               very process — no codegen step, no subprocess.")
  _ (println "    SCORE    : p/generate computed the log marginal likelihood,")
  _ (println "               the SAME operation that scores a hand-written")
  _ (println "               model. Coherence is one number, one yardstick.")
  _ (println "  Proposal, evaluation, and scoring share one runtime and one")
  _ (println "  representation. ClojureScript wrote ClojureScript, and the GFI")
  _ (println "  judged it on its own terms.")]

  (println "\nDone."))
