;; ============================================================================
;; DEMO 1 — State transitions as behavior; LLMs as distributions
;; ============================================================================
;;
;; Companion code for §4 of the Machine-Psychology-with-GenMLX paper, in
;; response to Griffiths et al. 2026 ("Whither symbols in the era of advanced
;; neural networks?", Trends in Cognitive Sciences).
;;
;; THE STRUCTURAL CLAIM
;; --------------------
;; In GenMLX, every distribution sample is a state transition.  The handler
;; that processes a (Bernoulli) coin flip is the same handler that processes
;; a (Gaussian) sample, which is the same handler that processes a token
;; sampled from `dist/categorical(logits)` where the logits happen to be
;; computed by a 35-billion-parameter language model.
;;
;; A behavior, in the operant-psychology sense, is a state transition in
;; relation to an event.  Each call to a gen function is exactly that:
;;   - the state is the handler's accumulated choicemap + score + key
;;   - the event is the parameters arriving at the trace site(s)
;;   - the transformation is sampling from a distribution
;;   - the output is the new state
;;
;; None of this changes when the distribution's parameters happen to be
;; computed by a neural network forward pass.  The LLM is "just" a
;; distribution whose parameters are computed by a particular kind of
;; function.  Wrapping it in the gen macro makes it a canonical-form
;; machine in exactly the sense a coin flip is a canonical-form machine.
;;
;; This demo shows three gen functions — Bernoulli, two-Gaussian, LLM —
;; with the SAME six GFI operations applied uniformly to each.
;;
;; Run:
;;   bun run --bun nbb examples/demo_state_transitions_as_behavior.cljs
;;
;; (Requires the Qwen3.6-35B-A3B-4bit model at ~/.cache/models/.
;;  For faster iteration, swap MODEL-NAME for "qwen3-0.6b-mlx-bf16".)
;; ============================================================================

(ns demo-state-transitions-as-behavior
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as llm-core]
            [promesa.core :as pr])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ----------------------------------------------------------------------------
;; Configuration
;; ----------------------------------------------------------------------------

(def MODEL-NAME "Qwen3.6-35B-A3B-4bit")
(def MODEL-DIR  (str (.-HOME js/process.env) "/.cache/models/" MODEL-NAME))

;; Number of tokens to generate in the LLM examples.  Small for demo brevity.
(def MAX-TOKENS 6)

;; ----------------------------------------------------------------------------
;; Three gen functions, increasing in substrate complexity.
;; The structural form is identical: each is `S × P → S × output`,
;; sampling at named trace sites from distributions parameterized by inputs.
;; ----------------------------------------------------------------------------

;; (1) A coin flip.  One trace site.  Bernoulli.
(def coin-flip
  (dyn/auto-key
    (gen [bias]
      (trace :outcome (dist/bernoulli bias)))))

;; (2) A two-step Gaussian process.  Two trace sites with a dependency.
;;     The second sample's mean is the first sample's value.
(def two-step
  (dyn/auto-key
    (gen [μ σ]
      (let [latent   (trace :z (dist/gaussian μ σ))
            observed (trace :y (dist/gaussian latent (mx/scalar 0.5)))]
        observed))))

;; (3) An LLM wrapped as a gen function.  Each token is a trace site
;;     :t0, :t1, ... sampling from dist/categorical(logits) where the
;;     logits come from the LLM forward pass.  Created in the main
;;     async block once the model is loaded.

;; ----------------------------------------------------------------------------
;; Helper utilities for printing results uniformly.
;; ----------------------------------------------------------------------------

(defn fmt
  "Format a number with limited precision."
  [x]
  (cond
    (number? x) (.toFixed x 4)
    :else (str x)))

(defn ->scalar
  "Materialize an MLX scalar (or pass through a number) to a JS number."
  [x]
  (if (number? x) x (mx/item x)))

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
;; Print the outcome of one trace, uniformly across gen functions.
;; ----------------------------------------------------------------------------

(defn show-coin-trace [label trace]
  (let [val (cm/get-value (cm/get-submap (:choices trace) :outcome))]
    (println (str "  " label
                  " :outcome=" (fmt (->scalar val))
                  "  log p=" (fmt (->scalar (:score trace)))))))

(defn show-two-step-trace [label trace]
  (let [z (cm/get-value (cm/get-submap (:choices trace) :z))
        y (cm/get-value (cm/get-submap (:choices trace) :y))]
    (println (str "  " label
                  " :z=" (fmt (->scalar z))
                  "  :y=" (fmt (->scalar y))
                  "  log p=" (fmt (->scalar (:score trace)))))))

(defn show-llm-trace [tokenizer label trace]
  (pr/let [text (llm-core/decode-trace tokenizer trace)]
    (println (str "  " label
                  "  log p=" (fmt (->scalar (:score trace)))))
    (println (str "    text: " (pr-str text)))))

;; ----------------------------------------------------------------------------
;; The six GFI operations, applied uniformly to all three gen functions.
;; ----------------------------------------------------------------------------

(defn demo-simulate
  "GFI op 1/6 — simulate: forward-sample all choices.
   Same operation; same return shape (a Trace with :choices, :score, :retval)."
  [llm-gf tokenizer {:keys [prompt-ids]}]
  (print-section "1. SIMULATE — forward-sample all choices")
  (println "  Returns: a Trace with :choices, :score, :retval")
  (println "  Same operation, same return shape, across all three substrates.")

  (print-subsection "(1) Bernoulli coin-flip with bias=0.7")
  (let [t (p/simulate coin-flip [(mx/scalar 0.7)])]
    (show-coin-trace "trace:" t))

  (print-subsection "(2) Two-step Gaussian with μ=0.0, σ=1.0")
  (let [t (p/simulate two-step [(mx/scalar 0.0) (mx/scalar 1.0)])]
    (show-two-step-trace "trace:" t))

  (print-subsection (str "(3) Qwen LLM, " MAX-TOKENS " tokens"))
  (let [t (p/simulate llm-gf [prompt-ids MAX-TOKENS])]
    ;; show-llm-trace returns a promise (decode-trace is async); await it
    ;; so the output prints in order with the rest of the demo.
    (show-llm-trace tokenizer "trace:" t)))

(defn demo-generate
  "GFI op 2/6 — generate: forward-sample with constraints.
   Returns {:trace :weight} where :weight = log p(constraints | model).
   Same operation; same return shape across substrates."
  [llm-gf tokenizer {:keys [prompt-ids paris-id]}]
  (print-section "2. GENERATE — forward-sample with constraints")
  (println "  Returns: {:trace Trace, :weight log-marginal-likelihood}")
  (println "  The weight is structurally identical across substrates:")
  (println "  log P(constraints | model).  This is COHERENCE in our framework.")

  (print-subsection "(1) Coin-flip — constrain :outcome to 1")
  (let [r (p/generate coin-flip
                      [(mx/scalar 0.7)]
                      (cm/set-value cm/EMPTY :outcome (mx/scalar 1.0)))]
    (println (str "  weight (log p(:outcome=1 | bias=0.7)): "
                  (fmt (->scalar (:weight r)))))
    (show-coin-trace "trace:" (:trace r)))

  (print-subsection "(2) Two-step — constrain :z to 1.0")
  (let [r (p/generate two-step
                      [(mx/scalar 0.0) (mx/scalar 1.0)]
                      (cm/set-value cm/EMPTY :z (mx/scalar 1.0)))]
    (println (str "  weight (log p(:z=1.0 | μ=0, σ=1)): "
                  (fmt (->scalar (:weight r)))))
    (show-two-step-trace "trace:" (:trace r)))

  (print-subsection "(3) LLM — constrain :t0 to ' Paris'")
  (let [r (p/generate llm-gf
                      [prompt-ids MAX-TOKENS]
                      (cm/set-value cm/EMPTY :t0
                                    (mx/scalar paris-id mx/int32)))]
    (println (str "  weight (log p(:t0=' Paris' | prompt)): "
                  (fmt (->scalar (:weight r)))))
    (show-llm-trace tokenizer "trace:" (:trace r))))

(defn demo-assess
  "GFI op 3/6 — assess: score fully-specified choices.
   Returns {:retval :weight} where :weight = log p(choices | model).
   No sampling occurs — pure scoring."
  [llm-gf _tokenizer {:keys [prompt-ids paris-id]}]
  (print-section "3. ASSESS — score a fully-specified choice")
  (println "  Returns: {:retval, :weight}")
  (println "  No sampling; the system answers 'how plausible is this?'")

  (print-subsection "(1) Coin-flip — score :outcome=1 under bias=0.7")
  (let [r (p/assess coin-flip
                    [(mx/scalar 0.7)]
                    (cm/set-value cm/EMPTY :outcome (mx/scalar 1.0)))]
    (println (str "  weight: " (fmt (->scalar (:weight r))))))

  (print-subsection "(2) Two-step — score :z=1.0, :y=1.5")
  (let [choices (-> cm/EMPTY
                    (cm/set-value :z (mx/scalar 1.0))
                    (cm/set-value :y (mx/scalar 1.5)))
        r (p/assess two-step
                    [(mx/scalar 0.0) (mx/scalar 1.0)]
                    choices)]
    (println (str "  weight: " (fmt (->scalar (:weight r))))))

  (print-subsection "(3) LLM — score the completion ' Paris' (one token)")
  ;; Assess just the first token; fix max-tokens to 1.
  (let [r (p/assess llm-gf
                    [prompt-ids 1]
                    (cm/set-value cm/EMPTY :t0
                                  (mx/scalar paris-id mx/int32)))]
    (println (str "  weight (log p(' Paris' | 'The capital of France is')): "
                  (fmt (->scalar (:weight r)))))))

(defn demo-update
  "GFI op 4/6 — update: modify a trace with new constraints.
   Returns {:trace :weight :discard}.  The same operation across substrates."
  [llm-gf tokenizer {:keys [prompt-ids paris-id london-id]}]
  (print-section "4. UPDATE — modify a trace with new constraints")
  (println "  Returns: {:trace, :weight, :discard}")
  (println "  Same operation across substrates: counterfactual modification.")

  (print-subsection "(1) Coin-flip — start from outcome=0, update to outcome=1")
  (let [t0 (p/generate coin-flip [(mx/scalar 0.7)]
                       (cm/set-value cm/EMPTY :outcome (mx/scalar 0.0)))
        r  (p/update coin-flip (:trace t0)
                     (cm/set-value cm/EMPTY :outcome (mx/scalar 1.0)))]
    (show-coin-trace "before:" (:trace t0))
    (show-coin-trace "after: " (:trace r))
    (println (str "  weight delta: " (fmt (->scalar (:weight r))))))

  (print-subsection "(2) Two-step — keep :z, change :y away from z")
  (let [t0 (p/generate two-step [(mx/scalar 0.0) (mx/scalar 1.0)]
                       (-> cm/EMPTY
                           (cm/set-value :z (mx/scalar 1.0))
                           (cm/set-value :y (mx/scalar 1.0))))
        r  (p/update two-step (:trace t0)
                     (cm/set-value cm/EMPTY :y (mx/scalar 3.0)))]
    (show-two-step-trace "before:" (:trace t0))
    (show-two-step-trace "after: " (:trace r))
    (println (str "  weight delta (less likely under N(z=1, σ=0.5)): "
                  (fmt (->scalar (:weight r))))))

  (print-subsection "(3) LLM — start with one completion, update :t0")
  (pr/let [t0 (p/generate llm-gf [prompt-ids MAX-TOKENS]
                          (cm/set-value cm/EMPTY :t0
                                        (mx/scalar paris-id mx/int32)))
           r  (p/update llm-gf (:trace t0)
                        (cm/set-value cm/EMPTY :t0
                                      (mx/scalar london-id mx/int32)))
           _  (show-llm-trace tokenizer "before:" (:trace t0))
           _  (show-llm-trace tokenizer "after: " (:trace r))]
    (println (str "  weight delta: " (fmt (->scalar (:weight r)))))))

(defn demo-regenerate
  "GFI op 5/6 — regenerate: resample selected addresses.
   Returns {:trace :weight}.  The remaining addresses keep their values."
  [llm-gf tokenizer {:keys [prompt-ids paris-id]}]
  (print-section "5. REGENERATE — resample at selected addresses")
  (println "  Returns: {:trace, :weight}")
  (println "  Selected addresses are resampled; others are preserved.")

  (print-subsection "(1) Coin-flip — regenerate :outcome")
  (let [t0 (p/simulate coin-flip [(mx/scalar 0.7)])
        r  (p/regenerate coin-flip t0 (sel/select :outcome))]
    (show-coin-trace "before:" t0)
    (show-coin-trace "after: " (:trace r)))

  (print-subsection "(2) Two-step — regenerate just :y, keep :z")
  (let [t0 (p/simulate two-step [(mx/scalar 0.0) (mx/scalar 1.0)])
        r  (p/regenerate two-step t0 (sel/select :y))]
    (show-two-step-trace "before:" t0)
    (show-two-step-trace "after: " (:trace r)))

  (print-subsection "(3) LLM — regenerate from :t1 onward, keep :t0")
  ;; First fix :t0 to ' Paris', then regenerate :t1, :t2, ...
  (pr/let [t0 (p/generate llm-gf [prompt-ids MAX-TOKENS]
                          (cm/set-value cm/EMPTY :t0
                                        (mx/scalar paris-id mx/int32)))
           regen-sel (apply sel/select
                            (mapv #(keyword (str "t" %))
                                  (range 1 MAX-TOKENS)))
           r  (p/regenerate llm-gf (:trace t0) regen-sel)
           _  (show-llm-trace tokenizer "before:" (:trace t0))
           _  (show-llm-trace tokenizer "after: " (:trace r))]))

(defn demo-project
  "GFI op 6/6 — project: log-prob of a selected sub-set of choices.
   Returns an MLX scalar.  Marginalizes the remaining addresses (in a sense)."
  [llm-gf tokenizer {:keys [prompt-ids]}]
  (print-section "6. PROJECT — log-prob of a selected subset")
  (println "  Returns: an MLX scalar = sum of log-probabilities at selected addresses")

  (print-subsection "(1) Coin-flip — project :outcome from a simulated trace")
  (let [t (p/simulate coin-flip [(mx/scalar 0.7)])
        w (p/project coin-flip t (sel/select :outcome))]
    (show-coin-trace "trace:" t)
    (println (str "  project weight: " (fmt (->scalar w)))))

  (print-subsection "(2) Two-step — project just :z (drop :y contribution)")
  (let [t (p/simulate two-step [(mx/scalar 0.0) (mx/scalar 1.0)])
        w-z   (p/project two-step t (sel/select :z))
        w-all (p/project two-step t (sel/select :z :y))]
    (show-two-step-trace "trace:" t)
    (println (str "  project :z only:   " (fmt (->scalar w-z))))
    (println (str "  project :z and :y: " (fmt (->scalar w-all)))))

  (print-subsection "(3) LLM — project just :t0")
  ;; Print the (sync) project weight first, then return the show-llm-trace
  ;; promise so the outer pr/do awaits its async print before continuing.
  (let [t (p/simulate llm-gf [prompt-ids MAX-TOKENS])
        w (p/project llm-gf t (sel/select :t0))]
    (println (str "  project :t0 only: " (fmt (->scalar w))))
    (show-llm-trace tokenizer "trace:" t)))

;; ----------------------------------------------------------------------------
;; Closing reflection — what the demo just demonstrated.
;; ----------------------------------------------------------------------------

(defn closing-remarks []
  (print-section "WHAT THIS DEMONSTRATES")
  (println "
  Six operations.  Three substrates.  One uniform structural treatment.

    SIMULATE     → Trace                              ← all three
    GENERATE     → {:trace, :weight}                  ← all three
    ASSESS       → {:retval, :weight}                 ← all three
    UPDATE       → {:trace, :weight, :discard}        ← all three
    REGENERATE   → {:trace, :weight}                  ← all three
    PROJECT      → log-probability scalar             ← all three

  The handler does not know that one substrate is a coin and another is a
  35-billion-parameter language model.  It does not need to know.  Each
  trace site receives a distribution; sampling that distribution given a
  PRNG key produces a value; the choice + score are accumulated into the
  handler's state.  The substrate's complexity is hidden behind the
  distribution interface.

  THE LLM IS, STRUCTURALLY, A DISTRIBUTION.  Its parameters happen to be
  computed by a neural network forward pass — but a categorical with
  LLM-computed logits is the same KIND of object as a Bernoulli with
  hand-set p, or a Gaussian with hand-set μ and σ.

  Every call to one of these gen functions is one cycle.  Each cycle is a
  state transition (the handler advances) in relation to an event (the
  parameters arriving at the trace site).  By the operant-psychology
  definition, this is a BEHAVIOR.  The completion produced is the
  behavior's OUTPUT, not the behavior itself.

  The structural identity dissolves the question Griffiths et al. organize
  their paper around.  There is no fact of the matter about whether the
  LLM is 'symbolic' or 'subsymbolic'.  At the canonical-form level —
  the level at which Machine Psychology operates — the LLM is a
  canonical-form machine that supports the GFI's operations, exactly as
  a coin flip does.  The interesting questions are structural:
    - Which operations does it support?  (all six)
    - With what schema?  (one trace site per token, categorical at each)
    - Under what AIKR?  (bounded context, finite parameters,
                         open input, real-time per cycle)

  See PAPER_MACHINE_PSYCHOLOGY.md §4 for the discussion.
"))

;; ============================================================================
;; Main async entry point
;; ============================================================================

(println "Loading model:" MODEL-NAME)
(println "  (this can take a moment for the 35B-A3B variant)")

(pr/let [m         (llm/load-model MODEL-DIR)
         _         (println "Model loaded.  Wrapping as gen function...")
         llm-gf    (llm-core/make-llm-gf m)
         tokenizer (:tokenizer m)

         ;; Encode the prompts and target tokens ONCE, then pass to each demo.
         raw-prompt (llm/encode tokenizer "The capital of France is")
         raw-paris  (llm/encode tokenizer " Paris")
         raw-london (llm/encode tokenizer " London")
         tokens     {:prompt-ids (vec raw-prompt)
                     :paris-id   (first (vec raw-paris))
                     :london-id  (first (vec raw-london))}
         _          (println "Ready.\n")]

  (pr/do
    (demo-simulate    llm-gf tokenizer tokens)
    (demo-generate    llm-gf tokenizer tokens)
    (demo-assess      llm-gf tokenizer tokens)
    (demo-update      llm-gf tokenizer tokens)
    (demo-regenerate  llm-gf tokenizer tokens)
    (demo-project     llm-gf tokenizer tokens)
    (closing-remarks)
    (println "\nDone.")))
