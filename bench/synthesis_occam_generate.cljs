(ns bench.synthesis-occam-generate
  "ONE-SHOT live-LLM generator for the synthesis-Occam experiment (genmlx-heaw).

   Runs the dense Qwen3-0.6B BASE model in knowledge mode (base model emits
   `name ~ dist(params)` lines -> Instaparse parse-math -> assemble-gen-fn ->
   eval-model) over a fixed task set, and FREEZES the authored programs to a
   checked-in fixture so the scoring bench (bench/synthesis_occam.cljs) is
   deterministic and reproducible.

   This is a provenance tool, NOT a registered experiment: LLM sampling at
   temperature>0 is non-deterministic, so we run it once and commit the result.
   Re-run it only to regenerate the fixture from scratch.

   Reachable conjugate family: knowledge-mode's grammar emits only
   gaussian/uniform/bernoulli/exponential, so the only conjugacy that can fire
   is normal-normal (gaussian prior + gaussian obs). Estimation tasks elicit
   that; binary/rate tasks elicit non-conjugate models (bernoulli/exponential
   likelihood) so the fire-rate denominator spans firing and non-firing models.

   Run: bun run --bun nbb bench/synthesis_occam_generate.cljs"
  (:require [clojure.string :as str]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.msa :as msa]
            [genmlx.mlx :as mx]
            [promesa.core :as pr]))

(def fs (js/require "fs"))
(def path-mod (js/require "path"))

(def model-path
  (str (.-HOME js/process.env) "/.cache/models/qwen3-0.6b-mlx-bf16"))

(def fixture-path
  (.resolve path-mod (js/process.cwd) "bench/fixtures/synthesis_occam_programs.edn"))

;; ---------------------------------------------------------------------------
;; Task set. Each task fixes a latent + observation variable name and supplies
;; observed data. Estimation tasks aim at normal-normal (conjugacy fires);
;; binary/rate tasks aim at bernoulli/exponential likelihoods (do not fire).
;; assemble-gen-fn always builds a model over exactly the task :variables, so
;; every candidate is a well-formed gen fn with the obs address present.
;; ---------------------------------------------------------------------------

(def tasks
  [;; --- estimation tasks: expect normal-normal (gaussian prior + gaussian obs) ---
   {:id :temperature
    :description (str "A thermometer measures a fixed but unknown room temperature with noise. "
                      "Write a model with a latent variable `mu` (the true temperature in Celsius) "
                      "and an observation `y` (the noisy reading). `y` must depend on `mu`.")
    :variables [:mu :y] :obs-addr :y :observations {:y 21.5}}
   {:id :scale-weight
    :description (str "A kitchen scale reports a noisy weight. Write a model with a latent `mu` "
                      "(the true weight in grams) and an observation `y` (the scale reading). "
                      "`y` must depend on `mu`.")
    :variables [:mu :y] :obs-addr :y :observations {:y 503.0}}
   {:id :distance-sensor
    :description (str "An ultrasonic sensor measures distance with noise. Write a model with a "
                      "latent `mu` (the true distance in centimeters) and an observation `y` "
                      "(the sensor reading). `y` must depend on `mu`.")
    :variables [:mu :y] :obs-addr :y :observations {:y 147.2}}
   {:id :test-score
    :description (str "A student has a latent ability and produces a noisy test score. Write a "
                      "model with a latent `mu` (the ability) and an observation `y` (the observed "
                      "score). `y` must depend on `mu`.")
    :variables [:mu :y] :obs-addr :y :observations {:y 88.0}}
   {:id :voltage-meter
    :description (str "A voltmeter reads a fixed voltage with measurement noise. Write a model with "
                      "a latent `mu` (the true voltage) and an observation `y` (the meter reading). "
                      "`y` must depend on `mu`.")
    :variables [:mu :y] :obs-addr :y :observations {:y 5.1}}
   {:id :gps-position
    :description (str "A GPS unit reports a noisy position coordinate. Write a model with a latent "
                      "`mu` (the true coordinate) and an observation `y` (the GPS reading). `y` must "
                      "depend on `mu`.")
    :variables [:mu :y] :obs-addr :y :observations {:y 12.7}}

   ;; --- binary / rate tasks: expect bernoulli / exponential (non-conjugate here) ---
   {:id :coin-bias
    :description (str "A coin has an unknown bias. Write a model with a latent `p` (the probability "
                      "of heads) and an observation `y` (1 if the flip was heads, 0 otherwise). "
                      "`y` must depend on `p`.")
    :variables [:p :y] :obs-addr :y :observations {:y 1.0}}
   {:id :defect-rate
    :description (str "A factory line has an unknown defect probability. Write a model with a latent "
                      "`p` (the defect probability) and an observation `y` (1 if a sampled item is "
                      "defective). `y` must depend on `p`.")
    :variables [:p :y] :obs-addr :y :observations {:y 0.0}}
   {:id :wait-time
    :description (str "Customers arrive and the wait time between arrivals is random. Write a model "
                      "with a latent `rate` (the arrival rate) and an observation `y` (an observed "
                      "wait time). `y` must depend on `rate`.")
    :variables [:rate :y] :obs-addr :y :observations {:y 3.4}}
   {:id :decay-time
    :description (str "A radioactive sample emits particles; the time until the next emission is "
                      "random. Write a model with a latent `rate` (the decay rate) and an "
                      "observation `y` (an observed time). `y` must depend on `rate`.")
    :variables [:rate :y] :obs-addr :y :observations {:y 1.8}}])

(def samples-per-task 5)
(def gen-temperature 0.7)
(def gen-max-tokens 80)
(def seed-base 4200)  ;; per-candidate seed = seed-base + global index (reproducible sampling)

;; Strong few-shot system prompt in the EXACT format parse-math accepts. The
;; ChatSession path leaves Qwen3 thinking tokens in the output (breaks parsing);
;; generate-text-raw injects the think-skip and builds the ChatML scaffold.
(def system-prompt
  (str "You write a probabilistic model as lines of the form:\n"
       "name ~ distribution(params)\n\n"
       "Allowed distributions ONLY:\n"
       "  gaussian(mean, std)\n  uniform(low, high)\n  bernoulli(p)\n  exponential(rate)\n\n"
       "When a variable depends on another, pass that variable's name as a parameter.\n"
       "Use exactly the variable names the user asks for. Output ONLY the lines, "
       "no prose, no code fences, no explanation.\n\n"
       "Example (latent `w`, observation `r` depending on `w`):\n"
       "w ~ gaussian(70, 10)\n"
       "r ~ gaussian(w, 2)"))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn ensure-dir [dir]
  (when-not (.existsSync fs dir)
    (.mkdirSync fs dir #js {:recursive true})))

(defn obs-depends-on-latent?
  "Did the LLM make the observation expression reference the latent variable?
   (A crude proxy for 'the model has real structure', independent of whether
   conjugacy fires.)"
  [dist-map obs-addr latent-addr]
  (let [expr (get dist-map obs-addr "")]
    (boolean (re-find (re-pattern (str "\\b" (name latent-addr) "\\b")) (str expr)))))

(defn write-fixture! [records]
  (ensure-dir (.dirname path-mod fixture-path))
  (.writeFileSync fs fixture-path
                  (str ";; Frozen LLM-authored programs for the synthesis-Occam experiment "
                       "(genmlx-heaw).\n"
                       ";; Generated by bench/synthesis_occam_generate.cljs against "
                       "qwen3-0.6b base (knowledge mode).\n"
                       ";; DO NOT hand-edit: regenerate with the generator to refresh provenance.\n"
                       ";; Each entry is one LLM-synthesized model; the scorer re-evals :code and\n"
                       ";; recomputes fire/exact/IS deterministically.\n\n"
                       (with-out-str (binding [*print-length* nil] (pr (vec records))))
                       "\n"))
  (println (str "\nWrote " (count records) " programs -> " fixture-path)))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(println "============================================================")
(println " synthesis-Occam: live-LLM program generation (freeze step)")
(println "============================================================")
(println "Model:" model-path)
(println "Tasks:" (count tasks) "x" samples-per-task "samples ="
         (* (count tasks) samples-per-task) "candidates\n")

(def jobs
  (vec (for [task tasks, i (range samples-per-task)] [task i])))

(pr/let
  [m  (do (println "[load] loading model...") (llm/load-model model-path))
   _  (println "[load] type:" (:type m) " vocab:" (llm/vocab-size (:tokenizer m)) "\n")
   records
   (pr/loop [idx 0, acc []]
     (if (>= idx (count jobs))
       acc
       (let [[task i] (nth jobs idx)
             {:keys [id description variables obs-addr observations]} task
             latent (first variables)
             seed   (+ seed-base idx)]
         (pr/let [raw      (-> (llm/generate-text-raw
                               m description
                               {:max-tokens gen-max-tokens
                                :temperature gen-temperature
                                :seed seed
                                :system-prompt system-prompt})
                              (pr/catch (fn [_] "")))
                  dist-map (or (msa/parse-math raw) {})
                  code     (msa/assemble-gen-fn variables dist-map)
                  gf       (msa/eval-model code)
                  scored   (if gf
                             (msa/score-model* gf observations {:n-particles 50})
                             {:method nil :log-ml ##-Inf})
                  method   (:method scored)
                  fired?   (boolean (#{:exact :kalman} method))
                  parsed?  (boolean (seq dist-map))
                  rec {:task-id id
                       :description description
                       :variables variables
                       :obs-addr obs-addr
                       :observations observations
                       :latent latent
                       :seed seed
                       :raw-text raw
                       :code code
                       :dist-map dist-map
                       :parsed? parsed?
                       :obs-depends? (obs-depends-on-latent? dist-map obs-addr latent)
                       :eval-ok? (boolean gf)
                       :score-method (when method (name method))
                       :fired? fired?}]
           (when (< idx 6)
             (println (str "    --- raw[" (name id) " #" i "] ---\n"
                           (str/trim raw) "\n    ---"))
             (println (str "    parsed dist-map: " (pr-str dist-map))))
           (println (str "[" (inc idx) "/" (count jobs) "] " (name id) " #" i
                         "  parsed=" (if parsed? "Y" "n")
                         " dep=" (if (:obs-depends? rec) "Y" "n")
                         " method=" (or (:score-method rec) "nil")
                         (when fired? "  <- FIRED")))
           (mx/force-gc!)
           (pr/recur (inc idx) (conj acc rec))))))
   _ (write-fixture! records)
   _ (let [n      (count records)
           valid  (count (filter :eval-ok? records))
           fired  (count (filter :fired? records))
           parsed (count (filter :parsed? records))]
       (println "\n------------------------------------------------------------")
       (println (str " candidates:   " n))
       (println (str " eval-ok:      " valid))
       (println (str " parsed:       " parsed " (" (.toFixed (* 100.0 (/ parsed (max 1 n))) 1) "%)"))
       (println (str " fired (exact):" fired " (" (.toFixed (* 100.0 (/ fired (max 1 valid))) 1)
                     "% of eval-ok)"))
       (println "------------------------------------------------------------")
       (when (< valid 30)
         (println (str "\n!! WARNING: only " valid " valid models (<30). "
                       "Increase samples-per-task or refine tasks."))))]
  (mx/force-gc!)
  (println "\nDone."))
