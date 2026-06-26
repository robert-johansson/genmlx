(ns fim-infill
  "FIM-infill synthesis: fill a HOLE in a GenMLX probabilistic program using the
   resident 80B Qwen3-Coder-Next in fill-in-the-middle mode, then validity-gate and
   rank the candidates with the GFI oracle. A three-layer filter:

     FIM proposes        — <|fim_prefix|> P <|fim_suffix|> S <|fim_middle|>  (semantic fit)
     eval-model gates     — SCI parse+eval of the assembled program          (syntactic/executable)
     score-model ranks    — Bayesian model evidence, log p(obs)              (does it explain data)

   This is the genmlx-bjmm experiment (FIM as a pure proposer scored by the oracle —
   no owned-forward / differentiability needed) on the native MoE forward wired by
   genmlx-2sh6, exercised REPL-style: the 80B is loaded ONCE and resident, the
   hole/template/data are live values you redefine and re-(synth) for free.

   REPL:   (load-file \"examples/fim_infill.cljs\")  (require it on the genmlx cp),
           then (p/then (load-80b!) (fn [_] (synth 8)))
   Script: FIM_DEMO=1 bunx --bun nbb@1.4.208 examples/fim_infill.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.msa :as msa]
            [clojure.string :as str]
            [promesa.core :as p]))

;; The 80B Qwen3-Coder-Next (model_type qwen3_next → native Qwen35MoeModel via
;; genmlx-2sh6). Resolve the HF snapshot dir that holds config.json.
(def ^:private repo-root
  (or (.-QWEN3_NEXT_DIR js/process.env)
      (str (.-HOME js/process.env) "/code/mlx/models/Qwen3-Coder-Next-4bit")))

(defn- resolve-snapshot [dir]
  (let [fs (js/require "fs")]
    (cond
      (.existsSync fs (str dir "/config.json")) dir
      (.existsSync fs (str dir "/snapshots"))
      (->> (.readdirSync fs (str dir "/snapshots"))
           (map #(str dir "/snapshots/" %))
           (filter #(.existsSync fs (str % "/config.json")))
           first)
      :else dir)))

(def model-map* (atom nil))
(def ^:private stops* (atom nil))

(defn load-80b!
  "Load the 80B FIM teacher ONCE (resident). Returns a promise of the model-map."
  []
  (if @model-map*
    (p/resolved @model-map*)
    (p/let [mm (llm/load-model (resolve-snapshot repo-root))]
      (reset! model-map* mm)
      (let [tk (:tokenizer mm)]
        ;; FIM generation ends at any of these (fim_pad / endoftext / im_end / eos).
        (reset! stops* (set (remove nil? [(llm/eos-token-id tk)
                                          (llm/token->id tk "<|fim_pad|>")
                                          (llm/token->id tk "<|endoftext|>")
                                          (llm/token->id tk "<|im_end|>")]))))
      (println (str "80B Qwen3-Coder-Next resident (" (name (:type mm)) ", native forward)."))
      mm)))

;; ── FIM generation ──────────────────────────────────────────────────────────
(defn- fim-prompt [prefix suffix]
  (str "<|fim_prefix|>" prefix "<|fim_suffix|>" suffix "<|fim_middle|>"))

(defn fim-generate
  "Generate the FIM middle string connecting `prefix` and `suffix`. temp 0 = greedy,
   >0 = sampled (seed for reproducibility). Stops at a FIM stop token or max-tokens.
   Returns a promise of the decoded middle."
  ([prefix suffix] (fim-generate prefix suffix {}))
  ([prefix suffix {:keys [max-tokens temp seed] :or {max-tokens 14 temp 0.0}}]
   (let [{:keys [model tokenizer]} @model-map*]
     (p/let [enc (llm/encode tokenizer (fim-prompt prefix suffix) true)
             ids (vec enc)]
       (llm/init-cache! model)
       (let [greedy? (<= temp 0)
             inv-t   (when-not greedy? (mx/scalar (/ 1.0 temp)))
             stops   @stops*
             pick    (if greedy?
                       (fn [lg rk] [(mx/item (mx/argmax lg)) rk])
                       (fn [lg rk] (let [[sk nk] (rng/split rk)]
                                     [(mx/item (rng/categorical sk (mx/multiply lg inv-t))) nk])))
             out (try
                   (loop [i 0, acc [], logits (llm/forward-prefill model ids)
                          rk (rng/ensure-key (when seed (rng/fresh-key seed)))]
                     (if (>= i max-tokens)
                       acc
                       (let [[tid nk] (pick logits rk)]
                         (if (contains? stops tid)
                           acc
                           (recur (inc i) (conj acc tid) (llm/forward-step model tid) nk)))))
                   (finally (llm/reset-cache! model)))]
         (llm/decode tokenizer (js/Uint32Array.from (clj->js out))))))))

;; ── The hole, the template, the data (all hot-reloadable values) ─────────────
;;
;; A conjugate linear-Gaussian regression with the slope's PRIOR STD-DEV punched
;; out as the hole. Data is generated from slope = 2 (noiseless: y = 2x), so the
;; marginal evidence log p(y) is maximised by a prior wide enough to reach slope 2
;; and penalised when it is too tight (slope pinned near 0) or needlessly diffuse.
(def template-prefix
  "(fn [trace]\n  (let [slope (trace :slope (dist/gaussian 0 ")
(def template-suffix
  (str "))]\n"
       "    (trace :y0 (dist/gaussian (mx/multiply slope (mx/scalar 1.0)) 1))\n"
       "    (trace :y1 (dist/gaussian (mx/multiply slope (mx/scalar 2.0)) 1))\n"
       "    (trace :y2 (dist/gaussian (mx/multiply slope (mx/scalar 3.0)) 1))\n"
       "    slope))"))
(def observations {:y0 2.0 :y1 4.0 :y2 6.0})   ; y = 2x → slope = 2

(defn assemble [fill] (str template-prefix fill template-suffix))

(defn- extract-number
  "Pull the leading numeric literal out of a FIM middle (the model may trail extra
   tokens). nil if the middle does not begin with a number."
  [middle]
  (some-> (re-find #"^\s*([0-9]+(?:\.[0-9]+)?)" (or middle "")) second))

;; ── The GFI oracle: validity-gate (eval-model) + rank (score-model log-ML) ──
(defn oracle
  "Splice `fill` into the hole, SCI-eval the assembled program, and score it against
   `observations`. Returns {:fill :valid? :log-ml :src}. eval-model returns nil on a
   parse/eval failure → that candidate is invalid (the syntactic/executable gate)."
  [fill]
  (let [src (assemble fill)
        gf  (msa/eval-model src)]
    {:fill fill
     :valid? (some? gf)
     :log-ml (if gf (msa/score-model gf observations) ##-Inf)
     :src src}))

(defn synth
  "FIM-propose `k` fills for the hole (sampled for diversity), validity-gate + rank
   by GFI log-ML. Returns a promise of the scored candidates, best first.
   Generates sequentially (one resident model, one KV cache)."
  ([k] (synth k {:temp 1.1}))
  ([k {:keys [temp] :or {temp 1.1}}]
   (p/let [middles (reduce (fn [pacc i]
                             (p/let [acc pacc
                                     m (fim-generate template-prefix template-suffix
                                                     {:temp temp :seed (+ 7 i) :max-tokens 8})]
                               (conj acc m)))
                           (p/resolved [])
                           (range k))]
     (let [fills  (->> middles (map extract-number) (remove nil?) distinct)
           scored (->> fills (map oracle) (sort-by :log-ml >))]
       {:middles middles :scored scored}))))

;; Reference σ grid scored by the oracle alone (no LLM) — the Bayesian evidence
;; LANDSCAPE the FIM proposals are competing on. Lets us see whether the code
;; model's syntactic guess matches the value the data actually prefers.
(def reference-sigmas ["0.1" "0.5" "1" "2" "3" "5" "10" "30" "100"])

(defn- fmt [x] (if (js/isFinite x) (.toFixed x 3) "-Inf"))
(defn- pad [s n] (str (str/join (repeat (max 0 (- n (count (str s)))) " ")) s))

(defn report [{:keys [middles scored]}]
  (let [proposed (set (map :fill scored))
        grid     (map (fn [s] (assoc (oracle s) :proposed? (contains? proposed s)))
                      reference-sigmas)
        grid-best (:fill (apply max-key :log-ml grid))]
    (println (str "\nFIM raw middles (" (count middles) " samples): "
                  (pr-str (vec (take 10 middles)))))
    (println "\n── GFI oracle evidence landscape (log-ML over a reference σ grid) ──")
    (println "  (★ = the Bayesian optimum; ◄FIM = a value FIM actually proposed)")
    (doseq [{:keys [fill log-ml proposed?]} grid]
      (println (str "  σ = " (pad fill 5) "   log-ML = " (pad (fmt log-ml) 9)
                    (when (= fill grid-best) "  ★")
                    (when proposed? "  ◄FIM"))))
    (println "\n── FIM proposals, validity-gated + ranked by the oracle ──")
    (doseq [{:keys [fill valid? log-ml]} scored]
      (println (str "  σ = " (pad fill 5) "   log-ML = " (pad (fmt log-ml) 9)
                    (when-not valid? "   (invalid — failed eval gate)"))))
    (when-let [best (first scored)]
      (println (str "\nFIM proposed " (pr-str (vec proposed))
                    "; among those the oracle's best is σ = " (:fill best)
                    " (log-ML " (fmt (:log-ml best)) ")."))
      (if (= (:fill best) grid-best)
        (println "That matches the Bayesian optimum on the reference grid ★.")
        (println (str "The Bayesian optimum on the grid is σ = " grid-best " ★ — FIM's syntactic guess "
                      "missed it (it copied the visible obs-noise). The oracle exposes the gap: FIM "
                      "COVERAGE is the bottleneck, exactly what wider sampling / the harvest loop feed.")))
      (println "\nAssembled, validity-gated, oracle-selected GenMLX model:\n")
      (println (:src best)))
    scored))

;; ── scripted demo ─────────────────────────────────────────────────────────
(when (.. js/process -env -FIM_DEMO)
  (println "* loading 80B Qwen3-Coder-Next (FIM teacher) — resident, paid ONCE ...")
  (-> (load-80b!)
      (.then (fn [_]
               (println "\nHole: the slope's prior std-dev in a linear-Gaussian regression.")
               (println "Data: y = 2x  → the oracle should prefer a prior wide enough to reach slope 2.")
               (synth 8)))
      (.then report)
      (.catch (fn [e] (println "DEMO ERR:" (str e) (when e (.-stack e)))))))
