(ns fim-coverage-probe
  "The DECISIVE coverage + identity probe (bean genmlx-ttm8, epic genmlx-fjuk).

   genmlx-fjuk's binding question is whether the 80B Qwen3-Coder-Next FIM proposer
   actually COVERS structure the deterministic grid cannot enumerate — and whether each
   assembled program ROUND-TRIPS through llm-proposer/parse-spec to a faithful spec (or
   is a :code-only stale-spec particle). The fim_infill.cljs PoC punched the hole at a
   single NUMBER (a prior σ), which a grid trivially enumerates — so it could not isolate
   FIM's contribution (and indeed FIM just copied the visible σ). This probe punches the
   hole at an obs-site MEAN EXPRESSION instead, where the needed answer is

       (mx/add (mx/multiply slope (mx/scalar 3.0)) intercept)

   — an expression no σ-grid can reach. The target is a linear-Gaussian regression whose
   data was generated WITH an intercept (y = 2x + 5), but whose visible obs (y0, y1) are
   rendered WITHOUT it: `(mx/multiply slope (mx/scalar x))`. The `intercept` latent is
   present-but-unused (bound in the let, referenced nowhere). So FIM must INVENT the use
   of an in-scope latent against the copy-the-visible-pattern pull — the genuinely hard
   case, mirroring the σ lesson.

   We measure, over K sampled fills:
     COVERAGE  — fraction whose fill wires `intercept` into an additive mean (mx/add ...)
     IDENTITY  — fraction that parse-spec round-trips WITHOUT dropping that structure
     EVIDENCE  — does the coherent full-intercept model's exact log-ML beat the
                 no-intercept init (confirming the oracle gradient FIM is chasing is real)

   Verdict (recorded back on genmlx-fjuk):
     (a) coverage fails        -> L1/L2 moot for cheap; the proposer is the wall.
     (b) covers but parse-spec drops it -> HIGH identity risk proven; fix before any loop.
     (c) both pass             -> build L1 (FIM-SMC over construction steps).

   NATIVE probe, Thor-only (native qwen3_next MoE forward SIGTRAPs on Metal). Touches NO
   search.cljs / smcp3 machinery — it is a standalone extension of examples/fim_infill.cljs.

   Run: FIM_PROBE=1 bunx --bun nbb@1.4.208 examples/fim_coverage_probe.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.msa-score :as score]
            [genmlx.world.llm-proposer :as lp]
            [genmlx.world.synth :as syn]
            [genmlx.codegen.eval :as ce]
            [clojure.string :as str]
            [promesa.core :as p]))

;; ── Resident 80B Qwen3-Coder-Next (model_type qwen3_next -> native forward) ────
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

(defn load-80b! []
  (if @model-map*
    (p/resolved @model-map*)
    (p/let [mm (llm/load-model (resolve-snapshot repo-root))]
      (reset! model-map* mm)
      (let [tk (:tokenizer mm)]
        (reset! stops* (set (remove nil? [(llm/eos-token-id tk)
                                          (llm/token->id tk "<|fim_pad|>")
                                          (llm/token->id tk "<|endoftext|>")
                                          (llm/token->id tk "<|im_end|>")]))))
      (println (str "80B Qwen3-Coder-Next resident (" (name (:type mm)) ", native forward)."))
      mm)))

;; ── FIM generation (one prefill + greedy/sampled steps, one resident cache) ───
(defn- fim-prompt [prefix suffix]
  (str "<|fim_prefix|>" prefix "<|fim_suffix|>" suffix "<|fim_middle|>"))

(defn fim-generate
  ([prefix suffix] (fim-generate prefix suffix {}))
  ([prefix suffix {:keys [max-tokens temp seed] :or {max-tokens 48 temp 0.0}}]
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

;; ── The target: a linear-Gaussian regression with the intercept latent
;;    present-but-unused, and the hole punched at the y2 obs MEAN. ─────────────
;;
;; Data is y = 2x + 5 (slope 2, intercept 5) at x = 1,2,3 — so the no-intercept
;; through-origin line CANNOT fit (no single slope hits 7,9,11), and the oracle
;; should prefer the model that wires the in-scope `intercept` into the mean.
(def observations {:y0 7.0 :y1 9.0 :y2 11.0})

(def template-prefix
  (str "(fn [trace]\n"
       "  (let [slope (trace :slope (dist/gaussian 0 10))\n"
       "        intercept (trace :intercept (dist/gaussian 0 10))]\n"
       "    {:y0 (trace :y0 (dist/gaussian (mx/multiply slope (mx/scalar 1.0)) 1.0))\n"
       "     :y1 (trace :y1 (dist/gaussian (mx/multiply slope (mx/scalar 2.0)) 1.0))\n"
       "     :y2 (trace :y2 (dist/gaussian "))
(def template-suffix " 1.0))}))")

(defn assemble [middle] (str template-prefix middle template-suffix))

;; The two coherent REFERENCE models (hand-built, not FIM) used only to confirm the
;; oracle gradient is real: the no-intercept init vs the full-intercept target. The
;; init binds ONLY slope (a clean single-latent linear-Gaussian -> :exact); the target
;; adds and USES intercept (joint two-latent linear-Gaussian -> :exact). An unused
;; intercept latent would defeat analytical elimination and force a noisy IS score, so
;; the honest simpler baseline omits it entirely.
(defn- ref-model [latents mean-fn]
  (str "(fn [trace]\n"
       "  (let [" latents "]\n"
       "    {:y0 (trace :y0 (dist/gaussian " (mean-fn 1.0) " 1.0))\n"
       "     :y1 (trace :y1 (dist/gaussian " (mean-fn 2.0) " 1.0))\n"
       "     :y2 (trace :y2 (dist/gaussian " (mean-fn 3.0) " 1.0))}))"))
(def no-intercept-init
  (ref-model "slope (trace :slope (dist/gaussian 0 10))"
             (fn [x] (str "(mx/multiply slope (mx/scalar " x "))"))))
(def full-intercept-ref
  (ref-model (str "slope (trace :slope (dist/gaussian 0 10))\n"
                  "        intercept (trace :intercept (dist/gaussian 0 10))")
             (fn [x] (str "(mx/add (mx/multiply slope (mx/scalar " x ")) intercept)"))))

;; ── Measurement helpers ──────────────────────────────────────────────────────
(defn- tree-has? [form sym]
  (boolean (some #(= % sym) (tree-seq coll? seq (if (coll? form) form (list form))))))

(defn- mean-of
  "The mean (first) arg form of the obs site `addr` in a parsed synth spec, or nil."
  [spec addr]
  (some #(when (= addr (:addr %)) (first (:args %))) (:obs spec)))

(defn- additive-intercept?
  "Does a mean form wire `intercept` into an additive (mx/add ...) mean — the needed
   structure, distinguished from a bare `intercept` or an unused mention?"
  [mean-form]
  (boolean (and mean-form (tree-has? mean-form 'intercept) (tree-has? mean-form 'mx/add))))

(defn measure
  "Score one FIM fill on all five probe axes. Returns a row map."
  [middle]
  (let [code     (assemble middle)
        gf       (score/eval-model code)
        valid?   (some? gf)
        spec     (lp/parse-spec code)                         ; identity round-trip
        spec'    (when spec (lp/parse-spec (syn/render spec))) ; render -> re-parse
        y2-mean  (when spec (mean-of spec :y2))
        ;; COVERAGE: the fill wires intercept additively into the y2 mean (parse-based,
        ;; with a raw-string fallback when the program is off-grammar for parse-spec).
        covers?  (if spec (additive-intercept? y2-mean)
                     (and (str/includes? middle "intercept") (str/includes? middle "mx/add")))
        ;; IDENTITY: parse-spec returns a spec that re-renders to the SAME spec AND does
        ;; not DROP the covered structure (the feared stale-spec failure: code covers,
        ;; spec doesn't). A covering fill that parse-spec mangles is the (b) risk.
        faithful? (boolean (and spec spec' (= spec spec')
                                (= covers? (additive-intercept? (mean-of spec :y2)))))
        {:keys [log-ml method]} (if gf (score/score-model* gf observations {}) {:log-ml ##-Inf :method nil})]
    {:middle (str/trim middle) :valid? valid? :covers? covers?
     :in-grammar? (some? spec) :faithful? faithful?
     :log-ml log-ml :method method}))

(defn- fmt [x] (if (and x (js/isFinite x)) (.toFixed (js/Number x) 3) "  -Inf"))
(defn- pad [s n] (let [s (str s)] (str (str/join (repeat (max 0 (- n (count s))) " ")) s)))

;; ── The probe ─────────────────────────────────────────────────────────────────
(defn probe
  "Generate K FIM fills for the y2-mean hole (one greedy anchor + sampled for
   diversity), measure each, score the two reference models, and print the verdict."
  ([] (probe 10))
  ([k]
   (p/let [;; one greedy fill (what the model most wants) + (k-1) sampled fills
           middles (reduce (fn [pacc i]
                             (p/let [acc pacc
                                     m (fim-generate template-prefix template-suffix
                                                     {:temp (if (zero? i) 0.0 0.8)
                                                      :seed (+ 11 i) :max-tokens 48})]
                               (conj acc m)))
                           (p/resolved [])
                           (range k))]
     (let [rows      (mapv measure middles)
           init-ll   (:log-ml (score/score-model* (score/eval-model no-intercept-init) observations {}))
           full      (score/score-model* (score/eval-model full-intercept-ref) observations {})
           n         (count rows)
           n-valid   (count (filter :valid? rows))
           n-cover   (count (filter :covers? rows))
           n-cov&val (count (filter #(and (:valid? %) (:covers? %)) rows))
           ;; identity is only meaningful for the fills that COVERED: of those, how many
           ;; round-trip faithfully (the rest are the stale-spec risk).
           covering  (filter :covers? rows)
           n-faith   (count (filter :faithful? covering))]
       (println "\n========================================================================")
       (println " genmlx-ttm8 — FIM coverage + identity probe (hole at the obs-site MEAN)")
       (println "========================================================================")
       (println (str "\nData: y = 2x + 5  -> " (pr-str observations)
                     "   (intercept latent present-but-unused in the prefix)"))
       (println "\nHole (y2 mean) — needed structure: (mx/add (mx/multiply slope (mx/scalar 3.0)) intercept)\n")
       (println (str "  " (pad "valid" 5) "  " (pad "covers" 6) "  " (pad "spec" 4) "  "
                     (pad "faithful" 8) "  " (pad "log-ML" 9) "  method      fill"))
       (doseq [{:keys [valid? covers? in-grammar? faithful? log-ml method middle]} rows]
         (println (str "  " (pad (if valid? "yes" "NO") 5) "  " (pad (if covers? "YES" "-") 6) "  "
                       (pad (if in-grammar? "ok" "off") 4) "  " (pad (if covers? (if faithful? "yes" "DROP") "-") 8) "  "
                       (pad (fmt log-ml) 9) "  " (pad (or (some-> method name) "-") 10) "  "
                       (subs middle 0 (min 56 (count middle))))))
       (println "\n── EVIDENCE gradient (coherent reference models, exact) ──")
       (println (str "  no-intercept init : log-ML = " (fmt init-ll)))
       (println (str "  full-intercept ref: log-ML = " (fmt (:log-ml full))
                     "   (method " (name (or (:method full) :?)) ")"))
       (let [grad-ok? (and (js/isFinite (:log-ml full)) (js/isFinite init-ll)
                           (> (:log-ml full) init-ll))]
         (println (str "  => intercept model " (if grad-ok? "BEATS" "does NOT beat")
                       " the no-intercept init by "
                       (fmt (when (and (js/isFinite (:log-ml full)) (js/isFinite init-ll))
                              (- (:log-ml full) init-ll))) " nats."))
         (println "\n── SUMMARY ──")
         (println (str "  COVERAGE : " n-cover "/" n " fills wire intercept additively into the mean ("
                       n-cov&val " also eval-valid)."))
         (println (str "  IDENTITY : " n-faith "/" (count covering)
                       " covering fills round-trip through parse-spec faithfully."))
         (println (str "  EVIDENCE : " (if grad-ok? "confirmed" "FAILED")
                       " — the oracle gradient points at the intercept."))
         (let [verdict (cond
                         (zero? n-cover)            "(a) COVERAGE FAILS — FIM never proposes the intercept term. L1/L2 are moot for cheap; the proposer is the wall."
                         (< n-faith (count covering)) "(b) COVERS but parse-spec DROPS structure — HIGH identity risk PROVEN. Fix parse-spec/particle-identity before any search loop."
                         grad-ok?                   "(c) BOTH PASS — FIM covers and round-trips, the oracle rewards it. Build L1 (FIM-SMC over construction steps)."
                         :else                      "INCONCLUSIVE — coverage+identity hold but the evidence gradient did not confirm; recheck the target/scoring.")]
           (println (str "\n  VERDICT: " verdict))
           {:rows rows :n n :n-valid n-valid :n-cover n-cover :n-cov&val n-cov&val
            :n-covering (count covering) :n-faithful n-faith
            :init-ll init-ll :full full :grad-ok? grad-ok? :verdict verdict}))))))

(defn report [_] nil)

;; ── scripted ──────────────────────────────────────────────────────────────────
(when (.. js/process -env -FIM_PROBE)
  (println "* loading 80B Qwen3-Coder-Next (FIM proposer) — resident, paid ONCE ...")
  (-> (load-80b!)
      (.then (fn [_] (probe 10)))
      (.catch (fn [e] (println "PROBE ERR:" (str e) (when e (.-stack e)))))))
