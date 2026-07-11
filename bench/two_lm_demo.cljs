(ns bench.two-lm-demo
  "Two language models, one algebra (genmlx-8dfk, paper E10).

   The strongest instantiation of 'LLMs as compositional components inside
   generative functions': TWO heterogeneous LMs in ONE joint generative
   program — a Qwen3.6-VLM generative function whose structured output
   conditions a Qwen3-Coder-Next generative function; both traced, both
   scored, one trace.

   The program (Box's loop with perception, in one value algebra):
     1. A scatter plot of the observed data is rendered to PNG (deterministic
        pure-CLJS encoder below — the rendered input is itself a frozen
        fixture).
     2. The VLM GF (owned qwen3_5_moe forward, image-conditioned prefix,
        genmlx-jq6l) looks at the plot and emits a STRUCTURED judgment of the
        trend — grammar-constrained to exactly one of linear|quadratic|
        constant, so the judgment is a traced, scored choice with support the
        program controls.
     3. The coder GF (native Qwen3-Coder-Next) proposes a probabilistic
        program CONDITIONED on that judgment (the label selects the coder's
        prompt — the composition seam is inside one gen body, so the
        conditioning is part of the joint trace).
     4. The exact evidence (msa score-model*: L3 analytical elimination when
        it fires) scores the proposed program against the observed data.

   FRAMING DISCIPLINE (roadmap E10): the VLM is an ADVISORY PRIOR —
   calibration-gated, never on the exact-verify path. A miscalibrated
   component is just a prior the evidence overrides; the reference ladder
   below (hand-written linear/quadratic/constant models, exactly scored)
   is what the data actually says, independent of either LLM.

   HONESTY (jitter): big-MoE forwards carry bounded kernel-level score
   jitter (genmlx-cnhi), so per-seed transcripts are FROZEN fixtures —
   generated once, committed; the scorer re-evals the frozen :code
   deterministically. No sampled TEXT is ever asserted.

   Run (Thor, heavy — TWO models resident, ~70 GB; battery discipline
   genmlx-h3p5 mandatory):
     ~/genmlx-guarded-run.sh two-lm-demo bunx --bun nbb@1.4.208 bench/two_lm_demo.cljs

   Env: GENMLX_VLM_MOE_MODEL / GENMLX_CODER_MODEL override checkpoints;
        SEEDS=11,12,13,14 ; MAX_CODER_TOKENS=512 ; SKIP_FORCED=1 skips the
        constrained-generate arm."
  (:require [clojure.string :as str]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as core]
            [genmlx.llm.grammar :as gram]
            [genmlx.llm.msa-score :as score]
            [genmlx.codegen.eval :as ceval]
            [promesa.core :as pr])
  (:require-macros [genmlx.gen :refer [gen]]))

(def fs (js/require "fs"))
(def path-mod (js/require "path"))
(def zlib (js/require "zlib"))
(def os (js/require "os"))
(def child-process (js/require "child_process"))

(def out-dir (.resolve path-mod (js/process.cwd) "results/two-lm-demo"))

(defn- env [k d] (or (aget (.-env js/process) k) d))

(def vlm-dir
  (or (env "GENMLX_VLM_MOE_MODEL" nil)
      (let [base (str (.-HOME js/process.env)
                      "/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots")]
        (when (.existsSync fs base)
          (str base "/" (first (js->clj (.readdirSync fs base))))))))

(def coder-dir
  (or (env "GENMLX_CODER_MODEL" nil)
      (let [base (str (.-HOME js/process.env) "/code/mlx/models/Qwen3-Coder-Next-4bit/snapshots")]
        (when (.existsSync fs base)
          (str base "/" (first (js->clj (.readdirSync fs base))))))))

(def seeds (mapv js/parseInt (str/split (env "SEEDS" "11,12,13,14") #",")))
(def max-coder-tokens (js/parseInt (env "MAX_CODER_TOKENS" "512")))
(def skip-forced? (= "1" (env "SKIP_FORCED" "0")))

;; ===========================================================================
;; The observed data (fixed literals — the experiment's ground truth).
;; y_i = 2*x_i + 5 + e_i with e_i one frozen N(0,1) draw per point; the TRUE
;; structure is linear.
;; ===========================================================================

(def xs [0 1 2 3 4 5 6 7])
(def ys [5.31 6.62 9.84 10.19 13.47 14.98 17.05 18.55])
(def observations (into {} (map-indexed (fn [i y] [(keyword (str "y" i)) y]) ys)))

;; ===========================================================================
;; Deterministic pure-CLJS PNG scatter renderer (the rendered input fixture).
;; Minimal spec-conformant encoder: IHDR + one zlib IDAT (filter 0) + IEND.
;; ===========================================================================

(def ^:private crc-table
  (let [t (js/Uint32Array. 256)]
    (dotimes [n 256]
      (loop [c n, k 0]
        (if (= k 8)
          (aset t n (unsigned-bit-shift-right c 0))
          (recur (if (odd? c) (bit-xor 0xEDB88320 (unsigned-bit-shift-right c 1))
                              (unsigned-bit-shift-right c 1))
                 (inc k)))))
    t))

(defn- crc32 [^js bytes]
  (loop [c 0xFFFFFFFF, i 0]
    (if (= i (.-length bytes))
      (unsigned-bit-shift-right (bit-xor c 0xFFFFFFFF) 0)
      (recur (bit-xor (aget crc-table (bit-and (bit-xor c (aget bytes i)) 0xFF))
                      (unsigned-bit-shift-right c 8))
             (inc i)))))

(defn- be32 [n]
  (js/Uint8Array.from #js [(bit-and (unsigned-bit-shift-right n 24) 0xFF)
                           (bit-and (unsigned-bit-shift-right n 16) 0xFF)
                           (bit-and (unsigned-bit-shift-right n 8) 0xFF)
                           (bit-and n 0xFF)]))

(defn- concat-bytes [arrs]
  (let [total (reduce + (map #(.-length %) arrs))
        out (js/Uint8Array. total)]
    (loop [off 0, [a & more] arrs]
      (if a
        (do (.set out a off) (recur (+ off (.-length a)) more))
        out))))

(defn- png-chunk [type-str ^js data]
  (let [type-bytes (js/Uint8Array.from (map #(.charCodeAt % 0) type-str))
        td (concat-bytes [type-bytes data])]
    (concat-bytes [(be32 (.-length data)) td (be32 (crc32 td))])))

(defn render-scatter-png
  "RGB scatter of (xs, ys): white bg, black axes, dark-blue 7x7 points.
   Returns PNG bytes (Uint8Array). Fully deterministic."
  [xs ys]
  (let [W 480 H 360 m 40
        px (js/Uint8Array. (* W H 3))
        set-px! (fn [x y r g b]
                  (when (and (>= x 0) (< x W) (>= y 0) (< y H))
                    (let [o (* 3 (+ (* y W) x))]
                      (aset px o r) (aset px (inc o) g) (aset px (+ o 2) b))))
        x-lo -0.5 x-hi 7.5 y-lo 0.0 y-hi 22.0
        sx (fn [x] (js/Math.round (+ m (* (- W (* 2 m)) (/ (- x x-lo) (- x-hi x-lo))))))
        sy (fn [y] (js/Math.round (- (- H m) (* (- H (* 2 m)) (/ (- y y-lo) (- y-hi y-lo))))))]
    (.fill px 255)                                        ; white background
    (doseq [x (range m (- W m))]                          ; x axis
      (set-px! x (- H m) 0 0 0))
    (doseq [y (range m (- H m))]                          ; y axis
      (set-px! m y 0 0 0))
    (doseq [[x y] (map vector xs ys)]                     ; points
      (let [cx (sx x) cy (sy y)]
        (doseq [dx (range -3 4) dy (range -3 4)]
          (set-px! (+ cx dx) (+ cy dy) 20 60 160))))
    (let [raw (js/Uint8Array. (* H (inc (* W 3))))]       ; filter-0 scanlines
      (dotimes [row H]
        (let [ro (* row (inc (* W 3)))]
          (aset raw ro 0)
          (.set raw (.subarray px (* row W 3) (* (inc row) W 3)) (inc ro))))
      (concat-bytes
       [(js/Uint8Array.from #js [137 80 78 71 13 10 26 10])
        (png-chunk "IHDR" (concat-bytes [(be32 W) (be32 H)
                                         (js/Uint8Array.from #js [8 2 0 0 0])]))
        (png-chunk "IDAT" (js/Uint8Array. (.deflateSync zlib raw)))
        (png-chunk "IEND" (js/Uint8Array. 0))]))))

;; ===========================================================================
;; Prompts
;; ===========================================================================

(def labels ["linear" "quadratic" "constant"])
(def enum-regex " ?(linear|quadratic|constant)")

(def vlm-question
  (str "The image is a scatter plot: the horizontal axis is the input x "
       "(0 through 7), the vertical axis is the observed value y. Which trend "
       "best describes the points? Answer with exactly one word: "
       "linear, quadratic, or constant."))

(def coder-system
  (str "You write probabilistic models as a single ClojureScript form "
       "(fn [trace] ...).\n"
       "Allowed distributions: (dist/gaussian mean std), (dist/uniform lo hi), "
       "(dist/exponential rate).\n"
       "Allowed arithmetic: (mx/add a b), (mx/multiply a b), (mx/scalar x).\n"
       "Latents are traced at keyword addresses; the eight observations are "
       "traced at :y0 :y1 :y2 :y3 :y4 :y5 :y6 :y7, for inputs x = 0 1 2 3 4 5 6 7.\n"
       "Example — a CONSTANT (no-trend) model:\n"
       "(fn [trace]\n"
       "  (let [mu (trace :mu (dist/gaussian 0 20))]\n"
       "    (trace :y0 (dist/gaussian mu 1))\n"
       "    (trace :y1 (dist/gaussian mu 1))\n"
       "    (trace :y2 (dist/gaussian mu 1))\n"
       "    (trace :y3 (dist/gaussian mu 1))\n"
       "    (trace :y4 (dist/gaussian mu 1))\n"
       "    (trace :y5 (dist/gaussian mu 1))\n"
       "    (trace :y6 (dist/gaussian mu 1))\n"
       "    (trace :y7 (dist/gaussian mu 1))))\n"
       "For a linear trend, use latents slope and intercept with wide gaussian "
       "priors and mean (mx/add (mx/multiply slope (mx/scalar x)) intercept) "
       "with the numeric x of each observation. For a quadratic trend, add a "
       "curvature latent multiplied by (mx/scalar (* x x)) — write the squared "
       "number directly. Observation noise std is 1.\n"
       "Output ONLY the (fn [trace] ...) form. No prose, no code fences."))

(defn coder-user [label]
  (str "A scatter plot of the eight observations shows a " (str/upper-case label)
       " trend. Write the probabilistic model."))

;; Hand-written reference ladder: what the evidence says about THIS data,
;; independent of either LLM (the calibration frame for the advisory prior).
(def reference-codes
  {"constant"
   (str "(fn [trace]\n  (let [mu (trace :mu (dist/gaussian 0 20))]\n"
        (apply str (for [i (range 8)]
                     (str "    (trace :y" i " (dist/gaussian mu 1))\n")))
        "    mu))")
   "linear"
   (str "(fn [trace]\n  (let [slope (trace :slope (dist/gaussian 0 10))\n"
        "        intercept (trace :intercept (dist/gaussian 0 10))]\n"
        (apply str (for [i (range 8)]
                     (str "    (trace :y" i " (dist/gaussian (mx/add (mx/multiply slope (mx/scalar "
                          i ".0)) intercept) 1))\n")))
        "    slope))")
   "quadratic"
   (str "(fn [trace]\n  (let [a (trace :a (dist/gaussian 0 5))\n"
        "        slope (trace :slope (dist/gaussian 0 10))\n"
        "        intercept (trace :intercept (dist/gaussian 0 10))]\n"
        (apply str (for [i (range 8)]
                     (str "    (trace :y" i " (dist/gaussian (mx/add (mx/multiply a (mx/scalar "
                          (* i i) ".0)) (mx/add (mx/multiply slope (mx/scalar "
                          i ".0)) intercept)) 1))\n")))
        "    slope))")})

;; ===========================================================================
;; Helpers
;; ===========================================================================

(defn- ensure-dir [dir]
  (when-not (.existsSync fs dir) (.mkdirSync fs dir #js {:recursive true})))

(defn- git-sha []
  (try (-> (.execSync child-process "git rev-parse --short HEAD") (.toString) (str/trim))
       (catch :default _ "unknown")))

(defn- t-addr [i] (keyword (str "t" i)))

(defn- gen-tail
  "The generated token ids of a make-llm-gf retval (strip the prompt prefix
   and a trailing eos)."
  [ctx prompt-len eos]
  (let [tail (vec (drop prompt-len ctx))]
    (if (= eos (peek tail)) (pop tail) tail)))

(defn- detok
  "Sync detokenize via a compile-constraint :token-index (ASCII enum only)."
  [token-index ids]
  (str/trim (apply str (map #(nth token-index % "") ids))))

(defn- first-fn-form
  "The first balanced (...) form starting at the first \"(fn\" — truncates any
   trailing prose a temp-1 coder appends after the model form. String-aware
   paren counting; nil when no (fn appears or the form never closes."
  [s]
  (when-let [start (some-> s (str/index-of "(fn"))]
    (loop [i start, depth 0, in-str? false]
      (when (< i (count s))
        (let [ch (.charAt s i)]
          (cond
            (and in-str? (= ch "\\")) (recur (+ i 2) depth true)
            (= ch "\"")               (recur (inc i) depth (not in-str?))
            in-str?                   (recur (inc i) depth true)
            (= ch "(")                (recur (inc i) (inc depth) false)
            (= ch ")")                (if (= depth 1)
                                        (subs s start (inc i))
                                        (recur (inc i) (dec depth) false))
            :else                     (recur (inc i) depth false)))))))

(defn- score-code
  "Deterministic scoring of a frozen code string against the observations."
  [code]
  (let [gf (score/eval-model code)
        {:keys [log-ml method]} (if gf
                                  (score/score-model* gf observations {:n-particles 500})
                                  {:log-ml ##-Inf :method nil})]
    {:eval-ok? (boolean gf)
     :method (when method (name method))
     :log-ml log-ml}))

;; ===========================================================================
;; Run
;; ===========================================================================

(when-not (and vlm-dir coder-dir)
  (println "FATAL: need both checkpoints —"
           "\n  VLM  :" (or vlm-dir "MISSING (GENMLX_VLM_MOE_MODEL)")
           "\n  coder:" (or coder-dir "MISSING (GENMLX_CODER_MODEL)"))
  (js/process.exit 1))

(println "============================================================")
(println " two-LM composition demo (E10): VLM prior GF -> coder GF")
(println "============================================================")
(println "VLM  :" vlm-dir)
(println "coder:" coder-dir)
(println "seeds:" (pr-str seeds))

(def t0 (.now js/Date))

;; --- fixture 1: the rendered input --------------------------------------
(ensure-dir out-dir)
(def png-bytes (render-scatter-png xs ys))
(.writeFileSync fs (str out-dir "/scatter.png") png-bytes)
(println (str "\n[render] scatter.png (" (.-length png-bytes) " bytes) — "
              (count xs) " points, true structure linear"))

(->
 (pr/let
 [;; --- load both models (coder native first: fixed footprint) ------------
  _ (println "\n[load] coder (native)…")
  mm-coder (llm/load-model coder-dir)
  _ (println "  type:" (:type mm-coder))
  _ (println "[load] VLM (owned)…")
  mm-vlm (llm/load-model vlm-dir)
  _ (println "  type:" (:type mm-vlm)
             " owned:" (llm/cljs-forward-model? (:model mm-vlm)))

  ;; --- prompts -------------------------------------------------------------
  vlm-chat (llm/render-chat [{:role "user" :content vlm-question :images 1}])
  vlm-ids-raw (llm/encode (:tokenizer mm-vlm) vlm-chat false)
  coder-prompt-ids                       ; label -> pre-encoded prompt ids
  (pr/loop [ls labels, acc {}]
    (if-not (seq ls)
      acc
      ;; think-skip? false: qwen3_next is a non-thinking coder and the proven
      ;; 80B text path (T1/T2 arm A via generate-text-raw+) renders without
      ;; the <think> skip for this family.
      (pr/let [ids (llm/encode (:tokenizer mm-coder)
                               (llm/render-chat
                                [{:role "system" :content coder-system}
                                 {:role "user" :content (coder-user (first ls))}]
                                {:think-skip? false})
                               false)]
        (pr/recur (rest ls) (assoc acc (first ls) (vec ids))))))]

 (let [vlm-prompt (vec vlm-ids-raw)
       constraint (gram/compile-constraint (:tokenizer mm-vlm) enum-regex)
       token-index (:token-index constraint)
       vlm-eos (llm/eos-token-id (:tokenizer mm-vlm))
       coder-eos (llm/eos-token-id (:tokenizer mm-coder))
       ;; :prefill-chunk bounds the VLM decoder-prefill transient — the 80B is
       ;; co-resident, so the single-slab headroom assumption doesn't hold here.
       vlm-gf (gram/constrain
               (core/make-llm-gf mm-vlm {:images [png-bytes] :prefill-chunk 256})
               constraint)
       coder-gf (core/make-llm-gf mm-coder)
       vlm-max 8
       label-of (fn [ctx]
                  (let [w (detok token-index (gen-tail ctx (count vlm-prompt) vlm-eos))]
                    (if (some #{w} labels) w :invalid)))
       ;; THE composite: two LMs, one trace. The VLM's structured judgment is
       ;; a traced choice (its token sites under :vlm); the coder's proposal
       ;; conditions on it (label selects the coder prompt) under :coder.
       composite
       (dyn/auto-key
        (gen []
             (let [vlm-ctx (splice :vlm vlm-gf vlm-prompt vlm-max)
                   label (label-of vlm-ctx)
                   cp (get coder-prompt-ids label
                           (get coder-prompt-ids "constant"))
                   coder-ctx (splice :coder coder-gf cp max-coder-tokens)]
               {:label label :vlm-ctx vlm-ctx :coder-ctx coder-ctx})))
       run-one
       (fn [seed forced-label]
         (let [key (rng/fresh-key seed)
               result
               (if forced-label
                 ;; constrained-generate arm: force the VLM's judgment and let
                 ;; the rest of the joint program respond — the GFI weight
                 ;; prices the forced prior against the VLM's actual belief.
                 (pr/let [ids (llm/encode (:tokenizer mm-vlm) forced-label false)]
                   (let [cmap (reduce (fn [c [i tid]]
                                        (cm/set-choice c [:vlm (t-addr i)]
                                                       (mx/scalar tid mx/int32)))
                                      cm/EMPTY
                                      (map-indexed vector (vec ids)))
                         {:keys [trace weight]} (p/generate (dyn/with-key composite key) [] cmap)]
                     {:trace trace :weight (mx/item weight)}))
                 {:trace (p/simulate (dyn/with-key composite key) [])})]
           (pr/let [{:keys [trace weight]} result
                    retval (:retval trace)
                    coder-text (llm/decode
                                (:tokenizer mm-coder)
                                (js/Uint32Array.from
                                 (clj->js (gen-tail (:coder-ctx retval)
                                                    (count (get coder-prompt-ids
                                                                (:label retval)
                                                                (get coder-prompt-ids "constant")))
                                                    coder-eos))))]
             (mx/force-gc!)
             (let [code (or (first-fn-form (ceval/extract-code coder-text))
                            (first-fn-form coder-text)
                            "")
                   scored (score-code code)]
               (merge {:seed seed
                       :forced-label forced-label
                       :vlm-label (:label retval)
                       :vlm-token-ids (gen-tail (:vlm-ctx retval) (count vlm-prompt) vlm-eos)
                       :trace-score (mx/item (:score trace))
                       :coder-raw coder-text
                       :code code}
                      scored
                      (when weight {:generate-weight weight}))))))]

   (println (str "\n[grammar] enum DFA over " (count labels) " labels; "
                 "vlm prompt " (count vlm-prompt) " tokens; coder prompts "
                 (pr-str (into {} (map (fn [[k v]] [k (count v)]) coder-prompt-ids)))))

   ;; --- reference ladder (deterministic, LLM-free) -------------------------
   (println "\n-- reference ladder (hand-written, exact evidence) --")
   (let [reference (into {}
                         (map (fn [[label code]]
                                (let [s (score-code code)]
                                  (println (str "  " label ": log-ml="
                                                (.toFixed (:log-ml s) 4)
                                                " method=" (:method s)))
                                  [label (assoc s :code code)])))
                         reference-codes)
         best-ref (key (apply max-key (comp :log-ml val) reference))]
     (println (str "  evidence selects: " best-ref))

     ;; --- per-seed simulate + one forced-prior generate ---------------------
     (pr/let [rows
              (pr/loop [ss seeds, acc []]
                (if-not (seq ss)
                  acc
                  (pr/let [_ (println (str "\n[seed " (first ss) "] simulate…"))
                           row (run-one (first ss) nil)]
                    (println (str "  vlm=" (:vlm-label row)
                                  "  eval-ok=" (:eval-ok? row)
                                  "  method=" (:method row)
                                  "  log-ml=" (.toFixed (:log-ml row) 3)
                                  "  trace-score=" (.toFixed (:trace-score row) 2)))
                    (pr/recur (rest ss) (conj acc row)))))
              forced-rows
              (if skip-forced?
                []
                (pr/let [_ (println (str "\n[forced] generate with :vlm forced"
                                         " to \"quadratic\" (seed " (first seeds) ")…"))
                         row (run-one (first seeds) "quadratic")]
                  (println (str "  vlm(forced)=" (:vlm-label row)
                                "  W=" (.toFixed (:generate-weight row) 3)
                                "  method=" (:method row)
                                "  log-ml=" (.toFixed (:log-ml row) 3)))
                  [row]))]

       ;; --- freeze fixtures --------------------------------------------------
       (let [all-rows (into rows forced-rows)
             label-counts (frequencies (map :vlm-label rows))
             agree (count (filter #(= best-ref (:vlm-label %)) rows))
             dur (- (.now js/Date) t0)
             meta {:experiment "two-lm-demo"
                   :bean "genmlx-8dfk"
                   :git_sha (git-sha)
                   :timestamp (.toISOString (js/Date.))
                   :hardware {:platform (.platform os) :arch (.arch os)
                              :cpus (count (.cpus os))
                              :gpu (if (mx/metal-is-available?) "Metal" "CUDA")}
                   :runtime {:engine "bun+nbb" :node (.-version js/process)}
                   :models {:vlm vlm-dir :coder coder-dir}
                   :duration_ms dur
                   :script "bench/two_lm_demo.cljs"}]
         (.writeFileSync fs (str out-dir "/transcripts.edn")
                         (str ";; Frozen per-seed two-LM composition transcripts (genmlx-8dfk, E10).\n"
                              ";; Generated once (MoE forwards jitter, genmlx-cnhi); the scorer\n"
                              ";; re-evals :code deterministically. DO NOT hand-edit.\n"
                              (with-out-str (binding [*print-length* nil]
                                              (pr {:meta meta
                                                   :reference reference
                                                   :rows all-rows})))
                              "\n"))
         (.writeFileSync fs (str out-dir "/metadata.json")
                         (js/JSON.stringify (clj->js meta) nil 2))
         (.writeFileSync
          fs (str out-dir "/data.json")
          (js/JSON.stringify
           (clj->js
            {:experiment "two-lm-demo"
             :description
             (str "Two heterogeneous LMs in one joint generative program: a "
                  "grammar-constrained VLM GF (trend judgment over a rendered "
                  "scatter) whose traced choice conditions a coder GF that "
                  "proposes probabilistic programs, scored by exact evidence. "
                  "The VLM is an advisory prior; the reference ladder is the "
                  "evidence's own verdict on the data.")
             :config {:xs xs :ys ys :true_structure "linear"
                      :labels labels :seeds seeds
                      :vlm_max_tokens vlm-max :coder_max_tokens max-coder-tokens
                      :jitter_note (str "Large-MoE forwards are kernel-level "
                                        "nondeterministic (genmlx-cnhi); "
                                        "transcripts are one-shot frozen "
                                        "fixtures, scores re-derived from "
                                        "frozen :code are deterministic.")}
             :reference {:ladder (into {} (map (fn [[l s]]
                                                 [l (select-keys s [:log-ml :method])]))
                                       reference)
                         :selected best-ref}
             :vlm_prior {:label_counts label-counts
                         :n_seeds (count rows)
                         :agrees_with_evidence agree}
             :rows (mapv #(dissoc % :coder-raw :vlm-token-ids) all-rows)
             :meta meta})
           nil 2))
         (println (str "\n[freeze] " out-dir "/{scatter.png, transcripts.edn, data.json, metadata.json}"))
         (println "\n============================================================")
         (println (str " VLM prior over " (count rows) " seeds: " (pr-str label-counts)
                       "  (evidence: " best-ref ")"))
         (println (str " candidates eval-ok " (count (filter :eval-ok? rows)) "/" (count rows)
                       "; exact-scored "
                       (count (filter #(#{"exact" "kalman"} (:method %)) rows)) "/" (count rows)))
         (println "============================================================")
         (mx/force-gc!)
         (js/process.exit 0))))))
 (pr/catch (fn [e]
             (println "\nFATAL:" (or (some-> e .-message) (pr-str e)))
             (when-let [d (ex-data e)] (println "  ex-data:" (pr-str d)))
             (js/process.exit 1))))
