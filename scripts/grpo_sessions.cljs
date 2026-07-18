(ns grpo-sessions
  "L4: GRPO on the child's own sessions (genmlx-qhy4, epic genmlx-z92i) —
   the overnight-loop training driver: pi session JSONLs -> decision-point
   prompts (genmlx.world.session-grpo) -> with-trainer/train-step! rounds
   -> checkpoint. Reward comes from the seam: a builtin
   (match-administered | tool-format) or an external oracle plugin (the
   maithri2 contract — see genmlx.world.session-reward/resolve-reward).

   Env:
     SESSIONS_DIR  a dir of session *.jsonl, or a parent of such dirs
                   (default ~/.mlx-node/agent/sessions — all projects)
     MODEL_DIR     policy checkpoint (default the 0.8b smoke model; the
                   35B trains via the frozen-experts path — same driver)
     REWARD        match-administered (default) | tool-format | plugin path
     MODE          terminal (default) | all
     STEPS         train steps (default 3); one prompt per step, cycling
     GROUP_SIZE    completions per step (default 4)
     MAX_COMPLETION completion cap (default 96)
     LR            learning rate (default 2e-6)
     TEMPERATURE   rollout temperature (default 0.9)
     SEED          model-thread RNG seed (optional)
     SYSTEM_PROMPT the deployed system prompt (render parity; optional)
     CKPT_OUT      default ~/genmlx-checkpoints/session-grpo-<model>-<date>
     METRICS_OUT   per-step JSONL (default ~/genmlx-battery-logs/, never
                   /tmp — reboot-safe logs are the point)
     SAVE          1 (default) to save weights+aux+optimizer; 0 to skip
     SERVE_CHECK   1 (default) to reload the SAVED checkpoint through the
                   owned forward and decode a greedy line (save->serve
                   closed in one run); 0 to skip

   Discipline (docs/thor-gpu-discipline.md): heavy runs go through
   scripts/guarded-run.sh; ONE GPU process — never concurrently with a
   serving agent. The genmlx-li1p rule is enforced: a step that silently
   skips its gradient apply FAILS the run (exit 1, no save) — a skipped
   apply means NaN grads or garbage ratios, and the artifact is tainted.

   Run (Thor):
     scripts/guarded-run.sh grpo-sessions \\
       bunx --bun nbb@1.4.208 scripts/grpo_sessions.cljs"
  (:require [clojure.set :as cset]
            [clojure.string :as str]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.pi-session :as ps]
            [genmlx.llm.qwen3-forward :as q3f]
            [genmlx.mlx :as mx]
            [genmlx.world.session-grpo :as sg]
            [genmlx.world.session-reward :as sr]
            [genmlx.world.train :as train]
            [promesa.core :as p]
            ["fs" :as fs]
            ["os" :as os]
            ["path" :as path]))

(defn- env
  ([k] (aget (.-env js/process) k))
  ([k default] (or (env k) default)))
(defn- env-int [k default] (if-let [v (env k)] (js/parseInt v) default))
(defn- env-num [k default] (if-let [v (env k)] (js/parseFloat v) default))
(defn- home [& parts] (apply path/join (os/homedir) parts))

(def sessions-dir (env "SESSIONS_DIR" (home ".mlx-node" "agent" "sessions")))
(def model-dir    (env "MODEL_DIR" (home ".cache" "models" "qwen3.5-0.8b-mlx-bf16")))
(def reward-spec  (env "REWARD" "match-administered"))
(def mode         (keyword (env "MODE" "terminal")))
(def steps        (env-int "STEPS" 3))
(def group-size   (env-int "GROUP_SIZE" 4))
(def max-completion (env-int "MAX_COMPLETION" 96))
(def save?        (not= "0" (env "SAVE" "1")))
(def serve-check? (not= "0" (env "SERVE_CHECK" "1")))
(def date-tag     (-> (.toISOString (js/Date.)) (subs 0 10) (str/replace "-" "")))
(def ckpt-out     (env "CKPT_OUT"
                       (home "genmlx-checkpoints"
                             (str "session-grpo-" (path/basename model-dir)
                                  "-" date-tag))))
(def metrics-out  (env "METRICS_OUT"
                       (home "genmlx-battery-logs"
                             (str "grpo-sessions-" date-tag ".jsonl"))))

(def grpo-cfg
  (merge {:learning-rate (env-num "LR" 2e-6)
          :temperature (env-num "TEMPERATURE" 0.9)
          :gradient-clip-norm 0.5
          :kl-coef 0.0
          :loss-type :grpo
          :enable-thinking false
          :lm-head-chunk-size 2
          :forward-chunk-size 4
          :group-size group-size
          :max-completion-length max-completion}
         (when-let [s (env "SEED")] {:seed (js/parseInt s)})))

(defn- session-files
  "Leaf dir of *.jsonl, or one level of subdirs of such files."
  [root]
  (let [names  (vec (.readdirSync fs root))
        direct (filterv #(str/ends-with? % ".jsonl") names)]
    (if (seq direct)
      (mapv #(path/join root %) (sort direct))
      (->> names
           (map #(path/join root %))
           (filter #(.isDirectory (.statSync fs %)))
           (mapcat (fn [d] (->> (.readdirSync fs d)
                                (filter (fn [f] (str/ends-with? f ".jsonl")))
                                (map (fn [f] (path/join d f))))))
           sort
           vec))))

(def aux-files ["config.json" "tokenizer.json" "tokenizer_config.json"
                "special_tokens_map.json" "vocab.json" "merges.txt"
                "generation_config.json" "added_tokens.json" "chat_template.jinja"])

(defn- copy-aux! [src dst]
  (doseq [f aux-files
          :let [s (path/join src f)]
          :when (.existsSync fs s)]
    (.copyFileSync fs s (path/join dst f))))

(defn- patch-quant-config!
  "The training engine dequantizes a quantized checkpoint at init (x76x)
   and saveModel writes DENSE weights, so the copied config.json must
   drop its quantization stanza or the reload misparses the artifact."
  [dir]
  (let [cfg-path (path/join dir "config.json")]
    (when (.existsSync fs cfg-path)
      (let [cfg (js/JSON.parse (.readFileSync fs cfg-path "utf8"))]
        (when (or (.-quantization cfg) (.-quantization_config cfg))
          (js-delete cfg "quantization")
          (js-delete cfg "quantization_config")
          (.writeFileSync fs cfg-path (js/JSON.stringify cfg nil 2))
          (println "  patched config.json: dropped quantization stanza"
                   "(trained save is dense)"))))))

(defn- metric-line! [m]
  (.mkdirSync fs (path/dirname metrics-out) #js {:recursive true})
  (fs/appendFileSync metrics-out (str (js/JSON.stringify (clj->js m)) "\n")))

(defn- wait-for-save!
  "The native saveModel promise can resolve before its detached
   weights.safetensors write lands on disk (observed on Thor) — poll until
   the file exists with a size stable across a 1s beat."
  []
  (let [f (path/join ckpt-out "weights.safetensors")]
    (p/loop [i 0, prev -1]
      (let [sz (if (.existsSync fs f) (.-size (.statSync fs f)) -2)]
        (cond
          (and (pos? sz) (= sz prev)) true
          (> i 180) (throw (ex-info "weights.safetensors never stabilized"
                                    {:genmlx/error :save-timeout}))
          :else (p/let [_ (p/delay 1000)] (p/recur (inc i) sz)))))))

(defn- keyset-parity!
  "The trained checkpoint (engine layout, remapped by the owned loader)
   must expose EXACTLY the base checkpoint's dequantized weight names —
   a wrong remap surfaces here by name, not as garbage generation."
  []
  (let [base-keys (set (keys (q3f/load-weights model-dir)))
        ckpt-keys (set (keys (q3f/load-weights ckpt-out)))
        missing   (sort (cset/difference base-keys ckpt-keys))
        extra     (sort (cset/difference ckpt-keys base-keys))]
    (println (str "  keyset parity: base " (count base-keys)
                  " / trained " (count ckpt-keys)
                  "  missing " (count missing) "  extra " (count extra)))
    (doseq [k (take 6 missing)] (println "    missing:" k))
    (doseq [k (take 6 extra)]   (println "    extra:  " k))
    (when (or (seq missing) (seq extra))
      (throw (ex-info "trained checkpoint keyset diverges from the base"
                      {:genmlx/error :keyset-mismatch
                       :missing (count missing) :extra (count extra)})))))

(defn- serve-check!
  "Load the SAVED checkpoint through the owned forward and decode a short
   greedy reply — the trained artifact serves."
  []
  (p/let [mm  (llm/load-model ckpt-out {:cljs-forward? true})
          ids (.applyChatTemplate (:tokenizer mm)
                                  (ps/messages->js [{:role "user" :content "Say ok."}])
                                  true js/undefined false)]
    (let [{:keys [model tokenizer]} mm
          eos (llm/eos-token-id tokenizer)
          ids (vec (js/Array.from ids))]
      (llm/init-cache! model)
      (let [gen (try
                  (loop [logits (llm/forward-prefill model ids), out []]
                    (let [tok (mx/item (mx/argmax logits))]
                      (cond
                        (== tok eos)         out
                        (>= (count out) 12)  out
                        :else (recur (llm/forward-step model tok)
                                     (conj out tok)))))
                  (finally (llm/reset-cache! model)))]
        (p/let [text (llm/decode tokenizer
                                 (js/Uint32Array.from (into-array gen)))]
          (println "  serve check reply:" (pr-str text))
          (when-not (seq gen)
            (throw (ex-info "serve check generated nothing"
                            {:genmlx/error :serve-check-empty})))
          text)))))

(println (str "### GRPO on sessions  model=" (path/basename model-dir)))
(println (str "  sessions: " sessions-dir))
(println (str "  reward=" reward-spec "  mode=" (name mode)
              "  steps=" steps "  group=" group-size
              "  max-completion=" max-completion))
(println (str "  metrics -> " metrics-out))
(when save? (println (str "  ckpt -> " ckpt-out)))

(def gcore (js/require "@genmlx/core"))

(->
 (p/let [files (p/resolved (session-files sessions-dir))
         conv  (p/resolved
                (sg/sessions->prompts files
                                      {:mode mode
                                       :skip-images? true
                                       :on-error :skip
                                       :system-prompt (env "SYSTEM_PROMPT")}))
         _ (p/resolved
            (do (println (str "  " (count files) " session files -> "
                              (count (:prompts conv)) " prompts ("
                              (:skipped conv) " image points skipped, "
                              (count (:failed conv)) " files unusable)"))
                (when (empty? (:prompts conv))
                  (throw (ex-info "no usable prompts from the sessions dir"
                                  {:genmlx/error :no-prompts})))))
         toolset   (p/resolved (sr/observed-toolset (:points conv)))
         reward-fn (sr/resolve-reward reward-spec
                                      {:points (:points conv)
                                       :toolset toolset
                                       :opts {}})
         model     (.load (.-Qwen35Model gcore) model-dir)
         _         (p/resolved (println "  policy model loaded; training ..."))
         history
         (train/with-trainer model grpo-cfg
           (fn [trainer]
             (p/let [hist
                     (p/loop [i 0, hist []]
                       (if (= i steps)
                         hist
                         (let [pidx   (mod i (count (:prompts conv)))
                               prompt (nth (:prompts conv) pidx)
                               prov   (sg/prompt-meta prompt)]
                           (p/let [r (train/train-step! trainer [prompt] reward-fn)]
                             (println (str "    step " i
                                           "  session=" (:session-id prov)
                                           "/" (:turn-index prov)
                                           "  reward-mean=" (.toFixed (:reward-mean r) 3)
                                           "  loss=" (.toFixed (:loss r) 4)
                                           "  applied=" (:gradients-applied? r)))
                             (metric-line!
                              {:step i :prompt-index pidx
                               :session-id (:session-id prov)
                               :turn-index (:turn-index prov)
                               :reward-mean (:reward-mean r)
                               :reward-std (:reward-std r)
                               :loss (:loss r)
                               :gradients-applied (:gradients-applied? r)
                               :rewards (:rewards r)
                               :total-tokens (:total-tokens r)
                               :generation-ms (:generation-ms r)
                               :training-ms (:training-ms r)})
                             (p/recur (inc i) (conj hist r))))))]
               ;; optimizer moments must be saved INSIDE the trainer scope
               (when (and save? (every? :gradients-applied? hist))
                 (.mkdirSync fs ckpt-out #js {:recursive true})
                 (train/save-optimizer-state!
                  trainer (path/join ckpt-out "optimizer_state.safetensors")))
               hist)))]
   (let [applied (count (filter :gradients-applied? history))
         all-rs  (mapcat :rewards history)]
     (println (str "  done: " (count history) " steps, " applied " applied, "
                   "reward-mean overall "
                   (.toFixed (/ (reduce + 0.0 all-rs) (max 1 (count all-rs))) 3)))
     (when-not (every? number? all-rs)
       (throw (ex-info "non-numeric reward escaped the seam"
                       {:genmlx/error :bad-reward})))
     (if (< applied (count history))
       ;; genmlx-li1p: a silent skip means NaN grads / garbage ratios —
       ;; fail loudly and do NOT save the tainted artifact.
       (do (println (str "FAIL: only " applied "/" (count history)
                         " steps applied gradients (genmlx-li1p) — not saving"))
           (set! (.-exitCode js/process) 1))
       (p/let [_ (when save?
                   (let [stale (path/join ckpt-out "weights.safetensors")]
                     ;; a pre-existing weights file would satisfy the
                     ;; stability poll before the NEW write starts
                     (when (.existsSync fs stale) (.unlinkSync fs stale)))
                   (p/let [_ (.saveModel model ckpt-out)
                           _ (wait-for-save!)]
                     (copy-aux! model-dir ckpt-out)
                     (patch-quant-config! ckpt-out)
                     (println "  saved:" ckpt-out)))
               _ (when (and save? serve-check?)
                   (p/let [_ (p/resolved (keyset-parity!))]
                     (serve-check!)))]
         (println "OK")))))
 (p/catch (fn [e]
            (println "ERROR:" (or (ex-message e) (str e)))
            (when-let [d (ex-data e)]
              (println "  " (pr-str (dissoc d :sci.impl/callstack))))
            (set! (.-exitCode js/process) 1))))
