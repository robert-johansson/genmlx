(ns bench.owned-branch
  "genmlx-7f93 gate 4: the free-branching claim on the OWNED forward,
   MEASURED (spec §8: measure before claiming). Three numbers:

     1. fork cost — branch-cache!/branch-from on the owned path is a
        persistent-map assoc of an immutable cache VALUE (no copy), measured
        in µs/fork
     2. per-step decode cost at growing prefix length T — forward-branch
        (O(1) in T) vs the replay fallback (re-forwards the whole prefix,
        O(T)/step), the R3 asymmetry as a ratio
     3. filter-level wall-time — token-SMC (N particles) on the branch
        decoder vs the replay decoder, same key

   Default model: Ornith-1.0-35B-4bit (owned qwen3_5_moe) when cached, else
   the dense 0.6b; GENMLX_OWNED_BENCH_MODEL overrides.
   Output: results/owned_branch/data.json
   Usage: bunx --bun nbb@1.4.208 bench/owned_branch.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.smc :as tsmc]
            [promesa.core :as pr]
            ["fs" :as fs]
            ["os" :as os]
            ["path" :as path]))

(def model-dir
  (or (some-> js/process .-env .-GENMLX_OWNED_BENCH_MODEL)
      (let [base (str (os/homedir)
                      "/.cache/huggingface/hub/models--mlx-community--Ornith-1.0-35B-4bit/snapshots")]
        (when (.existsSync fs base)
          (str base "/" (first (js->clj (.readdirSync fs base))))))
      (first (filter #(.existsSync fs (path/join % "tokenizer.json"))
                     [(path/join (os/homedir) ".cache" "models" "qwen3-0.6b-mlx-bf16")
                      (path/join (os/homedir) ".cache" "models" "qwen3-0.6b")]))))

(def moe? (boolean (re-find #"35B|moe|Moe|MoE" (or model-dir ""))))
(def Ts (if moe? [16 64] [64 256]))       ; prefix lengths for part 2
(def K 4)                                 ; decode steps timed per arm

(defn- mat [a] (mx/materialize! a) a)
(defn- now [] (js/Date.now))

(defn- tile-prompt [ids n]
  (vec (take n (cycle ids))))

(defn- bench-decode
  "Per-step decode cost at prefix length T: branch vs replay (ms/step).
   force-gc! between steps keeps the CUDA pool trimmed — a T=64 replay
   forward on the 35B MoE builds a multi-GB transient graph per step, and
   letting those accumulate as dark driver pages is the genmlx-h3p5 OOM
   cascade (this bench triggered reboot #3 on 2026-07-10 before the guards)."
  [model base-ids T]
  (let [prompt (tile-prompt base-ids T)]
    (llm/init-cache! model)
    (let [t0 (now)
          l0 (mat (llm/forward-prefill model prompt))
          prefill-ms (- (now) t0)
          _  (mx/force-gc!)
          b  (llm/branch-cache! model)
          ;; branch arm: K O(1) steps on the branch
          [br-ms toks]
          (loop [i 0, lg l0, acc [], ms 0]
            (if (>= i K)
              [ms acc]
              (let [tok (mx/item (mx/argmax lg))
                    t1 (now)
                    lg' (mat (llm/forward-branch model b tok))]
                (recur (inc i) lg' (conj acc tok) (+ ms (- (now) t1))))))
          _ (llm/dispose-branch! model b)
          _ (mx/force-gc!)
          ;; replay arm: the SAME tokens, each step re-forwards the prefix.
          ;; GC time is excluded from the timed window (it runs after each
          ;; step's clock stops), so the ratio stays a pure decode-cost ratio.
          rp-ms
          (loop [i 0, ctx prompt, ms 0]
            (if (>= i K)
              ms
              (let [ctx' (conj ctx (nth toks i))
                    t1 (now)
                    _ (mat (llm/forward-pass model ctx'))
                    ms' (+ ms (- (now) t1))]
                (mx/force-gc!)
                (recur (inc i) ctx' ms'))))]
      (llm/reset-cache! model)
      (mx/force-gc!)
      {:T T :prefill-ms prefill-ms
       :branch-ms-per-step (/ br-ms K)
       :replay-ms-per-step (/ rp-ms K)
       :replay-over-branch (/ rp-ms (max br-ms 1))})))

(defn- bench-fork
  "µs per fork on a prefilled cache (the free-branching number)."
  [model base-ids]
  (llm/init-cache! model)
  (mat (llm/forward-prefill model (tile-prompt base-ids 32)))
  (let [root (llm/branch-cache! model)
        n 200
        t0 (now)
        ids (vec (repeatedly n #(llm/branch-from model root)))
        us (/ (* 1000.0 (- (now) t0)) n)]
    (doseq [b ids] (llm/dispose-branch! model b))
    (llm/dispose-branch! model root)
    (llm/reset-cache! model)
    (mx/force-gc!)
    {:forks n :us-per-fork us}))

(defn- bench-filter
  "token-SMC wall-time, branch decoder vs replay decoder, same key."
  [mm model prompt]
  (let [run! (fn [decoder]
               (let [t0 (now)
                     r (tsmc/token-smc {:particles 4 :max-tokens 8
                                        :eos-id (llm/eos-token-id (:tokenizer mm))
                                        :decoder decoder :key (rng/fresh-key 7)}
                                       mm prompt)]
                 {:secs (/ (- (now) t0) 1000.0)
                  :log-ml (mx/realize (:log-ml-estimate r))}))
        br (run! (tsmc/native-decoder model))
        _  (mx/force-gc!)
        rp (run! (tsmc/replay-decoder model))
        _  (mx/force-gc!)]
    {:particles 4 :max-tokens 8
     :branch-secs (:secs br) :replay-secs (:secs rp)
     :replay-over-branch (/ (:secs rp) (max (:secs br) 1e-9))}))

(if-not model-dir
  (println "SKIP owned-branch bench — no model available")
  (pr/let [mm (llm/load-model model-dir)
           {:keys [model tokenizer]} mm
           enc (llm/encode tokenizer "The quick brown fox jumps over the lazy dog. ")]
    (when-not (llm/supports-branching? model)
      (println "model does not support branching — nothing to measure")
      (js/process.exit 1))
    (mx/force-gc!)
    (let [base-ids (vec enc)
          _ (println (str "== owned-branch bench on " model-dir))
          fork (bench-fork model base-ids)
          _ (println (str "  fork: " (.toFixed (:us-per-fork fork) 1) " µs/fork ("
                          (:forks fork) " forks of a T=32 prefix)"))
          decode (mapv (fn [T]
                         (let [r (bench-decode model base-ids T)]
                           (println (str "  T=" T ": prefill " (:prefill-ms r) " ms | "
                                         "branch " (.toFixed (:branch-ms-per-step r) 0)
                                         " ms/step | replay " (.toFixed (:replay-ms-per-step r) 0)
                                         " ms/step | ratio "
                                         (.toFixed (:replay-over-branch r) 1) "x"))
                           (mx/force-gc!)
                           r))
                       Ts)
          filt (bench-filter mm model (tile-prompt base-ids 12))
          _ (println (str "  token-SMC N=4 T<=8: branch " (.toFixed (:branch-secs filt) 1)
                          "s | replay " (.toFixed (:replay-secs filt) 1) "s | ratio "
                          (.toFixed (:replay-over-branch filt) 1) "x"))
          out {:model model-dir :moe? moe?
               :fork fork :decode decode :filter filt}]
      (when-not (.existsSync fs "results/owned_branch")
        (.mkdirSync fs "results/owned_branch" #js {:recursive true}))
      (.writeFileSync fs "results/owned_branch/data.json"
                      (js/JSON.stringify (clj->js out) nil 2))
      (println "wrote results/owned_branch/data.json"))))
