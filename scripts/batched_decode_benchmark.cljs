(ns batched-decode-benchmark
  "Speed/leak benchmark for the NATIVE batched decoder behind world.train/generate-batch
   (genmlx-d3yn — 'is the cheap proposer actually cheap?'). Replaces the never-committed
   Mac-only scripts/particle_benchmark.cljs with the two signals the bean names:

     1. CROSS-CALL GROWTH — CALLS sequential generate-batch calls on ONE engine, same
        prompts; per-call wall time should be flat. (Mac baseline 2026-06-22: 32.6 ->
        43.4 -> 54.3 s — native state accumulated across calls.)
     2. EOS EFFICIENCY — a prompt whose greedy-ish completion ends almost immediately
        vs one that runs long. Without EOS-stop both cost ~MAXTOK decode steps and the
        short/long time ratio is ~1; with per-sequence early stop, short << long.

   Pure inspection: no weight update (generateBatch, not train-step!). Reports
   s/call, s/particle, completion char stats, and the growth ratio last/first.

   Run (Thor: pin clocks first — sudo nvpmodel -m 0 && sudo jetson_clocks):
     bunx --bun nbb@1.4.208 scripts/batched_decode_benchmark.cljs
   Env: MODEL   checkpoint dir   (default ~/.cache/models/qwen3.5-0.8b-mlx-bf16)
        FAMILY  qwen35|qwen3     (default qwen35)
        K       group size       (default 8)
        MAXTOK  max completion   (default 128)
        CALLS   sequential calls (default 4)"
  (:require [genmlx.world.train :as train]
            [promesa.core :as p]))

(def os   (js/require "os"))
(def path (js/require "path"))
(def fs   (js/require "fs"))
(defn- env  [k d] (or (aget (.-env js/process) k) d))
(defn- envi [k d] (let [v (env k nil)] (if v (js/parseInt v 10) d)))

(def gcore (js/require "@genmlx/core"))

(def model-dir (env "MODEL" (.join path (.homedir os) ".cache" "models"
                                   "qwen3.5-0.8b-mlx-bf16")))
(def family  (keyword (env "FAMILY" "qwen35")))
(def K       (envi "K" 8))
(def maxtok  (envi "MAXTOK" 128))
(def calls   (envi "CALLS" 4))

;; The long prompt invites a completion that will not EOS before MAXTOK; the
;; short prompt should EOS within a few tokens. enable-thinking false keeps
;; Qwen3.5 from opening an unterminated <think> block on the short prompt.
(def long-prompt  "Explain Bayesian inference in detail, covering priors, likelihoods, posteriors, and a worked example.")
(def short-prompt "Reply with the single word OK and nothing else.")

(defn- now [] (js/Date.now))

(defn- run-calls
  "N sequential generate-batch calls; returns a promise of [ms ...] + completions of
   the last call."
  [tr prompts n]
  (p/loop [i 0 times [] last-comps nil]
    (if (>= i n)
      (p/resolved {:times times :completions last-comps})
      (let [t0 (now)]
        (p/let [comps (train/generate-batch tr prompts)]
          (p/recur (inc i) (conj times (- (now) t0)) comps))))))

(defn- fmt [x] (.toFixed (js/Number x) 1))

(defn- report [label {:keys [times completions]}]
  (let [n-seq (count completions)
        lens  (mapv count completions)
        s0    (/ (first times) 1000.0)
        sN    (/ (last times) 1000.0)]
    (println (str "\n-- " label " --"))
    (println (str "  per-call s: " (mapv #(js/Number (fmt (/ % 1000.0))) times)
                  "  growth last/first: " (fmt (/ sN (max s0 0.001)))))
    (println (str "  particles/call: " n-seq
                  "  s/particle (last call): " (fmt (/ sN (max n-seq 1)))))
    (println (str "  completion chars: min " (apply min lens)
                  " max " (apply max lens)
                  " mean " (fmt (/ (reduce + lens) (max (count lens) 1)))))
    {:s-per-call (mapv #(/ % 1000.0) times) :s-per-particle (/ sN (max n-seq 1))}))

(if-not (.existsSync fs (.join path model-dir "config.json"))
  (do (println "SKIP: no model at" model-dir) (js/process.exit 0))
  (p/let [_     (println (str "== batched-decode benchmark: " model-dir
                              " K=" K " MAXTOK=" maxtok " CALLS=" calls " =="))
          model (case family
                  :qwen35 (.load (.-Qwen35Model gcore) model-dir)
                  :qwen3  (.load (.-Qwen3Model gcore) model-dir))
          tr    (train/make-trainer! model
                                     {:group-size K :max-completion-length maxtok
                                      :seed 42 :enable-thinking false}
                                     {:family family})
          ;; warmup (JIT/PTX cache, first-touch allocations) — not measured
          _     (train/generate-batch tr [short-prompt])
          long-res  (run-calls tr [long-prompt] calls)
          short-res (run-calls tr [short-prompt] calls)
          l     (report (str "LONG prompt (should decode ~" maxtok " tokens/particle)") long-res)
          s     (report "SHORT prompt (should EOS in a few tokens)" short-res)
          ratio (/ (last (:s-per-call s)) (max (last (:s-per-call l)) 0.001))]
    (println (str "\n  EOS efficiency: short/long time ratio = " (fmt ratio)
                  "  (~1.0 = no EOS stop; << 1.0 = early stop works)"))
    (train/dispose! tr)
    (js/process.exit 0)))
