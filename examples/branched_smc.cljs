;; Branch-per-particle SMC for grammar-constrained program synthesis on the RESIDENT
;; 80B (Qwen3-Coder-Next), demonstrating the "second path": N concurrent program
;; hypotheses forked from ONE shared prefill, each extended under a grammar via the
;; native forward-branch, and multinomial-RESAMPLED (branch-from the survivors,
;; dispose the rest) toward a tempered target. Resource-rational program synthesis,
;; the Lisp-machine way, on one Thor — built on genmlx.llm.backend's Tier-2 branch
;; surface (the same primitive genmlx.llm.branched uses for GFI regenerate/MCMC).
;;
;;   export GENMLX_MOE_MODEL=/path/to/Qwen3-Coder-Next-4bit/snapshots/<hash>
;;   bunx --bun nbb@1.4.208 -cp src:examples:malli/src:instaparse/src examples/branched_smc.cljs
;;
;; Thor/CUDA-only + heavy (loads the 42GB MoE); skips cleanly if the model is absent.
(ns branched-smc
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.mlx.random :as rng]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.grammar :as gram]
            [promesa.core :as pr]
            ["fs" :as fs]))

(def model-dir
  (or (some-> js/process .-env .-GENMLX_MOE_MODEL)
      "/home/robert/code/mlx/models/Qwen3-Coder-Next-4bit/snapshots/7b9321eabb85ce79625cac3f61ea691e4ea984b5"))

(def N 12)      ; particles
(def T 18)      ; tokens
(def R 6)       ; resample every R steps
(def temp 0.55) ; target temperature (<1 -> resampling concentrates on confident paths)

(defn- mat [a] (mx/materialize! a) a)
(defn- now [] (js/Date.now))
(defn- amb [] (try (/ (mx/get-active-memory) 1048576.0) (catch :default _ -1)))
(defn- f1 [x] (js/Number (.toFixed x 1)))
(defn- lp [logits tok] (mx/item (dc/dist-log-prob (dist/categorical logits) tok)))
(defn- gmask [c dfa logits] (gram/apply-mask c dfa logits))
(defn- gadv  [c dfa tok] (if (not= tok (:eos-id c))
                           (gram/dfa-advance-string (:dfa c) dfa (nth (:token-index c) tok "")) dfa))
(defn- ptext [c p] (->> (:toks p) (map #(nth (:token-index c) % "")) (apply str)))

(if-not (.existsSync fs model-dir)
  (do (println "SKIP branched_smc — model dir not found:" model-dir) (js/process.exit 0))
  (pr/let [{:keys [model tokenizer]} (llm/load-model model-dir)
           constraint (gram/compile-constraint tokenizer "[a-z0-9 ()]+")
           enc (llm/encode tokenizer "; one line of clojure to add two numbers\n")
           prompt (vec enc)]
    (println (str "== branch-per-particle SMC on the resident 80B: N=" N " particles, T=" T
                  " tokens, resample/" R ", target-temp=" temp))
    (llm/init-cache! model)
    (let [l0 (mat (llm/forward-prefill model prompt))
          start (:start (:dfa constraint))
          base-mem (amb), t0 (now)
          init (vec (for [i (range N)]                   ; ONE prefill -> N particle branches
                      {:branch (llm/branch-cache! model) :toks [] :dfa start
                       :logits l0 :logw 0.0 :key (rng/fresh-key (+ 1 i)) :anc i}))
          peak (atom (amb)), genealogy (atom [])]
      (loop [t 0, ps init]
        (if (>= t T)
          (let [ranked (reverse (sort-by :logw ps))]
            (println (str "\n-- final population (" N " particles, " (f1 (- (now) t0)) "ms) --"))
            (doseq [p (take N ranked)]
              (println (str "  logw=" (.toFixed (:logw p) 2) " <anc" (:anc p) "> " (pr-str (ptext constraint p)))))
            (println (str "\n  resample genealogy (distinct survivors per resample): " @genealogy))
            (println (str "  peak mem +" (f1 (- @peak base-mem)) " MB for " N " particles (bounded)"))
            (println (str "  WINNER: " (pr-str (ptext constraint (first ranked)))))
            (doseq [p ps] (try (llm/dispose-branch! model (:branch p)) (catch :default _ nil)))
            (llm/reset-cache! model)
            (println "\n== SMC complete — N concurrent grammar-constrained hypotheses on the 80B via branching ==")
            (js/process.exit 0))
          (let [ps' (mapv (fn [p]
                            (let [[k1 k2] (rng/split (:key p))
                                  masked (gmask constraint (:dfa p) (:logits p))
                                  tok (mx/item (dc/dist-sample (dist/categorical masked) k2))
                                  w (lp masked tok)
                                  nl (mat (llm/forward-branch model (:branch p) tok))]
                              (assoc p :toks (conj (:toks p) tok) :dfa (gadv constraint (:dfa p) tok)
                                       :logits nl :key k1
                                       :logw (+ (:logw p) (* (- (/ 1.0 temp) 1.0) w)))))
                          ps)
                _ (reset! peak (max @peak (amb)))
                ps'' (if (and (pos? t) (zero? (mod t R)))     ; multinomial resample over tempered weights
                       (let [idx-logits (mx/array (mapv :logw ps'))
                             ancs (vec (for [i (range N)]
                                         (mx/item (dc/dist-sample (dist/categorical idx-logits)
                                                                  (rng/fresh-key (+ (* 1000 t) i))))))
                             newps (mapv (fn [a] (let [src (nth ps' a)]
                                                   (assoc src :branch (llm/branch-from model (:branch src)) :logw 0.0)))
                                         ancs)]
                         (swap! genealogy conj (count (distinct ancs)))
                         (doseq [p ps'] (try (llm/dispose-branch! model (:branch p)) (catch :default _ nil)))
                         newps)
                       ps')]
            (recur (inc t) ps'')))))))
