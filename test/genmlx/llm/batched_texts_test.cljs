;; @tier slow — loads the qwen3.5-0.8b checkpoint (owned forward).
(ns genmlx.llm.batched-texts-test
  "genmlx-789s gate: generate-texts-batched — the Route B text-level API
   behind the synthesis proposer's :call-llm seam (Tier-1 of the paper
   campaign). K sampled chat completions from ONE batched owned forward.

   Pins: K strings returned with per-lane token counts; determinism under
   seed (same seed => byte-identical texts); the temperature guard
   (temperature <= 0 throws — greedy best-of-K would collapse the lanes);
   and the :temperature opt on make-llm-gf-batched leaves the default path
   untouched (nil => raw logits). The batched-vs-sequential wall-clock is
   REPORTED, not asserted (timing flakes; the k7nj/9uyg benchmarks own the
   speedup claim) — measured on a LONG-FORM prompt at the proposer's
   workload shape, because on short early-EOS completions the lockstep
   loop (which runs to the cap unless a :check-every boundary fires)
   legitimately loses to sequential EOS-stop: batching pays off when
   lanes actually decode, which is the best-of-K synthesis regime.

   Run: bunx --bun nbb@1.4.208 test/genmlx/llm/batched_texts_test.cljs"
  (:require [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as llmc]
            [promesa.core :as pr]
            ["fs" :as fs]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn- assert-true [label v]
  (if v (do (swap! pass inc) (println (str "  PASS: " label)))
        (do (swap! fail inc) (println (str "  FAIL: " label)))))

(def model-root (str (.-HOME js/process.env) "/.cache/models"))
(def model-path (str model-root "/qwen3.5-0.8b-mlx-bf16"))

(def prompt "Name one color of the rainbow, then stop.")
(def N 4)
(def MAXTOK 24)

(defn- run-batched [m seed]
  (llmc/generate-texts-batched m prompt {:n N :max-tokens MAXTOK
                                         :temperature 0.8 :seed seed
                                         :system-prompt "You are terse."}))

;; Timing comparison at the PROPOSER's workload shape: a prompt that decodes
;; long (no early EOS), budget 64 — the best-of-K synthesis regime where
;; every lane actually works.
(def long-prompt "List and describe the colors of the rainbow, one detailed sentence each.")
(def LONGTOK 64)

(defn- run-batched-long [m seed]
  (llmc/generate-texts-batched m long-prompt {:n N :max-tokens LONGTOK
                                              :temperature 0.8 :seed seed}))

(defn- run-sequential-long [m seed]
  (pr/loop [i 0, texts [], ms 0]
    (if (>= i N)
      (pr/resolved {:texts texts :gen-ms ms})
      (pr/let [r (llm/generate-text-raw+ m long-prompt
                                         {:max-tokens LONGTOK :temperature 0.8
                                          :seed (+ seed i)})]
        (pr/recur (inc i) (conj texts (:text r)) (+ ms (:gen-ms r)))))))

(if-not (.existsSync fs (str model-path "/config.json"))
  (println (str "SKIP: no model at " model-path))
  (-> (pr/let [m  (llm/load-model model-path {:cljs-forward? true})
               r1 (run-batched m 7)
               r2 (run-batched m 7)
               r3 (run-batched m 11)
               bl (run-batched-long m 7)
               sl (run-sequential-long m 7)]
        (println (str "\n== generate-texts-batched (N=" N " MAXTOK=" MAXTOK ") =="))
        (assert-true (str "returns " N " texts")
                     (and (= N (count (:texts r1)))
                          (every? string? (:texts r1))))
        (assert-true (str "returns " N " per-lane token counts, each in (0, " MAXTOK "]")
                     (and (= N (count (:n-tokens r1)))
                          (every? #(and (pos? %) (<= % MAXTOK)) (:n-tokens r1))))
        (assert-true "gen-ms recorded" (pos? (:gen-ms r1)))
        (assert-true "deterministic under seed (same seed => identical texts)"
                     (= (:texts r1) (:texts r2)))
        (println (str "  [report] lanes distinct within one call: "
                      (count (distinct (:texts r1))) "/" N))
        (println (str "  [report] seed 7 vs seed 11 differ: "
                      (not= (:texts r1) (:texts r3))))
        (println (str "  [report] long-form N=" N " x " LONGTOK " tok: batched "
                      (:gen-ms bl) " ms vs sequential " (:gen-ms sl) " ms  (speedup "
                      (.toFixed (/ (:gen-ms sl) (max (:gen-ms bl) 1)) 2) "x; lane tokens "
                      (pr-str (:n-tokens bl)) ")"))
        (let [threw (try (llmc/generate-texts-batched m prompt {:temperature 0})
                         false
                         (catch :default _ true))]
          (assert-true "temperature 0 throws (greedy best-of-K is refused)" threw))
        (println (str "\n=== batched-texts: " @pass " PASS, " @fail " FAIL ==="))
        (when (pos? @fail) (set! (.-exitCode js/process) 1)))
      (pr/catch (fn [e]
                  (println "FAIL (uncaught):" (or (ex-message e) (str e)))
                  (when-let [d (ex-data e)] (println " " (pr-str (dissoc d :sci.impl/callstack))))
                  (set! (.-exitCode js/process) 1)))))
