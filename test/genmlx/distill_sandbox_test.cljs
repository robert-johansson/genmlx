;; @tier slow
(ns genmlx.distill-sandbox-test
  "Acceptance for genmlx.world.distill-sandbox — the process-isolation layer that makes
   the distillation filter robust to NON-TERMINATING / crashing untrusted teacher code
   (genmlx-8d15). It spawns REAL worker subprocesses (scripts/distill_check.cljs), so it
   is slow; run:
     bun run --bun nbb test/genmlx/distill_sandbox_test.cljs

   The canonical case: a batch of 3 counter-machine candidates where row 1 is an
   infinite loop `(fn [state action] (loop [] (recur)))`. The sandbox must process
   row 0, KILL the worker stuck on row 1 (recording :timeout), resume, and process
   row 2 — the whole batch completes and the corpus is never poisoned by the hanger."
  (:require [genmlx.world.distill-sandbox :as sb]
            [clojure.string :as str]
            [promesa.core :as p]))

(def fs   (js/require "fs"))
(def os   (js/require "os"))
(def path (js/require "path"))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v (do (swap! pass inc) (println "  PASS" label))
        (do (swap! fail inc) (println "  FAIL" label))))

(def tmp-dir (.join path (.tmpdir os) "genmlx-distill-sandbox-test"))
(when-not (.existsSync fs tmp-dir) (.mkdirSync fs tmp-dir #js {:recursive true}))

(def candidates
  [{:task_id "counter-machine" :sample_idx 0
    :raw_text "(fn [state action] (case action :inc (update state :count inc) :dec (update state :count dec) :reset (assoc state :count 0) state))"}
   {:task_id "counter-machine" :sample_idx 1
    :raw_text "(fn [state action] (loop [] (recur)))"}              ;; NON-TERMINATING
   {:task_id "counter-machine" :sample_idx 2
    :raw_text "(fn [state action] (update state :count inc))"}])    ;; wrong, but TERMINATES

(def cand-file (.join path tmp-dir "cands.jsonl"))
(.writeFileSync fs cand-file (str/join "\n" (map #(js/JSON.stringify (clj->js %)) candidates)))

(defn by-sample [verdicts s] (first (filter #(= s (:sample-idx %)) verdicts)))

(println "\n== sandbox over 3 candidates (row 1 is an infinite loop) ==")
(-> (sb/collect-verdicts cand-file
                         {:out-path   (.join path tmp-dir "verdicts.edn")
                          :eval-opts  {:n-particles 10}
                          :timeout-ms 8000 :poll-ms 300 :verbose? true})
    (p/then
      (fn [verdicts]
        (let [v0 (by-sample verdicts 0)
              v1 (by-sample verdicts 1)
              v2 (by-sample verdicts 2)]
          (assert-true "batch COMPLETED with a verdict for all 3 rows (no hang)" (= 3 (count verdicts)))
          (assert-true "row 0 (correct counter) is kept" (:kept? v0))
          (assert-true "row 1 (non-terminating) -> :timeout, not kept"
                       (and (some? v1) (= :timeout (:reason v1)) (not (:kept? v1))))
          (assert-true "row 1 timeout verdict keeps its provenance (task-id + sample-idx)"
                       (and (= "counter-machine" (:task-id v1)) (= 1 (:sample-idx v1))))
          (assert-true "row 2 (after the hang) still processed -> :test-fail, not kept"
                       (and (some? v2) (= :test-fail (:reason v2)) (not (:kept? v2)))))))
    (p/catch (fn [e] (swap! fail inc) (println "  FAIL (uncaught)" (.-message e))))
    (p/finally
      (fn [_ _]
        (println (str "\n== distill-sandbox (genmlx-8d15): " @pass " passed, " @fail " failed =="))
        (when (pos? @fail) (set! (.-exitCode js/process) 1)))))
