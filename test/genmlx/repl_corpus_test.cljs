;; @tier slow
(ns genmlx.repl-corpus-test
  "Acceptance for genmlx.world.repl-corpus — the REPL-trace SFT-corpus harvester
   (Phase 3, genmlx-oexl). NATIVE-FREE + deterministic (conjugate models score exact, no
   model loads): a successful trajectory becomes propose-eval-revise chat rows, the oracle
   filter drops non-improving transitions, and the leakage-safe task split holds.

   Run: bun run --bun nbb test/genmlx/repl_corpus_test.cljs"
  (:require [genmlx.world.repl-corpus :as rc]
            [genmlx.world.synth :as syn]
            [clojure.string :as str]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v (do (swap! pass inc) (println "  PASS" label))
      (do (swap! fail inc) (println "  FAIL" label))))

;; tightly-clustered data near 5 -> a tighter obs noise has higher exact evidence.
;; Models are RENDERED from specs (no hand-written code strings).
(def obs {:y0 5.0 :y1 5.2 :y2 4.8})
(defn- model [noise]
  (syn/render (syn/spec [(syn/latent 'mu "gaussian" [5 3])]
                        (for [k [:y0 :y1 :y2]] (syn/obs k "gaussian" ['mu noise])))))
(def crude (model 3))
(def tight (model 1))
(def task {:id :gtest :task-desc "Fit the y observations." :observations obs})

(println "\n-- trajectory->rows --")
(def rows1 (rc/trajectory->rows task [{:code crude} {:code tight}]))
(assert-true "an improving transition -> exactly one row" (= 1 (count rows1)))
(def r1 (first rows1))
(def msgs1 (:messages r1))
(def user1 (:content (second msgs1)))
(def asst1 (:content (nth msgs1 2)))
(assert-true "row carries the task-id (for the leakage split)" (= "gtest" (:task-id r1)))
(assert-true "row is a :repl-edit" (= :repl-edit (:kind r1)))
(assert-true "messages are system/user/assistant" (= ["system" "user" "assistant"] (map :role msgs1)))
(assert-true "user turn embeds the data" (str/includes? user1 ":y0 = 5"))
(assert-true "user turn embeds model_i (crude state)" (str/includes? user1 "dist/gaussian mu 3"))
(assert-true "user turn embeds the verifier feedback (evidence)" (str/includes? user1 "evidence"))
(assert-true "assistant turn is the ACCEPTED next model (fenced, tightened)"
             (and (str/includes? asst1 "dist/gaussian mu 1") (str/includes? asst1 "```clojure")))

(println "\n-- oracle filter --")
(assert-true "a NON-improving transition (same model) is dropped"
             (empty? (rc/trajectory->rows task [{:code crude} {:code crude}])))
(assert-true "a 3-step improving trajectory yields 2 rows"
             (= 2 (count (rc/trajectory->rows task [{:code (model 3)} {:code (model 2)} {:code (model 1)}]))))
(def broken "(fn [trace] (let [mu (dist/gaussian 0 5)] {:y0 (trace :y0 (dist/gaussian mu 1.0))}))")
(assert-true "a transition to a BROKEN model is dropped (target must score)"
             (empty? (rc/trajectory->rows task [{:code crude} {:code broken}])))

(println "\n-- build-corpus + leakage-safe split --")
(def runs [{:task task :trajectory [{:code crude} {:code tight}]}
           {:task {:id :held :task-desc "other" :observations obs} :trajectory [{:code crude} {:code tight}]}])
(def res (rc/build-corpus runs {:eval-ids #{"held"}}))
(assert-true "harvests rows from both runs" (= 2 (:n-rows res)))
(assert-true "per-task counts present" (= {"gtest" 1 "held" 1} (:per-task res)))
(assert-true "train-rows EXCLUDE the held-out task (no leakage)"
             (and (= 1 (count (:train-rows res))) (= "gtest" (:task-id (first (:train-rows res))))))
(assert-true "the held-out task's row is reported as dropped, not folded in"
             (= 1 (count (:dropped-eval res))))

(println (str "\n==== repl_corpus_test: " @pass " passed, " @fail " failed ===="))
(when (pos? @fail) (js/process.exit 1))
