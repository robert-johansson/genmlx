;; @tier medium
(ns genmlx.genmlx-tool-worker-test
  "genmlx-wdx0 (L5-A): the pi-tool child worker driven exactly as the
   extension drives it — a fresh subprocess per op, JSON on stdin, the
   GENMLX_RESULT marker line out. The ORACLE CHECK is the done-means
   heart: the worker's log-ML for a conjugate model equals a direct
   in-process score-model run (:exact — deterministic analytical
   evidence), so the bridge adds nothing and loses nothing.

   Run: bun run --bun nbb test/genmlx/genmlx_tool_worker_test.cljs"
  (:require [clojure.string :as str]
            [genmlx.llm.msa-score :as score]
            ["child_process" :as cp]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))
(defn assert-close [label expected actual tol]
  (let [d (js/Math.abs (- expected actual))]
    (if (<= d tol)
      (do (swap! pass inc) (println "  PASS" label (str "(|Δ| " (.toFixed d 6) ")")))
      (do (swap! fail inc)
          (println "  FAIL" label "expected" expected "actual" actual "tol" tol)))))

(def marker "GENMLX_RESULT:")

(defn- run-raw [input]
  (let [r (.spawnSync cp "bun"
                      #js ["run" "--bun" "nbb" "scripts/genmlx_tool_worker.cljs"]
                      #js {:input input :encoding "utf8" :timeout 300000})
        line (->> (str/split-lines (or (.-stdout r) ""))
                  (filter #(str/starts-with? % marker))
                  last)]
    (when line
      (js->clj (js/JSON.parse (subs line (count marker)))
               :keywordize-keys true))))

(defn- run-op [req] (run-raw (js/JSON.stringify (clj->js req))))

(def good-model
  "(fn [trace] (let [mu (trace :mu (dist/gaussian 0 3))] (trace :y0 (dist/gaussian mu 1)) (trace :y1 (dist/gaussian mu 1))))")
(def wrong-model
  "(fn [trace] (let [mu (trace :mu (dist/gaussian 100 0.1))] (trace :y0 (dist/gaussian mu 1)) (trace :y1 (dist/gaussian mu 1))))")
(def is-model
  "(fn [trace] (let [mu (trace :mu (dist/uniform 0 4))] (trace :y0 (dist/gaussian mu 1)) (trace :y1 (dist/gaussian mu 1))))")
(def garbage-model "(this is not clojure")
(def obs {:y0 2.0 :y1 1.7})

(println "\n-- eval-model --")
(let [r (run-op {:op "eval-model" :code good-model})]
  (assert-true "valid model evaluates"
               (and (:ok r) (:valid r)))
  (assert-true "schema carries the trace sites"
               (= [":mu" ":y0" ":y1"]
                  (mapv :addr (get-in r [:schema :traceSites]))))
  (assert-true "schema flags static + conjugate"
               (and (get-in r [:schema :static])
                    (get-in r [:schema :conjugate]))))
(let [r (run-op {:op "eval-model" :code garbage-model})]
  (assert-true "garbage code -> ok but invalid"
               (and (:ok r) (false? (:valid r)))))

(println "\n-- score-model + THE ORACLE CHECK --")
(let [r      (run-op {:op "score-model" :code good-model :observations obs})
      direct (score/score-model* (score/eval-model good-model) obs {})]
  (println "    worker:" (:logMl r) (:method r)
           "| direct:" (:log-ml direct) (:method direct))
  (assert-true "conjugate model scores finite via the exact method"
               (and (:ok r) (:valid r) (:finite r) (= "exact" (:method r))))
  (assert-close "ORACLE: worker log-ML == in-process score-model"
                (:log-ml direct) (:logMl r) 1e-3)
  (assert-true "methods agree" (= (name (:method direct)) (:method r))))
(let [r (run-op {:op "score-model" :code is-model :observations obs})]
  (assert-true "non-conjugate model falls back to IS, labeled honestly"
               (and (:ok r) (:finite r) (not= "exact" (:method r)))))
(let [r (run-op {:op "score-model" :code garbage-model :observations obs})]
  (assert-true "garbage code -> valid false, logMl null"
               (and (:ok r) (false? (:valid r)) (nil? (:logMl r))
                    (false? (:finite r)))))

(println "\n-- rank-models --")
(let [r (run-op {:op "rank-models"
                 :candidates [garbage-model wrong-model good-model]
                 :observations obs})
      ranking (:ranking r)]
  (println "    ranking:" (mapv (juxt :index :logMl :valid) ranking))
  (assert-true "three candidates ranked" (= 3 (count ranking)))
  (assert-true "the good model ranks first" (= 2 (:index (first ranking))))
  (assert-true "the wrong-prior model ranks second" (= 1 (:index (second ranking))))
  (assert-true "garbage ranks last as invalid"
               (and (= 0 (:index (last ranking)))
                    (false? (:valid (last ranking))))))

(println "\n-- protocol errors --")
(assert-true "malformed stdin -> ok false"
             (let [r (run-raw "{not json")]
               (and (false? (:ok r))
                    (str/includes? (str (:error r)) "malformed"))))
(assert-true "unknown op -> ok false naming the ops"
             (let [r (run-op {:op "explode"})]
               (and (false? (:ok r))
                    (str/includes? (str (:error r)) "score-model"))))
(assert-true "missing observations -> ok false"
             (false? (:ok (run-op {:op "score-model" :code good-model}))))

(println (str "\n== genmlx-tool-worker: " @pass " passed, " @fail " failed =="))
(when (pos? @fail) (set! (.-exitCode js/process) 1))
