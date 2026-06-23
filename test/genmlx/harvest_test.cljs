;; @tier slow
(ns genmlx.harvest-test
  "Acceptance for genmlx.world.harvest — the RUN side of the Phase-3 REPL-trace harvest
   (genmlx-oexl). NATIVE-FREE + deterministic: a pure structured proposer (no LLM, no model
   load) drives the greedy driver, harvest-task/harvest-tasks collect the trajectories in
   build-corpus's shape, and the shared loop-proposer unions the σ-grid refiner onto the
   (here mocked-empty) LLM half. Proves the orchestration the scaled 35B harvest reuses,
   without any model.

   Run: bun run --bun nbb test/genmlx/harvest_test.cljs"
  (:require [genmlx.world.harvest :as h]
            [genmlx.world.repl-corpus :as rc]
            [genmlx.world.synth :as syn]
            [clojure.string :as str]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v (do (swap! pass inc) (println "  PASS" label))
      (do (swap! fail inc) (println "  FAIL" label))))

;; Tightly-clustered data near 5 -> tightening the shared σ strictly raises exact evidence.
(defn- crude [obs]
  (syn/spec [(syn/latent 'mu "gaussian" [5 3])]
            (for [k (keys obs)] (syn/obs k "gaussian" ['mu 3.0]))))

;; A PURE native-free proposer: from the current shared σ, offer the NEXT-LOWER rung on a
;; descending ladder. Each accepted move tightens toward the data scale, so greedy
;; synthesize accepts a multi-step trajectory deterministically (the LLM's stand-in here).
(defn- ladder-proposer [spec _fb]
  (let [cur    (last (:args (first (:obs spec))))
        ladder [2.0 1.0 0.5 0.2]
        nxt    (first (filter #(< % cur) ladder))]
    (when nxt
      [{:edit :set-noise :desc (str "σ -> " nxt)
        :spec' (reduce #(syn/set-noise %1 %2 nxt) spec (map :addr (:obs spec)))}])))

;; ---------------------------------------------------------------------------
(println "\n-- harvest-task: greedy multi-step run in build-corpus shape --")
(def task {:id :t1 :task-desc "Fit the y observations."
           :observations {:y0 5.0 :y1 5.2 :y2 4.8} :solve-bar -5.0})
(def run (h/harvest-task task {:propose ladder-proposer
                               :init-spec (crude (:observations task)) :max-steps 6}))
(assert-true "trajectory is multi-step (init + >1 accepted edit)" (> (:steps run) 1))
(assert-true "final evidence is finite" (and (:final run) (js/isFinite (:final run))))
(assert-true ":task carries id/task-desc/observations build-corpus needs"
             (= #{:id :task-desc :observations} (set (keys (:task run)))))
(assert-true "every trajectory step carries :code (what build-corpus reads)"
             (every? :code (:trajectory run)))
(assert-true ":solved? reflects final >= solve-bar" (= (>= (:final run) -5.0) (:solved? run)))
(assert-true "harvest-task throws without :propose"
             (try (h/harvest-task task {:init-spec (crude (:observations task))}) false
                  (catch :default _ true)))

;; ---------------------------------------------------------------------------
(println "\n-- harvest-tasks: map + on-run streaming + leakage-safe build-corpus --")
(def task2 {:id :held :task-desc "Other y observations."
            :observations {:y0 5.1 :y1 4.9 :y2 5.0} :solve-bar -5.0})
(def seen (atom []))
(def runs (h/harvest-tasks [task task2] (constantly ladder-proposer)
                           {:init-spec-for #(crude (:observations %))
                            :on-run (fn [r idx _t] (swap! seen conj [idx (:steps r)]))}))
(assert-true "harvest-tasks returns one run per task" (= 2 (count runs)))
(assert-true "on-run fired once per task, in order" (= [0 1] (mapv first @seen)))
(assert-true "harvest-tasks throws without :init-spec-for"
             (try (h/harvest-tasks [task] (constantly ladder-proposer) {}) false
                  (catch :default _ true)))

(def corpus (rc/build-corpus runs {:eval-ids #{"held"}}))
(assert-true "build-corpus harvests rows from the harvest-tasks runs" (pos? (:n-rows corpus)))
(assert-true "train-rows EXCLUDE the held-out task (no leakage)"
             (every? #(not= "held" (:task-id %)) (:train-rows corpus)))
(assert-true "the held-out task's rows are reported as dropped, not folded in"
             (and (pos? (count (:dropped-eval corpus)))
                  (every? #(= "held" (:task-id %)) (:dropped-eval corpus))))
(def row0 (first (:train-rows corpus)))
(assert-true "a harvested row is a system/user/assistant chat row"
             (= ["system" "user" "assistant"] (map :role (:messages row0))))
(assert-true "the assistant turn is a fenced clojure form"
             (str/includes? (:content (nth (:messages row0) 2)) "```clojure"))

;; ---------------------------------------------------------------------------
(println "\n-- loop-proposer: the shared LLM ∪ σ-grid union wiring (mocked-empty LLM) --")
(def obs (:observations task))
(def crude-code (syn/render (crude obs)))
(def fb (syn/check crude-code obs {:n-particles 2000}))
(def mock-llm (fn [_req] {:completions []}))    ;; LLM contributes nothing this step
(def prop (h/loop-proposer {:call-llm mock-llm :task-desc "Fit y." :observations obs :revise 0}))
(def cands (prop (crude obs) fb))
(assert-true "loop-proposer still yields the σ-grid candidates with an empty LLM half"
             (pos? (count cands)))
(assert-true "every union candidate carries a :spec' (valid loop state)" (every? :spec' cands))
(assert-true "the candidates are the shared-σ refinements (set-noise edits)"
             (every? #(= :set-noise (:edit %)) cands))

(println (str "\n==== harvest_test: " @pass " passed, " @fail " failed ===="))
(when (pos? @fail) (js/process.exit 1))
