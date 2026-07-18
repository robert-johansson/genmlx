;; @tier fast
(ns genmlx.session-reward-plugin-test
  "Contract + discrimination tests for the heuristic-oracle reward plugin
   (genmlx-mvyp), loaded through the REAL resolve-reward seam exactly as
   grpo_sessions.cljs does (REWARD=scripts/rewards/heuristic_oracle_reward.cljs).

   Pins: seam contract shape, finiteness, floor on malformed, determinism,
   the discrimination ladder (well-formed declared call > declared call with
   undeclared params > undeclared call >= prose > malformed), code-parse
   grading, and non-degenerate spread across completion variants — the
   property match-administered lacks on self-imitating regenerations."
  (:require [genmlx.world.session-reward :as sr]
            [promesa.core :as p]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn- assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

(def toolset
  [{:name "get_weather" :params [{:name "location"}]}])

(def good
  (str "I will check.\n<tool_call>\n<function=get_weather>\n"
       "<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>"))
(def bad-param
  (str "<tool_call>\n<function=get_weather>\n"
       "<parameter=zip>\n75001\n</parameter>\n</function>\n</tool_call>"))
(def undeclared
  (str "<tool_call>\n<function=rm_rf>\n"
       "<parameter=path>\n/\n</parameter>\n</function>\n</tool_call>"))
(def malformed "<tool_call>\n<function=get_weather>")
(def prose "The weather in Paris is sunny.")
(def code-good (str prose "\n```clojure\n(+ 1 2)\n```"))
(def code-bad (str prose "\n```clojure\n(+ 1 2\n```"))

(defn- finite? [x] (and (number? x) (js/isFinite x)))

(-> (p/let [reward (sr/resolve-reward "scripts/rewards/heuristic_oracle_reward.cljs"
                                      {:points [] :toolset toolset :opts {}})]
      (assert-true "seam returns a reward-fn" (fn? reward))
      (let [r-good (reward "p" good)
            r-badp (reward "p" bad-param)
            r-und  (reward "p" undeclared)
            r-mal  (reward "p" malformed)
            r-pro  (reward "p" prose)
            r-cg   (reward "p" code-good)
            r-cb   (reward "p" code-bad)
            all    [r-good r-badp r-und r-mal r-pro r-cg r-cb]]
        (assert-true "every reward is finite (GRPO-poison guard)"
                     (every? finite? all))
        (assert-true "malformed block scores exactly the floor"
                     (= -1.0 r-mal))
        (assert-true "deterministic per completion"
                     (= r-good (reward "other-prompt" good)))
        (assert-true "ladder: well-formed declared call is best"
                     (> r-good r-badp))
        (assert-true "ladder: declared call w/ bad params > undeclared call"
                     (> r-badp r-und))
        (assert-true "ladder: undeclared call >= prose"
                     (>= r-und r-pro))
        (assert-true "ladder: everything beats malformed"
                     (every? #(> % r-mal) [r-good r-badp r-und r-pro r-cg r-cb]))
        (assert-true "code grading: parseable code > prose > broken code"
                     (and (> r-cg r-pro) (> r-pro r-cb)))
        (assert-true "non-degenerate spread (>=5 distinct values across variants)"
                     (>= (count (distinct all)) 5))))
    (p/then
     (fn [_]
       ;; empty-toolset behavior: tool use is credited, not penalized
       (p/let [reward (sr/resolve-reward "scripts/rewards/heuristic_oracle_reward.cljs"
                                         {:points [] :toolset [] :opts {}})]
         (assert-true "empty toolset: a parse-clean call is not penalized vs prose"
                      (> (reward "p" good) (reward "p" prose)))
         (println (str "\n== session-reward-plugin: " @pass " passed, " @fail " failed =="))
         (when (pos? @fail) (set! (.-exitCode js/process) 1)))))
    (p/catch (fn [e]
               (println "  FAIL (uncaught)" (or (ex-message e) (str e)))
               (set! (.-exitCode js/process) 1))))
