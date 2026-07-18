;; @tier fast
(ns genmlx.world-session-reward-test
  "genmlx-qhy4 (L4): reward builders over real converter output — pure.
   Covers observed-toolset derivation, tool-format well-formedness cases,
   match-administered against the provenance seam (exact / partial /
   floor / call-free / meta-less), the weights knob, and the external
   plugin contract of resolve-reward.

   Run: bun run --bun nbb test/genmlx/world_session_reward_test.cljs"
  (:require [clojure.string :as str]
            [genmlx.world.session-grpo :as sg]
            [genmlx.world.session-reward :as sr]
            [promesa.core :as pr]
            ["fs" :as fs]
            ["os" :as os]
            ["path" :as path]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))
(defn assert-equal [label expected actual]
  (if (= expected actual)
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc)
        (println "  FAIL" label "\n    expected:" (pr-str expected)
                 "\n    actual:  " (pr-str actual)))))

(defn- jl [o] (js/JSON.stringify (clj->js o)))

(def fixture
  (str/join "\n"
            [(jl {:type "session" :version 3 :id "sess-r" :timestamp "t" :cwd "/w"})
             (jl {:type "message" :id "m1" :parentId nil :timestamp "t"
                  :message {:role "user" :content "start"}})
             (jl {:type "message" :id "m2" :parentId "m1" :timestamp "t"
                  :message {:role "assistant" :stopReason "toolUse"
                            :content [{:type "text" :text "looking"}
                                      {:type "toolCall" :id "c1" :name "look"
                                       :arguments {:dir "n"}}]}})
             (jl {:type "message" :id "m3" :parentId "m2" :timestamp "t"
                  :message {:role "toolResult" :toolCallId "c1" :toolName "look"
                            :isError false
                            :content [{:type "text" :text "a wall"}]}})
             (jl {:type "message" :id "m4" :parentId "m3" :timestamp "t"
                  :message {:role "user" :content "answer"}})
             (jl {:type "message" :id "m5" :parentId "m4" :timestamp "t"
                  :message {:role "assistant" :stopReason "stop"
                            :content [{:type "text" :text "north"}]}})]))

(def dir (fs/mkdtempSync (path/join (os/tmpdir) "session-reward-")))
(def sfile (path/join dir "s.jsonl"))
(fs/writeFileSync sfile fixture)

(def points (:points (sg/session-file->points sfile {:mode :all})))
(def tool-point (first points))      ; the m2 tool-call decision
(def terminal-point (second points)) ; the m5 call-free answer

(def look-block
  "<tool_call>\n<function=look>\n<parameter=dir>\nn\n</parameter>\n</function>\n</tool_call>")

(println "\n-- observed-toolset --")
(assert-equal "toolset derived from administered calls"
              [{:name "look" :params [{:name "dir"}]}]
              (sr/observed-toolset points))

(println "\n-- tool-format-reward --")
(let [toolset (sr/observed-toolset points)
      r (sr/tool-format-reward toolset)]
  (assert-equal "well-formed declared call -> 1.0" 1.0 (r nil look-block))
  (assert-equal "undeclared function -> floor" -1.0
                (r nil (str/replace look-block "function=look" "function=nuke")))
  (assert-equal "undeclared parameter -> floor" -1.0
                (r nil (str/replace look-block "parameter=dir" "parameter=zap")))
  (assert-equal "unclosed block -> floor" -1.0
                (r nil "<tool_call>\n<function=look>"))
  (assert-equal "call-free prose -> no-call default 0.0" 0.0 (r nil "hello"))
  (assert-equal "no-call-reward knob" 0.3
                ((sr/tool-format-reward toolset {:no-call-reward 0.3}) nil "hi")))

(println "\n-- match-administered-reward --")
(let [r (sr/match-administered-reward)
      p (:prompt tool-point)]
  (assert-equal "exact regen of the administered action -> 1.0"
                1.0 (r p (str "looking\n\n" look-block)))
  (assert-equal "right tool, wrong argument -> parse+names 0.6"
                0.6 (r p (str/replace look-block "\nn\n" "\ns\n")))
  (assert-equal "wrong tool name -> parse only 0.25"
                0.25 (r p (str/replace look-block "look" "grab")))
  (assert-equal "unparseable -> floor" -1.0 (r p "<tool_call>\nbroken"))
  (assert-equal "prompt without provenance -> floor" -1.0
                (r [{:role "user" :content "x"}] "anything"))
  (let [tp (:prompt terminal-point)]
    (assert-equal "call-free admin, call-free regen -> 1.0" 1.0 (r tp "south"))
    (assert-equal "call-free admin, regen emits a call -> parse only 0.25"
                  0.25 (r tp look-block)))
  ;; wrong arg value: parse (0.5) + names (0.2), args component withheld
  (assert-equal "weights knob"
                0.7 ((sr/match-administered-reward
                      {:weights {:parse 0.5 :names 0.2 :args 0.2}})
                     p (str/replace look-block "\nn\n" "\nw\n"))))

(println "\n-- resolve-reward + the external plugin contract --")
(def plugin (path/join dir "oracle_stub.cljs"))
(fs/writeFileSync plugin
                  "(fn [ctx]\n  ;; ctx = {:points :toolset :opts}\n  (let [n (count (:points ctx))]\n    (fn [_prompt completion]\n      (if (pos? (count completion)) (/ n 10.0) -1.0))))")
(def bad-plugin (path/join dir "bad_plugin.cljs"))
(fs/writeFileSync bad-plugin "42")

(defn- genmlx-error [e]
  (loop [x e, n 0]
    (when (and x (< n 8))
      (or (:genmlx/error (ex-data x))
          (recur (ex-cause x) (inc n))))))

(-> (pr/let [tf  (sr/resolve-reward "tool-format" {:toolset []})
             ma  (sr/resolve-reward "match-administered" {})
             f   (sr/resolve-reward plugin {:points points :toolset []})
             ;; convert the expected rejection into a VALUE inside the
             ;; binding (pr/handle) — an inner pr/catch double-settles
             ;; under nbb (the genmlx-tb5f class)
             bad (pr/handle (sr/resolve-reward bad-plugin {})
                            (fn [_ e] (when e (genmlx-error e))))]
      (assert-true "builtin specs resolve to fns" (and (fn? tf) (fn? ma)))
      (assert-equal "plugin closes over ctx and scores" 0.2 (f nil "text"))
      (assert-equal "plugin floor path" -1.0 (f nil ""))
      (assert-equal "non-fn plugin -> typed error" :bad-reward-plugin bad))
    (pr/handle
     (fn [_ e]
       (when e
         (println "ERROR:" (str e))
         (swap! fail inc))
       (println (str "\n== world-session-reward: " @pass " passed, "
                     @fail " failed =="))
       (when (pos? @fail) (set! (.-exitCode js/process) 1)))))
