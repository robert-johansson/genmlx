;; @tier fast
(ns genmlx.world-session-grpo-test
  "genmlx-rlho (L4): pi sessions -> GRPO training prompts. Pure — fixture
   session files in a temp dir, no GPU, no model, no train.cljs.

   Covers: decision points (one per assistant turn, prompt = the decoded
   context, tool fields preserved, :images never in prompts), terminal
   mode, the reward seam (provenance rides the prompt vector's metadata,
   recoverable via prompt-meta inside a reward-fn), image handling (typed
   error vs :skip-images? drop+count), directory conversion with
   :on-error :skip for broken files.

   Run: bun run --bun nbb test/genmlx/world_session_grpo_test.cljs"
  (:require [clojure.string :as str]
            [genmlx.world.session-grpo :as sg]
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

(def s1-lines
  "A two-decision tool-loop session: user -> assistant(toolCall) ->
   toolResult -> user -> assistant(final)."
  [(jl {:type "session" :version 3 :id "sess-1" :timestamp "t" :cwd "/w"})
   (jl {:type "message" :id "m1" :parentId nil :timestamp "t"
        :message {:role "user" :content "start the trial"}})
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
        :message {:role "user" :content "answer now"}})
   (jl {:type "message" :id "m5" :parentId "m4" :timestamp "t"
        :message {:role "assistant" :stopReason "stop"
                  :content [{:type "text" :text "north"}]}})])

(def s2-lines
  "An image session: the image conditions BOTH decision points."
  [(jl {:type "session" :version 3 :id "sess-2" :timestamp "t" :cwd "/w"})
   (jl {:type "message" :id "n1" :parentId nil :timestamp "t"
        :message {:role "user"
                  :content [{:type "text" :text "what color?"}
                            {:type "image" :data "YWJj" :mimeType "image/png"}]}})
   (jl {:type "message" :id "n2" :parentId "n1" :timestamp "t"
        :message {:role "assistant" :stopReason "stop"
                  :content [{:type "text" :text "red"}]}})])

(def dir (fs/mkdtempSync (path/join (os/tmpdir) "session-grpo-")))
(fs/writeFileSync (path/join dir "a-s1.jsonl") (str/join "\n" s1-lines))
(fs/writeFileSync (path/join dir "b-s2.jsonl") (str/join "\n" s2-lines))
(fs/writeFileSync (path/join dir "c-broken.jsonl") "{not a header")

(println "\n-- decision points --")
(let [{:keys [points]} (sg/session-file->points
                        (path/join dir "a-s1.jsonl")
                        {:mode :all :system-prompt "SYS"})]
  (assert-equal "one point per assistant turn" 2 (count points))
  (let [p1 (first points), p2 (second points)]
    (assert-equal "first prompt = system + user"
                  [{:role "system" :content "SYS"}
                   {:role "user" :content "start the trial"}]
                  (:prompt p1))
    (assert-equal "second prompt spans the tool loop" 5 (count (:prompt p2)))
    (assert-equal "tool-call fields preserved in prompt context"
                  [{:id "c1" :name "look" :arguments "{\"dir\":\"n\"}"}]
                  (:toolCalls (nth (:prompt p2) 2)))
    (assert-equal "tool result keeps toolCallId + isError"
                  {:role "tool" :content "a wall" :toolCallId "c1" :isError false}
                  (nth (:prompt p2) 3))
    (assert-equal "completion is the recorded assistant message"
                  {:role "assistant" :content "north"}
                  (dissoc (get-in p2 [:meta :completion]) :toolCalls))
    (assert-equal "meta provenance"
                  {:session-id "sess-1" :cwd "/w" :turn-index 5}
                  (select-keys (:meta p2) [:session-id :cwd :turn-index]))
    (assert-true "reward seam: prompt-meta recovers provenance from the vector"
                 (= (:meta p2) (sg/prompt-meta (:prompt p2))))))

(println "\n-- terminal mode --")
(let [{:keys [points]} (sg/session-file->points (path/join dir "a-s1.jsonl") {})]
  (assert-equal "terminal = one point" 1 (count points))
  ;; no :system-prompt here, so the last assistant sits at index 4
  (assert-equal "terminal point is the LAST decision"
                4 (get-in (first points) [:meta :turn-index])))

(println "\n-- image handling --")
(assert-true "image prompt -> typed error by default"
             (try (sg/session-file->points (path/join dir "b-s2.jsonl") {})
                  false
                  (catch :default e
                    (= :images-unsupported (:genmlx/error (ex-data e))))))
(let [{:keys [points skipped]} (sg/session-file->points
                                (path/join dir "b-s2.jsonl")
                                {:skip-images? true})]
  (assert-equal "skip-images? drops the point" 0 (count points))
  (assert-equal "and counts it" 1 skipped))

(println "\n-- directory conversion --")
(assert-true "broken file -> throw by default"
             (try (sg/sessions->prompts dir {:skip-images? true}) false
                  (catch :default _ true)))
(let [{:keys [prompts points skipped failed]}
      (sg/sessions->prompts dir {:skip-images? true :on-error :skip
                                 :system-prompt "SYS"})]
  (assert-equal "one usable terminal prompt across the directory" 1 (count prompts))
  (assert-equal "image point counted skipped" 1 skipped)
  (assert-equal "broken file recorded" 1 (count failed))
  (assert-true "prompt carries the reward seam through the batch"
               (= "sess-1" (:session-id (sg/prompt-meta (first prompts)))))
  (assert-true "points align with prompts"
               (= (first prompts) (:prompt (first points))))
  (assert-true "no :images key ever reaches a prompt message"
               (not-any? #(contains? % :images) (first prompts))))

(println "\n-- unknown mode --")
(assert-true "unknown :mode -> typed error"
             (try (sg/session-file->points (path/join dir "a-s1.jsonl")
                                           {:mode :bogus})
                  false
                  (catch :default e
                    (= :unknown-mode (:genmlx/error (ex-data e))))))

(println (str "\n== world-session-grpo: " @pass " passed, " @fail " failed =="))
(when (pos? @fail) (set! (.-exitCode js/process) 1))
