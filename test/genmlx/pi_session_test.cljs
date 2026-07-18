;; @tier fast
(ns genmlx.pi-session-test
  "genmlx-opwh (L3-C1) Module A: the pi session JSONL reader + the
   convert-messages.ts mirror. Pure — no GPU, no model.

   Covers: header/entry parsing + typed errors, leaf-path (leaf = last
   line; abandoned branch arms excluded), session-tree branch points,
   path->messages conversion (\\n joins, thinking dropped, error/aborted
   assistant dropped with tool calls untracked, orphan repair, tool-result
   image hoist with the fixed text, base64 image decode), assistant
   indices, compaction typed error, and an env-gated smoke over every real
   session on this machine.

   Run: bun run --bun nbb test/genmlx/pi_session_test.cljs"
  (:require [clojure.string :as str]
            [genmlx.llm.pi-session :as ps]
            ["fs" :as fs]))

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

(defn- line [o] (js/JSON.stringify (clj->js o)))

(defn- thrown-error
  "Run f; return its ex-data :genmlx/error keyword, or nil if no throw."
  [f]
  (try (f) nil
       (catch :default e (:genmlx/error (ex-data e)))))

;; ---------------------------------------------------------------------------
;; fixture: a session with tool calls, drops, orphans, images, and a branch
;; ---------------------------------------------------------------------------

(def fixture-lines
  [(line {:type "session" :version 3 :id "s-1"
          :timestamp "2026-07-18T00:00:00.000Z" :cwd "/tmp/proj"})
   (line {:type "model_change" :id "e1" :parentId nil
          :timestamp "t" :provider "genmlx" :modelId "m"})
   (line {:type "thinking_level_change" :id "e2" :parentId "e1"
          :timestamp "t" :thinkingLevel "low"})
   (line {:type "message" :id "e3" :parentId "e2" :timestamp "t"
          :message {:role "user" :content "hello"}})
   ;; an ABANDONED branch arm off e3 (appended later in real files, placed
   ;; here to prove file order does not matter for tree structure)
   (line {:type "message" :id "x1" :parentId "e3" :timestamp "t"
          :message {:role "user" :content "abandoned arm"}})
   ;; the live arm continues from e3
   (line {:type "message" :id "e4" :parentId "e3" :timestamp "t"
          :message {:role "assistant" :stopReason "toolUse"
                    :usage {:input 5 :output 7 :totalTokens 12}
                    :content [{:type "text" :text "hi"}
                              {:type "thinking" :thinking "secret"}
                              {:type "toolCall" :id "call_1" :name "look"
                               :arguments {:b 2 :a 1}}]}})
   (line {:type "message" :id "e5" :parentId "e4" :timestamp "t"
          :message {:role "toolResult" :toolCallId "call_1" :toolName "look"
                    :isError false
                    :content [{:type "text" :text "seen"}
                              {:type "image" :data "YWJj"
                               :mimeType "image/png"}]}})
   ;; error-stopReason assistant: DROPPED, call_2 never tracked
   (line {:type "message" :id "e6" :parentId "e5" :timestamp "t"
          :message {:role "assistant" :stopReason "error"
                    :content [{:type "text" :text "partial"}
                              {:type "toolCall" :id "call_2" :name "x"
                               :arguments {}}]}})
   (line {:type "message" :id "e7" :parentId "e6" :timestamp "t"
          :message {:role "user"
                    :content [{:type "text" :text "a"}
                              {:type "text" :text "b"}
                              {:type "image" :data "YWJj"
                               :mimeType "image/png"}]}})
   ;; retained assistant whose call_3 never gets a result -> orphan repair
   (line {:type "message" :id "e8" :parentId "e7" :timestamp "t"
          :message {:role "assistant" :stopReason "toolUse"
                    :content [{:type "toolCall" :id "call_3" :name "y"
                               :arguments {:q 1}}]}})
   ;; a custom role rides the path and must be skipped without state change
   (line {:type "message" :id "e8b" :parentId "e8" :timestamp "t"
          :message {:role "bashExecution" :command "ls"}})
   (line {:type "message" :id "e9" :parentId "e8b" :timestamp "t"
          :message {:role "user" :content "next"}})
   (line {:type "label" :id "e9b" :parentId "e9" :timestamp "t"
          :targetId "e3" :label "start"})
   (line {:type "message" :id "e10" :parentId "e9b" :timestamp "t"
          :message {:role "assistant" :stopReason "stop"
                    :content [{:type "text" :text "done"}]}})])

(def fixture (str/join "\n" fixture-lines))

(println "\n-- parse-session --")
(let [{:keys [header entries]} (ps/parse-session fixture)]
  (assert-equal "header id" "s-1" (:id header))
  (assert-equal "header cwd" "/tmp/proj" (:cwd header))
  (assert-equal "header parent-session nil" nil (:parent-session header))
  (assert-equal "entry count" 13 (count entries))
  (assert-equal "entry types survive"
                "model_change" (:type (first entries)))
  (assert-equal "nil parentId -> nil parent-id" nil (:parent-id (first entries))))

(println "\n-- typed parse errors --")
(assert-equal "bad header -> :bad-session-header"
              :bad-session-header
              (thrown-error #(ps/parse-session (line {:type "message" :id "z"}))))
(assert-equal "empty file -> :bad-session-header"
              :bad-session-header
              (thrown-error #(ps/parse-session "  \n \n")))
(assert-equal "malformed line -> :malformed-session-line"
              :malformed-session-line
              (thrown-error #(ps/parse-session (str (first fixture-lines)
                                                    "\n{not json"))))

(println "\n-- leaf-path + session-tree --")
(let [session (ps/parse-session fixture)
      path    (ps/leaf-path session)
      tree    (ps/session-tree session)]
  (assert-equal "leaf is the last line" "e10" (:id (last path)))
  (assert-equal "path excludes the abandoned arm"
                ["e1" "e2" "e3" "e4" "e5" "e6" "e7" "e8" "e8b" "e9" "e9b" "e10"]
                (mapv :id path))
  (assert-equal "one root" 1 (count tree))
  (let [e3-node (->> (tree-seq map? :children (first tree))
                     (filter #(= "e3" (:id (:entry %))))
                     first)]
    (assert-equal "branch point e3 has two children"
                  #{"x1" "e4"} (into #{} (map #(:id (:entry %))) (:children e3-node))))
  (assert-equal "missing parent -> :broken-session-tree"
                :broken-session-tree
                (thrown-error
                 #(ps/leaf-path (ps/parse-session
                                 (str (first fixture-lines) "\n"
                                      (line {:type "message" :id "q"
                                             :parentId "ghost" :timestamp "t"
                                             :message {:role "user" :content "x"}})))))))

(println "\n-- path->messages (the convert-messages.ts mirror) --")
(let [session (ps/parse-session fixture)
      path    (ps/leaf-path session)
      msgs    (ps/path->messages path)]
  (assert-equal "message count" 9 (count msgs))
  (assert-equal "roles in order"
                ["user" "assistant" "tool" "user" "user" "assistant" "tool" "user" "assistant"]
                (mapv :role msgs))
  (assert-equal "user string content" "hello" (:content (nth msgs 0)))
  (let [a (nth msgs 1)]
    (assert-equal "assistant text (thinking dropped)" "hi" (:content a))
    (assert-equal "tool call id/name"
                  [{:id "call_1" :name "look"}]
                  (mapv #(select-keys % [:id :name]) (:toolCalls a)))
    (assert-equal "tool call arguments re-stringified with original key order"
                  "{\"b\":2,\"a\":1}"
                  (:arguments (first (:toolCalls a)))))
  (let [t (nth msgs 2)]
    (assert-equal "tool result shape"
                  {:role "tool" :content "seen" :toolCallId "call_1" :isError false}
                  (dissoc t :images)))
  (let [h (nth msgs 3)]
    (assert-equal "tool-image hoist text" ps/tool-image-hoist-text (:content h))
    (assert-equal "hoisted image bytes" [97 98 99]
                  (vec (js/Array.from (first (:images h))))))
  (let [u (nth msgs 4)]
    (assert-equal "user parts joined with newline" "a\nb" (:content u))
    (assert-equal "user image decoded" 3 (.-length (first (:images u)))))
  (assert-equal "error assistant dropped (no 'partial' text anywhere)"
                nil (some #(str/includes? (str (:content %)) "partial") msgs))
  (assert-equal "empty-text assistant content" "" (:content (nth msgs 5)))
  (let [orphan (nth msgs 6)]
    (assert-equal "orphan repair before the next user"
                  {:role "tool" :content "No result provided"
                   :toolCallId "call_3" :isError true}
                  orphan))
  (assert-equal "call_2 of the dropped assistant NOT repaired"
                nil (some #(= "call_2" (:toolCallId %)) msgs))
  (assert-equal "assistant indices" [1 5 8] (ps/assistant-indices msgs))
  ;; provenance metadata (genmlx-5v23): real messages carry their source
  ;; entry id; synthetic ones (orphan repair, image hoist) carry none
  (assert-equal "converted messages carry source entry ids"
                ["e3" "e4" "e5" "e7" "e8" "e9" "e10"]
                (into [] (keep ps/message-entry-id) msgs))
  (assert-equal "synthetic messages carry no entry id"
                [nil nil]
                [(ps/message-entry-id (nth msgs 3))    ; image hoist
                 (ps/message-entry-id (nth msgs 6))])  ; orphan repair
  ;; system prompt option
  (let [msgs' (ps/path->messages path {:system-prompt "SYS"})]
    (assert-equal "system prompt prepended"
                  {:role "system" :content "SYS"} (first msgs'))
    (assert-equal "assistant indices shift" [2 6 9] (ps/assistant-indices msgs'))))

(println "\n-- trailing orphan repair --")
(let [text (str/join "\n"
                     [(line {:type "session" :version 3 :id "s-2"
                             :timestamp "t" :cwd "/tmp"})
                      (line {:type "message" :id "a1" :parentId nil :timestamp "t"
                             :message {:role "assistant" :stopReason "toolUse"
                                       :content [{:type "toolCall" :id "c9"
                                                  :name "z" :arguments {}}]}})])
      msgs (ps/path->messages (ps/leaf-path (ps/parse-session text)))]
  (assert-equal "unanswered final tool call repaired at path end"
                {:role "tool" :content "No result provided"
                 :toolCallId "c9" :isError true}
                (peek msgs)))

(println "\n-- compaction --")
(let [text (str/join "\n"
                     [(line {:type "session" :version 3 :id "s-3"
                             :timestamp "t" :cwd "/tmp"})
                      (line {:type "message" :id "m1" :parentId nil :timestamp "t"
                             :message {:role "user" :content "x"}})
                      (line {:type "compaction" :id "c1" :parentId "m1"
                             :timestamp "t" :summary "..." :firstKeptEntryId "m1"
                             :tokensBefore 9})])
      session (ps/parse-session text)]
  (assert-equal "compaction on the path -> typed error"
                :compaction-unsupported
                (thrown-error #(ps/path->messages (ps/leaf-path session)))))

(println "\n-- message->js round trip shape --")
(let [o (ps/message->js {:role "assistant" :content "hi"
                         :toolCalls [{:id "c" :name "n" :arguments "{\"a\":1}"}]})]
  (assert-equal "role" "assistant" (.-role o))
  (assert-equal "toolCalls arguments string" "{\"a\":1}"
                (.-arguments (aget (.-toolCalls o) 0))))
(let [o (ps/message->js {:role "tool" :content "r" :toolCallId "c" :isError false})]
  (assert-true "isError false survives (not elided)" (false? (.-isError o))))

(println "\n-- real sessions on this machine (env-gated smoke) --")
(let [root (str (.. js/process -env -HOME) "/.mlx-node/agent/sessions")]
  (if-not (.existsSync fs root)
    (println "  SKIP — no session directory at" root)
    (let [files (->> (.readdirSync fs root)
                     (mapcat (fn [d]
                               (let [dir (str root "/" d)]
                                 (when (.isDirectory (.statSync fs dir))
                                   (->> (.readdirSync fs dir)
                                        (filter #(str/ends-with? % ".jsonl"))
                                        (map #(str dir "/" %)))))))
                     vec)
          results (mapv (fn [f]
                          (try
                            (let [s (ps/read-session f)
                                  p (ps/leaf-path s)]
                              ;; compaction is a legitimate typed refusal
                              (try (ps/path->messages p) :ok
                                   (catch :default e
                                     (if (= :compaction-unsupported
                                            (:genmlx/error (ex-data e)))
                                       :compacted
                                       (throw e)))))
                            (catch :default e [:fail f (ex-message e)])))
                        files)
          fails (filterv vector? results)]
      (println "  " (count files) "session files:"
               (count (filter #(= :ok %) results)) "converted,"
               (count (filter #(= :compacted %) results)) "compacted")
      (doseq [[_ f m] (take 3 fails)] (println "    FAIL" f "—" m))
      (assert-true "every real session parses (or is a typed compaction refusal)"
                   (empty? fails)))))

(println (str "\n== pi-session: " @pass " passed, " @fail " failed =="))
(when (pos? @fail) (set! (.-exitCode js/process) 1))
