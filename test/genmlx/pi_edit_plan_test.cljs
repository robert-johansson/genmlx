;; @tier fast
(ns genmlx.pi-edit-plan-test
  "genmlx-5v23 (L3-C2), the model-free half: tool-call span/prose text
   machinery, plan-edit (the pi branchWithSummary rejoin shape), and
   write-edit! (fork = verbatim copy + branch; in-place = append). The
   resample result is FABRICATED here; the decode half runs in
   pi_edit_test (slow, guarded).

   Run: bun run --bun nbb test/genmlx/pi_edit_plan_test.cljs"
  (:require [clojure.string :as str]
            [genmlx.llm.pi-edit :as pe]
            [genmlx.llm.pi-session :as ps]
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

(println "\n-- tool-call spans + prose --")
(let [block "<tool_call>\n<function=f>\n</function>\n</tool_call>"
      text  (str "Lead. " block " mid " block " tail")]
  (assert-equal "two complete blocks found" 2 (count (pe/tool-call-spans text)))
  (assert-equal "span slices reproduce the blocks"
                [block block]
                (mapv #(subs text (:start %) (:end %)) (pe/tool-call-spans text)))
  (assert-equal "prose strip keeps surrounding text"
                "Lead.  mid  tail" (pe/strip-tool-calls text)))
(assert-equal "unterminated opener contributes no span"
              [] (pe/tool-call-spans "x <tool_call>\n<function=f>"))
(assert-equal "opener without newline is not a block"
              [] (pe/tool-call-spans "<tool_call>x</tool_call>"))
(assert-equal "no blocks -> text trimmed only"
              "plain" (pe/strip-tool-calls "plain  \n"))
(assert-equal "trailing block trims the gap"
              "Answer:"
              (pe/strip-tool-calls
               "Answer:\n\n<tool_call>\n<function=f>\n</function>\n</tool_call>\n"))

;; ---------------------------------------------------------------------------
;; fixture session
;; ---------------------------------------------------------------------------

(defn- jl [o] (js/JSON.stringify (clj->js o)))

(def fixture-lines
  [(jl {:type "session" :version 3 :id "src-1" :timestamp "t" :cwd "/w"})
   (jl {:type "model_change" :id "e1" :parentId nil :timestamp "t"
        :provider "mlx" :modelId "ornith-x"})
   (jl {:type "message" :id "e2" :parentId "e1" :timestamp "t"
        :message {:role "user" :content "hello"}})
   (jl {:type "message" :id "e3" :parentId "e2" :timestamp "t"
        :message {:role "assistant" :stopReason "stop"
                  :content [{:type "text" :text "hi"}]}})
   (jl {:type "message" :id "e4" :parentId "e3" :timestamp "t"
        :message {:role "user" :content "again"}})
   (jl {:type "message" :id "e5" :parentId "e4" :timestamp "t"
        :message {:role "assistant" :stopReason "stop"
                  :content [{:type "text" :text "final"}]}})])
(def fixture (str (str/join "\n" fixture-lines) "\n"))

(def resample
  {:turn 1 :source-entry-id "e3" :boundary-token 0 :exact? true
   :kept-tokens [] :sampled-tokens [1 2 3] :new-tokens [1 2]
   :text "Point set.\n\n<tool_call>\n...\n</tool_call>"
   :prose "Point set."
   :tool-calls [{:name "set_point" :args {"xy" "1,2"}}]
   :tool-call-errors [] :finish-reason "toolUse" :suffix-logprob -3.5
   :usage {:input 10 :output 3}})

(println "\n-- plan-edit --")
(let [session (ps/parse-session fixture)
      {:keys [entries leaf-id]} (pe/plan-edit session resample)
      [e1 e2] entries]
  (assert-equal "branch_summary at the rejoin point (edited turn's parent)"
                ["branch_summary" "e2"] [(:type e1) (:parentId e1)])
  (assert-equal "fromId names the abandoned leaf" "e5" (:fromId e1))
  (assert-true "summary carries provenance"
               (str/includes? (:summary e1) "e3"))
  (assert-equal "message childs the summary" (:id e1) (:parentId e2))
  (assert-equal "leaf-id is the new message" (:id e2) leaf-id)
  (assert-true "fresh 8-hex ids, distinct"
               (and (re-matches #"[0-9a-f]{8}" (:id e1))
                    (re-matches #"[0-9a-f]{8}" (:id e2))
                    (not= (:id e1) (:id e2))))
  (let [m (:message e2)]
    (assert-equal "identity scraped from model_change"
                  ["mlx" "mlx" "ornith-x"]
                  [(:api m) (:provider m) (:model m)])
    (assert-equal "stopReason carried" "toolUse" (:stopReason m))
    (assert-equal "honest usage totals" 13 (get-in m [:usage :totalTokens]))
    (assert-equal "content = prose part + tool call"
                  [{:type "text" :text "Point set."}
                   {:type "toolCall" :id "call_edit_1" :name "set_point"
                    :arguments {"xy" "1,2"}}]
                  (:content m))))
(let [session (ps/parse-session fixture)
      {:keys [entries]} (pe/plan-edit session resample
                                      {:provider "genmlx" :model "m2"
                                       :summary "S"})
      [e1 e2] entries]
  (assert-equal "opts override identity + summary"
                ["S" "genmlx" "m2"]
                [(:summary e1) (get-in e2 [:message :provider])
                 (get-in e2 [:message :model])]))
(assert-true "unknown source entry -> :not-editable"
             (try (pe/plan-edit (ps/parse-session fixture)
                                (assoc resample :source-entry-id "ghost"))
                  false
                  (catch :default e
                    (= :not-editable (:genmlx/error (ex-data e))))))

(println "\n-- write-edit! fork --")
(def dir (fs/mkdtempSync (path/join (os/tmpdir) "pi-edit-")))
(def src (path/join dir "2026-07-18T00-00-00-000Z_src-1.jsonl"))
(fs/writeFileSync src fixture)
(let [session (ps/parse-session fixture)
      plan    (pe/plan-edit session resample)
      out     (pe/write-edit! src plan {})]
  (assert-true "fork written to a new pi-shaped file"
               (and (not= out src)
                    (re-matches #".*T.*Z_[0-9a-f-]{36}\.jsonl" out)))
  (let [text   (.readFileSync fs out "utf8")
        lines  (remove str/blank? (str/split-lines text))
        forked (ps/parse-session text)]
    (assert-equal "header parentSession = source" src
                  (get-in forked [:header :parent-session]))
    (assert-equal "header cwd copied" "/w" (get-in forked [:header :cwd]))
    (assert-true "fresh session id"
                 (not= "src-1" (get-in forked [:header :id])))
    (assert-equal "original entry lines copied VERBATIM"
                  (vec (rest fixture-lines)) (vec (take 5 (rest lines))))
    (let [path' (ps/leaf-path forked)]
      (assert-equal "leaf path rejoins at e2 and ends at the edit"
                    ["e1" "e2" "branch_summary" "message"]
                    (into (mapv :id (take 2 path'))
                          (mapv :type (drop 2 path'))))
      (let [msgs (ps/path->messages path')]
        ;; the edit ends at a PENDING tool call, so the mirror's orphan
        ;; repair rightly synthesizes the placeholder result at path end —
        ;; the same repair v1 applies when priming this history
        (assert-equal "edited conversation renders (incl. pending-call repair)"
                      [{:role "user" :content "hello"}
                       {:role "assistant" :content "Point set."
                        :toolCalls [{:id "call_edit_1" :name "set_point"
                                     :arguments "{\"xy\":\"1,2\"}"}]}
                       {:role "tool" :content "No result provided"
                        :toolCallId "call_edit_1" :isError true}]
                      msgs)
        (assert-equal "edited turn carries its entry id"
                      (:leaf-id plan)
                      (ps/message-entry-id (nth msgs 1)))
        (assert-true "the repair message is synthetic (no entry id)"
                     (nil? (ps/message-entry-id (nth msgs 2))))))
    (let [tree (ps/session-tree forked)
          e2n  (->> (tree-seq map? :children (first tree))
                    (filter #(= "e2" (:id (:entry %))))
                    first)]
      (assert-equal "e2 branches two ways (original + edit)"
                    2 (count (:children e2n))))))

(println "\n-- write-edit! in-place --")
(def src2 (path/join dir "in-place.jsonl"))
(fs/writeFileSync src2 fixture)
(let [session (ps/parse-session fixture)
      plan    (pe/plan-edit session resample)
      out     (pe/write-edit! src2 plan {:in-place? true})]
  (assert-equal "in-place returns the source path" src2 out)
  (let [text  (.readFileSync fs src2 "utf8")
        lines (remove str/blank? (str/split-lines text))]
    (assert-equal "two lines appended" (+ 2 (count fixture-lines)) (count lines))
    (assert-equal "original lines untouched"
                  fixture-lines (vec (take (count fixture-lines) lines)))
    (assert-equal "reparsed leaf is the edit"
                  (:leaf-id plan)
                  (:id (last (ps/leaf-path (ps/parse-session text)))))))

(println (str "\n== pi-edit-plan: " @pass " passed, " @fail " failed =="))
(when (pos? @fail) (set! (.-exitCode js/process) 1))
