;; @tier slow
(ns genmlx.pi-edit-test
  "genmlx-5v23 (L3-C2), the decode half on the 0.8b: boundary resolution
   against the real render, resample-from-boundary (whole turn, token,
   tool-call), the C1-scores-C2 law (decode-time suffix logprob vs
   forward-branch-scores over the same tokens), and edit-session! writing
   a fork that round-trips through read-session AND pa/session-scores.

     A  boundary: {:tool-call 0} keeps exactly the pre-block prefix
     B  whole-turn temp-0 resample is deterministic and == {:token 0}
     C  {:tool-call 0} resample preserves the kept prose byte-for-byte
     D  LAW: suffix-logprob == forward-branch-scores over ctx ++ sampled
        (bf16 graph-shape tolerance — the C1 instrument scoring the edit)
     E  edit-session! fork: round trip + session-scores on the edited arm
     F  error taxonomy + in-place mode

   Run: bun run --bun nbb test/genmlx/pi_edit_test.cljs (guarded on Thor)"
  (:require [clojure.string :as str]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.pi-assess :as pa]
            [genmlx.llm.pi-edit :as pe]
            [genmlx.llm.pi-session :as ps]
            [genmlx.mlx :as mx]
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
(defn assert-close [label expected actual tol]
  (let [d (js/Math.abs (- expected actual))]
    (if (<= d tol)
      (do (swap! pass inc) (println "  PASS" label (str "(|Δ| " (.toFixed d 5) ")")))
      (do (swap! fail inc)
          (println "  FAIL" label "expected" expected "actual" actual "tol" tol)))))

(def model-dir
  (let [d (path/join (os/homedir) ".cache" "models" "qwen3.5-0.8b-mlx-bf16")]
    (when (.existsSync fs (path/join d "tokenizer.json")) d)))

(defn- summary []
  (println (str "\n== pi-edit: " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

(def sys-text "You are a terse assistant.")

(defn- jl [o] (js/JSON.stringify (clj->js o)))

(def fixture
  (str (str/join
        "\n"
        [(jl {:type "session" :version 3 :id "edit-src" :timestamp "t"
              :cwd "/w"})
         (jl {:type "model_change" :id "e0" :parentId nil :timestamp "t"
              :provider "genmlx" :modelId "qwen3.5-0.8b"})
         (jl {:type "message" :id "e1" :parentId "e0" :timestamp "t"
              :message {:role "user"
                        :content "Set the point at row 12, column 34."}})
         (jl {:type "message" :id "e2" :parentId "e1" :timestamp "t"
              :message {:role "assistant" :stopReason "toolUse"
                        :content [{:type "text"
                                   :text "Setting the point now."}
                                  {:type "toolCall" :id "c1"
                                   :name "set_point"
                                   :arguments {:xy "12,34"}}]}})
         (jl {:type "message" :id "e3" :parentId "e2" :timestamp "t"
              :message {:role "toolResult" :toolCallId "c1"
                        :toolName "set_point" :isError false
                        :content [{:type "text" :text "ok"}]}})
         (jl {:type "message" :id "e4" :parentId "e3" :timestamp "t"
              :message {:role "user" :content "Thanks. Say done."}})
         (jl {:type "message" :id "e5" :parentId "e4" :timestamp "t"
              :message {:role "assistant" :stopReason "stop"
                        :content [{:type "text" :text "Done."}]}})])
       "\n"))

(defn- decode-toks [tokenizer toks]
  (llm/decode tokenizer (js/Uint32Array.from (into-array toks))))

(defn- genmlx-error
  "The :genmlx/error keyword of e or any error on its cause chain — a
   rejection that crossed a promise chain under nbb arrives wrapped in an
   :sci/error whose original ex-data sits on the cause."
  [e]
  (loop [x e, n 0]
    (when (and x (< n 8))
      (or (:genmlx/error (ex-data x))
          (recur (ex-cause x) (inc n))))))

(defn- rejects-with
  "Run thunk (sync throw or rejected promise); resolves to true iff it
   fails with the given :genmlx/error."
  [thunk kw]
  (try
    (-> (pr/let [_ (thunk)] false)
        (pr/catch (fn [e] (= kw (genmlx-error e)))))
    (catch :default e
      (pr/resolved (= kw (genmlx-error e))))))

(if-not model-dir
  (do (println "SKIP pi-edit — no qwen3.5-0.8b checkpoint") (summary))
  (->
   (pr/let [mm (llm/load-model model-dir {:cljs-forward? true})]
     (let [{:keys [model tokenizer]} mm
           dir  (fs/mkdtempSync (path/join (os/tmpdir) "pi-edit-gpu-"))
           src  (path/join dir "2026-07-18T00-00-00-000Z_edit-src.jsonl")
           _    (fs/writeFileSync src fixture)
           msgs (ps/path->messages (ps/leaf-path (ps/read-session src))
                                   {:system-prompt sys-text})
           k    2] ; [sys user asst1 tool user asst2] — asst1
       (pr/let [;; ---- A: boundary resolution -----------------------------
                rb        (pe/resolve-boundary tokenizer msgs k
                                               {:tool-call 0} {})
                kept-text (decode-toks tokenizer (:keep-tokens rb))
                span-text (let [[s e] (:span (:render rb))]
                            (decode-toks tokenizer
                                         (subvec (:full (:render rb)) s e)))]
         (println "  kept:" (pr-str kept-text))
         (assert-true "A: kept prefix ends before the block opener"
                      (and (pos? (:boundary-token rb))
                           (not (str/includes? kept-text "<tool_call>"))))
         (assert-true "A: the prose survives in the kept prefix"
                      (str/includes? kept-text "Setting the point now."))
         (assert-true "A: span text carries the rendered block"
                      (str/includes? span-text "<function=set_point>"))
         (assert-true "A: kept decode is a prefix of the span text"
                      (str/starts-with? span-text kept-text))
         (pr/let [;; ---- B/C: resamples ----------------------------------
                  r-a  (pe/resample-turn mm msgs k nil {:max-new 24})
                  r-b  (pe/resample-turn mm msgs k nil {:max-new 24})
                  r-t0 (pe/resample-turn mm msgs k {:token 0} {:max-new 24})
                  r-tc (pe/resample-turn mm msgs k {:tool-call 0}
                                         {:max-new 24})]
           (println "  whole-turn resample:" (pr-str (:text r-a)))
           (println "  from-tool-call:     " (pr-str (:text r-tc))
                    "| suffix lp" (.toFixed (:suffix-logprob r-tc) 3))
           (assert-true "B: temp-0 whole-turn resample is deterministic"
                        (= (:text r-a) (:text r-b)))
           (assert-true "B: {:token 0} == whole turn"
                        (= (:text r-a) (:text r-t0)))
           (assert-true "B: finish reason is terminal"
                        (contains? #{"stop" "length" "toolUse"}
                                   (:finish-reason r-a)))
           (assert-true "C: kept prose preserved byte-for-byte"
                        (str/starts-with? (:text r-tc) kept-text))
           (assert-true "C: suffix logprob finite and negative"
                        (and (js/isFinite (:suffix-logprob r-tc))
                             (neg? (:suffix-logprob r-tc))))
           ;; ---- D: the C1-scores-C2 law --------------------------------
           (let [s    (first (:span (:render rb)))
                 ctx  (into (vec (subvec (:full (:render rb)) 0 s))
                            (:keep-tokens rb))
                 seq' (into ctx (:sampled-tokens r-tc))
                 b    (llm/owned-branch! model {:cache nil :offset 0})
                 sc   (vec (mx/->clj (llm/forward-branch-scores model b seq')))
                 _    (llm/dispose-branch! model b)
                 tail (subvec sc (- (count sc)
                                    (count (:sampled-tokens r-tc))))]
             (assert-close "D: decode-time suffix lp == C1 scorer over the edit"
                           (reduce + 0.0 tail) (:suffix-logprob r-tc) 2.0)
             ;; ---- E: edit-session! fork + assess-the-edit ---------------
             (pr/let [res    (pe/edit-session! mm src k {:tool-call 0}
                                               {:system-prompt sys-text
                                                :max-new 24})
                      forked (pr/resolved (ps/read-session (:file res)))
                      msgs'  (pr/resolved
                              (ps/path->messages (ps/leaf-path forked)
                                                 {:system-prompt sys-text}))
                      scores (pa/session-scores mm msgs' {})]
               (assert-true "E: fork written to a new file"
                            (and (not= (:file res) src)
                                 (.existsSync fs (:file res))))
               (assert-true "E: fork header points home"
                            (= src (get-in forked [:header :parent-session])))
               (assert-true "E: leaf is the edited turn"
                            (= (:leaf-id res)
                               (:id (last (ps/leaf-path forked)))))
               (assert-true "E: edited turn renders with the kept prose"
                            (str/includes?
                             (:content (nth msgs' 2))
                             "Setting the point now."))
               (assert-true "E: session-scores runs on the edited arm (1 turn)"
                            (and (= 1 (count scores))
                                 (js/isFinite (:logprob (first scores)))
                                 (neg? (:logprob (first scores)))))
               ;; ---- F: errors + in-place --------------------------------
               (pr/let [e1 (rejects-with
                            #(pe/resample-turn mm msgs 1 nil {}) :not-assistant)
                        e2 (rejects-with
                            #(pe/resample-turn mm msgs 99 nil {}) :bad-turn-index)
                        e3 (rejects-with
                            #(pe/resample-turn mm msgs k {:tool-call 7} {})
                            :no-such-tool-call)
                        e4 (rejects-with
                            #(pe/resample-turn mm msgs k {:bogus 1} {})
                            :bad-boundary)
                        e5 (rejects-with
                            #(pe/resample-turn mm msgs k {:token 9999} {})
                            :bad-boundary-token)]
                 (assert-true "F: non-assistant turn refused" e1)
                 (assert-true "F: out-of-range turn refused" e2)
                 (assert-true "F: missing tool-call refused" e3)
                 (assert-true "F: unknown boundary refused" e4)
                 (assert-true "F: out-of-span token refused" e5)
                 (let [src2 (path/join dir "in-place.jsonl")]
                   (fs/copyFileSync src src2)
                   (pr/let [res2 (pe/edit-session! mm src2 k nil
                                                   {:system-prompt sys-text
                                                    :max-new 16
                                                    :in-place? true})]
                     (assert-true "F: in-place edit appends to the source"
                                  (= src2 (:file res2)))
                     (assert-true "F: in-place leaf is the edit"
                                  (= (:leaf-id res2)
                                     (:id (last (ps/leaf-path
                                                 (ps/read-session src2))))))
                     (summary))))))))))
   (pr/catch (fn [e]
               (println "ERROR:" (or (ex-message e) (str e)))
               (when-let [d (ex-data e)]
                 (println "  ex-data:" (pr-str (dissoc d :sci.impl/callstack))))
               (swap! fail inc)
               (summary)))))
