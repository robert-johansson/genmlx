;; @tier slow
(ns genmlx.pi-fork-test
  "genmlx-lin9 (L3-C3), engine side: newSession {\"forkFrom\": sid} on the
   0.8b — the O(1) counterfactual-administration substrate.

     P1 fork == continuation: the same next turn run on the fork and on
        the source is byte-identical, with identical cachedTokens (the
        shared committed prefix delta-prefills on BOTH arms — fork cost
        is ~0 prefill)
     P2 divergence: the arms take different turns and each keeps
        delta-prefilling over its OWN history
     P3 dispose independence: disposing the source leaves the fork alive
        (persistent cache values — reference sharing, not aliasing)
     P4 fork-of-fork chains
     P5 unknown forkFrom -> :unknown-session
     P6 forking a mid-turn session -> :fork-source-busy; after the turn
        resolves the fork succeeds

   Run: bun run --bun nbb test/genmlx/pi_fork_test.cljs (guarded on Thor)"
  (:require [genmlx.llm.pi-provider :as pp]
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

(def model-dir
  (let [d (path/join (os/homedir) ".cache" "models" "qwen3.5-0.8b-mlx-bf16")]
    (when (.existsSync fs (path/join d "tokenizer.json")) d)))

(defn- summary []
  (println (str "\n== pi-fork: " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

(def engine pp/engine)

(defn- run-turn [sid messages config]
  (-> (.turnStream engine sid
                   (js/JSON.stringify (clj->js messages))
                   (js/JSON.stringify (clj->js config))
                   (fn [_]) js/undefined js/undefined)
      (pr/then (fn [fj] (js->clj (js/JSON.parse fj) :keywordize-keys true)))))

(defn- fork! [sid]
  (.newSession engine (js/JSON.stringify #js {:forkFrom sid})))

(defn- thrown-error [f]
  (try (f) nil
       (catch :default e (:genmlx/error (ex-data e)))))

(def msgs-1
  [{:role "system" :content "You are a terse assistant."}
   {:role "user" :content "Reply with exactly one short English word."}])

(def cfg {:temperature 0 :maxNewTokens 24 :reasoningEffort "none"})

(if-not model-dir
  (do (println "SKIP pi-fork — no qwen3.5-0.8b checkpoint") (summary))
  (->
   (pr/let [info (.loadModel engine model-dir)
            _    (pr/resolved (assert-true "load resolves" (some? (.-path info))))
            sidA (pr/resolved (.newSession engine "{}"))
            f1   (run-turn sidA msgs-1 cfg)]
     (println "  turn-1:" (pr-str (:text f1)))
     (let [msgs-2 (conj msgs-1
                        {:role "assistant" :content (:text f1)}
                        {:role "user" :content "Now reply with a different word."})
           sidB   (fork! sidA)]
       (pr/let [;; P1: the same next turn on both arms
                fB (run-turn sidB msgs-2 cfg)
                fA (run-turn sidA msgs-2 cfg)]
         (println "  arm texts:" (pr-str (:text fA)) "/" (pr-str (:text fB))
                  "| cached" (:cachedTokens fA) "/" (:cachedTokens fB))
         (assert-true "P1: fork's turn == source's turn (byte-identical)"
                      (= (:text fA) (:text fB)))
         (assert-true "P1: identical delta-prefill accounting on both arms"
                      (and (= (:cachedTokens fA) (:cachedTokens fB))
                           (pos? (:cachedTokens fB))))
         (assert-true "P1: the fork reused the WHOLE shared prefix (~0-cost fork)"
                      (>= (:cachedTokens fB) (:promptTokens f1)))
         ;; P2: divergent turns, each arm delta-prefills its own history
         (let [msgs-3a (conj msgs-2
                             {:role "assistant" :content (:text fA)}
                             {:role "user" :content "Translate it to French."})
               msgs-3b (conj msgs-2
                             {:role "assistant" :content (:text fB)}
                             {:role "user" :content "Translate it to German."})]
           (pr/let [f3a (run-turn sidA msgs-3a cfg)
                    f3b (run-turn sidB msgs-3b cfg)]
             (println "  divergent:" (pr-str (:text f3a)) "/" (pr-str (:text f3b)))
             (assert-true "P2: both arms complete after divergence"
                          (and (contains? #{"stop" "length"} (:finishReason f3a))
                               (contains? #{"stop" "length"} (:finishReason f3b))))
             (assert-true "P2: each arm delta-prefills over its own arm"
                          (and (> (:cachedTokens f3a) (:cachedTokens fA))
                               (> (:cachedTokens f3b) (:cachedTokens fB))))
             ;; P3: dispose the source; the fork lives on
             (.dispose engine sidA)
             (let [msgs-4 (conj msgs-3b
                                {:role "assistant" :content (:text f3b)}
                                {:role "user" :content "Say ok."})]
               (pr/let [f4 (run-turn sidB msgs-4 cfg)]
                 (assert-true "P3: fork survives source disposal"
                              (contains? #{"stop" "length"} (:finishReason f4)))
                 (assert-true "P3: and still delta-prefills"
                              (> (:cachedTokens f4) 0))
                 ;; P4: fork of fork — P1 at depth 2: fork AFTER f4, then the
                 ;; same continuation on both arms
                 (let [sidC   (fork! sidB)
                       msgs-5 (conj msgs-4
                                    {:role "assistant" :content (:text f4)}
                                    {:role "user" :content "Say it again."})]
                   (pr/let [f5c (run-turn sidC msgs-5 cfg)
                            f5b (run-turn sidB msgs-5 cfg)]
                     (assert-true "P4: fork-of-fork turn == child's turn"
                                  (= (:text f5c) (:text f5b)))
                     (assert-true "P4: identical delta-prefill at depth 2"
                                  (and (= (:cachedTokens f5c) (:cachedTokens f5b))
                                       (pos? (:cachedTokens f5c))))
                     (.dispose engine sidC)
                   ;; P5: unknown source
                   (assert-true "P5: fork of unknown session -> :unknown-session"
                                (= :unknown-session (thrown-error #(fork! "nope"))))
                   ;; P6: fork mid-turn refused; after resolution it works
                   (let [eos  (.-eosTokenId info)
                         pin  (fn [logits _gen]
                                (let [vocab (first (mx/shape logits))]
                                  (mx/where (mx/equal (mx/arange 0 vocab 1)
                                                      (mx/scalar 100))
                                            logits (mx/scalar -1e30))))
                         _    (assert-true "P6: setup uses a non-eos pin"
                                           (not= 100 eos))
                         _    (pp/set-logit-mask! sidB pin)
                         turn (run-turn sidB
                                        (conj msgs-3b
                                              {:role "assistant" :content (:text f3b)}
                                              {:role "user" :content "Go on at length."})
                                        (assoc cfg :maxNewTokens 512))]
                     (pr/let [err (pr/delay 150 nil)
                              err (pr/resolved (thrown-error #(fork! sidB)))
                              _   (pr/resolved (.abort engine sidB))
                              ft  turn]
                       (assert-true "P6: mid-turn fork refused (:fork-source-busy)"
                                    (= :fork-source-busy err))
                       (assert-true "P6: the pinned turn was aborted (not raced)"
                                    (= "aborted" (:finishReason ft)))
                       (pp/set-logit-mask! sidB nil)
                       (let [sidD (fork! sidB)]
                         (assert-true "P6: fork succeeds once the turn resolved"
                                      (string? sidD))
                         (.dispose engine sidD)
                         (.dispose engine sidB)
                         (summary)))))))))))))
   (pr/catch (fn [e]
               (println "ERROR:" (str e))
               (swap! fail inc)
               (summary)))))
