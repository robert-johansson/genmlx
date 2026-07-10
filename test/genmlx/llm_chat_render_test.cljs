;; @tier fast
(ns genmlx.llm-chat-render-test
  "genmlx-jq6l: the owned-path chat template (llm/render-chat) + the
   forward-prefill :images arity's error contracts. Pure string checks and
   fake-model throws — no checkpoint, no GPU work beyond the addon load.

   render-chat is the single source of truth for the ChatML hand-building
   that generate-text-raw and the ornith vision demos previously duplicated:
   these gates pin it EXACTLY to those two formats (a drift here silently
   changes every owned-path prompt).

   Run: bunx --bun nbb@1.4.208 test/genmlx/llm_chat_render_test.cljs"
  (:require [genmlx.llm.backend :as llm]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))
(defn assert-eq [label expected actual]
  (if (= expected actual)
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc)
        (println "  FAIL" label)
        (println "    expected:" (pr-str expected))
        (println "    actual:  " (pr-str actual)))))

(println "\n-- render-chat: generate-text-raw text format (think-skip on) --")
(assert-eq "system+user, think-skip (the generate-text-raw layout)"
           (str "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\nHello<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")
           (llm/render-chat [{:role "system" :content "You are a helpful assistant."}
                             {:role "user" :content "Hello"}]))

(println "\n-- render-chat: the ornith_vision_owned demo layout (1 image) --")
(assert-eq "system+user-with-1-image, think-skip (the VLM demo layout)"
           (str "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                "What is shown?<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")
           (llm/render-chat [{:role "system" :content "You are a helpful assistant."}
                             {:role "user" :content "What is shown?" :images 1}]))

(println "\n-- render-chat: variants --")
(assert-true "2 images (as a seq of buffers) -> 2 markers"
             (= 2 (count (re-seq #"<\|image_pad\|>"
                                 (llm/render-chat [{:role "user" :content "q"
                                                    :images [(js/Uint8Array. 1)
                                                             (js/Uint8Array. 1)]}])))))
(assert-true ":think-skip? false leaves no think block"
             (not (re-find #"<think>"
                           (llm/render-chat [{:role "user" :content "q"}]
                                            {:think-skip? false}))))
(assert-true "always ends with the assistant opener"
             (.endsWith (llm/render-chat [{:role "user" :content "q"}]
                                         {:think-skip? false})
                        "<|im_start|>assistant\n"))

(println "\n-- forward-prefill :images error contracts --")
(assert-true "native model + :images -> :vlm-prefill-owned-only"
             (try (llm/forward-prefill #js {} [1 2] {:images [(js/Uint8Array. 1)]})
                  false
                  (catch :default e
                    (= :vlm-prefill-owned-only (:genmlx/error (ex-data e))))))
(assert-true "owned model without a vision tower + :images -> :no-vision-tower"
             (try (llm/forward-prefill
                   (llm/->CljsForwardModel {:vcfg nil} (atom nil)
                                           (atom {:next-id 1 :branches {}}))
                   [1 2] {:images [(js/Uint8Array. 1)]})
                  false
                  (catch :default e
                    (= :no-vision-tower (:genmlx/error (ex-data e))))))

(println (str "\n== llm-chat-render: " @pass " passed, " @fail " failed =="))
(when (pos? @fail) (set! (.-exitCode js/process) 1))
