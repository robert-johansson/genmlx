;; @tier exclude — loads Ornith-1.0-35B-4bit (~25 GB) and runs THREE full VLM
;; prefills (~630 tokens each, chunked). Run manually, guarded and solo:
;;   ~/genmlx-guarded-run.sh vlm-gf bunx --bun nbb@1.4.208 test/genmlx/llm_owned_vlm_gf_test.cljs
(ns genmlx.llm-owned-vlm-gf-test
  "genmlx-jq6l GATE: images through the backend/GF API on the owned forward.

   What w3og proved at the fwd layer (vis/vlm-prefill + q35/step) and 7f93
   proved at the branch layer, this proves at the API layers above:
     G1 render-chat -> encode carries exactly one image marker
     G2 llm/forward-prefill {:images [img]} installs the image-conditioned
        cache (offset = expanded seq-len, rope-delta stored) and plain
        llm/forward-step continues from it transparently (sensible greedy text)
     G3 make-llm-gf {:images [img]}: p/simulate builds a trace whose sites
        are the answer tokens; score finite; decoded text sensible
     G4 p/generate fully constrained to G3's choices: weight == that run's
        trace score (the same-run GFI identity, tight — no jitter excuse:
        both are computed from one forward execution)

   Text-level answers are PRINTED (recorded next to the native ones in the
   bean), never hard-asserted (genmlx-ba06 jitter)."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as core]
            [genmlx.llm.qwen35-vision-forward :as vfwd]
            [promesa.core :as pr]
            ["fs" :as fs]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

(def model-dir
  (or (some-> js/process .-env .-GENMLX_OWNED_MOE_MODEL)
      (let [base (str (.-HOME js/process.env)
                      "/.cache/huggingface/hub/models--mlx-community--Ornith-1.0-35B-4bit/snapshots")]
        (when (.existsSync fs base)
          (str base "/" (first (js->clj (.readdirSync fs base))))))))
(def image-path (str (.-HOME js/process.env) "/code/mlx/ornith/image.png"))

(def question "What is shown in this image? Answer in a few words.")

(defn- summary []
  (println (str "\n== llm-owned-vlm-gf: " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

(if-not (and model-dir (.existsSync fs image-path))
  (do (println "SKIP llm-owned-vlm-gf — checkpoint or test image absent") (summary))
  (->
   (pr/let [img (.readFileSync fs image-path)
            mm (llm/load-model model-dir)      ; smart default => owned
            {:keys [model tokenizer]} mm
            chat (llm/render-chat [{:role "user" :content question :images 1}])
            enc (llm/encode tokenizer chat false)]
     (mx/force-gc!)
     (let [prompt (vec enc)]
       (println (str "== owned VLM through the backend/GF API ("
                     (count prompt) "-token rendered prompt)"))

       ;; ---- G1 ----
       (assert-true "G1: rendered prompt carries exactly one image marker"
                    (= 1 (count (filter #(= % vfwd/image-token-id) prompt))))

       ;; ---- G2: backend level ----
       (llm/init-cache! model)
       (let [logits (llm/forward-prefill model prompt {:images [img]})
             {:keys [offset rope-delta]} @(:cache model)]
         (assert-true (str "G2: cache cell holds the expanded prefix (offset "
                           offset " > " (count prompt) ", rope-delta " rope-delta ")")
                      (and (> offset (count prompt)) (integer? rope-delta)))
         (pr/let [ids (loop [i 0, lg logits, acc []]
                        (if (>= i 8)
                          acc
                          (let [id (mx/item (mx/argmax lg))]
                            (recur (inc i) (llm/forward-step model id) (conj acc id)))))
                  text (llm/decode tokenizer (js/Uint32Array.from (into-array ids)))]
           (llm/reset-cache! model)
           (mx/force-gc!)
           (println (str "    [G2] greedy continuation via forward-step: " (pr-str text)))
           (assert-true "G2: forward-step continues the image prefix (non-empty text)"
                        (pos? (count text)))

           ;; ---- G3 + G4: GF level ----
           (let [gf (core/make-llm-gf mm {:images [img]})
                 tr (p/simulate (dyn/with-key gf (rng/fresh-key 7)) [prompt 10])
                 score (mx/realize (:score tr))]
             (mx/force-gc!)
             (pr/let [gen-text (core/decode-trace tokenizer tr)]
               (println (str "    [G3] simulate answer: " (pr-str gen-text)
                             " (score " (.toFixed score 3) ")"))
               (assert-true "G3: simulate over the image prompt yields a finite score"
                            (js/isFinite score))
               (assert-true "G3: trace has answer-token sites"
                            (pos? (count gen-text)))
               (let [{tr2 :trace w :weight}
                     (p/generate (dyn/with-key gf (rng/fresh-key 11))
                                 [prompt 10] (:choices tr))
                     w* (mx/realize w)
                     s2 (mx/realize (:score tr2))
                     d-same (js/Math.abs (- w* s2))
                     d-cross (js/Math.abs (- w* score))]
                 (mx/force-gc!)
                 (println (str "    [G4] generate weight " (.toFixed w* 3)
                               " | same-run score " (.toFixed s2 3)
                               " | cross-run Δ " (.toFixed d-cross 3)
                               " (jitter, informational)"))
                 (assert-true (str "G4: fully-constrained weight == same-run score (Δ "
                                   (.toExponential d-same 2) " < 1e-3)")
                              (< d-same 1e-3))
                 (summary))))))))
   (pr/catch (fn [e]
               (swap! fail inc)
               (println "  FAIL (uncaught)" (or (.-message e) e))
               (summary)))))
