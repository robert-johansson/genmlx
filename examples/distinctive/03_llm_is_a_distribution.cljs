(ns demo-llm-is-a-distribution
  "DISTINCTIVE FEATURE: an LLM is an ACTUAL generative function.

   The SAME ten-protocol Generative Function Interface (GFI) that drives a
   Bernoulli coin and a Gaussian also drives a multi-billion-parameter language
   model. p/simulate samples it. p/generate conditions it and returns an
   importance log-weight = log p(constraint). There is no special 'LLM API' —
   an LLM is not a special object, it is a distribution like any other."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as llm-core]
            [promesa.core :as pr])
  (:require-macros [genmlx.gen :refer [gen]]))

(def model-path
  (str (.-HOME js/process.env) "/.cache/models/qwen3-0.6b-mlx-bf16"))

(defn fixed [v] (.toFixed v 4))

;; ── Two ordinary tiny generative functions ───────────────────────────────────
;; A coin and a Gaussian. Each has ONE trace site :x. Standard gen functions.
(def coin     (gen [] (trace :x (dist/bernoulli 0.5))))
(def gaussian (gen [] (trace :x (dist/gaussian 0 1))))

(println "Loading the language model (this is the same kind of object as a coin)...")

(pr/let
  [m       (llm/load-model model-path)
   _       (println "Model loaded.\n")
   llm-gf  (llm-core/make-llm-gf m)        ; <- an LLM as a generative function
   tok     (:tokenizer m)
   prompt  "The best programming language is"
   raw-ids (llm/encode tok prompt)
   ids     (vec raw-ids)
   ;; first token id of " Clojure" — what we will CONSTRAIN the LLM to say
   clj-ids (llm/encode tok " Clojure")]

  (let [clj-id   (first (vec clj-ids))
        ;; max 8 new tokens — keep it fast
        llm-args [ids 8]]

    (println "Three generative functions, three wildly different distributions:")
    (println "  coin     : (gen [] (trace :x (dist/bernoulli 0.5)))   ~2 outcomes")
    (println "  gaussian : (gen [] (trace :x (dist/gaussian 0 1)))    continuous R")
    (println (str "  llm-gf   : Qwen3-0.6B over prompt \"" prompt "\"  ~151k-token vocab"))
    (println (str "             (" (count ids) " prompt tokens, generating up to 8 new tokens)\n"))

    ;; ════════════════════════════════════════════════════════════════════════
    ;; (b) p/simulate — the SAME operation samples a free draw from each prior.
    ;;     trace :score is log p(sample) in every case.
    ;; ════════════════════════════════════════════════════════════════════════
    (println "==================================================================")
    (println "(b) p/simulate — one free sample + its log p, from each GF")
    (println "==================================================================")

    (let [coin-tr (p/simulate (dyn/auto-key coin) [])
          gn-tr   (p/simulate (dyn/auto-key gaussian) [])
          llm-tr  (p/simulate llm-gf llm-args)]

      (println "  coin     sample :x =" (mx/item (cm/get-value (cm/get-submap (:choices coin-tr) :x)))
               "  log p =" (fixed (mx/item (:score coin-tr))))
      (println "  gaussian sample :x =" (fixed (mx/item (cm/get-value (cm/get-submap (:choices gn-tr) :x))))
               "  log p =" (fixed (mx/item (:score gn-tr))))

      ;; decode the LLM trace's sampled tokens back to text
      (pr/let [llm-text (llm-core/decode-trace tok llm-tr)]
        (println (str "  llm      sample text = \"" llm-text "\""))
        (println "  llm      log p(sequence) =" (fixed (mx/item (:score llm-tr))))
        (println "  (same :score field on all three — it is just log probability)\n")

        ;; ══════════════════════════════════════════════════════════════════════
        ;; (c) p/generate — the SAME operation CONSTRAINS the first outcome of
        ;;     each GF. :weight is the importance log-weight = log p(constraint).
        ;; ══════════════════════════════════════════════════════════════════════
        (println "==================================================================")
        (println "(c) p/generate — constrain the first outcome; read log p(constraint)")
        (println "==================================================================")

        (let [;; coin: force :x = 1  (heads)
              coin-res (p/generate (dyn/auto-key coin) []
                                   (cm/set-value cm/EMPTY :x (mx/scalar 1.0)))
              ;; gaussian: force :x = 1.5
              gn-res   (p/generate (dyn/auto-key gaussian) []
                                   (cm/set-value cm/EMPTY :x (mx/scalar 1.5)))
              ;; llm: force the FIRST generated token :t0 to be " Clojure"
              llm-res  (p/generate llm-gf llm-args
                                   (cm/set-value cm/EMPTY :t0 (mx/scalar clj-id mx/int32)))]

          (println "  coin     | force :x = 1 (heads)")
          (println "           log p(constraint) =" (fixed (mx/item (:weight coin-res)))
                   "  (= log 0.5)")
          (println "  gaussian | force :x = 1.5")
          (println "           log p(constraint) =" (fixed (mx/item (:weight gn-res)))
                   "  (= log N(1.5;0,1))")

          (pr/let [llm-ctext (llm-core/decode-trace tok (:trace llm-res))]
            (println (str "  llm      | force :t0 = \" Clojure\" (token id " clj-id ")"))
            (println (str "           generated text = \"" llm-ctext "\""))
            (println "           log p(constraint) =" (fixed (mx/item (:weight llm-res)))
                     "  (= log p(\" Clojure\" | prompt))\n")

            ;; ════════════════════════════════════════════════════════════════════
            ;; (d) The point.
            ;; ════════════════════════════════════════════════════════════════════
            (println "==================================================================")
            (println "(d) Same interface. Same operations. Three distributions.")
            (println "==================================================================")
            (println "  - p/simulate sampled a coin flip, a real number, and a")
            (println "    sentence — each returning a Trace with a :score = log p.")
            (println "  - p/generate conditioned the first outcome of all three and")
            (println "    returned an importance log-weight = log p(constraint).")
            (println "  - The LLM never used a special API: make-llm-gf produced an")
            (println "    ordinary DynamicGF whose trace sites :t0..:t7 sample from")
            (println "    dist/categorical(logits). An LLM is not a special object —")
            (println "    it is a generative function, like the coin and the Gaussian.")
            (println "\nDone.")))))))
