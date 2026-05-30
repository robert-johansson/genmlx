(ns demo-grammar-conditioning
  "DISTINCTIVE FEATURE: conditioning a sampled value on a GRAMMAR is a
   first-class GFI operation — and it rides the SAME Ring-style with-handler
   middleware that GenMLX uses for analytical/conjugacy inference.

   grammar/compile-constraint turns a regex into a DFA + per-state token masks.
   grammar/wrap-grammar is ring middleware: it intercepts EVERY dist/categorical
   draw, adds a -inf mask to the logits so only grammar-valid next-tokens have
   support, samples, then advances the DFA. grammar/constrain just installs that
   middleware via (dispatch/with-handler gf (wrap-grammar h/generate-transition c)).

   The exact same dispatch/with-handler mechanism carries analytical conjugacy
   updates — so grammar conditioning is NOT an LLM feature bolted on the side.
   It is the generic handler-middleware contract applied to categorical draws.
   It works on ANY gf that uses dist/categorical."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.dispatch :as dispatch]
            [genmlx.handler :as h]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as llm-core]
            [genmlx.llm.grammar :as grammar]
            [promesa.core :as pr])
  (:require-macros [genmlx.gen :refer [gen]]))

(def model-path
  (str (.-HOME js/process.env) "/.cache/models/qwen3-0.6b-mlx-bf16"))

;; Decode the generated suffix of a constrained trace back to text. A grammar
;; trace's :retval is the full context vector (prompt-ids ++ generated ids);
;; strip the prompt and EOS, map ids through the decoded token-index.
(defn decode-generated [constraint prompt-len trace]
  (let [{:keys [token-index eos-id]} constraint]
    (->> (subvec (:retval trace) prompt-len)
         (remove #(= % eos-id))
         (map #(nth token-index %))
         (apply str))))

(println "Loading the language model...")

(pr/let
  [m       (llm/load-model model-path)
   _       (println "Model loaded.\n")
   tok     (:tokenizer m)
   ;; Prompt: "Answer with yes or no: Is 7 a prime number? " — encoded raw.
   yn-raw  (llm/encode tok "Answer yes or no: Is 7 prime? ")
   yn-ids  (vec yn-raw)
   ;; Prompt for the numeric grammar.
   num-raw (llm/encode tok "Pick a number: ")
   num-ids (vec num-raw)]

  ;; ════════════════════════════════════════════════════════════════════════
  ;; (a) Compile a regex constraint to a DFA + precomputed token masks.
  ;;     Grammar #1: an enumerated yes/no set.  Grammar #2: a 1-3 digit number.
  ;; ════════════════════════════════════════════════════════════════════════
  (println "==================================================================")
  (println "(a) compile-constraint : regex --> DFA + per-state token masks")
  (println "==================================================================")

  (let [yn-c  (grammar/compile-constraint tok "(yes|no)" {})
        num-c (grammar/compile-constraint tok "[0-9]{1,3}" {})]

    (println (str "  regex \"(yes|no)\"   -> DFA with " (count (get-in yn-c [:dfa :alive]))
                  " alive states, accept=" (sort (get-in yn-c [:dfa :accept]))
                  ", masks precomputed? " (some? (:masks yn-c))))
    (println (str "  regex \"[0-9]{1,3}\" -> DFA with " (count (get-in num-c [:dfa :alive]))
                  " alive states, accept=" (sort (get-in num-c [:dfa :accept]))
                  ", masks precomputed? " (some? (:masks num-c))))

    ;; Sanity-check the DFA decides language membership (pure, no model needed).
    (println "  DFA membership: (yes|no) accepts \"yes\"="
             (grammar/dfa-accepts? (:dfa yn-c) "yes")
             " \"no\"=" (grammar/dfa-accepts? (:dfa yn-c) "no")
             " \"maybe\"=" (grammar/dfa-accepts? (:dfa yn-c) "maybe"))
    (println "  DFA membership: [0-9]{1,3} accepts \"42\"="
             (grammar/dfa-accepts? (:dfa num-c) "42")
             " \"7\"=" (grammar/dfa-accepts? (:dfa num-c) "7")
             " \"1234\"=" (grammar/dfa-accepts? (:dfa num-c) "1234")
             " \"x\"=" (grammar/dfa-accepts? (:dfa num-c) "x"))

    ;; ════════════════════════════════════════════════════════════════════════
    ;; (a') apply-mask : the masking primitive. At the DFA start state, apply the
    ;;      grammar mask to a real logit vector from the model and show that
    ;;      grammar-INVALID next-tokens are driven to -inf, valid ones survive.
    ;; ════════════════════════════════════════════════════════════════════════
    (println "\n==================================================================")
    (println "(a') apply-mask : invalid next-tokens get -inf, valid ones survive")
    (println "==================================================================")

    (pr/let [num-logits (llm/forward-pass (:model m) num-ids)
             _          (mx/eval! num-logits)]
      (let [start     (get-in num-c [:dfa :start])
            masked    (grammar/apply-mask num-c start num-logits)
            _         (mx/eval! masked)
            token-idx (:token-index num-c)
            vocab     (count token-idx)
            mask-vec  (grammar/get-mask num-c start)
            valid-ids (filterv #(zero? (aget mask-vec %)) (range vocab))
            ;; read logits to a JS typed array once, then scan in JS (fast).
            logit-arr (.toFloat32 num-logits)
            ;; pick the highest-logit GRAMMAR-INVALID token to show it gets -inf
            invalid-pick (loop [i 0, best -1, best-v js/Number.NEGATIVE_INFINITY]
                           (if (>= i vocab)
                             best
                             (if (and (not (zero? (aget mask-vec i)))
                                      (> (aget logit-arr i) best-v))
                               (recur (inc i) i (aget logit-arr i))
                               (recur (inc i) best best-v))))
            ;; the model's UNconstrained argmax vs the grammar-constrained argmax
            free-pick (mx/item (mx/argmax num-logits))
            grm-pick  (mx/item (mx/argmax masked))]
        (println (str "  vocab size = " vocab
                      ";  grammar-valid first tokens = " (count valid-ids)
                      " (the digits 0-9)"))
        (println (str "  valid first-token strings: "
                      (pr-str (mapv #(nth token-idx %) (take 12 valid-ids)))))
        (println (str "  unconstrained argmax token   id=" free-pick
                      "  -> " (pr-str (nth token-idx free-pick))
                      "  (logit " (.toFixed (mx/item (mx/index num-logits free-pick)) 3) ")"))
        (println (str "  highest invalid token        id=" invalid-pick
                      "  -> " (pr-str (nth token-idx invalid-pick))
                      "  (logit " (.toFixed (mx/item (mx/index num-logits invalid-pick)) 3) ")"))
        (println (str "  ...its logit AFTER apply-mask = "
                      (mx/item (mx/index masked invalid-pick))
                      "   (-inf  => the grammar removed it from support)"))
        (println (str "  grammar-constrained argmax   id=" grm-pick
                      "  -> " (pr-str (nth token-idx grm-pick))
                      "  (a valid digit)"))
        (println "  --> apply-mask added -inf to invalid logits; the categorical's")
        (println "      support collapses onto exactly the grammar-legal tokens.")

        ;; ════════════════════════════════════════════════════════════════════════
        ;; (b) constrain : wrap the gf so its dist/categorical draws are masked.
        ;;     Then p/simulate. Constrained output MATCHES the grammar; the
        ;;     unconstrained gf over the same prompt need not.
        ;; ════════════════════════════════════════════════════════════════════════
        (println "\n==================================================================")
        (println "(b) constrain + p/simulate : output is guaranteed grammar-valid")
        (println "==================================================================")

        (let [free-gf  (llm-core/make-llm-gf m)              ; unconstrained LLM gf
              num-gf   (grammar/constrain (llm-core/make-llm-gf m) num-c)  ; masked
              yn-gf    (grammar/constrain (llm-core/make-llm-gf m) yn-c)   ; masked
              num-re   #"[0-9]{1,3}"
              yn-re    #"(yes|no)"]

          ;; Unconstrained simulate over the numeric prompt: free-form text.
          (pr/let [free-tr (p/simulate free-gf [num-ids 6])
                   free-txt (llm-core/decode-trace tok free-tr)]
            (println (str "  UNCONSTRAINED simulate(\"Pick a number: \") -> "
                          (pr-str free-txt)))
            (println (str "    matches [0-9]{1,3}? "
                          (some? (re-matches num-re free-txt))
                          "   (free generation is under no obligation to)"))

            ;; Constrained simulate over the numeric prompt: a valid number.
            (pr/let [num-tr (p/simulate num-gf [num-ids 6])]
              (let [num-txt (decode-generated num-c (count num-ids) num-tr)]
                (println (str "  CONSTRAINED   simulate w/ [0-9]{1,3}      -> "
                              (pr-str num-txt)))
                (println (str "    matches [0-9]{1,3}? "
                              (some? (re-matches num-re num-txt))
                              "   log p(seq) = "
                              (.toFixed (mx/item (:score num-tr)) 3)))

                ;; Constrained simulate over the yes/no prompt: a valid choice.
                (pr/let [yn-tr (p/simulate yn-gf [yn-ids 4])]
                  (let [yn-txt (decode-generated yn-c (count yn-ids) yn-tr)]
                    (println (str "  CONSTRAINED   simulate w/ (yes|no)        -> "
                                  (pr-str yn-txt)))
                    (println (str "    matches (yes|no)?  "
                                  (some? (re-matches yn-re yn-txt))
                                  "   log p(seq) = "
                                  (.toFixed (mx/item (:score yn-tr)) 3)))

                    ;; A few more numeric draws to show variety within the grammar.
                    (pr/let [a (p/simulate num-gf [num-ids 6])
                             b (p/simulate num-gf [num-ids 6])
                             c (p/simulate num-gf [num-ids 6])]
                      (let [draws (mapv #(decode-generated num-c (count num-ids) %) [a b c])]
                        (println (str "  three more [0-9]{1,3} draws: " (pr-str draws)
                                      "  all valid? "
                                      (every? #(some? (re-matches num-re %)) draws)))

                        ;; ════════════════════════════════════════════════════════
                        ;; (c) The generality point — explicit.
                        ;; ════════════════════════════════════════════════════════
                        (println "\n==================================================================")
                        (println "(c) Why this is generic, not an LLM trick")
                        (println "==================================================================")
                        (println "  grammar/constrain is literally:")
                        (println "    (dispatch/with-handler gf")
                        (println "      (wrap-grammar h/generate-transition constraint))")
                        (println "  wrap-grammar is Ring-style middleware that intercepts ONLY")
                        (println "  :categorical draws, applies the DFA mask to the logits, then")
                        (println "  delegates to the base handler transition and advances the DFA.")
                        (println "  This is the SAME dispatch/with-handler contract GenMLX uses to")
                        (println "  install analytical/conjugacy middleware — so a grammar is just")
                        (println "  another handler transform over categorical, not LLM-specific.")

                        ;; Prove it on a NON-LLM categorical gf: a 5-way die whose
                        ;; categorical draw is masked by the same constrain machinery.
                        (println "\n  Proof on a plain (non-LLM) categorical gf:")
                        (let [;; uniform-ish logits over a tiny 5-symbol vocab
                              die     (dyn/auto-key
                                        (gen [] (trace :t0 (dist/categorical
                                                             (mx/array [0.0 0.0 0.0 0.0 0.0])))))
                              ;; token-index that maps ids 0..4 to single chars "0".."4";
                              ;; restrict the categorical to the literal symbol "3".
                              tiny-idx ["0" "1" "2" "3" "4"]
                              die-c    {:dfa     (grammar/compile-regex "3")
                                        :token-index tiny-idx
                                        :eos-id  -1   ; no EOS in this toy vocab
                                        :masks   nil}
                              die-gf   (grammar/constrain die die-c)]
                          (let []
                            (pr/let [d1 (p/simulate die-gf [])
                                     d2 (p/simulate die-gf [])
                                     d3 (p/simulate die-gf [])]
                              (let [picks (mapv #(mx/item (cm/get-value
                                                            (cm/get-submap (:choices %) :t0)))
                                                [d1 d2 d3])]
                                (println (str "    die :t0 draws under regex \"3\" = " (pr-str picks)))
                                (println (str "    all equal symbol-id 3? " (every? #(= 3 %) picks)
                                              "  (the mask zeroed every non-\"3\" symbol)"))
                                (println "\nDone.")))))))))))))))))
