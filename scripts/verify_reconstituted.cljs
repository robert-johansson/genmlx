;; verify_reconstituted.cljs — GPU verification for a reconstituted
;; frozen-experts checkpoint (genmlx-e2my, companion to
;; reconstitute_moe_checkpoint.py).
;;
;; Loads the SOURCE checkpoint and the MERGED (reconstituted) checkpoint
;; through the same owned forward, prefills an identical fixed prompt, and
;; diffs the last-position logits: the merged model must differ from source
;; (the trained non-expert delta is present) with finite logits, and must
;; generate coherent greedy text. Together with the script's build-time
;; assertions and the file-level expert/vision byte-identity check, a nonzero
;; finite logit delta here attributes ALL behavioral change to the trained
;; non-expert weights.
;;
;; Usage (Thor: ONE GPU process at a time; run under the guard for safety):
;;   FLOOR_MB=25000 ~/genmlx-guarded-run.sh verify-reconstituted \
;;     env SOURCE_DIR=<source snapshot> MERGED_DIR=<reconstituted dir> \
;;     bunx --bun nbb@1.4.208 scripts/verify_reconstituted.cljs
(ns verify-reconstituted
  (:require [genmlx.llm.backend :as llm]
            [genmlx.mlx :as mx]
            [promesa.core :as p]))

(def source-dir (or (aget (.-env js/process) "SOURCE_DIR")
                    (throw (ex-info "SOURCE_DIR required" {}))))
(def merged-dir (or (aget (.-env js/process) "MERGED_DIR")
                    (throw (ex-info "MERGED_DIR required" {}))))

(def probe-prompt "The Generative Function Interface treats probabilistic programs as")

(defn- last-logits!
  "Load dir -> prefill probe -> materialized last-position logits [vocab]."
  [dir label]
  (p/let [m   (llm/load-model dir)
          ids (llm/encode (:tokenizer m) probe-prompt true)]
    (llm/init-cache! (:model m))
    (let [logits (llm/forward-prefill (:model m) ids)]
      (mx/materialize! logits)
      (println (str "  " label ": loaded " dir))
      (println (str "  " label " argmax: ") (mx/item (mx/argmax logits)))
      {:logits logits :model-map m})))

(defn -main []
  (println "== verify_reconstituted (genmlx-e2my) ==")
  (p/let [src (last-logits! source-dir "source")
          _   (mx/force-gc!)
          rec (last-logits! merged-dir "merged")
          diff     (mx/abs (mx/subtract (:logits rec) (:logits src)))
          max-diff (mx/item (mx/amax diff))
          mean-diff (mx/item (mx/mean diff))]
    (println "  max |logit delta|  :" max-diff)
    (println "  mean |logit delta| :" mean-diff)
    (when-not (js/isFinite max-diff)
      (throw (ex-info "NON-FINITE logit delta — reconstitution broken" {:max max-diff})))
    (when (zero? max-diff)
      (throw (ex-info "ZERO logit delta — trained non-expert delta missing" {})))
    (println (str "  DELTA PRESENT & FINITE — "
                  "experts bit-identical (file-level), so the delta is exactly the non-expert delta."))
    (p/let [g (llm/generate-text-raw+ (:model-map rec) "What is a probability distribution?"
                                      {:max-tokens 48})]
      (println "  greedy smoke (" (:n-tokens g) "tokens," (:gen-ms g) "ms ):")
      (println "   " (pr-str (:text g)))
      (println "== verify_reconstituted: PASS =="))))

(-> (-main)
    (p/catch (fn [e]
               (println "FAIL:" (.-message e))
               (println (.-stack e))
               (set! (.-exitCode js/process) 1))))
