;; @tier slow
;; genmlx-5luk: the native qwen3_5_moe forward (256-expert gather_mm /
;; mlx_qwen35_moe_forward) hard-crashes the process with an uncatchable SIGTRAP
;; on real checkpoints (e.g. Qwen3.6-35B-A3B-4bit). load-model must REFUSE such a
;; model with a CATCHABLE ex-info BEFORE any native load, so callers get an
;; actionable error instead of a process kill. This test verifies the guard
;; predicate and that load-model rejects (without loading the 19GB weights).
;;
;; @tier slow because the integration case touches a model directory and uses
;; promesa I/O; it does NO GPU work (the guard fires before .loadModel), and the
;; predicate cases run regardless of which models are present.
(ns genmlx.llm-moe-guard-test
  (:require [genmlx.llm.backend :as llm]
            [promesa.core :as p]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn assert-true [label v]
  (if v
    (do (swap! pass inc)  (println (str "  PASS: " label)))
    (do (swap! fail inc)  (println (str "  FAIL: " label)))))

(def model-dir (str (.-HOME js/process.env) "/.cache/models"))
(def moe-path  (str model-dir "/Qwen3.6-35B-A3B-4bit"))

(defn- file-present? [path]
  (try ((.-existsSync (js/require "fs")) path)
       (catch :default _ false)))

;; ---------------------------------------------------------------------------
;; 1. Pure predicate — no model needed
;; ---------------------------------------------------------------------------
(println "\n== unsupported-native-moe? predicate ==")
(assert-true "qwen3_5_moe refused by default (empty opts)"
             (llm/unsupported-native-moe? "qwen3_5_moe" {}))
(assert-true "qwen3_5_moe refused with nil opts"
             (llm/unsupported-native-moe? "qwen3_5_moe" nil))
(assert-true "qwen3_5_moe allowed with {:allow-native-moe? true}"
             (not (llm/unsupported-native-moe? "qwen3_5_moe" {:allow-native-moe? true})))
(assert-true "dense qwen3_5 NOT refused (no over-blocking)"
             (not (llm/unsupported-native-moe? "qwen3_5" {})))
(assert-true "qwen3 NOT refused"
             (not (llm/unsupported-native-moe? "qwen3" {})))
(assert-true "gemma4 NOT refused"
             (not (llm/unsupported-native-moe? "gemma4" {})))

;; ---------------------------------------------------------------------------
;; 2. Integration — load-model rejects before the native load (no SIGTRAP)
;; ---------------------------------------------------------------------------
(println "\n== load-model guard (Qwen3.6-35B-A3B-4bit) ==")
(-> (if (file-present? (str moe-path "/config.json"))
      (-> (p/let [_ (llm/load-model moe-path)]
            ;; Should never reach here — the guard must reject first.
            (assert-true "load-model on qwen3_5_moe should have thrown" false))
          (p/catch
           (fn [e]
             (let [d (ex-data e)]
               (assert-true "rejected (did not SIGTRAP) with an ex-info" (some? d))
               (assert-true ":genmlx/error = :unsupported-model-type"
                            (= :unsupported-model-type (:genmlx/error d)))
               (assert-true ":model-type = :qwen3_5_moe"
                            (= :qwen3_5_moe (:model-type d)))
               (assert-true "message names the bean genmlx-5luk"
                            (boolean (re-find #"genmlx-5luk" (ex-message e))))
               (println (str "  (caught: " (ex-message e) ")"))))))
      (do (println "  SKIP: Qwen3.6-35B-A3B-4bit/config.json not present")
          (p/resolved nil)))
    (p/finally
     (fn [& _]
       (println (str "\n== " @pass " passed, " @fail " failed =="))
       (when (pos? @fail) (set! (.-exitCode js/process) 1)))))
