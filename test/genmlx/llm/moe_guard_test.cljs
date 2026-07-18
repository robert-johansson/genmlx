;; @tier slow
;; genmlx-5luk + genmlx-2sh6: the native MoE forward (256-expert gather_mm /
;; mlx_qwen35_moe_forward) hard-crashes the process with an uncatchable SIGTRAP
;; on Metal (e.g. Qwen3.6-35B-A3B-4bit). On CUDA the SAME native forward is
;; verified safe (mlx-2h4l), so the refusal is PLATFORM-GATED:
;;   - Metal: load-model must REFUSE a native-MoE model with a CATCHABLE ex-info
;;     BEFORE any native load, so callers get an actionable error not a process kill.
;;   - CUDA: native MoE (qwen3_5_moe, qwen3_next) is allowed — it is the default
;;     forward for the 80B Qwen3-Coder-Next. (The native-load smoke lives in
;;     llm_qwen3_next_native_test.cljs; here we only assert the guard predicate.)
;; This test verifies the platform-gated predicate and, on Metal, that load-model
;; rejects (without loading the weights).
;;
;; @tier slow because the integration case touches a model directory and uses
;; promesa I/O; it does NO GPU work (the guard fires before .load), and the
;; predicate cases run regardless of which models are present.
(ns genmlx.llm.moe-guard-test
  (:require [genmlx.llm.backend :as llm]
            [genmlx.mlx :as mx]
            [promesa.core :as p]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn assert-true [label v]
  (if v
    (do (swap! pass inc)  (println (str "  PASS: " label)))
    (do (swap! fail inc)  (println (str "  FAIL: " label)))))

(def ^:private metal? (mx/metal-is-available?))
(def model-dir (str (.-HOME js/process.env) "/.cache/models"))
(def moe-path  (str model-dir "/Qwen3.6-35B-A3B-4bit"))

(defn- file-present? [path]
  (try ((.-existsSync (js/require "fs")) path)
       (catch :default _ false)))

(println (str "\n== platform: " (if metal? "Metal (native MoE refused)"
                                    "CUDA / non-Metal (native MoE allowed)") " =="))

;; ---------------------------------------------------------------------------
;; 1. Predicate (reads only the platform via mx/metal-is-available?, no model).
;;    Platform-gated (genmlx-2sh6).
;; ---------------------------------------------------------------------------
(println "\n== unsupported-native-moe? predicate ==")
(doseq [mt ["qwen3_5_moe" "qwen3_next"]]
  (if metal?
    (do
      (assert-true (str mt " refused by default on Metal (empty opts)")
                   (llm/unsupported-native-moe? mt {}))
      (assert-true (str mt " refused by default on Metal (nil opts)")
                   (llm/unsupported-native-moe? mt nil))
      (assert-true (str mt " allowed on Metal with {:allow-native-moe? true}")
                   (not (llm/unsupported-native-moe? mt {:allow-native-moe? true}))))
    (do
      (assert-true (str mt " NOT refused on CUDA (empty opts)")
                   (not (llm/unsupported-native-moe? mt {})))
      (assert-true (str mt " NOT refused on CUDA (nil opts)")
                   (not (llm/unsupported-native-moe? mt nil)))
      (assert-true (str mt " NOT refused on CUDA with {:allow-native-moe? true}")
                   (not (llm/unsupported-native-moe? mt {:allow-native-moe? true}))))))

;; Dense families are never refused on either platform (no over-blocking).
(doseq [mt ["qwen3_5" "qwen3" "gemma4"]]
  (assert-true (str "dense " mt " NOT refused (any platform)")
               (not (llm/unsupported-native-moe? mt {}))))

;; ---------------------------------------------------------------------------
;; 2. Integration — Metal only: load-model rejects before the native load.
;;    On CUDA the guard does not fire, so we do NOT attempt to load 40GB here
;;    (the native-load smoke is llm_qwen3_next_native_test.cljs).
;; ---------------------------------------------------------------------------
(println "\n== load-model guard (Qwen3.6-35B-A3B-4bit) ==")
(-> (if (and metal? (file-present? (str moe-path "/config.json")))
      (-> (p/let [_ (llm/load-model moe-path)]
            ;; Should never reach here — the guard must reject first on Metal.
            (assert-true "load-model on native MoE (Metal) should have thrown" false))
          (p/catch
           (fn [e]
             (let [d (ex-data e)]
               (assert-true "rejected (did not SIGTRAP) with an ex-info" (some? d))
               (assert-true ":genmlx/error = :unsupported-model-type"
                            (= :unsupported-model-type (:genmlx/error d)))
               (assert-true "message names the bean genmlx-5luk"
                            (boolean (re-find #"genmlx-5luk" (ex-message e))))
               (println (str "  (caught: " (ex-message e) ")"))))))
      (do (println (str "  SKIP: "
                        (cond
                          (not metal?) "non-Metal backend — native MoE is allowed (see llm_qwen3_next_native_test)"
                          :else "Qwen3.6-35B-A3B-4bit/config.json not present")))
          (p/resolved nil)))
    (p/finally
     (fn [& _]
       (println (str "\n== " @pass " passed, " @fail " failed =="))
       (when (pos? @fail) (set! (.-exitCode js/process) 1)))))
