(ns genmlx.sandbox-child
  "Child-process evaluator for genmlx.sandbox (genmlx-uv9j). NOT a library —
   this script is spawned by genmlx.sandbox/eval-with-budget as
   `<runtime> <nbb-cli> src/genmlx/sandbox_child.cljs` with the form string on
   stdin. Do not require it from application code (a main-guard makes an
   accidental require inert, but it exports nothing useful).

   Protocol (must stay in sync with genmlx.sandbox):
     stdin:  the entire input is ONE ClojureScript form string
     stdout: whatever the form prints, then \\n<<<genmlx-sandbox-result>>>\\n
             followed by exactly one EDN map —
               {:value <edn> :ms n}
             | {:error :eval-error :message s :ms n}
             | {:error :unserializable :message s :ms n}

   Evaluation uses nbb.core/load-string: SCI underneath, but classpath-aware,
   so the form may (require '[genmlx.codegen.eval ...]) etc. against the
   repo's nbb.edn paths (child cwd = repo root by convention). Only
   EDN-round-trippable values cross back; the round-trip is checked HERE so
   the parent never has to guess why a read failed.

   The child exits explicitly (process.exit) after writing the result with a
   synchronous fd write: a candidate that starts timers (js/setInterval)
   must not keep this process alive, and stdout must not be truncated by the
   exit. A candidate that never returns is not handled here at all — it
   CANNOT be (single-threaded SCI); the parent's spawnSync timeout SIGKILLs
   this whole process instead."
  (:require [nbb.core :as nbb]
            [clojure.edn :as edn]
            [promesa.core :as p]))

(def ^:private fs (js/require "fs"))

;; Must match genmlx.sandbox/sentinel.
(def ^:private sentinel "<<<genmlx-sandbox-result>>>")

(defn- emit-and-exit!
  "Synchronously write the sentinel + one EDN result map to fd 1, then exit.
   fs.writeSync (not process.stdout.write) so the payload cannot be lost in
   an async buffer when process.exit tears the runtime down."
  [m]
  (.writeSync fs 1 (str "\n" sentinel "\n" (pr-str m) "\n"))
  (.exit js/process 0))

(defn- result-for
  "Build the success-or-unserializable result map for an evaluated value.
   EDN gatekeeping: the value must survive pr-str -> edn/read-string."
  [v ms]
  (let [printed (try (pr-str v) (catch :default _ ::unprintable))
        value   (when-not (= ::unprintable printed)
                  (try [(edn/read-string printed)]
                       (catch :default _ nil)))]
    (cond
      (= ::unprintable printed)
      {:error :unserializable :message "value cannot be printed" :ms ms}

      (nil? value)
      {:error :unserializable
       :message (str "value is not EDN-round-trippable: "
                     (subs printed 0 (min 500 (count printed))))
       :ms ms}

      :else {:value (first value) :ms ms})))

(defn- evaluate! []
  (let [code (.readFileSync fs 0 "utf8")
        t0   (js/Date.now)]
    (-> (nbb/load-string code)
        (p/then (fn [v] (emit-and-exit! (result-for v (- (js/Date.now) t0)))))
        (p/catch (fn [e]
                   (emit-and-exit!
                    {:error   :eval-error
                     :message (str (or (some-> e .-message) e))
                     :ms      (- (js/Date.now) t0)}))))))

;; Main guard: only read stdin + eval when this file is the invoked script,
;; so an accidental (require '[genmlx.sandbox-child]) can't block on stdin.
(when (= (nbb/invoked-file) nbb/*file*)
  (evaluate!))
