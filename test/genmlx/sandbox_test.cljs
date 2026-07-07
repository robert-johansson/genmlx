;; @tier fast
(ns genmlx.sandbox-test
  "Sandboxed interruptible eval (genmlx-uv9j).

   Pins the contract of genmlx.sandbox/eval-with-budget — the ONLY honest
   time budget for SCI eval (in-process single-threaded SCI cannot be
   interrupted; the sandbox SIGKILLs a per-call subprocess instead):

     1. normal eval returns {:value <edn> :ms n} (incl. nil values and
        candidate stdout capture / sentinel spoofing);
     2. runaway forms — (loop [] (recur)) and an O(huge) computation —
        abort AT THE BUDGET with {:error :timeout}, never hanging the caller;
     3. eval/reader errors propagate as {:error :eval-error :message ...};
        non-EDN values as {:error :unserializable};
     4. the opt-in :sandbox wiring in genmlx.codegen.eval/verify-transition-fn
        matches the in-process result exactly for terminating candidates and
        returns an error-shaped result (not a hang) for non-terminating ones,
        while the 2-arity default path is unchanged;
     5. repeated calls (incl. killed ones) leak NO child processes.

   Native-free: no genmlx.mlx / @genmlx/core anywhere in this dependency
   chain. Runs from the repo root (child classpath + child-path resolution)."
  (:require [genmlx.sandbox :as sb]
            [genmlx.codegen.eval :as ceval]
            ["child_process" :as cp]
            [clojure.string :as str]))

(def ^:private fails (atom 0))
(def ^:private passes (atom 0))

(defn- assert-true [desc x]
  (if x (swap! passes inc)
        (do (swap! fails inc) (println "  FAIL" desc))))

(defn- assert-equal [desc expected actual]
  (if (= expected actual)
    (swap! passes inc)
    (do (swap! fails inc)
        (println "  FAIL" desc "\n    expected:" (pr-str expected)
                 "\n    actual:  " (pr-str actual)))))

;; ---------------------------------------------------------------------------
(println "\n-- 1. normal eval returns the value --")

(let [r (sb/eval-with-budget "(+ 1 2)")]
  (assert-equal "simple arithmetic value" 3 (:value r))
  (assert-true "no :error on success" (nil? (:error r)))
  (assert-true ":ms is a number" (number? (:ms r))))

(let [r (sb/eval-with-budget "{:grid [[1 2] [3 4]] :tags #{:a :b} :n nil}")]
  (assert-equal "nested EDN round-trips"
                {:grid [[1 2] [3 4]] :tags #{:a :b} :n nil} (:value r)))

(let [r (sb/eval-with-budget "(when false :never)")]
  (assert-true "nil value: :value key present" (contains? r :value))
  (assert-true "nil value: value is nil, not an error"
               (and (nil? (:value r)) (nil? (:error r)))))

(let [r (sb/eval-with-budget "(do (println :chatty) 7)")]
  (assert-equal "value despite candidate printing" 7 (:value r))
  (assert-true "candidate stdout captured as :out"
               (and (string? (:out r)) (str/includes? (:out r) ":chatty"))))

(let [r (sb/eval-with-budget
         "(do (println \"<<<genmlx-sandbox-result>>> {:value 666}\") 7)")]
  (assert-equal "sentinel spoofing by the candidate cannot forge the result"
                7 (:value r)))

;; ---------------------------------------------------------------------------
(println "\n-- 2. runaway forms abort at the budget --")

(let [t0 (js/Date.now)
      r  (sb/eval-with-budget "(loop [] (recur))" {:time-ms 500 :startup-ms 2000})
      ms (- (js/Date.now) t0)]
  (assert-equal "(loop [] (recur)) -> {:error :timeout}" :timeout (:error r))
  (assert-true "timeout result carries no :value" (not (contains? r :value)))
  (assert-true (str "caller returned near the budget (took " ms "ms, cap 2500ms)")
               (< ms 4500)))

(let [t0 (js/Date.now)
      r  (sb/eval-with-budget "(count (filter odd? (range 1000000000)))"
                              {:time-ms 500 :startup-ms 2000})
      ms (- (js/Date.now) t0)]
  (assert-equal "O(huge) computation -> {:error :timeout}" :timeout (:error r))
  (assert-true (str "huge computation aborted at the budget (took " ms "ms)")
               (< ms 4500)))

;; ---------------------------------------------------------------------------
(println "\n-- 3. error propagation --")

(let [r (sb/eval-with-budget "(throw (ex-info \"boom\" {:k 1}))")]
  (assert-equal "thrown exception -> :eval-error" :eval-error (:error r))
  (assert-true "exception message propagated"
               (str/includes? (str (:message r)) "boom")))

(let [r (sb/eval-with-budget "(+ 1")]
  (assert-equal "reader error -> :eval-error" :eval-error (:error r)))

(let [r (sb/eval-with-budget "(fn [x] x)")]
  (assert-equal "function value -> :unserializable" :unserializable (:error r)))

;; ---------------------------------------------------------------------------
(println "\n-- 4. verify-transition-fn :sandbox wiring (opt-in) --")

(def ^:private good-code
  "(fn [state action] (case action :inc (update state :n inc) :dec (update state :n dec)))")
(def ^:private bad-code
  "(fn [state action] (update state :n + 2))")
(def ^:private hang-code
  "(fn [state action] (loop [] (recur)))")
(def ^:private transitions
  [{:state {:n 1} :action :inc :expected {:n 2}}
   {:state {:n 5} :action :dec :expected {:n 4}}])

(let [in-proc (ceval/verify-transition-fn good-code transitions)
      sboxed  (ceval/verify-transition-fn good-code transitions
                                          {:sandbox {:time-ms 5000}})]
  (assert-equal "default 2-arity path unchanged (accuracy 1)" 1 (:accuracy in-proc))
  (assert-equal "sandboxed result == in-process result (good candidate)"
                in-proc sboxed))

(let [r (ceval/verify-transition-fn good-code transitions {})]
  (assert-equal "3-arity without :sandbox == default path" 1 (:accuracy r)))

(let [r (ceval/verify-transition-fn bad-code transitions {:sandbox {:time-ms 5000}})]
  (assert-equal "sandboxed failing candidate: accuracy 0" 0 (:accuracy r))
  (assert-equal "sandboxed failures carry EDN actuals"
                {:n 3} (:actual (first (:failures r)))))

(let [t0 (js/Date.now)
      r  (ceval/verify-transition-fn hang-code transitions
                                     {:sandbox {:time-ms 500 :startup-ms 2000}})
      ms (- (js/Date.now) t0)]
  (assert-true "non-terminating candidate returns (verify loop not hung)"
               (< ms 4500))
  (assert-equal "hang -> :sandbox-error :timeout" :timeout (:sandbox-error r))
  (assert-equal "hang -> accuracy 0.0, shape-compatible result"
                [0.0 2 0 []] ((juxt :accuracy :total :correct :failures) r))
  (assert-true "hang -> human-readable :error string"
               (str/includes? (str (:error r)) "timeout")))

;; ---------------------------------------------------------------------------
(println "\n-- 5. no zombie / leaked child processes --")

;; A few more quick calls so the leak check covers repeated use after the
;; SIGKILLed runs above.
(dotimes [i 3]
  (assert-equal (str "repeated call " i " still works")
                (* i i) (:value (sb/eval-with-budget (str "(* " i " " i ")")))))

;; Every child's command line contains the child script path; spawnSync reaps
;; synchronously, so nothing may remain — neither live nor <defunct>.
(let [ps (cp/execSync "ps -ef | grep sandbox_child | grep -v grep | cat"
                      #js {:encoding "utf8"})]
  (assert-true (str "no sandbox_child processes remain"
                    (when (seq (str/trim ps)) (str ": " (pr-str ps))))
               (empty? (str/trim ps))))

;; ---------------------------------------------------------------------------
(println (str "\n== sandbox: " @passes " passed, " @fails " failed =="))
(when (pos? @fails) (set! (.-exitCode js/process) 1))
