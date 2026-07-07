(ns genmlx.sandbox
  "Sandboxed, interruptible evaluation of ClojureScript form strings with a
   wall-clock budget (genmlx-uv9j).

   ## The problem

   A synthesized candidate (free-form CLJS evaluated via SCI in the
   propose-verify loop, e.g. arc3-solver-u65s) can non-terminate or run
   O(huge) over a grid. SCI is single-threaded and has no interrupt or
   op-budget hook: once `sci/eval-string` (or `nbb.core/load-string`) enters a
   hot loop, NOTHING on that thread can stop it — no setTimeout fires, no
   exception can be injected. In-process eval is therefore uninterruptible by
   construction; the only honest budget is process-level.

   ## Isolation mechanism chosen: per-call subprocess, killed on timeout

   `eval-with-budget` spawns a fresh evaluator subprocess per call via
   Node's `child_process.spawnSync` with `:timeout` + `:killSignal SIGKILL`:

     - the form string is fed over stdin; the child
       (src/genmlx/sandbox_child.cljs) evaluates it with `nbb.core/load-string`
       (SCI underneath, classpath-aware) and prints one EDN result map after a
       sentinel line;
     - on timeout the OS kills the child mid-eval — the one thing in-process
       SCI cannot do — and the caller gets {:error :timeout};
     - spawnSync reaps the child synchronously, so no zombies are possible.

   Alternatives investigated and rejected on this box (Bun + nbb 1.4.208):
     - Bun Worker + terminate(): js/Worker exists, but the worker entry must
       be a JS file that can import SCI/nbb — nbb is not in the repo's
       node_modules (it lives at an unstable bunx cache path), and a
       synchronous wait would additionally need SharedArrayBuffer+Atomics.
       Strictly more moving parts for the same guarantee.
     - Reusable evaluator subprocess (spawn once, forms over stdin): needs a
       synchronous mid-flight read of the child's stdout, which neither Node
       nor Bun offers portably. Measured cold-spawn cost is ~150-350 ms on
       this box (nbb boots in ~130 ms under Bun), so per-call spawn is cheap
       enough that reuse buys nothing. If spawn ever becomes seconds-slow,
       revisit with an async protocol.
     - SCI op budget: SCI has no step/op-limit feature, so :time-ms is the
       only budget dimension (the bean's op-budget is 'optional if cheap' —
       it is not cheap).
     - DO NOT spawn through `bunx --bun nbb@...`: bunx launches the real
       evaluator as a GRANDCHILD, so the timeout SIGKILL hits only the
       wrapper and orphans the runaway at 99% CPU (observed live on this
       box). The default command therefore re-invokes process.argv[0]
       (the running bun/node binary) on process.argv[1] (the running nbb
       CLI script) directly, so the direct child IS the evaluator.

   ## What CAN and CANNOT be interrupted

   CAN:    anything the child does — infinite loops, huge SCI computations,
           runaway allocation, lingering timers (the child also calls
           process.exit explicitly after printing its result, so a candidate
           that starts a js/setInterval cannot keep the child alive).
   CANNOT: the parent's own thread. This namespace never evaluates the form
           in-process; if you call plain sci/eval-string yourself you are
           back to uninterruptible. Also, the kill is SIGKILL: the candidate
           gets no cleanup — acceptable because the child is stateless and
           its only observable effect is the result printed on stdout.

   ## Budget semantics (honest fine print)

   The subprocess timeout is (:time-ms + :startup-ms) because interpreter
   startup happens inside the same process and spawnSync cannot observe the
   moment eval begins. So :time-ms is a floor on the eval budget and
   (:time-ms + :startup-ms) is the hard wall-clock ceiling. On success, :ms
   in the result is the child-measured eval-only time.

   ## Serialization boundary

   Values cross the process boundary as EDN text. Only EDN-round-trippable
   values survive: maps/vectors/sets/lists of keywords, symbols, strings,
   numbers, booleans, nil. Functions, JS objects, records, MLX arrays do NOT —
   the child detects the failed round-trip and returns
   {:error :unserializable}. Check (:error r), not (:value r) truthiness:
   a form legitimately evaluating to nil yields {:value nil :ms n}.

   ## Sync by design

   'Sync math, async events' (CLAUDE.md): this subprocess boundary is genuine
   I/O, but the consumer is the synchronous verify loop
   (genmlx.codegen.eval/verify-transition-fn), so a blocking wait is the
   right shape and spawnSync provides it natively. An async
   (promesa-returning) variant via child_process.execFile with the same
   timeout/killSignal options would be a ~15-line addition if an async
   consumer ever needs one; deferred until then.

   ## Assumptions

   - The parent runs under nbb (process.argv = [bun-or-node, nbb-cli, ...]);
     otherwise pass :cmd explicitly.
   - cwd is the repo root (the repo-wide convention), so the default
     :child-path \"src/genmlx/sandbox_child.cljs\" and the child's classpath
     requires (nbb.edn paths) resolve. Both are overridable per call."
  (:require ["child_process" :as cp]
            [clojure.edn :as edn]
            [clojure.string :as str]))

;; Must match the sentinel in src/genmlx/sandbox_child.cljs. The child prints
;; it AFTER eval completes, so taking the LAST occurrence in stdout is robust
;; even against a candidate that prints the sentinel itself.
(def ^:private sentinel "<<<genmlx-sandbox-result>>>")

(def ^:private default-child-path "src/genmlx/sandbox_child.cljs")

(defn- runtime-cmd
  "[runtime-binary nbb-cli-script] of the CURRENTLY RUNNING nbb process, or
   nil if process.argv doesn't look like an nbb invocation. Re-invoking the
   same runtime avoids both version drift and the bunx grandchild-orphan trap
   (see ns docstring)."
  []
  (let [argv (.-argv js/process)
        exe  (aget argv 0)
        cli  (aget argv 1)]
    (when (and exe cli (str/includes? cli "nbb"))
      [exe cli])))

(defn- stderr-tail [r]
  (let [e (.-stderr r)]
    (when (and e (seq e))
      (let [t (str/trim e)]
        (subs t (max 0 (- (count t) 300)))))))

(defn eval-with-budget
  "Evaluate the ClojureScript string form-str in a killable subprocess with a
   wall-clock budget. Blocks the caller for at most
   (+ time-ms startup-ms) ms. NEVER hangs: a non-terminating or too-slow form
   is SIGKILLed at the budget.

   opts:
     :time-ms    eval budget in ms (default 2000)
     :startup-ms allowance for interpreter startup, added to the subprocess
                 timeout (default 3000; measured startup on this box is
                 ~150-350 ms — the default is a >8x safety margin)
     :child-path evaluator script (default \"src/genmlx/sandbox_child.cljs\",
                 resolved against :cwd / process cwd)
     :cwd        working directory for the child (default: inherit; must be
                 the repo root for classpath requires to resolve)
     :cmd        full command override as [exe & args] (the child script path
                 must be included); default is
                 [process.argv[0] process.argv[1] child-path]

   Returns exactly one of:
     {:value <edn> :ms n}          eval succeeded; :ms is child-measured
                                   eval-only time. May carry :out (string) —
                                   anything the form printed to stdout.
     {:error :timeout :time-ms n}  killed at the budget
     {:error :eval-error :message s :ms n}      the form threw / didn't parse
     {:error :unserializable :message s :ms n}  value not EDN-round-trippable
     {:error :spawn-error :message s}           child died without a result
                                                (spawn failure, OOM-kill, ...)

   Check (:error r) — a form evaluating to nil returns {:value nil}."
  ([form-str] (eval-with-budget form-str {}))
  ([form-str {:keys [time-ms startup-ms child-path cwd cmd]
              :or   {time-ms 2000 startup-ms 3000
                     child-path default-child-path}}]
   (when-not (string? form-str)
     (throw (ex-info "genmlx.sandbox/eval-with-budget: form-str must be a string"
                     {:got (type form-str)})))
   (let [[exe & args] (or cmd
                          (some-> (runtime-cmd) (conj child-path))
                          (throw (ex-info
                                  (str "genmlx.sandbox: process.argv doesn't look like an nbb "
                                       "invocation; pass :cmd [runtime nbb-cli child-path]")
                                  {:argv (js->clj (.-argv js/process))})))
         spawn-opts (cond-> {:input      form-str
                             :timeout    (+ time-ms startup-ms)
                             :killSignal "SIGKILL"
                             :encoding   "utf8"
                             :maxBuffer  (* 32 1024 1024)}
                      cwd (assoc :cwd cwd))
         r    (cp/spawnSync exe (clj->js (vec args)) (clj->js spawn-opts))
         out  (.-stdout r)
         err  (.-error r)
         code (some-> err .-code)
         idx  (when out (str/last-index-of out sentinel))]
     (cond
       (= "ETIMEDOUT" code)
       {:error :timeout :time-ms time-ms}

       (nil? idx)
       {:error :spawn-error
        :message (str "child produced no result"
                      (when err (str "; error: " (or code (.-message err))))
                      (when-let [s (.-status r)] (str "; exit status " s))
                      (when-let [sig (.-signal r)] (str "; signal " sig))
                      (when-let [tail (stderr-tail r)] (str "; stderr: " tail)))}

       :else
       (let [payload (str/trim (subs out (+ idx (count sentinel))))
             parsed  (try (edn/read-string payload)
                          (catch :default e
                            {:error :spawn-error
                             :message (str "unreadable child result: " (.-message e))}))
             printed (str/trim (subs out 0 idx))]
         (cond-> parsed
           (seq printed) (assoc :out printed)))))))
