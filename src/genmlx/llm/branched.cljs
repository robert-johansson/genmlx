(ns genmlx.llm.branched
  "Branch-using GFI for an LLM-as-generative-function on the native branchable KV
   cache (the second path: Bayesian inference over token programs on the resident
   80B Qwen3-Coder-Next, not just forward sampling).

   `make-llm-gf` (genmlx.llm.core) is the REPLAY oracle: every regenerate re-runs
   the whole body from token 0 (genmlx.dynamic `run-regen-general*` -> `run-body`).
   This namespace builds `make-llm-gf-branched`, which attaches a `::custom-dispatch`
   (genmlx.dispatch/with-dispatch) that drives the native fork primitive directly —
   so a token-MCMC / regenerate move SKIPS the prefix replay:

     simulate   -> forward the body once, forking the model-internal cache at EACH
                   token site into a fork-at-each-site LEDGER stored in trace meta
                   (`::ledger {:branches :logits :toks :dfas}`). SCORE-ONCE: the
                   raw next-token logits at each site are stored, so a later move
                   scores against them rather than re-forwarding (avoids injecting
                   the 4-bit MoE's run-to-run logit jitter into the MH ratio).
     regenerate -> resume O(1) at the divergence site k via `branch-from(ledger[k])`,
                   resample the selected site, and advance the isolated branch over
                   the retained suffix with `forward-branch` (AR-exact per-step, NOT
                   chunked-prefill), accumulating the retained-only weight
                   W = Σ_{retained j>k} [lp(tok_j; FRESH branch logits_j)
                                          − lp(tok_j; STORED ledger logits_j)].
                   Selected / fresh / removed sites cancel to 0 (matches the handler
                   general path `make-regen-result-general`). The new trace carries a
                   fresh ledger that SHARES the unchanged prefix branch ids [0,k).

   Grammar: a `genmlx.llm.grammar` constraint is threaded as a DFA state across the
   suffix; raw logits are masked (`apply-mask`) before BOTH sampling and scoring, so
   the renormalized constrained distribution is identical on the proposal and target
   sides (no chunked masking bias). constraint = nil -> identity mask (the plain path).

   MUTATION BOUNDARY (mirrors the KV-cache try/finally in genmlx.llm.backend, and
   world/net.cljs `with-server` / world/train.cljs `with-trainer`): the native fork
   ledger is a persistent native resource that `reset-cache!` does NOT free. Every
   branch id created here is registered in the scope atom `*branch-scope*`, and
   `with-llm-branches*` disposes them ALL in a `finally`. Run every inference sweep
   inside that scope. `llm-mh-chain` additionally disposes the loser ledger's unique
   branches each step, bounding live branches to ~2 ledgers over an arbitrarily long
   chain. The scope atom is created fresh per `with-llm-branches*` and never escapes."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.dispatch :as dispatch]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.protocols :as p]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.util :as u]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as core]
            [genmlx.llm.grammar :as gram]
            [clojure.set :as set]))

;; ---------------------------------------------------------------------------
;; Small helpers
;; ---------------------------------------------------------------------------

(defn- mat [a] (mx/materialize! a) a)
(defn- t-addr [i] (keyword (str "t" i)))
(defn- site-dist [logits] (dist/categorical logits))
(defn- site-sample [logits key] (dc/dist-sample (site-dist logits) key))
(defn- site-lp [logits tok] (dc/dist-log-prob (site-dist logits) tok))
(defn- leaf [choices i] (cm/get-value (cm/get-submap choices (t-addr i))))

;; Grammar masking — identity when constraint is nil (the no-grammar path is the
;; same code with a pass-through mask and a nil DFA state threaded but unused).
(defn- ginit    [constraint] (when constraint (:start (:dfa constraint))))
(defn- gmask    [constraint dfa logits] (if constraint (gram/apply-mask constraint dfa logits) logits))
(defn- gadvance [constraint dfa tok-id]
  (if (and constraint (not= tok-id (:eos-id constraint)))
    (gram/dfa-advance-string (:dfa constraint) dfa (nth (:token-index constraint) tok-id ""))
    dfa))
(defn- masked-lp [constraint dfa logits tok] (site-lp (gmask constraint dfa logits) tok))

;; ---------------------------------------------------------------------------
;; Blessed disposal scope (the lifecycle answer)
;; ---------------------------------------------------------------------------

(def ^:dynamic *branch-scope*
  "When bound (by with-llm-branches*), an atom holding the set of every live native
   branch id created in scope, so they can all be disposed in a finally. nil outside
   a scope -> branch ids are not tracked (the caller owns disposal)." nil)

(defn- track!   [bid] (when *branch-scope* (swap! *branch-scope* conj bid)) bid)
(defn- untrack! [bid] (when *branch-scope* (swap! *branch-scope* disj bid)) bid)
(defn- dispose! [model bid]
  (untrack! bid)
  (try (llm/dispose-branch! model bid) (catch :default _ nil)))

(defn with-llm-branches*
  "Run (f) with a fresh native-branch disposal scope, disposing EVERY branch id
   created inside (across simulate/regenerate ledgers) in a finally — the only safe
   teardown, since reset-cache! does not free branches. Returns (f)'s value. Run all
   branch-using inference inside this scope."
  [model f]
  (binding [*branch-scope* (atom #{})]
    (try (f)
         (finally (doseq [b @*branch-scope*]
                    (try (llm/dispose-branch! model b) (catch :default _ nil)))))))

(defn branch-scope-live
  "Number of live tracked branch ids in the current scope (nil outside a scope).
   Diagnostic — feeds the live-branch memory measurement (bean mlx-l5wy)."
  [] (when *branch-scope* (count @*branch-scope*)))

(defn release-branch!
  "Dispose a native branch id AND untrack it from the current scope. Public helper
   for consumers that manually discard candidate traces' branches (the loser path in
   llm-mh-chain does this internally). Use this, not the raw backend dispose-branch!,
   inside a with-llm-branches* scope so the live-branch count stays accurate."
  [model bid] (dispose! model bid))

;; ---------------------------------------------------------------------------
;; Trace + ledger construction
;; ---------------------------------------------------------------------------

(defn ledger
  "The fork-at-each-site branch ledger attached to a branched trace's metadata, or
   nil for a trace not produced by make-llm-gf-branched (e.g. the dense replay path)."
  [trace] (::ledger (meta trace)))

(defn- mk-trace [gf args choices retval score led]
  (cond-> (tr/with-score-type
            (tr/make-trace {:gen-fn gf :args args :choices choices :retval retval :score score})
            :joint)
    led (vary-meta assoc ::ledger led)))

;; ---------------------------------------------------------------------------
;; Branched SIMULATE — build the trace make-llm-gf would + the fork ledger
;; ---------------------------------------------------------------------------

(defn- branched-simulate [model eos constraint gf args key]
  (let [[prompt-ids max-tokens] args, prompt (vec prompt-ids)]
    (if (zero? max-tokens)
      (mk-trace gf args cm/EMPTY prompt (mx/scalar 0.0) nil)
      (do
        (llm/init-cache! model)
        (let [l0 (mat (llm/forward-prefill model prompt))]
          (loop [i 0, logits l0, dfa (ginit constraint), key key,
                 choices cm/EMPTY, score (mx/scalar 0.0), ctx prompt,
                 branches [], logitss [], toks [], dfas []]
            (if (>= i max-tokens)
              (mk-trace gf args choices ctx score
                        {:branches branches :logits logitss :toks toks :dfas dfas})
              (let [bid (track! (llm/branch-cache! model))    ; fork BEFORE sampling site i
                    masked (gmask constraint dfa logits)
                    [k1 k2] (rng/split key)
                    tok (site-sample masked k2), tok-id (mx/item tok)
                    lp (site-lp masked tok)
                    choices' (cm/set-value choices (t-addr i) tok), score' (mx/add score lp)
                    br' (conj branches bid), lo' (conj logitss logits)
                    tk' (conj toks tok-id), df' (conj dfas dfa)]
                (if (= tok-id eos)
                  (mk-trace gf args choices' (conj ctx tok-id) score'
                            {:branches br' :logits lo' :toks tk' :dfas df'})
                  (let [nl (mat (llm/forward-step model tok-id))]
                    (recur (inc i) nl (gadvance constraint dfa tok-id) k1
                           choices' score' (conj ctx tok-id) br' lo' tk' df')))))))))))

;; ---------------------------------------------------------------------------
;; Branched REGENERATE — resume at the divergence site, rescore the suffix
;; ---------------------------------------------------------------------------

(defn- branched-regenerate [model eos constraint gf trace selection key]
  (let [led (ledger trace)
        [prompt-ids max-tokens] (:args trace)
        oc (:choices trace)
        old-toks (:toks led), old-logits (:logits led)
        old-branches (:branches led), old-dfas (:dfas led)
        old-n (count old-toks)
        sel? (fn [j] (sel/selected? selection (t-addr j)))
        k (first (filter sel? (range old-n)))]
    (if (nil? k)
      {:trace trace :weight (mx/scalar 0.0)}            ; nothing selected -> identity
      (let [s (track! (llm/branch-from model (nth old-branches k)))
            ;; new ledger SHARES the unchanged prefix [0,k) (same branch ids/logits/dfas)
            pre-br (vec (subvec old-branches 0 k)), pre-lo (vec (subvec old-logits 0 k))
            pre-tk (vec (subvec old-toks 0 k)),     pre-df (vec (subvec old-dfas 0 k))
            pre-ch (reduce (fn [c i] (cm/set-value c (t-addr i) (leaf oc i))) cm/EMPTY (range k))
            pre-sc (reduce (fn [sc i] (mx/add sc (masked-lp constraint (nth old-dfas i) (nth old-logits i) (leaf oc i))))
                           (mx/scalar 0.0) (range k))]
        (loop [j k, logits (nth old-logits k), dfa (nth old-dfas k), key key,
               nbr pre-br, nlo pre-lo, ntk pre-tk, ndf pre-df,
               nch pre-ch, nsc pre-sc, weight (mx/scalar 0.0)]
          (let [snap (track! (llm/branch-from model s))   ; new-ledger fork for site j
                masked (gmask constraint dfa logits)
                retained? (and (< j old-n) (not (sel? j)))
                [k1 k2] (rng/split key)
                used-tok (if retained? (leaf oc j) (site-sample masked k2))
                used-id (mx/item used-tok)
                lp (site-lp masked used-tok)
                weight' (if retained?
                          (mx/add weight (mx/subtract lp (masked-lp constraint (nth old-dfas j)
                                                                    (nth old-logits j) used-tok)))
                          weight)
                nbr' (conj nbr snap), nlo' (conj nlo logits)
                ntk' (conj ntk used-id), ndf' (conj ndf dfa)
                nch' (cm/set-value nch (t-addr j) used-tok), nsc' (mx/add nsc lp)
                key' (if retained? key k1)]
            (if (or (= used-id eos) (>= (inc j) max-tokens))
              (do (dispose! model s)
                  {:trace (mk-trace gf (:args trace) nch' (into (vec prompt-ids) ntk') nsc'
                                    {:branches nbr' :logits nlo' :toks ntk' :dfas ndf'})
                   :weight weight'})
              (let [nl (mat (llm/forward-branch model s used-id))]
                (recur (inc j) nl (gadvance constraint dfa used-id) key'
                       nbr' nlo' ntk' ndf' nch' nsc' weight')))))))))

;; ---------------------------------------------------------------------------
;; make-llm-gf-branched
;; ---------------------------------------------------------------------------

(defn make-llm-gf-branched
  "An LLM-as-GF whose simulate + regenerate run on the native branchable cache
   (branch-using inference on the resident MoE). All OTHER GFI ops delegate to the
   plain make-llm-gf oracle (the handler replay path), so the branched gf is a full
   drop-in generative function. The model must expose the branch surface
   (llm/supports-branching? — the native MoE path, or the owned CljsForwardModel
   whose persistent-value caches fork by reference-sharing, genmlx-7f93; only the
   upstream DENSE natives need the replay path). Optional `constraint` is a genmlx.llm.grammar
   constraint threaded as grammar masking through both simulate and regenerate.

   Returns a DynamicGF; run inference inside (with-llm-branches* model ...)."
  ([model-map] (make-llm-gf-branched model-map nil))
  ([model-map constraint]
   (let [{:keys [model tokenizer]} model-map
         eos (llm/eos-token-id tokenizer)
         base (core/make-llm-gf model-map)
         oracle (if constraint (gram/constrain base constraint) base)
         df (fn [op gf args key opts]
              (case op
                :simulate   (branched-simulate model eos constraint gf args key)
                :regenerate (let [t (:trace opts), sl (:selection opts)]
                              (if (ledger t)
                                (branched-regenerate model eos constraint gf t sl key)
                                ;; no ledger (foreign trace) -> correct replay fallback
                                (p/regenerate (dyn/with-key oracle key) t sl)))
                :generate   (p/generate (dyn/with-key oracle key) args (:constraints opts))
                :assess     (p/assess   (dyn/with-key oracle key) args (:constraints opts))
                :project    (p/project  (dyn/with-key oracle key) (:trace opts) (:selection opts))
                :update     (p/update   (dyn/with-key oracle key) (:trace opts) (:constraints opts))
                :propose    (p/propose  (dyn/with-key oracle key) args)))]
     (dispatch/with-dispatch oracle df))))

;; ---------------------------------------------------------------------------
;; token-MCMC driver with loser-disposal (long chains, bounded live branches)
;; ---------------------------------------------------------------------------

(defn- suffix-ids [trace k] (set (subvec (:branches (ledger trace)) k)))

(defn llm-mh-chain
  "Single-site Metropolis-Hastings over a branched LLM-GF, using the standard GFI
   regenerate weight as the log MH acceptance ratio (matches inference/mcmc mh-step),
   and disposing the loser ledger's UNIQUE (suffix) branches each step so live
   branches stay bounded to ~2 ledgers over an arbitrarily long chain. Each step
   also force-gc!s: on the OWNED forward, disposing a loser ledger only drops
   CLJS references to multi-GB cache values that JS GC cannot size (the
   genmlx-h3p5 dark-page class) — the per-iteration cleanup CLAUDE.md sanctions
   for inference hot loops. Run inside with-llm-branches*.
   Returns {:trace :accept-rate :max-live}."
  [model gf init-trace selection n key]
  (loop [t init-trace, key key, i 0, accepts 0, max-live 0]
    (if (>= i n)
      {:trace t :accept-rate (/ accepts (double n)) :max-live max-live}
      (let [[rk ak nk] (rng/split-n key 3)
            k (first (filter #(sel/selected? selection (t-addr %))
                             (range (count (:toks (ledger t))))))
            result (p/regenerate (dyn/with-key gf rk) t selection)
            w (mx/item (:weight result))
            accept? (u/accept-mh? w ak)
            winner (if accept? (:trace result) t)
            loser  (if accept? t (:trace result))]
        (doseq [b (set/difference (suffix-ids loser k) (suffix-ids winner k))] (dispose! model b))
        (mx/force-gc!)
        (recur winner nk (inc i) (if accept? (inc accepts) accepts)
               (max max-live (or (branch-scope-live) 0)))))))
