(ns genmlx.llm.smc
  "Token-SMC over branchable KV caches (genmlx-5qk7) — particle filtering on
   the resident LLM as a first-class inference algorithm: the second path
   (vision-north-star) made concrete. N particles over one prompt pay prefill
   ONCE; per step each alive particle decodes one token on its own cache
   branch; resampling duplicates a cheap branch handle instead of recomputing;
   losers are disposed immediately.

   SEMANTICS (twisted SMC / SMC-steering):
     target at step t:  π_t(w_{1:t}) ∝ p_LM(w_{1:t}) · Π_{s<=t} φ_s(w_{1:s})
     proposal q_t:      model logits (bootstrap), grammar-masked renormalized
                        logits, or a caller fn transforming logits
     incremental w:     log φ_t + log(p_t/q_t). For the grammar-masked
                        proposal with the grammar as its own twist this is the
                        mask log-normalizer log Z_t(prefix) — the standard
                        SMC-steering weight.
     resample:          systematic (u/systematic-resample — all-(-Inf)
                        populations throw :degenerate-particles, genmlx-ng9t),
                        ESS-gated on the ALIVE count; winners FORK the branch,
                        losers are DISPOSED immediately; weights reset; log-ML
                        accumulates via the telescoped increment
                        (ismc/log-ml-increment-from — post-uxjm semantics).

   DECODER ABSTRACTION: the filter steps particles through ITokenDecoder, so
   the same algorithm runs on (a) the native branchable cache (NativeDecoder —
   branch-cache!/branch-from/forward-branch), (b) a dense model without the
   branch surface (ReplayDecoder — fork copies the token vector, each step
   re-forwards the whole prefix; correct, O(T) per step, for V5-class smokes),
   and (c) a synthetic table decoder (model-free tests with enumerable
   posteriors — the llm_token_mcmc_test pattern). Every decoder ledgers its
   live handles, so the R1 (bounded branches) and R2 (no leak) resource
   properties are assertable uniformly.

   RESOURCE CONTRACT:
     R1 live handles <= N + 1 (N particle heads + the prefill root) at every
        instant; transient resample peak stays within N + 1 because losers are
        disposed before winners fork.
     R2 token-smc disposes ALL handles on return (results are VALUES);
        with-token-smc* exposes live handles to a continuation inside a scope
        and tears everything down in a finally — mirroring with-llm-branches*.
     R3 prefill runs once per prompt; per-step cost is N × decode-step + O(N)
        bookkeeping. Fork cost is backend-dependent (owned CljsForwardModel:
        an O(1) persistent-value reference share; CUDA native flat path: a
        cache COPY; Metal paged path: a block-share) — measured numbers live
        in bench/owned_branch.cljs + bench/token_smc.cljs, not hidden here.
     R4 dropped per-step graphs are force-gc!'d every :gc-every rounds
        (default 1): JS GC cannot see native buffer sizes, so without this an
        N-particle filter accumulates N x T dead transient graphs (measured
        ~8 GB/s dark pages on a 35B — the genmlx-h3p5 OOM class).

   V1 DEVIATIONS (documented on the bean): rejuvenation runs at filter END
   (post-loop token-MCMC via the CONSTRAINED gf's regenerate — π-invariant for
   grammar twists; per-resample rejuvenation needs handle rebuilds and is v2);
   exported traces come from `particle->trace` (constrained generate — the
   wrap-model mechanism) rather than being carried live through the filter."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.inference.util :as u]
            [genmlx.inference.smc :as ismc]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.grammar :as gram]))

;; ===========================================================================
;; Decoder abstraction
;; ===========================================================================

(defprotocol ITokenDecoder
  (dec-prefill! [d prompt-ids]
    "Prefill the prompt once. Returns {:root handle :logits mx-logits}.")
  (dec-fork! [d handle] "Fork a handle (cheap branch duplicate). Returns handle'.")
  (dec-step! [d handle tok-id] "Advance handle by one token. Returns next logits.")
  (dec-dispose! [d handle] "Dispose a handle (frees the native branch).")
  (dec-live-handles [d] "Collection of live handles (the R1/R2 ledger)."))

(defn- mat [a] (mx/materialize! a) a)

(deftype NativeDecoder [model live]
  ITokenDecoder
  (dec-prefill! [_ prompt-ids]
    (llm/init-cache! model)
    (let [logits (mat (llm/forward-prefill model (vec prompt-ids)))
          root (llm/branch-cache! model)]
      (swap! live conj root)
      {:root root :logits logits}))
  (dec-fork! [_ handle]
    (let [b (llm/branch-from model handle)]
      (swap! live conj b)
      b))
  (dec-step! [_ handle tok-id]
    (mat (llm/forward-branch model handle tok-id)))
  (dec-dispose! [_ handle]
    (swap! live disj handle)
    (try (llm/dispose-branch! model handle) (catch :default _ nil)))
  (dec-live-handles [_] @live))

(defn native-decoder
  "Decoder over the branchable KV cache surface (requires
   llm/supports-branching?): the native MoE branch primitives, or the owned
   CljsForwardModel whose persistent-value caches make dec-fork! an O(1)
   reference share (genmlx-7f93)."
  [model]
  (->NativeDecoder model (atom #{})))

(deftype ReplayDecoder [model prompt-ref live counter]
  ITokenDecoder
  (dec-prefill! [_ prompt-ids]
    (vreset! prompt-ref (vec prompt-ids))
    (let [logits (mat (llm/forward-pass model (vec prompt-ids)))
          root (swap! counter inc)]
      (swap! live assoc root [])
      {:root root :logits logits}))
  (dec-fork! [_ handle]
    (let [b (swap! counter inc)]
      (swap! live assoc b (get @live handle))
      b))
  (dec-step! [_ handle tok-id]
    (let [toks (conj (get @live handle) tok-id)]
      (swap! live assoc handle toks)
      (mat (llm/forward-pass model (into @prompt-ref toks)))))
  (dec-dispose! [_ handle] (swap! live dissoc handle))
  (dec-live-handles [_] (keys @live)))

(defn replay-decoder
  "Correct-but-O(T)-per-step decoder for models WITHOUT the branch surface
   (upstream dense natives; since genmlx-7f93 the owned CljsForwardModel
   branches natively): a handle is a token vector; every step re-forwards
   prompt+tokens through the uncached forward. The asymmetry vs the branch
   decoder is the documented R3 cost difference, not hidden — and the
   parity/benchmark oracle for the owned branch path."
  [model]
  (->ReplayDecoder model (volatile! []) (atom {}) (atom 0)))

(deftype TableDecoder [logits-fn live counter]
  ITokenDecoder
  (dec-prefill! [_ prompt-ids]
    (let [root (swap! counter inc)]
      (swap! live assoc root [])
      {:root root :logits (logits-fn (vec prompt-ids) [])}))
  (dec-fork! [_ handle]
    (let [b (swap! counter inc)]
      (swap! live assoc b (get @live handle))
      b))
  (dec-step! [_ handle tok-id]
    (let [toks (conj (get @live handle) tok-id)]
      (swap! live assoc handle toks)
      (logits-fn nil toks)))
  (dec-dispose! [_ handle] (swap! live dissoc handle))
  (dec-live-handles [_] (keys @live)))

(defn table-decoder
  "Synthetic decoder for model-free tests: (logits-fn prompt-ids tokens) ->
   MLX logits for the next position. Enumerable posteriors, zero GPU."
  [logits-fn]
  (->TableDecoder logits-fn (atom {}) (atom 0)))

(defn decoder-for
  "Pick the decoder for a model-map: native when the branch surface exists,
   replay otherwise."
  [model-map]
  (let [model (:model model-map)]
    (if (llm/supports-branching? model)
      (native-decoder model)
      (replay-decoder model))))

;; ===========================================================================
;; Grammar helpers (nil constraint = identity mask)
;; ===========================================================================

(defn- ginit [constraint] (when constraint (:start (:dfa constraint))))
(defn- gmask [constraint dfa logits]
  (if constraint (gram/apply-mask constraint dfa logits) logits))
(defn- gadvance [constraint dfa tok-id]
  (if (and constraint (not= tok-id (:eos-id constraint)))
    (gram/dfa-advance-string (:dfa constraint) dfa
                             (nth (:token-index constraint) tok-id ""))
    dfa))

;; ===========================================================================
;; The filter
;; ===========================================================================

(defn- lse-item [logits]
  (mx/realize (mx/logsumexp logits)))

(defn- run-filter
  "Core loop shared by token-smc and with-token-smc*. Returns
   {:particles [{:handle :tokens :log-w :finished? :dfa}]
    :log-ml-estimate mx-scalar :ess-trajectory [..] :decoder d}.

   Per-round cleanup (:gc-every, default 1): mx/force-gc! after every round's
   particle steps — the R4 resource property. Each step DROPS the particle's
   previous logits/cache graph, but JS GC cannot see the native buffer sizes
   behind those references, so an N-particle filter otherwise accumulates
   N x T dead transient graphs before any collection runs. On a 35B owned
   model that measured ~8 GB/s of driver-side dark pages (the genmlx-h3p5
   OOM-cascade class, reboot #3 2026-07-10) — a filter-scale hazard, not a
   decoder-specific one. 0/false disables (model-free table-decoder tests)."
  [decoder {:keys [particles ess-threshold max-tokens eos-id proposal twist
                   constraint key callback gc-every]
            :or {particles 8 ess-threshold 0.5 proposal :model gc-every 1}}
   prompt-ids]
  (when (and (= proposal :grammar-masked) (nil? constraint))
    (throw (ex-info "proposal :grammar-masked requires a :constraint"
                    {:genmlx/error :missing-constraint})))
  (let [n particles
        key (rng/ensure-key key)
        {:keys [root logits]} (dec-prefill! decoder prompt-ids)
        init-p (fn [] {:handle (dec-fork! decoder root)
                       :tokens []
                       :logits logits
                       :dfa (ginit constraint)
                       :log-w (mx/scalar 0.0)
                       :finished? (zero? max-tokens)})
        ;; the root handle stays live for the run (heads = N, root = +1: R1)
        state0 {:ps (vec (repeatedly n init-p))
                :log-ml (mx/scalar 0.0)
                ;; weights at the start of the current resample segment
                :seg-w (vec (repeat n (mx/scalar 0.0)))
                :ess []}
        step-particle
        (fn [pt t kt]
          (if (:finished? pt)
            pt
            (let [{:keys [handle tokens logits dfa log-w]} pt
                  masked (gmask constraint dfa logits)
                  lse-masked (when constraint (lse-item masked))]
              (if (and constraint (= lse-masked js/Number.NEGATIVE_INFINITY))
                ;; mask deadlock: no valid token under the twist
                (assoc pt :finished? true :log-w (mx/scalar js/Number.NEGATIVE_INFINITY))
                (let [q-logits (case proposal
                                 :grammar-masked masked
                                 :model logits
                                 (proposal {:dfa dfa :step t :tokens tokens} logits))
                      tok (dc/dist-sample (dist/categorical q-logits) kt)
                      tok-id (mx/item tok)
                      ;; incremental weight: log p + log φ − log q. categorical
                      ;; log-probs normalize internally, so
                      ;;   grammar-masked: log p_masked/Z_p... reduces to the
                      ;;   mask log-normalizer  lse(masked) − lse(raw)
                      ;;   :model: p == q -> 0 (twist only)
                      ;;   fn proposal: lp under raw − lp under q
                      inc-w (case proposal
                              :grammar-masked (mx/scalar (- lse-masked (lse-item logits)))
                              :model (mx/scalar 0.0)
                              (mx/subtract (dc/dist-log-prob (dist/categorical logits) tok)
                                           (dc/dist-log-prob (dist/categorical q-logits) tok)))
                      tokens' (conj tokens tok-id)
                      dfa' (gadvance constraint dfa tok-id)
                      phi (when twist (twist {:dfa dfa' :step t} tokens'))
                      inc-w (if phi (mx/add inc-w (mx/ensure-array phi)) inc-w)
                      log-w' (mx/add log-w inc-w)
                      done? (or (= tok-id eos-id) (>= (count tokens') max-tokens))]
                  (if done?
                    (assoc pt :tokens tokens' :dfa dfa' :log-w log-w' :finished? true)
                    (assoc pt :tokens tokens' :dfa dfa' :log-w log-w'
                           :logits (dec-step! decoder handle tok-id))))))))
        resample!
        (fn [{:keys [ps log-ml seg-w ess]} kr]
          (let [w-arr (u/materialize-weights (mapv :log-w ps))
                prev-arr (u/materialize-weights seg-w)
                ml-inc (ismc/log-ml-increment-from w-arr prev-arr)
                indices (u/systematic-resample (mapv :log-w ps) n kr)
                counts (frequencies indices)
                ;; losers first, THEN fork winners: transient peak <= N + 1
                _ (doseq [i (range n) :when (nil? (counts i))]
                    (dec-dispose! decoder (:handle (nth ps i))))
                used (volatile! #{})
                ps' (mapv (fn [a]
                            (let [src (nth ps a)
                                  first? (not (contains? @used a))]
                              (vswap! used conj a)
                              (-> (if first? src
                                      (assoc src :handle (dec-fork! decoder (:handle src))))
                                  (assoc :log-w (mx/scalar 0.0)))))
                          indices)]
            {:ps ps' :log-ml (mx/add log-ml ml-inc)
             :seg-w (vec (repeat n (mx/scalar 0.0)))
             :ess ess}))]
    (loop [t 0, st state0, key key]
      (let [alive (count (remove :finished? (:ps st)))]
        (if (or (zero? alive) (>= t max-tokens))
          (let [w-arr (u/materialize-weights (mapv :log-w (:ps st)))
                prev-arr (u/materialize-weights (:seg-w st))
                final-ml (mx/add (:log-ml st) (ismc/log-ml-increment-from w-arr prev-arr))]
            {:particles (:ps st) :log-ml-estimate final-ml
             :ess-trajectory (:ess st) :decoder decoder :root root})
          (let [[kt kr knext] (rng/split-n key 3)
                kts (rng/split-n kt n)
                ps' (vec (map-indexed (fn [i pt] (step-particle pt t (nth kts i))) (:ps st)))
                ess (u/compute-ess (mapv :log-w ps'))
                st' (assoc st :ps ps' :ess (conj (:ess st) ess))
                resample? (< ess (* ess-threshold (max 1 alive)))
                st'' (if resample? (resample! st' kr) st')]
            (when callback (callback {:step t :ess ess :resampled? resample?}))
            (when (and gc-every (pos? gc-every) (zero? (mod (inc t) gc-every)))
              (mx/force-gc!))
            (recur (inc t) st'' knext)))))))

;; ===========================================================================
;; Batched-lane filter (genmlx-k7nj): K particles = ONE [K]-batched branch
;; ===========================================================================

(defn- lane-rows
  "Slice a [K V] logits array into K [V] rows."
  [logits-k n]
  (mapv #(mx/take-idx logits-k (mx/scalar % mx/int32) 0) (range n)))

(defn- run-filter-lanes
  "Batched-lane core loop (genmlx-k7nj): the K particles ride the lane axis
   of ONE [K]-batched owned branch — each round is a single
   forward-branch-batched step and resampling is one lane-axis gather
   (resample-branch-lanes!) instead of per-branch fork/replay loops. The
   weight algebra is identical to run-filter (grammar-mask log-normalizer
   increments; :model → 0), computed per lane on host-sliced rows, so a
   particle's weight is a deterministic function of its token sequence in
   both engines — the sweep-equivalence gate.

   Scope: :model and :grammar-masked proposals. fn proposals and :twist have
   per-particle host contracts and stay on the per-branch engine (token-smc
   routes them there; direct callers get the loud errors below).

   Returns {:particles [{:tokens :log-w :finished? :dfa}] :log-ml-estimate
   :ess-trajectory :root :lanes} — :root/:lanes are LIVE branch ids the
   caller must dispose (token-smc does, in its normal flow and on throw)."
  [model {:keys [particles ess-threshold max-tokens eos-id proposal
                 constraint key callback gc-every]
          :or {particles 8 ess-threshold 0.5 proposal :model gc-every 1}}
   prompt-ids]
  (when-not (contains? #{:model :grammar-masked} proposal)
    (throw (ex-info "lanes mode supports :model and :grammar-masked proposals only — fn proposals need a per-lane batched contract; use the per-branch engine."
                    {:genmlx/error :lanes-proposal-unsupported :proposal proposal})))
  (when (and (= proposal :grammar-masked) (nil? constraint))
    (throw (ex-info "proposal :grammar-masked requires a :constraint"
                    {:genmlx/error :missing-constraint})))
  (let [n particles
        key (rng/ensure-key key)
        _ (llm/init-cache! model)
        logits0 (mat (llm/forward-prefill model (vec prompt-ids)))
        root (llm/branch-cache! model)
        lanes (llm/branch-from model root)
        vocab (first (mx/shape logits0))
        pad-id (or eos-id 0)
        zeros-k (fn [] (let [z (mx/zeros [n])] (mx/materialize! z) z))]
    (if (zero? max-tokens)
      {:particles (vec (repeat n {:tokens [] :log-w (mx/scalar 0.0)
                                  :finished? true :dfa (ginit constraint)}))
       :log-ml-estimate (mx/scalar 0.0) :ess-trajectory []
       :root root :lanes lanes}
      (try
       (loop [t 0
             ;; round-0 lanes share the prefill logits; the first batched
             ;; step tiles the branch cache to K (forward-branch-batched)
             logits-k (mx/broadcast-to (mx/expand-dims logits0 0) [n vocab])
             tokens (vec (repeat n []))
             dfas (vec (repeat n (ginit constraint)))
             finished (vec (repeat n false))
             log-w (zeros-k)
             seg-w (zeros-k)
             log-ml (mx/scalar 0.0)
             ess-traj []
             key key]
        (let [alive (count (remove true? finished))]
          (if (or (zero? alive) (>= t max-tokens))
            (let [final-ml (mx/add log-ml (ismc/log-ml-increment-from log-w seg-w))
                  lw-host (mx/->clj log-w)]
              {:particles (mapv (fn [i]
                                  {:tokens (tokens i)
                                   :log-w (mx/scalar (nth lw-host i))
                                   :finished? (boolean (finished i))
                                   :dfa (dfas i)})
                                (range n))
               :log-ml-estimate final-ml :ess-trajectory ess-traj
               :root root :lanes lanes})
            (let [[kt kr knext] (rng/split-n key 3)
                  rows (when constraint (lane-rows logits-k n))
                  ;; per-lane grammar mask on host-sliced rows; finished (or
                  ;; deadlocked) lanes keep the raw row so batch sampling
                  ;; never sees an all--inf row
                  masked-info
                  (when constraint
                    (mapv (fn [i]
                            (if (finished i)
                              {:row (rows i) :dead? false}
                              (let [m (gram/apply-mask constraint (dfas i) (rows i))
                                    lse-m (lse-item m)]
                                (if (= lse-m js/Number.NEGATIVE_INFINITY)
                                  {:row (rows i) :dead? true}
                                  {:row m :dead? false :lse-masked lse-m}))))
                          (range n)))
                  q-logits-k (case proposal
                               :grammar-masked (mx/stack (mapv :row masked-info) 0)
                               :model logits-k)
                  tok (dc/dist-sample (dist/categorical q-logits-k) kt) ; [K]
                  tok-ids (vec (mx/->clj tok))
                  ;; per-lane state advance (host bookkeeping)
                  stepped
                  (mapv (fn [i]
                          (let [tok-id (long (nth tok-ids i))]
                            (cond
                              (finished i)
                              {:tokens (tokens i) :dfa (dfas i) :finished? true
                               :inc 0.0 :fed pad-id}

                              (and constraint (:dead? (masked-info i)))
                              ;; mask deadlock: no valid token under the twist
                              {:tokens (tokens i) :dfa (dfas i) :finished? true
                               :inc js/Number.NEGATIVE_INFINITY :fed pad-id}

                              :else
                              (let [inc-w (if (= proposal :grammar-masked)
                                            (- (:lse-masked (masked-info i))
                                               (lse-item (rows i)))
                                            0.0)
                                    tokens' (conj (tokens i) tok-id)
                                    dfa' (gadvance constraint (dfas i) tok-id)
                                    done? (or (= tok-id eos-id)
                                              (>= (count tokens') max-tokens))]
                                {:tokens tokens' :dfa dfa' :finished? done?
                                 :inc inc-w :fed (if done? pad-id tok-id)}))))
                        (range n))
                  log-w' (let [w (mx/add log-w (mx/array (mapv :inc stepped)))]
                           (mx/materialize! w) w)
                  tokens' (mapv :tokens stepped)
                  dfas' (mapv :dfa stepped)
                  finished' (mapv :finished? stepped)
                  ess (u/ess-from-log-weight-array log-w')
                  resample? (< ess (* ess-threshold (max 1 alive)))
                  any-alive? (some false? finished')
                  ;; ONE lockstep batched step for all K lanes (dead lanes
                  ;; feed pad; their rows are never read again)
                  logits-k' (when any-alive?
                              (mat (llm/forward-branch-batched
                                    model lanes
                                    (mx/array (mapv :fed stepped) mx/int32))))
                  st' (if resample?
                        (let [ml-inc (ismc/log-ml-increment-from log-w' seg-w)
                              lw-host (mx/->clj log-w')
                              idx (u/systematic-resample
                                   (mapv #(mx/scalar %) lw-host) n kr)
                              idx-arr (mx/array idx mx/int32)]
                          ;; the cache gather IS the resample step (lo6e D1)
                          (llm/resample-branch-lanes! model lanes idx)
                          {:logits (when logits-k'
                                     (mat (mx/take-idx logits-k' idx-arr 0)))
                           :tokens (mapv tokens' idx) :dfas (mapv dfas' idx)
                           :finished (mapv finished' idx)
                           :log-w (zeros-k) :seg-w (zeros-k)
                           :log-ml (mx/add log-ml ml-inc)})
                        {:logits logits-k' :tokens tokens' :dfas dfas'
                         :finished finished' :log-w log-w' :seg-w seg-w
                         :log-ml log-ml})]
              (when callback (callback {:step t :ess ess :resampled? resample?}))
              (when (and gc-every (pos? gc-every) (zero? (mod (inc t) gc-every)))
                (mx/force-gc!))
              (recur (inc t) (:logits st') (:tokens st') (:dfas st')
                     (:finished st') (:log-w st') (:seg-w st') (:log-ml st')
                     (conj ess-traj ess) knext)))))
       (catch :default e
         ;; this engine created root+lanes; free them (and ONLY them) on throw
         (doseq [id [lanes root]]
           (try (llm/dispose-branch! model id) (catch :default _ nil)))
         (throw e))))))

(defn- dispose-lanes! [model result]
  (doseq [id [(:lanes result) (:root result)]]
    (when id (try (llm/dispose-branch! model id) (catch :default _ nil)))))

(defn- dispose-all! [decoder result]
  (doseq [pt (:particles result)] (dec-dispose! decoder (:handle pt)))
  (dec-dispose! decoder (:root result)))

(defn- export-particles [result _model-map]
  ;; :text is nil here — tokenizer decode is an ASYNC IO boundary (sync math,
  ;; async events), so texts come from the separate decode-particles! helper.
  (mapv (fn [pt]
          {:tokens (:tokens pt)
           :text nil
           :log-w (:log-w pt)
           :finished? (boolean (:finished? pt))})
        (:particles result)))

(defn decode-particles!
  "Fill :text on a token-smc result's particles by decoding through the
   tokenizer (async — returns a promise of the updated result). The filter
   itself stays sync; decoding is the IO boundary."
  [model-map result]
  (let [tokenizer (:tokenizer model-map)
        ps (:particles result)]
    (-> (js/Promise.all
         (to-array (map #(llm/decode tokenizer (js/Uint32Array.from (to-array (:tokens %)))) ps)))
        (.then (fn [texts]
                 (assoc result :particles
                        (mapv (fn [pt t] (assoc pt :text t)) ps (vec texts))))))))

;; ===========================================================================
;; Rejuvenation (v1: at filter end, via the constrained gf's regenerate —
;; π-invariant for grammar twists)
;; ===========================================================================

(defn particle->trace
  "Export a particle as a standard GenMLX trace over token sites :t0..:tn —
   a fully-constrained generate of `gf` (the model or grammar-constrained
   model as a generative function), so assess == score holds by the GFI
   contract. WEIGHT SEMANTIC: the particle's :log-w is its TWISTED-target
   importance weight within the returned population (uniform after a final
   resample), NOT the trace score; the trace score is the model's own joint."
  [gf max-tokens particle key]
  (let [constraints (reduce (fn [c [i tok]]
                              (cm/set-value c (keyword (str "t" i)) (mx/array tok)))
                            cm/EMPTY
                            (map-indexed vector (:tokens particle)))]
    (:trace (p/generate (dyn/with-key gf key) [] constraints))))

(defn- rejuvenate-particles
  "K token-MCMC moves per unfinished-or-finished particle over `selection`,
   via the gf's regenerate (weight-preserving: an MH kernel invariant for the
   grammar-constrained target). Skips particles with no tokens."
  [result {:keys [gf steps selection key]}]
  (if (or (nil? gf) (nil? steps) (zero? steps))
    result
    (let [key (rng/ensure-key key)
          ps' (vec
               (map-indexed
                (fn [i pt]
                  (if (empty? (:tokens pt))
                    pt
                    (let [ki (rng/fresh-key (+ 7000 i))
                          tr0 (particle->trace gf (count (:tokens pt)) pt ki)
                          tr (loop [t tr0, k key, s 0]
                               (if (>= s steps)
                                 t
                                 (let [[rk ak nk] (rng/split-n k 3)
                                       {:keys [trace weight]} (p/regenerate (dyn/with-key gf rk) t selection)]
                                   (recur (if (u/accept-mh? (mx/realize weight) ak) trace t)
                                          nk (inc s)))))
                          toks (mapv (fn [j] (mx/item (cm/get-value
                                                       (cm/get-submap (:choices tr)
                                                                      (keyword (str "t" j))))))
                                     (range (count (:tokens pt))))]
                      (assoc pt :tokens toks))))
                (:particles result)))]
      (assoc result :particles ps'))))

;; ===========================================================================
;; Public API
;; ===========================================================================

(defn token-smc
  "Token-level SMC where each particle IS a cache branch.

   opts: {:particles N            ;; default 8
          :ess-threshold r        ;; resample when ESS < r * alive (default 0.5)
          :max-tokens T           ;; REQUIRED
          :eos-id id
          :proposal :model | :grammar-masked | (fn [state logits] logits')
          :constraint c           ;; genmlx.llm.grammar constraint (grammar twist)
          :twist (fn [state token-prefix] log-phi)
          :rejuvenation {:steps K :selection sel :gf gf}  ;; v1: at filter end
          :decoder d              ;; override (tests); default decoder-for
          :lanes? bool            ;; batched-lane engine (genmlx-k7nj): K
                                  ;; particles ride the lane axis of ONE
                                  ;; [K]-batched owned branch — one lockstep
                                  ;; forward per round, resample = one lane
                                  ;; gather. Owned forward only; :model /
                                  ;; :grammar-masked proposals; no :twist,
                                  ;; no :decoder override.
          :gc-every n             ;; force-gc! cadence in rounds (default 1; R4)
          :key k :callback fn}

   model-map: {:model :tokenizer} (llm/load-model) — or nil with :decoder.
   prompt-ids: vector of token ids.

   Returns {:particles [{:tokens :text :log-w :finished?}]
            :log-ml-estimate mx-scalar
            :ess-trajectory [..]}
   with ALL branches disposed before returning (results are values). T=0
   returns prompt-only particles and log-ml 0. All particles at -Inf weight
   throw :degenerate-particles (genmlx-ng9t) — all-impossible is loud."
  [opts model-map prompt-ids]
  (if (:lanes? opts)
    (let [model (:model model-map)]
      (when (:twist opts)
        (throw (ex-info "lanes mode does not support :twist (per-particle host contract) — use the per-branch engine."
                        {:genmlx/error :lanes-twist-unsupported})))
      (when (:decoder opts)
        (throw (ex-info "lanes mode drives the owned batched branch directly — :decoder override is a per-branch concept."
                        {:genmlx/error :lanes-decoder-unsupported})))
      (when-not (llm/supports-branching? model)
        (throw (ex-info "lanes mode requires the OWNED forward's branch ledger — load with {:cljs-forward? true} (or a supported family's smart default)."
                        {:genmlx/error :lanes-owned-only})))
      (let [result (run-filter-lanes model opts prompt-ids)
            out {:particles (export-particles
                             (rejuvenate-particles result (:rejuvenation opts))
                             model-map)
                 :log-ml-estimate (:log-ml-estimate result)
                 :ess-trajectory (:ess-trajectory result)}]
        (dispose-lanes! model result)
        out))
    (let [decoder (or (:decoder opts) (decoder-for model-map))
          result (try (run-filter decoder opts prompt-ids)
                      (catch :default e
                        ;; twist/proposal threw: dispose every live handle, rethrow
                        (doseq [h (vec (dec-live-handles decoder))]
                          (dec-dispose! decoder h))
                        (throw e)))
          result (rejuvenate-particles result (:rejuvenation opts))
          out {:particles (export-particles result model-map)
               :log-ml-estimate (:log-ml-estimate result)
               :ess-trajectory (:ess-trajectory result)}]
      (dispose-all! decoder result)
      out)))

(defn with-token-smc*
  "Run the filter and call (f {:particles [..with LIVE :handle..]
   :log-ml-estimate :ess-trajectory :decoder}) INSIDE the disposal scope —
   for composition (e.g. continuing decode on surviving branches). Everything
   is torn down in a finally regardless of f's outcome; f's value is
   returned. Mirrors with-llm-branches*."
  [opts model-map prompt-ids f]
  (let [decoder (or (:decoder opts) (decoder-for model-map))]
    (try
      (let [result (run-filter decoder opts prompt-ids)]
        (try
          (f result)
          (finally (dispose-all! decoder result))))
      (catch :default e
        (doseq [h (vec (dec-live-handles decoder))]
          (dec-dispose! decoder h))
        (throw e)))))

(defn live-handles
  "The decoder's live-handle count (R1/R2 assertions)."
  [decoder] (count (dec-live-handles decoder)))
