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
        bookkeeping. Fork cost is backend-dependent (CUDA flat path: a cache
        COPY; Metal paged path: a block-share) — measured numbers live in
        bench/token_smc_bench.cljs, not hidden here.

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

(deftype NativeDecoder [model live]
  ITokenDecoder
  (dec-prefill! [_ prompt-ids]
    (llm/init-cache! model)
    (let [logits (mx/materialize! (llm/forward-prefill model (vec prompt-ids)))
          root (llm/branch-cache! model)]
      (swap! live conj root)
      {:root root :logits logits}))
  (dec-fork! [_ handle]
    (let [b (llm/branch-from model handle)]
      (swap! live conj b)
      b))
  (dec-step! [_ handle tok-id]
    (mx/materialize! (llm/forward-branch model handle tok-id)))
  (dec-dispose! [_ handle]
    (swap! live disj handle)
    (try (llm/dispose-branch! model handle) (catch :default _ nil)))
  (dec-live-handles [_] @live))

(defn native-decoder
  "Decoder over the native branchable KV cache (requires
   llm/supports-branching?)."
  [model]
  (->NativeDecoder model (atom #{})))

(deftype ReplayDecoder [model prompt-ref live counter]
  ITokenDecoder
  (dec-prefill! [_ prompt-ids]
    (vreset! prompt-ref (vec prompt-ids))
    (let [logits (mx/materialize! (llm/forward-pass model (vec prompt-ids)))
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
      (mx/materialize! (llm/forward-pass model (into @prompt-ref toks)))))
  (dec-dispose! [_ handle] (swap! live dissoc handle))
  (dec-live-handles [_] (keys @live)))

(defn replay-decoder
  "Correct-but-O(T)-per-step decoder for models WITHOUT the branch surface
   (dense CljsForwardModel): a handle is a token vector; every step
   re-forwards prompt+tokens through the uncached forward. The asymmetry vs
   the native decoder is the documented R3 cost difference, not hidden."
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
    :log-ml-estimate mx-scalar :ess-trajectory [..] :decoder d}."
  [decoder {:keys [particles ess-threshold max-tokens eos-id proposal twist
                   constraint key callback]
            :or {particles 8 ess-threshold 0.5 proposal :model}}
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
            (recur (inc t) st'' knext)))))))

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
    out))

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
