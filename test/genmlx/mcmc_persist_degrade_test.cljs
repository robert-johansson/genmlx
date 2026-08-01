;; @tier fast
;; NOTE: the `aset` below MUST run before the ns form — `vmala-chunk-ops-override`
;; in genmlx.inference.mcmc is read at namespace-load time, and the ns form is
;; what loads it. Setting it here (rather than reaching into a private var) makes
;; the chunked-MALA rows exercise the REAL routing decision. Per-file process
;; isolation (test/run.sh) keeps this env write from leaking to other files.
(aset (.-env js/process) "GENMLX_VMALA_CHUNK_OPS" "4")

(ns genmlx.mcmc-persist-degrade-test
  "Positive oracles for two mcmc.cljs seam defects (2026-08-01 audit,
   parent genmlx-wkdl).

   genmlx-01e7 — persist-chain / persist-chain1 gained REAL persistent
   capture in 7e92912 (replacing mx/compile-fn, a documented identity
   pass-through), but not the untraceable-fn degrade catch that its sibling
   persist-point-fn got 11 minutes later. On CUDA that made every model whose
   score forces a host eval (host-side branching on mx/item, an explicit
   mx/eval!, the Mix combinator's (int (mx/item ...))) hard-THROW under
   compiled-mh / mala / hmc at their DEFAULTS. Handler-path parity (CLAUDE.md
   principle 5) says the compiled path must produce the same samples, not an
   exception, so the oracle here asserts sample COUNT + finiteness + that the
   degrade actually fired — never merely 'it did not throw'.

   genmlx-n896 — the N-chain MALA chain terminal stacked a host-side
   accumulator with no empty case, so a zero-sample chain (the chunked
   runner's burn-only [b 0] chunks, and equally a plain {:samples 0} sweep on
   the single-graph path — the SAME line) died on
   'stack requires at least one array'.

   Both rows below FAIL loudly with the respective fix reverted; see the bean
   Summary of Changes for the observed before/after."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.combinators :as comb]
            [genmlx.inference.mcmc :as mcmc]))

;; ---------------------------------------------------------------------------
;; Harness
;; ---------------------------------------------------------------------------

(defn- run-capturing
  "Run `f`, capturing both its *out* (the degrade notices are printlns) and
   any throw. Returns {:out str :res value :err error-or-nil}."
  [f]
  (let [res (volatile! nil)
        err (volatile! nil)
        out (with-out-str
              (try (vreset! res (f))
                   (catch :default e (vreset! err e))))]
    {:out out :res @res :err @err}))

(defn- err-msg [e] (if e (str (.-message e)) "<none>"))

;; A model whose score CANNOT be traced: `mu` is chosen by a HOST-SIDE branch
;; on (mx/item z), which throws inside an MLX compile trace. This is the same
;; class as the Mix combinator's (int (mx/item ...)) at combinators.cljs:171.
(def branch-model
  (gen []
    (let [z  (trace :z (dist/gaussian 0 1))
          mu (if (pos? (mx/item z)) 3.0 -3.0)
          x  (trace :x (dist/gaussian mu 1))]
      (trace :y (dist/gaussian x 0.5))
      x)))

(def branch-obs (cm/from-map {:y 1.0}))

(defn- keyed-branch-model [] (dyn/with-key branch-model (rng/fresh-key 11)))

;; A model with NO conjugate elimination of :mu, so fused-vectorized-mala
;; actually has a latent to sample (a conjugate one is eliminated away and the
;; [N,D] init-q stack degenerates for an unrelated reason).
(def quad-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0 1))]
      (trace :y (dist/gaussian (mx/multiply mu mu) 0.5))
      mu)))

(def quad-obs (cm/from-map {:y 1.0}))

(defn- keyed-quad-model [] (dyn/with-key quad-model (rng/fresh-key 7)))

;; ---------------------------------------------------------------------------
;; genmlx-01e7 — untraceable score fns must DEGRADE, not throw
;; ---------------------------------------------------------------------------

(def ^:private n-req 4)

(deftest untraceable-model-runs-under-all-three-samplers-test
  (testing "compiled-mh / mala / hmc at :compile? true defaults on a host-branch model"
    ;; `moves?` — whether the chain is expected to visit >1 distinct state.
    ;; MEASURED, not assumed: an untraceable model is also non-autodifferentiable
    ;; through MLX's transform, so MALA's proposal degenerates and its chain sits
    ;; still on this model. That is PRE-EXISTING, not an artifact of the degrade
    ;; catch: forcing persist-chain/persist-chain1 back to the pre-7e92912
    ;; identity path gives the same 4 identical MALA samples at the same key
    ;; (and byte-identical HMC samples). It deserves its own bean; the contract
    ;; this test pins is that all three RUN and return real samples.
    (doseq [[nm moves? run]
            [["compiled-mh" true
              #(mcmc/compiled-mh {:samples n-req :burn 2 :addresses [:z :x]
                                  :key (rng/fresh-key 99)}
                                 (keyed-branch-model) [] branch-obs)]
             ["mala" false
              #(mcmc/mala {:samples n-req :burn 2 :addresses [:z :x]
                           :key (rng/fresh-key 99)}
                          (keyed-branch-model) [] branch-obs)]
             ["hmc" true
              #(mcmc/hmc {:samples n-req :burn 2 :addresses [:z :x]
                          :key (rng/fresh-key 99)}
                         (keyed-branch-model) [] branch-obs)]]]
      (let [{:keys [out res err]} (run-capturing run)]
        ;; 1. It completed at all. (Pre-fix: "Attempting to eval an array
        ;;    during function transformations ... is not allowed".)
        (is (nil? err)
            (str nm ": must not throw on an untraceable score fn — got " (err-msg err)))
        ;; 2. POSITIVE: the requested number of samples actually came back.
        (is (= n-req (count res))
            (str nm ": expected " n-req " samples, got " (count res)))
        ;; 3. POSITIVE: every sampled coordinate is a real number.
        (is (every? (fn [s] (and (pos? (count s)) (every? h/finite? s)))
                    res)
            (str nm ": every sample must be non-empty and finite, got " (pr-str res)))
        ;; 4. POSITIVE: the chain MOVED — a degraded-but-broken path that
        ;;    returned the initial point S times would pass 1-3. (See the
        ;;    `moves?` note above for why MALA is exempt on this model.)
        (when moves?
          (is (< 1 (count (distinct (map vec res))))
              (str nm ": chain must move (distinct sample values), got " (pr-str res))))
        ;; 5. POSITIVE: the degrade path is what carried it — the chain
        ;;    wrapper, not only the point wrapper, announced itself.
        (is (re-find #"Note: persist-chain1? — fn is not traceable" out)
            (str nm ": expected a persist-chain/persist-chain1 degrade notice; out=" (pr-str out)))))))

(deftest degrade-notice-is-printed-once-per-wrapper-test
  (testing "a degraded wrapper announces itself exactly once and then stays on the raw fn"
    (let [persist-chain1 @(resolve 'genmlx.inference.mcmc/persist-chain1)
          persist-chain  @(resolve 'genmlx.inference.mcmc/persist-chain)
          ;; Untraceable single-output builder: mx/item inside.
          raw1 (fn [x] (mx/add x (mx/scalar (if (pos? (mx/item x)) 1.0 -1.0))))
          ;; Untraceable multi-output builder (persist-chain's #js convention).
          rawN (fn [x] #js [(mx/add x (mx/scalar (if (pos? (mx/item x)) 1.0 -1.0)))])]
      (doseq [[nm wrapped expect pick]
              [["persist-chain1" (persist-chain1 raw1 :mh 4) 3.0 identity]
               ["persist-chain"  (persist-chain rawN :mh 4)  3.0 #(aget % 0)]]]
        (let [{:keys [out res err]}
              (run-capturing
               (fn [] (mapv #(h/realize (pick (wrapped (mx/scalar %))))
                            [2.0 2.0 2.0])))]
          (is (nil? err) (str nm ": must degrade, not throw — got " (err-msg err)))
          (is (= 3 (count res)) (str nm ": all three calls must return"))
          (is (every? #(h/close? expect % 1e-5) res)
              (str nm ": degraded calls must equal the raw fn (2+1), got " (pr-str res)))
          (is (= 1 (count (re-seq #"is not traceable" out)))
              (str nm ": notice must print exactly ONCE across 3 calls, out=" (pr-str out))))))))

;; The bean's second named member of the class: the Mix combinator's own
;; (int (mx/item ...)) at combinators.cljs:171/2313/2503. It only bites when
;; :component-idx is an UNCONSTRAINED latent — constrained, the item reads an
;; already-evaluated observation and the score traces fine. Mix must be the
;; TOP-LEVEL gf: spliced inside a parent, prepare-mcmc-score rejects the
;; nested latents long before any compile trace (a separate, pre-existing
;; guard).
(def mix-model
  (comb/mix-combinator
   [(dyn/auto-key (gen [] (trace :v (dist/gaussian -2.0 1.0))))
    (dyn/auto-key (gen [] (trace :v (dist/gaussian  2.0 1.0))))]
   (mx/array [0.0 0.0])))

(deftest mix-combinator-runs-under-all-three-samplers-test
  (testing "Mix with an unconstrained :component-idx — the bean's named sibling model"
    (doseq [[nm run] [["compiled-mh"
                       #(mcmc/compiled-mh {:samples n-req :burn 2 :key (rng/fresh-key 21)}
                                          mix-model [] cm/EMPTY)]
                      ["mala"
                       #(mcmc/mala {:samples n-req :burn 2 :key (rng/fresh-key 21)}
                                   mix-model [] cm/EMPTY)]
                      ["hmc"
                       #(mcmc/hmc {:samples n-req :burn 2 :key (rng/fresh-key 21)}
                                  mix-model [] cm/EMPTY)]]]
      (let [{:keys [out res err]} (run-capturing run)]
        (is (nil? err) (str "Mix/" nm ": must not throw — got " (err-msg err)))
        (is (= n-req (count res))
            (str "Mix/" nm ": expected " n-req " samples, got " (count res)))
        (is (every? (fn [s] (and (= 2 (count s)) (every? h/finite? s))) res)
            (str "Mix/" nm ": each sample is [v component-idx], both finite; got " (pr-str res)))
        (is (re-find #"Note: persist-chain1? — fn is not traceable" out)
            (str "Mix/" nm ": expected a chain-wrapper degrade notice; out=" (pr-str out)))))))

(deftest traceable-model-still-takes-the-captured-path-test
  (testing "the degrade catch does not disarm capture for well-behaved models"
    (let [{:keys [out res err]}
          (run-capturing
           #(mcmc/mala {:samples n-req :burn 2 :addresses [:mu]}
                       (keyed-quad-model) [] quad-obs))]
      (is (nil? err) (str "traceable mala must not throw — got " (err-msg err)))
      (is (= n-req (count res)) "traceable mala returns the requested samples")
      (is (every? (fn [s] (every? h/finite? s)) res) "traceable mala samples are finite")
      (is (nil? (re-find #"is not traceable" out))
          (str "a traceable model must NOT print a degrade notice; out=" (pr-str out))))))

;; ---------------------------------------------------------------------------
;; genmlx-n896 — zero-sample chain terminals must not (mx/stack [])
;; ---------------------------------------------------------------------------

(defn- vmala [opts]
  (mcmc/fused-vectorized-mala
   (merge {:n-chains 3 :addresses [:mu] :key (rng/fresh-key 5)} opts)
   (keyed-quad-model) [] quad-obs))

(deftest chunked-vectorized-mala-with-burn-test
  (testing "chunked N-chain MALA with :burn > 0 (burn-only [b 0] chunks)"
    (let [{:keys [out res err]} (run-capturing #(vmala {:samples 2 :burn 6}))]
      (is (nil? err)
          (str "chunked MALA with burn>0 must not throw — got " (err-msg err)))
      ;; The routing we intended is the routing that ran.
      (is (re-find #"GENMLX_VMALA_CHUNK_OPS routing" out)
          (str "expected the chunked runner to be selected; out=" (pr-str out)))
      (is (= [2 3 1] (h/realize-shape (:samples res)))
          "chunked MALA returns [S,N,D] samples across the burn-only chunk seams")
      (is (every? h/finite? (flatten (h/realize-vec (:samples res))))
          "chunked MALA samples are finite")
      (is (h/finite? (:acceptance-rate res)) "acceptance rate is a real number"))))

(deftest chunked-vectorized-mala-zero-samples-test
  (testing "chunked N-chain MALA with {:samples 0}"
    (let [{:keys [res err]} (run-capturing #(vmala {:samples 0 :burn 6}))]
      (is (nil? err)
          (str "chunked MALA with samples=0 must not throw — got " (err-msg err)))
      (is (= [0 3 1] (h/realize-shape (:samples res)))
          "an all-burn sweep yields an EMPTY [0,N,D] result, not an error"))))

(deftest single-graph-vectorized-mala-zero-samples-test
  (testing "the pre-existing {:samples 0} defect on the single-graph path (same line)"
    ;; total-steps 1 → (* 2 1) = 2, under the CHUNK_OPS=4 threshold, so this
    ;; row rides the NON-chunked whole-sweep builder — the site n896 records
    ;; as pre-dating 158d4eb.
    (let [{:keys [out res err]} (run-capturing #(vmala {:samples 0 :burn 1}))]
      (is (nil? err)
          (str "single-graph MALA with samples=0 must not throw — got " (err-msg err)))
      (is (nil? (re-find #"GENMLX_VMALA_CHUNK_OPS routing" out))
          (str "this row must NOT be chunk-routed (it pins the other path); out=" (pr-str out)))
      (is (= [0 3 1] (h/realize-shape (:samples res)))
          "zero-sample single-graph sweep yields an empty [0,N,D] result"))))

(cljs.test/run-tests)
