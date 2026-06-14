;; @tier fast
(ns genmlx.hardening-sweep-2026-06-14-test
  "Independent-oracle regression tests for the 2026-06-14 foundation hardening
   sweep (epic genmlx-bjcq). Each block pins a bug found by the static bug-hunt
   to a closed-form / handler-parity oracle so the fix can never silently
   regress. One section per child bean, labelled with its id.

   No models are loaded; all LLM-layer findings are tracked elsewhere."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.combinators :as comb]
            [genmlx.inference.smcp3 :as smcp3]
            [genmlx.inference.vi :as vi]
            [genmlx.llm.grammar :as grammar]
            [genmlx.agents.biased-planners :as bp]
            [genmlx.test-helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn- v-at [trace addr]
  (mx/item (cm/get-value (cm/get-submap (:choices trace) addr))))

;; ===========================================================================
;; genmlx-abw8 (C26) — {0}/{0,0} repeat quantifier no longer crashes; matches ε
;; ===========================================================================

(deftest grammar-zero-repeat-quantifier
  (testing "a{0}b compiles and is equivalent to b (the atom matches empty)"
    (let [dfa (grammar/compile-regex "a{0}b")]
      (is (grammar/dfa-accepts? dfa "b") "a{0}b accepts \"b\"")
      (is (not (grammar/dfa-accepts? dfa "ab")) "a{0}b rejects \"ab\"")))
  (testing "a{0,0}c compiles and is equivalent to c"
    (let [dfa (grammar/compile-regex "a{0,0}c")]
      (is (grammar/dfa-accepts? dfa "c") "a{0,0}c accepts \"c\"")
      (is (not (grammar/dfa-accepts? dfa "aac")) "a{0,0}c rejects \"aac\"")))
  (testing "non-zero repeats still work (no regression)"
    (let [dfa (grammar/compile-regex "a{2}")]
      (is (grammar/dfa-accepts? dfa "aa"))
      (is (not (grammar/dfa-accepts? dfa "a")))
      (is (not (grammar/dfa-accepts? dfa "aaa"))))))

;; ===========================================================================
;; genmlx-lgun (C1) — broadcasted-normal batched log-prob keeps the particle axis
;; ===========================================================================

(deftest broadcasted-normal-batched-logprob
  (testing "[N,D] batch -> [N] per-row log-prob (oracle: per-element gaussian-lp)"
    (let [mu    [1.0 2.0 3.0]
          sigma [0.5 0.7 1.0]
          d     (dist/broadcasted-normal (mx/array mu) (mx/array sigma))
          rows  [[1.0 2.0 3.0] [0.0 0.0 0.0]]
          v     (mx/array rows)
          lp    (dc/dist-log-prob d v)]
      (is (= [2] (mx/shape lp)) "log-prob preserves the leading particle axis -> [N]")
      (doseq [[i row] (map-indexed vector rows)]
        (let [oracle (reduce + (map (fn [x m s] (h/gaussian-lp x m s)) row mu sigma))]
          (is (h/close? oracle (mx/item (mx/index lp i)) 1e-4)
              (str "row " i " log-prob = sum of per-element gaussian log-probs"))))))
  (testing "scalar (unbatched) mode still collapses to a scalar"
    (let [d  (dist/broadcasted-normal (mx/array [1.0 2.0 3.0]) (mx/array [0.5 0.7 1.0]))
          lp (dc/dist-log-prob d (mx/array [1.0 2.0 3.0]))]
      (is (= [] (mx/shape lp)) "unbatched [D] value -> scalar log-prob"))))

;; ===========================================================================
;; genmlx-lgun (C18) — geometric inverse-CDF cannot produce +Inf at the u=0 edge
;; ===========================================================================

(deftest geometric-u0-boundary
  (testing "u=0 yields a finite k=0, not +Inf (inverse-CDF uses 1-u)"
    (with-redefs [rng/uniform (fn [_key _shape] (mx/scalar 0.0))]
      (let [g (dyn/auto-key (gen [] (trace :g (dist/geometric 0.3))))
            t (p/simulate g [])
            v (v-at t :g)]
        (is (js/isFinite v) "geometric sample at u=0 is finite")
        (is (= 0.0 v) "geometric sample at u=0 is k=0")))))

;; ===========================================================================
;; genmlx-zek9 — Mix combinator update/regenerate weight + key
;; ===========================================================================

;; Components carry an UNCONSTRAINED latent :m and an obs :x. :x is INDEPENDENT
;; of :m (deliberately non-conjugate) so the components score with the joint
;; generate semantics (weight = constrained sites only), not the L3 analytical
;; marginal-weight path that a normal-normal pair would trigger.
(def ^:private mix-a (dyn/auto-key (gen [] (let [m (trace :m (dist/gaussian 0.0 1.0))]
                                             (trace :x (dist/gaussian 0.0 0.5))
                                             m))))
(def ^:private mix-b (dyn/auto-key (gen [] (let [m (trace :m (dist/gaussian 5.0 1.0))]
                                             (trace :x (dist/gaussian 0.0 0.5))
                                             m))))
(def ^:private mix2 (comb/mix-combinator [mix-a mix-b]
                                         (mx/array [(js/Math.log 0.5) (js/Math.log 0.5)])))

(deftest mix-component-flip-update-weight
  (testing "component-flip update weight excludes the fresh latent (oracle: score-delta - weight = log p(fresh m'))"
    (let [c0  (-> cm/EMPTY
                  (cm/set-choice [:component-idx] (mx/scalar 0 mx/int32))
                  (cm/set-choice [:x] (mx/scalar 0.3)))
          {t0 :trace} (p/generate mix2 [] c0)
          ;; flip to component 1; keep x constrained, resample its latent m
          c1  (-> cm/EMPTY
                  (cm/set-choice [:component-idx] (mx/scalar 1 mx/int32))
                  (cm/set-choice [:x] (mx/scalar 0.3)))
          {t1 :trace w :weight} (p/update mix2 t0 c1)   ; must NOT throw (latent keyed)
          m'  (v-at t1 :m)
          score-delta (- (mx/item (:score t1)) (mx/item (:score t0)))
          ;; fresh latent m' was drawn from component 1's prior N(5,1)
          oracle (h/gaussian-lp m' 5.0 1.0)]
      (is (h/close? oracle (- score-delta (mx/item w)) 1e-4)
          "score(t')-score(t)-weight == log p(fresh latent under new component prior)")
      ;; The pre-fix bug (weight = full new score) made this difference exactly 0.
      (is (> (Math/abs (- score-delta (mx/item w))) 1e-3)
          "weight is not the buggy full-score delta (which would give 0)"))))

(deftest mix-index-regen-retains-inner
  (testing "regenerate of :component-idx with one component retains the unselected inner choices, W=0"
    (let [mix1 (comb/mix-combinator [mix-a] (mx/array [0.0]))
          c0   (-> cm/EMPTY
                   (cm/set-choice [:component-idx] (mx/scalar 0 mx/int32))
                   (cm/set-choice [:x] (mx/scalar 0.3)))
          {t0 :trace} (p/generate mix1 [] c0)
          m0   (v-at t0 :m)
          x0   (v-at t0 :x)
          {t1 :trace w :weight} (p/regenerate mix1 t0 (sel/select :component-idx))]
      (is (= 0.0 (mx/item w)) "index-only regenerate weight is 0")
      (is (= m0 (v-at t1 :m))
          "unselected inner latent :m is retained bit-for-bit (not resampled)")
      (is (= x0 (v-at t1 :x)) "unselected inner obs :x is retained bit-for-bit"))))

;; ===========================================================================
;; genmlx-gc4w (C5) — branch-rewritten compiled ops must not leak the reserved
;; branch-cond key into the trace choicemap
;; ===========================================================================

(def ^:private branch-model
  (gen [flag]
    (if flag
      (trace :x (dist/gaussian 0.0 1.0))
      (trace :x (dist/gaussian 0.0 2.0)))))

(defn- has-branch-cond-leak? [trace]
  ;; cm/addresses returns paths (vectors); a leaked reserved key would appear as
  ;; a path element with the "genmlx.compiled" namespace.
  (some (fn [path]
          (some #(and (keyword? %) (= "genmlx.compiled" (namespace %))) path))
        (cm/addresses (:choices trace))))

(deftest branch-rewrite-no-cond-key-leak
  (testing "compiled branch-rewritten generate/update do not leak branch-cond keys"
    (let [gf  (dyn/with-key branch-model (rng/fresh-key 7))
          c   (cm/set-choice cm/EMPTY [:x] (mx/scalar 0.5))
          {t0 :trace} (p/generate gf [true] c)
          {t1 :trace} (p/update gf t0 (cm/set-choice cm/EMPTY [:x] (mx/scalar 0.9)))]
      (is (not (has-branch-cond-leak? t0)) "generate trace choicemap has no genmlx.compiled/* leaf")
      (is (not (has-branch-cond-leak? t1)) "update trace choicemap has no genmlx.compiled/* leaf")
      (is (= #{[:x]} (set (cm/addresses (:choices t0)))) "generate choices are exactly {:x}")
      (is (= #{[:x]} (set (cm/addresses (:choices t1)))) "update choices are exactly {:x}"))))

;; ===========================================================================
;; genmlx-1thx — conjugate analytical detection
;; ===========================================================================

;; A Gamma prior conjugate to BOTH a Poisson and an Exponential child: a single
;; family cannot eliminate it correctly, so the analytical path must decline and
;; the weight must equal the handler path (strip-analytical).
(def ^:private mixed-family-model
  (dyn/auto-key
    (gen []
      (let [rate (trace :rate (dist/gamma-dist 2.0 1.0))]
        (trace :c (dist/poisson rate))
        (trace :xx (dist/exponential rate))
        rate))))

(deftest conjugate-mixed-family-declines
  (testing "gamma prior with poisson+exponential children: analytical weight == handler weight"
    (let [obs (-> cm/EMPTY
                  (cm/set-choice [:c] (mx/scalar 3 mx/int32))
                  (cm/set-choice [:xx] (mx/scalar 0.8)))
          k   (rng/fresh-key 11)
          wa  (mx/item (:weight (p/generate (dyn/with-key mixed-family-model k) [] obs)))
          wh  (mx/item (:weight (p/generate (dyn/with-key (dyn/strip-analytical-path mixed-family-model) k) [] obs)))]
      (is (h/close? wh wa 1e-3)
          "mixed-family prior is declined to the handler (no wrong single-family marginal)"))))

;; A let-rebound (affine-shadowed) natural parameter: y ~ N(mu+5, 1). The schema
;; erases the +5, so :direct must NOT fire; the path declines to the handler.
(def ^:private affine-shadow-model
  (dyn/auto-key
    (gen []
      (let [mu (trace :mu (dist/gaussian 0.0 10.0))
            mu (mx/add mu (mx/scalar 5.0))]
        (trace :y (dist/gaussian mu 1.0))
        mu))))

(deftest conjugate-affine-shadow-not-direct
  (testing "let-rebound affine natural param: analytical weight == handler weight (offset not dropped)"
    (let [obs (cm/set-choice cm/EMPTY [:y] (mx/scalar 2.0))
          k   (rng/fresh-key 13)
          wa  (mx/item (:weight (p/generate (dyn/with-key affine-shadow-model k) [] obs)))
          wh  (mx/item (:weight (p/generate (dyn/with-key (dyn/strip-analytical-path affine-shadow-model) k) [] obs)))]
      (is (h/close? wh wa 1e-3)
          "shadowed natural param not scored as a direct (offset-0) conjugate"))))

(deftest conjugate-direct-still-detected
  (testing "a genuine direct conjugate (mu used directly) is still eliminated (no false negative)"
    (let [m (dyn/auto-key (gen [] (let [mu (trace :mu (dist/gaussian 0.0 10.0))]
                                    (trace :y (dist/gaussian mu 1.0))
                                    mu)))
          pairs (:conjugate-pairs (:schema m))]
      (is (= 1 (count pairs)) "direct normal-normal pair detected")
      (is (= :direct (:type (:dependency-type (first pairs)))) "classified :direct"))))

;; ===========================================================================
;; genmlx-9rwf (C22) — VIMCO objective is finite at K=1 (no /0 NaN)
;; ===========================================================================

(deftest vimco-k1-finite
  (testing "vimco-objective with a single sample is finite (reinforce term guarded)"
    (let [obj (vi/vimco-objective (fn [z] (mx/negative (mx/sum (mx/square z))))
                                  (fn [z] (mx/negative (mx/sum (mx/square z)))))
          samples (mx/reshape (mx/array [0.3 -0.2]) [1 2])  ; K=1, d=2
          loss (obj samples)]
      (is (js/isFinite (mx/item loss)) "K=1 VIMCO loss is finite (not NaN)"))))

;; ===========================================================================
;; genmlx-ivs0 — SMCP3 is reproducible under a fixed :key
;; ===========================================================================

(def ^:private smcp3-model
  (dyn/auto-key (gen [] (let [z (trace :z (dist/gaussian 0.0 1.0))]
                          (trace :y (dist/gaussian z 1.0))
                          z))))

(deftest smcp3-reproducible-under-key
  (testing "two SMCP3 runs with the same explicit :key produce identical particles"
    (let [obs-seq [(cm/choicemap :y (mx/scalar 2.0))]
          run (fn [] (smcp3/smcp3 {:particles 8 :key (rng/fresh-key 99)}
                                  smcp3-model [] obs-seq))
          r1 (run)
          r2 (run)
          zs (fn [r] (mapv #(v-at % :z) (:traces r)))]
      (is (= (zs r1) (zs r2)) "particle :z values are identical across keyed runs"))))

;; ===========================================================================
;; genmlx-m3nn — biased MDP agent: keyed rollout works + :fused is rejected
;; ===========================================================================

(deftest biased-mdp-keyed-rollout
  (testing "simulate-biased-mdp with :key runs (no arity crash) and is reproducible"
    (let [mdp   (bp/procrastination-mdp {})
          agent (bp/make-biased-mdp-agent {:mdp mdp :alpha 2.0 :gamma 1.0 :n-iters 14}
                                          {:bias :naive})
          k     (rng/fresh-key 5)
          r1    (bp/simulate-biased-mdp agent 0 10 {:key k})
          r2    (bp/simulate-biased-mdp agent 0 10 {:key k})]
      (is (vector? (:actions r1)) "keyed biased rollout returns actions (no ArityException)")
      (is (= (:actions r1) (:actions r2)) "same key -> same trajectory")))
  (testing ":rollout-mode :fused is rejected for biased agents (no nil-Q deref)"
    (let [mdp   (bp/procrastination-mdp {})
          agent (bp/make-biased-mdp-agent {:mdp mdp :alpha ##Inf :gamma 1.0 :n-iters 14}
                                          {:bias :naive})]
      (is (thrown? js/Error (bp/simulate-biased-mdp agent 0 5 {:rollout-mode :fused}))
          "fused rollout on a biased agent throws a clear error"))))

;; ===========================================================================
;; genmlx-flmw — SpliceSite schema shape matches schema.cljs/handle-splice
;; ===========================================================================

(deftest splice-site-schema-shape
  (testing "a real splice site uses :gf-form (not the non-existent :gf-sym)"
    (let [child (gen [] (trace :z (dist/gaussian 0.0 1.0)))
          host  (gen [] (let [a (splice :sub child)] a))
          sch   (:schema host)
          ss    (first (:splice-sites sch))]
      (is (some? ss) "host model has a splice site")
      (is (contains? ss :gf-form) "splice site has :gf-form")
      (is (not (contains? ss :gf-sym)) "splice site has no :gf-sym key")
      (is (every? #(contains? ss %) [:addr :addr-form :gf-form :splice-args :deps :static?])
          "splice site carries the full handle-splice key set"))))

(cljs.test/run-tests)
